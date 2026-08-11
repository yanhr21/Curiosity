#!/usr/bin/env python3
"""CPU structural audit for bounded native-tactile SUGAR fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sugar_rl.utils.native_tactile_training_bcppo import (
    NativeTactileTrainingBCPPO,
)
from sugar_rl.utils.reference_only_tactile_actor_critic import (
    ReferenceOnlyTactileActorCritic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    checkpoint = torch.load(
        args.teacher_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    source_state = checkpoint["model_state_dict"]
    tactile_group = "native_whole_hand_tactile_history"
    obs = {
        "policy": torch.zeros(2, 890),
        tactile_group: torch.zeros(2, 324000),
        "critic": torch.zeros(2, 890),
        "teacher": torch.zeros(2, 890),
    }
    obs_groups = {
        "policy": ["policy", tactile_group],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    torch.manual_seed(13011)
    policy = ReferenceOnlyTactileActorCritic(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=29,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        tactile_obs_group=tactile_group,
        tactile_grid_shape=(20, 25),
        tactile_num_hands=2,
        tactile_channels_per_hand=324,
        tactile_encoder_channels=(32, 64, 64),
        tactile_embedding_dim=128,
        tactile_preactivation_cap=0.15,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    warm_start = policy.load_sugar_warm_start(source_state)
    finetune = policy.configure_tactile_actor_finetune()

    first_layer = policy.actor[0]
    base_width = policy.num_actor_base_obs
    probe_base = torch.linspace(-1.0, 1.0, steps=2 * base_width).reshape(
        2, base_width
    )
    tactile_features = torch.linspace(-800.0, 800.0, steps=2 * 256).reshape(
        2, 256
    )
    raw_correction = F.linear(
        tactile_features, first_layer.weight[:, base_width:], None
    )
    bounded_correction = 0.15 * torch.tanh(raw_correction / 0.15)
    bounded_action = policy._actor_forward(
        torch.cat((probe_base, tactile_features), dim=-1)
    )

    zero_obs = {name: value.clone() for name, value in obs.items()}
    zero_action = policy.act_inference(zero_obs)
    direct_zero_action = policy.actor(
        torch.cat((zero_obs["policy"], torch.zeros(2, 256)), dim=-1)
    )

    contact_obs = {name: value.clone() for name, value in obs.items()}
    contact_obs["policy"] = probe_base.clone()
    contact_obs[tactile_group][:, :4000] = torch.linspace(
        -0.2, 0.2, steps=8000
    ).reshape(2, 4000)
    policy.zero_grad(set_to_none=True)
    contact_action = policy.act_inference(contact_obs)
    contact_action.square().mean().backward()
    base_gradient_abs_max = float(first_layer.weight.grad[:, :base_width].abs().max())
    tactile_gradient_l2 = float(first_layer.weight.grad[:, base_width:].norm())
    encoder_gradient_l2 = float(
        torch.sqrt(
            sum(
                parameter.grad.square().sum()
                for parameter in policy.actor_tactile_encoder.parameters()
                if parameter.grad is not None
            )
        )
    )

    reducer = NativeTactileTrainingBCPPO.__new__(NativeTactileTrainingBCPPO)
    reducer.contact_balanced_distillation = True
    reducer.policy = SimpleNamespace(tactile_obs_group=tactile_group)
    reducer._contact_distill_supported_samples = 0
    reducer._contact_distill_total_samples = 0
    reducer._contact_distill_supported_batches = 0
    per_sample_loss = torch.tensor([1.0, 10.0, 3.0, 20.0])
    reducer_obs = {tactile_group: torch.zeros(4, 324000)}
    reducer_obs[tactile_group][0, 0] = 1.0
    reducer_obs[tactile_group][2, 1] = -1.0
    reduced_loss = reducer._reduce_distill_loss(per_sample_loss, reducer_obs)

    checks = {
        "official_zero_tactile_actor_exact": (
            warm_start["actor_zero_tactile_max_abs_error"] == 0.0
        ),
        "zero_runtime_matches_direct_official_path": bool(
            torch.equal(zero_action, direct_zero_action)
        ),
        "declared_cap_is_0p15": policy.tactile_preactivation_cap == 0.15,
        "bounded_correction_respects_cap": float(bounded_correction.abs().max())
        <= 0.15000001,
        "bounded_correction_reaches_saturation_probe": float(
            bounded_correction.abs().max()
        )
        > 0.149,
        "bounded_forward_is_finite": bool(torch.isfinite(bounded_action).all()),
        "base_column_gradient_exact_zero": base_gradient_abs_max == 0.0,
        "tactile_column_gradient_nonzero": tactile_gradient_l2 > 0.0,
        "spatial_encoder_gradient_nonzero": encoder_gradient_l2 > 0.0,
        "contact_reducer_selects_supported_mean": float(reduced_loss) == 2.0,
        "contact_reducer_counts_samples": (
            reducer._contact_distill_supported_samples == 2
            and reducer._contact_distill_total_samples == 4
            and reducer._contact_distill_supported_batches == 1
        ),
        "finetune_report_declares_bounded_fusion": (
            finetune["tactile_fusion"]
            == "bounded_first_layer_preactivation_correction"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"bounded fusion structural audit failed: {checks}")
    report = {
        "schema": "bounded_native_whole_hand_tactile_fusion_audit_v1",
        "checks": checks,
        "dimensions": {
            "base_actor": base_width,
            "raw_tactile": 324000,
            "tactile_embedding": 256,
            "first_hidden": first_layer.out_features,
            "actions": 29,
        },
        "bounded_correction": {
            "cap": 0.15,
            "raw_abs_max_probe": float(raw_correction.abs().max()),
            "bounded_abs_max_probe": float(bounded_correction.abs().max()),
        },
        "gradients": {
            "base_columns_abs_max": base_gradient_abs_max,
            "tactile_columns_l2": tactile_gradient_l2,
            "spatial_encoder_l2": encoder_gradient_l2,
        },
        "contact_balanced_distillation": {
            "all_sample_mean_probe": float(per_sample_loss.mean()),
            "supported_sample_mean_probe": float(reduced_loss),
            "supported_samples": reducer._contact_distill_supported_samples,
            "total_samples": reducer._contact_distill_total_samples,
        },
        "claim_boundary": (
            "CPU structural evidence only; live contact, optimization, and "
            "frozen physical behavior require separate runs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
