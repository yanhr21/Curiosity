#!/usr/bin/env python3
"""CPU structural audit for action-residual-bounded tactile SUGAR fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

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
        tactile_action_residual_cap=0.1,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    warm_start = policy.load_sugar_warm_start(source_state)
    finetune = policy.configure_tactile_actor_finetune()

    base_width = policy.num_actor_base_obs
    first_layer = policy.actor[0]
    probe_base = torch.linspace(-1.0, 1.0, steps=2 * base_width).reshape(
        2, base_width
    )
    generator = torch.Generator().manual_seed(13012)
    strong_features = 5000.0 * torch.randn(4096, 256, generator=generator)
    strong_base = probe_base[:1].expand(strong_features.shape[0], -1)
    actor_input = torch.cat((strong_base, strong_features), dim=-1)
    with torch.no_grad():
        base_action = policy.actor(
            torch.cat((strong_base, torch.zeros_like(strong_features)), dim=-1)
        )
        tactile_action_before_output_bound = (
            policy._tactile_enhanced_actor_forward(actor_input)
        )
        bounded_action = policy._actor_forward(actor_input)
        raw_residual = tactile_action_before_output_bound - base_action
        applied_residual = bounded_action - base_action

    zero_action = policy.act_inference({name: value.clone() for name, value in obs.items()})
    direct_zero_action = policy.actor(
        torch.cat((obs["policy"], torch.zeros(2, 256)), dim=-1)
    )

    contact_obs = {name: value.clone() for name, value in obs.items()}
    contact_obs["policy"] = probe_base.clone()
    contact_obs[tactile_group][:, :4000] = torch.linspace(
        -0.2, 0.2, steps=8000
    ).reshape(2, 4000)
    policy.zero_grad(set_to_none=True)
    policy.act_inference(contact_obs).square().mean().backward()
    base_gradient_abs_max = float(
        first_layer.weight.grad[:, :base_width].abs().max()
    )
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

    raw_abs_max = float(raw_residual.abs().max())
    applied_abs_max = float(applied_residual.abs().max())
    checks = {
        "official_zero_tactile_actor_exact": (
            warm_start["actor_zero_tactile_max_abs_error"] == 0.0
        ),
        "zero_runtime_matches_direct_official_path": bool(
            torch.equal(zero_action, direct_zero_action)
        ),
        "hidden_preactivation_cap_is_0p15": (
            policy.tactile_preactivation_cap == 0.15
        ),
        "action_residual_cap_is_0p1": policy.tactile_action_residual_cap == 0.1,
        "action_residual_respects_cap": applied_abs_max <= 0.10000001,
        "strong_probe_reaches_action_residual_bound": applied_abs_max > 0.099,
        "action_bound_reduces_strong_probe": applied_abs_max < raw_abs_max,
        "bounded_forward_is_finite": bool(torch.isfinite(bounded_action).all()),
        "base_column_gradient_exact_zero": base_gradient_abs_max == 0.0,
        "tactile_column_gradient_nonzero": tactile_gradient_l2 > 0.0,
        "spatial_encoder_gradient_nonzero": encoder_gradient_l2 > 0.0,
        "finetune_report_declares_action_bound": (
            finetune["tactile_action_residual"]
            == "bounded_relative_to_exact_official_zero_tactile_action"
            and finetune["tactile_action_residual_cap"] == 0.1
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"action residual structural audit failed: {checks}")
    report = {
        "schema": "action_residual_native_whole_hand_tactile_fusion_audit_v1",
        "checks": checks,
        "dimensions": {
            "base_actor": base_width,
            "raw_tactile": 324000,
            "tactile_embedding": 256,
            "first_hidden": first_layer.out_features,
            "actions": 29,
        },
        "declared_bounds": {
            "hidden_preactivation_per_unit": 0.15,
            "normalized_action_residual_per_action": 0.1,
        },
        "strong_probe": {
            "raw_action_residual_abs_max": raw_abs_max,
            "applied_action_residual_abs_max": applied_abs_max,
        },
        "gradients": {
            "base_columns_abs_max": base_gradient_abs_max,
            "tactile_columns_l2": tactile_gradient_l2,
            "spatial_encoder_l2": encoder_gradient_l2,
        },
        "distillation_reduction": "official all-sample mean",
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
