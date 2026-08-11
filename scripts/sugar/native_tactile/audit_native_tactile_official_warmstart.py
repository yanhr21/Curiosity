#!/usr/bin/env python3
"""Audit the official-width whole-hand tactile warm start without Isaac Sim.

This constructs the exact serious SUGAR actor used by Plan 13, loads the
released Refiner, and checks the zero-tactile identity plus the tactile-only
actor gradient gate.  It is a structural gate, not policy-benefit evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sugar_rl.utils.reference_only_tactile_actor_critic import (
    ReferenceOnlyTactileActorCritic,
)


POLICY_WIDTH = 890
TACTILE_WIDTH = 324000
TACTILE_FEATURE_WIDTH = 256
ACTION_WIDTH = 29


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    device = torch.device(args.device)
    torch.manual_seed(13011)

    batch = 2
    observations = {
        "policy": torch.zeros(batch, POLICY_WIDTH, device=device),
        "native_whole_hand_tactile_history": torch.zeros(
            batch, TACTILE_WIDTH, device=device
        ),
        "critic": torch.zeros(batch, POLICY_WIDTH, device=device),
        "teacher": torch.zeros(batch, POLICY_WIDTH, device=device),
    }
    groups = {
        "policy": ["policy", "native_whole_hand_tactile_history"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    policy = ReferenceOnlyTactileActorCritic(
        obs=observations,
        obs_groups=groups,
        num_actions=ACTION_WIDTH,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.5,
        tactile_obs_group="native_whole_hand_tactile_history",
        tactile_grid_shape=(20, 25),
        tactile_num_hands=2,
        tactile_channels_per_hand=324,
        tactile_encoder_channels=(32, 64, 64),
        tactile_embedding_dim=128,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    source_state = checkpoint["model_state_dict"]
    warm_start = policy.load_sugar_warm_start(source_state)
    finetune = policy.configure_tactile_actor_finetune()

    policy.zero_grad(set_to_none=True)
    probe = {
        "policy": torch.linspace(
            -1.0, 1.0, steps=batch * POLICY_WIDTH, device=device
        ).reshape(batch, POLICY_WIDTH),
        "native_whole_hand_tactile_history": torch.linspace(
            -0.25, 0.25, steps=batch * TACTILE_WIDTH, device=device
        ).reshape(batch, TACTILE_WIDTH),
        "critic": observations["critic"],
        "teacher": observations["teacher"],
    }
    policy.act_inference(probe).square().mean().backward()
    first_gradient = policy.actor._modules["0"].weight.grad
    if first_gradient is None:
        raise RuntimeError("Missing actor input-layer gradient")
    base_gradient_abs_max = float(
        first_gradient[:, :POLICY_WIDTH].abs().max().item()
    )
    tactile_gradient_abs_max = float(
        first_gradient[:, POLICY_WIDTH:].abs().max().item()
    )
    encoder_gradient_l2 = float(
        sum(
            parameter.grad.detach().float().pow(2).sum()
            for parameter in policy.actor_tactile_encoder.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ).sqrt().item()
    )

    checks = {
        "official_actor_width_is_890": policy.num_actor_base_obs
        == POLICY_WIDTH,
        "official_critic_width_is_890": policy.num_critic_base_obs
        == POLICY_WIDTH,
        "raw_tactile_width_is_324000": (
            policy.actor_tactile_encoder.expected_flat_dim == TACTILE_WIDTH
        ),
        "tactile_feature_width_is_256": (
            policy.actor_tactile_encoder.output_dim == TACTILE_FEATURE_WIDTH
        ),
        "zero_tactile_embedding_is_exact_zero": (
            warm_start["zero_tactile_feature_abs_max"] == 0.0
        ),
        "zero_tactile_actor_recovers_official": (
            warm_start["actor_zero_tactile_max_abs_error"] <= 1.0e-6
        ),
        "critic_recovers_official": (
            warm_start["critic_zero_tactile_max_abs_error"] <= 1.0e-6
        ),
        "official_actor_base_gradient_is_zero": base_gradient_abs_max == 0.0,
        "tactile_actor_columns_receive_gradient": tactile_gradient_abs_max > 0.0,
        "spatial_tactile_encoder_receives_gradient": encoder_gradient_l2 > 0.0,
        "actor_normalization_disabled": not policy.actor_obs_normalization,
        "critic_normalization_disabled": not policy.critic_obs_normalization,
    }
    report = {
        "semantics": (
            "official-width structural warm-start and gradient-gate audit; "
            "not tactile usefulness or physical behavior evidence"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint.get("iter"),
        "warm_start": warm_start,
        "finetune": finetune,
        "gradient_probe": {
            "base_columns_abs_max": base_gradient_abs_max,
            "tactile_columns_abs_max": tactile_gradient_abs_max,
            "spatial_encoder_l2": encoder_gradient_l2,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, indent=2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
