#!/usr/bin/env python3
"""Exercise the serious Plan-15 policy and official Tracker warm start on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sugar_rl.utils.online_patch_tactile_actor_critic import (
    OnlinePatchTactileActorCritic,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tracker", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=256)
args = parser.parse_args()


def main() -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this preflight")
    device = torch.device("cuda:0")
    observations = {
        "policy": torch.zeros(args.batch_size, 504, device=device),
        "online_patch_tactile_history": torch.zeros(
            args.batch_size, 1944, device=device
        ),
        "critic": torch.zeros(args.batch_size, 890, device=device),
        "teacher": torch.zeros(args.batch_size, 890, device=device),
    }
    groups = {
        "policy": ["policy", "online_patch_tactile_history"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    # These unit scales are used only to exercise the architecture. Physical
    # normalization remains deliberately unfrozen until live mass traces exist.
    model = OnlinePatchTactileActorCritic(
        observations,
        groups,
        29,
        patch_channel_scales=[1.0] * 9,
        init_noise_std=0.5,
    ).to(device)
    checkpoint = torch.load(
        args.tracker.expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    warm_start = model.load_sugar_warm_start(checkpoint["model_state_dict"])
    finetune = model.configure_tactile_actor_finetune()

    tactile = observations["online_patch_tactile_history"].reshape(
        args.batch_size, 4, 2, 27, 9
    )
    tactile[..., 0] = 1.0
    tactile[..., 1:6] = torch.rand_like(tactile[..., 1:6])
    tactile[..., 6] = torch.rand_like(tactile[..., 6])
    tactile[..., 7:9] = (
        torch.rand_like(tactile[..., 7:9]) > 0.8
    ).to(torch.float32)
    action = model.act_inference(observations)
    value = model.evaluate(observations)
    loss = action.square().mean() + value.square().mean()
    loss.backward()
    encoder_gradient_l2 = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in model.actor_tactile_encoder.parameters()
                if parameter.grad is not None
            )
        ).item()
    )
    if not torch.isfinite(action).all() or not torch.isfinite(value).all():
        raise RuntimeError("non-finite policy output")
    if encoder_gradient_l2 <= 0.0:
        raise RuntimeError("patch Transformer received no actor gradient")

    result = {
        "schema": "plan15_online_patch_policy_gpu_preflight_v1",
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "actor_output_shape": list(action.shape),
        "critic_output_shape": list(value.shape),
        "encoder_gradient_l2": encoder_gradient_l2,
        "warm_start": warm_start,
        "finetune": finetune,
        "normalization_semantics": (
            "structural GPU preflight only; physical channel scales remain unfrozen"
        ),
        "proves_tactile_benefit": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
