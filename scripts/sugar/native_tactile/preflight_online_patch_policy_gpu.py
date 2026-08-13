#!/usr/bin/env python3
"""Exercise the serious Plan-15 policy and official Tracker warm start on GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sugar_rl.utils.online_patch_tactile_actor_critic import (
    OnlinePatchTactileActorCritic,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tracker", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument(
    "--iterations",
    type=int,
    default=1,
    help="Repeated forward/backward steps for CUDA-path stability measurement.",
)
parser.add_argument("--log-every", type=int, default=100)
args = parser.parse_args()


def main() -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("iterations and log-every must be positive")
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
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    encoder_gradient_l2_min = float("inf")
    encoder_gradient_l2_max = 0.0
    for iteration in range(1, args.iterations + 1):
        model.zero_grad(set_to_none=True)
        tactile[..., 1:7].uniform_(0.0, 1.0)
        tactile[..., 7:9].copy_(
            (torch.rand_like(tactile[..., 7:9]) > 0.8).to(torch.float32)
        )
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
            raise RuntimeError(f"non-finite policy output at iteration {iteration}")
        if not torch.isfinite(loss) or encoder_gradient_l2 <= 0.0:
            raise RuntimeError(
                f"invalid patch Transformer gradient at iteration {iteration}"
            )
        encoder_gradient_l2_min = min(
            encoder_gradient_l2_min, encoder_gradient_l2
        )
        encoder_gradient_l2_max = max(
            encoder_gradient_l2_max, encoder_gradient_l2
        )
        if iteration % args.log_every == 0 or iteration == args.iterations:
            print(
                f"iteration={iteration}/{args.iterations} "
                f"encoder_gradient_l2={encoder_gradient_l2:.8f}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started

    result = {
        "schema": "plan15_online_patch_policy_gpu_preflight_v1",
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "elapsed_s": elapsed_s,
        "samples_per_second": args.batch_size * args.iterations / elapsed_s,
        "actor_output_shape": list(action.shape),
        "critic_output_shape": list(value.shape),
        "encoder_gradient_l2": encoder_gradient_l2,
        "encoder_gradient_l2_min": encoder_gradient_l2_min,
        "encoder_gradient_l2_max": encoder_gradient_l2_max,
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
