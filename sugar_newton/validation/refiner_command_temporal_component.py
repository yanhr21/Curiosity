# SPDX-License-Identifier: BSD-3-Clause
"""Audit the serious command-conditioned Newton Refiner composer on CUDA.

This is a component gate, not a physical evaluation.  It proves that the exact
released Refiner remains the initialized deployed action, that the current 36-D
command changes the six-layer Transformer's hidden representation, and that a
training loss reaches only the new composer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sugar_newton.rl.train_bcppo import activate_rsl_rl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFINER = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts"
    / "refiner_model10000.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-refiner-checkpoint", type=Path, default=DEFAULT_REFINER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=171726)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    args = parser.parse_args()

    if not args.official_refiner_checkpoint.is_file():
        raise SystemExit(
            f"official Refiner checkpoint not found: {args.official_refiner_checkpoint}"
        )
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("command-conditioned Refiner audit requires a Slurm CUDA node")

    activate_rsl_rl(args.rsl_rl_root)
    sugar_rl_source = ROOT / "SUGAR/source/sugar_rl"
    if str(sugar_rl_source) not in sys.path:
        sys.path.insert(0, str(sugar_rl_source))
    from sugar_rl.utils.frozen_expert_transition_actor_critic import (
        ACTION_DIM,
        GENERATED_COMMAND_DIM,
        REFINER_OBSERVATION_DIM,
        TEMPORAL_HISTORY_STEPS,
        FrozenOfficialRefinerCausalTemporalComposer,
    )

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    actor = FrozenOfficialRefinerCausalTemporalComposer(
        args.official_refiner_checkpoint,
        additive_residual=True,
        command_conditioned=True,
    ).to(device)

    current = torch.randn(
        2, REFINER_OBSERVATION_DIM, generator=generator, device=device
    )
    history = torch.randn(
        2,
        TEMPORAL_HISTORY_STEPS,
        REFINER_OBSERVATION_DIM,
        generator=generator,
        device=device,
    )
    history[:, -1] = current
    command_a = torch.randn(
        2, GENERATED_COMMAND_DIM, generator=generator, device=device
    )
    command_b = command_a.clone()
    command_b[:, -7:] += 0.25

    def actor_input(command: torch.Tensor) -> torch.Tensor:
        return torch.cat((current, history.flatten(1), command), dim=-1)

    terms_a = actor.composition_terms(actor_input(command_a))
    endpoint_delta = float(
        (terms_a["composed_action"] - terms_a["selected_endpoint_action"])
        .abs()
        .max()
        .item()
    )
    residual_max = float(terms_a["bounded_residual_action"].abs().max().item())
    retention_delta = float((terms_a["expert_retention"] - 1.0).abs().max().item())
    latent_a = actor.temporal_composer.encode(history, command_a)
    latent_b = actor.temporal_composer.encode(history, command_b)
    command_latent_delta = float((latent_a - latent_b).abs().max().item())

    final = actor.temporal_composer.output[-1]
    output_head_exact_zero = bool(
        float(final.weight.abs().max().item()) == 0.0
        and float(final.bias.abs().max().item()) == 0.0
    )
    target = torch.randn(2, ACTION_DIM, generator=generator, device=device)
    loss = torch.nn.functional.mse_loss(terms_a["composed_action"], target)
    loss.backward()
    expert_grad_max = max(
        (float(parameter.grad.abs().max().item()) if parameter.grad is not None else 0.0)
        for parameter in actor.expert.parameters()
    )
    composer_grad_max = max(
        (float(parameter.grad.abs().max().item()) if parameter.grad is not None else 0.0)
        for parameter in actor.temporal_composer.parameters()
    )

    checks = {
        "cuda_execution": current.is_cuda,
        "output_dim_is_29": actor.temporal_composer.output_dim == ACTION_DIM,
        "command_dim_is_36": (
            actor.temporal_composer.condition_dim == GENERATED_COMMAND_DIM == 36
        ),
        "token_count_is_12": actor.temporal_composer.position_embedding.shape[1] == 12,
        "output_head_exact_zero": output_head_exact_zero,
        "initialized_action_is_exact_refiner": endpoint_delta == 0.0,
        "initialized_residual_is_exact_zero": residual_max == 0.0,
        "initialized_retention_is_exact_one": retention_delta == 0.0,
        "command_changes_hidden_representation": command_latent_delta > 0.0,
        "official_refiner_gradient_is_zero": expert_grad_max == 0.0,
        "composer_gradient_is_nonzero": composer_grad_max > 0.0,
    }
    result = {
        "protocol": "sugar_newton_refiner_command_temporal_component_v1",
        "seed": args.seed,
        "official_refiner_checkpoint": str(args.official_refiner_checkpoint.resolve()),
        "output_dim": actor.temporal_composer.output_dim,
        "command_dim": actor.temporal_composer.condition_dim,
        "token_count": int(actor.temporal_composer.position_embedding.shape[1]),
        "initialized_endpoint_max_delta": endpoint_delta,
        "initialized_residual_max": residual_max,
        "initialized_retention_delta_from_one": retention_delta,
        "command_hidden_max_delta": command_latent_delta,
        "official_refiner_gradient_max": expert_grad_max,
        "composer_gradient_max": composer_grad_max,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
