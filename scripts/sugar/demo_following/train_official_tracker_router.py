#!/usr/bin/env python3
"""Train a causal demo router over the two released SUGAR Tracker experts.

The CarryBox and KickBox Tracker actors are copied exactly from the released
checkpoints and remain frozen.  A small adapter reads only the frozen 798-D
causal selected-demo condition and chooses one expert.  This is an executable
skill-routing baseline, not a claim of arbitrary-demo imitation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_shared_full_tracker import (
    OFFICIAL_TRACKERS,
    TRACKER_POLICY_DIM,
    build_dataset,
    load_sequences,
)
from train_shared_topology_distillation import (
    ACTIONABLE_DEMO_CONDITIONING_DIM,
    ACTION_DIM,
    ROOT,
    TASKS,
    TRAINING_FRAMES,
)


DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/official_tracker_router_v1/seed161610/step_1000"
)


def _expert() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(TRACKER_POLICY_DIM, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, ACTION_DIM),
    )


class DemoConditionedOfficialTrackerRouter(nn.Module):
    """One checkpoint containing exact frozen experts and a causal router."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList((_expert(), _expert()))
        self.router = nn.Sequential(
            nn.LayerNorm(ACTIONABLE_DEMO_CONDITIONING_DIM),
            nn.Linear(ACTIONABLE_DEMO_CONDITIONING_DIM, 256),
            nn.ELU(),
            nn.Linear(256, len(TASKS)),
        )

    def routing_logits(self, condition: torch.Tensor) -> torch.Tensor:
        return self.router(condition)

    def routing_weights(self, condition: torch.Tensor) -> torch.Tensor:
        logits = self.routing_logits(condition)
        index = torch.argmax(logits, dim=-1)
        return torch.nn.functional.one_hot(index, len(TASKS)).to(logits.dtype)

    def act_inference(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        state = observation["policy"]
        condition = observation["demo_conditioning"]
        actions = torch.stack(tuple(expert(state) for expert in self.experts), dim=1)
        weights = self.routing_weights(condition)
        return torch.sum(actions * weights.unsqueeze(-1), dim=1)


def _load_exact_experts(policy: DemoConditionedOfficialTrackerRouter) -> None:
    for expert_index, task in enumerate(TASKS):
        source = torch.load(
            OFFICIAL_TRACKERS[task], map_location="cpu", weights_only=True
        )["model_state_dict"]
        state = {
            name.removeprefix("actor."): value
            for name, value in source.items()
            if name.startswith("actor.")
        }
        policy.experts[expert_index].load_state_dict(state, strict=True)
        policy.experts[expert_index].requires_grad_(False)


def construct_router_policy(device: torch.device) -> DemoConditionedOfficialTrackerRouter:
    policy = DemoConditionedOfficialTrackerRouter()
    _load_exact_experts(policy)
    return policy.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=161610)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.no_grad()
def _evaluate(
    policy: DemoConditionedOfficialTrackerRouter,
    condition: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, object]:
    logits = policy.routing_logits(condition)
    prediction = torch.argmax(logits, dim=-1)
    margin = logits.gather(1, target[:, None]).squeeze(1) - logits.gather(
        1, (1 - target)[:, None]
    ).squeeze(1)
    groups = {}
    for index, name in enumerate(("Carry45", "Kick21")):
        mask = target == index
        groups[name] = {
            "samples": int(mask.sum()),
            "accuracy": float((prediction[mask] == target[mask]).float().mean()),
            "mean_correct_logit_margin": float(margin[mask].mean()),
        }
    return {
        "accuracy": float((prediction == target).float().mean()),
        "minimum_correct_logit_margin": float(margin.min()),
        "mean_correct_logit_margin": float(margin.mean()),
        "groups": groups,
        "all_finite": bool(torch.isfinite(logits).all()),
    }


def main() -> None:
    args = parse_args()
    if args.steps != 1000 or args.batch_size != 512:
        raise ValueError("router training is fixed to 1000 steps and batch 512")
    output = args.output_dir.expanduser().resolve()
    if (ROOT / "experiments").resolve() not in output.parents:
        raise ValueError("output must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    if args.device.startswith("cuda") and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA fitting requires a retained compute allocation")
    for path in OFFICIAL_TRACKERS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset, condition_audits = build_dataset(load_sequences(), device)
    condition = dataset["demo_conditioning"]
    target = dataset["selected_option"]
    row_in_group = torch.arange(condition.shape[0], device=device) % TRAINING_FRAMES
    train_mask = row_in_group < 500
    validation_mask = ~train_mask
    policy = construct_router_policy(device)
    expert_before = {
        name: value.detach().clone()
        for name, value in policy.state_dict().items()
        if name.startswith("experts.")
    }
    before = _evaluate(policy, condition[validation_mask], target[validation_mask])
    optimizer = torch.optim.Adam(policy.router.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    train_indices = torch.nonzero(train_mask, as_tuple=False).squeeze(1)
    losses: list[float] = []
    policy.train()
    for step in range(args.steps):
        sample = torch.randint(
            train_indices.shape[0], (args.batch_size,), generator=generator
        ).to(device)
        index = train_indices[sample]
        logits = policy.routing_logits(condition[index])
        loss = torch.nn.functional.cross_entropy(logits, target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.router.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 200 == 0:
            print(
                f"ROUTER_STEP={step + 1}/{args.steps} loss={losses[-1]:.8f}",
                flush=True,
            )
    policy.eval()
    train_result = _evaluate(policy, condition[train_mask], target[train_mask])
    validation_result = _evaluate(
        policy, condition[validation_mask], target[validation_mask]
    )
    expert_delta = max(
        float(torch.abs(policy.state_dict()[name] - value).max())
        for name, value in expert_before.items()
    )
    checks = {
        "official_carry_and_kick_expert_weights_loaded": True,
        "official_experts_frozen_and_exact": expert_delta == 0.0,
        "router_reads_only_causal_demo_condition": True,
        "future_actions_absent_from_router_input": True,
        "frozen_predictor_conditioning_is_causal": all(
            audit["model_frozen"] and audit["future_actual_events_used"] is False
            for audit in condition_audits.values()
        ),
        "temporal_validation_accuracy_at_least_99_percent": (
            validation_result["accuracy"] >= 0.99
        ),
        "both_demo_routes_validate_at_least_99_percent": all(
            record["accuracy"] >= 0.99
            for record in validation_result["groups"].values()
        ),
        "positive_validation_margin": (
            validation_result["minimum_correct_logit_margin"] > 0.0
        ),
        "all_values_finite": bool(
            np.isfinite(losses).all()
            and train_result["all_finite"]
            and validation_result["all_finite"]
        ),
        "exactly_1000_optimizer_steps": len(losses) == 1000,
    }
    result = {
        "protocol": "sugar_demo_conditioned_official_tracker_router_fit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "seed": args.seed,
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "state_observation_contract": "exact_official_510D_Tracker_policy_observation",
        "router_input_contract": "frozen_798D_causal_selected_demo_condition_only",
        "expert_order": ["CarryBox", "KickBox"],
        "expert_checkpoints": {task: str(path) for task, path in OFFICIAL_TRACKERS.items()},
        "temporal_split": {
            "train_frames_per_sequence": 500,
            "validation_frames_per_sequence": TRAINING_FRAMES - 500,
        },
        "before_validation": before,
        "after_train": train_result,
        "after_validation": validation_result,
        "optimization": {
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
            "expert_parameter_max_abs_delta": expert_delta,
        },
        "claim_boundary": (
            "This is a causal selected-demo skill router over two released experts. "
            "It is not arbitrary-demo imitation or a learned replacement for either expert."
        ),
        "checkpoint": "policy.pt",
    }
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "protocol": "sugar_demo_conditioned_official_tracker_router_checkpoint_v1",
            "iteration": args.steps,
            "policy_state_dict": policy.state_dict(),
            "expert_order": ["CarryBox", "KickBox"],
        },
        output / "policy.pt",
    )
    (output / "proof.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": result["passed"],
                "validation": validation_result,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise RuntimeError("official Tracker router fit failed")


if __name__ == "__main__":
    main()
