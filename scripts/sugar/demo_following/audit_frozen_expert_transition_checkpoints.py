#!/usr/bin/env python3
"""Audit exact released experts and learned residuals in a transition pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--pair-root", type=Path)
group.add_argument("--shared-root", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--post-iteration", type=int, default=64)
args = parser.parse_args()


def _official_actor(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    return {
        name.removeprefix("actor."): value
        for name, value in state.items()
        if name.startswith("actor.")
    }


def main() -> None:
    pair_root = (args.pair_root or args.shared_root).expanduser().resolve()
    if (ROOT / "experiments").resolve() not in pair_root.parents:
        raise ValueError("pair root must remain under experiments/")
    official = (
        _official_actor(ROOT / "SUGAR/demo_ckpts/CarryBox/tracker.pt"),
        _official_actor(ROOT / "SUGAR/demo_ckpts/KickBox/tracker.pt"),
    )
    arms: dict[str, object] = {}
    checkpoint_roots = (
        {
            "correct_kick": pair_root / "correct_kick/train",
            "wrong_carry": pair_root / "wrong_carry/train",
        }
        if args.pair_root is not None
        else {"shared": pair_root / "train"}
    )
    for arm, checkpoint_root in checkpoint_roots.items():
        pre = torch.load(
            checkpoint_root / "model_pre_update.pt",
            map_location="cpu",
            weights_only=True,
        )["model_state_dict"]
        post = torch.load(
            checkpoint_root / f"model_{args.post_iteration}.pt",
            map_location="cpu",
            weights_only=True,
        )["model_state_dict"]
        official_error = 0.0
        for expert_index, expected in enumerate(official):
            for name, value in expected.items():
                key = f"actor.experts.{expert_index}.{name}"
                official_error = max(
                    official_error,
                    float(torch.abs(pre[key] - value).max()),
                    float(torch.abs(post[key] - value).max()),
                )
        if any(name.startswith("actor.composer.") for name in pre):
            policy_topology = "causal_action_composition"
            trainable_prefix = "actor.composer."
        elif any(name.startswith("actor.residual.") for name in pre):
            policy_topology = "selected_expert_residual"
            trainable_prefix = "actor.residual."
        else:
            raise RuntimeError(f"unknown transition topology in {checkpoint_root}")
        trainable_keys = [name for name in pre if name.startswith(trainable_prefix)]
        trainable_delta = max(
            float(torch.abs(post[name] - pre[name]).max())
            for name in trainable_keys
        )
        output_weight = f"{trainable_prefix}6.weight"
        output_bias = f"{trainable_prefix}6.bias"
        output_zero = bool(
            torch.count_nonzero(pre[output_weight]) == 0
            and torch.count_nonzero(pre[output_bias]) == 0
        )
        finite = all(
            bool(torch.isfinite(value).all())
            for value in post.values()
            if torch.is_tensor(value)
        )
        arms[arm] = {
            "official_expert_max_abs_error_pre_or_post": official_error,
            "policy_topology": policy_topology,
            "pre_update_trainable_output_layer_exact_zero": output_zero,
            "transition_trainable_max_parameter_delta": trainable_delta,
            "all_checkpoint_tensors_finite": finite,
            "pass": bool(
                official_error == 0.0
                and output_zero
                and trainable_delta > 0.0
                and finite
            ),
        }
    result = {
        "protocol": "sugar_frozen_expert_transition_checkpoint_audit_v2",
        "pair_root": str(pair_root),
        "post_iteration": args.post_iteration,
        "arms": arms,
        "overall_pass": all(bool(record["pass"]) for record in arms.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["overall_pass"]:
        raise RuntimeError("frozen-expert transition checkpoint audit failed")


if __name__ == "__main__":
    main()
