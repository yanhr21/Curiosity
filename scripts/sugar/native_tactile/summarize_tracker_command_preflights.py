#!/usr/bin/env python3
"""Summarize the matched live/zero Tracker-command one-update preflights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_report(root: Path) -> dict:
    return json.loads((root / "training_signal_update0.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tactile", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tactile = load_report(args.tactile)
    zero = load_report(args.zero)
    tactile_state = torch.load(
        args.tactile / "model_prelearn.pt", map_location="cpu", weights_only=False
    )["model_state_dict"]
    zero_state = torch.load(
        args.zero / "model_prelearn.pt", map_location="cpu", weights_only=False
    )["model_state_dict"]

    same_keys = set(tactile_state) == set(zero_state)
    differing = []
    if same_keys:
        differing = [
            name
            for name in tactile_state
            if not torch.equal(tactile_state[name], zero_state[name])
        ]

    expected_shapes = {
        "policy": [288, 1, 504],
        "native_whole_hand_tactile_history": [288, 1, 324000],
        "critic": [288, 1, 890],
        "teacher": [288, 1, 890],
    }
    tactile_gradient = max(
        entry["maximum_l2"] for entry in tactile["encoder_gradients"].values()
    )
    tactile_change = max(
        entry["delta_l2"]
        for entry in tactile["encoder_parameter_change"].values()
    )
    zero_gradient = max(
        entry["maximum_l2"] for entry in zero["encoder_gradients"].values()
    )
    zero_change = max(
        entry["delta_l2"] for entry in zero["encoder_parameter_change"].values()
    )
    checks = {
        "matched_prelearning_policy": same_keys and not differing,
        "tactile_runtime_shapes": tactile["observation_shapes"] == expected_shapes,
        "zero_runtime_shapes": zero["observation_shapes"] == expected_shapes,
        "actor_base_is_504d": (
            tactile["actor_contract"]["base_observation_width"] == 504
            and zero["actor_contract"]["base_observation_width"] == 504
        ),
        "raw_tactile_is_324000d": (
            tactile["actor_contract"]["raw_tactile_width"] == 324000
            and zero["actor_contract"]["raw_tactile_width"] == 324000
        ),
        "live_rollout_contains_native_tactile": (
            tactile["rollout"]["frames_with_any_signal"] > 0
            and tactile["rollout"]["abs_max"] > 0.0
        ),
        "zero_rollout_is_exact_zero": (
            zero["rollout"]["frames_with_any_signal"] == 0
            and zero["rollout"]["abs_max"] == 0.0
        ),
        "zero_input_does_not_optimize_tactile_encoder": (
            zero_gradient == 0.0 and zero_change == 0.0
        ),
        "live_tactile_reaches_encoder_optimization": (
            tactile_gradient > 0.0 and tactile_change > 0.0
        ),
    }
    result = {
        "schema": "tracker_command_native_tactile_preflight_pair_v1",
        "semantics": (
            "matched deployable 35-D Tracker command plus official five-frame "
            "proprioception/action history and phase; one full-trajectory "
            "BCPPO update per arm"
        ),
        "checks": checks,
        "overall_pass": all(checks.values()),
        "prelearning_differing_tensors": differing,
        "tactile": {
            "frames_with_signal": tactile["rollout"]["frames_with_any_signal"],
            "frame_count": tactile["rollout"]["frame_count"],
            "abs_max": tactile["rollout"]["abs_max"],
            "maximum_encoder_gradient_l2": tactile_gradient,
            "maximum_encoder_parameter_delta_l2": tactile_change,
            "losses": tactile["losses"],
        },
        "zero": {
            "frames_with_signal": zero["rollout"]["frames_with_any_signal"],
            "frame_count": zero["rollout"]["frame_count"],
            "abs_max": zero["rollout"]["abs_max"],
            "maximum_encoder_gradient_l2": zero_gradient,
            "maximum_encoder_parameter_delta_l2": zero_change,
            "losses": zero["losses"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
