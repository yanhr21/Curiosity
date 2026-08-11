#!/usr/bin/env python3
"""Summarize one matched frozen tactile-versus-zero evaluation pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "completed_steps",
    "cumulative_reward",
    "student_teacher_action_mae",
    "student_teacher_action_abs_max",
    "maximum_relative_lift_m",
    "final_relative_lift_m",
    "bilateral_physical_tactile_frames",
    "maximum_active_taxels_left",
    "maximum_active_taxels_right",
    "final_object_position_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tactile", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    missing = [key for key in METRICS if key not in result]
    if missing:
        raise ValueError(f"{path} is missing metrics: {missing}")
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    tactile = load(args.tactile)
    zero = load(args.zero)
    tactile_arm = tactile.get("arm")
    zero_arm = zero.get("arm")
    expected_zero_arm = {
        "tactile": "zero",
        "bounded_tactile": "bounded_zero",
        "residual_tactile": "residual_zero",
    }.get(tactile_arm)
    checks = {
        "declared_arms_form_matched_tactile_zero_pair": expected_zero_arm is not None
        and zero_arm == expected_zero_arm,
        "seed_equal": tactile.get("seed") == zero.get("seed"),
        "physical_condition_equal": tactile.get("physical_condition")
        == zero.get("physical_condition"),
        "reference_equal": tactile.get("reference") == zero.get("reference"),
        "disabled_events_equal": tactile.get("disabled_events")
        == zero.get("disabled_events"),
    }
    if not all(checks.values()):
        raise ValueError(f"pair is not matched: {checks}")

    tactile_metrics = {key: tactile[key] for key in METRICS}
    zero_metrics = {key: zero[key] for key in METRICS}
    result = {
        "schema": "native_whole_hand_tactile_frozen_pair_summary_v1",
        "tactile_result": str(args.tactile.resolve()),
        "zero_result": str(args.zero.resolve()),
        "checks": checks,
        "condition": tactile["physical_condition"],
        "seed": tactile["seed"],
        "reference": tactile["reference"],
        "tactile": tactile_metrics,
        "zero": zero_metrics,
        "tactile_minus_zero": {
            key: tactile_metrics[key] - zero_metrics[key] for key in METRICS
        },
        "termination_terms": {
            "tactile": tactile.get("termination_terms"),
            "zero": zero.get("termination_terms"),
        },
        "interpretation_boundary": (
            "A single matched rollout can establish a physical behavior difference, "
            "but not cross-seed tactile usefulness or generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
