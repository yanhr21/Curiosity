#!/usr/bin/env python3
"""Select the first finite, upright but unsuccessful Carry-to-Kick prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREDECLARED_PREFIX_STEPS = (9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97)
MINIMUM_UPRIGHT_ROOT_HEIGHT_M = 0.65
ADMITTED_SAFE_KICK_SUCCESSES = 10

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def main() -> None:
    records: list[dict[str, object]] = []
    for prefix_steps in PREDECLARED_PREFIX_STEPS:
        path = args.input_root / f"prefix_{prefix_steps:03d}" / "RESULT.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing predeclared frontier result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("protocol") != "sugar_cross_skill_recovery_frozen_eval_v2":
            raise RuntimeError(f"frontier protocol drift: {path}")
        if int(result["prefix"]["carry_steps"]) != prefix_steps:
            raise RuntimeError(f"frontier prefix mismatch: {path}")
        aggregate = result["aggregate"]
        handoff = result["handoff"]
        finite = bool(
            result.get("structurally_valid")
            and handoff["all_initial_values_finite"]
        )
        upright = bool(
            float(handoff["minimum_robot_root_height_m"])
            >= MINIMUM_UPRIGHT_ROOT_HEIGHT_M
        )
        safe_successes = int(aggregate["safe_kick_success_count"])
        records.append(
            {
                "carry_prefix_steps": prefix_steps,
                "finite": finite,
                "upright_handoff": upright,
                "minimum_robot_root_height_m": float(
                    handoff["minimum_robot_root_height_m"]
                ),
                "maximum_abs_policy_observation": float(
                    handoff["maximum_abs_policy_observation"]
                ),
                "kick_success_count": int(aggregate["kick_success_count"]),
                "safe_kick_success_count": safe_successes,
                "physical_fall_count": int(aggregate["physical_fall_count"]),
                "mean_planar_object_net_displacement_m": float(
                    aggregate["mean_planar_object_net_displacement_m"]
                ),
                "frontier_candidate": bool(
                    finite
                    and upright
                    and safe_successes < ADMITTED_SAFE_KICK_SUCCESSES
                ),
            }
        )

    selected = next(
        (record for record in records if record["frontier_candidate"]), None
    )
    upright_failures = [
        record
        for record in records
        if record["finite"]
        and record["upright_handoff"]
        and int(record["physical_fall_count"]) > 0
    ]
    recovery_boundary = (
        sorted(
            upright_failures,
            key=lambda record: (
                -int(record["physical_fall_count"]),
                int(record["carry_prefix_steps"]),
            ),
        )[0]
        if upright_failures
        else None
    )
    result = {
        "protocol": "sugar_cross_skill_prefix_frontier_v1",
        "predeclared_prefix_steps": list(PREDECLARED_PREFIX_STEPS),
        "selection_rule": {
            "all_trace_and_handoff_values_finite": True,
            "minimum_handoff_robot_root_height_m": MINIMUM_UPRIGHT_ROOT_HEIGHT_M,
            "safe_kick_success_count_strictly_below": ADMITTED_SAFE_KICK_SUCCESSES,
            "safe_kick_success_definition": (
                "at least 0.05 m planar object displacement, foot-to-box contact, "
                "and no >=0.35 m root-height loss"
            ),
        },
        "records": records,
        "selected_frontier": selected,
        "frontier_found": selected is not None,
        "selected_recovery_boundary": recovery_boundary,
        "recovery_boundary_found": recovery_boundary is not None,
        "recovery_boundary_rule": (
            "When the strict majority-failure frontier is absent, select the finite upright "
            "handoff with the largest physical-fall count; break ties by shortest prefix. "
            "This boundary is reported separately and is not relabeled as the strict frontier."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if selected is None and recovery_boundary is None:
        raise RuntimeError("prefix sweep found neither a strict frontier nor an upright failure boundary")


if __name__ == "__main__":
    main()
