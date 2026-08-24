#!/usr/bin/env python3
"""Summarize the Carry45 initialization-context by target-goal factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CELLS = {
    "carry_context_carry_goal": ("CarryBox", "small", "carry45"),
    "carry_context_kick_goal": ("CarryBox", "small", "kick21"),
    "kick_context_carry_goal": ("KickBox", "big", "carry45"),
    "kick_context_kick_goal": ("KickBox", "big", "kick21"),
}
RAW_ACTION_LIMIT = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    root = parse_args().input_root.expanduser().resolve()
    results: dict[str, dict[str, object]] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    seeds: set[int] = set()
    for name, (domain, asset, goal) in CELLS.items():
        result = json.loads((root / name / "RESULT.json").read_text(encoding="utf-8"))
        with np.load(root / name / "TRACE.npz", allow_pickle=False) as archive:
            trace = {key: np.asarray(archive[key]) for key in archive.files}
        if (
            result.get("domain") != domain
            or result.get("resolved_scene_object_asset") != asset
            or result.get("target_goal_source") != goal
            or result.get("routed_generator_skill") != "CarryBox"
            or result.get("selected_demo_option") != "correct"
        ):
            raise RuntimeError(f"context/goal contract drift: {name}")
        aggregate = result["aggregate"]
        seeds.add(int(result["seed"]))
        results[name] = {
            "initialization_context": domain,
            "matched_physical_asset": asset,
            "target_goal": goal,
            "carry_success_count": int(aggregate["carry_success_count"]),
            "physical_fall_count": int(aggregate["physical_fall_count"]),
            "mean_maximum_lift_m": float(aggregate["mean_maximum_lift_m"]),
            "mean_bilateral_hand_contact_fraction": float(
                aggregate["mean_bilateral_hand_contact_fraction"]
            ),
            "maximum_abs_raw_action": float(result["maximum_abs_raw_student_action"]),
            "behaviorally_admitted": bool(
                int(aggregate["carry_success_count"]) >= 10
                and int(aggregate["physical_fall_count"]) <= 2
                and float(result["maximum_abs_raw_student_action"])
                <= RAW_ACTION_LIMIT
            ),
        }
        traces[name] = trace
    if len(seeds) != 1:
        raise RuntimeError("context/goal cells do not share one seed")

    initial_keys = (
        "initial_robot_root_state_w",
        "initial_robot_joint_pos",
        "initial_robot_joint_vel",
        "initial_object_root_state_w",
        "initial_object_mass_kg",
    )
    initial_exact = {}
    for context in ("carry", "kick"):
        carry_goal = traces[f"{context}_context_carry_goal"]
        kick_goal = traces[f"{context}_context_kick_goal"]
        initial_exact[context] = all(
            np.array_equal(carry_goal[key], kick_goal[key]) for key in initial_keys
        )
    goal_checks = {
        "carry_and_kick_target_positions_are_distinct": not np.array_equal(
            traces["carry_context_carry_goal"]["initial_target_object_position_w"],
            traces["carry_context_kick_goal"]["initial_target_object_position_w"],
        ),
        "carry_goal_is_exact_across_initialization_contexts": np.array_equal(
            traces["carry_context_carry_goal"]["initial_target_object_position_w"],
            traces["kick_context_carry_goal"]["initial_target_object_position_w"],
        ),
        "kick_goal_is_exact_across_initialization_contexts": np.array_equal(
            traces["carry_context_kick_goal"]["initial_target_object_position_w"],
            traces["kick_context_kick_goal"]["initial_target_object_position_w"],
        ),
    }
    admitted = {
        name: bool(record["behaviorally_admitted"])
        for name, record in results.items()
    }
    carry_context_sensitive = (
        admitted["carry_context_carry_goal"]
        != admitted["carry_context_kick_goal"]
    )
    kick_context_sensitive = (
        admitted["kick_context_carry_goal"]
        != admitted["kick_context_kick_goal"]
    )
    if not carry_context_sensitive and not kick_context_sensitive:
        diagnosis = "target_goal_is_not_sufficient_initialization_geometry_context_dominates"
    elif carry_context_sensitive and kick_context_sensitive:
        diagnosis = "target_goal_changes_admission_in_both_initialization_contexts"
    else:
        diagnosis = "target_goal_effect_depends_on_initialization_geometry_context"

    checks = {
        "all_four_cells_present": len(results) == 4,
        "goal_swap_preserves_exact_carry_initial_state": initial_exact["carry"],
        "goal_swap_preserves_exact_kick_initial_state": initial_exact["kick"],
        **goal_checks,
        "all_trace_values_finite": all(
            all(
                np.isfinite(value).all()
                for value in trace.values()
                if value.dtype.kind in "f"
            )
            for trace in traces.values()
        ),
    }
    output = {
        "protocol": "sugar_carry_skill_context_goal_factorial_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "seed": next(iter(seeds)),
        "cells": results,
        "admission_pattern": admitted,
        "diagnosis": diagnosis,
        "goal_effect_on_admission": {
            "carry_context_sensitive": carry_context_sensitive,
            "kick_context_sensitive": kick_context_sensitive,
        },
        "claim_boundary": (
            "Within each context the robot/object initial state, physical asset, mass, "
            "seed and selected Carry45 Generator+Tracker are fixed; only the target pose "
            "installed before the common prefix changes. This separates target sensitivity "
            "from initialization/geometry compatibility."
        ),
    }
    (root / "FACTORIAL_RESULT.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise RuntimeError("context/goal structural proof failed")


if __name__ == "__main__":
    main()
