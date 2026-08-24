#!/usr/bin/env python3
"""Summarize the fixed 2x2 Carry-skill asset/motion-context factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CELLS = {
    "carry_motion_small_asset": ("CarryBox", "small"),
    "carry_motion_big_asset": ("CarryBox", "big"),
    "kick_motion_small_asset": ("KickBox", "small"),
    "kick_motion_big_asset": ("KickBox", "big"),
}
RAW_ACTION_LIMIT = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def delta(big: dict[str, object], small: dict[str, object]) -> dict[str, float]:
    return {
        "carry_success_count": float(big["carry_success_count"])
        - float(small["carry_success_count"]),
        "physical_fall_count": float(big["physical_fall_count"])
        - float(small["physical_fall_count"]),
        "mean_maximum_lift_m": float(big["mean_maximum_lift_m"])
        - float(small["mean_maximum_lift_m"]),
        "maximum_abs_raw_action": float(big["maximum_abs_raw_action"])
        - float(small["maximum_abs_raw_action"]),
    }


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    results: dict[str, dict[str, object]] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    seeds: set[int] = set()
    for name, (domain, asset) in CELLS.items():
        directory = root / name
        result = json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
        trace = load_trace(directory / "TRACE.npz")
        if (
            result.get("domain") != domain
            or result.get("resolved_scene_object_asset") != asset
            or result.get("selected_demo_option") != "correct"
            or result.get("routed_generator_skill") != "CarryBox"
            or result.get("routed_generator_with_expert") is not True
        ):
            raise RuntimeError(f"factorial contract drift: {name}")
        seeds.add(int(result["seed"]))
        aggregate = result["aggregate"]
        results[name] = {
            "domain_motion_context": domain,
            "physical_asset": asset,
            "passed": bool(result["passed"]),
            "carry_success_count": int(aggregate["carry_success_count"]),
            "kick_success_count": int(aggregate["kick_success_count"]),
            "physical_fall_count": int(aggregate["physical_fall_count"]),
            "mean_maximum_lift_m": float(aggregate["mean_maximum_lift_m"]),
            "mean_bilateral_hand_contact_fraction": float(
                aggregate["mean_bilateral_hand_contact_fraction"]
            ),
            "maximum_abs_raw_action": float(result["maximum_abs_raw_student_action"]),
            "mean_object_mass_kg": float(result["object_mass_kg"]["mean"]),
            "behaviorally_admitted": bool(
                int(aggregate["carry_success_count"]) >= 10
                and int(aggregate["physical_fall_count"]) <= 2
                and float(result["maximum_abs_raw_student_action"])
                <= RAW_ACTION_LIMIT
            ),
        }
        traces[name] = trace
    if len(seeds) != 1:
        raise RuntimeError("all factorial cells must share one startup seed")

    paired_keys = (
        "initial_robot_root_state_w",
        "initial_robot_joint_pos",
        "initial_robot_joint_vel",
        "initial_object_root_state_w",
        "prefix_action",
    )
    initial_state_checks = {}
    mass_ratio_checks = {}
    for motion in ("carry", "kick"):
        small_name = f"{motion}_motion_small_asset"
        big_name = f"{motion}_motion_big_asset"
        initial_state_checks[motion] = all(
            np.array_equal(traces[small_name][key], traces[big_name][key])
            for key in paired_keys
        )
        ratio = (
            float(results[big_name]["mean_object_mass_kg"])
            / float(results[small_name]["mean_object_mass_kg"])
        )
        mass_ratio_checks[motion] = {
            "big_over_small": ratio,
            "matches_nominal_1p5_ratio": bool(np.isclose(ratio, 1.5, atol=1.0e-5)),
        }

    carry_small = results["carry_motion_small_asset"]
    carry_big = results["carry_motion_big_asset"]
    kick_small = results["kick_motion_small_asset"]
    kick_big = results["kick_motion_big_asset"]
    admission_pattern = {
        name: bool(record["behaviorally_admitted"])
        for name, record in results.items()
    }
    if (
        not carry_big["behaviorally_admitted"]
        and not kick_small["behaviorally_admitted"]
        and kick_big["behaviorally_admitted"]
    ):
        diagnosis = "crossover_compatibility_mismatched_asset_motion_pairs_fail"
    elif carry_big["behaviorally_admitted"] and not kick_small["behaviorally_admitted"]:
        diagnosis = "kick_motion_initialization_and_goal_are_sufficient_to_break_carry"
    elif kick_small["behaviorally_admitted"] and not carry_big["behaviorally_admitted"]:
        diagnosis = "big_asset_is_sufficient_to_break_carry"
    elif not carry_big["behaviorally_admitted"] and not kick_small["behaviorally_admitted"]:
        diagnosis = "both_big_asset_and_kick_motion_context_independently_break_carry"
    elif not kick_big["behaviorally_admitted"]:
        diagnosis = "big_asset_and_kick_motion_context_fail_only_in_combination"
    else:
        diagnosis = "carry_skill_transfers_across_the_full_factorial"

    checks = {
        "all_four_cells_present": len(results) == 4,
        "one_shared_startup_seed": len(seeds) == 1,
        "asset_swap_preserves_exact_initial_state_within_carry_motion": initial_state_checks["carry"],
        "asset_swap_preserves_exact_initial_state_within_kick_motion": initial_state_checks["kick"],
        "mass_ratio_matches_asset_configs_in_both_motion_contexts": all(
            record["matches_nominal_1p5_ratio"]
            for record in mass_ratio_checks.values()
        ),
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
        "protocol": "sugar_carry_skill_asset_motion_factorial_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "seed": next(iter(seeds)),
        "fixed_selected_skill": "CarryBox45 released Generator+Tracker pair",
        "cells": results,
        "admission_pattern": admission_pattern,
        "diagnosis": diagnosis,
        "effects": {
            "big_minus_small_with_carry_motion": delta(carry_big, carry_small),
            "big_minus_small_with_kick_motion": delta(kick_big, kick_small),
            "kick_minus_carry_motion_with_small_asset": delta(kick_small, carry_small),
            "kick_minus_carry_motion_with_big_asset": delta(kick_big, carry_big),
        },
        "initial_state_exact_match": initial_state_checks,
        "mass_ratio": mass_ratio_checks,
        "claim_boundary": (
            "One fixed seed and 20 physics profiles per cell isolate the physical asset "
            "from the domain motion initialization/goal context. This diagnoses a causal "
            "failure source; it is not a learned safe-transition result."
        ),
    }
    (root / "FACTORIAL_RESULT.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise RuntimeError("Carry-skill factorial structural proof failed")


if __name__ == "__main__":
    main()
