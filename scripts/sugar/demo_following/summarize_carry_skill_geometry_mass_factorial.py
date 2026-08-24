#!/usr/bin/env python3
"""Summarize the Carry45 physical geometry-by-mass factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CELLS = {
    "small_geometry_small_mass": ("small", "small"),
    "small_geometry_big_mass": ("small", "big"),
    "big_geometry_small_mass": ("big", "small"),
    "big_geometry_big_mass": ("big", "big"),
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
    for name, (geometry, mass_source) in CELLS.items():
        result = json.loads((root / name / "RESULT.json").read_text(encoding="utf-8"))
        with np.load(root / name / "TRACE.npz", allow_pickle=False) as archive:
            trace = {key: np.asarray(archive[key]) for key in archive.files}
        if (
            result.get("domain") != "CarryBox"
            or result.get("resolved_scene_object_asset") != geometry
            or result.get("object_nominal_mass_source") != mass_source
            or result.get("routed_generator_skill") != "CarryBox"
            or result.get("selected_demo_option") != "correct"
        ):
            raise RuntimeError(f"geometry/mass contract drift: {name}")
        aggregate = result["aggregate"]
        seeds.add(int(result["seed"]))
        results[name] = {
            "geometry": geometry,
            "nominal_mass_source": mass_source,
            "mean_mass_kg": float(result["object_mass_kg"]["mean"]),
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
        raise RuntimeError("geometry/mass cells do not share one seed")

    paired_keys = (
        "initial_robot_root_state_w",
        "initial_robot_joint_pos",
        "initial_robot_joint_vel",
        "initial_object_root_state_w",
        "prefix_action",
    )
    baseline = traces["small_geometry_small_mass"]
    exact_initial = {
        name: all(np.array_equal(baseline[key], trace[key]) for key in paired_keys)
        for name, trace in traces.items()
    }
    small_small_mass = float(results["small_geometry_small_mass"]["mean_mass_kg"])
    big_small_mass = float(results["big_geometry_small_mass"]["mean_mass_kg"])
    small_big_mass = float(results["small_geometry_big_mass"]["mean_mass_kg"])
    big_big_mass = float(results["big_geometry_big_mass"]["mean_mass_kg"])
    mass_checks = {
        "small_mass_equal_across_geometry": bool(
            np.isclose(small_small_mass, big_small_mass, atol=1.0e-6)
        ),
        "big_mass_equal_across_geometry": bool(
            np.isclose(small_big_mass, big_big_mass, atol=1.0e-6)
        ),
        "big_over_small_mass_ratio_is_1p5": bool(
            np.isclose(small_big_mass / small_small_mass, 1.5, atol=1.0e-5)
        ),
    }
    admitted = {
        name: bool(record["behaviorally_admitted"])
        for name, record in results.items()
    }
    if not admitted["small_geometry_big_mass"] and not admitted["big_geometry_small_mass"]:
        diagnosis = "big_mass_and_big_geometry_each_independently_break_carry"
    elif not admitted["small_geometry_big_mass"]:
        diagnosis = "big_mass_is_sufficient_to_break_carry"
    elif not admitted["big_geometry_small_mass"]:
        diagnosis = "big_geometry_is_sufficient_to_break_carry"
    elif not admitted["big_geometry_big_mass"]:
        diagnosis = "big_geometry_and_big_mass_fail_only_in_combination"
    else:
        diagnosis = "carry_skill_transfers_across_geometry_and_mass"

    checks = {
        "all_four_cells_present": len(results) == 4,
        "all_cells_share_exact_initial_state_and_prefix": all(exact_initial.values()),
        "mass_and_inertia_override_readback_finite": all(
            np.isfinite(trace["initial_object_mass_kg"]).all()
            and np.isfinite(trace["initial_object_inertia"]).all()
            for trace in traces.values()
        ),
        **mass_checks,
    }
    output = {
        "protocol": "sugar_carry_skill_geometry_mass_factorial_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "seed": next(iter(seeds)),
        "cells": results,
        "admission_pattern": admitted,
        "diagnosis": diagnosis,
        "exact_initial_state_and_prefix": exact_initial,
        "claim_boundary": (
            "The same Carry45 Generator+Tracker, motion context, target, seed and exact "
            "initial state are used in all cells. Geometry and nominal mass/inertia are "
            "the only factorial variables. This is a physical compatibility diagnosis, "
            "not a learned mass-adaptation result."
        ),
    }
    (root / "FACTORIAL_RESULT.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise RuntimeError("geometry/mass structural proof failed")


if __name__ == "__main__":
    main()
