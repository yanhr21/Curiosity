#!/usr/bin/env python3
"""Summarize the matched direct-versus-causal-fallback SUGAR skill test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CELLS = (
    "carry_kick_direct",
    "carry_kick_safe",
    "kick_carry_direct",
    "kick_carry_safe",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    result = {
        cell: json.loads((root / cell / "RESULT.json").read_text(encoding="utf-8"))
        for cell in CELLS
    }
    trace = {
        cell: np.load(root / cell / "TRACE.npz", allow_pickle=False)
        for cell in CELLS
    }

    checks = {
        "all_four_cells_present": True,
        "carry_direct_safe_initial_state_exact": all(
            np.array_equal(
                trace["carry_kick_direct"][f"post_prefix_{name}"],
                trace["carry_kick_safe"][f"post_prefix_{name}"],
            )
            for name in (
                "robot_root_state_w",
                "robot_joint_pos",
                "robot_joint_vel",
                "object_root_state_w",
            )
        ),
        "kick_direct_safe_initial_state_exact": all(
            np.array_equal(
                trace["kick_carry_direct"][f"post_prefix_{name}"],
                trace["kick_carry_safe"][f"post_prefix_{name}"],
            )
            for name in (
                "robot_root_state_w",
                "robot_joint_pos",
                "robot_joint_vel",
                "object_root_state_w",
            )
        ),
        "carry_kick_direct_is_behaviorally_admitted": bool(
            result["carry_kick_direct"]["passed"]
        ),
        "all_cells_disable_observation_corruption_for_exact_equivalence": all(
            record["observation_corruption_disabled"] for record in result.values()
        ),
        "carry_kick_safe_has_no_falls": (
            result["carry_kick_safe"]["aggregate"]["physical_fall_count"] == 0
        ),
        "carry_kick_safe_retains_kick_behavior": (
            result["carry_kick_safe"]["aggregate"]["kick_success_count"] >= 10
        ),
        "no_fallback_carry_candidate_actions_match_direct": bool(
            np.allclose(
                trace["carry_kick_direct"]["student_action"],
                trace["carry_kick_safe"]["student_action"],
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ),
        "no_fallback_carry_physics_matches_direct": bool(
            np.allclose(
                trace["carry_kick_direct"]["object_root_state_w"],
                trace["carry_kick_safe"]["object_root_state_w"],
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ),
        "kick_carry_direct_is_rejected": bool(
            not result["kick_carry_direct"]["passed"]
        ),
        "kick_carry_direct_leaves_action_envelope": (
            result["kick_carry_direct"]["maximum_abs_raw_student_action"] > 25.0
        ),
        "kick_carry_safe_uses_fallback": (
            result["kick_carry_safe"]["safe_fallback_fraction"] > 0.0
        ),
        "kick_carry_first_candidate_action_matches_direct": bool(
            np.allclose(
                trace["kick_carry_direct"]["student_action"][0],
                trace["kick_carry_safe"]["student_action"][0],
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ),
        "kick_carry_safe_stays_inside_action_envelope": (
            result["kick_carry_safe"]["maximum_abs_executed_action"] <= 25.0
        ),
        "kick_carry_safe_has_no_falls": (
            result["kick_carry_safe"]["aggregate"]["physical_fall_count"] == 0
        ),
        "kick_carry_safe_retains_domain_kick_behavior": (
            result["kick_carry_safe"]["aggregate"]["kick_success_count"] >= 10
        ),
    }
    cells = {
        cell: {
            "passed": record["passed"],
            "carry_success_count": record["aggregate"]["carry_success_count"],
            "kick_success_count": record["aggregate"]["kick_success_count"],
            "physical_fall_count": record["aggregate"]["physical_fall_count"],
            "maximum_abs_raw_candidate_action": record[
                "maximum_abs_raw_student_action"
            ],
            "maximum_abs_executed_action": record["maximum_abs_executed_action"],
            "safe_fallback_fraction": record["safe_fallback_fraction"],
            "profiles_using_safe_fallback": record["profiles_using_safe_fallback"],
        }
        for cell, record in result.items()
    }
    passed = all(checks.values())
    summary = {
        "protocol": "sugar_official_skill_causal_safe_fallback_v1",
        "passed": passed,
        "checks": checks,
        "cells": cells,
        "diagnosis": (
            "current_action_envelope_blocks_catastrophic_cross_geometry_actions_"
            "while_preserving_the_released_domain_skill"
            if passed
            else "action_envelope_triggers_only_after_the_state_has_left_both_"
            "released_expert_distributions"
        ),
        "claim_boundary": (
            "Matched frozen-policy evidence for a causal safety composition of released "
            "SUGAR skills. It does not prove arbitrary-demo following, semantic transfer "
            "or a general learned latent skill model."
        ),
    }
    (root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise RuntimeError("official skill safe-fallback experiment failed")


if __name__ == "__main__":
    main()
