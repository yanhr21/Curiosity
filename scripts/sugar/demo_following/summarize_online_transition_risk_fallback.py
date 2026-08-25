#!/usr/bin/env python3
"""Summarize matched direct versus transition-risk-latched online behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    direct_result = json.loads(
        (root / "direct_carry_on_big/RESULT.json").read_text(encoding="utf-8")
    )
    risk_result = json.loads(
        (root / "risk_latched_fallback/RESULT.json").read_text(encoding="utf-8")
    )
    direct = np.load(root / "direct_carry_on_big/TRACE.npz", allow_pickle=False)
    risk = np.load(root / "risk_latched_fallback/TRACE.npz", allow_pickle=False)
    exact_keys = (
        "initial_robot_root_state_w",
        "initial_robot_joint_pos",
        "initial_robot_joint_vel",
        "initial_object_root_state_w",
        "post_prefix_robot_root_state_w",
        "post_prefix_robot_joint_pos",
        "post_prefix_robot_joint_vel",
        "post_prefix_object_root_state_w",
        "prefix_action",
    )
    predecision = slice(0, 50)
    invalid_path = root / "risk_latched_fallback/INVALID_TRANSITION.npz"
    invalid_record = None
    if invalid_path.exists():
        with np.load(invalid_path, allow_pickle=False) as invalid:
            invalid_envs = invalid["env_indices"].astype(np.int64)
            latched = invalid["latched_fallback"].astype(bool)
            probabilities = invalid["risk_probability"]
            invalid_record = {
                "frame": int(invalid["frame"]),
                "env_indices": invalid_envs.tolist(),
                "all_invalid_profiles_were_latched_fallback": bool(
                    np.all(latched[invalid_envs])
                ),
                "risk_probability_at_invalid_profiles": probabilities[
                    invalid_envs
                ].tolist(),
                "executed_action_all_finite": bool(
                    np.isfinite(invalid["executed_action"]).all()
                ),
                "candidate_action_all_finite": bool(
                    np.isfinite(invalid["candidate_action"]).all()
                ),
                "fallback_action_all_finite": bool(
                    np.isfinite(invalid["fallback_action"]).all()
                ),
                "maximum_abs_finite_candidate_action": float(
                    np.max(np.abs(invalid["candidate_action"]))
                ),
            }
    checks = {
        "matched_initial_and_prefix_state_exact": all(
            np.array_equal(direct[key], risk[key]) for key in exact_keys
        ),
        "candidate_actions_exact_before_decision": bool(
            np.array_equal(
                direct["student_action"][predecision],
                risk["student_action"][predecision],
            )
        ),
        "direct_cross_geometry_arm_is_rejected": not direct_result["passed"],
        "risk_fallback_evaluator_passes": risk_result["passed"],
        "risk_fallback_is_state_dependent": (
            0 < risk_result["transition_risk_latched_profile_count"] < 20
        ),
        "risk_fallback_has_no_falls": (
            risk_result["aggregate"]["physical_fall_count"] == 0
        ),
        "risk_fallback_actions_remain_in_envelope": (
            risk_result["maximum_abs_executed_action"] <= 25.0
        ),
        "risk_fallback_retains_physical_skill_behavior": (
            risk_result["aggregate"]["carry_success_count"]
            + risk_result["aggregate"]["kick_success_count"]
            >= 10
        ),
    }
    summary = {
        "protocol": "sugar_online_causal_transition_risk_fallback_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "direct": {
            "carry_success_count": direct_result["aggregate"]["carry_success_count"],
            "kick_success_count": direct_result["aggregate"]["kick_success_count"],
            "physical_fall_count": direct_result["aggregate"]["physical_fall_count"],
            "maximum_abs_executed_action": direct_result["maximum_abs_executed_action"],
        },
        "risk_fallback": {
            "carry_success_count": risk_result["aggregate"]["carry_success_count"],
            "kick_success_count": risk_result["aggregate"]["kick_success_count"],
            "physical_fall_count": risk_result["aggregate"]["physical_fall_count"],
            "maximum_abs_executed_action": risk_result["maximum_abs_executed_action"],
            "latched_profile_count": risk_result["transition_risk_latched_profile_count"],
            "threshold": risk_result["transition_risk_threshold"],
            "executed_valid_frames": risk_result.get("executed_valid_frames", 650),
            "first_invalid_transition": invalid_record,
        },
        "claim_boundary": (
            "Matched one-seed online evidence for a Carry45-only causal transition-risk "
            "fallback on BIGBOX. Offline early risk separation does not make an abrupt "
            "frame-49 endpoint switch safe: the first invalid profile was already "
            "risk-latched to the Kick fallback. This is a negative result and not a "
            "general multi-skill transition policy."
        ),
        "automatic_next_stage": (
            "learn a causal transition/recovery controller using official endpoint skills; "
            "do not tune the frozen classifier threshold"
            if invalid_record is not None
            else "inspect completed online composition"
        ),
    }
    (root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise RuntimeError("online transition-risk fallback gate failed")


if __name__ == "__main__":
    main()
