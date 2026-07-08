#!/usr/bin/env python3
"""Summarize a probe-then-carry direct Isaac diagnostic.

This is a heuristic selector report, not a learned policy and not a success
claim for full robot carrying.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize probe-then-carry diagnostic.")
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--carry-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-posture", required=True)
    parser.add_argument("--selected-stance-steps", type=int, required=True)
    parser.add_argument("--selected-step-length", type=float, required=True)
    parser.add_argument("--selection-rule", required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    probe = _load(args.probe_summary)
    carry = _load(args.carry_summary)
    failures: list[str] = []

    if not bool(probe.get("probe_belief_available")):
        failures.append("probe_belief_available is false")
    if bool(probe.get("probe_belief_uses_hidden_ground_truth")):
        failures.append("probe belief used hidden ground truth")
    if carry.get("carry_posture") != args.selected_posture:
        failures.append(
            f"carry posture mismatch: {carry.get('carry_posture')} != {args.selected_posture}"
        )
    if int(carry.get("fall_events") or 0) > 0:
        failures.append(f"carry fall_events {carry.get('fall_events')} > 0")
    if int(carry.get("box_drop_events") or 0) > 0:
        failures.append(f"carry box_drop_events {carry.get('box_drop_events')} > 0")
    if not bool(carry.get("root_shortcut_free")):
        failures.append("carry root_shortcut_free is false")
    if bool(carry.get("stance_anchor_fixed_to_world")):
        failures.append("carry stance_anchor_fixed_to_world is true")

    report = {
        "scene_type": "direct_isaac_probe_then_adaptive_carry_diagnostic",
        "status": "pass" if not failures else "fail",
        "success_claim": "heuristic_probe_then_carry_diagnostic_not_rl_not_full_walking_robot_success",
        "not_success_reason": (
            "uses current direct-Isaac support-foot scaffold and a hand-coded "
            "posture selector; this is not video-conditioned RL, not a learned "
            "policy, and not full humanoid free-walking carrying"
        ),
        "selector_type": "hand_coded_probe_risk_rule",
        "selection_rule": args.selection_rule,
        "selected_posture": args.selected_posture,
        "selected_stance_steps": int(args.selected_stance_steps),
        "selected_step_length_m": float(args.selected_step_length),
        "probe_summary": str(args.probe_summary),
        "carry_summary": str(args.carry_summary),
        "probe": {
            "completed_steps": probe.get("completed_steps"),
            "box_seed": probe.get("box_seed"),
            "box_randomized": probe.get("box_randomized"),
            "box_mass_kg": probe.get("box_mass_kg"),
            "box_size_m": probe.get("box_size_m"),
            "box_com_offset_m": probe.get("box_com_offset_m"),
            "probe_mode": probe.get("probe_mode"),
            "probe_belief_available": probe.get("probe_belief_available"),
            "probe_belief_source": probe.get("probe_belief_source"),
            "probe_belief_uses_hidden_ground_truth": probe.get("probe_belief_uses_hidden_ground_truth"),
            "probe_risk_score": probe.get("probe_risk_score"),
            "probe_load_risk_bucket": probe.get("probe_load_risk_bucket"),
            "probe_recommended_carry_adjustment": probe.get("probe_recommended_carry_adjustment"),
            "mean_probe_support_foot_x_measured_effort": probe.get("mean_probe_support_foot_x_measured_effort"),
            "mean_probe_support_foot_z_measured_effort": probe.get("mean_probe_support_foot_z_measured_effort"),
        },
        "carry": {
            "completed_steps": carry.get("completed_steps"),
            "carry_posture": carry.get("carry_posture"),
            "controller_mode": carry.get("controller_mode"),
            "backend_support_mode": carry.get("backend_support_mode"),
            "support_foot_mode": carry.get("support_foot_mode"),
            "max_box_travel_x_m": carry.get("max_box_travel_x_m"),
            "final_box_target_distance_x_m": carry.get("final_box_target_distance_x_m"),
            "final_post_settle_box_travel_x_m": carry.get("final_post_settle_box_travel_x_m"),
            "fall_events": carry.get("fall_events"),
            "box_drop_events": carry.get("box_drop_events"),
            "root_shortcut_free": carry.get("root_shortcut_free"),
            "stance_anchor_fixed_to_world": carry.get("stance_anchor_fixed_to_world"),
            "min_drive_near_ground_foot_count": carry.get("min_drive_near_ground_foot_count"),
            "drive_near_ground_zero_steps": carry.get("drive_near_ground_zero_steps"),
            "drive_near_ground_lt2_steps": carry.get("drive_near_ground_lt2_steps"),
            "min_commanded_stance_near_ground_foot_count": carry.get(
                "min_commanded_stance_near_ground_foot_count"
            ),
            "commanded_stance_near_ground_lt2_steps": carry.get(
                "commanded_stance_near_ground_lt2_steps"
            ),
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
