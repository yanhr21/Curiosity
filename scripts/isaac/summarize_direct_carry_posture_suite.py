#!/usr/bin/env python3
"""Summarize a multi-posture direct Isaac carry suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize direct carry posture suite.")
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-postures", type=int, default=3)
    parser.add_argument("--min-steps", type=int, default=3560)
    parser.add_argument("--min-box-travel-x", type=float, default=0.52)
    parser.add_argument("--max-target-distance-x", type=float, default=0.08)
    parser.add_argument("--max-tilt", type=float, default=0.14)
    parser.add_argument("--min-support-margin", type=float, default=0.12)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _count_bad_shortcuts(summary: dict) -> dict[str, int]:
    shortcuts = summary.get("root_shortcuts")
    if isinstance(shortcuts, dict):
        return {str(key): int(value or 0) for key, value in shortcuts.items() if int(value or 0) != 0}
    if not bool(summary.get("root_shortcut_free")):
        return {"root_shortcut_free": 1}
    return {}


def _max_dict_value(value: object) -> float | None:
    if not isinstance(value, dict) or not value:
        return None
    numeric = [float(item or 0.0) for item in value.values()]
    return max(numeric) if numeric else None


def main() -> int:
    args = parse_args()
    cases = []
    failures: list[str] = []

    for path in args.summary:
        summary = _load(path)
        posture = str(summary.get("carry_posture"))
        case_failures: list[str] = []
        bad_shortcuts = _count_bad_shortcuts(summary)
        completed_steps = int(summary.get("completed_steps") or 0)
        fall_events = int(summary.get("fall_events") or 0)
        box_drop_events = int(summary.get("box_drop_events") or 0)
        target_distance = float(summary.get("final_box_target_distance_x_m") or 999.0)
        box_travel = float(summary.get("max_box_travel_x_m") or 0.0)
        max_tilt = float(summary.get("max_tilt_rad") or 0.0)
        support_margin = summary.get("min_support_polygon_margin_m")
        support_margin_f = float(support_margin) if support_margin is not None else -999.0
        per_foot_speed = summary.get("per_foot_max_near_ground_xy_speed_mps")
        per_foot_slip = summary.get("per_foot_max_near_ground_xy_slip_m")
        max_near_ground_speed = _max_dict_value(per_foot_speed)
        max_near_ground_slip = _max_dict_value(per_foot_slip)
        if completed_steps < int(args.min_steps):
            case_failures.append(f"completed_steps {completed_steps} < {args.min_steps}")
        if fall_events:
            case_failures.append(f"fall_events {fall_events} > 0")
        if box_drop_events:
            case_failures.append(f"box_drop_events {box_drop_events} > 0")
        if bad_shortcuts:
            case_failures.append(f"shortcut counters nonzero: {bad_shortcuts}")
        if bool(summary.get("stance_anchor_fixed_to_world")):
            case_failures.append("stance_anchor_fixed_to_world is true")
        if box_travel < float(args.min_box_travel_x):
            case_failures.append(f"max_box_travel_x_m {box_travel} < {args.min_box_travel_x}")
        if target_distance > float(args.max_target_distance_x):
            case_failures.append(f"final_box_target_distance_x_m {target_distance} > {args.max_target_distance_x}")
        if max_tilt > float(args.max_tilt):
            case_failures.append(f"max_tilt_rad {max_tilt} > {args.max_tilt}")
        if support_margin_f < float(args.min_support_margin):
            case_failures.append(f"min_support_polygon_margin_m {support_margin_f} < {args.min_support_margin}")
        if int(summary.get("min_drive_near_ground_foot_count") or 0) < 2:
            case_failures.append("min_drive_near_ground_foot_count < 2")
        if int(summary.get("drive_near_ground_lt2_steps") or 0) != 0:
            case_failures.append("drive_near_ground_lt2_steps != 0")
        if int(summary.get("min_commanded_stance_near_ground_foot_count") or 0) < 2:
            case_failures.append("min_commanded_stance_near_ground_foot_count < 2")
        if int(summary.get("commanded_stance_near_ground_lt2_steps") or 0) != 0:
            case_failures.append("commanded_stance_near_ground_lt2_steps != 0")

        cases.append(
            {
                "summary": str(path),
                "carry_posture": posture,
                "completed_steps": completed_steps,
                "fall_events": fall_events,
                "box_drop_events": box_drop_events,
                "root_shortcut_free": not bad_shortcuts,
                "stance_anchor_fixed_to_world": bool(summary.get("stance_anchor_fixed_to_world")),
                "support_foot_mode": summary.get("support_foot_mode"),
                "support_foot_joint_count": summary.get("support_foot_joint_count"),
                "max_box_travel_x_m": box_travel,
                "final_box_target_distance_x_m": target_distance,
                "final_post_settle_box_travel_x_m": summary.get("final_post_settle_box_travel_x_m"),
                "max_tilt_rad": max_tilt,
                "min_support_polygon_margin_m": support_margin,
                "probe_mode": summary.get("probe_mode"),
                "probe_risk_score": summary.get("probe_risk_score"),
                "probe_belief_available": summary.get("probe_belief_available"),
                "probe_belief_uses_hidden_ground_truth": summary.get(
                    "probe_belief_uses_hidden_ground_truth"
                ),
                "online_probe_adaptive_support_enabled": summary.get(
                    "online_probe_adaptive_support_enabled"
                ),
                "online_probe_adaptive_support_risk_bucket": summary.get(
                    "online_probe_adaptive_support_risk_bucket"
                ),
                "online_probe_adaptive_support_profile": summary.get(
                    "online_probe_adaptive_support_profile"
                ),
                "online_probe_adaptive_support_uses_hidden_ground_truth": summary.get(
                    "online_probe_adaptive_support_uses_hidden_ground_truth"
                ),
                "online_probe_adaptive_hold_enabled": summary.get("online_probe_adaptive_hold_enabled"),
                "online_probe_adaptive_hold_risk_bucket": summary.get(
                    "online_probe_adaptive_hold_risk_bucket"
                ),
                "online_probe_adaptive_hold_profile": summary.get(
                    "online_probe_adaptive_hold_profile"
                ),
                "online_probe_adaptive_hold_closure_fraction": summary.get(
                    "online_probe_adaptive_hold_closure_fraction"
                ),
                "online_probe_adaptive_hold_collision_available": summary.get(
                    "online_probe_adaptive_hold_collision_available"
                ),
                "online_probe_adaptive_hold_collision_enabled": summary.get(
                    "online_probe_adaptive_hold_collision_enabled"
                ),
                "online_probe_adaptive_hold_collision_update_count": summary.get(
                    "online_probe_adaptive_hold_collision_update_count"
                ),
                "online_probe_adaptive_hold_uses_hidden_ground_truth": summary.get(
                    "online_probe_adaptive_hold_uses_hidden_ground_truth"
                ),
                "planted_stance_rail_propulsion_enabled": summary.get(
                    "planted_stance_rail_propulsion_enabled"
                ),
                "planted_stance_rail_propulsion_steps": summary.get(
                    "planted_stance_rail_propulsion_steps"
                ),
                "freeze_commanded_stance_foot_targets_enabled": summary.get(
                    "freeze_commanded_stance_foot_targets_enabled"
                ),
                "freeze_commanded_stance_foot_target_count": summary.get(
                    "freeze_commanded_stance_foot_target_count"
                ),
                "freeze_commanded_stance_foot_target_switch_count": summary.get(
                    "freeze_commanded_stance_foot_target_switch_count"
                ),
                "freeze_commanded_stance_foot_active_feet": summary.get(
                    "freeze_commanded_stance_foot_active_feet"
                ),
                "per_foot_max_near_ground_xy_speed_mps": summary.get(
                    "per_foot_max_near_ground_xy_speed_mps"
                ),
                "max_near_ground_foot_speed_mps": max_near_ground_speed,
                "per_foot_max_near_ground_xy_slip_m": summary.get(
                    "per_foot_max_near_ground_xy_slip_m"
                ),
                "max_near_ground_foot_slip_m": max_near_ground_slip,
                "min_drive_near_ground_foot_count": summary.get("min_drive_near_ground_foot_count"),
                "drive_near_ground_lt2_steps": summary.get("drive_near_ground_lt2_steps"),
                "min_commanded_stance_near_ground_foot_count": summary.get(
                    "min_commanded_stance_near_ground_foot_count"
                ),
                "commanded_stance_near_ground_lt2_steps": summary.get(
                    "commanded_stance_near_ground_lt2_steps"
                ),
                "failures": case_failures,
            }
        )
        failures.extend([f"{posture}: {failure}" for failure in case_failures])

    postures = sorted({case["carry_posture"] for case in cases})
    if len(postures) < int(args.min_postures):
        failures.append(f"posture count {len(postures)} < {args.min_postures}")

    report = {
        "scene_type": "direct_isaac_multi_posture_box_carry_suite",
        "status": "pass" if not failures else "fail",
        "success_claim": "direct_isaac_support_foot_robot_carry_suite_not_final_humanoid_or_rl_success",
        "not_success_reason": (
            "This verifies the current articulated support-foot robot scaffold "
            "across multiple carry postures. It is not yet a full humanoid "
            "controller, not video-conditioned RL, and not proof of arbitrary "
            "posture carrying beyond the declared suite."
        ),
        "postures": postures,
        "case_count": len(cases),
        "thresholds": {
            "min_postures": int(args.min_postures),
            "min_steps": int(args.min_steps),
            "min_box_travel_x_m": float(args.min_box_travel_x),
            "max_target_distance_x_m": float(args.max_target_distance_x),
            "max_tilt_rad": float(args.max_tilt),
            "min_support_polygon_margin_m": float(args.min_support_margin),
        },
        "cases": cases,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
