#!/usr/bin/env python3
"""Summarize active-probing task-runner carry episodes across postures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_POSTURES = ("front_mid", "low_front", "chest_high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize task-runner active-probe posture episodes.")
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--box-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--postures", nargs="+", default=list(DEFAULT_POSTURES))
    parser.add_argument(
        "--runner-root",
        type=Path,
        default=Path("experiments/outputs/direct_carry_task_runner"),
    )
    parser.add_argument(
        "--check-root",
        type=Path,
        default=Path("experiments/outputs/direct_carry_task_runner_checks"),
    )
    return parser.parse_args()


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _passed(summary: dict, check: dict | None) -> bool:
    checks = [
        check is not None and check.get("status") == "pass",
        int(summary.get("completed_steps") or 0) > 0,
        int(summary.get("fall_events") or 0) == 0,
        int(summary.get("box_drop_events") or 0) == 0,
        bool(summary.get("root_shortcut_free")),
        not bool(summary.get("stance_anchor_fixed_to_world")),
        bool(summary.get("support_foot_contact_report_available")),
        int(summary.get("support_foot_contact_report_error_count") or 0) == 0,
        int(summary.get("min_drive_contact_report_foot_count") or 0) >= 2,
        int(summary.get("drive_contact_report_lt2_steps") or 0) == 0,
        int(summary.get("min_commanded_stance_contact_report_foot_count") or 0) >= 2,
        int(summary.get("commanded_stance_contact_report_lt2_steps") or 0) == 0,
        int(summary.get("probe_steps_requested") or 0) > 0,
        bool(summary.get("probe_belief_available")),
        not bool(summary.get("probe_belief_uses_hidden_ground_truth")),
        float(summary.get("max_probe_box_travel_x_m") or 0.0) > 0.0,
        bool(summary.get("feedback_step_controller_enabled")),
        int(summary.get("feedback_step_applied_steps") or 0) > 0,
    ]
    return all(checks)


def main() -> int:
    args = parse_args()
    postures = []
    failures = []
    reference_box = None
    for posture in args.postures:
        case_stamp = f"{args.stamp}_{posture}"
        summary_path = args.runner_root / case_stamp / "direct_carry_task_physical_backend_summary.json"
        check_path = args.check_root / f"{case_stamp}_probecheck" / "direct_carry_task_runner_check.json"
        row_path = args.check_root / f"{case_stamp}_probecheck" / "direct_carry_task_runner_episode_table.jsonl"
        summary = _load(summary_path)
        check = _load(check_path)
        if summary is None:
            failures.append(f"{posture}: missing summary {summary_path}")
            continue
        box_signature = {
            "box_seed": summary.get("box_seed"),
            "box_mass_kg": summary.get("box_mass_kg"),
            "box_size_m": summary.get("box_size_m"),
            "box_com_offset_m": summary.get("box_com_offset_m"),
        }
        if reference_box is None:
            reference_box = box_signature
        elif box_signature != reference_box:
            failures.append(f"{posture}: randomized box signature differs from first posture")
        if int(summary.get("box_seed") or -1) != int(args.box_seed):
            failures.append(f"{posture}: box_seed {summary.get('box_seed')} != {args.box_seed}")
        passed = _passed(summary, check)
        if not passed:
            failures.append(f"{posture}: active-probe task-runner gate failed")
        postures.append(
            {
                "posture": posture,
                "passed": passed,
                "summary_path": str(summary_path),
                "check_path": str(check_path),
                "episode_row_path": str(row_path),
                "check_status": None if check is None else check.get("status"),
                "completed_steps": summary.get("completed_steps"),
                "box_seed": summary.get("box_seed"),
                "box_mass_kg": summary.get("box_mass_kg"),
                "box_size_m": summary.get("box_size_m"),
                "box_com_offset_m": summary.get("box_com_offset_m"),
                "fall_events": summary.get("fall_events"),
                "box_drop_events": summary.get("box_drop_events"),
                "final_post_settle_box_travel_x_m": summary.get("final_post_settle_box_travel_x_m"),
                "max_abs_post_settle_box_travel_x_m": summary.get("max_abs_post_settle_box_travel_x_m"),
                "max_target_directed_post_settle_box_travel_m": summary.get(
                    "max_target_directed_post_settle_box_travel_m"
                ),
                "final_post_settle_box_target_distance_x_m": summary.get(
                    "final_post_settle_box_target_distance_x_m"
                ),
                "support_foot_placement_mode": summary.get("support_foot_placement_mode"),
                "support_foot_placement_controller_enabled": summary.get(
                    "support_foot_placement_controller_enabled"
                ),
                "support_foot_directional_placement": summary.get("support_foot_directional_placement"),
                "stance_foot_world_lock_enabled": summary.get("stance_foot_world_lock_enabled"),
                "stance_foot_world_lock_joint_count": summary.get("stance_foot_world_lock_joint_count"),
                "stance_foot_world_lock_switch_count": summary.get("stance_foot_world_lock_switch_count"),
                "stance_foot_world_lock_pose_update_count": summary.get(
                    "stance_foot_world_lock_pose_update_count"
                ),
                "stance_foot_world_lock_active_feet": summary.get("stance_foot_world_lock_active_feet"),
                "freeze_locked_stance_foot_targets_enabled": summary.get(
                    "freeze_locked_stance_foot_targets_enabled"
                ),
                "freeze_locked_stance_foot_target_count": summary.get("freeze_locked_stance_foot_target_count"),
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
                "planted_stance_rail_propulsion_enabled": summary.get(
                    "planted_stance_rail_propulsion_enabled"
                ),
                "planted_stance_rail_propulsion_steps": summary.get("planted_stance_rail_propulsion_steps"),
                "feedback_step_controller_enabled": summary.get("feedback_step_controller_enabled"),
                "feedback_step_x_gain": summary.get("feedback_step_x_gain"),
                "feedback_step_x_limit_m": summary.get("feedback_step_x_limit_m"),
                "feedback_step_tilt_gain": summary.get("feedback_step_tilt_gain"),
                "feedback_step_tilt_limit_m": summary.get("feedback_step_tilt_limit_m"),
                "feedback_step_applied_steps": summary.get("feedback_step_applied_steps"),
                "max_abs_feedback_step_x_adjustment_m": summary.get("max_abs_feedback_step_x_adjustment_m"),
                "max_abs_feedback_step_tilt_adjustment_m": summary.get("max_abs_feedback_step_tilt_adjustment_m"),
                "probe_steps_requested": summary.get("probe_steps_requested"),
                "probe_belief_available": summary.get("probe_belief_available"),
                "probe_belief_source": summary.get("probe_belief_source"),
                "probe_belief_uses_hidden_ground_truth": summary.get("probe_belief_uses_hidden_ground_truth"),
                "probe_risk_score": summary.get("probe_risk_score"),
                "probe_load_risk_bucket": summary.get("probe_load_risk_bucket"),
                "probe_recommended_carry_adjustment": summary.get("probe_recommended_carry_adjustment"),
                "max_probe_box_travel_x_m": summary.get("max_probe_box_travel_x_m"),
                "max_probe_box_relative_error_m": summary.get("max_probe_box_relative_error_m"),
                "support_foot_contact_report_available": summary.get("support_foot_contact_report_available"),
                "support_foot_contact_report_error_count": summary.get("support_foot_contact_report_error_count"),
                "min_drive_contact_report_foot_count": summary.get("min_drive_contact_report_foot_count"),
                "drive_contact_report_lt2_steps": summary.get("drive_contact_report_lt2_steps"),
                "min_commanded_stance_contact_report_foot_count": summary.get(
                    "min_commanded_stance_contact_report_foot_count"
                ),
                "commanded_stance_contact_report_lt2_steps": summary.get(
                    "commanded_stance_contact_report_lt2_steps"
                ),
                "per_foot_max_near_ground_xy_speed_mps": summary.get(
                    "per_foot_max_near_ground_xy_speed_mps"
                ),
                "per_foot_max_near_ground_xy_slip_m": summary.get(
                    "per_foot_max_near_ground_xy_slip_m"
                ),
                "max_near_ground_foot_speed_mps": max(
                    [
                        float(value or 0.0)
                        for value in (summary.get("per_foot_max_near_ground_xy_speed_mps") or {}).values()
                    ]
                    or [0.0]
                ),
                "max_near_ground_foot_slip_m": max(
                    [
                        float(value or 0.0)
                        for value in (summary.get("per_foot_max_near_ground_xy_slip_m") or {}).values()
                    ]
                    or [0.0]
                ),
                "root_shortcut_free": summary.get("root_shortcut_free"),
                "stance_anchor_fixed_to_world": summary.get("stance_anchor_fixed_to_world"),
            }
        )
    report = {
        "scene_type": "direct_isaac_task_runner_active_probe_posture_sweep",
        "status": "pass" if not failures and len(postures) == len(args.postures) else "fail",
        "success_claim": "active_probe_task_runner_scaffold_not_full_walking_robot_success",
        "not_success_reason": (
            "current backend is still a scaffolded support-foot carrier, not a "
            "complete walking robot or learned loco-manipulation policy"
        ),
        "stamp": args.stamp,
        "box_seed": int(args.box_seed),
        "shared_randomized_box": reference_box,
        "postures": postures,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
