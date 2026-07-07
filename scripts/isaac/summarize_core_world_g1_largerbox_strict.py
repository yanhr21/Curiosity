#!/usr/bin/env python3
"""Summarize strict Core API G1 larger-box carrying diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strict G1 larger-box carrying diagnostics.")
    parser.add_argument(
        "--case-root",
        action="append",
        type=Path,
        default=[],
        help="Suite root containing case subdirectories such as agile_low_cradle_freebox_walk.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        default=[],
        help="Direct core_world_g1_box_scene_summary.json path.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _summary_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for root in args.case_root:
        if (root / "core_world_g1_box_scene_summary.json").exists():
            paths.append(root / "core_world_g1_box_scene_summary.json")
            continue
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            candidate = case_dir / "core_world_g1_box_scene_summary.json"
            if candidate.exists():
                paths.append(candidate)
    paths.extend(args.summary)
    return paths


def _check_path(summary_path: Path) -> Path:
    candidate = summary_path.with_name("check.json")
    if candidate.exists():
        return candidate
    return summary_path.with_name("strict_check_report.json")


def _count_rollout_writes(summary: dict[str, Any]) -> int:
    keys = (
        "root_pose_write_count_rollout",
        "root_velocity_write_count_rollout",
        "box_pose_write_count_rollout",
    )
    return sum(int(summary.get(key) or 0) for key in keys)


def _first_events_from_csv(summary_path: Path) -> dict[str, Any]:
    state_path = summary_path.with_name("core_world_g1_box_scene_state.csv")
    events: dict[str, Any] = {
        "first_fall_step": None,
        "first_fall_time_s": None,
        "first_box_drop_step": None,
        "first_box_drop_time_s": None,
    }
    if not state_path.exists():
        return events
    with state_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if events["first_fall_step"] is None and int(float(row.get("fall") or 0)):
                events["first_fall_step"] = int(float(row["step"]))
                events["first_fall_time_s"] = float(row["time_s"])
            if events["first_box_drop_step"] is None and int(float(row.get("drop") or 0)):
                events["first_box_drop_step"] = int(float(row["step"]))
                events["first_box_drop_time_s"] = float(row["time_s"])
            if events["first_fall_step"] is not None and events["first_box_drop_step"] is not None:
                break
    return events


def _case_record(summary_path: Path) -> dict[str, Any]:
    summary = _load(summary_path)
    check_path = _check_path(summary_path)
    check = _load(check_path) if check_path.exists() else {}
    check_status = check.get("status")
    rollout_writes = _count_rollout_writes(summary)
    fall_events = int(summary.get("fall_events") or 0)
    drop_events = int(summary.get("box_drop_events") or 0)
    first_events = _first_events_from_csv(summary_path)
    failures = list(check.get("failures") or [])
    passed = check_status == "pass" and fall_events == 0 and drop_events == 0 and rollout_writes == 0
    if check_status is None:
        failures.append("missing check status")
    if fall_events:
        failures.append(f"fall_events {fall_events} > 0")
    if drop_events:
        failures.append(f"box_drop_events {drop_events} > 0")
    if rollout_writes:
        failures.append(f"rollout root/box writes {rollout_writes} > 0")
    return {
        "case_dir": str(summary_path.parent),
        "summary_path": str(summary_path),
        "check_path": str(check_path) if check_path.exists() else None,
        "passed": passed,
        "check_status": check_status,
        "failures": failures,
        "completed_steps": summary.get("completed_steps"),
        "fall_events": fall_events,
        "box_drop_events": drop_events,
        "first_fall_step": summary.get("first_fall_step") if summary.get("first_fall_step") is not None else first_events["first_fall_step"],
        "first_fall_time_s": summary.get("first_fall_time_s") if summary.get("first_fall_time_s") is not None else first_events["first_fall_time_s"],
        "first_box_drop_step": summary.get("first_box_drop_step") if summary.get("first_box_drop_step") is not None else first_events["first_box_drop_step"],
        "first_box_drop_time_s": summary.get("first_box_drop_time_s") if summary.get("first_box_drop_time_s") is not None else first_events["first_box_drop_time_s"],
        "agile_command_hold_final_fall_events": summary.get("agile_command_hold_final_fall_events"),
        "agile_command_hold_final_box_drop_events": summary.get(
            "agile_command_hold_final_box_drop_events"
        ),
        "agile_command_hold_final_first_fall_step": summary.get(
            "agile_command_hold_final_first_fall_step"
        ),
        "agile_command_hold_final_first_box_drop_step": summary.get(
            "agile_command_hold_final_first_box_drop_step"
        ),
        "agile_command_hold_final_stand_fall_events": summary.get(
            "agile_command_hold_final_stand_fall_events"
        ),
        "agile_command_hold_final_stand_box_drop_events": summary.get(
            "agile_command_hold_final_stand_box_drop_events"
        ),
        "agile_command_hold_final_stand_first_fall_step": summary.get(
            "agile_command_hold_final_stand_first_fall_step"
        ),
        "agile_command_hold_final_stand_first_box_drop_step": summary.get(
            "agile_command_hold_final_stand_first_box_drop_step"
        ),
        "rollout_write_count_total": rollout_writes,
        "root_pose_write_count_rollout": summary.get("root_pose_write_count_rollout"),
        "root_velocity_write_count_rollout": summary.get("root_velocity_write_count_rollout"),
        "box_pose_write_count_rollout": summary.get("box_pose_write_count_rollout"),
        "box_mass_kg": summary.get("box_mass_kg"),
        "box_size_m": summary.get("box_size_m"),
        "final_robot_target_directed_travel_m": summary.get("final_robot_target_directed_travel_m"),
        "final_box_target_directed_travel_m": summary.get("final_box_target_directed_travel_m"),
        "max_robot_target_directed_travel_m": summary.get("max_robot_target_directed_travel_m"),
        "max_box_target_directed_travel_m": summary.get("max_box_target_directed_travel_m"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "max_abs_roll_rad": summary.get("max_abs_roll_rad"),
        "max_abs_pitch_rad": summary.get("max_abs_pitch_rad"),
        "max_box_tilt_rad": summary.get("max_box_tilt_rad"),
        "agile_command_hold_final_min_robot_z_m": summary.get(
            "agile_command_hold_final_min_robot_z_m"
        ),
        "agile_command_hold_final_min_box_z_m": summary.get(
            "agile_command_hold_final_min_box_z_m"
        ),
        "agile_command_hold_final_max_tilt_rad": summary.get(
            "agile_command_hold_final_max_tilt_rad"
        ),
        "agile_command_hold_final_max_box_tilt_rad": summary.get(
            "agile_command_hold_final_max_box_tilt_rad"
        ),
        "agile_command_hold_final_stand_min_robot_z_m": summary.get(
            "agile_command_hold_final_stand_min_robot_z_m"
        ),
        "agile_command_hold_final_stand_min_box_z_m": summary.get(
            "agile_command_hold_final_stand_min_box_z_m"
        ),
        "agile_command_hold_final_stand_max_tilt_rad": summary.get(
            "agile_command_hold_final_stand_max_tilt_rad"
        ),
        "agile_command_hold_final_stand_max_box_tilt_rad": summary.get(
            "agile_command_hold_final_stand_max_box_tilt_rad"
        ),
        "max_abs_box_roll_rad": summary.get("max_abs_box_roll_rad"),
        "max_abs_box_pitch_rad": summary.get("max_abs_box_pitch_rad"),
        "max_box_robot_relative_offset_error_m": summary.get("max_box_robot_relative_offset_error_m"),
        "final_box_robot_relative_offset_error_m": summary.get("final_box_robot_relative_offset_error_m"),
        "max_abs_robot_target_lateral_error_m": summary.get("max_abs_robot_target_lateral_error_m"),
        "max_abs_box_target_lateral_error_m": summary.get("max_abs_box_target_lateral_error_m"),
        "final_robot_target_lateral_error_m": summary.get("final_robot_target_lateral_error_m"),
        "final_box_target_lateral_error_m": summary.get("final_box_target_lateral_error_m"),
        "target_window_enabled": summary.get("target_window_enabled"),
        "target_window_center_m": summary.get("target_window_center_m"),
        "target_window_halfwidth_m": summary.get("target_window_halfwidth_m"),
        "target_window_robot_stable_steps": summary.get("target_window_robot_stable_steps"),
        "target_window_box_stable_steps": summary.get("target_window_box_stable_steps"),
        "target_window_both_stable_steps": summary.get("target_window_both_stable_steps"),
        "target_window_robot_longest_streak_steps": summary.get(
            "target_window_robot_longest_streak_steps"
        ),
        "target_window_box_longest_streak_steps": summary.get(
            "target_window_box_longest_streak_steps"
        ),
        "target_window_both_longest_streak_steps": summary.get(
            "target_window_both_longest_streak_steps"
        ),
        "target_window_both_streak_at_end_steps": summary.get(
            "target_window_both_streak_at_end_steps"
        ),
        "target_window_robot_first_stable_step": summary.get("target_window_robot_first_stable_step"),
        "target_window_box_first_stable_step": summary.get("target_window_box_first_stable_step"),
        "target_window_both_first_stable_step": summary.get("target_window_both_first_stable_step"),
        "target_window_both_final_hold_stable_steps": summary.get(
            "target_window_both_final_hold_stable_steps"
        ),
        "target_window_both_final_hold_longest_streak_steps": summary.get(
            "target_window_both_final_hold_longest_streak_steps"
        ),
        "target_window_both_final_hold_streak_at_end_steps": summary.get(
            "target_window_both_final_hold_streak_at_end_steps"
        ),
        "target_window_both_final_hold_first_stable_step": summary.get(
            "target_window_both_final_hold_first_stable_step"
        ),
        "target_window_both_final_stand_stable_steps": summary.get(
            "target_window_both_final_stand_stable_steps"
        ),
        "target_window_both_final_stand_longest_streak_steps": summary.get(
            "target_window_both_final_stand_longest_streak_steps"
        ),
        "target_window_both_final_stand_streak_at_end_steps": summary.get(
            "target_window_both_final_stand_streak_at_end_steps"
        ),
        "target_window_both_final_stand_first_stable_step": summary.get(
            "target_window_both_final_stand_first_stable_step"
        ),
        "target_window_both_stable_at_final_step": summary.get(
            "target_window_both_stable_at_final_step"
        ),
        "target_window_both_final_hold_stable_at_final_step": summary.get(
            "target_window_both_final_hold_stable_at_final_step"
        ),
        "target_window_both_final_stand_stable_at_final_step": summary.get(
            "target_window_both_final_stand_stable_at_final_step"
        ),
        "balance_roll_target_from_lateral": summary.get("balance_roll_target_from_lateral"),
        "balance_roll_target_lateral_source": summary.get("balance_roll_target_lateral_source"),
        "balance_roll_target_lateral_gain": summary.get("balance_roll_target_lateral_gain"),
        "balance_roll_target_lateral_limit": summary.get("balance_roll_target_lateral_limit"),
        "balance_roll_target_lateral_deadband": summary.get(
            "balance_roll_target_lateral_deadband"
        ),
        "balance_roll_target_lateral_sign": summary.get("balance_roll_target_lateral_sign"),
        "balance_roll_target_lateral_start_after_hold_steps": summary.get(
            "balance_roll_target_lateral_start_after_hold_steps"
        ),
        "balance_roll_target_lateral_ramp_steps": summary.get(
            "balance_roll_target_lateral_ramp_steps"
        ),
        "balance_roll_target_lateral_max_tilt": summary.get(
            "balance_roll_target_lateral_max_tilt"
        ),
        "balance_roll_target_lateral_max_box_tilt": summary.get(
            "balance_roll_target_lateral_max_box_tilt"
        ),
        "balance_roll_target_lateral_active_steps": summary.get(
            "balance_roll_target_lateral_active_steps"
        ),
        "balance_roll_target_lateral_first_active_step": summary.get(
            "balance_roll_target_lateral_first_active_step"
        ),
        "balance_roll_target_lateral_last_error_m": summary.get(
            "balance_roll_target_lateral_last_error_m"
        ),
        "balance_roll_target_lateral_last_target_rad": summary.get(
            "balance_roll_target_lateral_last_target_rad"
        ),
        "balance_roll_target_lateral_max_abs_target_rad": summary.get(
            "balance_roll_target_lateral_max_abs_target_rad"
        ),
        "balance_roll_target_lateral_suppressed_by_hold_delay_steps": summary.get(
            "balance_roll_target_lateral_suppressed_by_hold_delay_steps"
        ),
        "balance_roll_target_lateral_suppressed_by_tilt_steps": summary.get(
            "balance_roll_target_lateral_suppressed_by_tilt_steps"
        ),
        "box_retention_posture_controller_enabled": summary.get(
            "box_retention_posture_controller_enabled"
        ),
        "box_retention_rel_start_m": summary.get("box_retention_rel_start_m"),
        "box_retention_rel_stop_m": summary.get("box_retention_rel_stop_m"),
        "box_retention_tilt_start_rad": summary.get("box_retention_tilt_start_rad"),
        "box_retention_tilt_stop_rad": summary.get("box_retention_tilt_stop_rad"),
        "box_retention_blend_rate": summary.get("box_retention_blend_rate"),
        "box_retention_active_steps": summary.get("box_retention_active_steps"),
        "box_retention_first_active_step": summary.get("box_retention_first_active_step"),
        "box_retention_last_risk": summary.get("box_retention_last_risk"),
        "box_retention_max_risk": summary.get("box_retention_max_risk"),
        "agile_command_hold_final_tilt_escape_scale": summary.get(
            "agile_command_hold_final_tilt_escape_scale"
        ),
        "agile_command_hold_final_tilt_escape_tilt_rad": summary.get(
            "agile_command_hold_final_tilt_escape_tilt_rad"
        ),
        "agile_command_hold_final_tilt_escape_box_tilt_rad": summary.get(
            "agile_command_hold_final_tilt_escape_box_tilt_rad"
        ),
        "agile_command_hold_final_tilt_escape_suppress_after_target_window_streak": summary.get(
            "agile_command_hold_final_tilt_escape_suppress_after_target_window_streak"
        ),
        "agile_command_hold_final_tilt_escape_active_steps": summary.get(
            "agile_command_hold_final_tilt_escape_active_steps"
        ),
        "agile_command_hold_final_tilt_escape_first_active_step": summary.get(
            "agile_command_hold_final_tilt_escape_first_active_step"
        ),
        "agile_command_hold_final_tilt_escape_max_scale": summary.get(
            "agile_command_hold_final_tilt_escape_max_scale"
        ),
        "agile_command_hold_final_tilt_escape_suppressed_by_target_window_steps": summary.get(
            "agile_command_hold_final_tilt_escape_suppressed_by_target_window_steps"
        ),
        "agile_command_hold_final_brake_command_x": summary.get(
            "agile_command_hold_final_brake_command_x"
        ),
        "agile_command_hold_final_brake_delay_steps": summary.get(
            "agile_command_hold_final_brake_delay_steps"
        ),
        "agile_command_hold_final_brake_steps": summary.get(
            "agile_command_hold_final_brake_steps"
        ),
        "agile_command_hold_final_brake_active_steps": summary.get(
            "agile_command_hold_final_brake_active_steps"
        ),
        "agile_command_hold_final_brake_first_active_step": summary.get(
            "agile_command_hold_final_brake_first_active_step"
        ),
        "agile_command_hold_final_brake_last_active_step": summary.get(
            "agile_command_hold_final_brake_last_active_step"
        ),
        "agile_command_hold_final_brake_max_abs_command_x": summary.get(
            "agile_command_hold_final_brake_max_abs_command_x"
        ),
        "agile_command_hold_stand_target_overrides": summary.get(
            "agile_command_hold_stand_target_overrides"
        ),
        "agile_command_hold_applied_stand_joint_targets": summary.get(
            "agile_command_hold_applied_stand_joint_targets"
        ),
        "agile_command_hold_adaptive_scale_enabled": summary.get(
            "agile_command_hold_adaptive_scale_enabled"
        ),
        "agile_command_hold_adaptive_box_tilt_enabled": summary.get(
            "agile_command_hold_adaptive_box_tilt_enabled"
        ),
        "agile_command_hold_adaptive_min_observed_scale": summary.get(
            "agile_command_hold_adaptive_min_observed_scale"
        ),
        "agile_command_hold_adaptive_max_observed_scale": summary.get(
            "agile_command_hold_adaptive_max_observed_scale"
        ),
        "agile_command_hold_adaptive_last_risk": summary.get("agile_command_hold_adaptive_last_risk"),
        "agile_command_hold_lateral_max_abs_command": summary.get(
            "agile_command_hold_lateral_max_abs_command"
        ),
        "agile_command_hold_lateral_terminal_only": summary.get(
            "agile_command_hold_lateral_terminal_only"
        ),
        "agile_command_hold_lateral_error_start_m": summary.get(
            "agile_command_hold_lateral_error_start_m"
        ),
        "agile_command_hold_lateral_use_excess_error": summary.get(
            "agile_command_hold_lateral_use_excess_error"
        ),
        "agile_command_hold_lateral_max_tilt_rad": summary.get(
            "agile_command_hold_lateral_max_tilt_rad"
        ),
        "agile_command_hold_lateral_max_box_tilt_rad": summary.get(
            "agile_command_hold_lateral_max_box_tilt_rad"
        ),
        "agile_command_hold_lateral_suppressed_by_tilt_steps": summary.get(
            "agile_command_hold_lateral_suppressed_by_tilt_steps"
        ),
        "agile_command_stop_target_window_enabled": summary.get(
            "agile_command_stop_target_window_enabled"
        ),
        "agile_command_stop_target_window_min_step": summary.get(
            "agile_command_stop_target_window_min_step"
        ),
        "agile_command_stop_target_window_latched_step": summary.get(
            "agile_command_stop_target_window_latched_step"
        ),
        "agile_command_hold_terminal_latch_enabled": summary.get(
            "agile_command_hold_terminal_latch_enabled"
        ),
        "agile_command_hold_terminal_latched": summary.get(
            "agile_command_hold_terminal_latched"
        ),
        "agile_command_hold_terminal_latched_step": summary.get(
            "agile_command_hold_terminal_latched_step"
        ),
        "agile_command_hold_final_box_target_travel_m": summary.get(
            "agile_command_hold_final_box_target_travel_m"
        ),
        "agile_command_hold_final_scale": summary.get("agile_command_hold_final_scale"),
        "agile_command_hold_final_latch_enabled": summary.get(
            "agile_command_hold_final_latch_enabled"
        ),
        "agile_command_hold_final_zero_corrections_enabled": summary.get(
            "agile_command_hold_final_zero_corrections_enabled"
        ),
        "agile_command_hold_final_lateral_suppressed_steps": summary.get(
            "agile_command_hold_final_lateral_suppressed_steps"
        ),
        "agile_command_hold_final_yaw_suppressed_steps": summary.get(
            "agile_command_hold_final_yaw_suppressed_steps"
        ),
        "agile_command_hold_final_max_abs_command_x": summary.get(
            "agile_command_hold_final_max_abs_command_x"
        ),
        "agile_command_hold_final_max_abs_command_y": summary.get(
            "agile_command_hold_final_max_abs_command_y"
        ),
        "agile_command_hold_final_max_abs_command_yaw": summary.get(
            "agile_command_hold_final_max_abs_command_yaw"
        ),
        "agile_command_hold_final_last_command_xyz_yaw": summary.get(
            "agile_command_hold_final_last_command_xyz_yaw"
        ),
        "agile_command_hold_final_freeze_in_target_window": summary.get(
            "agile_command_hold_final_freeze_in_target_window"
        ),
        "agile_command_hold_final_freeze_latched": summary.get(
            "agile_command_hold_final_freeze_latched"
        ),
        "agile_command_hold_final_freeze_latched_step": summary.get(
            "agile_command_hold_final_freeze_latched_step"
        ),
        "agile_command_hold_final_freeze_active_steps": summary.get(
            "agile_command_hold_final_freeze_active_steps"
        ),
        "agile_command_hold_final_freeze_first_active_step": summary.get(
            "agile_command_hold_final_freeze_first_active_step"
        ),
        "agile_command_hold_rescue_overrides_final_freeze": summary.get(
            "agile_command_hold_rescue_overrides_final_freeze"
        ),
        "agile_command_hold_rescue_override_freeze_active_steps": summary.get(
            "agile_command_hold_rescue_override_freeze_active_steps"
        ),
        "agile_command_hold_rescue_override_freeze_first_active_step": summary.get(
            "agile_command_hold_rescue_override_freeze_first_active_step"
        ),
        "agile_command_hold_stand_overrides_final_freeze": summary.get(
            "agile_command_hold_stand_overrides_final_freeze"
        ),
        "agile_command_hold_stand_override_freeze_active_steps": summary.get(
            "agile_command_hold_stand_override_freeze_active_steps"
        ),
        "agile_command_hold_stand_override_freeze_first_active_step": summary.get(
            "agile_command_hold_stand_override_freeze_first_active_step"
        ),
        "agile_command_hold_final_stand_enabled": summary.get(
            "agile_command_hold_final_stand_enabled"
        ),
        "agile_command_hold_final_stand_delay_steps": summary.get(
            "agile_command_hold_final_stand_delay_steps"
        ),
        "agile_command_hold_final_latched": summary.get("agile_command_hold_final_latched"),
        "agile_command_hold_final_latched_step": summary.get(
            "agile_command_hold_final_latched_step"
        ),
        "agile_command_hold_final_active_steps": summary.get(
            "agile_command_hold_final_active_steps"
        ),
        "agile_command_hold_final_stand_active_steps": summary.get(
            "agile_command_hold_final_stand_active_steps"
        ),
        "agile_command_hold_final_stand_first_active_step": summary.get(
            "agile_command_hold_final_stand_first_active_step"
        ),
        "cradle_top_lid_enabled": summary.get("cradle_top_lid_enabled"),
        "cradle_top_lid_collision_enabled_step": summary.get("cradle_top_lid_collision_enabled_step"),
        "cradle_chest_pad_enabled": summary.get("cradle_chest_pad_enabled"),
        "cradle_chest_pad_collision_enabled_step": summary.get(
            "cradle_chest_pad_collision_enabled_step"
        ),
    }


def main() -> int:
    args = parse_args()
    paths = _summary_paths(args)
    cases = [_case_record(path) for path in paths]
    passed = [case for case in cases if bool(case["passed"])]
    failed = [case for case in cases if not bool(case["passed"])]
    status = "pass" if cases and not failed else "fail"
    report = {
        "scene_type": "core_world_g1_largerbox_strict_summary",
        "success_claim": "diagnostic_summary_only_not_final_carrying_success",
        "case_count": len(cases),
        "passed_case_count": len(passed),
        "failed_case_count": len(failed),
        "status": status,
        "cases": cases,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
