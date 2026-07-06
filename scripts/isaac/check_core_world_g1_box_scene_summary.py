#!/usr/bin/env python3
"""Check the direct Core API G1 + box scene summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Core API G1 box-scene diagnostic summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--expect-attach-box", default=None)
    parser.add_argument("--expect-torso-cradle", default=None)
    parser.add_argument("--expect-probe-mode", default=None)
    parser.add_argument("--expect-grasp-mode", default=None)
    parser.add_argument("--expect-grasp-body-path", default=None)
    parser.add_argument("--expect-active-grasp-body-path", default=None)
    parser.add_argument("--expect-box-support-mode", default=None)
    parser.add_argument("--expect-gait-mode", default=None)
    parser.add_argument("--expect-carry-box-spawned", choices=("true", "false"), default=None)
    parser.add_argument("--expect-box-collision-enabled", choices=("true", "false"), default=None)
    parser.add_argument("--expect-cradle-collision-enabled", choices=("true", "false"), default=None)
    parser.add_argument("--require-box-support-released", action="store_true")
    parser.add_argument("--min-box-support-actual-release-step", type=int, default=None)
    parser.add_argument("--min-cradle-piece-count", type=int, default=None)
    parser.add_argument("--min-joint-count", type=int, default=None)
    parser.add_argument("--require-stand-drive-gains", action="store_true")
    parser.add_argument("--min-stand-drive-gain-count", type=int, default=None)
    parser.add_argument("--max-fall-events", type=int, default=0)
    parser.add_argument("--max-box-drop-events", type=int, default=0)
    parser.add_argument("--max-final-hold-fall-events", type=int, default=None)
    parser.add_argument("--max-final-hold-box-drop-events", type=int, default=None)
    parser.add_argument("--max-final-stand-fall-events", type=int, default=None)
    parser.add_argument("--max-final-stand-box-drop-events", type=int, default=None)
    parser.add_argument("--min-robot-z", type=float, default=None)
    parser.add_argument("--min-box-z", type=float, default=None)
    parser.add_argument("--max-tilt", type=float, default=None)
    parser.add_argument("--max-box-tilt", type=float, default=None)
    parser.add_argument("--min-final-hold-robot-z", type=float, default=None)
    parser.add_argument("--min-final-hold-box-z", type=float, default=None)
    parser.add_argument("--max-final-hold-tilt", type=float, default=None)
    parser.add_argument("--max-final-hold-box-tilt", type=float, default=None)
    parser.add_argument("--min-final-stand-robot-z", type=float, default=None)
    parser.add_argument("--min-final-stand-box-z", type=float, default=None)
    parser.add_argument("--max-final-stand-tilt", type=float, default=None)
    parser.add_argument("--max-final-stand-box-tilt", type=float, default=None)
    parser.add_argument("--max-box-robot-relative-offset-error", type=float, default=None)
    parser.add_argument("--max-final-box-robot-relative-offset-error", type=float, default=None)
    parser.add_argument("--min-final-robot-target-directed-travel", type=float, default=None)
    parser.add_argument("--min-final-box-target-directed-travel", type=float, default=None)
    parser.add_argument("--max-final-robot-target-directed-travel", type=float, default=None)
    parser.add_argument("--max-final-box-target-directed-travel", type=float, default=None)
    parser.add_argument("--min-max-robot-target-directed-travel", type=float, default=None)
    parser.add_argument("--min-max-box-target-directed-travel", type=float, default=None)
    parser.add_argument("--max-robot-target-lateral-error", type=float, default=None)
    parser.add_argument("--max-box-target-lateral-error", type=float, default=None)
    parser.add_argument("--max-final-robot-target-lateral-error", type=float, default=None)
    parser.add_argument("--max-final-box-target-lateral-error", type=float, default=None)
    parser.add_argument("--min-target-window-robot-stable-steps", type=int, default=None)
    parser.add_argument("--min-target-window-box-stable-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-stable-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-longest-streak-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-streak-at-end-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-final-hold-stable-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-final-hold-longest-streak-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-final-hold-streak-at-end-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-final-stand-stable-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-final-stand-longest-streak-steps", type=int, default=None)
    parser.add_argument("--min-target-window-both-final-stand-streak-at-end-steps", type=int, default=None)
    parser.add_argument("--min-probe-active-steps", type=int, default=None)
    parser.add_argument("--min-final-probe-box-travel", type=float, default=None)
    parser.add_argument("--min-max-probe-box-travel", type=float, default=None)
    parser.add_argument("--min-final-probe-box-target-directed-travel", type=float, default=None)
    parser.add_argument("--require-probe-box-moved", action="store_true")
    parser.add_argument("--require-grasp-attached", action="store_true")
    parser.add_argument("--min-grasp-attach-step", type=int, default=None)
    parser.add_argument("--min-final-post-grasp-box-z-delta", type=float, default=None)
    parser.add_argument("--min-max-post-grasp-box-z-delta", type=float, default=None)
    parser.add_argument("--max-grasp-body-box-world-distance-at-attach", type=float, default=None)
    parser.add_argument("--max-root-pose-write-count-rollout", type=int, default=0)
    parser.add_argument("--max-root-velocity-write-count-rollout", type=int, default=0)
    parser.add_argument("--max-box-pose-write-count-rollout", type=int, default=0)
    parser.add_argument("--expect-diagnostic-root-drive", default=None)
    parser.add_argument("--min-diagnostic-root-drive-active-steps", type=int, default=None)
    parser.add_argument("--min-balance-target-active-steps", type=int, default=None)
    parser.add_argument("--min-agile-command-hold-final-active-steps", type=int, default=None)
    parser.add_argument("--min-agile-command-hold-final-stand-active-steps", type=int, default=None)
    parser.add_argument("--max-final-hold-command-x", type=float, default=None)
    parser.add_argument("--max-final-hold-command-y", type=float, default=None)
    parser.add_argument("--max-final-hold-command-yaw", type=float, default=None)
    parser.add_argument("--require-diagnostic-claim", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    if int(summary.get("completed_steps") or 0) < int(args.min_steps):
        failures.append(f"completed_steps {summary.get('completed_steps')} < {args.min_steps}")
    if args.expect_attach_box is not None and summary.get("attach_box") != args.expect_attach_box:
        failures.append(f"attach_box {summary.get('attach_box')} != {args.expect_attach_box}")
    if args.expect_torso_cradle is not None and summary.get("torso_cradle") != args.expect_torso_cradle:
        failures.append(f"torso_cradle {summary.get('torso_cradle')} != {args.expect_torso_cradle}")
    if args.expect_probe_mode is not None and summary.get("probe_mode") != args.expect_probe_mode:
        failures.append(f"probe_mode {summary.get('probe_mode')} != {args.expect_probe_mode}")
    if args.expect_grasp_mode is not None and summary.get("grasp_mode") != args.expect_grasp_mode:
        failures.append(f"grasp_mode {summary.get('grasp_mode')} != {args.expect_grasp_mode}")
    if args.expect_grasp_body_path is not None and summary.get("grasp_body_path") != args.expect_grasp_body_path:
        failures.append(f"grasp_body_path {summary.get('grasp_body_path')} != {args.expect_grasp_body_path}")
    if args.expect_active_grasp_body_path is not None and summary.get("active_grasp_body_path") != args.expect_active_grasp_body_path:
        failures.append(
            f"active_grasp_body_path {summary.get('active_grasp_body_path')} != {args.expect_active_grasp_body_path}"
        )
    if args.expect_box_support_mode is not None and summary.get("box_support_mode") != args.expect_box_support_mode:
        failures.append(f"box_support_mode {summary.get('box_support_mode')} != {args.expect_box_support_mode}")
    if args.expect_gait_mode is not None and summary.get("gait_mode") != args.expect_gait_mode:
        failures.append(f"gait_mode {summary.get('gait_mode')} != {args.expect_gait_mode}")
    if args.require_box_support_released and not bool(summary.get("box_support_released")):
        failures.append(f"box_support_released is false or missing: {summary.get('box_support_released')}")
    if args.min_box_support_actual_release_step is not None:
        value = summary.get("box_support_actual_release_step")
        if value is None or int(value) < int(args.min_box_support_actual_release_step):
            failures.append(f"box_support_actual_release_step {value} < {args.min_box_support_actual_release_step}")
    if args.expect_carry_box_spawned is not None:
        expected_box = args.expect_carry_box_spawned == "true"
        if bool(summary.get("carry_box_spawned")) != expected_box:
            failures.append(f"carry_box_spawned {summary.get('carry_box_spawned')} != {expected_box}")
    if args.expect_box_collision_enabled is not None:
        expected_box_collision = args.expect_box_collision_enabled == "true"
        if bool(summary.get("box_collision_enabled")) != expected_box_collision:
            failures.append(f"box_collision_enabled {summary.get('box_collision_enabled')} != {expected_box_collision}")
    if args.expect_cradle_collision_enabled is not None:
        expected_cradle_collision = args.expect_cradle_collision_enabled == "true"
        if bool(summary.get("cradle_collision_enabled")) != expected_cradle_collision:
            failures.append(f"cradle_collision_enabled {summary.get('cradle_collision_enabled')} != {expected_cradle_collision}")
    if args.min_joint_count is not None and int(summary.get("joint_count") or 0) < int(args.min_joint_count):
        failures.append(f"joint_count {summary.get('joint_count')} < {args.min_joint_count}")
    if args.min_cradle_piece_count is not None and int(summary.get("cradle_piece_count") or 0) < int(args.min_cradle_piece_count):
        failures.append(f"cradle_piece_count {summary.get('cradle_piece_count')} < {args.min_cradle_piece_count}")
    if args.require_stand_drive_gains and not bool(summary.get("stand_drive_gains_enabled")):
        failures.append("stand_drive_gains_enabled is false")
    if args.min_stand_drive_gain_count is not None:
        count = int(summary.get("applied_stand_drive_gain_count") or 0)
        if count < int(args.min_stand_drive_gain_count):
            failures.append(f"applied_stand_drive_gain_count {count} < {args.min_stand_drive_gain_count}")
    if int(summary.get("fall_events") or 0) > int(args.max_fall_events):
        failures.append(f"fall_events {summary.get('fall_events')} > {args.max_fall_events}")
    if int(summary.get("box_drop_events") or 0) > int(args.max_box_drop_events):
        failures.append(f"box_drop_events {summary.get('box_drop_events')} > {args.max_box_drop_events}")
    if args.max_final_hold_fall_events is not None:
        value = int(summary.get("agile_command_hold_final_fall_events") or 0)
        if value > int(args.max_final_hold_fall_events):
            failures.append(f"agile_command_hold_final_fall_events {value} > {args.max_final_hold_fall_events}")
    if args.max_final_hold_box_drop_events is not None:
        value = int(summary.get("agile_command_hold_final_box_drop_events") or 0)
        if value > int(args.max_final_hold_box_drop_events):
            failures.append(
                f"agile_command_hold_final_box_drop_events {value} > {args.max_final_hold_box_drop_events}"
            )
    if args.max_final_stand_fall_events is not None:
        value = int(summary.get("agile_command_hold_final_stand_fall_events") or 0)
        if value > int(args.max_final_stand_fall_events):
            failures.append(
                f"agile_command_hold_final_stand_fall_events {value} > {args.max_final_stand_fall_events}"
            )
    if args.max_final_stand_box_drop_events is not None:
        value = int(summary.get("agile_command_hold_final_stand_box_drop_events") or 0)
        if value > int(args.max_final_stand_box_drop_events):
            failures.append(
                "agile_command_hold_final_stand_box_drop_events "
                f"{value} > {args.max_final_stand_box_drop_events}"
            )
    if args.min_robot_z is not None and float(summary.get("min_robot_z_m") or -999.0) < float(args.min_robot_z):
        failures.append(f"min_robot_z_m {summary.get('min_robot_z_m')} < {args.min_robot_z}")
    if args.min_box_z is not None and float(summary.get("min_box_z_m") or -999.0) < float(args.min_box_z):
        failures.append(f"min_box_z_m {summary.get('min_box_z_m')} < {args.min_box_z}")
    if args.max_tilt is not None and float(summary.get("max_tilt_rad") or 999.0) > float(args.max_tilt):
        failures.append(f"max_tilt_rad {summary.get('max_tilt_rad')} > {args.max_tilt}")
    if args.max_box_tilt is not None and float(summary.get("max_box_tilt_rad") or 999.0) > float(args.max_box_tilt):
        failures.append(f"max_box_tilt_rad {summary.get('max_box_tilt_rad')} > {args.max_box_tilt}")
    if args.min_final_hold_robot_z is not None:
        value = summary.get("agile_command_hold_final_min_robot_z_m")
        if value is None or float(value) < float(args.min_final_hold_robot_z):
            failures.append(f"agile_command_hold_final_min_robot_z_m {value} < {args.min_final_hold_robot_z}")
    if args.min_final_hold_box_z is not None:
        value = summary.get("agile_command_hold_final_min_box_z_m")
        if value is None or float(value) < float(args.min_final_hold_box_z):
            failures.append(f"agile_command_hold_final_min_box_z_m {value} < {args.min_final_hold_box_z}")
    if args.max_final_hold_tilt is not None:
        value = float(summary.get("agile_command_hold_final_max_tilt_rad") or 999.0)
        if value > float(args.max_final_hold_tilt):
            failures.append(f"agile_command_hold_final_max_tilt_rad {value} > {args.max_final_hold_tilt}")
    if args.max_final_hold_box_tilt is not None:
        value = float(summary.get("agile_command_hold_final_max_box_tilt_rad") or 999.0)
        if value > float(args.max_final_hold_box_tilt):
            failures.append(
                f"agile_command_hold_final_max_box_tilt_rad {value} > {args.max_final_hold_box_tilt}"
            )
    if args.min_final_stand_robot_z is not None:
        value = summary.get("agile_command_hold_final_stand_min_robot_z_m")
        if value is None or float(value) < float(args.min_final_stand_robot_z):
            failures.append(
                f"agile_command_hold_final_stand_min_robot_z_m {value} < {args.min_final_stand_robot_z}"
            )
    if args.min_final_stand_box_z is not None:
        value = summary.get("agile_command_hold_final_stand_min_box_z_m")
        if value is None or float(value) < float(args.min_final_stand_box_z):
            failures.append(
                f"agile_command_hold_final_stand_min_box_z_m {value} < {args.min_final_stand_box_z}"
            )
    if args.max_final_stand_tilt is not None:
        value = float(summary.get("agile_command_hold_final_stand_max_tilt_rad") or 999.0)
        if value > float(args.max_final_stand_tilt):
            failures.append(
                f"agile_command_hold_final_stand_max_tilt_rad {value} > {args.max_final_stand_tilt}"
            )
    if args.max_final_stand_box_tilt is not None:
        value = float(summary.get("agile_command_hold_final_stand_max_box_tilt_rad") or 999.0)
        if value > float(args.max_final_stand_box_tilt):
            failures.append(
                "agile_command_hold_final_stand_max_box_tilt_rad "
                f"{value} > {args.max_final_stand_box_tilt}"
            )
    if args.max_box_robot_relative_offset_error is not None:
        rel_error = float(summary.get("max_box_robot_relative_offset_error_m") or 999.0)
        if rel_error > float(args.max_box_robot_relative_offset_error):
            failures.append(
                "max_box_robot_relative_offset_error_m "
                f"{summary.get('max_box_robot_relative_offset_error_m')} > {args.max_box_robot_relative_offset_error}"
            )
    if args.max_final_box_robot_relative_offset_error is not None:
        rel_error = summary.get("final_box_robot_relative_offset_error_m")
        if rel_error is None or float(rel_error) > float(args.max_final_box_robot_relative_offset_error):
            failures.append(
                "final_box_robot_relative_offset_error_m "
                f"{rel_error} > {args.max_final_box_robot_relative_offset_error}"
            )
    if args.min_final_robot_target_directed_travel is not None:
        value = float(summary.get("final_robot_target_directed_travel_m") or 0.0)
        if value < float(args.min_final_robot_target_directed_travel):
            failures.append(f"final_robot_target_directed_travel_m {value} < {args.min_final_robot_target_directed_travel}")
    if args.min_final_box_target_directed_travel is not None:
        value = float(summary.get("final_box_target_directed_travel_m") or 0.0)
        if value < float(args.min_final_box_target_directed_travel):
            failures.append(f"final_box_target_directed_travel_m {value} < {args.min_final_box_target_directed_travel}")
    if args.max_final_robot_target_directed_travel is not None:
        value = float(summary.get("final_robot_target_directed_travel_m") or 0.0)
        if value > float(args.max_final_robot_target_directed_travel):
            failures.append(f"final_robot_target_directed_travel_m {value} > {args.max_final_robot_target_directed_travel}")
    if args.max_final_box_target_directed_travel is not None:
        value = float(summary.get("final_box_target_directed_travel_m") or 0.0)
        if value > float(args.max_final_box_target_directed_travel):
            failures.append(f"final_box_target_directed_travel_m {value} > {args.max_final_box_target_directed_travel}")
    if args.min_max_robot_target_directed_travel is not None:
        value = float(summary.get("max_robot_target_directed_travel_m") or 0.0)
        if value < float(args.min_max_robot_target_directed_travel):
            failures.append(f"max_robot_target_directed_travel_m {value} < {args.min_max_robot_target_directed_travel}")
    if args.min_max_box_target_directed_travel is not None:
        value = float(summary.get("max_box_target_directed_travel_m") or 0.0)
        if value < float(args.min_max_box_target_directed_travel):
            failures.append(f"max_box_target_directed_travel_m {value} < {args.min_max_box_target_directed_travel}")
    if args.max_robot_target_lateral_error is not None:
        value = float(summary.get("max_abs_robot_target_lateral_error_m") or 999.0)
        if value > float(args.max_robot_target_lateral_error):
            failures.append(
                f"max_abs_robot_target_lateral_error_m {value} > {args.max_robot_target_lateral_error}"
            )
    if args.max_box_target_lateral_error is not None:
        value = float(summary.get("max_abs_box_target_lateral_error_m") or 999.0)
        if value > float(args.max_box_target_lateral_error):
            failures.append(
                f"max_abs_box_target_lateral_error_m {value} > {args.max_box_target_lateral_error}"
            )
    if args.max_final_robot_target_lateral_error is not None:
        value = abs(float(summary.get("final_robot_target_lateral_error_m") or 0.0))
        if value > float(args.max_final_robot_target_lateral_error):
            failures.append(
                f"abs(final_robot_target_lateral_error_m) {value} > {args.max_final_robot_target_lateral_error}"
            )
    if args.max_final_box_target_lateral_error is not None:
        raw_value = summary.get("final_box_target_lateral_error_m")
        value = 999.0 if raw_value is None else abs(float(raw_value))
        if value > float(args.max_final_box_target_lateral_error):
            failures.append(
                f"abs(final_box_target_lateral_error_m) {value} > {args.max_final_box_target_lateral_error}"
            )
    if args.min_target_window_robot_stable_steps is not None:
        value = int(summary.get("target_window_robot_stable_steps") or 0)
        if value < int(args.min_target_window_robot_stable_steps):
            failures.append(f"target_window_robot_stable_steps {value} < {args.min_target_window_robot_stable_steps}")
    if args.min_target_window_box_stable_steps is not None:
        value = int(summary.get("target_window_box_stable_steps") or 0)
        if value < int(args.min_target_window_box_stable_steps):
            failures.append(f"target_window_box_stable_steps {value} < {args.min_target_window_box_stable_steps}")
    if args.min_target_window_both_stable_steps is not None:
        value = int(summary.get("target_window_both_stable_steps") or 0)
        if value < int(args.min_target_window_both_stable_steps):
            failures.append(f"target_window_both_stable_steps {value} < {args.min_target_window_both_stable_steps}")
    if args.min_target_window_both_longest_streak_steps is not None:
        value = int(summary.get("target_window_both_longest_streak_steps") or 0)
        if value < int(args.min_target_window_both_longest_streak_steps):
            failures.append(
                "target_window_both_longest_streak_steps "
                f"{value} < {args.min_target_window_both_longest_streak_steps}"
            )
    if args.min_target_window_both_streak_at_end_steps is not None:
        value = int(summary.get("target_window_both_streak_at_end_steps") or 0)
        if value < int(args.min_target_window_both_streak_at_end_steps):
            failures.append(
                "target_window_both_streak_at_end_steps "
                f"{value} < {args.min_target_window_both_streak_at_end_steps}"
            )
    if args.min_target_window_both_final_hold_stable_steps is not None:
        value = int(summary.get("target_window_both_final_hold_stable_steps") or 0)
        if value < int(args.min_target_window_both_final_hold_stable_steps):
            failures.append(
                "target_window_both_final_hold_stable_steps "
                f"{value} < {args.min_target_window_both_final_hold_stable_steps}"
            )
    if args.min_target_window_both_final_hold_longest_streak_steps is not None:
        value = int(summary.get("target_window_both_final_hold_longest_streak_steps") or 0)
        if value < int(args.min_target_window_both_final_hold_longest_streak_steps):
            failures.append(
                "target_window_both_final_hold_longest_streak_steps "
                f"{value} < {args.min_target_window_both_final_hold_longest_streak_steps}"
            )
    if args.min_target_window_both_final_hold_streak_at_end_steps is not None:
        value = int(summary.get("target_window_both_final_hold_streak_at_end_steps") or 0)
        if value < int(args.min_target_window_both_final_hold_streak_at_end_steps):
            failures.append(
                "target_window_both_final_hold_streak_at_end_steps "
                f"{value} < {args.min_target_window_both_final_hold_streak_at_end_steps}"
            )
    if args.min_target_window_both_final_stand_stable_steps is not None:
        value = int(summary.get("target_window_both_final_stand_stable_steps") or 0)
        if value < int(args.min_target_window_both_final_stand_stable_steps):
            failures.append(
                "target_window_both_final_stand_stable_steps "
                f"{value} < {args.min_target_window_both_final_stand_stable_steps}"
            )
    if args.min_target_window_both_final_stand_longest_streak_steps is not None:
        value = int(summary.get("target_window_both_final_stand_longest_streak_steps") or 0)
        if value < int(args.min_target_window_both_final_stand_longest_streak_steps):
            failures.append(
                "target_window_both_final_stand_longest_streak_steps "
                f"{value} < {args.min_target_window_both_final_stand_longest_streak_steps}"
            )
    if args.min_target_window_both_final_stand_streak_at_end_steps is not None:
        value = int(summary.get("target_window_both_final_stand_streak_at_end_steps") or 0)
        if value < int(args.min_target_window_both_final_stand_streak_at_end_steps):
            failures.append(
                "target_window_both_final_stand_streak_at_end_steps "
                f"{value} < {args.min_target_window_both_final_stand_streak_at_end_steps}"
            )
    if args.min_probe_active_steps is not None:
        value = int(summary.get("probe_active_steps") or 0)
        if value < int(args.min_probe_active_steps):
            failures.append(f"probe_active_steps {value} < {args.min_probe_active_steps}")
    if args.min_final_probe_box_travel is not None:
        value = float(summary.get("final_probe_box_travel_xy_m") or 0.0)
        if value < float(args.min_final_probe_box_travel):
            failures.append(f"final_probe_box_travel_xy_m {value} < {args.min_final_probe_box_travel}")
    if args.min_max_probe_box_travel is not None:
        value = float(summary.get("max_probe_box_travel_xy_m") or 0.0)
        if value < float(args.min_max_probe_box_travel):
            failures.append(f"max_probe_box_travel_xy_m {value} < {args.min_max_probe_box_travel}")
    if args.min_final_probe_box_target_directed_travel is not None:
        value = float(summary.get("final_probe_box_target_directed_travel_m") or 0.0)
        if value < float(args.min_final_probe_box_target_directed_travel):
            failures.append(
                "final_probe_box_target_directed_travel_m "
                f"{value} < {args.min_final_probe_box_target_directed_travel}"
            )
    if args.require_probe_box_moved and not bool(summary.get("probe_box_moved")):
        failures.append(f"probe_box_moved is false or missing: {summary.get('probe_box_moved')}")
    if args.require_grasp_attached and not bool(summary.get("grasp_attached")):
        failures.append(f"grasp_attached is false or missing: {summary.get('grasp_attached')}")
    if args.min_grasp_attach_step is not None:
        value = summary.get("grasp_attach_step")
        if value is None or int(value) < int(args.min_grasp_attach_step):
            failures.append(f"grasp_attach_step {value} < {args.min_grasp_attach_step}")
    if args.min_final_post_grasp_box_z_delta is not None:
        value = float(summary.get("final_post_grasp_box_z_delta_m") or 0.0)
        if value < float(args.min_final_post_grasp_box_z_delta):
            failures.append(
                f"final_post_grasp_box_z_delta_m {value} < {args.min_final_post_grasp_box_z_delta}"
            )
    if args.min_max_post_grasp_box_z_delta is not None:
        value = float(summary.get("max_post_grasp_box_z_delta_m") or 0.0)
        if value < float(args.min_max_post_grasp_box_z_delta):
            failures.append(
                f"max_post_grasp_box_z_delta_m {value} < {args.min_max_post_grasp_box_z_delta}"
            )
    if args.max_grasp_body_box_world_distance_at_attach is not None:
        value = summary.get("grasp_body_box_world_distance_at_attach_m")
        if value is None or float(value) > float(args.max_grasp_body_box_world_distance_at_attach):
            failures.append(
                "grasp_body_box_world_distance_at_attach_m "
                f"{value} > {args.max_grasp_body_box_world_distance_at_attach}"
            )
    if int(summary.get("root_pose_write_count_rollout") or 0) > int(args.max_root_pose_write_count_rollout):
        failures.append("root_pose_write_count_rollout exceeded limit")
    if int(summary.get("root_velocity_write_count_rollout") or 0) > int(args.max_root_velocity_write_count_rollout):
        failures.append("root_velocity_write_count_rollout exceeded limit")
    if int(summary.get("box_pose_write_count_rollout") or 0) > int(args.max_box_pose_write_count_rollout):
        failures.append("box_pose_write_count_rollout exceeded limit")
    if args.expect_diagnostic_root_drive is not None and summary.get("diagnostic_root_drive") != args.expect_diagnostic_root_drive:
        failures.append(f"diagnostic_root_drive {summary.get('diagnostic_root_drive')} != {args.expect_diagnostic_root_drive}")
    if args.min_diagnostic_root_drive_active_steps is not None:
        value = int(summary.get("diagnostic_root_drive_active_steps") or 0)
        if value < int(args.min_diagnostic_root_drive_active_steps):
            failures.append(
                f"diagnostic_root_drive_active_steps {value} < {args.min_diagnostic_root_drive_active_steps}"
            )
    if args.min_balance_target_active_steps is not None:
        value = int(summary.get("balance_target_active_steps") or 0)
        if value < int(args.min_balance_target_active_steps):
            failures.append(f"balance_target_active_steps {value} < {args.min_balance_target_active_steps}")
    if args.min_agile_command_hold_final_active_steps is not None:
        value = int(summary.get("agile_command_hold_final_active_steps") or 0)
        if value < int(args.min_agile_command_hold_final_active_steps):
            failures.append(
                "agile_command_hold_final_active_steps "
                f"{value} < {args.min_agile_command_hold_final_active_steps}"
            )
    if args.min_agile_command_hold_final_stand_active_steps is not None:
        value = int(summary.get("agile_command_hold_final_stand_active_steps") or 0)
        if value < int(args.min_agile_command_hold_final_stand_active_steps):
            failures.append(
                "agile_command_hold_final_stand_active_steps "
                f"{value} < {args.min_agile_command_hold_final_stand_active_steps}"
            )
    if args.max_final_hold_command_x is not None:
        value = float(summary.get("agile_command_hold_final_max_abs_command_x") or 0.0)
        if value > float(args.max_final_hold_command_x):
            failures.append(f"agile_command_hold_final_max_abs_command_x {value} > {args.max_final_hold_command_x}")
    if args.max_final_hold_command_y is not None:
        value = float(summary.get("agile_command_hold_final_max_abs_command_y") or 0.0)
        if value > float(args.max_final_hold_command_y):
            failures.append(f"agile_command_hold_final_max_abs_command_y {value} > {args.max_final_hold_command_y}")
    if args.max_final_hold_command_yaw is not None:
        value = float(summary.get("agile_command_hold_final_max_abs_command_yaw") or 0.0)
        if value > float(args.max_final_hold_command_yaw):
            failures.append(
                f"agile_command_hold_final_max_abs_command_yaw {value} > {args.max_final_hold_command_yaw}"
            )
    if args.require_diagnostic_claim and "diagnostic" not in str(summary.get("success_claim", "")):
        failures.append(f"success_claim is not diagnostic-only: {summary.get('success_claim')}")
    if summary.get("error") is not None:
        failures.append(f"summary error: {summary.get('error')}")

    report = {
        "summary": str(args.summary),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "scene_type": summary.get("scene_type"),
        "success_claim": summary.get("success_claim"),
        "completed_steps": summary.get("completed_steps"),
        "attach_box": summary.get("attach_box"),
        "torso_cradle": summary.get("torso_cradle"),
        "probe_mode": summary.get("probe_mode"),
        "grasp_mode": summary.get("grasp_mode"),
        "grasp_body_path": summary.get("grasp_body_path"),
        "active_grasp_body_path": summary.get("active_grasp_body_path"),
        "grasp_body_wrapper_initialized": summary.get("grasp_body_wrapper_initialized"),
        "grasp_body_wrapper_error": summary.get("grasp_body_wrapper_error"),
        "grasp_attached": summary.get("grasp_attached"),
        "grasp_attach_step": summary.get("grasp_attach_step"),
        "grasp_lift_offset_z_m": summary.get("grasp_lift_offset_z_m"),
        "grasp_body_box_world_distance_at_attach_m": summary.get("grasp_body_box_world_distance_at_attach_m"),
        "max_post_grasp_box_z_delta_m": summary.get("max_post_grasp_box_z_delta_m"),
        "final_post_grasp_box_z_delta_m": summary.get("final_post_grasp_box_z_delta_m"),
        "probe_active_steps": summary.get("probe_active_steps"),
        "probe_box_moved": summary.get("probe_box_moved"),
        "max_probe_box_travel_xy_m": summary.get("max_probe_box_travel_xy_m"),
        "final_probe_box_travel_xy_m": summary.get("final_probe_box_travel_xy_m"),
        "max_probe_box_target_directed_travel_m": summary.get("max_probe_box_target_directed_travel_m"),
        "final_probe_box_target_directed_travel_m": summary.get("final_probe_box_target_directed_travel_m"),
        "carry_box_spawned": summary.get("carry_box_spawned"),
        "target_xy_m": summary.get("target_xy_m"),
        "box_support_mode": summary.get("box_support_mode"),
        "gait_mode": summary.get("gait_mode"),
        "policy_inference_count": summary.get("policy_inference_count"),
        "max_policy_raw_action_norm": summary.get("max_policy_raw_action_norm"),
        "agile_root_ang_vel_source": summary.get("agile_root_ang_vel_source"),
        "agile_root_ang_vel_read_failures": summary.get("agile_root_ang_vel_read_failures"),
        "agile_last_root_ang_vel_read_error": summary.get("agile_last_root_ang_vel_read_error"),
        "max_agile_root_ang_vel_norm": summary.get("max_agile_root_ang_vel_norm"),
        "agile_command_stop_step": summary.get("agile_command_stop_step"),
        "agile_command_stop_box_target_travel_m": summary.get("agile_command_stop_box_target_travel_m"),
        "agile_command_stop_robot_target_travel_m": summary.get("agile_command_stop_robot_target_travel_m"),
        "agile_command_hold_scale": summary.get("agile_command_hold_scale"),
        "agile_command_hold_adaptive_scale_enabled": summary.get("agile_command_hold_adaptive_scale_enabled"),
        "agile_command_hold_adaptive_min_scale": summary.get("agile_command_hold_adaptive_min_scale"),
        "agile_command_hold_adaptive_max_scale": summary.get("agile_command_hold_adaptive_max_scale"),
        "agile_command_hold_adaptive_tilt_start": summary.get("agile_command_hold_adaptive_tilt_start"),
        "agile_command_hold_adaptive_tilt_stop": summary.get("agile_command_hold_adaptive_tilt_stop"),
        "agile_command_hold_adaptive_rate_start": summary.get("agile_command_hold_adaptive_rate_start"),
        "agile_command_hold_adaptive_rate_stop": summary.get("agile_command_hold_adaptive_rate_stop"),
        "agile_command_hold_adaptive_rel_start": summary.get("agile_command_hold_adaptive_rel_start"),
        "agile_command_hold_adaptive_rel_stop": summary.get("agile_command_hold_adaptive_rel_stop"),
        "agile_command_hold_adaptive_box_tilt_enabled": summary.get(
            "agile_command_hold_adaptive_box_tilt_enabled"
        ),
        "agile_command_hold_adaptive_box_tilt_start": summary.get(
            "agile_command_hold_adaptive_box_tilt_start"
        ),
        "agile_command_hold_adaptive_box_tilt_stop": summary.get(
            "agile_command_hold_adaptive_box_tilt_stop"
        ),
        "agile_command_hold_adaptive_box_tilt_rate_start": summary.get(
            "agile_command_hold_adaptive_box_tilt_rate_start"
        ),
        "agile_command_hold_adaptive_box_tilt_rate_stop": summary.get(
            "agile_command_hold_adaptive_box_tilt_rate_stop"
        ),
        "agile_command_hold_adaptive_scale_smoothing": summary.get(
            "agile_command_hold_adaptive_scale_smoothing"
        ),
        "agile_command_hold_adaptive_active_steps": summary.get("agile_command_hold_adaptive_active_steps"),
        "agile_command_hold_adaptive_first_active_step": summary.get(
            "agile_command_hold_adaptive_first_active_step"
        ),
        "agile_command_hold_adaptive_min_observed_scale": summary.get(
            "agile_command_hold_adaptive_min_observed_scale"
        ),
        "agile_command_hold_adaptive_max_observed_scale": summary.get(
            "agile_command_hold_adaptive_max_observed_scale"
        ),
        "agile_command_hold_adaptive_last_risk": summary.get("agile_command_hold_adaptive_last_risk"),
        "agile_command_hold_lateral_correction_enabled": summary.get(
            "agile_command_hold_lateral_correction_enabled"
        ),
        "agile_command_hold_lateral_gain": summary.get("agile_command_hold_lateral_gain"),
        "agile_command_hold_lateral_limit": summary.get("agile_command_hold_lateral_limit"),
        "agile_command_hold_lateral_sign": summary.get("agile_command_hold_lateral_sign"),
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
        "agile_command_hold_lateral_active_steps": summary.get("agile_command_hold_lateral_active_steps"),
        "agile_command_hold_lateral_first_active_step": summary.get(
            "agile_command_hold_lateral_first_active_step"
        ),
        "agile_command_hold_lateral_max_abs_command": summary.get(
            "agile_command_hold_lateral_max_abs_command"
        ),
        "agile_command_hold_lateral_last_error_m": summary.get(
            "agile_command_hold_lateral_last_error_m"
        ),
        "agile_command_hold_yaw_correction_enabled": summary.get(
            "agile_command_hold_yaw_correction_enabled"
        ),
        "agile_command_hold_yaw_gain": summary.get("agile_command_hold_yaw_gain"),
        "agile_command_hold_yaw_limit": summary.get("agile_command_hold_yaw_limit"),
        "agile_command_hold_yaw_sign": summary.get("agile_command_hold_yaw_sign"),
        "agile_command_hold_yaw_active_steps": summary.get("agile_command_hold_yaw_active_steps"),
        "agile_command_hold_yaw_first_active_step": summary.get("agile_command_hold_yaw_first_active_step"),
        "agile_command_hold_yaw_max_abs_command": summary.get("agile_command_hold_yaw_max_abs_command"),
        "agile_command_hold_yaw_last_error_m": summary.get("agile_command_hold_yaw_last_error_m"),
        "agile_command_hold_terminal_box_target_travel_m": summary.get(
            "agile_command_hold_terminal_box_target_travel_m"
        ),
        "agile_command_hold_terminal_scale": summary.get("agile_command_hold_terminal_scale"),
        "agile_command_hold_terminal_latch_enabled": summary.get(
            "agile_command_hold_terminal_latch_enabled"
        ),
        "agile_command_hold_terminal_latched": summary.get("agile_command_hold_terminal_latched"),
        "agile_command_hold_terminal_latched_step": summary.get(
            "agile_command_hold_terminal_latched_step"
        ),
        "agile_command_hold_terminal_active_steps": summary.get(
            "agile_command_hold_terminal_active_steps"
        ),
        "agile_command_hold_terminal_first_active_step": summary.get(
            "agile_command_hold_terminal_first_active_step"
        ),
        "agile_command_hold_terminal_last_reason": summary.get(
            "agile_command_hold_terminal_last_reason"
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
        "agile_command_hold_final_reset_policy_state": summary.get(
            "agile_command_hold_final_reset_policy_state"
        ),
        "agile_command_hold_final_policy_state_reset_count": summary.get(
            "agile_command_hold_final_policy_state_reset_count"
        ),
        "agile_command_hold_final_last_policy_state_reset_error": summary.get(
            "agile_command_hold_final_last_policy_state_reset_error"
        ),
        "agile_command_hold_final_lateral_suppressed_steps": summary.get(
            "agile_command_hold_final_lateral_suppressed_steps"
        ),
        "agile_command_hold_final_yaw_suppressed_steps": summary.get(
            "agile_command_hold_final_yaw_suppressed_steps"
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
        "agile_command_hold_final_freeze_in_target_window": summary.get(
            "agile_command_hold_final_freeze_in_target_window"
        ),
        "agile_command_hold_final_freeze_max_tilt_rad": summary.get(
            "agile_command_hold_final_freeze_max_tilt_rad"
        ),
        "agile_command_hold_final_freeze_max_box_tilt_rad": summary.get(
            "agile_command_hold_final_freeze_max_box_tilt_rad"
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
        "agile_command_hold_final_first_active_step": summary.get(
            "agile_command_hold_final_first_active_step"
        ),
        "agile_command_hold_final_last_reason": summary.get(
            "agile_command_hold_final_last_reason"
        ),
        "agile_command_hold_final_stand_active_steps": summary.get(
            "agile_command_hold_final_stand_active_steps"
        ),
        "agile_command_hold_final_stand_first_active_step": summary.get(
            "agile_command_hold_final_stand_first_active_step"
        ),
        "agile_command_hold_mode": summary.get("agile_command_hold_mode"),
        "agile_command_hold_stand_blend_rate": summary.get("agile_command_hold_stand_blend_rate"),
        "agile_command_hold_policy_then_stand_delay_steps": summary.get(
            "agile_command_hold_policy_then_stand_delay_steps"
        ),
        "agile_command_hold_stand_target_overrides": summary.get("agile_command_hold_stand_target_overrides"),
        "agile_command_hold_applied_stand_joint_targets": summary.get("agile_command_hold_applied_stand_joint_targets"),
        "agile_command_hold_rescue_enabled": summary.get("agile_command_hold_rescue_enabled"),
        "agile_command_hold_rescue_forward_pitch_threshold_rad": summary.get(
            "agile_command_hold_rescue_forward_pitch_threshold_rad"
        ),
        "agile_command_hold_rescue_abs_roll_threshold_rad": summary.get(
            "agile_command_hold_rescue_abs_roll_threshold_rad"
        ),
        "agile_command_hold_rescue_blend_rate": summary.get("agile_command_hold_rescue_blend_rate"),
        "agile_command_hold_rescue_target_overrides": summary.get("agile_command_hold_rescue_target_overrides"),
        "agile_command_hold_applied_rescue_joint_targets": summary.get(
            "agile_command_hold_applied_rescue_joint_targets"
        ),
        "agile_command_hold_rescue_active": summary.get("agile_command_hold_rescue_active"),
        "agile_command_hold_rescue_first_active_step": summary.get(
            "agile_command_hold_rescue_first_active_step"
        ),
        "agile_command_hold_rescue_first_reason": summary.get("agile_command_hold_rescue_first_reason"),
        "agile_command_hold_rescue_active_steps": summary.get("agile_command_hold_rescue_active_steps"),
        "agile_command_hold_reset_policy_state": summary.get("agile_command_hold_reset_policy_state"),
        "agile_command_hold_active": summary.get("agile_command_hold_active"),
        "agile_command_hold_first_active_step": summary.get("agile_command_hold_first_active_step"),
        "agile_command_hold_first_reason": summary.get("agile_command_hold_first_reason"),
        "agile_command_hold_active_steps": summary.get("agile_command_hold_active_steps"),
        "agile_command_hold_stand_target_active_steps": summary.get("agile_command_hold_stand_target_active_steps"),
        "agile_command_hold_policy_state_reset_count": summary.get("agile_command_hold_policy_state_reset_count"),
        "agile_command_hold_last_policy_state_reset_error": summary.get("agile_command_hold_last_policy_state_reset_error"),
        "agile_last_command_xyz_yaw": summary.get("agile_last_command_xyz_yaw"),
        "box_support_released": summary.get("box_support_released"),
        "box_support_actual_release_step": summary.get("box_support_actual_release_step"),
        "cradle_piece_count": summary.get("cradle_piece_count"),
        "box_collision_enabled": summary.get("box_collision_enabled"),
        "cradle_collision_enabled": summary.get("cradle_collision_enabled"),
        "cradle_top_lid_enabled": summary.get("cradle_top_lid_enabled"),
        "cradle_top_lid_local_z_m": summary.get("cradle_top_lid_local_z_m"),
        "cradle_top_lid_thickness_m": summary.get("cradle_top_lid_thickness_m"),
        "cradle_top_lid_x_scale": summary.get("cradle_top_lid_x_scale"),
        "cradle_top_lid_y_scale": summary.get("cradle_top_lid_y_scale"),
        "cradle_top_lid_enable_on_hold": summary.get("cradle_top_lid_enable_on_hold"),
        "cradle_top_lid_collision_enabled_initial": summary.get("cradle_top_lid_collision_enabled_initial"),
        "cradle_top_lid_collision_enabled_step": summary.get("cradle_top_lid_collision_enabled_step"),
        "cradle_top_lid_collision_update_count": summary.get("cradle_top_lid_collision_update_count"),
        "cradle_top_lid_collision_update_error": summary.get("cradle_top_lid_collision_update_error"),
        "cradle_chest_pad_enabled": summary.get("cradle_chest_pad_enabled"),
        "cradle_chest_pad_local_pos0_m": summary.get("cradle_chest_pad_local_pos0_m"),
        "cradle_chest_pad_size_m": summary.get("cradle_chest_pad_size_m"),
        "cradle_chest_pad_enable_on_hold": summary.get("cradle_chest_pad_enable_on_hold"),
        "cradle_chest_pad_collision_enabled_initial": summary.get(
            "cradle_chest_pad_collision_enabled_initial"
        ),
        "cradle_chest_pad_collision_enabled_step": summary.get("cradle_chest_pad_collision_enabled_step"),
        "cradle_chest_pad_collision_update_count": summary.get("cradle_chest_pad_collision_update_count"),
        "cradle_chest_pad_collision_update_error": summary.get("cradle_chest_pad_collision_update_error"),
        "joint_count": summary.get("joint_count"),
        "stand_drive_gains_enabled": summary.get("stand_drive_gains_enabled"),
        "stand_drive_preset": summary.get("stand_drive_preset"),
        "applied_stand_drive_gain_count": summary.get("applied_stand_drive_gain_count"),
        "disable_usd_pelvis_xform": summary.get("disable_usd_pelvis_xform"),
        "balance_feedback_controller_enabled": summary.get("balance_feedback_controller_enabled"),
        "balance_pitch_gain": summary.get("balance_pitch_gain"),
        "balance_roll_gain": summary.get("balance_roll_gain"),
        "balance_pitch_rate_gain": summary.get("balance_pitch_rate_gain"),
        "balance_roll_rate_gain": summary.get("balance_roll_rate_gain"),
        "balance_adjustment_limit": summary.get("balance_adjustment_limit"),
        "balance_feedback_base": summary.get("balance_feedback_base"),
        "balance_start_on_agile_hold": summary.get("balance_start_on_agile_hold"),
        "balance_roll_left_ankle_scale": summary.get("balance_roll_left_ankle_scale"),
        "balance_roll_right_ankle_scale": summary.get("balance_roll_right_ankle_scale"),
        "balance_roll_left_hip_scale": summary.get("balance_roll_left_hip_scale"),
        "balance_roll_right_hip_scale": summary.get("balance_roll_right_hip_scale"),
        "balance_pitch_target": summary.get("balance_pitch_target"),
        "balance_roll_target": summary.get("balance_roll_target"),
        "balance_target_start_step": summary.get("balance_target_start_step"),
        "balance_target_end_step": summary.get("balance_target_end_step"),
        "balance_target_pulse_period_steps": summary.get("balance_target_pulse_period_steps"),
        "balance_target_pulse_width_steps": summary.get("balance_target_pulse_width_steps"),
        "balance_target_pulse_phase_step": summary.get("balance_target_pulse_phase_step"),
        "balance_target_active_steps": summary.get("balance_target_active_steps"),
        "balance_target_first_active_step": summary.get("balance_target_first_active_step"),
        "balance_pitch_sign": summary.get("balance_pitch_sign"),
        "balance_roll_sign": summary.get("balance_roll_sign"),
        "balance_start_step": summary.get("balance_start_step"),
        "balance_pitch_activation_threshold": summary.get("balance_pitch_activation_threshold"),
        "balance_roll_activation_threshold": summary.get("balance_roll_activation_threshold"),
        "balance_pitch_rate_activation_threshold": summary.get("balance_pitch_rate_activation_threshold"),
        "balance_roll_rate_activation_threshold": summary.get("balance_roll_rate_activation_threshold"),
        "balance_feedback_active_steps": summary.get("balance_feedback_active_steps"),
        "balance_feedback_first_active_step": summary.get("balance_feedback_first_active_step"),
        "gait_ramp_down_start_step": summary.get("gait_ramp_down_start_step"),
        "gait_ramp_down_end_step": summary.get("gait_ramp_down_end_step"),
        "gait_min_amplitude_scale": summary.get("gait_min_amplitude_scale"),
        "recovery_pitch_threshold": summary.get("recovery_pitch_threshold"),
        "recovery_pitch_rate_threshold": summary.get("recovery_pitch_rate_threshold"),
        "recovery_hip_pitch_offset": summary.get("recovery_hip_pitch_offset"),
        "recovery_knee_offset": summary.get("recovery_knee_offset"),
        "recovery_ankle_pitch_offset": summary.get("recovery_ankle_pitch_offset"),
        "recovery_waist_pitch_offset": summary.get("recovery_waist_pitch_offset"),
        "recovery_active_steps": summary.get("recovery_active_steps"),
        "recovery_first_active_step": summary.get("recovery_first_active_step"),
        "max_abs_roll_rad": summary.get("max_abs_roll_rad"),
        "max_abs_pitch_rad": summary.get("max_abs_pitch_rad"),
        "final_roll_rad": summary.get("final_roll_rad"),
        "final_pitch_rad": summary.get("final_pitch_rad"),
        "fall_events": summary.get("fall_events"),
        "box_drop_events": summary.get("box_drop_events"),
        "first_fall_step": summary.get("first_fall_step"),
        "first_fall_time_s": summary.get("first_fall_time_s"),
        "first_box_drop_step": summary.get("first_box_drop_step"),
        "first_box_drop_time_s": summary.get("first_box_drop_time_s"),
        "agile_command_hold_final_fall_events": summary.get(
            "agile_command_hold_final_fall_events"
        ),
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
        "min_robot_z_m": summary.get("min_robot_z_m"),
        "min_box_z_m": summary.get("min_box_z_m"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
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
        "final_box_roll_rad": summary.get("final_box_roll_rad"),
        "final_box_pitch_rad": summary.get("final_box_pitch_rad"),
        "max_box_robot_relative_offset_error_m": summary.get("max_box_robot_relative_offset_error_m"),
        "final_box_robot_relative_offset_error_m": summary.get("final_box_robot_relative_offset_error_m"),
        "max_robot_target_directed_travel_m": summary.get("max_robot_target_directed_travel_m"),
        "max_box_target_directed_travel_m": summary.get("max_box_target_directed_travel_m"),
        "final_robot_target_directed_travel_m": summary.get("final_robot_target_directed_travel_m"),
        "final_box_target_directed_travel_m": summary.get("final_box_target_directed_travel_m"),
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
        "final_robot_travel_xy_m": summary.get("final_robot_travel_xy_m"),
        "final_box_travel_xy_m": summary.get("final_box_travel_xy_m"),
        "diagnostic_root_drive": summary.get("diagnostic_root_drive"),
        "diagnostic_root_drive_active_steps": summary.get("diagnostic_root_drive_active_steps"),
        "diagnostic_root_drive_speed_mps": summary.get("diagnostic_root_drive_speed_mps"),
        "diagnostic_root_drive_final_commanded_xy_m": summary.get("diagnostic_root_drive_final_commanded_xy_m"),
        "root_pose_write_count_setup": summary.get("root_pose_write_count_setup"),
        "joint_state_write_count_setup": summary.get("joint_state_write_count_setup"),
        "joint_state_write_error": summary.get("joint_state_write_error"),
        "root_pose_write_count_rollout": summary.get("root_pose_write_count_rollout"),
        "root_velocity_write_count_rollout": summary.get("root_velocity_write_count_rollout"),
        "box_pose_write_count_rollout": summary.get("box_pose_write_count_rollout"),
        "error": summary.get("error"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
