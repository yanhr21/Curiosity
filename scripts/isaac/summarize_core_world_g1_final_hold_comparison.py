#!/usr/bin/env python3
"""Compare final-stop and final-stand G1 low-carry diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CASES = (
    (
        "full_stop_after_2m",
        "20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_strict_900_targetnegx1",
    ),
    (
        "final_stand_after_2m",
        "20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_strict_900_targetnegx1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/outputs/core_world_g1_agile_policy_low_cradle"),
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--min-box-travel", type=float, default=2.0)
    parser.add_argument("--max-box-travel", type=float, default=2.35)
    parser.add_argument("--max-fall-events", type=int, default=0)
    parser.add_argument("--max-drop-events", type=int, default=0)
    parser.add_argument("--max-tilt-rad", type=float, default=0.35)
    parser.add_argument("--max-box-tilt-rad", type=float, default=0.45)
    parser.add_argument("--max-final-rel-m", type=float, default=0.25)
    parser.add_argument("--max-final-robot-lateral-m", type=float, default=0.60)
    parser.add_argument("--max-final-box-lateral-m", type=float, default=0.60)
    parser.add_argument("--min-final-active-steps", type=int, default=120)
    parser.add_argument("--min-final-stand-active-steps", type=int, default=80)
    parser.add_argument("--max-final-hold-command-x", type=float, default=None)
    parser.add_argument("--max-final-hold-command-y", type=float, default=None)
    parser.add_argument("--max-final-hold-command-yaw", type=float, default=None)
    parser.add_argument("--min-final-hold-robot-z", type=float, default=None)
    parser.add_argument("--min-final-hold-box-z", type=float, default=None)
    parser.add_argument("--max-final-hold-tilt", type=float, default=None)
    parser.add_argument("--max-final-hold-box-tilt", type=float, default=None)
    parser.add_argument("--max-final-hold-fall-events", type=int, default=None)
    parser.add_argument("--max-final-hold-box-drop-events", type=int, default=None)
    parser.add_argument("--min-final-stand-robot-z", type=float, default=None)
    parser.add_argument("--min-final-stand-box-z", type=float, default=None)
    parser.add_argument("--max-final-stand-tilt", type=float, default=None)
    parser.add_argument("--max-final-stand-box-tilt", type=float, default=None)
    parser.add_argument("--max-final-stand-fall-events", type=int, default=None)
    parser.add_argument("--max-final-stand-box-drop-events", type=int, default=None)
    parser.add_argument("--min-target-window-both-stable-steps", type=int, default=0)
    parser.add_argument("--min-target-window-both-longest-streak-steps", type=int, default=0)
    parser.add_argument("--min-target-window-both-streak-at-end-steps", type=int, default=0)
    parser.add_argument("--min-target-window-both-final-hold-stable-steps", type=int, default=0)
    parser.add_argument("--min-target-window-both-final-hold-longest-streak-steps", type=int, default=0)
    parser.add_argument("--min-target-window-both-final-hold-streak-at-end-steps", type=int, default=0)
    parser.add_argument("--min-target-window-both-final-stand-stable-steps", type=int, default=0)
    parser.add_argument("--min-target-window-both-final-stand-longest-streak-steps", type=int, default=0)
    parser.add_argument("--min-target-window-both-final-stand-streak-at-end-steps", type=int, default=0)
    parser.add_argument("--max-root-pose-write-count-rollout", type=int, default=0)
    parser.add_argument("--max-root-velocity-write-count-rollout", type=int, default=0)
    parser.add_argument("--max-box-pose-write-count-rollout", type=int, default=0)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _first_events_from_csv(case_dir: Path) -> dict[str, Any]:
    events: dict[str, Any] = {
        "first_fall_step": None,
        "first_box_drop_step": None,
    }
    state_path = case_dir / "core_world_g1_box_scene_state.csv"
    if not state_path.exists():
        return events
    with state_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if events["first_fall_step"] is None and int(float(row.get("fall") or 0)):
                events["first_fall_step"] = int(float(row["step"]))
            if events["first_box_drop_step"] is None and int(float(row.get("drop") or 0)):
                events["first_box_drop_step"] = int(float(row["step"]))
            if events["first_fall_step"] is not None and events["first_box_drop_step"] is not None:
                break
    return events


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _case_record(root: Path, label: str, stamp: str, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = root / stamp / "agile_low_cradle_freebox_walk"
    summary_path = case_dir / "core_world_g1_box_scene_summary.json"
    check_path = case_dir / "check.json"
    if not summary_path.exists():
        return {
            "label": label,
            "stamp": stamp,
            "missing": True,
            "case_dir": str(case_dir),
            "status": "missing",
            "failures": ["summary missing"],
        }

    summary = _load_json(summary_path)
    check = _load_json(check_path) if check_path.exists() else {}
    first_events = _first_events_from_csv(case_dir)
    failures = list(check.get("failures") or [])
    if check_path.exists() and check.get("status") != "pass":
        failures.append(f"checker_status {check.get('status')} != pass")
    if not check_path.exists():
        failures.append("check.json missing")

    fall_events = _as_int(summary.get("fall_events"))
    drop_events = _as_int(summary.get("box_drop_events"))
    final_box_travel = _as_float(summary.get("final_box_target_directed_travel_m"))
    final_robot_travel = _as_float(summary.get("final_robot_target_directed_travel_m"))
    max_tilt = _as_float(summary.get("max_tilt_rad"), 999.0)
    max_box_tilt = _as_float(summary.get("max_box_tilt_rad"), 999.0)
    final_hold_min_robot_z = summary.get("agile_command_hold_final_min_robot_z_m")
    final_hold_min_box_z = summary.get("agile_command_hold_final_min_box_z_m")
    final_hold_max_tilt = summary.get("agile_command_hold_final_max_tilt_rad")
    final_hold_max_box_tilt = summary.get("agile_command_hold_final_max_box_tilt_rad")
    final_hold_fall_events = _as_int(summary.get("agile_command_hold_final_fall_events"))
    final_hold_drop_events = _as_int(summary.get("agile_command_hold_final_box_drop_events"))
    final_stand_min_robot_z = summary.get("agile_command_hold_final_stand_min_robot_z_m")
    final_stand_min_box_z = summary.get("agile_command_hold_final_stand_min_box_z_m")
    final_stand_max_tilt = summary.get("agile_command_hold_final_stand_max_tilt_rad")
    final_stand_max_box_tilt = summary.get("agile_command_hold_final_stand_max_box_tilt_rad")
    final_stand_fall_events = _as_int(summary.get("agile_command_hold_final_stand_fall_events"))
    final_stand_drop_events = _as_int(summary.get("agile_command_hold_final_stand_box_drop_events"))
    final_rel = _as_float(summary.get("final_box_robot_relative_offset_error_m"), 999.0)
    final_robot_lateral = abs(_as_float(summary.get("final_robot_target_lateral_error_m"), 999.0))
    final_box_lateral = abs(_as_float(summary.get("final_box_target_lateral_error_m"), 999.0))
    final_active_steps = _as_int(summary.get("agile_command_hold_final_active_steps"))
    final_stand_steps = _as_int(summary.get("agile_command_hold_final_stand_active_steps"))
    final_stand_enabled = bool(summary.get("agile_command_hold_final_stand_enabled"))
    final_command_x = _as_float(summary.get("agile_command_hold_final_max_abs_command_x"))
    final_command_y = _as_float(summary.get("agile_command_hold_final_max_abs_command_y"))
    final_command_yaw = _as_float(summary.get("agile_command_hold_final_max_abs_command_yaw"))
    target_window_both_steps = _as_int(summary.get("target_window_both_stable_steps"))
    target_window_both_streak = _as_int(summary.get("target_window_both_longest_streak_steps"))
    target_window_both_streak_at_end = _as_int(summary.get("target_window_both_streak_at_end_steps"))
    target_window_both_final_hold_steps = _as_int(
        summary.get("target_window_both_final_hold_stable_steps")
    )
    target_window_both_final_hold_streak = _as_int(
        summary.get("target_window_both_final_hold_longest_streak_steps")
    )
    target_window_both_final_hold_streak_at_end = _as_int(
        summary.get("target_window_both_final_hold_streak_at_end_steps")
    )
    target_window_both_final_stand_steps = _as_int(
        summary.get("target_window_both_final_stand_stable_steps")
    )
    target_window_both_final_stand_streak = _as_int(
        summary.get("target_window_both_final_stand_longest_streak_steps")
    )
    target_window_both_final_stand_streak_at_end = _as_int(
        summary.get("target_window_both_final_stand_streak_at_end_steps")
    )
    root_pose_writes = _as_int(summary.get("root_pose_write_count_rollout"))
    root_velocity_writes = _as_int(summary.get("root_velocity_write_count_rollout"))
    box_pose_writes = _as_int(summary.get("box_pose_write_count_rollout"))

    if fall_events > int(args.max_fall_events):
        failures.append(f"fall_events {fall_events} > {args.max_fall_events}")
    if drop_events > int(args.max_drop_events):
        failures.append(f"box_drop_events {drop_events} > {args.max_drop_events}")
    if final_box_travel < float(args.min_box_travel):
        failures.append(f"final_box_target_directed_travel_m {final_box_travel} < {args.min_box_travel}")
    if final_box_travel > float(args.max_box_travel):
        failures.append(f"final_box_target_directed_travel_m {final_box_travel} > {args.max_box_travel}")
    if max_tilt > float(args.max_tilt_rad):
        failures.append(f"max_tilt_rad {max_tilt} > {args.max_tilt_rad}")
    if max_box_tilt > float(args.max_box_tilt_rad):
        failures.append(f"max_box_tilt_rad {max_box_tilt} > {args.max_box_tilt_rad}")
    if final_rel > float(args.max_final_rel_m):
        failures.append(f"final_box_robot_relative_offset_error_m {final_rel} > {args.max_final_rel_m}")
    if final_robot_lateral > float(args.max_final_robot_lateral_m):
        failures.append(f"abs(final_robot_target_lateral_error_m) {final_robot_lateral} > {args.max_final_robot_lateral_m}")
    if final_box_lateral > float(args.max_final_box_lateral_m):
        failures.append(f"abs(final_box_target_lateral_error_m) {final_box_lateral} > {args.max_final_box_lateral_m}")
    if final_active_steps < int(args.min_final_active_steps):
        failures.append(f"agile_command_hold_final_active_steps {final_active_steps} < {args.min_final_active_steps}")
    if final_stand_enabled and final_stand_steps < int(args.min_final_stand_active_steps):
        failures.append(
            "agile_command_hold_final_stand_active_steps "
            f"{final_stand_steps} < {args.min_final_stand_active_steps}"
        )
    if args.max_final_hold_command_x is not None and final_command_x > float(args.max_final_hold_command_x):
        failures.append(
            f"agile_command_hold_final_max_abs_command_x {final_command_x} > {args.max_final_hold_command_x}"
        )
    if args.max_final_hold_command_y is not None and final_command_y > float(args.max_final_hold_command_y):
        failures.append(
            f"agile_command_hold_final_max_abs_command_y {final_command_y} > {args.max_final_hold_command_y}"
        )
    if args.max_final_hold_command_yaw is not None and final_command_yaw > float(args.max_final_hold_command_yaw):
        failures.append(
            "agile_command_hold_final_max_abs_command_yaw "
            f"{final_command_yaw} > {args.max_final_hold_command_yaw}"
        )
    if args.min_final_hold_robot_z is not None and (
        final_hold_min_robot_z is None or float(final_hold_min_robot_z) < float(args.min_final_hold_robot_z)
    ):
        failures.append(
            f"agile_command_hold_final_min_robot_z_m {final_hold_min_robot_z} < {args.min_final_hold_robot_z}"
        )
    if args.min_final_hold_box_z is not None and (
        final_hold_min_box_z is None or float(final_hold_min_box_z) < float(args.min_final_hold_box_z)
    ):
        failures.append(
            f"agile_command_hold_final_min_box_z_m {final_hold_min_box_z} < {args.min_final_hold_box_z}"
        )
    if args.max_final_hold_tilt is not None and (
        final_hold_max_tilt is None or float(final_hold_max_tilt) > float(args.max_final_hold_tilt)
    ):
        failures.append(
            f"agile_command_hold_final_max_tilt_rad {final_hold_max_tilt} > {args.max_final_hold_tilt}"
        )
    if args.max_final_hold_box_tilt is not None and (
        final_hold_max_box_tilt is None or float(final_hold_max_box_tilt) > float(args.max_final_hold_box_tilt)
    ):
        failures.append(
            "agile_command_hold_final_max_box_tilt_rad "
            f"{final_hold_max_box_tilt} > {args.max_final_hold_box_tilt}"
        )
    if args.max_final_hold_fall_events is not None and final_hold_fall_events > int(
        args.max_final_hold_fall_events
    ):
        failures.append(
            f"agile_command_hold_final_fall_events {final_hold_fall_events} > {args.max_final_hold_fall_events}"
        )
    if args.max_final_hold_box_drop_events is not None and final_hold_drop_events > int(
        args.max_final_hold_box_drop_events
    ):
        failures.append(
            "agile_command_hold_final_box_drop_events "
            f"{final_hold_drop_events} > {args.max_final_hold_box_drop_events}"
        )
    if final_stand_enabled and args.min_final_stand_robot_z is not None and (
        final_stand_min_robot_z is None or float(final_stand_min_robot_z) < float(args.min_final_stand_robot_z)
    ):
        failures.append(
            "agile_command_hold_final_stand_min_robot_z_m "
            f"{final_stand_min_robot_z} < {args.min_final_stand_robot_z}"
        )
    if final_stand_enabled and args.min_final_stand_box_z is not None and (
        final_stand_min_box_z is None or float(final_stand_min_box_z) < float(args.min_final_stand_box_z)
    ):
        failures.append(
            f"agile_command_hold_final_stand_min_box_z_m {final_stand_min_box_z} < {args.min_final_stand_box_z}"
        )
    if final_stand_enabled and args.max_final_stand_tilt is not None and (
        final_stand_max_tilt is None or float(final_stand_max_tilt) > float(args.max_final_stand_tilt)
    ):
        failures.append(
            f"agile_command_hold_final_stand_max_tilt_rad {final_stand_max_tilt} > {args.max_final_stand_tilt}"
        )
    if final_stand_enabled and args.max_final_stand_box_tilt is not None and (
        final_stand_max_box_tilt is None or float(final_stand_max_box_tilt) > float(args.max_final_stand_box_tilt)
    ):
        failures.append(
            "agile_command_hold_final_stand_max_box_tilt_rad "
            f"{final_stand_max_box_tilt} > {args.max_final_stand_box_tilt}"
        )
    if final_stand_enabled and args.max_final_stand_fall_events is not None and final_stand_fall_events > int(
        args.max_final_stand_fall_events
    ):
        failures.append(
            "agile_command_hold_final_stand_fall_events "
            f"{final_stand_fall_events} > {args.max_final_stand_fall_events}"
        )
    if (
        final_stand_enabled
        and args.max_final_stand_box_drop_events is not None
        and final_stand_drop_events > int(args.max_final_stand_box_drop_events)
    ):
        failures.append(
            "agile_command_hold_final_stand_box_drop_events "
            f"{final_stand_drop_events} > {args.max_final_stand_box_drop_events}"
        )
    if target_window_both_steps < int(args.min_target_window_both_stable_steps):
        failures.append(
            f"target_window_both_stable_steps {target_window_both_steps} < "
            f"{args.min_target_window_both_stable_steps}"
        )
    if target_window_both_streak < int(args.min_target_window_both_longest_streak_steps):
        failures.append(
            "target_window_both_longest_streak_steps "
            f"{target_window_both_streak} < {args.min_target_window_both_longest_streak_steps}"
        )
    if target_window_both_streak_at_end < int(args.min_target_window_both_streak_at_end_steps):
        failures.append(
            "target_window_both_streak_at_end_steps "
            f"{target_window_both_streak_at_end} < {args.min_target_window_both_streak_at_end_steps}"
        )
    if target_window_both_final_hold_steps < int(args.min_target_window_both_final_hold_stable_steps):
        failures.append(
            "target_window_both_final_hold_stable_steps "
            f"{target_window_both_final_hold_steps} < {args.min_target_window_both_final_hold_stable_steps}"
        )
    if target_window_both_final_hold_streak < int(
        args.min_target_window_both_final_hold_longest_streak_steps
    ):
        failures.append(
            "target_window_both_final_hold_longest_streak_steps "
            f"{target_window_both_final_hold_streak} < "
            f"{args.min_target_window_both_final_hold_longest_streak_steps}"
        )
    if target_window_both_final_hold_streak_at_end < int(
        args.min_target_window_both_final_hold_streak_at_end_steps
    ):
        failures.append(
            "target_window_both_final_hold_streak_at_end_steps "
            f"{target_window_both_final_hold_streak_at_end} < "
            f"{args.min_target_window_both_final_hold_streak_at_end_steps}"
        )
    if target_window_both_final_stand_steps < int(args.min_target_window_both_final_stand_stable_steps):
        failures.append(
            "target_window_both_final_stand_stable_steps "
            f"{target_window_both_final_stand_steps} < {args.min_target_window_both_final_stand_stable_steps}"
        )
    if target_window_both_final_stand_streak < int(
        args.min_target_window_both_final_stand_longest_streak_steps
    ):
        failures.append(
            "target_window_both_final_stand_longest_streak_steps "
            f"{target_window_both_final_stand_streak} < "
            f"{args.min_target_window_both_final_stand_longest_streak_steps}"
        )
    if target_window_both_final_stand_streak_at_end < int(
        args.min_target_window_both_final_stand_streak_at_end_steps
    ):
        failures.append(
            "target_window_both_final_stand_streak_at_end_steps "
            f"{target_window_both_final_stand_streak_at_end} < "
            f"{args.min_target_window_both_final_stand_streak_at_end_steps}"
        )
    if root_pose_writes > int(args.max_root_pose_write_count_rollout):
        failures.append(
            f"root_pose_write_count_rollout {root_pose_writes} > {args.max_root_pose_write_count_rollout}"
        )
    if root_velocity_writes > int(args.max_root_velocity_write_count_rollout):
        failures.append(
            "root_velocity_write_count_rollout "
            f"{root_velocity_writes} > {args.max_root_velocity_write_count_rollout}"
        )
    if box_pose_writes > int(args.max_box_pose_write_count_rollout):
        failures.append(
            f"box_pose_write_count_rollout {box_pose_writes} > {args.max_box_pose_write_count_rollout}"
        )

    status = "pass" if not failures else "fail"
    return {
        "label": label,
        "stamp": stamp,
        "missing": False,
        "status": status,
        "checker_status": check.get("status"),
        "failures": failures,
        "case_dir": str(case_dir),
        "summary_path": str(summary_path),
        "check_path": str(check_path) if check_path.exists() else None,
        "completed_steps": summary.get("completed_steps"),
        "fall_events": fall_events,
        "box_drop_events": drop_events,
        "first_fall_step": summary.get("first_fall_step")
        if summary.get("first_fall_step") is not None
        else first_events["first_fall_step"],
        "first_box_drop_step": summary.get("first_box_drop_step")
        if summary.get("first_box_drop_step") is not None
        else first_events["first_box_drop_step"],
        "final_robot_target_directed_travel_m": final_robot_travel,
        "final_box_target_directed_travel_m": final_box_travel,
        "max_tilt_rad": max_tilt,
        "max_box_tilt_rad": max_box_tilt,
        "agile_command_hold_final_min_robot_z_m": final_hold_min_robot_z,
        "agile_command_hold_final_min_box_z_m": final_hold_min_box_z,
        "agile_command_hold_final_max_tilt_rad": final_hold_max_tilt,
        "agile_command_hold_final_max_box_tilt_rad": final_hold_max_box_tilt,
        "agile_command_hold_final_fall_events": final_hold_fall_events,
        "agile_command_hold_final_box_drop_events": final_hold_drop_events,
        "agile_command_hold_final_first_fall_step": summary.get(
            "agile_command_hold_final_first_fall_step"
        ),
        "agile_command_hold_final_first_box_drop_step": summary.get(
            "agile_command_hold_final_first_box_drop_step"
        ),
        "agile_command_hold_final_stand_min_robot_z_m": final_stand_min_robot_z,
        "agile_command_hold_final_stand_min_box_z_m": final_stand_min_box_z,
        "agile_command_hold_final_stand_max_tilt_rad": final_stand_max_tilt,
        "agile_command_hold_final_stand_max_box_tilt_rad": final_stand_max_box_tilt,
        "agile_command_hold_final_stand_fall_events": final_stand_fall_events,
        "agile_command_hold_final_stand_box_drop_events": final_stand_drop_events,
        "agile_command_hold_final_stand_first_fall_step": summary.get(
            "agile_command_hold_final_stand_first_fall_step"
        ),
        "agile_command_hold_final_stand_first_box_drop_step": summary.get(
            "agile_command_hold_final_stand_first_box_drop_step"
        ),
        "final_box_robot_relative_offset_error_m": final_rel,
        "final_robot_target_lateral_error_m": summary.get("final_robot_target_lateral_error_m"),
        "final_box_target_lateral_error_m": summary.get("final_box_target_lateral_error_m"),
        "agile_command_hold_terminal_scale": summary.get("agile_command_hold_terminal_scale"),
        "agile_command_hold_final_scale": summary.get("agile_command_hold_final_scale"),
        "agile_command_hold_final_box_target_travel_m": summary.get(
            "agile_command_hold_final_box_target_travel_m"
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
        "agile_command_hold_final_active_steps": final_active_steps,
        "agile_command_hold_final_stand_enabled": final_stand_enabled,
        "agile_command_hold_final_stand_delay_steps": summary.get(
            "agile_command_hold_final_stand_delay_steps"
        ),
        "agile_command_hold_final_stand_active_steps": final_stand_steps,
        "target_window_enabled": summary.get("target_window_enabled"),
        "target_window_center_m": summary.get("target_window_center_m"),
        "target_window_halfwidth_m": summary.get("target_window_halfwidth_m"),
        "target_window_both_stable_steps": target_window_both_steps,
        "target_window_both_longest_streak_steps": target_window_both_streak,
        "target_window_both_streak_at_end_steps": target_window_both_streak_at_end,
        "target_window_both_first_stable_step": summary.get("target_window_both_first_stable_step"),
        "target_window_both_final_hold_stable_steps": target_window_both_final_hold_steps,
        "target_window_both_final_hold_longest_streak_steps": target_window_both_final_hold_streak,
        "target_window_both_final_hold_streak_at_end_steps": target_window_both_final_hold_streak_at_end,
        "target_window_both_final_hold_first_stable_step": summary.get(
            "target_window_both_final_hold_first_stable_step"
        ),
        "target_window_both_final_stand_stable_steps": target_window_both_final_stand_steps,
        "target_window_both_final_stand_longest_streak_steps": target_window_both_final_stand_streak,
        "target_window_both_final_stand_streak_at_end_steps": target_window_both_final_stand_streak_at_end,
        "target_window_both_final_stand_first_stable_step": summary.get(
            "target_window_both_final_stand_first_stable_step"
        ),
        "root_pose_write_count_rollout": root_pose_writes,
        "root_velocity_write_count_rollout": root_velocity_writes,
        "box_pose_write_count_rollout": box_pose_writes,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# G1 Final-Hold Comparison",
        "",
        "This compares final-stop and final-stand low-carry diagnostics. It is not a final success claim.",
        "",
        "| case | status | checker | fall/drop | first fall/drop | travel robot/box | tilt robot/box | final rel | final lat robot/box | final hold | target window | writes root/vel/box | failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for record in records:
        failures = "; ".join(str(v) for v in record.get("failures") or []) or "-"
        final_hold = (
            f"scale={_fmt(record.get('agile_command_hold_final_scale'))}, "
            f"at={_fmt(record.get('agile_command_hold_final_box_target_travel_m'))}, "
            f"steps={_fmt(record.get('agile_command_hold_final_active_steps'))}, "
            f"zero_corr={_fmt(record.get('agile_command_hold_final_zero_corrections_enabled'))}, "
            f"zero_lat/yaw={_fmt(record.get('agile_command_hold_final_lateral_suppressed_steps'))}/"
            f"{_fmt(record.get('agile_command_hold_final_yaw_suppressed_steps'))}, "
            f"cmd_max={_fmt(record.get('agile_command_hold_final_max_abs_command_x'))}/"
            f"{_fmt(record.get('agile_command_hold_final_max_abs_command_y'))}/"
            f"{_fmt(record.get('agile_command_hold_final_max_abs_command_yaw'))}, "
            f"stable={_fmt(record.get('agile_command_hold_final_min_robot_z_m'))}/"
            f"{_fmt(record.get('agile_command_hold_final_min_box_z_m'))}/"
            f"{_fmt(record.get('agile_command_hold_final_max_tilt_rad'))}/"
            f"{_fmt(record.get('agile_command_hold_final_max_box_tilt_rad'))}, "
            f"fall/drop={_fmt(record.get('agile_command_hold_final_fall_events'))}/"
            f"{_fmt(record.get('agile_command_hold_final_box_drop_events'))}, "
            f"stand={_fmt(record.get('agile_command_hold_final_stand_enabled'))}, "
            f"stand_delay={_fmt(record.get('agile_command_hold_final_stand_delay_steps'))}, "
            f"stand_steps={_fmt(record.get('agile_command_hold_final_stand_active_steps'))}, "
            f"stand_fall/drop={_fmt(record.get('agile_command_hold_final_stand_fall_events'))}/"
            f"{_fmt(record.get('agile_command_hold_final_stand_box_drop_events'))}"
        )
        target_window = (
            f"enabled={_fmt(record.get('target_window_enabled'))}, "
            f"center={_fmt(record.get('target_window_center_m'))}, "
            f"half={_fmt(record.get('target_window_halfwidth_m'))}, "
            f"both_steps={_fmt(record.get('target_window_both_stable_steps'))}, "
            f"both_streak={_fmt(record.get('target_window_both_longest_streak_steps'))}, "
            f"both_end={_fmt(record.get('target_window_both_streak_at_end_steps'))}, "
            f"final_hold_steps={_fmt(record.get('target_window_both_final_hold_stable_steps'))}, "
            f"final_hold_streak={_fmt(record.get('target_window_both_final_hold_longest_streak_steps'))}, "
            f"final_hold_end={_fmt(record.get('target_window_both_final_hold_streak_at_end_steps'))}, "
            f"final_stand_steps={_fmt(record.get('target_window_both_final_stand_stable_steps'))}, "
            f"final_stand_streak={_fmt(record.get('target_window_both_final_stand_longest_streak_steps'))}, "
            f"final_stand_end={_fmt(record.get('target_window_both_final_stand_streak_at_end_steps'))}"
        )
        lines.append(
            "| {label} | {status} | {checker} | {fall}/{drop} | {first_fall}/{first_drop} | "
            "{robot_travel}/{box_travel} | {tilt}/{box_tilt} | {final_rel} | "
            "{robot_lat}/{box_lat} | {final_hold} | {target_window} | {writes} | {failures} |".format(
                label=record["label"],
                status=record["status"],
                checker=_fmt(record.get("checker_status")),
                fall=_fmt(record.get("fall_events")),
                drop=_fmt(record.get("box_drop_events")),
                first_fall=_fmt(record.get("first_fall_step")),
                first_drop=_fmt(record.get("first_box_drop_step")),
                robot_travel=_fmt(record.get("final_robot_target_directed_travel_m")),
                box_travel=_fmt(record.get("final_box_target_directed_travel_m")),
                tilt=_fmt(record.get("max_tilt_rad")),
                box_tilt=_fmt(record.get("max_box_tilt_rad")),
                final_rel=_fmt(record.get("final_box_robot_relative_offset_error_m")),
                robot_lat=_fmt(record.get("final_robot_target_lateral_error_m")),
                box_lat=_fmt(record.get("final_box_target_lateral_error_m")),
                final_hold=final_hold,
                target_window=target_window,
                writes=(
                    f"{_fmt(record.get('root_pose_write_count_rollout'))}/"
                    f"{_fmt(record.get('root_velocity_write_count_rollout'))}/"
                    f"{_fmt(record.get('box_pose_write_count_rollout'))}"
                ),
                failures=failures.replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    records = [_case_record(args.root, label, stamp, args) for label, stamp in DEFAULT_CASES]
    report = {
        "report_type": "core_world_g1_final_hold_comparison",
        "success_claim": "diagnostic_comparison_only_not_final_carrying_success",
        "cases": records,
    }
    markdown = _markdown(records)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)

    print(markdown, end="")
    return 0 if all(record.get("status") == "pass" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
