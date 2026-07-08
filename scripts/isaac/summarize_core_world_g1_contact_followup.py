#!/usr/bin/env python3
"""Compare G1 low-carry contact follow-up runs.

This is a lightweight JSON summarizer.  It does not run Isaac; it only reads
existing summary/check files and reports whether a contact variant improves the
known low-carry baseline without hiding strict-gate failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("experiments/outputs/core_world_g1_agile_policy_low_cradle")
DEFAULT_CASES = (
    ("baseline_lowcarry_replay_record", "20260706_g1_lowcarry_168398_replay_record_retry2"),
    ("hold_contact_partial", "20260706_g1_lowcarry_followup_chestpad_hold_contact"),
    ("terminal_contact_pending", "20260707_gpu_contact_next_chestpad_terminal_contact"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        metavar=("LABEL", "STAMP"),
        default=[],
        help="Case label and suite stamp under --root.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-box-travel", type=float, default=2.0)
    parser.add_argument("--max-box-travel", type=float, default=2.35)
    parser.add_argument("--max-final-relative-error", type=float, default=0.25)
    parser.add_argument("--max-fall-events", type=int, default=0)
    parser.add_argument("--max-drop-events", type=int, default=0)
    parser.add_argument("--min-final-hold-active-steps", type=int, default=399)
    parser.add_argument("--min-target-window-end-streak", type=int, default=40)
    parser.add_argument("--min-target-window-final-hold-end-streak", type=int, default=40)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _case_dir(root: Path, stamp: str) -> Path:
    standard = root / stamp / "agile_low_cradle_freebox_walk"
    legacy = root / stamp
    if standard.exists() or not (legacy / "core_world_g1_box_scene_summary.json").exists():
        return standard
    return legacy


def _read_case(root: Path, label: str, stamp: str, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = _case_dir(root, stamp)
    summary_path = case_dir / "core_world_g1_box_scene_summary.json"
    check_path = case_dir / "check.json"
    if not summary_path.exists():
        return {
            "label": label,
            "stamp": stamp,
            "case_dir": str(case_dir),
            "status": "missing",
            "failures": [f"missing summary: {summary_path}"],
        }
    summary = _load_json(summary_path)
    check = _load_json(check_path) if check_path.exists() else {}
    failures = list(check.get("failures") or [])
    check_status = check.get("status")
    if check_status is None:
        failures.append("missing check.json/check status")
    elif check_status != "pass":
        failures.append(f"check_status {check_status} != pass")

    fall_events = _as_int(summary.get("fall_events"))
    drop_events = _as_int(summary.get("box_drop_events"))
    root_pose_writes = _as_int(summary.get("root_pose_write_count_rollout"))
    root_velocity_writes = _as_int(summary.get("root_velocity_write_count_rollout"))
    box_pose_writes = _as_int(summary.get("box_pose_write_count_rollout"))
    final_box_travel = _as_float(summary.get("final_box_target_directed_travel_m"))
    final_robot_travel = _as_float(summary.get("final_robot_target_directed_travel_m"))
    final_rel = _as_float(summary.get("final_box_robot_relative_offset_error_m"), 999.0)
    final_hold_steps = _as_int(summary.get("agile_command_hold_final_active_steps"))
    target_end = _as_int(summary.get("target_window_both_streak_at_end_steps"))
    final_hold_end = _as_int(summary.get("target_window_both_final_hold_streak_at_end_steps"))

    if fall_events > int(args.max_fall_events):
        failures.append(f"fall_events {fall_events} > {args.max_fall_events}")
    if drop_events > int(args.max_drop_events):
        failures.append(f"box_drop_events {drop_events} > {args.max_drop_events}")
    if root_pose_writes or root_velocity_writes or box_pose_writes:
        failures.append(
            "rollout shortcut writes "
            f"root_pose/root_velocity/box_pose={root_pose_writes}/{root_velocity_writes}/{box_pose_writes}"
        )
    if final_box_travel < float(args.min_box_travel):
        failures.append(f"final_box_target_directed_travel_m {final_box_travel} < {args.min_box_travel}")
    if final_box_travel > float(args.max_box_travel):
        failures.append(f"final_box_target_directed_travel_m {final_box_travel} > {args.max_box_travel}")
    if final_rel > float(args.max_final_relative_error):
        failures.append(f"final_relative_error_m {final_rel} > {args.max_final_relative_error}")
    if final_hold_steps < int(args.min_final_hold_active_steps):
        failures.append(f"final_hold_active_steps {final_hold_steps} < {args.min_final_hold_active_steps}")
    if target_end < int(args.min_target_window_end_streak):
        failures.append(f"target_window_end_streak {target_end} < {args.min_target_window_end_streak}")
    if final_hold_end < int(args.min_target_window_final_hold_end_streak):
        failures.append(
            "target_window_final_hold_end_streak "
            f"{final_hold_end} < {args.min_target_window_final_hold_end_streak}"
        )

    return {
        "label": label,
        "stamp": stamp,
        "case_dir": str(case_dir),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "check_status": check_status,
        "completed_steps": summary.get("completed_steps"),
        "fall_events": fall_events,
        "box_drop_events": drop_events,
        "root_pose_write_count_rollout": root_pose_writes,
        "root_velocity_write_count_rollout": root_velocity_writes,
        "box_pose_write_count_rollout": box_pose_writes,
        "final_robot_target_directed_travel_m": final_robot_travel,
        "final_box_target_directed_travel_m": final_box_travel,
        "final_relative_error_m": final_rel,
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "max_box_tilt_rad": summary.get("max_box_tilt_rad"),
        "final_hold_active_steps": final_hold_steps,
        "target_window_both_streak_at_end_steps": target_end,
        "target_window_both_final_hold_streak_at_end_steps": final_hold_end,
        "cradle_chest_pad_enabled": summary.get("cradle_chest_pad_enabled"),
        "cradle_chest_pad_enable_on_hold": summary.get("cradle_chest_pad_enable_on_hold"),
        "cradle_chest_pad_enable_on_terminal_hold": summary.get(
            "cradle_chest_pad_enable_on_terminal_hold"
        ),
        "cradle_chest_pad_collision_enabled_step": summary.get(
            "cradle_chest_pad_collision_enabled_step"
        ),
    }


def main() -> int:
    args = parse_args()
    requested = tuple((label, stamp) for label, stamp in args.case) or DEFAULT_CASES
    cases = [_read_case(args.root, label, stamp, args) for label, stamp in requested]
    present_cases = [case for case in cases if case["status"] != "missing"]
    passing_cases = [case for case in cases if case["status"] == "pass"]
    report = {
        "scene_type": "core_world_g1_contact_followup_comparison",
        "success_claim": "comparison_only_not_final_carrying_success",
        "status": "pass" if present_cases and len(passing_cases) == len(present_cases) else "fail",
        "case_count": len(cases),
        "present_case_count": len(present_cases),
        "passing_case_count": len(passing_cases),
        "cases": cases,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
