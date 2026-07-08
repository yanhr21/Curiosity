#!/usr/bin/env python3
"""Summarize current G1 low-carry 900-step control diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CASES = (
    ("lowcarry700_pass", "20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_strict_700_targetnegx1"),
    ("nolateral900", "20260706_g1_agile_largerbox_lowcarry_terminal_nolateral_strict_900_targetnegx1"),
    ("latched_zero900", "20260706_g1_agile_largerbox_lowcarry_nolateral_latchedzerostop_strict_900_targetnegx1"),
    ("latched_micro900", "20260706_g1_agile_largerbox_lowcarry_nolateral_latchedmicro_strict_900_targetnegx1"),
    (
        "terminal_lateral900",
        "20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_export_short_strict_900_targetnegx1",
    ),
    (
        "terminal_lateral_threshold_invalid611",
        "20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh_strict_900_targetnegx1",
    ),
    (
        "terminal_lateral_threshold_fix900",
        "20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh_fix_strict_900_targetnegx1",
    ),
    (
        "terminal_lateral_threshold045_tiltgate900",
        "20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_thresh045_tiltgate_strict_900_targetnegx1",
    ),
    (
        "terminal015_final006_2m900",
        "20260706_g1_agile_largerbox_lowcarry_terminal015_final006_2m_strict_900_targetnegx1",
    ),
    (
        "terminal015_final000_2m900",
        "20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_strict_900_targetnegx1",
    ),
    (
        "terminal015_final000_2m_finalstand900",
        "20260706_g1_agile_largerbox_lowcarry_terminal015_final000_2m_finalstand_strict_900_targetnegx1",
    ),
    (
        "terminal_lateral_excess_tiltgate_fallback900",
        "20260706_g1_agile_largerbox_lowcarry_latchedmicro_termlat_excess_tiltgate_strict_900_targetnegx1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/outputs/core_world_g1_agile_policy_low_cradle"),
    )
    parser.add_argument("--output", type=Path, default=None)
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


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _case_row(root: Path, label: str, stamp: str) -> dict[str, Any]:
    case_dir = root / stamp / "agile_low_cradle_freebox_walk"
    summary_path = case_dir / "core_world_g1_box_scene_summary.json"
    check_path = case_dir / "check.json"
    if not summary_path.exists():
        return {"label": label, "stamp": stamp, "missing": True}
    summary = _load_json(summary_path)
    check = _load_json(check_path) if check_path.exists() else {}
    first_events = _first_events_from_csv(case_dir)
    return {
        "label": label,
        "stamp": stamp,
        "missing": False,
        "check_status": check.get("status"),
        "fall_events": summary.get("fall_events"),
        "box_drop_events": summary.get("box_drop_events"),
        "first_fall_step": summary.get("first_fall_step")
        if summary.get("first_fall_step") is not None
        else first_events["first_fall_step"],
        "first_box_drop_step": summary.get("first_box_drop_step")
        if summary.get("first_box_drop_step") is not None
        else first_events["first_box_drop_step"],
        "final_robot_target_directed_travel_m": summary.get("final_robot_target_directed_travel_m"),
        "final_box_target_directed_travel_m": summary.get("final_box_target_directed_travel_m"),
        "final_robot_target_lateral_error_m": summary.get("final_robot_target_lateral_error_m"),
        "final_box_target_lateral_error_m": summary.get("final_box_target_lateral_error_m"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "max_box_tilt_rad": summary.get("max_box_tilt_rad"),
        "terminal_scale": summary.get("agile_command_hold_terminal_scale"),
        "terminal_latch": summary.get("agile_command_hold_terminal_latch_enabled"),
        "final_scale": summary.get("agile_command_hold_final_scale"),
        "final_threshold_m": summary.get("agile_command_hold_final_box_target_travel_m"),
        "final_latch": summary.get("agile_command_hold_final_latch_enabled"),
        "final_zero_corrections": summary.get("agile_command_hold_final_zero_corrections_enabled"),
        "final_lateral_suppressed_steps": summary.get(
            "agile_command_hold_final_lateral_suppressed_steps"
        ),
        "final_yaw_suppressed_steps": summary.get("agile_command_hold_final_yaw_suppressed_steps"),
        "final_max_command_x": summary.get("agile_command_hold_final_max_abs_command_x"),
        "final_max_command_y": summary.get("agile_command_hold_final_max_abs_command_y"),
        "final_max_command_yaw": summary.get("agile_command_hold_final_max_abs_command_yaw"),
        "final_hold_min_robot_z": summary.get("agile_command_hold_final_min_robot_z_m"),
        "final_hold_min_box_z": summary.get("agile_command_hold_final_min_box_z_m"),
        "final_hold_max_tilt": summary.get("agile_command_hold_final_max_tilt_rad"),
        "final_hold_max_box_tilt": summary.get("agile_command_hold_final_max_box_tilt_rad"),
        "final_hold_fall_events": summary.get("agile_command_hold_final_fall_events"),
        "final_hold_box_drop_events": summary.get("agile_command_hold_final_box_drop_events"),
        "final_hold_first_fall_step": summary.get("agile_command_hold_final_first_fall_step"),
        "final_hold_first_drop_step": summary.get(
            "agile_command_hold_final_first_box_drop_step"
        ),
        "final_stand_min_robot_z": summary.get("agile_command_hold_final_stand_min_robot_z_m"),
        "final_stand_min_box_z": summary.get("agile_command_hold_final_stand_min_box_z_m"),
        "final_stand_max_tilt": summary.get("agile_command_hold_final_stand_max_tilt_rad"),
        "final_stand_max_box_tilt": summary.get("agile_command_hold_final_stand_max_box_tilt_rad"),
        "final_stand_fall_events": summary.get("agile_command_hold_final_stand_fall_events"),
        "final_stand_box_drop_events": summary.get(
            "agile_command_hold_final_stand_box_drop_events"
        ),
        "final_active_steps": summary.get("agile_command_hold_final_active_steps"),
        "final_stand_enabled": summary.get("agile_command_hold_final_stand_enabled"),
        "final_stand_delay_steps": summary.get("agile_command_hold_final_stand_delay_steps"),
        "final_stand_active_steps": summary.get("agile_command_hold_final_stand_active_steps"),
        "target_window_enabled": summary.get("target_window_enabled"),
        "target_window_center_m": summary.get("target_window_center_m"),
        "target_window_halfwidth_m": summary.get("target_window_halfwidth_m"),
        "target_window_both_stable_steps": summary.get("target_window_both_stable_steps"),
        "target_window_both_longest_streak_steps": summary.get(
            "target_window_both_longest_streak_steps"
        ),
        "target_window_both_streak_at_end_steps": summary.get(
            "target_window_both_streak_at_end_steps"
        ),
        "target_window_both_final_hold_stable_steps": summary.get(
            "target_window_both_final_hold_stable_steps"
        ),
        "target_window_both_final_hold_longest_streak_steps": summary.get(
            "target_window_both_final_hold_longest_streak_steps"
        ),
        "target_window_both_final_hold_streak_at_end_steps": summary.get(
            "target_window_both_final_hold_streak_at_end_steps"
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
        "lateral_enabled": summary.get("agile_command_hold_lateral_correction_enabled"),
        "lateral_terminal_only": summary.get("agile_command_hold_lateral_terminal_only"),
        "lateral_error_start_m": summary.get("agile_command_hold_lateral_error_start_m"),
        "lateral_use_excess_error": summary.get("agile_command_hold_lateral_use_excess_error"),
        "lateral_max_tilt_rad": summary.get("agile_command_hold_lateral_max_tilt_rad"),
        "lateral_max_box_tilt_rad": summary.get("agile_command_hold_lateral_max_box_tilt_rad"),
        "lateral_suppressed_by_tilt_steps": summary.get(
            "agile_command_hold_lateral_suppressed_by_tilt_steps"
        ),
        "lateral_active_steps": summary.get("agile_command_hold_lateral_active_steps"),
        "lateral_max_abs_command": summary.get("agile_command_hold_lateral_max_abs_command"),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# G1 Low-Carry Control Matrix",
        "",
        "This is a diagnostic summary only, not a success claim.",
        "",
        "| case | check | fall/drop | first fall/drop | target travel robot/box | final lat robot/box | max tilt robot/box | terminal | final hold | lateral |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        if row.get("missing"):
            lines.append(f"| {row['label']} | missing | - | - | - | - | - | - | - | - |")
            continue
        terminal = (
            f"scale={_fmt(row['terminal_scale'])}, latch={_fmt(row['terminal_latch'])}, "
            f"final={_fmt(row['final_scale'])}@{_fmt(row['final_threshold_m'])}, "
            f"final_latch={_fmt(row['final_latch'])}, "
            f"zero_corr={_fmt(row['final_zero_corrections'])}, "
            f"zero_lat/yaw={_fmt(row['final_lateral_suppressed_steps'])}/"
            f"{_fmt(row['final_yaw_suppressed_steps'])}, "
            f"cmd_max={_fmt(row['final_max_command_x'])}/"
            f"{_fmt(row['final_max_command_y'])}/{_fmt(row['final_max_command_yaw'])}, "
            f"hold_z={_fmt(row['final_hold_min_robot_z'])}/{_fmt(row['final_hold_min_box_z'])}, "
            f"hold_tilt={_fmt(row['final_hold_max_tilt'])}/{_fmt(row['final_hold_max_box_tilt'])}, "
            f"hold_fall/drop={_fmt(row['final_hold_fall_events'])}/{_fmt(row['final_hold_box_drop_events'])}, "
            f"stand_z={_fmt(row['final_stand_min_robot_z'])}/{_fmt(row['final_stand_min_box_z'])}, "
            f"stand_tilt={_fmt(row['final_stand_max_tilt'])}/{_fmt(row['final_stand_max_box_tilt'])}, "
            f"stand_fall/drop={_fmt(row['final_stand_fall_events'])}/{_fmt(row['final_stand_box_drop_events'])}"
        )
        final_hold = (
            f"final_steps={_fmt(row['final_active_steps'])}, "
            f"final_stand={_fmt(row['final_stand_enabled'])}, "
            f"stand_delay={_fmt(row['final_stand_delay_steps'])}, "
            f"stand_steps={_fmt(row['final_stand_active_steps'])}, "
            f"target_window={_fmt(row['target_window_enabled'])}, "
            f"window={_fmt(row['target_window_center_m'])}+/-{_fmt(row['target_window_halfwidth_m'])}, "
            f"both_steps={_fmt(row['target_window_both_stable_steps'])}, "
            f"both_streak={_fmt(row['target_window_both_longest_streak_steps'])}, "
            f"both_end={_fmt(row['target_window_both_streak_at_end_steps'])}, "
            f"final_hold_steps={_fmt(row['target_window_both_final_hold_stable_steps'])}, "
            f"final_hold_streak={_fmt(row['target_window_both_final_hold_longest_streak_steps'])}, "
            f"final_hold_end={_fmt(row['target_window_both_final_hold_streak_at_end_steps'])}, "
            f"final_stand_steps={_fmt(row['target_window_both_final_stand_stable_steps'])}, "
            f"final_stand_streak={_fmt(row['target_window_both_final_stand_longest_streak_steps'])}, "
            f"final_stand_end={_fmt(row['target_window_both_final_stand_streak_at_end_steps'])}"
        )
        lateral = (
            f"on={_fmt(row['lateral_enabled'])}, terminal_only={_fmt(row['lateral_terminal_only'])}, "
            f"start={_fmt(row['lateral_error_start_m'])}, excess={_fmt(row['lateral_use_excess_error'])}, "
            f"tilt_gate={_fmt(row['lateral_max_tilt_rad'])}/{_fmt(row['lateral_max_box_tilt_rad'])}, "
            f"tilt_supp={_fmt(row['lateral_suppressed_by_tilt_steps'])}, "
            f"steps={_fmt(row['lateral_active_steps'])}, "
            f"max={_fmt(row['lateral_max_abs_command'])}"
        )
        lines.append(
            "| {label} | {check} | {fall}/{drop} | {first_fall}/{first_drop} | "
            "{robot_travel}/{box_travel} | {robot_lat}/{box_lat} | {tilt}/{box_tilt} | "
            "{terminal} | {final_hold} | {lateral} |".format(
                label=row["label"],
                check=_fmt(row["check_status"]),
                fall=_fmt(row["fall_events"]),
                drop=_fmt(row["box_drop_events"]),
                first_fall=_fmt(row["first_fall_step"]),
                first_drop=_fmt(row["first_box_drop_step"]),
                robot_travel=_fmt(row["final_robot_target_directed_travel_m"]),
                box_travel=_fmt(row["final_box_target_directed_travel_m"]),
                robot_lat=_fmt(row["final_robot_target_lateral_error_m"]),
                box_lat=_fmt(row["final_box_target_lateral_error_m"]),
                tilt=_fmt(row["max_tilt_rad"]),
                box_tilt=_fmt(row["max_box_tilt_rad"]),
                terminal=terminal,
                final_hold=final_hold,
                lateral=lateral,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = [_case_row(args.root, label, stamp) for label, stamp in DEFAULT_CASES]
    text = _markdown(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
