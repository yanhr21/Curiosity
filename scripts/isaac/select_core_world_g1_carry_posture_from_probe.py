#!/usr/bin/env python3
"""Select a G1 carry posture from observable probe telemetry.

This is a diagnostic heuristic, not a learned model. It uses visible box size
and logged probe displacement from a prior G1 probe run. It must not use hidden
payload mass as an input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOWCARRY_ENV = {
    "LARGERBOX_STRICT_MODE": "lowcarry",
    "AGILE_COMMAND_HOLD_YAW_CORRECTION": "1",
    "AGILE_COMMAND_HOLD_YAW_GAIN": "0.0",
    "AGILE_COMMAND_HOLD_YAW_LIMIT": "0.0",
    "AGILE_COMMAND_HOLD_YAW_SIGN": "-1.0",
    "AGILE_COMMAND_HOLD_LATERAL_CORRECTION": "1",
    "AGILE_COMMAND_HOLD_LATERAL_TERMINAL_ONLY": "1",
    "AGILE_COMMAND_HOLD_LATERAL_ERROR_START": "0.45",
    "AGILE_COMMAND_HOLD_LATERAL_USE_EXCESS_ERROR": "1",
    "AGILE_COMMAND_HOLD_LATERAL_GAIN": "0.006",
    "AGILE_COMMAND_HOLD_LATERAL_LIMIT": "0.0015",
    "AGILE_COMMAND_HOLD_LATERAL_SIGN": "1.0",
    "AGILE_COMMAND_HOLD_LATERAL_MAX_TILT": "0.30",
    "AGILE_COMMAND_HOLD_LATERAL_MAX_BOX_TILT": "0.35",
    "AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL": "0.65",
    "AGILE_COMMAND_HOLD_TERMINAL_SCALE": "0.015",
    "AGILE_COMMAND_HOLD_TERMINAL_LATCH": "1",
    "AGILE_COMMAND_HOLD_FINAL_BOX_TARGET_TRAVEL": "0.6",
    "AGILE_COMMAND_HOLD_FINAL_SCALE": "0.0",
    "AGILE_COMMAND_HOLD_FINAL_LATCH": "1",
}


CHESTPAD_ENV = {
    "LARGERBOX_STRICT_MODE": "chestpad",
    "AGILE_COMMAND_HOLD_YAW_CORRECTION": "1",
    "AGILE_COMMAND_HOLD_YAW_GAIN": "0.04",
    "AGILE_COMMAND_HOLD_YAW_LIMIT": "0.08",
    "AGILE_COMMAND_HOLD_YAW_SIGN": "-1.0",
    "AGILE_COMMAND_HOLD_TERMINAL_BOX_TARGET_TRAVEL": "1.05",
    "AGILE_COMMAND_HOLD_TERMINAL_SCALE": "0.015",
    "AGILE_COMMAND_HOLD_TERMINAL_LATCH": "1",
}


BOXTILT_ENV = {
    "LARGERBOX_STRICT_MODE": "boxtilt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select G1 carry posture from probe telemetry.")
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-env", type=Path, default=None)
    parser.add_argument("--resistant-probe-travel-threshold", type=float, default=0.015)
    parser.add_argument("--high-probe-travel-threshold", type=float, default=None)
    parser.add_argument("--tall-box-threshold", type=float, default=0.09)
    parser.add_argument("--wide-box-threshold", type=float, default=0.12)
    parser.add_argument("--max-probe-fall-events", type=int, default=None)
    parser.add_argument("--max-probe-box-drop-events", type=int, default=None)
    parser.add_argument("--max-probe-tilt", type=float, default=None)
    parser.add_argument("--max-probe-box-tilt", type=float, default=None)
    parser.add_argument("--probe-tilt-risk-threshold", type=float, default=None)
    parser.add_argument("--probe-box-tilt-risk-threshold", type=float, default=None)
    parser.add_argument("--probe-relative-offset-risk-threshold", type=float, default=None)
    parser.add_argument("--min-probe-completed-steps", type=int, default=None)
    return parser.parse_args()


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _write_env(path: Path, env: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"export {key}={value}\n" for key, value in sorted(env.items())]
    path.write_text("".join(lines))


def main() -> int:
    args = parse_args()
    summary = json.loads(args.probe_summary.read_text())
    failures: list[str] = []

    probe_mode = str(summary.get("probe_mode") or "none")
    probe_steps = int(summary.get("probe_active_steps") or 0)
    probe_travel = summary.get("final_probe_box_travel_xy_m")
    max_probe_travel = summary.get("max_probe_box_travel_xy_m")
    max_probe_target_travel = float(summary.get("max_probe_box_target_directed_travel_m") or 0.0)
    max_tilt = float(summary.get("max_tilt_rad") or 0.0)
    max_box_tilt = float(summary.get("max_box_tilt_rad") or 0.0)
    max_relative_offset = float(summary.get("max_box_robot_relative_offset_error_m") or 0.0)
    size = _float_list(summary.get("box_size_m"))

    if probe_mode == "none":
        failures.append("probe_mode is none; run a probe diagnostic before posture selection")
    if summary.get("error") is not None:
        failures.append(f"probe summary error: {summary.get('error')}")
    if probe_steps <= 0:
        failures.append("probe_active_steps is zero or missing")
    if args.min_probe_completed_steps is not None:
        completed_steps = int(summary.get("completed_steps") or 0)
        if completed_steps < int(args.min_probe_completed_steps):
            failures.append(
                f"probe completed_steps {completed_steps} < {args.min_probe_completed_steps}"
            )
    if probe_travel is None and max_probe_travel is None:
        failures.append("probe travel fields are missing")
    if len(size) != 3:
        failures.append("box_size_m must contain visible x/y/z dimensions")
    if args.max_probe_fall_events is not None:
        value = int(summary.get("fall_events") or 0)
        if value > int(args.max_probe_fall_events):
            failures.append(f"probe fall_events {value} > {args.max_probe_fall_events}")
    if args.max_probe_box_drop_events is not None:
        value = int(summary.get("box_drop_events") or 0)
        if value > int(args.max_probe_box_drop_events):
            failures.append(f"probe box_drop_events {value} > {args.max_probe_box_drop_events}")
    if args.max_probe_tilt is not None:
        value = float(summary.get("max_tilt_rad") or 999.0)
        if value > float(args.max_probe_tilt):
            failures.append(f"probe max_tilt_rad {value} > {args.max_probe_tilt}")
    if args.max_probe_box_tilt is not None:
        value = float(summary.get("max_box_tilt_rad") or 999.0)
        if value > float(args.max_probe_box_tilt):
            failures.append(f"probe max_box_tilt_rad {value} > {args.max_probe_box_tilt}")

    probe_motion = float(max_probe_travel if max_probe_travel is not None else probe_travel or 0.0)
    box_y = float(size[1]) if len(size) == 3 else 0.0
    box_z = float(size[2]) if len(size) == 3 else 0.0

    resistant = probe_motion <= float(args.resistant_probe_travel_threshold)
    high_motion = (
        args.high_probe_travel_threshold is not None
        and max_probe_target_travel >= float(args.high_probe_travel_threshold)
    )
    robot_tilt_risk = (
        args.probe_tilt_risk_threshold is not None
        and max_tilt >= float(args.probe_tilt_risk_threshold)
    )
    box_tilt_risk = (
        args.probe_box_tilt_risk_threshold is not None
        and max_box_tilt >= float(args.probe_box_tilt_risk_threshold)
    )
    relative_offset_risk = (
        args.probe_relative_offset_risk_threshold is not None
        and max_relative_offset >= float(args.probe_relative_offset_risk_threshold)
    )
    tall = box_z >= float(args.tall_box_threshold)
    wide = box_y >= float(args.wide_box_threshold)

    reasons: list[str] = []
    if resistant:
        reasons.append("probe_motion_below_threshold")
    if high_motion:
        reasons.append("probe_target_motion_above_threshold")
    if robot_tilt_risk:
        reasons.append("probe_robot_tilt_above_risk_threshold")
    if box_tilt_risk:
        reasons.append("probe_box_tilt_above_risk_threshold")
    if relative_offset_risk:
        reasons.append("probe_relative_offset_above_risk_threshold")
    if tall:
        reasons.append("visible_box_tall")
    if wide:
        reasons.append("visible_box_wide")

    if resistant or robot_tilt_risk or box_tilt_risk or relative_offset_risk or tall or wide:
        selected = "chestpad"
        env = dict(CHESTPAD_ENV)
    elif high_motion:
        selected = "boxtilt"
        env = dict(BOXTILT_ENV)
    else:
        selected = "lowcarry"
        env = dict(LOWCARRY_ENV)

    report = {
        "scene_type": "core_world_g1_probe_selected_posture_diagnostic",
        "success_claim": "diagnostic_selector_not_final_autonomous_or_learned_success",
        "probe_summary": str(args.probe_summary),
        "selection_uses_hidden_ground_truth": False,
        "hidden_box_mass_seen_but_ignored": summary.get("box_mass_kg") is not None,
        "probe_mode": probe_mode,
        "probe_active_steps": probe_steps,
        "probe_motion_m": probe_motion,
        "box_size_m": size,
        "thresholds": {
            "resistant_probe_travel_m": float(args.resistant_probe_travel_threshold),
            "tall_box_z_m": float(args.tall_box_threshold),
            "wide_box_y_m": float(args.wide_box_threshold),
            "high_probe_travel_m": args.high_probe_travel_threshold,
            "max_probe_fall_events": args.max_probe_fall_events,
            "max_probe_box_drop_events": args.max_probe_box_drop_events,
            "max_probe_tilt_rad": args.max_probe_tilt,
            "max_probe_box_tilt_rad": args.max_probe_box_tilt,
            "probe_tilt_risk_rad": args.probe_tilt_risk_threshold,
            "probe_box_tilt_risk_rad": args.probe_box_tilt_risk_threshold,
            "probe_relative_offset_risk_m": args.probe_relative_offset_risk_threshold,
            "min_probe_completed_steps": args.min_probe_completed_steps,
        },
        "decision_flags": {
            "resistant": resistant,
            "high_motion": high_motion,
            "robot_tilt_risk": robot_tilt_risk,
            "box_tilt_risk": box_tilt_risk,
            "relative_offset_risk": relative_offset_risk,
            "tall": tall,
            "wide": wide,
        },
        "selection_reasons": reasons or ["probe_motion_and_visible_size_low_risk"],
        "selected_posture": selected,
        "selected_env": env,
        "output_env": str(args.output_env) if args.output_env else None,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_env is not None:
        _write_env(args.output_env, env)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
