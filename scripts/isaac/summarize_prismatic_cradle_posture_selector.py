#!/usr/bin/env python3
"""Summarize prismatic-cradle posture candidates and choose a rule baseline.

This is a diagnostic selector for completed Isaac scaffold runs. It is not a
learned policy, not active probing, and not evidence of humanoid walking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHORTCUT_FIELDS = (
    "root_pose_write_count",
    "root_velocity_write_count",
    "root_angular_velocity_write_count",
    "body_root_pose_write_count",
    "body_root_velocity_command_count",
    "box_pose_write_count",
    "payload_pose_write_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prismatic cradle posture selector candidates.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--min-payload-z", type=float, default=0.70)
    parser.add_argument("--selector-min-height-margin", type=float, default=0.01)
    parser.add_argument("--max-tilt", type=float, default=0.13)
    parser.add_argument("--max-relative-offset", type=float, default=0.12)
    parser.add_argument("--max-target-error", type=float, default=0.03)
    parser.add_argument("--min-abs-post-settle-travel", type=float, default=0.15)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _float(data: dict[str, Any], field: str, default: float = 0.0) -> float:
    value = data.get(field)
    if value is None:
        return default
    return float(value)


def _int(data: dict[str, Any], field: str, default: int = 0) -> int:
    value = data.get(field)
    if value is None:
        return default
    return int(value)


def _posture_name(payload_x: float | None, payload_z: float | None) -> str:
    if payload_z is None:
        height = "unknown_height"
    elif payload_z < 0.15:
        height = "low"
    elif payload_z < 0.175:
        height = "mid"
    else:
        height = "high"
    if payload_x is None:
        reach = "unknown_reach"
    elif payload_x < 0.475:
        reach = "close"
    else:
        reach = "front"
    return f"{height}_{reach}"


def _evaluate_case(case: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = root / str(case["summary"])
    row: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "box_condition": case.get("box_condition"),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
    }
    if not summary_path.exists():
        row.update({"passed_gate": False, "failures": [f"missing summary: {summary_path}"]})
        return row

    data = _read_json(summary_path)
    failures: list[str] = []
    completed = _int(data, "completed_steps")
    requested = _int(data, "steps_requested")
    if data.get("error"):
        failures.append(f"summary error: {data.get('error')}")
    if requested <= 0 or completed < requested:
        failures.append(f"incomplete rollout: {completed}/{requested}")
    if _int(data, "fall_events") != 0:
        failures.append(f"fall events: {data.get('fall_events')}")
    if _int(data, "box_drop_events") != 0:
        failures.append(f"box drop events: {data.get('box_drop_events')}")
    if _int(data, "nonfinite_state_events") != 0:
        failures.append(f"nonfinite events: {data.get('nonfinite_state_events')}")
    shortcut_count = sum(_int(data, field) for field in SHORTCUT_FIELDS)
    if shortcut_count != 0:
        failures.append(f"shortcut writes: {shortcut_count}")
    if not bool(data.get("articulated_carrier_enabled")):
        failures.append("articulated carrier not enabled")
    if not bool(data.get("foot_contact_drive_enabled")):
        failures.append("foot contact drive not enabled")
    if _int(data, "articulated_joint_count") < 8:
        failures.append(f"joint count too low: {data.get('articulated_joint_count')}")

    min_payload_z = _float(data, "min_payload_z_m")
    max_tilt = _float(data, "max_tilt_rad")
    max_offset = _float(data, "max_payload_relative_offset_error_m")
    post_settle_offset = data.get("max_post_settle_payload_relative_offset_error_m")
    post_settle_offset_f = _float(data, "max_post_settle_payload_relative_offset_error_m", default=max_offset)
    target_error = _float(data, "final_post_settle_payload_target_distance_x_m", default=999.0)
    travel = _float(data, "max_abs_post_settle_payload_travel_x_m")
    if min_payload_z < float(args.min_payload_z):
        failures.append(f"payload z below gate: {min_payload_z:.5f}")
    if max_tilt > float(args.max_tilt):
        failures.append(f"tilt above gate: {max_tilt:.5f}")
    if max_offset > float(args.max_relative_offset):
        failures.append(f"relative offset above gate: {max_offset:.5f}")
    if target_error > float(args.max_target_error):
        failures.append(f"target error above gate: {target_error:.5f}")
    if travel < float(args.min_abs_post_settle_travel):
        failures.append(f"post-settle travel below gate: {travel:.5f}")

    payload_x = data.get("payload_local_x_m")
    payload_z = data.get("payload_local_z_m")
    payload_x_f = None if payload_x is None else float(payload_x)
    payload_z_f = None if payload_z is None else float(payload_z)
    height_margin = min_payload_z - float(args.min_payload_z)
    selector_eligible = not failures and height_margin >= float(args.selector_min_height_margin)
    selector_metric_cost = (
        100.0 * target_error
        + 12.0 * max_tilt
        + 30.0 * post_settle_offset_f
        + 15.0 * max_offset
    )
    row.update(
        {
            "passed_gate": not failures,
            "selector_eligible": selector_eligible,
            "failures": failures,
            "posture": _posture_name(payload_x_f, payload_z_f),
            "payload_local_x_m": payload_x_f,
            "payload_local_z_m": payload_z_f,
            "payload_mass_kg": data.get("payload_mass_kg"),
            "payload_size_m": data.get("payload_size_m"),
            "completed_steps": completed,
            "steps_requested": requested,
            "fall_events": data.get("fall_events"),
            "box_drop_events": data.get("box_drop_events"),
            "nonfinite_state_events": data.get("nonfinite_state_events"),
            "shortcut_write_count": shortcut_count,
            "min_payload_z_m": min_payload_z,
            "height_margin_m": height_margin,
            "max_tilt_rad": max_tilt,
            "max_payload_relative_offset_error_m": max_offset,
            "max_post_settle_payload_relative_offset_error_m": post_settle_offset,
            "final_post_settle_payload_target_distance_x_m": target_error,
            "max_abs_post_settle_payload_travel_x_m": travel,
            "final_post_settle_payload_travel_x_m": data.get("final_post_settle_payload_travel_x_m"),
            "gated_step_last_block_reason": data.get("gated_step_last_block_reason"),
            "selector_metric_cost_lower_is_better": selector_metric_cost,
            "selection_sort_key": [
                0 if selector_eligible else 1,
                999.0 if payload_z_f is None else payload_z_f,
                selector_metric_cost,
            ],
            "success_claim": data.get("success_claim"),
        }
    )
    return row


def _select_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("selector_eligible")]
    passed = [row for row in rows if row.get("passed_gate")]
    pool = eligible or passed
    if not pool:
        return {
            "status": "no_passing_candidate",
            "selected_case_id": None,
            "selected_posture": None,
            "reason": "no candidate passed the selector gate",
        }
    selected = sorted(pool, key=lambda row: row["selection_sort_key"])[0]
    return {
        "status": "selected" if eligible else "selected_without_height_margin",
        "selected_case_id": selected.get("case_id"),
        "selected_posture": selected.get("posture"),
        "selected_summary_path": selected.get("summary_path"),
        "selected_payload_local_x_m": selected.get("payload_local_x_m"),
        "selected_payload_local_z_m": selected.get("payload_local_z_m"),
        "selected_height_margin_m": selected.get("height_margin_m"),
        "selected_final_target_error_m": selected.get("final_post_settle_payload_target_distance_x_m"),
        "selected_max_tilt_rad": selected.get("max_tilt_rad"),
        "selected_max_relative_offset_m": selected.get("max_payload_relative_offset_error_m"),
        "selected_metric_cost_lower_is_better": selected.get("selector_metric_cost_lower_is_better"),
        "reason": (
            "lowest payload_local_z among candidates that pass safety/transport "
            "gates and meet the height-margin threshold; ties use target/tilt/"
            "relative-offset cost"
        ),
    }


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    manifest = _read_json(args.manifest)
    rows = [_evaluate_case(case, root, args) for case in manifest.get("cases", [])]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("box_condition")), []).append(row)
    selections = {
        condition: _select_group(sorted(group_rows, key=lambda row: str(row.get("case_id"))))
        for condition, group_rows in sorted(groups.items())
    }
    failures = []
    if not rows:
        failures.append("manifest contains no cases")
    for condition, selection in selections.items():
        if selection.get("selected_case_id") is None:
            failures.append(f"{condition}: no selected posture")

    report = {
        "scene_type": "direct_isaac_prismatic_cradle_posture_selector_diagnostic",
        "status": "pass" if not failures else "fail",
        "selector_type": "rule_based_metric_selector_not_learned_policy",
        "success_claim": "posture_choice_scaffold_diagnostic_not_complete_robot_success",
        "not_success_reason": (
            "uses completed prismatic scaffold rollouts; no active probing, no "
            "learned policy, no humanoid walking backend, and no video guidance"
        ),
        "manifest": str(args.manifest),
        "gate": {
            "min_payload_z_m": float(args.min_payload_z),
            "selector_min_height_margin_m": float(args.selector_min_height_margin),
            "max_tilt_rad": float(args.max_tilt),
            "max_relative_offset_m": float(args.max_relative_offset),
            "max_target_error_m": float(args.max_target_error),
            "min_abs_post_settle_travel_m": float(args.min_abs_post_settle_travel),
        },
        "selection_policy": (
            "choose the lowest passing carry height with at least the configured "
            "payload-height margin; break ties using target error, tilt, and "
            "payload relative offset"
        ),
        "selections": selections,
        "candidate_count": len(rows),
        "passed_candidate_count": sum(1 for row in rows if row.get("passed_gate")),
        "selector_eligible_count": sum(1 for row in rows if row.get("selector_eligible")),
        "candidates": sorted(rows, key=lambda row: (str(row.get("box_condition")), row["selection_sort_key"])),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.table_output.open("w") as table_file:
        for row in report["candidates"]:
            table_file.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
