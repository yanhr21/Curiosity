#!/usr/bin/env python3
"""Audit active-probe features from G1 probe-selected load validation cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _probe_window_csv_features(csv_path: Path | None, probe_start: int, probe_steps: int) -> dict[str, Any]:
    features: dict[str, Any] = {
        "probe_csv_path": str(csv_path) if csv_path else None,
        "probe_csv_exists": bool(csv_path and csv_path.is_file()),
    }
    if not csv_path or not csv_path.is_file():
        return features
    end_step = probe_start + probe_steps
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row.get("step") or 0))
            if probe_start <= step < end_step:
                rows.append(row)
    features["probe_csv_rows"] = len(rows)
    if not rows:
        return features

    def vals(key: str) -> list[float]:
        out = []
        for row in rows:
            raw = row.get(key)
            if raw not in (None, ""):
                out.append(float(raw))
        return out

    for key in (
        "tilt_rad",
        "box_tilt_rad",
        "box_robot_relative_offset_error_m",
        "robot_target_directed_travel_m",
        "box_target_directed_travel_m",
        "robot_target_lateral_error_m",
        "box_target_lateral_error_m",
    ):
        numbers = vals(key)
        if numbers:
            features[f"probe_min_{key}"] = min(numbers)
            features[f"probe_max_{key}"] = max(numbers)
            features[f"probe_final_{key}"] = numbers[-1]
    fall_values = vals("fall")
    drop_values = vals("drop")
    features["probe_fall_rows"] = int(sum(1 for v in fall_values if v > 0.5))
    features["probe_drop_rows"] = int(sum(1 for v in drop_values if v > 0.5))
    return features


def main() -> int:
    args = parse_args()
    aggregate_path = args.suite_dir / "probe_selected_load_validation_summary.json"
    aggregate = _load(aggregate_path)
    cases = []
    for case in aggregate.get("cases", []):
        pipeline_path = Path(case["summary_path"])
        pipeline = _load(pipeline_path)
        probe_summary = pipeline.get("probe_summary") or {}
        selected_summary = pipeline.get("selected_summary") or {}
        probe_summary_path = Path(pipeline.get("probe_summary_path") or "")
        probe_csv = probe_summary_path.with_name("core_world_g1_box_scene_state.csv")
        probe_start = int(probe_summary.get("probe_start_step") or 40)
        probe_steps = int(probe_summary.get("probe_active_steps") or 0)
        record = {
            "label": case.get("label"),
            "mass_kg": case.get("mass_kg"),
            "selector_decision": case.get("selector_decision"),
            "validation_status": case.get("validation_status"),
            "validation_fall_events": case.get("fall_events"),
            "validation_box_drop_events": case.get("box_drop_events"),
            "validation_target_window_end_streak": case.get("target_window_both_streak_at_end_steps"),
            "selection_uses_hidden_ground_truth": case.get("selection_uses_hidden_ground_truth"),
            "probe_summary_path": str(probe_summary_path),
            "probe_status": probe_summary.get("status"),
            "probe_fall_events": probe_summary.get("fall_events"),
            "probe_box_drop_events": probe_summary.get("box_drop_events"),
            "probe_completed_steps": probe_summary.get("completed_steps"),
            "probe_active_steps": probe_summary.get("probe_active_steps"),
            "probe_final_box_target_directed_travel_m": probe_summary.get("final_probe_box_target_directed_travel_m"),
            "probe_max_box_target_directed_travel_m": probe_summary.get("max_probe_box_target_directed_travel_m"),
            "probe_max_tilt_rad": probe_summary.get("max_tilt_rad"),
            "probe_max_box_tilt_rad": probe_summary.get("max_box_tilt_rad"),
            "probe_final_relative_error_m": probe_summary.get("final_box_robot_relative_offset_error_m"),
            "selected_final_robot_target_directed_travel_m": selected_summary.get(
                "final_robot_target_directed_travel_m"
            ),
            "selected_final_box_target_directed_travel_m": selected_summary.get(
                "final_box_target_directed_travel_m"
            ),
        }
        record.update(_probe_window_csv_features(probe_csv, probe_start, probe_steps))
        cases.append(record)

    observations: list[str] = []
    if cases and len({case.get("selector_decision") for case in cases}) == 1:
        observations.append("selector made the same decision for all masses")
    if any(int(case.get("probe_fall_events") or 0) > 0 for case in cases):
        observations.append("probe summaries already contain fall events for at least one mass")
    if any(int(case.get("probe_box_drop_events") or 0) > 0 for case in cases):
        observations.append("probe summaries already contain box-drop events for at least one mass")
    probe_travels = [case.get("probe_final_box_target_directed_travel_m") for case in cases]
    if len([v for v in probe_travels if v is not None]) == len(cases):
        observations.append("probe final target-directed box travel is not by itself a monotonic mass signal")

    report = {
        "scene_type": "core_world_g1_probe_selected_load_feature_audit",
        "success_claim": "feature_audit_only_not_final_carrying_success",
        "status": "pass",
        "suite_dir": str(args.suite_dir),
        "aggregate_summary": str(aggregate_path),
        "observations": observations,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
