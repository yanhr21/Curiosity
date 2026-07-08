#!/usr/bin/env python3
"""Summarize direct-Isaac anchored posture sweep diagnostics.

This is a diagnostic selector, not a learned policy.  It only ranks completed
Isaac runs by observable safety and task metrics so the scene can exercise
multiple carrying postures without waiting on external models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize anchored posture sweep results.")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["_summary_path"] = str(path)
    data["_candidate"] = path.parent.name
    return data


def _score_run(data: dict) -> float:
    if data.get("error"):
        return 1.0e9
    completed = int(data.get("completed_steps") or 0)
    requested = int(data.get("steps_requested") or 0)
    incomplete_penalty = max(0, requested - completed) * 100.0
    fall_drop_penalty = 100000.0 * (
        int(data.get("fall_events") or 0) + int(data.get("box_drop_events") or 0)
    )
    shortcut_penalty = 100000.0 * sum(
        int(data.get(field) or 0)
        for field in (
            "root_pose_write_count",
            "root_velocity_write_count",
            "root_angular_velocity_write_count",
            "body_root_pose_write_count",
            "body_root_velocity_command_count",
            "box_pose_write_count",
            "payload_pose_write_count",
        )
    )
    target_error = abs(float(data.get("final_target_distance_x_m") or 0.0))
    payload_target_error = abs(float(data.get("final_payload_target_distance_x_m") or 0.0))
    relative_error = abs(float(data.get("max_payload_relative_offset_error_m") or 0.0))
    tilt = abs(float(data.get("max_tilt_rad") or 0.0))
    probe_risk = float(data.get("probe_risk_score") or 0.0)
    z_effort = float(data.get("mean_probe_support_foot_z_measured_effort") or 0.0)
    x_effort = float(data.get("mean_probe_support_foot_x_measured_effort") or 0.0)
    effort_proxy = (x_effort + z_effort) / 100000.0
    return (
        fall_drop_penalty
        + shortcut_penalty
        + incomplete_penalty
        + 100.0 * target_error
        + 100.0 * payload_target_error
        + 40.0 * relative_error
        + 10.0 * tilt
        + 3.0 * probe_risk
        + effort_proxy
    )


def main() -> int:
    args = parse_args()
    summary_paths = sorted(args.sweep_dir.glob("*/core_world_anchored_footstep_carrier_summary.json"))
    runs = [_load_summary(path) for path in summary_paths]
    ranked = []
    for data in runs:
        ranked.append(
            {
                "candidate": data["_candidate"],
                "score_lower_is_better": _score_run(data),
                "summary_path": data["_summary_path"],
                "payload_local_x_m": data.get("payload_local_x_m"),
                "payload_local_z_m": data.get("payload_local_z_m"),
                "payload_mass_kg": data.get("payload_mass_kg"),
                "payload_size_m": data.get("payload_size_m"),
                "payload_com_offset_m": data.get("payload_com_offset_m"),
                "completed_steps": data.get("completed_steps"),
                "steps_requested": data.get("steps_requested"),
                "fall_events": data.get("fall_events"),
                "box_drop_events": data.get("box_drop_events"),
                "final_target_distance_x_m": data.get("final_target_distance_x_m"),
                "final_payload_target_distance_x_m": data.get("final_payload_target_distance_x_m"),
                "max_payload_relative_offset_error_m": data.get("max_payload_relative_offset_error_m"),
                "max_tilt_rad": data.get("max_tilt_rad"),
                "probe_risk_score": data.get("probe_risk_score"),
                "probe_load_risk_bucket": data.get("probe_load_risk_bucket"),
                "mean_probe_support_foot_x_measured_effort": data.get("mean_probe_support_foot_x_measured_effort"),
                "mean_probe_support_foot_z_measured_effort": data.get("mean_probe_support_foot_z_measured_effort"),
                "success_claim": data.get("success_claim"),
                "carrier_claim": data.get("carrier_claim"),
            }
        )
    ranked.sort(key=lambda item: float(item["score_lower_is_better"]))
    report = {
        "scene_type": "direct_isaac_anchored_posture_sweep_diagnostic",
        "status": "complete" if ranked else "no_runs_found",
        "selector_type": "metric_ranker_not_learned_policy",
        "not_success_reason": "anchored support diagnostic; not full free-walking robot or non-retargeting RL",
        "sweep_dir": str(args.sweep_dir),
        "best_candidate": ranked[0]["candidate"] if ranked else None,
        "ranked_candidates": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ranked else 1


if __name__ == "__main__":
    raise SystemExit(main())
