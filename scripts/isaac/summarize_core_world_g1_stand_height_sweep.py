#!/usr/bin/env python3
"""Summarize Core API G1 stand-with-box height sweep diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Core API G1 stand height sweep.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    cases = []
    failures = []
    for case_dir in sorted(args.root.glob("z_*")):
        summary_path = case_dir / "core_world_g1_box_scene_summary.json"
        check_path = case_dir / "strict_check_report.json"
        if not summary_path.exists():
            failures.append(f"{case_dir.name}: missing summary")
            continue
        summary = _load(summary_path)
        check = _load(check_path) if check_path.exists() else None
        passed = check is not None and check.get("status") == "pass"
        if not passed:
            failures.append(f"{case_dir.name}: strict stand gate failed")
        cases.append(
            {
                "case": case_dir.name,
                "passed": passed,
                "summary_path": str(summary_path),
                "check_report_path": str(check_path) if check_path.exists() else None,
                "completed_steps": summary.get("completed_steps"),
                "attach_box": summary.get("attach_box"),
                "joint_count": summary.get("joint_count"),
                "stand_drive_gains_enabled": summary.get("stand_drive_gains_enabled"),
                "applied_stand_drive_gain_count": summary.get("applied_stand_drive_gain_count"),
                "root_pose_write_count_setup": summary.get("root_pose_write_count_setup"),
                "root_pose_write_count_rollout": summary.get("root_pose_write_count_rollout"),
                "root_velocity_write_count_rollout": summary.get("root_velocity_write_count_rollout"),
                "box_pose_write_count_rollout": summary.get("box_pose_write_count_rollout"),
                "fall_events": summary.get("fall_events"),
                "box_drop_events": summary.get("box_drop_events"),
                "min_robot_z_m": summary.get("min_robot_z_m"),
                "min_box_z_m": summary.get("min_box_z_m"),
                "max_tilt_rad": summary.get("max_tilt_rad"),
                "max_robot_travel_xy_m": summary.get("max_robot_travel_xy_m"),
                "max_box_travel_xy_m": summary.get("max_box_travel_xy_m"),
                "error": summary.get("error"),
                "check_failures": (check or {}).get("failures"),
            }
        )
    passed_cases = [case for case in cases if case["passed"]]
    report = {
        "scene_type": "core_world_g1_stand_height_sweep_diagnostic",
        "status": "pass" if passed_cases else "fail",
        "success_claim": "g1_core_api_height_sweep_diagnostic_not_walking_or_carrying_success",
        "not_success_reason": (
            "tests only Core API G1 standing with a fixed-torso attached box; "
            "it is not walking, not free object grasping, and not a carrying policy"
        ),
        "case_count": len(cases),
        "passed_case_count": len(passed_cases),
        "best_passed_case": passed_cases[0] if passed_cases else None,
        "cases": cases,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
