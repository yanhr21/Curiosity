#!/usr/bin/env python3
"""Check MuJoCo quadruped payload diagnostic summaries.

This reads only JSON, so it is safe on the login node. Passing this checker is
only a diagnostic gate; it is not unknown-object grasping or final carrying.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MuJoCo quadruped payload summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expect-assist-mode", default=None)
    parser.add_argument("--max-fall-events", type=int, default=None)
    parser.add_argument("--min-travel-x", type=float, default=None)
    parser.add_argument("--max-tilt", type=float, default=None)
    parser.add_argument("--max-root-pose-writes", type=int, default=None)
    parser.add_argument("--max-root-velocity-writes", type=int, default=None)
    parser.add_argument("--min-external-force-writes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    if int(summary.get("completed_steps", 0)) < int(summary.get("steps_requested", 0)):
        failures.append(f"incomplete steps: {summary.get('completed_steps')} / {summary.get('steps_requested')}")
    if args.expect_assist_mode is not None and summary.get("assist_mode") != args.expect_assist_mode:
        failures.append(f"assist mode mismatch: {summary.get('assist_mode')} != {args.expect_assist_mode}")
    if args.max_fall_events is not None and int(summary.get("fall_events", 0)) > args.max_fall_events:
        failures.append(f"fall events too high: {summary.get('fall_events')}")
    if args.min_travel_x is not None and float(summary.get("max_travel_x_m", 0.0)) < args.min_travel_x:
        failures.append(f"x travel too low: {summary.get('max_travel_x_m')}")
    if args.max_tilt is not None and float(summary.get("max_tilt_rad", 0.0)) > args.max_tilt:
        failures.append(f"tilt too high: {summary.get('max_tilt_rad')}")
    if args.max_root_pose_writes is not None and int(summary.get("root_pose_write_count", 0)) > args.max_root_pose_writes:
        failures.append(f"root pose writes too high: {summary.get('root_pose_write_count')}")
    if args.max_root_velocity_writes is not None and int(summary.get("root_velocity_write_count", 0)) > args.max_root_velocity_writes:
        failures.append(f"root velocity writes too high: {summary.get('root_velocity_write_count')}")
    if args.min_external_force_writes is not None and int(summary.get("external_force_write_count", 0)) < args.min_external_force_writes:
        failures.append(f"external force writes too low: {summary.get('external_force_write_count')}")

    report = {
        "summary": str(args.summary),
        "scene_type": summary.get("scene_type"),
        "success_claim": summary.get("success_claim"),
        "payload_mode": summary.get("payload_mode"),
        "payload_mass_kg": summary.get("payload_mass_kg"),
        "assist_mode": summary.get("assist_mode"),
        "external_stabilizer_enabled": summary.get("external_stabilizer_enabled"),
        "root_pose_write_count": summary.get("root_pose_write_count"),
        "root_velocity_write_count": summary.get("root_velocity_write_count"),
        "external_force_write_count": summary.get("external_force_write_count"),
        "external_torque_write_count": summary.get("external_torque_write_count"),
        "completed_steps": summary.get("completed_steps"),
        "steps_requested": summary.get("steps_requested"),
        "max_travel_x_m": summary.get("max_travel_x_m"),
        "min_torso_z_m": summary.get("min_torso_z_m"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "fall_events": summary.get("fall_events"),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
