#!/usr/bin/env python3
"""Check Arena G1 stand/walk smoke summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Arena G1 stand/walk summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--max-fall-events", type=int, default=0)
    parser.add_argument("--min-forward-travel-x", type=float, default=None)
    parser.add_argument("--min-root-z", type=float, default=None)
    parser.add_argument("--max-tilt", type=float, default=None)
    parser.add_argument("--require-non-carry-claim", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    completed_steps = int(summary.get("completed_steps") or 0)
    if completed_steps < int(args.min_steps):
        failures.append(f"completed_steps too low: {completed_steps} < {args.min_steps}")
    fall_events = int(summary.get("fall_events") or 0)
    if fall_events > int(args.max_fall_events):
        failures.append(f"fall_events too high: {fall_events} > {args.max_fall_events}")
    if args.min_forward_travel_x is not None:
        travel = summary.get("final_forward_travel_x_m")
        if travel is None or float(travel) < float(args.min_forward_travel_x):
            failures.append(f"final_forward_travel_x_m too low: {travel} < {args.min_forward_travel_x}")
    if args.min_root_z is not None:
        min_root_z = summary.get("min_root_z_m")
        if min_root_z is None or float(min_root_z) < float(args.min_root_z):
            failures.append(f"min_root_z_m too low: {min_root_z} < {args.min_root_z}")
    if args.max_tilt is not None:
        max_tilt = summary.get("max_tilt_rad")
        if max_tilt is None or float(max_tilt) > float(args.max_tilt):
            failures.append(f"max_tilt_rad too high: {max_tilt} > {args.max_tilt}")
    if args.require_non_carry_claim:
        claim = str(summary.get("success_claim", ""))
        if "not_walking_or_carrying" in claim:
            failures.append(f"success_claim is stale and says not_walking_or_carrying: {claim}")
        if "carrying" in claim and "not" not in claim:
            failures.append(f"success_claim may overclaim carrying: {claim}")

    report = {
        "summary": str(args.summary),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "scene_type": summary.get("scene_type"),
        "success_claim": summary.get("success_claim"),
        "completed_steps": completed_steps,
        "command_xyz_yaw": summary.get("command_xyz_yaw"),
        "commanded_walk_steps": summary.get("commanded_walk_steps"),
        "final_forward_travel_x_m": summary.get("final_forward_travel_x_m"),
        "min_root_z_m": summary.get("min_root_z_m"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "fall_events": summary.get("fall_events"),
        "error": summary.get("error"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
