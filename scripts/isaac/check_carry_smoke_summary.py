#!/usr/bin/env python3
"""Check minimal carry-scene smoke summary JSON.

This is a lightweight post-run checker. It does not prove the full research
goal; it only guards basic diagnostics such as no fall, no dropped box, and
minimum travel distance for walk/payload smokes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check direct Isaac carry-scene smoke summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--mode", choices=("stand", "walk", "payload"), required=True)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--min-robot-travel", type=float, default=None)
    parser.add_argument("--max-fall-events", type=int, default=0)
    parser.add_argument("--max-box-drop-events", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text())
    failures: list[str] = []

    completed_steps = int(data.get("completed_steps", 0))
    fall_events = int(data.get("fall_events", 0))
    box_drop_events = int(data.get("box_drop_events", 0))
    robot_travel = float(data.get("max_robot_travel_xy_m", 0.0))

    if completed_steps < args.min_steps:
        failures.append(f"completed_steps {completed_steps} < {args.min_steps}")
    if fall_events > args.max_fall_events:
        failures.append(f"fall_events {fall_events} > {args.max_fall_events}")
    if box_drop_events > args.max_box_drop_events:
        failures.append(f"box_drop_events {box_drop_events} > {args.max_box_drop_events}")

    min_robot_travel = args.min_robot_travel
    if min_robot_travel is None:
        min_robot_travel = 0.0 if args.mode == "stand" else 0.25
    if robot_travel < min_robot_travel:
        failures.append(f"max_robot_travel_xy_m {robot_travel:.3f} < {min_robot_travel:.3f}")

    if failures:
        print("[FAIL] Carry-scene smoke summary check failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("[OK] Carry-scene smoke summary check passed.")
    print(f"[OK] mode={args.mode}")
    print(f"[OK] completed_steps={completed_steps}")
    print(f"[OK] fall_events={fall_events}")
    print(f"[OK] box_drop_events={box_drop_events}")
    print(f"[OK] max_robot_travel_xy_m={robot_travel:.3f}")


if __name__ == "__main__":
    main()
