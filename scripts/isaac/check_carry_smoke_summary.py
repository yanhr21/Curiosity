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
    parser.add_argument("--min-joint-count", type=int, default=None)
    parser.add_argument("--expect-wbc-mode", default=None)
    parser.add_argument("--expect-attach-box", default=None)
    parser.add_argument("--max-root-pose-writes-rollout", type=int, default=0)
    parser.add_argument("--max-root-velocity-writes-rollout", type=int, default=0)
    parser.add_argument("--max-box-pose-writes-rollout", type=int, default=0)
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

    if args.expect_wbc_mode is not None and data.get("wbc_mode") != args.expect_wbc_mode:
        failures.append(f"wbc_mode {data.get('wbc_mode')} != {args.expect_wbc_mode}")
    if args.expect_attach_box is not None and data.get("attach_box") != args.expect_attach_box:
        failures.append(f"attach_box {data.get('attach_box')} != {args.expect_attach_box}")
    if args.min_joint_count is not None:
        joint_count = int(data.get("articulated_joint_count", 0))
        if joint_count < args.min_joint_count:
            failures.append(f"articulated_joint_count {joint_count} < {args.min_joint_count}")
    for field, limit in (
        ("root_pose_write_count_rollout", args.max_root_pose_writes_rollout),
        ("root_velocity_write_count_rollout", args.max_root_velocity_writes_rollout),
        ("box_pose_write_count_rollout", args.max_box_pose_writes_rollout),
    ):
        value = int(data.get(field, 0))
        if value > limit:
            failures.append(f"{field} {value} > {limit}")

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
    print(f"[OK] articulated_joint_count={data.get('articulated_joint_count')}")
    print(f"[OK] root_pose_write_count_rollout={data.get('root_pose_write_count_rollout')}")
    print(f"[OK] root_velocity_write_count_rollout={data.get('root_velocity_write_count_rollout')}")


if __name__ == "__main__":
    main()
