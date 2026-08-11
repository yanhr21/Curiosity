#!/usr/bin/env python3
"""Fully decode the retained CarryBox tactile presentation videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


SCENARIOS = {
    "successful_grasp": (
        "world_carrybox.mp4",
        "successful_carrybox_whole_hand_tactile.mp4",
        "left_detail.mp4",
        "right_detail.mp4",
        "palm_optical.mp4",
        "force_kinematics_friction_complete.mp4",
    ),
    "failed_grasp": (
        "world_carrybox.mp4",
        "failed_carrybox_whole_hand_tactile.mp4",
        "left_detail.mp4",
        "right_detail.mp4",
        "palm_optical.mp4",
    ),
    "failed_closure": (
        "world_carrybox.mp4",
        "failed_closure_carrybox_whole_hand_tactile.mp4",
    ),
}


def decode(video: Path, expected_frames: int, expected_fps: float) -> None:
    if not video.is_file():
        raise FileNotFoundError(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or frame.shape[:2] != (height, width):
            raise RuntimeError(f"invalid decoded frame {frames} in {video}")
        frames += 1
    capture.release()
    if frames != expected_frames:
        raise RuntimeError(f"{video.name}: decoded={frames}, expected={expected_frames}")
    if abs(fps - expected_fps) > 0.01:
        raise RuntimeError(f"{video.name}: fps={fps}, expected={expected_fps}")
    print(f"{video.name} decoded={frames}/{expected_frames} {width}x{height} {fps:g}fps")


def validate(run_root: Path, scenario: str) -> None:
    videos = SCENARIOS[scenario]
    summary = json.loads((run_root / "summary.json").read_text())
    source_frames = int(summary["source_frames"])
    for name in videos:
        video = run_root / name
        render_record = video.with_suffix(".render.json")
        force_record = run_root / "force_kinematics_friction_complete.audit.json"
        if name == "world_carrybox.mp4":
            expected, expected_fps = source_frames, 50.0
        elif render_record.is_file():
            record = json.loads(render_record.read_text())
            expected, expected_fps = int(record["frames"]), float(record["fps"])
        elif name == "force_kinematics_friction_complete.mp4" and force_record.is_file():
            record = json.loads(force_record.read_text())
            expected = int(record["video_declared_frames"])
            expected_fps = 1.0 / float(record["control_dt_s"])
        else:
            raise FileNotFoundError(f"missing render record for {video}")
        decode(video, expected, expected_fps)
    print(f"CarryBox tactile bundle full-decode PASS: {run_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("scenario", choices=(*SCENARIOS, "all"))
    args = parser.parse_args()
    if args.scenario == "all":
        for scenario in SCENARIOS:
            validate(args.run_root / scenario, scenario)
        print(f"All retained CarryBox tactile bundles full-decode PASS: {args.run_root}")
    else:
        validate(args.run_root, args.scenario)


if __name__ == "__main__":
    main()
