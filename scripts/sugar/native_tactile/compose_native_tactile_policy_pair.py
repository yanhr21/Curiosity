#!/usr/bin/env python3
"""Compose two completed policy videos into one matched side-by-side H.264."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if declared < 1:
        raise RuntimeError(f"video has no declared frames: {path}")
    return declared


def main() -> None:
    args = parse_args()
    left = args.left.expanduser().resolve()
    right = args.right.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (left, right):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    left_frames = video_frames(left)
    right_frames = video_frames(right)
    expected = min(left_frames, right_frames)
    filter_graph = (
        "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:white[left];"
        "[1:v]scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:white[right];"
        "[left][right]hstack=inputs=2:shortest=1[out]"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-an",
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )

    capture = cv2.VideoCapture(str(output))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = 0
    shapes: set[tuple[int, ...]] = set()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        shapes.add(tuple(frame.shape))
    capture.release()
    if declared != expected or decoded != expected:
        raise RuntimeError(
            f"full decode failed: expected={expected}, declared={declared}, decoded={decoded}"
        )
    if shapes != {(720, 2560, 3)}:
        raise RuntimeError(f"unexpected decoded shapes: {shapes}")

    record = {
        "schema": "native_tactile_policy_pair_video_v1",
        "left": str(left),
        "right": str(right),
        "left_frames": left_frames,
        "right_frames": right_frames,
        "paired_frames": expected,
        "fps": args.fps,
        "resolution": [2560, 720],
        "full_decode_passed": True,
        "semantics": "Each panel retains its own title; the pair ends at the shorter rollout.",
    }
    output.with_suffix(".render.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
