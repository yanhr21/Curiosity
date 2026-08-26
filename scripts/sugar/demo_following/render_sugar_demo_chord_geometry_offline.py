#!/usr/bin/env python3
"""Render exact recorded G1 body/box poses with CHORD contact geometry.

This renderer is intentionally geometric: it uses the archived 35 G1 body
centres, exact object pose and reconstructed contact points.  It does not rerun
physics and does not pretend to be a photorealistic replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

from render_frozen_trace_behavior import (
    CONNECTION_NAMES,
    ROOT,
    SOURCE_HZ,
    draw_sequence,
    load_reference,
    project,
    recenter,
)


VIDEO_FPS = 20
BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "pelvis_contour_link", "right_hip_pitch_link",
    "waist_yaw_link", "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link",
    "right_knee_link", "head_link", "left_shoulder_pitch_link", "logo_link",
    "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_roll_link",
    "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link", "left_rubber_hand",
    "right_rubber_hand",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("CarryBox", "KickBox"), required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _decode(path: Path) -> dict[str, object]:
    process = __import__("subprocess").run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "info", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        stdout=__import__("subprocess").DEVNULL,
        stderr=__import__("subprocess").PIPE,
        text=True,
        check=False,
    )
    log = process.stderr
    return {
        "exit_code": process.returncode,
        "h264": "Video: h264" in log,
        "yuv420p": "yuv420p" in log,
        "passed": process.returncode == 0 and "Video: h264" in log and "yuv420p" in log,
    }


def _timeline(
    frame: np.ndarray,
    values: np.ndarray,
    y: int,
    color: tuple[int, int, int],
    current: int,
) -> None:
    x0, width = 915, 340
    cv2.rectangle(frame, (x0, y), (x0 + width, y + 18), (235, 235, 235), -1)
    for index in np.flatnonzero(values):
        first = x0 + round(width * index / max(1, values.size - 1))
        second = x0 + round(width * (index + 1) / max(1, values.size - 1))
        cv2.rectangle(frame, (first, y), (max(first + 1, second), y + 18), color, -1)
    marker = x0 + round(width * current / max(1, values.size - 1))
    cv2.line(frame, (marker, y - 3), (marker, y + 21), (20, 20, 20), 2)


def main() -> None:
    args = _arguments()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reference_raw = load_reference(args.motion_dir)
    focus_xy = 0.5 * (
        reference_raw["body"][0, 0, :2] + reference_raw["object_position"][0, :2]
    )
    reference = recenter(reference_raw)
    with np.load(args.geometry_dir / "contact_geometry.npz", allow_pickle=False) as archive:
        geometry = {name: np.asarray(archive[name]) for name in archive.files}
    result = json.loads((args.geometry_dir / "RESULT.json").read_text(encoding="utf-8"))
    frames = min(reference["body"].shape[0], geometry["frame"].size)
    if frames != 660 or result.get("passed") is not True:
        raise RuntimeError("unadmitted geometry input")
    name_to_id = {name: index for index, name in enumerate(BODY_NAMES)}
    missing = {name for pair in CONNECTION_NAMES for name in pair}.difference(name_to_id)
    if missing:
        raise RuntimeError(f"body order misses {sorted(missing)}")

    video_path = output / f"{args.task.lower()}_exact_demo_chord_contact_geometry.mp4"
    video_frames = math.ceil((frames - 1) * VIDEO_FPS / SOURCE_HZ) + 1
    with imageio.get_writer(
        video_path,
        fps=VIDEO_FPS,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="warning",
    ) as writer:
        for video_frame in range(video_frames):
            source = min(frames - 1, round(video_frame * SOURCE_HZ / VIDEO_FPS))
            frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
            draw_sequence(frame, reference, source, (120, 75), name_to_id)
            active = geometry["contact_active"][source]
            for side, color in ((0, (210, 105, 30)), (1, (45, 45, 220))):
                if not active[side]:
                    continue
                position = 0.5 * (
                    geometry["hand_contact_position_w"][source, side]
                    + geometry["object_contact_position_w"][source, side]
                )
                position = position.copy()
                position[:2] -= focus_xy
                uv = project(position[None], (120, 75))[0]
                cv2.circle(frame, tuple(uv), 13, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(uv), 9, color, -1, cv2.LINE_AA)

            cv2.putText(frame, f"EXACT SUGAR {args.task.upper()} DEMONSTRATION", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, "35 recorded G1 body centres + exact box pose", (190, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.line(frame, (890, 15), (890, 705), (185, 185, 185), 1)
            cv2.putText(frame, "CHORD CONTACT GEOMETRY", (910, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 2, cv2.LINE_AA)
            cv2.putText(frame, "official 1 cm mesh-proximity rule", (910, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (75, 75, 75), 1, cv2.LINE_AA)
            cv2.putText(frame, f"frame {source:03d}/659", (910, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)
            distances = 1000.0 * geometry["minimum_distance_m"][source]
            for row, (label, side, color) in enumerate((('LEFT', 0, (210, 105, 30)), ('RIGHT', 1, (45, 45, 220)))):
                y = 170 + 72 * row
                cv2.circle(frame, (930, y - 5), 10, color, -1, cv2.LINE_AA)
                state = "CONTACT" if active[side] else "no contact"
                cv2.putText(frame, f"{label}: {state}", (954, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 2, cv2.LINE_AA)
                cv2.putText(frame, f"distance {distances[side]:6.1f} mm", (954, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (75, 75, 75), 1, cv2.LINE_AA)
            stored = geometry["sugar_binary_contact_label"]
            cv2.putText(frame, f"stored binary label: {int(stored[source])}", (910, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(frame, "validation only - never geometry input", (910, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(frame, "stored label", (910, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (75, 75, 75), 1, cv2.LINE_AA)
            _timeline(frame, stored, 410, (150, 150, 150), source)
            cv2.putText(frame, "left geometry", (910, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (75, 75, 75), 1, cv2.LINE_AA)
            _timeline(frame, geometry["contact_active"][:, 0], 470, (210, 105, 30), source)
            cv2.putText(frame, "right geometry", (910, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (75, 75, 75), 1, cv2.LINE_AA)
            _timeline(frame, geometry["contact_active"][:, 1], 530, (45, 45, 220), source)
            timing = result["independent_binary_timing_check"]
            cv2.putText(frame, f"precision {100 * timing['precision']:5.1f}%", (910, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(frame, f"recall    {100 * timing['recall']:5.1f}%", (910, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
            note = "timing agrees" if timing["recall"] > 0.9 else "stored timing disagrees"
            color = (35, 120, 35) if timing["recall"] > 0.9 else (30, 30, 205)
            cv2.putText(frame, note, (910, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "geometry only - not force", (480, 717), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (85, 85, 85), 1, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    proof = {
        "passed": False,
        "task": args.task,
        "video": str(video_path),
        "frames": video_frames,
        "source_frames": frames,
        "full_demonstration_displayed": video_frames
        >= math.ceil((frames - 1) * VIDEO_FPS / SOURCE_HZ) + 1,
        "visualization_semantics": "exact recorded body centres and object pose; no physics replay",
        "decode": _decode(video_path),
    }
    proof["passed"] = bool(
        proof["full_demonstration_displayed"] and proof["decode"]["passed"]
    )
    (output / "RENDER_PROOF.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(proof, indent=2, sort_keys=True), flush=True)
    if not proof["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
