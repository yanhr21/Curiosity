#!/usr/bin/env python3
"""Render one synchronized CarryBox mass jump using 27 patch units per hand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


WIDTH = 1600
HEIGHT = 1000
WORLD_HEIGHT = 600
PATCH_NAMES = (
    *(f"P{row}{column}" for row in range(4) for column in range(3)),
    *(f"{digit}{segment}" for digit in ("T", "I", "M", "R", "L") for segment in ("P", "M", "D")),
)
SLIP_COLORS = {
    0: (170, 170, 170),
    1: (70, 150, 70),
    2: (0, 190, 255),
    3: (20, 20, 230),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scale-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Online whole-hand patch tactile mass jump")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--profile-index",
        type=int,
        default=0,
        help="Profile axis to select from a frozen-evaluation trace.",
    )
    return parser.parse_args()


def fit_world(frame: np.ndarray) -> np.ndarray:
    scale = min(WIDTH / frame.shape[1], WORLD_HEIGHT / frame.shape[0])
    resized = cv2.resize(
        frame,
        (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((WORLD_HEIGHT, WIDTH, 3), dtype=np.uint8)
    x = (WIDTH - resized.shape[1]) // 2
    y = (WORLD_HEIGHT - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def pressure_color(value: float, scale: float) -> tuple[int, int, int]:
    fraction = float(np.clip(value / max(scale, 1.0e-9), 0.0, 1.0))
    return (
        int(255 - 205 * fraction),
        int(255 - 225 * fraction),
        255,
    )


def draw_patch(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    feature: np.ndarray,
    slip_state: int,
    pressure_scale: float,
    shear_scale: float,
) -> None:
    x0, y0, x1, y1 = box
    contact, load, pressure, shear_x, shear_y, friction = feature
    fill = pressure_color(float(pressure), pressure_scale) if contact > 0.5 else (245, 245, 245)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
    border = SLIP_COLORS[int(slip_state)] if contact > 0.5 else (190, 190, 190)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, 3 if contact > 0.5 else 1)
    cv2.putText(canvas, label, (x0 + 4, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{load:.1f}N", (x0 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (30, 30, 30), 1, cv2.LINE_AA)
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    arrow_scale = 0.38 * min(x1 - x0, y1 - y0) / max(shear_scale, 1.0e-9)
    end = (
        int(round(center[0] + shear_x * arrow_scale)),
        int(round(center[1] - shear_y * arrow_scale)),
    )
    cv2.arrowedLine(canvas, center, end, (30, 80, 190), 2, cv2.LINE_AA, tipLength=0.25)
    if contact > 0.5:
        cv2.putText(canvas, f"U {friction:.1f}", (x1 - 43, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70, 70, 70), 1, cv2.LINE_AA)


def draw_hand(
    canvas: np.ndarray,
    origin_x: int,
    title: str,
    features: np.ndarray,
    slip_state: np.ndarray,
    pressure_scale: float,
    shear_scale: float,
) -> None:
    cv2.putText(canvas, title, (origin_x + 8, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    palm_x = origin_x + 18
    palm_y = 670
    palm_w, palm_h = 66, 58
    for row in range(4):
        for column in range(3):
            index = row * 3 + column
            x0 = palm_x + column * palm_w
            y0 = palm_y + row * palm_h
            draw_patch(
                canvas,
                (x0, y0, x0 + palm_w - 5, y0 + palm_h - 5),
                PATCH_NAMES[index],
                features[index],
                int(slip_state[index]),
                pressure_scale,
                shear_scale,
            )
    finger_x = origin_x + 250
    finger_y = 670
    finger_w, finger_h = 98, 72
    for digit in range(5):
        for display_row, segment in enumerate((2, 1, 0)):
            index = 12 + digit * 3 + segment
            x0 = finger_x + digit * finger_w
            y0 = finger_y + display_row * finger_h
            draw_patch(
                canvas,
                (x0, y0, x0 + finger_w - 7, y0 + finger_h - 7),
                PATCH_NAMES[index],
                features[index],
                int(slip_state[index]),
                pressure_scale,
                shear_scale,
            )


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    online_trace = root / "online_mass_jump_trace.npz"
    frozen_trace = root / "frozen_evaluation_trace.npz"
    if online_trace.is_file():
        trace_path = online_trace
        frozen_profile = False
    elif frozen_trace.is_file():
        trace_path = frozen_trace
        frozen_profile = True
    else:
        raise FileNotFoundError(
            f"expected {online_trace.name} or {frozen_trace.name} in {root}"
        )
    world_path = root / "world_carrybox.mp4"
    summary_path = root / "summary.json"
    with np.load(trace_path, allow_pickle=False) as source:
        trace = {key: source[key] for key in source.files}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if frozen_profile:
        profile_count = int(summary["profiles"])
        if not 0 <= args.profile_index < profile_count:
            raise ValueError(
                f"profile index {args.profile_index} is outside 0..{profile_count - 1}"
            )
        recorded_profile = summary.get("world_video_profile_index", 0)
        if args.profile_index != recorded_profile:
            raise RuntimeError(
                f"world video records profile {recorded_profile}, not "
                f"requested profile {args.profile_index}"
            )
        trace = {
            key: value[:, args.profile_index]
            for key, value in trace.items()
        }
    scales = json.loads(args.scale_file.resolve().read_text(encoding="utf-8"))[
        "patch_channel_scales"
    ]
    features = np.asarray(trace["patch_features"], dtype=np.float32)
    slip_state = np.asarray(trace["slip_state"], dtype=np.int64)
    if features.shape[1:] != (2, 27, 6) or slip_state.shape != features.shape[:3]:
        raise RuntimeError(
            f"expected patch [T,2,27,6] and slip [T,2,27], got {features.shape} and {slip_state.shape}"
        )
    frame_count = len(features)
    capture = cv2.VideoCapture(str(world_path))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count:
        raise RuntimeError("world video and patch trace frame counts differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
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
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    initial_z = float(trace["object_pos_w"][0, 2])
    try:
        for frame_index in range(frame_count):
            ok, world = capture.read()
            if not ok:
                raise RuntimeError(f"world decode stopped at frame {frame_index}")
            canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
            canvas[:WORLD_HEIGHT] = fit_world(world)
            cv2.rectangle(canvas, (15, 14), (WIDTH - 15, 62), (255, 255, 255), -1)
            cv2.putText(canvas, args.title, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (15, 15, 15), 2, cv2.LINE_AA)
            jump = bool(trace["jump_applied"][frame_index])
            mass = float(trace["mass_readback_kg"][frame_index])
            lift = float(trace["object_pos_w"][frame_index, 2] - initial_z)
            if frozen_profile and "teacher_control" in trace:
                controller = (
                    "official Refiner pickup"
                    if bool(trace["teacher_control"][frame_index])
                    else "frozen policy control"
                )
                if not bool(trace["valid_frame"][frame_index]):
                    controller = "terminated profile"
            else:
                controller = "fixed online controller"
            mass_phase = (
                "post-jump (mass overlay hidden from actor)"
                if jump
                else "pre-jump / placebo clock"
            )
            status = f"{controller} | {mass_phase}"
            cv2.rectangle(canvas, (15, 545), (WIDTH - 15, 592), (25, 25, 25), -1)
            cv2.putText(
                canvas,
                f"frame {frame_index:03d} | mass {mass:.3f} kg | box lift {lift:+.3f} m | {status}",
                (30, 578),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            draw_hand(canvas, 5, "LEFT HAND: 12 palm + 15 finger patches", features[frame_index, 0], slip_state[frame_index, 0], float(scales[2]), float(scales[3]))
            draw_hand(canvas, 805, "RIGHT HAND: 12 palm + 15 finger patches", features[frame_index, 1], slip_state[frame_index, 1], float(scales[2]), float(scales[3]))
            cv2.putText(
                canvas,
                "fill = mean pressure | arrow = signed XY shear | U = friction utilization | border: gray no contact, green stick, yellow incipient, red gross | value = patch load",
                (25, 974),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin is closed")
            process.stdin.write(np.ascontiguousarray(canvas).tobytes())
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    decoded = cv2.VideoCapture(str(args.output))
    decoded_frames = 0
    while True:
        ok, _ = decoded.read()
        if not ok:
            break
        decoded_frames += 1
    decoded.release()
    if decoded_frames != frame_count:
        raise RuntimeError(f"full decode failed: {decoded_frames}/{frame_count}")
    record = {
        "schema": "plan15_online_mass_patch_video_v2_frozen_handoff_compatible",
        "source_summary": summary,
        "source_trace": trace_path.name,
        "profile_index": args.profile_index if frozen_profile else None,
        "frames": frame_count,
        "fps": args.fps,
        "resolution": [WIDTH, HEIGHT],
        "policy_spatial_unit": "one physical patch; 27 per hand; no taxel display",
        "full_decode": True,
    }
    args.output.with_suffix(".render.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
