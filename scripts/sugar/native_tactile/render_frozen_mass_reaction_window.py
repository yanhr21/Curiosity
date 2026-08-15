#!/usr/bin/env python3
"""Render a camera rollout beside the formal camera-free reaction distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


WIDTH = 1920
HEIGHT = 1080
LEFT_X = 20
LEFT_Y = 96
LEFT_WIDTH = 1080
LEFT_HEIGHT = 675
PLOT_LEFT = 1180
PLOT_RIGHT = 1875
PLOT_TOP = 285
PLOT_BOTTOM = 825
EVENT_COLORS = {
    "continuous_patch_onset_frames": (210, 110, 15),
    "contact_binary_onset_frames": (115, 115, 115),
    "slip_onset_frames": (20, 145, 230),
    "sag_onset_frames": (175, 65, 140),
    "drop_onset_frames": (35, 35, 215),
}
EVENT_LABELS = {
    "continuous_patch_onset_frames": "continuous tactile",
    "contact_binary_onset_frames": "contact binary",
    "slip_onset_frames": "slip",
    "sag_onset_frames": "2 cm sag",
    "drop_onset_frames": "15 cm drop",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-summary", type=Path, required=True)
    parser.add_argument("--camera-video", type=Path, required=True)
    parser.add_argument("--camera-jump-frame", type=int, required=True)
    parser.add_argument("--mass-factor", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (25, 25, 25),
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_frame(frame: np.ndarray) -> np.ndarray:
    scale = min(LEFT_WIDTH / frame.shape[1], LEFT_HEIGHT / frame.shape[0])
    resized = cv2.resize(
        frame,
        (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((LEFT_HEIGHT, LEFT_WIDTH, 3), 245, dtype=np.uint8)
    x = (LEFT_WIDTH - resized.shape[1]) // 2
    y = (LEFT_HEIGHT - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def event_x(frame: float, maximum: int) -> int:
    return int(round(PLOT_LEFT + frame / maximum * (PLOT_RIGHT - PLOT_LEFT)))


def pale(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(round(channel * 0.25 + 255 * 0.75)) for channel in color)


def draw_marker(
    canvas: np.ndarray,
    key: str,
    x: int,
    y: int,
    active: bool,
) -> None:
    color = EVENT_COLORS[key] if active else pale(EVENT_COLORS[key])
    if key == "contact_binary_onset_frames":
        cv2.rectangle(canvas, (x - 3, y - 3), (x + 3, y + 3), color, -1)
    elif key == "sag_onset_frames":
        points = np.asarray([[x, y - 4], [x + 4, y], [x, y + 4], [x - 4, y]])
        cv2.fillConvexPoly(canvas, points, color)
    elif key == "drop_onset_frames":
        points = np.asarray([[x, y + 4], [x - 4, y - 4], [x + 4, y - 4]])
        cv2.fillConvexPoly(canvas, points, color)
    else:
        cv2.circle(canvas, (x, y), 4, color, -1, cv2.LINE_AA)


def draw_timeline(
    canvas: np.ndarray,
    records: list[dict[str, object]],
    summary: dict[str, object],
    relative_frame: int,
    maximum: int,
) -> None:
    put_text(canvas, "FORMAL CAMERA-FREE DISTRIBUTION", (1140, 126), 0.62)
    put_text(
        canvas,
        f"{len(records)} matched {summary['mass_factor']:g}x drop profiles | 50 Hz",
        (1140, 158),
        0.52,
        (70, 70, 70),
        1,
    )
    med = summary["factor_summary"][str(summary["mass_factor"])]["drop_profiles"]
    median_line = (
        f"median frames: continuous {med['continuous_patch_onset_median_frames']:g} | "
        f"binary {med['contact_binary_onset_median_frames']:g} | "
        f"slip {med['slip_onset_median_frames']:g} | "
        f"sag {med['sag_onset_median_frames']:g} | "
        f"drop {med['drop_onset_median_frames']:g}"
    )
    put_text(canvas, median_line, (1140, 190), 0.40, (35, 35, 35), 1)

    legend_x = 1140
    for key in EVENT_COLORS:
        draw_marker(canvas, key, legend_x, 221, True)
        put_text(canvas, EVENT_LABELS[key], (legend_x + 9, 226), 0.34, (45, 45, 45), 1)
        legend_x += 142

    cv2.rectangle(
        canvas,
        (PLOT_LEFT, PLOT_TOP),
        (PLOT_RIGHT, PLOT_BOTTOM),
        (205, 205, 205),
        1,
    )
    for tick in range(0, maximum + 1, 10):
        x = event_x(tick, maximum)
        cv2.line(canvas, (x, PLOT_TOP), (x, PLOT_BOTTOM), (230, 230, 230), 1)
        put_text(canvas, str(tick), (x - 8, PLOT_BOTTOM + 25), 0.35, (70, 70, 70), 1)
    put_text(canvas, "frames after mass jump", (1420, PLOT_BOTTOM + 55), 0.44)

    row_height = (PLOT_BOTTOM - PLOT_TOP - 12) / len(records)
    marker_offsets = (-4, -2, 0, 2, 4)
    for row, record in enumerate(records):
        y = int(round(PLOT_TOP + 7 + (row + 0.5) * row_height))
        cv2.line(canvas, (PLOT_LEFT, y), (PLOT_RIGHT, y), (244, 244, 244), 1)
        for offset, key in zip(marker_offsets, EVENT_COLORS):
            value = record[key]
            if value is None:
                continue
            event_frame = int(value)
            draw_marker(
                canvas,
                key,
                event_x(event_frame, maximum),
                y + offset,
                relative_frame >= event_frame,
            )

    if relative_frame >= 0:
        cursor = event_x(min(relative_frame, maximum), maximum)
        cv2.line(canvas, (cursor, PLOT_TOP - 7), (cursor, PLOT_BOTTOM + 7), (20, 20, 20), 2)
        status = f"current: jump +{relative_frame:02d} frames ({relative_frame / 50.0:.2f} s)"
    else:
        status = f"before jump: {-relative_frame} frames"
    put_text(canvas, status, (1370, 920), 0.62, (20, 20, 20), 2)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.reaction_summary.read_text(encoding="utf-8"))
    records = [
        record
        for record in payload["profiles"]
        if float(record["mass_factor"]) == args.mass_factor and bool(record["drop"])
    ]
    if not records:
        raise RuntimeError(f"no drop records for mass factor {args.mass_factor}")
    records.sort(key=lambda record: (record["training_seed"], record["profile"]))
    payload["mass_factor"] = args.mass_factor
    maximum = int(payload["post_event_window_frames"])

    capture = cv2.VideoCapture(str(args.camera_video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count < 1 or fps <= 0.0:
        raise RuntimeError("camera video is not readable")
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
            f"{fps:g}",
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
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"camera decode stopped at frame {frame_index}")
            canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
            put_text(
                canvas,
                "6x MASS JUMP: VISIBLE G1 DROP AND FORMAL REACTION WINDOW",
                (35, 55),
                0.94,
                (15, 15, 15),
                2,
            )
            canvas[LEFT_Y : LEFT_Y + LEFT_HEIGHT, LEFT_X : LEFT_X + LEFT_WIDTH] = fit_frame(frame)
            put_text(canvas, "CAMERA-ENABLED QUALITATIVE ROLLOUT", (35, 815), 0.58)
            put_text(
                canvas,
                "Complete G1 + CarryBox + bilateral 27-patch maps",
                (35, 848),
                0.48,
                (65, 65, 65),
                1,
            )
            draw_timeline(
                canvas,
                records,
                payload,
                frame_index - args.camera_jump_frame,
                maximum,
            )
            cv2.rectangle(canvas, (25, 970), (WIDTH - 25, 1045), (245, 245, 245), -1)
            put_text(
                canvas,
                "IMPORTANT: left and right are separate rollouts. Camera can perturb "
                "closed-loop PhysX; formal counts use camera-free traces only.",
                (45, 1005),
                0.50,
                (25, 25, 25),
                1,
            )
            put_text(
                canvas,
                "Mass/jump and failure markers are evaluation-only and hidden from the deployed actor.",
                (45, 1032),
                0.43,
                (70, 70, 70),
                1,
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
        "schema": "plan15_frozen_reaction_window_video_v1",
        "camera_video": str(args.camera_video.resolve()),
        "camera_jump_frame": args.camera_jump_frame,
        "reaction_summary": str(args.reaction_summary.resolve()),
        "mass_factor": args.mass_factor,
        "formal_drop_profiles": len(records),
        "frames": frame_count,
        "fps": fps,
        "resolution": [WIDTH, HEIGHT],
        "full_decode": True,
        "camera_and_formal_trace_are_separate_rollouts": True,
    }
    args.output.with_suffix(".render.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
