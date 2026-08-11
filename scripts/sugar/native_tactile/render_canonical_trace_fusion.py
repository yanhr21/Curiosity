#!/usr/bin/env python3
"""Render canonical CarryBox contact beside the current late-fusion telemetry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


WIDTH = 1920
HEIGHT = 1080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-start", type=int, default=230)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def put(
    canvas: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.62,
    thickness: int = 1,
    color: tuple[int, int, int] = (35, 35, 35),
) -> None:
    cv2.putText(
        canvas,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_series(
    canvas: np.ndarray,
    values: np.ndarray,
    current: int,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x0, y0, width, height = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (195, 195, 195), 1)
    maximum = max(float(np.max(values)), 1.0e-12)
    x = np.linspace(x0 + 2, x0 + width - 2, len(values)).astype(np.int32)
    y = (
        y0
        + height
        - 3
        - np.clip(values / maximum, 0.0, 1.0) * (height - 6)
    ).astype(np.int32)
    points = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
    marker_x = int(round(x0 + current * (width - 1) / max(len(values) - 1, 1)))
    cv2.line(canvas, (marker_x, y0), (marker_x, y0 + height), (20, 20, 20), 1)
    put(canvas, f"{label}  max {maximum:.4g}", (x0 + 8, y0 + 23), 0.48, 1, color)


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    metrics_path = args.metrics.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (video_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)

    with np.load(metrics_path, allow_pickle=False) as archive:
        metrics = {name: archive[name] for name in archive.files}
    frame_count = len(metrics["frame"])
    source_start = int(args.source_start)

    capture = cv2.VideoCapture(str(video_path))
    displayed_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_start + displayed_frames > frame_count:
        raise RuntimeError("presentation video exceeds metric trace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable")

    interval = slice(source_start, source_start + displayed_frames)
    rms_series = metrics["normalized_input_rms"][interval]
    action_series = metrics["zero_base_action_delta_abs_max"][interval]
    encoded_series = metrics["feature_l2"][interval]
    written = 0
    try:
        for video_index in range(displayed_frames):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video decode stopped at frame {video_index}")
            source_frame = source_start + video_index
            canvas = np.full((HEIGHT, WIDTH, 3), 250, dtype=np.uint8)
            main = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            canvas[100:820, 20:1300] = main

            put(canvas, "Plan 13: canonical CarryBox tactile entering the serious late-fusion actor", (24, 42), 0.86, 2)
            put(canvas, "Recorded physical taxels; no learning and no claim of policy benefit", (24, 76), 0.62, 1, (75, 75, 75))

            draw_series(
                canvas,
                rms_series,
                video_index,
                (20, 850, 625, 185),
                (190, 80, 25),
                "normalized 4-frame input RMS",
            )
            draw_series(
                canvas,
                action_series,
                video_index,
                (675, 850, 625, 185),
                (35, 90, 195),
                "standardized zero-base action delta max",
            )

            current_contact = bool(metrics["current_contact"][source_frame])
            history_contact = bool(metrics["history_contact"][source_frame])
            active = metrics["active_taxels_by_hand"][source_frame]
            status_color = (20, 140, 20) if current_contact else ((0, 145, 220) if history_contact else (95, 95, 95))
            status = "CURRENT CONTACT" if current_contact else ("CONTACT IN 4-FRAME HISTORY" if history_contact else "ZERO TACTILE HISTORY")

            x = 1340
            put(canvas, f"source frame {source_frame} / 659", (x, 135), 0.72, 2)
            put(canvas, status, (x, 180), 0.69, 2, status_color)
            cv2.line(canvas, (x, 200), (1885, 200), (180, 180, 180), 1)

            rows = [
                ("active taxels L / R", f"{int(active[0])} / {int(active[1])}"),
                ("normalized input RMS", f"{metrics['normalized_input_rms'][source_frame]:.5f}"),
                ("encoder feature L2", f"{encoded_series[video_index]:.5f}"),
                ("actor.0 tactile delta max", f"{metrics['first_layer_tactile_preactivation_abs_max'][source_frame]:.6f}"),
                ("zero-base action delta max", f"{action_series[video_index]:.6f}"),
                ("warm-start tactile gain", "0.01"),
            ]
            y = 250
            for label, value in rows:
                put(canvas, label, (x, y), 0.55, 1, (80, 80, 80))
                put(canvas, value, (x, y + 35), 0.78, 2, (25, 25, 25))
                y += 105

            cv2.line(canvas, (x, 880), (1885, 880), (180, 180, 180), 1)
            put(canvas, "Fusion: left/right shared spatial CNN", (x, 920), 0.53, 1)
            put(canvas, "2 x 128-D -> concatenate before actor.0", (x, 950), 0.53, 1)
            put(canvas, "Action delta uses a standardized zero base.", (x, 995), 0.50, 1, (85, 85, 85))
            put(canvas, "It is not the recorded rollout action difference.", (x, 1023), 0.50, 1, (85, 85, 85))

            process.stdin.write(np.ascontiguousarray(canvas).tobytes())
            written += 1
    finally:
        capture.release()
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0 or written != displayed_frames:
        raise RuntimeError(
            f"render failed: ffmpeg={return_code}, frames={written}/{displayed_frames}"
        )

    verify = cv2.VideoCapture(str(output_path))
    decoded = 0
    while True:
        ok, _ = verify.read()
        if not ok:
            break
        decoded += 1
    verify.release()
    if decoded != displayed_frames:
        raise RuntimeError(f"output decodes {decoded}/{displayed_frames} frames")
    record = {
        "semantics": "canonical physical tactile plus serious late-fusion telemetry",
        "source_video": str(video_path),
        "metric_trace": str(metrics_path),
        "source_interval": [source_start, source_start + displayed_frames],
        "resolution": [WIDTH, HEIGHT],
        "fps": args.fps,
        "decoded_frames": decoded,
        "claim_boundary": "fusion scale visualization; not closed-loop policy benefit",
    }
    output_path.with_suffix(".render.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
