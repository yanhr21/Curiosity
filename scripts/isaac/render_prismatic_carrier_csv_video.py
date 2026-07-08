#!/usr/bin/env python3
"""Render a diagnostic MP4 from a prismatic carrier CSV rollout.

This is a metrics visualization, not an Isaac viewport recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def _csv_value(value: str) -> float | str:
    if value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return value


def load_rows(path: Path, stride: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx % max(1, stride) != 0:
                continue
            rows.append({key: _csv_value(value) for key, value in row.items()})
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}")
    return rows


def make_canvas(width: int, height: int):
    import numpy as np

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    return canvas


def draw_rect(img, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], *, fill: bool = True) -> None:
    h, w = img.shape[:2]
    x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
    y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
    if fill:
        img[y0 : y1 + 1, x0 : x1 + 1] = color
        return
    img[y0 : y0 + 2, x0 : x1 + 1] = color
    img[y1 - 1 : y1 + 1, x0 : x1 + 1] = color
    img[y0 : y1 + 1, x0 : x0 + 2] = color
    img[y0 : y1 + 1, x1 - 1 : x1 + 1] = color


def draw_line(img, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    import numpy as np

    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).astype(int)
    ys = np.linspace(y0, y1, steps + 1).astype(int)
    h, w = img.shape[:2]
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[mask], xs[mask]] = color


def _num(row: dict[str, float | str], key: str, default: float) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def render_frame(rows: list[dict[str, float | str]], row_idx: int, summary: dict, width: int, height: int):
    img = make_canvas(width, height)
    row = rows[row_idx]
    top = (40, 40, width - 40, int(height * 0.58))
    side = (40, int(height * 0.64), width - 40, height - 40)
    draw_rect(img, top[0], top[1], top[2], top[3], (255, 255, 255), fill=True)
    draw_rect(img, top[0], top[1], top[2], top[3], (30, 30, 30), fill=False)
    draw_rect(img, side[0], side[1], side[2], side[3], (255, 255, 255), fill=True)
    draw_rect(img, side[0], side[1], side[2], side[3], (30, 30, 30), fill=False)

    torso_z_default = float(summary.get("torso_z_m") or 0.55)
    payload_z_default = torso_z_default + float(summary.get("payload_local_z_m") or 0.04)
    xs = [_num(r, "torso_x", 0.0) for r in rows] + [_num(r, "payload_x", 0.0) for r in rows]
    ys = [_num(r, "torso_y", 0.0) for r in rows] + [_num(r, "payload_y", 0.0) for r in rows]
    zs = [_num(r, "torso_z", torso_z_default) for r in rows] + [
        _num(r, "payload_z", payload_z_default) for r in rows
    ]
    x_min, x_max = min(xs) - 0.15, max(xs) + 0.15
    y_min, y_max = min(ys) - 0.25, max(ys) + 0.25
    z_min, z_max = min(0.0, min(zs) - 0.05), max(zs) + 0.15

    def map_top(x: float, y: float) -> tuple[int, int]:
        px = top[0] + int((x - x_min) / max(x_max - x_min, 1e-6) * (top[2] - top[0]))
        py = top[3] - int((y - y_min) / max(y_max - y_min, 1e-6) * (top[3] - top[1]))
        return px, py

    def map_side(x: float, z: float) -> tuple[int, int]:
        px = side[0] + int((x - x_min) / max(x_max - x_min, 1e-6) * (side[2] - side[0]))
        py = side[3] - int((z - z_min) / max(z_max - z_min, 1e-6) * (side[3] - side[1]))
        return px, py

    baseline_payload_x = float(summary.get("post_settle_baseline_payload_x_m") or _num(rows[0], "payload_x", 0.0))
    target_x = baseline_payload_x + float(summary.get("target_x_m") or 0.0)
    tx0, ty0 = map_top(target_x, y_min)
    tx1, ty1 = map_top(target_x, y_max)
    draw_line(img, tx0, ty0, tx1, ty1, (180, 60, 60))

    for prev, curr in zip(rows[:row_idx], rows[1 : row_idx + 1]):
        px0, py0 = map_top(_num(prev, "payload_x", 0.0), _num(prev, "payload_y", 0.0))
        px1, py1 = map_top(_num(curr, "payload_x", 0.0), _num(curr, "payload_y", 0.0))
        draw_line(img, px0, py0, px1, py1, (240, 170, 80))
        tx0, tz0 = map_side(_num(prev, "payload_x", 0.0), _num(prev, "payload_z", payload_z_default))
        tx1, tz1 = map_side(_num(curr, "payload_x", 0.0), _num(curr, "payload_z", payload_z_default))
        draw_line(img, tx0, tz0, tx1, tz1, (240, 170, 80))

    torso_size = summary.get("torso_size_m") or [0.56, 0.34, 0.16]
    payload_size = summary.get("payload_size_m") or [0.34, 0.24, 0.24]
    stance_x = float(summary.get("stance_half_length_m") or 0.65)
    stance_y = float(summary.get("stance_half_width_m") or 0.24)

    torso_x = _num(row, "torso_x", 0.0)
    torso_y = _num(row, "torso_y", 0.0)
    payload_x = _num(row, "payload_x", 0.0)
    payload_y = _num(row, "payload_y", 0.0)
    torso_z = _num(row, "torso_z", torso_z_default)
    payload_z = _num(row, "payload_z", payload_z_default)

    sx0, sy0 = map_top(torso_x - stance_x, torso_y - stance_y)
    sx1, sy1 = map_top(torso_x + stance_x, torso_y + stance_y)
    draw_rect(img, sx0, sy0, sx1, sy1, (90, 180, 110), fill=False)

    def draw_body_top(cx: float, cy: float, size_x: float, size_y: float, color: tuple[int, int, int]) -> None:
        x0, y0 = map_top(cx - size_x * 0.5, cy - size_y * 0.5)
        x1, y1 = map_top(cx + size_x * 0.5, cy + size_y * 0.5)
        draw_rect(img, x0, y0, x1, y1, color, fill=True)
        draw_rect(img, x0, y0, x1, y1, (30, 30, 30), fill=False)

    draw_body_top(torso_x, torso_y, float(torso_size[0]), float(torso_size[1]), (80, 130, 220))
    draw_body_top(payload_x, payload_y, float(payload_size[0]), float(payload_size[1]), (230, 150, 60))

    gx0, gy0 = map_side(x_min, 0.0)
    gx1, gy1 = map_side(x_max, 0.0)
    draw_line(img, gx0, gy0, gx1, gy1, (120, 120, 120))
    tpx, tpz = map_side(torso_x, torso_z)
    bpx, bpz = map_side(payload_x, payload_z)
    draw_rect(img, tpx - 14, tpz - 14, tpx + 14, tpz + 14, (80, 130, 220), fill=True)
    draw_rect(img, bpx - 14, bpz - 14, bpx + 14, bpz + 14, (230, 150, 60), fill=True)
    return img


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv, args.stride)
    summary = json.loads(args.summary.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("IMAGEIO_NO_INTERNET", "1")
    try:
        import cv2

        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            (int(args.width), int(args.height)),
        )
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter did not open")
        for idx in range(len(rows)):
            frame = render_frame(rows, idx, summary, int(args.width), int(args.height))
            writer.write(frame[:, :, ::-1])
        writer.release()
    except Exception as cv2_error:
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(str(args.output), fps=int(args.fps), macro_block_size=2) as writer:
                for idx in range(len(rows)):
                    writer.append_data(render_frame(rows, idx, summary, int(args.width), int(args.height)))
        except Exception as imageio_error:
            raise RuntimeError(f"Could not write MP4 via cv2 ({cv2_error}) or imageio ({imageio_error})") from imageio_error
    print(f"[INFO] Wrote MP4: {args.output}")


if __name__ == "__main__":
    main()
