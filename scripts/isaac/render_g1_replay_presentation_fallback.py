#!/usr/bin/env python3
"""Render a presentation fallback from a G1 replay CSV.

This is not an Isaac camera render. It draws a legible G1-like humanoid,
free box, target/path, and strict metrics from an already-recorded replay CSV.
Use it only when Kit viewport/Replicator rendering is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to generate visualization on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-csv", type=Path, required=True)
    parser.add_argument("--record-summary", type=Path, required=True)
    parser.add_argument("--checker-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="G1 low-carry replay fallback")
    parser.add_argument("--subtitle", default="schematic from recorded pass, not Isaac camera render")
    parser.add_argument("--gif-name", default="g1_lowcarry_replay_fallback.gif")
    parser.add_argument("--poster-name", default="g1_lowcarry_replay_fallback_poster.png")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gif-duration-ms", type=int, default=80)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _project(
    x: float,
    z: float,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    panel: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, right, bottom = panel
    px = left + int((x - x_min) / max(1e-6, x_max - x_min) * (right - left))
    py = bottom - int((z - z_min) / max(1e-6, z_max - z_min) * (bottom - top))
    return px, py


def _draw_limb(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int], width: int) -> None:
    draw.line([a, b], fill=color, width=width)
    r = max(3, width // 2)
    for p in (a, b):
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=color)


def _draw_robot(
    draw: ImageDraw.ImageDraw,
    row: dict[str, str],
    frame_idx: int,
    x_min: float,
    x_max: float,
    panel: tuple[int, int, int, int],
) -> None:
    root_x = _float(row, "robot_x")
    root_z = _float(row, "robot_z", 0.78)
    phase = 2.0 * math.pi * (frame_idx / 18.0)
    pelvis = _project(root_x, root_z, x_min, x_max, 0.0, 1.55, panel)
    chest = _project(root_x - 0.03, root_z + 0.33, x_min, x_max, 0.0, 1.55, panel)
    head = _project(root_x - 0.04, root_z + 0.54, x_min, x_max, 0.0, 1.55, panel)
    shoulder_l = _project(root_x - 0.06, root_z + 0.27, x_min, x_max, 0.0, 1.55, panel)
    shoulder_r = _project(root_x - 0.02, root_z + 0.27, x_min, x_max, 0.0, 1.55, panel)
    hand_l = _project(root_x - 0.25, root_z + 0.11, x_min, x_max, 0.0, 1.55, panel)
    hand_r = _project(root_x - 0.25, root_z + 0.05, x_min, x_max, 0.0, 1.55, panel)
    foot_l = _project(root_x + 0.10 * math.sin(phase), 0.06, x_min, x_max, 0.0, 1.55, panel)
    knee_l = _project(root_x + 0.05 * math.sin(phase + 0.5), root_z - 0.32, x_min, x_max, 0.0, 1.55, panel)
    foot_r = _project(root_x + 0.10 * math.sin(phase + math.pi), 0.06, x_min, x_max, 0.0, 1.55, panel)
    knee_r = _project(root_x + 0.05 * math.sin(phase + math.pi + 0.5), root_z - 0.32, x_min, x_max, 0.0, 1.55, panel)
    dark = (42, 49, 58)
    blue = (42, 99, 164)
    orange = (205, 116, 45)
    _draw_limb(draw, pelvis, chest, dark, 13)
    draw.ellipse((head[0] - 18, head[1] - 18, head[0] + 18, head[1] + 18), fill=(58, 68, 79))
    _draw_limb(draw, pelvis, knee_l, blue, 11)
    _draw_limb(draw, knee_l, foot_l, blue, 10)
    _draw_limb(draw, pelvis, knee_r, (34, 77, 130), 11)
    _draw_limb(draw, knee_r, foot_r, (34, 77, 130), 10)
    _draw_limb(draw, shoulder_l, hand_l, orange, 9)
    _draw_limb(draw, shoulder_r, hand_r, orange, 9)
    draw.rounded_rectangle((chest[0] - 25, chest[1] - 38, chest[0] + 25, chest[1] + 38), radius=8, fill=(63, 74, 88), outline=(18, 24, 32), width=2)


def _draw_frame(
    rows: list[dict[str, str]],
    row: dict[str, str],
    idx: int,
    summary: dict,
    checker: dict | None,
    title: str,
    subtitle: str,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (244, 245, 246))
    draw = ImageDraw.Draw(img)
    title_font = _font(34)
    label_font = _font(20)
    small_font = _font(16)
    panel = (70, 110, width - 70, height - 150)
    draw.rectangle(panel, fill=(235, 238, 240), outline=(42, 49, 58), width=2)
    for i in range(8):
        z = i / 7 * 1.4
        _, y = _project(x_min, z, x_min, x_max, 0.0, 1.55, panel)
        draw.line([(panel[0], y), (panel[2], y)], fill=(216, 221, 225), width=1)
    ground_y = _project(0, 0.02, x_min, x_max, 0.0, 1.55, panel)[1]
    draw.line([(panel[0], ground_y), (panel[2], ground_y)], fill=(93, 103, 113), width=3)

    path_points = [
        _project(_float(r, "box_x"), _float(r, "box_z", 0.55), x_min, x_max, 0.0, 1.55, panel)
        for r in rows[: idx + 1]
    ]
    if len(path_points) > 1:
        draw.line(path_points, fill=(199, 132, 42), width=4)

    box_x = _float(row, "box_x", _float(row, "robot_x") - 0.25)
    box_z = _float(row, "box_z", 0.75)
    bx, by = _project(box_x, box_z, x_min, x_max, 0.0, 1.55, panel)
    box_w = 110
    box_h = 78
    draw.rounded_rectangle((bx - box_w // 2, by - box_h // 2, bx + box_w // 2, by + box_h // 2), radius=6, fill=(166, 116, 63), outline=(92, 58, 26), width=3)
    draw.line([(bx - box_w // 2, by), (bx + box_w // 2, by)], fill=(130, 84, 40), width=2)
    _draw_robot(draw, row, idx, x_min, x_max, panel)

    robot_travel = summary.get("final_robot_target_directed_travel_m")
    box_travel = summary.get("final_box_target_directed_travel_m")
    rel = summary.get("final_box_robot_relative_offset_error_m")
    checker_status = summary.get("status")
    if checker:
        checker_status = checker.get("status", checker_status)
        cases = checker.get("cases")
        if isinstance(cases, list) and cases:
            first_case = cases[0]
            if isinstance(first_case, dict):
                checker_status = first_case.get("check_status", checker_status)
                if first_case.get("passed") is False:
                    checker_status = "fail"
    draw.text((64, 34), str(title), font=title_font, fill=(28, 35, 43))
    draw.text((66, 78), str(subtitle), font=label_font, fill=(94, 54, 35))
    metrics = [
        f"strict checker: {checker_status}",
        f"fall/drop: {summary.get('fall_events')}/{summary.get('box_drop_events')}",
        f"robot/box travel: {float(robot_travel or 0):.2f} / {float(box_travel or 0):.2f} m",
        f"final rel error: {float(rel or 0):.3f} m",
        f"frame: {idx + 1}/{len(rows)}",
    ]
    x = 80
    y = height - 118
    for text in metrics:
        draw.rounded_rectangle((x - 10, y - 8, x + 238, y + 28), radius=8, fill=(255, 255, 255), outline=(205, 211, 216))
        draw.text((x, y), text, font=small_font, fill=(39, 47, 56))
        x += 250
    return img


def main() -> int:
    _refuse_login_node()
    args = parse_args()
    if not args.replay_csv.is_file():
        raise FileNotFoundError(args.replay_csv)
    if not args.record_summary.is_file():
        raise FileNotFoundError(args.record_summary)
    if args.checker_summary is not None and not args.checker_summary.is_file():
        raise FileNotFoundError(args.checker_summary)
    rows_all = _load_rows(args.replay_csv)
    if not rows_all:
        raise RuntimeError("Replay CSV has no rows")
    stride = max(1, len(rows_all) // max(1, int(args.max_frames)))
    rows = rows_all[::stride][: int(args.max_frames)]
    summary = json.loads(args.record_summary.read_text())
    checker = json.loads(args.checker_summary.read_text()) if args.checker_summary is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    xs = [_float(r, "robot_x") for r in rows] + [_float(r, "box_x") for r in rows]
    x_min = min(xs) - 0.35
    x_max = max(xs) + 0.35
    frames: list[Image.Image] = []
    for idx, row in enumerate(rows):
        img = _draw_frame(
            rows,
            row,
            idx,
            summary,
            checker,
            str(args.title),
            str(args.subtitle),
            int(args.width),
            int(args.height),
            x_min,
            x_max,
        )
        frame_path = frame_dir / f"g1_replay_fallback_{idx:04d}.png"
        img.save(frame_path)
        frames.append(img)
    gif_path = args.output_dir / str(args.gif_name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(20, int(args.gif_duration_ms)),
        loop=0,
        optimize=False,
    )
    poster_path = args.output_dir / str(args.poster_name)
    frames[min(len(frames) - 1, len(frames) // 2)].save(poster_path)
    out_summary = {
        "scene_type": "g1_replay_presentation_fallback",
        "success_claim": "schematic_replay_visual_only_not_isaac_camera_render_not_new_control_evidence",
        "status": "pass",
        "replay_csv": str(args.replay_csv),
        "record_summary": str(args.record_summary),
        "checker_summary": str(args.checker_summary) if args.checker_summary is not None else None,
        "frame_count": len(frames),
        "frame_dir": str(frame_dir),
        "gif": str(gif_path),
        "poster": str(poster_path),
    }
    (args.output_dir / "g1_replay_presentation_fallback_summary.json").write_text(
        json.dumps(out_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
