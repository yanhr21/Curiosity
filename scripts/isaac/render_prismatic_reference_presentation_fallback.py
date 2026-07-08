#!/usr/bin/env python3
"""Render a prismatic no-root carrier presentation fallback.

This is a schematic visualization from an already completed prismatic carrier
CSV. It is not an Isaac camera render and not new control evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to generate visualization on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gif-duration-ms", type=int, default=80)
    return parser.parse_args()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


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


def _draw_leg(
    draw: ImageDraw.ImageDraw,
    hip: tuple[int, int],
    foot: tuple[int, int],
    lift: float,
    color: tuple[int, int, int],
) -> None:
    mid = ((hip[0] + foot[0]) // 2, (hip[1] + foot[1]) // 2 - int(28 * max(0.0, lift)))
    draw.line([hip, mid, foot], fill=color, width=8, joint="curve")
    draw.ellipse((foot[0] - 28, foot[1] - 7, foot[0] + 28, foot[1] + 7), fill=color)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    text_fill: tuple[int, int, int],
) -> int:
    x, y = xy
    tw, th = _text_size(draw, text, font)
    draw.rounded_rectangle((x, y, x + tw + 22, y + th + 14), radius=7, fill=fill, outline=outline, width=1)
    draw.text((x + 11, y + 7), text, font=font, fill=text_fill)
    return x + tw + 34


def _phase(row: dict[str, str], summary: dict) -> str:
    step = int(_f(row, "step", 0.0))
    probe_start = int(summary.get("active_probe_start_step") or 10**9)
    probe_end = int(summary.get("active_probe_end_step") or 10**9)
    carry_start = int(summary.get("carry_start_step") or 10**9)
    if step < probe_start:
        return "settle / stance"
    if probe_start <= step < probe_end:
        return "active probe"
    if step < carry_start:
        return "post-probe adapt"
    return "guarded carry"


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 4,
) -> None:
    draw.line([start, end], fill=color, width=width)
    direction = 1 if end[0] >= start[0] else -1
    draw.polygon(
        [
            (end[0], end[1]),
            (end[0] - direction * 18, end[1] - 9),
            (end[0] - direction * 18, end[1] + 9),
        ],
        fill=color,
    )


def _draw_frame(
    rows: list[dict[str, str]],
    row: dict[str, str],
    idx: int,
    summary: dict,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (244, 245, 246))
    draw = ImageDraw.Draw(image)
    title_font = _font(32)
    label_font = _font(19)
    small_font = _font(15)
    tiny_font = _font(13)
    panel = (72, 112, width - 72, height - 150)
    draw.rectangle(panel, fill=(235, 238, 240), outline=(39, 47, 56), width=2)
    draw.rectangle((panel[0], panel[1], panel[2], panel[1] + 44), fill=(226, 230, 233))
    for i in range(7):
        z = i / 6 * 1.1
        _, y = _project(x_min, z, x_min, x_max, 0.0, 1.15, panel)
        draw.line([(panel[0], y), (panel[2], y)], fill=(216, 221, 225), width=1)
    ground_y = _project(0.0, 0.02, x_min, x_max, 0.0, 1.15, panel)[1]
    draw.line([(panel[0], ground_y), (panel[2], ground_y)], fill=(93, 103, 113), width=3)

    target_x = float(summary.get("target_x_m") or 0.0)
    tx, _ = _project(target_x, 0.02, x_min, x_max, 0.0, 1.15, panel)
    draw.line([(tx, panel[1]), (tx, panel[3])], fill=(178, 61, 54), width=3)
    draw.text((tx + 8, panel[1] + 8), "target", font=small_font, fill=(145, 46, 40))
    _arrow(draw, (tx - 115, panel[1] + 28), (tx - 18, panel[1] + 28), (178, 61, 54), width=3)

    path = [
        _project(_f(r, "payload_x"), _f(r, "payload_z", 0.72), x_min, x_max, 0.0, 1.15, panel)
        for r in rows[: idx + 1]
    ]
    if len(path) > 1:
        draw.line(path, fill=(199, 132, 42), width=4)

    torso_x = _f(row, "torso_x")
    torso_z = _f(row, "torso_z", 0.62)
    payload_x = _f(row, "payload_x", torso_x + 0.5)
    payload_z = _f(row, "payload_z", torso_z + 0.18)
    lift = _f(row, "actual_leg_lift_m", 0.0) / 0.08

    torso = _project(torso_x, torso_z, x_min, x_max, 0.0, 1.15, panel)
    deck = _project(torso_x + 0.34, torso_z + 0.15, x_min, x_max, 0.0, 1.15, panel)
    payload = _project(payload_x, payload_z, x_min, x_max, 0.0, 1.15, panel)
    phase = _phase(row, summary)

    phase_color = {
        "settle / stance": (80, 91, 104),
        "active probe": (49, 118, 143),
        "post-probe adapt": (126, 91, 38),
        "guarded carry": (54, 125, 78),
    }.get(phase, (80, 91, 104))
    x_cursor = panel[0] + 18
    x_cursor = _pill(draw, (x_cursor, panel[1] + 8), phase, small_font, (255, 255, 255), phase_color, phase_color)
    _pill(
        draw,
        (x_cursor, panel[1] + 8),
        f"near-ground feet: {int(_f(row, 'near_ground_foot_count', 0.0))}/4",
        small_font,
        (255, 255, 255),
        (96, 105, 115),
        (43, 51, 61),
    )

    for dx, side_color in ((-0.22, (42, 99, 164)), (-0.05, (34, 77, 130)), (0.12, (42, 99, 164)), (0.29, (34, 77, 130))):
        hip = _project(torso_x + dx, torso_z - 0.06, x_min, x_max, 0.0, 1.15, panel)
        foot = _project(torso_x + dx + _f(row, "abs_actual_x_slide_m", 0.0) * 0.35, 0.045, x_min, x_max, 0.0, 1.15, panel)
        _draw_leg(draw, hip, foot, lift, side_color)

    draw.rounded_rectangle((torso[0] - 115, torso[1] - 42, torso[0] + 115, torso[1] + 42), radius=7, fill=(67, 78, 92), outline=(22, 28, 36), width=3)
    draw.text((torso[0] - 58, torso[1] - 12), "carrier body", font=tiny_font, fill=(238, 242, 245))
    draw.rounded_rectangle((deck[0] - 220, deck[1] - 18, deck[0] + 220, deck[1] + 18), radius=5, fill=(92, 112, 121), outline=(38, 52, 60), width=2)
    for wall_x in (deck[0] - 220, deck[0] + 220):
        draw.rectangle((wall_x - 10, deck[1] - 96, wall_x + 10, deck[1] + 18), fill=(85, 103, 112))
    draw.text((deck[0] - 79, deck[1] + 24), "physical cradle", font=tiny_font, fill=(57, 67, 76))

    draw.rounded_rectangle((payload[0] - 92, payload[1] - 58, payload[0] + 92, payload[1] + 58), radius=6, fill=(166, 116, 63), outline=(92, 58, 26), width=3)
    draw.line([(payload[0] - 92, payload[1]), (payload[0] + 92, payload[1])], fill=(130, 84, 40), width=2)
    draw.text((payload[0] - 43, payload[1] - 10), "free box", font=small_font, fill=(45, 29, 16))
    _arrow(draw, (payload[0] + 108, payload[1]), (payload[0] + 180, payload[1]), (199, 132, 42), width=4)

    draw.text((64, 31), "Current Isaac Progress: Physical Free-Box Carry Scaffold", font=title_font, fill=(28, 35, 43))
    draw.text((66, 75), "prismatic no-root carrier, schematic from Isaac state CSV, not camera render or final humanoid success", font=label_font, fill=(94, 54, 35))
    metrics = [
        f"payload: {float(summary.get('payload_mass_kg') or 0):.1f} kg free box",
        f"fall/drop: {summary.get('fall_events')}/{summary.get('box_drop_events')}",
        f"payload travel: {float(summary.get('final_post_settle_payload_travel_x_m') or 0):.3f} m",
        f"target dist: {float(summary.get('final_post_settle_payload_target_distance_x_m') or 0):.3f} m",
        f"root/box writes: {summary.get('body_root_pose_write_count')}/{summary.get('body_root_velocity_command_count')}/{summary.get('box_pose_write_count')}",
    ]
    x = 78
    y = height - 116
    for text in metrics:
        draw.rounded_rectangle((x - 9, y - 8, x + 230, y + 28), radius=7, fill=(255, 255, 255), outline=(205, 211, 216))
        draw.text((x, y), text, font=small_font, fill=(39, 47, 56))
        x += 242
    footer = (
        f"frame {idx + 1}/{len(rows)}  step {row.get('step')}  "
        f"tilt {_f(row, 'tilt'):.4f} rad  "
        f"post-settle target error {_f(row, 'post_settle_target_distance_x', 0.0):.4f} m"
    )
    draw.text((80, height - 66), footer, font=small_font, fill=(65, 73, 82))
    return image


def main() -> int:
    _refuse_login_node()
    args = parse_args()
    if not args.state_csv.is_file():
        raise FileNotFoundError(args.state_csv)
    if not args.summary.is_file():
        raise FileNotFoundError(args.summary)
    all_rows = _rows(args.state_csv)
    if not all_rows:
        raise RuntimeError("state CSV has no rows")
    stride = max(1, len(all_rows) // max(1, int(args.max_frames)))
    rows = all_rows[::stride][: int(args.max_frames)]
    summary = json.loads(args.summary.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    xs = [_f(r, "torso_x") for r in rows] + [_f(r, "payload_x") for r in rows] + [float(summary.get("target_x_m") or 0.0)]
    x_min = min(xs) - 0.45
    x_max = max(xs) + 0.45
    frames: list[Image.Image] = []
    for idx, row in enumerate(rows):
        frame = _draw_frame(rows, row, idx, summary, int(args.width), int(args.height), x_min, x_max)
        frame.save(frame_dir / f"prismatic_reference_{idx:04d}.png")
        frames.append(frame)
    gif_path = args.output_dir / "prismatic_reference_fallback.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=max(20, int(args.gif_duration_ms)), loop=0, optimize=False)
    poster_path = args.output_dir / "prismatic_reference_fallback_poster.png"
    frames[min(len(frames) - 1, len(frames) // 2)].save(poster_path)
    out = {
        "scene_type": "prismatic_reference_presentation_fallback",
        "success_claim": "schematic_visual_only_not_isaac_camera_render_not_final_humanoid_success",
        "status": "pass",
        "state_csv": str(args.state_csv),
        "summary": str(args.summary),
        "frame_count": len(frames),
        "frame_dir": str(frame_dir),
        "gif": str(gif_path),
        "poster": str(poster_path),
    }
    (args.output_dir / "prismatic_reference_presentation_fallback_summary.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
