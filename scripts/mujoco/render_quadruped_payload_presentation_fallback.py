#!/usr/bin/env python3
"""Render a MuJoCo quadruped payload diagnostic presentation GIF.

This intentionally renders a clear schematic from a deterministic MuJoCo
rollout. It is not camera evidence, not unknown free-box grasping, and not a
final carrying-success claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_quadruped_payload_carry import MJCF_TEMPLATE, _quat_to_roll_pitch


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to render visualization on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--payload-mass", type=float, default=4.0)
    parser.add_argument("--target-speed", type=float, default=0.24)
    parser.add_argument("--target-height", type=float, default=0.56)
    parser.add_argument("--fall-height", type=float, default=0.30)
    parser.add_argument("--max-tilt-rad", type=float, default=0.55)
    parser.add_argument("--assist-mode", choices=("body_force", "none"), default="body_force")
    parser.add_argument("--max-assist-force-x", type=float, default=115.0)
    parser.add_argument("--max-assist-force-z", type=float, default=320.0)
    parser.add_argument("--max-assist-torque", type=float, default=220.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gif-duration-ms", type=int, default=70)
    return parser.parse_args()


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


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _badge(
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
    draw.rounded_rectangle((x, y, x + tw + 22, y + th + 14), radius=7, fill=fill, outline=outline)
    draw.text((x + 11, y + 7), text, font=font, fill=text_fill)
    return x + tw + 34


def _draw_rotated_box(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    half_w: int,
    half_h: int,
    angle: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> list[tuple[int, int]]:
    cx, cy = center
    pts = []
    for x, y in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        rx = x * math.cos(angle) - y * math.sin(angle)
        ry = x * math.sin(angle) + y * math.cos(angle)
        pts.append((int(cx + rx), int(cy + ry)))
    draw.polygon(pts, fill=fill, outline=outline)
    draw.line([pts[0], pts[2]], fill=outline, width=2)
    return pts


def _simulate(args: argparse.Namespace) -> tuple[list[dict[str, float]], dict[str, float | int | str | bool | None]]:
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(MJCF_TEMPLATE.format(payload_mass=args.payload_mass))
    data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    joint_to_act = {model.actuator(i).name.replace("_pos", ""): i for i in range(model.nu)}
    gait_order = [
        ("fl_hip", "fl_knee", 0.0),
        ("rr_hip", "rr_knee", 0.0),
        ("fr_hip", "fr_knee", math.pi),
        ("rl_hip", "rl_knee", math.pi),
    ]
    dt = float(model.opt.timestep)
    stride = max(1, args.steps // max(1, args.max_frames))
    initial_x: float | None = None
    states: list[dict[str, float]] = []
    summary: dict[str, float | int | str | bool | None] = {
        "scene_type": "mujoco_dynamic_quadruped_assisted_payload_visual",
        "success_claim": "diagnostic_visual_only_body_force_welded_payload_not_unknown_box_grasp",
        "payload_mass_kg": float(args.payload_mass),
        "payload_mode": "welded_child_body",
        "assist_mode": args.assist_mode,
        "external_stabilizer_enabled": args.assist_mode == "body_force",
        "steps_requested": int(args.steps),
        "completed_steps": 0,
        "target_speed_mps": float(args.target_speed),
        "max_travel_x_m": 0.0,
        "min_torso_z_m": None,
        "max_tilt_rad": 0.0,
        "fall_events": 0,
        "root_pose_write_count": 0,
        "root_velocity_write_count": 0,
        "external_force_write_count": 0,
        "external_torque_write_count": 0,
    }
    for step in range(args.steps):
        t = step * dt
        for hip_name, knee_name, phase in gait_order:
            swing = math.sin(2.0 * math.pi * 1.6 * t + phase)
            data.ctrl[joint_to_act[hip_name]] = 0.28 * swing
            data.ctrl[joint_to_act[knee_name]] = -0.78 + 0.18 * max(0.0, swing)

        torso_z = float(data.qpos[2])
        vx = float(data.qvel[0])
        qw, qx, qy, qz = (float(data.qpos[3]), float(data.qpos[4]), float(data.qpos[5]), float(data.qpos[6]))
        roll, pitch = _quat_to_roll_pitch(qw, qx, qy, qz)
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        if args.assist_mode == "body_force":
            force_x = 240.0 * (args.target_speed - vx)
            force_z = 900.0 * (args.target_height - torso_z) - 55.0 * float(data.qvel[2])
            torque_x = -180.0 * roll - 25.0 * float(data.qvel[3])
            torque_y = -180.0 * pitch - 25.0 * float(data.qvel[4])
            data.qfrc_applied[0] = np.clip(force_x, -args.max_assist_force_x, args.max_assist_force_x)
            data.xfrc_applied[torso_id, 0] = np.clip(force_x, -args.max_assist_force_x, args.max_assist_force_x)
            data.xfrc_applied[torso_id, 2] = np.clip(force_z, -args.max_assist_force_z, args.max_assist_force_z)
            data.xfrc_applied[torso_id, 3] = np.clip(torque_x, -args.max_assist_torque, args.max_assist_torque)
            data.xfrc_applied[torso_id, 4] = np.clip(torque_y, -args.max_assist_torque, args.max_assist_torque)
            summary["external_force_write_count"] = int(summary["external_force_write_count"]) + 1
            summary["external_torque_write_count"] = int(summary["external_torque_write_count"]) + 1
        mujoco.mj_step(model, data)

        if initial_x is None:
            initial_x = float(data.qpos[0])
        if step % stride == 0 or step == args.steps - 1:
            torso_x = float(data.qpos[0])
            torso_z = float(data.qpos[2])
            qw, qx, qy, qz = (float(data.qpos[3]), float(data.qpos[4]), float(data.qpos[5]), float(data.qpos[6]))
            roll, pitch = _quat_to_roll_pitch(qw, qx, qy, qz)
            tilt = math.hypot(roll, pitch)
            travel = torso_x - initial_x
            fall = int(torso_z < args.fall_height or tilt > args.max_tilt_rad)
            summary["completed_steps"] = step + 1
            summary["max_travel_x_m"] = max(float(summary["max_travel_x_m"]), travel)
            summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), tilt)
            summary["min_torso_z_m"] = (
                torso_z if summary["min_torso_z_m"] is None else min(float(summary["min_torso_z_m"]), torso_z)
            )
            summary["fall_events"] = int(summary["fall_events"]) + fall
            states.append(
                {
                    "step": float(step),
                    "time_s": float(t),
                    "torso_x": torso_x,
                    "torso_z": torso_z,
                    "roll": roll,
                    "pitch": pitch,
                    "tilt": tilt,
                    "travel": travel,
                    "fall": float(fall),
                }
            )
    return states, summary


def _draw_frame(
    states: list[dict[str, float]],
    idx: int,
    summary: dict[str, float | int | str | bool | None],
    width: int,
    height: int,
) -> Image.Image:
    state = states[idx]
    title_font = _font(32)
    label_font = _font(18)
    small_font = _font(15)
    image = Image.new("RGB", (width, height), (244, 245, 246))
    draw = ImageDraw.Draw(image)
    panel = (68, 116, width - 68, height - 148)
    draw.rectangle(panel, fill=(234, 237, 239), outline=(38, 47, 57), width=2)
    # Include the welded payload in the right-hand margin so the final poster
    # does not crop the box when the torso reaches the end of the path.
    x_min = -0.08
    x_max = max(0.95, float(summary["max_travel_x_m"]) + 0.62)
    ground_y = _project(0.0, 0.02, x_min, x_max, 0.0, 0.85, panel)[1]
    draw.line([(panel[0], ground_y), (panel[2], ground_y)], fill=(85, 94, 103), width=4)
    for i in range(7):
        z = i / 6 * 0.85
        _, y = _project(0.0, z, x_min, x_max, 0.0, 0.85, panel)
        draw.line([(panel[0], y), (panel[2], y)], fill=(216, 221, 225), width=1)

    path = [_project(s["torso_x"], s["torso_z"], x_min, x_max, 0.0, 0.85, panel) for s in states[: idx + 1]]
    if len(path) > 1:
        draw.line(path, fill=(198, 127, 40), width=4)

    torso = _project(state["torso_x"], state["torso_z"], x_min, x_max, 0.0, 0.85, panel)
    pitch = state["pitch"]
    phase = 2.0 * math.pi * 1.6 * state["time_s"]
    leg_offsets = [(-0.18, 0.0, (52, 102, 160)), (0.18, 0.0, (52, 102, 160)), (-0.18, math.pi, (32, 74, 126)), (0.18, math.pi, (32, 74, 126))]
    for dx, ph, color in leg_offsets:
        swing = math.sin(phase + ph)
        hip = _project(state["torso_x"] + dx, state["torso_z"] - 0.08, x_min, x_max, 0.0, 0.85, panel)
        foot_x = state["torso_x"] + dx + 0.06 * swing
        foot_z = 0.035 + 0.06 * max(0.0, swing)
        knee = _project((state["torso_x"] + dx + foot_x) * 0.5, 0.27 + 0.05 * max(0.0, swing), x_min, x_max, 0.0, 0.85, panel)
        foot = _project(foot_x, foot_z, x_min, x_max, 0.0, 0.85, panel)
        draw.line([hip, knee, foot], fill=color, width=9, joint="curve")
        draw.ellipse((foot[0] - 26, foot[1] - 7, foot[0] + 26, foot[1] + 7), fill=(35, 42, 48))

    _draw_rotated_box(draw, torso, 112, 39, -pitch, (64, 77, 91), (19, 28, 38))
    draw.text((torso[0] - 45, torso[1] - 10), "torso", font=small_font, fill=(238, 242, 245))
    payload_center = _project(state["torso_x"] + 0.34, state["torso_z"] + 0.03, x_min, x_max, 0.0, 0.85, panel)
    _draw_rotated_box(draw, payload_center, 85, 58, -pitch, (166, 116, 63), (89, 56, 26))
    draw.text((payload_center[0] - 34, payload_center[1] - 9), "4kg box", font=small_font, fill=(48, 31, 16))

    status = "PASS DIAGNOSTIC" if int(summary["fall_events"]) == 0 and float(summary["max_travel_x_m"]) >= 0.20 else "FAIL DIAGNOSTIC"
    status_color = (43, 125, 76) if status.startswith("PASS") else (169, 58, 50)
    draw.text((64, 31), "MuJoCo Robot-like Payload Carry Diagnostic", font=title_font, fill=(28, 35, 43))
    draw.text(
        (66, 75),
        "dynamic quadruped body + joints + welded payload + explicit body-force stabilizer; not free-box grasp or final success",
        font=label_font,
        fill=(88, 54, 35),
    )
    x = panel[0] + 16
    x = _badge(draw, (x, panel[1] + 12), status, small_font, (255, 255, 255), status_color, status_color)
    x = _badge(draw, (x, panel[1] + 12), f"travel {float(summary['max_travel_x_m']):.3f} m", small_font, (255, 255, 255), (95, 105, 116), (43, 51, 61))
    _badge(draw, (x, panel[1] + 12), f"max tilt {float(summary['max_tilt_rad']):.3f} rad", small_font, (255, 255, 255), (95, 105, 116), (43, 51, 61))
    metrics = [
        f"payload: {float(summary['payload_mass_kg']):.1f} kg welded",
        f"target speed: {float(summary['target_speed_mps']):.2f} m/s",
        f"falls: {summary['fall_events']}",
        f"root writes: {summary['root_pose_write_count']}/{summary['root_velocity_write_count']}",
        f"external force writes: {summary['external_force_write_count']}",
    ]
    x = 78
    y = height - 112
    for text in metrics:
        draw.rounded_rectangle((x - 8, y - 8, x + 218, y + 28), radius=7, fill=(255, 255, 255), outline=(204, 211, 216))
        draw.text((x, y), text, font=small_font, fill=(39, 47, 56))
        x += 232
    footer = f"frame {idx + 1}/{len(states)}  step {int(state['step'])}  x {state['torso_x']:.3f}  z {state['torso_z']:.3f}  tilt {state['tilt']:.3f}"
    draw.text((80, height - 62), footer, font=small_font, fill=(65, 73, 82))
    return image


def main() -> int:
    _refuse_login_node()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states, summary = _simulate(args)
    frames = [_draw_frame(states, i, summary, args.width, args.height) for i in range(len(states))]
    gif_path = args.output_dir / "mujoco_quadruped_payload_fallback.gif"
    poster_path = args.output_dir / "mujoco_quadruped_payload_fallback_poster.png"
    summary_path = args.output_dir / "mujoco_quadruped_payload_visual_summary.json"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(args.gif_duration_ms),
        loop=0,
        optimize=True,
    )
    frames[-1].save(poster_path)
    summary.update(
        {
            "visualization_type": "schematic_rollout_replay_not_camera_render",
            "gif": str(gif_path),
            "poster": str(poster_path),
            "frame_count": len(frames),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] GIF: {gif_path}")
    print(f"[INFO] Poster: {poster_path}")
    print(f"[INFO] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
