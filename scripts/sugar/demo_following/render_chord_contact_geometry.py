#!/usr/bin/env python3
"""Render synchronized exact-trace evidence for the CHORD geometry diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_HZ = 50.0
VIDEO_FPS = 20
PLAYBACK_RATE = 0.5
WIDTH, HEIGHT = 1920, 1080
TOP_Y, TOP_H = 76, 700
PANEL_W = 620
PANEL_X = (10, 650, 1290)

CONNECTIONS = (
    ("pelvis", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("pelvis", "waist_yaw_link"),
    ("waist_yaw_link", "waist_roll_link"),
    ("waist_roll_link", "torso_link"),
    ("torso_link", "head_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("left_wrist_roll_link", "left_wrist_pitch_link"),
    ("left_wrist_pitch_link", "left_wrist_yaw_link"),
    ("left_wrist_yaw_link", "left_rubber_hand"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
    ("right_wrist_roll_link", "right_wrist_pitch_link"),
    ("right_wrist_pitch_link", "right_wrist_yaw_link"),
    ("right_wrist_yaw_link", "right_rubber_hand"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _project(points: np.ndarray, panel_x: int) -> np.ndarray:
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    scale = 255.0
    u = panel_x + PANEL_W / 2 + scale * (0.82 * x - 0.58 * y)
    v = TOP_Y + TOP_H - 65 - scale * (z + 0.24 * x + 0.18 * y)
    return np.stack((u, v), axis=-1).round().astype(np.int32)


def _recenter(points: np.ndarray, origin_xy: np.ndarray) -> np.ndarray:
    result = points.copy()
    result[..., :2] -= origin_xy
    return result


def _draw_ground(frame: np.ndarray, panel_x: int) -> None:
    for value in np.linspace(-1.15, 1.15, 8):
        for line in (
            np.asarray([[-1.15, value, 0.0], [1.15, value, 0.0]]),
            np.asarray([[value, -1.15, 0.0], [value, 1.15, 0.0]]),
        ):
            uv = _project(line, panel_x)
            cv2.line(frame, tuple(uv[0]), tuple(uv[1]), (229, 229, 229), 1, cv2.LINE_AA)


def _draw_box(
    frame: np.ndarray, position: np.ndarray, quaternion: np.ndarray, panel_x: int
) -> None:
    half = np.asarray([0.34, 0.20, 0.12], dtype=np.float32)
    corners = np.asarray(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float32,
    ) * half
    world = corners @ _quat_matrix(quaternion).T + position
    uv = _project(world, panel_x)
    for first, second in (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ):
        cv2.line(frame, tuple(uv[first]), tuple(uv[second]), (40, 80, 205), 4, cv2.LINE_AA)


def _draw_world(
    frame: np.ndarray,
    *,
    trace: dict[str, np.ndarray],
    step: int,
    profile: int,
    panel_x: int,
    force_roles: tuple[int, int],
    title: str,
) -> None:
    names = [str(name) for name in trace["robot_body_names"].tolist()]
    ids = {name: index for index, name in enumerate(names)}
    origin_xy = trace["object_root_state_w"][0, profile, :2]
    body = _recenter(trace["robot_body_position_w"][step, profile], origin_xy)
    obj = trace["object_root_state_w"][step, profile].copy()
    obj[:2] -= origin_xy
    position_key = (
        "foot_contact_position_w"
        if "foot_contact_position_w" in trace
        else "contact_position_w"
    )
    valid_key = (
        "foot_contact_position_valid"
        if "foot_contact_position_valid" in trace
        else "contact_position_valid"
    )
    force_key = (
        "foot_contact_force_w"
        if "foot_contact_force_w" in trace
        else "contact_force_w"
    )
    positions = _recenter(trace[position_key][step, profile, list(force_roles)], origin_xy)
    valid = trace[valid_key][step, profile, list(force_roles)]
    forces = trace[force_key][step, profile, list(force_roles)]

    _draw_ground(frame, panel_x)
    body_uv = _project(body, panel_x)
    for first, second in CONNECTIONS:
        cv2.line(frame, tuple(body_uv[ids[first]]), tuple(body_uv[ids[second]]), (60, 60, 60), 5, cv2.LINE_AA)
    for name, body_id in ids.items():
        if name.endswith("ankle_roll_link"):
            color, radius = (25, 150, 230), 8
        elif name.endswith("rubber_hand"):
            color, radius = (205, 115, 35), 7
        else:
            color, radius = (70, 70, 70), 4
        cv2.circle(frame, tuple(body_uv[body_id]), radius, color, -1, cv2.LINE_AA)
    _draw_box(frame, obj[:3], obj[3:7], panel_x)

    for side, (point, is_valid, force) in enumerate(zip(positions, valid, forces, strict=True)):
        magnitude = float(np.linalg.norm(force))
        if not is_valid or magnitude <= 0.1:
            continue
        endpoint = point + force / max(magnitude, 1.0e-6) * min(0.30, 0.035 + 0.002 * magnitude)
        uv = _project(np.stack((point, endpoint)), panel_x)
        color = (40, 175, 45) if side == 0 else (185, 70, 185)
        cv2.circle(frame, tuple(uv[0]), 9, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(frame, tuple(uv[0]), tuple(uv[1]), color, 4, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(frame, f"{magnitude:.1f} N", tuple(uv[0] + np.asarray([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(frame, title, (panel_x + 8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (panel_x, TOP_Y), (panel_x + PANEL_W, TOP_Y + TOP_H), (175, 175, 175), 1)


def _draw_bottom(
    frame: np.ndarray,
    *,
    deltas: np.ndarray,
    pre_curve: np.ndarray,
    learned_curve: np.ndarray,
    active: np.ndarray,
    step: int,
    selected_profile: int,
) -> None:
    chart_x, chart_y, chart_w, chart_h = 25, 830, 750, 180
    cv2.putText(frame, "Paired CWS change: learned - exact-pre (all 20 profiles)", (chart_x, 805), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 2, cv2.LINE_AA)
    limit = max(0.24, float(np.max(np.abs(deltas))) * 1.10)
    zero_y = chart_y + chart_h // 2
    cv2.line(frame, (chart_x, zero_y), (chart_x + chart_w, zero_y), (80, 80, 80), 1)
    bar_w = chart_w / len(deltas)
    for index, value in enumerate(deltas):
        x0 = round(chart_x + index * bar_w + 3)
        x1 = round(chart_x + (index + 1) * bar_w - 3)
        height = round(abs(float(value)) / limit * (chart_h / 2 - 8))
        y1 = zero_y - height if value >= 0 else zero_y + height
        color = (30, 145, 80) if value > 0 else (45, 70, 205) if value < 0 else (170, 170, 170)
        cv2.rectangle(frame, (x0, min(zero_y, y1)), (x1, max(zero_y, y1)), color, -1)
        if index == selected_profile:
            cv2.rectangle(frame, (x0 - 2, chart_y), (x1 + 2, chart_y + chart_h), (0, 0, 0), 2)
        cv2.putText(frame, str(index), (x0, chart_y + chart_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (50, 50, 50), 1)

    curve_x, curve_y, curve_w, curve_h = 825, 830, 1065, 180
    cv2.putText(frame, f"Official CHORD CWS over time, matched profile {selected_profile}", (curve_x, 805), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 2, cv2.LINE_AA)
    ymax = max(1.0, float(np.max([pre_curve.max(), learned_curve.max()])) * 1.05)
    for index in np.flatnonzero(active):
        x0 = round(curve_x + index / max(1, len(active) - 1) * curve_w)
        cv2.line(frame, (x0, curve_y), (x0, curve_y + curve_h), (238, 238, 210), 2)
    def points(values: np.ndarray) -> np.ndarray:
        x = curve_x + np.arange(len(values)) / max(1, len(values) - 1) * curve_w
        y = curve_y + curve_h - values / ymax * curve_h
        return np.stack((x, y), axis=-1).round().astype(np.int32)
    cv2.polylines(frame, [points(pre_curve)], False, (215, 125, 30), 3, cv2.LINE_AA)
    cv2.polylines(frame, [points(learned_curve)], False, (25, 145, 55), 3, cv2.LINE_AA)
    cursor_x = round(curve_x + step / max(1, len(active) - 1) * curve_w)
    cv2.line(frame, (cursor_x, curve_y), (cursor_x, curve_y + curve_h), (20, 20, 20), 2)
    cv2.putText(frame, "exact-pre", (curve_x + 15, curve_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 125, 30), 2)
    cv2.putText(frame, "learned", (curve_x + 125, curve_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 145, 55), 2)
    cv2.putText(frame, "yellow = Kick21 reference contact active", (curve_x + 245, curve_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (90, 90, 40), 1)


def _decode(path: Path) -> dict[str, object]:
    process = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "info", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "exit_code": process.returncode,
        "h264": "Video: h264" in process.stderr,
        "yuv420p": "yuv420p" in process.stderr,
    }


def main() -> None:
    args = _args()
    collection = args.collection_root.resolve()
    output = args.output_dir.resolve()
    if not collection.is_relative_to((ROOT / "experiments").resolve()) or not output.is_relative_to((ROOT / "experiments").resolve()):
        raise ValueError("inputs and output must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    pre = _load(collection / "pre_update_kick/trace.npz")
    learned = _load(collection / "learned_kick/trace.npz")
    native = _load(collection / "native_kick21/TRACE.npz")
    metrics = _load(collection / "official_chord_representation/METRICS.npz")
    result = json.loads((collection / "official_chord_representation/RESULT.json").read_text())
    deltas = np.asarray(
        [
            result["learned"]["profiles"][index]["mean_cws_on_reference_contact"]
            - result["pre_update"]["profiles"][index]["mean_cws_on_reference_contact"]
            for index in range(20)
        ],
        dtype=np.float32,
    )
    selected_profile = int(np.argmax(np.abs(deltas)))
    native_roles = [str(value) for value in native["contact_role_names"].tolist()]
    native_force_roles = (native_roles.index("left_foot"), native_roles.index("right_foot"))
    native_frame_to_step = {int(frame): step for step, frame in enumerate(native["motion_frame"][:, 0])}
    frames = pre["motion_frame"][:, selected_profile]
    if not np.array_equal(frames, learned["motion_frame"][:, selected_profile]):
        raise RuntimeError("matched transition motion clocks differ")
    if any(int(frame) not in native_frame_to_step for frame in frames):
        raise RuntimeError("native reference does not cover the transition clock")

    output_path = output / "chord_contact_geometry_exact_trace.mp4"
    temporary = output_path.with_suffix(".partial.mp4")
    frame_count = math.ceil((len(frames) - 1) * VIDEO_FPS / (SOURCE_HZ * PLAYBACK_RATE)) + 1
    with imageio.get_writer(
        temporary,
        fps=VIDEO_FPS,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="warning",
    ) as writer:
        for video_frame in range(frame_count):
            step = min(len(frames) - 1, round(video_frame * SOURCE_HZ * PLAYBACK_RATE / VIDEO_FPS))
            motion_frame = int(frames[step])
            native_step = native_frame_to_step[motion_frame]
            image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
            _draw_world(image, trace=native, step=native_step, profile=0, panel_x=PANEL_X[0], force_roles=native_force_roles, title="REFERENCE: RELEASED KICK21 EXPERT, PROFILE 0")
            _draw_world(image, trace=pre, step=step, profile=selected_profile, panel_x=PANEL_X[1], force_roles=(0, 1), title=f"EXACT-PRE POLICY, PROFILE {selected_profile}")
            _draw_world(image, trace=learned, step=step, profile=selected_profile, panel_x=PANEL_X[2], force_roles=(0, 1), title=f"LEARNED POLICY, PROFILE {selected_profile}")
            _draw_bottom(
                image,
                deltas=deltas,
                pre_curve=metrics["pre_cws"][:, selected_profile],
                learned_curve=metrics["learned_cws"][:, selected_profile],
                active=metrics["pre_reference_active"][:, selected_profile],
                step=step,
                selected_profile=selected_profile,
            )
            cv2.putText(image, f"SUGAR motion frame {motion_frame} | 0.5x playback | green/purple arrows: live filtered PhysX foot-box forces", (22, 1056), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (55, 55, 55), 1, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    temporary.replace(output_path)
    decode = _decode(output_path)
    proof = {
        "protocol": "sugar_chord_contact_geometry_exact_trace_video_v1",
        "passed": bool(decode["exit_code"] == 0 and decode["h264"] and decode["yuv420p"] and output_path.stat().st_size > 100_000),
        "video": str(output_path),
        "bytes": output_path.stat().st_size,
        "frames": frame_count,
        "fps": VIDEO_FPS,
        "selected_profile": selected_profile,
        "selection_rule": "largest absolute paired CWS change; all 20 paired deltas remain visible",
        "selected_delta": float(deltas[selected_profile]),
        "aggregate_delta": float(result["paired"]["mean_learned_minus_pre_cws"]),
        "decode": decode,
        "semantics": "Exact recorded body centers, object pose, contact points and filtered PhysX force vectors; no physics or camera replay. Box mesh is an outline marker.",
    }
    (output / "RENDER_PROOF.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    if not proof["passed"]:
        raise RuntimeError("CHORD exact-trace video proof failed")
    print(json.dumps(proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
