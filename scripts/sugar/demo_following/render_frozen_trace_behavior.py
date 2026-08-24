#!/usr/bin/env python3
"""Render exact frozen PhysX trajectories when cluster Vulkan cameras are unavailable.

This is a visualization-only fallback.  It draws the recorded robot body centers and
object pose; it neither reruns physics nor synthesizes policy state.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
import subprocess

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_HZ = 50.0
VIDEO_FPS = 20
PLAYBACK_RATE_ACTUAL = 0.25
CANVAS_SIZE = (1280, 720)
PANEL_ORIGINS = ((10, 80), (650, 80))
PANEL_SIZE = (620, 610)


CONNECTION_NAMES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct-trace", type=Path, required=True)
    parser.add_argument("--unrelated-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-env", type=int, default=20)
    parser.add_argument("--policy-update", type=int, default=64)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "robot_50hz.npz", allow_pickle=False) as archive:
        body = np.asarray(archive["body_pos_w"], dtype=np.float32)
    with (path / "obj_motion_global_50hz.pkl").open("rb") as stream:
        obj = {name: np.asarray(value) for name, value in pickle.load(stream).items()}
    length = min(body.shape[0], obj["obj_trans"].shape[0])
    rotation = np.asarray(obj["obj_rot"][:length], dtype=np.float32)
    return {
        "body": body[:length],
        "object_position": np.asarray(obj["obj_trans"][:length], dtype=np.float32),
        "object_rotation": rotation,
    }


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def first_episode(trace: dict[str, np.ndarray], source_env: int) -> dict[str, np.ndarray]:
    hits = np.flatnonzero(trace["done"][:, source_env])
    last_transition = int(hits[0]) if hits.size else trace["done"].shape[0] - 1
    count = last_transition + 2
    quaternions = trace["object_root_state_w"][:count, source_env, 3:7]
    rotations = np.stack(
        [quaternion_wxyz_to_matrix(quaternion) for quaternion in quaternions]
    )
    return {
        "body": trace["robot_body_position_w"][:count, source_env].astype(np.float32),
        "object_position": trace["object_root_state_w"][:count, source_env, :3].astype(np.float32),
        "object_rotation": rotations,
    }


def recenter(sequence: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    focus_xy = 0.5 * (
        sequence["body"][0, 0, :2] + sequence["object_position"][0, :2]
    )
    body = sequence["body"].copy()
    obj = sequence["object_position"].copy()
    body[..., :2] -= focus_xy
    obj[..., :2] -= focus_xy
    return {**sequence, "body": body, "object_position": obj}


def project(points: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    scale = 245.0
    u = origin[0] + 310.0 + scale * (0.80 * x - 0.60 * y)
    v = origin[1] + 565.0 - scale * (z + 0.25 * x + 0.20 * y)
    return np.stack((u, v), axis=-1).round().astype(np.int32)


def draw_ground(frame: np.ndarray, origin: tuple[int, int]) -> None:
    values = np.linspace(-1.2, 1.2, 9)
    for value in values:
        for points in (
            np.asarray([[-1.2, value, 0.0], [1.2, value, 0.0]]),
            np.asarray([[value, -1.2, 0.0], [value, 1.2, 0.0]]),
        ):
            uv = project(points, origin)
            cv2.line(frame, tuple(uv[0]), tuple(uv[1]), (225, 225, 225), 1, cv2.LINE_AA)


def draw_box(
    frame: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    origin: tuple[int, int],
) -> None:
    # Pose marker dimensions approximate the visible small-box asset.  The
    # center and orientation are exact values from the trace/reference.
    half = np.asarray([0.22, 0.16, 0.13], dtype=np.float32)
    corners = np.asarray(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float32,
    ) * half
    world = corners @ rotation.T + position
    uv = project(world, origin)
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for first, second in edges:
        cv2.line(frame, tuple(uv[first]), tuple(uv[second]), (35, 75, 210), 3, cv2.LINE_AA)


def draw_sequence(
    frame: np.ndarray,
    sequence: dict[str, np.ndarray],
    index: int,
    origin: tuple[int, int],
    name_to_id: dict[str, int],
) -> None:
    index = min(index, sequence["body"].shape[0] - 1)
    body = sequence["body"][index]
    body_uv = project(body, origin)
    draw_ground(frame, origin)
    trail_start = max(0, index - 100)
    trail = project(sequence["object_position"][trail_start:index + 1], origin)
    if trail.shape[0] > 1:
        cv2.polylines(frame, [trail], False, (150, 185, 245), 2, cv2.LINE_AA)
    for first_name, second_name in CONNECTION_NAMES:
        first, second = name_to_id[first_name], name_to_id[second_name]
        cv2.line(frame, tuple(body_uv[first]), tuple(body_uv[second]), (55, 55, 55), 5, cv2.LINE_AA)
    for name, body_id in name_to_id.items():
        if name.endswith("rubber_hand"):
            color, radius = (210, 105, 30), 7
        elif name.endswith("ankle_roll_link"):
            color, radius = (30, 145, 230), 7
        else:
            color, radius = (65, 65, 65), 4
        cv2.circle(frame, tuple(body_uv[body_id]), radius, color, -1, cv2.LINE_AA)
    draw_box(
        frame,
        sequence["object_position"][index],
        sequence["object_rotation"][index],
        origin,
    )


def decode(path: Path) -> dict[str, object]:
    process = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "info", "-i", str(path),
            "-map", "0:v:0", "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
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


def render_pair(
    reference: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    body_names: np.ndarray,
    demo_label: str,
    output: Path,
) -> dict[str, object]:
    reference, actual = recenter(reference), recenter(actual)
    name_to_id = {str(name): index for index, name in enumerate(body_names)}
    missing = sorted(
        {name for pair in CONNECTION_NAMES for name in pair}.difference(name_to_id)
    )
    if missing:
        raise KeyError(f"missing body names: {missing}")
    reference_frames = reference["body"].shape[0]
    actual_frames = actual["body"].shape[0]
    output_reference = math.ceil((reference_frames - 1) * VIDEO_FPS / SOURCE_HZ) + 1
    output_actual = (
        math.ceil(
            (actual_frames - 1) * VIDEO_FPS / (SOURCE_HZ * PLAYBACK_RATE_ACTUAL)
        )
        + 1
    )
    frame_count = max(output_reference, output_actual)
    temporary = output.with_suffix(".partial.mp4")
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
            reference_index = round(video_frame * SOURCE_HZ / VIDEO_FPS)
            actual_index = round(
                video_frame * SOURCE_HZ * PLAYBACK_RATE_ACTUAL / VIDEO_FPS
            )
            frame = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), 255, np.uint8)
            draw_sequence(frame, reference, reference_index, PANEL_ORIGINS[0], name_to_id)
            draw_sequence(frame, actual, actual_index, PANEL_ORIGINS[1], name_to_id)
            cv2.putText(frame, demo_label, (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, "ACTUAL FROZEN POLICY (UPDATE 64)", (658, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, "DEMO 1x (then holds)", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.putText(frame, "ACTUAL 0.25x (then holds)", (658, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.putText(frame, "Exact recorded body centers + box pose; no physics replay", (385, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.line(frame, (640, 72), (640, 698), (180, 180, 180), 1)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    temporary.replace(output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "frames": frame_count,
        "fps": VIDEO_FPS,
        "reference_source_frames": reference_frames,
        "actual_source_frames": actual_frames,
        "reference_fully_displayed": frame_count >= output_reference,
        "actual_fully_displayed": frame_count >= output_actual,
        "decode": decode(output),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    experiments = (ROOT / "experiments").resolve()
    traces = (args.correct_trace.resolve(), args.unrelated_trace.resolve())
    if not all(path.is_relative_to(experiments) for path in (*traces, output)):
        raise ValueError("all paths must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    if args.source_env != 20 or args.policy_update != 64:
        raise ValueError("the admitted endpoint visualization is fixed to update 64, env 20")

    expected_arms = ("same_teacher_correct_reward", "same_teacher_unrelated_reward")
    loaded: list[dict[str, np.ndarray]] = []
    admitted_results = []
    for trace_path, expected_arm in zip(traces, expected_arms, strict=True):
        result_path = trace_path.with_name("RESULT.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("passed") is not True
            or not all(result.get("checks", {}).values())
            or result.get("arm") != expected_arm
            or result.get("policy_updates") != [32, 64]
            or result.get("profiles_per_update") != 20
        ):
            raise RuntimeError(f"unadmitted frozen evaluation: {trace_path}")
        admitted_results.append(str(result_path))
        loaded.append(load_npz(trace_path))
    if not np.array_equal(loaded[0]["ordered_body_names"], loaded[1]["ordered_body_names"]):
        raise RuntimeError("body order differs between matched arms")
    if not all(trace["done"].shape[1] == 40 for trace in loaded):
        raise RuntimeError("unexpected multi-update profile layout")

    output.mkdir(parents=True)
    references = (
        load_reference(ROOT / "SUGAR/data/CarryBox/data_045"),
        load_reference(ROOT / "SUGAR/data/KickBox/data_021"),
    )
    actuals = tuple(first_episode(trace, args.source_env) for trace in loaded)
    videos = (
        render_pair(
            references[0], actuals[0], loaded[0]["ordered_body_names"],
            "INPUT DEMO: CARRYBOX MOTION 45", output / "01_correct_demo_and_actual_behavior.mp4",
        ),
        render_pair(
            references[1], actuals[1], loaded[1]["ordered_body_names"],
            "INPUT DEMO: UNRELATED KICKBOX MOTION 21", output / "02_unrelated_kickbox_demo_and_actual_behavior.mp4",
        ),
    )
    checks = {
        "matched_frozen_results_admitted": True,
        "update64_profile_selected": args.source_env == 20,
        "correct_video_decodes_h264_yuv420p": videos[0]["decode"]["passed"],
        "unrelated_video_decodes_h264_yuv420p": videos[1]["decode"]["passed"],
        "both_references_fully_displayed": all(video["reference_fully_displayed"] for video in videos),
        "both_actual_episodes_fully_displayed": all(video["actual_fully_displayed"] for video in videos),
        "nonempty_videos": all(video["bytes"] > 100_000 for video in videos),
    }
    proof = {
        "protocol": "sugar_phase_event_reward_matched_exact_trace_video_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "policy_update": args.policy_update,
        "source_env": args.source_env,
        "rendering_semantics": (
            "Offline visualization of exact frozen-evaluation robot body centers and object pose. "
            "No camera-enabled physics rerun; pose-marker box dimensions are illustrative."
        ),
        "videos": videos,
        "sources": [
            *(str(path) for path in traces),
            *admitted_results,
            str(ROOT / "SUGAR/data/CarryBox/data_045/robot_50hz.npz"),
            str(ROOT / "SUGAR/data/KickBox/data_021/robot_50hz.npz"),
            str(Path(__file__).resolve()),
        ],
    }
    temporary_proof = output / "RENDER_PROOF.partial.json"
    temporary_proof.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_proof.replace(output / "RENDER_PROOF.json")
    if not proof["passed"]:
        raise RuntimeError("exact trace behavior render failed")
    print(json.dumps(proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
