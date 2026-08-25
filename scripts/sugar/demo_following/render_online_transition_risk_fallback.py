#!/usr/bin/env python3
"""Render matched direct and causal transition-risk behavior from exact traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from render_frozen_trace_behavior import (
    PANEL_ORIGINS,
    ROOT,
    decode,
    draw_sequence,
    first_episode,
    load_npz,
    recenter,
)


SOURCE_HZ = 50.0
VIDEO_FPS = 20
PLAYBACK_RATE = 0.5
DECISION_FRAME = 49
CANVAS = (1280, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _profile_is_directly_unsafe(
    result: dict[str, object], trace: dict[str, np.ndarray], env: int
) -> bool:
    return bool(
        result["profiles"][env]["physical_robot_fall"]
        or np.max(np.abs(trace["student_action"][:, env])) > 25.0
    )


def _select_profile(
    direct_result: dict[str, object],
    direct: dict[str, np.ndarray],
    risk: dict[str, np.ndarray],
    invalid_path: Path,
) -> tuple[int, str]:
    if invalid_path.exists():
        with np.load(invalid_path, allow_pickle=False) as invalid:
            invalid_envs = invalid["env_indices"].astype(np.int64)
        if invalid_envs.size:
            return int(invalid_envs[0]), "first numerically invalid transition"
    latched = risk["transition_risk_latched_fallback"][-1].astype(bool)
    unsafe = np.asarray(
        [_profile_is_directly_unsafe(direct_result, direct, env) for env in range(20)]
    )
    intersection = np.flatnonzero(latched & unsafe)
    if intersection.size:
        return int(intersection[0]), "directly unsafe and risk-latched"
    selected = np.flatnonzero(latched)
    if selected.size:
        return int(selected[0]), "risk-latched"
    selected = np.flatnonzero(unsafe)
    if selected.size:
        return int(selected[0]), "directly unsafe (no profile latched)"
    return 0, "fixed fallback profile"


def _put(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = (35, 35, 35),
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def _draw_risk_bar(
    frame: np.ndarray, probability: float, threshold: float, latched: bool
) -> None:
    left, top, width, height = 1030, 60, 210, 10
    cv2.rectangle(frame, (left, top), (left + width, top + height), (220, 220, 220), -1)
    cv2.rectangle(
        frame,
        (left, top),
        (left + int(np.clip(probability, 0.0, 1.0) * width), top + height),
        (45, 75, 210) if latched else (50, 160, 80),
        -1,
    )
    threshold_x = left + int(threshold * width)
    cv2.line(
        frame, (threshold_x, top - 3), (threshold_x, top + height + 3),
        (20, 20, 20), 2, cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    if experiments not in root.parents or experiments not in output.parents:
        raise ValueError("trace and video paths must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    direct_dir = root / "direct_carry_on_big"
    risk_dir = root / "risk_latched_fallback"
    direct_result = json.loads(
        (direct_dir / "RESULT.json").read_text(encoding="utf-8")
    )
    risk_result = json.loads((risk_dir / "RESULT.json").read_text(encoding="utf-8"))
    direct = load_npz(direct_dir / "TRACE.npz")
    risk = load_npz(risk_dir / "TRACE.npz")
    paired_keys = (
        "initial_robot_root_state_w",
        "initial_robot_joint_pos",
        "initial_robot_joint_vel",
        "initial_object_root_state_w",
        "post_prefix_robot_root_state_w",
        "post_prefix_robot_joint_pos",
        "post_prefix_robot_joint_vel",
        "post_prefix_object_root_state_w",
        "prefix_action",
    )
    initial_exact = all(np.array_equal(direct[key], risk[key]) for key in paired_keys)
    candidate_exact = np.array_equal(
        direct["student_action"][: DECISION_FRAME + 1],
        risk["student_action"][: DECISION_FRAME + 1],
    )
    if not initial_exact or not candidate_exact:
        raise RuntimeError("direct and fallback traces are not a matched causal pair")
    invalid_path = risk_dir / "INVALID_TRANSITION.npz"
    env, selection_reason = _select_profile(
        direct_result, direct, risk, invalid_path
    )
    body_names = direct["robot_body_names"]
    if not np.array_equal(body_names, risk["robot_body_names"]):
        raise RuntimeError("robot body-name drift")
    name_to_id = {str(name): index for index, name in enumerate(body_names)}
    direct_sequence = recenter(first_episode(direct, env))
    risk_sequence = recenter(first_episode(risk, env))
    source_frames = min(
        direct_sequence["body"].shape[0], risk_sequence["body"].shape[0]
    )
    frame_count = math.ceil(
        (source_frames - 1) * VIDEO_FPS / (SOURCE_HZ * PLAYBACK_RATE)
    ) + 1
    threshold = float(risk_result["transition_risk_threshold"])
    invalid_frame = (
        risk_result.get("first_invalid_transition") or {}
    ).get("frame")
    probabilities = risk["transition_risk_probability"][:, env]
    latched = risk["transition_risk_latched_fallback"][:, env].astype(bool)
    direct_root = direct["robot_root_state_w"][:, env, :3]
    risk_root = risk["robot_root_state_w"][:, env, :3]
    direct_root_origin = direct_root[0].copy()
    risk_root_origin = risk_root[0].copy()
    output.mkdir(parents=True, exist_ok=False)
    video = output / "direct_vs_causal_risk_fallback.mp4"
    temporary = video.with_suffix(".partial.mp4")
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
            source_index = min(
                round(video_frame * SOURCE_HZ * PLAYBACK_RATE / VIDEO_FPS),
                source_frames - 1,
            )
            frame = np.full((CANVAS[1], CANVAS[0], 3), 255, dtype=np.uint8)
            draw_sequence(
                frame, direct_sequence, source_index, PANEL_ORIGINS[0], name_to_id
            )
            draw_sequence(
                frame, risk_sequence, source_index, PANEL_ORIGINS[1], name_to_id
            )
            route = "KICK DOMAIN FALLBACK" if latched[source_index] else "CARRY CANDIDATE"
            direct_xy = float(
                np.linalg.norm(direct_root[source_index, :2] - direct_root_origin[:2])
            )
            risk_xy = float(
                np.linalg.norm(risk_root[source_index, :2] - risk_root_origin[:2])
            )
            risk_z_loss = float(risk_root_origin[2] - risk_root[source_index, 2])
            _put(frame, "DIRECT CARRY45 ON BIGBOX", (18, 28), 0.60, thickness=2)
            _put(frame, "CAUSAL RISK FALLBACK", (658, 28), 0.60, thickness=2)
            _put(
                frame,
                f"frame {source_index:03d} | direct max |a| "
                f"{np.max(np.abs(direct['executed_action'][source_index, env])):6.2f} "
                f"| root xy {direct_xy:.2f} m",
                (18, 53),
                0.45,
            )
            _put(
                frame,
                f"risk {probabilities[source_index]:.3f} / threshold {threshold:.3f}",
                (658, 51),
                0.43,
                (30, 30, 170) if latched[source_index] else (30, 120, 50),
            )
            _put(
                frame,
                f"{route} | root xy {risk_xy:.2f} m | z loss {risk_z_loss:.2f} m",
                (658, 72),
                0.41,
                (30, 30, 170) if latched[source_index] else (30, 120, 50),
            )
            _draw_risk_bar(
                frame, float(probabilities[source_index]), threshold,
                bool(latched[source_index]),
            )
            if source_index <= DECISION_FRAME:
                _put(
                    frame, "EARLY WINDOW", (1110, 88), 0.35,
                    (70, 70, 70),
                )
            else:
                _put(
                    frame, "LATCHED AT FRAME 49", (1060, 88), 0.35,
                    (70, 70, 70),
                )
            if invalid_frame is not None:
                _put(
                    frame,
                    f"TRACE STOPS BEFORE INVALID TRANSITION AT FRAME {invalid_frame}",
                    (680, 108),
                    0.40,
                    (25, 25, 190),
                    1,
                )
            if risk_xy > 2.0 or risk_z_loss > 0.35:
                _put(
                    frame,
                    "ROBOT LEFT FIXED VIEW: TRANSITION IS UNSTABLE",
                    (720, 350),
                    0.58,
                    (20, 20, 190),
                    2,
                )
            cv2.line(frame, (640, 72), (640, 698), (180, 180, 180), 1)
            _put(
                frame,
                f"Exact recorded PhysX body centers + object pose | env {env} | "
                "0.5x playback | box wireframe size illustrative",
                (260, 712),
                0.42,
                (75, 75, 75),
            )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    temporary.replace(video)
    video_decode = decode(video)
    checks = {
        "matched_initial_and_prefix_state_exact": initial_exact,
        "candidate_actions_exact_through_decision": candidate_exact,
        "selected_profile_is_latched": bool(latched[-1]),
        "all_available_valid_risk_frames_displayed": (
            source_frames == risk["robot_body_position_w"].shape[0]
        ),
        "invalid_transition_is_explicitly_labelled": invalid_frame is not None,
        "h264_yuv420p_decodes": bool(video_decode["passed"]),
    }
    proof = {
        "protocol": "sugar_online_transition_risk_exact_trace_video_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "input_root": str(root),
        "source_env": env,
        "selection_reason": selection_reason,
        "method_result_passed": bool(risk_result["passed"]),
        "available_valid_source_frames": source_frames,
        "first_invalid_transition_frame": invalid_frame,
        "video": {
            "path": str(video),
            "bytes": video.stat().st_size,
            "frames": frame_count,
            "fps": VIDEO_FPS,
            "decode": video_decode,
        },
        "rendering_semantics": (
            "Both panels use exact recorded PhysX body centers and object poses from "
            "the matched frozen rollout. No physics replay is performed. The risk bar "
            "shows the online causal probability through frame 49 and its frozen "
            "nine-sample mean afterward."
        ),
    }
    (output / "RENDER_PROOF.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(proof, indent=2, sort_keys=True))
    if not proof["passed"]:
        raise RuntimeError("transition-risk exact-trace video proof failed")


if __name__ == "__main__":
    main()
