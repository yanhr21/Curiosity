#!/usr/bin/env python3
"""Render an actual frozen policy rollout with bilateral anatomical tactile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

from render_sugar_whole_hand_carrybox import (
    HAND_WIDTH,
    HEIGHT,
    WIDTH,
    WORLD_HEIGHT,
    close_crop,
    draw_hand,
    fit_world,
    put,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--scale-bundle-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Bundle included in a shared active-taxel 95th-percentile display "
            "scale. Repeat for a matched video cohort."
        ),
    )
    return parser.parse_args()


def shared_display_scale(bundle_roots: list[Path]) -> tuple[float, float]:
    normal_values: list[np.ndarray] = []
    shear_values: list[np.ndarray] = []
    for bundle_root in bundle_roots:
        trace_path = bundle_root.resolve() / "tactile_trace.npz"
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        with np.load(trace_path, allow_pickle=False) as source:
            normal = np.abs(np.asarray(source["normal_force"], dtype=np.float32))
            shear = np.linalg.norm(
                np.asarray(source["signed_shear"], dtype=np.float32), axis=-1
            )
            active = np.asarray(source["penetration"], dtype=np.float32) > 0.0
        if np.any(active):
            normal_values.append(normal[active])
            shear_values.append(shear[active])
    if not normal_values:
        raise RuntimeError("shared display-scale cohort has no active taxels")
    return (
        float(np.percentile(np.concatenate(normal_values), 95.0)),
        float(np.percentile(np.concatenate(shear_values), 95.0)),
    )


def main() -> None:
    args = parse_args()
    root = args.bundle_root.resolve()
    summary_path = root / "summary.json"
    trace_path = root / "tactile_trace.npz"
    world_path = root / "world_carrybox.mp4"
    for path in (summary_path, trace_path, world_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as source:
        trace = {key: source[key] for key in source.files}

    normal = np.asarray(trace["normal_force"], dtype=np.float32)
    shear = np.asarray(trace["signed_shear"], dtype=np.float32)
    penetration = np.asarray(trace["penetration"], dtype=np.float32)
    if normal.shape[1:] != (2, 27, 20, 25):
        raise RuntimeError(f"unexpected normal shape {normal.shape}")
    if shear.shape[1:] != (2, 27, 20, 25, 2):
        raise RuntimeError(f"unexpected shear shape {shear.shape}")
    if penetration.shape != normal.shape:
        raise RuntimeError(f"unexpected penetration shape {penetration.shape}")
    frame_count = len(normal)
    required = (
        "object_state_w",
        "cumulative_reward_before_action",
        "raw_actor_tactile_nonzero_values",
        "fed_actor_tactile_nonzero_values",
        "same_state_live_zero_action_abs_max",
        "same_state_live_patch_permuted_action_abs_max",
    )
    for key in required:
        if len(trace[key]) != frame_count:
            raise RuntimeError(f"trace length mismatch for {key}")

    capture = cv2.VideoCapture(str(world_path))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count:
        raise RuntimeError("world/tactile frame count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
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
    mode = str(summary["actor_tactile_mode"])
    source = str(summary.get("actor_tactile_source", ""))
    mode_label = {
        "exact_zero_no_sensor_read": (
            "ACTOR INPUT: EXACT ZERO / NO SENSOR READ "
            "(DISPLAY IS PHYSICAL DIAGNOSTIC)"
        ),
        "live_physical_tactile": "ACTOR INPUT: LIVE PHYSICAL TACTILE",
        "evaluation_time_exact_zero": (
            "ACTOR INPUT: EXACT ZERO (DISPLAY IS PHYSICAL SENSOR ONLY)"
        ),
        "fixed_anatomical_patch_permutation": (
            "ACTOR INPUT: FIXED ANATOMICAL PATCH PERMUTATION"
        ),
    }.get(
        source,
        {
            "live": "ACTOR INPUT: LIVE PHYSICAL TACTILE",
            "zeroed": "ACTOR INPUT: EXACT ZERO (DISPLAY IS PHYSICAL SENSOR ONLY)",
            "patch_permuted": "ACTOR INPUT: FIXED ANATOMICAL PATCH PERMUTATION",
        }[mode],
    )
    if args.scale_bundle_root:
        normal_max, shear_max = shared_display_scale(args.scale_bundle_root)
        scale_semantics = "shared matched-cohort active-taxel p95"
    else:
        normal_max = float(summary["normal_display_scale_n"])
        shear_max = float(summary["shear_display_scale_n"])
        scale_semantics = "bundle-declared fixed scale"
    initial_z = float(trace["object_state_w"][0, 2])
    cumulative_reward = np.asarray(
        trace["cumulative_reward_before_action"], dtype=np.float64
    )
    try:
        for step in range(frame_count):
            ok, world = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode world frame {step}")
            canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
            canvas[:WORLD_HEIGHT, :HAND_WIDTH] = fit_world(
                world, (HAND_WIDTH, WORLD_HEIGHT)
            )
            canvas[:WORLD_HEIGHT, HAND_WIDTH:] = close_crop(
                world, (HAND_WIDTH, WORLD_HEIGHT)
            )
            cv2.line(
                canvas,
                (HAND_WIDTH, 0),
                (HAND_WIDTH, WORLD_HEIGHT),
                (255, 255, 255),
                3,
            )
            cv2.rectangle(canvas, (14, 14), (WIDTH - 14, 62), (255, 255, 255), -1)
            put(canvas, args.title, (28, 47), 0.76, 2)
            cv2.rectangle(canvas, (14, 620), (WIDTH - 14, 664), (25, 25, 25), -1)
            put(
                canvas,
                (
                    f"{mode_label}   |   DISPLAY {scale_semantics}: "
                    f"|Z| {normal_max:.4f} N, XY {shear_max:.4f} N"
                ),
                (28, 650),
                0.55,
                2,
                (255, 255, 255),
            )
            lift = float(trace["object_state_w"][step, 2] - initial_z)
            raw_nonzero = int(trace["raw_actor_tactile_nonzero_values"][step])
            fed_nonzero = int(trace["fed_actor_tactile_nonzero_values"][step])
            live_zero = float(trace["same_state_live_zero_action_abs_max"][step])
            live_permuted = float(
                trace["same_state_live_patch_permuted_action_abs_max"][step]
            )
            cv2.rectangle(canvas, (14, 668), (WIDTH - 14, 712), (25, 25, 25), -1)
            put(
                canvas,
                (
                    f"step {step:03d}   lift {lift:+.3f} m   cumulative reward "
                    f"{cumulative_reward[step]:+.2f}   raw/fed nonzero "
                    f"{raw_nonzero}/{fed_nonzero}   same-state max action delta: "
                    f"live-zero {live_zero:.3f}, live-permuted {live_permuted:.3f}"
                ),
                (28, 698),
                0.50,
                1,
                (255, 255, 255),
            )
            draw_hand(
                canvas,
                0,
                normal[step, 0],
                shear[step, 0],
                penetration[step, 0],
                normal_max,
                shear_max,
            )
            draw_hand(
                canvas,
                1,
                normal[step, 1],
                shear[step, 1],
                penetration[step, 1],
                normal_max,
                shear_max,
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin closed")
            process.stdin.write(np.ascontiguousarray(canvas).tobytes())
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    decoded = cv2.VideoCapture(str(args.output))
    declared = int(decoded.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = 0
    shapes: set[tuple[int, ...]] = set()
    try:
        while True:
            ok, frame = decoded.read()
            if not ok:
                break
            decoded_frames += 1
            shapes.add(tuple(frame.shape))
    finally:
        decoded.release()
    if declared != frame_count or decoded_frames != frame_count:
        raise RuntimeError(
            f"full decode failed: expected={frame_count} declared={declared} "
            f"decoded={decoded_frames}"
        )
    if shapes != {(HEIGHT, WIDTH, 3)}:
        raise RuntimeError(f"unexpected decoded shapes {shapes}")
    record = {
        "schema": "native_whole_hand_tactile_policy_rollout_render_v1",
        "bundle": str(root),
        "actor_tactile_mode": mode,
        "actor_tactile_source": source or "legacy_bundle_mode_only",
        "frames": frame_count,
        "fps": args.fps,
        "resolution": [WIDTH, HEIGHT],
        "display_scale": {
            "semantics": scale_semantics,
            "normal_abs_n": normal_max,
            "shear_magnitude_n": shear_max,
            "cohort_bundles": [
                str(path.resolve()) for path in args.scale_bundle_root
            ],
        },
        "full_decode": {
            "passed": True,
            "declared_frames": declared,
            "decoded_frames": decoded_frames,
        },
        "display_semantics": (
            "World and physical tactile are same-step simulation outputs. The "
            "header separately identifies what tactile tensor entered the actor."
        ),
    }
    args.output.with_suffix(".render.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
