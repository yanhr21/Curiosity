#!/usr/bin/env python3
"""Render a continuous SUGAR G1 tactile replay in short EGL worker processes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np


RAW_KEYS = {
    "raw_contact_index": -1,
    "raw_contact_kind": -1,
    "raw_patch": -1,
    "raw_counterpart_shape": -1,
    "raw_counterpart_particle": -1,
    "raw_sensor_is_shape0": 0,
    "raw_point_world_m": 0,
    "raw_point_patch_m": 0,
    "raw_force_world_n": 0,
    "raw_force_patch_n": 0,
    "raw_native_wrench_body0": 0,
    "raw_penetration_m": 0,
}

STATIC_KEYS = {"patch_names", "patch_size_m", "backend", "optical_available"}


def _pad(array: np.ndarray, width: int, value: int) -> np.ndarray:
    padding = [(0, 0), (0, width - array.shape[1])] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, padding, mode="constant", constant_values=value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=660)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-scale-n", type=float, default=25.0)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--solver", choices=("vbd", "mujoco"), default="vbd")
    parser.add_argument("--physics-substeps", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--vbd-contact-ke", type=float, default=1200.0)
    parser.add_argument("--vbd-contact-kd", type=float, default=0.0)
    parser.add_argument("--contact-friction", type=float, default=2.0)
    parser.add_argument("--box-collision", choices=("outer-sdf", "bounding-box"), default="outer-sdf")
    parser.add_argument("--robot-collisions", choices=("official", "sensor-only"), default="sensor-only")
    parser.add_argument("--robot-state-trace", type=Path, required=True)
    parser.add_argument("--anatomical-patch-asset", type=Path, required=True)
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path(__file__).with_name("run_newton_sugar_g1_carrybox_tactile.py"),
    )
    args = parser.parse_args()
    if not (1 <= args.chunk_size <= 50):
        raise ValueError("chunk-size must be in [1, 50] for the server EGL runtime")
    if args.frame_start < 0 or args.frame_stop <= args.frame_start:
        raise ValueError("The selected source-frame interval is empty.")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    chunks_root = output_root / ".render_chunks"
    if chunks_root.exists() and not args.merge_only:
        shutil.rmtree(chunks_root)
    chunks_root.mkdir(exist_ok=True)

    chunk_dirs: list[Path] = []
    for index, start in enumerate(range(args.frame_start, args.frame_stop, args.chunk_size)):
        stop = min(start + args.chunk_size, args.frame_stop)
        chunk_dir = chunks_root / f"chunk_{index:03d}_{start:04d}_{stop:04d}"
        if args.merge_only:
            required = ("frames.json", "summary.json", "trace.npz", "newton_sugar_g1_carrybox_native_tactile.mp4")
            missing = [name for name in required if not (chunk_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete render chunk {chunk_dir}: missing {missing}")
            chunk_dirs.append(chunk_dir)
            continue
        command = [
            sys.executable,
            str(args.worker),
            "--output-root",
            str(chunk_dir),
            "--frame-start",
            str(args.frame_start),
            "--frame-stop",
            str(stop),
            "--render-frame-start",
            str(start),
            "--device",
            args.device,
            "--force-scale-n",
            str(args.force_scale_n),
            "--renderer-refresh-frames",
            "0",
            "--dynamic-box",
            "--solver",
            args.solver,
            "--physics-substeps",
            str(args.physics_substeps),
            "--solver-iterations",
            str(args.solver_iterations),
            "--render-stride",
            str(args.render_stride),
            "--vbd-contact-ke",
            str(args.vbd_contact_ke),
            "--vbd-contact-kd",
            str(args.vbd_contact_kd),
            "--contact-friction",
            str(args.contact_friction),
            "--box-collision",
            args.box_collision,
            "--robot-collisions",
            args.robot_collisions,
            "--robot-state-trace",
            str(args.robot_state_trace.resolve()),
            "--anatomical-patch-asset",
            str(args.anatomical_patch_asset.resolve()),
        ]
        print(f"render_chunk={index} source=[{start},{stop})", flush=True)
        subprocess.run(command, check=True)
        chunk_dirs.append(chunk_dir)

    frames: list[dict] = []
    summaries: list[dict] = []
    trace_parts: list[dict[str, np.ndarray]] = []
    for chunk_index, chunk_dir in enumerate(chunk_dirs):
        chunk_start = args.frame_start + chunk_index * args.chunk_size
        chunk_stop = min(chunk_start + args.chunk_size, args.frame_stop)
        chunk_rows = json.loads((chunk_dir / "frames.json").read_text(encoding="utf-8"))
        frames.extend(row for row in chunk_rows if chunk_start <= row["source_frame"] < chunk_stop)
        summaries.append(json.loads((chunk_dir / "summary.json").read_text(encoding="utf-8")))
        with np.load(chunk_dir / "trace.npz") as archive:
            start_index = chunk_start - args.frame_start
            trace_parts.append(
                {
                    key: (
                        np.asarray(archive[key])
                        if key in STATIC_KEYS
                        else np.asarray(archive[key])[start_index:]
                    )
                    for key in archive.files
                }
            )

    raw_width = max(part["raw_contact_index"].shape[1] for part in trace_parts)
    trace: dict[str, np.ndarray] = {}
    for key in trace_parts[0]:
        if key in STATIC_KEYS:
            trace[key] = trace_parts[0][key]
        elif key in RAW_KEYS:
            trace[key] = np.concatenate(
                [_pad(part[key], raw_width, RAW_KEYS[key]) for part in trace_parts], axis=0
            )
        else:
            trace[key] = np.concatenate([part[key] for part in trace_parts], axis=0)
    trace["tactile_sequence"] = np.arange(len(frames), dtype=np.int64)
    timestamps = trace["tactile_timestamp_s"]
    trace["tactile_dt_s"] = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.diff(timestamps).astype(np.float64)]
    )
    np.savez_compressed(output_root / "trace.npz", **trace)
    (output_root / "frames.json").write_text(json.dumps(frames, indent=2) + "\n", encoding="utf-8")

    summary = dict(summaries[0])
    summary["frames"] = len(frames)
    summary["video_frames"] = sum(item["video_frames"] for item in summaries)
    summary["source_frame_interval"] = [args.frame_start, args.frame_stop]
    force_active = (np.linalg.norm(trace["force_patch_n"], axis=-1) > 1.0e-8).any(axis=(-2, -1))
    summary["contact_frames_per_patch"] = force_active.sum(axis=0).tolist()
    summary["contact_frames_per_hand"] = force_active.reshape(
        len(force_active), 2, -1
    ).any(axis=2).sum(axis=0).tolist()
    summary["maximum_raw_samples_per_frame"] = max(
        item["maximum_raw_samples_per_frame"] for item in summaries
    )
    summary["maximum_force_conservation_residual_n"] = max(
        item["maximum_force_conservation_residual_n"] for item in summaries
    )
    summary["taxel_position_shape"] = list(trace["taxel_position_w_m"].shape)
    summary["taxel_orientation_shape"] = list(trace["taxel_orientation_w_xyzw"].shape)
    box_positions = trace["box_position_w_m"]
    summary["maximum_box_displacement_from_initial_m"] = float(
        np.linalg.norm(box_positions - box_positions[0], axis=1).max()
    )
    summary["maximum_box_lift_from_initial_m"] = float(
        np.max(box_positions[:, 2] - box_positions[0, 2])
    )
    summary["render_process_chunks"] = len(chunk_dirs)
    summary["render_chunk_size_frames"] = args.chunk_size
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    concat_file = chunks_root / "videos.txt"
    concat_file.write_text(
        "".join(
            f"file '{(chunk_dir / 'newton_sugar_g1_carrybox_native_tactile.mp4').resolve()}'\n"
            for chunk_dir in chunk_dirs
        ),
        encoding="utf-8",
    )
    video = output_root / "newton_sugar_g1_carrybox_native_tactile.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    shutil.rmtree(chunks_root)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
