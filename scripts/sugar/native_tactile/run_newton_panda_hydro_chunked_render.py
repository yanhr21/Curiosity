#!/usr/bin/env python3
"""Render a continuous Panda hydro tactile interval in short EGL workers."""

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
    parser.add_argument("--frame-stop", type=int, default=420)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=("cube", "pen"), default="cube")
    parser.add_argument("--normal-scale-n", type=float, default=5.0)
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path(__file__).with_name("run_newton_panda_hydro_tactile.py"),
    )
    args = parser.parse_args()
    if not (1 <= args.chunk_size <= 50):
        raise ValueError("chunk-size must be in [1, 50] for the server EGL runtime")
    if args.frame_start < 0 or args.frame_stop <= args.frame_start:
        raise ValueError("The selected source-frame interval is empty.")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    chunks_root = output_root / ".render_chunks"
    if chunks_root.exists():
        shutil.rmtree(chunks_root)
    chunks_root.mkdir()

    chunk_dirs: list[Path] = []
    for index, start in enumerate(range(args.frame_start, args.frame_stop, args.chunk_size)):
        stop = min(start + args.chunk_size, args.frame_stop)
        chunk_dir = chunks_root / f"chunk_{index:03d}_{start:04d}_{stop:04d}"
        print(f"render_chunk={index} simulation=[0,{stop}) output=[{start},{stop})", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(args.worker),
                "--output-root",
                str(chunk_dir),
                "--num-frames",
                str(stop),
                "--frame-start",
                str(start),
                "--frame-stop",
                str(stop),
                "--device",
                args.device,
                "--scene",
                args.scene,
                "--normal-scale-n",
                str(args.normal_scale_n),
            ],
            check=True,
        )
        chunk_dirs.append(chunk_dir)

    summaries: list[dict] = []
    frames: list[dict] = []
    parts: list[dict[str, np.ndarray]] = []
    for chunk_dir in chunk_dirs:
        summaries.append(json.loads((chunk_dir / "summary.json").read_text(encoding="utf-8")))
        frames.extend(json.loads((chunk_dir / "frames.json").read_text(encoding="utf-8")))
        with np.load(chunk_dir / "trace.npz") as archive:
            parts.append({key: np.asarray(archive[key]) for key in archive.files})

    raw_width = max(part["raw_contact_index"].shape[1] for part in parts)
    trace: dict[str, np.ndarray] = {}
    for key in parts[0]:
        if key in STATIC_KEYS:
            trace[key] = parts[0][key]
        elif key in RAW_KEYS:
            trace[key] = np.concatenate([_pad(part[key], raw_width, RAW_KEYS[key]) for part in parts])
        else:
            trace[key] = np.concatenate([part[key] for part in parts])
    np.savez_compressed(output_root / "trace.npz", **trace)
    (output_root / "frames.json").write_text(json.dumps(frames, indent=2) + "\n", encoding="utf-8")

    summary = dict(summaries[0])
    summary["frames"] = len(frames)
    summary["source_frame_interval"] = [args.frame_start, args.frame_stop]
    force_active = (np.linalg.norm(trace["force_patch_n"], axis=-1) > 1.0e-8).any(axis=(-2, -1))
    summary["contact_frames_per_patch"] = force_active.sum(axis=0).tolist()
    initial_z = float(trace["object_position_w_m"][0, 2] - frames[0]["object_lift_m"])
    summary["maximum_object_lift_m"] = float(trace["object_position_w_m"][:, 2].max() - initial_z)
    summary["maximum_force_conservation_residual_n"] = max(
        item["maximum_force_conservation_residual_n"] for item in summaries
    )
    summary["taxel_position_shape"] = list(trace["taxel_position_w_m"].shape)
    summary["taxel_orientation_shape"] = list(trace["taxel_orientation_w_xyzw"].shape)
    summary["render_process_chunks"] = len(chunk_dirs)
    summary["render_chunk_size_frames"] = args.chunk_size
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    concat = chunks_root / "videos.txt"
    concat.write_text(
        "".join(
            f"file '{(chunk / 'newton_panda_hydro_native_tactile.mp4').resolve()}'\n" for chunk in chunk_dirs
        ),
        encoding="utf-8",
    )
    video = output_root / "newton_panda_hydro_native_tactile.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
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
