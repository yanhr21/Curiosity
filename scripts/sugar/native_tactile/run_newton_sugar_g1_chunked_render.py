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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=660)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-scale-n", type=float, default=25.0)
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
    if chunks_root.exists():
        shutil.rmtree(chunks_root)
    chunks_root.mkdir()

    chunk_dirs: list[Path] = []
    for index, start in enumerate(range(args.frame_start, args.frame_stop, args.chunk_size)):
        stop = min(start + args.chunk_size, args.frame_stop)
        chunk_dir = chunks_root / f"chunk_{index:03d}_{start:04d}_{stop:04d}"
        command = [
            sys.executable,
            str(args.worker),
            "--output-root",
            str(chunk_dir),
            "--frame-start",
            str(start),
            "--frame-stop",
            str(stop),
            "--device",
            args.device,
            "--force-scale-n",
            str(args.force_scale_n),
            "--renderer-refresh-frames",
            "0",
        ]
        print(f"render_chunk={index} source=[{start},{stop})", flush=True)
        subprocess.run(command, check=True)
        chunk_dirs.append(chunk_dir)

    frames: list[dict] = []
    summaries: list[dict] = []
    trace_parts: list[dict[str, np.ndarray]] = []
    for chunk_dir in chunk_dirs:
        frames.extend(json.loads((chunk_dir / "frames.json").read_text(encoding="utf-8")))
        summaries.append(json.loads((chunk_dir / "summary.json").read_text(encoding="utf-8")))
        with np.load(chunk_dir / "trace.npz") as archive:
            trace_parts.append({key: np.asarray(archive[key]) for key in archive.files})

    trace: dict[str, np.ndarray] = {}
    for key in trace_parts[0]:
        if key == "patch_names":
            trace[key] = trace_parts[0][key]
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
    summary["source_frame_interval"] = [args.frame_start, args.frame_stop]
    summary["contact_frames_per_hand"] = np.sum(
        np.asarray([item["contact_frames_per_hand"] for item in summaries], dtype=np.int64), axis=0
    ).tolist()
    summary["maximum_raw_samples_per_frame"] = max(
        item["maximum_raw_samples_per_frame"] for item in summaries
    )
    summary["maximum_force_conservation_residual_n"] = max(
        item["maximum_force_conservation_residual_n"] for item in summaries
    )
    box_positions = trace["box_position_w_m"]
    summary["maximum_box_displacement_from_initial_m"] = float(
        np.linalg.norm(box_positions - box_positions[0], axis=1).max()
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
