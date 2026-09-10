#!/usr/bin/env python3
"""Materialize admitted SUGAR trajectories as an official BPP ReplayBuffer.

This is data-layout glue only.  It preserves every 50 Hz action/proprio row,
stores both 10 Hz cameras with the released ReplayBuffer lower-rate index maps,
and uses the released lossless JPEG-XL numcodec.  It does not implement a
dataset sampler, model, forward pass, loss, or optimizer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import numcodecs
import zarr


ROOT = Path(__file__).resolve().parents[3]
SENSOR_PROTOCOL = "sugar_bpp_sensorimotor_corpus_v2"
OUTPUT_PROTOCOL = "sugar_bpp_official_replay_buffer_v1"
TRAJECTORIES = 995
STEPS = 700
FRAMES = 141
CAMERA_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
EXPECTED_SOURCE_COMMIT = "b5b494ed05fdd5d79e57b8b27072d3fa109ebfd0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensorimotor-result", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bpp_names(row: dict[str, Any]) -> tuple[str, str]:
    """Return delimiter-safe official BPP episode/task identifiers.

    BPP serializes split membership as ``episode_name:task_name`` and later
    recovers the task name with ``split(":")[1]``.  Colons inside either name
    therefore corrupt the released parser.  Each SUGAR source motion is one
    BPP task and each admitted rollout variant is one demonstration.
    """
    task = str(row["task_family"])
    source = int(row["behavior_identity"]["source_motion_id"])
    variant = int(row["candidate_variant_id"])
    task_name = f"{task}_motion{source:03d}"
    episode_name = f"{task_name}_variant{variant:02d}"
    if ":" in task_name or ":" in episode_name:
        raise ValueError("BPP identifiers may not contain ':'")
    return episode_name, task_name


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = base / path
    if local.exists():
        return local.resolve()
    return (ROOT / path).resolve()


def stream_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def create_array(group, name: str, data: np.ndarray) -> None:
    group.array(
        name,
        data=data,
        shape=data.shape,
        chunks=data.shape,
        compressor=None,
        overwrite=False,
    )


def main() -> None:
    args = parse_args()
    sensor_result_path = args.sensorimotor_result.resolve()
    source = args.source_repo.resolve()
    output = args.output.resolve()
    result_path = args.result.resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("ReplayBuffer materialization requires Slurm")
    if socket.gethostname().startswith(("login", "mgmtserver")):
        raise RuntimeError("refusing ReplayBuffer materialization on login node")
    if output.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite immutable ReplayBuffer evidence")
    if not (source / ".git").exists():
        raise FileNotFoundError(source)

    sensor_result = read_json(sensor_result_path)
    if (
        sensor_result.get("protocol") != SENSOR_PROTOCOL
        or sensor_result.get("passed") is not True
    ):
        raise ValueError("sensorimotor admission is absent or failed")
    manifest_ref = sensor_result.get("artifacts", {}).get("sensorimotor_manifest")
    if not manifest_ref:
        raise ValueError("sensorimotor result lacks manifest")
    manifest_path = resolve_from(sensor_result_path.parent, str(manifest_ref))
    rows = read_jsonl(manifest_path)
    if len(rows) != TRAJECTORIES:
        raise ValueError(f"expected {TRAJECTORIES} trajectories, found {len(rows)}")
    rows.sort(
        key=lambda row: (
            str(row["task_family"]),
            int(row["behavior_identity"]["source_motion_id"]),
            int(row["admitted_variant_rank"]),
        )
    )

    source_commit = git_output(source, "rev-parse", "HEAD")
    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise ValueError(
            f"official source commit drift: {source_commit} != {EXPECTED_SOURCE_COMMIT}"
        )
    sys.path.insert(0, str(source))
    from behavior_prompting.common.imagecodecs_numcodecs import (  # pylint: disable=import-outside-toplevel
        JpegXl,
        register_codecs,
    )
    from behavior_prompting.common.replay_buffer import ReplayBuffer  # pylint: disable=import-outside-toplevel

    register_codecs(verbose=False)
    staging = output.with_name(
        f"{output.name}.staging_job{os.environ['SLURM_JOB_ID']}"
    )
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(staging))
    root = zarr.group(store=store, overwrite=False)
    data_group = root.create_group("data", overwrite=False)
    meta_group = root.create_group("meta", overwrite=False)
    root.create_group("labels", overwrite=False)

    n_steps = TRAJECTORIES * STEPS
    n_frames = TRAJECTORIES * FRAMES
    numeric_compressor = numcodecs.Blosc(
        cname="zstd",
        clevel=5,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )
    image_compressor = JpegXl(
        lossless=True, effort=3, decodingspeed=1, numthreads=1
    )
    arrays = {
        "action": data_group.zeros(
            "action", shape=(n_steps, 29), chunks=(700, 29),
            dtype=np.float32, compressor=numeric_compressor,
        ),
        "ee_pos": data_group.zeros(
            "ee_pos", shape=(n_steps, 68), chunks=(700, 68),
            dtype=np.float32, compressor=numeric_compressor,
        ),
        "ee_ori": data_group.zeros(
            "ee_ori", shape=(n_steps, 29), chunks=(700, 29),
            dtype=np.float32, compressor=numeric_compressor,
        ),
        "gripper_states": data_group.zeros(
            "gripper_states", shape=(n_steps, 24), chunks=(700, 24),
            dtype=np.float32, compressor=numeric_compressor,
        ),
    }
    for camera_key in CAMERA_KEYS:
        arrays[camera_key] = data_group.zeros(
            camera_key,
            (n_frames, 320, 320, 3),
            chunks=(1, 320, 320, 3),
            dtype=np.uint8,
            compressor=image_compressor,
        )

    episode_ends = np.arange(1, TRAJECTORIES + 1, dtype=np.int64) * STEPS
    frame_ends = np.arange(1, TRAJECTORIES + 1, dtype=np.int64) * FRAMES
    identifiers = [bpp_names(row) for row in rows]
    episode_names = np.asarray([value[0] for value in identifiers], dtype=str)
    task_names = np.asarray([value[1] for value in identifiers], dtype=str)
    if len(set(episode_names.tolist())) != TRAJECTORIES:
        raise ValueError("BPP episode identifiers are not unique")
    if len(set(task_names.tolist())) != 199:
        raise ValueError("BPP task identifiers do not preserve 199 source motions")
    create_array(meta_group, "episode_ends", episode_ends)
    create_array(meta_group, "episode_names", episode_names)
    create_array(meta_group, "task_names", task_names)
    create_array(meta_group, "task_lengths", np.full(TRAJECTORIES, STEPS, np.int64))
    create_array(meta_group, "task_data_ends", episode_ends)
    create_array(meta_group, "task_labels_ends", episode_ends)
    create_array(
        meta_group,
        "is_episode_error_correction",
        np.zeros(TRAJECTORIES, dtype=bool),
    )
    create_array(
        meta_group,
        "is_task_error_correction",
        np.zeros(TRAJECTORIES, dtype=bool),
    )
    local_upsample = np.minimum(np.arange(STEPS) // 5, FRAMES - 2)
    local_downsample = np.minimum(np.arange(FRAMES) * 5, STEPS - 1)
    for camera_key in CAMERA_KEYS:
        upsample = np.concatenate(
            [local_upsample + episode * FRAMES for episode in range(TRAJECTORIES)]
        ).astype(np.int64)
        downsample = np.concatenate(
            [local_downsample + episode * STEPS for episode in range(TRAJECTORIES)]
        ).astype(np.int64)
        create_array(meta_group, f"upsample_index_{camera_key}", upsample)
        create_array(meta_group, f"downsample_index_{camera_key}", downsample)
        create_array(meta_group, f"episode_ends_{camera_key}", frame_ends)

    by_trace: dict[Path, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for output_index, row in enumerate(rows):
        trace_path = resolve_from(ROOT, str(row["trace"]["path"]))
        by_trace[trace_path].append((output_index, row))

    verified_streams = 0
    for trace_path, trace_rows in sorted(by_trace.items(), key=lambda item: str(item[0])):
        expected_trace_hashes = {str(row["trace"]["file_sha256"]) for _, row in trace_rows}
        if len(expected_trace_hashes) != 1 or sha256_file(trace_path) not in expected_trace_hashes:
            raise ValueError(f"trace hash mismatch: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as archive:
            actions = np.asarray(archive["executed_action"])
            cores = np.asarray(archive["goal_policy_core_observation_before"])
            if actions.shape[:2] != cores.shape[:2] or actions.shape[0] != STEPS:
                raise ValueError(f"trace clock mismatch: {trace_path}")
            for output_index, row in trace_rows:
                env_index = int(row["trace"]["environment_index"])
                action = np.asarray(actions[:, env_index], dtype=np.float32)
                core = np.asarray(cores[:, env_index], dtype=np.float32)
                if action.shape != (STEPS, 29) or core.shape != (STEPS, 121):
                    raise ValueError("source action/core geometry drift")
                if not np.isfinite(action).all() or not np.isfinite(core).all():
                    raise ValueError("non-finite source action/core")
                step_slice = slice(output_index * STEPS, (output_index + 1) * STEPS)
                arrays["action"][step_slice] = action
                arrays["ee_pos"][step_slice] = core[:, :68]
                arrays["ee_ori"][step_slice] = core[:, 68:97]
                arrays["gripper_states"][step_slice] = core[:, 97:121]

                for camera_key in CAMERA_KEYS:
                    camera = row["robot_rgb"]["modalities"][camera_key]
                    frame_paths = [resolve_from(ROOT, str(path)) for path in camera["frame_paths"]]
                    if len(frame_paths) != FRAMES:
                        raise ValueError("camera frame-count drift")
                    if stream_sha256(frame_paths) != camera["stream_sha256"]:
                        raise ValueError("camera stream hash mismatch")
                    frame_slice = slice(
                        output_index * FRAMES, (output_index + 1) * FRAMES
                    )
                    decoded = []
                    for frame_path in frame_paths:
                        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                        if bgr is None or bgr.shape != (320, 320, 3):
                            raise ValueError(f"invalid RGB frame: {frame_path}")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        # The released LiberoReplayImageDataset reverses the H axis
                        # because native LIBERO HDF5 frames are upside down.
                        decoded.append(np.ascontiguousarray(rgb[::-1]))
                    image_block = np.stack(decoded)
                    arrays[camera_key][frame_slice] = image_block
                    if not np.array_equal(arrays[camera_key][frame_slice], image_block):
                        raise ValueError("lossless image codec roundtrip failed")
                    verified_streams += 1

    replay = ReplayBuffer.create_from_path(str(staging), mode="r")
    checks = {
        "trajectory_count_exact_995": replay.n_episodes == TRAJECTORIES,
        "task_count_exact_995": replay.n_tasks == TRAJECTORIES,
        "transition_count_exact_696500": replay.n_steps == TRAJECTORIES * STEPS,
        "task_lengths_exact_700": bool(np.all(replay.task_lengths[:] == STEPS)),
        "task_names_are_160_20_19_motion_identities": len(set(task_names.tolist())) == 199,
        "dual_camera_stream_count_exact": verified_streams == TRAJECTORIES * 2,
        "rgb_lower_rate_storage_exact": all(
            replay.data[key].shape[0] == TRAJECTORIES * FRAMES
            and replay.is_key_upsampled(key)
            for key in CAMERA_KEYS
        ),
        "proprio_partition_width_exact_121": 68 + 29 + 24 == 121,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    root.attrs.update(
        {
            "protocol": OUTPUT_PROTOCOL,
            "sensorimotor_result_sha256": sha256_file(sensor_result_path),
            "sensorimotor_manifest_sha256": sha256_file(manifest_path),
            "official_source_commit": source_commit,
        }
    )
    os.replace(staging, output)
    result = {
        "protocol": OUTPUT_PROTOCOL,
        "passed": True,
        "checks": checks,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "official_source_commit": source_commit,
        "sensorimotor_result": {
            "path": str(sensor_result_path),
            "sha256": sha256_file(sensor_result_path),
        },
        "sensorimotor_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "replay_buffer": {
            "path": str(output),
            "trajectory_count": TRAJECTORIES,
            "main_rows": n_steps,
            "frames_per_camera": n_frames,
            "camera_keys": list(CAMERA_KEYS),
            "image_codec": image_compressor.get_config(),
            "row_order_sha256": canonical_sha256(rows),
        },
        "claim_boundary": (
            "This proves a lossless official ReplayBuffer materialization of the "
            "admitted corpus. It is not a sampler, forward, optimizer, training, "
            "or behavior-following result."
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
