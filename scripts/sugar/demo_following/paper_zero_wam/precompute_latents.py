#!/usr/bin/env python3
"""Encode the complete 199-trajectory SUGAR corpus with the Wan-2.2 causal VAE."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig, repaired_overfit_config
from .prompt_coverage import prompt_frame_indices, prompt_coverage_record
from .data import (
    action_trace_geometry_summary,
    load_executed_actions,
    manifest_clock_is_exact,
)


def distributed_context() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world_size


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_video(paths: list[str]) -> torch.Tensor:
    frames: list[np.ndarray] = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if frame.shape != (320, 320, 3):
            raise ValueError(f"expected 320x320 RGB frame, got {frame.shape}: {path}")
        frames.append(frame)
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).float()
    return tensor.div_(127.5).sub_(1.0)


def cache_path(root: Path, row: dict[str, Any]) -> Path:
    return root / str(row["split"]) / str(row["task"]) / f'{int(row["source_motion_id"]):03d}.pt'


def build_action_statistics(rows: list[dict[str, Any]], output: Path) -> None:
    actions: list[np.ndarray] = []
    for row in rows:
        if row["split"] != "train":
            continue
        trace_path = row["action_target"]["trace_path"]
        environment_index = int(row["action_target"]["environment_index"])
        value = load_executed_actions(trace_path, environment_index)
        actions.append(value)
    merged = np.concatenate(actions, axis=0)
    lower = np.quantile(merged, 0.01, axis=0)
    upper = np.quantile(merged, 0.99, axis=0)
    if np.any(upper - lower <= 1.0e-8):
        raise ValueError("action quantile interval collapsed")
    payload = {
        "protocol": "paper_zero_wam_sugar_action_quantiles_v2",
        "normalization": "clip(2*(x-q01)/(q99-q01+1e-6)-1,-1.5,1.5)",
        "inverse": "((x+1)/2)*(q99-q01+1e-6)+q01",
        "train_rows": int(merged.shape[0]),
        "action_dim": int(merged.shape[1]),
        "q01": lower.tolist(),
        "q99": upper.tolist(),
        "hash_checks": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--repair-prompt-coverage", action="store_true")
    args = parser.parse_args()
    config = repaired_overfit_config() if args.repair_prompt_coverage else PaperZeroWAMConfig()
    rank, world_size = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("Wan VAE materialization requires a CUDA compute node")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    rows = read_manifest(config.resolved(config.manifest))
    if not manifest_clock_is_exact(rows):
        raise ValueError(
            "manifest must contain the exact 199-source 64/141-RGB causal clock"
        )
    cache_root = (args.cache_root or config.resolved(config.latent_cache)).resolve()
    original_cache = config.resolved(PaperZeroWAMConfig().latent_cache)
    if args.repair_prompt_coverage and cache_root == original_cache.resolve():
        raise ValueError("repaired prompt cache may not overwrite original latents")
    cache_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] | None = None
    if rank == 0:
        build_action_statistics(rows, cache_root / "ACTION_QUANTILES.json")
    if dist.is_initialized():
        dist.barrier()

    source_root = config.resolved(config.wan_source)
    package = types.ModuleType("wan")
    package.__path__ = [str(source_root / "wan")]
    package.__package__ = "wan"
    sys.modules["wan"] = package
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(source_root / "wan" / "modules")]
    modules_package.__package__ = "wan.modules"
    sys.modules["wan.modules"] = modules_package
    from wan.modules.vae2_2 import Wan2_2_VAE

    vae = Wan2_2_VAE(
        vae_pth=str(config.resolved(config.wan_checkpoint) / "Wan2.2_VAE.pth"),
        dtype=torch.bfloat16,
        device=str(device),
    )
    processed = 0
    for row_index in range(rank, len(rows), world_size):
        row = rows[row_index]
        output = cache_path(cache_root, row)
        if output.exists():
            try:
                payload = torch.load(output, map_location="cpu", weights_only=True)
            except (EOFError, KeyError, OSError, RuntimeError):
                payload = None
            if payload is not None and (
                (not args.repair_prompt_coverage or payload.get("prompt_coverage") == prompt_coverage_record())
                and
                tuple(payload.get("prompt_latents", torch.empty(0)).shape)
                == (48, 16, 20, 20)
                and tuple(payload.get("robot_latents", torch.empty(0)).shape)
                == (48, 36, 20, 20)
                and tuple(payload.get("reversed_prompt_latents", torch.empty(0)).shape)
                == (48, 16, 20, 20)
                and payload.get("split") == row["split"]
                and payload.get("task") == row["task"]
                and int(payload.get("source_motion_id", -1))
                == int(row["source_motion_id"])
                and all(
                    bool(torch.isfinite(payload[key]).all().item())
                    for key in (
                        "prompt_latents",
                        "reversed_prompt_latents",
                        "robot_latents",
                    )
                )
                and not torch.equal(
                    payload["prompt_latents"], payload["reversed_prompt_latents"]
                )
            ):
                continue
        paths = row["prompt"]["frame_paths"]
        if args.repair_prompt_coverage:
            paths = [paths[index] for index in prompt_frame_indices()]
        prompt = load_video(paths).to(device)
        reversed_prompt = prompt.flip(1)
        if args.repair_prompt_coverage:
            from .data import load_validated_latent_payload
            original = load_validated_latent_payload(cache_path(original_cache, row),
                split=row["split"], task=row["task"], source_motion_id=row["source_motion_id"])
            robot_latents = original["robot_latents"]
        else:
            robot = load_video(row["robot_target"]["frame_paths"]).to(device)
        with torch.inference_mode():
            prompt_latents = vae.encode([prompt])[0].to(torch.bfloat16).cpu()
            reversed_prompt_latents = vae.encode([reversed_prompt])[0].to(torch.bfloat16).cpu()
            if not args.repair_prompt_coverage:
                robot_latents = vae.encode([robot])[0].to(torch.bfloat16).cpu()
        if tuple(prompt_latents.shape) != (48, 16, 20, 20):
            raise ValueError(f"prompt latent geometry mismatch: {tuple(prompt_latents.shape)}")
        if tuple(robot_latents.shape) != (48, 36, 20, 20):
            raise ValueError(f"robot latent geometry mismatch: {tuple(robot_latents.shape)}")
        if tuple(reversed_prompt_latents.shape) != (48, 16, 20, 20):
            raise ValueError(
                f"reversed prompt latent geometry mismatch: {tuple(reversed_prompt_latents.shape)}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        torch.save(
            {
                "prompt_latents": prompt_latents,
                "reversed_prompt_latents": reversed_prompt_latents,
                "robot_latents": robot_latents,
                "split": row["split"],
                "task": row["task"],
                "source_motion_id": int(row["source_motion_id"]),
                **({"prompt_coverage": prompt_coverage_record()} if args.repair_prompt_coverage else {}),
            },
            temporary,
        )
        temporary.replace(output)
        processed += 1
        emit_json_best_effort(
            {
                "rank": rank,
                "row_index": row_index,
                "processed": processed,
                "output": str(output),
                "prompt_shape": list(prompt_latents.shape),
                "robot_shape": list(robot_latents.shape),
            }
        )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        cached = sorted(cache_root.glob("*/*/*.pt"))
        action_trace_geometry = action_trace_geometry_summary(rows)
        action_trace_geometry_exact = action_trace_geometry["passed"] is True
        expected_by_path = {
            cache_path(cache_root, row).resolve(): row for row in rows
        }
        actual_paths = {path.resolve() for path in cached}
        exact_manifest_path_set = (
            len(rows) == len(expected_by_path) == 199
            and actual_paths == set(expected_by_path)
        )
        counts: dict[str, int] = {}
        reversed_nonidentical = 0
        finite_trajectory_count = 0
        metadata_identity_exact_count = 0
        for path in cached:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if args.repair_prompt_coverage:
                if payload.get("prompt_coverage") != prompt_coverage_record():
                    raise ValueError(f"repaired prompt coverage missing: {path}")
                original = torch.load(original_cache / path.relative_to(cache_root),
                                      map_location="cpu", weights_only=True)
                if not torch.equal(payload["robot_latents"], original["robot_latents"]):
                    raise ValueError(f"unchanged robot target drift: {path}")
            if tuple(payload["prompt_latents"].shape) != (48, 16, 20, 20):
                raise ValueError(f"cached prompt geometry mismatch: {path}")
            if tuple(payload["reversed_prompt_latents"].shape) != (48, 16, 20, 20):
                raise ValueError(f"cached reversed-prompt geometry mismatch: {path}")
            if tuple(payload["robot_latents"].shape) != (48, 36, 20, 20):
                raise ValueError(f"cached robot geometry mismatch: {path}")
            label = f'{payload["split"]}/{payload["task"]}'
            counts[label] = counts.get(label, 0) + 1
            expected = expected_by_path.get(path.resolve())
            metadata_identity_exact_count += int(
                expected is not None
                and payload.get("split") == expected["split"]
                and payload.get("task") == expected["task"]
                and int(payload.get("source_motion_id", -1))
                == int(expected["source_motion_id"])
            )
            reversed_nonidentical += int(
                not torch.equal(payload["prompt_latents"], payload["reversed_prompt_latents"])
            )
            finite_trajectory_count += int(
                all(
                    bool(torch.isfinite(payload[key]).all().item())
                    for key in (
                        "prompt_latents",
                        "reversed_prompt_latents",
                        "robot_latents",
                    )
                )
            )
        action_stats = json.loads(
            (cache_root / "ACTION_QUANTILES.json").read_text(encoding="utf-8")
        )
        prompt_frame_paths = {
            str(value) for row in rows for value in row["prompt"]["frame_paths"]
        }
        robot_frame_paths = {
            str(value)
            for row in rows
            for value in row["robot_target"]["frame_paths"]
        }
        lower = np.asarray(action_stats.get("q01", []), dtype=np.float64)
        upper = np.asarray(action_stats.get("q99", []), dtype=np.float64)
        action_statistics_valid = bool(
            action_stats.get("protocol")
            == "paper_zero_wam_sugar_action_quantiles_v2"
            and int(action_stats.get("train_rows", -1)) == 112000
            and int(action_stats.get("action_dim", -1)) == 29
            and lower.shape == upper.shape == (29,)
            and np.isfinite(lower).all()
            and np.isfinite(upper).all()
            and np.all(upper > lower)
        )
        result = {
            "protocol": "paper_zero_wam_wan22_latent_materialization_v1",
            "passed": len(cached) == 199
            and action_trace_geometry_exact
            and exact_manifest_path_set
            and metadata_identity_exact_count == 199
            and reversed_nonidentical == 199
            and finite_trajectory_count == 199
            and action_statistics_valid,
            "trajectory_count": len(cached),
            "manifest_clock_exact": manifest_clock_is_exact(rows),
            "prompt_frame_path_count": len(prompt_frame_paths),
            "robot_frame_path_count": len(robot_frame_paths),
            "prompt_robot_frame_path_overlap_count": len(
                prompt_frame_paths & robot_frame_paths
            ),
            "action_trace_geometry_exact": action_trace_geometry_exact,
            "action_trace_file_count": action_trace_geometry["trace_file_count"],
            "action_trace_environment_pair_count": action_trace_geometry[
                "trace_environment_pair_count"
            ],
            "action_trace_shape_environment_counts": action_trace_geometry[
                "shape_environment_counts"
            ],
            "exact_manifest_path_set": exact_manifest_path_set,
            "metadata_identity_exact_count": metadata_identity_exact_count,
            "counts": counts,
            "prompt_shape": [48, 16, 20, 20],
            "robot_shape": [48, 36, 20, 20],
            "reversed_prompt_nonidentical_count": reversed_nonidentical,
            "finite_prompt_reversed_robot_trajectory_count": finite_trajectory_count,
            "all_latent_tensors_finite": finite_trajectory_count == 199,
            "action_statistics_valid": action_statistics_valid,
            "action_training_rows": int(action_stats.get("train_rows", -1)),
            "hash_checks": False,
            **({"prompt_coverage": prompt_coverage_record(),
                "prompt_coverage_exact_count": len(cached),
                "robot_latents_bitwise_unchanged_count": len(cached)}
               if args.repair_prompt_coverage else {}),
        }
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        assert result is not None
        write_json_atomic(cache_root / "LATENT_RESULT.json", result)
        if not result["passed"]:
            raise SystemExit("latent materialization geometry failed")


if __name__ == "__main__":
    main()
