#!/usr/bin/env python3
"""Summarize full Wan latent materialization using shapes and tensor equality only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PaperZeroWAMConfig
from .data import action_trace_geometry_summary, manifest_clock_is_exact
from .results import refresh_results_document_best_effort


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    config = PaperZeroWAMConfig()
    root = (args.cache_root or config.resolved(config.latent_cache)).resolve()
    cached = sorted(root.glob("*/*/*.pt"))
    manifest = read_manifest(config.resolved(config.manifest))
    manifest_clock_exact = manifest_clock_is_exact(manifest)
    action_trace_geometry = action_trace_geometry_summary(manifest)
    action_trace_geometry_exact = action_trace_geometry["passed"] is True
    expected_by_path = {
        (root / str(row["split"]) / str(row["task"]) / f'{int(row["source_motion_id"]):03d}.pt').resolve(): row
        for row in manifest
    }
    actual_paths = {path.resolve() for path in cached}
    exact_manifest_path_set = (
        len(manifest) == len(expected_by_path) == 199
        and actual_paths == set(expected_by_path)
    )
    counts: dict[str, int] = {}
    reversed_nonidentical = 0
    finite_trajectory_count = 0
    metadata_identity_exact_count = 0
    for path in cached:
        payload = torch.load(path, map_location="cpu", weights_only=True)
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
    expected = {
        "train/CarryBox": 80,
        "train/KickBox": 80,
        "validation/CarryBox": 10,
        "validation/KickBox": 10,
        "test/CarryBox": 10,
        "test/KickBox": 9,
    }
    action_stats = json.loads(
        (root / "ACTION_QUANTILES.json").read_text(encoding="utf-8")
    )
    prompt_frame_paths = {
        str(value) for row in manifest for value in row["prompt"]["frame_paths"]
    }
    robot_frame_paths = {
        str(value)
        for row in manifest
        for value in row["robot_target"]["frame_paths"]
    }
    lower = [float(value) for value in action_stats.get("q01", [])]
    upper = [float(value) for value in action_stats.get("q99", [])]
    action_statistics_valid = (
        action_stats.get("protocol") == "paper_zero_wam_sugar_action_quantiles_v2"
        and int(action_stats.get("train_rows", -1)) == 112000
        and int(action_stats.get("action_dim", -1)) == 29
        and len(lower) == len(upper) == 29
        and all(math.isfinite(left) and math.isfinite(right) and right > left
                for left, right in zip(lower, upper, strict=True))
    )
    result = {
        "protocol": "paper_zero_wam_wan22_latent_materialization_v1",
        "passed": len(cached) == 199
        and manifest_clock_exact
        and action_trace_geometry_exact
        and exact_manifest_path_set
        and metadata_identity_exact_count == 199
        and counts == expected
        and reversed_nonidentical == 199
        and finite_trajectory_count == 199
        and action_statistics_valid,
        "trajectory_count": len(cached),
        "manifest_clock_exact": manifest_clock_exact,
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
    }
    write_json_atomic(root / "LATENT_RESULT.json", result)
    refresh_results_document_best_effort()
    emit_json_best_effort(result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
