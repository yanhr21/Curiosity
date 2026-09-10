#!/usr/bin/env python3
"""Materialize the full SUGAR schedule without file hashes or authorization gates."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .artifacts import write_json_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig


PATTERN_A = {1: 4, 2: 4, 3: 5, 4: 2}
PATTERN_B = {1: 3, 2: 3, 3: 2, 4: 5}
PROMPT_LATENT_FRAMES = 16
TOKENS_PER_LATENT_FRAME = 100


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_overfit_schedule_row(
    formal_rows: list[dict[str, Any]], rank: int
) -> dict[str, Any]:
    """Resolve the one frozen full-width overfit sample assigned to a rank."""

    if rank < 0 or rank >= 8:
        raise ValueError(f"overfit rank must be in 0..7, got {rank}")
    desired_task = "CarryBox" if rank < 4 else "KickBox"
    candidates = [
        row
        for row in formal_rows
        if int(row["rank"]) == rank
        and int(row["epoch"]) == 0
        and str(row["task"]) == desired_task
        and int(row["chunk_size"]) == 2
        and all(bool(value) for value in row["ifp_valid"])
    ]
    if not candidates:
        raise ValueError(f"rank {rank} has no valid frozen overfit sample")
    return candidates[0]


def expand_pattern(pattern: dict[int, int]) -> list[int]:
    chunks = [size for size, count in pattern.items() for _ in range(count)]
    if sum(chunks) != 35:
        raise AssertionError("trajectory pattern must partition 35 latent transitions")
    return chunks


def trajectory_chunks(
    row: dict[str, Any], epoch: int, ordinal: int, config: PaperZeroWAMConfig
) -> list[dict[str, Any]]:
    task = str(row["task"])
    source_motion_id = int(row["source_motion_id"])
    pattern_name = "A" if ordinal % 2 == 0 else "B"
    chunk_sizes = expand_pattern(PATTERN_A if pattern_name == "A" else PATTERN_B)
    task_offset = 0 if task == "CarryBox" else 1_000_003
    rng = random.Random(
        config.schedule_seed + epoch * 10_000_019 + task_offset + source_motion_id * 101
    )
    rng.shuffle(chunk_sizes)

    chunks: list[dict[str, Any]] = []
    latent_start = 0
    for trajectory_chunk_index, chunk_size in enumerate(chunk_sizes):
        latent_stop = latent_start + chunk_size
        ifp_starts = [
            latent_start + chunk_size * (1 + (head * config.ifp_stride))
            for head in range(config.ifp_heads)
        ]
        ifp_valid = [start + chunk_size <= 35 for start in ifp_starts]
        main_sequence_tokens = (
            (PROMPT_LATENT_FRAMES + latent_stop + 1) * TOKENS_PER_LATENT_FRAME
            + latent_stop * 20
        )
        ifp_tokens_per_head = (
            (latent_start + 1 + 2 * chunk_size) * TOKENS_PER_LATENT_FRAME
            + latent_start * 20
        )
        ifp_sequence_tokens = [
            ifp_tokens_per_head if valid else 0 for valid in ifp_valid
        ]
        # FSDP requires every rank to execute all four wrapped heads even when
        # a rank-local future target falls beyond the trajectory.  Keep
        # supervised and physically executed token counts distinct.
        ifp_executed_sequence_tokens = [
            ifp_tokens_per_head for _ in range(config.ifp_heads)
        ]
        chunks.append(
            {
                "epoch": epoch,
                "task": task,
                "source_motion_id": source_motion_id,
                "trajectory_pattern": pattern_name,
                "trajectory_chunk_index": trajectory_chunk_index,
                "latent_start": latent_start,
                "latent_stop": latent_stop,
                "chunk_size": chunk_size,
                "rgb_interval_start": latent_start * config.rgb_intervals_per_latent,
                "rgb_interval_stop": latent_stop * config.rgb_intervals_per_latent,
                "action_start": latent_start
                * config.rgb_intervals_per_latent
                * config.action_per_rgb_interval,
                "action_stop": latent_stop
                * config.rgb_intervals_per_latent
                * config.action_per_rgb_interval,
                "ifp_latent_starts": ifp_starts,
                "ifp_valid": ifp_valid,
                "main_sequence_tokens_per_pass": main_sequence_tokens,
                "main_pass_count": 2,
                "ifp_sequence_tokens": ifp_sequence_tokens,
                "ifp_executed_sequence_tokens": ifp_executed_sequence_tokens,
                "supervised_transformer_token_exposures": 2 * main_sequence_tokens
                + sum(ifp_sequence_tokens),
                "transformer_token_exposures": 2 * main_sequence_tokens
                + sum(ifp_executed_sequence_tokens),
                "trace_path": row["action_target"]["trace_path"],
                "environment_index": int(row["action_target"]["environment_index"]),
            }
        )
        latent_start = latent_stop
    if latent_start != 35:
        raise AssertionError("trajectory chunks do not cover the full causal clock")
    return chunks


def materialize(output_dir: Path, config: PaperZeroWAMConfig) -> dict[str, Any]:
    config.validate()
    manifest_rows = read_manifest(config.resolved(config.manifest))
    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[str(row["task"])].append(row)
    for task in grouped:
        grouped[task].sort(key=lambda row: int(row["source_motion_id"]))
    if {task: len(rows) for task, rows in grouped.items()} != {
        "CarryBox": 80,
        "KickBox": 80,
    }:
        raise ValueError("expected exactly 80 train identities per task")

    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "TRAIN_SCHEDULE.jsonl"
    epoch_summaries: list[dict[str, Any]] = []
    global_step = 0
    with schedule_path.open("w", encoding="utf-8") as stream:
        for epoch in range(config.epochs):
            task_chunks: dict[str, list[dict[str, Any]]] = {}
            for task in ("CarryBox", "KickBox"):
                chunks: list[dict[str, Any]] = []
                for ordinal, row in enumerate(grouped[task]):
                    chunks.extend(trajectory_chunks(row, epoch, ordinal, config))
                rng = random.Random(config.schedule_seed + epoch * 1_000_003 + (0 if task == "CarryBox" else 17))
                rng.shuffle(chunks)
                task_chunks[task] = chunks
            if any(len(rows) != 1120 for rows in task_chunks.values()):
                raise AssertionError("each task must contribute 1120 chunks per epoch")

            epoch_rows: list[dict[str, Any]] = []
            for epoch_step in range(config.optimizer_steps_per_epoch):
                batch = (
                    task_chunks["CarryBox"][epoch_step * 4 : epoch_step * 4 + 4]
                    + task_chunks["KickBox"][epoch_step * 4 : epoch_step * 4 + 4]
                )
                rng = random.Random(config.schedule_seed + epoch * 1_000_003 + epoch_step * 997)
                rng.shuffle(batch)
                for rank, row in enumerate(batch):
                    prompt_enabled = (
                        random.Random(
                            config.noise_seed + global_step * config.world_size + rank
                        ).random()
                        >= config.prompt_dropout_probability
                    )
                    row.update(
                        {
                            "global_step": global_step,
                            "epoch_step": epoch_step,
                            "rank": rank,
                            "global_packed_sample_batch": 8,
                            "gradient_accumulation_steps": 1,
                            "prompt_enabled": prompt_enabled,
                        }
                    )
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    epoch_rows.append(row)
                global_step += 1

            chunk_counts = Counter(int(row["chunk_size"]) for row in epoch_rows)
            task_counts = Counter(str(row["task"]) for row in epoch_rows)
            rgb_intervals = sum(
                int(row["rgb_interval_stop"]) - int(row["rgb_interval_start"])
                for row in epoch_rows
            )
            actions = sum(int(row["action_stop"]) - int(row["action_start"]) for row in epoch_rows)
            main_tokens = sum(
                2 * int(row["main_sequence_tokens_per_pass"])
                for row in epoch_rows
            )
            ifp_tokens = sum(
                sum(int(value) for value in row["ifp_sequence_tokens"])
                for row in epoch_rows
            )
            ifp_executed_tokens = sum(
                sum(int(value) for value in row["ifp_executed_sequence_tokens"])
                for row in epoch_rows
            )
            video_loss_elements = sum(
                int(row["chunk_size"]) * config.visual_channels * 20 * 20
                for row in epoch_rows
            )
            action_loss_elements = sum(
                (int(row["action_stop"]) - int(row["action_start"]))
                * config.action_dim
                for row in epoch_rows
            )
            ifp_loss_elements = [
                sum(
                    int(row["chunk_size"]) * config.visual_channels * 20 * 20
                    for row in epoch_rows
                    if bool(row["ifp_valid"][head_index])
                )
                for head_index in range(config.ifp_heads)
            ]
            if chunk_counts != Counter({1: 560, 2: 560, 3: 560, 4: 560}):
                raise AssertionError(f"chunk-size balance failed: {chunk_counts}")
            if task_counts != Counter({"CarryBox": 1120, "KickBox": 1120}):
                raise AssertionError(f"task balance failed: {task_counts}")
            if rgb_intervals != 22_400 or actions != 112_000:
                raise AssertionError("epoch raw-data coverage failed")
            prompt_enabled_samples = sum(bool(row["prompt_enabled"]) for row in epoch_rows)
            epoch_summaries.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": config.optimizer_steps_per_epoch,
                    "packed_samples": len(epoch_rows),
                    "chunk_size_counts": dict(sorted(chunk_counts.items())),
                    "task_counts": dict(sorted(task_counts.items())),
                    "latent_transition_exposures": 5_600,
                    "rgb_interval_exposures": rgb_intervals,
                    "action_exposures": actions,
                    "two_pass_main_transformer_tokens": main_tokens,
                    "ifp_supervised_transformer_tokens": ifp_tokens,
                    "ifp_executed_transformer_tokens": ifp_executed_tokens,
                    "supervised_transformer_token_exposures": main_tokens + ifp_tokens,
                    "total_transformer_token_exposures": main_tokens
                    + ifp_executed_tokens,
                    "video_loss_elements": video_loss_elements,
                    "action_loss_elements": action_loss_elements,
                    "ifp_loss_elements": ifp_loss_elements,
                    "prompt_enabled_samples": prompt_enabled_samples,
                    "prompt_dropped_samples": len(epoch_rows) - prompt_enabled_samples,
                }
            )

    try:
        schedule_label = str(schedule_path.relative_to(PROJECT_ROOT))
    except ValueError:
        schedule_label = str(schedule_path)
    materialized_rows = read_manifest(schedule_path)
    overfit_case_fields = (
        "rank",
        "task",
        "source_motion_id",
        "trace_path",
        "environment_index",
        "trajectory_pattern",
        "trajectory_chunk_index",
        "latent_start",
        "latent_stop",
        "chunk_size",
        "action_start",
        "action_stop",
        "ifp_latent_starts",
        "ifp_valid",
        "main_sequence_tokens_per_pass",
        "ifp_sequence_tokens",
        "ifp_executed_sequence_tokens",
        "supervised_transformer_token_exposures",
        "transformer_token_exposures",
    )
    overfit_cases = [
        {
            **{
                field: selected[field]
                for field in overfit_case_fields
            },
            "prompt_enabled": True,
        }
        for rank in range(config.world_size)
        for selected in (select_overfit_schedule_row(materialized_rows, rank),)
    ]
    if (
        len(overfit_cases) != 8
        or len(
            {
                (row["task"], int(row["source_motion_id"]))
                for row in overfit_cases
            }
        )
        != 8
        or len(
            {
                (row["trace_path"], int(row["environment_index"]))
                for row in overfit_cases
            }
        )
        != 8
    ):
        raise ValueError("overfit schedule cases must be eight distinct source/action rows")
    overfit_two_pass_tokens = sum(
        2 * int(row["main_sequence_tokens_per_pass"])
        for row in overfit_cases
    )
    overfit_supervised_ifp_tokens = sum(
        sum(int(value) for value in row["ifp_sequence_tokens"])
        for row in overfit_cases
    )
    overfit_executed_ifp_tokens = sum(
        sum(int(value) for value in row["ifp_executed_sequence_tokens"])
        for row in overfit_cases
    )
    visual_elements_per_latent = config.visual_channels * 20 * 20
    overfit_video_elements = sum(
        int(row["chunk_size"]) * visual_elements_per_latent
        for row in overfit_cases
    )
    overfit_ifp_elements = [
        sum(
            int(row["chunk_size"]) * visual_elements_per_latent
            for row in overfit_cases
            if bool(row["ifp_valid"][head_index])
        )
        for head_index in range(config.ifp_heads)
    ]
    overfit_local_elements = [
        {
            "video": int(row["chunk_size"]) * visual_elements_per_latent,
            "action": (
                int(row["action_stop"]) - int(row["action_start"])
            )
            * config.action_dim,
            "ifp": [
                int(row["chunk_size"]) * visual_elements_per_latent
                if bool(row["ifp_valid"][head_index])
                else 0
                for head_index in range(config.ifp_heads)
            ],
        }
        for row in overfit_cases
    ]
    overfit_rank_evidence = [
        {
            "rank": int(row["rank"]),
            "local_elements": local,
            "scales": {
                "video": (
                    config.world_size * local["video"] / overfit_video_elements
                ),
                "action": (
                    config.world_size
                    * local["action"]
                    / sum(value["action"] for value in overfit_local_elements)
                ),
                "ifp": [
                    config.world_size
                    * local["ifp"][head_index]
                    / overfit_ifp_elements[head_index]
                    if overfit_ifp_elements[head_index] > 0
                    else 0.0
                    for head_index in range(config.ifp_heads)
                ],
            },
        }
        for row, local in zip(overfit_cases, overfit_local_elements, strict=True)
    ]
    result = {
        "protocol": "paper_zero_wam_sugar_formal_schedule_v1",
        "schedule": schedule_label,
        "architecture": {
            "video_expert": "30 layers x 3072",
            "action_expert": "30 layers x 3072",
            "mot": "separate QKV/FFN/output, shared attention every layer",
            "ifp": "4 one-layer heads, stride 2, weights 0.5/0.25/0.15/0.15",
        },
        "batch": {
            "world_size": 8,
            "packed_samples_per_rank": 1,
            "global_packed_samples": 8,
            "gradient_accumulation_steps": 1,
        },
        "overfit": {
            "optimizer_steps": 32,
            "global_packed_samples": 8,
            "same_case_repeated_on_each_rank": True,
            "cases": overfit_cases,
            "per_update": {
                "latent_transition_exposures": sum(
                    int(row["chunk_size"]) for row in overfit_cases
                ),
                "rgb_interval_exposures": sum(
                    int(row["chunk_size"]) * config.rgb_intervals_per_latent
                    for row in overfit_cases
                ),
                "action_exposures": sum(
                    int(row["action_stop"]) - int(row["action_start"])
                    for row in overfit_cases
                ),
                "two_pass_main_transformer_tokens": overfit_two_pass_tokens,
                "ifp_supervised_transformer_tokens": overfit_supervised_ifp_tokens,
                "ifp_executed_transformer_tokens": overfit_executed_ifp_tokens,
                "supervised_transformer_token_exposures": (
                    overfit_two_pass_tokens + overfit_supervised_ifp_tokens
                ),
                "total_transformer_token_exposures": (
                    overfit_two_pass_tokens + overfit_executed_ifp_tokens
                ),
                "video_loss_elements": overfit_video_elements,
                "action_loss_elements": sum(
                    (int(row["action_stop"]) - int(row["action_start"]))
                    * config.action_dim
                    for row in overfit_cases
                ),
                "ifp_loss_elements": overfit_ifp_elements,
                "ifp_zero_target_heads": [
                    head_index
                    for head_index, value in enumerate(overfit_ifp_elements)
                    if value == 0
                ],
                "prompt_enabled_samples": 8,
                "prompt_dropped_samples": 0,
                "loss_reduction": config.loss_reduction,
                "loss_rank_evidence": overfit_rank_evidence,
            },
        },
        "totals": {
            "complete_epochs": config.epochs,
            "optimizer_steps": global_step,
            "packed_samples": config.epochs * 2_240,
            "latent_transition_exposures": config.epochs * 5_600,
            "rgb_interval_exposures": config.epochs * 22_400,
            "action_exposures": config.epochs * 112_000,
            "two_pass_main_transformer_tokens": sum(
                row["two_pass_main_transformer_tokens"] for row in epoch_summaries
            ),
            "ifp_supervised_transformer_tokens": sum(
                row["ifp_supervised_transformer_tokens"] for row in epoch_summaries
            ),
            "ifp_executed_transformer_tokens": sum(
                row["ifp_executed_transformer_tokens"] for row in epoch_summaries
            ),
            "supervised_transformer_token_exposures": sum(
                row["supervised_transformer_token_exposures"]
                for row in epoch_summaries
            ),
            "total_transformer_token_exposures": sum(
                row["total_transformer_token_exposures"] for row in epoch_summaries
            ),
            "video_loss_elements": sum(
                row["video_loss_elements"] for row in epoch_summaries
            ),
            "action_loss_elements": sum(
                row["action_loss_elements"] for row in epoch_summaries
            ),
            "ifp_loss_elements": [
                sum(row["ifp_loss_elements"][head] for row in epoch_summaries)
                for head in range(config.ifp_heads)
            ],
            "prompt_enabled_samples": sum(
                row["prompt_enabled_samples"] for row in epoch_summaries
            ),
            "prompt_dropped_samples": sum(
                row["prompt_dropped_samples"] for row in epoch_summaries
            ),
        },
        "epoch_summaries": epoch_summaries,
        "hash_checks": False,
    }
    write_json_atomic(output_dir / "SCHEDULE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/demo_following/paper_zero_wam_v1/schedule",
    )
    args = parser.parse_args()
    result = materialize(args.output_dir.resolve(), PaperZeroWAMConfig())
    print(json.dumps(result["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
