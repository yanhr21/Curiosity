#!/usr/bin/env python3
"""Exercise the production formal-training trace admission without constructing a model."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .train import (
    checkpoint_presence_is_complete,
    expected_overfit_step_evidence,
    expected_overfit_trace_samples,
    formal_training_decision,
    loss_scales_from_element_counts,
    overfit_execution_decision,
    publish_formal_progress,
    retained_trace_prefix,
    validated_checkpoint_step,
)


SAMPLE_FIELDS = (
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
    "ifp_sequence_tokens",
    "ifp_executed_sequence_tokens",
    "supervised_transformer_token_exposures",
    "transformer_token_exposures",
    "prompt_enabled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/demo_following/paper_zero_wam_v1/schedule/TRAIN_SCHEDULE.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/demo_following/paper_zero_wam_v1/schedule/"
        "TRAINING_TRACE_CONTRACT_AUDIT.json",
    )
    return parser.parse_args()


def write_trace(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def make_records(
    schedule: list[dict[str, Any]], config: PaperZeroWAMConfig
) -> list[dict[str, Any]]:
    rows_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in schedule:
        rows_by_step[int(row["global_step"])].append(row)

    records: list[dict[str, Any]] = []
    for step in sorted(rows_by_step):
        rows = sorted(rows_by_step[step], key=lambda row: int(row["rank"]))
        epoch = int(rows[0]["epoch"])
        # Deterministic finite values make both tasks improve from epoch 0 to 14.
        scale = 1.0 - 0.02 * epoch
        task_losses = {}
        for task in ("CarryBox", "KickBox"):
            task_rows = [row for row in rows if row["task"] == task]
            video_value = scale * (1.0 if task == "CarryBox" else 1.1)
            action_value = scale * (0.8 if task == "CarryBox" else 0.9)
            ifp_head_value = scale * (0.6 if task == "CarryBox" else 0.7)
            task_ifp_elements = [
                sum(
                    int(row["chunk_size"]) * config.visual_channels * 20 * 20
                    for row in task_rows
                    if bool(row["ifp_valid"][head_index])
                )
                for head_index in range(config.ifp_heads)
            ]
            task_losses[task] = {
                "count": 4.0,
                "video_loss": video_value,
                "video_loss_elements": sum(
                    int(row["chunk_size"]) * config.visual_channels * 20 * 20
                    for row in task_rows
                ),
                "action_loss": action_value,
                "action_loss_elements": sum(
                    (int(row["action_stop"]) - int(row["action_start"]))
                    * config.action_dim
                    for row in task_rows
                ),
                "ifp_loss": sum(
                    float(weight) * ifp_head_value
                    for weight, elements in zip(
                        config.ifp_weights, task_ifp_elements, strict=True
                    )
                    if elements > 0
                ),
                "ifp_head_losses": [ifp_head_value] * config.ifp_heads,
                "ifp_loss_elements": task_ifp_elements,
            }
            task_losses[task]["video_loss_sum"] = (
                video_value * task_losses[task]["video_loss_elements"]
            )
            task_losses[task]["action_loss_sum"] = (
                action_value * task_losses[task]["action_loss_elements"]
            )
            task_losses[task]["ifp_loss_sums"] = [
                ifp_head_value * elements for elements in task_ifp_elements
            ]
        prompt_enabled = sum(bool(row["prompt_enabled"]) for row in rows)
        two_pass_tokens = sum(
            2 * int(row["main_sequence_tokens_per_pass"]) for row in rows
        )
        ifp_tokens = sum(
            sum(int(value) for value in row["ifp_sequence_tokens"]) for row in rows
        )
        ifp_executed_tokens = sum(
            sum(int(value) for value in row["ifp_executed_sequence_tokens"])
            for row in rows
        )
        video_loss_elements = sum(
            int(row["chunk_size"]) * config.visual_channels * 20 * 20
            for row in rows
        )
        action_loss_elements = sum(
            (int(row["action_stop"]) - int(row["action_start"]))
            * config.action_dim
            for row in rows
        )
        ifp_loss_elements = [
            sum(
                int(row["chunk_size"]) * config.visual_channels * 20 * 20
                for row in rows
                if bool(row["ifp_valid"][head_index])
            )
            for head_index in range(config.ifp_heads)
        ]
        loss_rank_evidence = []
        for row in rows:
            local_video = int(row["chunk_size"]) * config.visual_channels * 20 * 20
            local_action = (
                int(row["action_stop"]) - int(row["action_start"])
            ) * config.action_dim
            local_ifp = [
                local_video if bool(row["ifp_valid"][head_index]) else 0
                for head_index in range(config.ifp_heads)
            ]
            loss_rank_evidence.append(
                {
                    "rank": int(row["rank"]),
                    "local_elements": {
                        "video": local_video,
                        "action": local_action,
                        "ifp": local_ifp,
                    },
                    "scales": {
                        "video": config.world_size * local_video / video_loss_elements,
                        "action": config.world_size * local_action / action_loss_elements,
                        "ifp": [
                            config.world_size * local / global_
                            if global_ > 0
                            else 0.0
                            for local, global_ in zip(
                                local_ifp, ifp_loss_elements, strict=True
                            )
                        ],
                    },
                }
            )
        records.append(
            {
                "mode": "formal",
                "optimizer_step": step,
                "resumed_from_step": 0,
                "epoch": epoch,
                "epoch_step": int(rows[0]["epoch_step"]),
                "learning_rate": 1.0e-4,
                "world_size": 8,
                "packed_samples_per_rank": 1,
                "global_packed_sample_batch": 8,
                "gradient_accumulation_steps": 1,
                "main_pass_count_per_sample": 2,
                "latent_transition_exposures": sum(int(row["chunk_size"]) for row in rows),
                "rgb_interval_exposures": sum(4 * int(row["chunk_size"]) for row in rows),
                "action_exposures": sum(20 * int(row["chunk_size"]) for row in rows),
                "two_pass_main_transformer_tokens": two_pass_tokens,
                "ifp_supervised_transformer_tokens": ifp_tokens,
                "ifp_executed_transformer_tokens": ifp_executed_tokens,
                "supervised_transformer_token_exposures": two_pass_tokens + ifp_tokens,
                "total_transformer_token_exposures": two_pass_tokens
                + ifp_executed_tokens,
                "video_loss_elements": video_loss_elements,
                "action_loss_elements": action_loss_elements,
                "ifp_loss_elements": ifp_loss_elements,
                "ifp_zero_target_heads": [
                    index for index, value in enumerate(ifp_loss_elements) if value == 0
                ],
                "loss_reduction": config.loss_reduction,
                "loss_rank_evidence": loss_rank_evidence,
                "prompt_enabled_samples": prompt_enabled,
                "prompt_dropped_samples": 8 - prompt_enabled,
                "losses": {
                    "loss": 2.4 * scale,
                    "video_loss": 1.05 * scale,
                    "action_loss": 0.85 * scale,
                    "ifp_loss": 0.65 * scale,
                    "ifp_active_heads": 3.0,
                },
                "task_losses": task_losses,
                "samples": [
                    {field: row[field] for field in SAMPLE_FIELDS} for row in rows
                ],
                "gradient_norms": {"video": 1.0, "action": 1.0, "ifp": 1.0},
                "parameter_update_norms": {
                    "video": 1.0,
                    "action": 1.0,
                    "ifp": 1.0,
                },
                "all_trainable_parameters_finite": True,
                "optimizer_applied": True,
                "amp_scaler_skipped": False,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    config = PaperZeroWAMConfig()
    config.validate()
    with args.schedule.open("r", encoding="utf-8") as stream:
        schedule = [json.loads(line) for line in stream if line.strip()]
    records = make_records(schedule, config)

    checkpoint_absent = not checkpoint_presence_is_complete(0, 0, False)
    checkpoint_complete = checkpoint_presence_is_complete(1, 1, True)
    partial_checkpoint_rejected = False
    dangling_checkpoint_marker_rejected = False
    try:
        checkpoint_presence_is_complete(0, 1, True)
    except ValueError:
        partial_checkpoint_rejected = True
    try:
        checkpoint_presence_is_complete(0, 0, True)
    except ValueError:
        dangling_checkpoint_marker_rejected = True
    checkpoint_state = {
        "protocol": "paper_zero_wam_formal_checkpoint_v1",
        "mode": "formal",
        "completed_optimizer_steps": 280,
        "architecture_parameter_count": config.expected_parameter_count,
        "world_size": config.world_size,
        "global_packed_sample_batch": config.world_size * config.samples_per_rank,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "epochs": config.epochs,
        "optimizer_steps_per_epoch": config.optimizer_steps_per_epoch,
        "formal_optimizer_steps": config.optimizer_steps,
        "optimizer": {
            "name": "AdamW",
            "peak_learning_rate": config.peak_learning_rate,
            "minimum_learning_rate": config.minimum_learning_rate,
            "warmup_steps": config.warmup_steps,
            "betas": [config.adam_beta1, config.adam_beta2],
            "epsilon": config.adam_epsilon,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
        },
        "schedule_seed": config.schedule_seed,
        "noise_seed": config.noise_seed,
        "loss_reduction": config.loss_reduction,
        "precision": config.precision,
        "hash_checks": False,
    }
    checkpoint_semantics_passed = (
        validated_checkpoint_step(checkpoint_state, "formal", config) == 280
    )
    wrong_checkpoint_batch_rejected = False
    mutated_checkpoint_state = copy.deepcopy(checkpoint_state)
    mutated_checkpoint_state["global_packed_sample_batch"] = 7
    try:
        validated_checkpoint_step(mutated_checkpoint_state, "formal", config)
    except ValueError:
        wrong_checkpoint_batch_rejected = True

    with tempfile.TemporaryDirectory(prefix="paper_zero_wam_trace_audit_") as temporary:
        trace = Path(temporary) / "TRAIN_TRACE.jsonl"
        write_trace(trace, records)
        clean = formal_training_decision(trace, args.schedule, config)
        synchronized_wrong_schedule = copy.deepcopy(schedule)
        synchronized_wrong_environment = int(
            synchronized_wrong_schedule[0]["environment_index"]
        ) + 1
        synchronized_wrong_schedule[0][
            "environment_index"
        ] = synchronized_wrong_environment
        synchronized_wrong_records = copy.deepcopy(records)
        synchronized_wrong_records[0]["samples"][0][
            "environment_index"
        ] = synchronized_wrong_environment
        synchronized_schedule_path = Path(temporary) / "WRONG_SCHEDULE.jsonl"
        write_trace(synchronized_schedule_path, synchronized_wrong_schedule)
        write_trace(trace, synchronized_wrong_records)
        synchronized_wrong_action_shard = formal_training_decision(
            trace, synchronized_schedule_path, config
        )
        write_trace(trace, records)
        progress_root = Path(temporary) / "formal"
        publish_formal_progress(
            trace,
            progress_root,
            config.optimizer_steps_per_epoch,
            123.0,
            config,
            refresh_document=False,
        )
        progress = json.loads(
            (progress_root / "FORMAL_PROGRESS.json").read_text(encoding="utf-8")
        )
        progress_prefix = records[: config.optimizer_steps_per_epoch]
        progress_passed = (
            progress.get("protocol") == "paper_zero_wam_formal_progress_v1"
            and int(progress.get("complete_epochs", -1)) == 1
            and int(progress.get("optimizer_steps_completed", -1))
            == config.optimizer_steps_per_epoch
            and progress.get("batch", {}).get("global_packed_samples") == 8
            and progress.get("cumulative_exposures")
            == {
                "packed_samples": config.optimizer_steps_per_epoch * config.world_size,
                "latent_transitions": sum(
                    int(record["latent_transition_exposures"])
                    for record in progress_prefix
                ),
                "rgb_intervals": sum(
                    int(record["rgb_interval_exposures"])
                    for record in progress_prefix
                ),
                "actions": sum(
                    int(record["action_exposures"]) for record in progress_prefix
                ),
            }
            and progress.get("scientific_decision_included") is False
            and progress.get("optimizer_updates_added_by_reporting") == 0
        )

        overfit_records = copy.deepcopy(records[:32])
        frozen_overfit_samples = expected_overfit_trace_samples(args.schedule, config)
        frozen_overfit_evidence = expected_overfit_step_evidence(
            frozen_overfit_samples, config
        )
        for step, record in enumerate(overfit_records):
            record["mode"] = "overfit"
            record["optimizer_step"] = step
            record["samples"] = copy.deepcopy(frozen_overfit_samples)
            record.update(copy.deepcopy(frozen_overfit_evidence))
        write_trace(trace, overfit_records)
        clean_overfit = overfit_execution_decision(trace, args.schedule, 32, config)

        missing_overfit_row = overfit_records.pop()
        write_trace(trace, overfit_records)
        truncated_overfit = overfit_execution_decision(
            trace, args.schedule, 32, config
        )
        overfit_records.append(missing_overfit_row)

        original_overfit_batch = overfit_records[0]["global_packed_sample_batch"]
        overfit_records[0]["global_packed_sample_batch"] = 7
        write_trace(trace, overfit_records)
        wrong_overfit_batch = overfit_execution_decision(
            trace, args.schedule, 32, config
        )
        overfit_records[0]["global_packed_sample_batch"] = original_overfit_batch

        original_overfit_source = overfit_records[1]["samples"][0][
            "source_motion_id"
        ]
        overfit_records[1]["samples"][0]["source_motion_id"] = (
            int(original_overfit_source) + 1
        )
        write_trace(trace, overfit_records)
        drifting_overfit_sample = overfit_execution_decision(
            trace, args.schedule, 32, config
        )
        overfit_records[1]["samples"][0][
            "source_motion_id"
        ] = original_overfit_source

        original_overfit_action_exposures = overfit_records[2][
            "action_exposures"
        ]
        overfit_records[2]["action_exposures"] = (
            int(original_overfit_action_exposures) - 1
        )
        write_trace(trace, overfit_records)
        wrong_overfit_action_exposure = overfit_execution_decision(
            trace, args.schedule, 32, config
        )
        overfit_records[2]["action_exposures"] = original_overfit_action_exposures

        write_trace(trace, records[:32])
        with trace.open("a", encoding="utf-8") as stream:
            stream.write('{"optimizer_step":32,"interrupted"')
        recovered_prefix = retained_trace_prefix(trace, 32)

        write_trace(trace, records[:31])
        with trace.open("a", encoding="utf-8") as stream:
            stream.write('{"optimizer_step":31,"interrupted"')
        committed_prefix_corruption_rejected = False
        try:
            retained_trace_prefix(trace, 32)
        except ValueError:
            committed_prefix_corruption_rejected = True

        original_source = records[0]["samples"][0]["source_motion_id"]
        records[0]["samples"][0]["source_motion_id"] = int(original_source) + 1
        write_trace(trace, records)
        wrong_source = formal_training_decision(trace, args.schedule, config)
        records[0]["samples"][0]["source_motion_id"] = original_source

        original_environment_index = records[0]["samples"][0]["environment_index"]
        records[0]["samples"][0]["environment_index"] = (
            int(original_environment_index) + 1
        )
        write_trace(trace, records)
        wrong_action_shard = formal_training_decision(trace, args.schedule, config)
        records[0]["samples"][0]["environment_index"] = original_environment_index

        original_action_stop = records[0]["samples"][0]["action_stop"]
        records[0]["samples"][0]["action_stop"] = int(original_action_stop) + 1
        write_trace(trace, records)
        wrong_boundary = formal_training_decision(trace, args.schedule, config)
        records[0]["samples"][0]["action_stop"] = original_action_stop

        original_ifp_update = records[0]["parameter_update_norms"].pop("ifp")
        write_trace(trace, records)
        missing_branch = formal_training_decision(trace, args.schedule, config)
        records[0]["parameter_update_norms"]["ifp"] = original_ifp_update

        original_ifp_executed_tokens = records[0][
            "ifp_executed_transformer_tokens"
        ]
        records[0]["ifp_executed_transformer_tokens"] = (
            int(original_ifp_executed_tokens) + 1
        )
        write_trace(trace, records)
        wrong_executed_tokens = formal_training_decision(trace, args.schedule, config)
        records[0]["ifp_executed_transformer_tokens"] = original_ifp_executed_tokens

        original_video_loss_elements = records[0]["video_loss_elements"]
        records[0]["video_loss_elements"] = int(original_video_loss_elements) + 1
        write_trace(trace, records)
        wrong_loss_elements = formal_training_decision(trace, args.schedule, config)
        records[0]["video_loss_elements"] = original_video_loss_elements

        original_task_video_elements = records[0]["task_losses"]["CarryBox"][
            "video_loss_elements"
        ]
        records[0]["task_losses"]["CarryBox"]["video_loss_elements"] = (
            int(original_task_video_elements) + 1
        )
        write_trace(trace, records)
        wrong_task_loss_elements = formal_training_decision(trace, args.schedule, config)
        records[0]["task_losses"]["CarryBox"][
            "video_loss_elements"
        ] = original_task_video_elements

        original_rank_scale = records[0]["loss_rank_evidence"][0]["scales"]["video"]
        records[0]["loss_rank_evidence"][0]["scales"]["video"] = (
            float(original_rank_scale) + 0.125
        )
        write_trace(trace, records)
        wrong_rank_scale = formal_training_decision(trace, args.schedule, config)
        records[0]["loss_rank_evidence"][0]["scales"]["video"] = original_rank_scale

        records[0]["all_trainable_parameters_finite"] = False
        write_trace(trace, records)
        nonfinite_parameter_state = formal_training_decision(
            trace, args.schedule, config
        )
        records[0]["all_trainable_parameters_finite"] = True

    zero_ifp_steps = [
        record["optimizer_step"]
        for record in records
        if 3 in record["ifp_zero_target_heads"]
    ]
    zero_head_scale = loss_scales_from_element_counts(
        [19_200, 580, 19_200, 19_200, 19_200, 0],
        [153_600, 4_640, 153_600, 96_000, 57_600, 0],
        config.world_size,
    )[-1]

    passed = (
        clean["passed"] is True
        and synchronized_wrong_action_shard["passed"] is False
        and synchronized_wrong_action_shard["checks"][
            "schedule_action_sources_match_manifest"
        ]
        is False
        and progress_passed
        and checkpoint_absent
        and checkpoint_complete
        and partial_checkpoint_rejected
        and dangling_checkpoint_marker_rejected
        and checkpoint_semantics_passed
        and wrong_checkpoint_batch_rejected
        and clean_overfit["passed"] is True
        and truncated_overfit["passed"] is False
        and truncated_overfit["checks"]["exact_32_contiguous_optimizer_rows"] is False
        and wrong_overfit_batch["passed"] is False
        and wrong_overfit_batch["checks"]["full_width_batch_eight_every_update"] is False
        and len(recovered_prefix) == 32
        and [json.loads(line)["optimizer_step"] for line in recovered_prefix]
        == list(range(32))
        and committed_prefix_corruption_rejected
        and wrong_source["passed"] is False
        and wrong_source["checks"]["all_33600_schedule_rows_consumed_exactly"] is False
        and wrong_action_shard["passed"] is False
        and wrong_action_shard["checks"]["all_33600_schedule_rows_consumed_exactly"] is False
        and wrong_boundary["passed"] is False
        and wrong_boundary["checks"]["all_33600_schedule_rows_consumed_exactly"] is False
        and missing_branch["passed"] is False
        and missing_branch["checks"]["all_branch_updates_finite_positive"] is False
        and wrong_executed_tokens["passed"] is False
        and wrong_executed_tokens["checks"]["transformer_token_exposures_exact"] is False
        and wrong_loss_elements["passed"] is False
        and wrong_loss_elements["checks"][
            "global_target_element_loss_reduction_exact"
        ]
        is False
        and wrong_task_loss_elements["passed"] is False
        and wrong_task_loss_elements["checks"]["task_loss_element_counts_exact"] is False
        and wrong_rank_scale["passed"] is False
        and wrong_rank_scale["checks"]["rank_local_loss_scale_evidence_exact"] is False
        and nonfinite_parameter_state["passed"] is False
        and nonfinite_parameter_state["checks"][
            "all_trainable_parameters_finite_after_every_update"
        ]
        is False
        and len(zero_ifp_steps) == 15
        and zero_head_scale == 0.0
    )
    result = {
        "protocol": "paper_zero_wam_training_trace_contract_audit_v1",
        "passed": passed,
        "model_constructed": False,
        "optimizer_updates": 0,
        "schedule_rows": len(schedule),
        "optimizer_trace_rows": len(records),
        "clean_trace_passed": clean["passed"],
        "formal_epoch_progress_reporting_passed": progress_passed,
        "clean_checks": clean["checks"],
        "clean_overfit_execution_trace_passed": clean_overfit["passed"],
        "clean_overfit_execution_checks": clean_overfit["checks"],
        "mutation_rejections": {
            "partial_distributed_checkpoint": partial_checkpoint_rejected,
            "dangling_checkpoint_marker": dangling_checkpoint_marker_rejected,
            "wrong_checkpoint_semantic_batch": wrong_checkpoint_batch_rejected,
            "truncated_overfit_trace": not truncated_overfit["passed"],
            "wrong_overfit_global_batch": not wrong_overfit_batch["passed"],
            "overfit_source_phase_drift": not drifting_overfit_sample["passed"],
            "wrong_overfit_action_exposure": not wrong_overfit_action_exposure[
                "passed"
            ],
            "interrupted_tail_after_committed_prefix_is_truncated": (
                len(recovered_prefix) == 32
            ),
            "corruption_inside_committed_prefix": committed_prefix_corruption_rejected,
            "wrong_source_motion_id": not wrong_source["passed"],
            "wrong_action_trace_environment_shard": not wrong_action_shard["passed"],
            "schedule_and_trace_synchronized_wrong_action_shard": not synchronized_wrong_action_shard[
                "passed"
            ],
            "wrong_action_boundary": not wrong_boundary["passed"],
            "missing_ifp_update_branch": not missing_branch["passed"],
            "wrong_ifp_executed_token_count": not wrong_executed_tokens["passed"],
            "wrong_global_loss_element_count": not wrong_loss_elements["passed"],
            "wrong_task_loss_element_count": not wrong_task_loss_elements["passed"],
            "wrong_rank_local_loss_scale": not wrong_rank_scale["passed"],
            "nonfinite_trainable_parameter_state": not nonfinite_parameter_state[
                "passed"
            ],
        },
        "global_zero_ifp_head_3_steps": zero_ifp_steps,
        "global_zero_ifp_head_scale": zero_head_scale,
        "hash_checks": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("training trace contract audit failed")


if __name__ == "__main__":
    main()
