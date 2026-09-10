#!/usr/bin/env python3
"""Distributed overfit/formal trainer for the full paper Zero-WAM reconstruction."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
import resource
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .data import (
    ScheduledSamples,
    read_jsonl,
    schedule_action_sources_are_exact,
)
from .model import IFPHead, PaperMoTLayer, PaperZeroWAM
from .results import refresh_results_document_best_effort
from .schedule import select_overfit_schedule_row


TRACE_SAMPLE_FIELDS = (
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
    parser.add_argument("--mode", choices=("overfit", "formal"), required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overfit-steps", type=int, default=32)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise ValueError(f"paper Zero-WAM training requires 8 H200 GPUs, found {world_size}")
    torch.cuda.set_device(local_rank)
    # Full-shard epoch checkpoints write model and AdamW state on all ranks.
    # Shared-filesystem skew may legitimately exceed NCCL's short default
    # watchdog while a faster rank waits at the commit barrier; the Slurm wall
    # time and atomic eight-shard checkpoint protocol remain the outer bound.
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    device = torch.device("cuda", local_rank)
    if "H200" not in torch.cuda.get_device_name(device):
        raise RuntimeError(f"expected H200, found {torch.cuda.get_device_name(device)}")
    return rank, world_size, local_rank, device


def publish_runtime_milestone(
    output_dir: Path,
    *,
    mode: str,
    rank: int,
    world_size: int,
    device: torch.device,
    milestone: str,
    started_at: float,
    **evidence: Any,
) -> None:
    """Publish non-authoritative progress around the expensive real process.

    These per-rank files are diagnostic breadcrumbs only.  They are never read
    by an admission, recovery or scientific reducer, and an I/O failure here
    cannot stop or replay training.
    """

    try:
        payload = {
            "protocol": "paper_zero_wam_runtime_milestone_v1",
            "authoritative": False,
            "used_for_admission": False,
            "mode": mode,
            "rank": rank,
            "world_size": world_size,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "process_id": os.getpid(),
            "milestone": milestone,
            "elapsed_seconds": time.perf_counter() - started_at,
            "maximum_resident_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "hash_checks": False,
            **evidence,
        }
        write_json_atomic(
            output_dir / "runtime_milestones" / f"rank{rank:02d}.json",
            payload,
        )
    except Exception as error:
        emit_json_best_effort(
            {
                "warning": "runtime_milestone_write_failed",
                "mode": mode,
                "rank": rank,
                "milestone": milestone,
                "error_type": type(error).__name__,
                "error": str(error),
                "training_continues": True,
            },
            error=True,
        )


def reduce_scalar(value: torch.Tensor) -> float:
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return float((reduced / dist.get_world_size()).item())


def loss_scales_from_element_counts(
    local_counts: list[int], global_counts: list[int], world_size: int
) -> list[float]:
    """Return rank-local scales, including zero for a globally absent IFP target."""

    if len(local_counts) != 6 or len(global_counts) != 6:
        raise ValueError("loss element count geometry must be video/action plus four IFP heads")
    if world_size <= 0 or any(value < 0 for value in local_counts + global_counts):
        raise ValueError("loss element counts and world size must be nonnegative/positive")
    if any(local > global_ for local, global_ in zip(local_counts, global_counts, strict=True)):
        raise ValueError("local loss element count exceeds its global count")
    if min(global_counts[:2]) <= 0:
        raise ValueError("global video/action loss branches must have target elements")
    return [
        float(world_size * local / global_) if global_ > 0 else 0.0
        for local, global_ in zip(local_counts, global_counts, strict=True)
    ]


def install_global_loss_scales(
    batch: dict[str, Any], world_size: int
) -> dict[str, Any]:
    """Make FSDP's rank-average equal a true global target-element mean."""

    local = torch.tensor(
        [
            batch["video_target_latents"].numel(),
            batch["action_target"].numel(),
            *[
                target.numel() if bool(valid) else 0
                for target, valid in zip(
                    batch["ifp_target_latents"], batch["ifp_valid"], strict=True
                )
            ],
        ],
        dtype=torch.int64,
        device=batch["video_target_latents"].device,
    )
    global_counts = local.clone()
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    # A far-future IFP head can be invalid on all eight rank-local chunks at a
    # schedule boundary.  It must still execute on every rank to preserve FSDP
    # collective order, but its objective for that step is exactly zero.
    scale_values = loss_scales_from_element_counts(
        [int(value) for value in local.tolist()],
        [int(value) for value in global_counts.tolist()],
        world_size,
    )
    batch["loss_scales"] = {
        "video": scale_values[0],
        "action": scale_values[1],
        "ifp": scale_values[2:],
    }
    return {
        "local": {
            "video": int(local[0].item()),
            "action": int(local[1].item()),
            "ifp": [int(value) for value in local[2:].tolist()],
        },
        "scales": {
            "video": scale_values[0],
            "action": scale_values[1],
            "ifp": scale_values[2:],
        },
        "video": int(global_counts[0].item()),
        "action": int(global_counts[1].item()),
        "ifp": [int(value) for value in global_counts[2:].tolist()],
        "ifp_zero_target_heads": [
            index for index, value in enumerate(global_counts[2:].tolist()) if int(value) == 0
        ],
    }


def branch_name(name: str) -> str:
    if name.startswith("ifp_") or ".ifp_" in name:
        return "ifp"
    if name.startswith("action_") or ".action_" in name or ".action_block." in name:
        return "action"
    return "video"


def gradient_norms(model: FSDP) -> dict[str, float]:
    sums = {name: torch.zeros((), device=torch.cuda.current_device()) for name in ("video", "action", "ifp")}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        sums[branch_name(name)] += parameter.grad.detach().float().square().sum()
    values = torch.stack([sums["video"], sums["action"], sums["ifp"]])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "video": float(values[0].sqrt().item()),
        "action": float(values[1].sqrt().item()),
        "ifp": float(values[2].sqrt().item()),
    }


def clone_local_parameters(model: FSDP) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    return [
        (branch_name(name), parameter, parameter.detach().clone())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None and parameter.numel() > 0
    ]


def update_norms(before: list[tuple[str, torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    sums = {name: torch.zeros((), device=torch.cuda.current_device()) for name in ("video", "action", "ifp")}
    for branch, parameter, old in before:
        sums[branch] += (parameter.detach().float() - old.float()).square().sum()
    values = torch.stack([sums["video"], sums["action"], sums["ifp"]])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "video": float(values[0].sqrt().item()),
        "action": float(values[1].sqrt().item()),
        "ifp": float(values[2].sqrt().item()),
    }


def all_trainable_parameters_finite(model: FSDP) -> bool:
    """Check every local trainable shard, including parameters unused by ICL."""

    # Keep this scan asynchronous until the single collective; calling
    # ``.item()`` once per parameter would serialize hundreds of CUDA kernels
    # at every one of the 4,200 formal updates.
    global_finite = torch.ones(
        (), device=torch.cuda.current_device(), dtype=torch.int64
    )
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.numel() > 0:
            global_finite.mul_(
                torch.isfinite(parameter.detach()).all().to(dtype=torch.int64)
            )
    dist.all_reduce(global_finite, op=dist.ReduceOp.MIN)
    return bool(global_finite.item())


def learning_rate(
    step: int, total_steps: int, config: PaperZeroWAMConfig, mode: str
) -> float:
    if mode == "overfit":
        return config.peak_learning_rate
    if step < config.warmup_steps:
        return config.peak_learning_rate * float(step + 1) / float(config.warmup_steps)
    progress = float(step - config.warmup_steps) / float(max(1, total_steps - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.peak_learning_rate - config.minimum_learning_rate
    ) * cosine


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    rank: int,
    mode: str,
    config: PaperZeroWAMConfig,
) -> None:
    if mode != "formal":
        raise ValueError("only the resumable formal run may write a checkpoint")
    # Alternate between two complete distributed slots.  The public
    # ``latest_checkpoint`` link changes only after every rank has replaced
    # both of its files, so an interruption can never invalidate the previous
    # complete epoch checkpoint.
    slot = (step // config.optimizer_steps_per_epoch) % 2
    checkpoint_dir = output_dir / f"checkpoint_slot{slot}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    model_temporary = checkpoint_dir / f"model_rank{rank:02d}.tmp"
    model_target = checkpoint_dir / f"model_rank{rank:02d}.pt"
    optimizer_temporary = checkpoint_dir / f"optimizer_rank{rank:02d}.tmp"
    optimizer_target = checkpoint_dir / f"optimizer_rank{rank:02d}.pt"
    torch.save({"step": step, "model": model_state}, model_temporary)
    model_temporary.replace(model_target)
    torch.save({"step": step, "optimizer": optimizer_state}, optimizer_temporary)
    optimizer_temporary.replace(optimizer_target)
    dist.barrier()
    if rank == 0:
        state_temporary = checkpoint_dir / "STATE.tmp"
        state_temporary.write_text(
            json.dumps(
                {
                    "protocol": f"paper_zero_wam_{mode}_checkpoint_v1",
                    "mode": mode,
                    "completed_optimizer_steps": step,
                    "architecture_parameter_count": config.expected_parameter_count,
                    "world_size": config.world_size,
                    "global_packed_sample_batch": (
                        config.world_size * config.samples_per_rank
                    ),
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
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        state_temporary.replace(checkpoint_dir / "STATE.json")
        latest = output_dir / "latest_checkpoint"
        latest_temporary = output_dir / ".latest_checkpoint.tmp"
        latest_temporary.unlink(missing_ok=True)
        latest_temporary.symlink_to(checkpoint_dir.name, target_is_directory=True)
        latest_temporary.replace(latest)
    dist.barrier()


def checkpoint_presence_is_complete(
    minimum_present: int, maximum_present: int, marker_present: bool
) -> bool:
    """Classify an absent versus complete distributed checkpoint fail-closed."""

    if minimum_present not in (0, 1) or maximum_present not in (0, 1):
        raise ValueError("distributed checkpoint presence flags must be boolean")
    if minimum_present != maximum_present:
        raise ValueError("distributed checkpoint is present on only a subset of ranks")
    if minimum_present == 0:
        if marker_present:
            raise ValueError("checkpoint marker exists without all eight rank shards")
        return False
    if not marker_present:
        raise ValueError("all rank shards exist without a checkpoint marker")
    return True


def validated_checkpoint_step(
    state: dict[str, Any], mode: str, config: PaperZeroWAMConfig
) -> int:
    """Validate a checkpoint's explicit frozen-run semantics without digests."""

    if mode != "formal":
        raise ValueError(f"unsupported checkpoint mode: {mode}")
    expected = {
        "protocol": f"paper_zero_wam_{mode}_checkpoint_v1",
        "mode": mode,
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
    mismatches = {
        key: {"expected": value, "observed": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint semantic contract mismatch: {mismatches}")
    step = int(state.get("completed_optimizer_steps", -1))
    valid_step = (
        step > 0
        and step <= config.optimizer_steps
        and step % config.optimizer_steps_per_epoch == 0
    )
    if not valid_step:
        raise ValueError(f"invalid {mode} checkpoint boundary: {step}")
    return step


def load_checkpoint_if_present(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    rank: int,
    config: PaperZeroWAMConfig,
) -> int:
    checkpoint_dir = output_dir / "latest_checkpoint"
    state_path = checkpoint_dir / "STATE.json"
    model_path = checkpoint_dir / f"model_rank{rank:02d}.pt"
    optimizer_path = checkpoint_dir / f"optimizer_rank{rank:02d}.pt"
    presence = torch.tensor(
        [
            int(state_path.is_file() and model_path.is_file() and optimizer_path.is_file()),
            int(checkpoint_dir.exists() or checkpoint_dir.is_symlink()),
        ],
        device=torch.cuda.current_device(),
        dtype=torch.int64,
    )
    minimum_presence = presence.clone()
    maximum_presence = presence.clone()
    dist.all_reduce(minimum_presence, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum_presence, op=dist.ReduceOp.MAX)
    if int(minimum_presence[1].item()) != int(maximum_presence[1].item()):
        raise ValueError("checkpoint marker visibility differs across ranks")
    complete = checkpoint_presence_is_complete(
        int(minimum_presence[0].item()),
        int(maximum_presence[0].item()),
        bool(minimum_presence[1].item()),
    )
    if not complete:
        return 0
    model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
    optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(model_payload["model"], strict=True)
        optimizer_state = FSDP.optim_state_dict_to_load(
            model, optimizer, optimizer_payload["optimizer"]
        )
    optimizer.load_state_dict(optimizer_state)
    completed_steps = int(model_payload["step"])
    if completed_steps != int(optimizer_payload["step"]):
        raise ValueError(f"model/optimizer checkpoint step mismatch on rank {rank}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected = validated_checkpoint_step(state, "formal", config)
    if completed_steps != expected:
        raise ValueError(
            f"checkpoint step disagreement on rank {rank}: {completed_steps} vs {expected}"
        )
    dist.barrier()
    return completed_steps


def retained_trace_prefix(log_path: Path, completed_steps: int) -> list[str]:
    """Read exactly the committed optimizer prefix and discard an interrupted tail."""

    if completed_steps < 0:
        raise ValueError("completed checkpoint step cannot be negative")
    if completed_steps == 0:
        return []
    retained: list[str] = []
    with log_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if len(retained) == completed_steps:
                    break
                raise ValueError(
                    f"malformed training trace inside committed prefix at line {line_number}"
                ) from error
            optimizer_step = int(record.get("optimizer_step", -1))
            if optimizer_step >= completed_steps:
                break
            if optimizer_step != len(retained):
                raise ValueError(
                    "training trace committed prefix is not contiguous: "
                    f"expected step {len(retained)}, found {optimizer_step}"
                )
            retained.append(line)
    if len(retained) != completed_steps:
        raise ValueError(
            f"checkpoint expects {completed_steps} committed trace rows, found "
            f"{len(retained)}"
        )
    return retained


def publish_formal_progress(
    log_path: Path,
    output_dir: Path,
    completed_steps: int,
    elapsed_seconds: float,
    config: PaperZeroWAMConfig,
    *,
    refresh_document: bool = True,
) -> None:
    """Publish one observational epoch boundary without creating a decision gate."""

    lines = retained_trace_prefix(log_path, completed_steps)
    records = [json.loads(line) for line in lines]
    if completed_steps % config.optimizer_steps_per_epoch != 0:
        raise ValueError("formal progress can be published only at a complete epoch")
    if not records or int(records[-1]["optimizer_step"]) != completed_steps - 1:
        raise ValueError("formal progress trace does not end at the checkpoint boundary")
    progress = {
        "protocol": "paper_zero_wam_formal_progress_v1",
        "optimizer_steps_completed": completed_steps,
        "optimizer_steps_planned": config.optimizer_steps,
        "complete_epochs": completed_steps // config.optimizer_steps_per_epoch,
        "epochs_planned": config.epochs,
        "checkpoint_step": completed_steps,
        "batch": {
            "world_size": config.world_size,
            "packed_samples_per_rank": config.samples_per_rank,
            "global_packed_samples": config.world_size * config.samples_per_rank,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        },
        "cumulative_exposures": {
            "packed_samples": sum(
                int(record["global_packed_sample_batch"]) for record in records
            ),
            "latent_transitions": sum(
                int(record["latent_transition_exposures"]) for record in records
            ),
            "rgb_intervals": sum(
                int(record["rgb_interval_exposures"]) for record in records
            ),
            "actions": sum(int(record["action_exposures"]) for record in records),
        },
        "latest_losses": records[-1]["losses"],
        "latest_task_losses": {
            task: {
                name: records[-1]["task_losses"][task][name]
                for name in ("video_loss", "action_loss", "ifp_loss")
            }
            for task in ("CarryBox", "KickBox")
        },
        "current_process_elapsed_seconds": elapsed_seconds,
        "current_process_resume_from_step": int(records[-1]["resumed_from_step"]),
        "scientific_decision_included": False,
        "optimizer_updates_added_by_reporting": 0,
        "hash_checks": False,
    }
    write_json_atomic(output_dir / "FORMAL_PROGRESS.json", progress)
    if refresh_document:
        refresh_results_document_best_effort()


def task_loss_record(
    losses: dict[str, torch.Tensor],
    batch: dict[str, Any],
    task: str,
    config: PaperZeroWAMConfig,
) -> dict[str, dict[str, Any]]:
    task_index = 0 if task == "CarryBox" else 1
    # Columns are video numerator/denominator, action numerator/denominator,
    # four IFP numerator/denominator pairs, and packed-sample count.
    values = torch.zeros((2, 13), device=torch.cuda.current_device(), dtype=torch.float64)
    video_count = batch["video_target_latents"].numel()
    action_count = batch["action_target"].numel()
    values[task_index, 0] = losses["video_loss_unscaled"].detach().double() * video_count
    values[task_index, 1] = video_count
    values[task_index, 2] = losses["action_loss_unscaled"].detach().double() * action_count
    values[task_index, 3] = action_count
    for head_index, (target, valid) in enumerate(
        zip(batch["ifp_target_latents"], batch["ifp_valid"], strict=True)
    ):
        count = target.numel() if bool(valid) else 0
        values[task_index, 4 + 2 * head_index] = (
            losses[f"ifp_head_{head_index}_loss_unscaled"].detach().double() * count
        )
        values[task_index, 5 + 2 * head_index] = count
    values[task_index, 12] = 1.0
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(("CarryBox", "KickBox")):
        count = float(values[index, 12].item())
        if count != 4.0:
            raise ValueError(f"runtime global batch expected four {name} rows, found {count}")
        if float(values[index, 1].item()) <= 0.0 or float(values[index, 3].item()) <= 0.0:
            raise ValueError(f"runtime task {name} has no video/action loss elements")
        ifp_value = 0.0
        for head_index, weight in enumerate(config.ifp_weights):
            denominator = float(values[index, 5 + 2 * head_index].item())
            if denominator > 0.0:
                ifp_value += float(weight) * float(
                    values[index, 4 + 2 * head_index].item() / denominator
                )
        ifp_head_losses: list[float] = []
        ifp_elements: list[int] = []
        for head_index in range(config.ifp_heads):
            denominator = float(values[index, 5 + 2 * head_index].item())
            ifp_elements.append(int(denominator))
            ifp_head_losses.append(
                float(values[index, 4 + 2 * head_index].item() / denominator)
                if denominator > 0.0
                else 0.0
            )
        result[name] = {
            "count": count,
            "video_loss": float(values[index, 0].item() / values[index, 1].item()),
            "video_loss_sum": float(values[index, 0].item()),
            "video_loss_elements": int(values[index, 1].item()),
            "action_loss": float(values[index, 2].item() / values[index, 3].item()),
            "action_loss_sum": float(values[index, 2].item()),
            "action_loss_elements": int(values[index, 3].item()),
            "ifp_loss": ifp_value,
            "ifp_head_losses": ifp_head_losses,
            "ifp_loss_sums": [
                float(values[index, 4 + 2 * head_index].item())
                for head_index in range(config.ifp_heads)
            ],
            "ifp_loss_elements": ifp_elements,
        }
    return result


def overfit_prompt_gate(
    model: FSDP,
    samples: ScheduledSamples,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    interventions = ("wrong_task", "reversed", "same_task_alternate")
    conditions: dict[str, dict[str, float]] = {}
    task_conditions: dict[str, dict[str, dict[str, float]]] = {
        task: {} for task in ("CarryBox", "KickBox")
    }
    generated: dict[str, dict[str, torch.Tensor]] = {}
    loss_names = ("loss", "video_loss", "action_loss", "ifp_loss")
    # FSDP-created parameter views must support subsequent training.
    with torch.no_grad():
        for condition in ("matched", *interventions):
            batch = samples.conditioned_sample(0, device, condition)
            # Counterfactual payloads may hit different CPU-cache paths.  Seed
            # immediately before forward so data I/O can never perturb the
            # fixed-noise comparison.
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(batch)
            conditions[condition] = {
                name: reduce_scalar(losses[name])
                for name in loss_names
            }
            task_values = torch.zeros((2, len(loss_names) + 1), device=device)
            task_index = 0 if batch["meta"]["task"] == "CarryBox" else 1
            task_values[task_index, : len(loss_names)] = torch.stack(
                [losses[name].detach().float() for name in loss_names]
            )
            task_values[task_index, -1] = 1.0
            dist.all_reduce(task_values, op=dist.ReduceOp.SUM)
            for index, task in enumerate(("CarryBox", "KickBox")):
                count = float(task_values[index, -1].item())
                if count != 4.0:
                    raise ValueError(
                        f"overfit gate expected four {task} ranks, found {count}"
                    )
                task_conditions[task][condition] = {
                    name: float(task_values[index, loss_index].item() / count)
                    for loss_index, name in enumerate(loss_names)
                }
            # A one-step fixed-noise flow integration probes the deployed
            # video -> action bottleneck without shortening or replacing the
            # model.  The action pass itself isolates the human prefix.
            torch.manual_seed(seed + 1)
            torch.cuda.manual_seed_all(seed + 1)
            probe_batch = dict(batch)
            probe_batch.update(
                {
                    "inference": True,
                    "video_inference_steps": 1,
                    "action_inference_steps": 1,
                    "video_guidance_scale": 1.0,
                }
            )
            # Inputs are BF16 while FSDP retains FP32 master parameters.
            # Match the loss forward and deployed sampling autocast boundary;
            # inference_mode/no_grad alone does not cast convolution weights.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated[condition] = model(probe_batch)
    matched = conditions["matched"]
    directional = {
        condition: {
            name: conditions[condition][name] - matched[name]
            for name in ("loss", "video_loss", "action_loss", "ifp_loss")
        }
        for condition in interventions
    }
    task_directional = {
        task: {
            condition: {
                name: task_conditions[task][condition][name]
                - task_conditions[task]["matched"][name]
                for name in loss_names
            }
            for condition in interventions
        }
        for task in ("CarryBox", "KickBox")
    }
    task_index = 0 if batch["meta"]["task"] == "CarryBox" else 1
    task_connectivity: dict[str, dict[str, dict[str, float]]] = {
        task: {} for task in ("CarryBox", "KickBox")
    }
    for condition in interventions:
        values = torch.zeros((2, 3), device=device, dtype=torch.float64)
        values[task_index, 0] = (
            generated[condition]["predicted_video_latents"].double()
            - generated["matched"]["predicted_video_latents"].double()
        ).square().mean()
        values[task_index, 1] = (
            generated[condition]["predicted_normalized_actions"].double()
            - generated["matched"]["predicted_normalized_actions"].double()
        ).square().mean()
        values[task_index, 2] = 1.0
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        for index, task in enumerate(("CarryBox", "KickBox")):
            count = float(values[index, 2].item())
            if count != 4.0:
                raise ValueError(
                    f"connectivity gate expected four {task} ranks, found {count}"
                )
            task_connectivity[task][condition] = {
                "predicted_future_mse_vs_matched": float(values[index, 0].item() / count),
                "predicted_action_mse_vs_matched": float(values[index, 1].item() / count),
            }
    prompt_loss_passed = all(
        task_directional[task][condition][name] > 0.0
        for task in ("CarryBox", "KickBox")
        for condition in interventions
        for name in ("video_loss", "ifp_loss")
    )
    teacher_action_isolated = all(
        abs(task_directional[task][condition]["action_loss"])
        <= 1.0e-7 * max(1.0, abs(task_conditions[task]["matched"]["action_loss"]))
        for task in ("CarryBox", "KickBox")
        for condition in interventions
    )
    generated_path_active = all(
        task_connectivity[task][condition][name] > 1.0e-12
        for task in ("CarryBox", "KickBox")
        for condition in interventions
        for name in (
            "predicted_future_mse_vs_matched",
            "predicted_action_mse_vs_matched",
        )
    )
    passed = prompt_loss_passed and teacher_action_isolated and generated_path_active
    return {
        "passed": passed,
        "interventions": list(interventions),
        "condition_count": 1 + len(interventions),
        "loss_forward_count_per_rank": 1 + len(interventions),
        "one_step_connectivity_forward_count_per_rank": 1 + len(interventions),
        "model_forward_count_per_rank": 2 * (1 + len(interventions)),
        "optimizer_updates_added": 0,
        "prompt_loss_passed": prompt_loss_passed,
        "teacher_action_isolated": teacher_action_isolated,
        "generated_future_to_action_path_active": generated_path_active,
        "losses": conditions,
        "margins_vs_matched": directional,
        "task_losses": task_conditions,
        "task_margins_vs_matched": task_directional,
        "task_generated_connectivity": task_connectivity,
    }


def expected_overfit_trace_samples(
    schedule_path: Path, config: PaperZeroWAMConfig
) -> list[dict[str, Any]]:
    """Materialize the exact eight source/phase rows repeated by overfit."""

    formal_rows = read_jsonl(schedule_path)
    samples: list[dict[str, Any]] = []
    for rank in range(config.world_size):
        selected = select_overfit_schedule_row(formal_rows, rank)
        sample = {field: selected[field] for field in TRACE_SAMPLE_FIELDS}
        sample["prompt_enabled"] = True
        samples.append(sample)
    if (
        len(samples) != 8
        or len({(row["task"], int(row["source_motion_id"])) for row in samples}) != 8
        or len({(row["trace_path"], int(row["environment_index"])) for row in samples})
        != 8
    ):
        raise ValueError("overfit must contain eight distinct frozen source/action rows")
    return samples


def expected_overfit_step_evidence(
    samples: list[dict[str, Any]], config: PaperZeroWAMConfig
) -> dict[str, Any]:
    """Derive one update's data/loss geometry from the frozen eight cases."""

    visual_elements_per_latent = config.visual_channels * 20 * 20
    local_elements = [
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
        for row in samples
    ]
    global_video = sum(value["video"] for value in local_elements)
    global_action = sum(value["action"] for value in local_elements)
    global_ifp = [
        sum(value["ifp"][head_index] for value in local_elements)
        for head_index in range(config.ifp_heads)
    ]
    rank_evidence = [
        {
            "rank": int(row["rank"]),
            "local_elements": value,
            "scales": {
                "video": config.world_size * value["video"] / global_video,
                "action": config.world_size * value["action"] / global_action,
                "ifp": [
                    config.world_size * value["ifp"][head_index]
                    / global_ifp[head_index]
                    if global_ifp[head_index] > 0
                    else 0.0
                    for head_index in range(config.ifp_heads)
                ],
            },
        }
        for row, value in zip(samples, local_elements, strict=True)
    ]
    two_pass_tokens = sum(
        int(row["transformer_token_exposures"])
        - sum(int(value) for value in row["ifp_executed_sequence_tokens"])
        for row in samples
    )
    supervised_ifp_tokens = sum(
        sum(int(value) for value in row["ifp_sequence_tokens"])
        for row in samples
    )
    executed_ifp_tokens = sum(
        sum(int(value) for value in row["ifp_executed_sequence_tokens"])
        for row in samples
    )
    return {
        "latent_transition_exposures": sum(int(row["chunk_size"]) for row in samples),
        "rgb_interval_exposures": sum(
            int(row["chunk_size"]) * config.rgb_intervals_per_latent
            for row in samples
        ),
        "action_exposures": sum(
            int(row["action_stop"]) - int(row["action_start"])
            for row in samples
        ),
        "two_pass_main_transformer_tokens": two_pass_tokens,
        "ifp_supervised_transformer_tokens": supervised_ifp_tokens,
        "ifp_executed_transformer_tokens": executed_ifp_tokens,
        "supervised_transformer_token_exposures": (
            two_pass_tokens + supervised_ifp_tokens
        ),
        "total_transformer_token_exposures": two_pass_tokens + executed_ifp_tokens,
        "video_loss_elements": global_video,
        "action_loss_elements": global_action,
        "ifp_loss_elements": global_ifp,
        "ifp_zero_target_heads": [
            head_index
            for head_index, value in enumerate(global_ifp)
            if value == 0
        ],
        "prompt_enabled_samples": sum(bool(row["prompt_enabled"]) for row in samples),
        "prompt_dropped_samples": sum(
            not bool(row["prompt_enabled"]) for row in samples
        ),
        "loss_reduction": config.loss_reduction,
        "loss_rank_evidence": rank_evidence,
    }


def overfit_execution_decision(
    log_path: Path,
    schedule_path: Path,
    expected_steps: int,
    config: PaperZeroWAMConfig,
    *, execution_world_size: int = 8, accumulation_steps: int = 1,
) -> dict[str, Any]:
    """Prove that the sole full-width overfit applied every declared update."""

    if (execution_world_size, accumulation_steps) not in ((8, 1), (1, 8)):
        raise ValueError("execution must preserve eight samples per optimizer update")
    with log_path.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    expected_samples = expected_overfit_trace_samples(schedule_path, config)
    expected_step_evidence = expected_overfit_step_evidence(expected_samples, config)
    branch_names = {"video", "action", "ifp"}
    expected_loss_names = {
        "loss",
        "video_loss",
        "action_loss",
        "ifp_loss",
        "ifp_active_heads",
    }
    checks = {
        "exact_32_contiguous_optimizer_rows": (
            expected_steps == 32
            and len(records) == expected_steps
            and [int(record["optimizer_step"]) for record in records]
            == list(range(expected_steps))
        ),
        "full_width_batch_eight_every_update": all(
            record.get("mode") == "overfit"
            and int(record.get("world_size", -1)) == execution_world_size
            and int(record.get("packed_samples_per_rank", -1)) == 1
            and int(record.get("global_packed_sample_batch", -1))
            == config.world_size
            and int(record.get("gradient_accumulation_steps", -1)) == accumulation_steps
            and int(record.get("main_pass_count_per_sample", -1)) == 2
            for record in records
        ),
        "single_gpu_physical_microbatch_map_exact": execution_world_size != 1 or all(
            record.get("physical_execution_map") == [
                {"logical_schedule_slot": slot, "physical_rank": 0, "microbatch_index": slot}
                for slot in range(8)
            ] for record in records
        ),
        "eight_rank_four_carry_four_kick_samples_every_update": all(
            len(record.get("samples", [])) == config.world_size
            and [int(row["rank"]) for row in record["samples"]]
            == list(range(config.world_size))
            and sum(row["task"] == "CarryBox" for row in record["samples"]) == 4
            and sum(row["task"] == "KickBox" for row in record["samples"]) == 4
            for record in records
        ),
        "exact_frozen_source_phase_grid_every_update": all(
            record.get("samples") == expected_samples for record in records
        ),
        "exact_frozen_data_and_loss_geometry_every_update": all(
            all(record.get(name) == value for name, value in expected_step_evidence.items())
            for record in records
        ),
        "all_losses_finite": all(
            set(record.get("losses", {})) == expected_loss_names
            and all(
                math.isfinite(float(value))
                for value in record["losses"].values()
            )
            for record in records
        ),
        "all_three_branch_gradients_finite_positive": all(
            set(record.get("gradient_norms", {})) == branch_names
            and all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in record["gradient_norms"].values()
            )
            for record in records
        ),
        "all_three_branch_updates_finite_positive": all(
            set(record.get("parameter_update_norms", {})) == branch_names
            and all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in record["parameter_update_norms"].values()
            )
            for record in records
        ),
        "every_optimizer_application_real_and_finite": all(
            record.get("optimizer_applied") is True
            and record.get("amp_scaler_skipped") is False
            and record.get("all_trainable_parameters_finite") is True
            and math.isfinite(float(record.get("learning_rate", math.nan)))
            and float(record["learning_rate"]) > 0.0
            for record in records
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def formal_training_decision(
    log_path: Path, schedule_path: Path, config: PaperZeroWAMConfig,
    *, execution_world_size: int = 8, accumulation_steps: int = 1,
) -> dict[str, Any]:
    if (execution_world_size, accumulation_steps) not in ((8, 1), (1, 8)):
        raise ValueError("execution must preserve eight samples per optimizer update")
    with log_path.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    step_ids = [int(record["optimizer_step"]) for record in records]
    with schedule_path.open("r", encoding="utf-8") as stream:
        scheduled_rows = [json.loads(line) for line in stream if line.strip()]
    manifest_rows = read_jsonl(config.resolved(config.manifest))
    schedule_action_sources_exact = schedule_action_sources_are_exact(
        scheduled_rows, manifest_rows
    )
    expected_token_totals = {
        "two_pass_main_transformer_tokens": sum(
            2 * int(row["main_sequence_tokens_per_pass"])
            for row in scheduled_rows
        ),
        "ifp_supervised_transformer_tokens": sum(
            sum(int(value) for value in row["ifp_sequence_tokens"])
            for row in scheduled_rows
        ),
        "ifp_executed_transformer_tokens": sum(
            sum(int(value) for value in row["ifp_executed_sequence_tokens"])
            for row in scheduled_rows
        ),
        "supervised_transformer_token_exposures": sum(
            int(row["supervised_transformer_token_exposures"])
            for row in scheduled_rows
        ),
        "total_transformer_token_exposures": sum(
            int(row["transformer_token_exposures"])
            for row in scheduled_rows
        ),
    }
    observed_token_totals = {
        name: sum(int(record[name]) for record in records)
        for name in expected_token_totals
    }
    expected_prompt_enabled = sum(bool(row["prompt_enabled"]) for row in scheduled_rows)
    observed_prompt_enabled = sum(
        int(record["prompt_enabled_samples"]) for record in records
    )
    expected_samples_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scheduled_rows:
        expected_samples_by_step[int(row["global_step"])].append(
            {field: row[field] for field in TRACE_SAMPLE_FIELDS}
        )
    for rows in expected_samples_by_step.values():
        rows.sort(key=lambda row: int(row["rank"]))
    visual_elements_per_latent = (
        config.visual_channels * 20 * 20
    )
    expected_loss_elements_by_step = {
        step: {
            "video": sum(
                int(row["chunk_size"]) * visual_elements_per_latent for row in rows
            ),
            "action": sum(
                (int(row["action_stop"]) - int(row["action_start"]))
                * config.action_dim
                for row in rows
            ),
            "ifp": [
                sum(
                    int(row["chunk_size"]) * visual_elements_per_latent
                    for row in rows
                    if bool(row["ifp_valid"][head_index])
                )
                for head_index in range(config.ifp_heads)
            ],
        }
        for step, rows in expected_samples_by_step.items()
    }
    observed_samples_by_step = {
        int(record["optimizer_step"]): sorted(
            record.get("samples", []), key=lambda row: int(row["rank"])
        )
        for record in records
    }
    expected_branch_keys = {"video", "action", "ifp"}
    expected_loss_keys = {
        "loss",
        "video_loss",
        "action_loss",
        "ifp_loss",
        "ifp_active_heads",
    }
    expected_task_loss_keys = {
        "count",
        "video_loss",
        "video_loss_sum",
        "video_loss_elements",
        "action_loss",
        "action_loss_sum",
        "action_loss_elements",
        "ifp_loss",
        "ifp_head_losses",
        "ifp_loss_sums",
        "ifp_loss_elements",
    }
    task_epoch: dict[str, dict[int, dict[str, Any]]] = {
        task: defaultdict(
            lambda: {
                "video_numerator": 0.0,
                "video_denominator": 0,
                "action_numerator": 0.0,
                "action_denominator": 0,
                "ifp_numerators": [0.0] * config.ifp_heads,
                "ifp_denominators": [0] * config.ifp_heads,
            }
        )
        for task in ("CarryBox", "KickBox")
    }
    for record in records:
        epoch = int(record["epoch"])
        for task in ("CarryBox", "KickBox"):
            task_record = record["task_losses"][task]
            accumulator = task_epoch[task][epoch]
            video_elements = int(task_record["video_loss_elements"])
            action_elements = int(task_record["action_loss_elements"])
            accumulator["video_numerator"] += float(task_record["video_loss_sum"])
            accumulator["video_denominator"] += video_elements
            accumulator["action_numerator"] += float(task_record["action_loss_sum"])
            accumulator["action_denominator"] += action_elements
            for head_index in range(config.ifp_heads):
                elements = int(task_record["ifp_loss_elements"][head_index])
                accumulator["ifp_numerators"][head_index] += float(
                    task_record["ifp_loss_sums"][head_index]
                )
                accumulator["ifp_denominators"][head_index] += elements
    epoch_element_means: dict[str, dict[str, dict[str, float]]] = {}
    progress_checks: dict[str, dict[str, bool]] = {}
    for task in ("CarryBox", "KickBox"):
        def reduce_epoch(epoch: int) -> dict[str, float]:
            values = task_epoch[task][epoch]
            if values["video_denominator"] <= 0 or values["action_denominator"] <= 0:
                raise ValueError(f"task {task} epoch {epoch} has no video/action targets")
            if any(value <= 0 for value in values["ifp_denominators"]):
                raise ValueError(f"task {task} epoch {epoch} has an empty IFP branch")
            return {
                "video_loss": values["video_numerator"] / values["video_denominator"],
                "action_loss": values["action_numerator"] / values["action_denominator"],
                "ifp_loss": sum(
                    float(weight) * values["ifp_numerators"][head_index]
                    / values["ifp_denominators"][head_index]
                    for head_index, weight in enumerate(config.ifp_weights)
                ),
            }

        first = reduce_epoch(0)
        last = reduce_epoch(config.epochs - 1)
        epoch_element_means[task] = {"first_epoch": first, "last_epoch": last}
        progress_checks[task] = {
            "video_improved": last["video_loss"] < first["video_loss"],
            "action_improved": last["action_loss"] < first["action_loss"],
            "ifp_not_worse": last["ifp_loss"] <= first["ifp_loss"],
        }
    checks = {
        "schedule_action_sources_match_manifest": schedule_action_sources_exact,
        "optimizer_steps_exact": step_ids == list(range(config.optimizer_steps)),
        "scheduled_step_rank_task_grid_exact": (
            set(expected_samples_by_step) == set(range(config.optimizer_steps))
            and all(
                [int(row["rank"]) for row in rows] == list(range(config.world_size))
                and sum(row["task"] == "CarryBox" for row in rows) == 4
                and sum(row["task"] == "KickBox" for row in rows) == 4
                for rows in expected_samples_by_step.values()
            )
            and all(
                int(row["epoch"])
                == int(row["global_step"]) // config.optimizer_steps_per_epoch
                and int(row["epoch_step"])
                == int(row["global_step"]) % config.optimizer_steps_per_epoch
                and int(row["global_packed_sample_batch"]) == config.world_size
                and int(row["gradient_accumulation_steps"]) == 1
                and int(row["main_pass_count"]) == 2
                for row in scheduled_rows
            )
        ),
        "epoch_and_epoch_step_exact": all(
            int(record["epoch"]) == int(record["optimizer_step"]) // config.optimizer_steps_per_epoch
            and int(record["epoch_step"])
            == int(record["optimizer_step"]) % config.optimizer_steps_per_epoch
            for record in records
        ),
        "batch_semantics_exact": all(
            int(record["world_size"]) == execution_world_size
            and int(record["packed_samples_per_rank"]) == 1
            and int(record["global_packed_sample_batch"]) == 8
            and int(record["gradient_accumulation_steps"]) == accumulation_steps
            and all(
                float(record["task_losses"][task]["count"]) == 4.0
                for task in ("CarryBox", "KickBox")
            )
            for record in records
        ),
        "single_gpu_physical_microbatch_map_exact": execution_world_size != 1 or all(
            record.get("physical_execution_map") == [
                {"logical_schedule_slot": slot, "physical_rank": 0, "microbatch_index": slot}
                for slot in range(8)
            ] for record in records
        ),
        "all_33600_schedule_rows_consumed_exactly": (
            len(scheduled_rows) == 33_600
            and set(observed_samples_by_step) == set(expected_samples_by_step)
            and all(
                observed_samples_by_step[step] == expected_samples_by_step[step]
                for step in expected_samples_by_step
            )
        ),
        "all_updates_applied": all(record["optimizer_applied"] for record in records),
        "no_amp_skips": all(not record["amp_scaler_skipped"] for record in records),
        "positive_learning_rate": all(float(record["learning_rate"]) > 0 for record in records),
        "all_losses_finite": all(
            set(record["losses"]) == expected_loss_keys
            and all(math.isfinite(float(value)) for value in record["losses"].values())
            for record in records
        ),
        "all_task_losses_finite_and_exact": all(
            set(record["task_losses"]) == {"CarryBox", "KickBox"}
            and all(
                set(record["task_losses"][task]) == expected_task_loss_keys
                and float(record["task_losses"][task]["count"]) == 4.0
                and all(
                    math.isfinite(float(record["task_losses"][task][name]))
                    for name in ("video_loss", "action_loss", "ifp_loss")
                )
                and len(record["task_losses"][task]["ifp_head_losses"])
                == config.ifp_heads
                and len(record["task_losses"][task]["ifp_loss_sums"])
                == config.ifp_heads
                and len(record["task_losses"][task]["ifp_loss_elements"])
                == config.ifp_heads
                and all(
                    math.isfinite(float(value))
                    for name in ("video_loss_sum", "action_loss_sum")
                    for value in (record["task_losses"][task][name],)
                )
                and all(
                    math.isfinite(float(value))
                    for name in ("ifp_head_losses", "ifp_loss_sums")
                    for value in record["task_losses"][task][name]
                )
                and int(record["task_losses"][task]["video_loss_elements"]) > 0
                and int(record["task_losses"][task]["action_loss_elements"]) > 0
                and all(
                    int(value) >= 0
                    for value in record["task_losses"][task]["ifp_loss_elements"]
                )
                for task in ("CarryBox", "KickBox")
            )
            for record in records
        ),
        "task_loss_element_counts_exact": all(
            int(record["task_losses"][task]["video_loss_elements"])
            == sum(
                int(row["chunk_size"]) * visual_elements_per_latent
                for row in expected_samples_by_step[int(record["optimizer_step"])]
                if row["task"] == task
            )
            and int(record["task_losses"][task]["action_loss_elements"])
            == sum(
                (int(row["action_stop"]) - int(row["action_start"]))
                * config.action_dim
                for row in expected_samples_by_step[int(record["optimizer_step"])]
                if row["task"] == task
            )
            and [int(value) for value in record["task_losses"][task]["ifp_loss_elements"]]
            == [
                sum(
                    int(row["chunk_size"]) * visual_elements_per_latent
                    for row in expected_samples_by_step[int(record["optimizer_step"])]
                    if row["task"] == task and bool(row["ifp_valid"][head_index])
                )
                for head_index in range(config.ifp_heads)
            ]
            for record in records
            for task in ("CarryBox", "KickBox")
        ),
        "all_branch_gradients_finite_positive": all(
            set(record["gradient_norms"]) == expected_branch_keys
            and all(math.isfinite(float(value)) and float(value) > 0.0 for value in record["gradient_norms"].values())
            for record in records
        ),
        "all_branch_updates_finite_positive": all(
            set(record["parameter_update_norms"]) == expected_branch_keys
            and all(math.isfinite(float(value)) and float(value) > 0.0 for value in record["parameter_update_norms"].values())
            for record in records
        ),
        "all_trainable_parameters_finite_after_every_update": all(
            record.get("all_trainable_parameters_finite") is True
            for record in records
        ),
        "two_factorized_main_passes_every_step": all(
            int(record["main_pass_count_per_sample"]) == 2 for record in records
        ),
        "rgb_interval_exposures_exact": sum(
            int(record["rgb_interval_exposures"]) for record in records
        )
        == config.epochs * 22_400,
        "latent_transition_exposures_exact": sum(
            int(record["latent_transition_exposures"]) for record in records
        )
        == config.epochs * 5_600,
        "action_exposures_exact": sum(
            int(record["action_exposures"]) for record in records
        )
        == config.epochs * 112_000,
        "transformer_token_exposures_exact": (
            observed_token_totals == expected_token_totals
            and expected_token_totals["total_transformer_token_exposures"]
            == expected_token_totals["two_pass_main_transformer_tokens"]
            + expected_token_totals["ifp_executed_transformer_tokens"]
            and expected_token_totals["supervised_transformer_token_exposures"]
            == expected_token_totals["two_pass_main_transformer_tokens"]
            + expected_token_totals["ifp_supervised_transformer_tokens"]
            and expected_token_totals["ifp_executed_transformer_tokens"]
            >= expected_token_totals["ifp_supervised_transformer_tokens"]
        ),
        "global_target_element_loss_reduction_exact": all(
            record.get("loss_reduction") == config.loss_reduction
            and int(record.get("video_loss_elements", -1))
            == expected_loss_elements_by_step[int(record["optimizer_step"])]["video"]
            and int(record.get("action_loss_elements", -1))
            == expected_loss_elements_by_step[int(record["optimizer_step"])]["action"]
            and [int(value) for value in record.get("ifp_loss_elements", [])]
            == expected_loss_elements_by_step[int(record["optimizer_step"])]["ifp"]
            and [int(value) for value in record.get("ifp_zero_target_heads", [])]
            == [
                head_index
                for head_index, value in enumerate(
                    expected_loss_elements_by_step[int(record["optimizer_step"])]["ifp"]
                )
                if value == 0
            ]
            for record in records
            if int(record["optimizer_step"]) in expected_loss_elements_by_step
        )
        and len(records) == len(expected_loss_elements_by_step),
        "rank_local_loss_scale_evidence_exact": all(
            record.get("loss_rank_evidence")
            == [
                {
                    "rank": int(row["rank"]),
                    "local_elements": {
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
                    },
                    "scales": {
                        "video": config.world_size
                        * int(row["chunk_size"])
                        * visual_elements_per_latent
                        / expected_loss_elements_by_step[int(record["optimizer_step"])]["video"],
                        "action": config.world_size
                        * (int(row["action_stop"]) - int(row["action_start"]))
                        * config.action_dim
                        / expected_loss_elements_by_step[int(record["optimizer_step"])]["action"],
                        "ifp": [
                            (
                                config.world_size
                                * int(row["chunk_size"])
                                * visual_elements_per_latent
                                / expected_loss_elements_by_step[
                                    int(record["optimizer_step"])
                                ]["ifp"][head_index]
                                if bool(row["ifp_valid"][head_index])
                                and expected_loss_elements_by_step[
                                    int(record["optimizer_step"])
                                ]["ifp"][head_index]
                                > 0
                                else 0.0
                            )
                            for head_index in range(config.ifp_heads)
                        ],
                    },
                }
                for row in expected_samples_by_step[int(record["optimizer_step"])]
            ]
            for record in records
            if int(record["optimizer_step"]) in expected_samples_by_step
        )
        and len(records) == len(expected_samples_by_step),
        "scheduled_prompt_dropout_exact": (
            observed_prompt_enabled == expected_prompt_enabled
            and sum(int(record["prompt_dropped_samples"]) for record in records)
            == len(scheduled_rows) - expected_prompt_enabled
        ),
        "both_tasks_progress": all(
            all(task_checks.values()) for task_checks in progress_checks.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "task_epoch_element_means": epoch_element_means,
        "task_progress_checks": progress_checks,
        "expected_transformer_token_totals": expected_token_totals,
        "observed_transformer_token_totals": observed_token_totals,
        "expected_loss_element_totals": {
            "video": sum(value["video"] for value in expected_loss_elements_by_step.values()),
            "action": sum(value["action"] for value in expected_loss_elements_by_step.values()),
            "ifp": [
                sum(value["ifp"][head] for value in expected_loss_elements_by_step.values())
                for head in range(config.ifp_heads)
            ],
        },
        "prompt_dropout": {
            "scheduled_enabled_samples": expected_prompt_enabled,
            "observed_enabled_samples": observed_prompt_enabled,
            "scheduled_dropped_samples": len(scheduled_rows) - expected_prompt_enabled,
        },
    }


def main() -> None:
    args = parse_args()
    config = PaperZeroWAMConfig()
    config.validate()
    rank, world_size, local_rank, device = setup_distributed()
    runtime_started_at = time.perf_counter()
    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "CONFIG.json").write_text(
            json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    dist.barrier()
    publish_runtime_milestone(
        output_dir,
        mode=args.mode,
        rank=rank,
        world_size=world_size,
        device=device,
        milestone="distributed_initialized",
        started_at=runtime_started_at,
    )

    torch.manual_seed(config.noise_seed)
    torch.cuda.manual_seed_all(config.noise_seed)
    # Keep the sharded parameters themselves FP32.  Wan's time embeddings and
    # output heads explicitly disable the surrounding BF16 autocast and assert
    # FP32; FSDP param casting would turn their weights BF16 before those calls.
    # The outer autocast below still executes the eligible Transformer/linear
    # operators in BF16, while reductions remain BF16.
    model = PaperZeroWAM.from_wan_pretrained(config, dtype=torch.float32)
    full_parameter_count = model.parameter_count
    if full_parameter_count != config.expected_parameter_count:
        raise RuntimeError(
            f"expected {config.expected_parameter_count:,} parameters, "
            f"got {full_parameter_count:,}"
        )
    publish_runtime_milestone(
        output_dir,
        mode=args.mode,
        rank=rank,
        world_size=world_size,
        device=device,
        milestone="full_width_model_constructed",
        started_at=runtime_started_at,
        architecture_parameter_count=full_parameter_count,
    )
    mixed_precision = MixedPrecision(
        param_dtype=None,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=None,
    )
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({PaperMoTLayer, IFPHead}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    publish_runtime_milestone(
        output_dir,
        mode=args.mode,
        rank=rank,
        world_size=world_size,
        device=device,
        milestone="full_shard_fsdp_constructed",
        started_at=runtime_started_at,
        architecture_parameter_count=full_parameter_count,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.peak_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    publish_runtime_milestone(
        output_dir,
        mode=args.mode,
        rank=rank,
        world_size=world_size,
        device=device,
        milestone="adamw_constructed",
        started_at=runtime_started_at,
        architecture_parameter_count=full_parameter_count,
    )
    start_step = (
        load_checkpoint_if_present(model, optimizer, output_dir, rank, config)
        if args.mode == "formal"
        else 0
    )
    samples = ScheduledSamples(
        args.schedule.resolve(),
        config.resolved(config.latent_cache),
        rank,
        config,
        args.mode,
        args.overfit_steps,
    )
    publish_runtime_milestone(
        output_dir,
        mode=args.mode,
        rank=rank,
        world_size=world_size,
        device=device,
        milestone="schedule_and_data_admitted",
        started_at=runtime_started_at,
        resumed_from_step=start_step,
        scheduled_optimizer_steps=len(samples),
    )
    total_steps = len(samples)
    expected_steps = args.overfit_steps if args.mode == "overfit" else config.optimizer_steps
    if total_steps != expected_steps:
        raise ValueError(f"expected {expected_steps} steps, found {total_steps}")
    log_path = output_dir / "TRAIN_TRACE.jsonl"
    if rank == 0:
        retained_records: list[str] = []
        if start_step > 0:
            if not log_path.is_file():
                raise ValueError(
                    f"checkpoint at step {start_step} has no training trace"
                )
            retained_records = retained_trace_prefix(log_path, start_step)
        with log_path.open("w", encoding="utf-8") as stream:
            stream.writelines(retained_records)
    dist.barrier()
    start_time = time.perf_counter()
    first_losses: dict[str, float] | None = None
    last_losses: dict[str, float] | None = None
    if start_step > 0 and log_path.exists():
        with log_path.open("r", encoding="utf-8") as stream:
            retained = [json.loads(line) for line in stream if line.strip()]
        first_record = retained[0]
        last_record = retained[-1]
        first_losses = dict(first_record["losses"])
        last_losses = dict(last_record["losses"])
    epoch_accumulators: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    overfit_gate_seed = config.noise_seed + 9_999_991
    initial_prompt_gate = (
        overfit_prompt_gate(model, samples, device, overfit_gate_seed)
        if args.mode == "overfit"
        else None
    )
    model.train()
    for step in range(start_step, total_steps):
        batch = samples.sample(step, device)
        global_loss_elements = install_global_loss_scales(batch, world_size)
        torch.manual_seed(config.noise_seed + step * world_size + rank)
        torch.cuda.manual_seed(config.noise_seed + step * world_size + rank)
        lr = learning_rate(step, total_steps, config, args.mode)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model(batch)
        if float(losses["main_pass_count"].item()) != 2.0:
            raise RuntimeError("training forward did not execute both factorized main passes")
        finite_losses = all(torch.isfinite(value).all() for value in losses.values())
        finite_flag = torch.tensor(int(finite_losses), device=device)
        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        if not bool(finite_flag.item()):
            raise FloatingPointError(f"non-finite loss at step {step}")
        losses["loss"].backward()
        gradient = gradient_norms(model)
        if any(not math.isfinite(value) or value <= 0.0 for value in gradient.values()):
            raise FloatingPointError(f"inactive/non-finite gradient at step {step}: {gradient}")
        model.clip_grad_norm_(config.gradient_clip_norm)
        before = clone_local_parameters(model)
        optimizer.step()
        updates = update_norms(before)
        del before
        if any(not math.isfinite(value) or value <= 0.0 for value in updates.values()):
            raise FloatingPointError(f"inactive/non-finite update at step {step}: {updates}")
        parameters_finite = all_trainable_parameters_finite(model)
        if not parameters_finite:
            raise FloatingPointError(
                f"non-finite trainable parameter after optimizer step {step}"
            )

        aggregate = {
            name: reduce_scalar(losses[name])
            for name in ("loss", "video_loss", "action_loss", "ifp_loss", "ifp_active_heads")
        }
        if first_losses is None:
            first_losses = dict(aggregate)
        last_losses = dict(aggregate)
        task_losses = task_loss_record(
            losses, batch, batch["meta"]["task"], config
        )
        local_sample = {
            field: batch["meta"][field] for field in TRACE_SAMPLE_FIELDS
        }
        local_loss_evidence = {
            "rank": rank,
            "local_elements": global_loss_elements["local"],
            "scales": global_loss_elements["scales"],
        }
        gathered_samples: list[dict[str, Any] | None] | None = (
            [None] * world_size if rank == 0 else None
        )
        gathered_loss_evidence: list[dict[str, Any] | None] | None = (
            [None] * world_size if rank == 0 else None
        )
        dist.gather_object(local_sample, gathered_samples, dst=0)
        dist.gather_object(local_loss_evidence, gathered_loss_evidence, dst=0)
        composition = torch.tensor(
            [
                batch["meta"]["latent_transition_exposures"],
                batch["meta"]["rgb_interval_exposures"],
                batch["meta"]["action_exposures"],
                2 * batch["meta"]["main_sequence_tokens_per_pass"],
                sum(batch["meta"]["ifp_sequence_tokens"]),
                sum(batch["meta"]["ifp_executed_sequence_tokens"]),
                batch["meta"]["supervised_transformer_token_exposures"],
                batch["meta"]["transformer_token_exposures"],
                int(batch["meta"]["prompt_enabled"]),
            ],
            device=device,
            dtype=torch.long,
        )
        dist.all_reduce(composition, op=dist.ReduceOp.SUM)
        record: dict[str, Any] = {
            "mode": args.mode,
            "optimizer_step": step,
            "resumed_from_step": start_step,
            "epoch": int(batch["meta"]["epoch"]),
            "epoch_step": int(batch["meta"]["epoch_step"]),
            "learning_rate": lr,
            "world_size": world_size,
            "packed_samples_per_rank": 1,
            "global_packed_sample_batch": world_size,
            "gradient_accumulation_steps": 1,
            "main_pass_count_per_sample": 2,
            "latent_transition_exposures": int(composition[0].item()),
            "rgb_interval_exposures": int(composition[1].item()),
            "action_exposures": int(composition[2].item()),
            "two_pass_main_transformer_tokens": int(composition[3].item()),
            "ifp_supervised_transformer_tokens": int(composition[4].item()),
            "ifp_executed_transformer_tokens": int(composition[5].item()),
            "supervised_transformer_token_exposures": int(composition[6].item()),
            "total_transformer_token_exposures": int(composition[7].item()),
            "video_loss_elements": global_loss_elements["video"],
            "action_loss_elements": global_loss_elements["action"],
            "ifp_loss_elements": global_loss_elements["ifp"],
            "ifp_zero_target_heads": global_loss_elements["ifp_zero_target_heads"],
            "loss_reduction": config.loss_reduction,
            "loss_rank_evidence": (
                sorted(
                    (value for value in gathered_loss_evidence if value is not None),
                    key=lambda value: int(value["rank"]),
                )
                if rank == 0 and gathered_loss_evidence is not None
                else []
            ),
            "prompt_enabled_samples": int(composition[8].item()),
            "prompt_dropped_samples": world_size - int(composition[8].item()),
            "losses": aggregate,
            "task_losses": task_losses,
            "samples": (
                sorted(
                    (value for value in gathered_samples if value is not None),
                    key=lambda value: int(value["rank"]),
                )
                if rank == 0 and gathered_samples is not None
                else []
            ),
            "gradient_norms": gradient,
            "parameter_update_norms": updates,
            "all_trainable_parameters_finite": parameters_finite,
            "optimizer_applied": True,
            "amp_scaler_skipped": False,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        epoch = record["epoch"]
        for name in ("video_loss", "action_loss", "ifp_loss"):
            epoch_accumulators[epoch][name].append(aggregate[name])
        if rank == 0:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            emit_json_best_effort(record)
        if step == start_step:
            publish_runtime_milestone(
                output_dir,
                mode=args.mode,
                rank=rank,
                world_size=world_size,
                device=device,
                milestone="first_optimizer_update_applied",
                started_at=runtime_started_at,
                optimizer_step=step,
                effective_learning_rate=lr,
                optimizer_applied=True,
                amp_scaler_skipped=False,
            )

        epoch_complete = args.mode == "formal" and (step + 1) % config.optimizer_steps_per_epoch == 0
        if epoch_complete:
            save_checkpoint(
                model, optimizer, output_dir, step + 1, rank, args.mode, config
            )
            publish_runtime_milestone(
                output_dir,
                mode=args.mode,
                rank=rank,
                world_size=world_size,
                device=device,
                milestone="complete_epoch_checkpoint_committed",
                started_at=runtime_started_at,
                completed_optimizer_steps=step + 1,
                complete_epochs=(step + 1) // config.optimizer_steps_per_epoch,
            )
            if rank == 0:
                try:
                    publish_formal_progress(
                        log_path,
                        output_dir,
                        step + 1,
                        time.perf_counter() - start_time,
                        config,
                    )
                except Exception as error:
                    emit_json_best_effort(
                        {
                            "warning": "formal_progress_publication_failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "completed_checkpoint_step": step + 1,
                            "training_continues": True,
                        }
                    )
            # Keep every FSDP rank on the same epoch boundary while rank zero
            # publishes observational progress and refreshes the living page.
            dist.barrier()

    prompt_gate = (
        overfit_prompt_gate(model, samples, device, overfit_gate_seed)
        if args.mode == "overfit"
        else None
    )
    result: dict[str, Any] | None = None
    result_name: str | None = None
    if rank == 0:
        assert first_losses is not None and last_losses is not None
        stochastic_train_ratios = {
            name: last_losses[name] / first_losses[name]
            for name in ("video_loss", "action_loss", "ifp_loss")
        }
        ratios = (
            {
                name: prompt_gate["losses"]["matched"][name]
                / initial_prompt_gate["losses"]["matched"][name]
                for name in ("video_loss", "action_loss", "ifp_loss")
            }
            if args.mode == "overfit"
            else stochastic_train_ratios
        )
        if args.mode == "overfit":
            passed = all(
                math.isfinite(value) and value <= 0.5 for value in ratios.values()
            ) and bool(prompt_gate and prompt_gate["passed"])
            execution_decision = overfit_execution_decision(
                log_path, args.schedule.resolve(), total_steps, config
            )
            result_name = "OVERFIT_RESULT.json"
        else:
            formal_decision = formal_training_decision(
                log_path, args.schedule.resolve(), config
            )
            passed = bool(formal_decision["passed"])
            execution_checks = {
                name: value
                for name, value in formal_decision["checks"].items()
                if name != "both_tasks_progress"
            }
            execution_decision = {
                "passed": all(execution_checks.values()),
                "checks": execution_checks,
            }
            result_name = "FORMAL_TRAINING_RESULT.json"
        result = {
            "protocol": f"paper_zero_wam_sugar_{args.mode}_v1",
            "passed": passed,
            "execution_completed": execution_decision["passed"],
            "execution_decision": execution_decision,
            "architecture_parameter_count": full_parameter_count,
            "optimizer_steps": total_steps,
            "first_losses": first_losses,
            "last_losses": last_losses,
            "last_to_first_loss_ratios": ratios,
            "stochastic_train_last_to_first_loss_ratios": stochastic_train_ratios,
            "initial_prompt_gate": initial_prompt_gate,
            "prompt_gate": prompt_gate,
            "formal_decision": formal_decision if args.mode == "formal" else None,
            "batch": {
                "world_size": world_size,
                "packed_samples_per_rank": 1,
                "global_packed_samples": world_size,
                "gradient_accumulation_steps": 1,
            },
            "checkpoint_policy": (
                "none_diagnostic_only_formal_restarts_from_wan_base"
                if args.mode == "overfit"
                else "two_rotating_complete_epoch_slots"
            ),
            "hash_checks": False,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
    execution_ok = torch.tensor(
        int(rank == 0 and result is not None and result["execution_completed"]),
        device=device,
    )
    dist.broadcast(execution_ok, src=0)
    if not bool(execution_ok.item()):
        raise RuntimeError(
            f"{args.mode} completed its forward/update loop but failed its execution trace contract"
        )
    dist.barrier()
    dist.destroy_process_group()
    # No distributed operation remains after the terminal result appears.
    if rank == 0:
        assert result is not None and result_name is not None
        write_json_atomic(output_dir / result_name, result)
        refresh_results_document_best_effort()
    # A completed overfit is a scientific diagnostic, not an admission gate
    # that can shorten the already-declared formal run. Infrastructure errors
    # before the atomic result still exit nonzero through their original
    # exception path and are the only failures eligible for identical retry.


if __name__ == "__main__":
    main()
