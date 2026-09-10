#!/usr/bin/env python3
"""Two-update, 128-sample optimizer smoke on the exact adapted official HOST model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_host_official_prompt_gate import construct_model, require_h200
from host_sugar_dataset import HostSugarTrainingDataset, ROOT


CONTEXT = ROOT / (
    "experiments/demo_following/host_official_v1/sugar_adapter_v1/"
    "HOST_NEUTRAL_CONTEXT.pt"
)
PROMPT_GATE = ROOT / (
    "experiments/demo_following/host_official_v1/prompt_gate_v1/"
    "HOST_FROZEN_PROMPT_GATE.json"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/host_official_v1/optimizer_smoke_v1/"
    "HOST_OPTIMIZER_SMOKE.json"
)
INTERFACE_NAMES = (
    "mot.mixtures.action.action_encoder.weight",
    "mot.mixtures.action.head.weight",
    "mot.mixtures.action.head.bias",
    "proprio_encoder.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=271510)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def freeze_official_stage2(model: torch.nn.Module) -> dict[str, int]:
    """Apply the released trainer's exact stage-2 trainable-scope rule."""

    from self_grounded_prediction.trainer import Wan22Trainer

    Wan22Trainer._apply_dit_only_train_mode(model)
    for name, parameter in model.dit.named_parameters():
        if name.startswith("mixtures.video."):
            parameter.requires_grad_(False)
    if model.visual_encoder is not None:
        model.visual_encoder.requires_grad_(False)
        model.visual_encoder.eval()
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def state_digest(model: torch.nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    matched = 0
    with torch.no_grad():
        for name, value in sorted(model.state_dict().items()):
            if not name.startswith(prefixes):
                continue
            matched += 1
            tensor = value.detach().contiguous()
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    if matched == 0:
        raise RuntimeError(f"no official frozen tensors matched prefixes {prefixes}")
    return digest.hexdigest()


def build_sample_indices() -> list[int]:
    # 16 Carry + 16 Kick motions, each at four complete-horizon anchors.
    anchors = (0, 45, 90, 135)
    indices = []
    for local_motion in range(16):
        for target_motion in (local_motion, 80 + local_motion):
            indices.extend(target_motion * 140 + anchor for anchor in anchors)
    if len(indices) != 128 or len(set(indices)) != 128:
        raise RuntimeError("fixed HOST optimizer-smoke sample set drift")
    return indices


def evaluate_fixed(
    model: torch.nn.Module,
    dataset: HostSugarTrainingDataset,
    indices: list[int],
    seed: int,
) -> list[dict[str, float]]:
    was_training = model.training
    model.eval()
    records = []
    with torch.no_grad():
        for rank, index in enumerate(indices):
            sample = dataset.collate_fn([dataset[index]])
            private_seed = seed + rank
            torch.manual_seed(private_seed)
            torch.cuda.manual_seed_all(private_seed)
            total, losses = model.training_loss(sample)
            row = {
                "index": index,
                "total": float(total),
                "video": float(losses["loss_video"]),
                "action": float(losses["loss_action"]),
                "progress": float(losses["loss_progress"]),
            }
            if not np.isfinite(list(row.values())).all():
                raise RuntimeError("non-finite official HOST fixed smoke evaluation")
            records.append(row)
    if was_training:
        model.train()
        if model.visual_encoder is not None:
            model.visual_encoder.eval()
    return records


def main() -> None:
    args = parse_args()
    runtime = require_h200()
    gate = json.loads(PROMPT_GATE.read_text())
    if gate.get("status") != "pass" or not gate.get("full_predeclared_gate"):
        raise RuntimeError("optimizer smoke is closed until the full frozen prompt gate passes")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    model, load_audit = construct_model()
    scope = freeze_official_stage2(model)
    dataset = HostSugarTrainingDataset("train", context_path=CONTEXT)
    indices = build_sample_indices()
    evaluation_indices = [
        motion * 140 + 45 for motion in (*range(8), *range(80, 88))
    ]

    named_parameters = dict(model.named_parameters())
    for name in INTERFACE_NAMES:
        if name not in named_parameters or not named_parameters[name].requires_grad:
            raise RuntimeError(f"official adapted interface is not stage-2 trainable: {name}")
    action_block_names = [
        name
        for name, parameter in named_parameters.items()
        if name.startswith("mot.mixtures.action.blocks.") and parameter.requires_grad
    ]
    if not action_block_names:
        raise RuntimeError("official HOST action expert is not stage-2 trainable")

    interface_before = {
        name: named_parameters[name].detach().cpu().clone() for name in INTERFACE_NAMES
    }
    action_probe_name = action_block_names[len(action_block_names) // 2]
    action_probe_before = named_parameters[action_probe_name].detach().cpu().clone()
    frozen_prefixes = ("mot.mixtures.video.", "visual_encoder.")
    frozen_hash_before = state_digest(model, frozen_prefixes)
    fixed_before = evaluate_fixed(model, dataset, evaluation_indices, args.seed + 100_000)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    if model.visual_encoder is not None:
        model.visual_encoder.eval()

    gradient_accumulation = 64
    micro_records = []
    update_records = []
    for micro_step, index in enumerate(indices, start=1):
        sample = dataset.collate_fn([dataset[index]])
        private_seed = args.seed + micro_step
        torch.manual_seed(private_seed)
        torch.cuda.manual_seed_all(private_seed)
        total, losses = model.training_loss(sample)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite HOST optimizer smoke loss at micro-step {micro_step}")
        (total / gradient_accumulation).backward()
        micro_records.append(
            {
                "micro_step": micro_step,
                "index": index,
                "total": float(total.detach()),
                "video": float(losses["loss_video"]),
                "action": float(losses["loss_action"]),
                "progress": float(losses["loss_progress"]),
            }
        )
        if micro_step % gradient_accumulation != 0:
            continue

        interface_grad_norms = {}
        for name in INTERFACE_NAMES:
            gradient = named_parameters[name].grad
            norm = 0.0 if gradient is None else float(gradient.float().norm())
            if not math.isfinite(norm) or norm <= 0.0:
                raise RuntimeError(f"missing/non-finite interface gradient: {name}={norm}")
            interface_grad_norms[name] = norm
        action_grad_norms = [
            float(named_parameters[name].grad.float().norm())
            for name in action_block_names
            if named_parameters[name].grad is not None
        ]
        if not action_grad_norms or not np.isfinite(action_grad_norms).all() or max(action_grad_norms) <= 0:
            raise RuntimeError("official action expert has no finite nonzero gradient")
        if any(
            parameter.grad is not None
            for name, parameter in named_parameters.items()
            if name.startswith(frozen_prefixes)
        ):
            raise RuntimeError("frozen official video/visual parameter received a gradient")

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0))
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError(f"invalid HOST optimizer smoke global grad norm: {grad_norm}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_records.append(
            {
                "update": len(update_records) + 1,
                "micro_step": micro_step,
                "global_grad_norm_before_clip": grad_norm,
                "interface_grad_norms": interface_grad_norms,
                "maximum_action_expert_grad_norm": max(action_grad_norms),
            }
        )
        print(
            f"HOST_SMOKE_UPDATE={len(update_records)}/2 micro_step={micro_step} "
            f"loss={float(total.detach()):.6f} grad_norm={grad_norm:.6f}",
            flush=True,
        )

    if len(update_records) != 2:
        raise RuntimeError(f"HOST optimizer smoke applied {len(update_records)} updates, expected 2")
    fixed_after = evaluate_fixed(model, dataset, evaluation_indices, args.seed + 100_000)
    frozen_hash_after = state_digest(model, frozen_prefixes)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("official frozen video/visual tensors changed during optimizer smoke")

    interface_deltas = {
        name: float(
            torch.max(torch.abs(named_parameters[name].detach().cpu() - interface_before[name]))
        )
        for name in INTERFACE_NAMES
    }
    if not all(delta > 0.0 and math.isfinite(delta) for delta in interface_deltas.values()):
        raise RuntimeError(f"official adapted interface did not update: {interface_deltas}")
    action_probe_delta = float(
        torch.max(
            torch.abs(named_parameters[action_probe_name].detach().cpu() - action_probe_before)
        )
    )
    if not math.isfinite(action_probe_delta) or action_probe_delta <= 0.0:
        raise RuntimeError("official action-expert probe did not update")

    before_totals = np.asarray([row["total"] for row in fixed_before])
    after_totals = np.asarray([row["total"] for row in fixed_after])
    before_tail = float(np.median(np.sort(before_totals)[-8:]))
    after_tail = float(np.median(np.sort(after_totals)[-8:]))
    tail_pass = after_tail < before_tail
    if not tail_pass:
        raise RuntimeError(
            f"fixed joint tail-median loss did not decrease: {before_tail} -> {after_tail}"
        )

    report = {
        "status": "pass",
        "diagnostic_only_not_training": True,
        "runtime": runtime,
        "prompt_gate": str(PROMPT_GATE.relative_to(ROOT)),
        "official_load": load_audit,
        "official_stage": 2,
        "scope": scope,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "unique_causal_samples": len(set(indices)),
        "micro_steps": len(indices),
        "gradient_accumulation": gradient_accumulation,
        "applied_optimizer_updates": len(update_records),
        "balanced_task_samples": {"CarryBox": 64, "KickBox": 64},
        "micro_records": micro_records,
        "updates": update_records,
        "fixed_evaluation_before": fixed_before,
        "fixed_evaluation_after": fixed_after,
        "fixed_joint_tail_median_before": before_tail,
        "fixed_joint_tail_median_after": after_tail,
        "interface_parameter_max_abs_deltas": interface_deltas,
        "action_expert_probe": action_probe_name,
        "action_expert_probe_max_abs_delta": action_probe_delta,
        "frozen_video_visual_sha256_before": frozen_hash_before,
        "frozen_video_visual_sha256_after": frozen_hash_after,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "unique_causal_samples", "applied_optimizer_updates", "fixed_joint_tail_median_before", "fixed_joint_tail_median_after", "cuda_peak_allocated_bytes")}, indent=2))


if __name__ == "__main__":
    main()
