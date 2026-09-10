#!/usr/bin/env python3
"""Frozen selected-video causal gate for the exact official HOST checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

from audit_host_official_strict_load import (
    ADAPTER_INITIALIZATION_SEED,
    compare_contract,
    load_and_prove_equal,
    load_model_config,
)
from host_sugar_dataset import HostSugarDataset, ROOT


HOST_SOURCE = ROOT / (
    "experiments/demo_following/host_official_v1/"
    "official_source_9f3bba57792aa5053ac600b1b7a4625f96ea6662/policy_training"
)
CHECKPOINT = ROOT / (
    "experiments/demo_following/host_official_v1/"
    "official_snapshot_fcb38563ab7afe0bb0dcefd9431adb76112fa4b6/model.pt"
)
CONFIG = CHECKPOINT.with_name("config.yaml")
CONTEXT = ROOT / (
    "experiments/demo_following/host_official_v1/sugar_adapter_v1/"
    "HOST_NEUTRAL_CONTEXT.pt"
)
DINO_REPO = ROOT / (
    "experiments/demo_following/host_official_v1/official_backbones/dinov2_pinned"
)
DINO_WEIGHTS = ROOT / (
    "experiments/demo_following/host_official_v1/official_backbones/weights/"
    "host_dinov2_vitb14_exact.pt"
)
SIGLIP_WEIGHTS = ROOT / (
    "experiments/demo_following/host_official_v1/official_backbones/weights/"
    "host_siglip_so400m_14_exact.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/host_official_v1/prompt_gate_v1/"
    "HOST_FROZEN_PROMPT_GATE.json"
)
CONDITIONS = (
    "matching",
    "wrong_task",
    "reversed",
    "same_task_alternate",
    "masked_after_preprocess",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-motions-per-split", type=int)
    parser.add_argument("--anchors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=271509)
    return parser.parse_args()


def require_h200() -> dict[str, str]:
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("official HOST prompt gate requires a CUDA Slurm allocation")
    device = torch.cuda.get_device_name(0)
    if "H200" not in device.upper():
        raise RuntimeError(f"official HOST prompt gate requires H200, found {device}")
    return {"slurm_job_id": os.environ["SLURM_JOB_ID"], "cuda_device": device}


def construct_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    sys.path.insert(0, str(HOST_SOURCE / "src"))
    from self_grounded_prediction.runtime import create_self_grounded_predictor

    backbone_paths = {
        "backbone_local_repo": str(DINO_REPO),
        "backbone_weights_path": str(DINO_WEIGHTS),
        "siglip_local_weights_path": str(SIGLIP_WEIGHTS),
    }
    config = load_model_config(
        CONFIG,
        adapted=True,
        reuse_original_wan_vae=True,
        backbone_paths=backbone_paths,
    )
    torch.manual_seed(ADAPTER_INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(ADAPTER_INITIALIZATION_SEED)
    model = create_self_grounded_predictor(
        **config, model_dtype=torch.bfloat16, device="cuda"
    )
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    contract = compare_contract(model, payload, adapted=True)
    inheritance = load_and_prove_equal(model, payload, adapted=True)
    del payload
    gc.collect()
    model.eval().requires_grad_(False)
    return model, {"contract": contract, "inheritance": inheritance}


def batch_sample(item: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    tensor_keys = (
        "video",
        "task_video",
        "action",
        "proprio",
        "image_is_pad",
        "action_is_pad",
        "progress_gt",
    )
    sample = {key: item[key].unsqueeze(0) for key in tensor_keys}
    sample.update(
        {
            "context": context["context"].unsqueeze(0),
            "context_mask": context["mask"].unsqueeze(0),
            "task_video_dropped": [False],
            "prompt": [item["prompt"]],
        }
    )
    return sample


def training_loss_with_postprocess_mask(
    model: torch.nn.Module,
    sample: dict[str, Any],
    mask_task_latents: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not mask_task_latents:
        return model.training_loss(sample)
    original = model._encode_video_latents
    calls = 0

    def wrapped(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        nonlocal calls
        value = original(
            video_tensor,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        calls += 1
        return torch.zeros_like(value) if calls == 2 else value

    model._encode_video_latents = MethodType(wrapped, model)
    try:
        result = model.training_loss(sample)
    finally:
        model._encode_video_latents = original
    if calls != 2:
        raise RuntimeError(f"HOST preprocess-mask intervention expected two VAE calls, got {calls}")
    return result


def main() -> None:
    args = parse_args()
    if args.anchors < 1 or args.anchors > 140:
        raise ValueError("--anchors must be in [1, 140]")
    runtime = require_h200()
    torch.cuda.reset_peak_memory_stats()
    model, load_audit = construct_model()
    context = torch.load(CONTEXT, map_location="cpu", weights_only=True)
    if context["prompt"] != "Follow the demonstrated motion.":
        raise RuntimeError("neutral HOST context contract drift")

    datasets = {
        "matching": {
            split: HostSugarDataset(split, prompt_mode="matching_alternate", epoch=0)
            for split in ("validation", "test")
        },
        "wrong_task": {
            split: HostSugarDataset(split, prompt_mode="wrong_task", epoch=0)
            for split in ("validation", "test")
        },
        "reversed": {
            split: HostSugarDataset(split, prompt_mode="reversed_matching_alternate", epoch=0)
            for split in ("validation", "test")
        },
        "same_task_alternate": {
            split: HostSugarDataset(split, prompt_mode="matching_alternate", epoch=1)
            for split in ("validation", "test")
        },
        "masked_after_preprocess": {
            split: HostSugarDataset(split, prompt_mode="matching_alternate", epoch=0)
            for split in ("validation", "test")
        },
    }
    # Frozen prompt scoring requires a complete official 24-action/five-frame
    # future.  Intervals 136..139 are valid training rows with official padding,
    # but are not admissible as causal representation anchors.
    anchor_intervals = np.rint(np.linspace(0, 135, args.anchors)).astype(int).tolist()
    records = []
    with torch.inference_mode():
        for split in ("validation", "test"):
            motion_count = len(datasets["matching"][split].rows)
            if args.max_motions_per_split is not None:
                motion_count = min(motion_count, args.max_motions_per_split)
            for motion_index in range(motion_count):
                for anchor_rank, interval in enumerate(anchor_intervals):
                    base_index = motion_index * 140 + interval
                    reference_item = datasets["matching"][split][base_index]
                    if reference_item["valid_action_count"] != 24 or bool(
                        reference_item["action_is_pad"].any()
                    ) or bool(reference_item["image_is_pad"].any()):
                        raise RuntimeError("frozen prompt gate admitted a padded future anchor")
                    for condition in CONDITIONS:
                        item = datasets[condition][split][base_index]
                        if not torch.equal(item["video"], reference_item["video"]):
                            raise RuntimeError("prompt intervention changed target robot video")
                        if not torch.equal(item["action"], reference_item["action"]):
                            raise RuntimeError("prompt intervention changed target action")
                        if not torch.equal(item["proprio"], reference_item["proprio"]):
                            raise RuntimeError("prompt intervention changed target proprio")
                        sample = batch_sample(item, context)
                        private_seed = (
                            args.seed
                            + (0 if split == "validation" else 1_000_000)
                            + motion_index * 1000
                            + anchor_rank
                        )
                        torch.manual_seed(private_seed)
                        torch.cuda.manual_seed_all(private_seed)
                        total, losses = training_loss_with_postprocess_mask(
                            model,
                            sample,
                            mask_task_latents=condition == "masked_after_preprocess",
                        )
                        numeric = {
                            key: float(losses[key])
                            for key in ("loss_video", "loss_progress", "loss_action")
                        }
                        if not np.isfinite([float(total), *numeric.values()]).all():
                            raise RuntimeError("non-finite official frozen HOST loss")
                        records.append(
                            {
                                "split": split,
                                "task": item["target_task"],
                                "motion_id": item["target_source_motion_id"],
                                "anchor_interval": interval,
                                "private_seed": private_seed,
                                "condition": condition,
                                **numeric,
                            }
                        )
                        print(
                            f"PROMPT_GATE {split} motion={motion_index + 1}/{motion_count} "
                            f"anchor={anchor_rank + 1}/{len(anchor_intervals)} condition={condition} "
                            f"video={numeric['loss_video']:.6f} progress={numeric['loss_progress']:.6f}",
                            flush=True,
                        )

    grouped: dict[tuple[str, str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        for metric in ("loss_video", "loss_progress"):
            grouped[(row["split"], row["task"], row["motion_id"], metric)][row["condition"]].append(
                row[metric]
            )
    comparisons = {}
    full_gate = args.max_motions_per_split is None and args.anchors == 10
    passed = full_gate
    for split in ("validation", "test"):
        comparisons[split] = {}
        motion_keys = sorted(
            {(row["task"], row["motion_id"]) for row in records if row["split"] == split}
        )
        for metric in ("loss_video", "loss_progress"):
            comparisons[split][metric] = {}
            for alternative in ("wrong_task", "reversed", "masked_after_preprocess"):
                deltas = []
                for task, motion_id in motion_keys:
                    values = grouped[(split, task, motion_id, metric)]
                    deltas.append(
                        float(np.mean(values[alternative]) - np.mean(values["matching"]))
                    )
                wins = sum(delta > 0.0 for delta in deltas)
                required = int(np.ceil(0.9 * len(deltas)))
                criterion = wins >= required
                passed &= criterion
                comparisons[split][metric][alternative] = {
                    "matching_advantage_mean": float(np.mean(deltas)),
                    "motion_wins": wins,
                    "motion_count": len(deltas),
                    "required_wins": required,
                    "passed": criterion,
                }
            same_deltas = []
            for task, motion_id in motion_keys:
                values = grouped[(split, task, motion_id, metric)]
                same_deltas.append(
                    float(np.mean(values["same_task_alternate"]) - np.mean(values["matching"]))
                )
            comparisons[split][metric]["same_task_identity_report_only"] = {
                "matching_advantage_mean": float(np.mean(same_deltas)),
                "matching_motion_wins": sum(delta > 0.0 for delta in same_deltas),
                "motion_count": len(same_deltas),
            }

    report = {
        "status": "pass" if passed else ("diagnostic_complete" if not full_gate else "fail"),
        "full_predeclared_gate": full_gate,
        "runtime": runtime,
        "official_load": load_audit,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "conditions": list(CONDITIONS),
        "matched_private_noise": True,
        "anchors": anchor_intervals,
        "records": records,
        "comparisons": comparisons,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "full_predeclared_gate", "comparisons", "cuda_peak_allocated_bytes")}, indent=2))
    if full_gate and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
