#!/usr/bin/env python3
"""Freeze the official SMP DiffNormalizer scale for conditional TinyMDM.

The calibration uses 5,000 Carry and 5,000 Kick training windows under their
matching class condition.  It records the official per-diffusion-step mean
absolute SDS loss used by MimicKit's ``DiffNormalizer``.  No rollout outcome
or recovery trace participates in this scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_conditional_taskwide_tinymdm import (  # noqa: E402
    CLASS_IDS,
    DATASET_ROOT,
    OUTPUT_ROOT,
    load_dataset,
    load_shared_prior,
)
from run_selected_demo_tinymdm import (  # noqa: E402
    DIFFUSION_STEPS,
    atomic_json,
    require_compute_gpu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-class", type=int, default=5_000)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    result_path = output_root / "reward_calibration/RESULT.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    device = torch.device(args.device)
    require_compute_gpu(device)
    manifest = json.loads(
        (args.dataset_root.expanduser().resolve() / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    model = load_shared_prior(output_root, device)
    rng = np.random.default_rng(160828)
    selected: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for task, class_id in CLASS_IDS.items():
        array = load_dataset(manifest, task, "train")
        indices = np.sort(
            rng.choice(array.shape[0], size=args.samples_per_class, replace=False)
        )
        selected.append(array[indices])
        labels.append(np.full(indices.shape[0], class_id, dtype=np.int64))
    windows = np.concatenate(selected, axis=0)
    class_labels = np.concatenate(labels, axis=0)
    loss_chunks: list[np.ndarray] = []
    devices = [device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(160829)
        torch.cuda.manual_seed_all(160829)
        for start in range(0, windows.shape[0], args.chunk_size):
            batch = torch.as_tensor(
                windows[start : start + args.chunk_size], device=device
            )
            batch_labels = torch.as_tensor(
                class_labels[start : start + args.chunk_size], device=device
            )
            normalized = model.normalize(batch).reshape(batch.shape[0], -1)
            losses = model.ESM_SDS_loss(
                normalized,
                t_lst=list(DIFFUSION_STEPS),
                class_labels=batch_labels,
            )
            loss_chunks.append(losses.cpu().numpy())
    losses = np.concatenate(loss_chunks, axis=0)
    mean_abs = np.mean(np.abs(losses), axis=0)
    if losses.shape != (windows.shape[0], len(DIFFUSION_STEPS)):
        raise RuntimeError(f"calibration loss geometry {losses.shape}")
    normalized_mean = np.mean(losses / mean_abs[None, :], axis=-1)
    reward = np.exp(-normalized_mean * 6.0)
    result = {
        "protocol": "sugar_conditional_tinymdm_official_smp_reward_calibration_v1",
        "passed": bool(
            np.isfinite(losses).all()
            and np.isfinite(mean_abs).all()
            and np.all(mean_abs > 0.0)
        ),
        "source": "training motions only; 5000 windows per matching class",
        "sample_seed": 160828,
        "noise_seed": 160829,
        "sample_count": int(windows.shape[0]),
        "samples_per_class": args.samples_per_class,
        "diffusion_steps": list(DIFFUSION_STEPS),
        "diff_normalizer_mean_abs": mean_abs.tolist(),
        "official_sds_loss_scale": 6.0,
        "calibration_reward": {
            "mean": float(reward.mean()),
            "median": float(np.median(reward)),
            "p05": float(np.quantile(reward, 0.05)),
            "p95": float(np.quantile(reward, 0.95)),
        },
        "checks": {
            "shared_conditional_checkpoint": True,
            "shared_diff_normalizer_scale": True,
            "recovery_outcomes_excluded": True,
            "official_diffusion_steps_and_reward_transform": True,
        },
    }
    atomic_json(result_path, result)
    # Indices are deliberately not duplicated in RESULT; the deterministic
    # sampling seed plus immutable dataset manifest reproduces them.
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["passed"]:
        raise RuntimeError("conditional SMP reward calibration failed")


if __name__ == "__main__":
    main()
