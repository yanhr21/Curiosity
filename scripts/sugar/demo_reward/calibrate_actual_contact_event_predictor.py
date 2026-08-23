#!/usr/bin/env python3
"""Fit validation-only variance scaling and test a frozen event predictor."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "SUGAR/source/sugar_rl"))
sys.path.insert(0, str(ROOT / "scripts/sugar/demo_reward"))

from train_actual_contact_event_predictor import (  # noqa: E402
    EVENT_TARGET_NAMES,
    PairDataset,
    make_loader,
    model_from_normalization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.no_grad()
def collect(
    model,
    loader,
    demo_bank: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets, means, variances = [], [], []
    model.eval()
    for batch in loader:
        policy = batch["policy_prefix"].to(device, non_blocking=True)
        selected = batch["selected_demo"].to(device, non_blocking=True)
        demo = demo_bank.index_select(0, selected)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                policy_prefix=policy,
                selected_demo_condition=demo,
            )
        targets.append(model.encode_targets(target).cpu().numpy())
        means.append(output["mean_log1p_scaled"].float().cpu().numpy())
        variances.append(
            torch.exp(output["log_variance_log1p_scaled"].float()).cpu().numpy()
        )
    return tuple(np.concatenate(values) for values in (targets, means, variances))


def coverage_record(
    target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> dict[str, object]:
    half_width = 1.6448536269514722 * np.sqrt(variance)
    covered = np.abs(target - mean) <= half_width
    per_target = np.mean(covered, axis=0)
    nll = 0.5 * (
        np.square(target - mean) / variance + np.log(variance)
    )
    return {
        "row_count": len(target),
        "mean_90pct_coverage": float(np.mean(per_target)),
        "minimum_target_coverage": float(np.min(per_target)),
        "maximum_target_coverage": float(np.max(per_target)),
        "mean_transformed_gaussian_nll": float(np.mean(nll)),
        "per_target_90pct_coverage": {
            name: float(value)
            for name, value in zip(EVENT_TARGET_NAMES, per_target)
        },
    }


def main() -> None:
    args = parse_args()
    if socket.gethostname().startswith(("mgmtserver", "login")):
        raise RuntimeError("calibration must run inside a compute allocation")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("retained CUDA Slurm allocation required")
    dataset_root = args.dataset_root.expanduser().resolve()
    predictor_dir = args.predictor_dir.expanduser().resolve()
    result_path = predictor_dir / "CALIBRATION_RESULT.json"
    multiplier_path = predictor_dir / "UNCERTAINTY_CALIBRATION.npz"
    if result_path.exists() or multiplier_path.exists():
        raise FileExistsError("calibration artifacts already exist")
    device = torch.device(args.device)
    model = model_from_normalization(dataset_root / "NORMALIZATION.npz", device)
    checkpoint = torch.load(
        predictor_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    arrays = {}
    for split in ("validation", "test"):
        dataset = PairDataset(dataset_root, split)
        loader = make_loader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            workers=args.num_workers,
        )
        arrays[split] = collect(model, loader, dataset.demo_bank(device), device)

    validation_target, validation_mean, validation_variance = arrays["validation"]
    multiplier = np.mean(
        np.square(validation_target - validation_mean)
        / np.maximum(validation_variance, 1.0e-8),
        axis=0,
    )
    multiplier = np.clip(multiplier, 0.1, 1000.0).astype(np.float32)
    np.savez(
        multiplier_path,
        variance_multiplier=multiplier,
        target_names=np.asarray(EVENT_TARGET_NAMES, dtype="U48"),
    )
    records = {}
    for split, (target, mean, variance) in arrays.items():
        records[split] = coverage_record(
            target,
            mean,
            variance * multiplier[None],
        )
    checks = {
        "multiplier_fit_uses_validation_only": True,
        "test_labels_not_used_for_calibration": True,
        "validation_mean_coverage_between_0p78_and_0p98": (
            0.78 <= records["validation"]["mean_90pct_coverage"] <= 0.98
        ),
        "test_mean_coverage_between_0p78_and_0p98": (
            0.78 <= records["test"]["mean_90pct_coverage"] <= 0.98
        ),
        "test_minimum_target_coverage_at_least_0p65": (
            records["test"]["minimum_target_coverage"] >= 0.65
        ),
        "all_multipliers_finite_positive": bool(
            np.isfinite(multiplier).all() and np.all(multiplier > 0)
        ),
    }
    training_result = json.loads(
        (predictor_dir / "RESULT.json").read_text(encoding="utf-8")
    )
    result = {
        "protocol": "sugar_causal_contact_event_predictor_calibration_v1",
        "passed": bool(training_result["passed"] and all(checks.values())),
        "frozen_training_gate_passed": bool(training_result["passed"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checks": checks,
        "variance_multiplier": multiplier.tolist(),
        "coverage": records,
        "claim_boundary": (
            "Passing establishes validation-calibrated uncertainty on the frozen held-out test. "
            "It does not establish policy-level demo following."
        ),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
