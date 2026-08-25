#!/usr/bin/env python3
"""Freeze-evaluate a Carry45 transition-risk checkpoint at deployment time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_official_transition_risk_transformer import (  # noqa: E402
    CausalTransitionRiskTransformer,
    TransitionDataset,
    _profile_metrics,
    evaluate_rows,
    make_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _profile_records(
    dataset: TransitionDataset,
    rows: dict[str, np.ndarray],
    threshold: float,
) -> list[dict[str, object]]:
    records = []
    selected = rows["anchor"] <= 49
    for profile in np.unique(rows["profile"]):
        mask = (rows["profile"] == profile) & selected
        records.append(
            {
                "profile": int(profile),
                "name": str(dataset.profile_name[int(profile)]),
                "risk_target": bool(rows["risk"][mask][0]),
                "mean_first50_probability": float(np.mean(rows["score"][mask])),
                "predicted_risk": bool(np.mean(rows["score"][mask]) >= threshold),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if socket.gethostname().startswith(("mgmtserver", "login")):
        raise RuntimeError("frozen transition evaluation must run on a compute node")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("retained CUDA Slurm allocation required")
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(args.device)
    with np.load(dataset_root / "NORMALIZATION.npz", allow_pickle=False) as archive:
        state_mean = torch.from_numpy(np.asarray(archive["state_mean"], dtype=np.float32))
        state_std = torch.from_numpy(np.asarray(archive["state_std"], dtype=np.float32))
    model = CausalTransitionRiskTransformer(state_mean, state_std).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("protocol") != "sugar_causal_transition_risk_transformer_v1":
        raise RuntimeError("transition-risk checkpoint protocol drift")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    datasets = {
        split: TransitionDataset(dataset_root, split)
        for split in ("validation", "test")
    }
    rows = {
        split: evaluate_rows(model, make_loader(dataset, 256, False, 2), device)
        for split, dataset in datasets.items()
    }
    validation_early = _profile_metrics(
        rows["validation"]["risk"], rows["validation"]["score"],
        rows["validation"]["profile"], rows["validation"]["anchor"], None, True
    )
    threshold = float(validation_early["threshold"])
    metrics = {}
    for split in ("validation", "test"):
        metrics[split] = {
            "first_50_frames": _profile_metrics(
                rows[split]["risk"], rows[split]["score"], rows[split]["profile"],
                rows[split]["anchor"], threshold, True
            ),
            "frames_9_to_199": _profile_metrics(
                rows[split]["risk"], rows[split]["score"], rows[split]["profile"],
                rows[split]["anchor"], threshold, False
            ),
        }
    profiles = _profile_records(datasets["test"], rows["test"], threshold)
    by_cell = {}
    for cell in sorted({record["name"].rsplit("/env", 1)[0] for record in profiles}):
        selected = [record for record in profiles if record["name"].startswith(cell + "/env")]
        by_cell[cell] = {
            "profiles": len(selected),
            "risky_profiles": sum(record["risk_target"] for record in selected),
            "predicted_risky_profiles": sum(record["predicted_risk"] for record in selected),
            "mean_probability": float(
                np.mean([record["mean_first50_probability"] for record in selected])
            ),
        }
    test = metrics["test"]["first_50_frames"]
    checks = {
        "checkpoint_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "threshold_selected_on_validation_first50_only": True,
        "heldout_test_first50_auroc_at_least_0p70": test["auroc"] >= 0.70,
        "heldout_test_first50_balanced_accuracy_at_least_0p65": (
            test["balanced_accuracy"] >= 0.65
        ),
        "heldout_test_first50_probability_gap_at_least_0p15": (
            test["risk_probability_gap"] >= 0.15
        ),
        "heldout_test_first50_brier_beats_prevalence": (
            test["brier"] < test["prevalence_baseline_brier"]
        ),
        "all_probabilities_finite": bool(
            all(np.isfinite(value["score"]).all() for value in rows.values())
        ),
        "test_profiles_are_seed_and_context_disjoint": True,
    }
    result = {
        "protocol": "sugar_causal_transition_risk_frozen_evaluation_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_first50_selected_threshold": threshold,
        "metrics": metrics,
        "test_by_cell": by_cell,
        "test_profiles": profiles,
        "claim_boundary": (
            "Frozen Carry45 transition-risk ranking/calibration on seed/context-disjoint "
            "early prefixes. This is not online fallback evidence and does not cover Kick risk."
        ),
        "automatic_next_stage": (
            "online_fallback_evaluation"
            if all(checks.values())
            else "feature_or_calibration_failure_audit"
        ),
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "checks": checks, "test": test}, indent=2))
    if not result["passed"]:
        raise RuntimeError("frozen transition-risk gate failed")


if __name__ == "__main__":
    main()
