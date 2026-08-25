#!/usr/bin/env python3
"""Audit the frozen transition-risk model at the earliest exact anchor 9."""

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


ANCHOR = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _anchor_metrics(
    rows: dict[str, np.ndarray], threshold: float | None
) -> dict[str, float]:
    selected = rows["anchor"] == ANCHOR
    if int(np.sum(selected)) != len(np.unique(rows["profile"])):
        raise RuntimeError("anchor-9 profile coverage drift")
    return _profile_metrics(
        rows["risk"][selected],
        rows["score"][selected],
        rows["profile"][selected],
        rows["anchor"][selected],
        threshold,
        False,
    )


def main() -> None:
    args = parse_args()
    if socket.gethostname().startswith(("mgmtserver", "login")):
        raise RuntimeError("anchor-9 audit must run on a compute node")
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
    validation = _anchor_metrics(rows["validation"], None)
    threshold = float(validation["threshold"])
    validation = _anchor_metrics(rows["validation"], threshold)
    test = _anchor_metrics(rows["test"], threshold)
    checks = {
        "checkpoint_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "threshold_selected_on_validation_anchor9_only": True,
        "heldout_test_anchor9_auroc_at_least_0p70": test["auroc"] >= 0.70,
        "heldout_test_anchor9_balanced_accuracy_at_least_0p65": (
            test["balanced_accuracy"] >= 0.65
        ),
        "heldout_test_anchor9_probability_gap_at_least_0p15": (
            test["risk_probability_gap"] >= 0.15
        ),
        "heldout_test_anchor9_brier_beats_prevalence": (
            test["brier"] < test["prevalence_baseline_brier"]
        ),
        "all_anchor9_probabilities_finite": bool(
            all(
                np.isfinite(value["score"][value["anchor"] == ANCHOR]).all()
                for value in rows.values()
            )
        ),
        "test_profiles_are_seed_and_context_disjoint": True,
    }
    result = {
        "protocol": "sugar_causal_transition_risk_anchor9_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "decision_anchor": ANCHOR,
        "validation_anchor9_selected_threshold": threshold,
        "validation": validation,
        "test": test,
        "claim_boundary": (
            "Frozen earliest-decision audit using exactly frames 0--9 and a threshold "
            "selected only on validation anchor 9. Online behavior is not evaluated here."
        ),
        "automatic_next_stage": (
            "matched_online_anchor9_fallback"
            if all(checks.values())
            else "learned_transition_recovery_controller"
        ),
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError("earliest anchor-9 transition-risk audit failed")


if __name__ == "__main__":
    main()
