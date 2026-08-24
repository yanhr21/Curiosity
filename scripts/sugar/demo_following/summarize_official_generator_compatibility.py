#!/usr/bin/env python3
"""Audit whether released Generator training ranges detect unsafe context early."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stats(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "q50": float(np.quantile(value, 0.50)),
        "q95": float(np.quantile(value, 0.95)),
        "maximum": float(np.max(value)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    compatible = np.load(root / "compatible_kick_on_small/TRACE.npz", allow_pickle=False)
    incompatible = np.load(root / "incompatible_carry_on_big/TRACE.npz", allow_pickle=False)
    compatible_result = json.loads(
        (root / "compatible_kick_on_small/RESULT.json").read_text(encoding="utf-8")
    )
    incompatible_result = json.loads(
        (root / "incompatible_carry_on_big/RESULT.json").read_text(encoding="utf-8")
    )
    early = slice(0, 100)
    key_max = "selected_generator_normalized_max_abs"
    key_fraction = "selected_generator_outside_train_range_fraction"
    cmax = compatible[key_max][early]
    imax = incompatible[key_max][early]
    cfrac = compatible[key_fraction][early]
    ifrac = incompatible[key_fraction][early]
    checks = {
        "compatible_arm_is_exact_released_kick_behavior": (
            compatible_result["aggregate"]["kick_success_count"] == 20
            and compatible_result["safe_fallback_fraction"] == 0.0
        ),
        "incompatible_arm_reproduces_cross_geometry_failure": (
            not incompatible_result["passed"]
        ),
        "all_scores_finite": bool(
            np.isfinite(cmax).all()
            and np.isfinite(imax).all()
            and np.isfinite(cfrac).all()
            and np.isfinite(ifrac).all()
        ),
        "exact_100_frame_predeclared_early_window": cmax.shape == imax.shape == (100, 20),
    }
    separation = {
        "normalized_max_abs": {
            "compatible": _stats(cmax),
            "incompatible": _stats(imax),
        },
        "outside_training_minmax_fraction": {
            "compatible": _stats(cfrac),
            "incompatible": _stats(ifrac),
        },
    }
    natural_range_separates = bool(np.min(ifrac) > np.max(cfrac))
    summary = {
        "protocol": "sugar_official_generator_normalizer_compatibility_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "early_window_frames": 100,
        "separation": separation,
        "natural_training_range_strictly_separates": natural_range_separates,
        "policy_gate_supported": natural_range_separates,
        "claim_boundary": (
            "Uses only the selected released Generator's own causal input and released "
            "normalizer statistics. This is an OOD audit, not a learned skill latent."
        ),
    }
    (root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise RuntimeError("Generator compatibility audit is structurally invalid")


if __name__ == "__main__":
    main()
