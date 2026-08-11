#!/usr/bin/env python3
"""Independently reconstruct the held-out tactile teacher-residual result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


BASE_WIDTH = 890
ALLOWED_CHANGED = {
    "actor.0.weight",
    "actor_tactile_encoder.convolution.0.weight",
    "actor_tactile_encoder.convolution.1.weight",
    "actor_tactile_encoder.convolution.3.weight",
    "actor_tactile_encoder.convolution.4.weight",
    "actor_tactile_encoder.convolution.6.weight",
    "actor_tactile_encoder.convolution.7.weight",
    "actor_tactile_encoder.projection.1.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def prediction_metrics(path: Path) -> dict[str, float | int]:
    with np.load(path, allow_pickle=False) as arrays:
        teacher = arrays["teacher_action"]
        live = arrays["live_action"]
        zero = arrays["zero_action"]
        permuted = arrays["patch_permuted_action"]
        return {
            "rows": int(teacher.shape[0]),
            "values": int(teacher.size),
            "live_mae": float(np.mean(np.abs(live - teacher), dtype=np.float64)),
            "zero_baseline_mae": float(
                np.mean(np.abs(zero - teacher), dtype=np.float64)
            ),
            "patch_permuted_mae": float(
                np.mean(np.abs(permuted - teacher), dtype=np.float64)
            ),
            "rows_live_better_than_zero": int(
                np.count_nonzero(
                    np.mean(np.abs(live - teacher), axis=-1)
                    < np.mean(np.abs(zero - teacher), axis=-1)
                )
            ),
        }


def close(left: float, right: float, tolerance: float = 2.0e-8) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    args = parse_args()
    root = args.result_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = json.loads((root / "training/report.json").read_text(encoding="utf-8"))
    source_path = Path(report["model"]["initial_checkpoint"])
    final_path = root / "training/model_best.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    final_container = torch.load(final_path, map_location="cpu", weights_only=False)
    final = final_container["model_state_dict"]
    if source.keys() != final.keys():
        raise RuntimeError("source and selected checkpoint tensor keys differ")
    changed = {
        name for name in source if not torch.equal(source[name], final[name])
    }
    base_columns_equal = torch.equal(
        source["actor.0.weight"][:, :BASE_WIDTH],
        final["actor.0.weight"][:, :BASE_WIDTH],
    )
    encoder_biases_zero = all(
        torch.count_nonzero(value).item() == 0
        for name, value in final.items()
        if name.startswith("actor_tactile_encoder.") and name.endswith("bias")
    )

    prediction_paths = [
        Path(report["prediction_records"]["selection"]),
        *[Path(path) for path in report["prediction_records"]["tests"]],
    ]
    reconstructed = [prediction_metrics(path) for path in prediction_paths]
    recorded = [report["metrics"]["selection"], *report["metrics"]["tests"]]
    metric_checks = []
    for rebuilt, expected in zip(reconstructed, recorded, strict=True):
        metric_checks.append(
            {
                "rows_equal": rebuilt["rows"] == expected["rows"],
                "values_equal": rebuilt["values"] == expected["values"],
                "live_mae_equal": close(rebuilt["live_mae"], expected["live_mae"]),
                "zero_mae_equal": close(
                    rebuilt["zero_baseline_mae"], expected["zero_baseline_mae"]
                ),
                "permuted_mae_equal": close(
                    rebuilt["patch_permuted_mae"], expected["patch_permuted_mae"]
                ),
                "rows_better_equal": (
                    rebuilt["rows_live_better_than_zero"]
                    == expected["rows_live_better_than_zero"]
                ),
            }
        )

    test_values = sum(int(row["values"]) for row in reconstructed[1:])
    aggregate_live = sum(
        float(row["live_mae"]) * int(row["values"])
        for row in reconstructed[1:]
    ) / test_values
    aggregate_zero = sum(
        float(row["zero_baseline_mae"]) * int(row["values"])
        for row in reconstructed[1:]
    ) / test_values
    split_conditions = [
        row["condition"]["label"]
        for group in ("train", "selection", "test")
        for row in report["split"][group]
    ]
    checks = {
        "five_condition_labels_are_distinct": len(set(split_conditions)) == 5,
        "selected_checkpoint_is_after_step_zero": report["optimization"]["best_step"] > 0,
        "only_declared_adapter_tensors_changed": changed == ALLOWED_CHANGED,
        "official_actor_base_columns_are_bitwise_equal": base_columns_equal,
        "zero_preserving_encoder_biases_are_zero": encoder_biases_zero,
        "checkpoint_is_rsl_runner_loadable": (
            final_container.get("iter") == 0
            and isinstance(final_container.get("infos"), dict)
        ),
        "all_saved_prediction_metrics_reconstruct": all(
            all(row.values()) for row in metric_checks
        ),
        "selection_live_beats_zero": (
            reconstructed[0]["live_mae"] < reconstructed[0]["zero_baseline_mae"]
        ),
        "each_test_live_beats_zero": all(
            row["live_mae"] < row["zero_baseline_mae"]
            for row in reconstructed[1:]
        ),
        "each_test_live_beats_patch_permutation": all(
            row["live_mae"] < row["patch_permuted_mae"]
            for row in reconstructed[1:]
        ),
        "heldout_aggregate_live_beats_zero": aggregate_live < aggregate_zero,
        "heldout_aggregate_matches_report": (
            close(aggregate_live, report["heldout_aggregate"]["live_mae"])
            and close(
                aggregate_zero,
                report["heldout_aggregate"]["zero_baseline_mae"],
            )
        ),
        "reported_gate_passed": bool(report["gate_passed"]),
    }
    result = {
        "schema": "native_tactile_heldout_teacher_residual_independent_audit_v1",
        "result_root": str(root),
        "changed_tensors": sorted(changed),
        "reconstructed_prediction_metrics": reconstructed,
        "metric_checks": metric_checks,
        "heldout_aggregate": {
            "live_mae": aggregate_live,
            "zero_baseline_mae": aggregate_zero,
            "relative_mae_reduction": (aggregate_zero - aggregate_live) / aggregate_zero,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"independent teacher-residual audit failed: {checks}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
