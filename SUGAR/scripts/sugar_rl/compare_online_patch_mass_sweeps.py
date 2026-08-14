#!/usr/bin/env python3
"""Paired Plan-15 comparison for completed Z, P, and PS frozen sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TRAIN_TO_EVAL = {151014: 152014, 151015: 152015, 151016: 152016}
FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)
EXPECTED_KEYS = {
    (train_seed, eval_seed, factor, profile)
    for train_seed, eval_seed in TRAIN_TO_EVAL.items()
    for factor in FACTORS
    for profile in range(20)
}
METRICS = {
    "event_eligible": ("all", "higher"),
    "acceptable_hold_or_safe_lower": ("joint_eligible", "higher"),
    "hold_success": ("joint_eligible", "higher"),
    "drop": ("joint_eligible", "lower"),
    "safe_lower": ("joint_eligible", "higher"),
    "robot_fall": ("all", "lower"),
    "maximum_height_loss_m": ("joint_eligible", "lower"),
    "bilateral_patch_contact_fraction": ("joint_eligible", "higher"),
    "gross_slip_patch_fraction": ("joint_eligible", "lower"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--z-root", type=Path, required=True)
parser.add_argument("--p-root", type=Path, required=True)
parser.add_argument("--ps-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bootstrap-samples", type=int, default=10_000)
parser.add_argument("--bootstrap-seed", type=int, default=153015)
args = parser.parse_args()


def metric_value(row: dict[str, object], metric: str) -> float | None:
    if metric == "event_eligible":
        return float(bool(row["eligible_post_jump_window"]))
    if metric == "acceptable_hold_or_safe_lower":
        return float(bool(row["hold_success"]) or bool(row["safe_lower"]))
    if metric in {"hold_success", "drop", "safe_lower", "robot_fall"}:
        return float(bool(row[metric]))
    value = row.get(metric)
    return None if value is None else float(value)


def load_branch(root: Path, branch: str) -> dict[tuple[int, int, float, int], dict[str, object]]:
    rows: dict[tuple[int, int, float, int], dict[str, object]] = {}
    for path in sorted(root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["branch"] != branch:
            raise ValueError(f"{path} contains branch {summary['branch']}, expected {branch}")
        train_seed = summary.get("training_seed")
        if train_seed is None:
            raise ValueError(f"{path} does not record its training seed")
        train_seed = int(train_seed)
        eval_seed = int(summary["seed"])
        if TRAIN_TO_EVAL.get(train_seed) != eval_seed:
            raise ValueError(f"unexpected train/evaluation seed pairing in {path}")
        factor = float(summary["mass_factor"])
        if factor not in FACTORS:
            raise ValueError(f"unexpected mass factor {factor} in {path}")
        for profile, episode in enumerate(summary["episodes"]):
            key = (train_seed, eval_seed, factor, profile)
            if key in rows:
                raise ValueError(f"duplicate profile {key}")
            rows[key] = episode
    if set(rows) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - set(rows))
        unexpected = sorted(set(rows) - EXPECTED_KEYS)
        raise ValueError(
            f"{branch} does not contain the exact 3x5x20 matched design: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return rows


def hierarchical_interval(
    differences: dict[int, np.ndarray], samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    seeds = np.asarray(sorted(differences), dtype=np.int64)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        values = []
        for seed in selected_seeds:
            block = differences[int(seed)]
            values.append(rng.choice(block, size=len(block), replace=True))
        draws[index] = np.concatenate(values).mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def compare(
    first: dict[tuple[int, int, float, int], dict[str, object]],
    second: dict[tuple[int, int, float, int], dict[str, object]],
    rng: np.random.Generator,
) -> dict[str, object]:
    if set(first) != set(second):
        raise ValueError("frozen sweep profile keys do not match across branches")
    result: dict[str, object] = {}
    for factor in FACTORS:
        factor_rows = sorted(key for key in first if key[2] == factor)
        factor_result = {}
        for metric, (population, better) in METRICS.items():
            by_seed: dict[int, list[float]] = {}
            for key in factor_rows:
                left = first[key]
                right = second[key]
                if population == "joint_eligible" and not (
                    left["eligible_post_jump_window"]
                    and right["eligible_post_jump_window"]
                ):
                    continue
                left_value = metric_value(left, metric)
                right_value = metric_value(right, metric)
                if left_value is None or right_value is None:
                    continue
                by_seed.setdefault(key[0], []).append(left_value - right_value)
            arrays = {
                seed: np.asarray(values, dtype=np.float64)
                for seed, values in by_seed.items()
                if values
            }
            count = sum(len(values) for values in arrays.values())
            if count == 0 or set(arrays) != set(TRAIN_TO_EVAL):
                factor_result[metric] = {
                    "paired_profiles": count,
                    "mean_difference_first_minus_second": None,
                    "hierarchical_bootstrap_95ci": None,
                    "better_direction": better,
                }
                continue
            difference = float(np.concatenate(list(arrays.values())).mean())
            low, high = hierarchical_interval(
                arrays, int(args.bootstrap_samples), rng
            )
            factor_result[metric] = {
                "paired_profiles": count,
                "mean_difference_first_minus_second": difference,
                "hierarchical_bootstrap_95ci": [low, high],
                "better_direction": better,
            }
        result[str(factor)] = factor_result
    return result


def main() -> None:
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    branches = {
        "Z": load_branch(args.z_root, "Z"),
        "P": load_branch(args.p_root, "P"),
        "PS": load_branch(args.ps_root, "PS"),
    }
    expected_profiles = len(EXPECTED_KEYS)
    for branch, rows in branches.items():
        if len(rows) != expected_profiles:
            raise ValueError(
                f"{branch} contains {len(rows)} profiles, expected {expected_profiles}"
            )
    rng = np.random.default_rng(int(args.bootstrap_seed))
    output = {
        "schema": "plan15_paired_frozen_comparison_v1",
        "profile_count_per_branch": expected_profiles,
        "bootstrap": {
            "method": "paired hierarchical percentile bootstrap: training seed then profile",
            "samples": int(args.bootstrap_samples),
            "seed": int(args.bootstrap_seed),
        },
        "comparisons": {
            "P-Z": compare(branches["P"], branches["Z"], rng),
            "PS-P": compare(branches["PS"], branches["P"], rng),
            "PS-Z": compare(branches["PS"], branches["Z"], rng),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
