#!/usr/bin/env python3
"""Paired Plan-15 comparison for completed Z, P, and PS frozen sweeps."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean


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
    "strict_sugar_event_eligible": ("all", "higher"),
    "acceptable_hold_or_safe_lower": ("joint_eligible", "higher"),
    "hold_success": ("joint_eligible", "higher"),
    "strict_sugar_hold_success": ("all", "higher"),
    "drop": ("joint_eligible", "lower"),
    "safe_lower": ("joint_eligible", "higher"),
    "robot_fall": ("all", "lower"),
    "reference_robot_deviation": ("all", "lower"),
    "maximum_height_loss_m": ("joint_eligible", "lower"),
    "bilateral_patch_contact_fraction": ("joint_eligible", "higher"),
    "gross_slip_patch_fraction": ("joint_eligible", "lower"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--z-root", type=Path, nargs="+", required=True)
parser.add_argument("--p-root", type=Path, nargs="+", required=True)
parser.add_argument("--ps-root", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def metric_value(row: dict[str, object], metric: str) -> float | None:
    if metric == "event_eligible":
        return float(bool(row["eligible_post_jump_window"]))
    if metric == "strict_sugar_event_eligible":
        return float(bool(row["strict_sugar_eligible_post_jump_window"]))
    if metric == "acceptable_hold_or_safe_lower":
        return float(bool(row["hold_success"]) or bool(row["safe_lower"]))
    if metric in {
        "hold_success",
        "strict_sugar_hold_success",
        "drop",
        "safe_lower",
        "robot_fall",
        "reference_robot_deviation",
    }:
        return float(bool(row[metric]))
    value = row.get(metric)
    return None if value is None else float(value)


def load_branch(
    roots: list[Path], branch: str
) -> dict[tuple[int, int, float, int], dict[str, object]]:
    rows: dict[tuple[int, int, float, int], dict[str, object]] = {}
    paths = sorted(
        path for root in roots for path in root.glob("*/summary.json")
    )
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["branch"] != branch:
            raise ValueError(f"{path} contains branch {summary['branch']}, expected {branch}")
        if summary.get("evaluation_view") != "strict_sugar_reference":
            raise ValueError(
                f"{path} is not a strict SUGAR evaluation; continued "
                "physical-outcome diagnostics are not scoreable"
            )
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


def exact_seed_sign_flip_pvalue(seed_differences: list[float]) -> float:
    """Two-sided exact paired randomization test over independent train seeds."""

    observed = abs(mean(seed_differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(seed_differences)):
        randomized = [
            value * sign for value, sign in zip(seed_differences, signs)
        ]
        extreme += abs(mean(randomized)) >= observed - 1.0e-12
        total += 1
    return float(extreme / total)


def compare(
    first: dict[tuple[int, int, float, int], dict[str, object]],
    second: dict[tuple[int, int, float, int], dict[str, object]],
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
            arrays = {seed: values for seed, values in by_seed.items() if values}
            count = sum(len(values) for values in arrays.values())
            if count == 0 or set(arrays) != set(TRAIN_TO_EVAL):
                factor_result[metric] = {
                    "paired_profiles": count,
                    "mean_difference_first_minus_second": None,
                    "paired_training_seed_differences": None,
                    "exact_seed_sign_flip_pvalue": None,
                    "holm_familywise_pvalue": None,
                    "better_direction": better,
                }
                continue
            seed_differences = [mean(arrays[seed]) for seed in sorted(arrays)]
            difference = mean(seed_differences)
            factor_result[metric] = {
                "paired_profiles": count,
                "mean_difference_first_minus_second": difference,
                "paired_training_seed_differences": seed_differences,
                "exact_seed_sign_flip_pvalue": exact_seed_sign_flip_pvalue(
                    seed_differences
                ),
                "holm_familywise_pvalue": None,
                "better_direction": better,
            }
        result[str(factor)] = factor_result
    return result


def apply_holm_familywise_correction(
    comparisons: dict[str, object],
) -> int:
    """Adjust the complete comparison/factor/metric family in one step."""

    tests: list[tuple[float, dict[str, object]]] = []
    for comparison in comparisons.values():
        for factor in comparison.values():
            for metric in factor.values():
                pvalue = metric["exact_seed_sign_flip_pvalue"]
                if pvalue is not None:
                    tests.append((float(pvalue), metric))
    tests.sort(key=lambda item: item[0])
    family_size = len(tests)
    running = 0.0
    for rank, (pvalue, metric) in enumerate(tests):
        adjusted = min(1.0, pvalue * (family_size - rank))
        running = max(running, adjusted)
        metric["holm_familywise_pvalue"] = running
    return family_size


def main() -> None:
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
    comparisons = {
        "P-Z": compare(branches["P"], branches["Z"]),
        "PS-P": compare(branches["PS"], branches["P"]),
        "PS-Z": compare(branches["PS"], branches["Z"]),
    }
    family_size = apply_holm_familywise_correction(comparisons)
    output = {
        "schema": "plan15_paired_frozen_comparison_v3_strict_seed_randomization",
        "profile_count_per_branch": expected_profiles,
        "inference": {
            "independent_unit": "training seed",
            "method": "two-sided exact paired sign-flip randomization",
            "training_seed_count": len(TRAIN_TO_EVAL),
            "multiple_comparison_correction": "Holm familywise correction",
            "family_size": family_size,
            "minimum_attainable_two_sided_raw_pvalue": 0.25,
            "confidence_intervals": None,
            "reason_no_interval": (
                "three independent training seeds are insufficient for a "
                "reliable BCa or cluster-bootstrap confidence interval"
            ),
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
