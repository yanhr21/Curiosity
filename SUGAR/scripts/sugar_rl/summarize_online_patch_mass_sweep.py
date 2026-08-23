#!/usr/bin/env python3
"""Summarize one completed Plan-15 Z/P/PS frozen sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


TRAIN_TO_EVAL = {151014: 152014, 151015: 152015, 151016: 152016}
FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def average(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return mean(values) if values else None


def main() -> None:
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_root.glob("*/summary.json"))
    ]
    if not summaries:
        raise FileNotFoundError(f"no summary.json files below {args.input_root}")

    branches = {item["branch"] for item in summaries}
    if len(branches) != 1:
        raise ValueError(f"input contains multiple branches: {sorted(branches)}")

    grouped: dict[float, list[dict[str, object]]] = {}
    run_keys = set()
    for summary in summaries:
        if summary.get("evaluation_view") != "strict_sugar_reference":
            raise ValueError(
                "formal Plan-15 sweep requires strict SUGAR termination; "
                "continued physical-outcome diagnostics are not scoreable"
            )
        train_seed = int(summary["training_seed"])
        eval_seed = int(summary["seed"])
        factor = float(summary["mass_factor"])
        if TRAIN_TO_EVAL.get(train_seed) != eval_seed or factor not in FACTORS:
            raise ValueError(
                "unexpected checkpoint/evaluation/factor tuple: "
                f"{train_seed}, {eval_seed}, {factor}"
            )
        key = (train_seed, eval_seed, factor)
        if key in run_keys:
            raise ValueError(f"duplicate frozen run: {key}")
        if len(summary["episodes"]) != 20:
            raise ValueError(
                f"{key} contains {len(summary['episodes'])} profiles, expected 20"
            )
        run_keys.add(key)
        grouped.setdefault(factor, []).extend(summary["episodes"])
    expected_run_keys = {
        (train_seed, eval_seed, factor)
        for train_seed, eval_seed in TRAIN_TO_EVAL.items()
        for factor in FACTORS
    }
    if run_keys != expected_run_keys:
        raise ValueError("input does not contain the exact 3x5 matched run set")

    factors = {}
    for factor, episodes in sorted(grouped.items()):
        eligible = [row for row in episodes if row["eligible_post_jump_window"]]
        strict_sugar_eligible = [
            row
            for row in episodes
            if row["strict_sugar_eligible_post_jump_window"]
        ]
        factors[str(factor)] = {
            "profiles": len(episodes),
            "eligible_profiles": len(eligible),
            "strict_sugar_eligible_profiles": len(strict_sugar_eligible),
            "hold_success_count": sum(bool(row["hold_success"]) for row in eligible),
            "strict_sugar_hold_success_count": sum(
                bool(row["strict_sugar_hold_success"])
                for row in strict_sugar_eligible
            ),
            "drop_count": sum(bool(row["drop"]) for row in eligible),
            "safe_lower_count": sum(bool(row["safe_lower"]) for row in eligible),
            "robot_fall_count": sum(bool(row["robot_fall"]) for row in episodes),
            "reference_robot_deviation_count": sum(
                bool(row["reference_robot_deviation"]) for row in episodes
            ),
            "mean_maximum_height_loss_m": average(
                eligible, "maximum_height_loss_m"
            ),
            "mean_bilateral_patch_contact_fraction": average(
                eligible, "bilateral_patch_contact_fraction"
            ),
            "mean_gross_slip_patch_fraction": average(
                eligible, "gross_slip_patch_fraction"
            ),
        }

    result = {
        "schema": "plan15_frozen_sweep_summary_v3_strict_sugar",
        "branch": next(iter(branches)),
        "source_runs": len(summaries),
        "profiles": sum(len(item["episodes"]) for item in summaries),
        "training_seeds": list(TRAIN_TO_EVAL),
        "evaluation_seeds": list(TRAIN_TO_EVAL.values()),
        "checkpoint_evaluation_pairing": {
            str(key): value for key, value in TRAIN_TO_EVAL.items()
        },
        "factors": factors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
