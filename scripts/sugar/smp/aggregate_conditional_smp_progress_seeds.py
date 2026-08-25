#!/usr/bin/env python3
"""Aggregate matched conditional-TinyMDM reward pairs across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pair", type=Path, action="append", required=True)
parser.add_argument(
    "--reward-mode",
    choices=("progress", "contrastive_progress"),
    default="progress",
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


COUNT_FIELDS = (
    "kick_success_count",
    "physical_fall_count",
    "safe_kick_success_count",
)
CONTINUOUS_FIELDS = (
    "mean_mean_reward",
    "mean_planar_object_net_displacement_m",
    "mean_planar_object_path_m",
    "mean_any_foot_box_contact_fraction",
    "mean_maximum_robot_root_height_loss_m",
)


def main() -> None:
    if len(args.pair) < 2:
        raise RuntimeError("multi-seed aggregation requires at least two matched pairs")
    pairs = []
    seen_seeds: set[int] = set()
    for path in args.pair:
        result = json.loads(path.read_text(encoding="utf-8"))
        experiment = result.get("experiment", {})
        checks = result.get("checks", {})
        seed = int(experiment.get("same_training_seed", -1))
        if (
            result.get("protocol")
            != "sugar_conditional_smp_recovery_correct_wrong_matched_pair_v1"
            or experiment.get("reward_mode") != args.reward_mode
            or checks.get("matched_frozen_evaluation") is not True
            or checks.get("only_semantic_class_differs") is not True
            or seed < 0
            or seed in seen_seeds
        ):
            raise RuntimeError(f"invalid or duplicate matched reward pair: {path}")
        seen_seeds.add(seed)
        pairs.append(result)

    count_totals = {
        arm: {
            field: sum(int(pair[arm][field]) for pair in pairs)
            for field in COUNT_FIELDS
        }
        for arm in ("correct", "wrong")
    }
    count_deltas = {
        field: count_totals["correct"][field] - count_totals["wrong"][field]
        for field in COUNT_FIELDS
    }
    mean_deltas = {
        field: sum(float(pair["correct_minus_wrong"][field]) for pair in pairs)
        / len(pairs)
        for field in CONTINUOUS_FIELDS
    }
    sign_counts = {
        field: {
            "correct_better": sum(
                float(pair["correct_minus_wrong"][field]) > 0.0 for pair in pairs
            ),
            "equal": sum(
                float(pair["correct_minus_wrong"][field]) == 0.0 for pair in pairs
            ),
            "correct_worse": sum(
                float(pair["correct_minus_wrong"][field]) < 0.0 for pair in pairs
            ),
        }
        for field in (*COUNT_FIELDS, *CONTINUOUS_FIELDS)
    }
    # For these cost fields, a negative correct-minus-wrong delta is better.
    for field in (
        "physical_fall_count",
        "mean_planar_object_path_m",
        "mean_any_foot_box_contact_fraction",
        "mean_maximum_robot_root_height_loss_m",
    ):
        signs = sign_counts[field]
        signs["correct_better"], signs["correct_worse"] = (
            signs["correct_worse"],
            signs["correct_better"],
        )

    per_seed_physical_advantage = [
        bool(pair["checks"]["correct_condition_has_physical_advantage"])
        for pair in pairs
    ]
    result = {
        "protocol": (
            "sugar_conditional_smp_progress_three_seed_aggregate_v1"
            if args.reward_mode == "progress"
            else "sugar_conditional_smp_contrastive_progress_seed_aggregate_v1"
        ),
        "reward_mode": args.reward_mode,
        "training_seeds": sorted(seen_seeds),
        "num_training_seeds": len(pairs),
        "profiles_per_arm": 20 * len(pairs),
        "count_totals": count_totals,
        "correct_minus_wrong_count_totals": count_deltas,
        "mean_correct_minus_wrong": mean_deltas,
        "per_seed_sign_counts": sign_counts,
        "per_seed_physical_advantage": per_seed_physical_advantage,
        "checks": {
            "condition_effect_replicated_all_seeds": all(
                pair["checks"]["behavior_difference_detected"] for pair in pairs
            ),
            "physical_advantage_replicated_all_seeds": all(
                per_seed_physical_advantage
            ),
            "safe_kick_never_worse": all(
                float(pair["correct_minus_wrong"]["safe_kick_success_count"])
                >= 0.0
                for pair in pairs
            ),
            "fall_count_never_worse": all(
                float(pair["correct_minus_wrong"]["physical_fall_count"]) <= 0.0
                for pair in pairs
            ),
        },
        "conclusion": (
            "seed_robust_matched_physical_advantage"
            if all(per_seed_physical_advantage)
            else "condition_effect_replicated_without_seed_robust_physical_advantage"
        ),
        "claim_boundary": (
            f"{len(pairs)} matched fixed-prefix training seeds. Aggregate profile counts "
            "do not replace training-seed replication, and no general demo-following "
            "claim follows."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
