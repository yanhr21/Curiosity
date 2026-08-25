#!/usr/bin/env python3
"""Aggregate learned-versus-pre-update Kick comparisons across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def main() -> None:
    records: list[dict[str, object]] = []
    training_seeds: set[int] = set()
    evaluation_seeds: set[int] = set()
    for path in args.result:
        result = json.loads(path.read_text(encoding="utf-8"))
        training_seed = int(result["training_seed"])
        evaluation_seed = int(result["evaluation_seed"])
        checks = result.get("checks", {})
        if (
            result.get("protocol")
            != "sugar_shared_frozen_expert_transition_learning_v1"
            or training_seed in training_seeds
            or evaluation_seed in evaluation_seeds
            or checks.get("matched_evaluation_protocol") is not True
            or checks.get("initial_physics_elementwise_identical") is not True
            or checks.get("same_kick_condition") is not True
            or checks.get("exact_pre_update_endpoint") is not True
        ):
            raise RuntimeError(f"invalid or duplicate learning result: {path}")
        training_seeds.add(training_seed)
        evaluation_seeds.add(evaluation_seed)
        records.append(result)
    if len(records) < 2:
        raise RuntimeError("learning aggregate requires at least two seeds")

    count_fields = (
        "kick_success_count",
        "safe_kick_success_count",
        "physical_fall_count",
    )
    numeric_fields = (
        "mean_planar_object_net_displacement_m",
        "mean_planar_object_path_m",
        "mean_any_foot_box_contact_fraction",
        "mean_maximum_robot_root_height_loss_m",
        "mean_mean_reward",
    )
    totals = {
        condition: {
            field: int(sum(record[condition][field] for record in records))
            for field in count_fields
        }
        for condition in ("learned_kick", "exact_pre_update_kick")
    }
    mean_delta = {
        field: float(
            np.mean(
                [record["learned_minus_pre_update"][field] for record in records]
            )
        )
        for field in numeric_fields
    }
    improvements = [
        bool(record["checks"]["learned_kick_safety_improvement"])
        for record in records
    ]
    result = {
        "protocol": "sugar_shared_frozen_expert_transition_learning_aggregate_v1",
        "training_seeds": sorted(training_seeds),
        "evaluation_seeds": sorted(evaluation_seeds),
        "num_training_seeds": len(records),
        "profiles_per_endpoint": 20 * len(records),
        "count_totals": totals,
        "mean_learned_minus_pre_update": mean_delta,
        "per_seed_safety_improvement": improvements,
        "checks": {
            "independent_training_and_evaluation_seeds": True,
            "matched_exact_pre_update_comparison_all_seeds": True,
            "safety_improvement_replicated_all_seeds": all(improvements),
        },
        "conclusion": (
            "matched_kick_safety_improvement_replicated"
            if all(improvements)
            else "matched_kick_safety_improvement_not_replicated"
        ),
        "claim_boundary": (
            "Two seeds test the same released Carry/Kick endpoint pair at prefix41. "
            "This is a matched two-skill transition result, not arbitrary-video "
            "generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
