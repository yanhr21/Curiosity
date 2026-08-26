#!/usr/bin/env python3
"""Combine seen and interleaved frozen temporal-composer evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COUNT_FIELDS = (
    "kick_success_count",
    "physical_fall_count",
    "safe_kick_success_count",
)
MEAN_FIELDS = (
    "mean_mean_reward",
    "mean_planar_object_net_displacement_m",
    "mean_planar_object_path_m",
    "mean_any_foot_box_contact_fraction",
    "mean_maximum_robot_root_height_loss_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interleaved", type=Path, required=True)
    parser.add_argument("--seen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol")
        != "sugar_multi_context_transition_recovery_diagnostic_v1"
        or result.get("policy_topology")
        != "causal_temporal_action_composition"
    ):
        raise RuntimeError(f"invalid temporal-composer result: {path}")
    return result


def main() -> None:
    args = parse_args()
    interleaved = load(args.interleaved)
    seen = load(args.seen)
    for key in ("training_seed", "evaluation_seed", "training_prefix_schedule"):
        if interleaved[key] != seen[key]:
            raise RuntimeError(f"seen/interleaved comparison drift: {key}")
    if (
        interleaved["evaluation_prefix_schedule"] != [37, 45, 53, 61]
        or seen["evaluation_prefix_schedule"] != [33, 41, 49, 57, 65]
        or interleaved["training_prefix_schedule"] != [33, 41, 49, 57, 65]
        or interleaved.get("checks", {}).get(
            "evaluation_prefixes_disjoint_from_training"
        )
        is not True
        or seen.get("checks", {}).get("evaluation_prefixes_seen_in_training")
        is not True
    ):
        raise RuntimeError("complete prefix-grid contract failed")
    contexts = len(interleaved["contexts"]), len(seen["contexts"])
    total_contexts = sum(contexts)
    totals = {
        endpoint: {
            field: int(
                interleaved["count_totals"][endpoint][field]
                + seen["count_totals"][endpoint][field]
            )
            for field in COUNT_FIELDS
        }
        for endpoint in ("learned_kick", "exact_pre_update_kick")
    }
    mean_delta = {
        field: float(
            (
                contexts[0] * interleaved["mean_learned_minus_pre_update"][field]
                + contexts[1] * seen["mean_learned_minus_pre_update"][field]
            )
            / total_contexts
        )
        for field in MEAN_FIELDS
    }
    learned = totals["learned_kick"]
    pre = totals["exact_pre_update_kick"]
    combined_improvement = bool(
        (
            learned["safe_kick_success_count"] > pre["safe_kick_success_count"]
            and learned["physical_fall_count"] <= pre["physical_fall_count"]
        )
        or (
            learned["physical_fall_count"] < pre["physical_fall_count"]
            and learned["safe_kick_success_count"] >= pre["safe_kick_success_count"]
        )
    )
    interleaved_improvement = bool(
        interleaved["checks"]["aggregate_kick_safety_improvement"]
    )
    admitted = bool(combined_improvement and interleaved_improvement)
    result = {
        "protocol": "sugar_causal_temporal_composer_complete_prefix_grid_v1",
        "policy_topology": "causal_temporal_action_composition",
        "training_seed": interleaved["training_seed"],
        "evaluation_seed": interleaved["evaluation_seed"],
        "training_prefixes": [33, 41, 49, 57, 65],
        "complete_evaluation_prefixes": list(range(33, 66, 4)),
        "count_totals": totals,
        "weighted_mean_learned_minus_pre_update": mean_delta,
        "checks": {
            "same_checkpoint_seed_budget_and_training_schedule": True,
            "seen_and_interleaved_cover_complete_step4_grid": True,
            "interleaved_physical_safety_improvement": interleaved_improvement,
            "combined_physical_safety_improvement": combined_improvement,
            "temporal_controller_admitted": admitted,
        },
        "conclusion": (
            "temporal_controller_improves_seen_and_interleaved_prefix_safety"
            if admitted
            else "temporal_controller_does_not_generalize_across_complete_prefix_grid"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
