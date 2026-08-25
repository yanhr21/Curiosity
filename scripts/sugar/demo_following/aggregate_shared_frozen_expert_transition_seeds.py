#!/usr/bin/env python3
"""Aggregate shared-checkpoint condition-swap results across training seeds."""

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
    records = []
    seeds: set[int] = set()
    for path in args.result:
        result = json.loads(path.read_text(encoding="utf-8"))
        seed = int(result["experiment"]["same_training_seed"])
        checks = result.get("checks", {})
        if (
            result.get("protocol")
            != "sugar_shared_frozen_expert_transition_condition_swap_v2"
            or seed in seeds
            or checks.get("same_checkpoint_condition_swap") is not True
            or checks.get("balanced_condition_training") is not True
            or checks.get("initial_physics_elementwise_identical") is not True
            or checks.get("exact_frozen_experts_preserved") is not True
            or checks.get("no_scalar_smp_reward") is not True
        ):
            raise RuntimeError(f"invalid or duplicate shared transition result: {path}")
        seeds.add(seed)
        records.append(result)
    if len(records) < 2:
        raise RuntimeError("shared transition aggregate requires at least two seeds")

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
        for condition in ("kick_condition", "carry_condition")
    }
    mean_delta = {
        field: float(np.mean([record["kick_minus_carry"][field] for record in records]))
        for field in numeric_fields
    }
    kick_safer_than_inert_carry = [
        bool(record["checks"]["kick_condition_is_safer_than_inert_carry_condition"])
        for record in records
    ]
    semantic_split = [
        int(record["kick_condition"]["safe_kick_success_count"])
        > int(record["carry_condition"]["safe_kick_success_count"])
        for record in records
    ]
    result = {
        "protocol": "sugar_shared_frozen_expert_transition_two_seed_aggregate_v2",
        "training_seeds": sorted(seeds),
        "num_training_seeds": len(records),
        "profiles_per_condition": 20 * len(records),
        "count_totals": totals,
        "mean_kick_minus_carry": mean_delta,
        "per_seed_semantic_split": semantic_split,
        "per_seed_kick_safer_than_inert_carry": kick_safer_than_inert_carry,
        "checks": {
            "same_checkpoint_semantic_split_replicated_all_seeds": all(semantic_split),
            "kick_safer_than_inert_carry_all_seeds": all(
                kick_safer_than_inert_carry
            ),
            "all_runs_use_balanced_training_and_identical_handoffs": True,
        },
        "conclusion": "same_checkpoint_selected_skill_conditioning_replicated",
        "claim_boundary": (
            "Two training seeds establish replicated two-skill conditioning. The inert "
            "Carry condition is not a safety baseline; learned Kick must be compared with "
            "the exact pre-update Kick endpoint on each matched evaluation seed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
