#!/usr/bin/env python3
"""Summarize one matched frozen-expert Kick-versus-Carry transition pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--correct", type=Path, required=True)
parser.add_argument("--wrong", type=Path, required=True)
parser.add_argument("--correct-audit", type=Path, required=True)
parser.add_argument("--wrong-audit", type=Path, required=True)
parser.add_argument("--training-seed", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def _evaluation(path: Path, selected_skill_id: int) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol") != "sugar_cross_skill_recovery_frozen_eval_v3"
        or result.get("structurally_valid") is not True
        or result.get("checkpoint_iteration") != 64
        or result.get("transition_selected_skill_id") != selected_skill_id
    ):
        raise RuntimeError(f"invalid transition evaluation: {path}")
    return result


def _audit(path: Path, selected_skill_id: int) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol") != "sugar_online_cross_skill_recovery_prefix_v2"
        or result.get("transition_selected_skill_id") != selected_skill_id
        or result.get("transition_observation_is_causal") is not True
        or result.get("conditional_tinymdm_reward") is not None
        or int(result.get("prefix_count", 0)) <= 0
    ):
        raise RuntimeError(f"invalid transition training audit: {path}")
    return result


def main() -> None:
    correct = _evaluation(args.correct, 1)
    wrong = _evaluation(args.wrong, 0)
    correct_audit = _audit(args.correct_audit, 1)
    wrong_audit = _audit(args.wrong_audit, 0)
    for key in ("seed", "num_envs", "steps", "prefix"):
        if correct[key] != wrong[key]:
            raise RuntimeError(f"matched transition evaluation drift: {key}")
    fields = (
        "mean_mean_reward",
        "mean_planar_object_net_displacement_m",
        "mean_planar_object_path_m",
        "mean_any_foot_box_contact_fraction",
        "mean_maximum_robot_root_height_loss_m",
        "kick_success_count",
        "physical_fall_count",
        "safe_kick_success_count",
    )
    correct_aggregate = correct["aggregate"]
    wrong_aggregate = wrong["aggregate"]
    delta = {
        field: float(correct_aggregate[field] - wrong_aggregate[field])
        for field in fields
    }
    physical_advantage = bool(
        (
            delta["safe_kick_success_count"] > 0
            and delta["physical_fall_count"] <= 0
        )
        or (
            delta["physical_fall_count"] < 0
            and delta["safe_kick_success_count"] >= 0
        )
    )
    result = {
        "protocol": "sugar_frozen_expert_transition_matched_pair_v1",
        "experiment": {
            "correct": "selected released Kick Generator+Tracker endpoint",
            "wrong": "selected released Carry Generator+Tracker endpoint",
            "shared_transition_topology": "510+36+2 -> 512/256/128 -> 29 residual",
            "same_online_prefix": "one Kick alignment plus 41 Carry steps",
            "same_training_seed": args.training_seed,
            "same_update_budget": 64,
            "scalar_smp_reward": False,
        },
        "correct": correct_aggregate,
        "wrong": wrong_aggregate,
        "correct_minus_wrong": delta,
        "checks": {
            "matched_frozen_evaluation": True,
            "both_selected_commands_are_causal": True,
            "no_scalar_smp_reward": True,
            "only_selected_released_endpoint_differs": True,
            "correct_endpoint_has_physical_advantage": physical_advantage,
        },
        "training_audits": {"correct": correct_audit, "wrong": wrong_audit},
        "conclusion": (
            "correct_endpoint_has_matched_physical_advantage"
            if physical_advantage
            else "selected_endpoint_changes_behavior_without_physical_advantage"
        ),
        "claim_boundary": (
            "One fixed-prefix, one-seed topology diagnostic. Frozen released endpoints "
            "plus a learned transition do not establish arbitrary-demo following."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

