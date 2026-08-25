#!/usr/bin/env python3
"""Summarize the matched correct-Kick versus wrong-Carry SMP reward pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--correct", type=Path, required=True)
parser.add_argument("--wrong", type=Path, required=True)
parser.add_argument("--correct-audit", type=Path, required=True)
parser.add_argument("--wrong-audit", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def load_evaluation(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol") != "sugar_cross_skill_recovery_frozen_eval_v3"
        or result.get("structurally_valid") is not True
        or result.get("checkpoint_iteration") != 64
    ):
        raise RuntimeError(f"invalid update-64 frozen evaluation: {path}")
    return result


def load_audit(path: Path, expected_class: int) -> dict[str, object]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    reward = audit.get("conditional_tinymdm_reward")
    if (
        audit.get("protocol") != "sugar_online_cross_skill_recovery_prefix_v2"
        or not isinstance(reward, dict)
        or reward.get("protocol") != "sugar_online_conditional_tinymdm_reward_v1"
        or reward.get("class_id") != expected_class
        or int(reward.get("reward_calls", 0)) <= 0
        or audit.get("conditional_tinymdm_task_reward_weight") != 0.5
        or audit.get("conditional_tinymdm_smp_reward_weight") != 0.5
    ):
        raise RuntimeError(f"invalid online conditional reward audit: {path}")
    return audit


def main() -> None:
    correct = load_evaluation(args.correct)
    wrong = load_evaluation(args.wrong)
    correct_audit = load_audit(args.correct_audit, 1)
    wrong_audit = load_audit(args.wrong_audit, 0)
    for key in ("seed", "num_envs", "steps", "prefix"):
        if correct[key] != wrong[key]:
            raise RuntimeError(f"matched evaluation drift: {key}")
    correct_aggregate = correct["aggregate"]
    wrong_aggregate = wrong["aggregate"]
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
    delta = {
        name: float(correct_aggregate[name] - wrong_aggregate[name])
        for name in fields
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
    behavior_difference = any(abs(value) > 1.0e-8 for value in delta.values())
    result = {
        "protocol": "sugar_conditional_smp_recovery_correct_wrong_matched_pair_v1",
        "experiment": {
            "correct": "Kick condition (class 1), matching the post-prefix Kick task",
            "wrong": "Carry condition (class 0), conflicting with the post-prefix Kick task",
            "same_teacher": "released KickBox Tracker",
            "same_initialization": "one Kick alignment plus 41 online Carry steps",
            "same_training_seed": 171632,
            "same_update_budget": 64,
            "same_task_and_smp_weights": [0.5, 0.5],
        },
        "correct": correct_aggregate,
        "wrong": wrong_aggregate,
        "correct_minus_wrong": delta,
        "online_reward_audit": {
            "correct": correct_audit["conditional_tinymdm_reward"],
            "wrong": wrong_audit["conditional_tinymdm_reward"],
        },
        "checks": {
            "matched_frozen_evaluation": True,
            "both_online_rewards_called_during_training": True,
            "only_semantic_class_differs": True,
            "behavior_difference_detected": behavior_difference,
            "correct_condition_has_physical_advantage": physical_advantage,
        },
        "conclusion": (
            "correct_condition_has_matched_physical_advantage"
            if physical_advantage
            else (
                "demo_condition_changes_behavior_without_matched_physical_advantage"
                if behavior_difference
                else "no_detectable_condition_effect_at_update64"
            )
        ),
        "claim_boundary": (
            "One fixed-prefix, one-seed causal diagnostic. A difference establishes "
            "selected-condition use, not general demonstration following."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
