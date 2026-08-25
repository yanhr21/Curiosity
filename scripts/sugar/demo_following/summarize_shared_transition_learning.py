#!/usr/bin/env python3
"""Compare learned Kick behavior with its exact pre-update Kick endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--learned", type=Path, required=True)
parser.add_argument("--pre-update", type=Path, required=True)
parser.add_argument("--training-seed", type=int, required=True)
parser.add_argument("--learned-iteration", type=int, default=64)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def _load(path: Path, expected_iteration: int) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol") != "sugar_cross_skill_recovery_frozen_eval_v3"
        or result.get("structurally_valid") is not True
        or result.get("checkpoint_iteration") != expected_iteration
        or result.get("transition_selected_skill_id") != 1
    ):
        raise RuntimeError(f"invalid Kick evaluation: {path}")
    return result


def main() -> None:
    learned = _load(args.learned, args.learned_iteration)
    pre_update = _load(args.pre_update, -1)
    for key in ("seed", "num_envs", "steps", "prefix"):
        if learned[key] != pre_update[key]:
            raise RuntimeError(f"learned/pre-update protocol drift: {key}")

    learned_trace = np.load(args.learned.parent / "trace.npz")
    pre_update_trace = np.load(args.pre_update.parent / "trace.npz")
    initial_keys = (
        "initial_robot_root_state_w",
        "initial_robot_joint_pos",
        "initial_robot_joint_vel",
        "initial_object_root_state_w",
        "initial_policy_observation",
    )
    identical_initial_state = all(
        np.array_equal(learned_trace[key], pre_update_trace[key])
        for key in initial_keys
    )
    if not identical_initial_state:
        raise RuntimeError("learned/pre-update initial physics is not elementwise identical")

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
    learned_aggregate = learned["aggregate"]
    pre_update_aggregate = pre_update["aggregate"]
    delta = {
        field: float(learned_aggregate[field] - pre_update_aggregate[field])
        for field in fields
    }
    learned_action = learned_trace["action"]
    pre_update_action = pre_update_trace["action"]
    action_difference = np.abs(learned_action - pre_update_action)
    safety_improvement = bool(
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
        "protocol": "sugar_shared_frozen_expert_transition_learning_v1",
        "training_seed": args.training_seed,
        "learned_iteration": args.learned_iteration,
        "evaluation_seed": learned["seed"],
        "profiles": learned["num_envs"],
        "learned_kick": learned_aggregate,
        "exact_pre_update_kick": pre_update_aggregate,
        "learned_minus_pre_update": delta,
        "action_difference": {
            "first_step_mean_abs": float(action_difference[0].mean()),
            "first_step_max_abs": float(action_difference[0].max()),
            "full_trace_mean_abs": float(action_difference.mean()),
        },
        "checks": {
            "matched_evaluation_protocol": True,
            "initial_physics_elementwise_identical": identical_initial_state,
            "same_kick_condition": True,
            "exact_pre_update_endpoint": True,
            "learned_kick_safety_improvement": safety_improvement,
        },
        "conclusion": (
            "learned_transition_improves_matched_kick_safety"
            if safety_improvement
            else "learned_transition_does_not_improve_matched_kick_safety"
        ),
        "claim_boundary": (
            "One training seed with one disjoint evaluation seed. This isolates the "
            "effect of 64 updates from selected-skill conditioning and requires "
            "independent-seed replication before a robust safety claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
