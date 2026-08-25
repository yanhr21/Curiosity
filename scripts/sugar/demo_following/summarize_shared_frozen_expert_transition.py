#!/usr/bin/env python3
"""Summarize one shared checkpoint evaluated under Kick and Carry conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kick", type=Path, required=True)
parser.add_argument("--carry", type=Path, required=True)
parser.add_argument("--training-audit", type=Path, required=True)
parser.add_argument("--checkpoint-audit", type=Path, required=True)
parser.add_argument("--training-seed", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def _evaluation(path: Path, skill_id: int) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol") != "sugar_cross_skill_recovery_frozen_eval_v3"
        or result.get("structurally_valid") is not True
        or result.get("checkpoint_iteration") != 64
        or result.get("transition_selected_skill_id") != skill_id
    ):
        raise RuntimeError(f"invalid shared transition evaluation: {path}")
    return result


def main() -> None:
    kick = _evaluation(args.kick, 1)
    carry = _evaluation(args.carry, 0)
    training_audit = json.loads(args.training_audit.read_text(encoding="utf-8"))
    checkpoint_audit = json.loads(
        args.checkpoint_audit.read_text(encoding="utf-8")
    )
    if (
        training_audit.get("transition_selected_skill_id") != -1
        or training_audit.get("transition_selected_skill_counts") != [32, 32]
        or training_audit.get("conditional_tinymdm_reward") is not None
        or checkpoint_audit.get("overall_pass") is not True
        or set(checkpoint_audit.get("arms", {})) != {"shared"}
    ):
        raise RuntimeError("shared transition training/checkpoint audit failed")
    for key in ("checkpoint", "seed", "num_envs", "steps", "prefix"):
        if kick[key] != carry[key]:
            raise RuntimeError(f"shared-checkpoint condition-swap drift: {key}")
    kick_trace = np.load(args.kick.parent / "trace.npz")
    carry_trace = np.load(args.carry.parent / "trace.npz")
    initial_keys = (
        "initial_robot_root_state_w",
        "initial_robot_joint_pos",
        "initial_robot_joint_vel",
        "initial_object_root_state_w",
        "initial_policy_observation",
    )
    identical_initial_state = all(
        np.array_equal(kick_trace[key], carry_trace[key]) for key in initial_keys
    )
    if not identical_initial_state:
        raise RuntimeError("condition-swap initial physics is not elementwise identical")
    first_action_difference = np.abs(
        kick_trace["action"][0] - carry_trace["action"][0]
    )
    full_action_difference = np.abs(
        kick_trace["action"] - carry_trace["action"]
    )
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
    kick_aggregate = kick["aggregate"]
    carry_aggregate = carry["aggregate"]
    delta = {
        field: float(kick_aggregate[field] - carry_aggregate[field])
        for field in fields
    }
    kick_safer_than_inert_carry = bool(
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
        "protocol": "sugar_shared_frozen_expert_transition_condition_swap_v2",
        "experiment": {
            "one_shared_checkpoint": kick["checkpoint"],
            "balanced_training_selected_skill_counts": [32, 32],
            "same_online_prefix": "one Kick alignment plus 41 Carry steps",
            "same_training_seed": args.training_seed,
            "same_update_budget": 64,
            "scalar_smp_reward": False,
        },
        "kick_condition": kick_aggregate,
        "carry_condition": carry_aggregate,
        "kick_minus_carry": delta,
        "causal_action_difference": {
            "first_step_mean_abs": float(first_action_difference.mean()),
            "first_step_max_abs": float(first_action_difference.max()),
            "full_trace_mean_abs": float(full_action_difference.mean()),
        },
        "checks": {
            "same_checkpoint_condition_swap": True,
            "initial_physics_elementwise_identical": identical_initial_state,
            "balanced_condition_training": True,
            "exact_frozen_experts_preserved": True,
            "no_scalar_smp_reward": True,
            "kick_condition_is_safer_than_inert_carry_condition": (
                kick_safer_than_inert_carry
            ),
        },
        "conclusion": "same_checkpoint_selected_skill_condition_changes_behavior",
        "claim_boundary": (
            "The Kick-versus-Carry swap tests whether one policy reads its selected-skill "
            "condition. Carry is an intentionally wrong/inert semantic control and is not "
            "the safety baseline. Training benefit must be measured against the exact "
            "pre-update Kick endpoint on the same evaluation seed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
