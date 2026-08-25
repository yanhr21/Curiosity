#!/usr/bin/env python3
"""Summarize learned versus exact-pre-update Kick across physical handoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
INITIAL_KEYS = (
    "initial_robot_root_state_w",
    "initial_robot_joint_pos",
    "initial_robot_joint_vel",
    "initial_object_root_state_w",
    "initial_policy_observation",
)
MINIMUM_MEAN_COMPOSITION_DEVIATION = 1.0e-4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        nargs=3,
        metavar=("PREFIX", "LEARNED_RESULT", "PRE_UPDATE_RESULT"),
        required=True,
    )
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--expected-schedule", default="41,49,57")
    parser.add_argument("--expected-evaluation-schedule")
    parser.add_argument(
        "--expected-policy-topology",
        choices=("selected_expert_residual", "causal_action_composition"),
        default="selected_expert_residual",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _evaluation(
    path: Path, iteration: int, prefix: int, policy_topology: str
) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    recorded_topology = result.get(
        "policy_topology", "selected_expert_residual"
    )
    valid_protocols = (
        {"sugar_cross_skill_recovery_frozen_eval_v4"}
        if policy_topology == "causal_action_composition"
        else {
            "sugar_cross_skill_recovery_frozen_eval_v3",
            "sugar_cross_skill_recovery_frozen_eval_v4",
        }
    )
    if (
        result.get("protocol") not in valid_protocols
        or result.get("structurally_valid") is not True
        or result.get("checkpoint_iteration") != iteration
        or result.get("transition_selected_skill_id") != 1
        or recorded_topology != policy_topology
        or result.get("prefix", {}).get("carry_steps") != prefix
    ):
        raise RuntimeError(f"invalid prefix{prefix} Kick evaluation: {path}")
    if policy_topology == "causal_action_composition":
        contract = result.get("kick_success_contract", {})
        if contract != {
            "name": "foot_contact_coupled_planar_motion_v1",
            "minimum_planar_net_displacement_m": 0.05,
            "minimum_contact_adjacent_planar_path_m": 0.01,
            "minimum_post_first_contact_planar_path_m": 0.03,
            "legacy_any_contact_plus_net_displacement_reported_separately": True,
            "handoff_to_first_action_interval_included": True,
        }:
            raise RuntimeError(f"invalid strict Kick metric contract: {path}")
        fall_contract = result.get("physical_fall_contract", {})
        if fall_contract != {
            "name": "root_height_or_tilt_v1",
            "minimum_root_height_loss_m": 0.35,
            "minimum_root_tilt_deg": 60.0,
            "legacy_height_only_fall_reported_separately": True,
            "root_height_loss_referenced_to_handoff": True,
            "handoff_tilt_not_charged_to_policy": True,
        }:
            raise RuntimeError(f"invalid strict fall metric contract: {path}")
    return result


def _comparison(
    prefix: int, learned_path: Path, pre_path: Path, policy_topology: str
) -> dict[str, object]:
    learned = _evaluation(learned_path, 64, prefix, policy_topology)
    pre = _evaluation(pre_path, -1, prefix, policy_topology)
    for key in ("seed", "num_envs", "steps", "prefix"):
        if learned[key] != pre[key]:
            raise RuntimeError(f"prefix{prefix} learned/pre-update drift: {key}")
    learned_trace = np.load(learned_path.parent / "trace.npz")
    pre_trace = np.load(pre_path.parent / "trace.npz")
    initial_keys = INITIAL_KEYS
    if policy_topology == "causal_action_composition":
        initial_keys += (
            "initial_carry_skill_command",
            "initial_kick_skill_command",
            "initial_selected_skill_id",
        )
    if not all(
        np.array_equal(learned_trace[key], pre_trace[key]) for key in initial_keys
    ):
        raise RuntimeError(f"prefix{prefix} initial physics is not identical")
    learned_aggregate = learned["aggregate"]
    pre_aggregate = pre["aggregate"]
    delta = {
        field: float(learned_aggregate[field] - pre_aggregate[field])
        for field in (*COUNT_FIELDS, *MEAN_FIELDS)
    }
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
    record = {
        "carry_prefix_steps": prefix,
        "evaluation_seed": learned["seed"],
        "profiles": learned["num_envs"],
        "learned_kick": learned_aggregate,
        "exact_pre_update_kick": pre_aggregate,
        "learned_minus_pre_update": delta,
        "initial_physics_elementwise_identical": True,
        "safety_improvement": safety_improvement,
    }
    if policy_topology == "causal_action_composition":
        learned_composition = learned.get("action_composition")
        pre_composition = pre.get("action_composition")
        if (
            not isinstance(learned_composition, dict)
            or not isinstance(pre_composition, dict)
            or learned_composition.get("future_or_outcome_labels_used") is not False
            or pre_composition.get("future_or_outcome_labels_used") is not False
            or float(
                pre_composition.get(
                    "mean_abs_deviation_from_selected_endpoint", float("nan")
                )
            )
            != 0.0
            or float(
                pre_composition.get(
                    "mean_abs_mixed_minus_selected_endpoint_action",
                    float("nan"),
                )
            )
            != 0.0
            or float(
                pre_composition.get(
                    "mean_abs_bounded_residual_action", float("nan")
                )
            )
            != 0.0
            or float(
                pre_composition.get(
                    "mean_abs_composed_minus_selected_endpoint_action",
                    float("nan"),
                )
            )
            != 0.0
            or float(
                learned_composition.get(
                    "maximum_abs_deployed_minus_composed_action",
                    float("nan"),
                )
            )
            != 0.0
            or float(
                pre_composition.get(
                    "maximum_abs_deployed_minus_composed_action",
                    float("nan"),
                )
            )
            != 0.0
        ):
            raise RuntimeError(
                f"prefix{prefix} causal action-composition audit failed"
            )
        record["learned_action_composition"] = learned_composition
        record["exact_pre_update_action_composition"] = pre_composition
        record["initial_full_584d_actor_input_elementwise_identical"] = True
        record["learned_composition_weight_changes_online"] = bool(
            float(
                learned_composition["mean_abs_deviation_from_selected_endpoint"]
            )
            >= MINIMUM_MEAN_COMPOSITION_DEVIATION
        )
    return record


def main() -> None:
    args = _parse_args()
    expected_schedule = [
        int(value.strip()) for value in args.expected_schedule.split(",") if value.strip()
    ]
    expected_evaluation_schedule = (
        expected_schedule
        if args.expected_evaluation_schedule is None
        else [
            int(value.strip())
            for value in args.expected_evaluation_schedule.split(",")
            if value.strip()
        ]
    )
    supplied = [int(spec[0]) for spec in args.comparison]
    if (
        supplied != expected_evaluation_schedule
        or len(set(supplied)) != len(supplied)
        or not expected_schedule
        or len(set(expected_schedule)) != len(expected_schedule)
    ):
        raise RuntimeError("comparison prefixes do not match the predeclared schedule")

    training_audit = json.loads(args.training_audit.read_text(encoding="utf-8"))
    checkpoint_audit = json.loads(args.checkpoint_audit.read_text(encoding="utf-8"))
    install_counts = training_audit.get("carry_prefix_install_counts")
    if (
        training_audit.get("protocol")
        != "sugar_online_cross_skill_recovery_prefix_v3"
        or training_audit.get("carry_prefix_schedule") != expected_schedule
        or not isinstance(install_counts, list)
        or len(install_counts) != len(expected_schedule)
        or any(int(count) <= 0 for count in install_counts)
        or sum(int(count) for count in install_counts)
        != int(training_audit.get("prefix_count", -1))
        or training_audit.get("prefix_schedule_is_episode_boundary_online") is not True
        or training_audit.get("state_teleport") is not False
        or training_audit.get("offline_replay") is not False
        or training_audit.get("ppo_prefix_transitions") != 0
        or training_audit.get("transition_selected_skill_id") != -1
        or training_audit.get("transition_selected_skill_counts") != [32, 32]
    ):
        raise RuntimeError("multi-context online training audit failed")
    if args.expected_policy_topology == "causal_action_composition" and (
        training_audit.get("transition_selected_skill_assignment")
        != "env_parity_swapped_each_episode"
        or min(
            training_audit.get(
                "transition_selected_skill_exposure_min_per_env", [0, 0]
            )
        )
        <= 0
    ):
        raise RuntimeError("causal-composition condition exposure audit failed")
    reward_audit = training_audit.get("transition_recovery_reward", {})
    if (
        reward_audit.get("enabled") is not True
        or int(reward_audit.get("reward_calls", 0)) <= 0
        or float(reward_audit.get("maximum_abs_reward", 0.0)) <= 0.0
        or reward_audit.get("future_or_outcome_labels_used") is not False
        or reward_audit.get("actor_observation_augmented") is not False
        or checkpoint_audit.get("overall_pass") is not True
    ):
        raise RuntimeError("multi-context reward/checkpoint audit failed")

    records = [
        _comparison(
            int(prefix), Path(learned), Path(pre), args.expected_policy_topology
        )
        for prefix, learned, pre in args.comparison
    ]
    evaluation_seeds = {int(record["evaluation_seed"]) for record in records}
    if len(evaluation_seeds) != 1 or args.training_seed in evaluation_seeds:
        raise RuntimeError("evaluation must use one unseen seed across all contexts")

    totals = {
        endpoint: {
            field: int(sum(record[endpoint][field] for record in records))
            for field in COUNT_FIELDS
        }
        for endpoint in ("learned_kick", "exact_pre_update_kick")
    }
    mean_delta = {
        field: float(np.mean([record["learned_minus_pre_update"][field] for record in records]))
        for field in MEAN_FIELDS
    }
    learned = totals["learned_kick"]
    pre = totals["exact_pre_update_kick"]
    physical_aggregate_safety_improvement = bool(
        (
            learned["safe_kick_success_count"] > pre["safe_kick_success_count"]
            and learned["physical_fall_count"] <= pre["physical_fall_count"]
        )
        or (
            learned["physical_fall_count"] < pre["physical_fall_count"]
            and learned["safe_kick_success_count"] >= pre["safe_kick_success_count"]
        )
    )
    composition_used = bool(
        args.expected_policy_topology != "causal_action_composition"
        or any(
            record.get("learned_composition_weight_changes_online") is True
            for record in records
        )
    )
    aggregate_safety_improvement = bool(
        physical_aggregate_safety_improvement and composition_used
    )
    checks = {
        "all_predeclared_contexts_installed_online": True,
        "evaluation_prefixes_disjoint_from_training": bool(
            set(expected_evaluation_schedule).isdisjoint(expected_schedule)
        ),
        "exact_frozen_experts_preserved": True,
        "causal_reward_not_actor_input": True,
        "all_initial_physics_elementwise_identical": True,
        "unseen_seed_evaluation": True,
        "physical_aggregate_kick_safety_improvement": (
            physical_aggregate_safety_improvement
        ),
        "aggregate_kick_safety_improvement": aggregate_safety_improvement,
    }
    if args.expected_policy_topology == "causal_action_composition":
        checks.update(
            {
                "pre_update_exact_selected_action_composition": True,
                "initial_full_584d_actor_input_elementwise_identical_all_contexts": True,
                "strict_contact_coupled_kick_metric_all_contexts": True,
                "strict_height_or_tilt_fall_metric_all_contexts": True,
                "minimum_mean_composition_deviation": (
                    MINIMUM_MEAN_COMPOSITION_DEVIATION
                ),
                "learned_action_composition_used_online": composition_used,
            }
        )
    result = {
        "protocol": "sugar_multi_context_transition_recovery_diagnostic_v1",
        "training_seed": args.training_seed,
        "policy_topology": args.expected_policy_topology,
        "evaluation_seed": evaluation_seeds.pop(),
        "training_prefix_schedule": expected_schedule,
        "evaluation_prefix_schedule": expected_evaluation_schedule,
        "training_prefix_install_counts": install_counts,
        "profiles_per_endpoint": sum(int(record["profiles"]) for record in records),
        "contexts": records,
        "count_totals": totals,
        "mean_learned_minus_pre_update": mean_delta,
        "checks": checks,
        "conclusion": (
            "multi_context_training_improves_unseen_seed_kick_safety"
            if aggregate_safety_improvement
            else "multi_context_training_does_not_improve_unseen_seed_kick_safety"
        ),
        "claim_boundary": (
            "One multi-context training seed and one unseen evaluation seed across predeclared "
            "physical evaluation handoffs. A positive diagnostic requires aggregate safe/fall "
            "improvement over the exact pre-update Kick endpoint; reward and action changes are "
            "insufficient. Kick success requires foot-contact-coupled motion, not an unrelated "
            "one-frame touch; a fall is either 0.35 m root-height loss or 60 degree root tilt. "
            "A positive result still requires independent-seed replication."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
