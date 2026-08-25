#!/usr/bin/env python3
"""Aggregate matched multi-context transition-recovery diagnostics across seeds."""

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
COMPOSITION_MEAN_FIELDS = (
    "mean_abs_deviation_from_selected_endpoint",
    "mean_abs_mixed_minus_selected_endpoint_action",
    "mean_abs_bounded_residual_action",
    "mean_abs_composed_minus_selected_endpoint_action",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records: list[dict[str, object]] = []
    training_seeds: set[int] = set()
    evaluation_seeds: set[int] = set()
    expected_schedule: list[int] | None = None
    expected_policy_topology: str | None = None
    for path in args.result:
        result = json.loads(path.read_text(encoding="utf-8"))
        training_seed = int(result["training_seed"])
        evaluation_seed = int(result["evaluation_seed"])
        checks = result.get("checks", {})
        schedule = [int(value) for value in result["training_prefix_schedule"]]
        context_prefixes = [
            int(context["carry_prefix_steps"])
            for context in result.get("contexts", [])
        ]
        if expected_schedule is None:
            expected_schedule = schedule
        policy_topology = str(
            result.get("policy_topology", "selected_expert_residual")
        )
        if expected_policy_topology is None:
            expected_policy_topology = policy_topology
        if (
            result.get("protocol")
            != "sugar_multi_context_transition_recovery_diagnostic_v1"
            or schedule != expected_schedule
            or policy_topology != expected_policy_topology
            or context_prefixes != schedule
            or training_seed in training_seeds
            or evaluation_seed in evaluation_seeds
            or training_seed == evaluation_seed
            or checks.get("all_predeclared_contexts_installed_online") is not True
            or checks.get("exact_frozen_experts_preserved") is not True
            or checks.get("causal_reward_not_actor_input") is not True
            or checks.get("all_initial_physics_elementwise_identical") is not True
            or checks.get("unseen_seed_evaluation") is not True
        ):
            raise RuntimeError(f"invalid or duplicate multi-context result: {path}")
        training_seeds.add(training_seed)
        evaluation_seeds.add(evaluation_seed)
        records.append(result)
    if len(records) < 2:
        raise RuntimeError("multi-context aggregate requires at least two seeds")
    if training_seeds & evaluation_seeds:
        raise RuntimeError("training and evaluation seed sets must be disjoint")

    totals = {
        endpoint: {
            field: int(
                sum(record["count_totals"][endpoint][field] for record in records)
            )
            for field in COUNT_FIELDS
        }
        for endpoint in ("learned_kick", "exact_pre_update_kick")
    }
    mean_delta = {
        field: float(
            np.mean(
                [record["mean_learned_minus_pre_update"][field] for record in records]
            )
        )
        for field in MEAN_FIELDS
    }
    improvements = [
        bool(record["checks"]["aggregate_kick_safety_improvement"])
        for record in records
    ]
    action_composition = None
    composition_checks: dict[str, object] = {}
    if expected_policy_topology == "causal_action_composition":
        per_seed_composition = []
        for record in records:
            contexts = record.get("contexts", [])
            learned_terms = [
                context.get("learned_action_composition")
                for context in contexts
            ]
            pre_terms = [
                context.get("exact_pre_update_action_composition")
                for context in contexts
            ]
            if (
                len(contexts) != len(expected_schedule or [])
                or not all(isinstance(item, dict) for item in learned_terms)
                or not all(isinstance(item, dict) for item in pre_terms)
                or record.get("checks", {}).get(
                    "pre_update_exact_selected_action_composition"
                ) is not True
            ):
                raise RuntimeError("causal action-composition seed audit is incomplete")
            per_seed_composition.append(
                {
                    "training_seed": int(record["training_seed"]),
                    "learned_composition_used_online": bool(
                        record["checks"]["learned_action_composition_used_online"]
                    ),
                    **{
                        field: float(
                            np.mean([float(item[field]) for item in learned_terms])
                        )
                        for field in COMPOSITION_MEAN_FIELDS
                    },
                    "maximum_abs_deployed_minus_composed_action": float(
                        max(
                            float(item["maximum_abs_deployed_minus_composed_action"])
                            for item in learned_terms
                        )
                    ),
                }
            )
        action_composition = {
            "per_seed": per_seed_composition,
            "across_seed_mean": {
                field: float(
                    np.mean([float(item[field]) for item in per_seed_composition])
                )
                for field in COMPOSITION_MEAN_FIELDS
            },
        }
        composition_checks = {
            "pre_update_exact_selected_action_composition_all_seeds": True,
            "learned_action_composition_used_online_all_seeds": all(
                bool(item["learned_composition_used_online"])
                for item in per_seed_composition
            ),
            "composition_terms_match_deployed_action_all_seeds": all(
                float(item["maximum_abs_deployed_minus_composed_action"]) == 0.0
                for item in per_seed_composition
            ),
        }
    result = {
        "protocol": "sugar_multi_context_transition_recovery_aggregate_v1",
        "training_seeds": sorted(training_seeds),
        "evaluation_seeds": sorted(evaluation_seeds),
        "num_training_seeds": len(records),
        "training_prefix_schedule": expected_schedule,
        "policy_topology": expected_policy_topology,
        "profiles_per_endpoint": int(
            sum(record["profiles_per_endpoint"] for record in records)
        ),
        "count_totals": totals,
        "mean_learned_minus_pre_update": mean_delta,
        "per_seed_aggregate_safety_improvement": improvements,
        "action_composition": action_composition,
        "checks": {
            "independent_training_and_evaluation_seeds": True,
            "same_predeclared_context_schedule_all_seeds": True,
            "matched_exact_pre_update_comparison_all_seeds": True,
            "aggregate_safety_improvement_replicated_all_seeds": all(improvements),
            **composition_checks,
        },
        "conclusion": (
            "multi_context_kick_safety_improvement_replicated"
            if all(improvements)
            else "multi_context_kick_safety_improvement_not_replicated"
        ),
        "claim_boundary": (
            "Independent training/evaluation seeds test the same three predeclared physical "
            "Carry-to-Kick handoff contexts against each seed's elementwise-matched exact "
            "pre-update endpoint. This tests context-robust recovery learning for the released "
            "Carry/Kick pair, not arbitrary-video or arbitrary-skill generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
