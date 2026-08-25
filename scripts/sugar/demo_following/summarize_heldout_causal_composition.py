#!/usr/bin/env python3
"""Summarize frozen causal-composer learned/pre results on unseen prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COUNT_FIELDS = (
    "kick_success_count",
    "safe_kick_success_count",
    "physical_fall_count",
)
MEAN_FIELDS = (
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
    "initial_carry_skill_command",
    "initial_kick_skill_command",
    "initial_selected_skill_id",
)
KICK_CONTRACT = {
    "name": "foot_contact_coupled_planar_motion_v1",
    "minimum_planar_net_displacement_m": 0.05,
    "minimum_contact_adjacent_planar_path_m": 0.01,
    "minimum_post_first_contact_planar_path_m": 0.03,
    "legacy_any_contact_plus_net_displacement_reported_separately": True,
    "handoff_to_first_action_interval_included": True,
}
FALL_CONTRACT = {
    "name": "root_height_or_tilt_v1",
    "minimum_root_height_loss_m": 0.35,
    "minimum_root_tilt_deg": 60.0,
    "legacy_height_only_fall_reported_separately": True,
    "root_height_loss_referenced_to_handoff": True,
    "handoff_tilt_not_charged_to_policy": True,
}
MINIMUM_MEAN_COMPOSITION_DEVIATION = 1.0e-4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        action="append",
        nargs=7,
        metavar=(
            "TRAIN_SEED",
            "EVAL_SEED",
            "CHECKPOINT_AUDIT",
            "LEARNED_PREFIX33",
            "PRE_PREFIX33",
            "LEARNED_PREFIX65",
            "PRE_PREFIX65",
        ),
        required=True,
    )
    parser.add_argument("--heldout-prefixes", default="33,65")
    parser.add_argument("--training-prefixes", default="41,49,57")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _evaluation(
    path: Path, *, iteration: int, evaluation_seed: int, prefix: int
) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("protocol") != "sugar_cross_skill_recovery_frozen_eval_v4"
        or result.get("structurally_valid") is not True
        or result.get("checkpoint_iteration") != iteration
        or result.get("transition_selected_skill_id") != 1
        or result.get("policy_topology") != "causal_action_composition"
        or result.get("seed") != evaluation_seed
        or result.get("num_envs") != 20
        or result.get("steps") != 250
        or result.get("prefix", {}).get("carry_steps") != prefix
        or result.get("kick_success_contract") != KICK_CONTRACT
        or result.get("physical_fall_contract") != FALL_CONTRACT
        or result.get("checks", {}).get(
            "composition_terms_match_deployed_action"
        )
        is not True
    ):
        raise RuntimeError(f"invalid held-out frozen evaluation: {path}")
    return result


def _safety_improvement(delta: dict[str, float]) -> bool:
    return bool(
        (
            delta["safe_kick_success_count"] > 0
            and delta["physical_fall_count"] <= 0
        )
        or (
            delta["physical_fall_count"] < 0
            and delta["safe_kick_success_count"] >= 0
        )
    )


def _comparison(
    learned_path: Path,
    pre_path: Path,
    *,
    evaluation_seed: int,
    prefix: int,
) -> dict[str, object]:
    learned = _evaluation(
        learned_path, iteration=64, evaluation_seed=evaluation_seed, prefix=prefix
    )
    pre = _evaluation(
        pre_path, iteration=-1, evaluation_seed=evaluation_seed, prefix=prefix
    )
    for key in ("seed", "num_envs", "steps", "prefix"):
        if learned[key] != pre[key]:
            raise RuntimeError(f"held-out prefix{prefix} learned/pre drift: {key}")
    learned_trace = np.load(learned_path.parent / "trace.npz")
    pre_trace = np.load(pre_path.parent / "trace.npz")
    if not all(
        np.array_equal(learned_trace[key], pre_trace[key])
        for key in INITIAL_KEYS
    ):
        raise RuntimeError(f"held-out prefix{prefix} initial input drift")
    learned_aggregate = learned["aggregate"]
    pre_aggregate = pre["aggregate"]
    delta = {
        field: float(learned_aggregate[field] - pre_aggregate[field])
        for field in (*COUNT_FIELDS, *MEAN_FIELDS)
    }
    learned_profiles = {
        int(row["profile"]): row for row in learned["profiles"]
    }
    pre_profiles = {int(row["profile"]): row for row in pre["profiles"]}
    if set(learned_profiles) != set(pre_profiles):
        raise RuntimeError(f"held-out prefix{prefix} profile set drift")
    outcome_changes = []
    for profile in sorted(learned_profiles):
        learned_row = learned_profiles[profile]
        pre_row = pre_profiles[profile]
        before = (
            bool(pre_row["safe_kick_success"]),
            bool(pre_row["physical_robot_fall"]),
        )
        after = (
            bool(learned_row["safe_kick_success"]),
            bool(learned_row["physical_robot_fall"]),
        )
        if before != after:
            outcome_changes.append(
                {
                    "profile": profile,
                    "pre_safe": before[0],
                    "learned_safe": after[0],
                    "pre_fall": before[1],
                    "learned_fall": after[1],
                    "learned_minus_pre_net_displacement_m": float(
                        learned_row["planar_object_net_displacement_m"]
                        - pre_row["planar_object_net_displacement_m"]
                    ),
                }
            )
    composition = learned.get("action_composition")
    pre_composition = pre.get("action_composition")
    if (
        not isinstance(composition, dict)
        or not isinstance(pre_composition, dict)
        or composition.get("future_or_outcome_labels_used") is not False
        or pre_composition.get("future_or_outcome_labels_used") is not False
        or float(
            composition.get(
                "maximum_abs_deployed_minus_composed_action", float("nan")
            )
        )
        != 0.0
        or float(
            pre_composition.get(
                "maximum_abs_deployed_minus_composed_action", float("nan")
            )
        )
        != 0.0
        or any(
            float(pre_composition.get(field, float("nan"))) != 0.0
            for field in (
                "mean_abs_deviation_from_selected_endpoint",
                "mean_abs_mixed_minus_selected_endpoint_action",
                "mean_abs_bounded_residual_action",
                "mean_abs_composed_minus_selected_endpoint_action",
            )
        )
    ):
        raise RuntimeError(f"held-out prefix{prefix} action composition audit failed")
    gate_deviation = float(
        composition["mean_abs_deviation_from_selected_endpoint"]
    )
    mixed_action_deviation = float(
        composition["mean_abs_mixed_minus_selected_endpoint_action"]
    )
    composed_action_deviation = float(
        composition["mean_abs_composed_minus_selected_endpoint_action"]
    )
    return {
        "prefix": prefix,
        "initial_full_584d_input_elementwise_identical": True,
        "learned": learned_aggregate,
        "exact_pre_update": pre_aggregate,
        "learned_minus_pre": delta,
        "safety_improvement": _safety_improvement(delta),
        "outcome_changes": outcome_changes,
        "composition": composition,
        "exact_pre_update_composition": pre_composition,
        "learned_action_composition_used_online": bool(
            composed_action_deviation >= MINIMUM_MEAN_COMPOSITION_DEVIATION
        ),
        "endpoint_gap_amplification": (
            mixed_action_deviation / gate_deviation
            if gate_deviation > 0.0
            else 0.0
        ),
    }


def main() -> None:
    args = _parse_args()
    heldout_prefixes = [int(value) for value in args.heldout_prefixes.split(",")]
    training_prefixes = [int(value) for value in args.training_prefixes.split(",")]
    if heldout_prefixes != [33, 65] or set(heldout_prefixes) & set(training_prefixes):
        raise RuntimeError("held-out prefix contract drift")
    if len(args.seed) != 2:
        raise RuntimeError("held-out evaluation requires exactly two seed pairs")

    records = []
    for spec in args.seed:
        training_seed = int(spec[0])
        evaluation_seed = int(spec[1])
        audit = json.loads(Path(spec[2]).read_text(encoding="utf-8"))
        arm = audit.get("arms", {}).get("shared", {})
        if (
            audit.get("protocol")
            != "sugar_frozen_expert_transition_checkpoint_audit_v2"
            or audit.get("post_iteration") != 64
            or audit.get("overall_pass") is not True
            or arm.get("policy_topology") != "causal_action_composition"
            or arm.get("official_expert_max_abs_error_pre_or_post") != 0.0
            or arm.get("all_checkpoint_tensors_finite") is not True
            or arm.get("pre_update_trainable_output_layer_exact_zero") is not True
            or float(arm.get("transition_trainable_max_parameter_delta", 0.0))
            <= 0.0
        ):
            raise RuntimeError(f"invalid checkpoint audit for seed{training_seed}")
        paths = [Path(value) for value in spec[3:]]
        comparisons = [
            _comparison(
                paths[index * 2],
                paths[index * 2 + 1],
                evaluation_seed=evaluation_seed,
                prefix=prefix,
            )
            for index, prefix in enumerate(heldout_prefixes)
        ]
        totals = {
            endpoint: {
                field: int(
                    sum(comparison[endpoint][field] for comparison in comparisons)
                )
                for field in COUNT_FIELDS
            }
            for endpoint in ("learned", "exact_pre_update")
        }
        delta = {
            field: float(totals["learned"][field] - totals["exact_pre_update"][field])
            for field in COUNT_FIELDS
        }
        composition_used = any(
            comparison["learned_action_composition_used_online"]
            for comparison in comparisons
        )
        records.append(
            {
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "comparisons": comparisons,
                "count_totals": totals,
                "learned_action_composition_used_online": composition_used,
                "aggregate_safety_improvement": bool(
                    _safety_improvement(delta) and composition_used
                ),
            }
        )

    training_seeds = [record["training_seed"] for record in records]
    evaluation_seeds = [record["evaluation_seed"] for record in records]
    if (
        len(set(training_seeds)) != 2
        or len(set(evaluation_seeds)) != 2
        or set(training_seeds) & set(evaluation_seeds)
    ):
        raise RuntimeError("held-out training/evaluation seed contract drift")
    totals = {
        endpoint: {
            field: int(
                sum(record["count_totals"][endpoint][field] for record in records)
            )
            for field in COUNT_FIELDS
        }
        for endpoint in ("learned", "exact_pre_update")
    }
    replicated = all(record["aggregate_safety_improvement"] for record in records)
    composition_used_all_seeds = all(
        record["learned_action_composition_used_online"] for record in records
    )
    result = {
        "protocol": "sugar_causal_composition_heldout_prefix_eval_v1",
        "policy_topology": "causal_action_composition",
        "training_prefixes": training_prefixes,
        "heldout_prefixes": heldout_prefixes,
        "profiles_per_endpoint": 80,
        "records": records,
        "count_totals": totals,
        "checks": {
            "no_policy_training_or_optimizer_update": True,
            "heldout_prefixes_disjoint_from_training": True,
            "all_initial_full_584d_inputs_elementwise_identical": True,
            "strict_contact_coupled_kick_metric_all_runs": True,
            "strict_height_or_tilt_fall_metric_all_runs": True,
            "official_experts_exact_all_seeds": True,
            "pre_update_exact_selected_endpoint_all_runs": True,
            "composition_terms_match_deployed_action_all_runs": True,
            "minimum_mean_composition_deviation": (
                MINIMUM_MEAN_COMPOSITION_DEVIATION
            ),
            "learned_action_composition_used_online_all_seeds": (
                composition_used_all_seeds
            ),
            "heldout_safety_improvement_replicated_all_seeds": replicated,
        },
        "conclusion": (
            "heldout_prefix_safety_improvement_replicated"
            if replicated
            else "heldout_prefix_safety_improvement_not_replicated"
        ),
        "claim_boundary": (
            "Frozen learned/pre checkpoints on adjacent unseen physical prefix lengths 33/65. "
            "This tests local handoff-length transfer for the released Carry/Kick pair, not "
            "arbitrary-video, arbitrary-skill or cross-asset generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
