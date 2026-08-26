#!/usr/bin/env python3
"""Attribute dense-prefix composer outcomes to its gate and residual paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ("full", "gate_only", "residual_only", "exact_pre_update")
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
COUNT_FIELDS = (
    "safe_kick_success_count",
    "physical_fall_count",
    "kick_success_count",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        nargs=5,
        metavar=("PREFIX", "FULL", "GATE_ONLY", "RESIDUAL_ONLY", "PRE"),
        required=True,
    )
    parser.add_argument("--expected-prefixes", default="37,45,53,61")
    parser.add_argument("--evaluation-seed", type=int, default=181662)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--residual-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_result(path: Path, prefix: int, seed: int, iteration: int) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    checks = result.get("checks", {})
    if (
        result.get("protocol") != "sugar_cross_skill_recovery_frozen_eval_v4"
        or result.get("structurally_valid") is not True
        or result.get("policy_topology") != "causal_action_composition"
        or result.get("transition_selected_skill_id") != 1
        or result.get("checkpoint_iteration") != iteration
        or result.get("seed") != seed
        or result.get("num_envs") != 20
        or result.get("steps") != 250
        or result.get("prefix", {}).get("carry_steps") != prefix
        or checks.get("composition_terms_match_deployed_action") is not True
    ):
        raise RuntimeError(f"invalid frozen ablation evaluation: {path}")
    return result


def _transition_counts(result: dict, pre: dict) -> tuple[dict, list[dict]]:
    rows = {int(row["profile"]): row for row in result["profiles"]}
    pre_rows = {int(row["profile"]): row for row in pre["profiles"]}
    if set(rows) != set(pre_rows) or len(rows) != 20:
        raise RuntimeError("ablation profile set drift")
    counts = {
        "safe_kick_gained": 0,
        "safe_kick_lost": 0,
        "fall_prevented": 0,
        "fall_introduced": 0,
    }
    changes = []
    for profile in sorted(rows):
        row = rows[profile]
        pre_row = pre_rows[profile]
        old = (
            bool(pre_row["safe_kick_success"]),
            bool(pre_row["physical_robot_fall"]),
        )
        new = (
            bool(row["safe_kick_success"]),
            bool(row["physical_robot_fall"]),
        )
        counts["safe_kick_gained"] += int(new[0] and not old[0])
        counts["safe_kick_lost"] += int(old[0] and not new[0])
        counts["fall_prevented"] += int(old[1] and not new[1])
        counts["fall_introduced"] += int(new[1] and not old[1])
        if old != new:
            changes.append(
                {
                    "profile": profile,
                    "pre_safe_kick": old[0],
                    "arm_safe_kick": new[0],
                    "pre_fall": old[1],
                    "arm_fall": new[1],
                }
            )
    return counts, changes


def _check_ablation_audit(path: Path, mode: str) -> None:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (
        audit.get("protocol")
        != "sugar_causal_composer_frozen_ablation_checkpoint_v1"
        or audit.get("mode") != mode
        or audit.get("source_iteration") != 64
        or audit.get("policy_training_or_optimizer_updates") != 0
        or not all(
            audit.get(key) is True
            for key in (
                "all_non_output_tensors_elementwise_identical",
                "preserved_output_rows_elementwise_identical",
                "ablated_output_rows_exact_zero",
                "all_output_tensors_finite",
            )
        )
    ):
        raise RuntimeError(f"invalid {mode} checkpoint audit")


def main() -> None:
    args = _parse_args()
    prefixes = [int(value) for value in args.expected_prefixes.split(",")]
    supplied = [int(spec[0]) for spec in args.comparison]
    if supplied != prefixes or len(set(prefixes)) != len(prefixes):
        raise RuntimeError("ablation prefix schedule drift")
    _check_ablation_audit(args.gate_audit, "gate_only")
    _check_ablation_audit(args.residual_audit, "residual_only")

    contexts = []
    visualization_targets = []
    for spec in args.comparison:
        prefix = int(spec[0])
        paths = {arm: Path(path) for arm, path in zip(ARMS, spec[1:])}
        results = {
            arm: _load_result(
                path,
                prefix,
                args.evaluation_seed,
                -1 if arm == "exact_pre_update" else 64,
            )
            for arm, path in paths.items()
        }
        traces = {arm: np.load(path.parent / "trace.npz") for arm, path in paths.items()}
        if not all(
            np.array_equal(traces["exact_pre_update"][key], traces[arm][key])
            for arm in ARMS[:-1]
            for key in INITIAL_KEYS
        ):
            raise RuntimeError(f"prefix{prefix} ablation initial-input drift")

        full_terms = results["full"]["action_composition"]
        gate_terms = results["gate_only"]["action_composition"]
        residual_terms = results["residual_only"]["action_composition"]
        pre_terms = results["exact_pre_update"]["action_composition"]
        if (
            float(gate_terms["mean_abs_bounded_residual_action"]) != 0.0
            or float(residual_terms["mean_abs_deviation_from_selected_endpoint"])
            != 0.0
            or float(residual_terms["mean_abs_mixed_minus_selected_endpoint_action"])
            != 0.0
            or any(
                float(pre_terms[field]) != 0.0
                for field in (
                    "mean_abs_deviation_from_selected_endpoint",
                    "mean_abs_mixed_minus_selected_endpoint_action",
                    "mean_abs_bounded_residual_action",
                )
            )
            or not np.array_equal(
                traces["full"]["kick_composition_weight"][0],
                traces["gate_only"]["kick_composition_weight"][0],
            )
            or not np.array_equal(
                traces["full"]["bounded_residual_action"][0],
                traces["residual_only"]["bounded_residual_action"][0],
            )
        ):
            raise RuntimeError(f"prefix{prefix} gate/residual isolation failed")

        arm_records = {}
        for arm, result in results.items():
            counts, changes = _transition_counts(
                result, results["exact_pre_update"]
            )
            arm_records[arm] = {
                "aggregate": result["aggregate"],
                "action_composition": result["action_composition"],
                "initial_action_mean_abs_difference_from_pre": float(
                    np.mean(
                        np.abs(
                            traces[arm]["action"][0]
                            - traces["exact_pre_update"]["action"][0]
                        )
                    )
                ),
                "initial_action_max_abs_difference_from_pre": float(
                    np.max(
                        np.abs(
                            traces[arm]["action"][0]
                            - traces["exact_pre_update"]["action"][0]
                        )
                    )
                ),
                "outcome_transition_counts_vs_pre": counts,
                "outcome_changes_vs_pre": changes,
            }
        full_changed = arm_records["full"]["outcome_changes_vs_pre"]
        profile_rows = {
            arm: {int(row["profile"]): row for row in result["profiles"]}
            for arm, result in results.items()
        }
        full_outcome_attribution = []
        for change in full_changed:
            profile = int(change["profile"])
            full_outcome = (
                bool(profile_rows["full"][profile]["safe_kick_success"]),
                bool(profile_rows["full"][profile]["physical_robot_fall"]),
            )
            gate_outcome = (
                bool(profile_rows["gate_only"][profile]["safe_kick_success"]),
                bool(profile_rows["gate_only"][profile]["physical_robot_fall"]),
            )
            residual_outcome = (
                bool(
                    profile_rows["residual_only"][profile]["safe_kick_success"]
                ),
                bool(
                    profile_rows["residual_only"][profile]["physical_robot_fall"]
                ),
            )
            full_outcome_attribution.append(
                {
                    "profile": profile,
                    "full_safe_kick": full_outcome[0],
                    "full_fall": full_outcome[1],
                    "gate_only_matches_full_outcome": gate_outcome == full_outcome,
                    "residual_only_matches_full_outcome": (
                        residual_outcome == full_outcome
                    ),
                    "gate_only_safe_kick": gate_outcome[0],
                    "gate_only_fall": gate_outcome[1],
                    "residual_only_safe_kick": residual_outcome[0],
                    "residual_only_fall": residual_outcome[1],
                }
            )
        visualization_targets.extend(
            {"prefix": prefix, "profile": row["profile"]} for row in full_changed
        )
        contexts.append(
            {
                "prefix": prefix,
                "initial_full_584d_inputs_elementwise_identical_all_arms": True,
                "arms": arm_records,
                "full_initial_gate_equals_gate_only": True,
                "full_initial_residual_equals_residual_only": True,
                "full_outcome_attribution": full_outcome_attribution,
            }
        )

    totals = {
        arm: {
            field: int(
                sum(context["arms"][arm]["aggregate"][field] for context in contexts)
            )
            for field in COUNT_FIELDS
        }
        for arm in ARMS
    }
    all_full_outcome_attribution = [
        {"prefix": context["prefix"], **record}
        for context in contexts
        for record in context["full_outcome_attribution"]
    ]
    gate_only_explains_every_change = bool(
        all_full_outcome_attribution
        and all(
            record["gate_only_matches_full_outcome"]
            for record in all_full_outcome_attribution
        )
    )
    residual_only_explains_every_change = bool(
        all_full_outcome_attribution
        and all(
            record["residual_only_matches_full_outcome"]
            for record in all_full_outcome_attribution
        )
    )
    if not all_full_outcome_attribution:
        conclusion = "full_endpoint_has_no_outcome_change_vs_pre_update"
    elif gate_only_explains_every_change and not residual_only_explains_every_change:
        conclusion = "full_outcome_changes_persist_through_gate_path_only"
    elif residual_only_explains_every_change and not gate_only_explains_every_change:
        conclusion = "full_outcome_changes_persist_through_residual_path_only"
    elif gate_only_explains_every_change and residual_only_explains_every_change:
        conclusion = "full_outcome_changes_persist_independently_through_both_paths"
    else:
        conclusion = "full_outcome_changes_are_not_explained_by_one_isolated_path"
    result = {
        "protocol": "sugar_causal_composer_frozen_gate_residual_ablation_v1",
        "training_or_optimizer_updates": 0,
        "evaluation_seed": args.evaluation_seed,
        "prefixes": prefixes,
        "profiles_per_arm": 20 * len(prefixes),
        "contexts": contexts,
        "count_totals": totals,
        "full_outcome_attribution": all_full_outcome_attribution,
        "visualization_targets": visualization_targets,
        "checks": {
            "all_initial_584d_inputs_elementwise_identical": True,
            "gate_only_residual_exact_zero": True,
            "residual_only_gate_deviation_exact_zero": True,
            "pre_update_gate_and_residual_exact_zero": True,
            "full_first_action_gate_matches_gate_only": True,
            "full_first_action_residual_matches_residual_only": True,
            "no_policy_training_or_optimizer_updates": True,
        },
        "conclusion": conclusion,
        "claim_boundary": (
            "Frozen same-checkpoint causal attribution only. The diagnostic zeros output rows "
            "without policy training and identifies whether full-policy outcome changes persist "
            "through the learned gate path, residual path, both or neither. It does not establish "
            "a new policy improvement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
