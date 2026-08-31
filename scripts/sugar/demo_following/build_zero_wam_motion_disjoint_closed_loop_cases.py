#!/usr/bin/env python3
"""Freeze motion-disjoint Zero-WAM closed-loop evaluation cases.

This builder contains no model or scorer. It expands the immutable 19-motion
test split into matched, reversed, same-task-alternate and wrong-task prompt
conditions under ten paired physics profiles per target motion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL = "sugar_zero_wam_motion_disjoint_closed_loop_cases_v1"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "24cc2b99b26e3136acb1508e4c1d8a6193702d25a43005978e9920179fc366c8"
)
EVALUATION_SEED = 291_500
PROFILE_COUNT = 10
ROLLOUT_STEPS = 650
CONDITIONS = ("matched", "reversed", "same_task_alternate", "wrong_task")
EXPECTED_TEST_TASK_COUNTS = {"CarryBox": 10, "KickBox": 9}
EXPECTED_TEST_IDS = {
    "CarryBox": [9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
    "KickBox": [9, 19, 29, 39, 49, 59, 69, 79, 89],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return bytes_sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def deterministic_seed(namespace: str, task: str, motion_id: int, profile: int) -> int:
    payload = f"zero-wam-motion-disjoint-v1/{EVALUATION_SEED}/{namespace}/{task}/{motion_id}/{profile}"
    return int(bytes_sha256(payload.encode("utf-8"))[:8], 16) & 0x7FFF_FFFF


def prompt_spec(target: dict[str, Any], condition: str) -> tuple[str, int, list[int]]:
    if condition == "matched":
        return str(target["task"]), int(target["source_motion_id"]), list(range(64))
    fixed = target["fixed_counterfactual_prompts"][condition]
    frame_order = fixed.get("frame_order", list(range(64)))
    return str(fixed["task"]), int(fixed["source_motion_id"]), list(map(int, frame_order))


def build_cases(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_by_identity = {
        (str(row["task"]), int(row["source_motion_id"])): row for row in source_rows
    }
    targets = sorted(
        (row for row in source_rows if row.get("split") == "test"),
        key=lambda row: (str(row["task"]), int(row["source_motion_id"])),
    )
    cases: list[dict[str, Any]] = []
    for target in targets:
        target_task = str(target["task"])
        target_motion_id = int(target["source_motion_id"])
        target_identity = f"{target_task}/{target_motion_id}"
        for profile_index in range(PROFILE_COUNT):
            paired_group_id = f"test/{target_identity}/profile{profile_index:02d}"
            physics_seed = deterministic_seed(
                "physics", target_task, target_motion_id, profile_index
            )
            score_noise_seed = deterministic_seed(
                "score-noise", target_task, target_motion_id, profile_index
            )
            for condition in CONDITIONS:
                prompt_task, prompt_motion_id, frame_order = prompt_spec(target, condition)
                prompt_source = source_by_identity[(prompt_task, prompt_motion_id)]
                prompt_sequence_sha256 = prompt_source["prompt"]["sequence_sha256"]
                condition_fingerprint = canonical_sha256(
                    {
                        "task": prompt_task,
                        "source_motion_id": prompt_motion_id,
                        "prompt_sequence_sha256": prompt_sequence_sha256,
                        "frame_order": frame_order,
                    }
                )
                case_id = f"{paired_group_id}/{condition}"
                cases.append(
                    {
                        "protocol": PROTOCOL,
                        "case_id": case_id,
                        "paired_group_id": paired_group_id,
                        "evaluation_seed": EVALUATION_SEED,
                        "profile_index": profile_index,
                        "physics_profile_seed": physics_seed,
                        "matched_score_noise_seed": score_noise_seed,
                        "rollout_steps": ROLLOUT_STEPS,
                        "scene_asset": "SMALLBOX",
                        "target": {
                            "split": "test",
                            "task": target_task,
                            "source_motion_id": target_motion_id,
                            "source_manifest_row_sha256": canonical_sha256(target),
                            "robot_sequence_sha256": target["robot_target"][
                                "sequence_sha256_at_64_normalized_indices"
                            ],
                        },
                        "condition": condition,
                        "prompt": {
                            "split": prompt_source["split"],
                            "task": prompt_task,
                            "source_motion_id": prompt_motion_id,
                            "source_manifest_row_sha256": canonical_sha256(prompt_source),
                            "source_sequence_sha256": prompt_sequence_sha256,
                            "frame_order": frame_order,
                            "condition_fingerprint_sha256": condition_fingerprint,
                            "selected_robot_reference_sha256": prompt_source[
                                "robot_target"
                            ]["sequence_sha256_at_64_normalized_indices"],
                        },
                        "deployed_input_contract": {
                            "prompt_rgb": True,
                            "causal_robot_rgb_history": True,
                            "causal_executed_action_history_29d": True,
                            "causal_generator_command_history_36d": True,
                            "causal_tracker_observation_history_510d": True,
                            "future_target": False,
                            "contact_or_success_label": False,
                            "language": False,
                            "selected_demo_id_scalar": False,
                        },
                        "evaluation_only_targets": {
                            "target_robot_sequence_sha256": target["robot_target"][
                                "sequence_sha256_at_64_normalized_indices"
                            ],
                            "may_enter_deployed_model": False,
                            "official_matched_noise_flow_scoring_only": True,
                        },
                        "endpoint_baseline": {
                            "required": condition in ("matched", "wrong_task"),
                            "prompted_task": prompt_task,
                            "generator_checkpoint": prompt_source["expert"][
                                "generator_checkpoint"
                            ],
                            "tracker_checkpoint": prompt_source["expert"][
                                "tracker_checkpoint"
                            ],
                            "must_restore_identical_initial_physics": True,
                            "parameter_exact": True,
                        },
                    }
                )
    return cases


def validate_cases(
    source_rows: list[dict[str, Any]],
    source_manifest_sha256: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_identity = {
        (str(row["task"]), int(row["source_motion_id"])): row for row in source_rows
    }
    test_rows = [row for row in source_rows if row.get("split") == "test"]
    test_task_counts = dict(Counter(str(row["task"]) for row in test_rows))
    test_ids = {
        task: sorted(
            int(row["source_motion_id"])
            for row in test_rows
            if row.get("task") == task
        )
        for task in EXPECTED_TEST_IDS
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, int, int, str]] = set()
    all_rows_semantically_exact = True
    prompt_sources_test_only = True
    for case in cases:
        groups[str(case.get("paired_group_id"))].append(case)
        target = case.get("target", {})
        prompt = case.get("prompt", {})
        condition = case.get("condition")
        identity = (
            str(target.get("task")),
            int(target.get("source_motion_id", -1)),
            int(case.get("profile_index", -1)),
            str(condition),
        )
        identities.add(identity)
        target_source = source_by_identity.get((identity[0], identity[1]), {})
        expected_prompt_task, expected_prompt_id, expected_order = prompt_spec(
            target_source, str(condition)
        ) if target_source else ("", -1, [])
        prompt_source = source_by_identity.get((expected_prompt_task, expected_prompt_id), {})
        expected_prompt_fingerprint = canonical_sha256(
            {
                "task": expected_prompt_task,
                "source_motion_id": expected_prompt_id,
                "prompt_sequence_sha256": prompt_source.get("prompt", {}).get(
                    "sequence_sha256"
                ),
                "frame_order": expected_order,
            }
        )
        allowlist = case.get("deployed_input_contract", {})
        baseline = case.get("endpoint_baseline", {})
        all_rows_semantically_exact = all_rows_semantically_exact and (
            case.get("protocol") == PROTOCOL
            and case.get("evaluation_seed") == EVALUATION_SEED
            and case.get("rollout_steps") == ROLLOUT_STEPS
            and case.get("scene_asset") == "SMALLBOX"
            and target.get("split") == "test"
            and target.get("source_manifest_row_sha256") == canonical_sha256(target_source)
            and target.get("robot_sequence_sha256")
            == target_source.get("robot_target", {}).get(
                "sequence_sha256_at_64_normalized_indices"
            )
            and prompt.get("task") == expected_prompt_task
            and prompt.get("source_motion_id") == expected_prompt_id
            and prompt.get("source_manifest_row_sha256") == canonical_sha256(prompt_source)
            and prompt.get("source_sequence_sha256")
            == prompt_source.get("prompt", {}).get("sequence_sha256")
            and prompt.get("frame_order") == expected_order
            and sorted(prompt.get("frame_order", [])) == list(range(64))
            and prompt.get("condition_fingerprint_sha256") == expected_prompt_fingerprint
            and case.get("physics_profile_seed")
            == deterministic_seed("physics", identity[0], identity[1], identity[2])
            and case.get("matched_score_noise_seed")
            == deterministic_seed("score-noise", identity[0], identity[1], identity[2])
            and allowlist
            == {
                "prompt_rgb": True,
                "causal_robot_rgb_history": True,
                "causal_executed_action_history_29d": True,
                "causal_generator_command_history_36d": True,
                "causal_tracker_observation_history_510d": True,
                "future_target": False,
                "contact_or_success_label": False,
                "language": False,
                "selected_demo_id_scalar": False,
            }
            and case.get("evaluation_only_targets", {}).get("may_enter_deployed_model")
            is False
            and baseline.get("required")
            is (condition in ("matched", "wrong_task"))
            and baseline.get("prompted_task") == expected_prompt_task
            and baseline.get("generator_checkpoint")
            == prompt_source.get("expert", {}).get("generator_checkpoint")
            and baseline.get("tracker_checkpoint")
            == prompt_source.get("expert", {}).get("tracker_checkpoint")
            and baseline.get("must_restore_identical_initial_physics") is True
            and baseline.get("parameter_exact") is True
        )
        prompt_sources_test_only = prompt_sources_test_only and prompt_source.get("split") == "test"

    expected_identities = {
        (task, motion_id, profile, condition)
        for task, motion_ids in EXPECTED_TEST_IDS.items()
        for motion_id in motion_ids
        for profile in range(PROFILE_COUNT)
        for condition in CONDITIONS
    }
    group_contract_exact = len(groups) == 19 * PROFILE_COUNT
    for group_rows in groups.values():
        group_contract_exact = group_contract_exact and (
            len(group_rows) == len(CONDITIONS)
            and {row["condition"] for row in group_rows} == set(CONDITIONS)
            and len({row["physics_profile_seed"] for row in group_rows}) == 1
            and len({row["matched_score_noise_seed"] for row in group_rows}) == 1
            and len({row["target"]["source_manifest_row_sha256"] for row in group_rows}) == 1
            and len(
                {row["prompt"]["condition_fingerprint_sha256"] for row in group_rows}
            )
            == len(CONDITIONS)
        )

    endpoint_baseline_count = sum(
        bool(case["endpoint_baseline"]["required"]) for case in cases
    )
    checks = {
        "bound_to_exact_immutable_source_manifest": (
            source_manifest_sha256 == EXPECTED_SOURCE_MANIFEST_SHA256
        ),
        "exact_19_motion_disjoint_test_targets": len(test_rows) == 19,
        "exact_10_carry_9_kick_test_balance": test_task_counts == EXPECTED_TEST_TASK_COUNTS,
        "exact_predeclared_test_motion_ids": test_ids == EXPECTED_TEST_IDS,
        "exact_760_unique_cases": len(cases) == len(identities) == 760,
        "exact_190_four_condition_paired_groups": group_contract_exact,
        "every_case_matches_immutable_prompt_and_seed_contract": all_rows_semantically_exact,
        "all_counterfactual_prompt_sources_remain_test_only": prompt_sources_test_only,
        "case_identity_set_complete": identities == expected_identities,
        "exact_380_matched_initial_endpoint_baselines": endpoint_baseline_count == 380,
    }
    return {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "source_manifest_sha256": source_manifest_sha256,
        "test_motion_count": len(test_rows),
        "test_task_counts": test_task_counts,
        "profile_count_per_target": PROFILE_COUNT,
        "condition_count": len(CONDITIONS),
        "case_count": len(cases),
        "adapted_rollout_count": len(cases),
        "endpoint_baseline_rollout_count": endpoint_baseline_count,
        "total_trace_rollout_count": len(cases) + endpoint_baseline_count,
        "rollout_steps": ROLLOUT_STEPS,
        "adapted_closed_loop_frames": len(cases) * ROLLOUT_STEPS,
        "endpoint_baseline_frames": endpoint_baseline_count * ROLLOUT_STEPS,
        "total_closed_loop_frames": (
            len(cases) + endpoint_baseline_count
        ) * ROLLOUT_STEPS,
        "decision_contract": {
            "matched_task_success_per_source": "at_least_8_of_10_safe outcomes",
            "wrong_task_switch_per_source": "at_least_8_of_10 safe outcomes for prompted task",
            "fall_rule": "no per-source regression against matched exact released endpoint",
            "order_gate": "matched beats reversed official flow loss at source-motion level",
            "identity_gate": "matched beats same-task-alternate official flow loss at source-motion level",
            "statistics": "positive per-task mean and win-rate>0.5; one-sided exact sign tests with Holm correction across order and identity",
            "claim_boundary": "task, order and selected-motion identity must pass together",
        },
        "checks": checks,
    }


def run_self_test(
    source_rows: list[dict[str, Any]], source_manifest_sha256: str, cases: list[dict[str, Any]]
) -> None:
    positive = validate_cases(source_rows, source_manifest_sha256, cases)
    assert positive["passed"] is True, positive

    incomplete = validate_cases(source_rows, source_manifest_sha256, cases[:-1])
    assert incomplete["checks"]["exact_760_unique_cases"] is False

    duplicated = copy.deepcopy(cases)
    duplicated[1]["condition"] = "matched"
    duplicate_result = validate_cases(source_rows, source_manifest_sha256, duplicated)
    assert duplicate_result["checks"]["exact_190_four_condition_paired_groups"] is False

    wrong_seed = copy.deepcopy(cases)
    wrong_seed[0]["physics_profile_seed"] += 1
    seed_result = validate_cases(source_rows, source_manifest_sha256, wrong_seed)
    assert seed_result["checks"]["every_case_matches_immutable_prompt_and_seed_contract"] is False

    future_leak = copy.deepcopy(cases)
    future_leak[0]["deployed_input_contract"]["future_target"] = True
    leak_result = validate_cases(source_rows, source_manifest_sha256, future_leak)
    assert leak_result["checks"]["every_case_matches_immutable_prompt_and_seed_contract"] is False

    bad_reverse = copy.deepcopy(cases)
    reversed_case = next(row for row in bad_reverse if row["condition"] == "reversed")
    reversed_case["prompt"]["frame_order"] = list(range(64))
    reverse_result = validate_cases(source_rows, source_manifest_sha256, bad_reverse)
    assert reverse_result["checks"]["every_case_matches_immutable_prompt_and_seed_contract"] is False
    print(
        json.dumps(
            {
                "self_test_passed": True,
                "positive_real_case_count": len(cases),
                "positive_real_adapted_frame_budget": len(cases) * ROLLOUT_STEPS,
                "positive_real_total_frame_budget": (len(cases) + 380) * ROLLOUT_STEPS,
                "rejected": [
                    "missing_case",
                    "duplicate_condition",
                    "physics_seed_drift",
                    "future_target_input_leak",
                    "reversed_order_tamper",
                ],
                "fixture_claim_boundary": "contract mutation tests only; not model evidence",
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    source_bytes = args.source_manifest.read_bytes()
    source_manifest_sha256 = bytes_sha256(source_bytes)
    source_rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()]
    cases = build_cases(source_rows)
    result = validate_cases(source_rows, source_manifest_sha256, cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_path = args.output_dir / "MOTION_DISJOINT_CLOSED_LOOP_CASES.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    result["case_manifest_path"] = str(case_path)
    result["case_manifest_sha256"] = file_sha256(case_path)
    result_path = args.output_dir / "MOTION_DISJOINT_CLOSED_LOOP_CASES_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.self_test:
        run_self_test(source_rows, source_manifest_sha256, cases)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
