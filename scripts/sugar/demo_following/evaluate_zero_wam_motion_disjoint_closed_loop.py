#!/usr/bin/env python3
"""Audit the frozen motion-disjoint Zero-WAM closed-loop grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_zero_wam_smallbox_closed_loop import (
    audit_trace,
    file_sha256,
    valid_commit,
    valid_sha256,
)


PROTOCOL = "official_zero_wam_motion_disjoint_closed_loop_evidence_v1"
AUDIT_PROTOCOL = "official_zero_wam_motion_disjoint_closed_loop_audit_v1"
SMALLBOX_PROTOCOL = "official_zero_wam_sugar_smallbox_closed_loop_audit_v1"
COMPLETION_PROTOCOL = "official_zero_wam_sugar_bounded_posttraining_completion_audit_v1"
CASE_PROTOCOL = "sugar_zero_wam_motion_disjoint_closed_loop_cases_v1"
EXPECTED_CASE_MANIFEST_SHA256 = (
    "4be15b98fc8b0b71792dee053e657e39bdeaf0f8dd68840514c5b2d08f1d05e5"
)
EXPECTED_CASE_COUNT = 760
EXPECTED_ENDPOINT_BASELINES = 380
EXPECTED_FLOW_GROUPS = 190
PROFILE_COUNT = 10
ROLLOUT_STEPS = 650
SCORE_STEPS = tuple(range(49, ROLLOUT_STEPS, 50))
EXPECTED_FLOW_ROWS = EXPECTED_FLOW_GROUPS * len(SCORE_STEPS)
MINIMUM_SAFE_PER_SOURCE = 8
CONDITIONS = ("matched", "reversed", "same_task_alternate", "wrong_task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smallbox-audit", type=Path)
    parser.add_argument("--training-completion", type=Path)
    parser.add_argument("--case-result", type=Path)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument("--flow-scores", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(base: Path, reference: Any) -> tuple[Path, str]:
    if not isinstance(reference, dict):
        return Path("/__missing_motion_disjoint_trace__"), ""
    raw = reference.get("path")
    path = Path(raw) if isinstance(raw, str) else Path("/__missing_motion_disjoint_trace__")
    if not path.is_absolute():
        path = base.parent / path
    expected = reference.get("sha256")
    return path, expected if isinstance(expected, str) else ""


def exact_sign_pvalue(margins: list[float]) -> float:
    nonzero = [value for value in margins if value != 0.0]
    positives = sum(value > 0.0 for value in nonzero)
    n = len(nonzero)
    if n == 0:
        return 1.0
    return sum(math.comb(n, index) for index in range(positives, n + 1)) / (2**n)


def holm_pass(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    decisions: dict[str, bool] = {}
    family_pass = True
    total = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        family_pass = family_pass and pvalue < alpha / (total - rank)
        decisions[name] = family_pass
    return decisions


def prompted_safe(physical: dict[str, Any], prompted_task: str) -> bool:
    if prompted_task == "CarryBox":
        return bool(physical.get("safe_carry_success"))
    if prompted_task == "KickBox":
        return bool(physical.get("safe_kick_success"))
    return False


def aggregate(
    cases: list[dict[str, Any]],
    evidence_entries: list[dict[str, Any]],
    trace_audits: dict[tuple[str, str], dict[str, Any]],
    flow_rows: list[dict[str, Any]],
    checkpoint_sha256: str,
    prompt_registry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cases_by_id = {case["case_id"]: case for case in cases}
    expected_adapted = {(case_id, "adapted_zero_wam") for case_id in cases_by_id}
    expected_baselines = {
        (case["case_id"], "released_endpoint")
        for case in cases
        if case["endpoint_baseline"]["required"]
    }
    expected_entries = expected_adapted | expected_baselines
    entries_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    entry_contract = isinstance(evidence_entries, list)
    for entry in evidence_entries if isinstance(evidence_entries, list) else []:
        key = (str(entry.get("case_id")), str(entry.get("run_kind")))
        if key in entries_by_key:
            entry_contract = False
        entries_by_key[key] = entry
    entry_contract = entry_contract and set(entries_by_key) == expected_entries
    all_trace_audits_pass = entry_contract and set(trace_audits) == expected_entries and all(
        audit.get("passed") is True for audit in trace_audits.values()
    )
    trace_hashes = [audit.get("trace_sha256") for audit in trace_audits.values()]
    trace_hashes_unique = (
        len(trace_hashes) == len(set(trace_hashes)) == len(expected_entries)
        and all(valid_sha256(value) for value in trace_hashes)
    )

    fingerprint_counts = Counter(
        case["prompt"]["condition_fingerprint_sha256"] for case in cases
    )
    registry_by_fingerprint: dict[str, dict[str, Any]] = {}
    prompt_registry_contract = isinstance(prompt_registry, list)
    for record in prompt_registry if isinstance(prompt_registry, list) else []:
        fingerprint = str(record.get("condition_fingerprint_sha256"))
        if fingerprint in registry_by_fingerprint:
            prompt_registry_contract = False
        registry_by_fingerprint[fingerprint] = record
    prompt_registry_contract = prompt_registry_contract and set(registry_by_fingerprint) == set(
        fingerprint_counts
    )
    cache_hashes: set[str] = set()
    if prompt_registry_contract:
        for fingerprint, record in registry_by_fingerprint.items():
            prompt_registry_contract = prompt_registry_contract and (
                record.get("official_prompt_encoder") is True
                and record.get("encode_call_count") == 1
                and record.get("language_enabled") is False
                and valid_sha256(record.get("cache_sha256"))
                and record.get("reuse_rollout_count") == fingerprint_counts[fingerprint]
                and record.get("reuse_step_count") == fingerprint_counts[fingerprint] * 650
                and record.get("cache_sha256") not in cache_hashes
            )
            cache_hashes.add(str(record.get("cache_sha256")))

    initial_state_exact = entry_contract
    prompt_only_groups = entry_contract
    entries_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if entry_contract:
        for key, entry in entries_by_key.items():
            case = cases_by_id[key[0]]
            entries_by_group[case["paired_group_id"]].append(entry)
            initial_state_exact = initial_state_exact and (
                entry.get("adapted_checkpoint_sha256") == checkpoint_sha256
                if key[1] == "adapted_zero_wam"
                else entry.get("exact_released_endpoint") is True
            )
            if key[1] == "adapted_zero_wam":
                fingerprint = case["prompt"]["condition_fingerprint_sha256"]
                registry_record = registry_by_fingerprint.get(fingerprint, {})
                prompt_registry_contract = prompt_registry_contract and (
                    entry.get("prompt_condition_fingerprint_sha256") == fingerprint
                    and entry.get("prompt_cache_sha256")
                    == registry_record.get("cache_sha256")
                )
        for group_entries in entries_by_group.values():
            initial_state_exact = initial_state_exact and (
                len({entry.get("initial_physics_sha256") for entry in group_entries}) == 1
                and len({entry.get("initial_causal_history_sha256") for entry in group_entries}) == 1
                and all(valid_sha256(entry.get("initial_physics_sha256")) for entry in group_entries)
                and all(valid_sha256(entry.get("initial_causal_history_sha256")) for entry in group_entries)
            )
            adapted = [entry for entry in group_entries if entry.get("run_kind") == "adapted_zero_wam"]
            prompt_only_groups = prompt_only_groups and (
                len(adapted) == len(CONDITIONS)
                and len({entry.get("non_prompt_conditioning_sha256") for entry in adapted}) == 1
                and all(valid_sha256(entry.get("non_prompt_conditioning_sha256")) for entry in adapted)
            )

    per_source: dict[tuple[str, int], dict[str, Any]] = {}
    physical_contract = all_trace_audits_pass
    for task, motion_id in sorted(
        {(case["target"]["task"], case["target"]["source_motion_id"]) for case in cases}
    ):
        source_cases = [
            case
            for case in cases
            if case["target"]["task"] == task
            and case["target"]["source_motion_id"] == motion_id
        ]
        record: dict[str, Any] = {}
        for condition in ("matched", "wrong_task"):
            condition_cases = [case for case in source_cases if case["condition"] == condition]
            adapted_physical = [
                trace_audits.get((case["case_id"], "adapted_zero_wam"), {}).get(
                    "physical", {}
                )
                for case in condition_cases
            ]
            baseline_physical = [
                trace_audits.get((case["case_id"], "released_endpoint"), {}).get(
                    "physical", {}
                )
                for case in condition_cases
            ]
            prompted_task = condition_cases[0]["prompt"]["task"] if condition_cases else ""
            safe_count = sum(prompted_safe(row, prompted_task) for row in adapted_physical)
            adapted_falls = sum(bool(row.get("physical_fall")) for row in adapted_physical)
            baseline_falls = sum(bool(row.get("physical_fall")) for row in baseline_physical)
            record[f"{condition}_prompted_task"] = prompted_task
            record[f"{condition}_safe_count"] = safe_count
            record[f"{condition}_adapted_falls"] = adapted_falls
            record[f"{condition}_baseline_falls"] = baseline_falls
            physical_contract = physical_contract and (
                len(condition_cases) == PROFILE_COUNT
                and safe_count >= MINIMUM_SAFE_PER_SOURCE
                and adapted_falls <= baseline_falls
            )
        per_source[(task, motion_id)] = record

    expected_flow_keys = {
        (
            case["paired_group_id"],
            case["target"]["task"],
            case["target"]["source_motion_id"],
            case["profile_index"],
            score_step,
        )
        for case in cases
        if case["condition"] == "matched"
        for score_step in SCORE_STEPS
    }
    flow_by_key: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    flow_contract = isinstance(flow_rows, list)
    for row in flow_rows if isinstance(flow_rows, list) else []:
        try:
            key = (
                str(row.get("paired_group_id")),
                str(row.get("target_task")),
                int(row.get("target_source_motion_id", -1)),
                int(row.get("profile_index", -1)),
                int(row.get("score_step", -1)),
            )
        except (TypeError, ValueError):
            flow_contract = False
            continue
        if key in flow_by_key:
            flow_contract = False
        flow_by_key[key] = row
    flow_contract = flow_contract and set(flow_by_key) == expected_flow_keys
    source_margins: dict[tuple[str, int], dict[str, float]] = {}
    if flow_contract:
        matched_cases_by_group = {
            case["paired_group_id"]: case for case in cases if case["condition"] == "matched"
        }
        margins_by_profile: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(
            lambda: {"order": [], "identity": []}
        )
        for key, row in flow_by_key.items():
            matched_case = matched_cases_by_group.get(key[0], {})
            matched_case_id = matched_case.get("case_id")
            matched_trace_sha = trace_audits.get(
                (str(matched_case_id), "adapted_zero_wam"), {}
            ).get("trace_sha256")
            values = [
                row.get("matched_flow_loss"),
                row.get("reversed_flow_loss"),
                row.get("same_task_alternate_flow_loss"),
            ]
            row_valid = (
                row.get("model_checkpoint_sha256") == checkpoint_sha256
                and row.get("causal_trace_sha256") == matched_trace_sha
                and valid_sha256(matched_trace_sha)
                and row.get("matched_score_noise_seed")
                == matched_case.get("matched_score_noise_seed")
                and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
                and row.get("official_video_flow_loss") is True
                and row.get("language_enabled") is False
                and row.get("evaluation_target_used_as_model_input") is False
                and valid_sha256(row.get("matched_noise_sha256"))
                and row.get("matched_noise_sha256") == row.get("reversed_noise_sha256")
                and row.get("matched_noise_sha256")
                == row.get("same_task_alternate_noise_sha256")
                and valid_sha256(row.get("matched_predicted_future_sha256"))
                and valid_sha256(row.get("reversed_predicted_future_sha256"))
                and valid_sha256(row.get("same_task_alternate_predicted_future_sha256"))
                and row.get("matched_predicted_future_sha256")
                != row.get("reversed_predicted_future_sha256")
                and row.get("matched_predicted_future_sha256")
                != row.get("same_task_alternate_predicted_future_sha256")
            )
            flow_contract = flow_contract and row_valid
            if not row_valid:
                continue
            margins_by_profile[(key[1], key[2], key[3])]["order"].append(
                float(row["reversed_flow_loss"]) - float(row["matched_flow_loss"])
            )
            margins_by_profile[(key[1], key[2], key[3])]["identity"].append(
                float(row["same_task_alternate_flow_loss"])
                - float(row["matched_flow_loss"])
            )
        margins_by_source: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
            lambda: {"order": [], "identity": []}
        )
        for profile, margins in margins_by_profile.items():
            flow_contract = flow_contract and all(
                len(values) == len(SCORE_STEPS) for values in margins.values()
            )
            for name, values in margins.items():
                if values:
                    margins_by_source[(profile[0], profile[1])][name].append(
                        sum(values) / len(values)
                    )
        for source, margins in margins_by_source.items():
            flow_contract = flow_contract and all(
                len(values) == PROFILE_COUNT for values in margins.values()
            )
            if all(values for values in margins.values()):
                source_margins[source] = {
                    name: sum(values) / len(values) for name, values in margins.items()
                }

    statistical: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    stats_contract = flow_contract and len(source_margins) == 19
    for name in ("order", "identity"):
        values = [margins[name] for margins in source_margins.values()]
        pvalue = exact_sign_pvalue(values)
        pvalues[name] = pvalue
        per_task_means: dict[str, float] = {}
        for task in ("CarryBox", "KickBox"):
            task_values = [
                value[name] for source, value in source_margins.items() if source[0] == task
            ]
            per_task_means[task] = (
                sum(task_values) / len(task_values) if task_values else 0.0
            )
        statistical[name] = {
            "source_motion_count": len(values),
            "mean_margin": sum(values) / len(values) if values else 0.0,
            "win_rate": sum(value > 0.0 for value in values) / len(values) if values else 0.0,
            "one_sided_exact_sign_p": pvalue,
            "per_task_mean_margin": per_task_means,
        }
        stats_contract = stats_contract and (
            statistical[name]["mean_margin"] > 0.0
            and statistical[name]["win_rate"] > 0.5
            and all(value > 0.0 for value in per_task_means.values())
        )
    holm = holm_pass(pvalues)
    stats_contract = stats_contract and all(holm.values())
    for name in statistical:
        statistical[name]["holm_pass"] = holm.get(name, False)

    checks = {
        "exact_760_adapted_plus_380_endpoint_entries": (
            entry_contract and len(evidence_entries) == 1140
        ),
        "all_1140_trace_hashes_unique_and_content_audits_pass": (
            all_trace_audits_pass and trace_hashes_unique
        ),
        "paired_initial_physics_and_history_exact": initial_state_exact,
        "adapted_four_conditions_change_prompt_only": prompt_only_groups,
        "every_unique_prompt_encoded_once_and_cache_bound_to_traces": prompt_registry_contract,
        "every_source_matched_and_wrong_task_reaches_8_of_10_without_fall_regression": (
            physical_contract and len(per_source) == 19
        ),
        "exact_2470_full_horizon_matched_noise_flow_rows": flow_contract,
        "order_and_identity_source_motion_statistics_pass": stats_contract,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "per_source_physical": {
            f"{task}/{motion_id}": value for (task, motion_id), value in per_source.items()
        },
        "statistics": statistical,
        "trace_entry_count": len(evidence_entries),
        "flow_score_row_count": len(flow_rows),
        "flow_score_steps": list(SCORE_STEPS),
    }


def evaluate(
    smallbox_path: Path,
    completion_path: Path,
    case_result_path: Path,
    case_manifest_path: Path,
    project_root: Path,
    evidence_path: Path,
    flow_path: Path,
) -> dict[str, Any]:
    smallbox = read_json(smallbox_path)
    completion = read_json(completion_path)
    case_result = read_json(case_result_path)
    cases = read_jsonl(case_manifest_path)
    evidence = read_json(evidence_path)
    flow_rows = read_jsonl(flow_path)
    checkpoint = completion.get("final_checkpoint_sha256")
    model_commit = completion.get("model_commit")
    provenance_checks = {
        "evidence_protocol_exact": evidence.get("protocol") == PROTOCOL,
        "smallbox_gate_passed_same_checkpoint": (
            smallbox.get("protocol") == SMALLBOX_PROTOCOL
            and smallbox.get("passed") is True
            and smallbox.get("adapted_checkpoint_sha256") == checkpoint
            and evidence.get("smallbox_audit_sha256") == file_sha256(smallbox_path)
        ),
        "formal_completion_passed": (
            completion.get("protocol") == COMPLETION_PROTOCOL
            and completion.get("passed") is True
            and evidence.get("training_completion_sha256") == file_sha256(completion_path)
        ),
        "same_official_commit_and_checkpoint": (
            valid_commit(model_commit)
            and valid_sha256(checkpoint)
            and evidence.get("model_commit") == model_commit
            and evidence.get("adapted_checkpoint_sha256") == checkpoint
        ),
        "exact_frozen_case_contract": (
            case_result.get("protocol") == CASE_PROTOCOL
            and case_result.get("passed") is True
            and case_result.get("case_manifest_sha256") == EXPECTED_CASE_MANIFEST_SHA256
            and file_sha256(case_manifest_path) == EXPECTED_CASE_MANIFEST_SHA256
            and len(cases) == EXPECTED_CASE_COUNT
            and case_result.get("endpoint_baseline_rollout_count")
            == EXPECTED_ENDPOINT_BASELINES
            and evidence.get("case_result_sha256") == file_sha256(case_result_path)
            and evidence.get("case_manifest_sha256") == EXPECTED_CASE_MANIFEST_SHA256
        ),
        "flow_score_file_hash_exact": evidence.get("flow_scores_sha256") == file_sha256(flow_path),
        "zero_optimizer_updates_during_grid": evidence.get("optimizer_step_count") == 0,
        "parameters_unchanged_during_grid": (
            evidence.get("parameters_before_sha256") == checkpoint
            and evidence.get("parameters_after_sha256") == checkpoint
        ),
    }
    evidence_entries = evidence.get("trace_entries", [])
    trace_audits: dict[tuple[str, str], dict[str, Any]] = {}
    endpoint_checkpoint_hashes: dict[Path, str] = {}

    def cached_endpoint_sha256(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in endpoint_checkpoint_hashes:
            endpoint_checkpoint_hashes[resolved] = file_sha256(resolved)
        return endpoint_checkpoint_hashes[resolved]

    if isinstance(evidence_entries, list):
        cases_by_id = {case["case_id"]: case for case in cases}
        for entry in evidence_entries:
            case_id = str(entry.get("case_id"))
            run_kind = str(entry.get("run_kind"))
            case = cases_by_id.get(case_id, {})
            trace_path, trace_hash = resolve(evidence_path, entry.get("trace"))
            audit, _ = audit_trace(
                trace_path,
                trace_hash,
                run_kind,
                entry.get("prompt_cache_sha256") if run_kind == "adapted_zero_wam" else None,
                str(checkpoint),
            )
            if run_kind == "released_endpoint" and case:
                baseline = case["endpoint_baseline"]
                generator = project_root / baseline["generator_checkpoint"]
                tracker = project_root / baseline["tracker_checkpoint"]
                audit["passed"] = audit.get("passed") is True and (
                    generator.is_file()
                    and tracker.is_file()
                    and entry.get("generator_checkpoint_sha256")
                    == cached_endpoint_sha256(generator)
                    and entry.get("tracker_checkpoint_sha256")
                    == cached_endpoint_sha256(tracker)
                )
            trace_audits[(case_id, run_kind)] = audit
    aggregate_result = aggregate(
        cases,
        evidence_entries,
        trace_audits,
        flow_rows,
        str(checkpoint),
        evidence.get("prompt_caches"),
    )
    checks = {**provenance_checks, **aggregate_result["checks"]}
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "passed": passed,
        "motion_disjoint_demo_following_passed": passed,
        "automatic_next_branch": (
            "evaluate_cross_asset_then_official_humangen"
            if passed
            else "close_broad_demo_following_claim_without_sweep"
        ),
        "per_source_physical": aggregate_result["per_source_physical"],
        "statistics": aggregate_result["statistics"],
        "checks": checks,
        "claim_boundary": (
            "Passing proves motion-disjoint two-task SMALLBOX prompt following with physical, "
            "order and selected-motion gates together; it does not prove cross-asset or HumanGen."
        ),
    }


def make_self_test_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task, motion_ids in {
        "CarryBox": [9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
        "KickBox": [9, 19, 29, 39, 49, 59, 69, 79, 89],
    }.items():
        other = "KickBox" if task == "CarryBox" else "CarryBox"
        for motion_id in motion_ids:
            for profile in range(PROFILE_COUNT):
                group = f"test/{task}/{motion_id}/profile{profile:02d}"
                for condition in CONDITIONS:
                    prompted = other if condition == "wrong_task" else task
                    cases.append(
                        {
                            "case_id": f"{group}/{condition}",
                            "paired_group_id": group,
                            "profile_index": profile,
                            "matched_score_noise_seed": profile,
                            "target": {"task": task, "source_motion_id": motion_id},
                            "condition": condition,
                            "prompt": {
                                "task": prompted,
                                "condition_fingerprint_sha256": hashlib.sha256(
                                    f"prompt/{task}/{motion_id}/{condition}".encode()
                                ).hexdigest(),
                            },
                            "endpoint_baseline": {
                                "required": condition in ("matched", "wrong_task")
                            },
                        }
                    )
    return cases


def run_self_test() -> None:
    checkpoint = "a" * 64
    cases = make_self_test_cases()
    entries: list[dict[str, Any]] = []
    audits: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        run_kinds = ["adapted_zero_wam"]
        if case["endpoint_baseline"]["required"]:
            run_kinds.append("released_endpoint")
        for run_kind in run_kinds:
            entry = {
                "case_id": case["case_id"],
                "run_kind": run_kind,
                "adapted_checkpoint_sha256": checkpoint,
                "exact_released_endpoint": run_kind == "released_endpoint",
                "initial_physics_sha256": hashlib.sha256(
                    f"physics/{case['paired_group_id']}".encode()
                ).hexdigest(),
                "initial_causal_history_sha256": hashlib.sha256(
                    f"history/{case['paired_group_id']}".encode()
                ).hexdigest(),
                "non_prompt_conditioning_sha256": hashlib.sha256(
                    f"nonprompt/{case['paired_group_id']}".encode()
                ).hexdigest(),
            }
            if run_kind == "adapted_zero_wam":
                fingerprint = case["prompt"]["condition_fingerprint_sha256"]
                entry["prompt_condition_fingerprint_sha256"] = fingerprint
                entry["prompt_cache_sha256"] = hashlib.sha256(
                    f"cache/{fingerprint}".encode()
                ).hexdigest()
            entries.append(entry)
            prompted = case["prompt"]["task"]
            physical = {
                "safe_carry_success": prompted == "CarryBox",
                "safe_kick_success": prompted == "KickBox",
                "physical_fall": False,
            }
            audits[(case["case_id"], run_kind)] = {
                "passed": True,
                "trace_sha256": hashlib.sha256(
                    f"trace/{case['case_id']}/{run_kind}".encode()
                ).hexdigest(),
                "physical": physical,
            }
    flow_rows = []
    for case in cases:
        if case["condition"] != "matched":
            continue
        for score_step in SCORE_STEPS:
            flow_rows.append(
                {
                    "paired_group_id": case["paired_group_id"],
                    "target_task": case["target"]["task"],
                    "target_source_motion_id": case["target"]["source_motion_id"],
                    "profile_index": case["profile_index"],
                    "score_step": score_step,
                    "matched_score_noise_seed": case["matched_score_noise_seed"],
                    "model_checkpoint_sha256": checkpoint,
                    "causal_trace_sha256": audits[
                        (case["case_id"], "adapted_zero_wam")
                    ]["trace_sha256"],
                    "matched_flow_loss": 0.2,
                    "reversed_flow_loss": 0.4,
                    "same_task_alternate_flow_loss": 0.5,
                    "official_video_flow_loss": True,
                    "language_enabled": False,
                    "evaluation_target_used_as_model_input": False,
                    "matched_noise_sha256": hashlib.sha256(
                        f"noise/{case['paired_group_id']}/{score_step}".encode()
                    ).hexdigest(),
                    "matched_predicted_future_sha256": hashlib.sha256(
                        f"matched/{case['paired_group_id']}/{score_step}".encode()
                    ).hexdigest(),
                    "reversed_predicted_future_sha256": hashlib.sha256(
                        f"reversed/{case['paired_group_id']}/{score_step}".encode()
                    ).hexdigest(),
                    "same_task_alternate_predicted_future_sha256": hashlib.sha256(
                        f"alternate/{case['paired_group_id']}/{score_step}".encode()
                    ).hexdigest(),
                }
            )
    for row in flow_rows:
        row["reversed_noise_sha256"] = row["matched_noise_sha256"]
        row["same_task_alternate_noise_sha256"] = row["matched_noise_sha256"]
    fingerprint_counts = Counter(
        case["prompt"]["condition_fingerprint_sha256"] for case in cases
    )
    prompt_registry = [
        {
            "condition_fingerprint_sha256": fingerprint,
            "official_prompt_encoder": True,
            "encode_call_count": 1,
            "language_enabled": False,
            "cache_sha256": hashlib.sha256(f"cache/{fingerprint}".encode()).hexdigest(),
            "reuse_rollout_count": count,
            "reuse_step_count": count * 650,
        }
        for fingerprint, count in sorted(fingerprint_counts.items())
    ]
    positive = aggregate(cases, entries, audits, flow_rows, checkpoint, prompt_registry)
    assert positive["passed"] is True, positive

    missing = aggregate(cases, entries[:-1], audits, flow_rows, checkpoint, prompt_registry)
    assert missing["checks"]["exact_760_adapted_plus_380_endpoint_entries"] is False

    missing_audits = dict(audits)
    missing_audits.pop(next(iter(missing_audits)))
    missing_audit = aggregate(
        cases, entries, missing_audits, flow_rows, checkpoint, prompt_registry
    )
    assert missing_audit["checks"]["all_1140_trace_hashes_unique_and_content_audits_pass"] is False

    missing_flow = aggregate(
        cases, entries, audits, flow_rows[:-1], checkpoint, prompt_registry
    )
    assert (
        missing_flow["checks"]["exact_2470_full_horizon_matched_noise_flow_rows"]
        is False
    )

    bad_audits = dict(audits)
    matched_case = next(case for case in cases if case["condition"] == "matched")
    for profile_case in [
        case
        for case in cases
        if case["target"] == matched_case["target"] and case["condition"] == "matched"
    ][:3]:
        key = (profile_case["case_id"], "adapted_zero_wam")
        bad_audits[key] = dict(bad_audits[key])
        bad_audits[key]["physical"] = dict(bad_audits[key]["physical"])
        bad_audits[key]["physical"]["safe_carry_success"] = False
    physical_failure = aggregate(
        cases, entries, bad_audits, flow_rows, checkpoint, prompt_registry
    )
    assert (
        physical_failure["checks"][
            "every_source_matched_and_wrong_task_reaches_8_of_10_without_fall_regression"
        ]
        is False
    )

    bad_order = [dict(row) for row in flow_rows]
    for row in bad_order:
        row["reversed_flow_loss"] = 0.1
    order_failure = aggregate(cases, entries, audits, bad_order, checkpoint, prompt_registry)
    assert order_failure["checks"]["order_and_identity_source_motion_statistics_pass"] is False

    bad_identity = [dict(row) for row in flow_rows]
    for row in bad_identity:
        row["same_task_alternate_flow_loss"] = 0.1
    identity_failure = aggregate(
        cases, entries, audits, bad_identity, checkpoint, prompt_registry
    )
    assert identity_failure["checks"]["order_and_identity_source_motion_statistics_pass"] is False

    mismatched_noise = [dict(row) for row in flow_rows]
    mismatched_noise[0]["reversed_noise_sha256"] = "b" * 64
    noise_failure = aggregate(
        cases, entries, audits, mismatched_noise, checkpoint, prompt_registry
    )
    assert (
        noise_failure["checks"]["exact_2470_full_horizon_matched_noise_flow_rows"]
        is False
    )

    fall_audits = dict(audits)
    baseline_key = next(key for key in audits if key[1] == "released_endpoint")
    adapted_key = (baseline_key[0], "adapted_zero_wam")
    fall_audits[adapted_key] = dict(fall_audits[adapted_key])
    fall_audits[adapted_key]["physical"] = dict(fall_audits[adapted_key]["physical"])
    fall_audits[adapted_key]["physical"]["physical_fall"] = True
    fall_failure = aggregate(
        cases, entries, fall_audits, flow_rows, checkpoint, prompt_registry
    )
    assert (
        fall_failure["checks"][
            "every_source_matched_and_wrong_task_reaches_8_of_10_without_fall_regression"
        ]
        is False
    )

    reencoded_registry = [dict(record) for record in prompt_registry]
    reencoded_registry[0]["encode_call_count"] = 2
    cache_failure = aggregate(
        cases, entries, audits, flow_rows, checkpoint, reencoded_registry
    )
    assert (
        cache_failure["checks"]["every_unique_prompt_encoded_once_and_cache_bound_to_traces"]
        is False
    )
    print(
        json.dumps(
            {
                "self_test_passed": True,
                "positive": {
                    "adapted": 760,
                    "endpoint": 380,
                    "flow_rows": EXPECTED_FLOW_ROWS,
                    "flow_score_steps": list(SCORE_STEPS),
                },
                "rejected": [
                    "missing_trace_entry",
                    "missing_trace_audit",
                    "missing_full_horizon_flow_anchor",
                    "only_7_of_10_safe",
                    "order_margin_reversed",
                    "identity_margin_reversed",
                    "mismatched_diffusion_noise",
                    "fall_regression",
                    "prompt_cache_reencoded",
                ],
                "fixture_claim_boundary": "aggregate contract test only; not model evidence",
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    required = (
        args.smallbox_audit,
        args.training_completion,
        args.case_result,
        args.case_manifest,
        args.project_root,
        args.evidence_json,
        args.flow_scores,
        args.output_dir,
    )
    if any(value is None for value in required):
        raise SystemExit("all audit, case, evidence, flow, project-root and output arguments are required")
    assert args.smallbox_audit is not None
    assert args.training_completion is not None
    assert args.case_result is not None
    assert args.case_manifest is not None
    assert args.project_root is not None
    assert args.evidence_json is not None
    assert args.flow_scores is not None
    assert args.output_dir is not None
    result = evaluate(
        args.smallbox_audit,
        args.training_completion,
        args.case_result,
        args.case_manifest,
        args.project_root.resolve(),
        args.evidence_json,
        args.flow_scores,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "MOTION_DISJOINT_CLOSED_LOOP_AUDIT.json"
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
