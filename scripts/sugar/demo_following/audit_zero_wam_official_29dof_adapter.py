#!/usr/bin/env python3
"""Fail-closed evidence audit for an official Zero-WAM SUGAR 29-DoF adapter.

This file contains no model or adapter implementation.  It validates evidence
emitted by a future strict-loaded official release and rejects action-only
learning, local learned modules, parameter-scope drift and weak fixed-data
diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


EVIDENCE_PROTOCOL = "official_zero_wam_sugar_29dof_adapter_evidence_v1"
PROMPT_GATE_PROTOCOL = "official_zero_wam_frozen_sugar_prompt_gate_v1"
OUTPUT_PROTOCOL = "official_zero_wam_sugar_29dof_adapter_admission_v1"
EXPECTED_SUGAR_MANIFEST_SHA256 = (
    "24cc2b99b26e3136acb1508e4c1d8a6193702d25a43005978e9920179fc366c8"
)
EXPECTED_PROMPT_CASE_MANIFEST_SHA256 = (
    "035e554a94ecd506e4e4d287f32cf90d21f78f8a5211277f34daedf4516e37f5"
)
REQUIRED_MODULE_ROLES = {
    "video_transformer",
    "action_transformer",
    "ifp",
    "vae",
    "official_embodiment_adapter",
}
LOSS_NAMES = ("video_flow_loss", "action_flow_loss", "ifp_loss")
EXPECTED_OVERFIT_SOURCE_MOTION_IDS = {
    "CarryBox": [2, 7, 14, 21, 26, 33, 40, 45, 52, 57, 64, 71, 76, 83, 90, 95],
    "KickBox": [2, 7, 14, 21, 26, 33, 40, 45, 52, 57, 64, 71, 76, 83, 90, 95],
}
EXPECTED_OVERFIT_SELECTION_SHA256 = (
    "949346b5e54710f49d7b8ea3683be6d96397a5a7ad22a4c459e9b4137eec645a"
)
EXPECTED_OVERFIT_ANCHOR_CHUNK_INDICES = [21, 49, 77, 105]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument("--prompt-gate-result", type=Path)
    parser.add_argument("--expected-model-commit")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def valid_hash_map(value: Any, minimum_size: int = 1) -> bool:
    return (
        isinstance(value, dict)
        and len(value) >= minimum_size
        and all(isinstance(name, str) and name and valid_sha256(digest) for name, digest in value.items())
    )


def finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def loss_records_valid(records: Any, exact_count: int | None = None) -> bool:
    if not isinstance(records, list) or not records:
        return False
    if exact_count is not None and len(records) != exact_count:
        return False
    previous_step = -1
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("optimizer_step"), int):
            return False
        if int(record["optimizer_step"]) <= previous_step:
            return False
        previous_step = int(record["optimizer_step"])
        if not all(finite_nonnegative(record.get(name)) for name in LOSS_NAMES):
            return False
    return True


def tail_median(records: list[dict[str, Any]], name: str) -> float:
    tail_count = min(10, max(1, len(records) // 4))
    return float(statistics.median(float(record[name]) for record in records[-tail_count:]))


def run_self_test() -> None:
    assert finite_nonnegative(0.0)
    assert not finite_nonnegative(float("nan"))
    records = [
        {"optimizer_step": index, "video_flow_loss": 10 - index, "action_flow_loss": 8 - index / 2, "ifp_loss": 5 - index / 4}
        for index in range(5)
    ]
    assert loss_records_valid(records)
    assert tail_median(records, "video_flow_loss") == 6.0


def require_runtime_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "evidence_json",
            "prompt_gate_result",
            "expected_model_commit",
            "expected_checkpoint_sha256",
            "output_dir",
        )
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise ValueError(f"missing required runtime arguments: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        if args.evidence_json is None:
            return
    require_runtime_args(args)
    assert args.evidence_json is not None
    assert args.prompt_gate_result is not None
    assert args.output_dir is not None
    assert args.expected_model_commit is not None
    assert args.expected_checkpoint_sha256 is not None

    evidence = read_json(args.evidence_json)
    prompt_gate = read_json(args.prompt_gate_result)
    documentation = evidence.get("official_adapter_documentation", {})
    interface = evidence.get("interface", {})
    architecture = evidence.get("architecture", {})
    zero_update = evidence.get("zero_update", {})
    two_update = evidence.get("two_update", {})
    overfit = evidence.get("fixed_data_joint_overfit", {})

    full_commit = bool(re.fullmatch(r"[0-9a-f]{40}", args.expected_model_commit))
    checkpoint_hash_valid = valid_sha256(args.expected_checkpoint_sha256)
    provenance_exact = (
        evidence.get("protocol") == EVIDENCE_PROTOCOL
        and evidence.get("provenance") == "official_zero_wam_release"
        and evidence.get("model_commit") == args.expected_model_commit
        and evidence.get("checkpoint_sha256") == args.expected_checkpoint_sha256
        and evidence.get("sugar_manifest_sha256") == EXPECTED_SUGAR_MANIFEST_SHA256
        and full_commit
        and checkpoint_hash_valid
    )
    prompt_checks = prompt_gate.get("checks", {})
    prompt_gate_same_checkpoint = (
        prompt_gate.get("protocol") == PROMPT_GATE_PROTOCOL
        and prompt_gate.get("passed") is True
        and isinstance(prompt_checks, dict)
        and bool(prompt_checks)
        and all(value is True for value in prompt_checks.values())
        and prompt_gate.get("provenance") == "official_zero_wam_release"
        and prompt_gate.get("model_commit") == args.expected_model_commit
        and prompt_gate.get("checkpoint_sha256") == args.expected_checkpoint_sha256
        and prompt_gate.get("case_manifest_sha256") == EXPECTED_PROMPT_CASE_MANIFEST_SHA256
    )

    documentation_valid = (
        documentation.get("model_commit") == args.expected_model_commit
        and isinstance(documentation.get("repository_relative_path"), str)
        and bool(documentation.get("repository_relative_path"))
        and valid_sha256(documentation.get("file_sha256"))
        and isinstance(documentation.get("official_configuration_name"), str)
        and bool(documentation.get("official_configuration_name"))
        and documentation.get("supports_embodiment_adaptation") is True
    )
    interface_valid = (
        interface.get("continuous_action") is True
        and interface.get("action_dimension") == 29
        and interface.get("action_target") == "executed_sugar_action"
        and interface.get("action_chunking_source") == "official_zero_wam_adapter"
        and isinstance(interface.get("action_chunk_length"), int)
        and int(interface.get("action_chunk_length", 0)) > 0
        and interface.get("future_target_in_deployed_inputs") is False
        and interface.get("outcome_or_success_in_deployed_inputs") is False
        and interface.get("language_enabled_for_primary_icl") is False
    )

    module_roles = architecture.get("module_roles", {})
    architecture_valid = (
        isinstance(module_roles, dict)
        and set(module_roles.values()) >= REQUIRED_MODULE_ROLES
        and architecture.get("architecture_diff_from_official_release") == []
        and architecture.get("local_learned_module_paths") == []
        and isinstance(architecture.get("adapter_glue_code_paths"), list)
        and bool(architecture.get("adapter_glue_code_paths"))
        and architecture.get("uses_official_wan_vae") is True
        and architecture.get("uses_official_video_transformer") is True
        and architecture.get("uses_official_action_transformer") is True
        and architecture.get("uses_official_mot") is True
        and architecture.get("uses_official_ifp") is True
    )

    zero_before = zero_update.get("parameter_sha256_before", {})
    zero_after = zero_update.get("parameter_sha256_after", {})
    zero_update_valid = (
        zero_update.get("optimizer_steps") == 0
        and valid_hash_map(zero_before, minimum_size=5)
        and valid_hash_map(zero_after, minimum_size=5)
        and set(zero_before) == set(module_roles)
        and zero_before == zero_after
        and zero_update.get("parameter_max_abs_delta") == 0.0
        and valid_sha256(zero_update.get("robot_history_fingerprint"))
        and valid_sha256(zero_update.get("matched_noise_fingerprint"))
        and zero_update.get("prompt_only_swap") is True
        and zero_update.get("causal_mask_verified") is True
        and zero_update.get("future_target_excluded_from_inputs") is True
        and zero_update.get("policy_cpu_rng_preserved") is True
        and zero_update.get("policy_cuda_rng_preserved") is True
    )

    two_before = two_update.get("parameter_sha256_before", {})
    two_after = two_update.get("parameter_sha256_after", {})
    selected_modules = two_update.get("selected_trainable_modules", [])
    official_modules = two_update.get("official_config_trainable_modules", [])
    two_hash_maps_valid = (
        valid_hash_map(two_before, minimum_size=5)
        and valid_hash_map(two_after, minimum_size=5)
        and set(two_before) == set(two_after)
    )
    changed_modules = (
        {name for name in two_before if two_before[name] != two_after[name]}
        if two_hash_maps_valid
        else set()
    )
    role_by_module = module_roles if isinstance(module_roles, dict) else {}
    vae_modules = {name for name, role in role_by_module.items() if role == "vae"}
    two_loss_records = two_update.get("loss_records", [])
    gradients = two_update.get("gradient_norms", {})
    two_update_valid = (
        two_update.get("optimizer_steps") == 2
        and two_hash_maps_valid
        and set(two_before) == set(module_roles)
        and isinstance(selected_modules, list)
        and bool(selected_modules)
        and len(selected_modules) == len(set(selected_modules))
        and isinstance(official_modules, list)
        and len(official_modules) == len(set(official_modules))
        and set(selected_modules) == set(official_modules)
        and changed_modules == set(selected_modules)
        and bool(vae_modules)
        and not changed_modules.intersection(vae_modules)
        and loss_records_valid(two_loss_records, exact_count=2)
        and finite_nonnegative(gradients.get("video_transformer"))
        and float(gradients.get("video_transformer", 0.0)) > 0.0
        and finite_nonnegative(gradients.get("action_transformer"))
        and float(gradients.get("action_transformer", 0.0)) > 0.0
        and two_update.get("all_gradients_finite") is True
    )

    optimizer = two_update.get("optimizer", {})
    optimizer_valid = (
        optimizer.get("source") == "official_release_config"
        and isinstance(optimizer.get("name"), str)
        and bool(optimizer.get("name"))
        and isinstance(optimizer.get("schedule"), str)
        and bool(optimizer.get("schedule"))
        and isinstance(optimizer.get("repository_relative_config_path"), str)
        and bool(optimizer.get("repository_relative_config_path"))
        and valid_sha256(optimizer.get("config_sha256"))
    )

    overfit_dataset = overfit.get("dataset", {})
    overfit_records = overfit.get("loss_trace", [])
    overfit_records_valid = loss_records_valid(overfit_records)
    planned_steps = overfit.get("planned_optimizer_steps")
    completed_steps = overfit.get("completed_optimizer_steps")
    first = overfit_records[0] if overfit_records_valid else {}
    final_video = tail_median(overfit_records, "video_flow_loss") if overfit_records_valid else math.inf
    final_action = tail_median(overfit_records, "action_flow_loss") if overfit_records_valid else math.inf
    final_ifp = tail_median(overfit_records, "ifp_loss") if overfit_records_valid else math.inf
    video_ratio = final_video / max(float(first.get("video_flow_loss", 0.0)), 1e-12)
    action_ratio = final_action / max(float(first.get("action_flow_loss", 0.0)), 1e-12)
    ifp_ratio = final_ifp / max(float(first.get("ifp_loss", 0.0)), 1e-12)
    overfit_valid = (
        overfit_dataset.get("source_manifest_sha256") == EXPECTED_SUGAR_MANIFEST_SHA256
        and overfit_dataset.get("split") == "train"
        and overfit_dataset.get("source_motion_count") == 32
        and overfit_dataset.get("task_source_motion_counts")
        == {"CarryBox": 16, "KickBox": 16}
        and overfit_dataset.get("source_motion_ids_by_task")
        == EXPECTED_OVERFIT_SOURCE_MOTION_IDS
        and overfit_dataset.get("selection_sha256") == EXPECTED_OVERFIT_SELECTION_SHA256
        and overfit_dataset.get("anchor_chunk_indices")
        == EXPECTED_OVERFIT_ANCHOR_CHUNK_INDICES
        and overfit_dataset.get("causal_sample_count") == 128
        and overfit_dataset.get("future_target_in_inputs") is False
        and overfit_dataset.get("outcome_or_success_label_used") is False
        and isinstance(planned_steps, int)
        and planned_steps >= 2
        and completed_steps == planned_steps
        and overfit.get("plan_frozen_before_run") is True
        and overfit.get("stopping_rule_source") == "official_release_config"
        and overfit_records_valid
        and overfit_records[0]["optimizer_step"] == 0
        and overfit_records[-1]["optimizer_step"] == completed_steps
        and all(float(first.get(name, 0.0)) > 0.0 for name in LOSS_NAMES)
        and video_ratio <= 0.5
        and action_ratio <= 0.5
        and ifp_ratio <= 1.0
    )

    checks = {
        "official_release_provenance_and_data_identity": provenance_exact,
        "frozen_prompt_gate_passed_same_checkpoint": prompt_gate_same_checkpoint,
        "documented_official_embodiment_adapter": documentation_valid,
        "supports_29d_continuous_action_chunks": interface_valid,
        "official_architecture_preserved": architecture_valid,
        "zero_update_audit": zero_update_valid,
        "two_update_optimizer_smoke": two_update_valid,
        "joint_video_action_overfit": overfit_valid,
        "official_optimizer_and_schedule_recorded": optimizer_valid,
        "no_local_action_transformer": architecture.get("local_learned_module_paths") == []
        and architecture.get("architecture_diff_from_official_release") == [],
        "future_and_outcome_labels_excluded_from_deployed_inputs": (
            interface.get("future_target_in_deployed_inputs") is False
            and interface.get("outcome_or_success_in_deployed_inputs") is False
            and overfit_dataset.get("future_target_in_inputs") is False
            and overfit_dataset.get("outcome_or_success_label_used") is False
        ),
        "exact_predeclared_32_motion_128_sample_diagnostic": (
            overfit_dataset.get("source_motion_count") == 32
            and overfit_dataset.get("task_source_motion_counts")
            == {"CarryBox": 16, "KickBox": 16}
            and overfit_dataset.get("source_motion_ids_by_task")
            == EXPECTED_OVERFIT_SOURCE_MOTION_IDS
            and overfit_dataset.get("selection_sha256")
            == EXPECTED_OVERFIT_SELECTION_SHA256
            and overfit_dataset.get("anchor_chunk_indices")
            == EXPECTED_OVERFIT_ANCHOR_CHUNK_INDICES
            and overfit_dataset.get("causal_sample_count") == 128
        ),
    }
    passed = all(checks.values())
    result = {
        "protocol": OUTPUT_PROTOCOL,
        "passed": passed,
        "provenance": "official_zero_wam_release",
        "model_commit": args.expected_model_commit,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "sugar_manifest_sha256": EXPECTED_SUGAR_MANIFEST_SHA256,
        "evidence_sha256": file_sha256(args.evidence_json),
        "overfit_loss_ratios": {
            "video_flow_tail_median_over_initial": video_ratio,
            "action_flow_tail_median_over_initial": action_ratio,
            "ifp_tail_median_over_initial": ifp_ratio,
        },
        "changed_modules_after_two_updates": sorted(changed_modules),
        "checks": checks,
        "automatic_next_branch": (
            "run_predeclared_bounded_sugar_posttraining"
            if passed
            else "close_unsupported_or_unfaithful_sugar_adapter"
        ),
        "claim_boundary": (
            "Passing proves only that the official checkpoint exposes a faithful 29-DoF "
            "adaptation path and that joint video/action optimization is executable on a balanced "
            "fixed-data diagnostic. It is not a post-trained model or closed-loop result."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "OFFICIAL_29DOF_ADAPTER_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
