#!/usr/bin/env python3
"""Fail-closed model-fidelity and data-scale gate for Zero-WAM training.

The current SUGAR corpus may support a bounded two-task post-training audit only
after a strict-loaded official pretrained Zero-WAM checkpoint passes frozen
prompt-dependence and official embodiment-adapter gates.  It is intentionally
rejected as foundation pre-training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


RELEASE_PROTOCOL = "official_zero_wam_release_strict_admission_v1"
WAN_BASE_PROTOCOL = "official_wan22_ti2v_5b_base_audit_v1"
SUGAR_MANIFEST_PROTOCOL = "sugar_zero_wam_immutable_video_action_icl_manifest_v1"
SUGAR_DIVERSITY_PROTOCOL = "sugar_zero_wam_training_data_diversity_v1"
PROMPT_CASE_PROTOCOL = "sugar_zero_wam_frozen_prompt_gate_cases_v1"
TRAINING_SCHEDULE_PROTOCOL = "sugar_zero_wam_bounded_posttraining_schedule_v1"
PROMPT_GATE_PROTOCOL = "official_zero_wam_frozen_sugar_prompt_gate_v1"
ADAPTER_PROTOCOL = "official_zero_wam_sugar_29dof_adapter_admission_v1"
EXPECTED_SUGAR_ICL_MANIFEST_SHA256 = (
    "24cc2b99b26e3136acb1508e4c1d8a6193702d25a43005978e9920179fc366c8"
)
EXPECTED_FROZEN_PROMPT_CASE_MANIFEST_SHA256 = (
    "035e554a94ecd506e4e4d287f32cf90d21f78f8a5211277f34daedf4516e37f5"
)
EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256 = (
    "0f30252c0315855a1154d2c9f68d78b283cf35d2eee772a16692f5b0f1882ec1"
)

EXPECTED_SUGAR_SCALE = {
    "trajectory_count": 199,
    "train_trajectory_count": 160,
    "validation_trajectory_count": 20,
    "test_trajectory_count": 19,
    "task_count": 2,
    "transition_count": 139_300,
    "video_action_interval_count": 27_860,
    "train_video_action_interval_count": 22_400,
}
OFFICIAL_REPORTED_PRETRAIN_SCALE = {
    "task_diverse_robot_task_count_minimum": 6_000,
    "robot_trajectories_per_epoch_approximate": 400_000,
    "human_gen_task_count": 8_600,
    "human_robot_pair_count": 74_200,
    "pretraining_gpu_hours": 15_360,
}
PROMPT_CHECKS = (
    "exact_frozen_case_manifest",
    "complete_1950_condition_score_matrix",
    "official_release_model_and_checkpoint_provenance",
    "official_next_video_flow_loss",
    "matched_noise",
    "language_disabled",
    "source_motion_level_statistics_after_ten_anchor_reduction",
    "matched_beats_wrong_task_validation",
    "matched_beats_wrong_task_test",
    "matched_beats_reversed_validation",
    "matched_beats_reversed_test",
    "matched_beats_same_task_alternate_validation",
    "matched_beats_same_task_alternate_test",
    "matched_beats_masked_prompt_validation",
    "matched_beats_masked_prompt_test",
    "holm_corrected_sign_tests_pass",
    "predicted_future_changes_before_action_decoder",
)
ADAPTER_CHECKS = (
    "official_release_provenance_and_data_identity",
    "frozen_prompt_gate_passed_same_checkpoint",
    "documented_official_embodiment_adapter",
    "supports_29d_continuous_action_chunks",
    "official_architecture_preserved",
    "zero_update_audit",
    "two_update_optimizer_smoke",
    "joint_video_action_overfit",
    "official_optimizer_and_schedule_recorded",
    "no_local_action_transformer",
    "future_and_outcome_labels_excluded_from_deployed_inputs",
    "exact_predeclared_32_motion_128_sample_diagnostic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-audit", type=Path, required=True)
    parser.add_argument("--wan-base-audit", type=Path, required=True)
    parser.add_argument("--sugar-manifest-result", type=Path, required=True)
    parser.add_argument("--sugar-data-diversity-result", type=Path, required=True)
    parser.add_argument("--prompt-case-result", type=Path, required=True)
    parser.add_argument("--training-schedule-result", type=Path, required=True)
    parser.add_argument("--training-schedule", type=Path, required=True)
    parser.add_argument("--prompt-gate-result", type=Path)
    parser.add_argument("--adapter-audit-result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
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


def checks_all_true(value: dict[str, Any]) -> bool:
    checks = value.get("checks", {})
    return isinstance(checks, dict) and bool(checks) and all(item is True for item in checks.values())


def named_checks_true(value: dict[str, Any], names: tuple[str, ...]) -> bool:
    checks = value.get("checks", {})
    return isinstance(checks, dict) and all(checks.get(name) is True for name in names)


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def evaluate(
    release: dict[str, Any],
    wan_base: dict[str, Any],
    manifest: dict[str, Any],
    diversity: dict[str, Any],
    prompt_cases: dict[str, Any],
    training_schedule: dict[str, Any],
    training_schedule_file_sha256: str,
    prompt: dict[str, Any],
    adapter: dict[str, Any],
) -> dict[str, Any]:
    split_counts = manifest.get("split_counts", {})
    task_counts = manifest.get("task_counts", {})
    sugar_manifest_checks = {
        "manifest_protocol_exact": manifest.get("protocol") == SUGAR_MANIFEST_PROTOCOL,
        "manifest_passed": manifest.get("passed") is True,
        "all_manifest_checks_true": checks_all_true(manifest),
        "trajectory_count_exact": manifest.get("trajectory_count") == 199,
        "train_trajectory_count_exact": split_counts.get("train") == 160,
        "validation_trajectory_count_exact": split_counts.get("validation") == 20,
        "test_trajectory_count_exact": split_counts.get("test") == 19,
        "two_task_counts_exact": task_counts == {"CarryBox": 100, "KickBox": 99},
        "transition_count_exact": manifest.get("transition_count") == 139_300,
        "video_action_interval_count_exact": manifest.get("video_action_interval_count") == 27_860,
        "train_video_action_interval_count_exact": (
            manifest.get("train_video_action_interval_count") == 22_400
        ),
        "action_manifest_hash_recorded": valid_sha256(manifest.get("action_manifest_sha256")),
        "language_disabled": manifest.get("checks", {}).get("language_disabled_for_primary_icl") is True,
        "future_and_outcome_excluded": (
            manifest.get("checks", {}).get("outcome_and_future_fields_excluded_from_deployed_inputs")
            is True
        ),
    }
    sugar_manifest_ready = all(sugar_manifest_checks.values())
    sugar_diversity_checks = {
        "protocol_exact": diversity.get("protocol") == SUGAR_DIVERSITY_PROTOCOL,
        "passed": diversity.get("passed") is True,
        "all_checks_true": checks_all_true(diversity),
        "bound_to_exact_immutable_manifest": (
            diversity.get("manifest_sha256") == EXPECTED_SUGAR_ICL_MANIFEST_SHA256
        ),
        "split_trajectory_counts_exact": (
            diversity.get("split_trajectory_counts")
            == {"train": 160, "validation": 20, "test": 19}
        ),
        "train_task_trajectory_counts_exact": (
            diversity.get("train_task_trajectory_counts")
            == {"CarryBox": 80, "KickBox": 80}
        ),
        "train_task_chunk_counts_exact": (
            diversity.get("train_task_chunk_counts")
            == {"CarryBox": 11_200, "KickBox": 11_200}
        ),
        "exact_22400_train_chunks": diversity.get("train_action_chunk_count") == 22_400,
        "all_train_action_chunks_unique": (
            diversity.get("train_action_chunk_unique_fraction") == 1.0
        ),
        "all_train_observation_chunks_unique": (
            diversity.get("train_observation_chunk_unique_fraction") == 1.0
        ),
        "zero_exact_action_chunk_cross_split_overlap": (
            diversity.get("train_action_heldout_exact_chunk_overlap_count") == 0
        ),
        "zero_exact_observation_chunk_cross_split_overlap": (
            diversity.get("train_observation_heldout_exact_chunk_overlap_count") == 0
        ),
        "all_29_action_dimensions_vary": (
            diversity.get("variable_action_dimension_count_at_1e_4") == 29
        ),
        "action_covariance_numerical_rank_29": (
            diversity.get("action_covariance_numerical_rank_at_relative_1e_6") == 29
        ),
    }
    sugar_diversity_ready = all(sugar_diversity_checks.values())
    sugar_posttraining_data_ready = sugar_manifest_ready and sugar_diversity_ready

    prompt_case_checks = {
        "protocol_exact": prompt_cases.get("protocol") == PROMPT_CASE_PROTOCOL,
        "passed": prompt_cases.get("passed") is True,
        "all_checks_true": checks_all_true(prompt_cases),
        "bound_to_exact_immutable_source_manifest": (
            prompt_cases.get("source_manifest_sha256")
            == EXPECTED_SUGAR_ICL_MANIFEST_SHA256
        ),
        "exact_frozen_case_manifest": (
            prompt_cases.get("case_manifest_sha256")
            == EXPECTED_FROZEN_PROMPT_CASE_MANIFEST_SHA256
        ),
        "exact_39_heldout_source_motions": (
            prompt_cases.get("target_source_motion_count") == 39
        ),
        "exact_390_matched_noise_groups": prompt_cases.get("case_group_count") == 390,
        "exact_1950_condition_instances": (
            prompt_cases.get("condition_instance_count") == 1_950
        ),
    }
    prompt_case_contract_ready = all(prompt_case_checks.values())

    training_schedule_checks = {
        "protocol_exact": (
            training_schedule.get("protocol") == TRAINING_SCHEDULE_PROTOCOL
        ),
        "passed": training_schedule.get("passed") is True,
        "all_checks_true": checks_all_true(training_schedule),
        "bound_to_exact_immutable_source_manifest": (
            training_schedule.get("source_manifest_sha256")
            == EXPECTED_SUGAR_ICL_MANIFEST_SHA256
        ),
        "result_records_exact_schedule_hash": (
            training_schedule.get("schedule_sha256")
            == EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256
        ),
        "actual_schedule_file_hash_exact": (
            training_schedule_file_sha256
            == EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256
        ),
        "exact_160_training_trajectories": (
            training_schedule.get("training_trajectory_count") == 160
        ),
        "exact_80_80_task_balance": (
            training_schedule.get("task_trajectory_counts")
            == {"CarryBox": 80, "KickBox": 80}
        ),
        "minimum_10_complete_epochs": (
            training_schedule.get("minimum_full_epochs") == 10
        ),
        "exact_224000_atomic_interval_exposure_floor": (
            training_schedule.get("minimum_atomic_interval_exposures") == 224_000
        ),
        "exact_1120000_action_exposure_floor": (
            training_schedule.get("minimum_action_exposures") == 1_120_000
        ),
    }
    training_schedule_ready = all(training_schedule_checks.values())

    wan_base_checks = {
        "protocol_exact": wan_base.get("protocol") == WAN_BASE_PROTOCOL,
        "passed": wan_base.get("passed") is True,
        "all_checks_true": checks_all_true(wan_base),
        "exact_22_file_snapshot": wan_base.get("snapshot_file_count") == 22,
        "exact_snapshot_bytes": wan_base.get("snapshot_bytes") == 34_203_123_632,
        "no_incomplete_files": wan_base.get("incomplete_files") == [],
        "base_claim_not_zero_wam": "not Zero-WAM" in str(wan_base.get("claim_boundary", "")),
    }
    public_wan_base_ready = all(wan_base_checks.values())

    official_commit = release.get("commit")
    release_checks = {
        "protocol_exact": release.get("protocol") == RELEASE_PROTOCOL,
        "release_available": release.get("release_available") is True,
        "strict_admission_passed": release.get("passed") is True,
        "all_release_checks_true": checks_all_true(release),
        "full_commit_recorded": isinstance(official_commit, str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", official_commit)),
        "official_checkpoint_not_public_wan_only": (
            release.get("checks", {}).get("no_local_substitute") is True
        ),
    }
    official_release_ready = all(release_checks.values())

    prompt_checks = {
        "protocol_exact": prompt.get("protocol") == PROMPT_GATE_PROTOCOL,
        "passed": prompt.get("passed") is True,
        "all_checks_true": checks_all_true(prompt),
        "official_provenance": prompt.get("provenance") == "official_zero_wam_release",
        "same_official_commit": prompt.get("model_commit") == official_commit,
        "checkpoint_hash_recorded": valid_sha256(prompt.get("checkpoint_sha256")),
        "exact_frozen_case_manifest": (
            prompt.get("case_manifest_sha256")
            == EXPECTED_FROZEN_PROMPT_CASE_MANIFEST_SHA256
        ),
        "all_fixed_prompt_checks_true": named_checks_true(prompt, PROMPT_CHECKS),
    }
    frozen_prompt_gate_ready = all(prompt_checks.values())

    adapter_checks = {
        "protocol_exact": adapter.get("protocol") == ADAPTER_PROTOCOL,
        "passed": adapter.get("passed") is True,
        "all_checks_true": checks_all_true(adapter),
        "official_provenance": adapter.get("provenance") == "official_zero_wam_release",
        "same_official_commit": adapter.get("model_commit") == official_commit,
        "same_official_checkpoint_as_prompt_gate": (
            valid_sha256(adapter.get("checkpoint_sha256"))
            and adapter.get("checkpoint_sha256") == prompt.get("checkpoint_sha256")
        ),
        "exact_sugar_manifest": (
            adapter.get("sugar_manifest_sha256") == EXPECTED_SUGAR_ICL_MANIFEST_SHA256
        ),
        "all_fixed_adapter_checks_true": named_checks_true(adapter, ADAPTER_CHECKS),
    }
    official_adapter_ready = all(adapter_checks.values())

    bounded_posttraining_checks = {
        "public_wan_base_preflight_ready": public_wan_base_ready,
        "official_zero_wam_release_ready": official_release_ready,
        "sugar_two_task_posttraining_data_ready": sugar_posttraining_data_ready,
        "frozen_prompt_case_contract_ready": prompt_case_contract_ready,
        "bounded_posttraining_schedule_ready": training_schedule_ready,
        "frozen_official_prompt_gate_ready": frozen_prompt_gate_ready,
        "official_29dof_adapter_ready": official_adapter_ready,
    }
    bounded_posttraining_allowed = all(bounded_posttraining_checks.values())

    # The immutable SUGAR corpus is intentionally never admitted as a substitute
    # for Zero-WAM foundation pre-training.  Its two tasks and 199 trajectories
    # are orders of magnitude below the official reported task-balanced scale.
    foundation_pretraining_checks = {
        "at_least_6000_robot_tasks": len(task_counts) >= 6_000,
        "about_400k_robot_trajectories_per_epoch": manifest.get("trajectory_count", 0) >= 400_000,
        "at_least_8600_human_gen_tasks": manifest.get("human_gen_task_count", 0) >= 8_600,
        "at_least_74200_human_robot_pairs": manifest.get("human_robot_pair_count", 0) >= 74_200,
        "official_foundation_training_recipe_and_data": False,
    }
    foundation_pretraining_allowed = all(foundation_pretraining_checks.values())

    if not release.get("release_available"):
        next_branch = "recheck_canonical_official_repository"
    elif not official_release_ready:
        next_branch = "strict_load_official_zero_wam_checkpoint_and_example"
    elif not prompt_case_contract_ready:
        next_branch = "rebuild_exact_frozen_prompt_case_contract"
    elif not training_schedule_ready:
        next_branch = "rebuild_exact_bounded_posttraining_schedule"
    elif not frozen_prompt_gate_ready:
        next_branch = "run_frozen_official_sugar_prompt_gate"
    elif not official_adapter_ready:
        next_branch = "audit_official_sugar_29dof_adapter"
    elif not sugar_manifest_ready:
        next_branch = "repair_immutable_sugar_posttraining_manifest"
    elif not sugar_diversity_ready:
        next_branch = "repair_sugar_training_data_diversity_without_synthesis"
    else:
        next_branch = "run_predeclared_bounded_sugar_posttraining"

    return {
        "protocol": "zero_wam_model_data_training_admission_v1",
        "passed": bounded_posttraining_allowed,
        "training_allowed": bounded_posttraining_allowed,
        "automatic_next_branch": next_branch,
        "official_identity": {
            "model_commit": official_commit,
            "checkpoint_sha256": prompt.get("checkpoint_sha256"),
        },
        "official_reported_pretraining_scale": OFFICIAL_REPORTED_PRETRAIN_SCALE,
        "current_sugar_scale": EXPECTED_SUGAR_SCALE,
        "public_wan_base_preflight": {"ready": public_wan_base_ready, "checks": wan_base_checks},
        "official_zero_wam_release": {"ready": official_release_ready, "checks": release_checks},
        "sugar_bounded_posttraining_data": {
            "ready": sugar_posttraining_data_ready,
            "manifest": {"ready": sugar_manifest_ready, "checks": sugar_manifest_checks},
            "effective_diversity": {
                "ready": sugar_diversity_ready,
                "checks": sugar_diversity_checks,
            },
            "claim_boundary": (
                "Ready only for the fixed two-task, same-embodiment SUGAR post-training audit "
                "from a strict-loaded official pretrained Zero-WAM checkpoint."
            ),
        },
        "frozen_prompt_case_contract": {
            "ready": prompt_case_contract_ready,
            "checks": prompt_case_checks,
        },
        "bounded_posttraining_schedule": {
            "ready": training_schedule_ready,
            "checks": training_schedule_checks,
        },
        "frozen_official_prompt_gate": {"ready": frozen_prompt_gate_ready, "checks": prompt_checks},
        "official_29dof_adapter": {"ready": official_adapter_ready, "checks": adapter_checks},
        "bounded_sugar_posttraining": {
            "allowed": bounded_posttraining_allowed,
            "checks": bounded_posttraining_checks,
        },
        "sugar_foundation_pretraining": {
            "allowed": foundation_pretraining_allowed,
            "checks": foundation_pretraining_checks,
            "decision": "forbidden_do_not_train_5b_from_scratch_on_sugar",
        },
        "claim_boundary": (
            "This gate can admit only a bounded two-task SUGAR post-training run from the official "
            "pretrained Zero-WAM implementation. It never reclassifies the 199-trajectory SUGAR "
            "corpus as foundation-scale data and never admits a local or public-Wan-only substitute."
        ),
    }


def run_self_test() -> None:
    manifest = {
        "protocol": SUGAR_MANIFEST_PROTOCOL,
        "passed": True,
        "trajectory_count": 199,
        "split_counts": {"train": 160, "validation": 20, "test": 19},
        "task_counts": {"CarryBox": 100, "KickBox": 99},
        "transition_count": 139_300,
        "video_action_interval_count": 27_860,
        "train_video_action_interval_count": 22_400,
        "action_manifest_sha256": "a" * 64,
        "checks": {
            "language_disabled_for_primary_icl": True,
            "outcome_and_future_fields_excluded_from_deployed_inputs": True,
        },
    }
    diversity = {
        "protocol": SUGAR_DIVERSITY_PROTOCOL,
        "passed": True,
        "manifest_sha256": EXPECTED_SUGAR_ICL_MANIFEST_SHA256,
        "split_trajectory_counts": {"train": 160, "validation": 20, "test": 19},
        "train_task_trajectory_counts": {"CarryBox": 80, "KickBox": 80},
        "train_task_chunk_counts": {"CarryBox": 11_200, "KickBox": 11_200},
        "train_action_chunk_count": 22_400,
        "train_action_chunk_unique_fraction": 1.0,
        "train_observation_chunk_unique_fraction": 1.0,
        "train_action_heldout_exact_chunk_overlap_count": 0,
        "train_observation_heldout_exact_chunk_overlap_count": 0,
        "variable_action_dimension_count_at_1e_4": 29,
        "action_covariance_numerical_rank_at_relative_1e_6": 29,
        "checks": {"effective_diversity_passed": True},
    }
    prompt_cases = {
        "protocol": PROMPT_CASE_PROTOCOL,
        "passed": True,
        "source_manifest_sha256": EXPECTED_SUGAR_ICL_MANIFEST_SHA256,
        "case_manifest_sha256": EXPECTED_FROZEN_PROMPT_CASE_MANIFEST_SHA256,
        "target_source_motion_count": 39,
        "case_group_count": 390,
        "condition_instance_count": 1_950,
        "checks": {"frozen_contract_passed": True},
    }
    training_schedule = {
        "protocol": TRAINING_SCHEDULE_PROTOCOL,
        "passed": True,
        "source_manifest_sha256": EXPECTED_SUGAR_ICL_MANIFEST_SHA256,
        "schedule_sha256": EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256,
        "training_trajectory_count": 160,
        "task_trajectory_counts": {"CarryBox": 80, "KickBox": 80},
        "minimum_full_epochs": 10,
        "minimum_atomic_interval_exposures": 224_000,
        "minimum_action_exposures": 1_120_000,
        "checks": {"full_schedule_passed": True},
    }
    wan = {
        "protocol": WAN_BASE_PROTOCOL,
        "passed": True,
        "snapshot_file_count": 22,
        "snapshot_bytes": 34_203_123_632,
        "incomplete_files": [],
        "claim_boundary": "Passing is not Zero-WAM.",
        "checks": {"strict_cpu_load_passed": True},
    }
    commit = "b" * 40
    release = {
        "protocol": RELEASE_PROTOCOL,
        "passed": True,
        "release_available": True,
        "commit": commit,
        "checks": {"no_local_substitute": True},
    }
    prompt = {
        "protocol": PROMPT_GATE_PROTOCOL,
        "passed": True,
        "provenance": "official_zero_wam_release",
        "model_commit": commit,
        "checkpoint_sha256": "c" * 64,
        "case_manifest_sha256": EXPECTED_FROZEN_PROMPT_CASE_MANIFEST_SHA256,
        "checks": {name: True for name in PROMPT_CHECKS},
    }
    adapter = {
        "protocol": ADAPTER_PROTOCOL,
        "passed": True,
        "provenance": "official_zero_wam_release",
        "model_commit": commit,
        "checkpoint_sha256": "c" * 64,
        "sugar_manifest_sha256": EXPECTED_SUGAR_ICL_MANIFEST_SHA256,
        "checks": {name: True for name in ADAPTER_CHECKS},
    }
    admitted = evaluate(
        release,
        wan,
        manifest,
        diversity,
        prompt_cases,
        training_schedule,
        EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256,
        prompt,
        adapter,
    )
    assert admitted["bounded_sugar_posttraining"]["allowed"] is True
    assert admitted["sugar_foundation_pretraining"]["allowed"] is False

    no_release = evaluate(
        {},
        wan,
        manifest,
        diversity,
        prompt_cases,
        training_schedule,
        EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256,
        {},
        {},
    )
    assert no_release["sugar_bounded_posttraining_data"]["ready"] is True
    assert no_release["bounded_sugar_posttraining"]["allowed"] is False
    assert no_release["automatic_next_branch"] == "recheck_canonical_official_repository"

    too_small = dict(manifest)
    too_small["trajectory_count"] = 10
    rejected_data = evaluate(
        release,
        wan,
        too_small,
        diversity,
        prompt_cases,
        training_schedule,
        EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256,
        prompt,
        adapter,
    )
    assert rejected_data["sugar_bounded_posttraining_data"]["ready"] is False
    assert rejected_data["bounded_sugar_posttraining"]["allowed"] is False

    duplicated = dict(diversity)
    duplicated["train_action_chunk_unique_fraction"] = 0.5
    rejected_diversity = evaluate(
        release,
        wan,
        manifest,
        duplicated,
        prompt_cases,
        training_schedule,
        EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256,
        prompt,
        adapter,
    )
    assert rejected_diversity["sugar_bounded_posttraining_data"]["ready"] is False
    assert rejected_diversity["bounded_sugar_posttraining"]["allowed"] is False

    incomplete_cases = dict(prompt_cases)
    incomplete_cases["case_group_count"] = 389
    rejected_cases = evaluate(
        release,
        wan,
        manifest,
        diversity,
        incomplete_cases,
        training_schedule,
        EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256,
        prompt,
        adapter,
    )
    assert rejected_cases["frozen_prompt_case_contract"]["ready"] is False
    assert rejected_cases["bounded_sugar_posttraining"]["allowed"] is False
    assert rejected_cases["automatic_next_branch"] == (
        "rebuild_exact_frozen_prompt_case_contract"
    )

    undertrained = dict(training_schedule)
    undertrained["minimum_full_epochs"] = 1
    rejected_schedule = evaluate(
        release,
        wan,
        manifest,
        diversity,
        prompt_cases,
        undertrained,
        EXPECTED_BOUNDED_POSTTRAINING_SCHEDULE_SHA256,
        prompt,
        adapter,
    )
    assert rejected_schedule["bounded_posttraining_schedule"]["ready"] is False
    assert rejected_schedule["bounded_sugar_posttraining"]["allowed"] is False
    assert rejected_schedule["automatic_next_branch"] == (
        "rebuild_exact_bounded_posttraining_schedule"
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
    result = evaluate(
        read_json(args.release_audit),
        read_json(args.wan_base_audit),
        read_json(args.sugar_manifest_result),
        read_json(args.sugar_data_diversity_result),
        read_json(args.prompt_case_result),
        read_json(args.training_schedule_result),
        file_sha256(args.training_schedule),
        read_json(args.prompt_gate_result),
        read_json(args.adapter_audit_result),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "ZERO_WAM_TRAINING_ADMISSION.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
