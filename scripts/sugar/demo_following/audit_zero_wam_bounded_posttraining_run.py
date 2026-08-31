#!/usr/bin/env python3
"""Audit completion of a serious official Zero-WAM SUGAR post-training run.

This is an evidence auditor, not a model implementation or training substitute.
It admits a checkpoint to frozen evaluation only after the official training
entrypoint has consumed at least ten complete, predeclared SUGAR epochs with
joint video/action/IFP supervision on an H200 Slurm compute node.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PROTOCOL = "official_zero_wam_sugar_bounded_posttraining_run_v1"
ADMISSION_PROTOCOL = "zero_wam_model_data_training_admission_v1"
SCHEDULE_PROTOCOL = "sugar_zero_wam_bounded_posttraining_schedule_v1"
CONSUMPTION_PROTOCOL = "official_zero_wam_sugar_atomic_training_consumption_v1"
EXPECTED_SCHEDULE_SHA256 = (
    "0f30252c0315855a1154d2c9f68d78b283cf35d2eee772a16692f5b0f1882ec1"
)
EXPECTED_SEED = 271_500
MINIMUM_EPOCHS = 10
TRAJECTORIES_PER_EPOCH = 160
INTERVALS_PER_TRAJECTORY = 140
ACTIONS_PER_TRAJECTORY = 700
MINIMUM_INTERVAL_EXPOSURES = 224_000
MINIMUM_ACTION_EXPOSURES = 1_120_000
LOSS_FIELDS = ("video_flow_loss", "action_flow_loss", "ifp_loss")
GRADIENT_FIELDS = (
    "video_gradient_norm",
    "action_gradient_norm",
    "ifp_gradient_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-admission", type=Path)
    parser.add_argument("--training-schedule", type=Path)
    parser.add_argument("--run-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    """Hash one file or a recursively complete sharded-checkpoint directory."""
    if path.is_file():
        return file_sha256(path)
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(value for value in path.rglob("*") if value.is_file())
        for value in files:
            relative = value.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(file_sha256(value)))
        return digest.hexdigest()
    return ""


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def valid_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))


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


def resolve_artifact(evidence_path: Path, reference: Any) -> tuple[Path, str]:
    if not isinstance(reference, dict):
        return Path("/__missing_zero_wam_artifact__"), ""
    raw_path = reference.get("path")
    path = Path(raw_path) if isinstance(raw_path, str) else Path("/__missing_zero_wam_artifact__")
    if not path.is_absolute():
        path = evidence_path.parent / path
    expected_hash = reference.get("sha256")
    return path, expected_hash if isinstance(expected_hash, str) else ""


def finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0


def frozen_epoch_order(epoch_index: int, trajectory_ids: set[str]) -> list[str]:
    ids_by_task = {
        task: [value for value in trajectory_ids if value.startswith(f"train/{task}/")]
        for task in ("CarryBox", "KickBox")
    }

    def order_key(trajectory_id: str) -> str:
        payload = f"zero-wam-sugar-bounded-v1/{EXPECTED_SEED}/{epoch_index}/{trajectory_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    carry = sorted(ids_by_task["CarryBox"], key=order_key)
    kick = sorted(ids_by_task["KickBox"], key=order_key)
    first_task = "CarryBox" if epoch_index % 2 == 0 else "KickBox"
    order: list[str] = []
    for carry_id, kick_id in zip(carry, kick, strict=True):
        order.extend([carry_id, kick_id] if first_task == "CarryBox" else [kick_id, carry_id])
    return order


def evaluate(
    admission_path: Path,
    schedule_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    admission = read_json(admission_path)
    schedule = read_json(schedule_path)
    evidence = read_json(evidence_path)
    admission_hash = file_sha256(admission_path)
    schedule_hash = file_sha256(schedule_path)

    identity = evidence.get("official_identity", {})
    admitted_identity = admission.get("official_identity", {})
    artifacts = evidence.get("artifacts", {})
    epoch_path, epoch_hash = resolve_artifact(evidence_path, artifacts.get("epoch_exposure_log"))
    optimizer_path, optimizer_hash = resolve_artifact(
        evidence_path, artifacts.get("optimizer_trace")
    )
    module_path, module_hash = resolve_artifact(evidence_path, artifacts.get("module_hash_audit"))
    final_path, final_hash = resolve_artifact(evidence_path, artifacts.get("final_checkpoint"))
    consumption_audit_path, consumption_audit_hash = resolve_artifact(
        evidence_path, artifacts.get("atomic_consumption_audit")
    )
    consumption_log_path, consumption_log_hash = resolve_artifact(
        evidence_path, artifacts.get("atomic_consumption_log")
    )
    artifact_paths = (
        epoch_path,
        optimizer_path,
        module_path,
        final_path,
        consumption_audit_path,
        consumption_log_path,
    )
    artifact_expected_hashes = (
        epoch_hash,
        optimizer_hash,
        module_hash,
        final_hash,
        consumption_audit_hash,
        consumption_log_hash,
    )
    artifact_hashes_valid = all(valid_sha256(value) for value in artifact_expected_hashes)
    artifact_files_exist = (
        epoch_path.is_file()
        and optimizer_path.is_file()
        and module_path.is_file()
        and consumption_audit_path.is_file()
        and consumption_log_path.is_file()
        and (
            final_path.is_file()
            or (final_path.is_dir() and any(value.is_file() for value in final_path.rglob("*")))
        )
    )
    artifact_actual_hashes = (
        tuple(artifact_sha256(path) for path in artifact_paths)
        if artifact_files_exist
        else ("", "", "", "", "", "")
    )
    artifact_hashes_exact = artifact_files_exist and artifact_hashes_valid and (
        artifact_actual_hashes == artifact_expected_hashes
    )

    epoch_rows = read_jsonl(epoch_path) if epoch_path.is_file() else []
    optimizer_rows = read_jsonl(optimizer_path) if optimizer_path.is_file() else []
    module_audit = read_json(module_path) if module_path.is_file() else {}
    consumption_audit = (
        read_json(consumption_audit_path) if consumption_audit_path.is_file() else {}
    )

    admission_checks = {
        "admission_protocol_exact": admission.get("protocol") == ADMISSION_PROTOCOL,
        "formal_training_was_admitted": (
            admission.get("passed") is True and admission.get("training_allowed") is True
        ),
        "evidence_bound_to_exact_admission_file": (
            evidence.get("training_admission_sha256") == admission_hash
        ),
        "schedule_protocol_exact": schedule.get("protocol") == SCHEDULE_PROTOCOL,
        "exact_frozen_schedule_file": schedule_hash == EXPECTED_SCHEDULE_SHA256,
        "evidence_bound_to_exact_schedule_file": evidence.get("schedule_sha256") == schedule_hash,
    }

    identity_checks = {
        "run_protocol_exact": evidence.get("protocol") == PROTOCOL,
        "official_commit_recorded": valid_commit(identity.get("model_commit")),
        "official_checkpoint_recorded": valid_sha256(identity.get("initial_checkpoint_sha256")),
        "same_official_commit_as_admission": (
            identity.get("model_commit") == admitted_identity.get("model_commit")
        ),
        "same_official_checkpoint_as_admission": (
            identity.get("initial_checkpoint_sha256")
            == admitted_identity.get("checkpoint_sha256")
        ),
        "official_release_provenance": identity.get("provenance") == "official_zero_wam_release",
        "no_public_wan_only_or_local_substitute": (
            evidence.get("model_fidelity", {}).get("local_learned_modules") == []
            and evidence.get("model_fidelity", {}).get("public_wan_only_substitute") is False
            and evidence.get("model_fidelity", {}).get("official_architecture_preserved") is True
        ),
    }

    recipe = evidence.get("official_recipe", {})
    slurm = evidence.get("slurm_compute", {})
    data = evidence.get("data_use", {})
    execution_checks = {
        "exact_training_seed": evidence.get("training_seed") == EXPECTED_SEED,
        "official_entrypoint_hash_recorded": valid_sha256(recipe.get("entrypoint_sha256")),
        "official_config_hash_recorded": valid_sha256(recipe.get("config_sha256")),
        "official_recipe_unmodified": (
            recipe.get("unmodified") is True
            and recipe.get("repository_clean") is True
            and recipe.get("local_training_diff_paths") == []
        ),
        "official_budget_satisfied": recipe.get("released_budget_satisfied") is True,
        "h200_slurm_compute": (
            isinstance(slurm.get("job_id"), (str, int))
            and bool(str(slurm.get("job_id")))
            and isinstance(slurm.get("node"), str)
            and bool(slurm.get("node"))
            and "H200" in str(slurm.get("gpu_model", "")).upper()
            and slurm.get("ran_on_login_node") is False
        ),
        "train_split_only": (
            data.get("validation_trajectory_exposures") == 0
            and data.get("test_trajectory_exposures") == 0
            and data.get("future_target_input_exposures") == 0
            and data.get("outcome_label_input_exposures") == 0
        ),
        "no_validation_selected_early_stop": data.get("validation_selected_early_stop") is False,
        "all_artifacts_present_and_hash_exact": artifact_hashes_exact,
    }

    schedule_epochs = schedule.get("epochs", [])
    schedule_ids = {
        record.get("trajectory_id") for record in schedule.get("trajectory_records", [])
    }
    rows_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    epoch_indices_valid = True
    for row in epoch_rows:
        epoch_index = row.get("epoch_index")
        if not isinstance(epoch_index, int) or epoch_index < 0:
            epoch_indices_valid = False
            continue
        rows_by_epoch[epoch_index].append(row)
    completed_epoch_indices = sorted(rows_by_epoch)
    contiguous_epochs = completed_epoch_indices == list(range(len(completed_epoch_indices)))
    epoch_orders: list[list[str]] = []
    every_epoch_complete = epoch_indices_valid and bool(rows_by_epoch)
    train_only = True
    exposure_totals = {"atomic_intervals": 0, "actions": 0}
    for epoch_index in completed_epoch_indices:
        rows = sorted(rows_by_epoch[epoch_index], key=lambda row: row.get("position", -1))
        positions = [row.get("position") for row in rows]
        order = [row.get("trajectory_id") for row in rows]
        epoch_orders.append(order)
        every_epoch_complete = every_epoch_complete and (
            len(rows) == TRAJECTORIES_PER_EPOCH
            and positions == list(range(TRAJECTORIES_PER_EPOCH))
            and len(set(order)) == TRAJECTORIES_PER_EPOCH
            and set(order) == schedule_ids
            and all(row.get("atomic_interval_exposures") == INTERVALS_PER_TRAJECTORY for row in rows)
            and all(row.get("action_exposures") == ACTIONS_PER_TRAJECTORY for row in rows)
        )
        train_only = train_only and all(
            isinstance(value, str) and value.startswith("train/") for value in order
        )
        exposure_totals["atomic_intervals"] += sum(
            row.get("atomic_interval_exposures", 0)
            for row in rows
            if isinstance(row.get("atomic_interval_exposures"), int)
        )
        exposure_totals["actions"] += sum(
            row.get("action_exposures", 0)
            for row in rows
            if isinstance(row.get("action_exposures"), int)
        )
    first_ten_match = len(epoch_orders) >= MINIMUM_EPOCHS and all(
        epoch_orders[index] == schedule_epochs[index].get("trajectory_order")
        for index in range(MINIMUM_EPOCHS)
    )
    all_orders_match_frozen_rule = all(
        order == frozen_epoch_order(epoch_index, schedule_ids)
        for epoch_index, order in zip(completed_epoch_indices, epoch_orders, strict=True)
    )
    all_orders_distinct = len({tuple(order) for order in epoch_orders}) == len(epoch_orders)
    coverage_checks = {
        "at_least_10_contiguous_complete_epochs": (
            contiguous_epochs and len(completed_epoch_indices) >= MINIMUM_EPOCHS
        ),
        "every_epoch_has_all_160_train_trajectories_once": every_epoch_complete and train_only,
        "first_10_epoch_orders_match_frozen_schedule": first_ten_match,
        "all_epoch_orders_match_frozen_extension_rule": all_orders_match_frozen_rule,
        "all_completed_epoch_orders_distinct": all_orders_distinct,
        "at_least_224000_atomic_interval_exposures": (
            exposure_totals["atomic_intervals"] >= MINIMUM_INTERVAL_EXPOSURES
        ),
        "at_least_1120000_action_exposures": (
            exposure_totals["actions"] >= MINIMUM_ACTION_EXPOSURES
        ),
    }

    consumption_checks = {
        "atomic_consumption_protocol_exact": (
            consumption_audit.get("protocol") == CONSUMPTION_PROTOCOL
        ),
        "atomic_consumption_audit_passed": consumption_audit.get("passed") is True,
        "all_atomic_consumption_checks_true": (
            isinstance(consumption_audit.get("checks"), dict)
            and bool(consumption_audit.get("checks"))
            and all(value is True for value in consumption_audit["checks"].values())
        ),
        "atomic_consumption_bound_to_exact_schedule": (
            consumption_audit.get("schedule_sha256") == schedule_hash
        ),
        "atomic_consumption_log_hash_exact": (
            valid_sha256(consumption_log_hash)
            and artifact_actual_hashes[5] == consumption_log_hash
            and consumption_audit.get("consumption_log_sha256") == consumption_log_hash
        ),
        "at_least_10_atomic_consumption_epochs": (
            consumption_audit.get("completed_full_epochs", 0) >= MINIMUM_EPOCHS
        ),
        "atomic_audit_proves_224000_interval_exposures": (
            consumption_audit.get("atomic_interval_exposures", 0)
            >= MINIMUM_INTERVAL_EXPOSURES
        ),
        "atomic_audit_proves_1120000_action_exposures": (
            consumption_audit.get("action_exposures", 0) >= MINIMUM_ACTION_EXPOSURES
        ),
    }

    optimizer_steps = [row.get("optimizer_step") for row in optimizer_rows]
    optimizer_epoch_set = {row.get("epoch_index") for row in optimizer_rows}
    losses_finite = all(
        all(finite_nonnegative(row.get(field)) for field in LOSS_FIELDS) for row in optimizer_rows
    )
    gradients_finite = all(
        all(finite_nonnegative(row.get(field)) for field in GRADIENT_FIELDS)
        for row in optimizer_rows
    )
    positive_gradient_sums = {
        field: sum(float(row.get(field, 0.0)) for row in optimizer_rows) for field in GRADIENT_FIELDS
    }
    optimization_checks = {
        "optimizer_trace_nonempty_and_strictly_increasing": (
            bool(optimizer_steps)
            and all(isinstance(step, int) and step >= 0 for step in optimizer_steps)
            and optimizer_steps == sorted(set(optimizer_steps))
        ),
        "every_completed_epoch_has_optimizer_evidence": (
            set(completed_epoch_indices).issubset(optimizer_epoch_set)
        ),
        "optimizer_trace_exactly_matches_atomic_consumption_steps": (
            isinstance(consumption_audit.get("optimizer_step_min"), int)
            and isinstance(consumption_audit.get("optimizer_step_max"), int)
            and isinstance(consumption_audit.get("distinct_optimizer_step_count"), int)
            and optimizer_steps
            == list(
                range(
                    consumption_audit["optimizer_step_min"],
                    consumption_audit["optimizer_step_max"] + 1,
                )
            )
            and len(optimizer_steps) == consumption_audit["distinct_optimizer_step_count"]
        ),
        "joint_video_action_ifp_losses_finite": bool(optimizer_rows) and losses_finite,
        "joint_video_action_ifp_gradients_finite": bool(optimizer_rows) and gradients_finite,
        "video_action_ifp_all_receive_gradient": all(
            value > 0 for value in positive_gradient_sums.values()
        ),
    }

    modules = module_audit.get("modules", [])
    module_rows_valid = isinstance(modules, list) and bool(modules) and all(
        isinstance(row, dict)
        and isinstance(row.get("name"), str)
        and row.get("official_release_module") is True
        and row.get("role") in {"video_world_model", "action_decoder", "ifp_module", "video_vae", "other"}
        and isinstance(row.get("trainable"), bool)
        and valid_sha256(row.get("before_sha256"))
        and valid_sha256(row.get("after_sha256"))
        for row in modules
    )
    trainable_modules = [row for row in modules if row.get("trainable") is True]
    frozen_modules = [row for row in modules if row.get("trainable") is False]
    trainable_roles = {row.get("role") for row in trainable_modules}
    module_checks = {
        "module_hash_manifest_valid": module_rows_valid,
        "official_trainable_scope_exact": module_audit.get("official_trainable_scope_exact") is True,
        "all_and_only_official_trainable_modules_changed": (
            bool(trainable_modules)
            and all(row.get("before_sha256") != row.get("after_sha256") for row in trainable_modules)
            and all(row.get("before_sha256") == row.get("after_sha256") for row in frozen_modules)
        ),
        "video_world_model_and_action_decoder_trained": (
            {"video_world_model", "action_decoder"}.issubset(trainable_roles)
        ),
        "official_video_vae_frozen_exact": any(
            row.get("role") == "video_vae"
            and row.get("trainable") is False
            and row.get("before_sha256") == row.get("after_sha256")
            for row in modules
        ),
        "no_local_module_in_hash_manifest": all(
            row.get("official_release_module") is True for row in modules
        ),
        "final_checkpoint_hash_exact_and_changed": (
            (
                final_path.is_file()
                or (final_path.is_dir() and any(value.is_file() for value in final_path.rglob("*")))
            )
            and valid_sha256(final_hash)
            and artifact_actual_hashes[3] == final_hash
            and final_hash != identity.get("initial_checkpoint_sha256")
        ),
    }

    checks = {
        **admission_checks,
        **identity_checks,
        **execution_checks,
        **coverage_checks,
        **consumption_checks,
        **optimization_checks,
        **module_checks,
    }
    passed = all(checks.values())
    if passed:
        next_branch = "run_frozen_prompt_and_closed_loop_physical_evaluation"
    elif not all(admission_checks.values()) or not all(identity_checks.values()):
        next_branch = "reject_run_and_restore_exact_official_admission_identity"
    elif not all(coverage_checks.values()):
        next_branch = "reject_undertrained_or_off_schedule_run"
    elif not all(consumption_checks.values()):
        next_branch = "reject_unproven_atomic_training_data_consumption"
    elif not all(optimization_checks.values()):
        next_branch = "reject_non_joint_or_nonfinite_training_run"
    elif not all(module_checks.values()):
        next_branch = "reject_architecture_or_parameter_scope_drift"
    else:
        next_branch = "reject_noncompliant_execution_evidence"
    return {
        "protocol": "official_zero_wam_sugar_bounded_posttraining_completion_audit_v1",
        "passed": passed,
        "training_complete": passed,
        "ready_for_frozen_evaluation": passed,
        "automatic_next_branch": next_branch,
        "completed_full_epochs": len(completed_epoch_indices),
        "atomic_interval_exposures": exposure_totals["atomic_intervals"],
        "action_exposures": exposure_totals["actions"],
        "atomic_consumption_audit_passed": all(consumption_checks.values()),
        "optimizer_trace_records": len(optimizer_rows),
        "model_commit": identity.get("model_commit"),
        "initial_checkpoint_sha256": identity.get("initial_checkpoint_sha256"),
        "final_checkpoint_sha256": final_hash if passed else None,
        "checks": checks,
        "claim_boundary": (
            "Passing proves official bounded post-training execution and coverage, not selected-demo "
            "following or physical success; those require the frozen prompt and closed-loop gates."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_self_test() -> None:
    global EXPECTED_SCHEDULE_SHA256
    with tempfile.TemporaryDirectory(prefix="zero_wam_run_audit_") as temp:
        root = Path(temp)
        commit = "b" * 40
        initial_checkpoint = "c" * 64
        trajectory_ids = [
            f"train/{'CarryBox' if index % 2 == 0 else 'KickBox'}/{index}"
            for index in range(TRAJECTORIES_PER_EPOCH)
        ]
        epochs = []
        epoch_rows: list[dict[str, Any]] = []
        for epoch_index in range(MINIMUM_EPOCHS):
            order = frozen_epoch_order(epoch_index, set(trajectory_ids))
            epochs.append({"epoch_index": epoch_index, "trajectory_order": order})
            epoch_rows.extend(
                {
                    "epoch_index": epoch_index,
                    "position": position,
                    "trajectory_id": trajectory_id,
                    "atomic_interval_exposures": INTERVALS_PER_TRAJECTORY,
                    "action_exposures": ACTIONS_PER_TRAJECTORY,
                }
                for position, trajectory_id in enumerate(order)
            )
        schedule = {
            "protocol": SCHEDULE_PROTOCOL,
            "trajectory_records": [{"trajectory_id": value} for value in trajectory_ids],
            "epochs": epochs,
        }
        schedule_path = root / "schedule.json"
        write_json(schedule_path, schedule)
        # Synthetic fixtures test the decision logic without claiming the production
        # schedule identity; real evaluation keeps the immutable module constant.
        EXPECTED_SCHEDULE_SHA256 = file_sha256(schedule_path)

        admission = {
            "protocol": ADMISSION_PROTOCOL,
            "passed": True,
            "training_allowed": True,
            "official_identity": {
                "model_commit": commit,
                "checkpoint_sha256": initial_checkpoint,
            },
        }
        admission_path = root / "admission.json"
        write_json(admission_path, admission)
        epoch_path = root / "epochs.jsonl"
        optimizer_path = root / "optimizer.jsonl"
        module_path = root / "modules.json"
        final_path = root / "final_checkpoint"
        consumption_audit_path = root / "atomic_consumption_audit.json"
        consumption_log_path = root / "atomic_consumption.jsonl"
        evidence_path = root / "evidence.json"
        optimizer_rows = [
            {
                "epoch_index": epoch,
                "optimizer_step": epoch,
                "video_flow_loss": 1.0 / (epoch + 1),
                "action_flow_loss": 0.8 / (epoch + 1),
                "ifp_loss": 0.6 / (epoch + 1),
                "video_gradient_norm": 1.0,
                "action_gradient_norm": 0.5,
                "ifp_gradient_norm": 0.25,
            }
            for epoch in range(MINIMUM_EPOCHS)
        ]
        modules = {
            "official_trainable_scope_exact": True,
            "modules": [
                {"name": "world", "role": "video_world_model", "trainable": True, "official_release_module": True, "before_sha256": "1" * 64, "after_sha256": "2" * 64},
                {"name": "action", "role": "action_decoder", "trainable": True, "official_release_module": True, "before_sha256": "3" * 64, "after_sha256": "4" * 64},
                {"name": "ifp", "role": "ifp_module", "trainable": True, "official_release_module": True, "before_sha256": "5" * 64, "after_sha256": "6" * 64},
                {"name": "vae", "role": "video_vae", "trainable": False, "official_release_module": True, "before_sha256": "7" * 64, "after_sha256": "7" * 64},
            ],
        }
        final_path.mkdir()
        (final_path / "model-00001-of-00002.safetensors").write_bytes(b"official-zero-wam-shard-1")
        (final_path / "model-00002-of-00002.safetensors").write_bytes(b"official-zero-wam-shard-2")
        consumption_log_path.write_text('{"synthetic_contract_only":true}\n', encoding="utf-8")
        good_consumption = {
            "protocol": CONSUMPTION_PROTOCOL,
            "passed": True,
            "schedule_sha256": file_sha256(schedule_path),
            "consumption_log_sha256": file_sha256(consumption_log_path),
            "completed_full_epochs": MINIMUM_EPOCHS,
            "atomic_interval_exposures": MINIMUM_INTERVAL_EXPOSURES,
            "action_exposures": MINIMUM_ACTION_EXPOSURES,
            "optimizer_step_min": 0,
            "optimizer_step_max": MINIMUM_EPOCHS - 1,
            "distinct_optimizer_step_count": MINIMUM_EPOCHS,
            "checks": {"full_atomic_consumption_contract_passed": True},
        }

        def emit(
            epochs_value: list[dict[str, Any]],
            optimizer_value: list[dict[str, Any]],
            modules_value: dict[str, Any],
            *,
            heldout: int = 0,
            consumption_value: dict[str, Any] | None = None,
        ) -> None:
            write_jsonl(epoch_path, epochs_value)
            write_jsonl(optimizer_path, optimizer_value)
            write_json(module_path, modules_value)
            write_json(
                consumption_audit_path,
                good_consumption if consumption_value is None else consumption_value,
            )
            evidence = {
                "protocol": PROTOCOL,
                "training_admission_sha256": file_sha256(admission_path),
                "schedule_sha256": file_sha256(schedule_path),
                "training_seed": EXPECTED_SEED,
                "official_identity": {
                    "provenance": "official_zero_wam_release",
                    "model_commit": commit,
                    "initial_checkpoint_sha256": initial_checkpoint,
                },
                "model_fidelity": {
                    "local_learned_modules": [],
                    "public_wan_only_substitute": False,
                    "official_architecture_preserved": True,
                },
                "official_recipe": {
                    "entrypoint_sha256": "8" * 64,
                    "config_sha256": "9" * 64,
                    "unmodified": True,
                    "repository_clean": True,
                    "local_training_diff_paths": [],
                    "released_budget_satisfied": True,
                },
                "slurm_compute": {
                    "job_id": "1",
                    "node": "compute01",
                    "gpu_model": "NVIDIA H200",
                    "ran_on_login_node": False,
                },
                "data_use": {
                    "validation_trajectory_exposures": heldout,
                    "test_trajectory_exposures": 0,
                    "future_target_input_exposures": 0,
                    "outcome_label_input_exposures": 0,
                    "validation_selected_early_stop": False,
                },
                "artifacts": {
                    "epoch_exposure_log": {"path": epoch_path.name, "sha256": file_sha256(epoch_path)},
                    "optimizer_trace": {"path": optimizer_path.name, "sha256": file_sha256(optimizer_path)},
                    "module_hash_audit": {"path": module_path.name, "sha256": file_sha256(module_path)},
                    "final_checkpoint": {"path": final_path.name, "sha256": artifact_sha256(final_path)},
                    "atomic_consumption_audit": {
                        "path": consumption_audit_path.name,
                        "sha256": file_sha256(consumption_audit_path),
                    },
                    "atomic_consumption_log": {
                        "path": consumption_log_path.name,
                        "sha256": file_sha256(consumption_log_path),
                    },
                },
            }
            write_json(evidence_path, evidence)

        emit(epoch_rows, optimizer_rows, modules)
        positive = evaluate(admission_path, schedule_path, evidence_path)
        assert positive["passed"] is True, positive

        emit([row for row in epoch_rows if row["epoch_index"] < 9], optimizer_rows[:9], modules)
        undertrained = evaluate(admission_path, schedule_path, evidence_path)
        assert undertrained["passed"] is False
        assert undertrained["checks"]["at_least_10_contiguous_complete_epochs"] is False

        tampered_order = copy.deepcopy(epoch_rows)
        tampered_order[0]["trajectory_id"], tampered_order[1]["trajectory_id"] = (
            tampered_order[1]["trajectory_id"], tampered_order[0]["trajectory_id"]
        )
        emit(tampered_order, optimizer_rows, modules)
        order_failure = evaluate(admission_path, schedule_path, evidence_path)
        assert order_failure["checks"]["first_10_epoch_orders_match_frozen_schedule"] is False

        action_only = copy.deepcopy(optimizer_rows)
        for row in action_only:
            row["video_gradient_norm"] = 0.0
        emit(epoch_rows, action_only, modules)
        action_only_failure = evaluate(admission_path, schedule_path, evidence_path)
        assert action_only_failure["checks"]["video_action_ifp_all_receive_gradient"] is False

        missing_optimizer_step = optimizer_rows[:5] + optimizer_rows[6:]
        emit(epoch_rows, missing_optimizer_step, modules)
        optimizer_gap_failure = evaluate(admission_path, schedule_path, evidence_path)
        assert (
            optimizer_gap_failure["checks"][
                "optimizer_trace_exactly_matches_atomic_consumption_steps"
            ]
            is False
        )

        drifted = copy.deepcopy(modules)
        drifted["modules"][-1]["after_sha256"] = "0" * 64
        emit(epoch_rows, optimizer_rows, drifted)
        drift_failure = evaluate(admission_path, schedule_path, evidence_path)
        assert drift_failure["checks"]["official_video_vae_frozen_exact"] is False

        emit(epoch_rows, optimizer_rows, modules, heldout=1)
        heldout_failure = evaluate(admission_path, schedule_path, evidence_path)
        assert heldout_failure["checks"]["train_split_only"] is False

        underconsumed = dict(good_consumption)
        underconsumed["passed"] = False
        underconsumed["atomic_interval_exposures"] = MINIMUM_INTERVAL_EXPOSURES - 1
        underconsumed["checks"] = {"full_atomic_consumption_contract_passed": False}
        emit(epoch_rows, optimizer_rows, modules, consumption_value=underconsumed)
        consumption_failure = evaluate(admission_path, schedule_path, evidence_path)
        assert consumption_failure["checks"]["atomic_consumption_audit_passed"] is False
        assert consumption_failure["checks"]["atomic_audit_proves_224000_interval_exposures"] is False
        print(
            json.dumps(
                {
                    "self_test_passed": True,
                    "positive_fixture_passed": True,
                    "rejected": [
                        "undertrained_nine_epoch",
                        "off_schedule_order",
                        "action_only_gradient",
                        "optimizer_trace_gap",
                        "frozen_vae_drift",
                        "heldout_data_leak",
                        "unproven_atomic_consumption",
                    ],
                    "fixture_claim_boundary": "synthetic contract test only; not model evidence",
                },
                sort_keys=True,
            )
        )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    required = (args.training_admission, args.training_schedule, args.run_evidence, args.output_dir)
    if any(value is None for value in required):
        raise SystemExit(
            "--training-admission, --training-schedule, --run-evidence and --output-dir are required"
        )
    assert args.training_admission is not None
    assert args.training_schedule is not None
    assert args.run_evidence is not None
    assert args.output_dir is not None
    result = evaluate(args.training_admission, args.training_schedule, args.run_evidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "BOUNDED_POSTTRAINING_COMPLETION_AUDIT.json"
    write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
