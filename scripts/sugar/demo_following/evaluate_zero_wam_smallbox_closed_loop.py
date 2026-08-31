#!/usr/bin/env python3
"""Fail-closed SMALLBOX evaluation for one officially post-trained Zero-WAM.

The evaluator recomputes physical outcomes and the predicted-video-to-action
causal chain from immutable per-step traces. It contains no policy or model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL = "official_zero_wam_sugar_smallbox_closed_loop_evidence_v1"
COMPLETION_PROTOCOL = "official_zero_wam_sugar_bounded_posttraining_completion_audit_v1"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "24cc2b99b26e3136acb1508e4c1d8a6193702d25a43005978e9920179fc366c8"
)
EVALUATION_SEED = 281_500
PROFILE_COUNT = 20
ROLLOUT_STEPS = 650
ACTION_DIM = 29
ACTION_EQUALITY_ATOL = 1.0e-6
LIFT_THRESHOLD_M = 0.05
FALL_HEIGHT_LOSS_M = 0.35
FALL_ROOT_TILT_DEG = 60.0
KICK_NET_DISPLACEMENT_M = 0.05
KICK_CONTACT_COUPLED_PATH_M = 0.01
KICK_POST_CONTACT_PATH_M = 0.03
MINIMUM_MATCHED_SUCCESSES = 16
CONDITIONS = {
    "carry45": {"task": "CarryBox", "source_motion_id": 45},
    "kick21": {"task": "KickBox", "source_motion_id": 21},
}
RUN_KINDS = ("released_endpoint", "adapted_zero_wam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-completion", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def valid_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve(evidence_path: Path, reference: Any) -> tuple[Path, str]:
    if not isinstance(reference, dict):
        return Path("/__missing_zero_wam_trace__"), ""
    raw_path = reference.get("path")
    path = Path(raw_path) if isinstance(raw_path, str) else Path("/__missing_zero_wam_trace__")
    if not path.is_absolute():
        path = evidence_path.parent / path
    expected = reference.get("sha256")
    return path, expected if isinstance(expected, str) else ""


def finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def vector_max_abs_difference(left: list[Any], right: list[Any]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def planar_distance(left: list[Any], right: list[Any]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def summarize_physics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    objects = [row["object_position_xyz"] for row in rows]
    heights = [float(row["robot_root_height_m"]) for row in rows]
    quaternions = [row["robot_root_quaternion_wxyz"] for row in rows]
    bilateral = [
        bool(row["left_hand_box_contact"] and row["right_hand_box_contact"])
        for row in rows
    ]
    foot = [
        bool(row["left_foot_box_contact"] or row["right_foot_box_contact"])
        for row in rows
    ]
    baseline_z = sorted(float(value[2]) for value in objects[:25])[len(objects[:25]) // 2]
    maximum_lift = max(float(value[2]) - baseline_z for value in objects)
    root_height_loss = heights[0] - min(heights)
    maximum_tilt = 0.0
    for quaternion in quaternions:
        _, x, y, _ = map(float, quaternion)
        up_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
        maximum_tilt = max(maximum_tilt, math.degrees(math.acos(up_z)))
    physical_fall = root_height_loss >= FALL_HEIGHT_LOSS_M or maximum_tilt >= FALL_ROOT_TILT_DEG
    planar_net = planar_distance(objects[-1], objects[0])
    step_paths = [planar_distance(objects[index], objects[index - 1]) for index in range(1, len(objects))]
    contact_coupled = sum(
        step_paths[index - 1]
        for index in range(1, len(objects))
        if foot[index - 1] or foot[index]
    )
    if any(foot):
        first_contact = foot.index(True)
        post_contact = sum(step_paths[max(first_contact - 1, 0) :])
    else:
        post_contact = 0.0
    carry = any(bilateral) and maximum_lift >= LIFT_THRESHOLD_M
    kick = (
        any(foot)
        and planar_net >= KICK_NET_DISPLACEMENT_M
        and contact_coupled >= KICK_CONTACT_COUPLED_PATH_M
        and post_contact >= KICK_POST_CONTACT_PATH_M
    )
    return {
        "maximum_lift_m": maximum_lift,
        "bilateral_contact_frames": sum(bilateral),
        "foot_contact_frames": sum(foot),
        "planar_object_net_displacement_m": planar_net,
        "contact_coupled_planar_path_m": contact_coupled,
        "post_first_contact_planar_path_m": post_contact,
        "maximum_robot_root_height_loss_m": root_height_loss,
        "maximum_robot_root_tilt_deg": maximum_tilt,
        "physical_fall": physical_fall,
        "carry_success": carry,
        "safe_carry_success": carry and not physical_fall,
        "kick_success": kick,
        "safe_kick_success": kick and not physical_fall,
    }


def audit_trace(
    path: Path,
    expected_hash: str,
    run_kind: str,
    prompt_cache_sha256: str | None,
    adapted_checkpoint_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file() or not valid_sha256(expected_hash):
        return {"passed": False, "reason": "missing_or_unhashed_trace"}, []
    actual_hash = file_sha256(path)
    rows = read_jsonl(path)
    step_exact = len(rows) == ROLLOUT_STEPS and [row.get("step") for row in rows] == list(
        range(ROLLOUT_STEPS)
    )
    physical_valid = step_exact
    adapted_causal_valid = step_exact
    endpoint_exact = step_exact
    for row in rows:
        physical_valid = physical_valid and (
            isinstance(row.get("robot_root_height_m"), (int, float))
            and math.isfinite(float(row.get("robot_root_height_m")))
            and finite_vector(row.get("robot_root_quaternion_wxyz"), 4)
            and abs(sum(float(value) ** 2 for value in row["robot_root_quaternion_wxyz"]) - 1.0)
            <= 1.0e-3
            and finite_vector(row.get("object_position_xyz"), 3)
            and finite_vector(row.get("executed_action_29d"), ACTION_DIM)
            and all(
                isinstance(row.get(name), bool)
                for name in (
                    "left_hand_box_contact",
                    "right_hand_box_contact",
                    "left_foot_box_contact",
                    "right_foot_box_contact",
                    "reset_or_done",
                )
            )
            and row.get("reset_or_done") is False
        )
        if run_kind == "adapted_zero_wam":
            decoded = row.get("decoded_action_29d")
            executed = row.get("executed_action_29d")
            adapted_causal_valid = adapted_causal_valid and (
                row.get("model_checkpoint_sha256") == adapted_checkpoint_sha256
                and row.get("prompt_cache_sha256") == prompt_cache_sha256
                and valid_sha256(row.get("predicted_robot_future_sha256"))
                and row.get("action_decoder_input_future_sha256")
                == row.get("predicted_robot_future_sha256")
                and finite_vector(decoded, ACTION_DIM)
                and finite_vector(executed, ACTION_DIM)
                and vector_max_abs_difference(decoded, executed) <= ACTION_EQUALITY_ATOL
                and row.get("teacher_forced_future_used") is False
                and row.get("future_target_used") is False
                and row.get("outcome_label_used") is False
                and row.get("endpoint_router_used") is False
                and row.get("demo_reward_used") is False
                and row.get("official_video_prediction_used") is True
                and row.get("official_action_decoder_used") is True
            )
        else:
            endpoint_exact = endpoint_exact and row.get("exact_released_generator_tracker") is True
    checks = {
        "trace_file_hash_exact": actual_hash == expected_hash,
        "exact_650_contiguous_steps": step_exact,
        "finite_normalized_physics_and_29d_actions": physical_valid,
        "adapted_predicted_future_to_executed_action_chain_exact": (
            adapted_causal_valid if run_kind == "adapted_zero_wam" else True
        ),
        "released_endpoint_route_exact": endpoint_exact if run_kind == "released_endpoint" else True,
    }
    result = {
        "passed": all(checks.values()),
        "trace_sha256": actual_hash,
        "checks": checks,
    }
    if physical_valid:
        result["physical"] = summarize_physics(rows)
    return result, rows


def evaluate(
    completion_path: Path,
    source_manifest_path: Path,
    project_root: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    completion = read_json(completion_path)
    evidence = read_json(evidence_path)
    completion_hash = file_sha256(completion_path)
    source_hash = file_sha256(source_manifest_path)
    source_rows = read_jsonl(source_manifest_path)
    prompt_rows: dict[str, dict[str, Any]] = {}
    for condition, spec in CONDITIONS.items():
        matches = [
            row
            for row in source_rows
            if row.get("task") == spec["task"]
            and row.get("source_motion_id") == spec["source_motion_id"]
        ]
        if len(matches) == 1:
            prompt_rows[condition] = matches[0]

    final_checkpoint = completion.get("final_checkpoint_sha256")
    model_commit = completion.get("model_commit")
    identity_checks = {
        "evidence_protocol_exact": evidence.get("protocol") == PROTOCOL,
        "completion_protocol_exact": completion.get("protocol") == COMPLETION_PROTOCOL,
        "formal_training_completion_passed": (
            completion.get("passed") is True
            and completion.get("ready_for_frozen_evaluation") is True
        ),
        "bound_to_exact_completion_file": evidence.get("training_completion_sha256") == completion_hash,
        "exact_immutable_source_manifest": (
            source_hash == EXPECTED_SOURCE_MANIFEST_SHA256
            and evidence.get("source_manifest_sha256") == source_hash
        ),
        "same_official_model_commit": (
            valid_commit(model_commit) and evidence.get("model_commit") == model_commit
        ),
        "same_adapted_checkpoint": (
            valid_sha256(final_checkpoint)
            and evidence.get("adapted_checkpoint_sha256") == final_checkpoint
        ),
    }

    scene = evidence.get("scene", {})
    run_state = evidence.get("evaluation_run", {})
    execution_checks = {
        "fixed_smallbox_scene": (
            scene.get("asset") == "SMALLBOX"
            and scene.get("profile_count") == PROFILE_COUNT
            and scene.get("rollout_steps") == ROLLOUT_STEPS
            and valid_sha256(scene.get("physics_config_sha256"))
        ),
        "fixed_evaluation_seed": evidence.get("evaluation_seed") == EVALUATION_SEED,
        "zero_optimizer_steps": run_state.get("optimizer_step_count") == 0,
        "checkpoint_parameters_unchanged": (
            run_state.get("parameters_before_sha256") == final_checkpoint
            and run_state.get("parameters_after_sha256") == final_checkpoint
        ),
        "official_model_without_local_modules": (
            run_state.get("official_zero_wam_model") is True
            and run_state.get("local_learned_modules") == []
            and run_state.get("public_wan_only_substitute") is False
        ),
    }

    prompt_caches = evidence.get("prompt_caches", {})
    endpoint_identities = evidence.get("released_endpoints", {})
    prompt_checks: dict[str, bool] = {}
    endpoint_checks: dict[str, bool] = {}
    for condition, spec in CONDITIONS.items():
        source_row = prompt_rows.get(condition, {})
        cache = prompt_caches.get(condition, {})
        expected_prompt_sha = source_row.get("prompt", {}).get("sequence_sha256")
        prompt_checks[f"{condition}_exact_source_prompt"] = (
            source_row.get("task") == spec["task"]
            and source_row.get("source_motion_id") == spec["source_motion_id"]
            and cache.get("prompt_sequence_sha256") == expected_prompt_sha
            and valid_sha256(expected_prompt_sha)
        )
        prompt_checks[f"{condition}_encoded_once_and_cached"] = (
            cache.get("official_prompt_encoder") is True
            and cache.get("encode_call_count") == 1
            and cache.get("reuse_step_count") == PROFILE_COUNT * ROLLOUT_STEPS
            and valid_sha256(cache.get("cache_sha256"))
            and cache.get("language_enabled") is False
        )
        expert = source_row.get("expert", {})
        endpoint = endpoint_identities.get(condition, {})
        generator_path = project_root / str(expert.get("generator_checkpoint", ""))
        tracker_path = project_root / str(expert.get("tracker_checkpoint", ""))
        endpoint_checks[f"{condition}_released_endpoint_files_hash_exact"] = (
            generator_path.is_file()
            and tracker_path.is_file()
            and endpoint.get("generator_checkpoint_sha256") == file_sha256(generator_path)
            and endpoint.get("tracker_checkpoint_sha256") == file_sha256(tracker_path)
            and endpoint.get("parameter_exact") is True
        )
    prompt_checks["carry_and_kick_cache_differ"] = (
        prompt_caches.get("carry45", {}).get("cache_sha256")
        != prompt_caches.get("kick21", {}).get("cache_sha256")
    )

    trace_entries = evidence.get("traces", [])
    expected_keys = {
        (run_kind, condition, profile)
        for run_kind in RUN_KINDS
        for condition in CONDITIONS
        for profile in range(PROFILE_COUNT)
    }
    entries_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    entries_valid = isinstance(trace_entries, list)
    for entry in trace_entries if isinstance(trace_entries, list) else []:
        key = (entry.get("run_kind"), entry.get("condition"), entry.get("profile_index"))
        if key in entries_by_key:
            entries_valid = False
        if (
            isinstance(key[0], str)
            and isinstance(key[1], str)
            and isinstance(key[2], int)
        ):
            entries_by_key[key] = entry
        else:
            entries_valid = False
    entries_valid = entries_valid and set(entries_by_key) == expected_keys

    trace_audits: dict[tuple[str, str, int], dict[str, Any]] = {}
    trace_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    trace_hashes: set[str] = set()
    all_trace_hashes_unique = True
    all_traces_pass = entries_valid
    if entries_valid:
        for key in sorted(expected_keys):
            entry = entries_by_key[key]
            path, expected_hash = resolve(evidence_path, entry.get("trace"))
            audit, rows = audit_trace(
                path,
                expected_hash,
                key[0],
                prompt_caches.get(key[1], {}).get("cache_sha256"),
                str(final_checkpoint),
            )
            trace_audits[key] = audit
            trace_rows[key] = rows
            all_traces_pass = all_traces_pass and audit.get("passed") is True
            all_trace_hashes_unique = all_trace_hashes_unique and expected_hash not in trace_hashes
            trace_hashes.add(expected_hash)

    matched_initial_states = entries_valid
    adapted_prompt_only_swap = entries_valid
    future_precedes_action_profiles = 0
    if entries_valid and all_traces_pass:
        for profile in range(PROFILE_COUNT):
            profile_entries = [
                entries_by_key[(run_kind, condition, profile)]
                for run_kind in RUN_KINDS
                for condition in CONDITIONS
            ]
            matched_initial_states = matched_initial_states and (
                len({entry.get("initial_physics_sha256") for entry in profile_entries}) == 1
                and len({entry.get("initial_causal_history_sha256") for entry in profile_entries}) == 1
                and all(valid_sha256(entry.get("initial_physics_sha256")) for entry in profile_entries)
                and all(valid_sha256(entry.get("initial_causal_history_sha256")) for entry in profile_entries)
            )
            observed_initial_physics = []
            for run_kind in RUN_KINDS:
                for condition in CONDITIONS:
                    first = trace_rows[(run_kind, condition, profile)][0]
                    observed_initial_physics.append(
                        canonical_sha256(
                            {
                                "robot_root_height_m": first["robot_root_height_m"],
                                "robot_root_quaternion_wxyz": first[
                                    "robot_root_quaternion_wxyz"
                                ],
                                "object_position_xyz": first["object_position_xyz"],
                                "hand_contacts": [
                                    first["left_hand_box_contact"],
                                    first["right_hand_box_contact"],
                                ],
                                "foot_contacts": [
                                    first["left_foot_box_contact"],
                                    first["right_foot_box_contact"],
                                ],
                            }
                        )
                    )
            matched_initial_states = matched_initial_states and len(set(observed_initial_physics)) == 1
            adapted_entries = [
                entries_by_key[("adapted_zero_wam", condition, profile)]
                for condition in CONDITIONS
            ]
            adapted_prompt_only_swap = adapted_prompt_only_swap and (
                len({entry.get("non_prompt_conditioning_sha256") for entry in adapted_entries}) == 1
                and all(valid_sha256(entry.get("non_prompt_conditioning_sha256")) for entry in adapted_entries)
            )
            carry_rows = trace_rows[("adapted_zero_wam", "carry45", profile)]
            kick_rows = trace_rows[("adapted_zero_wam", "kick21", profile)]
            future_difference_steps = [
                step
                for step in range(ROLLOUT_STEPS)
                if carry_rows[step]["predicted_robot_future_sha256"]
                != kick_rows[step]["predicted_robot_future_sha256"]
            ]
            action_difference_steps = [
                step
                for step in range(ROLLOUT_STEPS)
                if vector_max_abs_difference(
                    carry_rows[step]["executed_action_29d"],
                    kick_rows[step]["executed_action_29d"],
                )
                > ACTION_EQUALITY_ATOL
            ]
            if (
                future_difference_steps
                and action_difference_steps
                and future_difference_steps[0] <= action_difference_steps[0]
            ):
                future_precedes_action_profiles += 1

    trace_checks = {
        "exact_80_unique_trace_entries": (
            entries_valid and len(trace_entries) == 80 and all_trace_hashes_unique
        ),
        "all_trace_content_checks_pass": all_traces_pass,
        "elementwise_matched_initial_physics_and_history": matched_initial_states,
        "adapted_conditions_change_prompt_only": adapted_prompt_only_swap,
        "predicted_future_changes_before_or_with_action_all_profiles": (
            future_precedes_action_profiles == PROFILE_COUNT
        ),
    }

    outcome_counts: dict[str, dict[str, int]] = {}
    if entries_valid and all_traces_pass:
        for run_kind in RUN_KINDS:
            outcome_counts[run_kind] = {}
            for condition in CONDITIONS:
                physical = [
                    trace_audits[(run_kind, condition, profile)]["physical"]
                    for profile in range(PROFILE_COUNT)
                ]
                outcome_counts[run_kind][f"{condition}_safe_carry"] = sum(
                    bool(row["safe_carry_success"]) for row in physical
                )
                outcome_counts[run_kind][f"{condition}_safe_kick"] = sum(
                    bool(row["safe_kick_success"]) for row in physical
                )
                outcome_counts[run_kind][f"{condition}_falls"] = sum(
                    bool(row["physical_fall"]) for row in physical
                )
    adapted = outcome_counts.get("adapted_zero_wam", {})
    baseline = outcome_counts.get("released_endpoint", {})
    physical_checks = {
        "released_carry_endpoint_capable": baseline.get("carry45_safe_carry", 0)
        >= MINIMUM_MATCHED_SUCCESSES,
        "released_kick_endpoint_capable": baseline.get("kick21_safe_kick", 0)
        >= MINIMUM_MATCHED_SUCCESSES,
        "adapted_carry45_at_least_16_of_20": adapted.get("carry45_safe_carry", 0)
        >= MINIMUM_MATCHED_SUCCESSES,
        "adapted_kick21_at_least_16_of_20": adapted.get("kick21_safe_kick", 0)
        >= MINIMUM_MATCHED_SUCCESSES,
        "carry_falls_do_not_regress": adapted.get("carry45_falls", PROFILE_COUNT + 1)
        <= baseline.get("carry45_falls", -1),
        "kick_falls_do_not_regress": adapted.get("kick21_falls", PROFILE_COUNT + 1)
        <= baseline.get("kick21_falls", -1),
        "carry_condition_has_carry_specific_advantage": (
            adapted.get("carry45_safe_carry", 0) > adapted.get("kick21_safe_carry", 0)
        ),
        "kick_condition_has_kick_specific_advantage": (
            adapted.get("kick21_safe_kick", 0) > adapted.get("carry45_safe_kick", 0)
        ),
    }

    checks = {
        **identity_checks,
        **execution_checks,
        **prompt_checks,
        **endpoint_checks,
        **trace_checks,
        **physical_checks,
    }
    passed = all(checks.values())
    return {
        "protocol": "official_zero_wam_sugar_smallbox_closed_loop_audit_v1",
        "passed": passed,
        "smallbox_same_checkpoint_demo_following_passed": passed,
        "automatic_next_branch": (
            "run_motion_disjoint_test_prompt_evaluation"
            if passed
            else "close_checkpoint_without_threshold_or_reward_sweep"
        ),
        "outcome_counts": outcome_counts,
        "future_precedes_action_profile_count": future_precedes_action_profiles,
        "checks": checks,
        "claim_boundary": (
            "Passing proves one official same-checkpoint Carry45/Kick21 SMALLBOX prompt switch. "
            "It does not prove arbitrary-demo, motion-identity or cross-asset following."
        ),
    }


def make_trace(condition: str, run_kind: str, profile: int, checkpoint: str, cache: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(ROLLOUT_STEPS):
        carry = condition == "carry45"
        object_x = 0.0 if carry or step < 100 else 0.12 * (step - 99) / (ROLLOUT_STEPS - 100)
        object_z = 0.50 + (0.06 if carry and step >= 100 else 0.0)
        action_value = 0.01 if carry else 0.02
        executed = [action_value] * ACTION_DIM
        row: dict[str, Any] = {
            "step": step,
            "profile_index": profile,
            "robot_root_height_m": 0.80,
            "robot_root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "object_position_xyz": [object_x, 0.0, object_z],
            "left_hand_box_contact": carry and 80 <= step <= 220,
            "right_hand_box_contact": carry and 80 <= step <= 220,
            "left_foot_box_contact": (not carry) and 100 <= step <= 150,
            "right_foot_box_contact": False,
            "executed_action_29d": executed,
            "reset_or_done": False,
        }
        if run_kind == "adapted_zero_wam":
            future_hash = hashlib.sha256(f"future/{condition}/{profile}/{step}".encode()).hexdigest()
            row.update(
                {
                    "model_checkpoint_sha256": checkpoint,
                    "prompt_cache_sha256": cache,
                    "predicted_robot_future_sha256": future_hash,
                    "action_decoder_input_future_sha256": future_hash,
                    "decoded_action_29d": executed,
                    "teacher_forced_future_used": False,
                    "future_target_used": False,
                    "outcome_label_used": False,
                    "endpoint_router_used": False,
                    "demo_reward_used": False,
                    "official_video_prediction_used": True,
                    "official_action_decoder_used": True,
                }
            )
        else:
            row["exact_released_generator_tracker"] = True
        rows.append(row)
    return rows


def run_self_test() -> None:
    global EXPECTED_SOURCE_MANIFEST_SHA256
    with tempfile.TemporaryDirectory(prefix="zero_wam_smallbox_audit_") as temp:
        root = Path(temp)
        expert_root = root / "experts"
        expert_root.mkdir()
        for name in ("carry_generator.ckpt", "carry_tracker.pt", "kick_generator.ckpt", "kick_tracker.pt"):
            (expert_root / name).write_bytes(f"official/{name}".encode())
        source_rows = []
        for condition, spec in CONDITIONS.items():
            prefix = "carry" if condition == "carry45" else "kick"
            source_rows.append(
                {
                    "task": spec["task"],
                    "source_motion_id": spec["source_motion_id"],
                    "prompt": {
                        "sequence_sha256": hashlib.sha256(f"prompt/{condition}".encode()).hexdigest()
                    },
                    "expert": {
                        "generator_checkpoint": f"experts/{prefix}_generator.ckpt",
                        "tracker_checkpoint": f"experts/{prefix}_tracker.pt",
                    },
                }
            )
        source_path = root / "source.jsonl"
        write_jsonl(source_path, source_rows)
        EXPECTED_SOURCE_MANIFEST_SHA256 = file_sha256(source_path)
        commit = "a" * 40
        checkpoint = "b" * 64
        completion = {
            "protocol": COMPLETION_PROTOCOL,
            "passed": True,
            "ready_for_frozen_evaluation": True,
            "model_commit": commit,
            "final_checkpoint_sha256": checkpoint,
        }
        completion_path = root / "completion.json"
        write_json(completion_path, completion)
        cache_hashes = {
            condition: hashlib.sha256(f"cache/{condition}".encode()).hexdigest()
            for condition in CONDITIONS
        }
        traces = []
        for run_kind in RUN_KINDS:
            for condition in CONDITIONS:
                for profile in range(PROFILE_COUNT):
                    trace_path = root / f"{run_kind}_{condition}_{profile:02d}.jsonl"
                    write_jsonl(
                        trace_path,
                        make_trace(condition, run_kind, profile, checkpoint, cache_hashes[condition]),
                    )
                    traces.append(
                        {
                            "run_kind": run_kind,
                            "condition": condition,
                            "profile_index": profile,
                            "initial_physics_sha256": hashlib.sha256(f"physics/{profile}".encode()).hexdigest(),
                            "initial_causal_history_sha256": hashlib.sha256(f"history/{profile}".encode()).hexdigest(),
                            "non_prompt_conditioning_sha256": hashlib.sha256(
                                f"nonprompt/{profile}".encode()
                            ).hexdigest(),
                            "trace": {"path": trace_path.name, "sha256": file_sha256(trace_path)},
                        }
                    )
        evidence = {
            "protocol": PROTOCOL,
            "training_completion_sha256": file_sha256(completion_path),
            "source_manifest_sha256": file_sha256(source_path),
            "model_commit": commit,
            "adapted_checkpoint_sha256": checkpoint,
            "evaluation_seed": EVALUATION_SEED,
            "scene": {
                "asset": "SMALLBOX",
                "profile_count": PROFILE_COUNT,
                "rollout_steps": ROLLOUT_STEPS,
                "physics_config_sha256": "c" * 64,
            },
            "evaluation_run": {
                "optimizer_step_count": 0,
                "parameters_before_sha256": checkpoint,
                "parameters_after_sha256": checkpoint,
                "official_zero_wam_model": True,
                "local_learned_modules": [],
                "public_wan_only_substitute": False,
            },
            "prompt_caches": {
                condition: {
                    "prompt_sequence_sha256": source_rows[index]["prompt"]["sequence_sha256"],
                    "official_prompt_encoder": True,
                    "encode_call_count": 1,
                    "reuse_step_count": PROFILE_COUNT * ROLLOUT_STEPS,
                    "cache_sha256": cache_hashes[condition],
                    "language_enabled": False,
                }
                for index, condition in enumerate(CONDITIONS)
            },
            "released_endpoints": {
                condition: {
                    "generator_checkpoint_sha256": file_sha256(
                        expert_root / ("carry_generator.ckpt" if condition == "carry45" else "kick_generator.ckpt")
                    ),
                    "tracker_checkpoint_sha256": file_sha256(
                        expert_root / ("carry_tracker.pt" if condition == "carry45" else "kick_tracker.pt")
                    ),
                    "parameter_exact": True,
                }
                for condition in CONDITIONS
            },
            "traces": traces,
        }
        evidence_path = root / "evidence.json"
        write_json(evidence_path, evidence)
        positive = evaluate(completion_path, source_path, root, evidence_path)
        assert positive["passed"] is True, positive

        evidence["adapted_checkpoint_sha256"] = "d" * 64
        write_json(evidence_path, evidence)
        assert evaluate(completion_path, source_path, root, evidence_path)["passed"] is False
        evidence["adapted_checkpoint_sha256"] = checkpoint

        evidence["traces"][41]["initial_physics_sha256"] = "e" * 64
        write_json(evidence_path, evidence)
        initial_failure = evaluate(completion_path, source_path, root, evidence_path)
        assert initial_failure["checks"]["elementwise_matched_initial_physics_and_history"] is False
        evidence["traces"][41]["initial_physics_sha256"] = hashlib.sha256(b"physics/1").hexdigest()

        evidence["prompt_caches"]["carry45"]["encode_call_count"] = 2
        write_json(evidence_path, evidence)
        cache_failure = evaluate(completion_path, source_path, root, evidence_path)
        assert cache_failure["checks"]["carry45_encoded_once_and_cached"] is False
        evidence["prompt_caches"]["carry45"]["encode_call_count"] = 1

        adapted_carry = next(
            entry
            for entry in evidence["traces"]
            if entry["run_kind"] == "adapted_zero_wam"
            and entry["condition"] == "carry45"
            and entry["profile_index"] == 0
        )
        trace_path, _ = resolve(evidence_path, adapted_carry["trace"])
        trace_rows = read_jsonl(trace_path)
        trace_rows[0]["teacher_forced_future_used"] = True
        write_jsonl(trace_path, trace_rows)
        adapted_carry["trace"]["sha256"] = file_sha256(trace_path)
        write_json(evidence_path, evidence)
        teacher_failure = evaluate(completion_path, source_path, root, evidence_path)
        assert teacher_failure["checks"]["all_trace_content_checks_pass"] is False
        trace_rows[0]["teacher_forced_future_used"] = False
        write_jsonl(trace_path, trace_rows)
        adapted_carry["trace"]["sha256"] = file_sha256(trace_path)

        trace_rows[0]["decoded_action_29d"][0] += 0.1
        write_jsonl(trace_path, trace_rows)
        adapted_carry["trace"]["sha256"] = file_sha256(trace_path)
        write_json(evidence_path, evidence)
        action_failure = evaluate(completion_path, source_path, root, evidence_path)
        assert action_failure["checks"]["all_trace_content_checks_pass"] is False
        trace_rows[0]["decoded_action_29d"][0] -= 0.1
        write_jsonl(trace_path, trace_rows)
        adapted_carry["trace"]["sha256"] = file_sha256(trace_path)

        adapted_kick = next(
            entry
            for entry in evidence["traces"]
            if entry["run_kind"] == "adapted_zero_wam"
            and entry["condition"] == "kick21"
            and entry["profile_index"] == 0
        )
        kick_trace_path, _ = resolve(evidence_path, adapted_kick["trace"])
        kick_rows = read_jsonl(kick_trace_path)
        carry_rows = read_jsonl(trace_path)
        for step in range(ROLLOUT_STEPS):
            same_future = carry_rows[step]["predicted_robot_future_sha256"]
            kick_rows[step]["predicted_robot_future_sha256"] = same_future
            kick_rows[step]["action_decoder_input_future_sha256"] = same_future
        write_jsonl(kick_trace_path, kick_rows)
        adapted_kick["trace"]["sha256"] = file_sha256(kick_trace_path)
        write_json(evidence_path, evidence)
        future_failure = evaluate(completion_path, source_path, root, evidence_path)
        assert (
            future_failure["checks"]["predicted_future_changes_before_or_with_action_all_profiles"]
            is False
        )
        kick_rows = make_trace("kick21", "adapted_zero_wam", 0, checkpoint, cache_hashes["kick21"])
        write_jsonl(kick_trace_path, kick_rows)
        adapted_kick["trace"]["sha256"] = file_sha256(kick_trace_path)

        kick_rows[200]["robot_root_height_m"] = 0.30
        write_jsonl(kick_trace_path, kick_rows)
        adapted_kick["trace"]["sha256"] = file_sha256(kick_trace_path)
        write_json(evidence_path, evidence)
        fall_failure = evaluate(completion_path, source_path, root, evidence_path)
        assert fall_failure["checks"]["kick_falls_do_not_regress"] is False
        kick_rows[200]["robot_root_height_m"] = 0.80
        write_jsonl(kick_trace_path, kick_rows)
        adapted_kick["trace"]["sha256"] = file_sha256(kick_trace_path)

        original_endpoint_hash = evidence["released_endpoints"]["carry45"][
            "generator_checkpoint_sha256"
        ]
        evidence["released_endpoints"]["carry45"]["generator_checkpoint_sha256"] = "f" * 64
        write_json(evidence_path, evidence)
        endpoint_failure = evaluate(completion_path, source_path, root, evidence_path)
        assert endpoint_failure["checks"]["carry45_released_endpoint_files_hash_exact"] is False
        evidence["released_endpoints"]["carry45"][
            "generator_checkpoint_sha256"
        ] = original_endpoint_hash

        for profile in range(5):
            entry = next(
                item
                for item in evidence["traces"]
                if item["run_kind"] == "adapted_zero_wam"
                and item["condition"] == "carry45"
                and item["profile_index"] == profile
            )
            path, _ = resolve(evidence_path, entry["trace"])
            rows = read_jsonl(path)
            for row in rows:
                row["left_hand_box_contact"] = False
            write_jsonl(path, rows)
            entry["trace"]["sha256"] = file_sha256(path)
        write_json(evidence_path, evidence)
        carry_failure = evaluate(completion_path, source_path, root, evidence_path)
        assert carry_failure["checks"]["adapted_carry45_at_least_16_of_20"] is False

        print(
            json.dumps(
                {
                    "self_test_passed": True,
                    "positive_fixture": "20 profiles x 2 conditions x adapted/baseline x 650 steps",
                    "rejected": [
                        "wrong_checkpoint",
                        "initial_state_mismatch",
                        "prompt_reencoded",
                        "teacher_forced_future",
                        "decoded_action_mismatch",
                        "action_changes_without_future_change",
                        "fall_regression_against_endpoint",
                        "endpoint_checkpoint_identity_tamper",
                        "carry_only_15_of_20",
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
    required = (
        args.training_completion,
        args.source_manifest,
        args.project_root,
        args.evidence_json,
        args.output_dir,
    )
    if any(value is None for value in required):
        raise SystemExit(
            "--training-completion, --source-manifest, --project-root, --evidence-json and --output-dir are required"
        )
    assert args.training_completion is not None
    assert args.source_manifest is not None
    assert args.project_root is not None
    assert args.evidence_json is not None
    assert args.output_dir is not None
    result = evaluate(
        args.training_completion,
        args.source_manifest,
        args.project_root.resolve(),
        args.evidence_json,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "SMALLBOX_CLOSED_LOOP_AUDIT.json"
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
