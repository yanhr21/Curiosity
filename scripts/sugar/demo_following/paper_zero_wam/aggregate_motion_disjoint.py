#!/usr/bin/env python3
"""Audit all 19 held-out four-condition paper Zero-WAM physical grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .materialize_motion_cases import prompt_record
from .physical_metrics import RESTORE_PAYLOAD_KEYS, physical_summary
from .results import refresh_results_document_best_effort


CONDITIONS = ("matched", "reversed", "same_task_alternate", "wrong_task")


TRACE_KEYS = (
    "robot_root_state_w",
    "robot_joint_pos",
    "robot_joint_vel",
    "object_root_state_w",
    "contact",
    "requested_action",
    "executed_action",
    "done",
)
INITIAL_KEYS = (
    "initial_robot_root_state_w",
    "initial_robot_joint_pos",
    "initial_robot_joint_vel",
    "initial_object_root_state_w",
    "initial_previous_action",
    "initial_action_manager_prev_action",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--heldout-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PaperZeroWAMConfig()
    config.validate()
    expected_inference = {
        "chunk_size": config.inference_chunk_size,
        "video_guidance_scale": config.video_guidance_scale,
        "action_guidance_scale": config.action_guidance_scale,
        "flow_integrator": config.flow_integrator,
        "video_steps": config.video_inference_steps,
        "action_steps": config.action_inference_steps,
        "video_snr_shift": config.video_snr_shift,
        "action_snr_shift": config.action_snr_shift,
    }
    test_rows = sorted(
        (
            row
            for row in read_jsonl(config.resolved(config.manifest))
            if row["split"] == "test"
        ),
        key=lambda row: (row["task"], int(row["source_motion_id"])),
    )
    if len(test_rows) != 19:
        raise ValueError("motion-disjoint aggregate requires all 19 test motions")
    root = args.physical_root.resolve()
    endpoint_files_by_task = {
        task: {
            "motion_folder": str(PROJECT_ROOT / f"SUGAR/data/{task}/data_{motion_id:03d}"),
            "generator": str(PROJECT_ROOT / f"SUGAR/demo_ckpts/{task}/generator.ckpt"),
            "tracker": str(PROJECT_ROOT / f"SUGAR/demo_ckpts/{task}/tracker.pt"),
        }
        for task, motion_id in (("CarryBox", 45), ("KickBox", 21))
    }
    if not all(Path(path).exists() for value in endpoint_files_by_task.values() for path in value.values()):
        raise FileNotFoundError("released motion-disjoint endpoint file is missing")
    source_results: list[dict[str, Any]] = []
    batch_results: list[dict[str, Any]] = []
    trace_checks: list[dict[str, Any]] = []
    profile_vectors: list[np.ndarray] = []
    all_rollouts: list[dict[str, Any]] = []
    recomputed_rollouts: list[dict[str, Any]] = []
    all_videos: list[str] = []
    all_video_frame_counts: list[int] = []
    for source_index, row in enumerate(test_rows):
        expected_prompt_specs = [
            prompt_record(row, condition) for condition in CONDITIONS
        ]
        expected_prompt_by_condition = {
            str(spec["condition"]): spec for spec in expected_prompt_specs
        }
        source_dir = (
            root
            / f"source{source_index:02d}_{row['task']}_{int(row['source_motion_id']):03d}"
        )
        source_result = read_json(source_dir / "SOURCE_RESULT.json")
        source_results.append(source_result)
        for profile_batch in range(5):
            batch_dir = source_dir / f"profile_batch{profile_batch:02d}"
            batch = read_json(batch_dir / "BATCH_RESULT.json")
            batch_results.append(batch)
            prompt_specs_exact = batch.get("prompt_specs") == expected_prompt_specs
            wrong_task = "KickBox" if row["task"] == "CarryBox" else "CarryBox"
            endpoint_files_exact = batch.get("released_endpoint_files") == {
                "matched_endpoint": endpoint_files_by_task[row["task"]],
                "wrong_task_endpoint": endpoint_files_by_task[wrong_task],
            }
            all_rollouts.extend(batch["rollouts"])
            all_videos.extend(batch["videos"])
            all_video_frame_counts.extend(
                int(value) for value in batch.get("video_frame_counts", [])
            )
            profile_vectors.extend(
                np.asarray(vector, dtype=np.float32)
                for vector in batch["profile_readback"]["profile_vectors"]
            )
            with np.load(batch_dir / "TRACE.npz", allow_pickle=False) as trace:
                restore_keys = {f"restore_{name}" for name in RESTORE_PAYLOAD_KEYS}
                missing = sorted(
                    (set(TRACE_KEYS) | set(INITIAL_KEYS) | restore_keys)
                    - set(trace.files)
                )
                geometry = not missing and all(
                    trace[name].shape[0] == 650 and trace[name].shape[1] == 8
                    for name in TRACE_KEYS
                ) and all(trace[name].shape[0] == 8 for name in INITIAL_KEYS) and all(
                    trace[f"restore_{name}"].shape[0] == 8
                    for name in RESTORE_PAYLOAD_KEYS
                )
                finite = not missing and all(
                    np.isfinite(trace[name]).all()
                    for name in (*TRACE_KEYS, *INITIAL_KEYS, *sorted(restore_keys))
                    if name not in ("contact", "done")
                )
                adapted_restore_payload = (
                    {
                        name: np.asarray(trace[f"restore_{name}"]).copy()
                        for name in RESTORE_PAYLOAD_KEYS
                    }
                    if not missing
                    else {}
                )
                zero_resets = not missing and not bool(np.asarray(trace["done"]).any())
                action_execution_exact = bool(
                    geometry
                    and np.array_equal(
                        trace["requested_action"], trace["executed_action"]
                    )
                )
                maximum_action_error = (
                    float(
                        np.max(
                            np.abs(
                                trace["requested_action"].astype(np.float64)
                                - trace["executed_action"].astype(np.float64)
                            )
                        )
                    )
                    if geometry
                    else float("inf")
                )
                summaries_exact = bool(geometry)
                if geometry:
                    for environment_index in range(8):
                        profile_id = profile_batch * 2 + environment_index // 4
                        condition = CONDITIONS[environment_index % 4]
                        candidates = [
                            value
                            for value in batch["rollouts"]
                            if int(value.get("profile_id", -1)) == profile_id
                            and value.get("condition") == condition
                        ]
                        if len(candidates) != 1:
                            summaries_exact = False
                            continue
                        recorded = candidates[0]
                        recomputed = physical_summary(
                            trace["robot_root_state_w"][:, environment_index],
                            trace["object_root_state_w"][:, environment_index],
                            trace["contact"][:, environment_index],
                        )
                        identity_exact = (
                            recorded.get("target_task") == row["task"]
                            and int(recorded.get("source_motion_id", -1))
                            == int(row["source_motion_id"])
                            and recorded.get("prompt_split")
                            == expected_prompt_by_condition[condition]["split"]
                            and recorded.get("prompt_task")
                            == expected_prompt_by_condition[condition]["task"]
                            and int(recorded.get("prompt_source_motion_id", -1))
                            == int(
                                expected_prompt_by_condition[condition][
                                    "source_motion_id"
                                ]
                            )
                            and recorded.get("prompt_latent_key")
                            == expected_prompt_by_condition[condition]["latent_key"]
                            and recorded.get("prompt_reversed")
                            is (condition == "reversed")
                            and int(recorded.get("done_count", -1))
                            == int(trace["done"][:, environment_index].sum())
                        )
                        summaries_exact = bool(
                            summaries_exact
                            and identity_exact
                            and recorded.get("physical") == recomputed
                        )
                        recomputed_rollouts.append(
                            {
                                **recorded,
                                "source_index": source_index,
                                "physical": recomputed,
                            }
                        )
            endpoint_path = batch_dir / "ENDPOINT_TRACES.npz"
            endpoint_missing: list[str] = []
            endpoint_geometry = False
            endpoint_finite = False
            endpoint_zero_resets = False
            endpoint_action_execution_exact = False
            endpoint_summaries_exact = False
            endpoint_maximum_action_error = float("inf")
            endpoint_conditions = ("matched_endpoint", "wrong_task_endpoint")
            with np.load(endpoint_path, allow_pickle=False) as endpoint_trace:
                expected_endpoint_keys = {
                    f"{condition}_{name}"
                    for condition in endpoint_conditions
                    for name in (
                        *TRACE_KEYS,
                        *INITIAL_KEYS,
                        *(f"restore_{key}" for key in RESTORE_PAYLOAD_KEYS),
                    )
                }
                endpoint_missing = sorted(
                    expected_endpoint_keys - set(endpoint_trace.files)
                )
                endpoint_geometry = not endpoint_missing and all(
                    endpoint_trace[f"{condition}_{name}"].shape[0] == 650
                    and endpoint_trace[f"{condition}_{name}"].shape[1] == 2
                    for condition in endpoint_conditions
                    for name in TRACE_KEYS
                ) and all(
                    endpoint_trace[f"{condition}_{name}"].shape[0] == 2
                    for condition in endpoint_conditions
                    for name in INITIAL_KEYS
                ) and all(
                    endpoint_trace[f"{condition}_restore_{name}"].shape[0] == 8
                    for condition in endpoint_conditions
                    for name in RESTORE_PAYLOAD_KEYS
                )
                endpoint_finite = not endpoint_missing and all(
                    np.isfinite(endpoint_trace[f"{condition}_{name}"]).all()
                    for condition in endpoint_conditions
                    for name in (
                        *TRACE_KEYS,
                        *INITIAL_KEYS,
                        *(f"restore_{key}" for key in RESTORE_PAYLOAD_KEYS),
                    )
                    if name not in ("contact", "done")
                )
                endpoint_restore_payload_exact = bool(
                    geometry
                    and endpoint_geometry
                    and all(
                        np.array_equal(
                            endpoint_trace[f"{condition}_restore_{name}"],
                            adapted_restore_payload[name],
                        )
                        for condition in endpoint_conditions
                        for name in RESTORE_PAYLOAD_KEYS
                    )
                )
                endpoint_zero_resets = not endpoint_missing and all(
                    not bool(endpoint_trace[f"{condition}_done"].any())
                    for condition in endpoint_conditions
                )
                endpoint_action_execution_exact = bool(
                    endpoint_geometry
                    and all(
                        np.array_equal(
                            endpoint_trace[f"{condition}_requested_action"],
                            endpoint_trace[f"{condition}_executed_action"],
                        )
                        for condition in endpoint_conditions
                    )
                )
                if endpoint_geometry:
                    endpoint_maximum_action_error = max(
                        float(
                            np.max(
                                np.abs(
                                    endpoint_trace[
                                        f"{condition}_requested_action"
                                    ].astype(np.float64)
                                    - endpoint_trace[
                                        f"{condition}_executed_action"
                                    ].astype(np.float64)
                                )
                            )
                        )
                        for condition in endpoint_conditions
                    )
                endpoint_summaries_exact = bool(endpoint_geometry)
                if endpoint_geometry:
                    for condition in endpoint_conditions:
                        for profile_offset, environment_index in enumerate((0, 4)):
                            profile_id = profile_batch * 2 + profile_offset
                            candidates = [
                                value
                                for value in batch["rollouts"]
                                if value.get("route") == "released_endpoint"
                                and value.get("condition") == condition
                                and int(value.get("profile_id", -1)) == profile_id
                                and int(value.get("environment_index", -1))
                                == environment_index
                            ]
                            if len(candidates) != 1:
                                endpoint_summaries_exact = False
                                continue
                            recorded = candidates[0]
                            recomputed = physical_summary(
                                endpoint_trace[f"{condition}_robot_root_state_w"][
                                    :, profile_offset
                                ],
                                endpoint_trace[f"{condition}_object_root_state_w"][
                                    :, profile_offset
                                ],
                                endpoint_trace[f"{condition}_contact"][:, profile_offset],
                            )
                            endpoint_summaries_exact = bool(
                                endpoint_summaries_exact
                                and recorded.get("target_task") == row["task"]
                                and int(recorded.get("source_motion_id", -1))
                                == int(row["source_motion_id"])
                                and int(recorded.get("done_count", -1))
                                == int(
                                    endpoint_trace[f"{condition}_done"][
                                        :, profile_offset
                                    ].sum()
                                )
                                and recorded.get("physical") == recomputed
                            )
                            recomputed_rollouts.append(
                                {
                                    **recorded,
                                    "source_index": source_index,
                                    "physical": recomputed,
                                }
                            )
            trace_checks.append(
                {
                    "source_index": source_index,
                    "profile_batch": profile_batch,
                    "missing_keys": missing,
                    "endpoint_missing_keys": endpoint_missing,
                    "geometry_passed": geometry,
                    "endpoint_geometry_passed": endpoint_geometry,
                    "finite_passed": finite,
                    "endpoint_finite_passed": endpoint_finite,
                    "zero_resets": zero_resets,
                    "endpoint_zero_resets": endpoint_zero_resets,
                    "requested_action_equals_physx_executed_action": action_execution_exact,
                    "endpoint_requested_action_equals_physx_executed_action": endpoint_action_execution_exact,
                    "maximum_requested_executed_action_error": max(
                        maximum_action_error, endpoint_maximum_action_error
                    ),
                    "all_eight_physical_summaries_recomputed_exact": summaries_exact,
                    "all_four_endpoint_summaries_recomputed_exact": endpoint_summaries_exact,
                    "adapted_endpoint_initial_state_exact": all(
                        (batch.get("adapted_endpoint_initial_state_exact") or {}).values()
                    ),
                    "adapted_endpoint_initial_rgb_exact": all(
                        (batch.get("adapted_endpoint_initial_rgb_exact") or {}).values()
                    ),
                    "adapted_endpoint_profile_readback_exact": all(
                        (batch.get("adapted_endpoint_profile_readback_exact") or {}).values()
                    ),
                    "adapted_endpoint_restore_payloads_reopened_exact": endpoint_restore_payload_exact,
                    "released_endpoint_tracker_history_reinitialized": all(
                        (batch.get("released_endpoint_tracker_history_reinitialized") or {}).values()
                    ),
                    "released_endpoint_files_exact": endpoint_files_exact,
                    "semantic_prompt_specs_exact": prompt_specs_exact,
                }
            )

    expected_adapted_grid = {
        (source_index, profile_id, condition)
        for source_index in range(19)
        for profile_id in range(10)
        for condition in CONDITIONS
    }
    expected_endpoint_grid = {
        (source_index, profile_id, condition)
        for source_index in range(19)
        for profile_id in range(10)
        for condition in ("matched_endpoint", "wrong_task_endpoint")
    }
    actual_adapted_grid = {
        (
            int(rollout["source_index"]),
            int(rollout["profile_id"]),
            str(rollout["condition"]),
        )
        for rollout in recomputed_rollouts
        if rollout["route"] == "adapted"
    }
    actual_endpoint_grid = {
        (
            int(rollout["source_index"]),
            int(rollout["profile_id"]),
            str(rollout["condition"]),
        )
        for rollout in recomputed_rollouts
        if rollout["route"] == "released_endpoint"
    }
    profile_vectors_valid = (
        len(profile_vectors) == 190
        and len({value.shape for value in profile_vectors}) == 1
        and all(np.isfinite(value).all() for value in profile_vectors)
    )
    all_profiles_distinct = profile_vectors_valid and all(
        not np.array_equal(profile_vectors[left], profile_vectors[right])
        for left in range(190)
        for right in range(left + 1, 190)
    )
    heldout = read_json(args.heldout_result.resolve())
    checkpoint_steps = {int(batch["checkpoint_step"]) for batch in batch_results}
    architecture_counts = {
        int(batch["architecture_parameter_count"]) for batch in batch_results
    }
    recomputed_source_outcomes: list[dict[str, Any]] = []
    for source_index, row in enumerate(test_rows):
        selected = [
            value
            for value in recomputed_rollouts
            if int(value["source_index"]) == source_index
        ]
        adapted = [value for value in selected if value["route"] == "adapted"]
        endpoints = [
            value for value in selected if value["route"] == "released_endpoint"
        ]
        target_success_key = (
            "safe_carry_success" if row["task"] == "CarryBox" else "safe_kick_success"
        )
        wrong_success_key = (
            "safe_kick_success" if row["task"] == "CarryBox" else "safe_carry_success"
        )
        matched = sum(
            bool(value["physical"][target_success_key])
            for value in adapted
            if value["condition"] == "matched"
        )
        wrong = sum(
            bool(value["physical"][wrong_success_key])
            for value in adapted
            if value["condition"] == "wrong_task"
        )
        falls_by_condition = {
            condition: sum(
                bool(value["physical"]["physical_fall"])
                for value in selected
                if value["condition"] == condition
            )
            for condition in (
                *CONDITIONS,
                "matched_endpoint",
                "wrong_task_endpoint",
            )
        }
        no_regression = (
            falls_by_condition["matched"] <= falls_by_condition["matched_endpoint"]
            and falls_by_condition["wrong_task"]
            <= falls_by_condition["wrong_task_endpoint"]
            and falls_by_condition["reversed"] == 0
            and falls_by_condition["same_task_alternate"] == 0
        )
        recomputed_source_outcomes.append(
            {
                "source_index": source_index,
                "target_task": row["task"],
                "source_motion_id": int(row["source_motion_id"]),
                "adapted_rollout_count": len(adapted),
                "released_endpoint_rollout_count": len(endpoints),
                "rollout_count": len(selected),
                "matched_prompt_task_successes": matched,
                "wrong_prompt_task_switch_successes": wrong,
                "falls": falls_by_condition,
                "fall_no_regression": no_regression,
                "matches_source_result": (
                    len(adapted) == 40
                    and len(endpoints) == 20
                    and matched
                    == int(source_results[source_index]["matched_prompt_task_successes"])
                    and wrong
                    == int(source_results[source_index]["wrong_prompt_task_switch_successes"])
                    and source_results[source_index].get("falls") == falls_by_condition
                    and bool(
                        source_results[source_index]
                        .get("checks", {})
                        .get("matched_falls_do_not_exceed_released_endpoint", False)
                    )
                    == (
                        falls_by_condition["matched"]
                        <= falls_by_condition["matched_endpoint"]
                    )
                    and bool(
                        source_results[source_index]
                        .get("checks", {})
                        .get("wrong_task_falls_do_not_exceed_released_endpoint", False)
                    )
                    == (
                        falls_by_condition["wrong_task"]
                        <= falls_by_condition["wrong_task_endpoint"]
                    )
                ),
            }
        )

    task_summary: dict[str, dict[str, int]] = {}
    for task in ("CarryBox", "KickBox"):
        selected = [
            row for row in recomputed_source_outcomes if row["target_task"] == task
        ]
        task_summary[task] = {
            "source_count": len(selected),
            "matched_successes": sum(int(row["matched_prompt_task_successes"]) for row in selected),
            "matched_trials": 10 * len(selected),
            "wrong_prompt_switch_successes": sum(
                int(row["wrong_prompt_task_switch_successes"]) for row in selected
            ),
            "wrong_prompt_switch_trials": 10 * len(selected),
            "adapted_falls": sum(
                sum(int(value) for key, value in row["falls"].items() if not key.endswith("endpoint"))
                for row in selected
            ),
            "released_endpoint_falls": sum(
                sum(int(value) for key, value in row["falls"].items() if key.endswith("endpoint"))
                for row in selected
            ),
        }
    checks = {
        "all_19_source_results_passed": len(source_results) == 19
        and all(
            result.get("protocol") == "paper_zero_wam_motion_disjoint_physical_source_v1"
            and result.get("passed") is True
            and int(result.get("source_index", -1)) == index
            and result.get("target_task") == test_rows[index]["task"]
            and int(result.get("source_motion_id", -1))
            == int(test_rows[index]["source_motion_id"])
            and int(result.get("adapted_rollout_count", -1)) == 40
            and int(result.get("released_endpoint_rollout_count", -1)) == 20
            and int(result.get("rollout_count", -1)) == 60
            and result.get("visualization_frame_counts") == [131] * 10
            and result.get("prompt_specs")
            == [prompt_record(test_rows[index], condition) for condition in CONDITIONS]
            for index, result in enumerate(source_results)
        ),
        "all_95_batch_execution_contracts_passed": len(batch_results) == 95
        and all(
            result.get("protocol")
            == "paper_zero_wam_motion_disjoint_physical_batch_v1"
            and result.get("passed_execution_contract") is True
            and int(result.get("source_index", -1)) == index // 5
            and result.get("target_task") == test_rows[index // 5]["task"]
            and int(result.get("source_motion_id", -1))
            == int(test_rows[index // 5]["source_motion_id"])
            and int(result.get("profile_batch", -1)) == index % 5
            and int(result.get("adapted_rollout_count", -1)) == 8
            and int(result.get("released_endpoint_rollout_count", -1)) == 4
            and int(result.get("rollout_count", -1)) == 12
            and result.get("video_frame_counts") == [131] * 2
            and result.get("prompt_specs")
            == [
                prompt_record(test_rows[index // 5], condition)
                for condition in CONDITIONS
            ]
            for index, result in enumerate(batch_results)
        ),
        "complete_760_adapted_rollout_grid": (
            len([row for row in all_rollouts if row["route"] == "adapted"]) == 760
            and actual_adapted_grid == expected_adapted_grid
        ),
        "complete_380_released_endpoint_grid": (
            len(
                [
                    row
                    for row in all_rollouts
                    if row["route"] == "released_endpoint"
                ]
            )
            == 380
            and actual_endpoint_grid == expected_endpoint_grid
        ),
        "all_95_adapted_and_endpoint_trace_archives_reopened": len(trace_checks) == 95
        and all(
            not row["missing_keys"]
            and not row["endpoint_missing_keys"]
            and row["geometry_passed"]
            and row["endpoint_geometry_passed"]
            and row["finite_passed"]
            and row["endpoint_finite_passed"]
            and row["zero_resets"]
            and row["endpoint_zero_resets"]
            and row["requested_action_equals_physx_executed_action"]
            and row["endpoint_requested_action_equals_physx_executed_action"]
            and row["all_eight_physical_summaries_recomputed_exact"]
            and row["all_four_endpoint_summaries_recomputed_exact"]
            and row["adapted_endpoint_initial_state_exact"]
            and row["adapted_endpoint_initial_rgb_exact"]
            and row["adapted_endpoint_profile_readback_exact"]
            and row["adapted_endpoint_restore_payloads_reopened_exact"]
            and row["released_endpoint_tracker_history_reinitialized"]
            and row["released_endpoint_files_exact"]
            and row["semantic_prompt_specs_exact"]
            for row in trace_checks
        ),
        "all_19_source_outcomes_recomputed_exact": (
            len(recomputed_source_outcomes) == 19
            and all(row["matches_source_result"] for row in recomputed_source_outcomes)
        ),
        "all_19_sources_pass_fall_no_regression": all(
            bool(row["fall_no_regression"]) for row in recomputed_source_outcomes
        ),
        "all_190_startup_profiles_exactly_distinct": all_profiles_distinct,
        "all_190_visualizations_nonempty": len(all_videos) == 190
        and all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in all_videos),
        "all_190_visualizations_have_131_frames": (
            all_video_frame_counts == [131] * 190
        ),
        "same_formal_checkpoint_step_4200": checkpoint_steps == {4200},
        "same_full_architecture": architecture_counts == {10_658_724_829},
        "all_95_batches_use_frozen_inference_protocol": all(
            batch.get("inference") == expected_inference for batch in batch_results
        ),
        "frozen_heldout_order_identity_gate_passed": (
            heldout.get("passed") is True
            and int(heldout.get("score_instance_count", -1)) == 1950
            and int(heldout.get("checkpoint_step", -1)) == 4200
        ),
    }
    result = {
        "protocol": "paper_zero_wam_motion_disjoint_physical_result_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "source_count": 19,
        "profile_count_per_source": 10,
        "condition_count": 4,
        "endpoint_route_count": 2,
        "adapted_rollout_count": 760,
        "released_endpoint_rollout_count": 380,
        "rollout_count": len(recomputed_rollouts),
        "executed_action_count": len(recomputed_rollouts) * 650,
        "requested_executed_action_rows": len(recomputed_rollouts) * 650,
        "maximum_requested_executed_action_error": max(
            row["maximum_requested_executed_action_error"]
            for row in trace_checks
        ),
        "visualization_count": len(all_videos),
        "visualization_frame_counts": all_video_frame_counts,
        "semantic_prompt_specs": [
            {
                "source_index": source_index,
                "target_task": row["task"],
                "source_motion_id": int(row["source_motion_id"]),
                "prompts": [
                    prompt_record(row, condition) for condition in CONDITIONS
                ],
            }
            for source_index, row in enumerate(test_rows)
        ],
        "task_summary": task_summary,
        "source_results": source_results,
        "recomputed_source_outcomes": recomputed_source_outcomes,
        "trace_checks": trace_checks,
        "inference": expected_inference,
        "videos": all_videos,
        "hash_checks": False,
        "claim_boundary": (
            "Combined motion-disjoint flow order/identity dependence and live same-embodiment "
            "task-switch execution over all 19 test sources; not cross-embodiment or open-ended proof."
        ),
    }
    write_json_atomic(args.output.resolve(), result)
    refresh_results_document_best_effort()
    emit_json_best_effort(
        {"passed": result["passed"], "checks": checks, "task_summary": task_summary}
    )
    if not result["passed"]:
        raise SystemExit("motion-disjoint physical aggregate failed")


if __name__ == "__main__":
    main()
