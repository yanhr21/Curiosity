#!/usr/bin/env python3
"""Aggregate all twenty paired SMALLBOX paper Zero-WAM profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .physical_metrics import RESTORE_PAYLOAD_KEYS, physical_summary
from .results import refresh_results_document_best_effort


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-root", type=Path, required=True)
    args = parser.parse_args()
    config = PaperZeroWAMConfig()
    config.validate()
    root = args.physical_root.resolve()
    expected_prompt_sources = {"CarryBox": 45, "KickBox": 21}
    expected_prompt_specs = [
        {
            "split": "train",
            "task": task,
            "source_motion_id": source_motion_id,
            "latent_key": "prompt_latents",
        }
        for task, source_motion_id in expected_prompt_sources.items()
    ]
    expected_endpoint_files = {
        task: {
            "motion_folder": str(PROJECT_ROOT / f"SUGAR/data/{task}/data_{motion_id:03d}"),
            "generator": str(PROJECT_ROOT / f"SUGAR/demo_ckpts/{task}/generator.ckpt"),
            "tracker": str(PROJECT_ROOT / f"SUGAR/demo_ckpts/{task}/tracker.pt"),
        }
        for task, motion_id in (("CarryBox", 45), ("KickBox", 21))
    }
    if not all(Path(path).exists() for value in expected_endpoint_files.values() for path in value.values()):
        raise FileNotFoundError("released SMALLBOX endpoint file is missing")
    batches = [
        read_json(root / f"profile_batch{batch:02d}" / "BATCH_RESULT.json")
        for batch in range(5)
    ]
    batch_contract = all(
        result.get("protocol") == "paper_zero_wam_smallbox_physical_batch_v1"
        and result.get("passed_execution_contract") is True
        and result.get("profile_batch") == batch
        and int(result.get("adapted_rollout_count", -1)) == 8
        and int(result.get("released_endpoint_rollout_count", -1)) == 8
        and int(result.get("rollout_count", -1)) == 16
        and result.get("prompt_specs") == expected_prompt_specs
        and result.get("video_frame_counts") == [131] * 4
        and result.get("released_endpoint_files") == expected_endpoint_files
        for batch, result in enumerate(batches)
    )
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
    trace_audits: list[dict[str, Any]] = []
    recomputed_rollouts: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        trace_path = root / f"profile_batch{batch_index:02d}" / "TRACE.npz"
        endpoint_path = (
            root / f"profile_batch{batch_index:02d}" / "ENDPOINT_TRACES.npz"
        )
        with np.load(trace_path, allow_pickle=False) as trace:
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
                    candidates = [
                        row
                        for row in batch["rollouts"]
                        if int(row.get("environment_index", -1)) == environment_index
                        and row.get("route") == "adapted"
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
                        int(recorded.get("profile_id", -1))
                        == batch_index * 4 + environment_index // 2
                        and recorded.get("prompt_split") == "train"
                        and recorded.get("prompt_task")
                        == ("CarryBox" if environment_index % 2 == 0 else "KickBox")
                        and int(recorded.get("prompt_source_motion_id", -1))
                        == expected_prompt_sources[
                            "CarryBox" if environment_index % 2 == 0 else "KickBox"
                        ]
                        and int(recorded.get("source_motion_id", -1))
                        == expected_prompt_sources[
                            "CarryBox" if environment_index % 2 == 0 else "KickBox"
                        ]
                        and recorded.get("prompt_latent_key") == "prompt_latents"
                        and recorded.get("prompt_reversed") is False
                        and int(recorded.get("done_count", -1))
                        == int(trace["done"][:, environment_index].sum())
                    )
                    summaries_exact = bool(
                        summaries_exact
                        and identity_exact
                        and recorded.get("physical") == recomputed
                    )
                    recomputed_rollouts.append({**recorded, "physical": recomputed})
        endpoint_missing: list[str] = []
        endpoint_geometry = False
        endpoint_finite = False
        endpoint_zero_resets = False
        endpoint_action_execution_exact = False
        endpoint_summaries_exact = False
        endpoint_maximum_action_error = float("inf")
        with np.load(endpoint_path, allow_pickle=False) as endpoint_trace:
            expected_endpoint_keys = {
                f"{task.lower()}_{name}"
                for task in ("CarryBox", "KickBox")
                for name in (
                    *TRACE_KEYS,
                    *INITIAL_KEYS,
                    *(f"restore_{key}" for key in RESTORE_PAYLOAD_KEYS),
                )
            }
            endpoint_missing = sorted(expected_endpoint_keys - set(endpoint_trace.files))
            endpoint_geometry = not endpoint_missing and all(
                endpoint_trace[f"{task.lower()}_{name}"].shape[0] == 650
                and endpoint_trace[f"{task.lower()}_{name}"].shape[1] == 4
                for task in ("CarryBox", "KickBox")
                for name in TRACE_KEYS
            ) and all(
                endpoint_trace[f"{task.lower()}_{name}"].shape[0] == 4
                for task in ("CarryBox", "KickBox")
                for name in INITIAL_KEYS
            ) and all(
                endpoint_trace[f"{task.lower()}_restore_{name}"].shape[0] == 8
                for task in ("CarryBox", "KickBox")
                for name in RESTORE_PAYLOAD_KEYS
            )
            endpoint_finite = not endpoint_missing and all(
                np.isfinite(endpoint_trace[f"{task.lower()}_{name}"]).all()
                for task in ("CarryBox", "KickBox")
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
                        endpoint_trace[f"{task.lower()}_restore_{name}"],
                        adapted_restore_payload[name],
                    )
                    for task in ("CarryBox", "KickBox")
                    for name in RESTORE_PAYLOAD_KEYS
                )
            )
            endpoint_zero_resets = not endpoint_missing and all(
                not bool(endpoint_trace[f"{task.lower()}_done"].any())
                for task in ("CarryBox", "KickBox")
            )
            endpoint_action_execution_exact = bool(
                endpoint_geometry
                and all(
                    np.array_equal(
                        endpoint_trace[f"{task.lower()}_requested_action"],
                        endpoint_trace[f"{task.lower()}_executed_action"],
                    )
                    for task in ("CarryBox", "KickBox")
                )
            )
            if endpoint_geometry:
                endpoint_maximum_action_error = max(
                    float(
                        np.max(
                            np.abs(
                                endpoint_trace[
                                    f"{task.lower()}_requested_action"
                                ].astype(np.float64)
                                - endpoint_trace[
                                    f"{task.lower()}_executed_action"
                                ].astype(np.float64)
                            )
                        )
                    )
                    for task in ("CarryBox", "KickBox")
                )
            endpoint_summaries_exact = bool(endpoint_geometry)
            if endpoint_geometry:
                for task in ("CarryBox", "KickBox"):
                    prefix = task.lower()
                    for profile_offset, environment_index in enumerate((0, 2, 4, 6)):
                        profile_id = batch_index * 4 + profile_offset
                        candidates = [
                            row
                            for row in batch["rollouts"]
                            if row.get("route") == "released_endpoint"
                            and row.get("prompt_task") == task
                            and int(row.get("profile_id", -1)) == profile_id
                            and int(row.get("environment_index", -1))
                            == environment_index
                        ]
                        if len(candidates) != 1:
                            endpoint_summaries_exact = False
                            continue
                        recorded = candidates[0]
                        recomputed = physical_summary(
                            endpoint_trace[f"{prefix}_robot_root_state_w"][
                                :, profile_offset
                            ],
                            endpoint_trace[f"{prefix}_object_root_state_w"][
                                :, profile_offset
                            ],
                            endpoint_trace[f"{prefix}_contact"][:, profile_offset],
                        )
                        endpoint_summaries_exact = bool(
                            endpoint_summaries_exact
                            and recorded.get("prompt_split") == "train"
                            and int(recorded.get("prompt_source_motion_id", -1))
                            == expected_prompt_sources[task]
                            and int(recorded.get("source_motion_id", -1))
                            == expected_prompt_sources[task]
                            and recorded.get("prompt_latent_key") == "prompt_latents"
                            and recorded.get("prompt_reversed") is False
                            and int(recorded.get("done_count", -1))
                            == int(
                                endpoint_trace[f"{prefix}_done"][
                                    :, profile_offset
                                ].sum()
                            )
                            and recorded.get("physical") == recomputed
                        )
                        recomputed_rollouts.append(
                            {**recorded, "physical": recomputed}
                        )
        trace_audits.append(
            {
                "profile_batch": batch_index,
                "adapted_trace": str(trace_path),
                "released_endpoint_trace": str(endpoint_path),
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
                "all_eight_endpoint_summaries_recomputed_exact": endpoint_summaries_exact,
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
                "semantic_prompt_specs_exact": batch.get("prompt_specs")
                == expected_prompt_specs,
            }
        )
    rollouts = recomputed_rollouts
    videos = [path for result in batches for path in result["videos"]]
    video_frame_counts = [
        int(value)
        for result in batches
        for value in result.get("video_frame_counts", [])
    ]
    expected_keys = {
        (profile, task, route)
        for profile in range(20)
        for task in ("CarryBox", "KickBox")
        for route in ("adapted", "released_endpoint")
    }
    actual_keys = {
        (int(row["profile_id"]), str(row["prompt_task"]), str(row["route"]))
        for row in rollouts
    }
    unique_complete_grid = len(rollouts) == 80 and actual_keys == expected_keys
    profile_vectors = [
        np.asarray(vector, dtype=np.float32)
        for result in batches
        for vector in result["profile_readback"]["profile_vectors"]
    ]
    profile_vectors_valid = (
        len(profile_vectors) == 20
        and len({value.shape for value in profile_vectors}) == 1
        and all(np.isfinite(value).all() for value in profile_vectors)
    )
    all_twenty_profiles_distinct = profile_vectors_valid and all(
        not np.array_equal(profile_vectors[left], profile_vectors[right])
        for left in range(20)
        for right in range(left + 1, 20)
    )
    by_task_route = {
        (task, route): [
            row
            for row in rollouts
            if row["prompt_task"] == task and row["route"] == route
        ]
        for task in ("CarryBox", "KickBox")
        for route in ("adapted", "released_endpoint")
    }
    by_profile_task = {
        (int(row["profile_id"]), str(row["prompt_task"])): row
        for row in rollouts
        if row["route"] == "adapted"
    }
    paired_topology: list[dict[str, Any]] = []
    if unique_complete_grid:
        for profile_id in range(20):
            carry = by_profile_task[(profile_id, "CarryBox")]["physical"]
            kick = by_profile_task[(profile_id, "KickBox")]["physical"]
            carry_lift_margin = float(carry["maximum_lift_m"]) - float(
                kick["maximum_lift_m"]
            )
            carry_bilateral_margin = int(carry["bilateral_contact_frames"]) - int(
                kick["bilateral_contact_frames"]
            )
            kick_displacement_margin = float(
                kick["planar_object_net_displacement_m"]
            ) - float(carry["planar_object_net_displacement_m"])
            kick_contact_path_margin = float(
                kick["contact_coupled_planar_path_m"]
            ) - float(carry["contact_coupled_planar_path_m"])
            paired_topology.append(
                {
                    "profile_id": profile_id,
                    "carry_lift_margin_m": carry_lift_margin,
                    "carry_bilateral_contact_margin_frames": carry_bilateral_margin,
                    "kick_net_displacement_margin_m": kick_displacement_margin,
                    "kick_contact_coupled_path_margin_m": kick_contact_path_margin,
                    "carry_topology_advantage": carry_lift_margin > 0.0
                    and carry_bilateral_margin > 0,
                    "kick_topology_advantage": kick_displacement_margin > 0.0
                    and kick_contact_path_margin > 0.0,
                }
            )
    carry_topology_wins = sum(
        bool(row["carry_topology_advantage"]) for row in paired_topology
    )
    kick_topology_wins = sum(
        bool(row["kick_topology_advantage"]) for row in paired_topology
    )
    outcomes = {
        "CarryBox": {
            "safe_successes": sum(
                bool(row["physical"]["safe_carry_success"])
                for row in by_task_route[("CarryBox", "adapted")]
            ),
            "falls": sum(
                bool(row["physical"]["physical_fall"])
                for row in by_task_route[("CarryBox", "adapted")]
            ),
            "released_endpoint_falls": sum(
                bool(row["physical"]["physical_fall"])
                for row in by_task_route[("CarryBox", "released_endpoint")]
            ),
        },
        "KickBox": {
            "safe_successes": sum(
                bool(row["physical"]["safe_kick_success"])
                for row in by_task_route[("KickBox", "adapted")]
            ),
            "falls": sum(
                bool(row["physical"]["physical_fall"])
                for row in by_task_route[("KickBox", "adapted")]
            ),
            "released_endpoint_falls": sum(
                bool(row["physical"]["physical_fall"])
                for row in by_task_route[("KickBox", "released_endpoint")]
            ),
        },
    }
    checks = {
        "five_execution_batches_passed": batch_contract,
        "all_five_full_traces_reopened_and_outcomes_recomputed": (
            len(trace_audits) == 5
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
                and row["all_eight_endpoint_summaries_recomputed_exact"]
                and row["adapted_endpoint_initial_state_exact"]
                and row["adapted_endpoint_initial_rgb_exact"]
                and row["adapted_endpoint_profile_readback_exact"]
                and row["adapted_endpoint_restore_payloads_reopened_exact"]
                and row["released_endpoint_tracker_history_reinitialized"]
                and row["semantic_prompt_specs_exact"]
                for row in trace_audits
            )
        ),
        "complete_20_profile_adapted_endpoint_grid": unique_complete_grid,
        "all_twenty_physx_profile_vectors_finite_and_same_geometry": profile_vectors_valid,
        "all_twenty_physx_profiles_exactly_distinct": all_twenty_profiles_distinct,
        "carry_safe_success_at_least_16_of_20": outcomes["CarryBox"]["safe_successes"] >= 16,
        "kick_safe_success_at_least_16_of_20": outcomes["KickBox"]["safe_successes"] >= 16,
        "carry_topology_advantage_at_least_16_of_20": carry_topology_wins >= 16,
        "kick_topology_advantage_at_least_16_of_20": kick_topology_wins >= 16,
        "carry_falls_do_not_exceed_released_endpoint": (
            outcomes["CarryBox"]["falls"]
            <= outcomes["CarryBox"]["released_endpoint_falls"]
        ),
        "kick_falls_do_not_exceed_released_endpoint": (
            outcomes["KickBox"]["falls"]
            <= outcomes["KickBox"]["released_endpoint_falls"]
        ),
        "twenty_nonempty_pair_videos": len(videos) == 20
        and all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in videos),
        "twenty_pair_videos_have_131_frames": video_frame_counts == [131] * 20,
        "same_formal_checkpoint_step_4200": {
            int(result["checkpoint_step"]) for result in batches
        } == {4200},
        "same_full_architecture": len(
            {int(result["architecture_parameter_count"]) for result in batches}
        ) == 1
        and int(batches[0]["architecture_parameter_count"]) == 10_658_724_829,
        "all_batches_use_frozen_inference_protocol": all(
            result.get("inference") == expected_inference for result in batches
        ),
    }
    result = {
        "protocol": "paper_zero_wam_smallbox_physical_result_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint_step": 4200,
        "architecture_parameter_count": 10_658_724_829,
        "outcomes": outcomes,
        "profile_count": 20,
        "adapted_rollout_count": 40,
        "released_endpoint_rollout_count": 40,
        "rollout_count": 80,
        "prompt_specs": expected_prompt_specs,
        "frame_count": 80 * 650,
        "requested_executed_action_rows": 80 * 650,
        "maximum_requested_executed_action_error": max(
            row["maximum_requested_executed_action_error"]
            for row in trace_audits
        ),
        "paired_topology": paired_topology,
        "trace_audits": trace_audits,
        "carry_topology_advantage_count": carry_topology_wins,
        "kick_topology_advantage_count": kick_topology_wins,
        "videos": videos,
        "visualization_frame_counts": video_frame_counts,
        "inference": expected_inference,
        "hash_checks": False,
        "automatic_next_stage": (
            "motion_disjoint_physical_prompt_grid"
            if all(checks.values())
            else "record_closed_negative_without_threshold_or_training_sweep"
        ),
        "claim_boundary": (
            "Passing proves a same-checkpoint Carry45/Kick21 task switch without fall "
            "regression against the exact released endpoints in paired SMALLBOX profiles. "
            "It does not by itself prove held-out motion identity following."
        ),
    }
    write_json_atomic(root / "SMALLBOX_RESULT.json", result)
    refresh_results_document_best_effort()
    emit_json_best_effort(
        {"passed": result["passed"], "outcomes": outcomes, "checks": checks}
    )


if __name__ == "__main__":
    main()
