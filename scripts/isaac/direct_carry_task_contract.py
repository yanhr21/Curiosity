#!/usr/bin/env python3
"""Direct Isaac carry-task episode contract helpers.

This module is intentionally lightweight: it defines the JSONL row shape used
to bridge the current transparent Isaac diagnostics to a future RL task. It
does not run simulation or train a policy.
"""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "direct_isaac_carry_task_episode_v1"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _reward_terms(summary: dict[str, Any]) -> dict[str, float]:
    target_x = _f(summary.get("target_x_m"), 0.0)
    target_direction = -1.0 if target_x < 0.0 else 1.0
    target_distance = _f(
        summary.get("final_post_settle_box_target_distance_x_m"),
        _f(summary.get("final_box_target_distance_x_m"), 1.0),
    )
    signed_progress = _f(
        summary.get("final_post_settle_box_travel_x_m"),
        _f(summary.get("max_box_travel_x_m"), 0.0),
    )
    progress = _f(
        summary.get("max_target_directed_post_settle_box_travel_m"),
        target_direction * signed_progress,
    )
    fall_events = _i(summary.get("fall_events"))
    drop_events = _i(summary.get("box_drop_events"))
    contact_lt2 = _i(summary.get("drive_contact_report_lt2_steps"))
    commanded_contact_lt2 = _i(summary.get("commanded_stance_contact_report_lt2_steps"))
    effort = summary.get("per_foot_max_support_measured_effort") or {}
    peak_effort = max((_f(v) for v in effort.values()), default=0.0)
    balance_margin = _f(summary.get("min_support_polygon_margin_m"))
    travel_loss = _f(summary.get("post_settle_box_travel_loss_after_peak_m"))
    rail_motion = _f(summary.get("max_rail_joint_motion_m"))

    return {
        "progress_reward": progress,
        "signed_progress_x_m": signed_progress,
        "target_direction": target_direction,
        "target_distance_penalty": target_distance,
        "fall_penalty": 100.0 * float(fall_events),
        "drop_penalty": 50.0 * float(drop_events),
        "contact_loss_penalty": 0.01 * float(contact_lt2 + commanded_contact_lt2),
        "effort_proxy_penalty": 0.0001 * peak_effort,
        "balance_margin_reward": balance_margin,
        "travel_loss_penalty": travel_loss,
        "rail_motion_penalty": rail_motion,
    }


def reward_proxy(summary: dict[str, Any]) -> float:
    terms = _reward_terms(summary)
    return (
        terms["progress_reward"]
        + 0.5 * terms["balance_margin_reward"]
        - terms["target_distance_penalty"]
        - terms["fall_penalty"]
        - terms["drop_penalty"]
        - terms["contact_loss_penalty"]
        - terms["effort_proxy_penalty"]
        - terms["travel_loss_penalty"]
        - terms["rail_motion_penalty"]
    )


def _strict_gate_passed(summary: dict[str, Any]) -> bool:
    if "passed" in summary:
        return bool(summary.get("passed"))
    if str(summary.get("status", "")).lower() == "pass":
        return True
    if str(summary.get("status", "")).lower() in {"fail", "error"}:
        return False
    if "task_runner_backend_returncode" in summary and _i(summary.get("task_runner_backend_returncode"), 1) != 0:
        return False
    required = [
        _i(summary.get("completed_steps")) > 0,
        _i(summary.get("fall_events")) == 0,
        _i(summary.get("box_drop_events")) == 0,
        bool(summary.get("root_shortcut_free")),
        not bool(summary.get("stance_anchor_fixed_to_world")),
        _i(summary.get("support_root_pose_write_count")) == 0,
        _i(summary.get("anchor_world_joint_retarget_count")) == 0,
        _i(summary.get("foot_pose_write_count")) == 0,
        _i(summary.get("stance_anchor_pose_write_count")) == 0,
    ]
    if summary.get("support_foot_contact_report_available") is not None:
        required.extend(
            [
                bool(summary.get("support_foot_contact_report_available")),
                _i(summary.get("support_foot_contact_report_error_count")) == 0,
                _i(summary.get("min_drive_contact_report_foot_count")) >= 2,
                _i(summary.get("drive_contact_report_lt2_steps")) == 0,
                _i(summary.get("min_commanded_stance_contact_report_foot_count")) >= 2,
                _i(summary.get("commanded_stance_contact_report_lt2_steps")) == 0,
            ]
        )
    return all(required)


def _legacy_backend_capabilities(summary: dict[str, Any], parent_summary: dict[str, Any]) -> dict[str, Any]:
    scene_type = str(summary.get("scene_type", parent_summary.get("scene_type", "")))
    controller_mode = str(summary.get("controller_mode", ""))
    support_mode = str(summary.get("backend_support_mode", summary.get("support_mode", "")))
    placement_mode = str(summary.get("support_foot_placement_mode", ""))
    is_direct_scaffold = (
        "direct_carry" in scene_type
        or "direct_isaac" in scene_type
        or "anchored" in scene_type
        or "physical_alternating_anchor_feet_cradle" in controller_mode
        or "physical_alternating_placement_feet_cradle" in controller_mode
        or "alternating_anchor" in support_mode
        or placement_mode == "alternating_directional_x"
    )
    if not is_direct_scaffold:
        return {
            "backend_id": "legacy_unknown_backend",
            "backend_family": "unknown_legacy_summary",
            "isaac_backend": bool("isaac" in scene_type.lower()),
            "free_dynamic_box": None,
            "randomized_box_properties": summary.get("box_randomized", parent_summary.get("box_randomized")),
            "active_probe_supported": bool(summary.get("probe_belief_available")),
            "trainable_policy_backend": False,
            "real_robot_morphology": False,
            "support_switching_supported": False,
            "video_conditioning_supported": False,
            "root_shortcut_audited": summary.get("root_shortcut_free") is not None,
            "root_shortcut_free_claimed": summary.get("root_shortcut_free"),
            "hidden_context_isolated": True,
            "scaffold_backend": True,
            "claim_limit": "Legacy summary without explicit backend capabilities; treat as scaffold evidence only.",
            "shortcut_audit_fields": [],
            "notes": ["Capability fields were inferred during export from a pre-capability summary."],
        }
    is_directional_placement = (
        "physical_alternating_placement_feet_cradle" in controller_mode
        or placement_mode == "alternating_directional_x"
    )
    return {
        "backend_id": (
            "physical_alternating_placement_feet_cradle_v1"
            if is_directional_placement
            else "physical_alternating_anchor_feet_cradle_v1"
        ),
        "backend_family": "directional_foot_placement_scaffold" if is_directional_placement else "anchored_support_scaffold",
        "isaac_backend": True,
        "free_dynamic_box": True,
        "randomized_box_properties": summary.get("box_randomized", parent_summary.get("box_randomized")),
        "active_probe_supported": bool(summary.get("probe_belief_available")),
        "trainable_policy_backend": False,
        "real_robot_morphology": False,
        "support_switching_supported": bool(is_directional_placement),
        "video_conditioning_supported": False,
        "root_shortcut_audited": True,
        "root_shortcut_free_claimed": summary.get("root_shortcut_free"),
        "hidden_context_isolated": True,
        "scaffold_backend": True,
        "claim_limit": (
            "Inferred legacy capability for direct Isaac scaffold backend; not a full walking robot, "
            "not RL, and not video-conditioned success."
        ),
        "shortcut_audit_fields": [
            "root_shortcut_free",
            "body_root_pose_write_count",
            "body_root_velocity_command_count",
            "box_pose_write_count",
            "box_kinematic_pose_write_count",
            "support_root_pose_write_count",
            "anchor_world_joint_retarget_count",
            "foot_pose_write_count",
            "stance_anchor_pose_write_count",
        ],
        "notes": ["Capability fields were inferred during export from a pre-capability summary."],
    }


def episode_row(
    *,
    source_summary: str,
    episode_id: str,
    summary: dict[str, Any],
    parent_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one direct carry diagnostic summary into one task-contract row."""

    parent_summary = parent_summary or {}
    terms = _reward_terms(summary)
    backend_capabilities = (
        summary.get("backend_capabilities")
        or parent_summary.get("backend_capabilities")
        or _legacy_backend_capabilities(summary, parent_summary)
    )
    box_context = {
        "box_randomized": summary.get("box_randomized", parent_summary.get("box_randomized")),
        "box_seed": summary.get("box_seed", parent_summary.get("box_seed")),
        "box_mass_kg": summary.get("box_mass_kg"),
        "box_size_m": summary.get("box_size_m"),
        "box_com_offset_m": summary.get("box_com_offset_m"),
        "box_mass_range_kg": summary.get("box_mass_range_kg"),
        "box_size_jitter_fraction": summary.get("box_size_jitter_fraction"),
    }
    observation = {
        "policy_observation_contract": "no_hidden_mass_or_com",
        "video_reference_id": parent_summary.get("video_reference_id"),
        "reference_embedding_id": parent_summary.get("reference_embedding_id"),
        "target_distance_x_m": parent_summary.get("target_distance_x_m"),
        "carry_posture_command": summary.get("carry_posture", summary.get("posture")),
        "probe_mode": summary.get("probe_mode"),
        "probe_belief_available": summary.get("probe_belief_available"),
        "probe_belief_source": summary.get("probe_belief_source"),
        "probe_belief_uses_hidden_ground_truth": summary.get("probe_belief_uses_hidden_ground_truth"),
        "probe_risk_score": summary.get("probe_risk_score"),
        "probe_load_risk_bucket": summary.get("probe_load_risk_bucket"),
        "probe_recommended_carry_adjustment": summary.get("probe_recommended_carry_adjustment"),
        "max_probe_support_foot_x_tracking_error_m": summary.get("max_probe_support_foot_x_tracking_error_m"),
        "max_probe_support_foot_z_tracking_error_m": summary.get("max_probe_support_foot_z_tracking_error_m"),
        "max_probe_support_foot_x_measured_effort": summary.get("max_probe_support_foot_x_measured_effort"),
        "max_probe_support_foot_z_measured_effort": summary.get("max_probe_support_foot_z_measured_effort"),
        "support_foot_contact_report_available": summary.get("support_foot_contact_report_available"),
        "per_foot_contact_report_steps": summary.get("per_foot_contact_report_steps"),
        "min_drive_contact_report_foot_count": summary.get("min_drive_contact_report_foot_count"),
        "min_commanded_stance_contact_report_foot_count": summary.get(
            "min_commanded_stance_contact_report_foot_count"
        ),
    }
    action = {
        "action_space_version": "posture_and_controller_params_v1",
        "carry_posture": summary.get("carry_posture", summary.get("posture")),
        "controller_mode": summary.get("controller_mode"),
        "backend_support_mode": summary.get("backend_support_mode"),
        "support_foot_mode": summary.get("support_foot_mode"),
        "feedback_step_controller_enabled": summary.get("feedback_step_controller_enabled"),
        "feedback_step_x_gain": summary.get("feedback_step_x_gain"),
        "feedback_step_x_limit_m": summary.get("feedback_step_x_limit_m"),
        "feedback_step_tilt_gain": summary.get("feedback_step_tilt_gain"),
        "feedback_step_tilt_limit_m": summary.get("feedback_step_tilt_limit_m"),
        "support_foot_double_support_fraction": summary.get("support_foot_double_support_fraction"),
        "probe_steps": summary.get("probe_steps_requested"),
        "probe_amplitude_x_m": summary.get("probe_x_amplitude_m"),
        "probe_amplitude_z_m": summary.get("probe_z_amplitude_m"),
    }
    gates = {
        "passed": _strict_gate_passed(summary),
        "completed_steps": summary.get("completed_steps"),
        "fall_events": summary.get("fall_events"),
        "box_drop_events": summary.get("box_drop_events"),
        "root_shortcut_free": summary.get("root_shortcut_free"),
        "stance_anchor_fixed_to_world": summary.get("stance_anchor_fixed_to_world"),
        "support_root_pose_write_count": summary.get("support_root_pose_write_count"),
        "anchor_world_joint_retarget_count": summary.get("anchor_world_joint_retarget_count"),
        "foot_pose_write_count": summary.get("foot_pose_write_count"),
        "stance_anchor_pose_write_count": summary.get("stance_anchor_pose_write_count"),
        "support_foot_contact_report_available": summary.get("support_foot_contact_report_available"),
        "support_foot_contact_report_error_count": summary.get("support_foot_contact_report_error_count"),
        "drive_contact_report_lt2_steps": summary.get("drive_contact_report_lt2_steps"),
        "commanded_stance_contact_report_lt2_steps": summary.get("commanded_stance_contact_report_lt2_steps"),
    }
    termination = {
        "done": True,
        "status": summary.get("status"),
        "completed_steps": summary.get("completed_steps"),
        "episode_completed": _i(summary.get("completed_steps")) > 0,
        "step_limit_reached": summary.get("step_limit_reached"),
        "fall_terminated": _i(summary.get("fall_events")) > 0,
        "drop_terminated": _i(summary.get("box_drop_events")) > 0,
        "nonfinite_terminated": _i(summary.get("nonfinite_state_events")) > 0,
        "backend_error": str(summary.get("status", "")).lower() == "error"
        or _i(summary.get("task_runner_backend_returncode"), 0) != 0,
    }
    metrics = {
        "reward_proxy": reward_proxy(summary),
        "reward_terms": terms,
        "max_box_travel_x_m": summary.get("max_box_travel_x_m"),
        "max_abs_box_travel_x_m": summary.get("max_abs_box_travel_x_m"),
        "max_target_directed_post_settle_box_travel_m": summary.get(
            "max_target_directed_post_settle_box_travel_m"
        ),
        "final_box_target_distance_x_m": summary.get("final_box_target_distance_x_m"),
        "final_post_settle_box_travel_x_m": summary.get("final_post_settle_box_travel_x_m"),
        "max_abs_post_settle_box_travel_x_m": summary.get("max_abs_post_settle_box_travel_x_m"),
        "final_post_settle_box_target_distance_x_m": summary.get("final_post_settle_box_target_distance_x_m"),
        "post_settle_box_travel_loss_after_peak_m": summary.get("post_settle_box_travel_loss_after_peak_m"),
        "min_support_polygon_margin_m": summary.get("min_support_polygon_margin_m"),
        "max_actual_support_foot_lift_m": summary.get("max_actual_support_foot_lift_m"),
        "per_foot_max_support_measured_effort": summary.get("per_foot_max_support_measured_effort"),
        "max_rail_joint_motion_m": summary.get("max_rail_joint_motion_m"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "source_summary": source_summary,
        "source_scene_type": summary.get("scene_type", parent_summary.get("scene_type")),
        "success_claim": summary.get("success_claim", parent_summary.get("success_claim")),
        "not_success_reason": summary.get("not_success_reason", parent_summary.get("not_success_reason")),
        "backend_capabilities": backend_capabilities,
        "observation": observation,
        "action": action,
        "metrics": metrics,
        "termination": termination,
        "gates": gates,
        "hidden_eval_context": box_context,
        "limitation_label": "direct_isaac_scaffold_not_full_walking_robot",
    }
