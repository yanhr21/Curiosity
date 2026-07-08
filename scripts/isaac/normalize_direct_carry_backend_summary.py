#!/usr/bin/env python3
"""Normalize a physical carry backend summary into the direct-task schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize direct carry-task backend summary.")
    parser.add_argument("--backend-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--controller-mode", default="physical_anchored_cradle")
    parser.add_argument("--backend-name", default="core_world_anchored_footstep_carrier")
    parser.add_argument("--backend-log", type=Path, default=None)
    parser.add_argument("--carry-posture", default=None)
    parser.add_argument("--backend-support-mode", default=None)
    parser.add_argument(
        "--non-success-reason",
        default="backend_is_not_final_robot_controller",
        help="Explicit limitation to put in the direct-task controller contract.",
    )
    return parser.parse_args()


def _count(summary: dict, field: str) -> int:
    return int(summary.get(field) or 0)


def _backend_carrier_claim(backend: dict) -> str | None:
    if bool(backend.get("stance_anchor_fixed_to_world")):
        return "free_torso_pulled_by_driven_prismatic_joint_to_world_fixed_support_frame"
    if bool(backend.get("support_feet_fixed_to_anchor")) and bool(backend.get("disable_support_reposition")):
        return "free_torso_pulled_by_driven_prismatic_joint_to_ground_contact_support_feet"
    return backend.get("carrier_claim")


def _fallback_abs_target_distance(target: object, travel: object) -> float | None:
    if target is None or travel is None:
        return None
    return abs(float(target) - float(travel))


def _fallback_travel_loss(peak: object, final: object) -> float | None:
    if peak is None or final is None:
        return None
    return max(0.0, float(peak) - float(final))


def _first_not_none(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def main() -> None:
    args = parse_args()
    backend = json.loads(args.backend_summary.read_text())
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    root_shortcuts = {
        "root_pose_write_count": _count(backend, "root_pose_write_count"),
        "root_velocity_write_count": _count(backend, "root_velocity_write_count"),
        "root_angular_velocity_write_count": _count(backend, "root_angular_velocity_write_count"),
        "body_root_pose_write_count": _count(backend, "body_root_pose_write_count"),
        "body_root_velocity_command_count": _count(backend, "body_root_velocity_command_count"),
        "box_pose_write_count": _count(backend, "box_pose_write_count"),
        "payload_pose_write_count": _count(backend, "payload_pose_write_count"),
    }
    shortcut_free = all(value == 0 for value in root_shortcuts.values())

    backend_support_mode = args.backend_support_mode
    if backend_support_mode is None:
        backend_support_mode = (
            "fixed_anchor"
            if bool(backend.get("stance_anchor_fixed_to_world"))
            else "replant_world_joint"
            if bool(backend.get("replant_anchor_world_joint"))
            else "dynamic_anchor"
        )

    normalized = {
        "scene_type": "direct_isaac_physical_backend_carry_task",
        "success_claim": "physical_backend_diagnostic_not_full_walking_robot_success",
        "controller_mode": str(args.controller_mode),
        "carry_posture": args.carry_posture,
        "backend_support_mode": backend_support_mode,
        "controller_contract": {
            "purpose": "task_scene_with_swappable_physical_backend_not_final_robot_controller",
            "replaceable_controller_inputs": [
                "phase",
                "box_pose",
                "target_pose",
                "morphology_limits",
                "estimated_load_belief",
            ],
            "expected_controller_outputs": [
                "robot_joint_or_task_targets",
                "contact_mode",
                "probing_action",
                "carry_posture_label",
            ],
            "non_success_reason": str(args.non_success_reason),
        },
        "backend_name": str(args.backend_name),
        "backend_summary": str(args.backend_summary),
        "backend_log": None if args.backend_log is None else str(args.backend_log),
        "backend_scene_type": backend.get("scene_type"),
        "backend_success_claim": backend.get("success_claim"),
        "backend_carrier_claim": _backend_carrier_claim(backend),
        "payload_mode": backend.get("payload_mode"),
        "completed_steps": backend.get("completed_steps"),
        "steps_requested": backend.get("steps_requested"),
        "probe_steps_requested": backend.get("probe_steps_requested"),
        "probe_mode": backend.get("probe_mode"),
        "probe_x_amplitude_m": backend.get("probe_x_amplitude_m"),
        "probe_z_amplitude_m": backend.get("probe_z_amplitude_m"),
        "probe_start_step": backend.get("probe_start_step"),
        "probe_end_step": backend.get("probe_end_step"),
        "probe_belief_available": backend.get("probe_belief_available"),
        "probe_belief_source": backend.get("probe_belief_source"),
        "probe_belief_uses_hidden_ground_truth": backend.get("probe_belief_uses_hidden_ground_truth"),
        "probe_compliance_proxy": backend.get("probe_compliance_proxy"),
        "probe_lag_proxy": backend.get("probe_lag_proxy"),
        "probe_support_foot_x_tracking_proxy": backend.get("probe_support_foot_x_tracking_proxy"),
        "probe_support_foot_z_tracking_proxy": backend.get("probe_support_foot_z_tracking_proxy"),
        "probe_support_foot_x_effort_proxy": backend.get("probe_support_foot_x_effort_proxy"),
        "probe_support_foot_z_effort_proxy": backend.get("probe_support_foot_z_effort_proxy"),
        "probe_risk_score": backend.get("probe_risk_score"),
        "probe_load_risk_bucket": backend.get("probe_load_risk_bucket"),
        "probe_recommended_carry_adjustment": backend.get("probe_recommended_carry_adjustment"),
        "probe_belief_policy_action_applied": backend.get("probe_belief_policy_action_applied"),
        "max_probe_torso_travel_x_m": backend.get("max_probe_torso_travel_x_m"),
        "max_probe_torso_travel_z_m": backend.get("max_probe_torso_travel_z_m"),
        "max_probe_box_travel_x_m": backend.get("max_probe_payload_travel_x_m"),
        "max_probe_box_travel_z_m": backend.get("max_probe_payload_travel_z_m"),
        "max_probe_box_relative_error_m": backend.get("max_probe_payload_relative_error_m"),
        "max_probe_support_foot_x_tracking_error_m": backend.get("max_probe_support_foot_x_tracking_error_m"),
        "mean_probe_support_foot_x_tracking_error_m": backend.get("mean_probe_support_foot_x_tracking_error_m"),
        "probe_support_foot_x_tracking_error_samples": backend.get("probe_support_foot_x_tracking_error_samples"),
        "max_probe_support_foot_z_tracking_error_m": backend.get("max_probe_support_foot_z_tracking_error_m"),
        "mean_probe_support_foot_z_tracking_error_m": backend.get("mean_probe_support_foot_z_tracking_error_m"),
        "probe_support_foot_z_tracking_error_samples": backend.get("probe_support_foot_z_tracking_error_samples"),
        "probe_joint_effort_available": backend.get("probe_joint_effort_available"),
        "probe_joint_effort_read_error_count": backend.get("probe_joint_effort_read_error_count"),
        "probe_joint_effort_first_error": backend.get("probe_joint_effort_first_error"),
        "max_probe_support_foot_x_measured_effort": backend.get("max_probe_support_foot_x_measured_effort"),
        "mean_probe_support_foot_x_measured_effort": backend.get("mean_probe_support_foot_x_measured_effort"),
        "probe_support_foot_x_measured_effort_samples": backend.get("probe_support_foot_x_measured_effort_samples"),
        "max_probe_support_foot_z_measured_effort": backend.get("max_probe_support_foot_z_measured_effort"),
        "mean_probe_support_foot_z_measured_effort": backend.get("mean_probe_support_foot_z_measured_effort"),
        "probe_support_foot_z_measured_effort_samples": backend.get("probe_support_foot_z_measured_effort_samples"),
        "final_probe_box_lag_x_m": backend.get("final_probe_payload_lag_x_m"),
        "final_probe_box_lag_z_m": backend.get("final_probe_payload_lag_z_m"),
        "box_seed": backend.get("box_seed"),
        "box_randomized": bool(backend.get("payload_randomized", False)),
        "box_mass_kg": backend.get("payload_mass_kg"),
        "box_mass_requested_kg": backend.get("payload_mass_requested_kg"),
        "box_mass_range_kg": backend.get("payload_mass_range_kg"),
        "box_size_m": backend.get("payload_size_m"),
        "box_size_requested_m": backend.get("payload_size_requested_m"),
        "box_size_jitter_fraction": backend.get("payload_size_jitter_fraction"),
        "box_com_offset_range_m": backend.get("payload_com_offset_range_m"),
        "box_com_offset_m": backend.get("payload_com_offset_m"),
        "payload_local_x_m": backend.get("payload_local_x_m"),
        "payload_local_z_m": backend.get("payload_local_z_m"),
        "torso_z_m": backend.get("torso_z_m"),
        "stance_half_length_m": backend.get("stance_half_length_m"),
        "stance_half_width_m": backend.get("stance_half_width_m"),
        "target_x_m": backend.get("target_x_m"),
        "max_torso_travel_x_m": backend.get("max_torso_travel_x_m"),
        "max_box_travel_x_m": backend.get("max_payload_travel_x_m"),
        "max_abs_torso_travel_x_m": backend.get("max_abs_torso_travel_x_m"),
        "max_abs_box_travel_x_m": backend.get("max_abs_payload_travel_x_m"),
        "max_post_settle_torso_travel_x_m": backend.get("max_post_settle_torso_travel_x_m"),
        "max_post_settle_box_travel_x_m": backend.get("max_post_settle_payload_travel_x_m"),
        "max_abs_post_settle_torso_travel_x_m": backend.get("max_abs_post_settle_torso_travel_x_m"),
        "max_abs_post_settle_box_travel_x_m": backend.get("max_abs_post_settle_payload_travel_x_m"),
        "max_target_directed_post_settle_torso_travel_m": backend.get(
            "max_target_directed_post_settle_torso_travel_m"
        ),
        "max_target_directed_post_settle_box_travel_m": backend.get(
            "max_target_directed_post_settle_payload_travel_m"
        ),
        "final_post_settle_torso_travel_x_m": backend.get("final_post_settle_torso_travel_x_m"),
        "final_post_settle_box_travel_x_m": backend.get("final_post_settle_payload_travel_x_m"),
        "final_post_settle_torso_target_distance_x_m": backend.get("final_post_settle_target_distance_x_m"),
        "final_post_settle_box_target_distance_x_m": _first_not_none(
            backend.get("final_post_settle_payload_target_distance_x_m"),
            _fallback_abs_target_distance(
                backend.get("target_x_m"),
                backend.get("final_post_settle_payload_travel_x_m"),
            ),
        ),
        "final_target_distance_x_m": backend.get("final_target_distance_x_m"),
        "final_box_target_distance_x_m": backend.get("final_payload_target_distance_x_m"),
        "final_post_settle_box_relative_error_m": backend.get(
            "final_post_settle_payload_relative_error_m",
            backend.get("payload_relative_error_m"),
        ),
        "max_box_relative_offset_error_m": backend.get("max_payload_relative_offset_error_m"),
        "min_box_z_m": backend.get("min_payload_z_m"),
        "max_tilt_rad": backend.get("max_tilt_rad"),
        "max_roll_rad": backend.get("max_roll_rad"),
        "max_pitch_rad": backend.get("max_pitch_rad"),
        "fall_events": backend.get("fall_events"),
        "box_drop_events": backend.get("box_drop_events"),
        "articulated_carrier_enabled": backend.get("articulated_carrier_enabled"),
        "articulated_joint_count": backend.get("articulated_joint_count"),
        "foot_contact_drive_enabled": backend.get("foot_contact_drive_enabled"),
        "motion_mode": backend.get("motion_mode"),
        "horizontal_legs_enabled": backend.get("horizontal_legs_enabled"),
        "step_length_m": backend.get("step_length_m"),
        "step_height_m": backend.get("step_height_m"),
        "gait_period_steps": backend.get("gait_period_steps"),
        "x_slide_limit_m": backend.get("x_slide_limit_m"),
        "rail_joint_count": backend.get("rail_joint_count"),
        "rail_capacity_m": backend.get("rail_capacity_m"),
        "rail_joint_indices": backend.get("rail_joint_indices"),
        "max_joint_motion_m": backend.get("max_joint_motion_m"),
        "max_rail_joint_motion_m": backend.get("max_rail_joint_motion_m"),
        "max_clamp_joint_motion_m": backend.get("max_clamp_joint_motion_m"),
        "max_cradle_joint_motion_m": backend.get("max_cradle_joint_motion_m"),
        "max_commanded_clamp_target_m": backend.get("max_commanded_clamp_target_m"),
        "final_commanded_clamp_target_m": backend.get("final_commanded_clamp_target_m"),
        "clamp_drive_target_update_count": backend.get("clamp_drive_target_update_count"),
        "max_commanded_cradle_target_m": backend.get("max_commanded_cradle_target_m"),
        "final_commanded_cradle_target_m": backend.get("final_commanded_cradle_target_m"),
        "cradle_drive_target_update_count": backend.get("cradle_drive_target_update_count"),
        "cycle_count": backend.get("cycle_count"),
        "stride_m": backend.get("stride_m"),
        "foot_pose_write_count": _count(backend, "foot_pose_write_count"),
        "stance_anchor_pose_write_count": _count(backend, "stance_anchor_pose_write_count"),
        "support_foot_mode": backend.get("support_foot_mode"),
        "support_feet_fixed_to_anchor": bool(backend.get("support_feet_fixed_to_anchor", False)),
        "support_foot_mass_kg": backend.get("support_foot_mass_kg"),
        "support_foot_joint_count": backend.get("support_foot_joint_count"),
        "support_foot_x_joint_count": backend.get("support_foot_x_joint_count"),
        "support_foot_x_joint_indices": backend.get("support_foot_x_joint_indices"),
        "support_foot_z_joint_count": backend.get("support_foot_z_joint_count"),
        "support_foot_z_joint_indices": backend.get("support_foot_z_joint_indices"),
        "support_foot_x_lower_m": backend.get("support_foot_x_lower_m"),
        "support_foot_x_upper_m": backend.get("support_foot_x_upper_m"),
        "support_foot_z_lower_m": backend.get("support_foot_z_lower_m"),
        "support_foot_z_upper_m": backend.get("support_foot_z_upper_m"),
        "use_support_foot_drive": bool(backend.get("use_support_foot_drive", False)),
        "support_foot_drive_direction_scale": backend.get("support_foot_drive_direction_scale"),
        "support_foot_placement_mode": backend.get("support_foot_placement_mode"),
        "support_foot_placement_controller_enabled": bool(
            backend.get("support_foot_placement_controller_enabled", False)
        ),
        "support_foot_directional_placement": bool(backend.get("support_foot_directional_placement", False)),
        "support_foot_step_height_m": backend.get("support_foot_step_height_m"),
        "support_foot_stance_x_m": backend.get("support_foot_stance_x_m"),
        "support_foot_swing_x_m": backend.get("support_foot_swing_x_m"),
        "support_foot_contact_z_threshold_m": backend.get("support_foot_contact_z_threshold_m"),
        "support_foot_contact_report_requested": bool(backend.get("support_foot_contact_report_requested", False)),
        "support_foot_contact_report_available": bool(backend.get("support_foot_contact_report_available", False)),
        "support_foot_contact_report_threshold": backend.get("support_foot_contact_report_threshold"),
        "support_foot_contact_report_enabled_paths": backend.get("support_foot_contact_report_enabled_paths"),
        "support_foot_contact_report_event_count": backend.get("support_foot_contact_report_event_count"),
        "support_foot_contact_report_error_count": backend.get("support_foot_contact_report_error_count"),
        "support_foot_contact_report_first_error": backend.get("support_foot_contact_report_first_error"),
        "support_foot_effort_contact_threshold": backend.get("support_foot_effort_contact_threshold"),
        "support_foot_double_support_fraction": backend.get("support_foot_double_support_fraction"),
        "support_foot_continuity_grace_steps": backend.get("support_foot_continuity_grace_steps"),
        "support_foot_continuity_start_step": backend.get("support_foot_continuity_start_step"),
        "stance_foot_world_lock_enabled": bool(backend.get("stance_foot_world_lock_enabled", False)),
        "stance_foot_world_lock_joint_count": backend.get("stance_foot_world_lock_joint_count"),
        "stance_foot_world_lock_switch_count": backend.get("stance_foot_world_lock_switch_count"),
        "stance_foot_world_lock_pose_update_count": backend.get("stance_foot_world_lock_pose_update_count"),
        "stance_foot_world_lock_active_feet": backend.get("stance_foot_world_lock_active_feet"),
        "freeze_locked_stance_foot_targets_enabled": bool(
            backend.get("freeze_locked_stance_foot_targets_enabled", False)
        ),
        "freeze_locked_stance_foot_target_count": backend.get("freeze_locked_stance_foot_target_count"),
        "freeze_commanded_stance_foot_targets_enabled": bool(
            backend.get("freeze_commanded_stance_foot_targets_enabled", False)
        ),
        "freeze_commanded_stance_foot_target_count": backend.get("freeze_commanded_stance_foot_target_count"),
        "freeze_commanded_stance_foot_target_switch_count": backend.get(
            "freeze_commanded_stance_foot_target_switch_count"
        ),
        "freeze_commanded_stance_foot_active_feet": backend.get("freeze_commanded_stance_foot_active_feet"),
        "planted_stance_rail_propulsion_enabled": bool(
            backend.get("planted_stance_rail_propulsion_enabled", False)
        ),
        "planted_stance_rail_propulsion_steps": backend.get("planted_stance_rail_propulsion_steps"),
        "feedback_step_controller_enabled": bool(backend.get("feedback_step_controller_enabled", False)),
        "feedback_step_x_gain": backend.get("feedback_step_x_gain"),
        "feedback_step_x_limit_m": backend.get("feedback_step_x_limit_m"),
        "feedback_step_tilt_gain": backend.get("feedback_step_tilt_gain"),
        "feedback_step_tilt_limit_m": backend.get("feedback_step_tilt_limit_m"),
        "feedback_step_applied_steps": backend.get("feedback_step_applied_steps"),
        "max_abs_feedback_step_x_adjustment_m": backend.get("max_abs_feedback_step_x_adjustment_m"),
        "max_abs_feedback_step_tilt_adjustment_m": backend.get("max_abs_feedback_step_tilt_adjustment_m"),
        "online_probe_adaptive_support_enabled": bool(
            backend.get("online_probe_adaptive_support_enabled", False)
        ),
        "online_probe_adaptive_support_decision_applied": bool(
            backend.get("online_probe_adaptive_support_decision_applied", False)
        ),
        "online_probe_adaptive_support_decision_step": backend.get(
            "online_probe_adaptive_support_decision_step"
        ),
        "online_probe_adaptive_support_uses_hidden_ground_truth": bool(
            backend.get("online_probe_adaptive_support_uses_hidden_ground_truth", False)
        ),
        "online_probe_adaptive_support_risk_score": backend.get("online_probe_adaptive_support_risk_score"),
        "online_probe_adaptive_support_risk_bucket": backend.get("online_probe_adaptive_support_risk_bucket"),
        "online_probe_adaptive_support_profile": backend.get("online_probe_adaptive_support_profile"),
        "online_probe_adaptive_support_step_height_m": backend.get(
            "online_probe_adaptive_support_step_height_m"
        ),
        "online_probe_adaptive_support_double_support_fraction": backend.get(
            "online_probe_adaptive_support_double_support_fraction"
        ),
        "online_probe_adaptive_support_stance_x_m": backend.get("online_probe_adaptive_support_stance_x_m"),
        "online_probe_adaptive_support_swing_x_m": backend.get("online_probe_adaptive_support_swing_x_m"),
        "online_probe_adaptive_support_medium_threshold": backend.get(
            "online_probe_adaptive_support_medium_threshold"
        ),
        "online_probe_adaptive_support_high_threshold": backend.get(
            "online_probe_adaptive_support_high_threshold"
        ),
        "online_probe_adaptive_hold_enabled": bool(backend.get("online_probe_adaptive_hold_enabled", False)),
        "online_probe_adaptive_hold_decision_applied": bool(
            backend.get("online_probe_adaptive_hold_decision_applied", False)
        ),
        "online_probe_adaptive_hold_decision_step": backend.get("online_probe_adaptive_hold_decision_step"),
        "online_probe_adaptive_hold_uses_hidden_ground_truth": bool(
            backend.get("online_probe_adaptive_hold_uses_hidden_ground_truth", False)
        ),
        "online_probe_adaptive_hold_risk_score": backend.get("online_probe_adaptive_hold_risk_score"),
        "online_probe_adaptive_hold_risk_bucket": backend.get("online_probe_adaptive_hold_risk_bucket"),
        "online_probe_adaptive_hold_profile": backend.get("online_probe_adaptive_hold_profile"),
        "online_probe_adaptive_hold_closure_fraction": backend.get(
            "online_probe_adaptive_hold_closure_fraction"
        ),
        "online_probe_adaptive_hold_actuated": bool(backend.get("online_probe_adaptive_hold_actuated", False)),
        "online_probe_adaptive_hold_collision_available": bool(
            backend.get("online_probe_adaptive_hold_collision_available", False)
        ),
        "online_probe_adaptive_hold_collision_paths": backend.get("online_probe_adaptive_hold_collision_paths"),
        "online_probe_adaptive_hold_collision_enabled": bool(
            backend.get("online_probe_adaptive_hold_collision_enabled", False)
        ),
        "online_probe_adaptive_hold_collision_update_count": backend.get(
            "online_probe_adaptive_hold_collision_update_count"
        ),
        "online_probe_adaptive_hold_low_closure_fraction": backend.get(
            "online_probe_adaptive_hold_low_closure_fraction"
        ),
        "online_probe_adaptive_hold_medium_closure_fraction": backend.get(
            "online_probe_adaptive_hold_medium_closure_fraction"
        ),
        "online_probe_adaptive_hold_high_closure_fraction": backend.get(
            "online_probe_adaptive_hold_high_closure_fraction"
        ),
        "alternating_support_foot_drive": bool(backend.get("alternating_support_foot_drive", False)),
        "max_support_foot_x_joint_motion_m": backend.get("max_support_foot_x_joint_motion_m"),
        "max_support_foot_z_joint_motion_m": backend.get("max_support_foot_z_joint_motion_m"),
        "max_commanded_support_foot_lift_m": backend.get("max_commanded_support_foot_lift_m"),
        "per_foot_max_commanded_x_m": backend.get("per_foot_max_commanded_x_m"),
        "per_foot_max_commanded_z_m": backend.get("per_foot_max_commanded_z_m"),
        "final_support_foot_x_joint_target_m": backend.get("final_support_foot_x_joint_target_m"),
        "final_support_foot_x_joint_target_m_by_foot": backend.get("final_support_foot_x_joint_target_m_by_foot"),
        "final_support_foot_z_joint_target_m_by_foot": backend.get("final_support_foot_z_joint_target_m_by_foot"),
        "disable_support_reposition": bool(backend.get("disable_support_reposition", False)),
        "final_anchor_travel_x_m": backend.get("final_anchor_travel_x_m"),
        "max_abs_anchor_travel_x_m": backend.get("max_abs_anchor_travel_x_m"),
        "max_anchor_travel_xy_m": backend.get("max_anchor_travel_xy_m"),
        "support_foot_min_z_m": backend.get("support_foot_min_z_m"),
        "support_foot_max_z_m": backend.get("support_foot_max_z_m"),
        "max_actual_support_foot_lift_m": backend.get("max_actual_support_foot_lift_m"),
        "per_foot_max_actual_lift_m": backend.get("per_foot_max_actual_lift_m"),
        "per_foot_min_z_m": backend.get("per_foot_min_z_m"),
        "per_foot_max_z_m": backend.get("per_foot_max_z_m"),
        "per_foot_near_ground_steps": backend.get("per_foot_near_ground_steps"),
        "per_foot_max_near_ground_xy_slip_m": backend.get("per_foot_max_near_ground_xy_slip_m"),
        "per_foot_max_near_ground_xy_speed_mps": backend.get("per_foot_max_near_ground_xy_speed_mps"),
        "min_near_ground_foot_count": backend.get("min_near_ground_foot_count"),
        "max_near_ground_foot_count": backend.get("max_near_ground_foot_count"),
        "near_ground_zero_steps": backend.get("near_ground_zero_steps"),
        "near_ground_lt2_steps": backend.get("near_ground_lt2_steps"),
        "min_drive_near_ground_foot_count": backend.get("min_drive_near_ground_foot_count"),
        "drive_near_ground_zero_steps": backend.get("drive_near_ground_zero_steps"),
        "drive_near_ground_lt2_steps": backend.get("drive_near_ground_lt2_steps"),
        "per_foot_contact_report_steps": backend.get("per_foot_contact_report_steps"),
        "min_contact_report_foot_count": backend.get("min_contact_report_foot_count"),
        "max_contact_report_foot_count": backend.get("max_contact_report_foot_count"),
        "contact_report_zero_steps": backend.get("contact_report_zero_steps"),
        "contact_report_lt2_steps": backend.get("contact_report_lt2_steps"),
        "min_drive_contact_report_foot_count": backend.get("min_drive_contact_report_foot_count"),
        "drive_contact_report_zero_steps": backend.get("drive_contact_report_zero_steps"),
        "drive_contact_report_lt2_steps": backend.get("drive_contact_report_lt2_steps"),
        "min_commanded_stance_contact_report_foot_count": backend.get(
            "min_commanded_stance_contact_report_foot_count"
        ),
        "commanded_stance_contact_report_lt2_steps": backend.get(
            "commanded_stance_contact_report_lt2_steps"
        ),
        "support_foot_effort_available": bool(backend.get("support_foot_effort_available", False)),
        "support_foot_effort_read_error_count": backend.get("support_foot_effort_read_error_count"),
        "support_foot_effort_first_error": backend.get("support_foot_effort_first_error"),
        "per_foot_max_support_x_measured_effort": backend.get("per_foot_max_support_x_measured_effort"),
        "per_foot_max_support_z_measured_effort": backend.get("per_foot_max_support_z_measured_effort"),
        "per_foot_max_support_measured_effort": backend.get("per_foot_max_support_measured_effort"),
        "min_drive_effort_supported_foot_count": backend.get("min_drive_effort_supported_foot_count"),
        "drive_effort_supported_zero_steps": backend.get("drive_effort_supported_zero_steps"),
        "drive_effort_supported_lt2_steps": backend.get("drive_effort_supported_lt2_steps"),
        "min_commanded_stance_effort_supported_foot_count": backend.get(
            "min_commanded_stance_effort_supported_foot_count"
        ),
        "commanded_stance_effort_supported_lt2_steps": backend.get(
            "commanded_stance_effort_supported_lt2_steps"
        ),
        "min_commanded_stance_near_ground_foot_count": backend.get("min_commanded_stance_near_ground_foot_count"),
        "commanded_stance_near_ground_lt2_steps": backend.get("commanded_stance_near_ground_lt2_steps"),
        "min_support_polygon_margin_x_m": backend.get("min_support_polygon_margin_x_m"),
        "min_support_polygon_margin_y_m": backend.get("min_support_polygon_margin_y_m"),
        "min_support_polygon_margin_m": backend.get("min_support_polygon_margin_m"),
        "max_abs_support_foot_travel_x_m": backend.get("max_abs_support_foot_travel_x_m"),
        "max_support_foot_travel_xy_m": backend.get("max_support_foot_travel_xy_m"),
        "final_support_foot_travel_x_m": backend.get("final_support_foot_travel_x_m"),
        "quasistatic_compensate_settle_drift": backend.get("quasistatic_compensate_settle_drift"),
        "quasistatic_effective_target_x_m": backend.get("quasistatic_effective_target_x_m"),
        "post_settle_box_travel_loss_after_peak_m": _first_not_none(
            backend.get("post_settle_payload_travel_loss_after_peak_m"),
            _fallback_travel_loss(
                backend.get("max_post_settle_payload_travel_x_m"),
                backend.get("final_post_settle_payload_travel_x_m"),
            ),
        ),
        "gated_step_max_travel_loss_m": backend.get("gated_step_max_travel_loss_m"),
        "gated_step_recovery_phase": backend.get("gated_step_recovery_phase"),
        "gated_step_loss_rebaseline_steps": backend.get("gated_step_loss_rebaseline_steps"),
        "gated_step_loss_rebaseline_count": backend.get("gated_step_loss_rebaseline_count"),
        "gated_step_motion_step_final": backend.get("gated_step_motion_step_final"),
        "gated_step_hold_steps": backend.get("gated_step_hold_steps"),
        "gated_step_release_steps": backend.get("gated_step_release_steps"),
        "gated_step_recovery_steps": backend.get("gated_step_recovery_steps"),
        "gated_step_last_safe": backend.get("gated_step_last_safe"),
        "gated_step_last_block_reason": backend.get("gated_step_last_block_reason"),
        "gated_step_peak_post_settle_box_travel_x_m": backend.get("gated_step_peak_post_settle_payload_travel_x_m"),
        "gated_step_peak_step": backend.get("gated_step_peak_step"),
        "gated_step_travel_loss_after_peak_m": backend.get("gated_step_travel_loss_after_peak_m"),
        "prelift_reset_lift_fraction": backend.get("prelift_reset_lift_fraction"),
        "prelift_reset_lower_fraction": backend.get("prelift_reset_lower_fraction"),
        "prelift_stance_overdrive": backend.get("prelift_stance_overdrive"),
        "guarded_step_target_tolerance_m": backend.get("guarded_step_target_tolerance_m"),
        "max_commanded_leg_lift_m": backend.get("max_commanded_leg_lift_m"),
        "max_abs_commanded_x_slide_target_m": backend.get("max_abs_commanded_x_slide_target_m"),
        "swing_x_force_scale": backend.get("swing_x_force_scale"),
        "swing_x_force_scaled_steps": backend.get("swing_x_force_scaled_steps"),
        "per_leg_swing_x_force_scaled_steps": backend.get("per_leg_swing_x_force_scaled_steps"),
        "stance_foot_latch_enabled": backend.get("stance_foot_latch_enabled"),
        "stance_foot_latch_is_scaffold": backend.get("stance_foot_latch_is_scaffold"),
        "stance_foot_latch_lift_threshold_m": backend.get("stance_foot_latch_lift_threshold_m"),
        "stance_foot_latch_enable_count": backend.get("stance_foot_latch_enable_count"),
        "stance_foot_latch_disable_count": backend.get("stance_foot_latch_disable_count"),
        "stance_foot_latch_retarget_count": backend.get("stance_foot_latch_retarget_count"),
        "per_leg_stance_latch_enabled_steps": backend.get("per_leg_stance_latch_enabled_steps"),
        "per_leg_stance_latch_enable_count": backend.get("per_leg_stance_latch_enable_count"),
        "per_leg_stance_latch_disable_count": backend.get("per_leg_stance_latch_disable_count"),
        "per_leg_stance_latch_retarget_count": backend.get("per_leg_stance_latch_retarget_count"),
        "max_actual_leg_lift_m": backend.get("max_actual_leg_lift_m"),
        "max_abs_actual_x_slide_m": backend.get("max_abs_actual_x_slide_m"),
        "min_foot_z_m": backend.get("min_foot_z_m"),
        "max_foot_z_m": backend.get("max_foot_z_m"),
        "foot_contact_z_threshold_m": backend.get("foot_contact_z_threshold_m"),
        "per_leg_near_ground_steps": backend.get("per_leg_near_ground_steps"),
        "per_leg_min_foot_z_m": backend.get("per_leg_min_foot_z_m"),
        "per_leg_max_foot_z_m": backend.get("per_leg_max_foot_z_m"),
        "per_leg_max_commanded_lift_m": backend.get("per_leg_max_commanded_lift_m"),
        "per_leg_max_abs_commanded_x_m": backend.get("per_leg_max_abs_commanded_x_m"),
        "per_leg_max_actual_lift_m": backend.get("per_leg_max_actual_lift_m"),
        "per_leg_max_abs_actual_x_m": backend.get("per_leg_max_abs_actual_x_m"),
        "root_shortcuts": root_shortcuts,
        "root_shortcut_free": shortcut_free,
        "support_root_pose_write_count": _count(backend, "support_root_pose_write_count"),
        "anchor_world_joint_retarget_count": _count(backend, "anchor_world_joint_retarget_count"),
        "stance_anchor_fixed_to_world": bool(backend.get("stance_anchor_fixed_to_world", False)),
        "stance_anchor_kinematic": bool(backend.get("stance_anchor_kinematic", False)),
        "stance_anchor_dynamic_high_mass": bool(backend.get("stance_anchor_dynamic_high_mass", False)),
        "stance_anchor_as_articulation_root": bool(backend.get("stance_anchor_as_articulation_root", False)),
        "articulation_root_path": backend.get("articulation_root_path"),
        "replant_anchor_world_joint": bool(backend.get("replant_anchor_world_joint", False)),
        "cumulative_cycle_target": backend.get("cumulative_cycle_target"),
        "robot_proxy_pose_write_count": 0,
        "box_kinematic_pose_write_count": 0,
    }
    args.output_summary.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Normalized direct backend summary: {args.output_summary}")


if __name__ == "__main__":
    main()
