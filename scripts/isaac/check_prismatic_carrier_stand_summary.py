#!/usr/bin/env python3
"""Check no-root prismatic carrier stand summaries.

This checker reads only JSON/log text, so it is safe on the login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check prismatic carrier stand summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--max-fall-events", type=int, default=0)
    parser.add_argument("--max-box-drop-events", type=int, default=0)
    parser.add_argument("--max-root-pose-writes", type=int, default=0)
    parser.add_argument("--max-root-velocity-writes", type=int, default=0)
    parser.add_argument("--max-root-angular-velocity-writes", type=int, default=0)
    parser.add_argument("--max-body-root-pose-writes", type=int, default=0)
    parser.add_argument("--max-body-root-velocity-commands", type=int, default=0)
    parser.add_argument("--max-box-pose-writes", type=int, default=0)
    parser.add_argument("--max-payload-pose-writes", type=int, default=0)
    parser.add_argument("--max-tilt", type=float, default=0.85)
    parser.add_argument("--max-stand-drift", type=float, default=None)
    parser.add_argument("--min-torso-travel-x", type=float, default=None)
    parser.add_argument("--min-payload-travel-x", type=float, default=None)
    parser.add_argument("--min-abs-torso-travel-x", type=float, default=None)
    parser.add_argument("--min-abs-payload-travel-x", type=float, default=None)
    parser.add_argument("--min-abs-post-settle-torso-travel-x", type=float, default=None)
    parser.add_argument("--min-abs-post-settle-payload-travel-x", type=float, default=None)
    parser.add_argument("--max-final-target-distance-x", type=float, default=None)
    parser.add_argument("--max-final-payload-target-distance-x", type=float, default=None)
    parser.add_argument("--max-final-post-settle-target-distance-x", type=float, default=None)
    parser.add_argument("--max-final-post-settle-payload-target-distance-x", type=float, default=None)
    parser.add_argument("--max-payload-relative-offset-error", type=float, default=None)
    parser.add_argument("--max-post-settle-payload-relative-offset-error", type=float, default=None)
    parser.add_argument("--min-payload-z", type=float, default=None)
    parser.add_argument("--expect-motion-mode", default=None)
    parser.add_argument("--expect-payload-mode", default=None)
    parser.add_argument("--min-joint-count", type=int, default=1)
    parser.add_argument("--min-joint-motion", type=float, default=None)
    parser.add_argument("--max-nonfinite-events", type=int, default=0)
    parser.add_argument("--require-articulated-carrier", action="store_true")
    parser.add_argument("--require-foot-contact-drive", action="store_true")
    parser.add_argument("--require-active-probe", action="store_true")
    parser.add_argument("--require-probe-belief", action="store_true")
    parser.add_argument("--require-no-hidden-probe-gt", action="store_true")
    parser.add_argument("--min-active-probe-steps", type=int, default=None)
    parser.add_argument("--require-probe-adaptive-gait-decision", action="store_true")
    parser.add_argument("--expect-probe-adaptive-risk-bucket", default=None)
    parser.add_argument("--expect-probe-adaptive-gait-drive-scale", type=float, default=None)
    parser.add_argument("--probe-adaptive-scale-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--require-probe-adaptive-posture-decision", action="store_true")
    parser.add_argument("--expect-probe-adaptive-posture-strategy", default=None)
    parser.add_argument("--expect-probe-adaptive-posture-risk-bucket", default=None)
    parser.add_argument("--expect-probe-adaptive-posture-leg-target-offset", type=float, default=None)
    parser.add_argument("--probe-adaptive-posture-offset-tolerance", type=float, default=1.0e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    if summary.get("error"):
        failures.append(f"summary error: {summary['error']}")
    if int(summary.get("completed_steps", 0)) < int(summary.get("steps_requested", 0)):
        failures.append(f"incomplete steps: {summary.get('completed_steps')} / {summary.get('steps_requested')}")
    if args.expect_motion_mode is not None and summary.get("motion_mode") != args.expect_motion_mode:
        failures.append(f"motion mode mismatch: {summary.get('motion_mode')} != {args.expect_motion_mode}")
    if args.expect_payload_mode is not None and summary.get("payload_mode") != args.expect_payload_mode:
        failures.append(f"payload mode mismatch: {summary.get('payload_mode')} != {args.expect_payload_mode}")
    if args.require_articulated_carrier and not bool(summary.get("articulated_carrier_enabled")):
        failures.append(f"articulated carrier not enabled: {summary.get('articulated_carrier_enabled')}")
    if int(summary.get("articulated_joint_count") or 0) < args.min_joint_count:
        failures.append(f"joint count too low: {summary.get('articulated_joint_count')}")
    if args.require_foot_contact_drive and not bool(summary.get("foot_contact_drive_enabled")):
        failures.append(f"foot contact drive not enabled: {summary.get('foot_contact_drive_enabled')}")
    if args.require_active_probe and not bool(summary.get("active_probe_enabled")):
        failures.append(f"active probe not enabled: {summary.get('active_probe_enabled')}")
    if args.require_probe_belief and not bool(summary.get("active_probe_belief_available")):
        failures.append(f"active probe belief not available: {summary.get('active_probe_belief_available')}")
    if args.require_no_hidden_probe_gt and bool(summary.get("active_probe_uses_hidden_ground_truth")):
        failures.append("active probe uses hidden ground truth")
    if args.min_active_probe_steps is not None:
        observed_probe_steps = int(summary.get("active_probe_steps_observed") or 0)
        if observed_probe_steps < int(args.min_active_probe_steps):
            failures.append(f"active probe steps too low: {observed_probe_steps}")
    if args.require_probe_adaptive_gait_decision and not bool(summary.get("probe_adaptive_gait_decision_available")):
        failures.append(
            f"probe-adaptive gait decision not available: {summary.get('probe_adaptive_gait_decision_available')}"
        )
    if (
        args.expect_probe_adaptive_risk_bucket is not None
        and summary.get("probe_adaptive_risk_bucket") != args.expect_probe_adaptive_risk_bucket
    ):
        failures.append(
            "probe-adaptive risk bucket mismatch: "
            f"{summary.get('probe_adaptive_risk_bucket')} != {args.expect_probe_adaptive_risk_bucket}"
        )
    if args.expect_probe_adaptive_gait_drive_scale is not None:
        observed_scale = summary.get("probe_adaptive_gait_drive_scale")
        if observed_scale is None or abs(float(observed_scale) - args.expect_probe_adaptive_gait_drive_scale) > float(
            args.probe_adaptive_scale_tolerance
        ):
            failures.append(
                "probe-adaptive gait drive scale mismatch: "
                f"{observed_scale} != {args.expect_probe_adaptive_gait_drive_scale}"
            )
    if args.require_probe_adaptive_posture_decision and not bool(
        summary.get("probe_adaptive_posture_decision_available")
    ):
        failures.append(
            "probe-adaptive posture decision not available: "
            f"{summary.get('probe_adaptive_posture_decision_available')}"
        )
    if (
        args.expect_probe_adaptive_posture_strategy is not None
        and summary.get("probe_adaptive_posture_strategy") != args.expect_probe_adaptive_posture_strategy
    ):
        failures.append(
            "probe-adaptive posture strategy mismatch: "
            f"{summary.get('probe_adaptive_posture_strategy')} != {args.expect_probe_adaptive_posture_strategy}"
        )
    if (
        args.expect_probe_adaptive_posture_risk_bucket is not None
        and summary.get("probe_adaptive_posture_risk_bucket") != args.expect_probe_adaptive_posture_risk_bucket
    ):
        failures.append(
            "probe-adaptive posture risk bucket mismatch: "
            f"{summary.get('probe_adaptive_posture_risk_bucket')} != "
            f"{args.expect_probe_adaptive_posture_risk_bucket}"
        )
    if args.expect_probe_adaptive_posture_leg_target_offset is not None:
        observed_offset = summary.get("probe_adaptive_posture_leg_target_offset_m")
        if observed_offset is None or abs(
            float(observed_offset) - args.expect_probe_adaptive_posture_leg_target_offset
        ) > float(args.probe_adaptive_posture_offset_tolerance):
            failures.append(
                "probe-adaptive posture leg target offset mismatch: "
                f"{observed_offset} != {args.expect_probe_adaptive_posture_leg_target_offset}"
            )
    if int(summary.get("fall_events", 0)) > args.max_fall_events:
        failures.append(f"fall events too high: {summary.get('fall_events')}")
    if int(summary.get("box_drop_events", 0)) > args.max_box_drop_events:
        failures.append(f"box drop events too high: {summary.get('box_drop_events')}")
    for field, limit in (
        ("root_pose_write_count", args.max_root_pose_writes),
        ("root_velocity_write_count", args.max_root_velocity_writes),
        ("root_angular_velocity_write_count", args.max_root_angular_velocity_writes),
        ("body_root_pose_write_count", args.max_body_root_pose_writes),
        ("body_root_velocity_command_count", args.max_body_root_velocity_commands),
        ("box_pose_write_count", args.max_box_pose_writes),
        ("payload_pose_write_count", args.max_payload_pose_writes),
    ):
        value = int(summary.get(field) or 0)
        if value > int(limit):
            failures.append(f"{field} too high: {value}")
    if float(summary.get("max_tilt_rad", 0.0)) > args.max_tilt:
        failures.append(f"tilt too high: {summary.get('max_tilt_rad')}")
    if args.max_stand_drift is not None and float(summary.get("max_torso_drift_xy_m", 0.0)) > args.max_stand_drift:
        failures.append(f"stand drift too high: {summary.get('max_torso_drift_xy_m')}")
    if args.min_torso_travel_x is not None and float(summary.get("max_torso_travel_x_m", 0.0)) < args.min_torso_travel_x:
        failures.append(f"torso travel x too low: {summary.get('max_torso_travel_x_m')}")
    if args.min_payload_travel_x is not None and float(summary.get("max_payload_travel_x_m", 0.0)) < args.min_payload_travel_x:
        failures.append(f"payload travel x too low: {summary.get('max_payload_travel_x_m')}")
    if args.min_abs_torso_travel_x is not None and float(summary.get("max_abs_torso_travel_x_m", 0.0)) < args.min_abs_torso_travel_x:
        failures.append(f"absolute torso travel x too low: {summary.get('max_abs_torso_travel_x_m')}")
    if args.min_abs_payload_travel_x is not None and float(summary.get("max_abs_payload_travel_x_m", 0.0)) < args.min_abs_payload_travel_x:
        failures.append(f"absolute payload travel x too low: {summary.get('max_abs_payload_travel_x_m')}")
    if args.min_abs_post_settle_torso_travel_x is not None:
        post_settle_torso_travel = summary.get("max_abs_post_settle_torso_travel_x_m")
        if post_settle_torso_travel is None or float(post_settle_torso_travel) < args.min_abs_post_settle_torso_travel_x:
            failures.append(f"absolute post-settle torso travel x too low: {post_settle_torso_travel}")
    if args.min_abs_post_settle_payload_travel_x is not None:
        post_settle_payload_travel = summary.get("max_abs_post_settle_payload_travel_x_m")
        if post_settle_payload_travel is None or float(post_settle_payload_travel) < args.min_abs_post_settle_payload_travel_x:
            failures.append(f"absolute post-settle payload travel x too low: {post_settle_payload_travel}")
    if args.max_final_target_distance_x is not None:
        target_distance = summary.get("final_target_distance_x_m")
        if target_distance is None or float(target_distance) > args.max_final_target_distance_x:
            failures.append(f"final target distance x too high: {target_distance}")
    if args.max_final_payload_target_distance_x is not None:
        payload_target_distance = summary.get("final_payload_target_distance_x_m")
        if payload_target_distance is None or float(payload_target_distance) > args.max_final_payload_target_distance_x:
            failures.append(f"final payload target distance x too high: {payload_target_distance}")
    if args.max_final_post_settle_target_distance_x is not None:
        post_settle_target_distance = summary.get("final_post_settle_target_distance_x_m")
        if post_settle_target_distance is None or float(post_settle_target_distance) > args.max_final_post_settle_target_distance_x:
            failures.append(f"final post-settle target distance x too high: {post_settle_target_distance}")
    if args.max_final_post_settle_payload_target_distance_x is not None:
        post_settle_payload_target_distance = summary.get("final_post_settle_payload_target_distance_x_m")
        if post_settle_payload_target_distance is None or float(post_settle_payload_target_distance) > args.max_final_post_settle_payload_target_distance_x:
            failures.append(f"final post-settle payload target distance x too high: {post_settle_payload_target_distance}")
    if args.max_payload_relative_offset_error is not None:
        payload_offset_error = summary.get("max_payload_relative_offset_error_m", summary.get("payload_relative_error_m"))
        if payload_offset_error is None or float(payload_offset_error) > args.max_payload_relative_offset_error:
            failures.append(f"payload relative offset error too high: {payload_offset_error}")
    if args.max_post_settle_payload_relative_offset_error is not None:
        post_settle_payload_offset_error = summary.get("max_post_settle_payload_relative_offset_error_m")
        if post_settle_payload_offset_error is None or float(post_settle_payload_offset_error) > args.max_post_settle_payload_relative_offset_error:
            failures.append(f"post-settle payload relative offset error too high: {post_settle_payload_offset_error}")
    if args.min_payload_z is not None:
        min_payload_z = summary.get("min_payload_z_m")
        if min_payload_z is None or float(min_payload_z) < args.min_payload_z:
            failures.append(f"payload z too low: {min_payload_z}")
    if int(summary.get("nonfinite_state_events", 0)) > args.max_nonfinite_events:
        failures.append(f"nonfinite state events too high: {summary.get('nonfinite_state_events')}")
    if args.min_joint_motion is not None:
        joint_motion = summary.get("max_joint_motion_m")
        if joint_motion is None or float(joint_motion) < args.min_joint_motion:
            failures.append(f"joint motion too low: {joint_motion}")

    disjoint_warning = False
    if args.log is not None and args.log.exists():
        log_text = args.log.read_text(errors="ignore")
        disjoint_warning = "disjointed body transforms" in log_text or "disjoint body transforms" in log_text
        if disjoint_warning:
            failures.append("disjoint fixed-joint warning present in log")

    report = {
        "summary": str(args.summary),
        "log": None if args.log is None else str(args.log),
        "scene_type": summary.get("scene_type"),
        "success_claim": summary.get("success_claim"),
        "motion_mode": summary.get("motion_mode"),
        "payload_mode": summary.get("payload_mode"),
        "horizontal_legs_enabled": summary.get("horizontal_legs_enabled"),
        "completed_steps": summary.get("completed_steps"),
        "steps_requested": summary.get("steps_requested"),
        "articulated_carrier_enabled": summary.get("articulated_carrier_enabled"),
        "articulated_joint_count": summary.get("articulated_joint_count"),
        "foot_contact_drive_enabled": summary.get("foot_contact_drive_enabled"),
        "active_probe_enabled": summary.get("active_probe_enabled"),
        "active_probe_uses_hidden_ground_truth": summary.get("active_probe_uses_hidden_ground_truth"),
        "active_probe_steps_observed": summary.get("active_probe_steps_observed"),
        "active_probe_belief_available": summary.get("active_probe_belief_available"),
        "active_probe_observed_load_risk_bucket": summary.get("active_probe_observed_load_risk_bucket"),
        "active_probe_observed_risk_score": summary.get("active_probe_observed_risk_score"),
        "active_probe_payload_lift_response_m": summary.get("active_probe_payload_lift_response_m"),
        "active_probe_max_relative_offset_error_m": summary.get("active_probe_max_relative_offset_error_m"),
        "probe_adaptive_gait_enabled": summary.get("probe_adaptive_gait_enabled"),
        "probe_adaptive_gait_decision_available": summary.get("probe_adaptive_gait_decision_available"),
        "probe_adaptive_gait_decision_step": summary.get("probe_adaptive_gait_decision_step"),
        "probe_adaptive_risk_bucket": summary.get("probe_adaptive_risk_bucket"),
        "probe_adaptive_risk_score": summary.get("probe_adaptive_risk_score"),
        "probe_adaptive_gait_drive_scale": summary.get("probe_adaptive_gait_drive_scale"),
        "probe_adaptive_base_gait_drive_target_x_m": summary.get("probe_adaptive_base_gait_drive_target_x_m"),
        "probe_adaptive_effective_gait_drive_target_x_m": summary.get(
            "probe_adaptive_effective_gait_drive_target_x_m"
        ),
        "probe_adaptive_keeps_real_task_target": summary.get("probe_adaptive_keeps_real_task_target"),
        "probe_adaptive_posture_enabled": summary.get("probe_adaptive_posture_enabled"),
        "probe_adaptive_posture_decision_available": summary.get(
            "probe_adaptive_posture_decision_available"
        ),
        "probe_adaptive_posture_decision_step": summary.get("probe_adaptive_posture_decision_step"),
        "probe_adaptive_posture_strategy": summary.get("probe_adaptive_posture_strategy"),
        "probe_adaptive_posture_risk_bucket": summary.get("probe_adaptive_posture_risk_bucket"),
        "probe_adaptive_posture_risk_score": summary.get("probe_adaptive_posture_risk_score"),
        "probe_adaptive_posture_leg_target_offset_m": summary.get(
            "probe_adaptive_posture_leg_target_offset_m"
        ),
        "probe_adaptive_posture_effective_leg_target_m": summary.get(
            "probe_adaptive_posture_effective_leg_target_m"
        ),
        "payload_mass_kg": summary.get("payload_mass_kg"),
        "tray_local_x_m": summary.get("tray_local_x_m"),
        "tray_local_z_m": summary.get("tray_local_z_m"),
        "tray_size_m": summary.get("tray_size_m"),
        "root_pose_write_count": summary.get("root_pose_write_count"),
        "root_velocity_write_count": summary.get("root_velocity_write_count"),
        "root_angular_velocity_write_count": summary.get("root_angular_velocity_write_count"),
        "body_root_pose_write_count": summary.get("body_root_pose_write_count"),
        "body_root_velocity_command_count": summary.get("body_root_velocity_command_count"),
        "box_pose_write_count": summary.get("box_pose_write_count"),
        "payload_pose_write_count": summary.get("payload_pose_write_count"),
        "fall_events": summary.get("fall_events"),
        "box_drop_events": summary.get("box_drop_events"),
        "nonfinite_state_events": summary.get("nonfinite_state_events"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "max_torso_drift_xy_m": summary.get("max_torso_drift_xy_m"),
        "max_payload_drift_xy_m": summary.get("max_payload_drift_xy_m"),
        "max_torso_travel_x_m": summary.get("max_torso_travel_x_m"),
        "max_payload_travel_x_m": summary.get("max_payload_travel_x_m"),
        "min_torso_travel_x_m": summary.get("min_torso_travel_x_m"),
        "min_payload_travel_x_m": summary.get("min_payload_travel_x_m"),
        "max_abs_torso_travel_x_m": summary.get("max_abs_torso_travel_x_m"),
        "max_abs_payload_travel_x_m": summary.get("max_abs_payload_travel_x_m"),
        "max_abs_post_settle_torso_travel_x_m": summary.get("max_abs_post_settle_torso_travel_x_m"),
        "max_abs_post_settle_payload_travel_x_m": summary.get("max_abs_post_settle_payload_travel_x_m"),
        "final_target_distance_x_m": summary.get("final_target_distance_x_m"),
        "final_payload_target_distance_x_m": summary.get("final_payload_target_distance_x_m"),
        "final_post_settle_target_distance_x_m": summary.get("final_post_settle_target_distance_x_m"),
        "final_post_settle_payload_target_distance_x_m": summary.get("final_post_settle_payload_target_distance_x_m"),
        "max_joint_motion_m": summary.get("max_joint_motion_m"),
        "max_commanded_leg_lift_m": summary.get("max_commanded_leg_lift_m"),
        "max_abs_commanded_x_slide_target_m": summary.get("max_abs_commanded_x_slide_target_m"),
        "sync_inchworm_min_cycles": summary.get("sync_inchworm_min_cycles"),
        "sync_inchworm_cycle_count": summary.get("sync_inchworm_cycle_count"),
        "sync_inchworm_stride_m": summary.get("sync_inchworm_stride_m"),
        "sync_inchworm_stride_override_m": summary.get("sync_inchworm_stride_override_m"),
        "payload_local_x_m": summary.get("payload_local_x_m"),
        "payload_local_z_m": summary.get("payload_local_z_m"),
        "torso_z_m": summary.get("torso_z_m"),
        "stance_half_length_m": summary.get("stance_half_length_m"),
        "stance_half_width_m": summary.get("stance_half_width_m"),
        "final_commanded_leg_lift_m": summary.get("final_commanded_leg_lift_m"),
        "final_abs_commanded_x_slide_target_m": summary.get("final_abs_commanded_x_slide_target_m"),
        "max_actual_leg_lift_m": summary.get("max_actual_leg_lift_m"),
        "max_abs_actual_x_slide_m": summary.get("max_abs_actual_x_slide_m"),
        "final_actual_leg_lift_m": summary.get("final_actual_leg_lift_m"),
        "final_abs_actual_x_slide_m": summary.get("final_abs_actual_x_slide_m"),
        "payload_relative_distance_m": summary.get("payload_relative_distance_m"),
        "payload_relative_error_m": summary.get("payload_relative_error_m"),
        "max_payload_relative_offset_error_m": summary.get("max_payload_relative_offset_error_m"),
        "post_settle_payload_relative_error_m": summary.get("post_settle_payload_relative_error_m"),
        "max_post_settle_payload_relative_offset_error_m": summary.get(
            "max_post_settle_payload_relative_offset_error_m"
        ),
        "min_payload_z_m": summary.get("min_payload_z_m"),
        "feedback_motion_step_final": summary.get("feedback_motion_step_final"),
        "feedback_hold_steps": summary.get("feedback_hold_steps"),
        "feedback_release_steps": summary.get("feedback_release_steps"),
        "feedback_last_safe": summary.get("feedback_last_safe"),
        "feedback_last_block_reason": summary.get("feedback_last_block_reason"),
        "control_errors": summary.get("control_errors"),
        "disjoint_warning": disjoint_warning,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
