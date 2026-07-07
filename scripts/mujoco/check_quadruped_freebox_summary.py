#!/usr/bin/env python3
"""Check MuJoCo quadruped free-box contact-carry diagnostic summaries.

This reads JSON only and is safe on login nodes. Passing is still a diagnostic
gate, not final unknown-object carrying.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expect-payload-mode", default="free_body_contact_tray")
    parser.add_argument("--expect-assist-mode", default=None)
    parser.add_argument("--expect-leg-drive-mode", default=None)
    parser.add_argument("--expect-support-controller-mode", default=None)
    parser.add_argument("--min-support-joint-torque-writes", type=int, default=None)
    parser.add_argument("--require-closed-loop-foot-placement", action="store_true")
    parser.add_argument("--require-hold-capture-point-foot-placement", action="store_true")
    parser.add_argument("--min-hold-capture-active-steps", type=int, default=None)
    parser.add_argument("--max-fall-events", type=int, default=None)
    parser.add_argument("--max-box-drop-events", type=int, default=None)
    parser.add_argument("--min-box-travel-x", type=float, default=None)
    parser.add_argument("--min-final-box-travel-x", type=float, default=None)
    parser.add_argument("--max-tilt", type=float, default=None)
    parser.add_argument("--min-box-z", type=float, default=None)
    parser.add_argument("--max-relative-offset-error", type=float, default=None)
    parser.add_argument("--max-final-relative-offset-error", type=float, default=None)
    parser.add_argument("--max-root-pose-writes", type=int, default=None)
    parser.add_argument("--max-root-velocity-writes", type=int, default=None)
    parser.add_argument("--max-box-pose-writes", type=int, default=None)
    parser.add_argument("--max-box-velocity-writes", type=int, default=None)
    parser.add_argument("--min-external-force-writes", type=int, default=None)
    parser.add_argument("--expect-retention-force-mode", default=None)
    parser.add_argument("--min-retention-force-writes", type=int, default=None)
    parser.add_argument("--require-target-stop-latched", action="store_true")
    parser.add_argument("--min-target-stop-hold-steps", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    if int(summary.get("completed_steps", 0)) < int(summary.get("steps_requested", 0)):
        failures.append(f"incomplete steps: {summary.get('completed_steps')} / {summary.get('steps_requested')}")
    if args.expect_payload_mode is not None and summary.get("payload_mode") != args.expect_payload_mode:
        failures.append(f"payload mode mismatch: {summary.get('payload_mode')} != {args.expect_payload_mode}")
    if args.expect_assist_mode is not None and summary.get("assist_mode") != args.expect_assist_mode:
        failures.append(f"assist mode mismatch: {summary.get('assist_mode')} != {args.expect_assist_mode}")
    if args.expect_leg_drive_mode is not None and summary.get("leg_drive_mode") != args.expect_leg_drive_mode:
        failures.append(f"leg drive mode mismatch: {summary.get('leg_drive_mode')} != {args.expect_leg_drive_mode}")
    if args.expect_support_controller_mode is not None and summary.get("support_controller_mode") != args.expect_support_controller_mode:
        failures.append(
            f"support controller mode mismatch: {summary.get('support_controller_mode')} != {args.expect_support_controller_mode}"
        )
    if args.min_support_joint_torque_writes is not None and int(summary.get("support_joint_torque_write_count", 0)) < args.min_support_joint_torque_writes:
        failures.append(f"support joint torque writes too low: {summary.get('support_joint_torque_write_count')}")
    if args.require_closed_loop_foot_placement and not bool(summary.get("closed_loop_foot_placement", False)):
        failures.append("closed-loop foot placement was not enabled")
    if args.require_hold_capture_point_foot_placement and not bool(
        summary.get("hold_capture_point_foot_placement", False)
    ):
        failures.append("hold capture-point foot placement was not enabled")
    if args.min_hold_capture_active_steps is not None:
        value = int(summary.get("hold_capture_active_steps", 0))
        if value < int(args.min_hold_capture_active_steps):
            failures.append(f"hold capture active steps too low: {value}")
    if args.max_fall_events is not None and int(summary.get("fall_events", 0)) > args.max_fall_events:
        failures.append(f"fall events too high: {summary.get('fall_events')}")
    if args.max_box_drop_events is not None and int(summary.get("box_drop_events", 0)) > args.max_box_drop_events:
        failures.append(f"box drop events too high: {summary.get('box_drop_events')}")
    if args.min_box_travel_x is not None and float(summary.get("max_box_travel_x_m", 0.0)) < args.min_box_travel_x:
        failures.append(f"box x travel too low: {summary.get('max_box_travel_x_m')}")
    if args.min_final_box_travel_x is not None and float(summary.get("final_box_travel_x_m", 0.0)) < args.min_final_box_travel_x:
        failures.append(f"final box x travel too low: {summary.get('final_box_travel_x_m')}")
    if args.max_tilt is not None and float(summary.get("max_tilt_rad", 0.0)) > args.max_tilt:
        failures.append(f"tilt too high: {summary.get('max_tilt_rad')}")
    if args.min_box_z is not None and float(summary.get("min_box_z_m", 0.0)) < args.min_box_z:
        failures.append(f"box z too low: {summary.get('min_box_z_m')}")
    if args.max_relative_offset_error is not None and float(summary.get("max_box_torso_relative_offset_error_m", 0.0)) > args.max_relative_offset_error:
        failures.append(f"relative offset error too high: {summary.get('max_box_torso_relative_offset_error_m')}")
    if args.max_final_relative_offset_error is not None and float(summary.get("final_box_torso_relative_offset_error_m", 0.0)) > args.max_final_relative_offset_error:
        failures.append(f"final relative offset error too high: {summary.get('final_box_torso_relative_offset_error_m')}")
    if args.max_root_pose_writes is not None and int(summary.get("root_pose_write_count", 0)) > args.max_root_pose_writes:
        failures.append(f"root pose writes too high: {summary.get('root_pose_write_count')}")
    if args.max_root_velocity_writes is not None and int(summary.get("root_velocity_write_count", 0)) > args.max_root_velocity_writes:
        failures.append(f"root velocity writes too high: {summary.get('root_velocity_write_count')}")
    if args.max_box_pose_writes is not None and int(summary.get("box_pose_write_count", 0)) > args.max_box_pose_writes:
        failures.append(f"box pose writes too high: {summary.get('box_pose_write_count')}")
    if args.max_box_velocity_writes is not None and int(summary.get("box_velocity_write_count", 0)) > args.max_box_velocity_writes:
        failures.append(f"box velocity writes too high: {summary.get('box_velocity_write_count')}")
    if args.min_external_force_writes is not None and int(summary.get("external_force_write_count", 0)) < args.min_external_force_writes:
        failures.append(f"external force writes too low: {summary.get('external_force_write_count')}")
    if args.expect_retention_force_mode is not None and summary.get("box_retention_force_mode") != args.expect_retention_force_mode:
        failures.append(
            f"retention force mode mismatch: {summary.get('box_retention_force_mode')} != {args.expect_retention_force_mode}"
        )
    if args.min_retention_force_writes is not None and int(summary.get("box_retention_force_write_count", 0)) < args.min_retention_force_writes:
        failures.append(f"retention force writes too low: {summary.get('box_retention_force_write_count')}")
    if args.require_target_stop_latched and not bool(summary.get("target_stop_latched", False)):
        failures.append("target stop was not latched")
    if args.min_target_stop_hold_steps is not None and int(summary.get("target_stop_hold_steps", 0)) < args.min_target_stop_hold_steps:
        failures.append(f"target stop hold steps too low: {summary.get('target_stop_hold_steps')}")

    report = {
        "summary": str(args.summary),
        "scene_type": summary.get("scene_type"),
        "success_claim": summary.get("success_claim"),
        "payload_mode": summary.get("payload_mode"),
        "box_mass_kg": summary.get("box_mass_kg"),
        "assist_mode": summary.get("assist_mode"),
        "actuator_kp": summary.get("actuator_kp"),
        "actuator_kv": summary.get("actuator_kv"),
        "leg_drive_mode": summary.get("leg_drive_mode"),
        "support_controller_mode": summary.get("support_controller_mode"),
        "support_joint_torque_write_count": summary.get("support_joint_torque_write_count"),
        "closed_loop_foot_placement": summary.get("closed_loop_foot_placement"),
        "hold_capture_point_foot_placement": summary.get("hold_capture_point_foot_placement"),
        "hold_capture_active_steps": summary.get("hold_capture_active_steps"),
        "max_abs_hold_capture_x_adjust_m": summary.get("max_abs_hold_capture_x_adjust_m"),
        "max_abs_hold_capture_y_signal_m": summary.get("max_abs_hold_capture_y_signal_m"),
        "external_stabilizer_enabled": summary.get("external_stabilizer_enabled"),
        "root_pose_write_count": summary.get("root_pose_write_count"),
        "root_velocity_write_count": summary.get("root_velocity_write_count"),
        "box_pose_write_count": summary.get("box_pose_write_count"),
        "box_velocity_write_count": summary.get("box_velocity_write_count"),
        "external_force_write_count": summary.get("external_force_write_count"),
        "external_torque_write_count": summary.get("external_torque_write_count"),
        "box_retention_force_mode": summary.get("box_retention_force_mode"),
        "box_retention_force_write_count": summary.get("box_retention_force_write_count"),
        "box_retention_equal_opposite_force": summary.get("box_retention_equal_opposite_force"),
        "target_stop_latched": summary.get("target_stop_latched"),
        "target_stop_step": summary.get("target_stop_step"),
        "target_stop_hold_steps": summary.get("target_stop_hold_steps"),
        "completed_steps": summary.get("completed_steps"),
        "steps_requested": summary.get("steps_requested"),
        "max_box_travel_x_m": summary.get("max_box_travel_x_m"),
        "final_box_travel_x_m": summary.get("final_box_travel_x_m"),
        "min_box_z_m": summary.get("min_box_z_m"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "fall_events": summary.get("fall_events"),
        "box_drop_events": summary.get("box_drop_events"),
        "max_box_torso_relative_offset_error_m": summary.get("max_box_torso_relative_offset_error_m"),
        "final_box_torso_relative_offset_error_m": summary.get("final_box_torso_relative_offset_error_m"),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
