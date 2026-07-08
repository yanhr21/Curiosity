#!/usr/bin/env python3
"""Check custom articulated quadruped carry diagnostic summaries.

This checker reads only JSON/log text, so it is safe on the login node.  It is
for diagnostic gates only; passing it is not a final locomotion or carrying
success claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dynamic quadruped carry summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--expect-payload-mode", default=None)
    parser.add_argument("--expect-staged-attach-mode", default=None)
    parser.add_argument("--expect-base-assist-mode", default=None)
    parser.add_argument("--require-attach", action="store_true")
    parser.add_argument("--forbid-disjoint-warning", action="store_true")
    parser.add_argument("--max-fall-events", type=int, default=0)
    parser.add_argument("--max-box-drop-events", type=int, default=0)
    parser.add_argument("--min-target-hold-steps", type=int, default=None)
    parser.add_argument("--max-target-distance", type=float, default=None)
    parser.add_argument("--max-relative-error", type=float, default=None)
    parser.add_argument("--max-peak-relative-error", type=float, default=None)
    parser.add_argument("--min-torso-travel", type=float, default=None)
    parser.add_argument("--min-box-travel", type=float, default=None)
    parser.add_argument("--max-tilt", type=float, default=None)
    parser.add_argument("--min-joint-motion", type=float, default=None)
    parser.add_argument("--max-control-errors", type=int, default=None)
    parser.add_argument("--require-contact-proxy", action="store_true")
    parser.add_argument("--max-contact-proxy-gap", type=float, default=None)
    parser.add_argument("--max-root-pose-writes", type=int, default=None)
    parser.add_argument("--max-root-velocity-writes", type=int, default=None)
    parser.add_argument("--max-root-angular-velocity-writes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    if int(summary.get("completed_steps", 0)) < int(summary.get("steps_requested", 0)):
        failures.append(f"incomplete steps: {summary.get('completed_steps')} / {summary.get('steps_requested')}")
    if args.expect_payload_mode is not None and summary.get("payload_mode") != args.expect_payload_mode:
        failures.append(f"payload mode mismatch: {summary.get('payload_mode')} != {args.expect_payload_mode}")
    if args.expect_staged_attach_mode is not None and summary.get("staged_attach_mode") != args.expect_staged_attach_mode:
        failures.append(
            f"staged attach mode mismatch: {summary.get('staged_attach_mode')} != {args.expect_staged_attach_mode}"
        )
    if args.expect_base_assist_mode is not None and summary.get("base_assist_mode") != args.expect_base_assist_mode:
        failures.append(f"base assist mode mismatch: {summary.get('base_assist_mode')} != {args.expect_base_assist_mode}")
    if args.require_attach and not bool(summary.get("attached")):
        failures.append("attach missing")
    if args.require_attach and summary.get("attach_step") is None:
        failures.append("attach_step missing")
    if int(summary.get("fall_events", 0)) > args.max_fall_events:
        failures.append(f"fall events too high: {summary.get('fall_events')}")
    if int(summary.get("box_drop_events", 0)) > args.max_box_drop_events:
        failures.append(f"box drop events too high: {summary.get('box_drop_events')}")
    if args.min_target_hold_steps is not None and int(summary.get("target_hold_steps", 0)) < args.min_target_hold_steps:
        failures.append(f"target hold steps too low: {summary.get('target_hold_steps')}")
    if args.max_target_distance is not None:
        target_distance = summary.get("final_box_target_distance_xy_m")
        if target_distance is None or float(target_distance) > args.max_target_distance:
            failures.append(f"target distance too high: {target_distance}")
    if args.max_relative_error is not None:
        rel_error = summary.get("box_relative_error_m_after_attach")
        if rel_error is None or float(rel_error) > args.max_relative_error:
            failures.append(f"relative error too high: {rel_error}")
    if args.max_peak_relative_error is not None:
        peak_error = summary.get("max_box_relative_error_m_after_attach")
        if peak_error is None or float(peak_error) > args.max_peak_relative_error:
            failures.append(f"peak relative error too high: {peak_error}")
    if args.min_torso_travel is not None and float(summary.get("max_torso_travel_xy_m", 0.0)) < args.min_torso_travel:
        failures.append(f"torso travel too low: {summary.get('max_torso_travel_xy_m')}")
    if args.min_box_travel is not None and float(summary.get("max_box_travel_xy_m", 0.0)) < args.min_box_travel:
        failures.append(f"box travel too low: {summary.get('max_box_travel_xy_m')}")
    if args.max_tilt is not None and float(summary.get("max_tilt_rad", 0.0)) > args.max_tilt:
        failures.append(f"tilt too high: {summary.get('max_tilt_rad')}")
    if args.min_joint_motion is not None:
        joint_motion = summary.get("max_joint_motion_rad")
        if joint_motion is None or float(joint_motion) < args.min_joint_motion:
            failures.append(f"joint motion too low: {joint_motion}")
    if args.max_control_errors is not None:
        control_errors = summary.get("control_errors") or []
        if len(control_errors) > args.max_control_errors:
            failures.append(f"control errors too many: {control_errors}")
    if args.require_contact_proxy and not bool(summary.get("contact_proxy_enabled")):
        failures.append(f"contact proxy not enabled: {summary.get('contact_proxy_enabled')}")
    if args.max_contact_proxy_gap is not None:
        proxy_gap = summary.get("max_contact_proxy_grip_gap_m")
        if proxy_gap is None or float(proxy_gap) > args.max_contact_proxy_gap:
            failures.append(f"contact proxy gap too high: {proxy_gap}")
    if args.max_root_pose_writes is not None:
        root_pose_writes = int(summary.get("root_pose_write_count", 0))
        if root_pose_writes > args.max_root_pose_writes:
            failures.append(f"root pose writes too high: {root_pose_writes}")
    if args.max_root_velocity_writes is not None:
        root_velocity_writes = int(summary.get("root_velocity_write_count", 0))
        if root_velocity_writes > args.max_root_velocity_writes:
            failures.append(f"root velocity writes too high: {root_velocity_writes}")
    if args.max_root_angular_velocity_writes is not None:
        root_angular_velocity_writes = int(summary.get("root_angular_velocity_write_count", 0))
        if root_angular_velocity_writes > args.max_root_angular_velocity_writes:
            failures.append(f"root angular velocity writes too high: {root_angular_velocity_writes}")

    disjoint_warning = False
    if args.log is not None and args.log.exists():
        log_text = args.log.read_text(errors="ignore")
        disjoint_warning = "disjointed body transforms" in log_text or "disjoint body transforms" in log_text
        if args.forbid_disjoint_warning and disjoint_warning:
            failures.append("disjoint fixed-joint warning present in log")

    report = {
        "summary": str(args.summary),
        "log": None if args.log is None else str(args.log),
        "completed_steps": summary.get("completed_steps"),
        "steps_requested": summary.get("steps_requested"),
        "payload_mode": summary.get("payload_mode"),
        "payload_mass_kg": summary.get("payload_mass_kg"),
        "staged_attach_mode": summary.get("staged_attach_mode"),
        "base_assist_mode": summary.get("base_assist_mode"),
        "root_pose_write_enabled": summary.get("root_pose_write_enabled"),
        "root_velocity_write_enabled": summary.get("root_velocity_write_enabled"),
        "root_angular_velocity_write_enabled": summary.get("root_angular_velocity_write_enabled"),
        "base_post_step_velocity_assist": summary.get("base_post_step_velocity_assist"),
        "support_drive_enabled": summary.get("support_drive_enabled"),
        "support_drive_claim": summary.get("support_drive_claim"),
        "support_drive_gain": summary.get("support_drive_gain"),
        "support_drive_max_speed": summary.get("support_drive_max_speed"),
        "support_pad_pose_write_count": summary.get("support_pad_pose_write_count"),
        "support_pad_velocity_write_count": summary.get("support_pad_velocity_write_count"),
        "root_pose_write_count": summary.get("root_pose_write_count"),
        "root_velocity_write_count": summary.get("root_velocity_write_count"),
        "root_angular_velocity_write_count": summary.get("root_angular_velocity_write_count"),
        "base_x_gain": summary.get("base_x_gain"),
        "base_max_x_speed": summary.get("base_max_x_speed"),
        "base_x_command_scale": summary.get("base_x_command_scale"),
        "base_lateral_gain": summary.get("base_lateral_gain"),
        "base_height_gain": summary.get("base_height_gain"),
        "base_max_z_speed": summary.get("base_max_z_speed"),
        "base_upright_gain": summary.get("base_upright_gain"),
        "base_max_angular_speed": summary.get("base_max_angular_speed"),
        "attached": summary.get("attached"),
        "attach_step": summary.get("attach_step"),
        "target_hold_steps": summary.get("target_hold_steps"),
        "target_hold_latched": summary.get("target_hold_latched"),
        "target_hold_radius_m": summary.get("target_hold_radius_m"),
        "gait_frequency_hz": summary.get("gait_frequency_hz"),
        "hip_neutral_deg": summary.get("hip_neutral_deg"),
        "knee_neutral_deg": summary.get("knee_neutral_deg"),
        "hip_amplitude_deg": summary.get("hip_amplitude_deg"),
        "knee_amplitude_deg": summary.get("knee_amplitude_deg"),
        "torso_z_m": summary.get("torso_z_m"),
        "stance_half_length_m": summary.get("stance_half_length_m"),
        "stance_half_width_m": summary.get("stance_half_width_m"),
        "foot_length_m": summary.get("foot_length_m"),
        "foot_width_m": summary.get("foot_width_m"),
        "foot_height_m": summary.get("foot_height_m"),
        "static_friction": summary.get("static_friction"),
        "dynamic_friction": summary.get("dynamic_friction"),
        "hip_stiffness": summary.get("hip_stiffness"),
        "hip_damping": summary.get("hip_damping"),
        "hip_max_force": summary.get("hip_max_force"),
        "knee_stiffness": summary.get("knee_stiffness"),
        "knee_damping": summary.get("knee_damping"),
        "knee_max_force": summary.get("knee_max_force"),
        "target_body_margin_m": summary.get("target_body_margin_m"),
        "min_hold_torso_travel_m": summary.get("min_hold_torso_travel_m"),
        "target_body_x_m": summary.get("target_body_x_m"),
        "target_hold_body_ready": summary.get("target_hold_body_ready"),
        "fall_events": summary.get("fall_events"),
        "box_drop_events": summary.get("box_drop_events"),
        "max_torso_travel_xy_m": summary.get("max_torso_travel_xy_m"),
        "max_box_travel_xy_m": summary.get("max_box_travel_xy_m"),
        "final_box_target_distance_xy_m": summary.get("final_box_target_distance_xy_m"),
        "box_relative_error_m_after_attach": summary.get("box_relative_error_m_after_attach"),
        "max_box_relative_error_m_after_attach": summary.get("max_box_relative_error_m_after_attach"),
        "max_tilt_rad": summary.get("max_tilt_rad"),
        "max_joint_motion_rad": summary.get("max_joint_motion_rad"),
        "control_errors": summary.get("control_errors"),
        "carry_local_x_m": summary.get("carry_local_x_m"),
        "carry_local_z_m": summary.get("carry_local_z_m"),
        "contact_proxy_gain": summary.get("contact_proxy_gain"),
        "contact_proxy_max_speed": summary.get("contact_proxy_max_speed"),
        "contact_proxy_enabled": summary.get("contact_proxy_enabled"),
        "contact_proxy_grip_gap_m": summary.get("contact_proxy_grip_gap_m"),
        "max_contact_proxy_grip_gap_m": summary.get("max_contact_proxy_grip_gap_m"),
        "disjoint_warning": disjoint_warning,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
