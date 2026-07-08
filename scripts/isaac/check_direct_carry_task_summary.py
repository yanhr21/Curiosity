#!/usr/bin/env python3
"""Check the direct Isaac carry-task interface summary.

This checker is intentionally strict about success semantics.  The direct
task scene is allowed to be a kinematic proxy interface, but it must report the
proxy pose writes that make it ineligible as robot-carrying evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check direct carry-task interface summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--expect-controller-mode", default="kinematic_proxy")
    parser.add_argument("--expect-carry-posture", default=None)
    parser.add_argument("--expect-backend-support-mode", default=None)
    parser.add_argument("--require-box-randomized", action="store_true")
    parser.add_argument("--expect-box-seed", type=int, default=None)
    parser.add_argument("--min-probe-steps", type=int, default=None)
    parser.add_argument("--require-probe-belief", action="store_true")
    parser.add_argument("--forbid-probe-hidden-ground-truth", action="store_true")
    parser.add_argument("--min-probe-box-travel-x", type=float, default=None)
    parser.add_argument("--min-box-travel", type=float, default=None)
    parser.add_argument("--min-abs-box-travel-x", type=float, default=None)
    parser.add_argument("--min-post-settle-box-travel-x", type=float, default=None)
    parser.add_argument("--min-abs-post-settle-box-travel-x", type=float, default=None)
    parser.add_argument("--max-final-post-settle-box-target-distance-x", type=float, default=None)
    parser.add_argument("--max-final-box-target-distance-x", type=float, default=None)
    parser.add_argument("--max-post-settle-box-travel-loss-after-peak", type=float, default=None)
    parser.add_argument("--max-fall-events", type=int, default=None)
    parser.add_argument("--max-box-drop-events", type=int, default=0)
    parser.add_argument("--require-kinematic-proxy-writes", action="store_true")
    parser.add_argument("--require-root-shortcut-free", action="store_true")
    parser.add_argument("--max-anchor-world-joint-retarget-count", type=int, default=None)
    parser.add_argument("--max-support-root-pose-write-count", type=int, default=None)
    parser.add_argument("--max-foot-pose-write-count", type=int, default=None)
    parser.add_argument("--max-stance-anchor-pose-write-count", type=int, default=None)
    parser.add_argument("--expect-support-foot-mode", default=None)
    parser.add_argument("--expect-support-foot-placement-mode", default=None)
    parser.add_argument("--require-support-foot-placement-controller", action="store_true")
    parser.add_argument("--require-directional-foot-placement", action="store_true")
    parser.add_argument("--require-feedback-step-controller", action="store_true")
    parser.add_argument("--min-feedback-step-applied-steps", type=int, default=None)
    parser.add_argument("--require-online-probe-adaptive-support", action="store_true")
    parser.add_argument("--expect-online-probe-adaptive-support-bucket", default=None)
    parser.add_argument("--expect-online-probe-adaptive-support-profile", default=None)
    parser.add_argument("--expect-online-probe-adaptive-support-step-height", type=float, default=None)
    parser.add_argument("--expect-online-probe-adaptive-support-double-support", type=float, default=None)
    parser.add_argument("--forbid-online-probe-adaptive-hidden-ground-truth", action="store_true")
    parser.add_argument("--require-online-probe-adaptive-hold", action="store_true")
    parser.add_argument("--require-online-probe-adaptive-hold-actuated", action="store_true")
    parser.add_argument("--require-online-probe-adaptive-hold-collision-available", action="store_true")
    parser.add_argument("--expect-online-probe-adaptive-hold-collision-enabled", type=int, choices=(0, 1), default=None)
    parser.add_argument("--min-online-probe-adaptive-hold-collision-updates", type=int, default=None)
    parser.add_argument("--expect-online-probe-adaptive-hold-bucket", default=None)
    parser.add_argument("--expect-online-probe-adaptive-hold-profile", default=None)
    parser.add_argument("--expect-online-probe-adaptive-hold-closure-fraction", type=float, default=None)
    parser.add_argument("--forbid-online-probe-adaptive-hold-hidden-ground-truth", action="store_true")
    parser.add_argument("--max-rail-joint-motion", type=float, default=None)
    parser.add_argument("--min-clamp-joint-motion", type=float, default=None)
    parser.add_argument("--min-cradle-joint-motion", type=float, default=None)
    parser.add_argument("--min-support-foot-joint-count", type=int, default=None)
    parser.add_argument("--min-support-foot-x-joint-motion", type=float, default=None)
    parser.add_argument("--min-support-foot-z-joint-count", type=int, default=None)
    parser.add_argument("--min-support-foot-z-joint-motion", type=float, default=None)
    parser.add_argument("--min-actual-support-foot-lift", type=float, default=None)
    parser.add_argument("--min-near-ground-foot-count", type=int, default=None)
    parser.add_argument("--min-drive-near-ground-foot-count", type=int, default=None)
    parser.add_argument("--max-drive-near-ground-zero-steps", type=int, default=None)
    parser.add_argument("--max-drive-near-ground-lt2-steps", type=int, default=None)
    parser.add_argument("--require-support-foot-contact-report-evidence", action="store_true")
    parser.add_argument("--min-drive-contact-report-foot-count", type=int, default=None)
    parser.add_argument("--max-drive-contact-report-zero-steps", type=int, default=None)
    parser.add_argument("--max-drive-contact-report-lt2-steps", type=int, default=None)
    parser.add_argument("--min-commanded-stance-contact-report-foot-count", type=int, default=None)
    parser.add_argument("--max-commanded-stance-contact-report-lt2-steps", type=int, default=None)
    parser.add_argument("--require-support-foot-effort-evidence", action="store_true")
    parser.add_argument("--min-drive-effort-supported-foot-count", type=int, default=None)
    parser.add_argument("--max-drive-effort-supported-zero-steps", type=int, default=None)
    parser.add_argument("--max-drive-effort-supported-lt2-steps", type=int, default=None)
    parser.add_argument("--min-commanded-stance-effort-supported-foot-count", type=int, default=None)
    parser.add_argument("--max-commanded-stance-effort-supported-lt2-steps", type=int, default=None)
    parser.add_argument("--min-commanded-stance-near-ground-foot-count", type=int, default=None)
    parser.add_argument("--max-commanded-stance-near-ground-lt2-steps", type=int, default=None)
    parser.add_argument("--min-support-polygon-margin", type=float, default=None)
    parser.add_argument("--max-near-ground-foot-speed", type=float, default=None)
    parser.add_argument("--max-near-ground-foot-slip", type=float, default=None)
    parser.add_argument("--require-stance-foot-world-lock", action="store_true")
    parser.add_argument("--min-stance-foot-world-lock-switches", type=int, default=None)
    parser.add_argument("--require-freeze-locked-stance-foot-targets", action="store_true")
    parser.add_argument("--require-freeze-commanded-stance-foot-targets", action="store_true")
    parser.add_argument("--require-planted-stance-rail-propulsion", action="store_true")
    parser.add_argument("--max-abs-anchor-travel-x", type=float, default=None)
    parser.add_argument("--max-abs-support-foot-travel-x", type=float, default=None)
    parser.add_argument("--forbid-fixed-world-support", action="store_true")
    parser.add_argument("--require-non-success-claim", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    completed_steps = int(summary.get("completed_steps", 0))
    if completed_steps < int(args.min_steps):
        failures.append(f"completed_steps too low: {completed_steps} < {args.min_steps}")
    if args.expect_controller_mode is not None and summary.get("controller_mode") != args.expect_controller_mode:
        failures.append(f"controller mode mismatch: {summary.get('controller_mode')} != {args.expect_controller_mode}")
    if args.expect_carry_posture is not None and summary.get("carry_posture") != args.expect_carry_posture:
        failures.append(f"carry posture mismatch: {summary.get('carry_posture')} != {args.expect_carry_posture}")
    if args.expect_backend_support_mode is not None and summary.get("backend_support_mode") != args.expect_backend_support_mode:
        failures.append(
            f"backend support mode mismatch: {summary.get('backend_support_mode')} != {args.expect_backend_support_mode}"
        )
    if args.require_box_randomized and not bool(summary.get("box_randomized")):
        failures.append(f"box_randomized is false or missing: {summary.get('box_randomized')}")
    if args.expect_box_seed is not None:
        box_seed = summary.get("box_seed")
        if box_seed is None or int(box_seed) != int(args.expect_box_seed):
            failures.append(f"box seed mismatch: {box_seed} != {args.expect_box_seed}")
    if args.min_probe_steps is not None:
        probe_steps = int(summary.get("probe_steps_requested") or 0)
        if probe_steps < int(args.min_probe_steps):
            failures.append(f"probe_steps_requested too low: {probe_steps} < {args.min_probe_steps}")
    if args.require_probe_belief and not bool(summary.get("probe_belief_available")):
        failures.append(f"probe_belief_available is false or missing: {summary.get('probe_belief_available')}")
    if args.forbid_probe_hidden_ground_truth and bool(summary.get("probe_belief_uses_hidden_ground_truth")):
        failures.append("probe_belief_uses_hidden_ground_truth is true")
    if args.min_probe_box_travel_x is not None:
        probe_box_travel = summary.get("max_probe_box_travel_x_m")
        if probe_box_travel is None or float(probe_box_travel) < float(args.min_probe_box_travel_x):
            failures.append(
                f"max_probe_box_travel_x_m too low: {probe_box_travel} < {args.min_probe_box_travel_x}"
            )
    if args.expect_support_foot_mode is not None and summary.get("support_foot_mode") != args.expect_support_foot_mode:
        failures.append(f"support foot mode mismatch: {summary.get('support_foot_mode')} != {args.expect_support_foot_mode}")
    if (
        args.expect_support_foot_placement_mode is not None
        and summary.get("support_foot_placement_mode") != args.expect_support_foot_placement_mode
    ):
        failures.append(
            "support foot placement mode mismatch: "
            f"{summary.get('support_foot_placement_mode')} != {args.expect_support_foot_placement_mode}"
        )
    if args.require_support_foot_placement_controller and not bool(
        summary.get("support_foot_placement_controller_enabled")
    ):
        failures.append(
            "support_foot_placement_controller_enabled is false or missing: "
            f"{summary.get('support_foot_placement_controller_enabled')}"
        )
    if args.require_directional_foot_placement and not bool(summary.get("support_foot_directional_placement")):
        failures.append(
            "support_foot_directional_placement is false or missing: "
            f"{summary.get('support_foot_directional_placement')}"
        )
    if args.require_feedback_step_controller and not bool(summary.get("feedback_step_controller_enabled")):
        failures.append(
            f"feedback_step_controller_enabled is false or missing: "
            f"{summary.get('feedback_step_controller_enabled')}"
        )
    if args.min_feedback_step_applied_steps is not None:
        applied_steps = int(summary.get("feedback_step_applied_steps") or 0)
        if applied_steps < int(args.min_feedback_step_applied_steps):
            failures.append(
                f"feedback_step_applied_steps too low: "
                f"{applied_steps} < {args.min_feedback_step_applied_steps}"
            )
    if args.require_online_probe_adaptive_support:
        if not bool(summary.get("online_probe_adaptive_support_enabled")):
            failures.append(
                "online_probe_adaptive_support_enabled is false or missing: "
                f"{summary.get('online_probe_adaptive_support_enabled')}"
            )
        if not bool(summary.get("online_probe_adaptive_support_decision_applied")):
            failures.append(
                "online_probe_adaptive_support_decision_applied is false or missing: "
                f"{summary.get('online_probe_adaptive_support_decision_applied')}"
            )
        if summary.get("online_probe_adaptive_support_decision_step") is None:
            failures.append("online_probe_adaptive_support_decision_step is missing")
        if not bool(summary.get("probe_belief_policy_action_applied")):
            failures.append(
                "probe_belief_policy_action_applied is false or missing: "
                f"{summary.get('probe_belief_policy_action_applied')}"
            )
    if args.forbid_online_probe_adaptive_hidden_ground_truth and bool(
        summary.get("online_probe_adaptive_support_uses_hidden_ground_truth")
    ):
        failures.append("online_probe_adaptive_support_uses_hidden_ground_truth is true")
    if (
        args.expect_online_probe_adaptive_support_bucket is not None
        and summary.get("online_probe_adaptive_support_risk_bucket")
        != args.expect_online_probe_adaptive_support_bucket
    ):
        failures.append(
            "online support bucket mismatch: "
            f"{summary.get('online_probe_adaptive_support_risk_bucket')} "
            f"!= {args.expect_online_probe_adaptive_support_bucket}"
        )
    if (
        args.expect_online_probe_adaptive_support_profile is not None
        and summary.get("online_probe_adaptive_support_profile")
        != args.expect_online_probe_adaptive_support_profile
    ):
        failures.append(
            "online support profile mismatch: "
            f"{summary.get('online_probe_adaptive_support_profile')} "
            f"!= {args.expect_online_probe_adaptive_support_profile}"
        )
    if args.expect_online_probe_adaptive_support_step_height is not None:
        actual = summary.get("online_probe_adaptive_support_step_height_m")
        if actual is None or abs(float(actual) - float(args.expect_online_probe_adaptive_support_step_height)) > 1e-6:
            failures.append(
                "online support step height mismatch: "
                f"{actual} != {args.expect_online_probe_adaptive_support_step_height}"
            )
    if args.expect_online_probe_adaptive_support_double_support is not None:
        actual = summary.get("online_probe_adaptive_support_double_support_fraction")
        if actual is None or abs(float(actual) - float(args.expect_online_probe_adaptive_support_double_support)) > 1e-6:
            failures.append(
                "online support double support mismatch: "
                f"{actual} != {args.expect_online_probe_adaptive_support_double_support}"
            )
    if args.require_online_probe_adaptive_hold:
        if not bool(summary.get("online_probe_adaptive_hold_enabled")):
            failures.append(
                "online_probe_adaptive_hold_enabled is false or missing: "
                f"{summary.get('online_probe_adaptive_hold_enabled')}"
            )
        if not bool(summary.get("online_probe_adaptive_hold_decision_applied")):
            failures.append(
                "online_probe_adaptive_hold_decision_applied is false or missing: "
                f"{summary.get('online_probe_adaptive_hold_decision_applied')}"
            )
        if summary.get("online_probe_adaptive_hold_decision_step") is None:
            failures.append("online_probe_adaptive_hold_decision_step is missing")
        if not bool(summary.get("probe_belief_policy_action_applied")):
            failures.append(
                "probe_belief_policy_action_applied is false or missing for online hold: "
                f"{summary.get('probe_belief_policy_action_applied')}"
            )
    if args.require_online_probe_adaptive_hold_actuated and not bool(
        summary.get("online_probe_adaptive_hold_actuated")
    ):
        failures.append(
            "online_probe_adaptive_hold_actuated is false or missing: "
            f"{summary.get('online_probe_adaptive_hold_actuated')}"
        )
    if args.require_online_probe_adaptive_hold_collision_available and not bool(
        summary.get("online_probe_adaptive_hold_collision_available")
    ):
        failures.append(
            "online_probe_adaptive_hold_collision_available is false or missing: "
            f"{summary.get('online_probe_adaptive_hold_collision_available')}"
        )
    if args.expect_online_probe_adaptive_hold_collision_enabled is not None:
        expected_enabled = bool(args.expect_online_probe_adaptive_hold_collision_enabled)
        if bool(summary.get("online_probe_adaptive_hold_collision_enabled")) != expected_enabled:
            failures.append(
                "online hold collision enabled mismatch: "
                f"{summary.get('online_probe_adaptive_hold_collision_enabled')} != {expected_enabled}"
            )
    if args.min_online_probe_adaptive_hold_collision_updates is not None:
        update_count = int(summary.get("online_probe_adaptive_hold_collision_update_count") or 0)
        if update_count < int(args.min_online_probe_adaptive_hold_collision_updates):
            failures.append(
                "online_probe_adaptive_hold_collision_update_count too low: "
                f"{update_count} < {args.min_online_probe_adaptive_hold_collision_updates}"
            )
    if args.forbid_online_probe_adaptive_hold_hidden_ground_truth and bool(
        summary.get("online_probe_adaptive_hold_uses_hidden_ground_truth")
    ):
        failures.append("online_probe_adaptive_hold_uses_hidden_ground_truth is true")
    if (
        args.expect_online_probe_adaptive_hold_bucket is not None
        and summary.get("online_probe_adaptive_hold_risk_bucket")
        != args.expect_online_probe_adaptive_hold_bucket
    ):
        failures.append(
            "online hold bucket mismatch: "
            f"{summary.get('online_probe_adaptive_hold_risk_bucket')} "
            f"!= {args.expect_online_probe_adaptive_hold_bucket}"
        )
    if (
        args.expect_online_probe_adaptive_hold_profile is not None
        and summary.get("online_probe_adaptive_hold_profile")
        != args.expect_online_probe_adaptive_hold_profile
    ):
        failures.append(
            "online hold profile mismatch: "
            f"{summary.get('online_probe_adaptive_hold_profile')} "
            f"!= {args.expect_online_probe_adaptive_hold_profile}"
        )
    if args.expect_online_probe_adaptive_hold_closure_fraction is not None:
        actual = summary.get("online_probe_adaptive_hold_closure_fraction")
        if actual is None or abs(float(actual) - float(args.expect_online_probe_adaptive_hold_closure_fraction)) > 1e-6:
            failures.append(
                "online hold closure fraction mismatch: "
                f"{actual} != {args.expect_online_probe_adaptive_hold_closure_fraction}"
            )
    if args.max_rail_joint_motion is not None:
        rail_motion = summary.get("max_rail_joint_motion_m", summary.get("max_joint_motion_m"))
        if rail_motion is None or float(rail_motion) > float(args.max_rail_joint_motion):
            failures.append(f"max_rail_joint_motion_m too high: {rail_motion} > {args.max_rail_joint_motion}")
    if args.min_clamp_joint_motion is not None:
        clamp_motion = summary.get("max_clamp_joint_motion_m")
        if clamp_motion is None or float(clamp_motion) < float(args.min_clamp_joint_motion):
            failures.append(f"max_clamp_joint_motion_m too low: {clamp_motion} < {args.min_clamp_joint_motion}")
    if args.min_cradle_joint_motion is not None:
        cradle_motion = summary.get("max_cradle_joint_motion_m")
        if cradle_motion is None or float(cradle_motion) < float(args.min_cradle_joint_motion):
            failures.append(f"max_cradle_joint_motion_m too low: {cradle_motion} < {args.min_cradle_joint_motion}")
    if args.min_support_foot_joint_count is not None:
        support_foot_joint_count = int(summary.get("support_foot_joint_count") or 0)
        if support_foot_joint_count < int(args.min_support_foot_joint_count):
            failures.append(
                f"support_foot_joint_count too low: {support_foot_joint_count} < {args.min_support_foot_joint_count}"
            )
    if args.max_abs_anchor_travel_x is not None:
        anchor_travel = summary.get("max_abs_anchor_travel_x_m")
        if anchor_travel is None or float(anchor_travel) > float(args.max_abs_anchor_travel_x):
            failures.append(f"max_abs_anchor_travel_x_m too high: {anchor_travel} > {args.max_abs_anchor_travel_x}")
    if args.max_abs_support_foot_travel_x is not None:
        support_foot_travel = summary.get("max_abs_support_foot_travel_x_m")
        if support_foot_travel is None or float(support_foot_travel) > float(args.max_abs_support_foot_travel_x):
            failures.append(
                f"max_abs_support_foot_travel_x_m too high: "
                f"{support_foot_travel} > {args.max_abs_support_foot_travel_x}"
            )
    if args.min_support_foot_x_joint_motion is not None:
        support_foot_joint_motion = summary.get("max_support_foot_x_joint_motion_m")
        if support_foot_joint_motion is None or float(support_foot_joint_motion) < float(args.min_support_foot_x_joint_motion):
            failures.append(
                f"max_support_foot_x_joint_motion_m too low: "
                f"{support_foot_joint_motion} < {args.min_support_foot_x_joint_motion}"
            )
    if args.min_support_foot_z_joint_count is not None:
        support_foot_z_joint_count = int(summary.get("support_foot_z_joint_count") or 0)
        if support_foot_z_joint_count < int(args.min_support_foot_z_joint_count):
            failures.append(
                f"support_foot_z_joint_count too low: "
                f"{support_foot_z_joint_count} < {args.min_support_foot_z_joint_count}"
            )
    if args.min_support_foot_z_joint_motion is not None:
        support_foot_z_joint_motion = summary.get("max_support_foot_z_joint_motion_m")
        if support_foot_z_joint_motion is None or float(support_foot_z_joint_motion) < float(args.min_support_foot_z_joint_motion):
            failures.append(
                f"max_support_foot_z_joint_motion_m too low: "
                f"{support_foot_z_joint_motion} < {args.min_support_foot_z_joint_motion}"
            )
    if args.min_actual_support_foot_lift is not None:
        actual_lift = summary.get("max_actual_support_foot_lift_m")
        if actual_lift is None or float(actual_lift) < float(args.min_actual_support_foot_lift):
            failures.append(
                f"max_actual_support_foot_lift_m too low: "
                f"{actual_lift} < {args.min_actual_support_foot_lift}"
            )
    if args.min_near_ground_foot_count is not None:
        near_ground_count = summary.get("min_near_ground_foot_count")
        if near_ground_count is None or int(near_ground_count) < int(args.min_near_ground_foot_count):
            failures.append(
                f"min_near_ground_foot_count too low: "
                f"{near_ground_count} < {args.min_near_ground_foot_count}"
            )
    if args.min_drive_near_ground_foot_count is not None:
        drive_near_ground_count = summary.get("min_drive_near_ground_foot_count")
        if drive_near_ground_count is None or int(drive_near_ground_count) < int(args.min_drive_near_ground_foot_count):
            failures.append(
                f"min_drive_near_ground_foot_count too low: "
                f"{drive_near_ground_count} < {args.min_drive_near_ground_foot_count}"
            )
    if args.max_drive_near_ground_zero_steps is not None:
        zero_steps = int(summary.get("drive_near_ground_zero_steps") or 0)
        if zero_steps > int(args.max_drive_near_ground_zero_steps):
            failures.append(
                f"drive_near_ground_zero_steps too high: "
                f"{zero_steps} > {args.max_drive_near_ground_zero_steps}"
            )
    if args.max_drive_near_ground_lt2_steps is not None:
        lt2_steps = int(summary.get("drive_near_ground_lt2_steps") or 0)
        if lt2_steps > int(args.max_drive_near_ground_lt2_steps):
            failures.append(
                f"drive_near_ground_lt2_steps too high: "
                f"{lt2_steps} > {args.max_drive_near_ground_lt2_steps}"
            )
    if args.require_support_foot_contact_report_evidence and not bool(summary.get("support_foot_contact_report_available")):
        failures.append(
            "support_foot_contact_report_available is false or missing: "
            f"{summary.get('support_foot_contact_report_available')}"
        )
    if args.min_drive_contact_report_foot_count is not None:
        contact_count = summary.get("min_drive_contact_report_foot_count")
        if contact_count is None or int(contact_count) < int(args.min_drive_contact_report_foot_count):
            failures.append(
                f"min_drive_contact_report_foot_count too low: "
                f"{contact_count} < {args.min_drive_contact_report_foot_count}"
            )
    if args.max_drive_contact_report_zero_steps is not None:
        zero_steps = int(summary.get("drive_contact_report_zero_steps") or 0)
        if zero_steps > int(args.max_drive_contact_report_zero_steps):
            failures.append(
                f"drive_contact_report_zero_steps too high: "
                f"{zero_steps} > {args.max_drive_contact_report_zero_steps}"
            )
    if args.max_drive_contact_report_lt2_steps is not None:
        lt2_steps = int(summary.get("drive_contact_report_lt2_steps") or 0)
        if lt2_steps > int(args.max_drive_contact_report_lt2_steps):
            failures.append(
                f"drive_contact_report_lt2_steps too high: "
                f"{lt2_steps} > {args.max_drive_contact_report_lt2_steps}"
            )
    if args.min_commanded_stance_contact_report_foot_count is not None:
        contact_count = summary.get("min_commanded_stance_contact_report_foot_count")
        if contact_count is None or int(contact_count) < int(args.min_commanded_stance_contact_report_foot_count):
            failures.append(
                f"min_commanded_stance_contact_report_foot_count too low: "
                f"{contact_count} < {args.min_commanded_stance_contact_report_foot_count}"
            )
    if args.max_commanded_stance_contact_report_lt2_steps is not None:
        lt2_steps = int(summary.get("commanded_stance_contact_report_lt2_steps") or 0)
        if lt2_steps > int(args.max_commanded_stance_contact_report_lt2_steps):
            failures.append(
                f"commanded_stance_contact_report_lt2_steps too high: "
                f"{lt2_steps} > {args.max_commanded_stance_contact_report_lt2_steps}"
            )
    if args.require_support_foot_effort_evidence and not bool(summary.get("support_foot_effort_available")):
        failures.append(
            f"support_foot_effort_available is false or missing: {summary.get('support_foot_effort_available')}"
        )
    if args.min_drive_effort_supported_foot_count is not None:
        effort_count = summary.get("min_drive_effort_supported_foot_count")
        if effort_count is None or int(effort_count) < int(args.min_drive_effort_supported_foot_count):
            failures.append(
                f"min_drive_effort_supported_foot_count too low: "
                f"{effort_count} < {args.min_drive_effort_supported_foot_count}"
            )
    if args.max_drive_effort_supported_zero_steps is not None:
        zero_steps = int(summary.get("drive_effort_supported_zero_steps") or 0)
        if zero_steps > int(args.max_drive_effort_supported_zero_steps):
            failures.append(
                f"drive_effort_supported_zero_steps too high: "
                f"{zero_steps} > {args.max_drive_effort_supported_zero_steps}"
            )
    if args.max_drive_effort_supported_lt2_steps is not None:
        lt2_steps = int(summary.get("drive_effort_supported_lt2_steps") or 0)
        if lt2_steps > int(args.max_drive_effort_supported_lt2_steps):
            failures.append(
                f"drive_effort_supported_lt2_steps too high: "
                f"{lt2_steps} > {args.max_drive_effort_supported_lt2_steps}"
            )
    if args.min_commanded_stance_effort_supported_foot_count is not None:
        effort_count = summary.get("min_commanded_stance_effort_supported_foot_count")
        if effort_count is None or int(effort_count) < int(args.min_commanded_stance_effort_supported_foot_count):
            failures.append(
                f"min_commanded_stance_effort_supported_foot_count too low: "
                f"{effort_count} < {args.min_commanded_stance_effort_supported_foot_count}"
            )
    if args.max_commanded_stance_effort_supported_lt2_steps is not None:
        lt2_steps = int(summary.get("commanded_stance_effort_supported_lt2_steps") or 0)
        if lt2_steps > int(args.max_commanded_stance_effort_supported_lt2_steps):
            failures.append(
                f"commanded_stance_effort_supported_lt2_steps too high: "
                f"{lt2_steps} > {args.max_commanded_stance_effort_supported_lt2_steps}"
            )
    if args.min_commanded_stance_near_ground_foot_count is not None:
        commanded_count = summary.get("min_commanded_stance_near_ground_foot_count")
        if commanded_count is None or int(commanded_count) < int(args.min_commanded_stance_near_ground_foot_count):
            failures.append(
                f"min_commanded_stance_near_ground_foot_count too low: "
                f"{commanded_count} < {args.min_commanded_stance_near_ground_foot_count}"
            )
    if args.max_commanded_stance_near_ground_lt2_steps is not None:
        commanded_lt2_steps = int(summary.get("commanded_stance_near_ground_lt2_steps") or 0)
        if commanded_lt2_steps > int(args.max_commanded_stance_near_ground_lt2_steps):
            failures.append(
                f"commanded_stance_near_ground_lt2_steps too high: "
                f"{commanded_lt2_steps} > {args.max_commanded_stance_near_ground_lt2_steps}"
            )
    if args.min_support_polygon_margin is not None:
        support_margin = summary.get("min_support_polygon_margin_m")
        if support_margin is None or float(support_margin) < float(args.min_support_polygon_margin):
            failures.append(
                f"min_support_polygon_margin_m too low: "
                f"{support_margin} < {args.min_support_polygon_margin}"
            )
    if args.max_near_ground_foot_speed is not None:
        speed_by_foot = summary.get("per_foot_max_near_ground_xy_speed_mps") or {}
        if not isinstance(speed_by_foot, dict) or not speed_by_foot:
            failures.append("per_foot_max_near_ground_xy_speed_mps missing")
        else:
            too_fast = {
                foot: speed
                for foot, speed in speed_by_foot.items()
                if float(speed or 0.0) > float(args.max_near_ground_foot_speed)
            }
            if too_fast:
                failures.append(
                    f"near-ground foot speeds too high: "
                    f"{too_fast} > {args.max_near_ground_foot_speed}"
                )
    if args.max_near_ground_foot_slip is not None:
        slip_by_foot = summary.get("per_foot_max_near_ground_xy_slip_m") or {}
        if not isinstance(slip_by_foot, dict) or not slip_by_foot:
            failures.append("per_foot_max_near_ground_xy_slip_m missing")
        else:
            too_much_slip = {
                foot: slip
                for foot, slip in slip_by_foot.items()
                if float(slip or 0.0) > float(args.max_near_ground_foot_slip)
            }
            if too_much_slip:
                failures.append(
                    f"near-ground foot slip too high: "
                    f"{too_much_slip} > {args.max_near_ground_foot_slip}"
                )
    if args.require_stance_foot_world_lock:
        if not bool(summary.get("stance_foot_world_lock_enabled")):
            failures.append(
                "stance_foot_world_lock_enabled is false or missing: "
                f"{summary.get('stance_foot_world_lock_enabled')}"
            )
        if int(summary.get("stance_foot_world_lock_joint_count") or 0) <= 0:
            failures.append(
                "stance_foot_world_lock_joint_count missing or zero: "
                f"{summary.get('stance_foot_world_lock_joint_count')}"
            )
    if args.min_stance_foot_world_lock_switches is not None:
        switch_count = int(summary.get("stance_foot_world_lock_switch_count") or 0)
        if switch_count < int(args.min_stance_foot_world_lock_switches):
            failures.append(
                f"stance_foot_world_lock_switch_count too low: "
                f"{switch_count} < {args.min_stance_foot_world_lock_switches}"
            )
    if args.require_freeze_locked_stance_foot_targets:
        if not bool(summary.get("freeze_locked_stance_foot_targets_enabled")):
            failures.append(
                "freeze_locked_stance_foot_targets_enabled is false or missing: "
                f"{summary.get('freeze_locked_stance_foot_targets_enabled')}"
            )
        if int(summary.get("freeze_locked_stance_foot_target_count") or 0) <= 0:
            failures.append(
                "freeze_locked_stance_foot_target_count missing or zero: "
                f"{summary.get('freeze_locked_stance_foot_target_count')}"
            )
    if args.require_freeze_commanded_stance_foot_targets:
        if not bool(summary.get("freeze_commanded_stance_foot_targets_enabled")):
            failures.append(
                "freeze_commanded_stance_foot_targets_enabled is false or missing: "
                f"{summary.get('freeze_commanded_stance_foot_targets_enabled')}"
            )
        if int(summary.get("freeze_commanded_stance_foot_target_count") or 0) <= 0:
            failures.append(
                "freeze_commanded_stance_foot_target_count missing or zero: "
                f"{summary.get('freeze_commanded_stance_foot_target_count')}"
            )
        if int(summary.get("freeze_commanded_stance_foot_target_switch_count") or 0) <= 0:
            failures.append(
                "freeze_commanded_stance_foot_target_switch_count missing or zero: "
                f"{summary.get('freeze_commanded_stance_foot_target_switch_count')}"
            )
    if args.require_planted_stance_rail_propulsion:
        if not bool(summary.get("planted_stance_rail_propulsion_enabled")):
            failures.append(
                "planted_stance_rail_propulsion_enabled is false or missing: "
                f"{summary.get('planted_stance_rail_propulsion_enabled')}"
            )
        if int(summary.get("planted_stance_rail_propulsion_steps") or 0) <= 0:
            failures.append(
                "planted_stance_rail_propulsion_steps missing or zero: "
                f"{summary.get('planted_stance_rail_propulsion_steps')}"
            )
    if args.max_fall_events is not None and int(summary.get("fall_events", 0)) > int(args.max_fall_events):
        failures.append(f"fall_events too high: {summary.get('fall_events')}")
    if int(summary.get("box_drop_events", 0)) > int(args.max_box_drop_events):
        failures.append(f"box_drop_events too high: {summary.get('box_drop_events')}")
    max_box_travel = summary.get("max_box_travel_xy_m", summary.get("max_box_travel_x_m", 0.0))
    if args.min_box_travel is not None and float(max_box_travel or 0.0) < float(args.min_box_travel):
        failures.append(f"box travel too low: {max_box_travel} < {args.min_box_travel}")
    if args.min_abs_box_travel_x is not None:
        max_abs_box_travel = summary.get("max_abs_box_travel_x_m", summary.get("max_box_travel_x_m"))
        if max_abs_box_travel is None or float(max_abs_box_travel) < float(args.min_abs_box_travel_x):
            failures.append(f"absolute box travel x too low: {max_abs_box_travel} < {args.min_abs_box_travel_x}")
    if args.min_post_settle_box_travel_x is not None:
        post_settle_box_travel = summary.get("max_post_settle_box_travel_x_m")
        if post_settle_box_travel is None or float(post_settle_box_travel) < float(args.min_post_settle_box_travel_x):
            failures.append(f"post-settle box travel x too low: {post_settle_box_travel}")
    if args.min_abs_post_settle_box_travel_x is not None:
        post_settle_abs_box_travel = summary.get(
            "max_abs_post_settle_box_travel_x_m",
            summary.get("max_post_settle_box_travel_x_m"),
        )
        if (
            post_settle_abs_box_travel is None
            or float(post_settle_abs_box_travel) < float(args.min_abs_post_settle_box_travel_x)
        ):
            failures.append(
                f"absolute post-settle box travel x too low: "
                f"{post_settle_abs_box_travel} < {args.min_abs_post_settle_box_travel_x}"
            )
    if args.max_final_box_target_distance_x is not None:
        final_box_target_distance = summary.get("final_box_target_distance_x_m", summary.get("final_box_target_distance_xy_m"))
        if final_box_target_distance is None or float(final_box_target_distance) > float(args.max_final_box_target_distance_x):
            failures.append(f"final box target distance too high: {final_box_target_distance}")
    if args.max_final_post_settle_box_target_distance_x is not None:
        final_post_settle_box_target_distance = summary.get("final_post_settle_box_target_distance_x_m")
        if (
            final_post_settle_box_target_distance is None
            or float(final_post_settle_box_target_distance) > float(args.max_final_post_settle_box_target_distance_x)
        ):
            failures.append(
                f"final post-settle box target distance too high: {final_post_settle_box_target_distance}"
            )
    if args.max_post_settle_box_travel_loss_after_peak is not None:
        travel_loss_after_peak = summary.get("post_settle_box_travel_loss_after_peak_m")
        if travel_loss_after_peak is None or float(travel_loss_after_peak) > float(args.max_post_settle_box_travel_loss_after_peak):
            failures.append(f"post-settle box travel loss after peak too high: {travel_loss_after_peak}")
    if args.require_kinematic_proxy_writes:
        if int(summary.get("robot_proxy_pose_write_count", 0)) <= 0:
            failures.append("robot_proxy_pose_write_count missing or zero")
        if int(summary.get("box_kinematic_pose_write_count", 0)) <= 0:
            failures.append("box_kinematic_pose_write_count missing or zero")
    if args.require_root_shortcut_free:
        root_shortcuts = summary.get("root_shortcuts")
        if isinstance(root_shortcuts, dict):
            bad_shortcuts = {key: value for key, value in root_shortcuts.items() if int(value or 0) != 0}
            if bad_shortcuts:
                failures.append(f"root/body/box shortcut counters nonzero: {bad_shortcuts}")
        elif not bool(summary.get("root_shortcut_free")):
            failures.append(f"root_shortcut_free is false or missing: {summary.get('root_shortcut_free')}")
    if args.max_anchor_world_joint_retarget_count is not None:
        retarget_count = int(summary.get("anchor_world_joint_retarget_count") or 0)
        if retarget_count > int(args.max_anchor_world_joint_retarget_count):
            failures.append(
                f"anchor_world_joint_retarget_count too high: {retarget_count} > {args.max_anchor_world_joint_retarget_count}"
            )
    if args.max_support_root_pose_write_count is not None:
        support_root_writes = int(summary.get("support_root_pose_write_count") or 0)
        if support_root_writes > int(args.max_support_root_pose_write_count):
            failures.append(
                f"support_root_pose_write_count too high: {support_root_writes} > {args.max_support_root_pose_write_count}"
            )
    if args.max_foot_pose_write_count is not None:
        foot_writes = int(summary.get("foot_pose_write_count") or 0)
        if foot_writes > int(args.max_foot_pose_write_count):
            failures.append(f"foot_pose_write_count too high: {foot_writes} > {args.max_foot_pose_write_count}")
    if args.max_stance_anchor_pose_write_count is not None:
        stance_anchor_writes = int(summary.get("stance_anchor_pose_write_count") or 0)
        if stance_anchor_writes > int(args.max_stance_anchor_pose_write_count):
            failures.append(
                f"stance_anchor_pose_write_count too high: "
                f"{stance_anchor_writes} > {args.max_stance_anchor_pose_write_count}"
            )
    if args.forbid_fixed_world_support:
        if bool(summary.get("stance_anchor_fixed_to_world")):
            failures.append("stance_anchor_fixed_to_world is true")
        if bool(summary.get("stance_foot_world_lock_enabled")):
            failures.append("stance_foot_world_lock_enabled is true")
    if args.require_non_success_claim:
        success_claim = str(summary.get("success_claim", ""))
        contract_reason = str(summary.get("controller_contract", {}).get("non_success_reason", ""))
        if "diagnostic" not in success_claim:
            failures.append(f"success_claim is not diagnostic-only: {success_claim}")
        known_limitations = ("kinematic_proxy", "not_free_walking", "not_final_robot_controller")
        if not any(marker in contract_reason for marker in known_limitations):
            failures.append(f"controller contract does not expose proxy limitation: {contract_reason}")

    report = {
        "summary": str(args.summary),
        "scene_type": summary.get("scene_type"),
        "success_claim": summary.get("success_claim"),
        "controller_mode": summary.get("controller_mode"),
        "carry_posture": summary.get("carry_posture"),
        "backend_support_mode": summary.get("backend_support_mode"),
        "completed_steps": completed_steps,
        "probe_steps_requested": summary.get("probe_steps_requested"),
        "probe_mode": summary.get("probe_mode"),
        "probe_x_amplitude_m": summary.get("probe_x_amplitude_m"),
        "probe_z_amplitude_m": summary.get("probe_z_amplitude_m"),
        "max_probe_torso_travel_x_m": summary.get("max_probe_torso_travel_x_m"),
        "max_probe_torso_travel_z_m": summary.get("max_probe_torso_travel_z_m"),
        "max_probe_box_travel_x_m": summary.get("max_probe_box_travel_x_m"),
        "max_probe_box_travel_z_m": summary.get("max_probe_box_travel_z_m"),
        "max_probe_box_relative_error_m": summary.get("max_probe_box_relative_error_m"),
        "max_probe_support_foot_x_tracking_error_m": summary.get("max_probe_support_foot_x_tracking_error_m"),
        "mean_probe_support_foot_x_tracking_error_m": summary.get("mean_probe_support_foot_x_tracking_error_m"),
        "probe_support_foot_x_tracking_error_samples": summary.get("probe_support_foot_x_tracking_error_samples"),
        "max_probe_support_foot_z_tracking_error_m": summary.get("max_probe_support_foot_z_tracking_error_m"),
        "mean_probe_support_foot_z_tracking_error_m": summary.get("mean_probe_support_foot_z_tracking_error_m"),
        "probe_support_foot_z_tracking_error_samples": summary.get("probe_support_foot_z_tracking_error_samples"),
        "probe_joint_effort_available": summary.get("probe_joint_effort_available"),
        "probe_joint_effort_read_error_count": summary.get("probe_joint_effort_read_error_count"),
        "probe_joint_effort_first_error": summary.get("probe_joint_effort_first_error"),
        "max_probe_support_foot_x_measured_effort": summary.get("max_probe_support_foot_x_measured_effort"),
        "mean_probe_support_foot_x_measured_effort": summary.get("mean_probe_support_foot_x_measured_effort"),
        "probe_support_foot_x_measured_effort_samples": summary.get("probe_support_foot_x_measured_effort_samples"),
        "max_probe_support_foot_z_measured_effort": summary.get("max_probe_support_foot_z_measured_effort"),
        "mean_probe_support_foot_z_measured_effort": summary.get("mean_probe_support_foot_z_measured_effort"),
        "probe_support_foot_z_measured_effort_samples": summary.get("probe_support_foot_z_measured_effort_samples"),
        "final_probe_box_lag_x_m": summary.get("final_probe_box_lag_x_m"),
        "final_probe_box_lag_z_m": summary.get("final_probe_box_lag_z_m"),
        "probe_belief_available": summary.get("probe_belief_available"),
        "probe_belief_source": summary.get("probe_belief_source"),
        "probe_belief_uses_hidden_ground_truth": summary.get("probe_belief_uses_hidden_ground_truth"),
        "probe_compliance_proxy": summary.get("probe_compliance_proxy"),
        "probe_lag_proxy": summary.get("probe_lag_proxy"),
        "probe_support_foot_x_tracking_proxy": summary.get("probe_support_foot_x_tracking_proxy"),
        "probe_support_foot_z_tracking_proxy": summary.get("probe_support_foot_z_tracking_proxy"),
        "probe_support_foot_x_effort_proxy": summary.get("probe_support_foot_x_effort_proxy"),
        "probe_support_foot_z_effort_proxy": summary.get("probe_support_foot_z_effort_proxy"),
        "probe_risk_score": summary.get("probe_risk_score"),
        "probe_load_risk_bucket": summary.get("probe_load_risk_bucket"),
        "probe_recommended_carry_adjustment": summary.get("probe_recommended_carry_adjustment"),
        "probe_belief_policy_action_applied": summary.get("probe_belief_policy_action_applied"),
        "box_mass_kg": summary.get("box_mass_kg"),
        "box_randomized": summary.get("box_randomized"),
        "box_mass_range_kg": summary.get("box_mass_range_kg"),
        "box_size_m": summary.get("box_size_m"),
        "box_size_jitter_fraction": summary.get("box_size_jitter_fraction"),
        "box_com_offset_m": summary.get("box_com_offset_m"),
        "box_seed": summary.get("box_seed"),
        "max_torso_travel_xy_m": summary.get("max_torso_travel_xy_m"),
        "max_box_travel_xy_m": summary.get("max_box_travel_xy_m"),
        "max_box_travel_x_m": summary.get("max_box_travel_x_m"),
        "max_abs_box_travel_x_m": summary.get("max_abs_box_travel_x_m"),
        "max_post_settle_box_travel_x_m": summary.get("max_post_settle_box_travel_x_m"),
        "max_abs_post_settle_box_travel_x_m": summary.get("max_abs_post_settle_box_travel_x_m"),
        "max_target_directed_post_settle_box_travel_m": summary.get(
            "max_target_directed_post_settle_box_travel_m"
        ),
        "final_post_settle_box_travel_x_m": summary.get("final_post_settle_box_travel_x_m"),
        "final_post_settle_box_target_distance_x_m": summary.get("final_post_settle_box_target_distance_x_m"),
        "post_settle_box_travel_loss_after_peak_m": summary.get("post_settle_box_travel_loss_after_peak_m"),
        "final_box_target_distance_xy_m": summary.get("final_box_target_distance_xy_m"),
        "final_box_target_distance_x_m": summary.get("final_box_target_distance_x_m"),
        "box_drop_events": summary.get("box_drop_events"),
        "fall_events": summary.get("fall_events"),
        "motion_mode": summary.get("motion_mode"),
        "quasistatic_compensate_settle_drift": summary.get("quasistatic_compensate_settle_drift"),
        "quasistatic_effective_target_x_m": summary.get("quasistatic_effective_target_x_m"),
        "gated_step_hold_steps": summary.get("gated_step_hold_steps"),
        "gated_step_release_steps": summary.get("gated_step_release_steps"),
        "gated_step_recovery_steps": summary.get("gated_step_recovery_steps"),
        "gated_step_last_block_reason": summary.get("gated_step_last_block_reason"),
        "gated_step_peak_post_settle_box_travel_x_m": summary.get("gated_step_peak_post_settle_box_travel_x_m"),
        "gated_step_travel_loss_after_peak_m": summary.get("gated_step_travel_loss_after_peak_m"),
        "max_commanded_leg_lift_m": summary.get("max_commanded_leg_lift_m"),
        "max_actual_leg_lift_m": summary.get("max_actual_leg_lift_m"),
        "max_abs_actual_x_slide_m": summary.get("max_abs_actual_x_slide_m"),
        "per_leg_near_ground_steps": summary.get("per_leg_near_ground_steps"),
        "per_leg_max_actual_lift_m": summary.get("per_leg_max_actual_lift_m"),
        "per_leg_max_abs_actual_x_m": summary.get("per_leg_max_abs_actual_x_m"),
        "robot_proxy_pose_write_count": summary.get("robot_proxy_pose_write_count"),
        "box_kinematic_pose_write_count": summary.get("box_kinematic_pose_write_count"),
        "root_shortcut_free": summary.get("root_shortcut_free"),
        "support_root_pose_write_count": summary.get("support_root_pose_write_count"),
        "anchor_world_joint_retarget_count": summary.get("anchor_world_joint_retarget_count"),
        "foot_pose_write_count": summary.get("foot_pose_write_count"),
        "stance_anchor_pose_write_count": summary.get("stance_anchor_pose_write_count"),
        "stance_anchor_fixed_to_world": summary.get("stance_anchor_fixed_to_world"),
        "rail_joint_count": summary.get("rail_joint_count"),
        "rail_capacity_m": summary.get("rail_capacity_m"),
        "rail_joint_indices": summary.get("rail_joint_indices"),
        "max_joint_motion_m": summary.get("max_joint_motion_m"),
        "max_rail_joint_motion_m": summary.get("max_rail_joint_motion_m"),
        "max_clamp_joint_motion_m": summary.get("max_clamp_joint_motion_m"),
        "max_cradle_joint_motion_m": summary.get("max_cradle_joint_motion_m"),
        "max_commanded_clamp_target_m": summary.get("max_commanded_clamp_target_m"),
        "final_commanded_clamp_target_m": summary.get("final_commanded_clamp_target_m"),
        "clamp_drive_target_update_count": summary.get("clamp_drive_target_update_count"),
        "max_commanded_cradle_target_m": summary.get("max_commanded_cradle_target_m"),
        "final_commanded_cradle_target_m": summary.get("final_commanded_cradle_target_m"),
        "cradle_drive_target_update_count": summary.get("cradle_drive_target_update_count"),
        "support_foot_mode": summary.get("support_foot_mode"),
        "support_foot_joint_count": summary.get("support_foot_joint_count"),
        "support_foot_x_joint_count": summary.get("support_foot_x_joint_count"),
        "support_foot_z_joint_count": summary.get("support_foot_z_joint_count"),
        "max_support_foot_x_joint_motion_m": summary.get("max_support_foot_x_joint_motion_m"),
        "max_support_foot_z_joint_motion_m": summary.get("max_support_foot_z_joint_motion_m"),
        "max_commanded_support_foot_lift_m": summary.get("max_commanded_support_foot_lift_m"),
        "support_foot_placement_mode": summary.get("support_foot_placement_mode"),
        "support_foot_placement_controller_enabled": summary.get("support_foot_placement_controller_enabled"),
        "support_foot_directional_placement": summary.get("support_foot_directional_placement"),
        "alternating_support_foot_drive": summary.get("alternating_support_foot_drive"),
        "final_support_foot_x_joint_target_m": summary.get("final_support_foot_x_joint_target_m"),
        "max_abs_anchor_travel_x_m": summary.get("max_abs_anchor_travel_x_m"),
        "max_abs_support_foot_travel_x_m": summary.get("max_abs_support_foot_travel_x_m"),
        "support_foot_min_z_m": summary.get("support_foot_min_z_m"),
        "support_foot_max_z_m": summary.get("support_foot_max_z_m"),
        "support_foot_contact_report_requested": summary.get("support_foot_contact_report_requested"),
        "support_foot_contact_report_available": summary.get("support_foot_contact_report_available"),
        "support_foot_contact_report_threshold": summary.get("support_foot_contact_report_threshold"),
        "support_foot_contact_report_enabled_paths": summary.get("support_foot_contact_report_enabled_paths"),
        "support_foot_contact_report_event_count": summary.get("support_foot_contact_report_event_count"),
        "support_foot_contact_report_error_count": summary.get("support_foot_contact_report_error_count"),
        "support_foot_contact_report_first_error": summary.get("support_foot_contact_report_first_error"),
        "per_foot_contact_report_steps": summary.get("per_foot_contact_report_steps"),
        "min_contact_report_foot_count": summary.get("min_contact_report_foot_count"),
        "max_contact_report_foot_count": summary.get("max_contact_report_foot_count"),
        "contact_report_zero_steps": summary.get("contact_report_zero_steps"),
        "contact_report_lt2_steps": summary.get("contact_report_lt2_steps"),
        "min_drive_contact_report_foot_count": summary.get("min_drive_contact_report_foot_count"),
        "drive_contact_report_zero_steps": summary.get("drive_contact_report_zero_steps"),
        "drive_contact_report_lt2_steps": summary.get("drive_contact_report_lt2_steps"),
        "min_commanded_stance_contact_report_foot_count": summary.get(
            "min_commanded_stance_contact_report_foot_count"
        ),
        "commanded_stance_contact_report_lt2_steps": summary.get(
            "commanded_stance_contact_report_lt2_steps"
        ),
        "support_foot_effort_contact_threshold": summary.get("support_foot_effort_contact_threshold"),
        "support_foot_effort_available": summary.get("support_foot_effort_available"),
        "support_foot_effort_read_error_count": summary.get("support_foot_effort_read_error_count"),
        "support_foot_effort_first_error": summary.get("support_foot_effort_first_error"),
        "per_foot_max_support_x_measured_effort": summary.get("per_foot_max_support_x_measured_effort"),
        "per_foot_max_support_z_measured_effort": summary.get("per_foot_max_support_z_measured_effort"),
        "per_foot_max_support_measured_effort": summary.get("per_foot_max_support_measured_effort"),
        "min_drive_effort_supported_foot_count": summary.get("min_drive_effort_supported_foot_count"),
        "drive_effort_supported_zero_steps": summary.get("drive_effort_supported_zero_steps"),
        "drive_effort_supported_lt2_steps": summary.get("drive_effort_supported_lt2_steps"),
        "min_commanded_stance_effort_supported_foot_count": summary.get(
            "min_commanded_stance_effort_supported_foot_count"
        ),
        "commanded_stance_effort_supported_lt2_steps": summary.get(
            "commanded_stance_effort_supported_lt2_steps"
        ),
        "max_actual_support_foot_lift_m": summary.get("max_actual_support_foot_lift_m"),
        "per_foot_max_actual_lift_m": summary.get("per_foot_max_actual_lift_m"),
        "min_near_ground_foot_count": summary.get("min_near_ground_foot_count"),
        "max_near_ground_foot_count": summary.get("max_near_ground_foot_count"),
        "support_foot_double_support_fraction": summary.get("support_foot_double_support_fraction"),
        "support_foot_continuity_grace_steps": summary.get("support_foot_continuity_grace_steps"),
        "support_foot_continuity_start_step": summary.get("support_foot_continuity_start_step"),
        "stance_foot_world_lock_enabled": summary.get("stance_foot_world_lock_enabled"),
        "stance_foot_world_lock_joint_count": summary.get("stance_foot_world_lock_joint_count"),
        "stance_foot_world_lock_switch_count": summary.get("stance_foot_world_lock_switch_count"),
        "stance_foot_world_lock_pose_update_count": summary.get("stance_foot_world_lock_pose_update_count"),
        "stance_foot_world_lock_active_feet": summary.get("stance_foot_world_lock_active_feet"),
        "freeze_locked_stance_foot_targets_enabled": summary.get("freeze_locked_stance_foot_targets_enabled"),
        "freeze_locked_stance_foot_target_count": summary.get("freeze_locked_stance_foot_target_count"),
        "freeze_commanded_stance_foot_targets_enabled": summary.get(
            "freeze_commanded_stance_foot_targets_enabled"
        ),
        "freeze_commanded_stance_foot_target_count": summary.get("freeze_commanded_stance_foot_target_count"),
        "freeze_commanded_stance_foot_target_switch_count": summary.get(
            "freeze_commanded_stance_foot_target_switch_count"
        ),
        "freeze_commanded_stance_foot_active_feet": summary.get("freeze_commanded_stance_foot_active_feet"),
        "planted_stance_rail_propulsion_enabled": summary.get("planted_stance_rail_propulsion_enabled"),
        "planted_stance_rail_propulsion_steps": summary.get("planted_stance_rail_propulsion_steps"),
        "feedback_step_controller_enabled": summary.get("feedback_step_controller_enabled"),
        "feedback_step_x_gain": summary.get("feedback_step_x_gain"),
        "feedback_step_x_limit_m": summary.get("feedback_step_x_limit_m"),
        "feedback_step_tilt_gain": summary.get("feedback_step_tilt_gain"),
        "feedback_step_tilt_limit_m": summary.get("feedback_step_tilt_limit_m"),
        "feedback_step_applied_steps": summary.get("feedback_step_applied_steps"),
        "max_abs_feedback_step_x_adjustment_m": summary.get("max_abs_feedback_step_x_adjustment_m"),
        "max_abs_feedback_step_tilt_adjustment_m": summary.get("max_abs_feedback_step_tilt_adjustment_m"),
        "online_probe_adaptive_support_enabled": summary.get("online_probe_adaptive_support_enabled"),
        "online_probe_adaptive_support_decision_applied": summary.get(
            "online_probe_adaptive_support_decision_applied"
        ),
        "online_probe_adaptive_support_decision_step": summary.get(
            "online_probe_adaptive_support_decision_step"
        ),
        "online_probe_adaptive_support_uses_hidden_ground_truth": summary.get(
            "online_probe_adaptive_support_uses_hidden_ground_truth"
        ),
        "online_probe_adaptive_support_risk_score": summary.get("online_probe_adaptive_support_risk_score"),
        "online_probe_adaptive_support_risk_bucket": summary.get("online_probe_adaptive_support_risk_bucket"),
        "online_probe_adaptive_support_profile": summary.get("online_probe_adaptive_support_profile"),
        "online_probe_adaptive_support_step_height_m": summary.get(
            "online_probe_adaptive_support_step_height_m"
        ),
        "online_probe_adaptive_support_double_support_fraction": summary.get(
            "online_probe_adaptive_support_double_support_fraction"
        ),
        "online_probe_adaptive_support_stance_x_m": summary.get("online_probe_adaptive_support_stance_x_m"),
        "online_probe_adaptive_support_swing_x_m": summary.get("online_probe_adaptive_support_swing_x_m"),
        "online_probe_adaptive_hold_enabled": summary.get("online_probe_adaptive_hold_enabled"),
        "online_probe_adaptive_hold_decision_applied": summary.get(
            "online_probe_adaptive_hold_decision_applied"
        ),
        "online_probe_adaptive_hold_decision_step": summary.get("online_probe_adaptive_hold_decision_step"),
        "online_probe_adaptive_hold_uses_hidden_ground_truth": summary.get(
            "online_probe_adaptive_hold_uses_hidden_ground_truth"
        ),
        "online_probe_adaptive_hold_risk_score": summary.get("online_probe_adaptive_hold_risk_score"),
        "online_probe_adaptive_hold_risk_bucket": summary.get("online_probe_adaptive_hold_risk_bucket"),
        "online_probe_adaptive_hold_profile": summary.get("online_probe_adaptive_hold_profile"),
        "online_probe_adaptive_hold_closure_fraction": summary.get(
            "online_probe_adaptive_hold_closure_fraction"
        ),
        "online_probe_adaptive_hold_actuated": summary.get("online_probe_adaptive_hold_actuated"),
        "online_probe_adaptive_hold_collision_available": summary.get(
            "online_probe_adaptive_hold_collision_available"
        ),
        "online_probe_adaptive_hold_collision_paths": summary.get("online_probe_adaptive_hold_collision_paths"),
        "online_probe_adaptive_hold_collision_enabled": summary.get(
            "online_probe_adaptive_hold_collision_enabled"
        ),
        "online_probe_adaptive_hold_collision_update_count": summary.get(
            "online_probe_adaptive_hold_collision_update_count"
        ),
        "near_ground_zero_steps": summary.get("near_ground_zero_steps"),
        "near_ground_lt2_steps": summary.get("near_ground_lt2_steps"),
        "min_drive_near_ground_foot_count": summary.get("min_drive_near_ground_foot_count"),
        "drive_near_ground_zero_steps": summary.get("drive_near_ground_zero_steps"),
        "drive_near_ground_lt2_steps": summary.get("drive_near_ground_lt2_steps"),
        "min_commanded_stance_near_ground_foot_count": summary.get("min_commanded_stance_near_ground_foot_count"),
        "commanded_stance_near_ground_lt2_steps": summary.get("commanded_stance_near_ground_lt2_steps"),
        "min_support_polygon_margin_m": summary.get("min_support_polygon_margin_m"),
        "per_foot_near_ground_steps": summary.get("per_foot_near_ground_steps"),
        "per_foot_max_near_ground_xy_speed_mps": summary.get("per_foot_max_near_ground_xy_speed_mps"),
        "per_foot_max_near_ground_xy_slip_m": summary.get("per_foot_max_near_ground_xy_slip_m"),
        "backend_carrier_claim": summary.get("backend_carrier_claim"),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
