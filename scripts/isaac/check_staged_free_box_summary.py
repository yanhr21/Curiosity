#!/usr/bin/env python3
"""Check staged free-box carry smoke outputs.

This is a lightweight summary/log checker.  It does not run Isaac or load
simulation libraries, so it is safe to run on the login node after a compute
run has finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check staged free-box carry summary and log.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--max-target-distance", type=float, default=0.03)
    parser.add_argument("--max-relative-error", type=float, default=0.05)
    parser.add_argument("--max-peak-relative-error", type=float, default=None)
    parser.add_argument("--max-contact-proxy-gap", type=float, default=None)
    parser.add_argument("--min-body-travel", type=float, default=0.0)
    parser.add_argument("--min-box-travel", type=float, default=0.0)
    parser.add_argument("--min-support-margin", type=float, default=None)
    parser.add_argument("--min-support-margin-after-attach", type=float, default=None)
    parser.add_argument("--min-stance-count", type=float, default=None)
    parser.add_argument("--min-target-hold-steps", type=int, default=None)
    parser.add_argument("--max-body-z-deviation", type=float, default=None)
    parser.add_argument("--expect-body-vertical-mode", default=None)
    parser.add_argument("--expect-physical-support-mode", default=None)
    parser.add_argument("--expect-support-deck-gap", type=float, default=None)
    parser.add_argument("--expect-strategy", default=None)
    parser.add_argument("--expect-attachment-mode", default=None)
    parser.add_argument("--expect-carrier-mode", default=None)
    parser.add_argument("--expect-carry-geometry-mode", default=None)
    parser.add_argument("--require-contact-proxy", action="store_true")
    parser.add_argument("--require-dynamic-contact-proxy", action="store_true")
    parser.add_argument("--require-articulated-carrier", action="store_true")
    parser.add_argument("--require-no-root-shortcut", action="store_true")
    parser.add_argument("--max-body-root-velocity-commands", type=int, default=None)
    parser.add_argument("--max-body-root-pose-writes", type=int, default=None)
    parser.add_argument("--max-box-pose-writes", type=int, default=None)
    parser.add_argument("--require-attach", action="store_true")
    parser.add_argument("--forbid-disjoint-warning", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    if summary.get("error"):
        failures.append(f"summary error: {summary['error']}")
    if int(summary.get("completed_steps", 0)) < int(summary.get("steps_requested", 0)):
        failures.append(f"incomplete steps: {summary.get('completed_steps')} / {summary.get('steps_requested')}")
    if int(summary.get("fall_events", 0)) != 0:
        failures.append(f"fall events: {summary.get('fall_events')}")
    if int(summary.get("box_drop_events", 0)) != 0:
        failures.append(f"box drop events: {summary.get('box_drop_events')}")
    if args.require_attach and summary.get("attach_step") is None:
        failures.append("attach_step missing")
    if args.expect_strategy is not None and summary.get("strategy") != args.expect_strategy:
        failures.append(f"strategy mismatch: {summary.get('strategy')} != {args.expect_strategy}")
    if args.expect_attachment_mode is not None and summary.get("attachment_mode") != args.expect_attachment_mode:
        failures.append(f"attachment mode mismatch: {summary.get('attachment_mode')} != {args.expect_attachment_mode}")
    if args.expect_carrier_mode is not None and summary.get("carrier_mode") != args.expect_carrier_mode:
        failures.append(f"carrier mode mismatch: {summary.get('carrier_mode')} != {args.expect_carrier_mode}")
    if args.expect_carry_geometry_mode is not None and summary.get("carry_geometry_mode") != args.expect_carry_geometry_mode:
        failures.append(
            f"carry geometry mode mismatch: {summary.get('carry_geometry_mode')} != {args.expect_carry_geometry_mode}"
        )
    if args.expect_body_vertical_mode is not None and summary.get("body_vertical_mode") != args.expect_body_vertical_mode:
        failures.append(f"body vertical mode mismatch: {summary.get('body_vertical_mode')} != {args.expect_body_vertical_mode}")
    if args.expect_physical_support_mode is not None and summary.get("physical_support_mode") != args.expect_physical_support_mode:
        failures.append(f"physical support mode mismatch: {summary.get('physical_support_mode')} != {args.expect_physical_support_mode}")
    if args.expect_support_deck_gap is not None:
        support_deck_gap = summary.get("support_deck_gap_m")
        if support_deck_gap is None or abs(float(support_deck_gap) - args.expect_support_deck_gap) > 1e-6:
            failures.append(f"support deck gap mismatch: {support_deck_gap} != {args.expect_support_deck_gap}")
    if args.require_contact_proxy and not bool(summary.get("contact_proxy_enabled")):
        failures.append(f"contact proxy not enabled: {summary.get('contact_proxy_enabled')}")
    if args.require_dynamic_contact_proxy and not bool(summary.get("dynamic_contact_proxy_enabled")):
        failures.append(f"dynamic contact proxy not enabled: {summary.get('dynamic_contact_proxy_enabled')}")
    if args.require_articulated_carrier:
        if not bool(summary.get("articulated_carrier_enabled")):
            failures.append(f"articulated carrier not enabled: {summary.get('articulated_carrier_enabled')}")
        if int(summary.get("articulated_joint_count") or 0) <= 0:
            failures.append(f"articulated joint count too low: {summary.get('articulated_joint_count')}")
        if not bool(summary.get("foot_contact_drive_enabled")):
            failures.append(f"foot contact drive not enabled: {summary.get('foot_contact_drive_enabled')}")
    if args.require_no_root_shortcut:
        if not bool(summary.get("articulated_carrier_enabled")):
            failures.append("no-root shortcut gate requires articulated carrier")
        if int(summary.get("body_root_velocity_command_count") or 0) != 0:
            failures.append(f"body root velocity commands present: {summary.get('body_root_velocity_command_count')}")
        if int(summary.get("body_root_pose_write_count") or 0) != 0:
            failures.append(f"body root pose writes present: {summary.get('body_root_pose_write_count')}")
        if int(summary.get("box_pose_write_count") or 0) != 0:
            failures.append(f"box pose writes present: {summary.get('box_pose_write_count')}")
    if args.max_body_root_velocity_commands is not None:
        body_velocity_commands = int(summary.get("body_root_velocity_command_count") or 0)
        if body_velocity_commands > args.max_body_root_velocity_commands:
            failures.append(f"body root velocity commands too high: {body_velocity_commands}")
    if args.max_body_root_pose_writes is not None:
        body_pose_writes = int(summary.get("body_root_pose_write_count") or 0)
        if body_pose_writes > args.max_body_root_pose_writes:
            failures.append(f"body root pose writes too high: {body_pose_writes}")
    if args.max_box_pose_writes is not None:
        box_pose_writes = int(summary.get("box_pose_write_count") or 0)
        if box_pose_writes > args.max_box_pose_writes:
            failures.append(f"box pose writes too high: {box_pose_writes}")
    if float(summary.get("body_travel_x_m", 0.0)) < args.min_body_travel:
        failures.append(f"body travel too low: {summary.get('body_travel_x_m')}")
    if float(summary.get("box_travel_x_m", 0.0)) < args.min_box_travel:
        failures.append(f"box travel too low: {summary.get('box_travel_x_m')}")
    if args.min_support_margin is not None:
        support_margin = summary.get("min_support_margin_m")
        if support_margin is None or float(support_margin) < args.min_support_margin:
            failures.append(f"support margin too low: {support_margin}")
    if args.min_support_margin_after_attach is not None:
        support_margin_after_attach = summary.get("min_support_margin_after_attach_m")
        if support_margin_after_attach is None or float(support_margin_after_attach) < args.min_support_margin_after_attach:
            failures.append(f"post-attach support margin too low: {support_margin_after_attach}")
    if args.min_stance_count is not None:
        stance_count = summary.get("min_stance_count")
        if stance_count is None or float(stance_count) < args.min_stance_count:
            failures.append(f"stance count too low: {stance_count}")
    if args.min_target_hold_steps is not None:
        target_hold_steps = summary.get("target_hold_steps")
        if target_hold_steps is None or int(target_hold_steps) < args.min_target_hold_steps:
            failures.append(f"target hold steps too low: {target_hold_steps}")
    if args.max_body_z_deviation is not None:
        body_z_deviation = summary.get("max_body_z_deviation_m")
        if body_z_deviation is None or float(body_z_deviation) > args.max_body_z_deviation:
            failures.append(f"body z deviation too high: {body_z_deviation}")

    target_dist = summary.get("final_box_target_distance_xy_m")
    if target_dist is None or float(target_dist) > args.max_target_distance:
        failures.append(f"target distance too high: {target_dist}")

    rel_err = summary.get("box_relative_error_m_after_attach")
    if rel_err is None or float(rel_err) > args.max_relative_error:
        failures.append(f"post-attach relative error too high: {rel_err}")
    peak_rel_err = summary.get("max_box_relative_error_m_after_attach")
    if args.max_peak_relative_error is not None:
        if peak_rel_err is None or float(peak_rel_err) > args.max_peak_relative_error:
            failures.append(f"peak post-attach relative error too high: {peak_rel_err}")
    grip_gap = summary.get("contact_proxy_grip_gap_m")
    peak_grip_gap = summary.get("max_contact_proxy_grip_gap_m")
    if args.max_contact_proxy_gap is not None:
        if peak_grip_gap is None or float(peak_grip_gap) > args.max_contact_proxy_gap:
            failures.append(f"contact proxy grip gap too high: {peak_grip_gap}")

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
        "attach_step": summary.get("attach_step"),
        "strategy": summary.get("strategy"),
        "attachment_mode": summary.get("attachment_mode"),
        "carrier_mode": summary.get("carrier_mode"),
        "carrier_evidence_mode": summary.get("carrier_evidence_mode"),
        "articulated_carrier_requested": summary.get("articulated_carrier_requested"),
        "articulated_carrier_enabled": summary.get("articulated_carrier_enabled"),
        "articulated_joint_count": summary.get("articulated_joint_count"),
        "foot_contact_drive_enabled": summary.get("foot_contact_drive_enabled"),
        "body_root_velocity_command_count": summary.get("body_root_velocity_command_count"),
        "body_root_pose_write_count": summary.get("body_root_pose_write_count"),
        "box_pose_write_count": summary.get("box_pose_write_count"),
        "box_velocity_command_count": summary.get("box_velocity_command_count"),
        "carry_geometry_mode": summary.get("carry_geometry_mode"),
        "carry_clearance_m": summary.get("carry_clearance_m"),
        "carry_z_offset_m": summary.get("carry_z_offset_m"),
        "contact_proxy_gain": summary.get("contact_proxy_gain"),
        "contact_proxy_max_speed": summary.get("contact_proxy_max_speed"),
        "actual_staged_carry_x_m": summary.get("actual_staged_carry_x_m"),
        "approach_body_x_m": summary.get("approach_body_x_m"),
        "body_vertical_mode": summary.get("body_vertical_mode"),
        "body_height_gain": summary.get("body_height_gain"),
        "body_height_max_z_speed": summary.get("body_height_max_z_speed"),
        "physical_support_mode": summary.get("physical_support_mode"),
        "support_deck_gap_m": summary.get("support_deck_gap_m"),
        "body_vertical_velocity_preserve_available": summary.get("body_vertical_velocity_preserve_available"),
        "contact_proxy_enabled": summary.get("contact_proxy_enabled"),
        "dynamic_contact_proxy_enabled": summary.get("dynamic_contact_proxy_enabled"),
        "target_hold_steps": summary.get("target_hold_steps"),
        "target_hold_latched": summary.get("target_hold_latched"),
        "target_hold_radius_m": summary.get("target_hold_radius_m"),
        "target_slow_radius_m": summary.get("target_slow_radius_m"),
        "target_body_margin_m": summary.get("target_body_margin_m"),
        "target_body_x_m": summary.get("target_body_x_m"),
        "carry_phase_steps": summary.get("carry_phase_steps"),
        "final_box_target_distance_xy_m": target_dist,
        "box_relative_error_m_after_attach": rel_err,
        "max_box_relative_error_m_after_attach": peak_rel_err,
        "contact_proxy_grip_gap_m": grip_gap,
        "max_contact_proxy_grip_gap_m": peak_grip_gap,
        "body_travel_x_m": summary.get("body_travel_x_m"),
        "box_travel_x_m": summary.get("box_travel_x_m"),
        "min_support_margin_m": summary.get("min_support_margin_m"),
        "min_support_margin_after_attach_m": summary.get("min_support_margin_after_attach_m"),
        "min_stance_count": summary.get("min_stance_count"),
        "max_command_speed_mps": summary.get("max_command_speed_mps"),
        "initial_body_z_m": summary.get("initial_body_z_m"),
        "min_body_z_m": summary.get("min_body_z_m"),
        "max_body_z_deviation_m": summary.get("max_body_z_deviation_m"),
        "fall_events": summary.get("fall_events"),
        "box_drop_events": summary.get("box_drop_events"),
        "disjoint_warning": disjoint_warning,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
