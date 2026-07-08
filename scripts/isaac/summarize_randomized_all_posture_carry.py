#!/usr/bin/env python3
"""Summarize randomized all-posture strict carry diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POSTURES = ("front_mid", "low_front", "chest_high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize randomized all-posture carry results.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-seed", type=int, required=True)
    return parser.parse_args()


def _load_summary(root: Path, posture: str) -> dict:
    path = root / posture / "direct_carry_task_physical_backend_summary.json"
    data = json.loads(path.read_text())
    data["_summary_path"] = str(path)
    return data


def _posture_passed(data: dict) -> bool:
    required = [
        int(data.get("completed_steps") or 0) >= 3560,
        bool(data.get("box_randomized")),
        int(data.get("fall_events") or 0) == 0,
        int(data.get("box_drop_events") or 0) == 0,
        bool(data.get("root_shortcut_free")),
        not bool(data.get("stance_anchor_fixed_to_world")),
        int(data.get("support_root_pose_write_count") or 0) == 0,
        int(data.get("anchor_world_joint_retarget_count") or 0) == 0,
        int(data.get("foot_pose_write_count") or 0) == 0,
        int(data.get("stance_anchor_pose_write_count") or 0) == 0,
        int(data.get("min_drive_near_ground_foot_count") or 0) >= 2,
        int(data.get("drive_near_ground_zero_steps") or 0) == 0,
        int(data.get("drive_near_ground_lt2_steps") or 0) == 0,
        bool(data.get("support_foot_contact_report_available")),
        int(data.get("min_drive_contact_report_foot_count") or 0) >= 2,
        int(data.get("drive_contact_report_zero_steps") or 0) == 0,
        int(data.get("drive_contact_report_lt2_steps") or 0) == 0,
        int(data.get("min_commanded_stance_contact_report_foot_count") or 0) >= 2,
        int(data.get("commanded_stance_contact_report_lt2_steps") or 0) == 0,
        int(data.get("min_commanded_stance_near_ground_foot_count") or 0) >= 2,
        int(data.get("commanded_stance_near_ground_lt2_steps") or 0) == 0,
        float(data.get("max_box_travel_x_m") or 0.0) >= 0.52,
        float(data.get("final_box_target_distance_x_m") or 999.0) <= 0.18,
    ]
    return all(required)


def main() -> int:
    args = parse_args()
    postures = []
    failures = []
    reference_box = None
    for posture in POSTURES:
        try:
            data = _load_summary(args.root, posture)
        except FileNotFoundError as exc:
            failures.append(f"{posture}: missing summary {exc}")
            continue
        box_signature = {
            "box_seed": data.get("box_seed"),
            "box_mass_kg": data.get("box_mass_kg"),
            "box_size_m": data.get("box_size_m"),
            "box_com_offset_m": data.get("box_com_offset_m"),
        }
        if reference_box is None:
            reference_box = box_signature
        elif box_signature != reference_box:
            failures.append(f"{posture}: randomized box signature differs from first posture")
        if int(data.get("box_seed") or -1) != int(args.box_seed):
            failures.append(f"{posture}: box_seed {data.get('box_seed')} != {args.box_seed}")
        passed = _posture_passed(data)
        if not passed:
            failures.append(f"{posture}: posture strict gate failed")
        postures.append(
            {
                "posture": posture,
                "passed": passed,
                "summary_path": data["_summary_path"],
                "completed_steps": data.get("completed_steps"),
                "box_randomized": data.get("box_randomized"),
                "box_seed": data.get("box_seed"),
                "box_mass_kg": data.get("box_mass_kg"),
                "box_size_m": data.get("box_size_m"),
                "box_com_offset_m": data.get("box_com_offset_m"),
                "max_box_travel_x_m": data.get("max_box_travel_x_m"),
                "final_box_target_distance_x_m": data.get("final_box_target_distance_x_m"),
                "final_post_settle_box_travel_x_m": data.get("final_post_settle_box_travel_x_m"),
                "fall_events": data.get("fall_events"),
                "box_drop_events": data.get("box_drop_events"),
                "root_shortcut_free": data.get("root_shortcut_free"),
                "stance_anchor_fixed_to_world": data.get("stance_anchor_fixed_to_world"),
                "min_drive_near_ground_foot_count": data.get("min_drive_near_ground_foot_count"),
                "drive_near_ground_zero_steps": data.get("drive_near_ground_zero_steps"),
                "drive_near_ground_lt2_steps": data.get("drive_near_ground_lt2_steps"),
                "support_foot_contact_report_available": data.get("support_foot_contact_report_available"),
                "support_foot_contact_report_event_count": data.get("support_foot_contact_report_event_count"),
                "support_foot_contact_report_error_count": data.get("support_foot_contact_report_error_count"),
                "per_foot_contact_report_steps": data.get("per_foot_contact_report_steps"),
                "min_drive_contact_report_foot_count": data.get("min_drive_contact_report_foot_count"),
                "drive_contact_report_zero_steps": data.get("drive_contact_report_zero_steps"),
                "drive_contact_report_lt2_steps": data.get("drive_contact_report_lt2_steps"),
                "min_commanded_stance_contact_report_foot_count": data.get(
                    "min_commanded_stance_contact_report_foot_count"
                ),
                "commanded_stance_contact_report_lt2_steps": data.get(
                    "commanded_stance_contact_report_lt2_steps"
                ),
                "min_commanded_stance_near_ground_foot_count": data.get(
                    "min_commanded_stance_near_ground_foot_count"
                ),
                "commanded_stance_near_ground_lt2_steps": data.get(
                    "commanded_stance_near_ground_lt2_steps"
                ),
                "max_actual_support_foot_lift_m": data.get("max_actual_support_foot_lift_m"),
                "min_support_polygon_margin_m": data.get("min_support_polygon_margin_m"),
            }
        )
    report = {
        "scene_type": "direct_isaac_randomized_all_posture_strict_carry_diagnostic",
        "status": "pass" if not failures and len(postures) == len(POSTURES) else "fail",
        "success_claim": "all_posture_strict_support_diagnostic_not_full_robot_success",
        "not_success_reason": (
            "current carrier is still a direct-Isaac support-foot scaffold, not "
            "a learned policy or complete humanoid walking controller"
        ),
        "box_seed": int(args.box_seed),
        "shared_randomized_box": reference_box,
        "postures": postures,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
