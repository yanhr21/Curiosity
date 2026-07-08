#!/usr/bin/env python3
"""Summarize a direct Isaac probe plus posture/gait parameter search.

This is a transparent scaffold search report. It is not RL, not a learned
policy, and not evidence for full humanoid walking or video-conditioned
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize probe parameter-search carry diagnostic.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--box-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_score_terms(summary: dict) -> dict[str, float]:
    final_distance = float(summary.get("final_box_target_distance_x_m") or 999.0)
    travel_loss = float(summary.get("post_settle_box_travel_loss_after_peak_m") or 0.0)
    falls = int(summary.get("fall_events") or 0)
    drops = int(summary.get("box_drop_events") or 0)
    support_penalty = 0.0
    if int(summary.get("min_drive_near_ground_foot_count") or 0) < 2:
        support_penalty += 10.0
    if int(summary.get("drive_near_ground_lt2_steps") or 0) > 0:
        support_penalty += 10.0
    if not bool(summary.get("root_shortcut_free")):
        support_penalty += 10.0
    if bool(summary.get("stance_anchor_fixed_to_world")):
        support_penalty += 10.0

    tilt_proxy = _as_float(summary.get("max_tilt_rad"))
    margin = _as_float(summary.get("min_support_polygon_margin_m"), default=0.16)
    margin_shortfall = max(0.0, 0.16 - margin)
    support_lift_proxy = _as_float(summary.get("max_actual_support_foot_lift_m"))
    support_motion_proxy = _as_float(summary.get("max_support_foot_x_joint_motion_m")) + _as_float(
        summary.get("max_support_foot_z_joint_motion_m")
    )
    measured_effort_proxy = (
        _as_float(summary.get("mean_probe_support_foot_x_measured_effort"))
        + _as_float(summary.get("mean_probe_support_foot_z_measured_effort"))
        + 0.25 * _as_float(summary.get("max_probe_support_foot_x_measured_effort"))
        + 0.25 * _as_float(summary.get("max_probe_support_foot_z_measured_effort"))
    )
    effort_proxy_penalty = 1.0e-5 * measured_effort_proxy
    kinematic_effort_proxy_penalty = (
        0.50 * tilt_proxy
        + 2.0 * margin_shortfall
        + 0.05 * support_lift_proxy
        + 0.02 * support_motion_proxy
    )

    return {
        "final_distance": final_distance,
        "travel_loss_penalty": 0.25 * travel_loss,
        "fall_penalty": 100.0 * falls,
        "drop_penalty": 50.0 * drops,
        "support_penalty": support_penalty,
        "effort_proxy_penalty": effort_proxy_penalty,
        "kinematic_effort_proxy_penalty": kinematic_effort_proxy_penalty,
        "tilt_proxy_rad": tilt_proxy,
        "support_margin_shortfall_m": margin_shortfall,
        "support_lift_proxy_m": support_lift_proxy,
        "support_motion_proxy_m": support_motion_proxy,
        "measured_effort_proxy": measured_effort_proxy,
    }


def _candidate_score(summary: dict) -> float:
    return sum(
        value
        for key, value in _candidate_score_terms(summary).items()
        if key.endswith("_penalty") or key == "final_distance"
    )


def _strict_pass(summary: dict, check_report: dict | None) -> bool:
    if check_report is not None and check_report.get("status") != "pass":
        return False
    required = [
        int(summary.get("completed_steps") or 0) >= 3560,
        bool(summary.get("box_randomized")),
        int(summary.get("fall_events") or 0) == 0,
        int(summary.get("box_drop_events") or 0) == 0,
        bool(summary.get("root_shortcut_free")),
        not bool(summary.get("stance_anchor_fixed_to_world")),
        int(summary.get("support_root_pose_write_count") or 0) == 0,
        int(summary.get("anchor_world_joint_retarget_count") or 0) == 0,
        int(summary.get("foot_pose_write_count") or 0) == 0,
        int(summary.get("stance_anchor_pose_write_count") or 0) == 0,
        int(summary.get("min_drive_near_ground_foot_count") or 0) >= 2,
        int(summary.get("drive_near_ground_zero_steps") or 0) == 0,
        int(summary.get("drive_near_ground_lt2_steps") or 0) == 0,
        int(summary.get("min_commanded_stance_near_ground_foot_count") or 0) >= 2,
        int(summary.get("commanded_stance_near_ground_lt2_steps") or 0) == 0,
        float(summary.get("max_box_travel_x_m") or 0.0) >= 0.52,
        float(summary.get("final_box_target_distance_x_m") or 999.0) <= 0.18,
    ]
    return all(required)


def main() -> int:
    args = parse_args()
    probe = _load_json(args.probe_summary)
    candidate_root = args.root / "candidates"
    candidates = []
    failures: list[str] = []

    for config_path in sorted(candidate_root.glob("*/candidate_config.json")):
        case_dir = config_path.parent
        config = _load_json(config_path)
        summary_path = case_dir / "direct_carry_task_physical_backend_summary.json"
        check_path = case_dir / "strict_check_report.json"
        wrapper_status_path = case_dir / "wrapper_status.txt"
        check_status_path = case_dir / "check_status.txt"
        if not summary_path.exists():
            failures.append(f"{config['candidate_id']}: missing direct summary")
            candidates.append(
                {
                    **config,
                    "passed": False,
                    "failure": "missing direct summary",
                    "summary_path": str(summary_path),
                }
            )
            continue
        summary = _load_json(summary_path)
        check_report = _load_json(check_path) if check_path.exists() else None
        passed = _strict_pass(summary, check_report)
        if int(summary.get("box_seed") or -1) != int(args.box_seed):
            passed = False
            failures.append(f"{config['candidate_id']}: box seed mismatch")
        if not bool(summary.get("box_randomized")):
            passed = False
            failures.append(f"{config['candidate_id']}: box was not randomized")
        if not passed:
            failures.append(f"{config['candidate_id']}: strict candidate gate failed")

        wrapper_status = wrapper_status_path.read_text().strip() if wrapper_status_path.exists() else None
        check_status = check_status_path.read_text().strip() if check_status_path.exists() else None
        candidates.append(
            {
                **config,
                "passed": passed,
                "score": _candidate_score(summary),
                "score_terms": _candidate_score_terms(summary),
                "summary_path": str(summary_path),
                "check_report_path": str(check_path) if check_path.exists() else None,
                "wrapper_status": wrapper_status,
                "check_status": check_status,
                "completed_steps": summary.get("completed_steps"),
                "box_seed": summary.get("box_seed"),
                "box_mass_kg": summary.get("box_mass_kg"),
                "box_size_m": summary.get("box_size_m"),
                "box_com_offset_m": summary.get("box_com_offset_m"),
                "torso_z_m": summary.get("torso_z_m", config.get("torso_z_m")),
                "payload_local_x_m": summary.get("payload_local_x_m", config.get("payload_local_x_m")),
                "payload_local_z_m": summary.get("payload_local_z_m", config.get("payload_local_z_m")),
                "stance_half_length_m": summary.get("stance_half_length_m", config.get("stance_half_length_m")),
                "stance_half_width_m": summary.get("stance_half_width_m", config.get("stance_half_width_m")),
                "max_box_travel_x_m": summary.get("max_box_travel_x_m"),
                "final_box_target_distance_x_m": summary.get("final_box_target_distance_x_m"),
                "final_post_settle_box_travel_x_m": summary.get("final_post_settle_box_travel_x_m"),
                "post_settle_box_travel_loss_after_peak_m": summary.get(
                    "post_settle_box_travel_loss_after_peak_m"
                ),
                "fall_events": summary.get("fall_events"),
                "box_drop_events": summary.get("box_drop_events"),
                "root_shortcut_free": summary.get("root_shortcut_free"),
                "stance_anchor_fixed_to_world": summary.get("stance_anchor_fixed_to_world"),
                "min_drive_near_ground_foot_count": summary.get("min_drive_near_ground_foot_count"),
                "drive_near_ground_zero_steps": summary.get("drive_near_ground_zero_steps"),
                "drive_near_ground_lt2_steps": summary.get("drive_near_ground_lt2_steps"),
                "min_commanded_stance_near_ground_foot_count": summary.get(
                    "min_commanded_stance_near_ground_foot_count"
                ),
                "commanded_stance_near_ground_lt2_steps": summary.get(
                    "commanded_stance_near_ground_lt2_steps"
                ),
                "max_actual_support_foot_lift_m": summary.get("max_actual_support_foot_lift_m"),
                "support_foot_step_height_m": summary.get(
                    "support_foot_step_height_m", config.get("support_foot_step_height_m")
                ),
                "support_foot_double_support_fraction": summary.get(
                    "support_foot_double_support_fraction",
                    config.get("support_foot_double_support_fraction"),
                ),
                "min_support_polygon_margin_m": summary.get("min_support_polygon_margin_m"),
            }
        )

    passed_candidates = [candidate for candidate in candidates if candidate.get("passed")]
    best = min(passed_candidates, key=lambda item: float(item["score"])) if passed_candidates else None
    if not passed_candidates:
        failures.append("no candidate passed strict support gate")

    reference_box = None
    for candidate in candidates:
        if candidate.get("box_seed") is None:
            continue
        signature = {
            "box_seed": candidate.get("box_seed"),
            "box_mass_kg": candidate.get("box_mass_kg"),
            "box_size_m": candidate.get("box_size_m"),
            "box_com_offset_m": candidate.get("box_com_offset_m"),
        }
        if reference_box is None:
            reference_box = signature
        elif signature != reference_box:
            failures.append(f"{candidate['candidate_id']}: randomized box signature differs from first candidate")

    report = {
        "scene_type": "direct_isaac_probe_parameter_search_carry_diagnostic",
        "status": "pass" if passed_candidates else "fail",
        "success_claim": "parameter_search_scaffold_not_rl_not_full_robot_success",
        "not_success_reason": (
            "uses a hand-authored candidate search over the current direct-Isaac "
            "support-foot scaffold; this is not a learned policy, not video-conditioned "
            "RL, and not complete humanoid walking or balance control"
        ),
        "selector_type": "transparent_parameter_search_over_hand_authored_candidates",
        "selection_score": (
            "final_box_target_distance_x_m + 0.25 * post_settle_box_travel_loss_after_peak_m "
            "+ effort-aware proxy terms from measured support-foot effort when available, "
            "tilt, support-margin shortfall, support-foot lift/motion, and large penalties "
            "for falls, drops, support discontinuity, or shortcuts"
        ),
        "box_seed": int(args.box_seed),
        "shared_randomized_box": reference_box,
        "probe_summary": str(args.probe_summary),
        "probe": {
            "completed_steps": probe.get("completed_steps"),
            "box_seed": probe.get("box_seed"),
            "box_randomized": probe.get("box_randomized"),
            "box_mass_kg": probe.get("box_mass_kg"),
            "box_size_m": probe.get("box_size_m"),
            "box_com_offset_m": probe.get("box_com_offset_m"),
            "probe_mode": probe.get("probe_mode"),
            "probe_belief_available": probe.get("probe_belief_available"),
            "probe_belief_source": probe.get("probe_belief_source"),
            "probe_belief_uses_hidden_ground_truth": probe.get("probe_belief_uses_hidden_ground_truth"),
            "probe_risk_score": probe.get("probe_risk_score"),
            "probe_load_risk_bucket": probe.get("probe_load_risk_bucket"),
            "probe_recommended_carry_adjustment": probe.get("probe_recommended_carry_adjustment"),
        },
        "best_candidate": best,
        "candidate_count": len(candidates),
        "passed_candidate_count": len(passed_candidates),
        "candidates": candidates,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
