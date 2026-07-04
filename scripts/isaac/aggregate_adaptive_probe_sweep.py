#!/usr/bin/env python3
"""Aggregate adaptive-probe carry scaffold sweep summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate adaptive carry sweep outputs.")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = []
    failures = []
    for summary_path in sorted(args.sweep_dir.glob("*/adaptive_probe_carry_scene_summary.json")):
        case_name = summary_path.parent.name
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:  # noqa: BLE001
            failures.append({"case": case_name, "summary_path": str(summary_path), "error": repr(exc)})
            continue

        strategy = summary.get("selected_strategy", {})
        belief = summary.get("belief", {})
        cases.append(
            {
                "case": case_name,
                "summary_path": str(summary_path),
                "completed_steps": summary.get("completed_steps"),
                "steps_requested": summary.get("steps_requested"),
                "box_mass_kg": summary.get("box_mass_kg"),
                "box_size_m": summary.get("box_size_m"),
                "box_com_offset_m": summary.get("box_com_offset_m"),
                "robot_height_m": summary.get("robot_height_m"),
                "robot_mass_kg": summary.get("robot_mass_kg"),
                "arm_length_m": summary.get("arm_length_m"),
                "max_payload_kg": summary.get("max_payload_kg"),
                "estimated_mass_kg": belief.get("estimated_mass_kg"),
                "estimated_com_x_m": belief.get("estimated_com_x_m"),
                "strategy": strategy.get("name"),
                "mass_ratio": strategy.get("mass_ratio"),
                "arm_reach_ratio": strategy.get("arm_reach_ratio"),
                "final_phase": summary.get("final_phase"),
                "box_drop_events": summary.get("box_drop_events"),
                "final_box_target_distance_xy_m": summary.get("final_box_target_distance_xy_m"),
                "max_box_travel_xy_m": summary.get("max_box_travel_xy_m"),
                "min_support_margin_m": summary.get("min_support_margin_m"),
                "energy_proxy": summary.get("energy_proxy"),
                "success_claim": summary.get("success_claim"),
            }
        )

    strategy_counts: dict[str, int] = {}
    completed_cases = 0
    drop_cases = 0
    target_cases = 0
    margins = []
    for case in cases:
        strategy = str(case.get("strategy"))
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        if case.get("completed_steps") == case.get("steps_requested"):
            completed_cases += 1
        if float(case.get("box_drop_events") or 0) > 0:
            drop_cases += 1
        target_distance = case.get("final_box_target_distance_xy_m")
        if target_distance is not None and float(target_distance) <= 0.08:
            target_cases += 1
        margin = case.get("min_support_margin_m")
        if margin is not None:
            margins.append(float(margin))

    aggregate = {
        "sweep_type": "adaptive_direct_isaac_carry_scaffold_parameter_sweep",
        "success_claim": "diagnostic_only_kinematic_proxy_not_dynamic_robot_or_learned_policy",
        "case_count": len(cases),
        "failed_summary_count": len(failures),
        "completed_case_count": completed_cases,
        "drop_case_count": drop_cases,
        "target_reached_case_count_threshold_0p08m": target_cases,
        "strategy_counts": strategy_counts,
        "min_support_margin_m_over_cases": min(margins) if margins else None,
        "cases": cases,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Wrote aggregate summary: {args.output}")


if __name__ == "__main__":
    main()
