#!/usr/bin/env python3
"""Export probe parameter-search summaries as RL-interface episode rows.

This exporter does not train RL.  It converts the current transparent search
summaries into a stable JSONL contract: observation proxies, action
parameters, reward terms, strict gates, and limitation labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export parameter-search candidate episodes to JSONL.")
    parser.add_argument("--summary", action="append", type=Path, default=[], help="Per-seed parameter-search summary.")
    parser.add_argument(
        "--multiseed-summary",
        action="append",
        type=Path,
        default=[],
        help="Multi-seed summary whose run entries point to per-seed summaries.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_summary_paths(args: argparse.Namespace) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in args.summary:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield path
    for multiseed_path in args.multiseed_summary:
        multiseed = _load(multiseed_path)
        for run in multiseed.get("runs", []):
            summary_path = run.get("summary_path")
            if not summary_path:
                continue
            path = Path(summary_path)
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def _score_terms(candidate: dict) -> dict:
    terms = dict(candidate.get("score_terms") or {})
    if not terms:
        terms = {
            "final_distance": candidate.get("final_box_target_distance_x_m"),
            "support_penalty": 0.0 if candidate.get("passed") else 10.0,
            "fall_penalty": 100.0 * float(candidate.get("fall_events") or 0),
            "drop_penalty": 50.0 * float(candidate.get("box_drop_events") or 0),
        }
    return terms


def _reward_from_score(score: object) -> float | None:
    if score is None:
        return None
    try:
        return -float(score)
    except (TypeError, ValueError):
        return None


def _episode_row(summary_path: Path, summary: dict, candidate: dict) -> dict:
    probe = summary.get("probe") or {}
    shared_box = summary.get("shared_randomized_box") or {}
    score_terms = _score_terms(candidate)
    action = {
        "carry_posture": candidate.get("carry_posture"),
        "stance_steps": candidate.get("stance_steps"),
        "step_length_m": candidate.get("step_length_m"),
        "support_foot_stance_x_m": candidate.get("support_foot_stance_x_m"),
        "support_foot_swing_x_m": candidate.get("support_foot_swing_x_m"),
        "support_foot_step_height_m": candidate.get("support_foot_step_height_m"),
        "support_foot_double_support_fraction": candidate.get("support_foot_double_support_fraction"),
        "torso_z_m": candidate.get("torso_z_m"),
        "payload_local_x_m": candidate.get("payload_local_x_m"),
        "payload_local_z_m": candidate.get("payload_local_z_m"),
        "stance_half_length_m": candidate.get("stance_half_length_m"),
        "stance_half_width_m": candidate.get("stance_half_width_m"),
    }
    observation = {
        "box_seed": summary.get("box_seed"),
        "box_mass_kg": shared_box.get("box_mass_kg"),
        "box_size_m": shared_box.get("box_size_m"),
        "box_com_offset_m": shared_box.get("box_com_offset_m"),
        "probe_mode": probe.get("probe_mode"),
        "probe_risk_score": probe.get("probe_risk_score"),
        "probe_load_risk_bucket": probe.get("probe_load_risk_bucket"),
        "probe_recommended_carry_adjustment": probe.get("probe_recommended_carry_adjustment"),
        "probe_belief_uses_hidden_ground_truth": probe.get("probe_belief_uses_hidden_ground_truth"),
    }
    gates = {
        "passed": bool(candidate.get("passed")),
        "strict_check_status": candidate.get("check_status"),
        "wrapper_status": candidate.get("wrapper_status"),
        "fall_events": candidate.get("fall_events"),
        "box_drop_events": candidate.get("box_drop_events"),
        "root_shortcut_free": candidate.get("root_shortcut_free"),
        "stance_anchor_fixed_to_world": candidate.get("stance_anchor_fixed_to_world"),
        "min_drive_near_ground_foot_count": candidate.get("min_drive_near_ground_foot_count"),
        "drive_near_ground_zero_steps": candidate.get("drive_near_ground_zero_steps"),
        "drive_near_ground_lt2_steps": candidate.get("drive_near_ground_lt2_steps"),
        "min_commanded_stance_near_ground_foot_count": candidate.get(
            "min_commanded_stance_near_ground_foot_count"
        ),
        "commanded_stance_near_ground_lt2_steps": candidate.get("commanded_stance_near_ground_lt2_steps"),
    }
    metrics = {
        "completed_steps": candidate.get("completed_steps"),
        "score": candidate.get("score"),
        "reward_proxy": _reward_from_score(candidate.get("score")),
        "score_terms": score_terms,
        "max_box_travel_x_m": candidate.get("max_box_travel_x_m"),
        "final_box_target_distance_x_m": candidate.get("final_box_target_distance_x_m"),
        "final_post_settle_box_travel_x_m": candidate.get("final_post_settle_box_travel_x_m"),
        "post_settle_box_travel_loss_after_peak_m": candidate.get("post_settle_box_travel_loss_after_peak_m"),
        "max_actual_support_foot_lift_m": candidate.get("max_actual_support_foot_lift_m"),
        "min_support_polygon_margin_m": candidate.get("min_support_polygon_margin_m"),
    }
    return {
        "schema_version": "direct_isaac_probe_parameter_episode_v1",
        "source_summary": str(summary_path),
        "candidate_id": candidate.get("candidate_id"),
        "selector_type": summary.get("selector_type"),
        "success_claim": summary.get("success_claim"),
        "not_success_reason": summary.get("not_success_reason"),
        "observation": observation,
        "action": action,
        "metrics": metrics,
        "gates": gates,
        "candidate_summary_path": candidate.get("summary_path"),
    }


def main() -> int:
    args = parse_args()
    rows = []
    for summary_path in _iter_summary_paths(args):
        summary = _load(summary_path)
        for candidate in summary.get("candidates", []):
            rows.append(_episode_row(summary_path, summary, candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "episode_count": len(rows)}, indent=2, sort_keys=True))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
