#!/usr/bin/env python3
"""Select a direct carry posture from probe telemetry.

This is a scaffold selector: it uses only logged probe belief fields, not
hidden payload mass/shape ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select carry posture from direct probe summary.")
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--medium-threshold", type=float, default=0.58)
    parser.add_argument("--high-threshold", type=float, default=0.75)
    parser.add_argument("--low-posture", default="front_reach")
    parser.add_argument("--medium-posture", default="close_mid")
    parser.add_argument("--high-posture", default="chest_high")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.probe_summary.read_text())
    failures: list[str] = []

    if not bool(summary.get("probe_belief_available")):
        failures.append("probe_belief_available is false or missing")
    if bool(summary.get("probe_belief_uses_hidden_ground_truth")):
        failures.append("probe_belief_uses_hidden_ground_truth is true")
    risk_raw = summary.get("probe_risk_score")
    if risk_raw is None:
        failures.append("probe_risk_score is missing")
        risk = 0.0
    else:
        risk = float(risk_raw)

    medium_threshold = float(args.medium_threshold)
    high_threshold = float(args.high_threshold)
    if high_threshold <= medium_threshold:
        failures.append("high_threshold must be greater than medium_threshold")

    if risk >= high_threshold:
        selected = str(args.high_posture)
        bucket = "high"
    elif risk >= medium_threshold:
        selected = str(args.medium_posture)
        bucket = "medium"
    else:
        selected = str(args.low_posture)
        bucket = "low"

    report = {
        "scene_type": "direct_isaac_probe_selected_posture_scaffold",
        "success_claim": "probe_selected_posture_diagnostic_not_final_humanoid_or_rl_success",
        "probe_summary": str(args.probe_summary),
        "probe_belief_available": bool(summary.get("probe_belief_available")),
        "probe_belief_uses_hidden_ground_truth": bool(summary.get("probe_belief_uses_hidden_ground_truth")),
        "probe_belief_source": summary.get("probe_belief_source"),
        "probe_mode": summary.get("probe_mode"),
        "probe_steps_requested": summary.get("probe_steps_requested"),
        "probe_risk_score": risk_raw,
        "probe_load_risk_bucket": summary.get("probe_load_risk_bucket"),
        "thresholds": {
            "medium": medium_threshold,
            "high": high_threshold,
        },
        "selected_bucket": bucket,
        "selected_carry_posture": selected,
        "selection_uses_hidden_ground_truth": False,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
