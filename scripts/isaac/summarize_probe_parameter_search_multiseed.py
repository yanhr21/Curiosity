#!/usr/bin/env python3
"""Summarize multiple direct Isaac probe-parameter-search runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed parameter-search diagnostics.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/outputs/probe_parameter_search_carry"))
    parser.add_argument("--stamp", action="append", required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    runs = []
    failures: list[str] = []
    best_ids = []
    best_postures = []

    for stamp in args.stamp:
        path = args.run_root / stamp / "probe_parameter_search_carry_summary.json"
        if not path.exists():
            failures.append(f"{stamp}: missing summary {path}")
            runs.append({"stamp": stamp, "status": "missing", "summary_path": str(path)})
            continue
        summary = _load(path)
        best = summary.get("best_candidate") or {}
        if summary.get("status") != "pass":
            failures.append(f"{stamp}: summary status {summary.get('status')}")
        if not best:
            failures.append(f"{stamp}: missing best candidate")
        else:
            best_ids.append(str(best.get("candidate_id")))
            best_postures.append(str(best.get("carry_posture")))
        runs.append(
            {
                "stamp": stamp,
                "status": summary.get("status"),
                "summary_path": str(path),
                "box_seed": summary.get("box_seed"),
                "shared_randomized_box": summary.get("shared_randomized_box"),
                "probe_risk_score": (summary.get("probe") or {}).get("probe_risk_score"),
                "probe_load_risk_bucket": (summary.get("probe") or {}).get("probe_load_risk_bucket"),
                "candidate_count": summary.get("candidate_count"),
                "passed_candidate_count": summary.get("passed_candidate_count"),
                "best_candidate_id": best.get("candidate_id"),
                "best_carry_posture": best.get("carry_posture"),
                "best_score": best.get("score"),
                "best_final_box_target_distance_x_m": best.get("final_box_target_distance_x_m"),
                "best_fall_events": best.get("fall_events"),
                "best_box_drop_events": best.get("box_drop_events"),
                "best_min_drive_near_ground_foot_count": best.get("min_drive_near_ground_foot_count"),
                "best_drive_near_ground_lt2_steps": best.get("drive_near_ground_lt2_steps"),
                "rejected_candidate_ids": [
                    candidate.get("candidate_id")
                    for candidate in summary.get("candidates", [])
                    if not candidate.get("passed")
                ],
            }
        )

    best_id_counts = dict(Counter(best_ids))
    best_posture_counts = dict(Counter(best_postures))
    report = {
        "scene_type": "direct_isaac_probe_parameter_search_multiseed_diagnostic",
        "status": "pass" if not failures and len(runs) == len(args.stamp) else "fail",
        "success_claim": "multiseed_parameter_search_scaffold_not_rl_not_full_robot_success",
        "not_success_reason": (
            "multi-seed wrapper around a hand-authored parameter search on the current "
            "direct-Isaac support-foot scaffold; this is not a learned policy, not "
            "video-conditioned RL, and not complete humanoid walking or balance control"
        ),
        "seed_count": len(args.stamp),
        "completed_run_count": sum(1 for run in runs if run.get("status") == "pass"),
        "best_candidate_id_counts": best_id_counts,
        "best_carry_posture_counts": best_posture_counts,
        "best_candidate_varied": len(best_id_counts) > 1,
        "best_posture_varied": len(best_posture_counts) > 1,
        "runs": runs,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
