#!/usr/bin/env python3
"""Summarize a predeclared failure-rich transition overfit curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--learning-result", type=Path, action="append", required=True)
parser.add_argument("--checkpoint-audit", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def main() -> None:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.learning_result
    ]
    iterations = [64, 128, 192, 256]
    if [record.get("learned_iteration") for record in records] != iterations:
        raise RuntimeError("overfit endpoints must be exactly 64/128/192/256")
    for record in records:
        if (
            record.get("protocol")
            != "sugar_shared_frozen_expert_transition_learning_v1"
            or record.get("training_seed") != 181630
            or record.get("evaluation_seed") != 181630
            or record.get("checks", {}).get("initial_physics_elementwise_identical")
            is not True
        ):
            raise RuntimeError("invalid failure-rich overfit learning record")
    audit = json.loads(args.checkpoint_audit.read_text(encoding="utf-8"))
    if audit.get("overall_pass") is not True or audit.get("post_iteration") != 256:
        raise RuntimeError("failure-rich checkpoint audit failed")

    pre = records[0]["exact_pre_update_kick"]
    if any(record["exact_pre_update_kick"] != pre for record in records[1:]):
        raise RuntimeError("pre-update baseline drift across endpoints")
    rows = []
    for record in records:
        learned = record["learned_kick"]
        improvement = bool(
            learned["physical_fall_count"] < pre["physical_fall_count"]
            and learned["safe_kick_success_count"] >= pre["safe_kick_success_count"]
        )
        rows.append(
            {
                "iteration": record["learned_iteration"],
                "safe_kick_success_count": learned["safe_kick_success_count"],
                "physical_fall_count": learned["physical_fall_count"],
                "mean_planar_object_net_displacement_m": learned[
                    "mean_planar_object_net_displacement_m"
                ],
                "strict_learnability_pass": improvement,
            }
        )
    passing = [row for row in rows if row["strict_learnability_pass"]]
    result = {
        "protocol": "sugar_frozen_expert_transition_failure_overfit_v1",
        "diagnostic_only": True,
        "training_and_evaluation_seed": 181630,
        "carry_prefix_steps": 41,
        "selected_skill": "Kick",
        "pre_update": pre,
        "endpoints": rows,
        "checks": {
            "predeclared_endpoint_curve_complete": True,
            "exact_frozen_experts_preserved": True,
            "failure_rich_baseline_present": pre["physical_fall_count"] > 0,
            "strict_learnability_pass": bool(passing),
        },
        "selected_endpoint": passing[0] if passing else None,
        "conclusion": (
            "failure_rich_transition_is_learnable"
            if passing
            else "failure_rich_transition_not_learned_by_update256"
        ),
        "claim_boundary": (
            "Training and frozen evaluation intentionally reuse seed181630. This is a "
            "fixed-context learnability diagnostic and cannot establish generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
