#!/usr/bin/env python3
"""Compare matched pre-update and learned Carry9-to-Kick recovery rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--trained", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def main() -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    trained = json.loads(args.trained.read_text(encoding="utf-8"))
    for result in (baseline, trained):
        if (
            result.get("protocol")
            not in {
                "sugar_carry9_to_kick_recovery_frozen_eval_v1",
                "sugar_cross_skill_recovery_frozen_eval_v2",
                "sugar_cross_skill_recovery_frozen_eval_v3",
            }
            or result.get("structurally_valid") is not True
        ):
            raise RuntimeError("pair contains an invalid frozen evaluation")
    if (
        baseline["seed"] != trained["seed"]
        or baseline["num_envs"] != trained["num_envs"]
        or baseline["steps"] != trained["steps"]
        or baseline["prefix"] != trained["prefix"]
    ):
        raise RuntimeError("recovery pair is not matched")
    before = baseline["aggregate"]
    after = trained["aggregate"]
    delta = {
        name: float(after[name] - before[name])
        for name in (
            "mean_mean_reward",
            "mean_planar_object_net_displacement_m",
            "mean_planar_object_path_m",
            "mean_any_foot_box_contact_fraction",
            "mean_maximum_robot_root_height_loss_m",
            "kick_success_count",
            "physical_fall_count",
            "safe_kick_success_count",
        )
        if name in before and name in after
    }
    checks = {
        "matched_seed_profiles_steps_and_online_prefix": True,
        "mean_reward_improves": delta["mean_mean_reward"] > 0.0,
        "kick_success_count_does_not_decrease": delta["kick_success_count"] >= 0,
        "mean_planar_displacement_does_not_decrease": (
            delta["mean_planar_object_net_displacement_m"] >= 0.0
        ),
        "physical_fall_count_does_not_increase": delta["physical_fall_count"] <= 0,
        "safe_kick_success_count_does_not_decrease": (
            delta.get("safe_kick_success_count", 0.0) >= 0
        ),
        "physical_recovery_improves": bool(
            delta.get("safe_kick_success_count", delta["kick_success_count"]) > 0
            or delta["physical_fall_count"] < 0
        ),
    }
    result = {
        "protocol": "sugar_cross_skill_recovery_matched_pair_v2",
        "carry_prefix_steps": int(baseline["prefix"]["carry_steps"]),
        "baseline": before,
        "trained": after,
        "trained_minus_baseline": delta,
        "checks": checks,
        "learned_recovery_improves_matched_endpoint": all(checks.values()),
        "claim_boundary": (
            "A fixed-condition serious recovery run after an exact online Carry prefix. "
            "It is not yet a general cross-skill transition controller."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
