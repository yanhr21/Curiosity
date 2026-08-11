#!/usr/bin/env python3
"""Apply the frozen two-condition policy gate after tactile adapter pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heavy-pair", type=Path, required=True)
    parser.add_argument("--heavy-common", type=Path, required=True)
    parser.add_argument("--low-friction-pair", type=Path, required=True)
    parser.add_argument("--low-friction-common", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def condition_result(pair_path: Path, common_path: Path) -> dict[str, object]:
    pair = load(pair_path)
    common = load(common_path)
    tactile_steps = int(pair["tactile"]["completed_steps"])
    zero_steps = int(pair["zero"]["completed_steps"])
    reward_delta = float(common["tactile_minus_zero"]["cumulative_reward"])
    mean_position_delta = float(
        common["tactile_minus_zero"]["mean_object_position_error_m"]
    )
    checks = {
        "common_horizon_reward_higher_with_live_tactile": reward_delta > 0.0,
        "common_horizon_mean_position_error_lower_with_live_tactile": (
            mean_position_delta < 0.0
        ),
        "live_tactile_does_not_terminate_earlier": tactile_steps >= zero_steps,
    }
    return {
        "condition": pair["condition"],
        "tactile_steps": tactile_steps,
        "zero_steps": zero_steps,
        "common_steps": int(common["common_steps"]),
        "common_horizon": common,
        "termination_terms": pair["termination_terms"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    heavy = condition_result(args.heavy_pair, args.heavy_common)
    low_friction = condition_result(
        args.low_friction_pair, args.low_friction_common
    )
    result = {
        "schema": "native_tactile_teacher_residual_frozen_policy_gate_v1",
        "conditions": {
            "heldout_heavy": heavy,
            "heldout_low_friction": low_friction,
        },
        "gate_passed": bool(heavy["passed"] and low_friction["passed"]),
        "decision_rule": (
            "The same frozen pretrained checkpoint must give higher common-"
            "horizon reward, lower mean position error, and no earlier "
            "termination with live tactile than with exact-zero tactile on "
            "each untouched physical condition."
        ),
        "claim_boundary": (
            "A pass authorizes a later matched PPO experiment; it does not by "
            "itself establish multi-seed policy improvement."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
