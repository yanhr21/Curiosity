#!/usr/bin/env python3
"""Summarize live, zeroed, and patch-permuted frozen tactile-policy rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BEHAVIOR_METRICS = (
    "completed_steps",
    "cumulative_reward",
    "student_teacher_action_mae",
    "maximum_relative_lift_m",
    "final_relative_lift_m",
    "bilateral_physical_tactile_frames",
    "final_object_position_error_m",
)
INITIAL_KEYS = (
    "robot_root_state_w",
    "joint_pos",
    "joint_vel",
    "object_root_state_w",
    "last_action",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--zeroed", type=Path, required=True)
    parser.add_argument("--patch-permuted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    trace_path = Path(result["trace"])
    with np.load(trace_path, allow_pickle=False) as source:
        trace = {key: source[key] for key in source.files}
    return result, trace


def first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return int(indices[0]) if len(indices) else None


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = {
        "live": args.live,
        "zeroed": args.zeroed,
        "patch_permuted": args.patch_permuted,
    }
    loaded = {mode: load_result(path) for mode, path in paths.items()}
    results = {mode: value[0] for mode, value in loaded.items()}
    traces = {mode: value[1] for mode, value in loaded.items()}

    checks = {
        "declared_modes_match": all(
            results[mode].get("actor_tactile_mode") == mode for mode in paths
        ),
        "same_checkpoint": len({results[mode]["checkpoint"] for mode in paths}) == 1,
        "same_seed": len({results[mode]["seed"] for mode in paths}) == 1,
        "same_physical_condition": all(
            results[mode]["physical_condition"] == results["live"]["physical_condition"]
            for mode in paths
        ),
        "same_reference": all(
            results[mode]["reference"] == results["live"]["reference"]
            for mode in paths
        ),
        "same_disabled_events": all(
            results[mode]["disabled_events"] == results["live"]["disabled_events"]
            for mode in paths
        ),
        "same_patch_permutation": all(
            results[mode]["patch_permutation"] == results["live"]["patch_permutation"]
            for mode in paths
        ),
        "initial_physical_state_exact": all(
            np.array_equal(traces[mode][key], traces["live"][key])
            for mode in ("zeroed", "patch_permuted")
            for key in INITIAL_KEYS
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"dependence rollouts are not matched: {checks}")

    selected_action_keys = {
        "live": "same_state_action_live",
        "zeroed": "same_state_action_zeroed",
        "patch_permuted": "same_state_action_patch_permuted",
    }
    selected_action_checks = {}
    for mode, key in selected_action_keys.items():
        selected_action_checks[mode] = bool(
            np.array_equal(traces[mode]["action"], traces[mode][key])
        )
    checks["actual_action_matches_selected_input_mode"] = all(
        selected_action_checks.values()
    )
    if not checks["actual_action_matches_selected_input_mode"]:
        raise ValueError(f"selected action mismatch: {selected_action_checks}")

    live_supported = traces["live"]["raw_actor_tactile_nonzero_values"][:, 0] > 0
    first_supported = first_true(live_supported)
    comparisons = {}
    live_action = traces["live"]["action"][:, 0]
    live_object = traces["live"]["object_pos_w"][:, 0]
    for mode in ("zeroed", "patch_permuted"):
        common = min(len(live_action), len(traces[mode]["action"]))
        action_delta = np.max(
            np.abs(live_action[:common] - traces[mode]["action"][:common, 0]),
            axis=-1,
        )
        object_delta = np.linalg.norm(
            live_object[:common] - traces[mode]["object_pos_w"][:common, 0],
            axis=-1,
        )
        live_reward_common = float(traces["live"]["reward"][:common, 0].sum())
        mode_reward_common = float(traces[mode]["reward"][:common, 0].sum())
        live_initial_z = float(traces["live"]["object_root_state_w"][0, 2])
        mode_initial_z = float(traces[mode]["object_root_state_w"][0, 2])
        comparisons[mode] = {
            "common_steps": int(common),
            "first_action_difference_step": first_true(action_delta > 0.0),
            "action_abs_max": float(action_delta.max()),
            "action_abs_max_mean": float(action_delta.mean()),
            "object_position_difference_m_max": float(object_delta.max()),
            "object_position_difference_m_final_common_step": float(object_delta[-1]),
            "common_horizon_cumulative_reward": {
                "live": live_reward_common,
                mode: mode_reward_common,
                "live_minus_other": live_reward_common - mode_reward_common,
            },
            "common_horizon_mean_object_position_error_m": {
                "live": float(
                    traces["live"]["object_pos_error"][:common, 0].mean()
                ),
                mode: float(traces[mode]["object_pos_error"][:common, 0].mean()),
            },
            "final_common_step_relative_lift_m": {
                "live": float(live_object[common - 1, 2] - live_initial_z),
                mode: float(
                    traces[mode]["object_pos_w"][common - 1, 0, 2]
                    - mode_initial_z
                ),
            },
            "pre_live_contact_actions_exact": (
                bool(np.all(action_delta[:first_supported] == 0.0))
                if first_supported is not None
                else True
            ),
        }

    metrics = {
        mode: {key: results[mode][key] for key in BEHAVIOR_METRICS}
        for mode in paths
    }
    output = {
        "schema": "native_whole_hand_tactile_frozen_dependence_summary_v1",
        "inputs": {mode: str(path.resolve()) for mode, path in paths.items()},
        "checks": checks,
        "selected_action_checks": selected_action_checks,
        "checkpoint": results["live"]["checkpoint"],
        "seed": results["live"]["seed"],
        "condition": results["live"]["physical_condition"],
        "patch_permutation": results["live"]["patch_permutation"],
        "first_live_tactile_supported_step": first_supported,
        "same_state_actor_dependence": {
            "live_vs_zeroed_action_abs_max": results["live"][
                "same_state_live_vs_zeroed_action_abs_max"
            ],
            "live_vs_zeroed_action_abs_max_supported_mean": results["live"][
                "same_state_live_vs_zeroed_action_abs_max_supported_mean"
            ],
            "live_vs_patch_permuted_action_abs_max": results["live"][
                "same_state_live_vs_patch_permuted_action_abs_max"
            ],
            "live_vs_patch_permuted_action_abs_max_supported_mean": results["live"][
                "same_state_live_vs_patch_permuted_action_abs_max_supported_mean"
            ],
        },
        "closed_loop_behavior": metrics,
        "closed_loop_comparison_to_live": comparisons,
        "interpretation_boundary": (
            "Action and trajectory changes establish dependence on the tactile "
            "input. Better behavior requires consistent matched physical outcomes; "
            "one seed cannot establish general tactile usefulness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
