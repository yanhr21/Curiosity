#!/usr/bin/env python3
"""Validate and summarize the frozen tactile-column authority curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DECLARED_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
INITIAL_KEYS = (
    "robot_root_state_w",
    "joint_pos",
    "joint_vel",
    "object_root_state_w",
    "last_action",
)
METRICS = (
    "completed_steps",
    "cumulative_reward",
    "student_teacher_action_mae",
    "maximum_relative_lift_m",
    "final_relative_lift_m",
    "bilateral_physical_tactile_frames",
    "final_object_position_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("zero", "025", "050", "075", "100"):
        parser.add_argument(f"--scale-{label}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    with np.load(Path(result["trace"]), allow_pickle=False) as source:
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
        0.0: args.scale_zero,
        0.25: args.scale_025,
        0.5: args.scale_050,
        0.75: args.scale_075,
        1.0: args.scale_100,
    }
    loaded = {scale: load(path) for scale, path in paths.items()}
    results = {scale: value[0] for scale, value in loaded.items()}
    traces = {scale: value[1] for scale, value in loaded.items()}
    reference_result = results[0.0]
    reference_trace = traces[0.0]

    checks = {
        "declared_scales_exact": all(
            results[scale].get("tactile_authority_scale") == scale
            for scale in DECLARED_SCALES
        ),
        "all_actor_modes_live": all(
            results[scale].get("actor_tactile_mode") == "live"
            for scale in DECLARED_SCALES
        ),
        "same_checkpoint": len(
            {results[scale]["checkpoint"] for scale in DECLARED_SCALES}
        )
        == 1,
        "same_seed": len({results[scale]["seed"] for scale in DECLARED_SCALES})
        == 1,
        "same_physical_condition": all(
            results[scale]["physical_condition"]
            == reference_result["physical_condition"]
            for scale in DECLARED_SCALES
        ),
        "same_reference": all(
            results[scale]["reference"] == reference_result["reference"]
            for scale in DECLARED_SCALES
        ),
        "same_disabled_events": all(
            results[scale]["disabled_events"]
            == reference_result["disabled_events"]
            for scale in DECLARED_SCALES
        ),
        "same_patch_permutation": all(
            results[scale]["patch_permutation"]
            == reference_result["patch_permutation"]
            for scale in DECLARED_SCALES
        ),
        "initial_physical_state_exact": all(
            np.array_equal(traces[scale][key], reference_trace[key])
            for scale in DECLARED_SCALES[1:]
            for key in INITIAL_KEYS
        ),
        "scale_zero_live_equals_zeroed_same_state": bool(
            np.array_equal(
                reference_trace["same_state_action_live"],
                reference_trace["same_state_action_zeroed"],
            )
        ),
    }

    first_contact_by_scale = {}
    for scale in DECLARED_SCALES:
        supported = traces[scale]["raw_actor_tactile_nonzero_values"][:, 0] > 0
        first_contact_by_scale[str(scale)] = first_true(supported)
    checks["first_tactile_supported_step_equal"] = (
        len(set(first_contact_by_scale.values())) == 1
    )
    first_contact = next(iter(first_contact_by_scale.values()))
    if first_contact is None:
        raise RuntimeError("authority curve never reaches tactile support")
    checks["precontact_actions_exact"] = all(
        np.array_equal(
            traces[scale]["action"][:first_contact],
            reference_trace["action"][:first_contact],
        )
        for scale in DECLARED_SCALES[1:]
    )
    if not all(checks.values()):
        raise RuntimeError(f"authority curve mismatch: {checks}")

    common_steps = min(
        int(results[scale]["completed_steps"]) for scale in DECLARED_SCALES
    )
    rows = {}
    for scale in DECLARED_SCALES:
        trace = traces[scale]
        initial_z = float(trace["object_root_state_w"][0, 2])
        final_common_z = float(trace["object_pos_w"][common_steps - 1, 0, 2])
        rows[str(scale)] = {
            **{name: results[scale][name] for name in METRICS},
            "same_state_live_zero_action_abs_max": results[scale][
                "same_state_live_vs_zeroed_action_abs_max"
            ],
            "same_state_live_zero_action_abs_max_supported_mean": results[scale][
                "same_state_live_vs_zeroed_action_abs_max_supported_mean"
            ],
            "termination_terms": results[scale]["termination_terms"],
            "common_horizon": {
                "steps": common_steps,
                "cumulative_reward": float(trace["reward"][:common_steps, 0].sum()),
                "mean_object_position_error_m": float(
                    trace["object_pos_error"][:common_steps, 0].mean()
                ),
                "final_relative_lift_m": final_common_z - initial_z,
            },
        }

    output = {
        "schema": "native_whole_hand_tactile_authority_curve_summary_v1",
        "declared_scales": list(DECLARED_SCALES),
        "inputs": {str(scale): str(paths[scale].resolve()) for scale in DECLARED_SCALES},
        "checks": checks,
        "first_tactile_supported_step": first_contact,
        "common_horizon_steps": common_steps,
        "rows": rows,
        "interpretation_boundary": (
            "This frozen no-learning curve diagnoses action authority. It does "
            "not establish cross-seed tactile usefulness or replace matched "
            "training with an explicitly bounded fusion architecture."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
