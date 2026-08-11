#!/usr/bin/env python3
"""Summarize frozen teacher alignment on physically tactile-supported states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        left = left[mask]
        right = right[mask]
    if left.shape[0] == 0:
        raise ValueError("teacher-alignment summary has no selected rows")
    return float(np.abs(left - right).mean())


def main() -> None:
    args = parse_args()
    trace = args.trace.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with np.load(trace) as arrays:
        required = {
            "action",
            "teacher_action",
            "same_state_action_live",
            "same_state_action_zeroed",
            "same_state_action_patch_permuted",
            "raw_actor_tactile_nonzero_values",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"{trace} is missing arrays: {missing}")
        action = arrays["action"][:, 0]
        teacher = arrays["teacher_action"][:, 0]
        live = arrays["same_state_action_live"][:, 0]
        zeroed = arrays["same_state_action_zeroed"][:, 0]
        permuted = arrays["same_state_action_patch_permuted"][:, 0]
        supported = arrays["raw_actor_tactile_nonzero_values"][:, 0] > 0

    if not np.any(supported):
        raise ValueError("trace contains no physically tactile-supported states")
    live_mae = mae(live, teacher, supported)
    zeroed_mae = mae(zeroed, teacher, supported)
    permuted_mae = mae(permuted, teacher, supported)
    result = {
        "schema": "native_tactile_contact_teacher_alignment_v1",
        "trace": str(trace),
        "completed_steps": int(action.shape[0]),
        "physically_tactile_supported_steps": int(np.count_nonzero(supported)),
        "chosen_action_teacher_mae_all_steps": mae(action, teacher),
        "supported_state_teacher_mae": {
            "live_tactile": live_mae,
            "zeroed_tactile": zeroed_mae,
            "patch_permuted_tactile": permuted_mae,
            "live_minus_zeroed": live_mae - zeroed_mae,
            "live_minus_patch_permuted": live_mae - permuted_mae,
        },
        "interpretation": (
            "Same visited physical states and the same privileged teacher; only the "
            "counterfactual tactile tensor entering the frozen actor changes."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
