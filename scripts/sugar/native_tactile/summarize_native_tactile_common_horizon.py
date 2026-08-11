#!/usr/bin/env python3
"""Compare two frozen tactile-policy traces over their common horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tactile", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(arrays: np.lib.npyio.NpzFile, steps: int) -> dict[str, float]:
    reward = arrays["reward"][:steps, 0]
    position_error = arrays["object_pos_error"][:steps, 0]
    object_z = arrays["object_pos_w"][:steps, 0, 2]
    initial_z = float(arrays["object_root_state_w"][0, 2])
    lift = object_z - initial_z
    return {
        "cumulative_reward": float(reward.sum()),
        "mean_object_position_error_m": float(position_error.mean()),
        "final_object_position_error_m": float(position_error[-1]),
        "maximum_relative_lift_m": float(lift.max()),
        "final_relative_lift_m": float(lift[-1]),
    }


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    tactile_path = args.tactile.expanduser().resolve()
    zero_path = args.zero.expanduser().resolve()
    with np.load(tactile_path) as tactile_arrays, np.load(zero_path) as zero_arrays:
        tactile_steps = int(tactile_arrays["reward"].shape[0])
        zero_steps = int(zero_arrays["reward"].shape[0])
        common_steps = min(tactile_steps, zero_steps)
        tactile = summarize(tactile_arrays, common_steps)
        zero = summarize(zero_arrays, common_steps)
    result = {
        "schema": "native_tactile_frozen_common_horizon_v1",
        "tactile_trace": str(tactile_path),
        "zero_trace": str(zero_path),
        "completed_steps": {"tactile": tactile_steps, "zero": zero_steps},
        "common_steps": common_steps,
        "tactile": tactile,
        "zero": zero,
        "tactile_minus_zero": {
            name: tactile[name] - zero[name] for name in tactile
        },
        "interpretation": (
            "Reward and tracking use the same number of transitions; endpoint "
            "behavior after the shorter arm terminates is excluded."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
