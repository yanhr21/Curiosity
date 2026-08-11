#!/usr/bin/env python3
"""Compare the matched native-tactile and exact-zero training endpoints.

This is a small checkpoint accounting utility.  It proves that the two arms
started from the same policy parameters and reports how their learned policy
parameters differ.  It does not claim tactile benefit; that requires the
separate frozen physical rollout comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


MODEL_PATTERN = re.compile(r"model_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tactile", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def latest_checkpoint(directory: Path) -> tuple[int, Path]:
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("model_*.pt"):
        match = MODEL_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No model_N.pt checkpoint in {directory}")
    return max(candidates, key=lambda item: item[0])


def load_state(path: Path) -> tuple[int, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Missing model_state_dict in {path}")
    return int(payload.get("iter", -1)), state


def compare_states(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, object]:
    left_keys = set(left)
    right_keys = set(right)
    shared = sorted(left_keys & right_keys)
    mismatched_shapes = [
        key for key in shared if tuple(left[key].shape) != tuple(right[key].shape)
    ]
    comparable = [key for key in shared if key not in mismatched_shapes]
    exact = [key for key in comparable if torch.equal(left[key], right[key])]
    changed = [key for key in comparable if key not in exact]
    squared_l2 = 0.0
    maximum_abs = 0.0
    for key in changed:
        delta = left[key].to(torch.float64) - right[key].to(torch.float64)
        squared_l2 += float(torch.sum(delta * delta).item())
        maximum_abs = max(maximum_abs, float(delta.abs().max().item()))
    return {
        "left_only_keys": sorted(left_keys - right_keys),
        "right_only_keys": sorted(right_keys - left_keys),
        "shape_mismatch_keys": mismatched_shapes,
        "comparable_tensor_count": len(comparable),
        "exact_equal_tensor_count": len(exact),
        "changed_tensor_count": len(changed),
        "changed_tensor_names": changed,
        "parameter_delta_l2": squared_l2**0.5,
        "parameter_delta_abs_max": maximum_abs,
    }


def main() -> None:
    args = parse_args()
    tactile_dir = args.tactile.expanduser().resolve()
    zero_dir = args.zero.expanduser().resolve()
    output = args.output.expanduser().resolve()

    tactile_pre_iter, tactile_pre = load_state(tactile_dir / "model_prelearn.pt")
    zero_pre_iter, zero_pre = load_state(zero_dir / "model_prelearn.pt")
    tactile_index, tactile_path = latest_checkpoint(tactile_dir)
    zero_index, zero_path = latest_checkpoint(zero_dir)
    tactile_iter, tactile_final = load_state(tactile_path)
    zero_iter, zero_final = load_state(zero_path)

    initial = compare_states(tactile_pre, zero_pre)
    final = compare_states(tactile_final, zero_final)
    initial_exact = (
        not initial["left_only_keys"]
        and not initial["right_only_keys"]
        and not initial["shape_mismatch_keys"]
        and initial["changed_tensor_count"] == 0
    )

    result = {
        "semantics": (
            "matched checkpoint accounting only; frozen physical behavior is "
            "required before claiming a tactile advantage"
        ),
        "tactile_directory": str(tactile_dir),
        "zero_directory": str(zero_dir),
        "prelearn_checkpoint_iterations": [tactile_pre_iter, zero_pre_iter],
        "initial_model_state_exactly_equal": initial_exact,
        "initial_comparison": initial,
        "endpoint_files": [str(tactile_path), str(zero_path)],
        "endpoint_filename_indices": [tactile_index, zero_index],
        "endpoint_payload_iterations": [tactile_iter, zero_iter],
        "endpoint_comparison": final,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
