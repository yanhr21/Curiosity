#!/usr/bin/env python3
"""Create a frozen gate-only or residual-only causal-composer diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


FINAL_WEIGHT = "actor.composer.6.weight"
FINAL_BIAS = "actor.composer.6.bias"
ACTION_DIM = 29


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("gate_only", "residual_only"), required=True
    )
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise RuntimeError("source checkpoint has no model_state_dict")
    if checkpoint.get("iter") != 64:
        raise RuntimeError("ablation source must be the frozen update-64 endpoint")
    if FINAL_WEIGHT not in source_state or FINAL_BIAS not in source_state:
        raise RuntimeError("causal composer output layer is missing")
    if not any(name.startswith("actor.experts.0.") for name in source_state) or not any(
        name.startswith("actor.experts.1.") for name in source_state
    ):
        raise RuntimeError("causal composer does not embed both released experts")
    weight = source_state[FINAL_WEIGHT]
    bias = source_state[FINAL_BIAS]
    if tuple(weight.shape) != (1 + ACTION_DIM, 128) or tuple(bias.shape) != (
        1 + ACTION_DIM,
    ):
        raise RuntimeError("causal composer output geometry drift")
    if not all(torch.isfinite(value).all() for value in source_state.values()):
        raise RuntimeError("source checkpoint contains non-finite tensors")

    state = {name: value.detach().clone() for name, value in source_state.items()}
    if args.mode == "gate_only":
        state[FINAL_WEIGHT][1:].zero_()
        state[FINAL_BIAS][1:].zero_()
        preserved_rows = [0]
        zeroed_rows = list(range(1, 1 + ACTION_DIM))
    else:
        state[FINAL_WEIGHT][0].zero_()
        state[FINAL_BIAS][0].zero_()
        preserved_rows = list(range(1, 1 + ACTION_DIM))
        zeroed_rows = [0]

    changed = {FINAL_WEIGHT, FINAL_BIAS}
    if not all(
        torch.equal(state[name], source_state[name])
        for name in state
        if name not in changed
    ):
        raise RuntimeError("ablation changed a tensor outside the output layer")
    if not torch.equal(
        state[FINAL_WEIGHT][preserved_rows], weight[preserved_rows]
    ) or not torch.equal(state[FINAL_BIAS][preserved_rows], bias[preserved_rows]):
        raise RuntimeError("ablation changed a preserved composer row")
    if torch.count_nonzero(state[FINAL_WEIGHT][zeroed_rows]).item() != 0 or torch.count_nonzero(
        state[FINAL_BIAS][zeroed_rows]
    ).item() != 0:
        raise RuntimeError("ablation rows are not exact zero")

    checkpoint["model_state_dict"] = state
    source_infos = checkpoint.get("infos")
    checkpoint["infos"] = dict(source_infos) if isinstance(source_infos, dict) else {}
    checkpoint["infos"].update(
        {
            "diagnostic_only": True,
            "causal_composer_ablation": args.mode,
            "policy_training_or_optimizer_updates": 0,
            "source_checkpoint": str(args.source.resolve()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    audit = {
        "protocol": "sugar_causal_composer_frozen_ablation_checkpoint_v1",
        "mode": args.mode,
        "source_checkpoint": str(args.source.resolve()),
        "output_checkpoint": str(args.output.resolve()),
        "source_iteration": 64,
        "policy_training_or_optimizer_updates": 0,
        "all_non_output_tensors_elementwise_identical": True,
        "preserved_output_rows_elementwise_identical": True,
        "ablated_output_rows_exact_zero": True,
        "all_output_tensors_finite": bool(
            torch.isfinite(state[FINAL_WEIGHT]).all()
            and torch.isfinite(state[FINAL_BIAS]).all()
        ),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
