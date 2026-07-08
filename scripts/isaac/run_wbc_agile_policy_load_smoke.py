#!/usr/bin/env python3
"""Compute-node WBC-AGILE policy loading diagnostic.

This deliberately does not start Isaac.  It isolates whether local official
WBC-AGILE policy artifacts can be loaded through TorchScript, training
checkpoint reconstruction, or the repository PolicyWrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import importlib.util
from pathlib import Path


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to load policy/model on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WBC-AGILE policy loading smoke.")
    parser.add_argument(
        "--mode",
        choices=("torchscript", "checkpoint_direct", "policy_wrapper"),
        required=True,
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--io-config",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/external/WBC-AGILE/agile/data/policy/"
            "velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml"
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _alarm_handler(signum: int, frame: object) -> None:
    del signum, frame
    raise TimeoutError("policy load timed out")


def main() -> int:
    _refuse_login_node()
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "mode": str(args.mode),
        "artifact": str(args.artifact),
        "artifact_size_bytes": args.artifact.stat().st_size if args.artifact.exists() else None,
        "io_config": str(args.io_config),
        "status": "not_started",
        "error": None,
    }

    def write() -> None:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    try:
        if not args.artifact.is_file():
            raise FileNotFoundError(args.artifact)
        if not args.io_config.is_file():
            raise FileNotFoundError(args.io_config)
        import torch
        import yaml

        policy_path = Path("/public/home/yanhongru/Curiosity/external/WBC-AGILE/agile/sim2mujoco/policy.py")
        spec = importlib.util.spec_from_file_location("wbc_agile_policy_direct", policy_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load policy module spec from {policy_path}")
        policy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy_module)
        CheckpointPolicyWrapper = policy_module.CheckpointPolicyWrapper
        PolicyWrapper = policy_module.PolicyWrapper

        device = torch.device("cpu")
        config = yaml.safe_load(args.io_config.read_text())
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(args.timeout_sec)))
        summary["status"] = "loading"
        write()
        print(f"[PROGRESS] loading mode={args.mode} artifact={args.artifact}", flush=True)

        if args.mode == "torchscript":
            model = torch.jit.load(args.artifact, map_location=device)
            model.eval()
            summary["loaded_type"] = type(model).__name__
        elif args.mode == "checkpoint_direct":
            wrapper = CheckpointPolicyWrapper.from_checkpoint(args.artifact, config, device)
            summary["loaded_type"] = type(wrapper).__name__
            summary["model_type"] = type(wrapper.model).__name__
        else:
            wrapper = PolicyWrapper.from_config(args.artifact, config, device)
            summary["loaded_type"] = type(wrapper).__name__

        signal.alarm(0)
        summary["status"] = "loaded"
        print(f"[PROGRESS] loaded mode={args.mode}", flush=True)
        write()
        return 0
    except BaseException as exc:
        signal.alarm(0)
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)
        write()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
