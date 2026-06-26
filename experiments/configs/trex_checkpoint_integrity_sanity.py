#!/usr/bin/env python3
"""Current-state T-Rex checkpoint integrity sanity.

This script runs on a compute node and checks the official released checkpoint
files already staged under `checkpoints/`. It does not instantiate a replacement
model, train anything, or create synthetic T-Rex data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ROOT = Path("/public/home/yanhongru/Curiosity")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _state_dict_summary(model_pt: Path, required_prefixes: list[str]) -> dict[str, Any]:
    state = torch.load(model_pt, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"{model_pt} did not load to a state-dict-like object")
    keys = sorted(str(key) for key in state.keys())
    prefix_hits = {prefix: any(key.startswith(prefix) for key in keys) for prefix in required_prefixes}
    tensor_count = sum(1 for value in state.values() if torch.is_tensor(value))
    tensor_numel = sum(int(value.numel()) for value in state.values() if torch.is_tensor(value))
    return {
        "key_count": len(keys),
        "tensor_count": tensor_count,
        "tensor_numel": tensor_numel,
        "required_prefix_hits": prefix_hits,
        "sample_keys": keys[:20],
    }


def _qwen_safetensors_summary(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        sample = {}
        for key in keys[:20]:
            tensor = handle.get_tensor(key)
            sample[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    return {
        "key_count": len(keys),
        "sample_tensors": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "SLURM_JOB_ID" not in os.environ:
        raise SystemExit("must run inside an existing Slurm allocation")

    root = args.root
    trex_repo = root / "external" / "T-Rex"
    midtrain = root / "checkpoints" / "trex" / "midtrain"
    pretrain = root / "checkpoints" / "trex" / "pretrain" / "checkpoint-0-610000"
    qwen = root / "checkpoints" / "qwen" / "Qwen3-VL-2B-Instruct"

    mid_args = _load_json(midtrain / "training_args.json")
    pre_args = _load_json(pretrain / "training_args.json")
    qwen_cfg = _load_json(qwen / "config.json")

    mid_state = _state_dict_summary(
        midtrain / "model.pt",
        [
            "tactile_vqvae.",
            "tactile_code_embedder",
            "deform_encoder",
            "deform_proj",
            "tacf6_vqvae_min",
            "tacf6_vqvae_max",
            "tacf6_vqvae_mask",
        ],
    )
    pre_state = _state_dict_summary(
        pretrain / "model.pt",
        [
            "model.",
            "visual",
            "state_embedder",
            "final_layer",
            "final_layer_tactile",
            "flare_proj",
        ],
    )
    qwen_state = _qwen_safetensors_summary(qwen / "model.safetensors")

    checks = {
        "official_trex_repo_exists": trex_repo.is_dir(),
        "midtrain_model_exists": (midtrain / "model.pt").is_file(),
        "midtrain_has_embedded_vqvae_args": bool(mid_args.get("use_tactile_vqvae")) and isinstance(mid_args.get("vqvae_config"), dict),
        "midtrain_action_contract": mid_args.get("action_dim") == 62 and mid_args.get("action_chunk") == 16,
        "midtrain_required_weight_prefixes": all(mid_state["required_prefix_hits"].values()),
        "pretrain_model_exists": (pretrain / "model.pt").is_file(),
        "pretrain_action_contract": pre_args.get("action_dim") == 62 and pre_args.get("action_chunk") == 16,
        "pretrain_training_stage_1": pre_args.get("training_stage") == 1,
        "pretrain_required_weight_prefixes": all(pre_state["required_prefix_hits"].values()),
        "qwen_model_exists": (qwen / "model.safetensors").is_file(),
        "qwen_config_model_type": qwen_cfg.get("model_type") == "qwen3_vl",
        "qwen_safetensors_nonempty": qwen_state["key_count"] > 0,
    }
    failures = [name for name, ok in checks.items() if not ok]
    result = {
        "classification": "official_trex_checkpoint_integrity_sanity",
        "status": "pass" if not failures else "fail",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "host": os.uname().nodename,
        "official_repo": {
            "path": str(trex_repo),
            "commit": os.popen(f"git -C {trex_repo} rev-parse HEAD").read().strip(),
            "dirty_status": os.popen(f"git -C {trex_repo} status --short").read().splitlines(),
        },
        "files": {
            "midtrain_model": _file(midtrain / "model.pt"),
            "midtrain_training_args": _file(midtrain / "training_args.json"),
            "midtrain_stats": _file(midtrain / "stats_data.json"),
            "pretrain_model": _file(pretrain / "model.pt"),
            "pretrain_training_args": _file(pretrain / "training_args.json"),
            "pretrain_stats": _file(pretrain / "stats_data.json"),
            "qwen_model": _file(qwen / "model.safetensors"),
            "qwen_config": _file(qwen / "config.json"),
        },
        "training_args": {
            "midtrain": {
                "action_dim": mid_args.get("action_dim"),
                "action_chunk": mid_args.get("action_chunk"),
                "training_stage": mid_args.get("training_stage"),
                "use_tactile_code": mid_args.get("use_tactile_code"),
                "use_tactile_vqvae": mid_args.get("use_tactile_vqvae"),
                "vqvae_config": mid_args.get("vqvae_config"),
                "cascaded_total_steps": mid_args.get("cascaded_total_steps"),
                "cascaded_split_step": mid_args.get("cascaded_split_step"),
            },
            "pretrain": {
                "action_dim": pre_args.get("action_dim"),
                "action_chunk": pre_args.get("action_chunk"),
                "training_stage": pre_args.get("training_stage"),
                "use_robot_state": pre_args.get("use_robot_state"),
            },
        },
        "state_dicts": {
            "midtrain": mid_state,
            "pretrain": pre_state,
            "qwen_safetensors": qwen_state,
        },
        "checks": checks,
        "failures": failures,
        "no_training": True,
        "no_placeholder_model": True,
        "generated_trex_fields": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
