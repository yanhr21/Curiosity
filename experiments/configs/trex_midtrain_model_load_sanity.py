#!/usr/bin/env python3
"""Compute-node-only official T-Rex midtrain model-load sanity.

This imports official `external/T-Rex/scripts/test.py` and calls its
`model_load(args)`. It does not implement a model, replace a checkpoint, or run
training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path("/public/home/yanhongru/Curiosity")


def _load_official_test_module(repo_root: Path):
    scripts_dir = repo_root / "scripts"
    test_py = scripts_dir / "test.py"
    if not test_py.is_file():
        raise FileNotFoundError(f"official test.py not found: {test_py}")
    for path in (repo_root, scripts_dir):
        sp = str(path)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    spec = importlib.util.spec_from_file_location("trex_official_test_current", test_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official test.py: {test_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args_in = parser.parse_args()

    if "SLURM_JOB_ID" not in os.environ:
        raise SystemExit("must run inside an existing Slurm allocation")

    workspace = args_in.workspace.resolve()
    repo_root = workspace / "external" / "T-Rex"
    ckpt = workspace / "checkpoints" / "trex" / "midtrain"
    base = workspace / "checkpoints" / "qwen" / "Qwen3-VL-2B-Instruct"

    official_test = _load_official_test_module(repo_root)
    model_args = SimpleNamespace(
        checkpoint_path=str(ckpt),
        base_model_path=str(base),
        stats_path="",
        dataset_name="rlbench",
        action_dim=62,
        action_chunk=16,
        use_robot_state=0,
        use_tactile_deform=1,
        use_tactile_vec=1,
        tactile_intermediate_size=0,
        n_flare_tokens_per_frame=0,
        n_flare_steps=0,
        cuda=args_in.cuda,
        port=5678,
        image_size=[384, 288],
        cascaded_total_steps=10,
        cascaded_split_step=6,
        disable_tactile=0,
        use_tactile_code=0,
        vqvae_codebook_size=64,
        vqvae_ckpt="",
        vqvae_config=None,
    )

    started = time.time()
    model, processor, statistic = official_test.model_load(model_args)
    elapsed = time.time() - started
    checks = {
        "has_deform_encoder": hasattr(model, "deform_encoder"),
        "has_deform_proj": hasattr(model, "deform_proj"),
        "has_tactile_vqvae": getattr(model, "tactile_vqvae", None) is not None,
        "has_tactile_code_embedder": hasattr(model, "tactile_code_embedder"),
        "stats_has_tacf6": all(key in statistic for key in ("tacf6_mask", "tacf6_min", "tacf6_max")),
        "processor_loaded": processor is not None,
    }
    failures = [name for name, ok in checks.items() if not ok]
    result: dict[str, Any] = {
        "classification": "official_trex_midtrain_model_load_sanity",
        "status": "pass" if not failures else "fail",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
        "official_model_load_source": str(repo_root / "scripts" / "test.py"),
        "official_repo_commit": os.popen(f"git -C {repo_root} rev-parse HEAD").read().strip(),
        "checkpoint_path": str(ckpt),
        "base_model_path": str(base),
        "elapsed_seconds": elapsed,
        "checks": checks,
        "failures": failures,
        "no_training": True,
        "no_placeholder_model": True,
        "generated_trex_fields": [],
        "interpretation": "The released midtrain checkpoint restores the official T-Rex model-load path with embedded tactile VQ-VAE components present.",
    }
    args_in.output.parent.mkdir(parents=True, exist_ok=True)
    with args_in.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args_in.output), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
