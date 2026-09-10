#!/usr/bin/env python3
"""Audit the pinned official HOST checkpoint without constructing a substitute model.

This is an artifact/inventory gate.  Full model construction and strict state-dict
comparison are a separate gate because they require HOST's official external backbones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


HOST_SOURCE_COMMIT = "9f3bba57792aa5053ac600b1b7a4625f96ea6662"
HOST_CHECKPOINT_COMMIT = "fcb38563ab7afe0bb0dcefd9431adb76112fa4b6"
HOST_CHECKPOINT_BYTES = 17_027_987_184
HOST_CHECKPOINT_SHA256 = (
    "89382e2a48c1d4a4c5b1791baff49c8cb5ded731a4b25209f0118e6e5016e86f"
)
EXPECTED_TOP_LEVEL_KEYS = {
    "mot",
    "proprio_encoder",
    "progress_encoder",
    "progress_decoder",
    "visual_encoder",
    "step",
    "torch_dtype",
}
HOST_TENSOR_COUNT = 3_148
HOST_PARAMETER_NUMEL = 8_513_682_648
HOST_TENSOR_MANIFEST_SHA256 = (
    "abb0fbbcab09aa99a135029c595d89df7d5b85e6779249beae29fdcdebd498d0"
)
HOST_COMPONENT_CONTRACT = {
    "mot": (2_609, 7_915_332_630),
    "proprio_encoder": (2, 86_016),
    "progress_encoder": (4, 9_449_472),
    "progress_decoder": (6, 9_452_546),
    "visual_encoder": (527, 579_361_984),
}
HOST_INTERFACE_CONTRACT = {
    "mot.mixtures.action.action_encoder.weight": [1024, 22],
    "mot.mixtures.action.head.weight": [22, 1024],
    "mot.mixtures.action.head.bias": [22],
    "proprio_encoder.weight": [4096, 20],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run only small audit-fixture tests; this never claims a HOST load.",
    )
    return parser.parse_args()


def require_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("checkpoint", "config", "source_tree", "output_dir")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError("missing required arguments: " + ", ".join(missing))


def require_h200_slurm() -> dict[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("HOST audit must run inside a Slurm compute allocation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
    device_name = torch.cuda.get_device_name(0)
    if "H200" not in device_name.upper():
        raise RuntimeError(f"HOST audit requires H200, found {device_name!r}")
    return {"slurm_job_id": job_id, "cuda_device": device_name}


def sha256_file(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(source_tree: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_tree), *args], text=True
    ).strip()


def validate_source(source_tree: Path) -> dict[str, Any]:
    commit = git_output(source_tree, "rev-parse", "HEAD")
    if commit != HOST_SOURCE_COMMIT:
        raise RuntimeError(
            f"official source commit mismatch: expected {HOST_SOURCE_COMMIT}, got {commit}"
        )
    tracked_files = git_output(source_tree, "ls-files").splitlines()
    required = {
        "policy_training/src/self_grounded_prediction/runtime.py",
        "policy_training/src/self_grounded_prediction/models/wan22/self_grounded_predictor.py",
    }
    absent = sorted(required.difference(tracked_files))
    if absent:
        raise RuntimeError(f"official sparse checkout lacks required files: {absent}")
    tree_hash = hashlib.sha256("\n".join(tracked_files).encode()).hexdigest()
    return {
        "commit": commit,
        "tracked_file_count": len(tracked_files),
        "tracked_file_list_sha256": tree_hash,
    }


def nested(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"official config is missing {path}")
        value = value[part]
    return value


def validate_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    expected = {
        "data.train.frames": 2,
        "data.train.action_frames": 15,
        "data.train.processor.action_output_dim": 22,
        "data.train.processor.proprio_output_dim": 20,
        "model.proprio_dim": 20,
        "model.video_dit_config.action_dim": 22,
        "model.video_dit_config.hidden_dim": 3072,
        "model.video_dit_config.num_layers": 30,
        "model.action_dit_config.action_dim": 22,
        "model.action_dit_config.hidden_dim": 1024,
        "model.action_dit_config.num_layers": 30,
        "model.visual_encoder.camera_input_size": 224,
    }
    actual = {path: nested(config, path) for path in expected}
    failures = {
        path: {"expected": expected[path], "actual": actual[path]}
        for path in expected
        if actual[path] != expected[path]
    }
    if failures:
        raise RuntimeError(f"official config contract mismatch: {failures}")
    return {
        "path": str(config_path.resolve()),
        "sha256": sha256_file(config_path),
        "contract": actual,
    }


def walk_tensors(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_tensors(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from walk_tensors(item, child)


def summarize_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    keys = set(checkpoint)
    if keys != EXPECTED_TOP_LEVEL_KEYS:
        raise RuntimeError(
            "official checkpoint top-level key mismatch: "
            f"missing={sorted(EXPECTED_TOP_LEVEL_KEYS - keys)}, "
            f"unexpected={sorted(keys - EXPECTED_TOP_LEVEL_KEYS)}"
        )
    tensors = list(walk_tensors(checkpoint))
    if not tensors:
        raise RuntimeError("official checkpoint contains no tensors")
    dtype_counts = Counter(str(tensor.dtype) for _, tensor in tensors)
    module_summary: dict[str, dict[str, Any]] = {}
    for top_key in sorted(EXPECTED_TOP_LEVEL_KEYS):
        component = [
            (name, tensor)
            for name, tensor in tensors
            if name == top_key or name.startswith(top_key + ".")
        ]
        module_summary[top_key] = {
            "tensor_count": len(component),
            "numel": sum(tensor.numel() for _, tensor in component),
            "dtypes": dict(sorted(Counter(str(t.dtype) for _, t in component).items())),
        }
    tensor_manifest = [
        {
            "key": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
        }
        for name, tensor in tensors
    ]
    manifest_bytes = json.dumps(
        tensor_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "top_level_keys": sorted(keys),
        "step": checkpoint["step"],
        "torch_dtype": str(checkpoint["torch_dtype"]),
        "tensor_count": len(tensors),
        "numel": sum(tensor.numel() for _, tensor in tensors),
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "components": module_summary,
        "tensor_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tensor_manifest": tensor_manifest,
    }


def validate_official_inventory(inventory: dict[str, Any]) -> None:
    failures: dict[str, Any] = {}
    for name, expected, actual in (
        ("tensor_count", HOST_TENSOR_COUNT, inventory["tensor_count"]),
        ("numel", HOST_PARAMETER_NUMEL, inventory["numel"]),
        (
            "tensor_manifest_sha256",
            HOST_TENSOR_MANIFEST_SHA256,
            inventory["tensor_manifest_sha256"],
        ),
        ("step", 950_000, inventory["step"]),
        ("torch_dtype", "torch.bfloat16", inventory["torch_dtype"]),
    ):
        if actual != expected:
            failures[name] = {"expected": expected, "actual": actual}
    for component, (tensor_count, numel) in HOST_COMPONENT_CONTRACT.items():
        actual = inventory["components"][component]
        if (actual["tensor_count"], actual["numel"]) != (tensor_count, numel):
            failures[f"component.{component}"] = {
                "expected": [tensor_count, numel],
                "actual": [actual["tensor_count"], actual["numel"]],
            }
    manifest = {item["key"]: item for item in inventory["tensor_manifest"]}
    for key, shape in HOST_INTERFACE_CONTRACT.items():
        actual = manifest.get(key)
        if actual is None or actual["shape"] != shape or actual["dtype"] != "torch.bfloat16":
            failures[f"interface.{key}"] = {
                "expected_shape": shape,
                "expected_dtype": "torch.bfloat16",
                "actual": actual,
            }
    if failures:
        raise RuntimeError(f"official tensor inventory mismatch: {failures}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_self_test() -> None:
    fixture = {
        "mot": {"weight": torch.zeros(2, 3)},
        "proprio_encoder": {"weight": torch.ones(4, 2)},
        "progress_encoder": {"weight": torch.ones(1)},
        "progress_decoder": {"bias": torch.zeros(1)},
        "visual_encoder": {"weight": torch.ones(2)},
        "step": 7,
        "torch_dtype": torch.bfloat16,
    }
    report = summarize_checkpoint(fixture)
    assert report["tensor_count"] == 5
    assert report["numel"] == 18
    invalid = dict(fixture)
    invalid["unexpected"] = {}
    try:
        summarize_checkpoint(invalid)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unexpected top-level key was not rejected")
    print("HOST checkpoint audit self-test passed (synthetic audit fixture only)")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    require_args(args)
    runtime = require_h200_slurm()
    checkpoint_path = args.checkpoint.resolve()
    if checkpoint_path.stat().st_size != HOST_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"checkpoint size mismatch: expected {HOST_CHECKPOINT_BYTES}, "
            f"got {checkpoint_path.stat().st_size}"
        )
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != HOST_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: expected {HOST_CHECKPOINT_SHA256}, "
            f"got {checkpoint_hash}"
        )
    source = validate_source(args.source_tree.resolve())
    config = validate_config(args.config.resolve())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint root must be dict, got {type(checkpoint).__name__}")
    inventory = summarize_checkpoint(checkpoint)
    validate_official_inventory(inventory)
    report = {
        "audit": "HOST official checkpoint artifact inventory",
        "claim_boundary": "robot-to-robot checkpoint artifact; no strict model load claim",
        "passed": True,
        "official": {
            "source_commit": HOST_SOURCE_COMMIT,
            "checkpoint_commit": HOST_CHECKPOINT_COMMIT,
            "checkpoint_bytes": HOST_CHECKPOINT_BYTES,
            "checkpoint_sha256": HOST_CHECKPOINT_SHA256,
        },
        "runtime": runtime,
        "source": source,
        "config": config,
        "checkpoint": inventory,
    }
    output_path = args.output_dir.resolve() / "HOST_OFFICIAL_CHECKPOINT_AUDIT.json"
    atomic_write_json(output_path, report)
    print(json.dumps({
        "passed": True,
        "output": str(output_path),
        "tensor_count": inventory["tensor_count"],
        "numel": inventory["numel"],
        "step": inventory["step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
