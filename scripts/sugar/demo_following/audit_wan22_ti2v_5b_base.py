#!/usr/bin/env python3
"""Audit the exact official Wan2.2-TI2V-5B source and checkpoint base."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CONFIG = {
    "vae_stride": (4, 16, 16),
    "patch_size": (1, 2, 2),
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "num_heads": 24,
    "num_layers": 30,
    "window_size": (-1, -1),
    "qk_norm": True,
    "cross_attn_norm": True,
    "sample_fps": 24,
    "frame_num": 121,
}
EXPECTED_ASSETS = {
    "diffusion_pytorch_model-00001-of-00003.safetensors": 9_000_000_000,
    "diffusion_pytorch_model-00002-of-00003.safetensors": 9_000_000_000,
    "diffusion_pytorch_model-00003-of-00003.safetensors": 100_000_000,
    "diffusion_pytorch_model.safetensors.index.json": 1_000,
    "Wan2.2_VAE.pth": 2_000_000_000,
    "models_t5_umt5-xxl-enc-bf16.pth": 10_000_000_000,
    "google/umt5-xxl/spiece.model": 1_000_000,
    "config.json": 100,
}
EXPECTED_CHECKPOINT_CONFIG = {
    "_class_name": "WanModel",
    "_diffusers_version": "0.33.0",
    "model_type": "ti2v",
    "text_len": 512,
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
}
EXPECTED_SNAPSHOT_FILE_COUNT = 22
EXPECTED_SNAPSHOT_BYTES = 34_203_123_632
EXPECTED_SNAPSHOT_PATHS = {
    ".gitattributes",
    ".modelscope.diff.ignore",
    "README.md",
    "Wan2.2_VAE.pth",
    "assets/comp_effic.png",
    "assets/logo.png",
    "assets/moe_2.png",
    "assets/moe_arch.png",
    "assets/performance.png",
    "assets/vae.png",
    "config.json",
    "configuration.json",
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "examples/i2v_input.JPG",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict-load-result", type=Path)
    parser.add_argument("--hash-files", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_config(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
            continue
        if target.value.id != "ti2v_5B":
            continue
        try:
            values[target.attr] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config_path = args.source_root / "wan/configs/wan_ti2v_5B.py"
    config = parse_config(config_path) if config_path.is_file() else {}
    config_match = all(config.get(name) == value for name, value in EXPECTED_CONFIG.items())

    assets: dict[str, dict[str, Any]] = {}
    assets_complete = True
    for relpath, minimum_bytes in EXPECTED_ASSETS.items():
        path = args.checkpoint_root / relpath
        size = path.stat().st_size if path.is_file() else 0
        ok = size >= minimum_bytes
        assets_complete = assets_complete and ok
        record: dict[str, Any] = {"bytes": size, "minimum_bytes": minimum_bytes, "passed": ok}
        if ok and args.hash_files:
            record["sha256"] = sha256(path)
        assets[relpath] = record

    incomplete_files = sorted(
        path.relative_to(args.checkpoint_root).as_posix()
        for path in args.checkpoint_root.rglob("*.incomplete")
        if path.is_file()
    )
    snapshot_paths = sorted(
        path
        for path in args.checkpoint_root.rglob("*")
        if path.is_file() and not path.name.endswith(".incomplete")
    )
    snapshot_files: dict[str, dict[str, Any]] = {}
    for path in snapshot_paths:
        relpath = path.relative_to(args.checkpoint_root).as_posix()
        record = {"bytes": path.stat().st_size}
        if args.hash_files:
            record["sha256"] = assets.get(relpath, {}).get("sha256") or sha256(path)
        snapshot_files[relpath] = record
    snapshot_bytes = sum(record["bytes"] for record in snapshot_files.values())

    checkpoint_config = {}
    checkpoint_config_path = args.checkpoint_root / "config.json"
    if checkpoint_config_path.is_file():
        checkpoint_config = read_json(checkpoint_config_path)
    checkpoint_config_match = all(
        checkpoint_config.get(name) == value for name, value in EXPECTED_CHECKPOINT_CONFIG.items()
    )

    strict = read_json(args.strict_load_result) if args.strict_load_result else {}
    strict_checks = strict.get("checks", {}) if isinstance(strict.get("checks", {}), dict) else {}
    checkpoint_hashes_recorded = args.hash_files and all(
        "sha256" in record for record in assets.values() if record["passed"]
    ) and all(record["passed"] for record in assets.values())
    checks = {
        "official_source_commit_is_full_sha": len(args.source_commit) == 40
        and all(char in "0123456789abcdef" for char in args.source_commit),
        "official_generate_entrypoint_present": (args.source_root / "generate.py").is_file(),
        "official_source_config_exact": config_match,
        "official_checkpoint_assets_complete": assets_complete,
        "official_snapshot_exact_file_count": len(snapshot_files) == EXPECTED_SNAPSHOT_FILE_COUNT,
        "official_snapshot_exact_paths": set(snapshot_files) == EXPECTED_SNAPSHOT_PATHS,
        "official_snapshot_exact_bytes": snapshot_bytes == EXPECTED_SNAPSHOT_BYTES,
        "no_incomplete_checkpoint_files": not incomplete_files,
        "official_checkpoint_sha256_recorded": checkpoint_hashes_recorded,
        "official_snapshot_sha256_recorded": args.hash_files
        and len(snapshot_files) == EXPECTED_SNAPSHOT_FILE_COUNT
        and all("sha256" in record for record in snapshot_files.values()),
        "checkpoint_config_exact": checkpoint_config_match,
        "strict_cpu_load_passed": bool(strict_checks.get("strict_cpu_load")),
        "exact_parameter_count_recorded": isinstance(strict.get("parameter_count"), int)
        and 4_000_000_000 < strict.get("parameter_count", 0) < 6_000_000_000,
        "h200_bfloat16_residency_passed": bool(strict_checks.get("h200_bfloat16_residency")),
        "exact_model_minimal_forward_passed": bool(strict_checks.get("exact_model_minimal_forward")),
        "official_model_class_verified": strict.get("model_class") == "wan.modules.model.WanModel",
    }
    passed = all(checks.values())
    result = {
        "protocol": "official_wan22_ti2v_5b_base_audit_v1",
        "passed": passed,
        "source_commit": args.source_commit,
        "source_config": config,
        "expected_config": EXPECTED_CONFIG,
        "expected_checkpoint_config": EXPECTED_CHECKPOINT_CONFIG,
        "checkpoint_config": checkpoint_config,
        "assets": assets,
        "snapshot_file_count": len(snapshot_files),
        "snapshot_bytes": snapshot_bytes,
        "expected_snapshot_file_count": EXPECTED_SNAPSHOT_FILE_COUNT,
        "expected_snapshot_bytes": EXPECTED_SNAPSHOT_BYTES,
        "expected_snapshot_paths": sorted(EXPECTED_SNAPSHOT_PATHS),
        "snapshot_files": snapshot_files,
        "incomplete_files": incomplete_files,
        "checks": checks,
        "strict_load_result": str(args.strict_load_result) if args.strict_load_result else None,
        "automatic_next_branch": (
            "wait_for_zero_wam_release_and_verify_its_wan_derivation" if passed
            else "complete_exact_wan_base_download_and_strict_load"
        ),
        "claim_boundary": (
            "Passing proves the exact public Wan2.2-TI2V-5B base and H200 residency only. "
            "It is not Zero-WAM, has no action branch or IFP proof, and cannot open SUGAR training."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "WAN22_BASE_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
