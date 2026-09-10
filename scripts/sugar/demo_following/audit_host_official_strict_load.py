#!/usr/bin/env python3
"""Construct official HOST and enforce strict original/adapted checkpoint loading.

No model is implemented here.  The script imports the pinned HOST package and only
performs key/shape/equality checks around its released constructor.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ORIGINAL_ACTION_DIM = 22
ORIGINAL_PHYSICAL_ACTION_DIM = 20
ORIGINAL_PROPRIO_DIM = 20
SUGAR_EXECUTED_ACTION_DIM = 29
SUGAR_ACTION_DIM = SUGAR_EXECUTED_ACTION_DIM + 2
SUGAR_PROPRIO_DIM = 121
COMPONENTS = (
    "mot",
    "proprio_encoder",
    "progress_encoder",
    "progress_decoder",
    "visual_encoder",
)
ADAPTER_ALLOWLIST = {
    "mot.mixtures.action.action_encoder.weight": ([1024, 22], [1024, 31]),
    "mot.mixtures.action.head.weight": ([22, 1024], [31, 1024]),
    "mot.mixtures.action.head.bias": ([22], [31]),
    "proprio_encoder.weight": ([4096, 20], [4096, 121]),
}
ADAPTER_INITIALIZATION_SEED = 20260831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("original", "adapted", "both"), default="both"
    )
    parser.add_argument(
        "--reuse-original-wan-vae",
        action="store_true",
        help=(
            "Load the official Wan2.2_VAE.pth cache instead of the redirected "
            "DiffSynth safetensors mirror. Both paths use HOST's released "
            "hash-checked Wan VAE loader and state-dict converter."
        ),
    )
    parser.add_argument("--dino-local-repo", type=Path)
    parser.add_argument("--dino-weights", type=Path)
    parser.add_argument("--siglip-weights", type=Path)
    return parser.parse_args()


def require_h200() -> dict[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not torch.cuda.is_available():
        raise RuntimeError("strict-load audit requires a CUDA Slurm allocation")
    device = torch.cuda.get_device_name(0)
    if "H200" not in device.upper():
        raise RuntimeError(f"strict-load audit requires H200, found {device!r}")
    return {"slurm_job_id": job_id, "cuda_device": device}


def load_model_config(
    path: Path,
    adapted: bool,
    reuse_original_wan_vae: bool,
    backbone_paths: dict[str, str] | None,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["model"]
    config.pop("_target_", None)
    config["load_text_encoder"] = False
    # The released HOST checkpoint contains the complete video and action experts.
    # This official flag avoids downloading a redundant pre-HOST initialization.
    config["skip_dit_load_from_pretrain"] = True
    config["action_dit_pretrained_path"] = None
    if reuse_original_wan_vae:
        config["redirect_common_files"] = False
    if backbone_paths is not None:
        config["visual_encoder"].update(backbone_paths)
    if adapted:
        config["proprio_dim"] = SUGAR_PROPRIO_DIM
        config["video_dit_config"]["action_dim"] = SUGAR_ACTION_DIM
        config["action_dit_config"]["action_dim"] = SUGAR_ACTION_DIM
    return config


def flattened_shapes(model: torch.nn.Module) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for component in COMPONENTS:
        module = getattr(model, component)
        for key, tensor in module.state_dict().items():
            shapes[f"{component}.{key}"] = list(tensor.shape)
    return shapes


def checkpoint_shapes(payload: dict[str, Any]) -> dict[str, list[int]]:
    return {
        f"{component}.{key}": list(tensor.shape)
        for component in COMPONENTS
        for key, tensor in payload[component].items()
    }


def compare_contract(
    model: torch.nn.Module, payload: dict[str, Any], adapted: bool
) -> dict[str, Any]:
    actual = flattened_shapes(model)
    expected = checkpoint_shapes(payload)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = {
        key: {"checkpoint": expected[key], "model": actual[key]}
        for key in sorted(set(actual).intersection(expected))
        if actual[key] != expected[key]
    }
    if adapted:
        if missing or unexpected or set(mismatched) != set(ADAPTER_ALLOWLIST):
            raise RuntimeError(
                f"adapted key/shape contract failed: missing={missing}, "
                f"unexpected={unexpected}, mismatched={mismatched}"
            )
        for key, (old_shape, new_shape) in ADAPTER_ALLOWLIST.items():
            if mismatched[key] != {"checkpoint": old_shape, "model": new_shape}:
                raise RuntimeError(f"adapter shape mismatch for {key}: {mismatched[key]}")
    elif missing or unexpected or mismatched:
        raise RuntimeError(
            f"original strict contract failed: missing={missing}, "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    return {
        "model_tensor_count": len(actual),
        "checkpoint_tensor_count": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": mismatched,
    }


@torch.no_grad()
def copy_shared_action_auxiliary_channels(
    model: torch.nn.Module, payload: dict[str, Any]
) -> dict[str, int]:
    """Preserve HOST's progress/mask channels without inventing robot mappings."""
    source = payload["mot"]
    target = model.mot.state_dict()
    old_aux = slice(ORIGINAL_PHYSICAL_ACTION_DIM, ORIGINAL_ACTION_DIM)
    new_aux = slice(SUGAR_EXECUTED_ACTION_DIM, SUGAR_ACTION_DIM)
    copies = (
        (
            "mixtures.action.action_encoder.weight",
            (slice(None), old_aux),
            (slice(None), new_aux),
        ),
        (
            "mixtures.action.head.weight",
            (old_aux, slice(None)),
            (new_aux, slice(None)),
        ),
        ("mixtures.action.head.bias", (old_aux,), (new_aux,)),
    )
    copied_numel = 0
    for key, source_index, target_index in copies:
        source_slice = source[key][source_index]
        target[key][target_index].copy_(
            source_slice.to(device=target[key].device, dtype=target[key].dtype)
        )
        if not torch.equal(
            target[key][target_index], source_slice.to(target[key].device)
        ):
            raise RuntimeError(f"shared auxiliary action copy failed for {key}")
        copied_numel += source_slice.numel()
    return {
        "bitwise_copied_auxiliary_tensors": len(copies),
        "bitwise_copied_auxiliary_numel": copied_numel,
    }


@torch.no_grad()
def load_and_prove_equal(
    model: torch.nn.Module, payload: dict[str, Any], adapted: bool
) -> dict[str, int]:
    equal_tensors = 0
    equal_numel = 0
    skipped: set[str] = set()
    for component in COMPONENTS:
        module = getattr(model, component)
        source = payload[component]
        if adapted:
            local_allowlist = {
                key.removeprefix(component + ".")
                for key in ADAPTER_ALLOWLIST
                if key.startswith(component + ".")
            }
            filtered = {key: value for key, value in source.items() if key not in local_allowlist}
            result = module.load_state_dict(filtered, strict=False)
            if set(result.missing_keys) != local_allowlist or result.unexpected_keys:
                raise RuntimeError(
                    f"adapted {component} load failed: missing={result.missing_keys}, "
                    f"unexpected={result.unexpected_keys}"
                )
            skipped.update(f"{component}.{key}" for key in local_allowlist)
        else:
            module.load_state_dict(source, strict=True)
        current = module.state_dict()
        for key, expected_tensor in source.items():
            full_key = f"{component}.{key}"
            if full_key in skipped:
                continue
            actual_tensor = current[key]
            if actual_tensor.dtype != expected_tensor.dtype:
                raise RuntimeError(
                    f"dtype mismatch after load for {full_key}: "
                    f"{actual_tensor.dtype} != {expected_tensor.dtype}"
                )
            if not torch.equal(actual_tensor, expected_tensor.to(actual_tensor.device)):
                raise RuntimeError(f"non-bitwise inherited tensor after load: {full_key}")
            equal_tensors += 1
            equal_numel += expected_tensor.numel()
    if adapted and skipped != set(ADAPTER_ALLOWLIST):
        raise RuntimeError(f"adapter skipped tensor set mismatch: {sorted(skipped)}")
    new_interface_numel = sum(
        getattr(model, key.split(".", 1)[0])
        .state_dict()[key.split(".", 1)[1]]
        .numel()
        for key in skipped
    )
    auxiliary_copy = (
        copy_shared_action_auxiliary_channels(model, payload)
        if adapted
        else {
            "bitwise_copied_auxiliary_tensors": 0,
            "bitwise_copied_auxiliary_numel": 0,
        }
    )
    return {
        "bitwise_equal_inherited_tensors": equal_tensors,
        "bitwise_equal_inherited_numel": equal_numel,
        "new_interface_tensors": len(skipped),
        "new_interface_numel": new_interface_numel,
        **auxiliary_copy,
        "fresh_embodiment_interface_numel": (
            new_interface_numel
            - auxiliary_copy["bitwise_copied_auxiliary_numel"]
        ),
    }


def audit_one(
    payload: dict[str, Any],
    config_path: Path,
    adapted: bool,
    reuse_original_wan_vae: bool,
    backbone_paths: dict[str, str] | None,
) -> dict[str, Any]:
    from self_grounded_prediction.runtime import create_self_grounded_predictor

    torch.manual_seed(ADAPTER_INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(ADAPTER_INITIALIZATION_SEED)
    torch.cuda.reset_peak_memory_stats()
    config = load_model_config(
        config_path,
        adapted=adapted,
        reuse_original_wan_vae=reuse_original_wan_vae,
        backbone_paths=backbone_paths,
    )
    model = create_self_grounded_predictor(
        **config, model_dtype=torch.bfloat16, device="cuda"
    )
    contract = compare_contract(model, payload, adapted=adapted)
    equality = load_and_prove_equal(model, payload, adapted=adapted)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "mode": (
            "adapted_31_action_channels_29_executed_121_proprio"
            if adapted
            else "original_22_action_channels_20_physical_20_proprio"
        ),
        "passed": True,
        "contract": contract,
        "equality": equality,
        "model_parameter_count": parameter_count,
        "constructor_seed": ADAPTER_INITIALIZATION_SEED,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "model_paths": model.model_paths,
        "vae_cache_variant": (
            "official_Wan2.2_VAE.pth"
            if reuse_original_wan_vae
            else "official_DiffSynth_redirected_safetensors"
        ),
        "visual_backbone_bootstrap": backbone_paths,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    args = parse_args()
    runtime = require_h200()
    raw_backbone_paths = (
        args.dino_local_repo,
        args.dino_weights,
        args.siglip_weights,
    )
    if any(path is not None for path in raw_backbone_paths) and not all(
        path is not None for path in raw_backbone_paths
    ):
        raise ValueError(
            "--dino-local-repo, --dino-weights and --siglip-weights must be "
            "provided together"
        )
    backbone_paths = None
    if all(path is not None for path in raw_backbone_paths):
        resolved_paths = [path.resolve() for path in raw_backbone_paths]
        if not resolved_paths[0].is_dir() or not all(
            path.is_file() for path in resolved_paths[1:]
        ):
            raise FileNotFoundError(f"invalid visual backbone paths: {resolved_paths}")
        backbone_paths = {
            "backbone_local_repo": str(resolved_paths[0]),
            "backbone_weights_path": str(resolved_paths[1]),
            "siglip_local_weights_path": str(resolved_paths[2]),
        }
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    modes = {
        "original": (False,),
        "adapted": (True,),
        "both": (False, True),
    }[args.mode]
    reports = [
        audit_one(
            payload,
            args.config.resolve(),
            adapted,
            reuse_original_wan_vae=args.reuse_original_wan_vae,
            backbone_paths=backbone_paths,
        )
        for adapted in modes
    ]
    output = {
        "audit": "official HOST constructor strict-load and SUGAR interface inheritance",
        "passed": True,
        "runtime": runtime,
        "adapter_allowlist": ADAPTER_ALLOWLIST,
        "reports": reports,
    }
    output_path = args.output_dir.resolve() / "HOST_OFFICIAL_STRICT_LOAD_AUDIT.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(json.dumps({"passed": True, "output": str(output_path), "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
