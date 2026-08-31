#!/usr/bin/env python3
"""Strict-load the exact public Wan2.2-TI2V-5B DiT and place it on H200.

This is a base-runtime audit only.  It intentionally does not add an action
branch, IFP head, robot adapter, or any other local approximation of Zero-WAM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def tensor_probe(model: Any) -> list[dict[str, Any]]:
    named = list(model.named_parameters())
    indices = sorted(set((0, len(named) // 2, len(named) - 1)))
    records = []
    for index in indices:
        name, parameter = named[index]
        sample = parameter.detach().reshape(-1)[:1024].float().cpu()
        records.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "finite": bool(sample.isfinite().all().item()),
                "sample_sum": float(sample.sum().item()),
                "sample_abs_max": float(sample.abs().max().item()),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "STRICT_LOAD_RESULT.json"
    result: dict[str, Any] = {
        "protocol": "official_wan22_ti2v_5b_h200_strict_load_v1",
        "passed": False,
        "provenance": "official_wan22_ti2v_5b_base",
        "source_commit": args.source_commit,
        "source_root": str(args.source_root.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "model_class": None,
        "parameter_count": 0,
        "checks": {},
        "claim_boundary": (
            "This audits the public Wan2.2-TI2V-5B video backbone only. It is not Zero-WAM, "
            "does not validate an action branch or IFP, and cannot open SUGAR training."
        ),
    }
    try:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("strict-load audit must run inside Slurm")
        source_root = args.source_root.resolve()
        checkpoint_root = args.checkpoint_root.resolve()
        sys.path.insert(0, str(source_root))

        import diffusers
        import flash_attn
        import torch
        import transformers

        # Upstream wan/__init__.py eagerly imports S2V/Animate and therefore
        # requires unrelated audio/video packages even for TI2V.  Register the
        # unmodified official wan/ directory as a namespace package so Python
        # imports the exact model module without executing optional entrypoints.
        wan_package = types.ModuleType("wan")
        wan_package.__path__ = [str(source_root / "wan")]
        wan_package.__package__ = "wan"
        sys.modules["wan"] = wan_package
        from wan.modules.model import WanModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_name = torch.cuda.get_device_name(0)
        if "H200" not in device_name:
            raise RuntimeError(f"expected H200, found {device_name}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        load_start = time.perf_counter()
        loaded = WanModel.from_pretrained(
            checkpoint_root,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        if isinstance(loaded, tuple):
            model, loading_info = loaded
        else:
            model, loading_info = loaded, {}
        cpu_load_seconds = time.perf_counter() - load_start
        missing = list(loading_info.get("missing_keys", []))
        unexpected = list(loading_info.get("unexpected_keys", []))
        mismatched = list(loading_info.get("mismatched_keys", []))
        errors = list(loading_info.get("error_msgs", []))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        cpu_probe = tensor_probe(model)

        residency_start = time.perf_counter()
        model = model.to(device="cuda:0", dtype=torch.bfloat16)
        model.eval()
        torch.cuda.synchronize(0)
        residency_seconds = time.perf_counter() - residency_start
        gpu_probe = tensor_probe(model)
        all_parameters_cuda_bf16 = all(
            parameter.device.type == "cuda" and parameter.dtype == torch.bfloat16
            for parameter in model.parameters()
        )
        peak_bytes = torch.cuda.max_memory_allocated(0)
        allocated_bytes = torch.cuda.memory_allocated(0)

        # Exact-model kernel smoke at the smallest valid latent grid.  This runs
        # every released DiT block; it is not presented as a video-quality or
        # Zero-WAM inference result.
        forward_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = [torch.zeros((48, 1, 2, 2), device="cuda:0", dtype=torch.bfloat16)]
            timestep = torch.tensor([500.0], device="cuda:0", dtype=torch.float32)
            context = [torch.zeros((1, 4096), device="cuda:0", dtype=torch.bfloat16)]
            forward_output = model(latent, timestep, context, seq_len=1)
        torch.cuda.synchronize(0)
        forward_seconds = time.perf_counter() - forward_start
        minimal_forward_ok = (
            len(forward_output) == 1
            and tuple(forward_output[0].shape) == (48, 1, 2, 2)
            and bool(forward_output[0].isfinite().all().item())
        )
        peak_bytes = torch.cuda.max_memory_allocated(0)

        checks = {
            "inside_slurm": True,
            "h200_device": "H200" in device_name,
            "torch_cuda_runtime_exact": torch.__version__ == "2.7.0+cu128",
            "diffusers_version_exact": diffusers.__version__ == "0.33.0",
            "transformers_version_exact": transformers.__version__ == "4.51.3",
            "flash_attention_version_exact": flash_attn.__version__ == "2.8.3.post1",
            "strict_cpu_load": not missing and not unexpected and not mismatched and not errors,
            "parameter_count_is_5b_scale": 4_000_000_000 < parameter_count < 6_000_000_000,
            "cpu_parameter_probes_finite": all(record["finite"] for record in cpu_probe),
            "h200_bfloat16_residency": all_parameters_cuda_bf16,
            "gpu_parameter_probes_finite": all(record["finite"] for record in gpu_probe),
            "exact_model_minimal_forward": minimal_forward_ok,
            "official_config_identity": (
                model.dim == 3072
                and model.ffn_dim == 14336
                and model.num_heads == 24
                and model.num_layers == 30
                and tuple(model.patch_size) == (1, 2, 2)
            ),
        }
        result.update(
            {
                "passed": all(checks.values()),
                "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "parameter_count": parameter_count,
                "buffer_count": sum(buffer.numel() for buffer in model.buffers()),
                "device": device_name,
                "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "diffusers_version": diffusers.__version__,
                "flash_attn_version": flash_attn.__version__,
                "transformers_version": transformers.__version__,
                "compatibility_glue": "namespace_package_bypass_of_eager_optional_wan_imports",
                "cpu_load_seconds": cpu_load_seconds,
                "h200_residency_seconds": residency_seconds,
                "minimal_forward_seconds": forward_seconds,
                "minimal_forward_shape": list(forward_output[0].shape),
                "cuda_allocated_bytes": allocated_bytes,
                "cuda_peak_allocated_bytes": peak_bytes,
                "loading_info": {
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                    "mismatched_keys": mismatched,
                    "error_msgs": errors,
                },
                "cpu_parameter_probes": cpu_probe,
                "gpu_parameter_probes": gpu_probe,
                "checks": checks,
            }
        )
        del model
        torch.cuda.empty_cache()
    except Exception as error:  # keep a machine-readable failure artifact
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
