#!/usr/bin/env python3
"""Materialize HOST's exact official UMT5 embedding for the neutral SUGAR prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
HOST_ROOT = ROOT / (
    "experiments/demo_following/host_official_v1/"
    "official_source_9f3bba57792aa5053ac600b1b7a4625f96ea6662/policy_training"
)
WAN_ROOT = ROOT / (
    "experiments/demo_following/zero_wam_official_v1/wan_base_runtime_v1/"
    "Wan2.2-TI2V-5B"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/host_official_v1/sugar_adapter_v1/"
    "HOST_NEUTRAL_CONTEXT.pt"
)
PROMPT = "Follow the demonstrated motion."
CONTEXT_LEN = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("official HOST neutral-context materialization requires Slurm H200")
    sys.path.insert(0, str(HOST_ROOT / "src"))
    from self_grounded_prediction.models.wan22.helpers.loader import _load_registered_model
    from self_grounded_prediction.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    encoder_path = WAN_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = WAN_ROOT / "google/umt5-xxl"
    if not encoder_path.is_file() or not tokenizer_path.is_dir():
        raise FileNotFoundError("exact official Wan UMT5 artifacts are incomplete")

    torch.cuda.reset_peak_memory_stats()
    encoder = _load_registered_model(
        str(encoder_path),
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device="cuda",
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=str(tokenizer_path), seq_len=CONTEXT_LEN, clean="whitespace"
    )
    with torch.inference_mode():
        ids, mask = tokenizer([PROMPT], return_mask=True, add_special_tokens=True)
        ids = ids.cuda()
        mask = mask.to(device="cuda", dtype=torch.bool)
        context = encoder(ids, mask)
    if context.shape != (1, CONTEXT_LEN, 4096) or mask.shape != (1, CONTEXT_LEN):
        raise RuntimeError(f"official UMT5 output geometry drift: {context.shape} {mask.shape}")
    if not torch.isfinite(context).all() or not mask.any():
        raise RuntimeError("official UMT5 neutral context is invalid")

    payload = {
        "context": context[0].cpu().to(torch.bfloat16).contiguous(),
        "mask": mask[0].cpu().contiguous(),
        "prompt": PROMPT,
        "context_len": CONTEXT_LEN,
        "encoder_sha256": sha256(encoder_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    loaded = torch.load(args.output, map_location="cpu", weights_only=True)
    if not torch.equal(payload["context"], loaded["context"]) or not torch.equal(
        payload["mask"], loaded["mask"]
    ):
        raise RuntimeError("neutral context cache round-trip mismatch")
    report = {
        "status": "pass",
        "slurm_job_id": job_id,
        "cuda_device": torch.cuda.get_device_name(0),
        "prompt": PROMPT,
        "context_shape": list(payload["context"].shape),
        "mask_shape": list(payload["mask"].shape),
        "valid_tokens": int(payload["mask"].sum()),
        "context_dtype": str(payload["context"].dtype),
        "encoder_path": str(encoder_path.relative_to(ROOT)),
        "encoder_size": encoder_path.stat().st_size,
        "encoder_sha256": payload["encoder_sha256"],
        "cache_path": str(args.output.relative_to(ROOT)),
        "cache_sha256": sha256(args.output),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    report_path = args.output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
