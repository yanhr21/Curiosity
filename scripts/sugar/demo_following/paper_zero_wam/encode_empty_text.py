#!/usr/bin/env python3
"""Encode the official empty-text UMT5 context that repaired conditioning needs.

``repaired_overfit_config()`` requires ``neutral_text_cache``: the raw UMT5-XXL
encoder output for the empty string, which the model zero-pads to text_len and
feeds through Wan's inherited learned ``text_embedding``.  Before the repair the
cross-attention modules were skipped entirely (text_context=None), so ~22.7% of
the loaded Wan computation was absent from the forward and received no
gradient.  This restores it.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan-source", type=Path, required=True)
    ap.add_argument("--wan-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.wan_source.resolve()
    pkg = types.ModuleType("wan")
    pkg.__path__ = [str(root / "wan")]
    pkg.__package__ = "wan"
    sys.modules["wan"] = pkg
    mod = types.ModuleType("wan.modules")
    mod.__path__ = [str(root / "wan" / "modules")]
    mod.__package__ = "wan.modules"
    sys.modules["wan.modules"] = mod
    from wan.modules.t5 import T5EncoderModel

    ckpt = args.wan_checkpoint.resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(ckpt / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(ckpt / "google" / "umt5-xxl"),
    )
    with torch.no_grad():
        context = encoder([""], device)[0]
    raw = context.float().cpu()
    # The model validates: 2-D, 4096-wide, 0 < rows <= 512, all finite.
    if raw.ndim != 2 or raw.shape[1] != 4096 or not 0 < raw.shape[0] <= 512:
        raise ValueError(f"unexpected empty-text context geometry: {tuple(raw.shape)}")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("empty-text context is not finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"context": raw, "text": "", "encoder": "official_Wan_UMT5_XXL"}, args.output
    )
    print(f"wrote {tuple(raw.shape)} -> {args.output}")


if __name__ == "__main__":
    main()
