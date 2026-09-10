"""Materialize official empty UMT5 and repaired full-span demo latents on held H200."""
import os
import sys
import torch

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import repaired_overfit_config
from .model import import_wan_model


def main():
    if not os.environ.get("SLURM_STEP_ID") or not torch.cuda.is_available():
        raise RuntimeError("preparation requires retained Slurm compute GPU")
    config = repaired_overfit_config()
    config.validate()
    cache = config.resolved(config.neutral_text_cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        import_wan_model(config.resolved(config.wan_source))
        from wan.modules.t5 import T5EncoderModel
        base = config.resolved(config.wan_checkpoint)
        encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device="cuda:0",
            checkpoint_path=str(base / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(base / "google/umt5-xxl"))
        with torch.no_grad():
            raw = encoder([""], torch.device("cuda:0"))[0].cpu()
            repeated = encoder([""], torch.device("cuda:0"))[0].cpu()
        if raw.ndim != 2 or raw.shape[1] != 4096 or not torch.equal(raw, repeated) or not torch.isfinite(raw).all():
            raise RuntimeError("official empty text encoding failed exact repeat")
        payload = {"context": raw, "text": "", "encoder": "official_Wan_UMT5_XXL"}
        torch.save(payload, cache)
        reloaded = torch.load(cache, map_location="cpu", weights_only=True)
        if not torch.equal(raw, reloaded["context"]):
            raise RuntimeError("empty context cache round-trip failed")
        write_json_atomic(cache.with_suffix(".json"), {"text": "", "shape": list(raw.shape),
            "official_source": config.wan_source, "checkpoint": str(base / "models_t5_umt5-xxl-enc-bf16.pth"),
            "exact_repeat_and_disk_roundtrip": True, "hash_checks": False})
        del encoder, raw, repeated
        torch.cuda.empty_cache()
    emit_json_best_effort({"official_empty_text_cache_ready": str(cache)})
    from . import precompute_latents
    sys.argv = ["precompute_latents", "--repair-prompt-coverage"]
    precompute_latents.main()


if __name__ == "__main__":
    main()
