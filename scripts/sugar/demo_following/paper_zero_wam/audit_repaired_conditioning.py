"""No-update numerical check using the actual released 5B/30-layer Wan backbone.

This tests inherited video/block computation, not Zero-WAM task success. It does
not construct or train a reduced replacement network.
"""
import os
import json
from types import SimpleNamespace
import warnings

import torch
import torch.nn.functional as F

from .artifacts import write_json_atomic, emit_json_best_effort
from .config import repaired_overfit_config
from .model import (PaperMoTLayer, PaperZeroWAM, IFPHead, import_wan_model, modulation,
                    official_mask_groups, official_masked_attention)


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("official full-width numerical audit requires a GPU")
    if not os.environ.get("SLURM_STEP_ID") and not os.environ.get("PZW_ALLOW_NON_SLURM"):
        raise RuntimeError(
            "run inside a retained srun step, or set PZW_ALLOW_NON_SLURM=1"
        )
    config = repaired_overfit_config()
    output = config.resolved(config.latent_cache) / "OFFICIAL_FORWARD_AUDIT.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if (previous.get("passed") is True and previous.get("mask_forward_backward_passed") is True
                and previous.get("official_layers") == 30 and previous.get("video_output_relative_l2") == 0
                and previous.get("ifp_output_relative_l2") == 0):
            emit_json_best_effort({"completed_official_forward_audit_reused": str(output), "optimizer_updates": 0})
            return
        raise RuntimeError("refusing to overwrite or bypass a failed official forward audit")
    warnings.filterwarnings("ignore", message="In CUDA autocast.*")
    WanModel = import_wan_model(config.resolved(config.wan_source))
    backbone, info = WanModel.from_pretrained(config.resolved(config.wan_checkpoint),
        torch_dtype=torch.float32, low_cpu_mem_usage=True, output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"strict Wan initialization failed: {info}")
    assert sum(p.numel() for p in backbone.parameters()) == 4_999_787_712
    backbone = backbone.cuda().eval()
    context_raw = torch.load(config.resolved(config.neutral_text_cache),
                            map_location="cuda", weights_only=True)["context"].float()
    torch.manual_seed(291701)
    torch.cuda.manual_seed_all(291701)
    latents = torch.randn(1, 48, 2, 4, 4, device="cuda")
    time = torch.full((1, 8), 0.5, device="cuda")
    owner = SimpleNamespace(config=config, num_heads=config.num_heads, official_precision=True)
    freqs = backbone.freqs.cuda()
    positions = torch.tensor([[t, y, x] for t in range(2) for y in range(2) for x in range(2)], device="cuda")
    grid = torch.tensor([[2, 2, 2]])
    lengths = torch.tensor([8])
    records = []
    # The tolerance is declared before outcomes: BF16 flash/SDPA paths may
    # round differently, but both must agree within 2% relative L2.
    relative_l2_limit = 0.02
    def error(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8))
    # Tensor-only forward/backward test of actual MoT visibility, at full
    # 24x128 attention width; no auxiliary learned network or optimizer.
    owner._video_positions = lambda *args: PaperZeroWAM._video_positions(owner, *args)
    owner._action_positions = PaperZeroWAM._action_positions
    mask_records = []
    for enabled in (False, True):
        layout = PaperZeroWAM._layout(owner, 2, 3, 2, 1, 1, 4, 4, torch.device("cuda"), enabled)
        length = layout.attention_mask.shape[0]
        with torch.enable_grad():
            inputs = [torch.randn(1, length, 24, 128, device="cuda", dtype=torch.bfloat16,
                                  requires_grad=True) for _ in range(3)]
            adapted_out = official_masked_attention(*inputs, layout.official_attention_groups)
            reference_out = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in inputs),
                attn_mask=layout.attention_mask[None, None], dropout_p=0.0).transpose(1, 2)
            adapted_grad = torch.autograd.grad(adapted_out.float().square().mean(), inputs)
            reference_grad = torch.autograd.grad(reference_out.float().square().mean(), inputs)
        mask_records.append({"prompt_enabled": enabled, "forward_relative_l2": error(adapted_out, reference_out),
            "qkv_gradient_relative_l2": [error(a, b) for a, b in zip(adapted_grad, reference_grad)]})
    mask_passed = all(row["forward_relative_l2"] <= relative_l2_limit
        and all(value <= relative_l2_limit for value in row["qkv_gradient_relative_l2"]) for row in mask_records)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        context = backbone.text_embedding(F.pad(context_raw, (0, 0, 0, 512 - context_raw.shape[0])).unsqueeze(0))
        e, e0 = PaperZeroWAM._time_state(owner, time, backbone.time_embedding, backbone.time_projection)
        reference = backbone.patch_embedding(latents).flatten(2).transpose(1, 2)
        adapted = reference.clone()
        cross_effects = []
        for index, block in enumerate(backbone.blocks):
            block_input = reference
            reference = block(reference, e0, lengths, grid, freqs, context, None)
            q, k, v, gates = PaperMoTLayer._attention_io(owner, block, adapted, e0, positions, freqs)
            groups = official_mask_groups(torch.ones(8, 8, dtype=torch.bool, device="cuda"),
                                          ("full_width_video_audit",))
            attended = official_masked_attention(q, k, v, groups).flatten(2)
            adapted = PaperMoTLayer._post(block, adapted, attended, gates, context)
            # Isolate the restored post-attention path using identical inputs.
            gates_raw = modulation(block, e0)
            attention = block.self_attn(block.norm1(block_input).float() * (1 + gates_raw[1]) + gates_raw[0],
                                       lengths, grid, freqs)
            residual = block_input.float() + attention.float() * gates_raw[2]
            cross = block.cross_attn(block.norm3(residual), context, None)
            cross_effects.append(float(cross.float().norm()))
            row = {"layer": index, "relative_l2": error(adapted, reference),
                   "cross_attention_output_norm": cross_effects[-1]}
            records.append(row)
        reference_video = backbone.unpatchify(backbone.head(reference, e), grid)[0]
        adapted_video = PaperZeroWAM._unpatchify(owner, backbone.head(adapted, e), 2, 2, 2)[0]
        output_error = error(adapted_video, reference_video)
        # Real IFP clone, full width, compared with the corresponding official
        # last block + official output head on exactly the same inputs.
        ifp = IFPHead(backbone.blocks[-1], backbone.head, config.num_heads, True).cuda().eval()
        actual_ifp = ifp(block_input, e0, e, positions, freqs, slice(0, 8), context)
        reference_ifp = backbone.head(backbone.blocks[-1](block_input, e0, lengths, grid, freqs, context, None), e)
        ifp_error = error(actual_ifp, reference_ifp)
    passed = (mask_passed and len(records) == 30 and all(row["relative_l2"] <= relative_l2_limit for row in records)
              and output_error <= relative_l2_limit and ifp_error <= relative_l2_limit
              and all(value > 0 for value in cross_effects)
              and all(torch.isfinite(t).all().item() for t in (adapted_video, reference_video, actual_ifp)))
    result = {"protocol": "repaired_official_full_width_wan_forward_audit_v1", "passed": passed,
              "official_parameters": 4_999_787_712, "official_layers": 30, "width": 3072,
              "optimizer_updates": 0, "relative_l2_limit": relative_l2_limit,
              "layer_comparisons": records, "video_output_relative_l2": output_error,
              "ifp_output_relative_l2": ifp_error,
              "mask_forward_backward_passed": mask_passed, "mask_forward_backward_comparisons": mask_records,
              "inherited_text_projection_parameters": sum(p.numel() for p in backbone.text_embedding.parameters()),
              "scope": "actual_official_video_path_and_full_IFP_block_not_demo_following", "hash_checks": False}
    write_json_atomic(output, result)
    emit_json_best_effort(result)
    if not passed:
        raise RuntimeError("restored official forward numerical check failed")


if __name__ == "__main__":
    main()
