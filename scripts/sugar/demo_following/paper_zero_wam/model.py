"""Full-width paper reconstruction of Zero-WAM on the Wan-2.2-TI2V-5B base.

This module implements the architectural claims in Zero-WAM v2 rather than a
small proxy: two independent 30-layer, 3072-wide experts; per-layer shared
attention with modality-owned QKV/FFN/output projections; a human-video RoPE
height offset; flow matching in video/action space; and four training-only IFP
heads initialized from the final video layer.
This is not the authors' released code. Disabling ``ifp_trunk_gradient`` is
an explicitly local training variant, not the paper's representation loss.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

from .config import PaperZeroWAMConfig


def import_wan_model(source_root: Path) -> type[nn.Module]:
    source_root = source_root.resolve()
    package = types.ModuleType("wan")
    package.__path__ = [str(source_root / "wan")]
    package.__package__ = "wan"
    sys.modules["wan"] = package
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(source_root / "wan" / "modules")]
    modules_package.__package__ = "wan.modules"
    sys.modules["wan.modules"] = modules_package
    from wan.modules.model import WanModel

    return WanModel


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("sinusoidal embedding dimension must be even")
    half = dim // 2
    position = position.to(torch.float64)
    scale = torch.pow(
        10000,
        -torch.arange(half, device=position.device, dtype=torch.float64) / half,
    )
    sinusoid = torch.outer(position.reshape(-1), scale)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).float()


def shifted_flow_time(uniform_time: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the released causal-VA rational SNR shift to a unit flow time."""

    if shift <= 0.0 or not math.isfinite(shift):
        raise ValueError("flow SNR shift must be finite and positive")
    return shift * uniform_time / (1.0 + (shift - 1.0) * uniform_time)


def inference_sigmas(
    steps: int, shift: float, *, device: torch.device
) -> torch.Tensor:
    """Return the released FlowMatchScheduler Euler boundaries, including zero."""

    if steps <= 0:
        raise ValueError("flow integration steps must be positive")
    base = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    return shifted_flow_time(base, shift)


def apply_rope(
    tensor: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor,
    official_precision: bool = False,
) -> torch.Tensor:
    """Apply Wan's 3-axis RoPE to explicit (time, height, width) positions."""

    if tensor.ndim != 4:
        raise ValueError("RoPE input must be [B,S,H,D]")
    complex_dim = tensor.shape[-1] // 2
    splits = [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3]
    freq_parts = freqs.to(tensor.device).split(splits, dim=1)
    # Wan's table stores exp(i * position * omega).  Recover omega from row
    # one so the same implementation also supports LingBot-VA's fractional
    # frame coordinate for actions and its -1/-1 action spatial sentinel.
    multiplier_parts: list[torch.Tensor] = []
    calculation_dtype = torch.float64 if official_precision else torch.float32
    for axis in range(3):
        omega = torch.angle(freq_parts[axis][1]).to(calculation_dtype)
        coordinate = positions[:, axis].to(calculation_dtype)
        phase = coordinate.unsqueeze(1) * omega.unsqueeze(0)
        multiplier = torch.polar(torch.ones_like(phase), phase)
        if official_precision:
            # Exact released table for integer video coordinates, including
            # the demo spatial offset; double precision for fractional action
            # coordinates and the negative spatial sentinel.
            integer = coordinate.long()
            in_table = (coordinate == integer) & (integer >= 0) & (integer < len(freq_parts[axis]))
            table = freq_parts[axis][integer.clamp(0, len(freq_parts[axis]) - 1)]
            multiplier = torch.where(in_table[:, None], table, multiplier)
        multiplier_parts.append(multiplier)
    multipliers = torch.cat(multiplier_parts, dim=1).unsqueeze(1)
    complex_tensor = torch.view_as_complex(
        tensor.to(calculation_dtype).reshape(*tensor.shape[:-1], complex_dim, 2)
    )
    rotated = torch.view_as_real(complex_tensor * multipliers).flatten(-2)
    return rotated.float() if official_precision else rotated.to(tensor.dtype)


def modulation(block: nn.Module, time_mod: torch.Tensor) -> tuple[torch.Tensor, ...]:
    base = block.modulation.to(device=time_mod.device, dtype=time_mod.dtype)
    values = (base.unsqueeze(0) + time_mod).chunk(6, dim=2)
    return tuple(value.squeeze(2) for value in values)


def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value.float() * (1.0 + scale.float()) + shift.float()


@dataclass
class TokenLayout:
    prompt: slice
    robot_history: slice
    video_target: slice
    action_history: slice
    action_target: slice
    video_positions: torch.Tensor
    action_positions: torch.Tensor
    attention_mask: torch.Tensor
    grid_height: int
    grid_width: int
    target_latent_frames: int
    official_attention_groups: Any = None
    self_only_queries: Any = None


_ATTENTION_GROUP_CACHE: dict[tuple, list] = {}


def official_mask_groups(mask: torch.Tensor, cache_key: tuple) -> list:
    """Group equal visible-key sets without changing any Boolean mask entry."""
    key = (str(mask.device), *cache_key)
    if key not in _ATTENTION_GROUP_CACHE:
        cpu = mask.detach().cpu().numpy()
        packed = np.packbits(cpu, axis=1)
        groups: dict[bytes, list[int]] = {}
        for index, row in enumerate(packed):
            groups.setdefault(row.tobytes(), []).append(index)
        plan = []
        for queries in groups.values():
            keys = np.flatnonzero(cpu[queries[0]])
            if not len(keys) or not np.array_equal(cpu[queries], np.broadcast_to(cpu[queries[0]], cpu[queries].shape)):
                raise AssertionError("official attention grouping changed visibility")
            plan.append((torch.tensor(queries, device=mask.device, dtype=torch.long),
                         torch.tensor(keys, device=mask.device, dtype=torch.long)))
        _ATTENTION_GROUP_CACHE[key] = plan
    return _ATTENTION_GROUP_CACHE[key]


_SELF_ONLY_CACHE: dict[tuple, Any] = {}


def self_only_query_index(mask: torch.Tensor, cache_key: tuple):
    """Queries that attend to exactly themselves and nothing else.

    With the prompt dropped, ``_layout`` sets the prompt block to an identity
    mask, so each of the 1600 prompt tokens forms its own visibility group.
    Routed through the grouped-varlen adapter that becomes ~1700 groups and
    ~214 FlashAttention launches per layer, versus 14 when the prompt is on --
    the dominant cost of a training step.

    Attention over a single visible key is analytically the identity on the
    value: softmax of one logit is exactly 1.0, so the output equals v[i].
    Handling these queries directly is bit-exact, not an approximation, and it
    changes no mask entry.
    """

    key = (str(mask.device), *cache_key)
    if key not in _SELF_ONLY_CACHE:
        counts = mask.sum(dim=1)
        single = counts == 1
        arange = torch.arange(mask.shape[0], device=mask.device)
        diagonal = mask[arange, arange]
        selected = single & diagonal
        _SELF_ONLY_CACHE[key] = (
            arange[selected] if bool(selected.any()) else None
        )
    return _SELF_ONLY_CACHE[key]


def official_masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              groups: list, self_only=None) -> torch.Tensor:
    """Adapter from the exact MoT mask to the released Wan FlashAttention API.

    Up to eight visibility groups share one varlen call. Q padding is discarded;
    k_lens excludes every padded key. No learned replacement attention is added.
    ``self_only`` carries the analytically-identity queries described above.
    """
    from wan.modules.attention import flash_attention
    if q.shape[0] != 1:
        raise ValueError("repaired attention requires microbatch one")
    result = torch.zeros_like(q)
    skip = None
    if self_only is not None and len(self_only):
        # ``result`` follows q's dtype (FP32 after the official-precision RoPE)
        # while v may still be BF16; cast so the identity copy is well-typed.
        result[0, self_only] = v[0, self_only].to(result.dtype)
        skip = set(self_only.tolist())
    pending = [
        (queries, keys) for queries, keys in groups
        if skip is None or len(keys) != 1 or int(queries[0]) not in skip
    ]
    for start in range(0, len(pending), 8):
        chunk = pending[start:start + 8]
        q_size = max(len(queries) for queries, _ in chunk)
        k_size = max(len(keys) for _, keys in chunk)
        queries_padded, keys_padded, values_padded = [], [], []
        for queries, keys in chunk:
            queries_padded.append(F.pad(q[0, queries], (0, 0, 0, 0, 0, q_size - len(queries))))
            keys_padded.append(F.pad(k[0, keys], (0, 0, 0, 0, 0, k_size - len(keys))))
            values_padded.append(F.pad(v[0, keys], (0, 0, 0, 0, 0, k_size - len(keys))))
        output = flash_attention(torch.stack(queries_padded), torch.stack(keys_padded),
            torch.stack(values_padded), k_lens=torch.tensor([len(keys) for _, keys in chunk],
                                                          device=q.device, dtype=torch.int32))
        for index, (queries, _) in enumerate(chunk):
            result[0, queries] = output[index, :len(queries)]
    return result


def ifp_head_execution_plan(
    validity: list[bool] | tuple[bool, ...], head_count: int
) -> tuple[tuple[int, bool], ...]:
    """Return a rank-invariant head call order plus rank-local loss validity."""

    if len(validity) != head_count:
        raise ValueError(
            f"expected {head_count} IFP validity entries, found {len(validity)}"
        )
    return tuple((head_index, bool(validity[head_index])) for head_index in range(head_count))


class ActionFlowHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, eps: float = 1.0e-6,
                 zero_init: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.projection = nn.Linear(hidden_dim, action_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)
        if zero_init:
            # Zero velocity at initialization blocks the upstream derivative
            # through this projection on the first backward pass. Once the
            # head updates, that path can open. This does not prove that
            # fitting within any fixed number of updates is impossible.
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        else:
            # Nonzero scaled initialization; its actual output scale and
            # benefit must be measured, not inferred from this formula alone.
            nn.init.normal_(self.projection.weight, std=hidden_dim**-0.5)
            nn.init.zeros_(self.projection.bias)

    def forward(self, tokens: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if time_embedding.dtype != torch.float32:
            raise TypeError("action timestep state must remain FP32")
        # Mirror Wan's video Head numerical boundary for the action-flow
        # velocity: norm, modulation and the shape-new projection are FP32.
        with torch.autocast(device_type="cuda", enabled=False):
            shift, scale = (
                self.modulation.unsqueeze(0) + time_embedding.unsqueeze(2)
            ).chunk(2, dim=2)
            value = self.norm(tokens.float()) * (1.0 + scale.squeeze(2))
            value = value + shift.squeeze(2)
            return self.projection(value)


class PaperMoTLayer(nn.Module):
    """One full-width pair of Wan blocks joined only at self-attention."""

    def __init__(self, video_block: nn.Module, action_block: nn.Module, num_heads: int,
                 official_precision: bool = False):
        super().__init__()
        self.video_block = video_block
        self.action_block = action_block
        self.num_heads = num_heads
        self.official_precision = official_precision

    def _attention_io(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        time_mod: torch.Tensor,
        positions: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation(
            block, time_mod
        )
        value = modulate(block.norm1(tokens), shift_msa, scale_msa).to(tokens.dtype)
        batch, sequence, _ = value.shape
        head_dim = value.shape[-1] // self.num_heads
        q = block.self_attn.norm_q(block.self_attn.q(value)).view(
            batch, sequence, self.num_heads, head_dim
        )
        k = block.self_attn.norm_k(block.self_attn.k(value)).view(
            batch, sequence, self.num_heads, head_dim
        )
        v = block.self_attn.v(value).view(batch, sequence, self.num_heads, head_dim)
        q = apply_rope(q, positions, freqs, self.official_precision)
        k = apply_rope(k, positions, freqs, self.official_precision)
        return q, k, v, (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    @staticmethod
    def _post(
        block: nn.Module,
        tokens: torch.Tensor,
        attention_output: torch.Tensor,
        gates: tuple[torch.Tensor, ...],
        text_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_msa, shift_mlp, scale_mlp, gate_mlp = gates
        # Projection/FFN matmuls remain under the caller's BF16 autocast while
        # their FSDP-sharded parameters stay FP32.  Only modulation and
        # residual accumulation cross the official Wan FP32 boundary.
        attention_projection = block.self_attn.o(attention_output)
        with torch.autocast(device_type="cuda", enabled=False):
            tokens = tokens.float() + attention_projection.float() * gate_msa.float()
        if text_context is not None:
            # Execute the inherited official module, including norm3 and the
            # ungated cross-attention residual, before the gated FFN.
            tokens = tokens + block.cross_attn(block.norm3(tokens), text_context, None)
        with torch.autocast(device_type="cuda", enabled=False):
            feedforward_input = modulate(block.norm2(tokens), shift_mlp, scale_mlp)
        feedforward = block.ffn(feedforward_input)
        with torch.autocast(device_type="cuda", enabled=False):
            tokens = tokens + feedforward.float() * gate_mlp.float()
        # Wan retains this gated residual state in FP32 between blocks.  The
        # surrounding autocast still runs eligible QKV/FFN matmuls in BF16.
        return tokens

    def forward(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_time_mod: torch.Tensor,
        action_time_mod: torch.Tensor,
        video_positions: torch.Tensor,
        action_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: torch.Tensor,
        text_context: torch.Tensor | None = None,
        attention_groups: Any = None,
        self_only: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vq, vk, vv, vgates = self._attention_io(
            self.video_block, video_tokens, video_time_mod, video_positions, freqs
        )
        aq, ak, av, agates = self._attention_io(
            self.action_block, action_tokens, action_time_mod, action_positions, freqs
        )
        q = torch.cat([vq, aq], dim=1).transpose(1, 2)
        k = torch.cat([vk, ak], dim=1).transpose(1, 2)
        v = torch.cat([vv, av], dim=1).transpose(1, 2)
        if self.official_precision:
            if attention_groups is None:
                raise ValueError("repaired MoT requires exact official mask grouping")
            attended = official_masked_attention(q.transpose(1, 2), k.transpose(1, 2),
                v.transpose(1, 2), attention_groups, self_only).flatten(2)
        else:
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0).transpose(1, 2).flatten(2)
        video_length = video_tokens.shape[1]
        video_attention = attended[:, :video_length]
        action_attention = attended[:, video_length:]
        return (
            self._post(self.video_block, video_tokens, video_attention, vgates, text_context),
            self._post(self.action_block, action_tokens, action_attention, agates, text_context),
        )


class IFPHead(nn.Module):
    """One paper IFP denoiser: one final-video-layer clone plus video head."""

    def __init__(self, block: nn.Module, output_head: nn.Module, num_heads: int,
                 official_precision: bool = False):
        super().__init__()
        self.block = copy.deepcopy(block)
        self.output_head = copy.deepcopy(output_head)
        self.num_heads = num_heads
        self.official_precision = official_precision

    def forward(
        self,
        tokens: torch.Tensor,
        time_mod: torch.Tensor,
        time_embedding: torch.Tensor,
        positions: torch.Tensor,
        freqs: torch.Tensor,
        target_slice: slice,
        text_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation(
            self.block, time_mod
        )
        value = modulate(self.block.norm1(tokens), shift_msa, scale_msa).to(tokens.dtype)
        batch, sequence, width = value.shape
        head_dim = width // self.num_heads
        q = self.block.self_attn.norm_q(self.block.self_attn.q(value)).view(
            batch, sequence, self.num_heads, head_dim
        )
        k = self.block.self_attn.norm_k(self.block.self_attn.k(value)).view(
            batch, sequence, self.num_heads, head_dim
        )
        v = self.block.self_attn.v(value).view(batch, sequence, self.num_heads, head_dim)
        q = apply_rope(q, positions, freqs, self.official_precision).transpose(1, 2)
        k = apply_rope(k, positions, freqs, self.official_precision).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.official_precision:
            from wan.modules.attention import flash_attention
            attended = flash_attention(q.transpose(1, 2), k.transpose(1, 2),
                                       v.transpose(1, 2)).flatten(2)
        else:
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).transpose(1, 2).flatten(2)
        tokens = PaperMoTLayer._post(self.block, tokens, attended,
            (gate_msa, shift_mlp, scale_mlp, gate_mlp), text_context)
        target_tokens = tokens[:, target_slice]
        target_time = time_embedding[:, target_slice]
        return self.output_head(target_tokens, target_time)


class PaperZeroWAM(nn.Module):
    def __init__(self, video_backbone: nn.Module, config: PaperZeroWAMConfig):
        super().__init__()
        config.validate()
        self.config = config
        if (
            video_backbone.dim != config.hidden_dim
            or video_backbone.ffn_dim != config.ffn_dim
            or video_backbone.num_layers != config.num_layers
            or video_backbone.num_heads != config.num_heads
        ):
            raise ValueError("loaded Wan backbone does not match Zero-WAM v2 geometry")

        self.video_patch_embedding = video_backbone.patch_embedding
        self.video_time_embedding = video_backbone.time_embedding
        self.video_time_projection = video_backbone.time_projection
        self.video_head = video_backbone.head
        if config.repaired_conditioning:
            self.video_text_embedding = video_backbone.text_embedding
            context = torch.load(config.resolved(config.neutral_text_cache),
                                 map_location="cpu", weights_only=True)
            raw = context["context"]
            if (context.get("text") != "" or context.get("encoder") != "official_Wan_UMT5_XXL"
                    or raw.ndim != 2 or raw.shape[1] != 4096 or not 0 < raw.shape[0] <= 512
                    or not bool(torch.isfinite(raw).all())):
                raise ValueError("invalid official empty-text conditioning cache")
            # Match WanModel.forward: zero-pad RAW UMT5 output to text_len
            # before the inherited learned text_embedding, context_lens=None.
            padded = F.pad(raw.float(), (0, 0, 0, 512 - raw.shape[0])).unsqueeze(0)
            self.register_buffer("neutral_text_context", padded, persistent=True)
        self.register_buffer("freqs", video_backbone.freqs, persistent=False)

        original_blocks = list(video_backbone.blocks)
        action_blocks = [copy.deepcopy(block) for block in original_blocks]
        self.mot_layers = nn.ModuleList(
            [
                PaperMoTLayer(video_block, action_block, config.num_heads, config.repaired_conditioning)
                for video_block, action_block in zip(original_blocks, action_blocks, strict=True)
            ]
        )
        self.action_encoder = nn.Linear(config.action_dim, config.action_hidden_dim)
        self.action_time_embedding = copy.deepcopy(video_backbone.time_embedding)
        self.action_time_projection = copy.deepcopy(video_backbone.time_projection)
        self.action_head = ActionFlowHead(config.action_hidden_dim, config.action_dim,
                                  zero_init=config.zero_init_action_head)

        fusion_width = len(config.ifp_fusion_layers) * config.hidden_dim
        self.ifp_fusion = nn.Sequential(
            nn.Linear(fusion_width, config.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.ifp_heads = nn.ModuleList(
            [
                IFPHead(original_blocks[-1], self.video_head, config.num_heads, config.repaired_conditioning)
                for _ in range(config.ifp_heads)
            ]
        )
        self.gradient_checkpointing = True

    @classmethod
    def from_wan_pretrained(
        cls,
        config: PaperZeroWAMConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "PaperZeroWAM":
        WanModel = import_wan_model(config.resolved(config.wan_source))
        # The shared FUSE mount delivers the 32 GB checkpoint at ~35 MB/s, which
        # costs ~8 minutes per launch.  PZW_WAN_CHECKPOINT allows a locally
        # staged copy of the same files.
        checkpoint = os.environ.get("PZW_WAN_CHECKPOINT") or config.resolved(
            config.wan_checkpoint
        )
        loaded = WanModel.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        video_backbone, loading_info = loaded if isinstance(loaded, tuple) else (loaded, {})
        failures = {
            key: loading_info.get(key, [])
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(key)
        }
        if failures:
            raise RuntimeError(f"Wan initialization failed: {failures}")
        return cls(video_backbone, config)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def upcast_precision_critical_modules(self) -> dict[str, int]:
        """Keep Wan's FP32 numerical boundary intact under BF16 residency.

        When the bulk of the model is resident in BF16 (needed to fit an 80 GB
        card), the modules that Wan deliberately evaluates in FP32 -- timestep
        embeddings/projections, every LayerNorm/RMSNorm, the AdaLN modulation
        tables and the output heads -- must stay FP32 or ``_time_state`` and
        the disabled-autocast blocks raise on dtype mismatch.  These are ~140M
        parameters (~0.5 GB), so the memory cost is negligible.
        """

        upcast = 0
        for module in self.modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or type(
                module
            ).__name__ in ("WanLayerNorm", "WanRMSNorm", "Head", "ActionFlowHead"):
                module.float()
                upcast += sum(p.numel() for p in module.parameters(recurse=False))
        for module in (
            self.video_time_embedding,
            self.video_time_projection,
            self.action_time_embedding,
            self.action_time_projection,
            self.video_head,
            self.action_head,
        ):
            module.float()
        for name, parameter in self.named_parameters():
            if name.endswith("modulation") and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
                upcast += parameter.numel()
        total = sum(
            p.numel() for p in self.parameters() if p.dtype == torch.float32
        )
        return {"fp32_parameter_count": total}

    def _text_context(self) -> torch.Tensor | None:
        if not self.config.repaired_conditioning:
            return None
        return self.video_text_embedding(self.neutral_text_context)

    def _time_state(
        self,
        values: torch.Tensor,
        embedding: nn.Module,
        projection: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence = values.shape
        # Match Wan's numerical boundary: the caller runs BF16 autocast, while
        # timestep embeddings/modulations are explicitly produced in FP32.
        # CUDA autocast only accepts FP16/BF16 as its target dtype.  Disable
        # the surrounding BF16 autocast explicitly for Wan's FP32 timestep
        # embedding/modulation path instead of requesting unsupported FP32
        # autocast (which merely warns and disables itself at runtime).
        with torch.autocast(device_type="cuda", enabled=False):
            encoded = embedding(
                sinusoidal_embedding_1d(256, values.reshape(-1) * 1000.0)
                .reshape(batch, sequence, 256)
                .float()
            )
            projected = projection(encoded).reshape(
                batch, sequence, 6, self.config.hidden_dim
            )
        if encoded.dtype != torch.float32 or projected.dtype != torch.float32:
            raise TypeError("Wan timestep state must remain FP32")
        return encoded, projected

    def _patchify(self, latents: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        value = self.video_patch_embedding(latents)
        grid_height, grid_width = value.shape[-2:]
        return value.flatten(2).transpose(1, 2), grid_height, grid_width

    def _video_positions(
        self,
        prompt_frames: int,
        robot_history_frames: int,
        target_frames: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions: list[list[int]] = []
        for time in range(prompt_frames):
            positions.extend(
                [
                    [time, y + self.config.human_rope_height_offset, x]
                    for y in range(height)
                    for x in range(width)
                ]
            )
        for time in range(robot_history_frames + target_frames):
            positions.extend(
                [[time, y, x] for y in range(height) for x in range(width)]
            )
        return torch.tensor(positions, dtype=torch.long, device=device)

    @staticmethod
    def _action_positions(
        length: int, device: torch.device, actions_per_latent: int = 20
    ) -> torch.Tensor:
        """LingBot-VA action grid: latent frame plus an intra-frame fraction."""

        if actions_per_latent <= 0:
            raise ValueError("actions_per_latent must be positive")
        indices = torch.arange(length, device=device, dtype=torch.float32)
        frame = torch.div(indices, actions_per_latent, rounding_mode="floor")
        within = torch.remainder(indices, actions_per_latent)
        time = frame + (within + 1.0) / float(actions_per_latent + 1)
        sentinel = torch.full_like(time, -1.0)
        return torch.stack([time, sentinel, sentinel], dim=1)

    def _layout(
        self,
        prompt_frames: int,
        robot_history_frames: int,
        target_frames: int,
        grid_height: int,
        grid_width: int,
        action_history: int,
        action_target: int,
        device: torch.device,
        prompt_enabled: bool,
    ) -> TokenLayout:
        tokens_per_frame = grid_height * grid_width
        p0, p1 = 0, prompt_frames * tokens_per_frame
        h0, h1 = p1, p1 + robot_history_frames * tokens_per_frame
        v0, v1 = h1, h1 + target_frames * tokens_per_frame
        video_length = v1
        ah0, ah1 = video_length, video_length + action_history
        at0, at1 = ah1, ah1 + action_target
        total = at1
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        if prompt_enabled:
            mask[p0:p1, p0:p1] = True
        else:
            mask[p0:p1, p0:p1] = torch.eye(p1 - p0, dtype=torch.bool, device=device)

        for frame in range(robot_history_frames):
            query = slice(h0 + frame * tokens_per_frame, h0 + (frame + 1) * tokens_per_frame)
            if prompt_enabled:
                mask[query, p0:p1] = True
            mask[query, h0 : h0 + (frame + 1) * tokens_per_frame] = True
            action_limit = min(action_history, frame * 20)
            mask[query, ah0 : ah0 + action_limit] = True

        if prompt_enabled:
            mask[v0:v1, p0:p1] = True
        mask[v0:v1, h0:h1] = True
        mask[v0:v1, v0:v1] = True
        mask[v0:v1, ah0:ah1] = True

        for action_index in range(action_history):
            query = ah0 + action_index
            visible_robot_frames = min(robot_history_frames, action_index // 20 + 1)
            mask[query, h0 : h0 + visible_robot_frames * tokens_per_frame] = True
            mask[query, ah0 : query + 1] = True

        mask[at0:at1, h0:h1] = True
        mask[at0:at1, v0:v1] = True
        mask[at0:at1, ah0:ah1] = True
        mask[at0:at1, at0:at1] = True
        # Action-target queries deliberately never see the prompt (p0:p1).
        # This is the paper's factorization: task intent must reach the action
        # branch through the predicted robot future, not by a direct shortcut
        # to the demonstration.  Combined with the hardcoded prompt_enabled
        # =False on the action pass, it means the action expert receives no
        # prompt gradient -- which is intended, not a bug.
        if not mask.any(dim=1).all():
            raise AssertionError("attention mask contains an empty query row")
        return TokenLayout(
            prompt=slice(p0, p1),
            robot_history=slice(h0, h1),
            video_target=slice(v0, v1),
            action_history=slice(0, action_history),
            action_target=slice(action_history, action_history + action_target),
            video_positions=self._video_positions(
                prompt_frames,
                robot_history_frames,
                target_frames,
                grid_height,
                grid_width,
                device,
            ),
            action_positions=self._action_positions(action_history + action_target, device),
            attention_mask=mask,
            grid_height=grid_height,
            grid_width=grid_width,
            target_latent_frames=target_frames,
            official_attention_groups=(official_mask_groups(mask,
                (prompt_frames, robot_history_frames, target_frames, grid_height, grid_width,
                 action_history, action_target, prompt_enabled)) if self.config.repaired_conditioning else None),
            self_only_queries=(self_only_query_index(mask,
                (prompt_frames, robot_history_frames, target_frames, grid_height, grid_width,
                 action_history, action_target, prompt_enabled)) if self.config.repaired_conditioning else None),
        )

    def _unpatchify(
        self, tokens: torch.Tensor, frames: int, height: int, width: int
    ) -> torch.Tensor:
        patch_t, patch_h, patch_w = self.config.patch_size
        batch = tokens.shape[0]
        value = tokens.view(
            batch,
            frames,
            height,
            width,
            patch_t,
            patch_h,
            patch_w,
            self.config.visual_channels,
        )
        value = torch.einsum("bfhwpqrc->bcfphqwr", value)
        return value.reshape(
            batch,
            self.config.visual_channels,
            frames * patch_t,
            height * patch_h,
            width * patch_w,
        )

    def _run_ifp(
        self,
        raw_robot_history_tokens: torch.Tensor,
        raw_action_history_tokens: torch.Tensor,
        fused_current: torch.Tensor,
        future_target: torch.Tensor,
        flow_time: torch.Tensor,
        future_time_index: int,
        head_index: int,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        noisy_future, _, _ = self._patchify(future_target)
        batch = noisy_future.shape[0]
        context = torch.cat(
            [raw_robot_history_tokens, raw_action_history_tokens, fused_current, noisy_future], dim=1
        )
        target_start = context.shape[1] - noisy_future.shape[1]
        target_slice = slice(target_start, context.shape[1])
        zero_count = context.shape[1] - noisy_future.shape[1]
        time_values = torch.cat(
            [
                torch.zeros((batch, zero_count), device=context.device),
                flow_time[:, None].expand(batch, noisy_future.shape[1]),
            ],
            dim=1,
        )
        time_embedding, time_mod = self._time_state(
            time_values, self.video_time_embedding, self.video_time_projection
        )
        history_frames = raw_robot_history_tokens.shape[1] // (grid_height * grid_width)
        fused_frames = fused_current.shape[1] // (grid_height * grid_width)
        future_frames = noisy_future.shape[1] // (grid_height * grid_width)
        def spatial_positions(times: list[int]) -> torch.Tensor:
            return torch.tensor(
                [
                    [time, y, x]
                    for time in times
                    for y in range(grid_height)
                    for x in range(grid_width)
                ],
                device=context.device,
                dtype=torch.long,
            )

        history_positions = spatial_positions(list(range(history_frames)))
        fused_positions = spatial_positions(
            list(range(history_frames, history_frames + fused_frames))
        )
        future_positions = spatial_positions(
            list(range(future_time_index + 1, future_time_index + future_frames + 1))
        )
        visual_count = raw_robot_history_tokens.shape[1] + fused_current.shape[1] + noisy_future.shape[1]
        action_count = raw_action_history_tokens.shape[1]
        positions = torch.cat(
            [
                history_positions,
                self._action_positions(action_count, context.device),
                fused_positions,
                future_positions,
            ],
            dim=0,
        )
        if positions.shape[0] != context.shape[1] or visual_count + action_count != context.shape[1]:
            raise AssertionError("IFP position layout mismatch")
        output_tokens = self.ifp_heads[head_index](
            context,
            time_mod,
            time_embedding,
            positions,
            self.freqs,
            target_slice,
            self._text_context(),
        )
        return self._unpatchify(output_tokens, future_frames, grid_height, grid_width)

    def _predict_main_velocities(
        self,
        batch: dict[str, Any],
        video_state: torch.Tensor,
        action_state: torch.Tensor,
        video_time_value: torch.Tensor,
        action_time_value: torch.Tensor,
        prompt_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the deployed main branches without any IFP module."""

        prompt_latents = batch["prompt_latents"]
        robot_history = batch["robot_history_latents"]
        action_history = batch["action_history"]
        video_latents = torch.cat([prompt_latents, robot_history, video_state], dim=2)
        video_tokens, grid_height, grid_width = self._patchify(video_latents)
        action_values = torch.cat([action_history, action_state], dim=1)
        action_tokens = self.action_encoder(action_values)
        layout = self._layout(
            prompt_latents.shape[2],
            robot_history.shape[2],
            video_state.shape[2],
            grid_height,
            grid_width,
            action_history.shape[1],
            action_state.shape[1],
            video_tokens.device,
            prompt_enabled,
        )
        video_time = torch.zeros((1, video_tokens.shape[1]), device=video_tokens.device)
        video_time[:, layout.video_target] = video_time_value
        action_time = torch.zeros((1, action_tokens.shape[1]), device=action_tokens.device)
        action_time[:, layout.action_target] = action_time_value
        video_time_embedding, video_time_mod = self._time_state(
            video_time, self.video_time_embedding, self.video_time_projection
        )
        action_time_embedding, action_time_mod = self._time_state(
            action_time, self.action_time_embedding, self.action_time_projection
        )
        text_context = self._text_context()
        for layer in self.mot_layers:
            args = (
                video_tokens,
                action_tokens,
                video_time_mod,
                action_time_mod,
                layout.video_positions,
                layout.action_positions,
                layout.attention_mask,
                self.freqs,
                text_context,
                layout.official_attention_groups,
                layout.self_only_queries,
            )
            if self.gradient_checkpointing and self.training:
                video_tokens, action_tokens = checkpoint(
                    layer, *args, use_reentrant=False
                )
            else:
                video_tokens, action_tokens = layer(*args)
        video_output_tokens = self.video_head(
            video_tokens[:, layout.video_target],
            video_time_embedding[:, layout.video_target],
        )
        video_velocity = self._unpatchify(
            video_output_tokens,
            video_state.shape[2],
            grid_height,
            grid_width,
        )
        action_velocity = self.action_head(
            action_tokens[:, layout.action_target],
            action_time_embedding[:, layout.action_target],
        )
        return video_velocity, action_velocity

    @torch.no_grad()
    def _sample_next_chunk(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Generate video first, then decode action only from that generated future."""

        video_steps = int(
            batch.get("video_inference_steps", self.config.video_inference_steps)
        )
        action_steps = int(
            batch.get("action_inference_steps", self.config.action_inference_steps)
        )
        guidance = float(
            batch.get("video_guidance_scale", self.config.video_guidance_scale)
        )
        action_guidance = float(
            batch.get("action_guidance_scale", self.config.action_guidance_scale)
        )
        if video_steps <= 0 or action_steps <= 0:
            raise ValueError("flow integration step counts must be positive")
        if action_guidance != 1.0:
            raise ValueError("paper action CFG is fixed at 1.0; no action CFG branch exists")
        video = torch.randn_like(batch["video_target_latents"])
        # These target-action tokens have no path to video under the audited
        # mask. Their value is not a causal video-conditioning repair. Retain
        # the selected RNG draw convention for reproducible action sampling.
        if self.config.inactive_action_token_noise:
            inactive_action_target = torch.randn_like(batch["action_target"])
        else:
            inactive_action_target = torch.zeros_like(batch["action_target"])
        video_sigmas = inference_sigmas(
            video_steps, self.config.video_snr_shift, device=video.device
        )
        for index in range(video_steps):
            time_value = video_sigmas[index : index + 1]
            action_time = video.new_ones((1,))
            conditional, _ = self._predict_main_velocities(
                batch,
                video,
                inactive_action_target,
                time_value,
                action_time,
                True,
            )
            if guidance != 1.0:
                unconditional, _ = self._predict_main_velocities(
                    batch,
                    video,
                    inactive_action_target,
                    time_value,
                    action_time,
                    False,
                )
                velocity = unconditional + guidance * (conditional - unconditional)
            else:
                velocity = conditional
            delta = video_sigmas[index + 1] - video_sigmas[index]
            video = (video + velocity.to(video.dtype) * delta.to(video.dtype)).to(
                video.dtype
            )

        action = torch.randn_like(batch["action_target"])
        video_time = video.new_zeros((1,))
        action_sigmas = inference_sigmas(
            action_steps, self.config.action_snr_shift, device=action.device
        )
        for index in range(action_steps):
            action_time = action_sigmas[index : index + 1]
            # The human-video prefix is isolated in the inverse-dynamics pass.
            # Its only deployed route to action is the generated robot future.
            _, velocity = self._predict_main_velocities(
                batch,
                video,
                action,
                video_time,
                action_time,
                False,
            )
            delta = action_sigmas[index + 1] - action_sigmas[index]
            action = (action + velocity.to(action.dtype) * delta.to(action.dtype)).to(
                action.dtype
            )
        return {
            "predicted_video_latents": video,
            "predicted_normalized_actions": action,
        }

    @torch.no_grad()
    def _score_prompt_condition(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Score prompt-conditioned video, then action through its predicted future.

        This fixed-noise score is the evaluation counterpart of the paper's
        video-first factorization.  It estimates the clean next-video chunk
        from a single flow state and supplies only that estimate—not the human
        prefix—to the inverse-dynamics pass.
        """

        video_clean = batch["video_target_latents"]
        action_clean = batch["action_target"]
        generator = batch.get("generator")
        video_noise = torch.randn(
            video_clean.shape,
            device=video_clean.device,
            dtype=video_clean.dtype,
            generator=generator,
        )
        action_noise = torch.randn(
            action_clean.shape,
            device=action_clean.device,
            dtype=action_clean.dtype,
            generator=generator,
        )
        video_t = shifted_flow_time(
            torch.rand((1,), device=video_clean.device, generator=generator),
            self.config.video_snr_shift,
        )
        action_t = shifted_flow_time(
            torch.rand((1,), device=action_clean.device, generator=generator),
            self.config.action_snr_shift,
        )
        video_noisy = (
            (1.0 - video_t[:, None, None, None, None]) * video_clean
            + video_t[:, None, None, None, None] * video_noise
        )
        action_noisy = (
            (1.0 - action_t[:, None, None]) * action_clean
            + action_t[:, None, None] * action_noise
        )
        video_velocity_target = video_noise - video_clean
        action_velocity_target = action_noise - action_clean
        inactive_action = (
            torch.randn_like(action_clean)
            if self.config.inactive_action_token_noise
            else torch.zeros_like(action_clean)
        )
        video_prediction, _ = self._predict_main_velocities(
            batch,
            video_noisy,
            inactive_action,
            video_t,
            video_t.new_ones((1,)),
            bool(batch["prompt_enabled"]),
        )
        # NOTE ON LEAKAGE: x_t - t*v_hat expands to (1-t)*x0 + t*(eps - v_hat).
        # It therefore carries a (1-t)-weighted verbatim copy of the ground
        # truth future regardless of model quality (mean weight ~0.25 at
        # video_snr_shift=5).  Conditioning the action pass on it and calling
        # the result "generated-future-conditioned" is not supportable.
        # Callers that need a genuine open-loop number must use the real
        # two-phase sampler in _sample_next_chunk.  The tensor is still
        # returned for diagnostics, but under an explicit name.
        predicted_clean_future = video_noisy - (
            video_t[:, None, None, None, None] * video_prediction.to(video_noisy.dtype)
        )
        _, action_prediction = self._predict_main_velocities(
            batch,
            predicted_clean_future,
            action_noisy,
            video_t.new_zeros((1,)),
            action_t,
            False,
        )
        return {
            "video_loss": F.mse_loss(
                video_prediction.float(), video_velocity_target.float()
            ),
            "action_loss": F.mse_loss(
                action_prediction.float(), action_velocity_target.float()
            ),
            "predicted_future_latents": predicted_clean_future,
            "ground_truth_weight_in_action_condition": float(
                (1.0 - video_t).reshape(-1)[0]
            ),
            "action_condition_is_ground_truth_contaminated": True,
            "predicted_action_velocity": action_prediction,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if bool(batch.get("prompt_score", False)):
            return self._score_prompt_condition(batch)
        if bool(batch.get("inference", False)):
            return self._sample_next_chunk(batch)
        prompt_latents = batch["prompt_latents"]
        robot_history = batch["robot_history_latents"]
        video_clean = batch["video_target_latents"]
        action_history = batch["action_history"]
        action_clean = batch["action_target"]
        if prompt_latents.shape[0] != 1:
            raise ValueError("formal FSDP layout expects one packed sample per rank")

        generator = batch.get("generator")
        video_noise = torch.randn(video_clean.shape, device=video_clean.device, dtype=video_clean.dtype, generator=generator)
        action_noise = torch.randn(action_clean.shape, device=action_clean.device, dtype=action_clean.dtype, generator=generator)
        video_t = shifted_flow_time(
            torch.rand((1,), device=video_clean.device, generator=generator),
            self.config.video_snr_shift,
        )
        action_t = shifted_flow_time(
            torch.rand((1,), device=action_clean.device, generator=generator),
            self.config.action_snr_shift,
        )
        video_noisy = (1.0 - video_t[:, None, None, None, None]) * video_clean + video_t[:, None, None, None, None] * video_noise
        action_noisy = (1.0 - action_t[:, None, None]) * action_clean + action_t[:, None, None] * action_noise
        video_velocity = video_noise - video_clean
        action_velocity = action_noise - action_clean

        video_latents = torch.cat([prompt_latents, robot_history, video_noisy], dim=2)
        video_tokens, grid_height, grid_width = self._patchify(video_latents)
        action_values = torch.cat([action_history, action_noisy], dim=1)
        action_tokens = self.action_encoder(action_values)
        prompt_frames = prompt_latents.shape[2]
        history_frames = robot_history.shape[2]
        target_frames = video_clean.shape[2]
        prompt_enabled = bool(batch["prompt_enabled"])
        layout = self._layout(
            prompt_frames,
            history_frames,
            target_frames,
            grid_height,
            grid_width,
            action_history.shape[1],
            action_clean.shape[1],
            video_tokens.device,
            prompt_enabled,
        )

        video_time = torch.zeros((1, video_tokens.shape[1]), device=video_tokens.device)
        video_time[:, layout.video_target] = video_t
        action_time = torch.zeros((1, action_tokens.shape[1]), device=action_tokens.device)
        action_time[:, layout.action_target] = action_t
        video_time_embedding, video_time_mod = self._time_state(
            video_time, self.video_time_embedding, self.video_time_projection
        )
        action_time_embedding, action_time_mod = self._time_state(
            action_time, self.action_time_embedding, self.action_time_projection
        )
        raw_robot_history = video_tokens[:, layout.robot_history]
        raw_action_history = action_tokens[:, layout.action_history]
        if not self.config.ifp_trunk_gradient:
            # Detach this history route only. The IFP future-token path still
            # uses shared embedding/conditioning modules, so this is not a
            # complete isolation of all IFP gradients from shared parameters.
            raw_robot_history = raw_robot_history.detach()
            raw_action_history = raw_action_history.detach()
        fusion_features: list[torch.Tensor] = []

        text_context = self._text_context()
        for layer_index, layer in enumerate(self.mot_layers):
            args = (
                video_tokens,
                action_tokens,
                video_time_mod,
                action_time_mod,
                layout.video_positions,
                layout.action_positions,
                layout.attention_mask,
                self.freqs,
                text_context,
                layout.official_attention_groups,
                layout.self_only_queries,
            )
            if self.gradient_checkpointing and self.training:
                video_tokens, action_tokens = checkpoint(layer, *args, use_reentrant=False)
            else:
                video_tokens, action_tokens = layer(*args)
            if layer_index in self.config.ifp_fusion_layers:
                tap = video_tokens[:, layout.video_target]
                # Paper IFP supervises the main robot-video representation.
                # Detaching this tap changes that objective; it is a local
                # gradient-interference diagnostic, not a faithful repair or
                # an established explanation of the historical video failure.
                fusion_features.append(tap if self.config.ifp_trunk_gradient else tap.detach())

        video_output_tokens = self.video_head(
            video_tokens[:, layout.video_target],
            video_time_embedding[:, layout.video_target],
        )
        video_prediction = self._unpatchify(
            video_output_tokens, target_frames, grid_height, grid_width
        )
        # The paper factorizes video prediction before inverse dynamics.  The
        # action-flow training pass is therefore teacher-forced with the clean
        # next robot-video latent, not the independently noised video-flow
        # state used above.  Inference replaces this clean future with the
        # generated video from the first sampling phase.  The ICL prefix is
        # isolated in this pass, matching L_a(ell): it must not create a
        # shortcut around the future-video bottleneck.
        teacher_video_prediction, action_prediction = self._predict_main_velocities(
            batch,
            video_clean,
            action_noisy,
            video_clean.new_zeros((1,)),
            action_t,
            False,
        )
        del teacher_video_prediction
        video_loss_unscaled = F.mse_loss(
            video_prediction.float(), video_velocity.float()
        )
        action_loss_unscaled = F.mse_loss(
            action_prediction.float(), action_velocity.float()
        )
        loss_scales = batch.get("loss_scales", {})
        video_scale = float(loss_scales.get("video", 1.0))
        action_scale = float(loss_scales.get("action", 1.0))
        if not math.isfinite(video_scale) or video_scale <= 0.0:
            raise ValueError("video global-element loss scale must be finite and positive")
        if not math.isfinite(action_scale) or action_scale <= 0.0:
            raise ValueError("action global-element loss scale must be finite and positive")
        video_loss = video_loss_unscaled * video_scale
        action_loss = action_loss_unscaled * action_scale

        fused_current = self.ifp_fusion(torch.cat(fusion_features, dim=-1))
        ifp_loss = video_loss.new_zeros(())
        ifp_active = 0
        ifp_losses: list[torch.Tensor] = []
        ifp_losses_unscaled: list[torch.Tensor] = []
        ifp_scales = loss_scales.get("ifp")
        if ifp_scales is not None and len(ifp_scales) != self.config.ifp_heads:
            raise ValueError("IFP global-element loss scale geometry changed")
        for head_index, valid in ifp_head_execution_plan(
            batch["ifp_valid"], self.config.ifp_heads
        ):
            future_clean = batch["ifp_target_latents"][head_index]
            weight = self.config.ifp_weights[head_index]
            # Every rank must enter every FSDP-wrapped IFP head in the same
            # order.  Validity differs across the rank-local variable-length
            # chunks, so conditionally skipping a head would desynchronize
            # FSDP collectives.  Invalid targets still traverse an exact-zero
            # autograd path; they contribute neither supervision nor gradient.
            future_noise = torch.randn(
                future_clean.shape,
                device=future_clean.device,
                dtype=future_clean.dtype,
                generator=generator,
            )
            future_t = shifted_flow_time(
                torch.rand((1,), device=future_clean.device, generator=generator),
                self.config.video_snr_shift,
            )
            future_noisy = (1.0 - future_t[:, None, None, None, None]) * future_clean + future_t[:, None, None, None, None] * future_noise
            future_velocity = future_noise - future_clean
            prediction = self._run_ifp(
                raw_robot_history,
                raw_action_history,
                fused_current,
                future_noisy,
                future_t,
                int(batch["ifp_latent_starts"][head_index]),
                head_index,
                grid_height,
                grid_width,
            )
            head_loss_unscaled = F.mse_loss(
                prediction.float(), future_velocity.float()
            )
            head_scale = (
                float(ifp_scales[head_index])
                if ifp_scales is not None
                else float(bool(valid))
            )
            if not math.isfinite(head_scale) or head_scale < 0.0:
                raise ValueError("IFP global-element loss scale must be finite and nonnegative")
            if bool(valid) != (head_scale > 0.0):
                raise ValueError("IFP loss scale does not match rank-local target validity")
            head_loss = head_loss_unscaled * head_scale
            ifp_losses_unscaled.append(head_loss_unscaled)
            ifp_losses.append(head_loss)
            ifp_loss = ifp_loss + float(weight) * head_loss
            ifp_active += int(bool(valid))

        total = (
            self.config.lambda_video * video_loss
            + self.config.lambda_action * action_loss
            + self.config.lambda_ifp * ifp_loss
        )
        return {
            "loss": total,
            "video_loss": video_loss,
            "action_loss": action_loss,
            "ifp_loss": ifp_loss,
            "video_loss_unscaled": video_loss_unscaled,
            "action_loss_unscaled": action_loss_unscaled,
            "ifp_active_heads": torch.tensor(float(ifp_active), device=total.device),
            "main_pass_count": torch.tensor(2.0, device=total.device),
            **{f"ifp_head_{index}_loss": value for index, value in enumerate(ifp_losses)},
            **{
                f"ifp_head_{index}_loss_unscaled": value
                for index, value in enumerate(ifp_losses_unscaled)
            },
        }
