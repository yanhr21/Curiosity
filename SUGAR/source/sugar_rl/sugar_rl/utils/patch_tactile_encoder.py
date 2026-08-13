# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Anatomical patch-token encoder for Plan-15 online whole-hand tactile."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class AnatomicalPatchTactileEncoder(nn.Module):
    """Encode ``[history=4, hand=2, patch=27, channel=9]`` into 128-D.

    Official TacSL taxels are reduced before this boundary.  The encoder has
    no taxel-grid dimension and treats each physical anatomical patch as one
    token.  Channel scales are mandatory frozen physical normalization
    constants selected before matched training.
    """

    def __init__(
        self,
        channel_scales: Sequence[float],
        *,
        history_steps: int = 4,
        num_hands: int = 2,
        patches_per_hand: int = 27,
        channels_per_patch: int = 9,
        embedding_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_dim: int = 256,
    ) -> None:
        super().__init__()
        expected = (4, 2, 27, 9, 128, 3, 4, 256)
        actual = (
            history_steps,
            num_hands,
            patches_per_hand,
            channels_per_patch,
            embedding_dim,
            num_layers,
            num_heads,
            feedforward_dim,
        )
        if actual != expected:
            raise ValueError(
                "Plan-15 patch encoder geometry is frozen as "
                f"{expected}, got {actual}"
            )
        scales = torch.as_tensor(channel_scales, dtype=torch.float32)
        if tuple(scales.shape) != (channels_per_patch,):
            raise ValueError(
                f"channel_scales must have shape ({channels_per_patch},)"
            )
        if not torch.isfinite(scales).all() or torch.any(scales <= 0.0):
            raise ValueError("channel_scales must be positive and finite")

        self.history_steps = history_steps
        self.num_hands = num_hands
        self.patches_per_hand = patches_per_hand
        self.channels_per_patch = channels_per_patch
        self.embedding_dim = embedding_dim
        self.token_count = history_steps * num_hands * patches_per_hand
        self.expected_flat_dim = self.token_count * channels_per_patch
        self.register_buffer("channel_scales", scales, persistent=True)

        self.patch_projection = nn.Linear(
            channels_per_patch, embedding_dim, bias=False
        )
        # Identities are tied to the frozen anatomical/time ordering.  Their
        # values remain learned model parameters shared by Z/P/PS.
        self.time_embedding = nn.Parameter(
            torch.empty(history_steps, embedding_dim)
        )
        self.hand_embedding = nn.Parameter(torch.empty(num_hands, embedding_dim))
        self.patch_embedding = nn.Parameter(
            torch.empty(patches_per_hand, embedding_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.pool_query = nn.Parameter(torch.empty(embedding_dim))
        self._reset_identity_parameters()

    def _reset_identity_parameters(self) -> None:
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.hand_embedding, std=0.02)
        nn.init.normal_(self.patch_embedding, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)

    @property
    def output_dim(self) -> int:
        return self.embedding_dim

    def architecture_contract(self) -> dict[str, object]:
        return {
            "policy_unit": "physical_anatomical_patch",
            "input_shape_without_batch": [4, 2, 27, 9],
            "expected_flat_dim": self.expected_flat_dim,
            "patch_projection": [9, 128],
            "transformer_layers": 3,
            "attention_heads": 4,
            "feedforward_dim": 256,
            "output_dim": self.output_dim,
            "taxel_policy_dimension": False,
            "zero_input_maps_to_exact_zero": True,
        }

    def forward(self, tactile_obs: torch.Tensor) -> torch.Tensor:
        if tactile_obs.ndim != 2 or tactile_obs.shape[-1] != self.expected_flat_dim:
            raise ValueError(
                "Patch tactile observation shape mismatch: expected "
                f"(batch, {self.expected_flat_dim}), got {tuple(tactile_obs.shape)}"
            )
        if not torch.isfinite(tactile_obs).all():
            raise ValueError("patch tactile observation contains non-finite values")
        batch_size = tactile_obs.shape[0]
        patches = tactile_obs.reshape(
            batch_size,
            self.history_steps,
            self.num_hands,
            self.patches_per_hand,
            self.channels_per_patch,
        )
        active = patches.abs().amax(dim=-1) > 0.0
        normalized = patches / self.channel_scales
        tokens = self.patch_projection(normalized)
        tokens = (
            tokens
            + self.time_embedding[None, :, None, None, :]
            + self.hand_embedding[None, None, :, None, :]
            + self.patch_embedding[None, None, None, :, :]
        ).reshape(batch_size, self.token_count, self.embedding_dim)
        active = active.reshape(batch_size, self.token_count)

        # PyTorch attention requires at least one unmasked token.  Empty Z-arm
        # rows temporarily expose one token, then the final explicit gate maps
        # their embedding to exact zero.
        any_active = active.any(dim=-1)
        safe_active = active.clone()
        safe_active[~any_active, 0] = True
        encoded = self.transformer(
            tokens, src_key_padding_mask=~safe_active
        )
        scores = torch.einsum("btd,d->bt", encoded, self.pool_query)
        scores = scores / math.sqrt(float(self.embedding_dim))
        scores = scores.masked_fill(~safe_active, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bt,btd->bd", weights, encoded)
        return torch.where(any_active[:, None], pooled, torch.zeros_like(pooled))
