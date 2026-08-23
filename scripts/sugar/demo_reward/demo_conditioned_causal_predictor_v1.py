#!/usr/bin/env python3
"""Causal selected-demo trajectory-mismatch predictor for SUGAR.

This is a new project model, not a T-REX, XIRL, RoboCLIP, or SMP
implementation.  It reuses SUGAR's exact spatial TacSL encoder and PyTorch's
standard TransformerEncoder.  Runtime inputs are limited to a fixed numeric
demo condition, a causal actor-visible prefix, its validity mask, and direct
spatial TacSL history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from sugar_rl.utils.tactile_actor_critic import SpatialTactileEncoder


COMPONENT_NAMES = (
    "body",
    "box_position",
    "box_rotation_6d",
    "box_velocity",
)

EVENT_TARGET_NAMES = (
    "body_mse",
    "box_position_mse",
    "box_rotation_6d_mse",
    "box_velocity_mse",
    "left_hand_contact_mismatch",
    "right_hand_contact_mismatch",
    "left_foot_contact_mismatch",
    "right_foot_contact_mismatch",
    "left_hand_duration_mismatch",
    "right_hand_duration_mismatch",
    "left_foot_duration_mismatch",
    "right_foot_duration_mismatch",
    "motion_regime_mismatch",
)


class DemoConditionedCausalPredictorV1(nn.Module):
    """Predict future selected-demo mismatch and aleatoric uncertainty."""

    def __init__(
        self,
        *,
        policy_dim: int,
        policy_history_steps: int,
        demo_windows: int,
        demo_window_steps: int,
        demo_feature_dim: int,
        tactile_history_steps: int,
        tactile_num_hands: int,
        tactile_channels_per_hand: int,
        tactile_grid_shape: Sequence[int],
        tactile_encoder_channels: Sequence[int],
        tactile_embedding_dim_per_hand: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        demo_mean: torch.Tensor,
        demo_std: torch.Tensor,
        tactile_rms: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if tuple(target_scale.shape) != (len(COMPONENT_NAMES),):
            raise ValueError("target_scale must have one entry per component")
        if tuple(state_mean.shape) != (policy_dim,):
            raise ValueError("state_mean shape mismatch")
        if tuple(state_std.shape) != (policy_dim,):
            raise ValueError("state_std shape mismatch")
        if tuple(demo_mean.shape) != (demo_feature_dim,):
            raise ValueError("demo_mean shape mismatch")
        if tuple(demo_std.shape) != (demo_feature_dim,):
            raise ValueError("demo_std shape mismatch")
        if tuple(tactile_rms.shape) != (tactile_channels_per_hand,):
            raise ValueError("tactile_rms shape mismatch")

        self.policy_dim = int(policy_dim)
        self.policy_history_steps = int(policy_history_steps)
        self.demo_windows = int(demo_windows)
        self.demo_window_steps = int(demo_window_steps)
        self.demo_feature_dim = int(demo_feature_dim)
        self.tactile_history_steps = int(tactile_history_steps)
        self.tactile_num_hands = int(tactile_num_hands)
        self.tactile_channels_per_hand = int(tactile_channels_per_hand)
        self.tactile_grid_shape = (
            int(tactile_grid_shape[0]),
            int(tactile_grid_shape[1]),
        )
        self.d_model = int(d_model)
        self.total_tokens = (
            1
            + self.demo_windows
            + self.policy_history_steps
            + self.tactile_history_steps
        )

        self.register_buffer("state_mean", state_mean.float().clone())
        self.register_buffer(
            "state_std", torch.clamp(state_std.float().clone(), min=1.0e-6)
        )
        self.register_buffer("demo_mean", demo_mean.float().clone())
        self.register_buffer(
            "demo_std", torch.clamp(demo_std.float().clone(), min=1.0e-6)
        )
        # Scale-only TacSL normalization preserves the exact zero field.
        self.register_buffer(
            "tactile_rms",
            torch.clamp(tactile_rms.float().clone(), min=1.0e-8),
        )
        self.register_buffer(
            "target_scale",
            torch.clamp(target_scale.float().clone(), min=1.0e-8),
        )

        self.demo_projection = nn.Sequential(
            nn.Linear(
                self.demo_window_steps * self.demo_feature_dim,
                self.d_model,
            ),
            nn.LayerNorm(self.d_model),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(self.policy_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.tactile_encoder = SpatialTactileEncoder(
            channels_per_hand=self.tactile_channels_per_hand,
            num_hands=self.tactile_num_hands,
            grid_shape=self.tactile_grid_shape,
            encoder_channels=tactile_encoder_channels,
            embedding_dim=int(tactile_embedding_dim_per_hand),
            activation="elu",
        )
        self.tactile_projection = nn.Sequential(
            nn.Linear(self.tactile_encoder.output_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.total_tokens, self.d_model)
        )
        self.token_type_embedding = nn.Embedding(4, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.component_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.d_model, 2)
                for name in COMPONENT_NAMES
            }
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def _validate_inputs(
        self,
        policy_prefix: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
        tactile_history: torch.Tensor,
        demo_condition: torch.Tensor,
    ) -> None:
        batch_size = policy_prefix.shape[0]
        expected = {
            "policy_prefix": (
                batch_size,
                self.policy_history_steps,
                self.policy_dim,
            ),
            "prefix_valid_mask": (
                batch_size,
                self.policy_history_steps,
            ),
            "tactile_history": (
                batch_size,
                self.tactile_history_steps,
                self.tactile_num_hands,
                self.tactile_channels_per_hand,
                self.tactile_grid_shape[0],
                self.tactile_grid_shape[1],
            ),
            "demo_condition": (
                batch_size,
                self.demo_windows,
                self.demo_window_steps,
                self.demo_feature_dim,
            ),
        }
        actual = {
            "policy_prefix": tuple(policy_prefix.shape),
            "prefix_valid_mask": tuple(prefix_valid_mask.shape),
            "tactile_history": tuple(tactile_history.shape),
            "demo_condition": tuple(demo_condition.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(
                    f"{name} shape mismatch: {actual[name]} != {shape}"
                )
        if prefix_valid_mask.dtype is not torch.bool:
            raise TypeError("prefix_valid_mask must be bool")

    def forward(
        self,
        *,
        policy_prefix: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
        tactile_history: torch.Tensor,
        demo_condition: torch.Tensor,
        zero_demo: bool = False,
        zero_tactile: bool = False,
    ) -> Mapping[str, torch.Tensor]:
        self._validate_inputs(
            policy_prefix,
            prefix_valid_mask,
            tactile_history,
            demo_condition,
        )
        batch_size = policy_prefix.shape[0]

        state = (policy_prefix - self.state_mean) / self.state_std
        state = state * prefix_valid_mask.unsqueeze(-1)
        state_tokens = self.state_projection(state)

        demo = (demo_condition - self.demo_mean) / self.demo_std
        if zero_demo:
            demo = torch.zeros_like(demo)
        demo_tokens = self.demo_projection(
            demo.reshape(
                batch_size,
                self.demo_windows,
                self.demo_window_steps * self.demo_feature_dim,
            )
        )

        tactile = tactile_history / self.tactile_rms.view(
            1, 1, 1, self.tactile_channels_per_hand, 1, 1
        )
        if zero_tactile:
            tactile = torch.zeros_like(tactile)
        tactile_flat = tactile.reshape(
            batch_size * self.tactile_history_steps, -1
        )
        tactile_tokens = self.tactile_projection(
            self.tactile_encoder(tactile_flat)
        ).reshape(batch_size, self.tactile_history_steps, self.d_model)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat(
            (cls, demo_tokens, state_tokens, tactile_tokens), dim=1
        )
        type_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=tokens.device),
                torch.ones(
                    self.demo_windows,
                    dtype=torch.long,
                    device=tokens.device,
                ),
                torch.full(
                    (self.policy_history_steps,),
                    2,
                    dtype=torch.long,
                    device=tokens.device,
                ),
                torch.full(
                    (self.tactile_history_steps,),
                    3,
                    dtype=torch.long,
                    device=tokens.device,
                ),
            )
        )
        tokens = (
            tokens
            + self.position_embedding
            + self.token_type_embedding(type_ids).unsqueeze(0)
        )
        padding_mask = torch.zeros(
            (batch_size, self.total_tokens),
            dtype=torch.bool,
            device=tokens.device,
        )
        state_begin = 1 + self.demo_windows
        padding_mask[
            :, state_begin : state_begin + self.policy_history_steps
        ] = ~prefix_valid_mask
        encoded = self.transformer(
            tokens, src_key_padding_mask=padding_mask
        )
        representation = self.readout(encoded[:, 0])

        raw = torch.stack(
            [
                self.component_heads[name](representation)
                for name in COMPONENT_NAMES
            ],
            dim=1,
        )
        return {
            "mean_log1p_scaled": raw[..., 0],
            "log_variance_log1p_scaled": torch.clamp(
                raw[..., 1], min=-10.0, max=8.0
            ),
            "representation": representation,
        }

    def encode_targets(self, component_mse: torch.Tensor) -> torch.Tensor:
        if component_mse.shape[-1] != len(COMPONENT_NAMES):
            raise ValueError("component target width mismatch")
        if torch.any(component_mse < 0):
            raise ValueError("component MSE targets must be nonnegative")
        return torch.log1p(component_mse / self.target_scale)

    def decode_mean(
        self, mean_log1p_scaled: torch.Tensor
    ) -> torch.Tensor:
        return torch.expm1(torch.clamp(mean_log1p_scaled, min=0.0)) * (
            self.target_scale
        )

    @staticmethod
    def gaussian_nll(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            torch.exp(-log_variance) * torch.square(target - mean)
            + log_variance
        )


class DemoConditionedCausalEventPredictorV2(nn.Module):
    """Serious causal Transformer for trajectory and contact-event mismatch.

    This is the event extension of the existing project predictor.  It uses a
    past-only official Tracker observation prefix and the fixed numeric
    selected-demo condition.  No actual future event, task ID or motion ID is
    accepted by ``forward``.
    """

    def __init__(
        self,
        *,
        policy_dim: int,
        policy_history_steps: int,
        demo_windows: int,
        demo_window_steps: int,
        demo_feature_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        demo_mean: torch.Tensor,
        demo_std: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if tuple(state_mean.shape) != (policy_dim,):
            raise ValueError("state_mean shape mismatch")
        if tuple(state_std.shape) != (policy_dim,):
            raise ValueError("state_std shape mismatch")
        if tuple(demo_mean.shape) != (demo_feature_dim,):
            raise ValueError("demo_mean shape mismatch")
        if tuple(demo_std.shape) != (demo_feature_dim,):
            raise ValueError("demo_std shape mismatch")
        if tuple(target_scale.shape) != (len(EVENT_TARGET_NAMES),):
            raise ValueError("target_scale shape mismatch")
        self.policy_dim = int(policy_dim)
        self.policy_history_steps = int(policy_history_steps)
        self.demo_windows = int(demo_windows)
        self.demo_window_steps = int(demo_window_steps)
        self.demo_feature_dim = int(demo_feature_dim)
        self.d_model = int(d_model)
        self.total_tokens = 1 + self.demo_windows + self.policy_history_steps

        self.register_buffer("state_mean", state_mean.float().clone())
        self.register_buffer("state_std", torch.clamp(state_std.float().clone(), min=1.0e-6))
        self.register_buffer("demo_mean", demo_mean.float().clone())
        self.register_buffer("demo_std", torch.clamp(demo_std.float().clone(), min=1.0e-6))
        self.register_buffer("target_scale", torch.clamp(target_scale.float().clone(), min=1.0e-8))

        self.demo_projection = nn.Sequential(
            nn.Linear(self.demo_window_steps * self.demo_feature_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(self.policy_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.total_tokens, self.d_model)
        )
        self.token_type_embedding = nn.Embedding(3, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.target_heads = nn.ModuleDict(
            {name: nn.Linear(self.d_model, 2) for name in EVENT_TARGET_NAMES}
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        policy_prefix: torch.Tensor,
        selected_demo_condition: torch.Tensor,
        zero_demo: bool = False,
    ) -> Mapping[str, torch.Tensor]:
        batch = policy_prefix.shape[0]
        if tuple(policy_prefix.shape) != (
            batch,
            self.policy_history_steps,
            self.policy_dim,
        ):
            raise ValueError("policy_prefix shape mismatch")
        if tuple(selected_demo_condition.shape) != (
            batch,
            self.demo_windows,
            self.demo_window_steps,
            self.demo_feature_dim,
        ):
            raise ValueError("selected_demo_condition shape mismatch")
        state = (policy_prefix - self.state_mean) / self.state_std
        demo = (selected_demo_condition - self.demo_mean) / self.demo_std
        if zero_demo:
            demo = torch.zeros_like(demo)
        state_tokens = self.state_projection(state)
        demo_tokens = self.demo_projection(
            demo.reshape(
                batch,
                self.demo_windows,
                self.demo_window_steps * self.demo_feature_dim,
            )
        )
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, demo_tokens, state_tokens), dim=1)
        type_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=tokens.device),
                torch.ones(self.demo_windows, dtype=torch.long, device=tokens.device),
                torch.full(
                    (self.policy_history_steps,),
                    2,
                    dtype=torch.long,
                    device=tokens.device,
                ),
            )
        )
        encoded = self.transformer(
            tokens
            + self.position_embedding
            + self.token_type_embedding(type_ids).unsqueeze(0)
        )
        representation = self.readout(encoded[:, 0])
        raw = torch.stack(
            [self.target_heads[name](representation) for name in EVENT_TARGET_NAMES],
            dim=1,
        )
        return {
            "mean_log1p_scaled": raw[..., 0],
            "log_variance_log1p_scaled": torch.clamp(raw[..., 1], min=-10.0, max=8.0),
            "representation": representation,
        }

    def encode_targets(self, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-1] != len(EVENT_TARGET_NAMES):
            raise ValueError("event target width mismatch")
        if torch.any(target < 0):
            raise ValueError("event mismatch targets must be nonnegative")
        return torch.log1p(target / self.target_scale)

    def decode_mean(self, mean: torch.Tensor) -> torch.Tensor:
        return torch.expm1(torch.clamp(mean, min=0.0)) * self.target_scale

    @staticmethod
    def gaussian_nll(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            torch.exp(-log_variance) * torch.square(target - mean)
            + log_variance
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
