# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous-action adapter of the original Intrinsic Curiosity Module.

The algorithmic reference is Pathak et al. (2017) and the official
``pathak22/noreward-rl`` ``StateActionPredictor``.  This adapter preserves the
shared encoder, inverse/forward objectives, feature-summed prediction error,
and detached intrinsic reward.  It changes only the observation encoder and
the discrete inverse-action likelihood required by SUGAR's continuous 29-D
applied joint targets.

No task result, reward, success, lift height, slip label, clamp similarity,
hidden physics value, simulator oracle, or future reference is accepted by
this module.  Those signals must remain outside ICM.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sugar_rl.utils.tactile_actor_critic import SpatialTactileEncoder


ICM_VECTOR_FIELD_DIMS: tuple[tuple[str, int], ...] = (
    ("projected_gravity", 3),
    ("base_height", 1),
    ("base_linear_velocity_body", 3),
    ("base_angular_velocity_body", 3),
    ("joint_position_relative", 29),
    ("joint_velocity", 29),
    ("previous_applied_action_policy_units", 29),
    ("box_position_body", 3),
    ("box_orientation_tangent_normal_body", 6),
    ("box_linear_velocity_body", 3),
    ("box_angular_velocity_body", 3),
    ("goal_position_body", 3),
)


def assemble_icm_vector_observation(
    fields: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Assemble only the frozen non-privileged ICM vector fields."""

    expected_names = tuple(name for name, _ in ICM_VECTOR_FIELD_DIMS)
    if set(fields) != set(expected_names):
        missing = sorted(set(expected_names) - set(fields))
        unexpected = sorted(set(fields) - set(expected_names))
        raise ValueError(
            f"ICM vector fields mismatch: missing={missing}, unexpected={unexpected}"
        )
    batch_size: int | None = None
    ordered = []
    for name, dim in ICM_VECTOR_FIELD_DIMS:
        value = fields[name]
        if value.ndim != 2 or value.shape[-1] != dim:
            raise ValueError(f"ICM field {name} must be (batch,{dim}), got {tuple(value.shape)}")
        if batch_size is None:
            batch_size = value.shape[0]
        elif value.shape[0] != batch_size:
            raise ValueError(f"ICM field {name} has a different batch size")
        if not torch.isfinite(value).all():
            raise ValueError(f"ICM field {name} contains non-finite values")
        ordered.append(value)
    return torch.cat(ordered, dim=-1)


@dataclass(frozen=True)
class OriginalICMContinuousCfg:
    """Frozen architecture and loss contract for SUGAR's ICM adapter."""

    action_dim: int = 29
    tactile_history_steps: int = 4
    tactile_num_hands: int = 2
    tactile_channels_per_frame: int = 3
    tactile_grid_shape: tuple[int, int] = (20, 25)
    tactile_encoder_channels: tuple[int, ...] = (32, 64, 64)
    tactile_embedding_dim_per_hand: int = 128
    vector_embedding_dim: int = 256
    feature_dim: int = 288
    predictor_hidden_dim: int = 256
    forward_loss_weight: float = 0.2
    prediction_loss_scale: float = 10.0
    intrinsic_reward_scale_eta: float = 0.01
    inverse_log_std_min: float = -5.0
    inverse_log_std_max: float = 2.0
    normalizer_clip: float = 10.0
    normalizer_epsilon: float = 1.0e-6
    action_normalizer_clip: float = 10.0
    applied_action_safety_abs_limit: float = 100.0

    @property
    def vector_obs_dim(self) -> int:
        return sum(dim for _, dim in ICM_VECTOR_FIELD_DIMS)

    def __post_init__(self) -> None:
        if self.action_dim != 29:
            raise ValueError("the SUGAR ICM contract is frozen to 29 actions")
        if self.tactile_history_steps < 2:
            raise ValueError("ICM requires at least two tactile-history frames")
        if self.tactile_num_hands != 2 or self.tactile_channels_per_frame != 3:
            raise ValueError("ICM requires two direct pressure+2D-shear R15 streams")
        if self.tactile_grid_shape != (20, 25):
            raise ValueError("ICM requires the validated official 20x25 R15 grid")
        if self.feature_dim != 288 or self.predictor_hidden_dim != 256:
            raise ValueError("feature/hidden dimensions preserve original ICM's 288/256 contract")
        if not 0.0 <= self.forward_loss_weight <= 1.0:
            raise ValueError("forward_loss_weight must lie in [0, 1]")
        if self.intrinsic_reward_scale_eta < 0.0:
            raise ValueError("intrinsic_reward_scale_eta must be non-negative")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["vector_obs_dim"] = self.vector_obs_dim
        payload["vector_field_dims"] = [list(field) for field in ICM_VECTOR_FIELD_DIMS]
        return payload


class StructuredICMObservationNormalizer(nn.Module):
    """Checkpointed running normalization without destroying taxel geometry."""

    def __init__(self, cfg: OriginalICMContinuousCfg):
        super().__init__()
        self.vector_dim = cfg.vector_obs_dim
        self.clip = float(cfg.normalizer_clip)
        self.epsilon = float(cfg.normalizer_epsilon)
        self.register_buffer("vector_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("vector_mean", torch.zeros(self.vector_dim, dtype=torch.float64))
        self.register_buffer("vector_m2", torch.zeros(self.vector_dim, dtype=torch.float64))
        # Pressure and the two signed shear components have separate moments.
        self.register_buffer("tactile_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("tactile_mean", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("tactile_m2", torch.zeros(3, dtype=torch.float64))

    @staticmethod
    @torch.no_grad()
    def _merge_moments(
        count: torch.Tensor,
        mean: torch.Tensor,
        m2: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        values64 = values.detach().to(dtype=torch.float64)
        batch_count = torch.as_tensor(values64.shape[0], dtype=torch.float64, device=values.device)
        batch_mean = values64.mean(dim=0)
        batch_m2 = torch.square(values64 - batch_mean).sum(dim=0)
        if count.item() == 0:
            count.copy_(batch_count)
            mean.copy_(batch_mean)
            m2.copy_(batch_m2)
            return
        delta = batch_mean - mean
        total = count + batch_count
        mean.add_(delta * (batch_count / total))
        m2.add_(batch_m2 + torch.square(delta) * count * batch_count / total)
        count.copy_(total)

    @torch.no_grad()
    def update(self, vector_obs: torch.Tensor, tactile_history: torch.Tensor) -> None:
        self._validate(vector_obs, tactile_history)
        self._merge_moments(
            self.vector_count, self.vector_mean, self.vector_m2, vector_obs
        )
        # Keep the pressure/shear channel axis and combine batch, history, hand,
        # and spatial positions. This retains the 20x25 layout in forward().
        tactile_channels = tactile_history.permute(0, 1, 2, 4, 5, 3).reshape(-1, 3)
        self._merge_moments(
            self.tactile_count,
            self.tactile_mean,
            self.tactile_m2,
            tactile_channels,
        )

    def _stats(
        self, count: torch.Tensor, mean: torch.Tensor, m2: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if count.item() <= 0:
            raise RuntimeError("ICM observation normalizer must be updated before use")
        denominator = torch.clamp(count, min=1.0)
        variance = m2 / denominator
        std = torch.sqrt(torch.clamp(variance, min=self.epsilon**2))
        return mean.to(dtype=dtype), std.to(dtype=dtype)

    def _validate(self, vector_obs: torch.Tensor, tactile_history: torch.Tensor) -> None:
        if vector_obs.ndim != 2 or vector_obs.shape[-1] != self.vector_dim:
            raise ValueError(
                f"ICM vector observation must be (batch,{self.vector_dim}), got {tuple(vector_obs.shape)}"
            )
        if tactile_history.ndim != 6 or tactile_history.shape[2:4] != (2, 3):
            raise ValueError(
                "ICM tactile history must be (batch,history,2,3,20,25), got "
                f"{tuple(tactile_history.shape)}"
            )
        if tactile_history.shape[-2:] != (20, 25):
            raise ValueError("ICM tactile history is not the official 20x25 R15 grid")
        if tactile_history.shape[0] != vector_obs.shape[0]:
            raise ValueError("ICM vector/tactile batch mismatch")
        if not torch.isfinite(vector_obs).all() or not torch.isfinite(tactile_history).all():
            raise ValueError("ICM observations contain non-finite values")

    def forward(
        self, vector_obs: torch.Tensor, tactile_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(vector_obs, tactile_history)
        vector_mean, vector_std = self._stats(
            self.vector_count, self.vector_mean, self.vector_m2, vector_obs.dtype
        )
        tactile_mean, tactile_std = self._stats(
            self.tactile_count, self.tactile_mean, self.tactile_m2, tactile_history.dtype
        )
        normalized_vector = torch.clamp(
            (vector_obs - vector_mean) / vector_std, -self.clip, self.clip
        )
        channel_shape = (1, 1, 1, 3, 1, 1)
        normalized_tactile = torch.clamp(
            (tactile_history - tactile_mean.view(channel_shape))
            / tactile_std.view(channel_shape),
            -self.clip,
            self.clip,
        )
        return normalized_vector, normalized_tactile


class ICMActionNormalizer(nn.Module):
    """Checkpointed per-joint normalization of official SUGAR action units.

    Official SUGAR does not clip its policy output to ``[-1, 1]`` before the
    per-joint scale/offset transform.  Standardizing those exact policy-unit
    actions is the continuous-action analogue of feeding the original ICM a
    fixed-scale one-hot action; it does not alter what the robot executes.
    """

    def __init__(self, cfg: OriginalICMContinuousCfg):
        super().__init__()
        self.action_dim = cfg.action_dim
        self.clip = float(cfg.action_normalizer_clip)
        self.epsilon = float(cfg.normalizer_epsilon)
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(self.action_dim, dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(self.action_dim, dtype=torch.float64))

    @torch.no_grad()
    def update(self, action: torch.Tensor) -> None:
        self._validate(action)
        StructuredICMObservationNormalizer._merge_moments(
            self.count, self.mean, self.m2, action
        )

    def _validate(self, action: torch.Tensor) -> None:
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"ICM action must be (batch,{self.action_dim}), got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise ValueError("ICM applied action contains non-finite values")

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        self._validate(action)
        if self.count.item() <= 0:
            raise RuntimeError("ICM action normalizer must be updated before use")
        variance = self.m2 / torch.clamp(self.count, min=1.0)
        std = torch.sqrt(torch.clamp(variance, min=self.epsilon**2))
        return torch.clamp(
            (action - self.mean.to(action.dtype)) / std.to(action.dtype),
            -self.clip,
            self.clip,
        )


def _normalized_columns_(linear: nn.Linear, std: float = 0.01) -> None:
    """PyTorch equivalent of official ICM normalized_columns_initializer."""

    with torch.no_grad():
        weight = torch.randn_like(linear.weight)
        weight.mul_(std / torch.sqrt(torch.square(weight).sum(dim=1, keepdim=True)))
        linear.weight.copy_(weight)
        if linear.bias is not None:
            linear.bias.zero_()


class StructuredICMEncoder(nn.Module):
    """Shared phi encoder for declared proprio/task vector and direct R15 history."""

    def __init__(self, cfg: OriginalICMContinuousCfg):
        super().__init__()
        self.cfg = cfg
        self.vector_encoder = nn.Sequential(
            nn.Linear(cfg.vector_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, cfg.vector_embedding_dim),
            nn.LayerNorm(cfg.vector_embedding_dim),
            nn.ELU(),
        )
        self.tactile_encoder = SpatialTactileEncoder(
            channels_per_hand=(
                cfg.tactile_history_steps * cfg.tactile_channels_per_frame
            ),
            num_hands=cfg.tactile_num_hands,
            grid_shape=cfg.tactile_grid_shape,
            encoder_channels=cfg.tactile_encoder_channels,
            embedding_dim=cfg.tactile_embedding_dim_per_hand,
            activation="elu",
        )
        fusion_input_dim = (
            cfg.vector_embedding_dim
            + cfg.tactile_num_hands * cfg.tactile_embedding_dim_per_hand
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.feature_dim),
            nn.ELU(),
        )

    def forward(
        self, vector_obs: torch.Tensor, tactile_history: torch.Tensor
    ) -> torch.Tensor:
        if tactile_history.shape[1] != self.cfg.tactile_history_steps:
            raise ValueError(
                f"ICM expected {self.cfg.tactile_history_steps} tactile frames, "
                f"got {tactile_history.shape[1]}"
            )
        batch_size = tactile_history.shape[0]
        # The existing validated spatial encoder accepts per-hand channels.
        tactile = tactile_history.permute(0, 2, 1, 3, 4, 5).reshape(
            batch_size,
            self.cfg.tactile_num_hands
            * self.cfg.tactile_history_steps
            * self.cfg.tactile_channels_per_frame
            * self.cfg.tactile_grid_shape[0]
            * self.cfg.tactile_grid_shape[1],
        )
        vector_feature = self.vector_encoder(vector_obs)
        tactile_feature = self.tactile_encoder(tactile)
        return self.fusion(torch.cat((vector_feature, tactile_feature), dim=-1))


class OriginalICMContinuous(nn.Module):
    """Original ICM equations with a continuous 29-D inverse likelihood."""

    def __init__(self, cfg: OriginalICMContinuousCfg):
        super().__init__()
        self.cfg = cfg
        self.normalizer = StructuredICMObservationNormalizer(cfg)
        self.action_normalizer = ICMActionNormalizer(cfg)
        self.encoder = StructuredICMEncoder(cfg)

        # Original g(phi_t, phi_t+1): one 256-unit ReLU layer then action head.
        self.inverse_hidden = nn.Linear(2 * cfg.feature_dim, cfg.predictor_hidden_dim)
        self.inverse_output = nn.Linear(cfg.predictor_hidden_dim, 2 * cfg.action_dim)
        # Original f(phi_t, a_t): one 256-unit ReLU layer then phi prediction.
        self.forward_hidden = nn.Linear(
            cfg.feature_dim + cfg.action_dim, cfg.predictor_hidden_dim
        )
        self.forward_output = nn.Linear(cfg.predictor_hidden_dim, cfg.feature_dim)
        for layer in (
            self.inverse_hidden,
            self.inverse_output,
            self.forward_hidden,
            self.forward_output,
        ):
            _normalized_columns_(layer, std=0.01)

    @torch.no_grad()
    def update_normalizer(
        self, vector_obs: torch.Tensor, tactile_history: torch.Tensor
    ) -> None:
        self.normalizer.update(vector_obs, tactile_history)

    @torch.no_grad()
    def update_action_normalizer(self, applied_action_policy_units: torch.Tensor) -> None:
        self.action_normalizer.update(applied_action_policy_units)

    def encode(
        self, vector_obs: torch.Tensor, tactile_history: torch.Tensor
    ) -> torch.Tensor:
        normalized_vector, normalized_tactile = self.normalizer(
            vector_obs, tactile_history
        )
        return self.encoder(normalized_vector, normalized_tactile)

    def transition(
        self,
        vector_obs_t: torch.Tensor,
        tactile_history_t: torch.Tensor,
        applied_action_policy_units_t: torch.Tensor,
        vector_obs_tp1: torch.Tensor,
        tactile_history_tp1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute predictor losses and ungated per-transition discovery bonus.

        ``applied_action_policy_units_t`` must be the exact policy-unit action
        whose official per-joint scale/offset transform produced the applied
        SUGAR joint target—not a future reference action. It is standardized
        only inside ICM and is never changed before robot execution.
        """

        if applied_action_policy_units_t.ndim != 2 or applied_action_policy_units_t.shape[-1] != self.cfg.action_dim:
            raise ValueError(
                f"ICM action must be (batch,{self.cfg.action_dim}), got "
                f"{tuple(applied_action_policy_units_t.shape)}"
            )
        if not torch.isfinite(applied_action_policy_units_t).all():
            raise ValueError("ICM applied action contains non-finite values")
        if torch.abs(applied_action_policy_units_t).max() > self.cfg.applied_action_safety_abs_limit:
            raise ValueError(
                "ICM action exceeds the declared official-policy safety audit limit"
            )
        if applied_action_policy_units_t.shape[0] != vector_obs_t.shape[0]:
            raise ValueError("ICM action/observation batch mismatch")

        # One shared phi is used for s_t and s_t+1 exactly as in official ICM.
        phi_t = self.encode(vector_obs_t, tactile_history_t)
        phi_tp1 = self.encode(vector_obs_tp1, tactile_history_tp1)
        fixed_action = self.action_normalizer(
            applied_action_policy_units_t.detach()
        )

        inverse_feature = F.relu(
            self.inverse_hidden(torch.cat((phi_t, phi_tp1), dim=-1))
        )
        inverse_parameters = self.inverse_output(inverse_feature)
        inverse_mean, inverse_log_std = inverse_parameters.chunk(2, dim=-1)
        inverse_log_std = torch.clamp(
            inverse_log_std,
            min=self.cfg.inverse_log_std_min,
            max=self.cfg.inverse_log_std_max,
        )
        inverse_nll_per_dim = (
            0.5
            * torch.square(
                (fixed_action - inverse_mean) / torch.exp(inverse_log_std)
            )
            + inverse_log_std
            + 0.5 * math.log(2.0 * math.pi)
        )
        # Mean over action dimensions is the continuous analogue of the
        # original scalar categorical cross-entropy.
        inverse_loss_per_transition = inverse_nll_per_dim.mean(dim=-1)
        inverse_loss = inverse_loss_per_transition.mean()

        forward_feature = F.relu(
            self.forward_hidden(torch.cat((phi_t, fixed_action), dim=-1))
        )
        predicted_phi_tp1 = self.forward_output(forward_feature)
        # Official code computes 0.5*mean(square)*288. This is exactly
        # 0.5*sum(feature error^2), averaged only across the batch.
        forward_error_per_transition = 0.5 * torch.square(
            predicted_phi_tp1 - phi_tp1
        ).sum(dim=-1)
        forward_loss = forward_error_per_transition.mean()

        mixed_predictor_loss = (
            (1.0 - self.cfg.forward_loss_weight) * inverse_loss
            + self.cfg.forward_loss_weight * forward_loss
        )
        predictor_loss = self.cfg.prediction_loss_scale * mixed_predictor_loss
        intrinsic_reward = (
            self.cfg.intrinsic_reward_scale_eta
            * forward_error_per_transition.detach()
        )
        return {
            "predictor_loss": predictor_loss,
            "inverse_loss": inverse_loss,
            "forward_loss": forward_loss,
            "inverse_loss_per_transition": inverse_loss_per_transition.detach(),
            "forward_error_per_transition": forward_error_per_transition.detach(),
            "intrinsic_reward": intrinsic_reward,
            "inverse_action_mean": inverse_mean.detach(),
            "inverse_action_log_std": inverse_log_std.detach(),
            "normalized_applied_action": fixed_action.detach(),
            "phi_t": phi_t.detach(),
            "phi_tp1": phi_tp1.detach(),
            "predicted_phi_tp1": predicted_phi_tp1.detach(),
        }

    def predictor_parameters(self) -> Sequence[nn.Parameter]:
        """Return only ICM parameters for its independent optimizer."""

        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
