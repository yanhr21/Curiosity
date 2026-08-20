# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Asymmetric SUGAR tactile actor-critic for the reference-only actor branch.

The actor retains the official-width SUGAR input and the existing serious
spatial TacSL encoder.  The critic retains the original official SUGAR input
width and receives no role-dependent tactile feature, so full/zero/pressure
differ only at the actor tactile boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.networks import EmpiricalNormalization, MLP

from sugar_rl.utils.tactile_actor_critic import TactileActorCritic


class ReferenceOnlyTactileActorCritic(TactileActorCritic):
    """Spatial tactile actor with an exact-state, non-tactile SUGAR critic."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        tactile_obs_group: str = "tactile",
        tactile_grid_shape: Sequence[int] = (20, 25),
        tactile_num_hands: int = 2,
        tactile_channels_per_hand: int = 3,
        tactile_encoder_channels: Sequence[int] = (32, 64, 64),
        tactile_embedding_dim: int = 128,
        warm_start_tactile_gain: float = 0.01,
        tactile_preactivation_cap: float | None = None,
        tactile_action_residual_cap: float | None = None,
        **kwargs,
    ):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            tactile_obs_group=tactile_obs_group,
            tactile_grid_shape=tactile_grid_shape,
            tactile_num_hands=tactile_num_hands,
            tactile_channels_per_hand=tactile_channels_per_hand,
            tactile_encoder_channels=tactile_encoder_channels,
            tactile_embedding_dim=tactile_embedding_dim,
            warm_start_tactile_gain=warm_start_tactile_gain,
            **kwargs,
        )
        self.tactile_preactivation_cap = (
            None
            if tactile_preactivation_cap is None
            else float(tactile_preactivation_cap)
        )
        if self.tactile_preactivation_cap is not None and not (
            self.tactile_preactivation_cap > 0.0
        ):
            raise ValueError("tactile_preactivation_cap must be positive")
        self.tactile_action_residual_cap = (
            None
            if tactile_action_residual_cap is None
            else float(tactile_action_residual_cap)
        )
        if self.tactile_action_residual_cap is not None and not (
            self.tactile_action_residual_cap > 0.0
        ):
            raise ValueError("tactile_action_residual_cap must be positive")

        # The inherited constructor builds a tactile critic for the active
        # exact-state branch. Replace it with the original-width official SUGAR
        # critic. The warm-start loader then copies every critic tensor exactly.
        self.critic_tactile_encoder = nn.Identity()
        self.critic = MLP(
            self.num_critic_base_obs,
            1,
            list(critic_hidden_dims),
            activation,
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(self.num_critic_base_obs)
            if critic_obs_normalization
            else nn.Identity()
        )
        self.critic_uses_tactile = False
        print(f"SUGAR reference-only privileged critic MLP: {self.critic}")

    def get_critic_obs(self, obs) -> torch.Tensor:
        """Return only the official privileged SUGAR critic group."""
        return self._concat_groups(obs, self.critic_base_groups)

    def _tactile_enhanced_actor_forward(
        self, actor_obs: torch.Tensor
    ) -> torch.Tensor:
        """Run the actor with the optional hidden preactivation bound."""
        if self.tactile_preactivation_cap is None:
            return self.actor(actor_obs)
        if actor_obs.ndim != 2 or actor_obs.shape[-1] <= self.num_actor_base_obs:
            raise RuntimeError(f"unexpected actor observation {tuple(actor_obs.shape)}")
        first_layer = self.actor._modules.get("0")
        if not isinstance(first_layer, nn.Linear):
            raise RuntimeError("Expected the accepted SUGAR actor input layer at actor.0")
        base_obs = actor_obs[:, : self.num_actor_base_obs]
        tactile_features = actor_obs[:, self.num_actor_base_obs :]
        base_preactivation = F.linear(
            base_obs,
            first_layer.weight[:, : self.num_actor_base_obs],
            first_layer.bias,
        )
        raw_tactile_correction = F.linear(
            tactile_features,
            first_layer.weight[:, self.num_actor_base_obs :],
            None,
        )
        cap = self.tactile_preactivation_cap
        tactile_correction = cap * torch.tanh(raw_tactile_correction / cap)
        value = base_preactivation + tactile_correction
        for name, module in self.actor._modules.items():
            if name == "0":
                continue
            value = module(value)
        return value

    def _actor_forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Run the official base plus an optionally bounded tactile residual."""

        tactile_action = self._tactile_enhanced_actor_forward(actor_obs)
        if self.tactile_action_residual_cap is None:
            return tactile_action
        base_obs = actor_obs[:, : self.num_actor_base_obs]
        tactile_features = actor_obs[:, self.num_actor_base_obs :]
        zero_features = torch.zeros_like(tactile_features)
        base_action = self.actor(torch.cat((base_obs, zero_features), dim=-1))
        cap = self.tactile_action_residual_cap
        raw_residual = tactile_action - base_action
        bounded_residual = cap * torch.tanh(raw_residual / cap)
        return base_action + bounded_residual

    def update_distribution(self, obs):
        mean = self._actor_forward(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self._actor_forward(actor_obs)

    def configure_tactile_actor_finetune(self) -> dict[str, object]:
        report = super().configure_tactile_actor_finetune()
        report.update(
            {
                "critic_observation": "official_exact_state_without_tactile",
                "critic_tactile_role_dependence": False,
                "tactile_fusion": (
                    "unbounded_late_concatenation"
                    if self.tactile_preactivation_cap is None
                    else "bounded_first_layer_preactivation_correction"
                ),
                "tactile_preactivation_cap": self.tactile_preactivation_cap,
                "tactile_action_residual": (
                    "unbounded"
                    if self.tactile_action_residual_cap is None
                    else "bounded_relative_to_exact_official_zero_tactile_action"
                ),
                "tactile_action_residual_cap": self.tactile_action_residual_cap,
            }
        )
        return report

    def _audit_zero_tactile_warm_start(
        self,
        source_state: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Prove exact official actor/critic recovery at zero actor tactile."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        def source_mlp_forward(name: str, probe: torch.Tensor, target_mlp: nn.Module) -> torch.Tensor:
            value = probe
            for child_name, module in target_mlp._modules.items():
                if isinstance(module, nn.Linear):
                    weight_key = f"{name}.{child_name}.weight"
                    bias_key = f"{name}.{child_name}.bias"
                    if weight_key not in source_state or bias_key not in source_state:
                        raise RuntimeError(
                            f"Official SUGAR warm-start is missing {weight_key} or {bias_key}"
                        )
                    value = F.linear(
                        value,
                        source_state[weight_key].to(device=device, dtype=dtype),
                        source_state[bias_key].to(device=device, dtype=dtype),
                    )
                else:
                    value = module(value)
            return value

        with torch.no_grad():
            tactile_zeros = torch.zeros(
                2,
                self.actor_tactile_encoder.expected_flat_dim,
                device=device,
                dtype=dtype,
            )
            actor_tactile_features = self.actor_tactile_encoder(tactile_zeros)
            tactile_zero_abs_max = float(actor_tactile_features.abs().max().item())
            actor_probe = torch.linspace(
                -1.0,
                1.0,
                steps=2 * self.num_actor_base_obs,
                device=device,
                dtype=dtype,
            ).reshape(2, self.num_actor_base_obs)
            critic_probe = torch.linspace(
                1.0,
                -1.0,
                steps=2 * self.num_critic_base_obs,
                device=device,
                dtype=dtype,
            ).reshape(2, self.num_critic_base_obs)
            source_actor = source_mlp_forward("actor", actor_probe, self.actor)
            source_critic = source_mlp_forward("critic", critic_probe, self.critic)
            target_actor = self._actor_forward(
                torch.cat((actor_probe, actor_tactile_features), dim=-1)
            )
            target_critic = self.critic(critic_probe)
            actor_error = float((target_actor - source_actor).abs().max().item())
            critic_error = float((target_critic - source_critic).abs().max().item())

        tolerance = 1.0e-6
        if tactile_zero_abs_max != 0.0 or actor_error > tolerance or critic_error > tolerance:
            raise RuntimeError(
                "Reference-only zero-tactile warm-start equivalence failed: "
                f"feature_abs_max={tactile_zero_abs_max}, actor_error={actor_error}, "
                f"critic_error={critic_error}, tolerance={tolerance}"
            )
        return {
            "zero_tactile_feature_abs_max": tactile_zero_abs_max,
            "actor_zero_tactile_max_abs_error": actor_error,
            "critic_zero_tactile_max_abs_error": critic_error,
            "zero_tactile_equivalence_tolerance": tolerance,
            "critic_zero_tactile_semantics": "official_non_tactile_privileged_critic",
        }
