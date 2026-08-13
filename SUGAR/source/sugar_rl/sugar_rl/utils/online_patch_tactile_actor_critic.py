# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Serious SUGAR actor with the Plan-15 anatomical patch Transformer."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.networks import EmpiricalNormalization, MLP

from sugar_rl.utils.patch_tactile_encoder import (
    AnatomicalPatchTactileEncoder,
)
from sugar_rl.utils.reference_only_tactile_actor_critic import (
    ReferenceOnlyTactileActorCritic,
)


class OnlinePatchTactileActorCritic(ReferenceOnlyTactileActorCritic):
    """504-D deployable base + 128-D patch embedding + SUGAR 512/256/128 MLP."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        patch_channel_scales: Sequence[float],
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        tactile_obs_group: str = "online_patch_tactile_history",
        warm_start_tactile_gain: float = 0.01,
        **kwargs,
    ) -> None:
        if tuple(actor_hidden_dims) != (512, 256, 128):
            raise ValueError("Plan-15 actor must retain SUGAR 512/256/128")
        if num_actions != 29:
            raise ValueError("Plan-15 actor must retain the 29-D SUGAR action")
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
            warm_start_tactile_gain=warm_start_tactile_gain,
            **kwargs,
        )
        if self.num_actor_base_obs != 504:
            raise ValueError(
                f"Plan-15 deployed base observation must be 504-D, got {self.num_actor_base_obs}"
            )
        if self.num_critic_base_obs != 890:
            raise ValueError(
                f"Plan-15 privileged critic must be 890-D, got {self.num_critic_base_obs}"
            )
        if tuple(obs[tactile_obs_group].shape[1:]) != (1944,):
            raise ValueError(
                "Plan-15 patch observation must be flat [B,1944], got "
                f"{tuple(obs[tactile_obs_group].shape)}"
            )

        self.actor_tactile_encoder = AnatomicalPatchTactileEncoder(
            patch_channel_scales
        )
        self.actor = MLP(
            self.num_actor_base_obs + self.actor_tactile_encoder.output_dim,
            num_actions,
            list(actor_hidden_dims),
            activation,
        )
        actor_input_dim = (
            self.num_actor_base_obs + self.actor_tactile_encoder.output_dim
        )
        self.actor_obs_normalizer = (
            EmpiricalNormalization(actor_input_dim)
            if actor_obs_normalization
            else nn.Identity()
        )
        first_layer = self.actor[0]
        with torch.no_grad():
            first_layer.weight[:, self.num_actor_base_obs :].mul_(
                self.warm_start_tactile_gain
            )

    def configure_tactile_actor_finetune(self) -> dict[str, object]:
        """Train the full serious student while retaining the privileged critic."""

        for parameter in self.actor.parameters():
            parameter.requires_grad_(True)
        for parameter in self.actor_tactile_encoder.parameters():
            parameter.requires_grad_(True)
        if self.noise_std_type == "scalar":
            self.std.requires_grad_(True)
        else:
            self.log_std.requires_grad_(True)
        return {
            "mode": "full_sugar_student_with_anatomical_patch_transformer",
            "actor_base_width": self.num_actor_base_obs,
            "actor_patch_embedding_width": self.actor_tactile_encoder.output_dim,
            "actor_input_width": self.actor[0].in_features,
            "actor_hidden_dims": [512, 256, 128],
            "action_width": 29,
            "critic_observation": "training_only_privileged_890d",
            "patch_encoder": self.actor_tactile_encoder.architecture_contract(),
        }

    def load_sugar_warm_start(
        self, source_state: dict[str, torch.Tensor]
    ) -> dict[str, object]:
        """Map the released 510-D Tracker into the proxy-free 504-D base."""

        source_first = source_state.get("actor.0.weight")
        if source_first is None or tuple(source_first.shape) != (512, 510):
            raise ValueError("warm start requires the released 510-D Tracker actor")
        target_first = self.actor[0]
        if tuple(target_first.weight.shape) != (512, 632):
            raise ValueError(
                f"unexpected Plan-15 actor input {tuple(target_first.weight.shape)}"
            )
        device = target_first.weight.device
        dtype = target_first.weight.dtype

        def source(name: str) -> torch.Tensor:
            if name not in source_state:
                raise KeyError(f"released Tracker checkpoint is missing {name}")
            return source_state[name].to(device=device, dtype=dtype)

        with torch.no_grad():
            # Retain the independently initialized low-gain patch columns.
            target_first.weight[:, : self.num_actor_base_obs].zero_()
            target_first.bias.copy_(source("actor.0.bias"))
            target_first.weight[:, 0:35].copy_(source_first[:, 0:35])
            target_first.weight[:, 35:500].copy_(source_first[:, 36:501])
            # target 500:504 (base lin vel + phase) intentionally start with
            # zero authority. Source 35 and 501:510 are deliberately excluded.
            for name, parameter in self.actor.named_parameters():
                if name in ("0.weight", "0.bias"):
                    continue
                parameter.copy_(source(f"actor.{name}"))
            for name, parameter in self.critic.named_parameters():
                parameter.copy_(source(f"critic.{name}"))
            if self.noise_std_type == "scalar":
                self.std.copy_(source("std"))
            else:
                self.log_std.copy_(source("log_std"))

        equivalence = self._audit_tracker_zero_patch_equivalence(source_state)
        return {
            "source_policy": "released_official_carrybox_tracker_510d",
            "target_actor_base_width": 504,
            "target_patch_embedding_width": 128,
            "target_actor_input_width": 632,
            "direct_tracker_command_columns": 35,
            "direct_robot_action_gravity_history_columns": 465,
            "new_zero_authority_columns": [500, 504],
            "excluded_source_columns": {
                "contact_label": [35, 36],
                "measured_object_position_orientation": [501, 510],
            },
            "actor_receives_excluded_source_values": False,
            **equivalence,
        }

    def _audit_tracker_zero_patch_equivalence(
        self, source_state: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        def source_mlp(
            prefix: str, value: torch.Tensor, target: nn.Module
        ) -> torch.Tensor:
            for child_name, module in target._modules.items():
                if isinstance(module, nn.Linear):
                    value = F.linear(
                        value,
                        source_state[f"{prefix}.{child_name}.weight"].to(
                            device=device, dtype=dtype
                        ),
                        source_state[f"{prefix}.{child_name}.bias"].to(
                            device=device, dtype=dtype
                        ),
                    )
                else:
                    value = module(value)
            return value

        with torch.no_grad():
            base = torch.linspace(
                -1.0, 1.0, steps=2 * 504, device=device, dtype=dtype
            ).reshape(2, 504)
            source_actor_obs = torch.zeros(2, 510, device=device, dtype=dtype)
            source_actor_obs[:, 0:35] = base[:, 0:35]
            source_actor_obs[:, 36:501] = base[:, 35:500]
            patch_zero = torch.zeros(2, 1944, device=device, dtype=dtype)
            embedding = self.actor_tactile_encoder(patch_zero)
            target_actor = self.actor(torch.cat((base, embedding), dim=-1))
            source_actor = source_mlp("actor", source_actor_obs, self.actor)

            critic_obs = torch.linspace(
                1.0, -1.0, steps=2 * 890, device=device, dtype=dtype
            ).reshape(2, 890)
            target_critic = self.critic(critic_obs)
            source_critic = source_mlp("critic", critic_obs, self.critic)
            actor_error = float((target_actor - source_actor).abs().max().item())
            critic_error = float((target_critic - source_critic).abs().max().item())
            zero_abs_max = float(embedding.abs().max().item())

        # The 510-wide source and 632-wide zero-patch target select different
        # GEMM kernels on H200.  Their measured float32 rounding remains below
        # the project's existing canonical action gate of 2e-6.
        tolerance = 2.0e-6
        if zero_abs_max != 0.0 or actor_error > tolerance or critic_error > tolerance:
            raise RuntimeError(
                "Plan-15 zero-patch warm-start equivalence failed: "
                f"embedding={zero_abs_max}, actor={actor_error}, critic={critic_error}"
            )
        return {
            "zero_patch_embedding_abs_max": zero_abs_max,
            "actor_zero_patch_max_abs_error": actor_error,
            "critic_max_abs_error": critic_error,
            "equivalence_tolerance": tolerance,
        }
