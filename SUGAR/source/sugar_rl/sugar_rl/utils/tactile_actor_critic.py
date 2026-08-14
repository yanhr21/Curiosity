# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL actor-critic that adds a spatial tactile branch to the SUGAR policy.

The original SUGAR actor and critic MLP dimensions are retained after a
per-hand convolutional encoder.  This lets a tactile policy warm-start from an
official SUGAR checkpoint without flattening the taxel maps before spatial
feature extraction.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class SpatialTactileEncoder(nn.Module):
    """Shared convolutional encoder for independent left/right taxel maps."""

    def __init__(
        self,
        channels_per_hand: int,
        num_hands: int,
        grid_shape: Sequence[int],
        encoder_channels: Sequence[int],
        embedding_dim: int,
        activation: str = "elu",
    ):
        super().__init__()
        if len(grid_shape) != 2:
            raise ValueError(f"Expected a 2-D tactile grid, got {tuple(grid_shape)}")
        if len(encoder_channels) < 2:
            raise ValueError("The tactile encoder requires at least two convolutional stages")

        self.channels_per_hand = int(channels_per_hand)
        self.num_hands = int(num_hands)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.embedding_dim = int(embedding_dim)
        self.expected_flat_dim = (
            self.channels_per_hand * self.num_hands * self.grid_shape[0] * self.grid_shape[1]
        )

        activation_classes = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if activation not in activation_classes:
            raise ValueError(
                f"Unsupported tactile activation '{activation}'; "
                f"expected one of {sorted(activation_classes)}"
            )
        activation_cls = activation_classes[activation]
        layers: list[nn.Module] = []
        in_channels = self.channels_per_hand
        for index, out_channels in enumerate(encoder_channels):
            stride = 1 if index == 0 else 2
            groups = min(8, int(out_channels))
            while int(out_channels) % groups != 0:
                groups -= 1
            layers.extend(
                [
                    nn.Conv2d(in_channels, int(out_channels), kernel_size=3, stride=stride, padding=1),
                    nn.GroupNorm(groups, int(out_channels)),
                    activation_cls(),
                ]
            )
            in_channels = int(out_channels)
        # For the fixed 20x25 TacSL grid, the convolution stack above always
        # produces 5x7. AdaptiveAvgPool2d((2, 2)) therefore uses exactly the
        # bins represented by kernel=(3, 4), stride=(2, 3). The explicit
        # AvgPool2d is mathematically equivalent for this locked geometry and,
        # unlike adaptive_avg_pool2d_backward_cuda, has a deterministic CUDA
        # backward implementation.
        layers.append(nn.AvgPool2d(kernel_size=(3, 4), stride=(2, 3)))
        self.convolution = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4, self.embedding_dim),
            activation_cls(),
        )

        # A zero taxel field must map to an exactly zero feature vector.  This
        # is the identity condition needed to warm-start from the accepted
        # SUGAR policy without changing its actions before tactile learning.
        # Keep all weights normally initialized so real spatial input has a
        # gradient immediately; only additive biases are removed.
        for module in self.modules():
            bias = getattr(module, "bias", None)
            if bias is not None:
                nn.init.zeros_(bias)

    @property
    def output_dim(self) -> int:
        return self.num_hands * self.embedding_dim

    def forward(self, tactile_obs: torch.Tensor) -> torch.Tensor:
        if tactile_obs.ndim != 2 or tactile_obs.shape[-1] != self.expected_flat_dim:
            raise ValueError(
                "Tactile observation shape mismatch: expected "
                f"(batch, {self.expected_flat_dim}), got {tuple(tactile_obs.shape)}"
            )
        batch_size = tactile_obs.shape[0]
        tactile_maps = tactile_obs.reshape(
            batch_size * self.num_hands,
            self.channels_per_hand,
            self.grid_shape[0],
            self.grid_shape[1],
        )
        embeddings = self.projection(self.convolution(tactile_maps))
        return embeddings.reshape(batch_size, self.output_dim)


class TactileActorCritic(nn.Module):
    """SUGAR actor-critic with separate spatial encoders for actor and critic."""

    is_recurrent = False

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
        **kwargs,
    ):
        if kwargs:
            print(
                "TactileActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(sorted(kwargs.keys()))
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.tactile_obs_group = tactile_obs_group
        self.warm_start_tactile_gain = float(warm_start_tactile_gain)

        if tactile_obs_group not in obs:
            raise KeyError(f"Missing tactile observation group '{tactile_obs_group}'. Available groups: {list(obs.keys())}")

        self.actor_base_groups = [name for name in obs_groups["policy"] if name != tactile_obs_group]
        self.critic_base_groups = [name for name in obs_groups["critic"] if name != tactile_obs_group]
        if not self.actor_base_groups or not self.critic_base_groups:
            raise ValueError("TactileActorCritic requires non-tactile SUGAR policy and critic observation groups")

        self.num_actor_base_obs = self._group_dim(obs, self.actor_base_groups)
        self.num_critic_base_obs = self._group_dim(obs, self.critic_base_groups)

        encoder_kwargs = {
            "channels_per_hand": tactile_channels_per_hand,
            "num_hands": tactile_num_hands,
            "grid_shape": tactile_grid_shape,
            "encoder_channels": tactile_encoder_channels,
            "embedding_dim": tactile_embedding_dim,
            "activation": activation,
        }
        self.actor_tactile_encoder = SpatialTactileEncoder(**encoder_kwargs)
        self.critic_tactile_encoder = SpatialTactileEncoder(**encoder_kwargs)
        tactile_feature_dim = self.actor_tactile_encoder.output_dim

        self.actor = MLP(
            self.num_actor_base_obs + tactile_feature_dim,
            num_actions,
            list(actor_hidden_dims),
            activation,
        )
        self.critic = MLP(
            self.num_critic_base_obs + tactile_feature_dim,
            1,
            list(critic_hidden_dims),
            activation,
        )

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        actor_input_dim = self.num_actor_base_obs + tactile_feature_dim
        critic_input_dim = self.num_critic_base_obs + tactile_feature_dim
        self.actor_obs_normalizer = (
            EmpiricalNormalization(actor_input_dim) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_input_dim) if critic_obs_normalization else nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type '{noise_std_type}'")

        self.distribution = None
        self._actor_base_gradient_hook = None
        Normal.set_default_validate_args(False)
        if getattr(self, "_print_inherited_spatial_tactile_architecture", True):
            print(f"SUGAR tactile actor MLP: {self.actor}")
            print(f"SUGAR tactile critic MLP: {self.critic}")
            print(f"Shared per-hand tactile encoder shape: {tuple(tactile_grid_shape)}, feature dim: {tactile_feature_dim}")

    @staticmethod
    def _group_dim(obs, groups: Sequence[str]) -> int:
        total = 0
        for group in groups:
            if obs[group].ndim != 2:
                raise ValueError(f"Observation group '{group}' must be flat, got shape {tuple(obs[group].shape)}")
            total += int(obs[group].shape[-1])
        return total

    @staticmethod
    def _concat_groups(obs, groups: Sequence[str]) -> torch.Tensor:
        return torch.cat([obs[group] for group in groups], dim=-1)

    def get_actor_obs(self, obs) -> torch.Tensor:
        base_obs = self._concat_groups(obs, self.actor_base_groups)
        tactile_features = self.actor_tactile_encoder(obs[self.tactile_obs_group])
        return torch.cat((base_obs, tactile_features), dim=-1)

    def get_critic_obs(self, obs) -> torch.Tensor:
        base_obs = self._concat_groups(obs, self.critic_base_groups)
        tactile_features = self.critic_tactile_encoder(obs[self.tactile_obs_group])
        return torch.cat((base_obs, tactile_features), dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(actor_obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def load_sugar_warm_start(self, source_state: dict[str, torch.Tensor]) -> dict[str, object]:
        """Load an official SUGAR checkpoint while adapting only the two input layers."""
        target_state = self.state_dict()
        copied: list[str] = []
        adapted: list[str] = []
        skipped: list[str] = []
        input_weights = {
            "actor.0.weight": self.num_actor_base_obs,
            "critic.0.weight": self.num_critic_base_obs,
        }

        for key, source_tensor in source_state.items():
            if key not in target_state:
                skipped.append(key)
                continue
            target_tensor = target_state[key]
            if target_tensor.shape == source_tensor.shape:
                target_tensor.copy_(source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))
                copied.append(key)
                continue
            if key in input_weights:
                expected_base_dim = input_weights[key]
                if source_tensor.ndim != 2 or source_tensor.shape[1] != expected_base_dim:
                    raise RuntimeError(
                        f"Official SUGAR {key} input width is {source_tensor.shape[1]}, "
                        f"but the preserved base observation width is {expected_base_dim}"
                    )
                if target_tensor.shape[0] != source_tensor.shape[0] or target_tensor.shape[1] <= expected_base_dim:
                    raise RuntimeError(
                        f"Cannot adapt {key}: source={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
                    )
                target_tensor[:, :expected_base_dim].copy_(
                    source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
                )
                target_tensor[:, expected_base_dim:].mul_(self.warm_start_tactile_gain)
                adapted.append(key)
                continue
            skipped.append(key)

        required = {"actor.0.weight", "critic.0.weight", "actor.6.weight", "critic.6.weight"}
        if self.noise_std_type == "scalar":
            required.add("std")
        else:
            required.add("log_std")
        missing_required = sorted(required.difference(copied).difference(adapted))
        if missing_required:
            raise RuntimeError(f"Warm-start checkpoint is missing required SUGAR parameters: {missing_required}")

        nn.Module.load_state_dict(self, target_state, strict=True)
        equivalence = self._audit_zero_tactile_warm_start(source_state)
        return {
            "copied_count": len(copied),
            "adapted": sorted(adapted),
            "skipped_source_keys": sorted(skipped),
            "actor_base_obs_dim": self.num_actor_base_obs,
            "critic_base_obs_dim": self.num_critic_base_obs,
            "tactile_feature_dim": self.actor_tactile_encoder.output_dim,
            "tactile_gain": self.warm_start_tactile_gain,
            **equivalence,
        }

    def configure_tactile_actor_finetune(self) -> dict[str, object]:
        """Freeze the accepted SUGAR actor and expose only its tactile adapter path.

        The first actor layer contains the original SUGAR observation columns
        followed by the tactile embedding columns.  A gradient mask keeps the
        former bitwise frozen while allowing the latter and the spatial encoder
        weights to learn.  Encoder biases remain frozen at zero, preserving the
        invariant that a zero taxel field produces the accepted SUGAR action.
        """

        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        if self.noise_std_type == "scalar":
            self.std.requires_grad_(False)
        else:
            self.log_std.requires_grad_(False)

        for parameter in self.actor_tactile_encoder.parameters():
            parameter.requires_grad_(True)
        frozen_encoder_biases: list[str] = []
        for module_name, module in self.actor_tactile_encoder.named_modules():
            bias = getattr(module, "bias", None)
            if bias is not None:
                bias.requires_grad_(False)
                frozen_encoder_biases.append(f"{module_name}.bias" if module_name else "bias")

        first_layer = self.actor._modules.get("0")
        if not isinstance(first_layer, nn.Linear):
            raise RuntimeError("Expected the accepted SUGAR actor input layer at actor.0")
        first_layer.weight.requires_grad_(True)
        if self._actor_base_gradient_hook is not None:
            self._actor_base_gradient_hook.remove()

        base_width = self.num_actor_base_obs

        def mask_base_columns(gradient: torch.Tensor) -> torch.Tensor:
            masked = gradient.clone()
            masked[:, :base_width] = 0.0
            return masked

        self._actor_base_gradient_hook = first_layer.weight.register_hook(mask_base_columns)
        trainable_actor_parameters = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and (name.startswith("actor.") or name.startswith("actor_tactile_encoder."))
        ]
        return {
            "mode": "frozen_official_sugar_actor_with_spatial_tactile_adapter",
            "actor_base_columns_frozen": base_width,
            "actor_tactile_columns_trainable": first_layer.in_features - base_width,
            "trainable_actor_parameters": trainable_actor_parameters,
            "frozen_zero_preserving_encoder_biases": sorted(frozen_encoder_biases),
            "action_noise_frozen": True,
            "critic_training": "unchanged_full_critic_optimization",
            "causal_invariant": "zero taxel field preserves accepted SUGAR actor throughout finetuning",
        }

    def _audit_zero_tactile_warm_start(
        self, source_state: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Prove that zero tactile preserves the official SUGAR MLP mapping."""

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        def source_mlp_forward(name: str, probe: torch.Tensor, target_mlp: nn.Module) -> torch.Tensor:
            value = probe
            # RSL-RL registers the same stateless activation instance at
            # several indices. ``named_children()`` de-duplicates shared
            # modules, so iterate the registered sequence verbatim.
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
            critic_tactile_features = self.critic_tactile_encoder(tactile_zeros)
            tactile_zero_abs_max = max(
                float(actor_tactile_features.abs().max().item()),
                float(critic_tactile_features.abs().max().item()),
            )

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
            target_actor = self.actor(torch.cat((actor_probe, actor_tactile_features), dim=-1))
            target_critic = self.critic(torch.cat((critic_probe, critic_tactile_features), dim=-1))
            actor_error = float((target_actor - source_actor).abs().max().item())
            critic_error = float((target_critic - source_critic).abs().max().item())

        tolerance = 1.0e-6
        if tactile_zero_abs_max != 0.0 or actor_error > tolerance or critic_error > tolerance:
            raise RuntimeError(
                "Zero-tactile SUGAR warm-start equivalence failed: "
                f"feature_abs_max={tactile_zero_abs_max}, actor_error={actor_error}, "
                f"critic_error={critic_error}, tolerance={tolerance}"
            )
        return {
            "zero_tactile_feature_abs_max": tactile_zero_abs_max,
            "actor_zero_tactile_max_abs_error": actor_error,
            "critic_zero_tactile_max_abs_error": critic_error,
            "zero_tactile_equivalence_tolerance": tolerance,
        }
