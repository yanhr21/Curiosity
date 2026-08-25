# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Causal transition policy over exact released SUGAR Tracker experts.

The two released Tracker actors remain parameter-exact and frozen.  The only
trainable actor component is a full SUGAR-topology residual that reads the
current Tracker observation, the causal command produced by the selected
released Generator, and the selected-skill one-hot vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


TRACKER_OBSERVATION_DIM = 510
GENERATED_COMMAND_DIM = 36
SELECTED_SKILL_DIM = 2
ACTION_DIM = 29
OFFICIAL_HIDDEN_DIMS = (512, 256, 128)
DUAL_COMMAND_INPUT_DIM = (
    TRACKER_OBSERVATION_DIM
    + 2 * GENERATED_COMMAND_DIM
    + SELECTED_SKILL_DIM
)


def _released_tracker(
    checkpoint: str | Path, device: torch.device | str = "cpu"
) -> tuple[MLP, torch.Tensor]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"released Tracker is missing model_state_dict: {path}")
    actor_state = {
        name.removeprefix("actor."): value
        for name, value in state.items()
        if name.startswith("actor.")
    }
    actor = MLP(
        TRACKER_OBSERVATION_DIM,
        ACTION_DIM,
        list(OFFICIAL_HIDDEN_DIMS),
        "elu",
    ).to(device)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval().requires_grad_(False)
    if "std" in state:
        std = state["std"].detach().to(device)
    elif "log_std" in state:
        std = state["log_std"].detach().to(device).exp()
    else:
        raise KeyError(f"released Tracker is missing std/log_std: {path}")
    if tuple(std.shape) != (ACTION_DIM,) or not torch.isfinite(std).all():
        raise RuntimeError(f"released Tracker std geometry drift: {path}")
    return actor, std


class FrozenSelectedTrackerResidual(nn.Module):
    """Exact selected expert plus a bounded, trainable transition residual."""

    def __init__(
        self,
        carry_tracker_checkpoint: str | Path,
        kick_tracker_checkpoint: str | Path,
        residual_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        residual_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if tuple(int(value) for value in residual_hidden_dims) != OFFICIAL_HIDDEN_DIMS:
            raise ValueError(
                "transition residual must retain the official 512/256/128 topology"
            )
        if not 0.0 < float(residual_limit) <= 1.0:
            raise ValueError("residual_limit must lie in (0, 1]")
        carry, carry_std = _released_tracker(carry_tracker_checkpoint)
        kick, kick_std = _released_tracker(kick_tracker_checkpoint)
        self.experts = nn.ModuleList((carry, kick))
        self.register_buffer("expert_std", torch.stack((carry_std, kick_std)))
        self.residual = MLP(
            TRACKER_OBSERVATION_DIM + GENERATED_COMMAND_DIM + SELECTED_SKILL_DIM,
            ACTION_DIM,
            list(OFFICIAL_HIDDEN_DIMS),
            "elu",
        )
        final = self.residual[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("transition residual output layer drift")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.residual_limit = float(residual_limit)

    @staticmethod
    def _split(actor_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            TRACKER_OBSERVATION_DIM
            + GENERATED_COMMAND_DIM
            + SELECTED_SKILL_DIM
        )
        if actor_input.ndim != 2 or actor_input.shape[-1] != expected:
            raise RuntimeError(
                f"transition actor input drift: {tuple(actor_input.shape)}"
            )
        observation = actor_input[:, :TRACKER_OBSERVATION_DIM]
        command_start = TRACKER_OBSERVATION_DIM
        command = actor_input[
            :, command_start : command_start + GENERATED_COMMAND_DIM
        ]
        skill = actor_input[:, -SELECTED_SKILL_DIM:]
        if not torch.isfinite(actor_input).all():
            raise RuntimeError("transition actor input is non-finite")
        if not torch.allclose(
            skill.sum(dim=-1), torch.ones_like(skill[:, 0]), atol=1.0e-6, rtol=0.0
        ) or torch.any((skill < -1.0e-6) | (skill > 1.0 + 1.0e-6)):
            raise RuntimeError("selected skill must be a causal two-way one-hot vector")
        selected_observation = observation.clone()
        selected_observation[:, :GENERATED_COMMAND_DIM] = command
        return selected_observation, skill

    def endpoint_action(self, actor_input: torch.Tensor) -> torch.Tensor:
        selected_observation, skill = self._split(actor_input)
        actions = torch.stack(
            tuple(expert(selected_observation) for expert in self.experts), dim=1
        )
        return torch.sum(actions * skill.unsqueeze(-1), dim=1)

    def endpoint_std(self, actor_input: torch.Tensor) -> torch.Tensor:
        _, skill = self._split(actor_input)
        return torch.sum(self.expert_std.unsqueeze(0) * skill.unsqueeze(-1), dim=1)

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        endpoint = self.endpoint_action(actor_input)
        residual = self.residual_limit * torch.tanh(self.residual(actor_input))
        return endpoint + residual


class FrozenExpertTransitionActorCritic(ActorCritic):
    """RSL-RL interface for the frozen-expert transition controller."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        carry_tracker_checkpoint: str,
        kick_tracker_checkpoint: str,
        transition_residual_limit: float = 1.0,
        actor_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        **kwargs,
    ) -> None:
        if num_actions != ACTION_DIM:
            raise RuntimeError(f"transition action geometry drift: {num_actions}")
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            **kwargs,
        )
        self.actor = FrozenSelectedTrackerResidual(
            carry_tracker_checkpoint,
            kick_tracker_checkpoint,
            actor_hidden_dims,
            transition_residual_limit,
        ).to(next(self.critic.parameters()).device)

    def distillation_teacher(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact selected released expert for repository BCPPO."""

        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.actor_obs_normalization:
            raise RuntimeError(
                "transition actor normalization would alter exact endpoint inputs"
            )
        with torch.no_grad():
            return (
                self.actor.endpoint_action(actor_input),
                self.actor.endpoint_std(actor_input),
            )


class FrozenExpertCausalActionComposer(nn.Module):
    """State-dependent composition of exact Carry/Kick actions plus residual."""

    def __init__(
        self,
        carry_tracker_checkpoint: str | Path,
        kick_tracker_checkpoint: str | Path,
        composer_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        residual_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if tuple(int(value) for value in composer_hidden_dims) != OFFICIAL_HIDDEN_DIMS:
            raise ValueError(
                "action composer must retain the official 512/256/128 topology"
            )
        if not 0.0 < float(residual_limit) <= 1.0:
            raise ValueError("residual_limit must lie in (0, 1]")
        carry, carry_std = _released_tracker(carry_tracker_checkpoint)
        kick, kick_std = _released_tracker(kick_tracker_checkpoint)
        self.experts = nn.ModuleList((carry, kick))
        self.register_buffer("expert_std", torch.stack((carry_std, kick_std)))
        # Output 0 is a signed adjustment of the exact selected endpoint's
        # Kick weight.  Outputs 1: are a bounded action residual.  A zero final
        # layer is therefore exactly the selected released expert, while the
        # clamp boundary retains a nonzero gradient for a causal transition.
        self.composer = MLP(
            DUAL_COMMAND_INPUT_DIM,
            1 + ACTION_DIM,
            list(OFFICIAL_HIDDEN_DIMS),
            "elu",
        )
        final = self.composer[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("action composer output layer drift")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.residual_limit = float(residual_limit)

    @staticmethod
    def _split(
        actor_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if actor_input.ndim != 2 or actor_input.shape[-1] != DUAL_COMMAND_INPUT_DIM:
            raise RuntimeError(
                f"causal action composer input drift: {tuple(actor_input.shape)}"
            )
        if not torch.isfinite(actor_input).all():
            raise RuntimeError("causal action composer input is non-finite")
        observation = actor_input[:, :TRACKER_OBSERVATION_DIM]
        carry_start = TRACKER_OBSERVATION_DIM
        kick_start = carry_start + GENERATED_COMMAND_DIM
        carry_command = actor_input[
            :, carry_start : carry_start + GENERATED_COMMAND_DIM
        ]
        kick_command = actor_input[:, kick_start : kick_start + GENERATED_COMMAND_DIM]
        skill = actor_input[:, -SELECTED_SKILL_DIM:]
        if not torch.allclose(
            skill.sum(dim=-1), torch.ones_like(skill[:, 0]), atol=1.0e-6, rtol=0.0
        ) or torch.any((skill < -1.0e-6) | (skill > 1.0 + 1.0e-6)):
            raise RuntimeError("selected skill must be a causal two-way one-hot vector")
        carry_observation = observation.clone()
        kick_observation = observation.clone()
        carry_observation[:, :GENERATED_COMMAND_DIM] = carry_command
        kick_observation[:, :GENERATED_COMMAND_DIM] = kick_command
        return carry_observation, kick_observation, skill, actor_input

    def expert_actions(
        self, actor_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        carry_observation, kick_observation, skill, _ = self._split(actor_input)
        return (
            self.experts[0](carry_observation),
            self.experts[1](kick_observation),
            skill,
        )

    def endpoint_action(self, actor_input: torch.Tensor) -> torch.Tensor:
        carry_action, kick_action, skill = self.expert_actions(actor_input)
        return carry_action * skill[:, :1] + kick_action * skill[:, 1:2]

    def endpoint_std(self, actor_input: torch.Tensor) -> torch.Tensor:
        _, _, skill, _ = self._split(actor_input)
        return torch.sum(self.expert_std.unsqueeze(0) * skill.unsqueeze(-1), dim=1)

    def kick_weight(self, actor_input: torch.Tensor) -> torch.Tensor:
        _, _, skill, full_input = self._split(actor_input)
        gate_logit = self.composer(full_input)[:, :1]
        # At zero initialization this is exactly the selected one-hot endpoint:
        # Carry=0, Kick=1.  Positive logits delay a selected Kick transition;
        # negative logits begin a selected Carry transition toward Kick.
        return torch.clamp(
            skill[:, 1:2] - 0.5 * torch.tanh(gate_logit), 0.0, 1.0
        )

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        carry_action, kick_action, skill = self.expert_actions(actor_input)
        composer_output = self.composer(actor_input)
        kick_weight = torch.clamp(
            skill[:, 1:2] - 0.5 * torch.tanh(composer_output[:, :1]),
            0.0,
            1.0,
        )
        residual = self.residual_limit * torch.tanh(composer_output[:, 1:])
        return (
            carry_action * (1.0 - kick_weight)
            + kick_action * kick_weight
            + residual
        )


class FrozenExpertCausalActionComposerActorCritic(ActorCritic):
    """RSL-RL policy for causal state-dependent frozen-expert composition."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        carry_tracker_checkpoint: str,
        kick_tracker_checkpoint: str,
        transition_residual_limit: float = 1.0,
        actor_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        **kwargs,
    ) -> None:
        if num_actions != ACTION_DIM:
            raise RuntimeError(f"transition action geometry drift: {num_actions}")
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            **kwargs,
        )
        self.actor = FrozenExpertCausalActionComposer(
            carry_tracker_checkpoint,
            kick_tracker_checkpoint,
            actor_hidden_dims,
            transition_residual_limit,
        ).to(next(self.critic.parameters()).device)

    def _actor_input(self, obs) -> torch.Tensor:
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.actor_obs_normalization:
            raise RuntimeError(
                "action-composer normalization would alter exact endpoint inputs"
            )
        return actor_input

    def distillation_teacher(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact selected released expert for repository BCPPO."""

        actor_input = self._actor_input(obs)
        with torch.no_grad():
            return (
                self.actor.endpoint_action(actor_input),
                self.actor.endpoint_std(actor_input),
            )

    def composition_kick_weight(self, obs) -> torch.Tensor:
        """Expose the deployed causal mixture weight for frozen audit only."""

        return self.actor.kick_weight(self._actor_input(obs))
