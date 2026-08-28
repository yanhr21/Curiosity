# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Causal transition policy over exact released SUGAR Tracker experts.

The two released Tracker actors remain parameter-exact and frozen.  The older
residual topology reads the selected command, while the current causal action
composer reads the current Tracker observation, both causal released-Generator
commands and the selected-skill one-hot.  Only the full SUGAR-topology
residual/composer module is trainable.
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
REFINER_OBSERVATION_DIM = 890
OFFICIAL_HIDDEN_DIMS = (512, 256, 128)
DUAL_COMMAND_INPUT_DIM = (
    TRACKER_OBSERVATION_DIM
    + 2 * GENERATED_COMMAND_DIM
    + SELECTED_SKILL_DIM
)
TEMPORAL_HISTORY_STEPS = 10
TEMPORAL_MODEL_DIM = 384
TEMPORAL_ACTOR_INPUT_DIM = DUAL_COMMAND_INPUT_DIM * (1 + TEMPORAL_HISTORY_STEPS)
REFINER_TEMPORAL_ACTOR_INPUT_DIM = REFINER_OBSERVATION_DIM * (
    1 + TEMPORAL_HISTORY_STEPS
)
REFINER_TEMPORAL_COMMAND_ACTOR_INPUT_DIM = (
    REFINER_TEMPORAL_ACTOR_INPUT_DIM + GENERATED_COMMAND_DIM
)
REFINER_ACTION_CHUNK_KNOTS = 7
REFINER_ACTION_CHUNK_DIM = REFINER_ACTION_CHUNK_KNOTS * ACTION_DIM


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
        # negative logits begin a selected Carry transition toward Kick.  The
        # unit tanh adjustment makes the complete Carry/Kick convex segment
        # reachable from either selected endpoint.
        return torch.clamp(
            skill[:, 1:2] - torch.tanh(gate_logit), 0.0, 1.0
        )

    def composition_terms(
        self, actor_input: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return deployed action terms for frozen causal attribution."""

        carry_action, kick_action, skill = self.expert_actions(actor_input)
        composer_output = self.composer(actor_input)
        kick_weight = torch.clamp(
            skill[:, 1:2] - torch.tanh(composer_output[:, :1]),
            0.0,
            1.0,
        )
        residual = self.residual_limit * torch.tanh(composer_output[:, 1:])
        selected_endpoint = (
            carry_action * skill[:, :1] + kick_action * skill[:, 1:2]
        )
        mixed_endpoint = (
            carry_action * (1.0 - kick_weight) + kick_action * kick_weight
        )
        composed_action = mixed_endpoint + residual
        return {
            "kick_weight": kick_weight,
            "selected_endpoint_action": selected_endpoint,
            "mixed_endpoint_action": mixed_endpoint,
            "bounded_residual_action": residual,
            "composed_action": composed_action,
        }

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.composition_terms(actor_input)["composed_action"]


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

    def composition_audit_terms(self, obs) -> dict[str, torch.Tensor]:
        """Expose exact deployed terms to camera-free frozen evaluation only."""

        return self.actor.composition_terms(self._actor_input(obs))


class _CausalTemporalComposerCore(nn.Module):
    """Six-layer past-only transition model with an exact-zero output head."""

    def __init__(
        self,
        frame_input_dim: int = DUAL_COMMAND_INPUT_DIM,
        output_dim: int = 1 + ACTION_DIM,
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        if frame_input_dim < 1 or output_dim < 1 or condition_dim < 0:
            raise ValueError("temporal composer dimensions must be positive")
        self.frame_input_dim = int(frame_input_dim)
        self.output_dim = int(output_dim)
        self.condition_dim = int(condition_dim)
        self.frame_projection = nn.Sequential(
            nn.Linear(self.frame_input_dim, TEMPORAL_MODEL_DIM),
            nn.LayerNorm(TEMPORAL_MODEL_DIM),
        )
        self.condition_projection = (
            nn.Sequential(
                nn.Linear(self.condition_dim, TEMPORAL_MODEL_DIM),
                nn.LayerNorm(TEMPORAL_MODEL_DIM),
            )
            if self.condition_dim
            else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, TEMPORAL_MODEL_DIM))
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                TEMPORAL_HISTORY_STEPS + 1 + int(bool(self.condition_dim)),
                TEMPORAL_MODEL_DIM,
            )
        )
        layer = nn.TransformerEncoderLayer(
            d_model=TEMPORAL_MODEL_DIM,
            nhead=8,
            dim_feedforward=4 * TEMPORAL_MODEL_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=6,
            norm=nn.LayerNorm(TEMPORAL_MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.output = MLP(
            TEMPORAL_MODEL_DIM,
            self.output_dim,
            list(OFFICIAL_HIDDEN_DIMS),
            "elu",
        )
        final = self.output[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("temporal composer output layer drift")
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def encode(
        self,
        history: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (
            TEMPORAL_HISTORY_STEPS,
            self.frame_input_dim,
        ):
            raise RuntimeError(
                f"causal temporal history drift: {tuple(history.shape)}"
            )
        tokens = self.frame_projection(history)
        cls = self.cls_token.expand(history.shape[0], -1, -1)
        if self.condition_projection is None:
            if condition is not None:
                raise RuntimeError("unconditioned temporal composer received a condition")
            tokens = torch.cat((cls, tokens), dim=1)
        else:
            if condition is None or tuple(condition.shape) != (
                history.shape[0],
                self.condition_dim,
            ):
                raise RuntimeError(
                    f"causal temporal condition drift: "
                    f"{None if condition is None else tuple(condition.shape)}"
                )
            if not torch.isfinite(condition).all():
                raise RuntimeError("causal temporal condition is non-finite")
            condition_token = self.condition_projection(condition).unsqueeze(1)
            tokens = torch.cat((cls, condition_token, tokens), dim=1)
        tokens = tokens + self.position_embedding
        encoded = self.transformer(tokens)
        return encoded[:, 0]

    def forward(
        self,
        history: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.output(self.encode(history, condition))


class FrozenExpertCausalTemporalActionComposer(
    FrozenExpertCausalActionComposer
):
    """Temporal exact-endpoint composer over an explicit causal 10-frame history."""

    def __init__(
        self,
        carry_tracker_checkpoint: str | Path,
        kick_tracker_checkpoint: str | Path,
        composer_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        residual_limit: float = 1.0,
    ) -> None:
        super().__init__(
            carry_tracker_checkpoint,
            kick_tracker_checkpoint,
            composer_hidden_dims,
            residual_limit,
        )
        del self.composer
        self.temporal_composer = _CausalTemporalComposerCore()

    @staticmethod
    def _split_temporal_input(
        actor_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actor_input.ndim != 2 or actor_input.shape[-1] != TEMPORAL_ACTOR_INPUT_DIM:
            raise RuntimeError(
                f"causal temporal actor input drift: {tuple(actor_input.shape)}"
            )
        if not torch.isfinite(actor_input).all():
            raise RuntimeError("causal temporal actor input is non-finite")
        current = actor_input[:, :DUAL_COMMAND_INPUT_DIM]
        history = actor_input[:, DUAL_COMMAND_INPUT_DIM:].reshape(
            actor_input.shape[0], TEMPORAL_HISTORY_STEPS, DUAL_COMMAND_INPUT_DIM
        )
        if not torch.equal(history[:, -1], current):
            raise RuntimeError("temporal history does not end at the deployed current state")
        return current, history

    def expert_actions(
        self, actor_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current, _ = self._split_temporal_input(actor_input)
        return super().expert_actions(current)

    def endpoint_std(self, actor_input: torch.Tensor) -> torch.Tensor:
        current, _ = self._split_temporal_input(actor_input)
        return super().endpoint_std(current)

    def _temporal_output(self, actor_input: torch.Tensor) -> torch.Tensor:
        _, history = self._split_temporal_input(actor_input)
        return self.temporal_composer(history)

    def kick_weight(self, actor_input: torch.Tensor) -> torch.Tensor:
        current, _ = self._split_temporal_input(actor_input)
        _, _, skill, _ = super()._split(current)
        gate_logit = self._temporal_output(actor_input)[:, :1]
        return torch.clamp(skill[:, 1:2] - torch.tanh(gate_logit), 0.0, 1.0)

    def composition_terms(
        self, actor_input: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        current, _ = self._split_temporal_input(actor_input)
        carry_action, kick_action, skill = self.expert_actions(actor_input)
        composer_output = self._temporal_output(actor_input)
        kick_weight = torch.clamp(
            skill[:, 1:2] - torch.tanh(composer_output[:, :1]), 0.0, 1.0
        )
        residual = self.residual_limit * torch.tanh(composer_output[:, 1:])
        selected_endpoint = carry_action * skill[:, :1] + kick_action * skill[:, 1:2]
        mixed_endpoint = carry_action * (1.0 - kick_weight) + kick_action * kick_weight
        composed_action = mixed_endpoint + residual
        return {
            "kick_weight": kick_weight,
            "selected_endpoint_action": selected_endpoint,
            "mixed_endpoint_action": mixed_endpoint,
            "bounded_residual_action": residual,
            "composed_action": composed_action,
            "temporal_history_last_frame": current,
        }


class FrozenExpertCausalTemporalActionComposerActorCritic(
    FrozenExpertCausalActionComposerActorCritic
):
    """RSL-RL policy for exact-endpoint temporal frozen-expert composition."""

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
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            carry_tracker_checkpoint=carry_tracker_checkpoint,
            kick_tracker_checkpoint=kick_tracker_checkpoint,
            transition_residual_limit=transition_residual_limit,
            actor_hidden_dims=actor_hidden_dims,
            **kwargs,
        )
        self.actor = FrozenExpertCausalTemporalActionComposer(
            carry_tracker_checkpoint,
            kick_tracker_checkpoint,
            actor_hidden_dims,
            transition_residual_limit,
        ).to(next(self.critic.parameters()).device)


def _released_refiner(
    checkpoint: str | Path, device: torch.device | str = "cpu"
) -> tuple[MLP, torch.Tensor]:
    """Load only the exact released Refiner actor/std for adapter composition."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"released Refiner is missing model_state_dict: {path}")
    actor_state = {
        name.removeprefix("actor."): value
        for name, value in state.items()
        if name.startswith("actor.")
    }
    actor = MLP(
        REFINER_OBSERVATION_DIM,
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
        raise KeyError(f"released Refiner is missing std/log_std: {path}")
    if tuple(std.shape) != (ACTION_DIM,) or not torch.isfinite(std).all():
        raise RuntimeError(f"released Refiner std geometry drift: {path}")
    return actor, std


class FrozenOfficialRefinerResidual(nn.Module):
    """Exact official 890-D Refiner plus a zero-start bounded residual adapter."""

    def __init__(
        self,
        official_refiner_checkpoint: str | Path,
        residual_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        residual_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if tuple(int(value) for value in residual_hidden_dims) != OFFICIAL_HIDDEN_DIMS:
            raise ValueError("Refiner residual must retain the official 512/256/128 topology")
        if not 0.0 < float(residual_limit) <= 1.0:
            raise ValueError("residual_limit must lie in (0, 1]")
        expert, expert_std = _released_refiner(official_refiner_checkpoint)
        self.expert = expert
        self.register_buffer("expert_std", expert_std)
        self.residual = MLP(
            REFINER_OBSERVATION_DIM,
            ACTION_DIM,
            list(OFFICIAL_HIDDEN_DIMS),
            "elu",
        )
        final = self.residual[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("Refiner residual output layer drift")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.residual_limit = float(residual_limit)

    @staticmethod
    def _validate(actor_input: torch.Tensor) -> None:
        if actor_input.ndim != 2 or actor_input.shape[-1] != REFINER_OBSERVATION_DIM:
            raise RuntimeError(f"Refiner residual input drift: {tuple(actor_input.shape)}")
        if not torch.isfinite(actor_input).all():
            raise RuntimeError("Refiner residual input is non-finite")

    def endpoint_action(self, actor_input: torch.Tensor) -> torch.Tensor:
        self._validate(actor_input)
        return self.expert(actor_input)

    def endpoint_std(self, actor_input: torch.Tensor) -> torch.Tensor:
        self._validate(actor_input)
        return self.expert_std.expand(actor_input.shape[0], -1)

    def composition_terms(self, actor_input: torch.Tensor) -> dict[str, torch.Tensor]:
        endpoint = self.endpoint_action(actor_input)
        residual = self.residual_limit * torch.tanh(self.residual(actor_input))
        return {
            "selected_endpoint_action": endpoint,
            "bounded_residual_action": residual,
            "composed_action": endpoint + residual,
        }

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.composition_terms(actor_input)["composed_action"]


class FrozenOfficialRefinerResidualActorCritic(ActorCritic):
    """RSL-RL interface for the exact Refiner plus trainable official-scale adapter."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        official_refiner_checkpoint: str,
        transition_residual_limit: float = 1.0,
        actor_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        **kwargs,
    ) -> None:
        if num_actions != ACTION_DIM:
            raise RuntimeError(f"Refiner residual action geometry drift: {num_actions}")
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            **kwargs,
        )
        self.actor = FrozenOfficialRefinerResidual(
            official_refiner_checkpoint,
            actor_hidden_dims,
            transition_residual_limit,
        ).to(next(self.critic.parameters()).device)

    def _actor_input(self, obs) -> torch.Tensor:
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.actor_obs_normalization:
            raise RuntimeError("Refiner residual normalization would alter exact inputs")
        return actor_input

    def distillation_teacher(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        actor_input = self._actor_input(obs)
        with torch.no_grad():
            return self.actor.endpoint_action(actor_input), self.actor.endpoint_std(actor_input)

    def composition_audit_terms(self, obs) -> dict[str, torch.Tensor]:
        return self.actor.composition_terms(self._actor_input(obs))


class FrozenOfficialRefinerTrackerSupervisedResidualActorCritic(
    FrozenOfficialRefinerResidualActorCritic
):
    """Frozen Refiner adapter supervised by the exact released Tracker action.

    The deployed actor remains the official 890-D Refiner plus its bounded
    official-scale residual.  The 510-D Tracker is a frozen, training-only
    distillation teacher queried from a separately named observation group.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        official_refiner_checkpoint: str,
        released_tracker_checkpoint: str,
        **kwargs,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            official_refiner_checkpoint=official_refiner_checkpoint,
            **kwargs,
        )
        device = next(self.critic.parameters()).device
        teacher, teacher_std = _released_tracker(
            released_tracker_checkpoint, device=device
        )
        self.tracker_teacher = teacher
        self.register_buffer("tracker_teacher_std", teacher_std)

    def distillation_teacher(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_groups = self.obs_groups.get("teacher")
        if not teacher_groups:
            raise RuntimeError("Tracker supervision requires a teacher observation group")
        teacher_input = torch.cat([obs[name] for name in teacher_groups], dim=-1)
        if teacher_input.ndim != 2 or teacher_input.shape[-1] != TRACKER_OBSERVATION_DIM:
            raise RuntimeError(
                f"Tracker teacher observation drift: {tuple(teacher_input.shape)}"
            )
        if not torch.isfinite(teacher_input).all():
            raise RuntimeError("Tracker teacher observation is non-finite")
        with torch.no_grad():
            mean = self.tracker_teacher(teacher_input)
            std = self.tracker_teacher_std.expand(teacher_input.shape[0], -1)
        return mean, std


class FrozenOfficialRefinerCausalTemporalComposer(nn.Module):
    """Exact official Refiner plus a serious past-only transition composer.

    The deployed input is the current official 890-D Refiner observation followed by an
    explicit causal 10-frame history ending at that same tensor.  The six-layer 384-D
    Transformer produces an expert-retention scalar and a bounded 29-D action correction.
    Its output head is exact zero, so initialization is bitwise the released Refiner.
    """

    def __init__(
        self,
        official_refiner_checkpoint: str | Path,
        residual_limit: float = 1.0,
        additive_residual: bool = False,
        command_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < float(residual_limit) <= 1.0:
            raise ValueError("residual_limit must lie in (0, 1]")
        expert, expert_std = _released_refiner(official_refiner_checkpoint)
        self.expert = expert
        self.register_buffer("expert_std", expert_std)
        self.additive_residual = bool(additive_residual)
        self.command_conditioned = bool(command_conditioned)
        if self.command_conditioned and not self.additive_residual:
            raise ValueError(
                "current-command conditioning is admitted only for the exact "
                "additive Refiner correction"
            )
        self.temporal_composer = _CausalTemporalComposerCore(
            REFINER_OBSERVATION_DIM,
            ACTION_DIM if self.additive_residual else 1 + ACTION_DIM,
            GENERATED_COMMAND_DIM if self.command_conditioned else 0,
        )
        self.residual_limit = float(residual_limit)

    def _split_temporal_input(
        self,
        actor_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        expected_dim = (
            REFINER_TEMPORAL_COMMAND_ACTOR_INPUT_DIM
            if self.command_conditioned
            else REFINER_TEMPORAL_ACTOR_INPUT_DIM
        )
        if actor_input.ndim != 2 or actor_input.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Refiner temporal actor input drift: {tuple(actor_input.shape)}"
            )
        if not torch.isfinite(actor_input).all():
            raise RuntimeError("Refiner temporal actor input is non-finite")
        current = actor_input[:, :REFINER_OBSERVATION_DIM]
        history_end = REFINER_TEMPORAL_ACTOR_INPUT_DIM
        history = actor_input[
            :, REFINER_OBSERVATION_DIM:history_end
        ].reshape(
            actor_input.shape[0],
            TEMPORAL_HISTORY_STEPS,
            REFINER_OBSERVATION_DIM,
        )
        if not torch.equal(history[:, -1], current):
            raise RuntimeError("Refiner temporal history does not end at current state")
        command = actor_input[:, history_end:] if self.command_conditioned else None
        if command is not None and tuple(command.shape) != (
            actor_input.shape[0],
            GENERATED_COMMAND_DIM,
        ):
            raise RuntimeError(f"Refiner current command drift: {tuple(command.shape)}")
        return current, history, command

    def endpoint_action(self, actor_input: torch.Tensor) -> torch.Tensor:
        current, _, _ = self._split_temporal_input(actor_input)
        return self.expert(current)

    def endpoint_std(self, actor_input: torch.Tensor) -> torch.Tensor:
        current, _, _ = self._split_temporal_input(actor_input)
        return self.expert_std.expand(current.shape[0], -1)

    def composition_terms(self, actor_input: torch.Tensor) -> dict[str, torch.Tensor]:
        current, history, command = self._split_temporal_input(actor_input)
        endpoint = self.expert(current)
        composer_output = self.temporal_composer(history, command)
        if self.additive_residual:
            expert_retention = torch.ones_like(composer_output[:, :1])
            residual = self.residual_limit * torch.tanh(composer_output)
            composed = endpoint + residual
        else:
            expert_retention = torch.clamp(
                1.0 - torch.tanh(composer_output[:, :1]),
                0.0,
                1.0,
            )
            residual = self.residual_limit * torch.tanh(composer_output[:, 1:])
            composed = expert_retention * endpoint + residual
        return {
            "expert_retention": expert_retention,
            "selected_endpoint_action": endpoint,
            "bounded_residual_action": residual,
            "composed_action": composed,
            "temporal_history_last_frame": history[:, -1],
            **(
                {"current_reference_command": command}
                if command is not None
                else {}
            ),
        }

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.composition_terms(actor_input)["composed_action"]


class FrozenOfficialRefinerCausalTemporalComposerActorCritic(ActorCritic):
    """RSL-RL interface for the exact Refiner and causal temporal composer."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        official_refiner_checkpoint: str,
        transition_residual_limit: float = 1.0,
        temporal_additive_residual: bool = False,
        temporal_command_conditioned: bool = False,
        actor_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        **kwargs,
    ) -> None:
        if num_actions != ACTION_DIM:
            raise RuntimeError(f"Refiner temporal action geometry drift: {num_actions}")
        if tuple(int(value) for value in actor_hidden_dims) != OFFICIAL_HIDDEN_DIMS:
            raise ValueError("Refiner temporal composer must retain 512/256/128 output MLP")
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            **kwargs,
        )
        self.actor = FrozenOfficialRefinerCausalTemporalComposer(
            official_refiner_checkpoint,
            transition_residual_limit,
            additive_residual=temporal_additive_residual,
            command_conditioned=temporal_command_conditioned,
        ).to(next(self.critic.parameters()).device)

    def _actor_input(self, obs) -> torch.Tensor:
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.actor_obs_normalization:
            raise RuntimeError("Refiner temporal normalization would alter exact inputs")
        return actor_input

    def distillation_teacher(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        actor_input = self._actor_input(obs)
        with torch.no_grad():
            return self.actor.endpoint_action(actor_input), self.actor.endpoint_std(actor_input)

    def composition_audit_terms(self, obs) -> dict[str, torch.Tensor]:
        return self.actor.composition_terms(self._actor_input(obs))


class RefinerReferencePhaseCausalTemporalActor(nn.Module):
    """Serious past-only controller for one frozen-Refiner reference phase.

    The environment, rather than this module, owns and freezes the released
    Refiner.  This actor consumes the exact current 890-D observation followed
    by the same explicit ten-frame causal history used by the admitted temporal
    composer.  Its single exact-zero-initialized output retimes only the future
    reference fields supplied to that Refiner.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temporal_composer = _CausalTemporalComposerCore(
            REFINER_OBSERVATION_DIM,
            1,
        )

    @staticmethod
    def _history(actor_input: torch.Tensor) -> torch.Tensor:
        if actor_input.ndim != 2 or actor_input.shape[-1] != REFINER_TEMPORAL_ACTOR_INPUT_DIM:
            raise RuntimeError(
                f"Refiner reference-phase actor input drift: {tuple(actor_input.shape)}"
            )
        if not torch.isfinite(actor_input).all():
            raise RuntimeError("Refiner reference-phase actor input is non-finite")
        current = actor_input[:, :REFINER_OBSERVATION_DIM]
        history = actor_input[:, REFINER_OBSERVATION_DIM:].reshape(
            actor_input.shape[0],
            TEMPORAL_HISTORY_STEPS,
            REFINER_OBSERVATION_DIM,
        )
        if not torch.equal(history[:, -1], current):
            raise RuntimeError(
                "Refiner reference-phase history does not end at current state"
            )
        return history

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.temporal_composer(self._history(actor_input))


class RefinerReferencePhaseCausalTemporalActorCritic(ActorCritic):
    """RSL-RL interface for the one-scalar causal reference-phase controller."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        actor_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        **kwargs,
    ) -> None:
        if num_actions != 1:
            raise RuntimeError(
                f"Refiner reference-phase action geometry drift: {num_actions}"
            )
        if tuple(int(value) for value in actor_hidden_dims) != OFFICIAL_HIDDEN_DIMS:
            raise ValueError(
                "Refiner reference-phase controller must retain 512/256/128 output MLP"
            )
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            **kwargs,
        )
        self.actor = RefinerReferencePhaseCausalTemporalActor().to(
            next(self.critic.parameters()).device
        )

    def _actor_input(self, obs) -> torch.Tensor:
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.actor_obs_normalization:
            raise RuntimeError(
                "Refiner reference-phase normalization would alter exact inputs"
            )
        return actor_input

    def distillation_teacher(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        actor_input = self._actor_input(obs)
        zero_phase = torch.zeros(
            actor_input.shape[0], 1, device=actor_input.device, dtype=actor_input.dtype
        )
        return zero_phase, torch.full_like(zero_phase, 0.05)


class RefinerActionChunkCausalTemporalActor(nn.Module):
    """Past-only 35-step transition planner with an exact-zero output head."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal_composer = _CausalTemporalComposerCore(
            REFINER_OBSERVATION_DIM,
            REFINER_ACTION_CHUNK_DIM,
        )

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        history = RefinerReferencePhaseCausalTemporalActor._history(actor_input)
        return self.temporal_composer(history)


class RefinerActionChunkCausalTemporalActorCritic(ActorCritic):
    """RSL-RL interface for one latched seven-knot Refiner correction plan."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        actor_hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        **kwargs,
    ) -> None:
        if num_actions != REFINER_ACTION_CHUNK_DIM:
            raise RuntimeError(
                f"Refiner action-chunk geometry drift: {num_actions}"
            )
        if tuple(int(value) for value in actor_hidden_dims) != OFFICIAL_HIDDEN_DIMS:
            raise ValueError(
                "Refiner action-chunk controller must retain 512/256/128 output MLP"
            )
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            **kwargs,
        )
        self.actor = RefinerActionChunkCausalTemporalActor().to(
            next(self.critic.parameters()).device
        )

    def _actor_input(self, obs) -> torch.Tensor:
        actor_input = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.actor_obs_normalization:
            raise RuntimeError(
                "Refiner action-chunk normalization would alter exact inputs"
            )
        return actor_input

    def distillation_teacher(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        actor_input = self._actor_input(obs)
        zero_plan = torch.zeros(
            actor_input.shape[0],
            REFINER_ACTION_CHUNK_DIM,
            device=actor_input.device,
            dtype=actor_input.dtype,
        )
        return zero_plan, torch.full_like(zero_plan, 0.35)


class FrozenOfficialRefinerTrackerSupervisedCausalTemporalComposerActorCritic(
    FrozenOfficialRefinerCausalTemporalComposerActorCritic
):
    """Causal frozen-Refiner composer supervised by the released Tracker.

    The deployed actor remains the exact official Refiner plus the admitted
    six-layer causal temporal composer.  The released 510-D Tracker is a
    frozen, training-only current-state action teacher and is not queried by
    ``act_inference``.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        official_refiner_checkpoint: str,
        released_tracker_checkpoint: str,
        **kwargs,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            official_refiner_checkpoint=official_refiner_checkpoint,
            **kwargs,
        )
        device = next(self.critic.parameters()).device
        teacher, teacher_std = _released_tracker(
            released_tracker_checkpoint, device=device
        )
        self.tracker_teacher = teacher
        self.register_buffer("tracker_teacher_std", teacher_std)

    def distillation_teacher(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_groups = self.obs_groups.get("teacher")
        if not teacher_groups:
            raise RuntimeError("Tracker supervision requires a teacher observation group")
        teacher_input = torch.cat([obs[name] for name in teacher_groups], dim=-1)
        if teacher_input.ndim != 2 or teacher_input.shape[-1] != TRACKER_OBSERVATION_DIM:
            raise RuntimeError(
                f"Tracker teacher observation drift: {tuple(teacher_input.shape)}"
            )
        if not torch.isfinite(teacher_input).all():
            raise RuntimeError("Tracker teacher observation is non-finite")
        with torch.no_grad():
            mean = self.tracker_teacher(teacher_input)
            std = self.tracker_teacher_std.expand(teacher_input.shape[0], -1)
        return mean, std
