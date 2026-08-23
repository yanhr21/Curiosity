# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Independent optimizer/checkpoint boundary for continuous original ICM.

This trainer consumes only aligned ``o_t, a_t, o_t+1`` tensors. It has no task
reward or outcome argument, so task success, lift progress, slip penalties, and
strategy labels cannot gate discovery learning. The validity mask is solely for
excluding transitions whose next observation was overwritten by an automatic
environment reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sugar_rl.utils.original_icm_continuous import (
    OriginalICMContinuous,
    OriginalICMContinuousCfg,
)


@dataclass
class ICMTransitionBatch:
    vector_obs_t: torch.Tensor
    tactile_history_t: torch.Tensor
    applied_action_policy_units_t: torch.Tensor
    vector_obs_tp1: torch.Tensor
    tactile_history_tp1: torch.Tensor
    transition_valid: torch.Tensor

    def select_valid(self) -> "ICMTransitionBatch":
        if self.transition_valid.ndim != 1 or self.transition_valid.dtype != torch.bool:
            raise ValueError("ICM transition_valid must be a one-dimensional bool tensor")
        batch_size = self.transition_valid.shape[0]
        tensors = (
            self.vector_obs_t,
            self.tactile_history_t,
            self.applied_action_policy_units_t,
            self.vector_obs_tp1,
            self.tactile_history_tp1,
        )
        if any(tensor.shape[0] != batch_size for tensor in tensors):
            raise ValueError("ICM transition tensors and validity mask are misaligned")
        if not self.transition_valid.any():
            raise ValueError("ICM update contains no valid temporal transitions")
        mask = self.transition_valid
        return ICMTransitionBatch(
            vector_obs_t=self.vector_obs_t[mask],
            tactile_history_t=self.tactile_history_t[mask],
            applied_action_policy_units_t=self.applied_action_policy_units_t[mask],
            vector_obs_tp1=self.vector_obs_tp1[mask],
            tactile_history_tp1=self.tactile_history_tp1[mask],
            transition_valid=torch.ones(
                int(mask.sum().item()), dtype=torch.bool, device=mask.device
            ),
        )


class OriginalICMTrainer:
    """Original ICM predictor optimizer, separate from the policy optimizer."""

    def __init__(
        self,
        cfg: OriginalICMContinuousCfg,
        device: torch.device | str,
        learning_rate: float = 1.0e-4,
        gradient_norm_clip: float = 40.0,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.learning_rate = float(learning_rate)
        self.gradient_norm_clip = float(gradient_norm_clip)
        if self.learning_rate != 1.0e-4:
            raise ValueError("the initial ICM adapter preserves official Adam lr=1e-4")
        if self.gradient_norm_clip != 40.0:
            raise ValueError("the initial ICM adapter preserves official grad clip=40")
        self.module = OriginalICMContinuous(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.module.predictor_parameters(), lr=self.learning_rate
        )
        self.optimizer_updates = 0
        self.transitions_trained = 0
        self.transitions_scored = 0
        self.post_reset_pairs_excluded = 0

    @torch.no_grad()
    def bootstrap_normalizer(self, batch: ICMTransitionBatch) -> dict[str, int]:
        """Initialize/update observation moments without training or rewards."""

        valid = batch.select_valid()
        self.module.update_normalizer(
            torch.cat((valid.vector_obs_t, valid.vector_obs_tp1), dim=0),
            torch.cat((valid.tactile_history_t, valid.tactile_history_tp1), dim=0),
        )
        self.module.update_action_normalizer(valid.applied_action_policy_units_t)
        excluded = int((~batch.transition_valid).sum().item())
        self.post_reset_pairs_excluded += excluded
        return {
            "valid_transitions": int(valid.transition_valid.numel()),
            "post_reset_pairs_excluded": excluded,
        }

    @torch.no_grad()
    def discovery_signal(self, batch: ICMTransitionBatch) -> torch.Tensor:
        """Score novelty before the predictor learns from this transition batch."""

        valid = batch.select_valid()
        output = self.module.transition(
            valid.vector_obs_t,
            valid.tactile_history_t,
            valid.applied_action_policy_units_t,
            valid.vector_obs_tp1,
            valid.tactile_history_tp1,
        )
        rewards = torch.zeros(
            batch.transition_valid.shape[0],
            dtype=output["intrinsic_reward"].dtype,
            device=output["intrinsic_reward"].device,
        )
        rewards[batch.transition_valid] = output["intrinsic_reward"]
        self.transitions_scored += int(valid.transition_valid.numel())
        return rewards.detach()

    def update(self, batch: ICMTransitionBatch) -> dict[str, float | int]:
        """Train inverse/forward models after the pre-update bonus is recorded."""

        valid = batch.select_valid()
        # Moments are learned from observations, not from outcomes. Updating
        # them here makes each rollout's bonus depend only on prior knowledge.
        self.module.update_normalizer(
            torch.cat((valid.vector_obs_t, valid.vector_obs_tp1), dim=0),
            torch.cat((valid.tactile_history_t, valid.tactile_history_tp1), dim=0),
        )
        self.module.update_action_normalizer(valid.applied_action_policy_units_t)
        output = self.module.transition(
            valid.vector_obs_t,
            valid.tactile_history_t,
            valid.applied_action_policy_units_t,
            valid.vector_obs_tp1,
            valid.tactile_history_tp1,
        )
        self.optimizer.zero_grad(set_to_none=True)
        output["predictor_loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.module.predictor_parameters(), self.gradient_norm_clip
        )
        self.optimizer.step()

        valid_count = int(valid.transition_valid.numel())
        excluded = int((~batch.transition_valid).sum().item())
        self.optimizer_updates += 1
        self.transitions_trained += valid_count
        self.post_reset_pairs_excluded += excluded
        return {
            "icm_predictor_loss": float(output["predictor_loss"].detach()),
            "icm_inverse_loss": float(output["inverse_loss"].detach()),
            "icm_forward_loss": float(output["forward_loss"].detach()),
            "icm_forward_error_mean": float(
                output["forward_error_per_transition"].mean()
            ),
            "icm_forward_error_std": float(
                output["forward_error_per_transition"].std(unbiased=False)
            ),
            "icm_gradient_norm_before_clip": float(gradient_norm),
            "icm_valid_transitions": valid_count,
            "icm_post_reset_pairs_excluded": excluded,
            "icm_optimizer_updates": self.optimizer_updates,
            "icm_transitions_trained": self.transitions_trained,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "protocol": "sugar_original_icm_continuous_trainer_v2",
            "config": self.cfg.to_dict(),
            "module_state_dict": self.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "gradient_norm_clip": self.gradient_norm_clip,
            "optimizer_updates": self.optimizer_updates,
            "transitions_trained": self.transitions_trained,
            "transitions_scored": self.transitions_scored,
            "post_reset_pairs_excluded": self.post_reset_pairs_excluded,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("protocol") != "sugar_original_icm_continuous_trainer_v2":
            raise ValueError("unexpected ICM trainer checkpoint protocol")
        if state.get("config") != self.cfg.to_dict():
            raise ValueError("ICM checkpoint config/schema drift")
        if state.get("learning_rate") != self.learning_rate:
            raise ValueError("ICM checkpoint learning-rate drift")
        if state.get("gradient_norm_clip") != self.gradient_norm_clip:
            raise ValueError("ICM checkpoint gradient-clip drift")
        self.module.load_state_dict(state["module_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.optimizer_updates = int(state["optimizer_updates"])
        self.transitions_trained = int(state["transitions_trained"])
        self.transitions_scored = int(state["transitions_scored"])
        self.post_reset_pairs_excluded = int(state["post_reset_pairs_excluded"])
