# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Separate paper-CWS reward ledger around unchanged SMP/original-ICM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from sugar_rl.utils.paper_cws_runtime_reward import (
    OfficialTacSLPaperCWSReward,
)


@dataclass
class PaperCWSAugmentedStepSignals:
    policy_reward: torch.Tensor
    task_outcome_reward: torch.Tensor
    external_constraint_reward: torch.Tensor
    smp_reward: torch.Tensor
    icm_discovery_reward: torch.Tensor
    smp_raw_sds_mean: torch.Tensor
    smp_raw_sds_by_step: torch.Tensor
    reward_terms: dict[str, torch.Tensor]
    transition_valid: torch.Tensor
    icm_normalizer_bootstrap: bool
    paper_cws_reward: torch.Tensor
    paper_cws_weighted_reward: torch.Tensor
    paper_cws_lower_violation_squared: torch.Tensor
    paper_cws_upper_violation_squared: torch.Tensor
    paper_cws_missed_contact: torch.Tensor
    paper_cws_unintended_contact: torch.Tensor
    paper_cws_reference_index: torch.Tensor
    paper_cws_reference_index_clamped: torch.Tensor
    paper_cws_reference_valid: torch.Tensor
    paper_cws_active_taxels: torch.Tensor


class PaperCWSAugmentedSMPICMRolloutIntegrator:
    """Add paper-CWS guidance without changing original ICM learning."""

    protocol = "paper_cws_augmented_smp_original_icm_integration_v3"

    def __init__(
        self,
        *,
        base: Any,
        paper_cws: OfficialTacSLPaperCWSReward,
        guidance_weight: float,
    ) -> None:
        if guidance_weight < 0.0:
            raise ValueError("paper-CWS guidance weight must be non-negative")
        if base.device != paper_cws.device:
            raise ValueError(
                "base and paper-CWS integrations must share one device"
            )
        if base.env is not paper_cws.env:
            raise ValueError(
                "base and paper-CWS integrations must share one environment"
            )
        self.base = base
        self.paper_cws = paper_cws
        self.guidance_weight = float(guidance_weight)
        self.last_base_signals: Any | None = None

    @property
    def at_rollout_boundary(self) -> bool:
        return self.base.at_rollout_boundary

    def begin(self) -> torch.Tensor:
        return self.base.begin()

    @torch.no_grad()
    def process_step(
        self,
        *,
        observation_t: Mapping[str, torch.Tensor],
        applied_action_policy_units_t: torch.Tensor,
        observation_tp1: Mapping[str, torch.Tensor],
        external_reward: torch.Tensor,
        dones: torch.Tensor,
    ) -> PaperCWSAugmentedStepSignals:
        base_signals = self.base.process_step(
            observation_t=observation_t,
            applied_action_policy_units_t=applied_action_policy_units_t,
            observation_tp1=observation_tp1,
            external_reward=external_reward,
            dones=dones,
        )
        self.last_base_signals = base_signals
        valid = base_signals.transition_valid
        cws = self.paper_cws.score(environment_valid=valid)
        reward = torch.where(
            valid, cws.reward, torch.zeros_like(cws.reward)
        )
        weighted = self.guidance_weight * reward
        policy_reward = base_signals.policy_reward + weighted
        if not torch.isfinite(policy_reward).all():
            raise RuntimeError("non-finite paper-CWS augmented policy reward")
        base_values = dict(vars(base_signals))
        base_values["policy_reward"] = policy_reward.detach().clone()
        return PaperCWSAugmentedStepSignals(
            **base_values,
            paper_cws_reward=reward.detach().clone(),
            paper_cws_weighted_reward=weighted.detach().clone(),
            paper_cws_lower_violation_squared=(
                cws.lower_violation_squared
            ),
            paper_cws_upper_violation_squared=(
                cws.upper_violation_squared
            ),
            paper_cws_missed_contact=cws.missed_contact,
            paper_cws_unintended_contact=cws.unintended_contact,
            paper_cws_reference_index=cws.reference_index,
            paper_cws_reference_index_clamped=(
                cws.reference_index_clamped
            ),
            paper_cws_reference_valid=cws.reference_valid,
            paper_cws_active_taxels=cws.active_taxels,
        )

    def finish_rollout(self) -> dict[str, Any]:
        metrics = self.base.finish_rollout()
        return {
            **metrics,
            "paper_cws_model_updated": False,
            "paper_cws_is_external_to_original_icm": True,
            "paper_cws_steps_scored": self.paper_cws.steps_scored,
        }

    def state_dict(self) -> dict[str, Any]:
        if not self.at_rollout_boundary:
            raise RuntimeError(
                "checkpoint paper-CWS integration at a rollout boundary"
            )
        return {
            "protocol": self.protocol,
            "base": self.base.state_dict(),
            "paper_cws": self.paper_cws.audit_state(),
            "guidance_weight": self.guidance_weight,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("protocol") != self.protocol:
            raise ValueError(
                "unexpected paper-CWS integration checkpoint protocol"
            )
        if float(state["guidance_weight"]) != self.guidance_weight:
            raise ValueError("paper-CWS guidance weight drift")
        declared = state["paper_cws"]
        current = self.paper_cws.audit_state()
        stable_keys = (
            "protocol",
            "status",
            "training_only_privileged_reward",
            "actor_receives_sdf_normals",
            "original_icm_receives_sdf_normals",
            "reference_arrays_path",
            "reference_arrays_sha256",
            "config",
            "paper_config",
            "reference_schedule_shape",
            "out_of_support_reward_policy",
            "invalid_transition_reward_policy",
        )
        if any(declared[key] != current[key] for key in stable_keys):
            raise ValueError("paper-CWS checkpoint/config binding drift")
        self.base.load_state_dict(state["base"])
        self.paper_cws.steps_scored = int(declared["steps_scored"])
        self.paper_cws.clamped_environment_steps = int(
            declared["clamped_environment_steps"]
        )
        self.paper_cws.invalid_environment_steps_masked = int(
            declared["invalid_environment_steps_masked"]
        )
        self.paper_cws.invalid_active_taxels_masked = int(
            declared["invalid_active_taxels_masked"]
        )
        self.paper_cws.invalid_active_taxels_with_missing_normals_masked = int(
            declared[
                "invalid_active_taxels_with_missing_normals_masked"
            ]
        )
