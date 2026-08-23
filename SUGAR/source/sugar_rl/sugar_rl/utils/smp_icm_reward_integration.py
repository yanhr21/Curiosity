# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout-boundary integration for frozen SMP and independently learned ICM.

This module is deliberately policy-optimizer agnostic.  PPO/A3C may consume
the returned policy reward, but neither optimizer defines curiosity.  The ICM
input API contains only aligned ``(o_t, a_t, o_t+1)`` transitions and the ICM
bonus is scored before the predictor update.  Task outcomes, tactile slip,
failed-strategy memory, and safety remain separate external ledgers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from sugar_rl.utils.official_smp_scorer import (
    OfficialSMPScorerCfg,
    OfficialSugarSMPScorer,
)
from sugar_rl.utils.original_icm_continuous import OriginalICMContinuousCfg
from sugar_rl.utils.original_icm_trainer import (
    ICMTransitionBatch,
    OriginalICMTrainer,
)
from sugar_rl.utils.sugar_smp_feature_window import SugarSMPFeatureWindowBuffer


OUTCOME_REWARD_TERMS = (
    "goal_position",
    "goal_orientation",
    "lift_fraction",
    "goal_stability",
)
EXTERNAL_CONSTRAINT_TERMS = (
    "tactile_slip",
    "repeated_failed_strategy",
    "joint_acc",
    "joint_torque",
    "action_rate_l2",
    "joint_limit",
    "feet_slide",
    "undesired_contacts",
)


@dataclass(frozen=True)
class SMPICMRewardMixCfg:
    """Predeclared task/prior/discovery/constraint reward mix.

    Task outcomes and external constraints are reconstructed from disjoint
    manager-reward terms before they are mixed with SMP and ICM.  Pure
    discovery keeps ``task_outcome_weight`` at zero and requires the task
    terms themselves to be zero.  Goal-recovery training explicitly enables
    the task component without changing ICM's inputs, target, update order, or
    outcome independence.
    """

    task_outcome_weight: float = 0.0
    smp_reward_weight: float = 0.5
    icm_reward_weight: float = 1.0
    external_constraint_weight: float = 1.0
    require_zero_outcome_rewards: bool = True
    require_no_success_termination: bool = True

    def __post_init__(self) -> None:
        if self.task_outcome_weight < 0.0:
            raise ValueError("task outcome weight must be non-negative")
        if self.smp_reward_weight < 0.0:
            raise ValueError("SMP reward weight must be non-negative")
        if self.icm_reward_weight < 0.0:
            raise ValueError("ICM reward weight must be non-negative")
        if self.external_constraint_weight < 0.0:
            raise ValueError("external constraint weight must be non-negative")


@dataclass
class SMPICMStepSignals:
    """Separate tensors for one environment step."""

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


def _reshape_tactile(observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
    tactile = observation["tactile_history"]
    if tactile.ndim != 2 or tactile.shape[-1] != 4 * 2 * 3 * 20 * 25:
        raise ValueError(
            "Stage-H tactile observation must be flattened "
            f"(batch,{4 * 2 * 3 * 20 * 25}), got {tuple(tactile.shape)}"
        )
    return tactile.reshape(-1, 4, 2, 3, 20, 25)


def _clone_batch(batch: ICMTransitionBatch) -> ICMTransitionBatch:
    return ICMTransitionBatch(
        vector_obs_t=batch.vector_obs_t.detach().clone(),
        tactile_history_t=batch.tactile_history_t.detach().clone(),
        applied_action_policy_units_t=(
            batch.applied_action_policy_units_t.detach().clone()
        ),
        vector_obs_tp1=batch.vector_obs_tp1.detach().clone(),
        tactile_history_tp1=batch.tactile_history_tp1.detach().clone(),
        transition_valid=batch.transition_valid.detach().clone(),
    )


def _concat_batches(batches: list[ICMTransitionBatch]) -> ICMTransitionBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty ICM rollout")
    return ICMTransitionBatch(
        vector_obs_t=torch.cat([batch.vector_obs_t for batch in batches], dim=0),
        tactile_history_t=torch.cat(
            [batch.tactile_history_t for batch in batches], dim=0
        ),
        applied_action_policy_units_t=torch.cat(
            [batch.applied_action_policy_units_t for batch in batches], dim=0
        ),
        vector_obs_tp1=torch.cat(
            [batch.vector_obs_tp1 for batch in batches], dim=0
        ),
        tactile_history_tp1=torch.cat(
            [batch.tactile_history_tp1 for batch in batches], dim=0
        ),
        transition_valid=torch.cat(
            [batch.transition_valid for batch in batches], dim=0
        ),
    )


class SMPICMRolloutIntegrator:
    """Score and learn Stage-H signals at auditable rollout boundaries."""

    def __init__(
        self,
        env,
        prior_dir: str,
        mix_cfg: SMPICMRewardMixCfg = SMPICMRewardMixCfg(),
        smp_cfg: OfficialSMPScorerCfg = OfficialSMPScorerCfg(),
        icm_cfg: OriginalICMContinuousCfg = OriginalICMContinuousCfg(),
    ) -> None:
        self.env = env
        self.device = torch.device(env.device)
        if self.device.type != "cuda":
            raise ValueError("Stage-H integration must run on a compute GPU")
        self.mix_cfg = mix_cfg
        self.smp_cfg = smp_cfg
        self.icm_cfg = icm_cfg
        self._audit_environment_reward_boundary()

        self.smp_window = SugarSMPFeatureWindowBuffer(env)
        self.smp_scorer = OfficialSugarSMPScorer(
            prior_dir=prior_dir,
            device=self.device,
            cfg=smp_cfg,
        )
        self.icm_trainer = OriginalICMTrainer(icm_cfg, device=self.device)
        self._rollout_batches: list[ICMTransitionBatch] = []
        self.rollouts_completed = 0
        self.policy_transitions_scored = 0
        self.icm_bootstrap_steps = 0
        self._started = False

    def _audit_environment_reward_boundary(self) -> None:
        manager = self.env.reward_manager
        active = set(manager.active_terms)
        classified = set(OUTCOME_REWARD_TERMS) | set(EXTERNAL_CONSTRAINT_TERMS)
        unclassified = sorted(active - classified)
        if unclassified:
            raise RuntimeError(
                f"reward terms are not assigned to a separate ledger: {unclassified}"
            )
        missing_outcomes = sorted(set(OUTCOME_REWARD_TERMS) - active)
        if missing_outcomes:
            raise RuntimeError(
                f"environment lacks declared task outcomes {missing_outcomes}"
            )
        missing_constraints = sorted(set(EXTERNAL_CONSTRAINT_TERMS) - active)
        if missing_constraints:
            raise RuntimeError(
                f"Stage-H environment lacks external constraints {missing_constraints}"
            )
        if self.mix_cfg.require_zero_outcome_rewards:
            nonzero_outcomes = {
                name: float(manager.get_term_cfg(name).weight)
                for name in OUTCOME_REWARD_TERMS
                if name in active and manager.get_term_cfg(name).weight != 0.0
            }
            if nonzero_outcomes:
                raise RuntimeError(
                    "pure-discovery policy reward still contains task outcomes: "
                    f"{nonzero_outcomes}"
                )
        if self.mix_cfg.require_no_success_termination:
            termination_names = set(self.env.termination_manager.active_terms)
            if "success" in termination_names:
                raise RuntimeError(
                    "pure-discovery phase must not terminate on task success"
                )

    @property
    def at_rollout_boundary(self) -> bool:
        return not self._rollout_batches

    def _icm_normalizer_ready(self) -> bool:
        module = self.icm_trainer.module
        return bool(
            module.normalizer.vector_count.item() > 0
            and module.normalizer.tactile_count.item() > 0
            and module.action_normalizer.count.item() > 0
        )

    @torch.no_grad()
    def begin(self) -> torch.Tensor:
        """Initialize the live SMP window at the current reset state."""

        if self._started:
            raise RuntimeError("Stage-H rollout integrator was already started")
        features = self.smp_window.update().detach().clone()
        self._started = True
        return features

    def _reward_term_ledger(
        self, external_reward: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        manager = self.env.reward_manager
        dt = float(self.env.step_dt)
        ledger = {
            name: manager._step_reward[:, index].detach().clone() * dt
            for index, name in enumerate(manager.active_terms)
        }
        reconstructed = torch.stack(tuple(ledger.values()), dim=0).sum(dim=0)
        if not torch.allclose(
            reconstructed,
            external_reward,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            maximum = float(torch.abs(reconstructed - external_reward).max())
            raise RuntimeError(
                "external reward does not equal the per-term ledger; "
                f"max_abs={maximum}"
            )
        return ledger

    def _transition(
        self,
        observation_t: Mapping[str, torch.Tensor],
        applied_action_policy_units_t: torch.Tensor,
        observation_tp1: Mapping[str, torch.Tensor],
        transition_valid: torch.Tensor,
    ) -> ICMTransitionBatch:
        return ICMTransitionBatch(
            vector_obs_t=observation_t["icm_vector"],
            tactile_history_t=_reshape_tactile(observation_t),
            applied_action_policy_units_t=applied_action_policy_units_t,
            vector_obs_tp1=observation_tp1["icm_vector"],
            tactile_history_tp1=_reshape_tactile(observation_tp1),
            transition_valid=transition_valid,
        )

    @torch.no_grad()
    def process_step(
        self,
        observation_t: Mapping[str, torch.Tensor],
        applied_action_policy_units_t: torch.Tensor,
        observation_tp1: Mapping[str, torch.Tensor],
        external_reward: torch.Tensor,
        dones: torch.Tensor,
    ) -> SMPICMStepSignals:
        """Score one transition before any ICM predictor update."""

        if not self._started:
            raise RuntimeError("call begin() before processing Stage-H transitions")
        dones = dones.to(device=self.device, dtype=torch.bool).reshape(-1)
        external_reward = external_reward.to(self.device).reshape(-1)
        transition_valid = ~dones
        batch = self._transition(
            observation_t,
            applied_action_policy_units_t,
            observation_tp1,
            transition_valid,
        )
        if not all(
            tensor.device == self.device
            for tensor in (
                batch.vector_obs_t,
                batch.tactile_history_t,
                batch.applied_action_policy_units_t,
                batch.vector_obs_tp1,
                batch.tactile_history_tp1,
                batch.transition_valid,
            )
        ):
            raise ValueError("Stage-H transition tensors must stay on the environment GPU")

        smp_output = self.smp_scorer.score(
            self.smp_window.update(), record_normalizer=True
        )
        # IsaacLab returns a reset observation for automatic-reset transitions.
        # Do not attribute the new episode's SMP window to the terminal action.
        smp_reward = smp_output["smp_reward"].detach().clone()
        smp_reward[~transition_valid] = 0.0
        raw_sds_mean = smp_output["raw_sds_mean"].detach().clone()
        raw_sds_by_step = smp_output["raw_sds_by_step"].detach().clone()

        bootstrapped = False
        if transition_valid.any():
            if not self._icm_normalizer_ready():
                self.icm_trainer.bootstrap_normalizer(batch)
                icm_discovery = torch.zeros_like(external_reward)
                self.icm_bootstrap_steps += 1
                bootstrapped = True
            else:
                # This is the actual independently learned curiosity signal:
                # pre-update forward prediction error in inverse-model features.
                icm_discovery = self.icm_trainer.discovery_signal(batch)
        else:
            icm_discovery = torch.zeros_like(external_reward)

        reward_terms = self._reward_term_ledger(external_reward)
        task_outcome_reward = torch.stack(
            tuple(reward_terms[name] for name in OUTCOME_REWARD_TERMS),
            dim=0,
        ).sum(dim=0)
        external_constraint_reward = torch.stack(
            tuple(reward_terms[name] for name in EXTERNAL_CONSTRAINT_TERMS),
            dim=0,
        ).sum(dim=0)
        if not torch.allclose(
            task_outcome_reward + external_constraint_reward,
            external_reward,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            maximum = float(
                torch.abs(
                    task_outcome_reward
                    + external_constraint_reward
                    - external_reward
                ).max()
            )
            raise RuntimeError(
                "task and constraint ledgers do not reconstruct manager reward; "
                f"max_abs={maximum}"
            )
        policy_reward = (
            self.mix_cfg.task_outcome_weight * task_outcome_reward
            + self.mix_cfg.external_constraint_weight
            * external_constraint_reward
            + self.mix_cfg.smp_reward_weight * smp_reward
            + self.mix_cfg.icm_reward_weight * icm_discovery
        )
        if not torch.isfinite(policy_reward).all():
            raise RuntimeError("non-finite Stage-H policy reward")

        self._rollout_batches.append(_clone_batch(batch))
        self.policy_transitions_scored += int(transition_valid.sum().item())
        return SMPICMStepSignals(
            policy_reward=policy_reward,
            task_outcome_reward=task_outcome_reward.detach().clone(),
            external_constraint_reward=(
                external_constraint_reward.detach().clone()
            ),
            smp_reward=smp_reward,
            icm_discovery_reward=icm_discovery.detach().clone(),
            smp_raw_sds_mean=raw_sds_mean,
            smp_raw_sds_by_step=raw_sds_by_step,
            reward_terms=reward_terms,
            transition_valid=transition_valid.detach().clone(),
            icm_normalizer_bootstrap=bootstrapped,
        )

    def finish_rollout(self) -> dict[str, float | int | bool]:
        """Update ICM and the SDS normalizer only after all bonuses are fixed."""

        if not self._rollout_batches:
            raise RuntimeError("cannot finish an empty Stage-H rollout")
        batch = _concat_batches(self._rollout_batches)
        valid_count = int(batch.transition_valid.sum().item())
        if valid_count > 0:
            icm_metrics = self.icm_trainer.update(batch)
        else:
            icm_metrics = {
                "icm_update_skipped_all_transitions_invalid": True,
                "icm_valid_transitions": 0,
            }
        self.smp_scorer.commit_rollout_normalizer()
        self._rollout_batches.clear()
        self.rollouts_completed += 1
        return {
            **icm_metrics,
            "rollouts_completed": self.rollouts_completed,
            "rollout_valid_transitions": valid_count,
            "smp_normalizer_updates": self.smp_scorer.normalizer_updates,
            "icm_scored_before_update": True,
            "icm_has_no_outcome_input": True,
        }

    def state_dict(self) -> dict[str, Any]:
        if not self.at_rollout_boundary:
            raise RuntimeError("checkpoint Stage-H integration only at rollout boundary")
        return {
            "protocol": "sugar_smp_original_icm_rollout_integration_v1",
            "semantic_boundary": (
                "ICM is independent pre-update action-conditioned feature "
                "prediction error; PPO/A3C is only a possible consumer"
            ),
            "mix_config": asdict(self.mix_cfg),
            "smp_scorer": self.smp_scorer.state_dict(),
            "icm_trainer": self.icm_trainer.state_dict(),
            "rollouts_completed": self.rollouts_completed,
            "policy_transitions_scored": self.policy_transitions_scored,
            "icm_bootstrap_steps": self.icm_bootstrap_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not self.at_rollout_boundary:
            raise RuntimeError("load Stage-H integration only at rollout boundary")
        if state.get("protocol") != "sugar_smp_original_icm_rollout_integration_v1":
            raise ValueError("unexpected Stage-H integration checkpoint protocol")
        if state.get("mix_config") != asdict(self.mix_cfg):
            raise ValueError("Stage-H reward-mix checkpoint drift")
        self.smp_scorer.load_state_dict(state["smp_scorer"])
        self.icm_trainer.load_state_dict(state["icm_trainer"])
        self.rollouts_completed = int(state["rollouts_completed"])
        self.policy_transitions_scored = int(state["policy_transitions_scored"])
        self.icm_bootstrap_steps = int(state["icm_bootstrap_steps"])
