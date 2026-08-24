# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Online runtime for the frozen causal trajectory/contact/event predictor."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

from sugar_rl.utils.demo_event_reward_potential import (
    DEFAULT_EVENT_WEIGHTS,
    calibrated_event_risk,
    compatibility_potential,
    event_internal_reward,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
MODEL_SOURCE = (
    WORKSPACE_ROOT
    / "scripts/sugar/demo_reward/demo_conditioned_causal_predictor_v1.py"
)
GOAL_POLICY_CORE_DIM = 121
GOAL_POLICY_CORE_TERM_NAMES = (
    "projected_gravity",
    "base_height",
    "base_linear_velocity_body",
    "base_angular_velocity_body",
    "joint_position_relative",
    "joint_velocity",
    "previous_applied_action_policy_units",
    "box_position_body",
    "box_orientation_tangent_normal_body",
    "box_linear_velocity_body",
    "box_angular_velocity_body",
    "goal_position_body",
    "goal_orientation_tangent_normal_body",
)


def extract_goal_policy_core(
    policy_observation: torch.Tensor,
    active_term_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Extract the exact non-tactile prefix exposed by the goal-policy task."""

    if policy_observation.ndim != 2 or policy_observation.shape[1] < GOAL_POLICY_CORE_DIM:
        raise ValueError("goal-policy observation is narrower than 121 dimensions")
    if active_term_names is not None and tuple(active_term_names[:13]) != GOAL_POLICY_CORE_TERM_NAMES:
        raise ValueError("goal-policy observation term order does not match the predictor corpus")
    return policy_observation[:, :GOAL_POLICY_CORE_DIM]


def _load_model_module():
    name = "_curiosity_frozen_demo_event_predictor"
    spec = importlib.util.spec_from_file_location(name, MODEL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FrozenDemoEventRewardCfg:
    dataset_root: str
    predictor_dir: str
    selected_task: str
    selected_motion_id: int
    compatibility_baseline: float
    eta: float = 1.0
    uncertainty_beta: float = 1.0
    reward_clip: float = 1.0
    per_target_risk_clip: float = 5.0
    target_weights: tuple[float, ...] = DEFAULT_EVENT_WEIGHTS


@dataclass
class DemoEventRewardSignals:
    reward: torch.Tensor
    unit_eta_reward: torch.Tensor
    next_potential: torch.Tensor
    next_risk: torch.Tensor
    next_weighted_uncertainty: torch.Tensor
    next_ready: torch.Tensor
    done: torch.Tensor
    failure_done: torch.Tensor
    selected_demo_phase: torch.Tensor


class FrozenDemoEventReward:
    """Maintain a reset-safe causal prefix and score one fixed numeric demo."""

    history_steps = 10

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        cfg: FrozenDemoEventRewardCfg,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self.dataset_root = Path(cfg.dataset_root).expanduser().resolve()
        self.predictor_dir = Path(cfg.predictor_dir).expanduser().resolve()
        self._load_frozen_model()
        self._load_selected_demo()
        self.policy_prefix = torch.zeros(
            self.num_envs,
            self.history_steps,
            self.policy_dim,
            device=self.device,
        )
        self.valid_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.started = False

    def _load_frozen_model(self) -> None:
        training = json.loads(
            (self.predictor_dir / "RESULT.json").read_text(encoding="utf-8")
        )
        calibration = json.loads(
            (self.predictor_dir / "CALIBRATION_RESULT.json").read_text(
                encoding="utf-8"
            )
        )
        if training.get("passed") is not True or calibration.get("passed") is not True:
            raise RuntimeError("predictor training and calibration must both pass")
        manifest = json.loads(
            (self.dataset_root / "MANIFEST.json").read_text(encoding="utf-8")
        )
        if manifest.get("alignment_mode") != "clock_phase":
            raise RuntimeError("runtime rejects the free-window alignment loophole")
        with np.load(self.dataset_root / "NORMALIZATION.npz", allow_pickle=False) as archive:
            statistics = {
                name: torch.from_numpy(np.asarray(archive[name], dtype=np.float32))
                for name in archive.files
            }
        self.policy_dim = int(statistics["state_mean"].numel())
        if self.policy_dim != GOAL_POLICY_CORE_DIM:
            raise RuntimeError("deployable event reward requires the 121-D goal-policy core")
        module = _load_model_module()
        self.model = module.DemoConditionedCausalEventPredictorV3(
            policy_dim=self.policy_dim,
            policy_history_steps=self.history_steps,
            demo_windows=32,
            demo_window_steps=10,
            demo_feature_dim=132,
            d_model=384,
            nhead=8,
            num_layers=6,
            dim_feedforward=1536,
            dropout=0.1,
            state_mean=statistics["state_mean"],
            state_std=statistics["state_std"],
            demo_mean=statistics["demo_mean"],
            demo_std=statistics["demo_std"],
            target_scale=statistics["target_scale"],
        ).to(self.device)
        checkpoint = torch.load(
            self.predictor_dir / "best.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval().requires_grad_(False)
        with np.load(
            self.predictor_dir / "UNCERTAINTY_CALIBRATION.npz",
            allow_pickle=False,
        ) as archive:
            multiplier = np.asarray(archive["variance_multiplier"], dtype=np.float32)
        self.variance_multiplier = torch.from_numpy(multiplier).to(self.device)

    def _load_selected_demo(self) -> None:
        if self.cfg.selected_task not in {"CarryBox", "KickBox"}:
            raise ValueError("selected_task must be CarryBox or KickBox")
        split = self.dataset_root / "train"
        with np.load(split / "routing.npz", allow_pickle=False) as routing:
            task = np.asarray(routing["demo_task"], dtype=np.int64)
            motion = np.asarray(routing["demo_source_motion_id"], dtype=np.int64)
        task_index = 0 if self.cfg.selected_task == "CarryBox" else 1
        rows = np.flatnonzero(
            (task == task_index) & (motion == int(self.cfg.selected_motion_id))
        )
        if rows.size != 1:
            raise ValueError("selected demo must identify exactly one train-split motion")
        bank = np.load(split / "demo_bank.npy", mmap_mode="r", allow_pickle=False)
        numeric = np.array(bank[int(rows[0])], dtype=np.float32, copy=True)
        if numeric.shape != (32, 10, 132) or not np.isfinite(numeric).all():
            raise RuntimeError("selected numeric demo has invalid geometry or values")
        self.selected_demo_row = int(rows[0])
        self.demo_condition = (
            torch.from_numpy(numeric)
            .to(self.device)
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1, -1)
        )

    def _validate_policy_core(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device=self.device, dtype=torch.float32)
        if value.shape != (self.num_envs, self.policy_dim):
            raise ValueError(
                f"policy core must have shape {(self.num_envs, self.policy_dim)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("policy core contains non-finite values")
        return value

    def _push(self, value: torch.Tensor, reset_mask: torch.Tensor) -> None:
        if reset_mask.shape != (self.num_envs,) or reset_mask.dtype is not torch.bool:
            raise ValueError("reset mask geometry drift")
        if reset_mask.any():
            self.policy_prefix[reset_mask] = 0.0
            self.valid_count[reset_mask] = 0
        self.policy_prefix[:, :-1] = self.policy_prefix[:, 1:].clone()
        self.policy_prefix[:, -1] = value
        self.valid_count = torch.clamp(self.valid_count + 1, max=self.history_steps)

    @torch.no_grad()
    def _predict(self, selected_demo_phase: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.model(
            policy_prefix=self.policy_prefix,
            selected_demo_condition=self.demo_condition,
            selected_demo_phase=selected_demo_phase,
        )
        risk = calibrated_event_risk(
            output["mean_log1p_scaled"],
            output["log_variance_log1p_scaled"],
            self.variance_multiplier,
            uncertainty_beta=self.cfg.uncertainty_beta,
            target_weights=self.cfg.target_weights,
            per_target_risk_clip=self.cfg.per_target_risk_clip,
        )
        risk["potential"] = compatibility_potential(risk["risk"])
        return risk

    @torch.no_grad()
    def begin(self, policy_core: torch.Tensor) -> None:
        if self.started:
            raise RuntimeError("demo event reward already started")
        value = self._validate_policy_core(policy_core)
        reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._push(value, reset)
        self.started = True

    @torch.no_grad()
    def process_step(
        self,
        policy_core_tp1: torch.Tensor,
        selected_demo_phase_tp1: torch.Tensor,
        done: torch.Tensor,
        failure_done: torch.Tensor,
    ) -> DemoEventRewardSignals:
        if not self.started:
            raise RuntimeError("call begin() before process_step()")
        done = done.to(device=self.device, dtype=torch.bool).reshape(-1)
        failure_done = failure_done.to(device=self.device, dtype=torch.bool).reshape(-1)
        if done.shape != (self.num_envs,) or failure_done.shape != (self.num_envs,):
            raise ValueError("terminal masks must have one value per environment")
        selected_demo_phase_tp1 = selected_demo_phase_tp1.to(
            device=self.device, dtype=torch.float32
        ).reshape(-1)
        if (
            selected_demo_phase_tp1.shape != (self.num_envs,)
            or not torch.isfinite(selected_demo_phase_tp1).all()
            or torch.any(selected_demo_phase_tp1 < 0)
            or torch.any(selected_demo_phase_tp1 > 1)
        ):
            raise ValueError("selected demo phase must be finite [env] values in [0,1]")
        self._push(self._validate_policy_core(policy_core_tp1), done)
        next_ready = self.valid_count == self.history_steps
        if next_ready.any():
            prediction = self._predict(selected_demo_phase_tp1)
        else:
            zero = torch.zeros(self.num_envs, device=self.device)
            prediction = {
                "potential": zero,
                "risk": zero,
                "weighted_uncertainty": zero,
            }
        next_potential = torch.where(
            next_ready,
            prediction["potential"],
            torch.zeros_like(prediction["potential"]),
        )
        reward = event_internal_reward(
            next_potential,
            next_ready,
            done,
            failure_done,
            compatibility_baseline=self.cfg.compatibility_baseline,
            eta=self.cfg.eta,
            reward_clip=self.cfg.reward_clip,
        )
        if self.cfg.eta <= 0.0:
            raise ValueError("event reward eta must be positive")
        unit_reward = reward / float(self.cfg.eta)
        return DemoEventRewardSignals(
            reward=reward,
            unit_eta_reward=unit_reward,
            next_potential=next_potential,
            next_risk=torch.where(
                next_ready, prediction["risk"], torch.zeros_like(prediction["risk"])
            ),
            next_weighted_uncertainty=torch.where(
                next_ready,
                prediction["weighted_uncertainty"],
                torch.zeros_like(prediction["weighted_uncertainty"]),
            ),
            next_ready=next_ready,
            done=done,
            failure_done=failure_done,
            selected_demo_phase=selected_demo_phase_tp1,
        )

    def audit(self) -> dict[str, Any]:
        return {
            "model_training": self.model.training,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            "policy_dim": self.policy_dim,
            "selected_task": self.cfg.selected_task,
            "selected_motion_id": int(self.cfg.selected_motion_id),
            "selected_demo_row": self.selected_demo_row,
            "history_steps": self.history_steps,
            "alignment_mode": "clock_phase",
        }


@dataclass(frozen=True)
class FrozenPhaseAwareDemoEventScorerCfg:
    runtime_config_path: str
    selected_option: str
    phase_horizon_steps: int = 650


class FrozenPhaseAwareDemoEventScorer:
    """Adapt the frozen event runtime to SUGAR rollout-boundary integration."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        cfg: FrozenPhaseAwareDemoEventScorerCfg,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        if cfg.phase_horizon_steps <= self.history_steps:
            raise ValueError("phase horizon must exceed the causal history")
        config_path = Path(cfg.runtime_config_path).expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("protocol") != "sugar_dense_demo_event_feedback_runtime_v1":
            raise ValueError("unexpected demo-event runtime protocol")
        if payload.get("potential_difference_shaping_used") is not False:
            raise ValueError("phase event reward must remain dense compatibility feedback")
        options = payload.get("selected_demo_options", {})
        if cfg.selected_option not in options:
            raise ValueError("selected demo option is not declared by the runtime config")
        selected = options[cfg.selected_option]
        event_cfg = FrozenDemoEventRewardCfg(
            dataset_root=str(payload["dataset_root"]),
            predictor_dir=str(payload["predictor_dir"]),
            selected_task=str(selected["selected_task"]),
            selected_motion_id=int(selected["selected_motion_id"]),
            compatibility_baseline=float(payload["compatibility_baseline"]),
            eta=float(payload["eta"]),
            uncertainty_beta=float(payload["uncertainty_beta"]),
            reward_clip=float(payload["reward_clip"]),
            per_target_risk_clip=float(payload["per_target_risk_clip"]),
            target_weights=tuple(float(value) for value in payload["target_weights"]),
        )
        self.runtime_config_path = config_path
        self.runtime = FrozenDemoEventReward(
            num_envs=self.num_envs,
            device=self.device,
            cfg=event_cfg,
        )
        self.episode_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.transitions_scored = 0
        self.started = False

    @property
    def history_steps(self) -> int:
        return FrozenDemoEventReward.history_steps

    def _core(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "policy" not in observation:
            raise KeyError("goal-policy observation is missing")
        return extract_goal_policy_core(observation["policy"])

    @torch.no_grad()
    def begin(
        self,
        observation: Mapping[str, torch.Tensor],
        *,
        initial_episode_steps: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("phase-aware demo event scorer already started")
        self.runtime.begin(self._core(observation))
        if initial_episode_steps is None:
            self.episode_steps.zero_()
        else:
            initial_episode_steps = initial_episode_steps.to(
                device=self.device,
                dtype=torch.long,
            ).reshape(-1)
            if (
                initial_episode_steps.shape != (self.num_envs,)
                or torch.any(initial_episode_steps < 0)
                or torch.any(
                    initial_episode_steps >= self.cfg.phase_horizon_steps
                )
            ):
                raise ValueError(
                    "initial demo-event phase steps must be one bounded value "
                    "per environment"
                )
            self.episode_steps.copy_(initial_episode_steps)
        self.started = True
        return self.frozen_model_audit()

    @torch.no_grad()
    def process_step(
        self,
        observation_tp1: Mapping[str, torch.Tensor],
        dones: torch.Tensor,
    ) -> DemoEventRewardSignals:
        if not self.started:
            raise RuntimeError("call begin() before phase-aware event scoring")
        dones = dones.to(device=self.device, dtype=torch.bool).reshape(-1)
        if dones.shape != (self.num_envs,):
            raise ValueError("done mask geometry drift")
        self.episode_steps += 1
        phase = torch.clamp(
            (self.episode_steps.float() + 1.0) / float(self.cfg.phase_horizon_steps),
            min=0.0,
            max=1.0,
        )
        phase = torch.where(
            dones,
            torch.full_like(phase, 1.0 / float(self.cfg.phase_horizon_steps)),
            phase,
        )
        signals = self.runtime.process_step(
            self._core(observation_tp1),
            phase,
            dones,
            torch.zeros_like(dones),
        )
        self.episode_steps[dones] = 0
        self.transitions_scored += self.num_envs
        return signals

    def frozen_model_audit(self) -> dict[str, Any]:
        audit = self.runtime.audit()
        return {
            **audit,
            "model_frozen": (
                audit["model_training"] is False
                and int(audit["trainable_parameters"]) == 0
            ),
            "transitions_scored": self.transitions_scored,
            "selected_option": self.cfg.selected_option,
            "phase_horizon_steps": int(self.cfg.phase_horizon_steps),
            "phase_source": (
                "reset_reference_frame_plus_causal_control_clock"
            ),
            "future_actual_events_used": False,
        }

    def state_dict(self) -> dict[str, Any]:
        if not self.started:
            raise RuntimeError("cannot checkpoint an unstarted event scorer")
        return {
            "protocol": "sugar_phase_aware_demo_event_scorer_state_v1",
            "selected_option": self.cfg.selected_option,
            "phase_horizon_steps": int(self.cfg.phase_horizon_steps),
            "policy_prefix": self.runtime.policy_prefix.detach().clone(),
            "valid_count": self.runtime.valid_count.detach().clone(),
            "episode_steps": self.episode_steps.detach().clone(),
            "transitions_scored": int(self.transitions_scored),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("protocol") != "sugar_phase_aware_demo_event_scorer_state_v1":
            raise ValueError("unexpected phase-aware event scorer checkpoint")
        if (
            state.get("selected_option") != self.cfg.selected_option
            or int(state.get("phase_horizon_steps", -1))
            != int(self.cfg.phase_horizon_steps)
        ):
            raise ValueError("phase-aware scorer checkpoint config drift")
        for name, destination in (
            ("policy_prefix", self.runtime.policy_prefix),
            ("valid_count", self.runtime.valid_count),
            ("episode_steps", self.episode_steps),
        ):
            source = state[name].to(device=destination.device, dtype=destination.dtype)
            if source.shape != destination.shape:
                raise ValueError(f"phase-aware scorer checkpoint {name} shape drift")
            destination.copy_(source)
        self.transitions_scored = int(state["transitions_scored"])
        self.runtime.started = True
        self.started = True


@dataclass
class DemoEventRewardAugmentedStepSignals:
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
    demo_reward: torch.Tensor
    demo_unit_eta_reward: torch.Tensor
    demo_event_potential: torch.Tensor
    demo_event_risk: torch.Tensor
    demo_event_uncertainty: torch.Tensor
    demo_event_ready: torch.Tensor
    demo_event_phase: torch.Tensor


class DemoEventRewardAugmentedSMPICMRolloutIntegrator:
    """Add phase-aware dense demo feedback after unchanged SMP/ICM scoring."""

    def __init__(self, *, base: Any, demo: FrozenPhaseAwareDemoEventScorer) -> None:
        if base.device != demo.device or base.env.num_envs != demo.num_envs:
            raise ValueError("base and demo-event integrations differ")
        self.base = base
        self.demo = demo
        self.last_base_signals: Any | None = None

    @property
    def at_rollout_boundary(self) -> bool:
        return self.base.at_rollout_boundary

    def begin(
        self,
        observation: Mapping[str, torch.Tensor],
        *,
        initial_episode_steps: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return {
            "smp_window": self.base.begin(),
            "demo_event": self.demo.begin(
                observation,
                initial_episode_steps=initial_episode_steps,
            ),
        }

    @torch.no_grad()
    def process_step(
        self,
        *,
        observation_t: Mapping[str, torch.Tensor],
        applied_action_policy_units_t: torch.Tensor,
        observation_tp1: Mapping[str, torch.Tensor],
        external_reward: torch.Tensor,
        dones: torch.Tensor,
    ) -> DemoEventRewardAugmentedStepSignals:
        base_signals = self.base.process_step(
            observation_t=observation_t,
            applied_action_policy_units_t=applied_action_policy_units_t,
            observation_tp1=observation_tp1,
            external_reward=external_reward,
            dones=dones,
        )
        self.last_base_signals = base_signals
        event = self.demo.process_step(observation_tp1, dones)
        policy_reward = base_signals.policy_reward + event.reward
        if not torch.isfinite(policy_reward).all():
            raise RuntimeError("non-finite phase-aware demo-event policy reward")
        values = dict(vars(base_signals))
        values["policy_reward"] = policy_reward.detach().clone()
        return DemoEventRewardAugmentedStepSignals(
            **values,
            demo_reward=event.reward,
            demo_unit_eta_reward=event.unit_eta_reward,
            demo_event_potential=event.next_potential,
            demo_event_risk=event.next_risk,
            demo_event_uncertainty=event.next_weighted_uncertainty,
            demo_event_ready=event.next_ready,
            demo_event_phase=event.selected_demo_phase,
        )

    def finish_rollout(self) -> dict[str, Any]:
        return {
            **self.base.finish_rollout(),
            "demo_event_predictor_updated": False,
            "demo_event_model_frozen": self.demo.frozen_model_audit()["model_frozen"],
        }

    def state_dict(self) -> dict[str, Any]:
        if not self.at_rollout_boundary:
            raise RuntimeError("checkpoint event integration at rollout boundary")
        return {
            "protocol": "sugar_demo_event_reward_augmented_smp_icm_v1",
            "base": self.base.state_dict(),
            "demo_event": self.demo.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("protocol") != "sugar_demo_event_reward_augmented_smp_icm_v1":
            raise ValueError("unexpected demo-event integration checkpoint")
        self.base.load_state_dict(state["base"])
        self.demo.load_state_dict(state["demo_event"])
