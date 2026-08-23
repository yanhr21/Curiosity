# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Online runtime for the frozen causal trajectory/contact/event predictor."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

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
        unit_reward = event_internal_reward(
            next_potential,
            next_ready,
            done,
            failure_done,
            compatibility_baseline=self.cfg.compatibility_baseline,
            eta=1.0,
            reward_clip=self.cfg.reward_clip,
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
