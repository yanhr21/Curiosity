# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen causal demo-predictor runtime and separate reward-ledger adapter.

This is glue around the hash-bound positive V2 predictor.  It does not define
or update ICM, does not change SMP, and does not infer slip from a proxy.  The
only failure input is the causal ``failure_closed`` bit already produced by
the separately audited direct-TacSL strategy runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

from sugar_rl.utils.demo_reward_potential import potential_difference_reward


WORKSPACE_ROOT = Path("/public/home/yanhongru/Curiosity")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class FrozenDemoRewardRuntimeCfg:
    config_path: str
    gamma: float
    eta: float
    policy_dim: int = 175
    policy_history_steps: int = 10
    failure_closed_policy_index: int = 174
    tactile_history_steps: int = 4
    tactile_num_hands: int = 2
    tactile_channels_per_hand: int = 3
    tactile_rows: int = 20
    tactile_cols: int = 25

    def __post_init__(self) -> None:
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("demo reward gamma must lie in (0,1]")
        if not np.isfinite(float(self.eta)) or float(self.eta) < 0.0:
            raise ValueError("demo reward eta must be finite and nonnegative")
        if self.policy_dim != 175 or self.policy_history_steps != 10:
            raise ValueError("V2 predictor requires the frozen 10x175 prefix")
        if self.failure_closed_policy_index != 174:
            raise ValueError("failure_closed must remain policy field 174")
        if (
            self.tactile_history_steps,
            self.tactile_num_hands,
            self.tactile_channels_per_hand,
            self.tactile_rows,
            self.tactile_cols,
        ) != (4, 2, 3, 20, 25):
            raise ValueError("V2 predictor requires frozen direct-TacSL shape")


@dataclass
class DemoRewardStepSignals:
    reward: torch.Tensor
    unit_eta_reward: torch.Tensor
    component_mse: torch.Tensor
    current_phi: torch.Tensor
    next_phi: torch.Tensor
    current_psi: torch.Tensor
    next_psi: torch.Tensor
    current_imitation_active: torch.Tensor
    next_imitation_active: torch.Tensor
    failure_closed_next: torch.Tensor
    transition_done: torch.Tensor


@dataclass
class DemoRewardAugmentedStepSignals:
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
    demo_component_mse: torch.Tensor
    demo_current_phi: torch.Tensor
    demo_next_phi: torch.Tensor
    demo_current_psi: torch.Tensor
    demo_next_psi: torch.Tensor
    demo_current_imitation_active: torch.Tensor
    demo_next_imitation_active: torch.Tensor
    demo_failure_closed_next: torch.Tensor


class FrozenDemoRewardScorer:
    """Causal prefix buffer plus frozen selected-demo V2 inference."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device | str,
        cfg: FrozenDemoRewardRuntimeCfg,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("demo predictor runtime requires the compute GPU")
        self.cfg = cfg
        self.config_path = _resolve(cfg.config_path)
        overlay = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.base_config_path: Path | None = None
        if "base_config" in overlay:
            self.base_config_path = _resolve(overlay["base_config"])
            base = json.loads(
                self.base_config_path.read_text(encoding="utf-8")
            )
            self.config = _deep_update(base, overlay)
        else:
            self.config = overlay
        if self.config.get("protocol") not in {
            "sugar_demo_reward_runtime_preflight_v1",
            "sugar_demo_reward_runtime_frozen_scale_v1",
        }:
            raise ValueError("unexpected demo runtime protocol")
        self._load_frozen_predictor()
        self.policy_prefix = torch.zeros(
            self.num_envs,
            cfg.policy_history_steps,
            cfg.policy_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.prefix_valid_mask = torch.zeros(
            self.num_envs,
            cfg.policy_history_steps,
            dtype=torch.bool,
            device=self.device,
        )
        self.current_component_mse = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.imitation_active = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.started = False
        self.transitions_scored = 0
        self.failure_boundaries = 0
        self.postfailure_zero_rewards = 0

    def _load_frozen_predictor(self) -> None:
        files = {
            name: _resolve(record["path"])
            for name, record in self.config["artifacts"].items()
            if isinstance(record, dict) and "path" in record
        }
        missing = [name for name, path in files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"demo runtime artifacts are missing: {sorted(missing)}"
            )
        self.artifact_paths = {
            name: str(path) for name, path in sorted(files.items())
        }
        predictor_cfg = json.loads(
            files["predictor_config"].read_text(encoding="utf-8")
        )
        model_module = _load_module(
            files["model_source"],
            "_curiosity_frozen_demo_reward_predictor_model",
        )
        with np.load(files["normalization"], allow_pickle=False) as archive:
            statistics = {
                name: torch.from_numpy(np.asarray(archive[name])).to(self.device)
                for name in (
                    "state_mean",
                    "state_std",
                    "demo_mean",
                    "demo_std",
                    "tactile_rms",
                    "target_scale",
                )
            }
        inputs = predictor_cfg["model_inputs"]
        architecture = predictor_cfg["architecture"]
        self.model = model_module.DemoConditionedCausalPredictorV1(
            policy_dim=int(inputs["policy_prefix_shape"][1]),
            policy_history_steps=int(inputs["policy_prefix_shape"][0]),
            demo_windows=int(inputs["selected_demo_condition_shape"][0]),
            demo_window_steps=int(inputs["selected_demo_condition_shape"][1]),
            demo_feature_dim=int(inputs["selected_demo_condition_shape"][2]),
            tactile_history_steps=int(inputs["direct_tacsl_history_shape"][0]),
            tactile_num_hands=int(inputs["direct_tacsl_history_shape"][1]),
            tactile_channels_per_hand=int(
                inputs["direct_tacsl_history_shape"][2]
            ),
            tactile_grid_shape=inputs["direct_tacsl_history_shape"][3:],
            tactile_encoder_channels=architecture["tactile_encoder_channels"],
            tactile_embedding_dim_per_hand=int(
                architecture["tactile_embedding_dim_per_hand"]
            ),
            d_model=int(architecture["d_model"]),
            nhead=int(architecture["nhead"]),
            num_layers=int(architecture["num_layers"]),
            dim_feedforward=int(architecture["dim_feedforward"]),
            dropout=float(architecture["dropout"]),
            state_mean=statistics["state_mean"],
            state_std=statistics["state_std"],
            demo_mean=statistics["demo_mean"],
            demo_std=statistics["demo_std"],
            tactile_rms=statistics["tactile_rms"],
            target_scale=statistics["target_scale"],
        ).to(self.device)
        checkpoint = torch.load(
            files["checkpoint"], map_location=self.device, weights_only=True
        )
        endpoint = json.loads(
            files["frozen_endpoint"].read_text(encoding="utf-8")
        )
        if not (
            checkpoint.get("protocol")
            == "sugar_demo_reward_prefailure_crossdemo_checkpoint_v2"
            and int(checkpoint["epoch"]) == int(endpoint["best_epoch"]) == 20
        ):
            raise ValueError("V2 predictor checkpoint/endpoint identity drift")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("frozen demo predictor still has trainable parameters")
        self.frozen_model_state = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        self.target_scale = statistics["target_scale"].detach().clone()

        routing_path = files["demo_routing"]
        bank = np.load(files["demo_condition_bank"], mmap_mode="r", allow_pickle=False)
        with np.load(routing_path, allow_pickle=False) as routing:
            motion_id = np.asarray(routing["motion_id"], dtype=np.int64)
            bank_row = np.asarray(routing["local_demo_bank_row"], dtype=np.int64)
        selected = int(self.config["selected_demo"]["motion_id"])
        expected_row = int(self.config["selected_demo"]["bank_row"])
        condition_mode = str(
            self.config["selected_demo"].get(
                "condition_mode", "selected_numeric"
            )
        )
        if condition_mode not in {
            "selected_numeric",
            "zero_normalized",
            "external_numeric",
        }:
            raise ValueError(
                f"unsupported selected-demo condition mode: {condition_mode}"
            )
        if condition_mode == "external_numeric":
            if "external_demo_condition" not in files:
                raise ValueError(
                    "external_numeric requires a hash-bound external_demo_condition artifact"
                )
            numeric_demo = np.load(
                files["external_demo_condition"], allow_pickle=False
            ).astype(np.float32, copy=False)
        else:
            selected_rows = np.flatnonzero(motion_id == selected)
            if selected_rows.size == 0 or set(bank_row[selected_rows].tolist()) != {
                expected_row
            }:
                raise ValueError("selected demo motion-to-bank routing mismatch")
            numeric_demo = np.array(bank[expected_row], dtype=np.float32, copy=True)
        if numeric_demo.shape != (32, 10, 216) or not np.isfinite(numeric_demo).all():
            raise ValueError("selected numeric demo condition shape/value drift")
        self.demo_condition = (
            torch.from_numpy(numeric_demo)
            .to(self.device)
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1, -1)
        )
        self.selected_demo_motion_id = selected
        self.selected_demo_bank_row = expected_row
        self.demo_condition_mode = condition_mode

    def _policy(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        policy = observation["policy"]
        if policy.shape != (self.num_envs, self.cfg.policy_dim):
            raise ValueError(
                f"demo policy vector shape drift: {tuple(policy.shape)}"
            )
        if not torch.isfinite(policy).all():
            raise ValueError("demo policy vector contains non-finite values")
        return policy

    def _tactile(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tactile = observation["tactile_history"]
        expected_flat = (
            self.cfg.tactile_history_steps
            * self.cfg.tactile_num_hands
            * self.cfg.tactile_channels_per_hand
            * self.cfg.tactile_rows
            * self.cfg.tactile_cols
        )
        if tactile.shape != (self.num_envs, expected_flat):
            raise ValueError(
                f"demo direct-TacSL history shape drift: {tuple(tactile.shape)}"
            )
        tactile = tactile.reshape(
            self.num_envs,
            self.cfg.tactile_history_steps,
            self.cfg.tactile_num_hands,
            self.cfg.tactile_channels_per_hand,
            self.cfg.tactile_rows,
            self.cfg.tactile_cols,
        )
        if not torch.isfinite(tactile).all():
            raise ValueError("demo direct-TacSL history contains non-finite values")
        return tactile

    def _push_policy(
        self, policy: torch.Tensor, reset_mask: torch.Tensor
    ) -> None:
        if reset_mask.shape != (self.num_envs,) or reset_mask.dtype is not torch.bool:
            raise ValueError("demo prefix reset mask shape/type drift")
        if reset_mask.any():
            self.policy_prefix[reset_mask] = 0.0
            self.prefix_valid_mask[reset_mask] = False
        self.policy_prefix[:, :-1] = self.policy_prefix[:, 1:].clone()
        self.prefix_valid_mask[:, :-1] = self.prefix_valid_mask[:, 1:].clone()
        self.policy_prefix[:, -1] = policy
        self.prefix_valid_mask[:, -1] = True

    @torch.no_grad()
    def _predict(
        self, tactile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(
            policy_prefix=self.policy_prefix,
            prefix_valid_mask=self.prefix_valid_mask,
            tactile_history=tactile,
            demo_condition=self.demo_condition,
            # Match the predictor's audited zero-demo ablation exactly: zero
            # after normalization, not a raw all-zero tensor that would become
            # nonzero after subtracting the learned demo mean.
            zero_demo=self.demo_condition_mode == "zero_normalized",
        )
        component_mse = self.model.decode_mean(output["mean_log1p_scaled"])
        if component_mse.shape != (self.num_envs, 4):
            raise RuntimeError("demo predictor component shape drift")
        if torch.any(component_mse < 0) or not torch.isfinite(component_mse).all():
            raise RuntimeError("demo predictor produced invalid component MSE")
        return component_mse, output["log_variance_log1p_scaled"]

    @torch.no_grad()
    def begin(self, observation: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.started:
            raise RuntimeError("demo scorer already started")
        policy = self._policy(observation)
        self._push_policy(
            policy,
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
        )
        failure_closed = (
            policy[:, self.cfg.failure_closed_policy_index] > 0.5
        )
        if failure_closed.any():
            raise RuntimeError("new demo scorer episode begins failure-closed")
        self.imitation_active.fill_(True)
        self.current_component_mse, log_variance = self._predict(
            self._tactile(observation)
        )
        self.started = True
        return {
            "component_mse": self.current_component_mse.detach().clone(),
            "log_variance_log1p_scaled": log_variance.detach().clone(),
            "imitation_active": self.imitation_active.detach().clone(),
        }

    @torch.no_grad()
    def process_step(
        self,
        observation_tp1: Mapping[str, torch.Tensor],
        dones: torch.Tensor,
    ) -> DemoRewardStepSignals:
        if not self.started:
            raise RuntimeError("call demo scorer begin() first")
        dones = dones.to(device=self.device, dtype=torch.bool).reshape(-1)
        if dones.shape != (self.num_envs,):
            raise ValueError("demo done mask shape drift")
        next_policy = self._policy(observation_tp1)
        self._push_policy(next_policy, dones)
        next_component_mse, _log_variance = self._predict(
            self._tactile(observation_tp1)
        )
        failure_closed_next = (
            next_policy[:, self.cfg.failure_closed_policy_index] > 0.5
        )
        shaping_next_active = (
            self.imitation_active & ~failure_closed_next & ~dones
        )
        output = potential_difference_reward(
            current_component_mse=self.current_component_mse,
            next_component_mse=next_component_mse,
            target_scale=self.target_scale,
            current_imitation_active=self.imitation_active,
            next_imitation_active=shaping_next_active,
            next_done=dones,
            gamma=self.cfg.gamma,
            eta=self.cfg.eta,
        )
        unit = potential_difference_reward(
            current_component_mse=self.current_component_mse,
            next_component_mse=next_component_mse,
            target_scale=self.target_scale,
            current_imitation_active=self.imitation_active,
            next_imitation_active=shaping_next_active,
            next_done=dones,
            gamma=self.cfg.gamma,
            eta=1.0,
        )
        reward = output["reward"]
        if torch.any(~self.imitation_active & (reward != 0.0)):
            raise RuntimeError("postfailure demo reward is not exactly zero")
        newly_closed = self.imitation_active & ~shaping_next_active & ~dones
        self.failure_boundaries += int(newly_closed.sum().item())
        self.postfailure_zero_rewards += int(
            ((~self.imitation_active) & (reward == 0.0)).sum().item()
        )
        current_active = self.imitation_active.detach().clone()
        self.current_component_mse = next_component_mse.detach().clone()
        self.imitation_active = torch.where(
            dones,
            torch.ones_like(shaping_next_active),
            shaping_next_active,
        )
        self.transitions_scored += self.num_envs
        return DemoRewardStepSignals(
            reward=reward.detach().clone(),
            unit_eta_reward=unit["reward"].detach().clone(),
            component_mse=next_component_mse.detach().clone(),
            current_phi=output["current_phi"].detach().clone(),
            next_phi=output["next_phi"].detach().clone(),
            current_psi=output["current_psi"].detach().clone(),
            next_psi=output["next_psi"].detach().clone(),
            current_imitation_active=current_active,
            next_imitation_active=shaping_next_active.detach().clone(),
            failure_closed_next=failure_closed_next.detach().clone(),
            transition_done=dones.detach().clone(),
        )

    def frozen_model_audit(self) -> dict[str, Any]:
        current_state = self.model.state_dict()
        model_unchanged = set(current_state) == set(self.frozen_model_state) and all(
            torch.equal(current_state[name], self.frozen_model_state[name])
            for name in self.frozen_model_state
        )
        return {
            "model_bitwise_frozen": model_unchanged,
            "all_parameters_require_grad_false": not any(
                parameter.requires_grad for parameter in self.model.parameters()
            ),
            "training_mode_false": self.model.training is False,
            "selected_demo_motion_id": self.selected_demo_motion_id,
            "selected_demo_bank_row": self.selected_demo_bank_row,
            "demo_condition_mode": self.demo_condition_mode,
            "transitions_scored": self.transitions_scored,
            "failure_boundaries": self.failure_boundaries,
            "postfailure_zero_rewards": self.postfailure_zero_rewards,
        }

    def state_dict(self) -> dict[str, Any]:
        if not self.started:
            raise RuntimeError("cannot checkpoint an unstarted demo scorer")
        return {
            "protocol": "sugar_frozen_demo_reward_scorer_state_v1",
            "runtime_config": asdict(self.cfg),
            "config_path": str(self.config_path),
            "base_config_path": (
                str(self.base_config_path)
                if self.base_config_path is not None
                else None
            ),
            "artifact_paths": dict(self.artifact_paths),
            "selected_demo_motion_id": self.selected_demo_motion_id,
            "selected_demo_bank_row": self.selected_demo_bank_row,
            "policy_prefix": self.policy_prefix.detach().clone(),
            "prefix_valid_mask": self.prefix_valid_mask.detach().clone(),
            "current_component_mse": self.current_component_mse.detach().clone(),
            "imitation_active": self.imitation_active.detach().clone(),
            "transitions_scored": self.transitions_scored,
            "failure_boundaries": self.failure_boundaries,
            "postfailure_zero_rewards": self.postfailure_zero_rewards,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("protocol") != "sugar_frozen_demo_reward_scorer_state_v1":
            raise ValueError("unexpected demo scorer state protocol")
        if state.get("runtime_config") != asdict(self.cfg):
            raise ValueError("demo runtime config checkpoint drift")
        if state.get("config_path") != str(self.config_path):
            raise ValueError("demo runtime config path changed")
        expected_base_path = (
            str(self.base_config_path)
            if self.base_config_path is not None
            else None
        )
        if state.get("base_config_path") != expected_base_path:
            raise ValueError("demo base runtime config path changed")
        if state.get("artifact_paths") != self.artifact_paths:
            raise ValueError("demo runtime artifact checkpoint drift")
        if (
            int(state.get("selected_demo_motion_id", -1))
            != self.selected_demo_motion_id
            or int(state.get("selected_demo_bank_row", -2))
            != self.selected_demo_bank_row
            or not self.frozen_model_audit()["model_bitwise_frozen"]
        ):
            raise ValueError("demo predictor identity drift")
        expected = {
            "policy_prefix": self.policy_prefix,
            "prefix_valid_mask": self.prefix_valid_mask,
            "current_component_mse": self.current_component_mse,
            "imitation_active": self.imitation_active,
        }
        for name, destination in expected.items():
            source = state[name].to(device=self.device, dtype=destination.dtype)
            if source.shape != destination.shape:
                raise ValueError(f"demo scorer checkpoint shape drift: {name}")
            destination.copy_(source)
        self.transitions_scored = int(state["transitions_scored"])
        self.failure_boundaries = int(state["failure_boundaries"])
        self.postfailure_zero_rewards = int(state["postfailure_zero_rewards"])
        self.started = True


class DemoRewardAugmentedSMPICMRolloutIntegrator:
    """Add a frozen demo ledger after the unchanged SMP/ICM integrator."""

    def __init__(
        self,
        *,
        base: Any,
        demo: FrozenDemoRewardScorer,
    ) -> None:
        if base.device != demo.device:
            raise ValueError("base and demo integrations must share one device")
        if base.env.num_envs != demo.num_envs:
            raise ValueError("base and demo environment counts differ")
        self.base = base
        self.demo = demo
        self.last_base_signals: Any | None = None

    @property
    def at_rollout_boundary(self) -> bool:
        return self.base.at_rollout_boundary

    def begin(
        self, observation: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        return {
            "smp_window": self.base.begin(),
            "demo": self.demo.begin(observation),
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
    ) -> DemoRewardAugmentedStepSignals:
        base_signals = self.base.process_step(
            observation_t=observation_t,
            applied_action_policy_units_t=applied_action_policy_units_t,
            observation_tp1=observation_tp1,
            external_reward=external_reward,
            dones=dones,
        )
        self.last_base_signals = base_signals
        demo_signals = self.demo.process_step(observation_tp1, dones)
        policy_reward = base_signals.policy_reward + demo_signals.reward
        if not torch.isfinite(policy_reward).all():
            raise RuntimeError("non-finite demo-augmented policy reward")
        base_values = dict(vars(base_signals))
        base_values["policy_reward"] = policy_reward.detach().clone()
        return DemoRewardAugmentedStepSignals(
            **base_values,
            demo_reward=demo_signals.reward,
            demo_unit_eta_reward=demo_signals.unit_eta_reward,
            demo_component_mse=demo_signals.component_mse,
            demo_current_phi=demo_signals.current_phi,
            demo_next_phi=demo_signals.next_phi,
            demo_current_psi=demo_signals.current_psi,
            demo_next_psi=demo_signals.next_psi,
            demo_current_imitation_active=(
                demo_signals.current_imitation_active
            ),
            demo_next_imitation_active=demo_signals.next_imitation_active,
            demo_failure_closed_next=demo_signals.failure_closed_next,
        )

    def finish_rollout(self) -> dict[str, Any]:
        metrics = self.base.finish_rollout()
        return {
            **metrics,
            "demo_predictor_updated": False,
            "demo_predictor_bitwise_frozen": self.demo.frozen_model_audit()[
                "model_bitwise_frozen"
            ],
        }

    def state_dict(self) -> dict[str, Any]:
        if not self.at_rollout_boundary:
            raise RuntimeError("checkpoint augmented integration at rollout boundary")
        return {
            "protocol": "sugar_demo_reward_augmented_smp_icm_integration_v1",
            "base": self.base.state_dict(),
            "demo": self.demo.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("protocol") != "sugar_demo_reward_augmented_smp_icm_integration_v1":
            raise ValueError("unexpected augmented integration checkpoint protocol")
        self.base.load_state_dict(state["base"])
        self.demo.load_state_dict(state["demo"])
