# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Audited upstream RSL PPO on the native SUGAR action boundary.

This module is policy-optimizer telemetry and checkpoint glue. The policy
update itself is executed by ``rsl_rl.algorithms.PPO.update``. Runtime wrappers
observe its unchanged mini-batches, gradient clipping, Adam steps, and
adaptive-KL learning-rate changes; they do not replace the upstream equations.

Original ICM is deliberately absent from this class. The Stage-H integrator
learns and scores ICM independently, then supplies its already separated
pre-update discovery ledger to this ordinary policy optimizer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import inspect
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from sugar_rl.utils.tactile_actor_critic import TactileActorCritic


PINNED_RSL_RL_VERSION = "3.0.1"
PINNED_RSL_PPO_SHA256 = (
    "deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().flatten()
    if flat.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    quantiles = torch.quantile(
        flat,
        torch.tensor(
            [0.05, 0.5, 0.95], device=flat.device, dtype=flat.dtype
        ),
    )
    return {
        "mean": float(flat.mean()),
        "min": float(flat.min()),
        "q05": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "max": float(flat.max()),
    }


def _gradient_l2(parameters: Iterable[nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float64)
        total += torch.square(gradient).sum()
    return float(torch.sqrt(total))


def _snapshot_parameters(
    parameters: Iterable[nn.Parameter],
) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _parameter_delta_l2(
    reference: list[torch.Tensor], parameters: Iterable[nn.Parameter]
) -> float:
    current = list(parameters)
    if len(reference) != len(current):
        raise RuntimeError("policy parameter schema changed during PPO update")
    total = torch.zeros((), dtype=torch.float64)
    for before, after in zip(reference, current, strict=True):
        difference = (
            after.detach().to(device="cpu", dtype=torch.float64)
            - before.to(device="cpu", dtype=torch.float64)
        )
        total += torch.square(difference).sum()
    return float(torch.sqrt(total))


def _explained_variance(
    targets: torch.Tensor, predictions: torch.Tensor
) -> float:
    targets = targets.detach().flatten()
    predictions = predictions.detach().flatten()
    target_variance = torch.var(targets, unbiased=False)
    if float(target_variance) <= 1.0e-12:
        return 0.0
    residual_variance = torch.var(
        targets - predictions, unbiased=False
    )
    return float(1.0 - residual_variance / target_variance)


def _all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(child) for child in value)
    return True


class SugarNativeTactileActorCritic(TactileActorCritic):
    """Official SUGAR BasePPO MLP with the admitted direct spatial R15 branch."""

    def __init__(self, *args, **kwargs) -> None:
        expected = {
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "tactile_obs_group": "tactile_history",
            "tactile_grid_shape": (20, 25),
            "tactile_num_hands": 2,
            "tactile_channels_per_hand": 12,
            "tactile_encoder_channels": [32, 64, 64],
            "tactile_embedding_dim": 128,
        }
        drift: dict[str, dict[str, Any]] = {}
        for name, target in expected.items():
            actual = kwargs.get(name)
            if isinstance(target, (tuple, list)):
                matches = tuple(actual) == tuple(target)
            else:
                matches = actual == target
            if not matches:
                drift[name] = {"actual": actual, "expected": target}
        if drift:
            raise ValueError(f"official SUGAR BasePPO policy config drift: {drift}")
        super().__init__(*args, **kwargs)


class SugarNativeZeroPreservingTactileActorCritic(
    SugarNativeTactileActorCritic
):
    """Fresh SUGAR-native policy whose zero-taxel feature stays exactly zero."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frozen_tactile_encoder_bias_names: tuple[str, ...] = tuple(
            sorted(self._freeze_zero_tactile_biases())
        )
        if len(self.frozen_tactile_encoder_bias_names) != 14:
            raise RuntimeError(
                "expected seven frozen additive biases per tactile encoder, "
                f"got {self.frozen_tactile_encoder_bias_names}"
            )
        audit = self.zero_tactile_causal_audit()
        if not audit["passed"]:
            raise RuntimeError(
                f"initial zero-taxel causal invariant failed: {audit}"
            )
        self.residual_mean_initialized_exact_zero = False

    def _freeze_zero_tactile_biases(self) -> list[str]:
        frozen: list[str] = []
        for encoder_name in (
            "actor_tactile_encoder",
            "critic_tactile_encoder",
        ):
            encoder = getattr(self, encoder_name)
            for module_name, module in encoder.named_modules():
                bias = getattr(module, "bias", None)
                if bias is None:
                    continue
                if torch.count_nonzero(bias.detach()) != 0:
                    raise RuntimeError(
                        f"{encoder_name}.{module_name}.bias is not zero"
                    )
                bias.requires_grad_(False)
                relative = f".{module_name}" if module_name else ""
                frozen.append(f"{encoder_name}{relative}.bias")
        return frozen

    @torch.no_grad()
    def zero_tactile_causal_audit(self) -> dict[str, Any]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        zeros = torch.zeros(
            2,
            self.actor_tactile_encoder.expected_flat_dim,
            device=device,
            dtype=dtype,
        )
        actor_features = self.actor_tactile_encoder(zeros)
        critic_features = self.critic_tactile_encoder(zeros)
        named_parameters = dict(self.named_parameters())
        bias_abs_max = max(
            (
                float(torch.abs(named_parameters[name]).max())
                for name in self.frozen_tactile_encoder_bias_names
            ),
            default=0.0,
        )
        all_biases_frozen = all(
            not named_parameters[name].requires_grad
            for name in self.frozen_tactile_encoder_bias_names
        )
        actor_abs_max = float(torch.abs(actor_features).max())
        critic_abs_max = float(torch.abs(critic_features).max())
        return {
            "actor_zero_taxel_feature_abs_max": actor_abs_max,
            "critic_zero_taxel_feature_abs_max": critic_abs_max,
            "tactile_encoder_bias_abs_max": bias_abs_max,
            "frozen_bias_count": len(
                self.frozen_tactile_encoder_bias_names
            ),
            "all_tactile_encoder_biases_frozen": all_biases_frozen,
            "passed": (
                actor_abs_max == 0.0
                and critic_abs_max == 0.0
                and bias_abs_max == 0.0
                and all_biases_frozen
                and len(self.frozen_tactile_encoder_bias_names) == 14
            ),
        }

    @torch.no_grad()
    def initialize_residual_mean_exact_zero(self) -> dict[str, Any]:
        """Zero only the existing actor output layer for residual control.

        This does not replace or shrink the admitted SUGAR-native
        state/direct-TacSL architecture.  It makes the initial deterministic
        residual mean exactly zero for every causal observation while leaving
        the upstream Gaussian sampling and PPO log-probability variable
        unchanged.
        """

        linear_layers = [
            (name, module)
            for name, module in self.actor.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            raise RuntimeError("SUGAR-native actor has no linear output layer")
        output_name, output_layer = linear_layers[-1]
        if output_layer.out_features != int(self.std.numel()):
            raise RuntimeError(
                "actor output/action dimension drift: "
                f"{output_layer.out_features} != {int(self.std.numel())}"
            )
        output_layer.weight.zero_()
        if output_layer.bias is None:
            raise RuntimeError("residual actor output layer requires a bias")
        output_layer.bias.zero_()
        self.residual_mean_initialized_exact_zero = True
        return self.residual_mean_zero_audit()

    @torch.no_grad()
    def residual_mean_zero_audit(self) -> dict[str, Any]:
        linear_layers = [
            (name, module)
            for name, module in self.actor.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            raise RuntimeError("SUGAR-native actor has no linear output layer")
        output_name, output_layer = linear_layers[-1]
        weight_abs_max = float(torch.abs(output_layer.weight).max())
        bias_abs_max = float(torch.abs(output_layer.bias).max())
        return {
            "architecture_class": type(self).__name__,
            "actor_output_layer": output_name,
            "actor_output_weight_abs_max": weight_abs_max,
            "actor_output_bias_abs_max": bias_abs_max,
            "initialized": bool(
                self.residual_mean_initialized_exact_zero
            ),
            "passed": (
                self.residual_mean_initialized_exact_zero
                and weight_abs_max == 0.0
                and bias_abs_max == 0.0
            ),
        }


class SugarNativeCuriosityPPO(PPO):
    """Unchanged upstream RSL PPO plus read-only stability telemetry."""

    contract_name = "sugar_native_base_ppo"
    expected_initial_learning_rate = 1.0e-3
    expected_schedule = "adaptive"
    expected_desired_kl: float | None = 0.01

    def __init__(
        self,
        policy: SugarNativeTactileActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.005,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        expected = {
            "num_learning_epochs": (num_learning_epochs, 5),
            "num_mini_batches": (num_mini_batches, 4),
            "clip_param": (clip_param, 0.2),
            "gamma": (gamma, 0.99),
            "lam": (lam, 0.95),
            "value_loss_coef": (value_loss_coef, 1.0),
            "entropy_coef": (entropy_coef, 0.005),
            "learning_rate": (
                learning_rate,
                self.expected_initial_learning_rate,
            ),
            "max_grad_norm": (max_grad_norm, 1.0),
            "use_clipped_value_loss": (use_clipped_value_loss, True),
            "schedule": (schedule, self.expected_schedule),
            "desired_kl": (desired_kl, self.expected_desired_kl),
            "normalize_advantage_per_mini_batch": (
                normalize_advantage_per_mini_batch,
                False,
            ),
        }
        drift = {
            name: {"actual": actual, "expected": target}
            for name, (actual, target) in expected.items()
            if actual != target
        }
        if drift:
            raise ValueError(f"official SUGAR BasePPO config drift: {drift}")
        if rnd_cfg is not None:
            raise ValueError("original ICM is independent; upstream RND must be absent")
        if symmetry_cfg is not None:
            raise ValueError("HN0/HN1 do not use RSL symmetry augmentation")
        if multi_gpu_cfg is not None:
            raise ValueError("HN0/HN1 are locked to one GPU")
        if not isinstance(policy, SugarNativeTactileActorCritic):
            raise TypeError("SUGAR-native PPO requires its audited actor-critic")

        self._audit_upstream_source()
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=(
                normalize_advantage_per_mini_batch
            ),
            rnd_cfg=None,
            symmetry_cfg=None,
            multi_gpu_cfg=None,
        )
        self.optimizer_steps = 0
        self.completed_updates = 0
        self._active_batch: tuple | None = None
        self._active_parameter_before: list[torch.Tensor] | None = None
        self._active_learning_rate_before: float | None = None
        self._active_raw_gradient_norm: float | None = None
        self._active_raw_actor_gradient_norm: float | None = None
        self._active_raw_critic_gradient_norm: float | None = None
        self._active_top_gradient_parameters: list[dict[str, float]] | None = None
        self._active_actor_parameter_before: list[torch.Tensor] | None = None
        self._active_critic_parameter_before: list[torch.Tensor] | None = None
        self._active_batch_coordinates: tuple[int, int] | None = None
        self._mini_batch_telemetry: list[dict[str, Any]] = []

    @staticmethod
    def _audit_upstream_source() -> None:
        ppo_path = Path(inspect.getfile(PPO)).resolve()
        actual = _sha256(ppo_path)
        if actual != PINNED_RSL_PPO_SHA256:
            raise RuntimeError(
                "upstream RSL PPO source drift: "
                f"path={ppo_path}, actual={actual}, "
                f"expected={PINNED_RSL_PPO_SHA256}"
            )

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
    ) -> None:
        if num_transitions_per_env != 24:
            raise ValueError("official SUGAR BasePPO requires 24-step rollouts")
        super().init_storage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
        )

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Persist upstream Adam and explicit update accounting."""

        return {
            "protocol": "sugar_native_curiosity_ppo_v1",
            "contract_name": self.contract_name,
            "upstream_rsl_rl_version": PINNED_RSL_RL_VERSION,
            "upstream_ppo_sha256": PINNED_RSL_PPO_SHA256,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "learning_rate": float(self.learning_rate),
            "optimizer_steps": int(self.optimizer_steps),
            "completed_updates": int(self.completed_updates),
        }

    def load_checkpoint_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("protocol") != "sugar_native_curiosity_ppo_v1":
            raise ValueError("unexpected SUGAR-native PPO checkpoint")
        if state.get("contract_name") != self.contract_name:
            raise ValueError(
                "SUGAR-native PPO checkpoint contract drift: "
                f"{state.get('contract_name')} != {self.contract_name}"
            )
        if state.get("upstream_rsl_rl_version") != PINNED_RSL_RL_VERSION:
            raise ValueError("RSL-RL checkpoint version drift")
        if state.get("upstream_ppo_sha256") != PINNED_RSL_PPO_SHA256:
            raise ValueError("RSL-RL PPO source SHA drift")
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.learning_rate = float(state["learning_rate"])
        self.optimizer_steps = int(state["optimizer_steps"])
        self.completed_updates = int(state["completed_updates"])
        if (
            self.optimizer_steps < 0
            or self.completed_updates < 0
            or not (1.0e-5 <= self.learning_rate <= 1.0e-2)
        ):
            raise ValueError("invalid SUGAR-native PPO checkpoint accounting")
        group_rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if group_rates != {self.learning_rate}:
            raise ValueError(
                "optimizer learning-rate state disagrees with PPO accounting: "
                f"{group_rates} != {self.learning_rate}"
            )

    def _set_policy_distribution_without_sampling(self, observations) -> None:
        actor_input = self.policy.actor_obs_normalizer(
            self.policy.get_actor_obs(observations)
        )
        self.policy.update_distribution(actor_input)

    def _actor_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.policy.named_parameters()
            if name.startswith("actor.")
            or name.startswith("actor_tactile_encoder.")
            or name in {"std", "log_std"}
        ]

    def _critic_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.policy.named_parameters()
            if name.startswith("critic.")
            or name.startswith("critic_tactile_encoder.")
        ]

    def _top_gradient_parameters(
        self, count: int = 8
    ) -> list[dict[str, float]]:
        values: list[tuple[str, float]] = []
        for name, parameter in self.policy.named_parameters():
            if parameter.grad is None:
                continue
            norm = float(
                torch.linalg.vector_norm(
                    parameter.grad.detach().to(
                        device="cpu", dtype=torch.float64
                    )
                )
            )
            values.append((name, norm))
        values.sort(key=lambda item: item[1], reverse=True)
        return [
            {"name": name, "gradient_l2": norm}
            for name, norm in values[:count]
        ]

    def _preupdate_rollout_telemetry(self) -> dict[str, Any]:
        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        old_log_probability = self.storage.actions_log_prob.flatten(0, 1).squeeze(
            -1
        )
        returns = self.storage.returns.flatten(0, 1)
        advantages = self.storage.advantages.flatten(0, 1)
        old_values = self.storage.values.flatten(0, 1)
        old_mean = self.storage.mu.flatten(0, 1)
        old_sigma = self.storage.sigma.flatten(0, 1)
        with torch.no_grad():
            self._set_policy_distribution_without_sampling(observations)
            log_probability = self.policy.get_actions_log_prob(actions)
            log_ratio = log_probability - old_log_probability
            ratio = torch.exp(log_ratio)
            values = self.policy.evaluate(observations)
        return {
            "sample_count": int(actions.shape[0]),
            "preupdate_importance_ratio_mean": float(ratio.mean()),
            "preupdate_importance_ratio_max_abs_from_one": float(
                torch.abs(ratio - 1.0).max()
            ),
            "preupdate_clip_fraction": float(
                (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
            ),
            "actions": _distribution_summary(actions),
            "action_abs_gt_one_fraction": float(
                (torch.abs(actions) > 1.0).float().mean()
            ),
            "old_action_mean": _distribution_summary(old_mean),
            "old_action_std": _distribution_summary(old_sigma),
            "returns": _distribution_summary(returns),
            "advantages": _distribution_summary(advantages),
            "values": _distribution_summary(old_values),
            "explained_variance_before": _explained_variance(returns, values),
        }

    def _record_pre_step(self) -> None:
        if (
            self._active_batch is None
            or self._active_parameter_before is None
            or self._active_learning_rate_before is None
            or self._active_batch_coordinates is None
        ):
            raise RuntimeError("missing active upstream PPO mini-batch")
        (
            observations,
            actions,
            target_values,
            advantages,
            returns,
            old_log_probability,
            old_mean,
            old_sigma,
            _hidden_states,
            _masks,
        ) = self._active_batch
        epoch_index, mini_batch_index = self._active_batch_coordinates
        with torch.no_grad():
            # Upstream PPO has already called policy.act on this exact batch.
            log_probability = self.policy.get_actions_log_prob(actions)
            current_mean = self.policy.action_mean
            current_sigma = self.policy.action_std
            entropy = self.policy.entropy
            values = self.policy.evaluate(observations)

            log_ratio = log_probability - old_log_probability.squeeze(-1)
            ratio = torch.exp(log_ratio)
            surrogate = -advantages.squeeze(-1) * ratio
            surrogate_clipped = -advantages.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(
                surrogate, surrogate_clipped
            ).mean()

            value_clipped = target_values + (values - target_values).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = torch.square(values - returns)
            value_losses_clipped = torch.square(value_clipped - returns)
            value_loss = torch.maximum(
                value_losses, value_losses_clipped
            ).mean()
            total_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )
            sampled_approx_kl = ((ratio - 1.0) - log_ratio).mean()
            analytic_kl = torch.sum(
                torch.log(current_sigma / old_sigma + 1.0e-5)
                + (
                    torch.square(old_sigma)
                    + torch.square(old_mean - current_mean)
                )
                / (2.0 * torch.square(current_sigma))
                - 0.5,
                dim=-1,
            ).mean()

        group_rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(group_rates) != 1:
            raise RuntimeError(f"upstream Adam group-rate divergence: {group_rates}")
        gradient_norm_after_clip = _gradient_l2(self.policy.parameters())
        raw_gradient_norm = self._active_raw_gradient_norm
        raw_actor_gradient_norm = self._active_raw_actor_gradient_norm
        raw_critic_gradient_norm = self._active_raw_critic_gradient_norm
        top_gradient_parameters = self._active_top_gradient_parameters
        if (
            raw_gradient_norm is None
            or raw_actor_gradient_norm is None
            or raw_critic_gradient_norm is None
            or top_gradient_parameters is None
        ):
            raise RuntimeError("upstream clip_grad_norm_ was not observed")
        clipped_actor_gradient_norm = _gradient_l2(self._actor_parameters())
        clipped_critic_gradient_norm = _gradient_l2(
            self._critic_parameters()
        )
        telemetry = {
            "epoch": epoch_index,
            "mini_batch": mini_batch_index,
            "importance_ratio_mean": float(ratio.mean()),
            "importance_ratio_min": float(ratio.min()),
            "importance_ratio_max": float(ratio.max()),
            "clip_fraction": float(
                (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
            ),
            "sampled_approx_kl": float(sampled_approx_kl),
            "analytic_gaussian_kl": float(analytic_kl),
            "surrogate_loss": float(surrogate_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy.mean()),
            "total_loss": float(total_loss),
            "raw_gradient_l2": float(raw_gradient_norm),
            "raw_actor_gradient_l2": float(raw_actor_gradient_norm),
            "raw_critic_gradient_l2": float(raw_critic_gradient_norm),
            "top_raw_gradient_parameters": top_gradient_parameters,
            "clipped_gradient_l2": float(gradient_norm_after_clip),
            "clipped_actor_gradient_l2": clipped_actor_gradient_norm,
            "clipped_critic_gradient_l2": clipped_critic_gradient_norm,
            "gradient_was_clipped": bool(raw_gradient_norm > self.max_grad_norm),
            "learning_rate_before_adaptation": float(
                self._active_learning_rate_before
            ),
            "learning_rate_at_step": group_rates.pop(),
            "action_mean": _distribution_summary(current_mean),
            "action_std": _distribution_summary(current_sigma),
        }
        telemetry["all_finite"] = _all_finite(telemetry)
        self._mini_batch_telemetry.append(telemetry)

    def _record_post_step(self) -> None:
        if (
            self._active_parameter_before is None
            or self._active_actor_parameter_before is None
            or self._active_critic_parameter_before is None
        ):
            raise RuntimeError("missing parameter snapshot after Adam step")
        if not self._mini_batch_telemetry:
            raise RuntimeError("missing pre-step PPO telemetry")
        self._mini_batch_telemetry[-1]["parameter_delta_l2"] = (
            _parameter_delta_l2(
                self._active_parameter_before, self.policy.parameters()
            )
        )
        self._mini_batch_telemetry[-1]["actor_parameter_delta_l2"] = (
            _parameter_delta_l2(
                self._active_actor_parameter_before,
                self._actor_parameters(),
            )
        )
        self._mini_batch_telemetry[-1]["critic_parameter_delta_l2"] = (
            _parameter_delta_l2(
                self._active_critic_parameter_before,
                self._critic_parameters(),
            )
        )
        self.optimizer_steps += 1
        self._active_parameter_before = None
        self._active_actor_parameter_before = None
        self._active_critic_parameter_before = None

    @staticmethod
    def _epoch_telemetry(
        mini_batches: list[dict[str, Any]], num_epochs: int
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for epoch_index in range(num_epochs):
            selected = [
                item for item in mini_batches if item["epoch"] == epoch_index
            ]
            if not selected:
                raise RuntimeError(f"missing PPO epoch {epoch_index}")

            def mean(name: str) -> float:
                return sum(float(item[name]) for item in selected) / len(selected)

            output.append(
                {
                    "epoch": epoch_index,
                    "mini_batches": len(selected),
                    "clip_fraction_mean": mean("clip_fraction"),
                    "clip_fraction_max": max(
                        float(item["clip_fraction"]) for item in selected
                    ),
                    "sampled_approx_kl_mean": mean("sampled_approx_kl"),
                    "sampled_approx_kl_max": max(
                        float(item["sampled_approx_kl"]) for item in selected
                    ),
                    "analytic_gaussian_kl_mean": mean(
                        "analytic_gaussian_kl"
                    ),
                    "analytic_gaussian_kl_max": max(
                        float(item["analytic_gaussian_kl"])
                        for item in selected
                    ),
                    "surrogate_loss_mean": mean("surrogate_loss"),
                    "value_loss_mean": mean("value_loss"),
                    "entropy_mean": mean("entropy"),
                    "raw_gradient_l2_mean": mean("raw_gradient_l2"),
                    "raw_gradient_l2_max": max(
                        float(item["raw_gradient_l2"]) for item in selected
                    ),
                    "raw_actor_gradient_l2_max": max(
                        float(item["raw_actor_gradient_l2"])
                        for item in selected
                    ),
                    "raw_critic_gradient_l2_max": max(
                        float(item["raw_critic_gradient_l2"])
                        for item in selected
                    ),
                    "clipped_gradient_l2_max": max(
                        float(item["clipped_gradient_l2"])
                        for item in selected
                    ),
                    "parameter_delta_l2_sum": sum(
                        float(item["parameter_delta_l2"])
                        for item in selected
                    ),
                    "actor_parameter_delta_l2_sum": sum(
                        float(item["actor_parameter_delta_l2"])
                        for item in selected
                    ),
                    "critic_parameter_delta_l2_sum": sum(
                        float(item["critic_parameter_delta_l2"])
                        for item in selected
                    ),
                    "learning_rate_start": float(
                        selected[0]["learning_rate_before_adaptation"]
                    ),
                    "learning_rate_end": float(
                        selected[-1]["learning_rate_at_step"]
                    ),
                    "all_finite": all(
                        bool(item["all_finite"]) for item in selected
                    ),
                }
            )
        return output

    def update(self) -> dict[str, Any]:
        """Run upstream PPO unchanged while observing its internal steps."""

        if self.policy.is_recurrent:
            raise ValueError("HN0/HN1 are locked to a feed-forward SUGAR policy")
        expected_steps = self.num_learning_epochs * self.num_mini_batches
        rollout_telemetry = self._preupdate_rollout_telemetry()
        update_parameter_before = _snapshot_parameters(self.policy.parameters())
        optimizer_steps_before = self.optimizer_steps
        learning_rate_start = float(self.learning_rate)
        self._mini_batch_telemetry = []

        original_generator = self.storage.mini_batch_generator
        original_clip_grad_norm = nn.utils.clip_grad_norm_
        batch_counter = 0

        def telemetry_generator(num_mini_batches, num_epochs=8):
            nonlocal batch_counter
            for batch in original_generator(num_mini_batches, num_epochs):
                self._active_batch = batch
                self._active_batch_coordinates = (
                    batch_counter // self.num_mini_batches,
                    batch_counter % self.num_mini_batches,
                )
                self._active_parameter_before = _snapshot_parameters(
                    self.policy.parameters()
                )
                self._active_actor_parameter_before = _snapshot_parameters(
                    self._actor_parameters()
                )
                self._active_critic_parameter_before = _snapshot_parameters(
                    self._critic_parameters()
                )
                rates = {
                    float(group["lr"]) for group in self.optimizer.param_groups
                }
                if len(rates) != 1:
                    raise RuntimeError(
                        f"upstream Adam group-rate divergence: {rates}"
                    )
                self._active_learning_rate_before = rates.pop()
                self._active_raw_gradient_norm = None
                self._active_raw_actor_gradient_norm = None
                self._active_raw_critic_gradient_norm = None
                self._active_top_gradient_parameters = None
                yield batch
                batch_counter += 1

        def observed_clip_grad_norm(parameters, max_norm, *args, **kwargs):
            self._active_raw_actor_gradient_norm = _gradient_l2(
                self._actor_parameters()
            )
            self._active_raw_critic_gradient_norm = _gradient_l2(
                self._critic_parameters()
            )
            self._active_top_gradient_parameters = (
                self._top_gradient_parameters()
            )
            result = original_clip_grad_norm(
                parameters, max_norm, *args, **kwargs
            )
            self._active_raw_gradient_norm = float(result)
            return result

        def optimizer_pre_hook(_optimizer, _args, _kwargs):
            self._record_pre_step()

        def optimizer_post_hook(_optimizer, _args, _kwargs):
            self._record_post_step()

        # Instance replacement delegates every yielded tensor to the upstream
        # generator. The global gradient wrapper calls the upstream operation
        # once and only records its returned pre-clipping norm.
        self.storage.mini_batch_generator = telemetry_generator
        nn.utils.clip_grad_norm_ = observed_clip_grad_norm
        pre_handle = self.optimizer.register_step_pre_hook(optimizer_pre_hook)
        post_handle = self.optimizer.register_step_post_hook(optimizer_post_hook)
        try:
            upstream_losses = super().update()
        finally:
            pre_handle.remove()
            post_handle.remove()
            nn.utils.clip_grad_norm_ = original_clip_grad_norm
            del self.storage.mini_batch_generator
            self._active_batch = None
            self._active_parameter_before = None
            self._active_actor_parameter_before = None
            self._active_critic_parameter_before = None
            self._active_learning_rate_before = None
            self._active_raw_gradient_norm = None
            self._active_raw_actor_gradient_norm = None
            self._active_raw_critic_gradient_norm = None
            self._active_top_gradient_parameters = None
            self._active_batch_coordinates = None

        if batch_counter != expected_steps:
            raise RuntimeError(
                f"upstream PPO yielded {batch_counter} batches, "
                f"expected {expected_steps}"
            )
        if self.optimizer_steps - optimizer_steps_before != expected_steps:
            raise RuntimeError("upstream Adam step accounting drift")
        if len(self._mini_batch_telemetry) != expected_steps:
            raise RuntimeError("incomplete upstream PPO mini-batch telemetry")

        self.completed_updates += 1
        epoch_telemetry = self._epoch_telemetry(
            self._mini_batch_telemetry, self.num_learning_epochs
        )
        output: dict[str, Any] = {
            **upstream_losses,
            "optimizer_implementation": "rsl_rl.algorithms.PPO.update",
            "upstream_rsl_rl_version": PINNED_RSL_RL_VERSION,
            "upstream_ppo_sha256": PINNED_RSL_PPO_SHA256,
            "optimizer_steps_this_update": expected_steps,
            "optimizer_steps_total": self.optimizer_steps,
            "completed_updates": self.completed_updates,
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": float(self.learning_rate),
            "parameter_delta_l2": _parameter_delta_l2(
                update_parameter_before, self.policy.parameters()
            ),
            "clip_fraction": sum(
                float(item["clip_fraction"])
                for item in self._mini_batch_telemetry
            )
            / expected_steps,
            "sampled_approx_kl": sum(
                float(item["sampled_approx_kl"])
                for item in self._mini_batch_telemetry
            )
            / expected_steps,
            "analytic_gaussian_kl": sum(
                float(item["analytic_gaussian_kl"])
                for item in self._mini_batch_telemetry
            )
            / expected_steps,
            "rollout_telemetry": rollout_telemetry,
            "epoch_telemetry": epoch_telemetry,
            "mini_batch_telemetry": self._mini_batch_telemetry,
        }
        output["all_finite"] = _all_finite(output)
        return output


class SugarNativeTactileFloorLrPPO(SugarNativeCuriosityPPO):
    """Named upstream-PPO study starting at its official adaptive LR floor."""

    contract_name = "sugar_native_tactile_floor_lr"
    expected_initial_learning_rate = 1.0e-5


class SugarNativeZeroPreservingTactileFloorLrPPO(
    SugarNativeTactileFloorLrPPO
):
    """Floor-LR upstream PPO requiring the zero-preserving tactile policy."""

    contract_name = "sugar_native_zero_preserving_tactile_floor_lr"

    def __init__(self, policy, *args, **kwargs) -> None:
        if not isinstance(
            policy, SugarNativeZeroPreservingTactileActorCritic
        ):
            raise TypeError(
                "zero-preserving PPO requires its audited tactile policy"
            )
        super().__init__(policy, *args, **kwargs)


class SugarNativeZeroPreservingTactileFixedLowLrPPO(
    SugarNativeZeroPreservingTactileFloorLrPPO
):
    """Zero-preserving upstream PPO using its supported fixed-LR mode."""

    contract_name = "sugar_native_zero_preserving_tactile_fixed_low_lr"
    expected_schedule = "fixed"
    expected_desired_kl = None
