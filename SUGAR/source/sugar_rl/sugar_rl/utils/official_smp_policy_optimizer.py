# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Faithful MimicKit SMP PPO semantics adapted to SUGAR observations/actions.

This is policy-optimizer glue, not a curiosity model.  It preserves the
official public SMP architecture and optimizer schedule (two 1024-unit ReLU
layers, fixed 0.05 Gaussian standard deviation, separate SGD optimizers,
32-step rollouts, actor 5x/4-env batches, critic 2x/2-env batches, PPO 0.2,
GAE 0.99/0.95, and clipped normalized advantages).  The policy additionally
uses the validated spatial R15 encoder because the SUGAR research task has a
direct tactile modality that the public MimicKit environment does not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from sugar_rl.utils.tactile_actor_critic import TactileActorCritic


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
PINNED_MIMICKIT_POLICY_SHA256 = {
    "MimicKit/mimickit/learning/ppo_agent.py": (
        "29d3310ef23bdd88adeb8748940d5cd352594c322d3d5635faed3da5684a10ce"
    ),
    "MimicKit/mimickit/learning/mp_optimizer.py": (
        "d6dd56e8e29da43a4fa149b1b42a125f03267096f03d5c8dc27c2928106ab652"
    ),
    "MimicKit/data/agents/smp_humanoid_agent.yaml": (
        "62460e1366fa6d244ce2479e71f9bfde97cc9557fecf858fbe234e02e15828b0"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_linear(module: nn.Module) -> nn.Linear:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linears:
        raise RuntimeError("policy network has no linear output layer")
    return linears[-1]


class OfficialSMPTactileActorCritic(TactileActorCritic):
    """Official SMP MLP contract plus the direct spatial tactile adapter."""

    def __init__(self, *args, **kwargs) -> None:
        hidden_actor = tuple(kwargs.get("actor_hidden_dims", ()))
        hidden_critic = tuple(kwargs.get("critic_hidden_dims", ()))
        activation = kwargs.get("activation")
        noise_std = float(kwargs.get("init_noise_std", -1.0))
        noise_type = kwargs.get("noise_std_type", "scalar")
        if hidden_actor != (1024, 1024) or hidden_critic != (1024, 1024):
            raise ValueError("official SMP actor/critic require two 1024-unit layers")
        if activation != "relu":
            raise ValueError("official SMP actor/critic activation is ReLU")
        if noise_std != 0.05 or noise_type != "scalar":
            raise ValueError("official SMP uses fixed scalar action std=0.05")
        super().__init__(*args, **kwargs)
        if not hasattr(self, "std"):
            raise RuntimeError("official SMP fixed-standard-deviation policy requires std")
        self.std.requires_grad_(False)
        actor_output = _last_linear(self.actor)
        nn.init.uniform_(actor_output.weight, -0.01, 0.01)
        nn.init.zeros_(actor_output.bias)
        critic_output = _last_linear(self.critic)
        nn.init.zeros_(critic_output.bias)
        if not isinstance(self.actor_obs_normalizer, nn.Module) or not hasattr(
            self.actor_obs_normalizer, "_mean"
        ):
            raise ValueError("official SMP adapter requires actor observation normalization")
        if not isinstance(self.critic_obs_normalizer, nn.Module) or not hasattr(
            self.critic_obs_normalizer, "_mean"
        ):
            raise ValueError("official SMP adapter requires critic observation normalization")
        self.register_buffer(
            "_pending_actor_sum",
            torch.zeros_like(self.actor_obs_normalizer._mean),
        )
        self.register_buffer(
            "_pending_actor_sum_square",
            torch.zeros_like(self.actor_obs_normalizer._mean),
        )
        self.register_buffer(
            "_pending_actor_count", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "_pending_critic_sum",
            torch.zeros_like(self.critic_obs_normalizer._mean),
        )
        self.register_buffer(
            "_pending_critic_sum_square",
            torch.zeros_like(self.critic_obs_normalizer._mean),
        )
        self.register_buffer(
            "_pending_critic_count", torch.zeros((), dtype=torch.long)
        )

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.actor_tactile_encoder.parameters()
        yield from self.actor.parameters()

    def critic_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.critic_tactile_encoder.parameters()
        yield from self.critic.parameters()

    @staticmethod
    @torch.no_grad()
    def _record_moments(
        values: torch.Tensor,
        pending_sum: torch.Tensor,
        pending_sum_square: torch.Tensor,
        pending_count: torch.Tensor,
    ) -> None:
        pending_sum.add_(values.detach().sum(dim=0, keepdim=True))
        pending_sum_square.add_(
            torch.square(values.detach()).sum(dim=0, keepdim=True)
        )
        pending_count.add_(values.shape[0])

    @torch.no_grad()
    def record_rollout_normalization(self, obs) -> None:
        """Record pre-action observations without changing live policy stats."""

        actor_input = self.get_actor_obs(obs)
        critic_input = self.get_critic_obs(obs)
        self._record_moments(
            actor_input,
            self._pending_actor_sum,
            self._pending_actor_sum_square,
            self._pending_actor_count,
        )
        self._record_moments(
            critic_input,
            self._pending_critic_sum,
            self._pending_critic_sum_square,
            self._pending_critic_count,
        )

    @staticmethod
    @torch.no_grad()
    def _commit_moments(
        normalizer,
        pending_sum: torch.Tensor,
        pending_sum_square: torch.Tensor,
        pending_count: torch.Tensor,
    ) -> int:
        new_count = int(pending_count.item())
        if new_count == 0:
            return 0
        new_mean = pending_sum / float(new_count)
        new_var = torch.clamp(
            pending_sum_square / float(new_count) - torch.square(new_mean),
            min=0.0,
        )
        old_count = int(normalizer.count.item())
        if old_count == 0:
            combined_mean = new_mean
            combined_var = new_var
        else:
            total = old_count + new_count
            delta = new_mean - normalizer._mean
            combined_mean = (
                float(old_count) * normalizer._mean
                + float(new_count) * new_mean
            ) / float(total)
            combined_var = (
                float(old_count) * normalizer._var
                + float(new_count) * new_var
                + torch.square(delta)
                * (float(old_count * new_count) / float(total))
            ) / float(total)
        normalizer._mean.copy_(combined_mean)
        normalizer._var.copy_(combined_var)
        normalizer._std.copy_(torch.sqrt(torch.clamp(combined_var, min=0.0)))
        normalizer.count.add_(new_count)
        pending_sum.zero_()
        pending_sum_square.zero_()
        pending_count.zero_()
        return new_count

    @torch.no_grad()
    def commit_rollout_normalization(self) -> dict[str, int]:
        """Match MimicKit: update observation moments after the policy update."""

        actor_count = self._commit_moments(
            self.actor_obs_normalizer,
            self._pending_actor_sum,
            self._pending_actor_sum_square,
            self._pending_actor_count,
        )
        critic_count = self._commit_moments(
            self.critic_obs_normalizer,
            self._pending_critic_sum,
            self._pending_critic_sum_square,
            self._pending_critic_count,
        )
        if actor_count != critic_count:
            raise RuntimeError("actor/critic rollout normalizer counts diverged")
        return {
            "policy_normalizer_samples_committed": actor_count,
            "policy_normalizer_total_samples": int(
                self.actor_obs_normalizer.count.item()
            ),
        }

    def update_normalization(self, obs) -> None:
        raise RuntimeError(
            "official SMP adapter records pre-action observations and commits "
            "normalization only after the rollout policy update"
        )


class _DualOptimizerCheckpoint:
    """Minimal state boundary expected by the upstream RSL runner."""

    def __init__(
        self,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
    ) -> None:
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def state_dict(self) -> dict[str, Any]:
        return {
            "protocol": "official_smp_separate_sgd_v1",
            "actor": self.actor_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("protocol") != "official_smp_separate_sgd_v1":
            raise ValueError("unexpected SMP policy-optimizer checkpoint")
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])


class OfficialSMPPolicyOptimizerAdapter(PPO):
    """RSL storage adapter for the official public MimicKit PPO update."""

    def __init__(
        self,
        policy: OfficialSMPTactileActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 8,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1.0e-4,
        max_grad_norm: float = 0.0,
        use_clipped_value_loss: bool = False,
        schedule: str = "fixed",
        desired_kl: float | None = None,
        normalize_advantage_per_mini_batch: bool = False,
        critic_num_learning_epochs: int = 2,
        critic_num_mini_batches: int = 16,
        normalized_advantage_clip: float = 4.0,
        action_bound_weight: float = 10.0,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self._audit_official_sources()
        expected = {
            "num_learning_epochs": (num_learning_epochs, 5),
            "num_mini_batches": (num_mini_batches, 8),
            "clip_param": (clip_param, 0.2),
            "gamma": (gamma, 0.99),
            "lam": (lam, 0.95),
            "value_loss_coef": (value_loss_coef, 1.0),
            "entropy_coef": (entropy_coef, 0.0),
            "learning_rate": (learning_rate, 1.0e-4),
            "max_grad_norm": (max_grad_norm, 0.0),
            "use_clipped_value_loss": (use_clipped_value_loss, False),
            "schedule": (schedule, "fixed"),
            "desired_kl": (desired_kl, None),
            "critic_num_learning_epochs": (critic_num_learning_epochs, 2),
            "critic_num_mini_batches": (critic_num_mini_batches, 16),
            "normalized_advantage_clip": (normalized_advantage_clip, 4.0),
            "action_bound_weight": (action_bound_weight, 10.0),
        }
        drift = {
            name: {"actual": actual, "expected": target}
            for name, (actual, target) in expected.items()
            if actual != target
        }
        if drift:
            raise ValueError(f"official SMP policy optimizer config drift: {drift}")
        if rnd_cfg is not None or symmetry_cfg is not None:
            raise ValueError("official SMP policy adapter does not use RND or symmetry")
        if multi_gpu_cfg is not None:
            raise ValueError("initial Stage-H adapter is single-GPU only")
        if not isinstance(policy, OfficialSMPTactileActorCritic):
            raise TypeError("official SMP optimizer requires its audited actor-critic")

        # Construct the upstream storage/transition boundary, then replace its
        # Adam optimizer with MimicKit's separate momentum-SGD optimizers.
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
        del self.optimizer
        self.actor_num_learning_epochs = num_learning_epochs
        self.actor_num_mini_batches = num_mini_batches
        self.critic_num_learning_epochs = critic_num_learning_epochs
        self.critic_num_mini_batches = critic_num_mini_batches
        self.normalized_advantage_clip = normalized_advantage_clip
        self.action_bound_weight = action_bound_weight

        actor_parameters = list(policy.actor_parameters())
        critic_parameters = list(policy.critic_parameters())
        if not actor_parameters or not critic_parameters:
            raise RuntimeError("empty SMP actor or critic parameter set")
        actor_ids = {id(parameter) for parameter in actor_parameters}
        critic_ids = {id(parameter) for parameter in critic_parameters}
        if actor_ids & critic_ids:
            raise RuntimeError("official SMP requires disjoint actor/critic parameters")
        self.actor_optimizer = torch.optim.SGD(
            actor_parameters, lr=learning_rate, momentum=0.9
        )
        self.critic_optimizer = torch.optim.SGD(
            critic_parameters, lr=learning_rate, momentum=0.9
        )
        self.optimizer = _DualOptimizerCheckpoint(
            self.actor_optimizer, self.critic_optimizer
        )
        self.actor_optimizer_steps = 0
        self.critic_optimizer_steps = 0

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Persist optimizer tensors and update accounting together."""

        return {
            "protocol": "official_smp_policy_optimizer_adapter_v1",
            "optimizer_state_dict": self.optimizer.state_dict(),
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "critic_optimizer_steps": self.critic_optimizer_steps,
        }

    def load_checkpoint_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("protocol") != "official_smp_policy_optimizer_adapter_v1":
            raise ValueError("unexpected official SMP policy optimizer checkpoint")
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.actor_optimizer_steps = int(state["actor_optimizer_steps"])
        self.critic_optimizer_steps = int(state["critic_optimizer_steps"])
        if self.actor_optimizer_steps < 0 or self.critic_optimizer_steps < 0:
            raise ValueError("negative official SMP optimizer step count")

    @staticmethod
    def _audit_official_sources() -> None:
        drift = {
            relative: {
                "actual": _sha256(WORKSPACE_ROOT / relative),
                "expected": expected,
            }
            for relative, expected in PINNED_MIMICKIT_POLICY_SHA256.items()
            if _sha256(WORKSPACE_ROOT / relative) != expected
        }
        if drift:
            raise RuntimeError(f"pinned official MimicKit PPO source drift: {drift}")

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
    ) -> None:
        if num_transitions_per_env != 32:
            raise ValueError("official SMP policy updates require 32-step rollouts")
        super().init_storage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
        )

    def compute_returns(self, obs) -> None:
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=True,
        )
        self.storage.advantages.clamp_(
            -self.normalized_advantage_clip,
            self.normalized_advantage_clip,
        )

    def act(self, obs):
        self.policy.record_rollout_normalization(obs)
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        """RSL transition storage without its per-step normalizer mutation."""

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    @staticmethod
    def _flat_storage(storage):
        return {
            "observations": storage.observations.flatten(0, 1),
            "actions": storage.actions.flatten(0, 1),
            "returns": storage.returns.flatten(0, 1),
            "old_log_prob": storage.actions_log_prob.flatten(0, 1),
            "advantages": storage.advantages.flatten(0, 1),
        }

    @staticmethod
    def _epoch_batches(
        sample_count: int,
        num_mini_batches: int,
        epochs: int,
        device: torch.device | str,
    ):
        if sample_count % num_mini_batches != 0:
            raise ValueError("official SMP mini-batch division must be exact")
        mini_batch_size = sample_count // num_mini_batches
        for epoch_index in range(epochs):
            permutation = torch.randperm(sample_count, device=device)
            for mini_batch_index in range(num_mini_batches):
                begin = mini_batch_index * mini_batch_size
                yield (
                    epoch_index,
                    mini_batch_index,
                    permutation[begin : begin + mini_batch_size],
                )

    @staticmethod
    def _gradient_l2(parameters: Iterable[nn.Parameter]) -> float:
        total = torch.zeros((), dtype=torch.float64)
        for parameter in parameters:
            if parameter.grad is not None:
                gradient = parameter.grad.detach().to(
                    device="cpu", dtype=torch.float64
                )
                total += torch.square(gradient).sum()
        return float(torch.sqrt(total))

    @staticmethod
    def _snapshot_parameters(
        parameters: Iterable[nn.Parameter],
    ) -> list[torch.Tensor]:
        return [parameter.detach().clone() for parameter in parameters]

    @staticmethod
    def _parameter_delta_l2(
        before: list[torch.Tensor],
        parameters: Iterable[nn.Parameter],
    ) -> float:
        current = list(parameters)
        if len(before) != len(current):
            raise RuntimeError("parameter schema changed during optimizer update")
        total = torch.zeros((), dtype=torch.float64)
        for reference, parameter in zip(before, current, strict=True):
            difference = (
                parameter.detach().to(device="cpu", dtype=torch.float64)
                - reference.to(device="cpu", dtype=torch.float64)
            )
            total += torch.square(difference).sum()
        return float(torch.sqrt(total))

    @staticmethod
    def _explained_variance(
        targets: torch.Tensor, predictions: torch.Tensor
    ) -> float:
        targets = targets.detach().flatten()
        predictions = predictions.detach().flatten()
        target_variance = torch.var(targets, unbiased=False)
        if float(target_variance) <= 1.0e-12:
            return 0.0
        residual_variance = torch.var(targets - predictions, unbiased=False)
        return float(1.0 - residual_variance / target_variance)

    @staticmethod
    def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
        values = values.detach().flatten()
        quantiles = torch.quantile(
            values,
            torch.tensor(
                [0.05, 0.5, 0.95], device=values.device, dtype=values.dtype
            ),
        )
        return {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "q05": float(quantiles[0]),
            "q50": float(quantiles[1]),
            "q95": float(quantiles[2]),
            "max": float(values.max()),
        }

    @staticmethod
    def _epoch_aggregates(
        mini_batches: list[dict[str, float | int | bool]],
        epochs: int,
        parameter_delta_by_epoch: list[float],
    ) -> list[dict[str, float | int | bool]]:
        output: list[dict[str, float | int | bool]] = []
        for epoch_index in range(epochs):
            selected = [
                item
                for item in mini_batches
                if item["epoch"] == epoch_index
            ]
            if not selected:
                raise RuntimeError(f"missing optimizer telemetry for epoch {epoch_index}")
            output.append(
                {
                    "epoch": epoch_index,
                    "mini_batches": len(selected),
                    "importance_ratio_mean": sum(
                        float(item["importance_ratio_mean"])
                        for item in selected
                    )
                    / len(selected),
                    "importance_ratio_min": min(
                        float(item["importance_ratio_min"])
                        for item in selected
                    ),
                    "importance_ratio_max": max(
                        float(item["importance_ratio_max"])
                        for item in selected
                    ),
                    "clip_fraction_mean": sum(
                        float(item["clip_fraction"]) for item in selected
                    )
                    / len(selected),
                    "clip_fraction_max": max(
                        float(item["clip_fraction"]) for item in selected
                    ),
                    "approx_kl_mean": sum(
                        float(item["approx_kl"]) for item in selected
                    )
                    / len(selected),
                    "approx_kl_max": max(
                        float(item["approx_kl"]) for item in selected
                    ),
                    "gradient_l2_mean": sum(
                        float(item["gradient_l2"]) for item in selected
                    )
                    / len(selected),
                    "gradient_l2_max": max(
                        float(item["gradient_l2"]) for item in selected
                    ),
                    "parameter_delta_l2": parameter_delta_by_epoch[epoch_index],
                    "all_finite": all(bool(item["all_finite"]) for item in selected),
                }
            )
        return output

    @staticmethod
    def _critic_epoch_aggregates(
        mini_batches: list[dict[str, float | int | bool]],
        epochs: int,
        parameter_delta_by_epoch: list[float],
    ) -> list[dict[str, float | int | bool]]:
        output: list[dict[str, float | int | bool]] = []
        for epoch_index in range(epochs):
            selected = [
                item
                for item in mini_batches
                if item["epoch"] == epoch_index
            ]
            if not selected:
                raise RuntimeError(
                    f"missing critic telemetry for epoch {epoch_index}"
                )
            output.append(
                {
                    "epoch": epoch_index,
                    "mini_batches": len(selected),
                    "critic_loss_mean": sum(
                        float(item["critic_loss"]) for item in selected
                    )
                    / len(selected),
                    "critic_loss_max": max(
                        float(item["critic_loss"]) for item in selected
                    ),
                    "gradient_l2_mean": sum(
                        float(item["gradient_l2"]) for item in selected
                    )
                    / len(selected),
                    "gradient_l2_max": max(
                        float(item["gradient_l2"]) for item in selected
                    ),
                    "parameter_delta_l2": parameter_delta_by_epoch[epoch_index],
                    "all_finite": all(bool(item["all_finite"]) for item in selected),
                }
            )
        return output

    def update(self) -> dict[str, Any]:
        data = self._flat_storage(self.storage)
        sample_count = self.storage.num_envs * self.storage.num_transitions_per_env
        actor_parameters = list(self.policy.actor_parameters())
        critic_parameters = list(self.policy.critic_parameters())
        actor_update_start = self._snapshot_parameters(actor_parameters)
        critic_update_start = self._snapshot_parameters(critic_parameters)
        with torch.no_grad():
            self.policy.act(data["observations"])
            preupdate_log_prob = self.policy.get_actions_log_prob(data["actions"])
            preupdate_ratio = torch.exp(
                preupdate_log_prob - data["old_log_prob"].squeeze(-1)
            )
            preupdate_ratio_mean = float(preupdate_ratio.mean())
            preupdate_ratio_max_abs_from_one = float(
                torch.abs(preupdate_ratio - 1.0).max()
            )
            preupdate_clip_fraction = float(
                (torch.abs(preupdate_ratio - 1.0) > self.clip_param)
                .float()
                .mean()
            )
            critic_predictions_before = self.policy.evaluate(
                data["observations"]
            ).squeeze(-1)
            explained_variance_before = self._explained_variance(
                data["returns"], critic_predictions_before
            )
        critic_loss_sum = 0.0
        critic_mini_batch_telemetry: list[dict[str, float | int | bool]] = []
        critic_parameter_delta_by_epoch: list[float] = []
        critic_epoch_start: list[torch.Tensor] | None = None
        for epoch_index, mini_batch_index, indices in self._epoch_batches(
            sample_count,
            self.critic_num_mini_batches,
            self.critic_num_learning_epochs,
            self.device,
        ):
            if mini_batch_index == 0:
                critic_epoch_start = self._snapshot_parameters(critic_parameters)
            values = self.policy.evaluate(data["observations"][indices])
            critic_loss = torch.square(
                data["returns"][indices] - values
            ).mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            gradient_l2 = self._gradient_l2(critic_parameters)
            self.critic_optimizer.step()
            critic_loss_sum += float(critic_loss.detach())
            self.critic_optimizer_steps += 1
            critic_mini_batch_telemetry.append(
                {
                    "epoch": epoch_index,
                    "mini_batch": mini_batch_index,
                    "critic_loss": float(critic_loss.detach()),
                    "gradient_l2": gradient_l2,
                    "all_finite": bool(
                        torch.isfinite(critic_loss.detach()).item()
                        and torch.isfinite(
                            torch.tensor(gradient_l2, dtype=torch.float64)
                        ).item()
                    ),
                }
            )
            if mini_batch_index == self.critic_num_mini_batches - 1:
                if critic_epoch_start is None:
                    raise RuntimeError("missing critic epoch parameter snapshot")
                critic_parameter_delta_by_epoch.append(
                    self._parameter_delta_l2(
                        critic_epoch_start, critic_parameters
                    )
                )

        actor_loss_sum = 0.0
        action_bound_loss_sum = 0.0
        clip_fraction_sum = 0.0
        importance_ratio_sum = 0.0
        actor_mini_batch_telemetry: list[dict[str, float | int | bool]] = []
        actor_parameter_delta_by_epoch: list[float] = []
        actor_epoch_start: list[torch.Tensor] | None = None
        for epoch_index, mini_batch_index, indices in self._epoch_batches(
            sample_count,
            self.actor_num_mini_batches,
            self.actor_num_learning_epochs,
            self.device,
        ):
            if mini_batch_index == 0:
                actor_epoch_start = self._snapshot_parameters(actor_parameters)
            observations = data["observations"][indices]
            actions = data["actions"][indices]
            self.policy.act(observations)
            log_prob = self.policy.get_actions_log_prob(actions)
            log_ratio = log_prob - data["old_log_prob"][indices].squeeze(-1)
            ratio = torch.exp(log_ratio)
            advantages = data["advantages"][indices].squeeze(-1)
            unclipped = advantages * ratio
            clipped = advantages * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = -torch.minimum(unclipped, clipped).mean()

            mean_action = self.policy.action_mean
            below = torch.clamp_max(mean_action + 1.0, 0.0)
            above = torch.clamp_min(mean_action - 1.0, 0.0)
            action_bound_loss = (
                torch.square(below).sum(dim=-1)
                + torch.square(above).sum(dim=-1)
            ).mean()
            actor_loss = (
                surrogate_loss + self.action_bound_weight * action_bound_loss
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            gradient_l2 = self._gradient_l2(actor_parameters)
            ratio_summary = self._distribution_summary(ratio)
            clip_fraction = float(
                (torch.abs(ratio - 1.0) > self.clip_param)
                .float()
                .mean()
                .detach()
            )
            approximate_kl = float(
                ((ratio - 1.0) - log_ratio).mean().detach()
            )
            entropy = float(self.policy.entropy.mean().detach())
            action_saturation = float(
                (torch.abs(actions) >= 1.0).float().mean().detach()
            )
            mean_action_saturation = float(
                (torch.abs(mean_action) >= 1.0).float().mean().detach()
            )
            self.actor_optimizer.step()

            actor_loss_sum += float(surrogate_loss.detach())
            action_bound_loss_sum += float(action_bound_loss.detach())
            clip_fraction_sum += clip_fraction
            importance_ratio_sum += ratio_summary["mean"]
            self.actor_optimizer_steps += 1
            actor_mini_batch_telemetry.append(
                {
                    "epoch": epoch_index,
                    "mini_batch": mini_batch_index,
                    "surrogate_loss": float(surrogate_loss.detach()),
                    "action_bound_loss": float(action_bound_loss.detach()),
                    "importance_ratio_mean": ratio_summary["mean"],
                    "importance_ratio_min": ratio_summary["min"],
                    "importance_ratio_q05": ratio_summary["q05"],
                    "importance_ratio_q50": ratio_summary["q50"],
                    "importance_ratio_q95": ratio_summary["q95"],
                    "importance_ratio_max": ratio_summary["max"],
                    "clip_fraction": clip_fraction,
                    "approx_kl": approximate_kl,
                    "gradient_l2": gradient_l2,
                    "entropy": entropy,
                    "action_saturation_fraction": action_saturation,
                    "mean_action_saturation_fraction": mean_action_saturation,
                    "all_finite": bool(
                        torch.isfinite(
                            torch.tensor(
                                [
                                    float(surrogate_loss.detach()),
                                    float(action_bound_loss.detach()),
                                    ratio_summary["mean"],
                                    ratio_summary["min"],
                                    ratio_summary["max"],
                                    clip_fraction,
                                    approximate_kl,
                                    gradient_l2,
                                    entropy,
                                    action_saturation,
                                    mean_action_saturation,
                                ],
                                dtype=torch.float64,
                            )
                        )
                        .all()
                        .item()
                    ),
                }
            )
            if mini_batch_index == self.actor_num_mini_batches - 1:
                if actor_epoch_start is None:
                    raise RuntimeError("missing actor epoch parameter snapshot")
                actor_parameter_delta_by_epoch.append(
                    self._parameter_delta_l2(actor_epoch_start, actor_parameters)
                )

        critic_updates = (
            self.critic_num_learning_epochs * self.critic_num_mini_batches
        )
        actor_updates = (
            self.actor_num_learning_epochs * self.actor_num_mini_batches
        )
        with torch.no_grad():
            critic_predictions_after = self.policy.evaluate(
                data["observations"]
            ).squeeze(-1)
            explained_variance_after = self._explained_variance(
                data["returns"], critic_predictions_after
            )
        actor_epoch_telemetry = self._epoch_aggregates(
            actor_mini_batch_telemetry,
            self.actor_num_learning_epochs,
            actor_parameter_delta_by_epoch,
        )
        critic_epoch_telemetry = self._critic_epoch_aggregates(
            critic_mini_batch_telemetry,
            self.critic_num_learning_epochs,
            critic_parameter_delta_by_epoch,
        )
        rollout_telemetry = {
            "returns": self._distribution_summary(data["returns"]),
            "advantages": self._distribution_summary(data["advantages"]),
            "actions": self._distribution_summary(data["actions"]),
            "action_saturation_fraction": float(
                (torch.abs(data["actions"]) >= 1.0).float().mean()
            ),
            "fixed_action_std_mean": float(self.policy.std.detach().mean()),
            "fixed_action_std_min": float(self.policy.std.detach().min()),
            "fixed_action_std_max": float(self.policy.std.detach().max()),
            "explained_variance_before": explained_variance_before,
            "explained_variance_after": explained_variance_after,
            "actor_parameter_delta_l2": self._parameter_delta_l2(
                actor_update_start, actor_parameters
            ),
            "critic_parameter_delta_l2": self._parameter_delta_l2(
                critic_update_start, critic_parameters
            ),
        }
        self.storage.clear()
        normalization_metrics = self.policy.commit_rollout_normalization()
        return {
            "critic": critic_loss_sum / critic_updates,
            "actor_surrogate": actor_loss_sum / actor_updates,
            "action_bound": action_bound_loss_sum / actor_updates,
            "clip_fraction": clip_fraction_sum / actor_updates,
            "importance_ratio": importance_ratio_sum / actor_updates,
            "actor_optimizer_steps_total": self.actor_optimizer_steps,
            "critic_optimizer_steps_total": self.critic_optimizer_steps,
            "preupdate_importance_ratio_mean": preupdate_ratio_mean,
            "preupdate_importance_ratio_max_abs_from_one": (
                preupdate_ratio_max_abs_from_one
            ),
            "preupdate_clip_fraction": preupdate_clip_fraction,
            "telemetry_schema": "official_smp_policy_update_telemetry_v1",
            "actor_epoch_telemetry": actor_epoch_telemetry,
            "actor_mini_batch_telemetry": actor_mini_batch_telemetry,
            "critic_epoch_telemetry": critic_epoch_telemetry,
            "critic_mini_batch_telemetry": critic_mini_batch_telemetry,
            "rollout_telemetry": rollout_telemetry,
            **normalization_metrics,
        }
