# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen accepted-SUGAR Refiner teacher and residual action mapping.

This is adapter glue around the exact official Refiner observation config,
official RSL-RL ``ActorCritic``, and accepted checkpoint.  It does not define
curiosity, train a replacement policy, or modify the official architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
from rsl_rl.modules import ActorCritic

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.managers import ObservationManager
from isaaclab.utils import class_to_dict

from sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.base_refiner_env_cfg import (
    BaseObservationsCfg,
)


ACCEPTED_REFINER_SHA256 = (
    "a398a7293fcea0ef948234e5de47b990fa586d2efd4e54ad7e481151c16124c3"
)
ACCEPTED_REFINER_ITERATION = 10000
OFFICIAL_REFINER_OBSERVATION_DIM = 890
OFFICIAL_REFINER_ACTION_DIM = 29


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TeacherResidualAction:
    """One fully declared residual-action transformation."""

    official_observation: torch.Tensor
    teacher_action: torch.Tensor
    residual_action: torch.Tensor
    teacher_coefficient: torch.Tensor
    residual_scale: torch.Tensor
    executed_action: torch.Tensor


class FrozenOfficialRefinerTeacher:
    """Exact online inference adapter for the accepted official Refiner."""

    def __init__(
        self,
        env,
        checkpoint: str | Path,
        *,
        expected_sha256: str | None = ACCEPTED_REFINER_SHA256,
    ) -> None:
        self.env = env
        self.device = torch.device(env.device)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if expected_sha256 is None:
            self.checkpoint_sha256 = None
        else:
            actual_sha256 = _sha256(self.checkpoint)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "accepted official Refiner checkpoint drift: "
                    f"{actual_sha256} != {expected_sha256}"
                )
            self.checkpoint_sha256 = actual_sha256

        observation_cfg = BaseObservationsCfg()
        observation_cfg.policy.enable_corruption = False
        self.observation_manager = ObservationManager(
            {"policy": observation_cfg.policy}, env
        )
        observation = self.observation_manager.compute()
        policy_observation = observation["policy"]
        if tuple(policy_observation.shape) != (
            env.num_envs,
            OFFICIAL_REFINER_OBSERVATION_DIM,
        ):
            raise RuntimeError(
                "official Refiner observation geometry drift: "
                f"{tuple(policy_observation.shape)}"
            )

        runner_cfg = BasePPORunnerCfg()
        policy_cfg = class_to_dict(runner_cfg.policy)
        class_name = policy_cfg.pop("class_name")
        if class_name != "ActorCritic":
            raise RuntimeError(f"official policy class drift: {class_name}")
        policy_cfg.setdefault("actor_obs_normalization", False)
        policy_cfg.setdefault("critic_obs_normalization", False)
        self.actor = ActorCritic(
            observation,
            {"policy": ["policy"], "critic": ["policy"]},
            OFFICIAL_REFINER_ACTION_DIM,
            **policy_cfg,
        ).to(self.device)
        payload = torch.load(
            self.checkpoint,
            map_location=self.device,
            weights_only=False,
        )
        self.checkpoint_iteration = int(payload["iter"])
        if self.checkpoint_iteration != ACCEPTED_REFINER_ITERATION:
            raise RuntimeError(
                "accepted Refiner iteration drift: "
                f"{self.checkpoint_iteration}"
            )
        self.actor.load_state_dict(payload["model_state_dict"], strict=True)
        self.actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self._initial_state = {
            name: value.detach().clone()
            for name, value in self.actor.state_dict().items()
        }

    @torch.inference_mode()
    def action(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact official policy observation and deterministic mean action."""

        observation = self.observation_manager.compute()["policy"]
        if tuple(observation.shape) != (
            self.env.num_envs,
            OFFICIAL_REFINER_OBSERVATION_DIM,
        ):
            raise RuntimeError("official Refiner observation shape changed")
        if not torch.isfinite(observation).all():
            raise RuntimeError("official Refiner observation is non-finite")
        action = self.actor.act_inference({"policy": observation})
        if tuple(action.shape) != (
            self.env.num_envs,
            OFFICIAL_REFINER_ACTION_DIM,
        ):
            raise RuntimeError("official Refiner action shape changed")
        if not torch.isfinite(action).all():
            raise RuntimeError("official Refiner action is non-finite")
        return observation, action

    @torch.inference_mode()
    def frozen_audit(self) -> dict[str, Any]:
        current = self.actor.state_dict()
        changed = [
            name
            for name, initial in self._initial_state.items()
            if not torch.equal(current[name], initial)
        ]
        requires_grad = [
            name
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad
        ]
        gradients = [
            name
            for name, parameter in self.actor.named_parameters()
            if parameter.grad is not None
        ]
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_iteration": self.checkpoint_iteration,
            "changed_state_keys": changed,
            "requires_grad_parameter_names": requires_grad,
            "gradient_parameter_names": gradients,
            "optimizer_constructed": False,
            "passed": not changed and not requires_grad and not gradients,
        }


class OfficialTeacherResidualActionTransform:
    """Map a causal policy residual into the native SUGAR action.

    PPO samples and records ``residual_action``.  The environment receives
    ``teacher_coefficient * teacher_action + residual_scale * residual_action``.
    Thus a coefficient of one and an exact-zero residual reproduces the
    accepted teacher, while a coefficient of zero releases the teacher and
    leaves the causal residual policy in control.
    """

    def __init__(
        self,
        teacher: FrozenOfficialRefinerTeacher,
        *,
        residual_scale: float,
    ) -> None:
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        self.teacher = teacher
        self.residual_scale = float(residual_scale)

    @torch.inference_mode()
    def apply(
        self,
        residual_action: torch.Tensor,
        teacher_coefficient: float | torch.Tensor,
        residual_scale_override: float | torch.Tensor | None = None,
    ) -> TeacherResidualAction:
        if tuple(residual_action.shape) != (
            self.teacher.env.num_envs,
            OFFICIAL_REFINER_ACTION_DIM,
        ):
            raise ValueError(
                f"residual action shape drift: {tuple(residual_action.shape)}"
            )
        if not torch.isfinite(residual_action).all():
            raise ValueError("residual action is non-finite")
        observation, teacher_action = self.teacher.action()
        coefficient = torch.as_tensor(
            teacher_coefficient,
            device=teacher_action.device,
            dtype=teacher_action.dtype,
        )
        if coefficient.ndim == 0:
            coefficient = coefficient.expand(teacher_action.shape[0])
        if coefficient.ndim == 1:
            coefficient = coefficient.reshape(-1, 1)
        elif coefficient.ndim != 2 or coefficient.shape[1] not in (
            1,
            OFFICIAL_REFINER_ACTION_DIM,
        ):
            raise ValueError(
                "teacher coefficient must be scalar, [env], [env,1], or "
                f"[env,{OFFICIAL_REFINER_ACTION_DIM}]"
            )
        if coefficient.shape[0] != teacher_action.shape[0]:
            raise ValueError("teacher coefficient batch size drift")
        if not torch.isfinite(coefficient).all():
            raise ValueError("teacher coefficient is non-finite")
        if bool(torch.any((coefficient < 0.0) | (coefficient > 1.0))):
            raise ValueError("teacher coefficient must stay within [0, 1]")
        scale_value = (
            self.residual_scale
            if residual_scale_override is None
            else residual_scale_override
        )
        scale = torch.as_tensor(
            scale_value,
            device=teacher_action.device,
            dtype=teacher_action.dtype,
        )
        if scale.ndim == 0:
            scale = scale.expand_as(coefficient)
        elif scale.ndim == 1:
            scale = scale.reshape(-1, 1)
        if scale.shape[0] != teacher_action.shape[0]:
            raise ValueError("residual scale batch size drift")
        if scale.shape[1] not in (1, OFFICIAL_REFINER_ACTION_DIM):
            raise ValueError(
                "residual scale must be scalar, [env], [env,1], or "
                f"[env,{OFFICIAL_REFINER_ACTION_DIM}]"
            )
        if not torch.isfinite(scale).all():
            raise ValueError("residual scale is non-finite")
        if bool(torch.any((scale <= 0.0) | (scale > 1.0))):
            raise ValueError("residual scale must stay within (0, 1]")
        executed = coefficient * teacher_action + scale * residual_action
        if not torch.isfinite(executed).all():
            raise RuntimeError("executed residual action is non-finite")
        return TeacherResidualAction(
            official_observation=observation,
            teacher_action=teacher_action,
            residual_action=residual_action,
            teacher_coefficient=coefficient,
            residual_scale=scale,
            executed_action=executed,
        )


class DirectTacslFailureTeacherRelease:
    """Causal teacher coefficient driven only by the locked failure boundary.

    The caller supplies the direct-TacSL strategy runtime's
    ``initial_strategy_failed`` latch and same-step ``failure_closed`` event
    after an environment transition.  The returned coefficient applies to the
    next action.  Once release starts, teacher authority cannot re-arm before
    an explicit environment reset.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        mode: str,
        linear_release_steps: int = 4,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        if mode not in ("immediate", "linear", "fixed_one"):
            raise ValueError(
                "teacher release mode must be immediate, linear, or fixed_one"
            )
        if int(linear_release_steps) <= 0:
            raise ValueError("linear_release_steps must be positive")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.mode = mode
        self.linear_release_steps = int(linear_release_steps)
        self.release_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.release_progress = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.coefficient = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def update(
        self,
        *,
        initial_strategy_failed: torch.Tensor,
        failure_closed: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.num_envs,)
        for name, value in (
            ("initial_strategy_failed", initial_strategy_failed),
            ("failure_closed", failure_closed),
            ("reset_mask", reset_mask),
        ):
            if value.shape != expected or value.dtype != torch.bool:
                raise ValueError(
                    f"{name} must be bool with shape {expected}, got "
                    f"{value.dtype} {tuple(value.shape)}"
                )
            if value.device != self.device:
                raise ValueError(f"{name} device drift: {value.device}")

        self.release_latched[reset_mask] = False
        self.release_progress[reset_mask] = 0
        self.coefficient[reset_mask] = 1.0

        if self.mode == "fixed_one":
            self.release_latched.zero_()
            self.release_progress.zero_()
            self.coefficient.fill_(1.0)
            return self.coefficient.clone()

        valid = ~reset_mask
        trigger = (
            valid
            & ~self.release_latched
            & initial_strategy_failed
            & failure_closed
        )
        self.release_latched[trigger] = True
        if self.mode == "immediate":
            self.release_progress[self.release_latched] = 1
            self.coefficient[self.release_latched] = 0.0
        else:
            advancing = valid & self.release_latched
            self.release_progress[advancing] += 1
            progress = self.release_progress.to(torch.float32)
            self.coefficient = torch.where(
                self.release_latched,
                torch.clamp(
                    1.0 - progress / float(self.linear_release_steps),
                    min=0.0,
                    max=1.0,
                ),
                torch.ones_like(self.coefficient),
            )
        if not torch.isfinite(self.coefficient).all():
            raise RuntimeError("teacher coefficient became non-finite")
        if bool(
            torch.any(
                (self.coefficient < 0.0) | (self.coefficient > 1.0)
            )
        ):
            raise RuntimeError("teacher coefficient left [0,1]")
        return self.coefficient.clone()

    @torch.no_grad()
    def audit_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "linear_release_steps": self.linear_release_steps,
            "release_latched": self.release_latched.tolist(),
            "release_progress": self.release_progress.tolist(),
            "coefficient": self.coefficient.tolist(),
        }


@dataclass(frozen=True)
class ResidualRuntimeStep:
    """Auditable routing record for one real environment control step."""

    residual_action: torch.Tensor
    teacher_action: torch.Tensor
    teacher_coefficient: torch.Tensor
    residual_scale: torch.Tensor
    executed_action: torch.Tensor
    action_manager_raw_action: torch.Tensor
    processed_joint_target: torch.Tensor
    initial_strategy_failed: torch.Tensor
    failure_closed: torch.Tensor
    reset_mask: torch.Tensor
    next_teacher_coefficient: torch.Tensor
    teacher_reference_frame: torch.Tensor
    next_teacher_reference_frame: torch.Tensor
    advancing_teacher_action: torch.Tensor
    support_hold_valid: torch.Tensor
    next_support_hold_valid: torch.Tensor
    support_hold_action: torch.Tensor
    next_support_hold_action: torch.Tensor
    support_hold_trigger: torch.Tensor
    support_hold_trigger_control_step: torch.Tensor


@dataclass(frozen=True)
class BoundedDropGraceStep:
    """Raw and effective drop termination state for one control step."""

    episode_step: torch.Tensor
    raw_dropped_after_lift: torch.Tensor
    window_started: torch.Tensor
    window_active: torch.Tensor
    raw_drop_suppressed: torch.Tensor
    effective_dropped_after_lift: torch.Tensor


class OfficialRefinerResidualVecEnvWrapper(RslRlVecEnvWrapper):
    """Execute frozen-teacher plus causal residual while PPO stores residual.

    The RSL algorithm calls ``act`` and stores/log-scores its sampled residual
    before this wrapper is entered.  Only ``step`` maps that residual to the
    native SUGAR action.  The independently learned ICM remains outside this
    wrapper and must be passed the actual applied action recovered from the
    environment.
    """

    protocol = "sugar_official_refiner_residual_vecenv_v1"

    def __init__(
        self,
        env,
        checkpoint: str | Path,
        *,
        residual_scale: float,
        release_mode: str,
        linear_release_steps: int = 4,
        teacher_release_scope: str = "full_body",
        support_teacher_mode: str = "advancing",
        drop_grace_steps: int = 0,
        post_release_residual_scale: float | None = None,
        teacher_reference_advance_mode: str = "legacy_pre_step",
        clip_actions: float | None = None,
    ) -> None:
        super().__init__(env, clip_actions=clip_actions)
        if self.num_actions != OFFICIAL_REFINER_ACTION_DIM:
            raise RuntimeError(
                f"official residual wrapper requires 29 actions, got "
                f"{self.num_actions}"
            )
        self.teacher = FrozenOfficialRefinerTeacher(
            self.unwrapped, checkpoint
        )
        self.action_transform = OfficialTeacherResidualActionTransform(
            self.teacher, residual_scale=residual_scale
        )
        if post_release_residual_scale is not None and not (
            residual_scale <= post_release_residual_scale <= 1.0
        ):
            raise ValueError(
                "post_release_residual_scale must be within "
                "[residual_scale, 1]"
            )
        self.post_release_residual_scale = (
            None
            if post_release_residual_scale is None
            else float(post_release_residual_scale)
        )
        if teacher_reference_advance_mode not in (
            "legacy_pre_step",
            "command_manager_only",
            "goal_teacher_post_step_once",
        ):
            raise ValueError(
                "teacher_reference_advance_mode must be legacy_pre_step, "
                "command_manager_only, or goal_teacher_post_step_once"
            )
        self.teacher_reference_advance_mode = (
            teacher_reference_advance_mode
        )
        self.reference_advance_nonreset_env_steps = 0
        self.reference_advance_nonreset_exact = True
        self.reference_advance_reset_env_steps = 0
        self.reference_advance_delta_counts: dict[int, int] = {}
        self.release = DirectTacslFailureTeacherRelease(
            self.num_envs,
            self.device,
            mode=release_mode,
            linear_release_steps=linear_release_steps,
        )
        if teacher_release_scope not in ("full_body", "arm_only"):
            raise ValueError(
                "teacher_release_scope must be full_body or arm_only"
            )
        self.teacher_release_scope = teacher_release_scope
        action_term = self.unwrapped.action_manager.get_term(
            "JointPositionAction"
        )
        self.teacher_joint_names = tuple(action_term._joint_names)
        if (
            len(self.teacher_joint_names) != OFFICIAL_REFINER_ACTION_DIM
            or len(set(self.teacher_joint_names))
            != OFFICIAL_REFINER_ACTION_DIM
        ):
            raise RuntimeError(
                "official teacher action term must expose 29 unique joints"
            )
        support_tokens = ("hip", "knee", "ankle", "waist")
        arm_tokens = ("shoulder", "elbow", "wrist")
        self.teacher_support_indices = tuple(
            index
            for index, name in enumerate(self.teacher_joint_names)
            if any(token in name for token in support_tokens)
        )
        self.teacher_manipulation_indices = tuple(
            index
            for index, name in enumerate(self.teacher_joint_names)
            if any(token in name for token in arm_tokens)
        )
        if (
            not self.teacher_support_indices
            or not self.teacher_manipulation_indices
            or set(self.teacher_support_indices)
            & set(self.teacher_manipulation_indices)
            or set(self.teacher_support_indices)
            | set(self.teacher_manipulation_indices)
            != set(range(OFFICIAL_REFINER_ACTION_DIM))
        ):
            raise RuntimeError(
                "named support/manipulation partition is incomplete"
            )
        partition_text = "\n".join(
            f"{index}:{name}:"
            f"{'support' if index in self.teacher_support_indices else 'arm'}"
            for index, name in enumerate(self.teacher_joint_names)
        )
        self.teacher_partition_sha256 = hashlib.sha256(
            partition_text.encode("utf-8")
        ).hexdigest()
        if support_teacher_mode not in (
            "advancing",
            "failure_latched",
        ):
            raise ValueError(
                "support_teacher_mode must be advancing or failure_latched"
            )
        if (
            support_teacher_mode == "failure_latched"
            and teacher_release_scope != "arm_only"
        ):
            raise ValueError(
                "failure-latched support requires arm-only teacher release"
            )
        self.support_teacher_mode = support_teacher_mode
        self._support_hold_action = torch.zeros(
            self.num_envs,
            len(self.teacher_support_indices),
            dtype=torch.float32,
            device=self.device,
        )
        self._support_hold_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._support_hold_trigger_control_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._support_hold_counts = {
            "triggers": 0,
            "applied_env_steps": 0,
            "reset_rearms": 0,
        }
        self.latest_step: ResidualRuntimeStep | None = None
        self.control_steps = 0
        self.drop_grace_steps = int(drop_grace_steps)
        if self.drop_grace_steps < 0:
            raise ValueError("drop_grace_steps must be nonnegative")
        self.latest_drop_grace: BoundedDropGraceStep | None = None
        self._drop_grace_window_started = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._drop_grace_start_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._drop_grace_counts = {
            "evaluation_calls": 0,
            "window_starts": 0,
            "raw_true": 0,
            "window_active": 0,
            "suppressed": 0,
            "effective_true": 0,
            "resets_while_active": 0,
        }
        self._termination_identities_before = (
            self._termination_identities()
        )
        self._termination_identities_after = (
            self._termination_identities_before
        )
        if self.drop_grace_steps > 0:
            self._install_bounded_drop_grace()

    @torch.no_grad()
    def _applied_teacher_coefficient(
        self, scalar: torch.Tensor
    ) -> torch.Tensor:
        if scalar.shape != (self.num_envs,):
            raise ValueError("teacher release scalar shape drift")
        if self.teacher_release_scope == "full_body":
            return scalar.reshape(-1, 1)
        coefficient = torch.ones(
            self.num_envs,
            OFFICIAL_REFINER_ACTION_DIM,
            dtype=scalar.dtype,
            device=scalar.device,
        )
        manipulation_indices = {
            int(index) for index in self.teacher_manipulation_indices
        }
        manipulation_mask = torch.tensor(
            [
                index in manipulation_indices
                for index in range(OFFICIAL_REFINER_ACTION_DIM)
            ],
            dtype=torch.bool,
            device=scalar.device,
        ).reshape(1, -1)
        return torch.where(
            manipulation_mask,
            scalar.reshape(-1, 1),
            coefficient,
        )

    @torch.no_grad()
    def _applied_residual_scale(
        self, teacher_coefficient: torch.Tensor
    ) -> torch.Tensor:
        """Return causal per-joint residual authority.

        The legacy contract remains an exact fixed scale.  The optional
        successor keeps support authority fixed at that scale and expands only
        released manipulation joints toward native policy authority.
        """

        if tuple(teacher_coefficient.shape) not in {
            (self.num_envs, 1),
            (self.num_envs, OFFICIAL_REFINER_ACTION_DIM),
        }:
            raise ValueError("teacher coefficient shape drift for scale routing")
        base = torch.full(
            (self.num_envs, OFFICIAL_REFINER_ACTION_DIM),
            self.action_transform.residual_scale,
            dtype=teacher_coefficient.dtype,
            device=teacher_coefficient.device,
        )
        if self.post_release_residual_scale is None:
            return base
        if self.teacher_release_scope != "arm_only":
            raise RuntimeError(
                "post-release native authority requires arm-only release"
            )
        manipulation_indices = {
            int(index) for index in self.teacher_manipulation_indices
        }
        manipulation_mask = torch.tensor(
            [
                index in manipulation_indices
                for index in range(OFFICIAL_REFINER_ACTION_DIM)
            ],
            dtype=torch.bool,
            device=teacher_coefficient.device,
        ).reshape(1, -1)
        released_scale = (
            self.action_transform.residual_scale
            + (1.0 - teacher_coefficient)
            * (
                self.post_release_residual_scale
                - self.action_transform.residual_scale
            )
        )
        return torch.where(manipulation_mask, released_scale, base)

    @torch.no_grad()
    def teacher_partition_audit_state(self) -> dict[str, Any]:
        return {
            "scope": self.teacher_release_scope,
            "support_teacher_mode": self.support_teacher_mode,
            "ordered_joint_names": list(self.teacher_joint_names),
            "support_indices": list(self.teacher_support_indices),
            "support_joint_names": [
                self.teacher_joint_names[index]
                for index in self.teacher_support_indices
            ],
            "manipulation_indices": list(
                self.teacher_manipulation_indices
            ),
            "manipulation_joint_names": [
                self.teacher_joint_names[index]
                for index in self.teacher_manipulation_indices
            ],
            "partition_sha256": self.teacher_partition_sha256,
            "complete_disjoint_partition": (
                set(self.teacher_support_indices)
                | set(self.teacher_manipulation_indices)
                == set(range(OFFICIAL_REFINER_ACTION_DIM))
                and not (
                    set(self.teacher_support_indices)
                    & set(self.teacher_manipulation_indices)
                )
            ),
        }

    @torch.no_grad()
    def teacher_reference_advance_audit_state(self) -> dict[str, Any]:
        return {
            "mode": self.teacher_reference_advance_mode,
            "nonreset_environment_steps": (
                self.reference_advance_nonreset_env_steps
            ),
            "nonreset_exactly_one_native_frame": (
                self.reference_advance_nonreset_exact
            ),
            "reset_environment_steps": (
                self.reference_advance_reset_env_steps
            ),
            "nonreset_delta_counts": {
                str(key): value
                for key, value in sorted(
                    self.reference_advance_delta_counts.items()
                )
            },
        }

    @torch.no_grad()
    def _effective_teacher_action(
        self, advancing_teacher_action: torch.Tensor
    ) -> torch.Tensor:
        if tuple(advancing_teacher_action.shape) != (
            self.num_envs,
            OFFICIAL_REFINER_ACTION_DIM,
        ):
            raise ValueError("advancing teacher action shape drift")
        if (
            self.support_teacher_mode == "advancing"
            or not bool(self._support_hold_valid.any())
        ):
            return advancing_teacher_action
        effective = advancing_teacher_action.clone()
        valid = self._support_hold_valid
        support_columns = torch.as_tensor(
            self.teacher_support_indices,
            dtype=torch.long,
            device=effective.device,
        )
        valid_envs = torch.nonzero(valid, as_tuple=False).reshape(-1)
        effective[
            valid_envs[:, None], support_columns[None, :]
        ] = self._support_hold_action[valid_envs]
        return effective

    @torch.no_grad()
    def support_hold_audit_state(self) -> dict[str, Any]:
        return {
            "mode": self.support_teacher_mode,
            "valid": self._support_hold_valid.detach().cpu().tolist(),
            "action": self._support_hold_action.detach().cpu().tolist(),
            "trigger_control_step": (
                self._support_hold_trigger_control_step.detach().cpu().tolist()
            ),
            "counts": dict(self._support_hold_counts),
        }

    def _termination_identities(self) -> dict[str, dict[str, Any]]:
        manager = self.unwrapped.termination_manager
        return {
            name: {
                "func": (
                    f"{manager.get_term_cfg(name).func.__module__}."
                    f"{manager.get_term_cfg(name).func.__qualname__}"
                ),
                "params_repr": repr(manager.get_term_cfg(name).params),
                "time_out": bool(manager.get_term_cfg(name).time_out),
            }
            for name in manager.active_terms
        }

    def _install_bounded_drop_grace(self) -> None:
        manager = self.unwrapped.termination_manager
        if "dropped_after_lift" not in manager.active_terms:
            raise RuntimeError(
                "bounded drop grace requires dropped_after_lift termination"
            )
        original_cfg = manager.get_term_cfg("dropped_after_lift")
        original_func = original_cfg.func

        def bounded_dropped_after_lift(env_instance, **params):
            raw = original_func(env_instance, **params).to(dtype=torch.bool)
            episode_step = (
                env_instance.episode_length_buf.detach().clone()
            )
            new_window = raw & ~self._drop_grace_window_started
            self._drop_grace_window_started[new_window] = True
            self._drop_grace_start_episode_step[new_window] = (
                episode_step[new_window]
            )
            window_age = (
                episode_step - self._drop_grace_start_episode_step
            )
            active = (
                self._drop_grace_window_started
                & (window_age >= 0)
                & (window_age < self.drop_grace_steps)
            )
            suppressed = raw & active
            effective = raw & ~active
            self.latest_drop_grace = BoundedDropGraceStep(
                episode_step=episode_step,
                raw_dropped_after_lift=raw.detach().clone(),
                window_started=(
                    self._drop_grace_window_started.detach().clone()
                ),
                window_active=active.detach().clone(),
                raw_drop_suppressed=suppressed.detach().clone(),
                effective_dropped_after_lift=effective.detach().clone(),
            )
            self._drop_grace_counts["evaluation_calls"] += 1
            self._drop_grace_counts["window_starts"] += int(
                new_window.sum()
            )
            self._drop_grace_counts["raw_true"] += int(raw.sum())
            self._drop_grace_counts["window_active"] += int(active.sum())
            self._drop_grace_counts["suppressed"] += int(
                suppressed.sum()
            )
            self._drop_grace_counts["effective_true"] += int(
                effective.sum()
            )
            return effective

        grace_cfg = manager.get_term_cfg("dropped_after_lift")
        grace_cfg.func = bounded_dropped_after_lift
        manager.set_term_cfg("dropped_after_lift", grace_cfg)
        self._termination_identities_after = self._termination_identities()

    @torch.no_grad()
    def _advance_teacher_reference(
        self,
        env_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        command = self.unwrapped.command_manager.get_term("motion")
        before = command.time_steps.detach().clone()
        motion_lengths = command.motion.time_step_total_permotion[
            command.motion_id
        ].to(device=self.device, dtype=torch.long)
        after = torch.minimum(
            before + 1, torch.clamp(motion_lengths - 1, min=0)
        )
        if env_mask is not None:
            if (
                env_mask.dtype != torch.bool
                or tuple(env_mask.shape) != (self.num_envs,)
            ):
                raise ValueError(
                    "teacher-reference advance mask shape/dtype drift"
                )
            after = torch.where(env_mask, after, before)
        command.time_steps.copy_(after)
        return before, after

    @staticmethod
    def _runtime_failure_fields(base_env) -> tuple[torch.Tensor, torch.Tensor]:
        entry = getattr(
            base_env, "_sugar_goal_tactile_strategy_runtime", None
        )
        if entry is None or not isinstance(entry, tuple) or len(entry) != 2:
            raise RuntimeError(
                "goal direct-TacSL strategy runtime was not constructed"
            )
        output = entry[1].update()
        initial = output["strategy/initial_strategy_failed"]
        closed = output["strategy/failure_closed"]
        if initial.dtype != torch.bool or closed.dtype != torch.bool:
            raise RuntimeError("failure-release runtime field dtype drift")
        return initial.detach().clone(), closed.detach().clone()

    def step(self, residual_action: torch.Tensor):
        scalar_coefficient = self.release.coefficient.detach().clone()
        coefficient = self._applied_teacher_coefficient(
            scalar_coefficient
        )
        residual_scale = self._applied_residual_scale(coefficient)
        transformed = self.action_transform.apply(
            residual_action,
            coefficient,
            residual_scale_override=residual_scale,
        )
        advancing_teacher_action = transformed.teacher_action
        support_hold_valid = self._support_hold_valid.detach().clone()
        support_hold_action = self._support_hold_action.detach().clone()
        effective_teacher_action = self._effective_teacher_action(
            advancing_teacher_action
        )
        if effective_teacher_action is not advancing_teacher_action:
            executed_action = (
                transformed.teacher_coefficient
                * effective_teacher_action
                + transformed.residual_scale
                * transformed.residual_action
            )
            if not torch.isfinite(executed_action).all():
                raise RuntimeError(
                    "failure-latched executed action is non-finite"
                )
            transformed = TeacherResidualAction(
                official_observation=transformed.official_observation,
                teacher_action=effective_teacher_action,
                residual_action=transformed.residual_action,
                teacher_coefficient=transformed.teacher_coefficient,
                residual_scale=transformed.residual_scale,
                executed_action=executed_action,
            )
        self._support_hold_counts["applied_env_steps"] += int(
            support_hold_valid.sum()
        )
        if self.teacher_reference_advance_mode == "legacy_pre_step":
            reference_before, reference_after = (
                self._advance_teacher_reference()
            )
        else:
            command = self.unwrapped.command_manager.get_term("motion")
            reference_before = command.time_steps.detach().clone()
            reference_after = reference_before.clone()
        observations, rewards, dones, extras = super().step(
            transformed.executed_action
        )
        if self.teacher_reference_advance_mode == "command_manager_only":
            command = self.unwrapped.command_manager.get_term("motion")
            reference_after = command.time_steps.detach().clone()
        elif (
            self.teacher_reference_advance_mode
            == "goal_teacher_post_step_once"
        ):
            _, reference_after = self._advance_teacher_reference(
                ~dones.to(dtype=torch.bool)
            )
        reset_mask = dones.to(dtype=torch.bool)
        if self.teacher_reference_advance_mode in (
            "command_manager_only",
            "goal_teacher_post_step_once",
        ):
            nonreset = ~reset_mask
            delta = reference_after - reference_before
            self.reference_advance_nonreset_env_steps += int(
                nonreset.sum()
            )
            self.reference_advance_reset_env_steps += int(reset_mask.sum())
            if bool(nonreset.any()):
                unique_delta, unique_count = torch.unique(
                    delta[nonreset], return_counts=True
                )
                for value, count in zip(
                    unique_delta.detach().cpu().tolist(),
                    unique_count.detach().cpu().tolist(),
                    strict=True,
                ):
                    key = int(value)
                    self.reference_advance_delta_counts[key] = (
                        self.reference_advance_delta_counts.get(key, 0)
                        + int(count)
                    )
                self.reference_advance_nonreset_exact &= bool(
                    torch.all(delta[nonreset] == 1)
                )
        initial_failed, failure_closed = self._runtime_failure_fields(
            self.unwrapped
        )
        support_hold_trigger = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.support_teacher_mode == "failure_latched":
            support_hold_trigger = (
                ~reset_mask
                & ~self._support_hold_valid
                & initial_failed
                & failure_closed
            )
            reset_rearms = reset_mask & self._support_hold_valid
            self._support_hold_counts["reset_rearms"] += int(
                reset_rearms.sum()
            )
            self._support_hold_valid[reset_mask] = False
            self._support_hold_action[reset_mask] = 0.0
            self._support_hold_trigger_control_step[reset_mask] = -1
            if bool(support_hold_trigger.any()):
                support_columns = torch.as_tensor(
                    self.teacher_support_indices,
                    dtype=torch.long,
                    device=self.device,
                )
                self._support_hold_action[support_hold_trigger] = (
                    advancing_teacher_action[
                        support_hold_trigger
                    ][:, support_columns]
                )
                self._support_hold_valid[support_hold_trigger] = True
                self._support_hold_trigger_control_step[
                    support_hold_trigger
                ] = self.control_steps
                self._support_hold_counts["triggers"] += int(
                    support_hold_trigger.sum()
                )
        next_coefficient = self.release.update(
            initial_strategy_failed=initial_failed,
            failure_closed=failure_closed,
            reset_mask=reset_mask,
        )
        action_term = self.unwrapped.action_manager.get_term(
            "JointPositionAction"
        )
        self.latest_step = ResidualRuntimeStep(
            residual_action=transformed.residual_action.detach().clone(),
            teacher_action=transformed.teacher_action.detach().clone(),
            teacher_coefficient=(
                transformed.teacher_coefficient.detach().clone()
            ),
            residual_scale=transformed.residual_scale.detach().clone(),
            executed_action=transformed.executed_action.detach().clone(),
            action_manager_raw_action=(
                self.unwrapped.action_manager.action.detach().clone()
            ),
            processed_joint_target=(
                action_term.processed_actions.detach().clone()
            ),
            initial_strategy_failed=initial_failed,
            failure_closed=failure_closed,
            reset_mask=reset_mask.detach().clone(),
            next_teacher_coefficient=next_coefficient.detach().clone(),
            teacher_reference_frame=reference_before,
            next_teacher_reference_frame=reference_after,
            advancing_teacher_action=(
                advancing_teacher_action.detach().clone()
            ),
            support_hold_valid=support_hold_valid,
            next_support_hold_valid=(
                self._support_hold_valid.detach().clone()
            ),
            support_hold_action=support_hold_action,
            next_support_hold_action=(
                self._support_hold_action.detach().clone()
            ),
            support_hold_trigger=support_hold_trigger.detach().clone(),
            support_hold_trigger_control_step=(
                self._support_hold_trigger_control_step.detach().clone()
            ),
        )
        if self.drop_grace_steps > 0:
            if self.latest_drop_grace is None:
                raise RuntimeError(
                    "bounded drop grace omitted its termination ledger"
                )
            active_at_reset = (
                reset_mask & self.latest_drop_grace.window_active
            )
            self._drop_grace_counts["resets_while_active"] += int(
                active_at_reset.sum()
            )
            self._drop_grace_window_started[reset_mask] = False
            self._drop_grace_start_episode_step[reset_mask] = -1
        self.control_steps += 1
        return observations, rewards, dones, extras

    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = {
            "protocol": self.protocol,
            "teacher_checkpoint_sha256": self.teacher.checkpoint_sha256,
            "residual_scale": self.action_transform.residual_scale,
            "post_release_residual_scale": (
                self.post_release_residual_scale
            ),
            "release_mode": self.release.mode,
            "linear_release_steps": self.release.linear_release_steps,
            "release_latched": self.release.release_latched.detach().clone(),
            "release_progress": self.release.release_progress.detach().clone(),
            "teacher_coefficient": self.release.coefficient.detach().clone(),
            "control_steps": int(self.control_steps),
        }
        if self.teacher_reference_advance_mode != "legacy_pre_step":
            state.update(
                {
                    "teacher_reference_advance_mode": (
                        self.teacher_reference_advance_mode
                    ),
                    "reference_advance_nonreset_env_steps": (
                        self.reference_advance_nonreset_env_steps
                    ),
                    "reference_advance_nonreset_exact": (
                        self.reference_advance_nonreset_exact
                    ),
                    "reference_advance_reset_env_steps": (
                        self.reference_advance_reset_env_steps
                    ),
                    "reference_advance_delta_counts": dict(
                        self.reference_advance_delta_counts
                    ),
                }
            )
        if self.teacher_release_scope != "full_body":
            state.update(
                {
                    "teacher_release_scope": self.teacher_release_scope,
                    "teacher_joint_names": self.teacher_joint_names,
                    "teacher_support_indices": (
                        self.teacher_support_indices
                    ),
                    "teacher_manipulation_indices": (
                        self.teacher_manipulation_indices
                    ),
                    "teacher_partition_sha256": (
                        self.teacher_partition_sha256
                    ),
                    "applied_teacher_coefficient": (
                        self._applied_teacher_coefficient(
                            self.release.coefficient
                        )
                        .detach()
                        .clone()
                    ),
                }
            )
        if self.support_teacher_mode != "advancing":
            state.update(
                {
                    "support_teacher_mode": self.support_teacher_mode,
                    "support_hold_action": (
                        self._support_hold_action.detach().clone()
                    ),
                    "support_hold_valid": (
                        self._support_hold_valid.detach().clone()
                    ),
                    "support_hold_trigger_control_step": (
                        self._support_hold_trigger_control_step.detach().clone()
                    ),
                    "support_hold_counts": dict(self._support_hold_counts),
                }
            )
        if self.drop_grace_steps > 0:
            state.update(
                {
                    "drop_grace_steps": self.drop_grace_steps,
                    "drop_grace_window_started": (
                        self._drop_grace_window_started.detach().clone()
                    ),
                    "drop_grace_start_episode_step": (
                        self._drop_grace_start_episode_step.detach().clone()
                    ),
                    "drop_grace_counts": dict(self._drop_grace_counts),
                }
            )
        return state

    @torch.inference_mode()
    def load_checkpoint_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "protocol": self.protocol,
            "teacher_checkpoint_sha256": self.teacher.checkpoint_sha256,
            "residual_scale": self.action_transform.residual_scale,
            "post_release_residual_scale": (
                self.post_release_residual_scale
            ),
            "release_mode": self.release.mode,
            "linear_release_steps": self.release.linear_release_steps,
            "teacher_reference_advance_mode": (
                self.teacher_reference_advance_mode
            ),
        }
        state = dict(state)
        state.setdefault(
            "teacher_reference_advance_mode", "legacy_pre_step"
        )
        drift = {
            name: {"actual": state.get(name), "expected": value}
            for name, value in expected.items()
            if state.get(name) != value
        }
        if drift:
            raise ValueError(
                f"official residual-wrapper checkpoint drift: {drift}"
            )
        checkpoint_drop_grace_steps = int(
            state.get("drop_grace_steps", 0)
        )
        if checkpoint_drop_grace_steps != self.drop_grace_steps:
            raise ValueError(
                "official residual-wrapper drop-grace checkpoint drift: "
                f"{checkpoint_drop_grace_steps} != {self.drop_grace_steps}"
            )
        checkpoint_scope = state.get(
            "teacher_release_scope", "full_body"
        )
        if checkpoint_scope != self.teacher_release_scope:
            raise ValueError(
                "official residual-wrapper teacher-release scope drift: "
                f"{checkpoint_scope} != {self.teacher_release_scope}"
            )
        checkpoint_support_mode = state.get(
            "support_teacher_mode", "advancing"
        )
        if checkpoint_support_mode != self.support_teacher_mode:
            raise ValueError(
                "official residual-wrapper support-teacher mode drift: "
                f"{checkpoint_support_mode} != {self.support_teacher_mode}"
            )
        tensors = (
            ("release_latched", self.release.release_latched),
            ("release_progress", self.release.release_progress),
            ("teacher_coefficient", self.release.coefficient),
        )
        for name, target in tensors:
            value = state[name].to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(
                    f"residual-wrapper checkpoint {name} shape drift"
                )
            target.copy_(value)
        self.control_steps = int(state["control_steps"])
        if self.control_steps < 0:
            raise ValueError("negative residual-wrapper control-step count")
        if self.teacher_reference_advance_mode != "legacy_pre_step":
            self.reference_advance_nonreset_env_steps = int(
                state["reference_advance_nonreset_env_steps"]
            )
            self.reference_advance_nonreset_exact = bool(
                state["reference_advance_nonreset_exact"]
            )
            self.reference_advance_reset_env_steps = int(
                state["reference_advance_reset_env_steps"]
            )
            self.reference_advance_delta_counts = {
                int(key): int(value)
                for key, value in state[
                    "reference_advance_delta_counts"
                ].items()
            }
            if (
                self.reference_advance_nonreset_env_steps < 0
                or self.reference_advance_reset_env_steps < 0
                or any(
                    count < 0
                    for count in self.reference_advance_delta_counts.values()
                )
            ):
                raise ValueError(
                    "negative teacher-reference advance accounting"
                )
        if self.drop_grace_steps > 0:
            grace_tensors = (
                (
                    "drop_grace_window_started",
                    self._drop_grace_window_started,
                ),
                (
                    "drop_grace_start_episode_step",
                    self._drop_grace_start_episode_step,
                ),
            )
            for name, target in grace_tensors:
                value = state[name].to(
                    device=target.device, dtype=target.dtype
                )
                if value.shape != target.shape:
                    raise ValueError(
                        f"residual-wrapper checkpoint {name} shape drift"
                    )
                target.copy_(value)
            counts = {
                name: int(value)
                for name, value in state["drop_grace_counts"].items()
            }
            if set(counts) != set(self._drop_grace_counts):
                raise ValueError(
                    "residual-wrapper drop-grace counter schema drift"
                )
            if any(value < 0 for value in counts.values()):
                raise ValueError("negative drop-grace checkpoint counter")
            self._drop_grace_counts = counts
        if self.teacher_release_scope != "full_body":
            expected_partition = {
                "teacher_joint_names": self.teacher_joint_names,
                "teacher_support_indices": self.teacher_support_indices,
                "teacher_manipulation_indices": (
                    self.teacher_manipulation_indices
                ),
                "teacher_partition_sha256": self.teacher_partition_sha256,
            }
            drift = {
                name: {
                    "actual": state.get(name),
                    "expected": value,
                }
                for name, value in expected_partition.items()
                if state.get(name) != value
            }
            if drift:
                raise ValueError(
                    f"teacher joint-partition checkpoint drift: {drift}"
                )
            expected_applied = self._applied_teacher_coefficient(
                self.release.coefficient
            )
            applied = state["applied_teacher_coefficient"].to(
                device=expected_applied.device,
                dtype=expected_applied.dtype,
            )
            if not torch.equal(applied, expected_applied):
                raise ValueError(
                    "per-joint teacher coefficient checkpoint drift"
                )
        if self.support_teacher_mode != "advancing":
            support_tensors = (
                ("support_hold_action", self._support_hold_action),
                ("support_hold_valid", self._support_hold_valid),
                (
                    "support_hold_trigger_control_step",
                    self._support_hold_trigger_control_step,
                ),
            )
            for name, target in support_tensors:
                value = state[name].to(
                    device=target.device, dtype=target.dtype
                )
                if value.shape != target.shape:
                    raise ValueError(
                        f"support-hold checkpoint {name} shape drift"
                    )
                target.copy_(value)
            counts = {
                name: int(value)
                for name, value in state["support_hold_counts"].items()
            }
            if set(counts) != set(self._support_hold_counts):
                raise ValueError(
                    "support-hold checkpoint counter schema drift"
                )
            if any(value < 0 for value in counts.values()):
                raise ValueError("negative support-hold checkpoint counter")
            self._support_hold_counts = counts

    @torch.no_grad()
    def drop_grace_audit_state(self) -> dict[str, Any]:
        before = self._termination_identities_before
        after = self._termination_identities_after
        unchanged_non_drop_terms = all(
            before[name] == after[name]
            for name in before
            if name != "dropped_after_lift"
        )
        drop_config_unchanged = (
            before.get("dropped_after_lift", {}).get("params_repr")
            == after.get("dropped_after_lift", {}).get("params_repr")
            and before.get("dropped_after_lift", {}).get("time_out")
            == after.get("dropped_after_lift", {}).get("time_out")
        )
        return {
            "enabled": self.drop_grace_steps > 0,
            "drop_grace_steps": self.drop_grace_steps,
            "counts": dict(self._drop_grace_counts),
            "window_started": (
                self._drop_grace_window_started.detach().cpu().tolist()
            ),
            "start_episode_step": (
                self._drop_grace_start_episode_step.detach().cpu().tolist()
            ),
            "termination_identities_before": before,
            "termination_identities_after": after,
            "unchanged_non_drop_terms": unchanged_non_drop_terms,
            "drop_config_unchanged": drop_config_unchanged,
        }
