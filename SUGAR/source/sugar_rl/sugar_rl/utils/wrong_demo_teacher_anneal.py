"""Official-Refiner adapter for a wrong-reference teacher with global anneal.

The task command continues to use its CarryBox motion.  Only the frozen
official Refiner's seven future-reference observation terms are redirected to
``MotionCommand.teacher_motion``.  No model or action is replaced.  A global
linear schedule then removes the full-body teacher by a declared control step.
"""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.managers import ObservationManager

import sugar_rl.tasks.locomanip.mdp as mdp
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.base_refiner_env_cfg import (
    BaseObservationsCfg,
)
from sugar_rl.utils.official_refiner_nominal_teacher import (
    FrozenOfficialRefinerTeacher,
    OFFICIAL_REFINER_OBSERVATION_DIM,
    OfficialRefinerResidualVecEnvWrapper,
    OfficialTeacherResidualActionTransform,
)


class FrozenOfficialRefinerWrongReferenceTeacher(FrozenOfficialRefinerTeacher):
    """Accepted official Refiner reading only the declared teacher motion."""

    def __init__(self, env, checkpoint) -> None:
        command = env.command_manager.get_term("motion")
        if command.teacher_motion is None:
            raise RuntimeError("wrong-reference teacher motion was not loaded")
        super().__init__(env, checkpoint)
        cfg = BaseObservationsCfg()
        policy = cfg.policy
        policy.enable_corruption = False
        replacements = {
            "joint_pos_vel_future": mdp.teacher_joint_pos_vel_future,
            "motion_anchor_pos_b_future": mdp.teacher_motion_anchor_pos_b_future,
            "motion_anchor_ori_b_future": mdp.teacher_motion_anchor_ori_b_future,
            "ref_obj_pos_b_future": mdp.teacher_obj_motion_pos_future,
            "ref_obj_ori_b_future": mdp.teacher_obj_motion_ori_future,
            "ref_obj_lin_vel_b_future": mdp.teacher_ref_obj_lin_vel_b_future,
            "ref_obj_ang_vel_b_future": mdp.teacher_ref_obj_ang_vel_b_future,
        }
        for name, func in replacements.items():
            getattr(policy, name).func = func
        self.observation_manager = ObservationManager({"policy": policy}, env)
        observation = self.observation_manager.compute()["policy"]
        if tuple(observation.shape) != (
            env.num_envs,
            OFFICIAL_REFINER_OBSERVATION_DIM,
        ):
            raise RuntimeError("wrong-reference official observation geometry drift")
        self.reference_term_names = tuple(replacements)
        self.teacher_motion_folder = str(command.cfg.teacher_motion_folder)

    @torch.inference_mode()
    def frozen_audit(self) -> dict[str, Any]:
        audit = super().frozen_audit()
        audit.update(
            {
                "reference_source": "motion_command.teacher_motion",
                "teacher_motion_folder": self.teacher_motion_folder,
                "redirected_reference_terms": list(self.reference_term_names),
                "all_seven_reference_terms_redirected": len(self.reference_term_names)
                == 7,
            }
        )
        audit["passed"] = bool(
            audit["passed"] and audit["all_seven_reference_terms_redirected"]
        )
        return audit


class GlobalLinearTeacherAnneal:
    """Full-body teacher coefficient shared across resets and environments."""

    mode = "global_linear_schedule"

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        total_control_steps: int,
        final_coefficient: float = 0.0,
    ) -> None:
        if total_control_steps <= 0:
            raise ValueError("teacher anneal requires positive total_control_steps")
        if not 0.0 <= final_coefficient < 1.0:
            raise ValueError("teacher final coefficient must stay in [0, 1)")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.linear_release_steps = int(total_control_steps)
        self.final_coefficient = float(final_coefficient)
        self.release_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.release_progress = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.coefficient = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.global_control_steps = 0

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
            if value.shape != expected or value.dtype is not torch.bool:
                raise ValueError(f"{name} schedule input geometry drift")
        self.global_control_steps += 1
        progress = min(
            self.global_control_steps / float(self.linear_release_steps), 1.0
        )
        coefficient = self.final_coefficient + (
            1.0 - self.final_coefficient
        ) * (1.0 - progress)
        self.coefficient.fill_(coefficient)
        self.release_latched.fill_(self.global_control_steps > 0)
        self.release_progress.fill_(
            min(self.global_control_steps, self.linear_release_steps)
        )
        return self.coefficient.clone()

    @torch.no_grad()
    def audit_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "linear_release_steps": self.linear_release_steps,
            "final_coefficient": self.final_coefficient,
            "global_control_steps": self.global_control_steps,
            "coefficient": self.coefficient.tolist(),
        }


class WrongReferenceScheduledOfficialRefinerResidualVecEnvWrapper(
    OfficialRefinerResidualVecEnvWrapper
):
    """Exact official teacher with wrong reference and full-body annealing."""

    protocol = "sugar_wrong_reference_scheduled_official_refiner_v1"

    def __init__(
        self,
        env,
        checkpoint,
        *,
        residual_scale: float,
        teacher_anneal_control_steps: int,
        teacher_final_coefficient: float = 0.0,
        clip_actions=None,
    ) -> None:
        super().__init__(
            env,
            checkpoint,
            residual_scale=residual_scale,
            release_mode="fixed_one",
            linear_release_steps=teacher_anneal_control_steps,
            teacher_release_scope="full_body",
            support_teacher_mode="advancing",
            drop_grace_steps=0,
            post_release_residual_scale=None,
            teacher_reference_advance_mode="goal_teacher_post_step_once",
            clip_actions=clip_actions,
        )
        self.teacher = FrozenOfficialRefinerWrongReferenceTeacher(
            self.unwrapped, checkpoint
        )
        self.action_transform = OfficialTeacherResidualActionTransform(
            self.teacher, residual_scale=residual_scale
        )
        self.release = GlobalLinearTeacherAnneal(
            self.num_envs,
            self.device,
            total_control_steps=teacher_anneal_control_steps,
            final_coefficient=teacher_final_coefficient,
        )
        self.teacher_anneal_control_steps = int(teacher_anneal_control_steps)
        self.teacher_final_coefficient = float(teacher_final_coefficient)

    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = super().checkpoint_state_dict()
        state.update(
            {
                "teacher_anneal_control_steps": self.teacher_anneal_control_steps,
                "teacher_final_coefficient": self.teacher_final_coefficient,
                "teacher_reference_source": "motion_command.teacher_motion",
                "teacher_motion_folder": self.teacher.teacher_motion_folder,
                "global_schedule_control_steps": self.release.global_control_steps,
            }
        )
        return state

    @torch.inference_mode()
    def load_checkpoint_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("teacher_anneal_control_steps", -1)) != (
            self.teacher_anneal_control_steps
        ):
            raise ValueError("wrong-reference teacher anneal checkpoint drift")
        if float(state.get("teacher_final_coefficient", 0.0)) != (
            self.teacher_final_coefficient
        ):
            raise ValueError("wrong-reference teacher floor checkpoint drift")
        super().load_checkpoint_state_dict(state)
        self.release.global_control_steps = int(
            state["global_schedule_control_steps"]
        )


class WrongReferenceFixedOfficialRefinerResidualVecEnvWrapper(
    OfficialRefinerResidualVecEnvWrapper
):
    """Exact official Refiner reading a declared demo with fixed authority."""

    protocol = "sugar_wrong_reference_fixed_official_refiner_v1"

    def __init__(
        self,
        env,
        checkpoint,
        *,
        residual_scale: float,
        clip_actions=None,
    ) -> None:
        super().__init__(
            env,
            checkpoint,
            residual_scale=residual_scale,
            release_mode="fixed_one",
            linear_release_steps=4,
            teacher_release_scope="full_body",
            support_teacher_mode="advancing",
            drop_grace_steps=0,
            post_release_residual_scale=None,
            teacher_reference_advance_mode="goal_teacher_post_step_once",
            clip_actions=clip_actions,
        )
        self.teacher = FrozenOfficialRefinerWrongReferenceTeacher(
            self.unwrapped, checkpoint
        )
        self.action_transform = OfficialTeacherResidualActionTransform(
            self.teacher, residual_scale=residual_scale
        )

    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = super().checkpoint_state_dict()
        state.update(
            {
                "teacher_reference_source": "motion_command.teacher_motion",
                "teacher_motion_folder": self.teacher.teacher_motion_folder,
                "teacher_authority_contract": "fixed_one",
            }
        )
        return state

    @torch.inference_mode()
    def load_checkpoint_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("teacher_authority_contract") != "fixed_one":
            raise ValueError("wrong-reference fixed-teacher checkpoint drift")
        super().load_checkpoint_state_dict(state)
