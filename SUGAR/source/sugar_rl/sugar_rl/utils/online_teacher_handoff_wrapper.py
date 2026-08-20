# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL wrapper that executes the official Refiner before live handoff."""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from sugar_rl.utils.official_refiner_nominal_teacher import (
    FrozenOfficialRefinerTeacher,
)


class OnlineTeacherHandoffVecEnvWrapper(RslRlVecEnvWrapper):
    """Replace policy actions only until the physical lift gate is satisfied."""

    def __init__(self, env, *, clip_actions: float, teacher_checkpoint: str | Path):
        super().__init__(env, clip_actions=clip_actions)
        self.base_env = env.unwrapped
        self.teacher = FrozenOfficialRefinerTeacher(
            self.base_env,
            teacher_checkpoint,
            expected_sha256=None,
        )
        self.last_policy_action = None
        self.last_teacher_action = None
        self.last_executed_action = None
        self.last_teacher_control_mask = None
        self.cumulative_teacher_control_steps = torch.zeros(
            self.base_env.num_envs,
            dtype=torch.long,
            device=self.base_env.device,
        )
        self.cumulative_policy_control_steps = torch.zeros_like(
            self.cumulative_teacher_control_steps
        )
        self.base_env._online_teacher_handoff_wrapper = self

    @torch.inference_mode()
    def step(self, actions: torch.Tensor):
        controller = self.base_env._online_teacher_handoff_controller
        teacher_control = ~controller.handoff_active
        _, teacher_action = self.teacher.action()
        executed = torch.where(teacher_control[:, None], teacher_action, actions)
        self.last_policy_action = actions.detach().clone()
        self.last_teacher_action = teacher_action.detach().clone()
        self.last_executed_action = executed.detach().clone()
        self.last_teacher_control_mask = teacher_control.detach().clone()
        self.cumulative_teacher_control_steps += teacher_control.to(torch.long)
        self.cumulative_policy_control_steps += (~teacher_control).to(torch.long)
        return super().step(executed)
