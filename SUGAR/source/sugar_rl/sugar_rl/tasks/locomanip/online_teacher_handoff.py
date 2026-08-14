# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Live official-Refiner pickup followed by a no-reset policy handoff."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TeacherHandoffConfig:
    minimum_lift_m: float = 0.05
    stable_lift_frames: int = 10


class OnlineTeacherHandoffController:
    """Track which vector environments have reached a physical held-box state."""

    def __init__(self, env, asset_name: str, config: TeacherHandoffConfig):
        self.env = env
        self.asset_name = asset_name
        self.asset = env.scene[asset_name]
        self.config = config
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        self.initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.initial_height_m = torch.zeros(self.num_envs, device=self.device)
        self.stable_lift_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.handoff_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.handoff_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.cumulative_handoffs = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _ids(self, env_ids) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
        return torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)

    def reset(self, env_ids=None) -> None:
        ids = self._ids(env_ids)
        self.initialized[ids] = False
        self.stable_lift_count[ids] = 0
        self.handoff_active[ids] = False
        self.handoff_step[ids] = -1

    def advance(self, *, control_step: int) -> torch.Tensor:
        height = self.asset.data.root_pos_w[:, 2]
        first = ~self.initialized
        self.initial_height_m[first] = height[first]
        self.initialized[first] = True
        lifted = (
            height - self.initial_height_m
            >= float(self.config.minimum_lift_m)
        )
        waiting = ~self.handoff_active
        self.stable_lift_count = torch.where(
            waiting & lifted,
            self.stable_lift_count + 1,
            torch.where(
                waiting,
                torch.zeros_like(self.stable_lift_count),
                self.stable_lift_count,
            ),
        )
        ready = waiting & (
            self.stable_lift_count >= int(self.config.stable_lift_frames)
        )
        ids = ready.nonzero(as_tuple=False).flatten()
        if ids.numel() > 0:
            self.handoff_active[ids] = True
            self.handoff_step[ids] = int(control_step)
            self.cumulative_handoffs[ids] += 1
        return ids

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            "handoff_active": self.handoff_active.clone(),
            "handoff_step": self.handoff_step.clone(),
            "stable_lift_count": self.stable_lift_count.clone(),
            "cumulative_handoffs": self.cumulative_handoffs.clone(),
        }


def _controller(
    env,
    asset_name: str,
    config: TeacherHandoffConfig,
) -> OnlineTeacherHandoffController:
    controller = getattr(env, "_online_teacher_handoff_controller", None)
    if controller is None:
        controller = OnlineTeacherHandoffController(env, asset_name, config)
        env._online_teacher_handoff_controller = controller
    return controller


def reset_online_teacher_handoff(
    env,
    env_ids,
    asset_name: str = "obj",
    minimum_lift_m: float = 0.05,
    stable_lift_frames: int = 10,
) -> None:
    config = TeacherHandoffConfig(
        minimum_lift_m=minimum_lift_m,
        stable_lift_frames=stable_lift_frames,
    )
    _controller(env, asset_name, config).reset(env_ids)


def step_online_teacher_handoff(
    env,
    env_ids,
    asset_name: str = "obj",
    minimum_lift_m: float = 0.05,
    stable_lift_frames: int = 10,
) -> None:
    del env_ids
    config = TeacherHandoffConfig(
        minimum_lift_m=minimum_lift_m,
        stable_lift_frames=stable_lift_frames,
    )
    controller = _controller(env, asset_name, config)
    controller.advance(control_step=int(env.common_step_counter))
    env._online_teacher_handoff_diagnostics = controller.diagnostics()


def online_teacher_handoff_training_mask(env) -> torch.Tensor:
    """Training-only PPO mask; this observation group is not an actor input."""

    controller = getattr(env, "_online_teacher_handoff_controller", None)
    if controller is None:
        return torch.zeros(
            (env.num_envs, 1), dtype=torch.float32, device=env.device
        )
    return controller.handoff_active.to(torch.float32).unsqueeze(-1)
