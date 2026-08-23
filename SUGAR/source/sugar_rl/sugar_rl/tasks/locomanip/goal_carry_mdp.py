# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-based CarryBox terms that do not track a prescribed body trajectory.

Official SUGAR motion is used only to reset a valid G1+box scene and to obtain
the final box goal. No per-frame joint, body, hand, or object reference enters
these observations, rewards, or terminations.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_error_magnitude,
    subtract_frame_transforms,
)

from sugar_rl.tasks.locomanip.mdp.commands import MotionCommand, MotionCommandCfg


class GoalCarryMotionCommand(MotionCommand):
    """Reference-seeded reset command with outcome-only box goals."""

    cfg: "GoalCarryMotionCommandCfg"

    def __init__(self, cfg: "GoalCarryMotionCommandCfg", env):
        super().__init__(cfg, env)
        self.episode_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.initial_obj_height_w = torch.zeros(self.num_envs, device=self.device)
        self.initial_obj_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_robot_root_height_w = torch.zeros(
            self.num_envs, device=self.device
        )
        self.goal_stable_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.ever_lifted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.metrics["goal_position_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["goal_orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["goal_linear_speed"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["lift_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_stable_steps"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.obj_target_pos_w, self.obj_target_quat_w), dim=-1)

    def _record_reference_targets(self, env_ids: Sequence[int]):
        # The final reference box pose is an outcome goal only. It does not
        # prescribe the path, contacts, posture, or timing used to reach it.
        super()._record_reference_targets(env_ids)
        if len(env_ids) == 0:
            return
        motion_ids = self.motion_id[env_ids]
        initial_positions = (
            self.motion.obj_pos[motion_ids, 0] + self._env.scene.env_origins[env_ids]
        )
        self.initial_obj_pos_w[env_ids] = initial_positions
        self.initial_obj_height_w[env_ids] = initial_positions[:, 2]
        # Record the physical root-height reference at the sampled motion
        # boundary.  Unlike the final box goal, fall detection must be relative
        # to the posture from which this particular episode starts.
        self.initial_robot_root_height_w[env_ids] = (
            self.motion.body_pos_w[
                motion_ids,
                self.time_steps[env_ids],
                0,
                2,
            ]
            + self._env.scene.env_origins[env_ids, 2]
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        self.episode_steps[env_ids] = 0
        self.goal_stable_counter[env_ids] = 0
        self.ever_lifted[env_ids] = False

    def _update_command(self):
        # Deliberately do not advance the SUGAR reference trajectory or update
        # its state pool. The reference is a reset source, not a tracking target.
        self.episode_steps += 1
        lift_height = self.obj_pos_w[:, 2] - self.initial_obj_height_w
        self.ever_lifted |= lift_height >= self.cfg.lifted_height_threshold
        position_error = torch.linalg.vector_norm(
            self.obj_pos_w - self.obj_target_pos_w, dim=-1
        )
        orientation_error = quat_error_magnitude(
            self.obj_quat_w, self.obj_target_quat_w
        )
        linear_speed = torch.linalg.vector_norm(self.obj_lin_vel_w, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.obj_ang_vel_w, dim=-1)
        stable = (
            (position_error <= self.cfg.success_position_tolerance)
            & (orientation_error <= self.cfg.success_orientation_tolerance)
            & (linear_speed <= self.cfg.success_linear_speed_tolerance)
            & (angular_speed <= self.cfg.success_angular_speed_tolerance)
        )
        self.goal_stable_counter = torch.where(
            stable,
            self.goal_stable_counter + 1,
            torch.zeros_like(self.goal_stable_counter),
        )

    def _update_metrics(self):
        position_error = torch.linalg.vector_norm(
            self.obj_pos_w - self.obj_target_pos_w, dim=-1
        )
        orientation_error = quat_error_magnitude(
            self.obj_quat_w, self.obj_target_quat_w
        )
        self.metrics["goal_position_error"] = position_error
        self.metrics["goal_orientation_error"] = orientation_error
        self.metrics["goal_linear_speed"] = torch.linalg.vector_norm(
            self.obj_lin_vel_w, dim=-1
        )
        self.metrics["lift_height"] = self.obj_pos_w[:, 2] - self.initial_obj_height_w
        self.metrics["goal_stable_steps"] = self.goal_stable_counter.float()


@configclass
class GoalCarryMotionCommandCfg(MotionCommandCfg):
    class_type: type = GoalCarryMotionCommand
    lifted_height_threshold: float = 0.10
    success_position_tolerance: float = 0.12
    success_orientation_tolerance: float = 0.45
    success_linear_speed_tolerance: float = 0.20
    success_angular_speed_tolerance: float = 0.40
    success_stable_steps: int = 20


def base_height(env, command_name: str) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    return (
        command.robot_anchor_pos_w[:, 2:3]
        - env.scene.env_origins[:, 2:3]
    )


def box_position_body(env, command_name: str) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    position, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )
    return position


def box_orientation_tangent_normal_body(env, command_name: str) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    _, orientation = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )
    rotation = matrix_from_quat(orientation)
    return torch.cat((rotation[..., :, 0], rotation[..., :, 2]), dim=-1)


def box_linear_velocity_body(env, command_name: str) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    return quat_apply_inverse(command.robot_anchor_quat_w, command.obj_lin_vel_w)


def box_angular_velocity_body(env, command_name: str) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    return quat_apply_inverse(command.robot_anchor_quat_w, command.obj_ang_vel_w)


def goal_position_body(env, command_name: str) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    position, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_target_pos_w,
        command.obj_target_quat_w,
    )
    return position


def goal_orientation_tangent_normal_body(env, command_name: str) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    _, orientation = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_target_pos_w,
        command.obj_target_quat_w,
    )
    rotation = matrix_from_quat(orientation)
    return torch.cat((rotation[..., :, 0], rotation[..., :, 2]), dim=-1)


def previous_applied_action_policy_units(
    env, action_name: str = "JointPositionAction"
) -> torch.Tensor:
    """Recover official policy units from the actual applied joint target."""

    action_term = env.action_manager.get_term(action_name)
    processed = action_term.processed_actions
    scale = action_term._scale
    offset = action_term._offset
    policy_units = (processed - offset) / scale
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        policy_units = policy_units.clone()
        policy_units[reset_mask] = 0.0
    if not torch.isfinite(policy_units).all():
        raise RuntimeError("non-finite applied SUGAR action in policy units")
    if policy_units.abs().max() > 100.0:
        raise RuntimeError("applied SUGAR action exceeds the declared safety audit limit")
    return policy_units


def goal_position_reward(env, command_name: str, std: float) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    squared_error = torch.square(command.obj_pos_w - command.obj_target_pos_w).sum(dim=-1)
    return torch.exp(-squared_error / std**2)


def goal_orientation_reward(env, command_name: str, std: float) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.obj_quat_w, command.obj_target_quat_w)
    return torch.exp(-torch.square(error) / std**2)


def lift_fraction_reward(env, command_name: str, target_height: float) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    lift = command.obj_pos_w[:, 2] - command.initial_obj_height_w
    return torch.clamp(lift / target_height, min=0.0, max=1.0)


def goal_stability_reward(
    env, command_name: str, position_tolerance: float, speed_tolerance: float
) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    position_error = torch.linalg.vector_norm(
        command.obj_pos_w - command.obj_target_pos_w, dim=-1
    )
    linear_speed = torch.linalg.vector_norm(command.obj_lin_vel_w, dim=-1)
    return (
        (position_error <= position_tolerance) & (linear_speed <= speed_tolerance)
    ).float()


def goal_reached(env, command_name: str) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    return command.goal_stable_counter >= command.cfg.success_stable_steps


def unsafe_robot_fall(
    env,
    command_name: str,
    maximum_root_height_loss_m: float,
) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    root_height_loss = (
        command.initial_robot_root_height_w
        - command.robot.data.root_pos_w[:, 2]
    )
    # Carry45 contains a deliberate recoverable forward fold: the admitted
    # official rollout reaches root up-z 0.191 while losing only 0.143 m of
    # root height, then returns upright and carries the box.  Orientation alone
    # therefore cannot define a fall for this goal task.
    return root_height_loss >= maximum_root_height_loss_m


def dropped_after_lift(
    env, command_name: str, maximum_height_above_start: float
) -> torch.Tensor:
    command: GoalCarryMotionCommand = env.command_manager.get_term(command_name)
    lift = command.obj_pos_w[:, 2] - command.initial_obj_height_w
    return command.ever_lifted & (lift < maximum_height_above_start)


def box_out_of_workspace(
    env, asset_cfg: SceneEntityCfg, maximum_distance: float
) -> torch.Tensor:
    obj = env.scene[asset_cfg.name]
    local_position = obj.data.root_pos_w - env.scene.env_origins
    return torch.linalg.vector_norm(local_position, dim=-1) > maximum_distance
