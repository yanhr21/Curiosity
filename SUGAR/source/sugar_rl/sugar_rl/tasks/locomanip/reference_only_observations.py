# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Leak-free reference-plan object observations for the tactile actor.

These terms preserve the official SUGAR observation widths and ordering while
removing feedback from the simulated box state.  They are intentionally kept
outside the active MDP export until the exact-state branch completes its final
evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from sugar_rl.tasks.locomanip.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reference_plan_obj_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the current reference box position in the robot-anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    position_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_ref_pos_w,
        command.obj_ref_quat_w,
    )
    return position_b.reshape(env.num_envs, -1)


def reference_plan_obj_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the current reference box orientation in the robot-anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, orientation_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_ref_pos_w,
        command.obj_ref_quat_w,
    )
    rotation = matrix_from_quat(orientation_b)
    return rotation[..., :2].reshape(env.num_envs, -1)


def _reference_velocity_b(
    env: ManagerBasedEnv,
    command_name: str,
    *,
    angular: bool,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_inverse = command.robot_anchor_quat_w.clone()
    anchor_inverse[:, :3] = -anchor_inverse[:, :3]
    velocity_w = command.obj_ref_ang_vel_w if angular else command.obj_ref_lin_vel_w
    velocity_b = torch.bmm(
        matrix_from_quat(anchor_inverse),
        velocity_w.unsqueeze(-1),
    ).squeeze(-1)
    return velocity_b.reshape(env.num_envs, -1)


def reference_plan_obj_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return reference linear velocity without reading the simulated box."""
    return _reference_velocity_b(env, command_name, angular=False)


def reference_plan_obj_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return reference angular velocity without reading the simulated box."""
    return _reference_velocity_b(env, command_name, angular=True)


def _future_reference_pose(
    env: ManagerBasedEnv,
    command_name: str,
) -> tuple[MotionCommand, torch.Tensor, torch.Tensor, int]:
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    frame_count = int(future_timesteps.shape[-1])
    future_position_w = (
        command.motion.obj_pos[batch_motion_ids, future_timesteps]
        + env.scene.env_origins.unsqueeze(1)
    )
    future_orientation_w = command.motion.obj_quat[batch_motion_ids, future_timesteps]
    return command, future_position_w, future_orientation_w, frame_count


def reference_plan_obj_motion_pos_future(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Express future reference positions in the current reference box frame."""
    command, future_position_w, future_orientation_w, frame_count = _future_reference_pose(
        env,
        command_name,
    )
    current_position_w = command.obj_ref_pos_w.unsqueeze(1).expand(-1, frame_count, -1)
    current_orientation_w = command.obj_ref_quat_w.unsqueeze(1).expand(-1, frame_count, -1)
    position_ref, _ = subtract_frame_transforms(
        current_position_w.reshape(-1, 3),
        current_orientation_w.reshape(-1, 4),
        future_position_w.reshape(-1, 3),
        future_orientation_w.reshape(-1, 4),
    )
    return position_ref.reshape(env.num_envs, frame_count * 3)


def reference_plan_obj_motion_ori_future(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Express future reference orientations in the current reference frame."""
    command, future_position_w, future_orientation_w, frame_count = _future_reference_pose(
        env,
        command_name,
    )
    current_position_w = command.obj_ref_pos_w.unsqueeze(1).expand(-1, frame_count, -1)
    current_orientation_w = command.obj_ref_quat_w.unsqueeze(1).expand(-1, frame_count, -1)
    _, orientation_ref = subtract_frame_transforms(
        current_position_w.reshape(-1, 3),
        current_orientation_w.reshape(-1, 4),
        future_position_w.reshape(-1, 3),
        future_orientation_w.reshape(-1, 4),
    )
    rotation = matrix_from_quat(orientation_ref)
    return rotation[..., :2].reshape(env.num_envs, frame_count * 6)
