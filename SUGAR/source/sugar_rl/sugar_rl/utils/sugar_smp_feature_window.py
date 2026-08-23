# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Live SUGAR state to the admitted official TinyMDM `10 x 216` contract."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

from isaaclab.utils.math import matrix_from_quat


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
SCHEMA_DIR = WORKSPACE_ROOT / "scripts/sugar/smp"
MIMICKIT_PYTHON = WORKSPACE_ROOT / "MimicKit/mimickit"
for path in (SCHEMA_DIR, MIMICKIT_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sugar_g1_box_schema import (  # noqa: E402
    CHARACTER_FEATURE_DIM,
    FEATURE_DIM,
    G1_JOINT_AXES,
    G1_JOINT_NAMES,
    KEY_BODY_INDICES,
    ROOT_BODY_INDEX,
    TRACKED_BODY_NAMES,
    WINDOW_SIZE,
)
from envs.amp_env import compute_disc_obs  # noqa: E402
from util import torch_util  # noqa: E402


def _wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion[..., (1, 2, 3, 0)]


class SugarSMPFeatureWindowBuffer:
    """Reset-safe live window using the same official feature call as export."""

    def __init__(self, env, robot_name: str = "robot", object_name: str = "obj") -> None:
        self.env = env
        self.robot = env.scene[robot_name]
        self.obj = env.scene[object_name]
        if list(self.robot.joint_names) != list(G1_JOINT_NAMES):
            raise RuntimeError("live SUGAR articulation order differs from admitted SMP schema")
        body_ids, body_names = self.robot.find_bodies(
            list(TRACKED_BODY_NAMES), preserve_order=True
        )
        if list(body_names) != list(TRACKED_BODY_NAMES):
            raise RuntimeError("live SUGAR tracked-body order differs from SMP schema")
        self.body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=env.device)
        self.joint_axes = torch.as_tensor(
            G1_JOINT_AXES, dtype=torch.float32, device=env.device
        )
        self.history: dict[str, torch.Tensor] | None = None
        self.last_step = -1

    def _current_frame(self) -> dict[str, torch.Tensor]:
        robot = self.robot.data
        obj = self.obj.data
        frame = {
            "body_pos": robot.body_pos_w[:, self.body_ids],
            "body_quat_xyzw": _wxyz_to_xyzw(robot.body_quat_w[:, self.body_ids]),
            "body_lin_vel": robot.body_lin_vel_w[:, self.body_ids],
            "body_ang_vel": robot.body_ang_vel_w[:, self.body_ids],
            "joint_pos": robot.joint_pos,
            "joint_vel": robot.joint_vel,
            "obj_pos": obj.root_pos_w,
            "obj_rot": matrix_from_quat(obj.root_quat_w),
            "obj_lin_vel": obj.root_lin_vel_w,
            "obj_ang_vel": obj.root_ang_vel_w,
        }
        if not all(torch.isfinite(value).all() for value in frame.values()):
            raise RuntimeError("non-finite live SUGAR state at the SMP boundary")
        return frame

    @torch.no_grad()
    def update(self) -> torch.Tensor:
        step = int(self.env.common_step_counter)
        current = self._current_frame()
        if self.history is None:
            self.history = {
                name: value[:, None].repeat(
                    (1, WINDOW_SIZE) + (1,) * (value.ndim - 1)
                )
                for name, value in current.items()
            }
        elif self.last_step != step:
            for name, value in current.items():
                self.history[name][:, :-1] = self.history[name][:, 1:].clone()
                self.history[name][:, -1] = value
        self.last_step = step

        reset_mask = self.env.episode_length_buf == 0
        if reset_mask.any():
            for name, value in current.items():
                history = self.history[name]
                reset_view = reset_mask.reshape(
                    self.env.num_envs,
                    *(1 for _ in range(history.ndim - 1)),
                )
                self.history[name] = torch.where(
                    reset_view,
                    value[:, None],
                    history,
                )
        return self.compute_features()

    @torch.no_grad()
    def compute_features(self) -> torch.Tensor:
        if self.history is None:
            raise RuntimeError("SMP feature history has not been initialized")
        history = self.history
        batch_size = self.env.num_envs
        root_pos = history["body_pos"][:, :, ROOT_BODY_INDEX]
        root_rot = history["body_quat_xyzw"][:, :, ROOT_BODY_INDEX]
        root_vel = history["body_lin_vel"][:, :, ROOT_BODY_INDEX]
        root_ang_vel = history["body_ang_vel"][:, :, ROOT_BODY_INDEX]
        key_pos = history["body_pos"][:, :, list(KEY_BODY_INDICES)]
        axes = self.joint_axes.view(1, 1, len(G1_JOINT_NAMES), 3).expand(
            batch_size, WINDOW_SIZE, -1, -1
        )
        joint_rot = torch_util.axis_angle_to_quat(axes, history["joint_pos"])
        ref_root_pos = root_pos[:, -1]
        ref_root_rot = root_rot[:, -1]
        character = compute_disc_obs(
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=history["joint_vel"],
            key_pos=key_pos,
            global_obs=False,
            root_height_obs=True,
            dof_vel_obs=False,
        ).reshape(batch_size, WINDOW_SIZE, -1)
        if character.shape[-1] != CHARACTER_FEATURE_DIM:
            raise RuntimeError(f"live official character feature shape {character.shape}")

        heading_inv = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_steps = heading_inv[:, None].expand(-1, WINDOW_SIZE, -1)
        obj_pos_local = torch_util.quat_rotate(
            heading_inv_steps, history["obj_pos"] - root_pos
        )
        obj_tangent_local = torch_util.quat_rotate(
            heading_inv_steps, history["obj_rot"][..., :, 0]
        )
        obj_normal_local = torch_util.quat_rotate(
            heading_inv_steps, history["obj_rot"][..., :, 2]
        )
        obj_lin_vel_local = torch_util.quat_rotate(
            heading_inv_steps, history["obj_lin_vel"]
        )
        obj_ang_vel_local = torch_util.quat_rotate(
            heading_inv_steps, history["obj_ang_vel"]
        )
        features = torch.cat(
            (
                character,
                obj_pos_local,
                obj_tangent_local,
                obj_normal_local,
                obj_lin_vel_local,
                obj_ang_vel_local,
            ),
            dim=-1,
        )
        if features.shape != (batch_size, WINDOW_SIZE, FEATURE_DIM):
            raise RuntimeError(f"live SUGAR SMP feature shape {features.shape}")
        if not torch.isfinite(features).all():
            raise RuntimeError("non-finite live SUGAR TinyMDM features")
        return features
