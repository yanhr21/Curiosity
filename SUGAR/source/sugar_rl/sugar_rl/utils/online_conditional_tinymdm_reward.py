"""Causal online reward from the official conditional MimicKit TinyMDM."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[5]
MIMICKIT_PYTHON = PROJECT_ROOT / "MimicKit/mimickit"
SMP_ADAPTER_ROOT = PROJECT_ROOT / "scripts/sugar/smp"
for path in (MIMICKIT_PYTHON, SMP_ADAPTER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs.amp_env import compute_disc_obs  # noqa: E402
from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: E402
from sugar_g1_box_schema import (  # noqa: E402
    CHARACTER_FEATURE_DIM,
    FEATURE_DIM,
    G1_JOINT_AXES,
    G1_JOINT_NAMES,
    KEY_BODY_INDICES,
    OBJECT_FEATURE_DIM,
    ROOT_BODY_INDEX,
    TRACKED_BODY_NAMES,
    WINDOW_SIZE,
)
from util import torch_util  # noqa: E402


class OnlineConditionalTinyMDMReward:
    """Maintain raw state history and compute a causal official SMP reward."""

    def __init__(
        self,
        base_env: Any,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        calibration_path: str | Path,
        class_id: int,
        reward_seed: int,
    ) -> None:
        if class_id not in (0, 1):
            raise ValueError("conditional TinyMDM class must be Carry=0 or Kick=1")
        self.base_env = base_env
        self.device = torch.device(base_env.device)
        self.num_envs = int(base_env.num_envs)
        self.class_id = int(class_id)
        config_path = Path(config_path).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        calibration_path = Path(calibration_path).expanduser().resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["env_config"] = str(config_path.parent / "env_config.yaml")
        if (
            config.get("arch_name") != "CondDiT"
            or int(config.get("num_class", 0)) != 2
            or int(config.get("input_channel", 0)) != FEATURE_DIM
            or int(config.get("num_disc_obs_steps", 0)) != WINDOW_SIZE
        ):
            raise RuntimeError("conditional TinyMDM configuration drift")
        self.model = TinyMDMModel(config, self.device).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True),
            strict=True,
        )
        self.model.eval().requires_grad_(False)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (
            calibration.get("passed") is not True
            or calibration.get("protocol")
            != "sugar_conditional_tinymdm_official_smp_reward_calibration_v1"
        ):
            raise RuntimeError("conditional SMP calibration is not admitted")
        self.diffusion_steps = tuple(
            int(value) for value in calibration["diffusion_steps"]
        )
        self.diff_mean_abs = torch.as_tensor(
            calibration["diff_normalizer_mean_abs"],
            dtype=torch.float32,
            device=self.device,
        )
        default_scale = float(calibration["official_sds_loss_scale"])
        default_median_reward = float(calibration["calibration_reward"]["median"])
        normalized_median = -math.log(default_median_reward) / default_scale
        self.loss_scale = math.log(2.0) / normalized_median
        self.class_labels = torch.full(
            (self.num_envs,), self.class_id, dtype=torch.long, device=self.device
        )
        robot = self.base_env.scene["robot"]
        self.body_ids = torch.as_tensor(
            [robot.body_names.index(name) for name in TRACKED_BODY_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.joint_ids = torch.as_tensor(
            [robot.joint_names.index(name) for name in G1_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.joint_axes = torch.as_tensor(
            G1_JOINT_AXES, dtype=torch.float32, device=self.device
        )
        self.history: dict[str, torch.Tensor] = {}
        self.observation_count = 0
        with torch.random.fork_rng(devices=[self.device.index or 0]):
            torch.manual_seed(reward_seed)
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(reward_seed)
            self._cpu_rng_state = torch.get_rng_state().clone()
            self._cuda_rng_state = torch.cuda.get_rng_state(self.device).clone()
        self.reward_calls = 0
        self.reward_sum = 0.0
        self.reward_min = float("inf")
        self.reward_max = float("-inf")
        self.reset_history()

    def _current_state(self) -> dict[str, torch.Tensor]:
        robot = self.base_env.scene["robot"].data
        obj = self.base_env.scene["obj"].data
        return {
            "body_pos": robot.body_pos_w.index_select(1, self.body_ids),
            "body_quat": robot.body_quat_w.index_select(1, self.body_ids),
            "body_lin_vel": robot.body_lin_vel_w.index_select(1, self.body_ids),
            "body_ang_vel": robot.body_ang_vel_w.index_select(1, self.body_ids),
            "joint_pos": robot.joint_pos.index_select(1, self.joint_ids),
            "joint_vel": robot.joint_vel.index_select(1, self.joint_ids),
            "object_root": obj.root_state_w,
        }

    @torch.no_grad()
    def reset_history(self) -> None:
        current = self._current_state()
        self.history = {
            name: value[:, None].expand(-1, WINDOW_SIZE, *value.shape[1:]).clone()
            for name, value in current.items()
        }
        self.observation_count = 0

    @torch.no_grad()
    def observe_current_state(self) -> None:
        current = self._current_state()
        for name, history in self.history.items():
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = current[name]
        self.observation_count += 1

    def _features(self) -> torch.Tensor:
        body_pos = self.history["body_pos"]
        body_quat_wxyz = self.history["body_quat"]
        body_quat = body_quat_wxyz[..., (1, 2, 3, 0)]
        body_lin_vel = self.history["body_lin_vel"]
        body_ang_vel = self.history["body_ang_vel"]
        dof_pos = self.history["joint_pos"]
        dof_vel = self.history["joint_vel"]
        root_pos = body_pos[:, :, ROOT_BODY_INDEX]
        root_rot = body_quat[:, :, ROOT_BODY_INDEX]
        axes = self.joint_axes.view(1, 1, len(G1_JOINT_NAMES), 3).expand(
            self.num_envs, WINDOW_SIZE, -1, -1
        )
        joint_rot = torch_util.axis_angle_to_quat(axes, dof_pos)
        character = compute_disc_obs(
            ref_root_pos=root_pos[:, -1],
            ref_root_rot=root_rot[:, -1],
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=body_lin_vel[:, :, ROOT_BODY_INDEX],
            root_ang_vel=body_ang_vel[:, :, ROOT_BODY_INDEX],
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=body_pos[:, :, list(KEY_BODY_INDICES)],
            global_obs=False,
            root_height_obs=True,
            dof_vel_obs=False,
        ).reshape(self.num_envs, WINDOW_SIZE, -1)
        if character.shape[-1] != CHARACTER_FEATURE_DIM:
            raise RuntimeError("online official character feature geometry drift")

        object_root = self.history["object_root"]
        object_position = object_root[..., 0:3]
        object_quat = object_root[..., (4, 5, 6, 3)]
        object_linear_velocity = object_root[..., 7:10]
        object_angular_velocity = object_root[..., 10:13]
        heading_inv = torch_util.calc_heading_quat_inv(root_rot[:, -1])
        heading_steps = heading_inv[:, None].expand(-1, WINDOW_SIZE, -1)
        x_axis = torch.zeros_like(object_position)
        x_axis[..., 0] = 1.0
        z_axis = torch.zeros_like(object_position)
        z_axis[..., 2] = 1.0
        object_features = torch.cat(
            (
                torch_util.quat_rotate(
                    heading_steps, object_position - root_pos
                ),
                torch_util.quat_rotate(
                    heading_steps,
                    torch_util.quat_rotate(object_quat, x_axis),
                ),
                torch_util.quat_rotate(
                    heading_steps,
                    torch_util.quat_rotate(object_quat, z_axis),
                ),
                torch_util.quat_rotate(heading_steps, object_linear_velocity),
                torch_util.quat_rotate(heading_steps, object_angular_velocity),
            ),
            dim=-1,
        )
        if object_features.shape[-1] != OBJECT_FEATURE_DIM:
            raise RuntimeError("online object feature geometry drift")
        features = torch.cat((character, object_features), dim=-1)
        if features.shape != (self.num_envs, WINDOW_SIZE, FEATURE_DIM):
            raise RuntimeError(f"online TinyMDM feature geometry {features.shape}")
        if not torch.isfinite(features).all():
            raise RuntimeError("online TinyMDM features became non-finite")
        return features

    @torch.no_grad()
    def reward(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.observation_count < WINDOW_SIZE:
            raise RuntimeError("conditional SMP reward requested before causal history warmup")
        features = self._features()
        normalized = self.model.normalize(features).reshape(self.num_envs, -1)
        global_cpu_state = torch.get_rng_state()
        global_cuda_state = torch.cuda.get_rng_state(self.device)
        try:
            torch.set_rng_state(self._cpu_rng_state)
            torch.cuda.set_rng_state(self._cuda_rng_state, self.device)
            losses = self.model.ESM_SDS_loss(
                normalized,
                t_lst=list(self.diffusion_steps),
                class_labels=self.class_labels,
            )
            self._cpu_rng_state = torch.get_rng_state().clone()
            self._cuda_rng_state = torch.cuda.get_rng_state(self.device).clone()
        finally:
            torch.set_rng_state(global_cpu_state)
            torch.cuda.set_rng_state(global_cuda_state, self.device)
        normalized_loss = torch.mean(losses / self.diff_mean_abs[None], dim=-1)
        reward = torch.exp(-normalized_loss * self.loss_scale)
        if not torch.isfinite(reward).all():
            raise RuntimeError("online conditional SMP reward became non-finite")
        self.reward_calls += 1
        self.reward_sum += float(reward.mean().item())
        self.reward_min = min(self.reward_min, float(reward.min().item()))
        self.reward_max = max(self.reward_max, float(reward.max().item()))
        return reward, losses.mean(dim=-1)

    def audit(self) -> dict[str, Any]:
        return {
            "protocol": "sugar_online_conditional_tinymdm_reward_v1",
            "class_id": self.class_id,
            "feature_geometry": [WINDOW_SIZE, FEATURE_DIM],
            "diffusion_steps": list(self.diffusion_steps),
            "loss_scale": self.loss_scale,
            "reward_calls": self.reward_calls,
            "reward_mean": (
                self.reward_sum / self.reward_calls if self.reward_calls else None
            ),
            "reward_min": self.reward_min if self.reward_calls else None,
            "reward_max": self.reward_max if self.reward_calls else None,
            "global_policy_rng_restored_after_each_call": True,
            "future_or_outcome_labels_used": False,
        }
