# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Causal reset glue for the SUGAR reference-waypoint foundation.

This module does not define a controller, policy, tactile proxy, or reward.
It restores hash-bound official SUGAR/TacSL source states, their real
four-frame spatial pressure/signed-shear history, the preceding official
action, source-consistent physics, and an object-only official reference
waypoint.  The native SUGAR command, action, sensor, reward, and termination
managers remain responsible for every subsequent transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sugar_rl.tasks.locomanip.direct_tactile_history import (
    direct_tactile_force_history,
)
from sugar_rl.tasks.locomanip.latent_contact_dynamics_events import (
    apply_stratified_latent_contact_dynamics,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_smp_icm_goal_env_cfg import (
    TACTILE_RUNTIME_PARAMS,
)


_SENSOR_NAMES = ("left_palm_tactile", "right_palm_tactile")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceWaypointSource:
    """One exact official source/waypoint/physics branch."""

    source_id: str
    path: Path
    sha256: str
    initial_frame: int
    reference_frame: int
    waypoint_reference_frame: int | None
    waypoint_relative_lift_m: float | None
    mass_scale: float
    static_friction: float = 0.6
    dynamic_friction: float = 0.5
    com_y_m: float = 0.0

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], workspace_root: Path
    ) -> "ReferenceWaypointSource":
        path = Path(str(payload["path"])).expanduser()
        if not path.is_absolute():
            path = workspace_root / path
        return cls(
            source_id=str(payload["source_id"]),
            path=path.resolve(),
            sha256=str(payload["sha256"]),
            initial_frame=int(payload["initial_frame"]),
            reference_frame=int(payload["reference_frame"]),
            waypoint_reference_frame=(
                int(payload["waypoint_reference_frame"])
                if payload.get("waypoint_reference_frame") is not None
                else None
            ),
            waypoint_relative_lift_m=(
                float(payload["waypoint_relative_lift_m"])
                if payload.get("waypoint_relative_lift_m") is not None
                else None
            ),
            mass_scale=float(payload["mass_scale"]),
            static_friction=float(payload.get("static_friction", 0.6)),
            dynamic_friction=float(payload.get("dynamic_friction", 0.5)),
            com_y_m=float(payload.get("com_y_m", 0.0)),
        )


class ReferenceWaypointFoundationReset:
    """Restore alternating exact source branches at every episode boundary."""

    protocol = "sugar_reference_waypoint_foundation_reset_v1"

    def __init__(
        self,
        env,
        sources: Sequence[ReferenceWaypointSource],
    ) -> None:
        self.env = env
        self.device = torch.device(env.device)
        self.sources = tuple(sources)
        if len(self.sources) != 2:
            raise ValueError(
                "reference-waypoint foundation requires exactly two sources"
            )
        if env.num_envs < 2 or env.num_envs % len(self.sources) != 0:
            raise ValueError(
                "foundation environments must divide evenly across sources"
            )
        if len({source.source_id for source in self.sources}) != 2:
            raise ValueError("foundation source IDs must be unique")
        self.source_index_by_env = (
            torch.arange(env.num_envs, device=self.device)
            % len(self.sources)
        )
        self._archives = tuple(
            self._load_source(source) for source in self.sources
        )
        self.reset_calls = 0
        self.reset_environment_steps = 0
        self.last_reset_env_ids = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        self._original_reset_idx = None
        self._configure_source_consistent_physics()

    @staticmethod
    def _load_source(
        spec: ReferenceWaypointSource,
    ) -> dict[str, np.ndarray]:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        actual_sha256 = _sha256(spec.path)
        if actual_sha256 != spec.sha256:
            raise RuntimeError(
                f"foundation source hash drift for {spec.source_id}: "
                f"{actual_sha256} != {spec.sha256}"
            )
        with np.load(spec.path, allow_pickle=False) as archive:
            payload = {
                name: np.asarray(archive[name]) for name in archive.files
            }
        required = {
            "robot_root_state_w",
            "robot_joint_pos",
            "robot_joint_vel",
            "object_root_state_w",
            "normal_force",
            "shear_force",
            "policy_actions_unclipped",
            "applied_actions_policy_units",
            "motion_frame",
            "source_environment_origin_w",
            "selected_motion_id",
            "native_sample_phase",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise KeyError(
                f"foundation source {spec.source_id} is missing {missing}"
            )
        frame = spec.initial_frame
        if frame < 3 or frame >= payload["robot_root_state_w"].shape[0]:
            raise ValueError(
                f"foundation frame is out of range for {spec.source_id}"
            )
        if int(payload["selected_motion_id"].reshape(-1)[0]) != 45:
            raise ValueError("foundation sources are locked to motion 45")
        if str(payload["native_sample_phase"].reshape(-1)[0]) != "pre_action":
            raise ValueError("foundation source must use pre-action samples")
        if (
            (spec.waypoint_reference_frame is None)
            == (spec.waypoint_relative_lift_m is None)
        ):
            raise ValueError(
                "foundation source requires exactly one absolute or relative "
                "object waypoint"
            )
        if (
            spec.waypoint_relative_lift_m is not None
            and spec.waypoint_relative_lift_m != 0.04
        ):
            raise ValueError(
                "corrected relative foundation lift must be exactly 0.04 m"
            )
        if int(payload["motion_frame"].reshape(-1)[frame]) != spec.reference_frame:
            raise ValueError(
                f"reference-frame drift for {spec.source_id}"
            )
        if payload["normal_force"][frame - 3 : frame + 1].shape != (
            4,
            2,
            20,
            25,
        ):
            raise ValueError("foundation normal-force history shape drift")
        if payload["shear_force"][frame - 3 : frame + 1].shape != (
            4,
            2,
            20,
            25,
            2,
        ):
            raise ValueError("foundation shear-force history shape drift")
        if payload["policy_actions_unclipped"].shape[1:] != (29,):
            raise ValueError("foundation official action shape drift")
        roundtrip = np.abs(
            np.asarray(payload["policy_actions_unclipped"], dtype=np.float32)
            - np.asarray(
                payload["applied_actions_policy_units"], dtype=np.float32
            )
        )
        if not np.isfinite(roundtrip).all() or float(roundtrip.max()) > 2.0e-6:
            raise RuntimeError(
                "foundation source action conversion exceeds tolerance"
            )
        normal = np.asarray(payload["normal_force"][frame], dtype=np.float32)
        shear = np.asarray(payload["shear_force"][frame], dtype=np.float32)
        if (
            not np.isfinite(normal).all()
            or not np.isfinite(shear).all()
            or float(normal.min()) < 0.0
            or np.any(normal.sum(axis=(-2, -1)) <= 0.0)
        ):
            raise RuntimeError(
                f"foundation source {spec.source_id} is not finite bilateral "
                "direct TacSL"
            )
        return payload

    def _configure_source_consistent_physics(self) -> None:
        term_cfg = self.env.event_manager.get_term_cfg(
            "latent_contact_dynamics"
        )
        term = term_cfg.func
        if not isinstance(term, apply_stratified_latent_contact_dynamics):
            raise TypeError(
                "foundation requires the coherent latent-dynamics event"
            )
        mass_scale = torch.empty(self.env.num_envs, dtype=torch.float32)
        static_friction = torch.empty_like(mass_scale)
        dynamic_friction = torch.empty_like(mass_scale)
        com_y_m = torch.empty_like(mass_scale)
        for source_index, source in enumerate(self.sources):
            mask = (
                self.source_index_by_env.detach().cpu() == source_index
            )
            mass_scale[mask] = source.mass_scale
            static_friction[mask] = source.static_friction
            dynamic_friction[mask] = source.dynamic_friction
            com_y_m[mask] = source.com_y_m
        if not torch.all(dynamic_friction <= static_friction):
            raise ValueError(
                "foundation dynamic friction exceeds static friction"
            )
        term._tuple_cpu = {
            "mass_scale": mass_scale,
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "com_y_m": com_y_m,
            "pulse_delta_velocity_w_mps": torch.zeros(
                self.env.num_envs, 3, dtype=torch.float32
            ),
        }
        env_ids = torch.arange(
            self.env.num_envs, dtype=torch.long, device=self.device
        )
        term(self.env, env_ids, **term_cfg.params)
        observed = term.tuple_for_device("cpu")
        if not all(
            torch.equal(observed[name], expected)
            for name, expected in term._tuple_cpu.items()
        ):
            raise RuntimeError(
                "source-consistent foundation physics tuple drift"
            )

    @staticmethod
    def _source_history(
        source: Mapping[str, np.ndarray], frame: int
    ) -> np.ndarray:
        normal = np.asarray(
            source["normal_force"][frame - 3 : frame + 1],
            dtype=np.float32,
        )
        shear = np.asarray(
            source["shear_force"][frame - 3 : frame + 1],
            dtype=np.float32,
        )
        history = np.concatenate(
            (normal[:, :, None], shear.transpose(0, 1, 4, 2, 3)),
            axis=2,
        )
        return (
            history
            / float(TACTILE_RUNTIME_PARAMS["taxel_area_m2"])
            * float(TACTILE_RUNTIME_PARAMS["stress_scale"])
        ).astype(np.float32, copy=False)

    @torch.no_grad()
    def restore(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        if ids.numel() == 0:
            return
        if (
            bool(torch.any(ids < 0))
            or bool(torch.any(ids >= self.env.num_envs))
            or int(torch.unique(ids).numel()) != int(ids.numel())
        ):
            raise ValueError("foundation reset environment IDs are invalid")
        target_origins = self.env.scene.env_origins.detach().cpu().numpy()
        command = self.env.command_manager.get_term("motion")
        previous_action = self.env.action_manager.action.detach().clone()
        desired_history = torch.zeros(
            (
                ids.numel(),
                4,
                2,
                3,
                20,
                25,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        for source_index, (spec, source) in enumerate(
            zip(self.sources, self._archives, strict=True)
        ):
            selected = ids[
                self.source_index_by_env[ids] == source_index
            ]
            if selected.numel() == 0:
                continue
            selected_cpu = selected.detach().cpu().numpy()
            source_origin = np.asarray(
                source["source_environment_origin_w"], dtype=np.float32
            )
            translations = (
                target_origins[selected_cpu] - source_origin[None, :]
            )
            frame = spec.initial_frame
            count = int(selected.numel())
            robot_root = np.repeat(
                np.asarray(
                    source["robot_root_state_w"][frame : frame + 1],
                    dtype=np.float32,
                ),
                count,
                axis=0,
            ).copy()
            object_root = np.repeat(
                np.asarray(
                    source["object_root_state_w"][frame : frame + 1],
                    dtype=np.float32,
                ),
                count,
                axis=0,
            ).copy()
            robot_root[:, :3] += translations
            object_root[:, :3] += translations
            joint_pos = np.repeat(
                np.asarray(
                    source["robot_joint_pos"][frame : frame + 1],
                    dtype=np.float32,
                ),
                count,
                axis=0,
            )
            joint_vel = np.repeat(
                np.asarray(
                    source["robot_joint_vel"][frame : frame + 1],
                    dtype=np.float32,
                ),
                count,
                axis=0,
            )
            self.env.scene["robot"].write_root_state_to_sim(
                torch.as_tensor(robot_root, device=self.device),
                env_ids=selected,
            )
            self.env.scene["robot"].write_joint_state_to_sim(
                torch.as_tensor(joint_pos, device=self.device),
                torch.as_tensor(joint_vel, device=self.device),
                env_ids=selected,
            )
            self.env.scene["obj"].write_root_state_to_sim(
                torch.as_tensor(object_root, device=self.device),
                env_ids=selected,
            )

            command.motion_id[selected] = 45
            command.time_steps[selected] = spec.reference_frame
            command._use_motion_data[selected] = True
            command._record_reference_targets(selected)
            if spec.waypoint_reference_frame is not None:
                command.obj_target_pos_w[selected] = (
                    command.motion.obj_pos[
                        command.motion_id[selected],
                        spec.waypoint_reference_frame,
                    ]
                    + self.env.scene.env_origins[selected]
                )
                command.obj_target_quat_w[selected] = (
                    command.motion.obj_quat[
                        command.motion_id[selected],
                        spec.waypoint_reference_frame,
                    ]
                )
            else:
                command.obj_target_pos_w[selected] = torch.as_tensor(
                    object_root[:, :3], device=self.device
                )
                command.obj_target_pos_w[selected, 2] += float(
                    spec.waypoint_relative_lift_m
                )
                command.obj_target_quat_w[selected] = torch.as_tensor(
                    object_root[:, 3:7], device=self.device
                )
            position = torch.as_tensor(
                object_root[:, :3], device=self.device
            )
            command.initial_obj_pos_w[selected] = position
            command.initial_obj_height_w[selected] = position[:, 2]
            command.ever_lifted[selected] = False
            command.goal_stable_counter[selected] = 0
            command.episode_steps[selected] = 0
            if hasattr(command, "last_reset_timestep"):
                command.last_reset_timestep[selected] = spec.reference_frame
            if hasattr(command, "last_reset_motion_id"):
                command.last_reset_motion_id[selected] = 45

            previous = torch.as_tensor(
                source[
                    "policy_actions_unclipped"
                ][frame - 1 : frame],
                dtype=torch.float32,
                device=self.device,
            ).expand(count, -1)
            previous_action[selected] = previous

            history = torch.as_tensor(
                self._source_history(source, frame),
                dtype=torch.float32,
                device=self.device,
            )
            local_rows = torch.nonzero(
                ids[:, None] == selected[None, :],
                as_tuple=False,
            )[:, 0]
            desired_history[local_rows] = history.unsqueeze(0).expand(
                count, -1, -1, -1, -1, -1
            )
            for hand, sensor_name in enumerate(_SENSOR_NAMES):
                sensor_data = self.env.scene[sensor_name].data
                sensor_data.tactile_normal_force[selected] = torch.as_tensor(
                    source["normal_force"][frame, hand].reshape(-1),
                    dtype=torch.float32,
                    device=self.device,
                )
                sensor_data.tactile_shear_force[selected] = torch.as_tensor(
                    source["shear_force"][frame, hand].reshape(-1, 2),
                    dtype=torch.float32,
                    device=self.device,
                )

        self.env.action_manager.process_action(previous_action)
        self.env.episode_length_buf[ids] = 1
        direct_tactile_force_history(
            self.env, **TACTILE_RUNTIME_PARAMS
        )
        cache = getattr(
            self.env, "_sugar_direct_tactile_history_cache", None
        )
        if not isinstance(cache, dict) or len(cache) != 1:
            raise RuntimeError(
                "foundation direct-TacSL history cache geometry drift"
            )
        entry = next(iter(cache.values()))
        history = entry["history"]
        if tuple(history.shape) != (
            self.env.num_envs,
            4,
            2,
            3,
            20,
            25,
        ):
            raise RuntimeError(
                "foundation direct-TacSL history tensor shape drift"
            )
        history[ids] = desired_history
        entry["step"] = int(self.env.common_step_counter)
        self.reset_calls += 1
        self.reset_environment_steps += int(ids.numel())
        self.last_reset_env_ids = ids.detach().clone()

    def install(self) -> None:
        if self._original_reset_idx is not None:
            raise RuntimeError("foundation reset hook is already installed")
        self._original_reset_idx = self.env._reset_idx

        def reset_then_restore(env_ids):
            self._original_reset_idx(env_ids)
            self.restore(env_ids)

        self.env._reset_idx = reset_then_restore

    def restore_original_reset(self) -> None:
        if self._original_reset_idx is not None:
            self.env._reset_idx = self._original_reset_idx
            self._original_reset_idx = None

    def audit_state(self) -> dict[str, Any]:
        term = self.env.event_manager.get_term_cfg(
            "latent_contact_dynamics"
        ).func
        values = term.tuple_for_device("cpu")
        return {
            "protocol": self.protocol,
            "source_ids": [
                source.source_id for source in self.sources
            ],
            "source_paths": [
                str(source.path) for source in self.sources
            ],
            "source_sha256": [
                source.sha256 for source in self.sources
            ],
            "source_index_by_env": (
                self.source_index_by_env.detach().cpu().tolist()
            ),
            "reference_frame_by_source": [
                source.reference_frame for source in self.sources
            ],
            "waypoint_reference_frame_by_source": [
                source.waypoint_reference_frame
                for source in self.sources
            ],
            "waypoint_relative_lift_m_by_source": [
                source.waypoint_relative_lift_m
                for source in self.sources
            ],
            "mass_scale_by_env": values["mass_scale"].tolist(),
            "static_friction_by_env": (
                values["static_friction"].tolist()
            ),
            "dynamic_friction_by_env": (
                values["dynamic_friction"].tolist()
            ),
            "com_y_m_by_env": values["com_y_m"].tolist(),
            "pulse_delta_velocity_w_mps_by_env": (
                values["pulse_delta_velocity_w_mps"].tolist()
            ),
            "reset_calls": self.reset_calls,
            "reset_environment_steps": self.reset_environment_steps,
            "last_reset_env_ids": (
                self.last_reset_env_ids.detach().cpu().tolist()
            ),
            "hook_installed": self._original_reset_idx is not None,
        }
