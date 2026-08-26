# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Online CHORD contact-wrench reward for the SUGAR KickBox transition.

This adapter deliberately calls the pinned official CHORD tensor kernels.  It
does not change the actor observation and it does not infer contacts from a
binary label.  Reference supports come from the geometry-reconstructed
KickBox21 demonstration; current supports come from the live PhysX
foot-to-box contact points and forces produced by the current rollout.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import quat_from_matrix


OFFICIAL_CHORD_COMMIT = "5654c50edc1f3dea8e3145bf2dbfc277dbf27b4c"
NUM_BASIS = 512
NUM_FRICTION_EDGES = 8
FRICTION_COEFFICIENT = 0.1
SUPPORT_THRESHOLD = 1.0e-3
BASIS_SEED = 26070033


def _load_official_utils(checkout: Path):
    source = (
        checkout
        / "robotic_grounding/source/robotic_grounding/robotic_grounding"
        / "tasks/v2d/mdp/utils_jit.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"pinned official CHORD kernels are missing: {source}")
    spec = importlib.util.spec_from_file_location(
        "sugar_official_chord_runtime_utils_jit", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned official CHORD kernels: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def _object_mesh_radius(path: Path) -> float:
    """Compute the same maximum vertex radius used by official CHORD."""

    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"cannot open CHORD object mesh: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points: list[np.ndarray] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        values = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not values:
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        points.append(
            np.asarray(
                [
                    transform.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                    for point in values
                ],
                dtype=np.float64,
            )
        )
    if not points:
        raise RuntimeError(f"CHORD object mesh has no readable vertices: {path}")
    vertices = np.concatenate(points, axis=0)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(centered, axis=-1).max())
    if not math.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("CHORD object mesh radius is invalid")
    return radius


def _basis(device: torch.device) -> torch.Tensor:
    """Use the fixed CPU draw already used by the frozen CHORD diagnostic."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(BASIS_SEED)
    basis = torch.randn(NUM_BASIS, 6, generator=generator, dtype=torch.float32)
    basis[:, 3:] = basis[:, 3:] / 1.0
    basis = basis / basis.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return basis.to(device=device)


class OfficialChordKickReward:
    """Official CHORD reward using a fixed KickBox21 geometry command."""

    def __init__(
        self,
        env,
        *,
        official_chord_root: str | Path,
        reference_geometry: str | Path,
        object_usd: str | Path,
        cws_weight: float = 10.0,
        unintended_weight: float = -10.0,
        missed_weight: float = -1.0,
    ) -> None:
        self.env = env
        self.device = torch.device(env.device)
        self.chord, self.official_source = _load_official_utils(
            Path(official_chord_root).expanduser().resolve()
        )
        self.reference_path = Path(reference_geometry).expanduser().resolve()
        self.object_usd = Path(object_usd).expanduser().resolve()
        self.radius = _object_mesh_radius(self.object_usd)
        self.cws_weight = float(cws_weight)
        self.unintended_weight = float(unintended_weight)
        self.missed_weight = float(missed_weight)
        for value in (self.cws_weight, self.unintended_weight, self.missed_weight):
            if not math.isfinite(value):
                raise ValueError("CHORD reward weights must be finite")

        self.basis = _basis(self.device)
        theta = torch.linspace(
            0.0,
            2.0 * torch.pi,
            steps=NUM_FRICTION_EDGES + 1,
            device=self.device,
            dtype=torch.float32,
        )[:-1]
        self.cos_t = torch.cos(theta).view(1, -1, 1)
        self.sin_t = torch.sin(theta).view(1, -1, 1)
        self.reference_supports = self._load_reference_supports()
        self.calls = 0
        self.active_reference_calls = 0
        self.maximum_abs_reward = 0.0

    def _supports(
        self,
        positions_w: torch.Tensor,
        forces_or_normals_w: torch.Tensor,
        object_position_w: torch.Tensor,
        object_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        count = positions_w.shape[0]
        points_com, normals_com = self.chord.wrench_preprocess_jit(
            contact_positions_w=positions_w.reshape(count, 1, 1, 3),
            contact_forces_first_hist_w=forces_or_normals_w.reshape(count, 1, 1, 3),
            object_com_position_w=object_position_w.reshape(count, 1, 1, 3),
            object_com_orientation_w=object_quat_w.reshape(count, 1, 1, 4),
            num_envs=count,
            num_bodies=1,
            num_robot_contacts=1,
        )
        return self.chord.wrench_support_one_body_jit(
            contact_points=points_com[:, 0],
            contact_normals=normals_com[:, 0],
            cos_t=self.cos_t,
            sin_t=self.sin_t,
            basis=self.basis,
            rc=self.radius,
            friction_coefficients=FRICTION_COEFFICIENT,
        )

    def _load_reference_supports(self) -> torch.Tensor:
        if not self.reference_path.is_file():
            raise FileNotFoundError(
                f"KickBox21 CHORD geometry is missing: {self.reference_path}"
            )
        with np.load(self.reference_path, allow_pickle=False) as archive:
            required = {
                "frame",
                "contact_active",
                "hand_contact_position_w",
                "hand_contact_normal_w",
                "object_position_w",
                "object_rotation_w",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise RuntimeError(f"KickBox21 CHORD geometry misses fields: {missing}")
            frames = np.asarray(archive["frame"])
            active_np = np.asarray(archive["contact_active"], dtype=np.bool_)
            positions_np = np.asarray(archive["hand_contact_position_w"], dtype=np.float32)
            normals_np = np.asarray(archive["hand_contact_normal_w"], dtype=np.float32)
            object_position_np = np.asarray(archive["object_position_w"], dtype=np.float32)
            object_rotation_np = np.asarray(archive["object_rotation_w"], dtype=np.float32)
        horizon = int(frames.shape[0])
        if not np.array_equal(frames, np.arange(horizon, dtype=frames.dtype)):
            raise RuntimeError("KickBox21 CHORD reference frames are not contiguous")
        if active_np.shape != (horizon, 2) or positions_np.shape != (horizon, 2, 3):
            raise RuntimeError("KickBox21 CHORD role geometry drift")
        if normals_np.shape != positions_np.shape:
            raise RuntimeError("KickBox21 CHORD normal geometry drift")
        if object_position_np.shape != (horizon, 3) or object_rotation_np.shape != (
            horizon,
            3,
            3,
        ):
            raise RuntimeError("KickBox21 CHORD object geometry drift")

        active = torch.as_tensor(active_np, device=self.device)
        positions = torch.as_tensor(positions_np, device=self.device)
        normals = torch.as_tensor(normals_np, device=self.device)
        positions = torch.where(active.unsqueeze(-1), positions, torch.zeros_like(positions))
        normals = torch.where(active.unsqueeze(-1), normals, torch.zeros_like(normals))
        object_position = torch.as_tensor(object_position_np, device=self.device)
        object_rotation = torch.as_tensor(object_rotation_np, device=self.device)
        object_quat = quat_from_matrix(object_rotation)
        sides = [
            self._supports(
                positions[:, side], normals[:, side], object_position, object_quat
            )
            for side in range(2)
        ]
        supports = torch.stack(sides, dim=1)
        if supports.shape != (horizon, 2, NUM_BASIS) or not torch.isfinite(supports).all():
            raise RuntimeError("KickBox21 CHORD reference support computation failed")
        return supports

    @staticmethod
    def _live_contact(sensor) -> tuple[torch.Tensor, torch.Tensor]:
        force = sensor.data.force_matrix_w_history
        position = sensor.data.contact_pos_w
        if force is None or force.ndim != 5 or force.shape[2:4] != (1, 1):
            raise RuntimeError("CHORD live foot force geometry drift")
        if position is None or position.ndim != 4 or position.shape[1:3] != (1, 1):
            raise RuntimeError("CHORD live foot contact-point geometry drift")
        force = force[:, -1, 0, 0, :]
        position = position[:, 0, 0, :]
        valid = (
            torch.isfinite(position).all(dim=-1)
            & torch.isfinite(force).all(dim=-1)
            & (torch.linalg.vector_norm(force, dim=-1) > 0.1)
        )
        return (
            torch.where(valid.unsqueeze(-1), position, torch.zeros_like(position)),
            torch.where(valid.unsqueeze(-1), force, torch.zeros_like(force)),
        )

    @torch.inference_mode()
    def reward(
        self, motion_frames: torch.Tensor, kick_mask: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if motion_frames.shape != (self.env.num_envs,):
            raise RuntimeError("CHORD motion-frame geometry drift")
        if torch.any(motion_frames < 0) or torch.any(
            motion_frames >= self.reference_supports.shape[0]
        ):
            raise RuntimeError("CHORD motion frame left the KickBox21 reference horizon")

        left_position, left_force = self._live_contact(
            self.env.scene.sensors["left_foot_forces"]
        )
        right_position, right_force = self._live_contact(
            self.env.scene.sensors["right_foot_forces"]
        )
        object_position = self.env.scene["obj"].data.root_pos_w
        object_quat = self.env.scene["obj"].data.root_quat_w
        current = torch.stack(
            (
                self._supports(left_position, left_force, object_position, object_quat),
                self._supports(right_position, right_force, object_position, object_quat),
            ),
            dim=1,
        )
        command = self.reference_supports[motion_frames.to(dtype=torch.long)]
        cmd_active = command > SUPPORT_THRESHOLD
        cur_active = current > SUPPORT_THRESHOLD
        cmd_body = cmd_active.any(dim=-1, keepdim=True)
        cur_body = cur_active.any(dim=-1, keepdim=True)
        cws = self.chord.contact_wrench_support_reward_jit(
            right_cmd_active=cmd_active[:, 1].unsqueeze(1),
            right_cur_active=cur_active[:, 1].unsqueeze(1),
            left_cmd_active=cmd_active[:, 0].unsqueeze(1),
            left_cur_active=cur_active[:, 0].unsqueeze(1),
            right_cmd_active_per_body=cmd_body[:, 1],
            left_cmd_active_per_body=cmd_body[:, 0],
            right_cmd_supports=command[:, 1].unsqueeze(1),
            right_cur_supports=current[:, 1].unsqueeze(1),
            left_cmd_supports=command[:, 0].unsqueeze(1),
            left_cur_supports=current[:, 0].unsqueeze(1),
            tolerance=0.1,
            var=0.1,
        )
        unintended = self.chord.unintended_contact_penalty_jit(
            right_cmd_active_per_body=cmd_body[:, 1],
            right_cur_active_per_body=cur_body[:, 1],
            left_cmd_active_per_body=cmd_body[:, 0],
            left_cur_active_per_body=cur_body[:, 0],
            right_cur_supports=current[:, 1].unsqueeze(1),
            left_cur_supports=current[:, 0].unsqueeze(1),
            num_bodies=1,
        )
        missed = self.chord.missed_contact_penalty_jit(
            right_cmd_active=cmd_active[:, 1].unsqueeze(1),
            right_cur_active=cur_active[:, 1].unsqueeze(1),
            left_cmd_active=cmd_active[:, 0].unsqueeze(1),
            left_cur_active=cur_active[:, 0].unsqueeze(1),
            right_cmd_active_per_body=cmd_body[:, 1],
            left_cmd_active_per_body=cmd_body[:, 0],
        )
        selected = kick_mask.to(dtype=cws.dtype) * (~dones).to(dtype=cws.dtype)
        reward = selected * (
            self.cws_weight * cws
            + self.unintended_weight * unintended
            + self.missed_weight * missed
        )
        if not torch.isfinite(reward).all():
            raise RuntimeError("official CHORD runtime reward became non-finite")
        self.calls += 1
        reference_active = cmd_body.any(dim=1).squeeze(-1)
        self.active_reference_calls += int(
            (selected.bool() & reference_active).sum().item()
        )
        self.maximum_abs_reward = max(
            self.maximum_abs_reward, float(torch.amax(torch.abs(reward)).item())
        )
        return reward, {"cws": cws, "unintended": unintended, "missed": missed}

    def audit(self) -> dict[str, object]:
        return {
            "enabled": True,
            "official_commit": OFFICIAL_CHORD_COMMIT,
            "official_source": str(self.official_source.resolve()),
            "reference_geometry": str(self.reference_path),
            "object_usd": str(self.object_usd),
            "reference_frames": int(self.reference_supports.shape[0]),
            "num_basis": NUM_BASIS,
            "num_friction_edges": NUM_FRICTION_EDGES,
            "friction_coefficient": FRICTION_COEFFICIENT,
            "support_threshold": SUPPORT_THRESHOLD,
            "weights": {
                "cws": self.cws_weight,
                "unintended": self.unintended_weight,
                "missed": self.missed_weight,
            },
            "reward_calls": self.calls,
            "active_reference_env_calls": self.active_reference_calls,
            "maximum_abs_reward": self.maximum_abs_reward,
            "live_physx_contact_points_and_forces": True,
            "actor_observation_augmented": False,
            "binary_contact_label_used": False,
            "future_or_outcome_labels_used": False,
        }
