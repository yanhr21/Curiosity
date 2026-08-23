# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime paper-CWS guidance on official TacSL geometry.

This is project glue around the public CHORD paper equations.  It is not
NVIDIA's unreleased implementation.  The scorer is training-only privileged
reward logic: exact SDF normals are never exposed to the actor or to original
ICM.  Slip, task outcome, safety, SMP, and ICM remain separate signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sugar_rl.utils.paper_contact_wrench_support import (
    PaperCWSConfig,
    contact_wrench_support,
    paper_cws_terms,
    sample_paper_support_directions,
)
from sugar_rl.utils.tacsl_paper_cws_adapter import (
    tacsl_contacts_to_object_frame,
)


_SENSOR_NAMES = ("left_palm_tactile", "right_palm_tactile")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PaperCWSRuntimeCfg:
    """Explicit runtime settings and hash-bound reference schedule."""

    reference_arrays_path: str
    reference_arrays_sha256: str
    reference_motion_frame_offset: int
    friction_coefficient: float
    friction_cone_edges: int
    relative_tolerance: float
    reward_variance: float
    support_direction_seed: int
    reference_support_key: str = "support_aggregate_nominal"
    reference_normal_force_key: str = "normal_force_nominal"
    active_force_epsilon: float = 0.0
    support_direction_count: int = 512
    support_basis_chunk_size: int = 64
    contact_epsilon: float = 1.0e-9

    def paper_config(self) -> PaperCWSConfig:
        return PaperCWSConfig(
            friction_coefficient=self.friction_coefficient,
            friction_cone_edges=self.friction_cone_edges,
            relative_tolerance=self.relative_tolerance,
            reward_variance=self.reward_variance,
            support_direction_seed=self.support_direction_seed,
            support_direction_count=self.support_direction_count,
            support_basis_chunk_size=self.support_basis_chunk_size,
            contact_epsilon=self.contact_epsilon,
        )


@dataclass(frozen=True)
class PaperCWSRuntimeSignals:
    reward: torch.Tensor
    lower_violation_squared: torch.Tensor
    upper_violation_squared: torch.Tensor
    missed_contact: torch.Tensor
    unintended_contact: torch.Tensor
    reference_index: torch.Tensor
    reference_index_clamped: torch.Tensor
    reference_valid: torch.Tensor
    active_taxels: torch.Tensor
    robot_support: torch.Tensor
    reference_support: torch.Tensor


class OfficialTacSLPaperCWSReward:
    """Score current official TacSL contact capability against a demo schedule."""

    protocol = "official_tacsl_paper_cws_runtime_reward_v3"

    def __init__(
        self,
        env,
        cfg: PaperCWSRuntimeCfg,
    ) -> None:
        self.env = env
        self.device = torch.device(env.device)
        if self.device.type != "cuda":
            raise ValueError("runtime paper-CWS scoring requires a compute GPU")
        self.cfg = cfg
        self.paper_cfg = cfg.paper_config()
        reference_path = Path(cfg.reference_arrays_path).expanduser().resolve()
        if (
            not reference_path.is_file()
            or _sha256(reference_path) != cfg.reference_arrays_sha256
        ):
            raise RuntimeError("paper-CWS reference array binding drift")
        with np.load(reference_path, allow_pickle=False) as archive:
            if (
                cfg.reference_support_key not in archive
                or cfg.reference_normal_force_key not in archive
            ):
                raise KeyError("paper-CWS reference schedule fields are absent")
            support = np.asarray(
                archive[cfg.reference_support_key], dtype=np.float32
            )
            normal = np.asarray(
                archive[cfg.reference_normal_force_key], dtype=np.float32
            )
        if (
            support.ndim != 2
            or support.shape[1] != 512
            or normal.ndim != 4
            or normal.shape[0] != support.shape[0]
            or normal.shape[1:] != (2, 20, 25)
            or not np.isfinite(support).all()
            or not np.isfinite(normal).all()
            or float(normal.min()) < 0.0
        ):
            raise RuntimeError("paper-CWS reference schedule geometry drift")
        self.reference_path = reference_path
        self.reference_support_schedule = torch.as_tensor(
            support, device=self.device
        )
        self.reference_has_contact_schedule = torch.as_tensor(
            normal.reshape(normal.shape[0], -1).sum(axis=-1) > 0.0,
            dtype=torch.bool,
            device=self.device,
        )
        self.basis = sample_paper_support_directions(
            count=self.paper_cfg.support_direction_count,
            seed=self.paper_cfg.support_direction_seed,
            dtype=torch.float32,
            device=self.device,
        )
        self.steps_scored = 0
        self.clamped_environment_steps = 0
        self.invalid_environment_steps_masked = 0
        self.invalid_active_taxels_masked = 0
        self.invalid_active_taxels_with_missing_normals_masked = 0

    def _current_contacts(
        self, environment_valid: torch.Tensor | None = None
    ):
        points = []
        normals = []
        normal_force = []
        transforms = []
        for sensor_name in _SENSOR_NAMES:
            sensor = self.env.scene[sensor_name]
            data = sensor.data
            points.append(data.tactile_points_pos_w)
            normals.append(data.tactile_contact_normal_w)
            normal_force.append(data.tactile_normal_force)
            transforms.append(
                sensor._contact_object_body_view.get_transforms()
            )
        points_w = torch.stack(points, dim=1)
        normals_w = torch.stack(normals, dim=1)
        force = torch.stack(normal_force, dim=1)
        object_transform = torch.stack(transforms, dim=1)
        if (
            points_w.shape[:2] != (self.env.num_envs, 2)
            or points_w.shape[-1] != 3
            or force.shape != points_w.shape[:-1]
            or normals_w.shape != points_w.shape
            or object_transform.shape != (self.env.num_envs, 2, 7)
        ):
            raise RuntimeError("live official TacSL paper-CWS geometry drift")
        if not torch.equal(
            object_transform[:, 0], object_transform[:, 1]
        ):
            maximum = float(
                torch.abs(
                    object_transform[:, 0] - object_transform[:, 1]
                ).max()
            )
            raise RuntimeError(
                "left/right TacSL contact-object transforms disagree; "
                f"max_abs={maximum}"
            )
        if environment_valid is None:
            environment_valid = torch.ones(
                self.env.num_envs, dtype=torch.bool, device=self.device
            )
        if (
            environment_valid.shape != (self.env.num_envs,)
            or environment_valid.dtype != torch.bool
            or environment_valid.device != self.device
        ):
            raise ValueError(
                "paper-CWS environment-valid mask geometry drift"
            )
        invalid = ~environment_valid
        raw_active = force > self.cfg.active_force_epsilon
        normal_norm = torch.linalg.vector_norm(normals_w, dim=-1)
        self.invalid_environment_steps_masked += int(invalid.sum())
        self.invalid_active_taxels_masked += int(
            (raw_active & invalid[:, None, None]).sum()
        )
        self.invalid_active_taxels_with_missing_normals_masked += int(
            (
                raw_active
                & (normal_norm <= 1.0e-6)
                & invalid[:, None, None]
            ).sum()
        )
        # After an environment reset the hash-bound tactile history is
        # restored before the next physical sensor update.  That observation
        # is not a valid PPO/SMP/ICM transition, so it must contribute neither
        # inferred contact capability nor CWS reward.
        force = torch.where(
            environment_valid[:, None, None], force, torch.zeros_like(force)
        )
        contacts = tacsl_contacts_to_object_frame(
            tactile_points_pos_w=points_w,
            tactile_sdf_outward_normal_w=normals_w,
            object_transform_xyzw=object_transform,
            normal_force=force,
            active_force_epsilon=self.cfg.active_force_epsilon,
        )
        return contacts

    @torch.no_grad()
    def score(
        self, environment_valid: torch.Tensor | None = None
    ) -> PaperCWSRuntimeSignals:
        if environment_valid is None:
            environment_valid = torch.ones(
                self.env.num_envs, dtype=torch.bool, device=self.device
            )
        contacts = self._current_contacts(environment_valid)
        positions = contacts.positions_object.flatten(1, -2)
        normals = contacts.force_normals_object.flatten(1, -2)
        active = contacts.active_contact.flatten(1, -1)
        robot_support = contact_wrench_support(
            positions,
            normals,
            active,
            self.basis,
            friction_coefficient=self.paper_cfg.friction_coefficient,
            edge_count=self.paper_cfg.friction_cone_edges,
            basis_chunk_size=self.paper_cfg.support_basis_chunk_size,
        )
        command = self.env.command_manager.get_term("motion")
        raw_index = (
            command.time_steps.to(dtype=torch.long)
            - int(self.cfg.reference_motion_frame_offset)
        )
        clamped = (raw_index < 0) | (
            raw_index >= self.reference_support_schedule.shape[0]
        )
        reference_index = raw_index.clamp(
            0, self.reference_support_schedule.shape[0] - 1
        )
        reference_support = self.reference_support_schedule[reference_index]
        reference_has_contact = self.reference_has_contact_schedule[
            reference_index
        ]
        robot_has_contact = active.any(dim=-1)
        terms = paper_cws_terms(
            reference_support,
            robot_support,
            reference_has_contact,
            robot_has_contact,
            relative_tolerance=self.paper_cfg.relative_tolerance,
            reward_variance=self.paper_cfg.reward_variance,
            contact_epsilon=self.paper_cfg.contact_epsilon,
        )
        reference_valid = (~clamped) & environment_valid
        reward = torch.where(
            reference_valid, terms.reward, torch.zeros_like(terms.reward)
        )
        lower_violation_squared = torch.where(
            reference_valid,
            terms.lower_violation_squared,
            torch.zeros_like(terms.lower_violation_squared),
        )
        upper_violation_squared = torch.where(
            reference_valid,
            terms.upper_violation_squared,
            torch.zeros_like(terms.upper_violation_squared),
        )
        missed_contact = reference_valid & terms.missed_contact
        unintended_contact = reference_valid & terms.unintended_contact
        if not torch.isfinite(reward).all():
            raise RuntimeError("paper-CWS runtime reward is non-finite")
        self.steps_scored += 1
        self.clamped_environment_steps += int(clamped.sum())
        return PaperCWSRuntimeSignals(
            reward=reward.detach().clone(),
            lower_violation_squared=(
                lower_violation_squared.detach().clone()
            ),
            upper_violation_squared=(
                upper_violation_squared.detach().clone()
            ),
            missed_contact=missed_contact.detach().clone(),
            unintended_contact=unintended_contact.detach().clone(),
            reference_index=reference_index.detach().clone(),
            reference_index_clamped=clamped.detach().clone(),
            reference_valid=reference_valid.detach().clone(),
            active_taxels=active.sum(dim=-1).detach().clone(),
            robot_support=robot_support.detach().clone(),
            reference_support=reference_support.detach().clone(),
        )

    def audit_state(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "status": "paper_formula_reproduction_not_official_chord_code",
            "training_only_privileged_reward": True,
            "actor_receives_sdf_normals": False,
            "original_icm_receives_sdf_normals": False,
            "reference_arrays_path": str(self.reference_path),
            "reference_arrays_sha256": self.cfg.reference_arrays_sha256,
            "config": asdict(self.cfg),
            "paper_config": self.paper_cfg.to_dict(),
            "reference_schedule_shape": list(
                self.reference_support_schedule.shape
            ),
            "steps_scored": self.steps_scored,
            "clamped_environment_steps": self.clamped_environment_steps,
            "out_of_support_reward_policy": "strict_zero",
            "invalid_transition_reward_policy": (
                "strict_zero_and_excluded_from_contact_geometry"
            ),
            "invalid_environment_steps_masked": (
                self.invalid_environment_steps_masked
            ),
            "invalid_active_taxels_masked": (
                self.invalid_active_taxels_masked
            ),
            "invalid_active_taxels_with_missing_normals_masked": (
                self.invalid_active_taxels_with_missing_normals_masked
            ),
        }
