# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Posture-capable authority routing around the frozen official Refiner.

This is action-routing glue, not a replacement policy or a hand-authored
strategy.  The existing 29-DoF SUGAR-native tactile policy still chooses every
residual action.  Direct TacSL and actor-visible robot/object dynamics are the
only way for it to infer load or friction; no mass, friction, COM, success, or
simulator-oracle field is introduced here.

The admitted arm-only wrapper intentionally keeps every hip, knee, ankle, and
waist joint close to the advancing official teacher.  That makes a materially
lower squat or shifted whole-body support posture structurally unavailable.
This successor preserves the official teacher as the balance backbone while
opening bounded policy authority on hips, knees, and waist:

* arm joints: the existing causal failure release to native residual authority;
* posture joints (hips, knees, waist): preserve the admitted nominal routing
  before failure, then open bounded residual authority after failure while
  retaining a nonzero official-teacher floor;
* balance joints (ankles): the unchanged official teacher plus the original
  low residual scale.

The routing is deterministic and checkpointed.  Whether a learned policy
actually discovers a safe lower/asymmetric posture remains an empirical gate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from sugar_rl.utils.official_refiner_nominal_teacher import (
    OFFICIAL_REFINER_ACTION_DIM,
    OfficialRefinerResidualVecEnvWrapper,
)


class PostureAdaptiveOfficialRefinerResidualVecEnvWrapper(
    OfficialRefinerResidualVecEnvWrapper
):
    """Add bounded hips/knees/waist authority to arm-only failure release."""

    protocol = (
        "sugar_official_refiner_residual_posture_adaptive_vecenv_v1"
    )

    def __init__(
        self,
        env,
        checkpoint: str | Path,
        *,
        residual_scale: float,
        release_mode: str,
        linear_release_steps: int = 4,
        post_release_residual_scale: float = 1.0,
        posture_pre_failure_residual_scale: float = 0.05,
        posture_post_failure_residual_scale: float = 0.40,
        posture_post_failure_teacher_floor: float = 0.65,
        drop_grace_steps: int = 0,
        clip_actions: float | None = None,
    ) -> None:
        if not (
            residual_scale
            <= posture_pre_failure_residual_scale
            <= posture_post_failure_residual_scale
            <= 1.0
        ):
            raise ValueError(
                "posture residual scales must satisfy "
                "base <= pre-failure <= post-failure <= 1"
            )
        if not 0.0 < posture_post_failure_teacher_floor <= 1.0:
            raise ValueError(
                "posture post-failure teacher floor must be in (0,1]"
            )
        self.posture_pre_failure_residual_scale = float(
            posture_pre_failure_residual_scale
        )
        self.posture_post_failure_residual_scale = float(
            posture_post_failure_residual_scale
        )
        self.posture_post_failure_teacher_floor = float(
            posture_post_failure_teacher_floor
        )
        super().__init__(
            env,
            checkpoint,
            residual_scale=residual_scale,
            release_mode=release_mode,
            linear_release_steps=linear_release_steps,
            teacher_release_scope="arm_only",
            support_teacher_mode="advancing",
            drop_grace_steps=drop_grace_steps,
            post_release_residual_scale=post_release_residual_scale,
            clip_actions=clip_actions,
        )
        posture_tokens = ("hip", "knee", "waist")
        balance_tokens = ("ankle",)
        self.teacher_posture_indices = tuple(
            index
            for index, name in enumerate(self.teacher_joint_names)
            if any(token in name for token in posture_tokens)
        )
        self.teacher_balance_indices = tuple(
            index
            for index, name in enumerate(self.teacher_joint_names)
            if any(token in name for token in balance_tokens)
        )
        if (
            len(self.teacher_posture_indices) != 11
            or len(self.teacher_balance_indices) != 4
            or set(self.teacher_posture_indices)
            & set(self.teacher_balance_indices)
            or (
                set(self.teacher_posture_indices)
                | set(self.teacher_balance_indices)
            )
            != set(self.teacher_support_indices)
        ):
            raise RuntimeError(
                "posture/balance partition does not exactly refine the "
                "official 15-joint support partition"
            )
        partition_text = "\n".join(
            f"{index}:{name}:"
            f"{self._authority_group(index)}"
            for index, name in enumerate(self.teacher_joint_names)
        )
        self.posture_authority_partition_sha256 = hashlib.sha256(
            partition_text.encode("utf-8")
        ).hexdigest()

    def _authority_group(self, index: int) -> str:
        if index in self.teacher_manipulation_indices:
            return "manipulation"
        if index in getattr(self, "teacher_posture_indices", ()):
            return "posture"
        if index in getattr(self, "teacher_balance_indices", ()):
            return "balance"
        return "unclassified"

    @torch.no_grad()
    def _applied_teacher_coefficient(
        self, scalar: torch.Tensor
    ) -> torch.Tensor:
        coefficient = super()._applied_teacher_coefficient(scalar)
        if not hasattr(self, "teacher_posture_indices"):
            # The superclass may query routing while it is still constructing
            # the named joint partition.
            return coefficient
        posture_columns = torch.as_tensor(
            self.teacher_posture_indices,
            dtype=torch.long,
            device=coefficient.device,
        )
        if tuple(scalar.shape) != (self.num_envs,):
            raise ValueError("posture teacher release scalar shape drift")
        posture_coefficient = (
            self.posture_post_failure_teacher_floor
            + (
                1.0 - self.posture_post_failure_teacher_floor
            )
            * scalar
        )
        posture_mask = torch.zeros(
            OFFICIAL_REFINER_ACTION_DIM,
            dtype=torch.bool,
            device=coefficient.device,
        )
        posture_mask[posture_columns] = True
        return torch.where(
            posture_mask.reshape(1, -1),
            posture_coefficient.reshape(-1, 1),
            coefficient,
        )

    @torch.no_grad()
    def _applied_residual_scale(
        self, teacher_coefficient: torch.Tensor
    ) -> torch.Tensor:
        scale = super()._applied_residual_scale(teacher_coefficient)
        if not hasattr(self, "teacher_posture_indices"):
            return scale
        manipulation_column = self.teacher_manipulation_indices[0]
        release_scalar = teacher_coefficient[:, manipulation_column]
        posture_scale = (
            self.posture_pre_failure_residual_scale
            + (1.0 - release_scalar)
            * (
                self.posture_post_failure_residual_scale
                - self.posture_pre_failure_residual_scale
            )
        )
        posture_columns = torch.as_tensor(
            self.teacher_posture_indices,
            dtype=torch.long,
            device=scale.device,
        )
        posture_mask = torch.zeros(
            OFFICIAL_REFINER_ACTION_DIM,
            dtype=torch.bool,
            device=scale.device,
        )
        posture_mask[posture_columns] = True
        return torch.where(
            posture_mask.reshape(1, -1),
            posture_scale.reshape(-1, 1),
            scale,
        )

    @torch.no_grad()
    def teacher_partition_audit_state(self) -> dict[str, Any]:
        state = super().teacher_partition_audit_state()
        state.update(
            {
                "authority_protocol": self.protocol,
                "posture_indices": list(self.teacher_posture_indices),
                "posture_joint_names": [
                    self.teacher_joint_names[index]
                    for index in self.teacher_posture_indices
                ],
                "balance_indices": list(self.teacher_balance_indices),
                "balance_joint_names": [
                    self.teacher_joint_names[index]
                    for index in self.teacher_balance_indices
                ],
                "posture_pre_failure_residual_scale": (
                    self.posture_pre_failure_residual_scale
                ),
                "posture_post_failure_residual_scale": (
                    self.posture_post_failure_residual_scale
                ),
                "posture_post_failure_teacher_floor": (
                    self.posture_post_failure_teacher_floor
                ),
                "posture_authority_partition_sha256": (
                    self.posture_authority_partition_sha256
                ),
                "three_way_complete_disjoint_partition": (
                    set(self.teacher_manipulation_indices)
                    | set(self.teacher_posture_indices)
                    | set(self.teacher_balance_indices)
                    == set(range(OFFICIAL_REFINER_ACTION_DIM))
                    and not (
                        set(self.teacher_manipulation_indices)
                        & set(self.teacher_posture_indices)
                    )
                    and not (
                        set(self.teacher_manipulation_indices)
                        & set(self.teacher_balance_indices)
                    )
                    and not (
                        set(self.teacher_posture_indices)
                        & set(self.teacher_balance_indices)
                    )
                ),
            }
        )
        return state

    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = super().checkpoint_state_dict()
        state.update(
            {
                "posture_pre_failure_residual_scale": (
                    self.posture_pre_failure_residual_scale
                ),
                "posture_post_failure_residual_scale": (
                    self.posture_post_failure_residual_scale
                ),
                "posture_post_failure_teacher_floor": (
                    self.posture_post_failure_teacher_floor
                ),
                "teacher_posture_indices": self.teacher_posture_indices,
                "teacher_balance_indices": self.teacher_balance_indices,
                "posture_authority_partition_sha256": (
                    self.posture_authority_partition_sha256
                ),
            }
        )
        return state

    @torch.inference_mode()
    def load_checkpoint_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "posture_pre_failure_residual_scale": (
                self.posture_pre_failure_residual_scale
            ),
            "posture_post_failure_residual_scale": (
                self.posture_post_failure_residual_scale
            ),
            "posture_post_failure_teacher_floor": (
                self.posture_post_failure_teacher_floor
            ),
            "teacher_posture_indices": self.teacher_posture_indices,
            "teacher_balance_indices": self.teacher_balance_indices,
            "posture_authority_partition_sha256": (
                self.posture_authority_partition_sha256
            ),
        }
        drift = {
            name: {"actual": state.get(name), "expected": value}
            for name, value in expected.items()
            if state.get(name) != value
        }
        if drift:
            raise ValueError(
                f"posture-authority checkpoint drift: {drift}"
            )
        super().load_checkpoint_state_dict(state)
