# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Matched observation ablations for the official TacSL policy branch.

Every mode starts from the same taxel-resolved official TacSL observation.
Masking happens only at the policy-observation boundary so robot/object
geometry, SDF queries, sensor update cost, actor architecture, parameter count,
rewards, and dynamics remain matched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from sugar_rl.tasks.locomanip.mdp.observations import tactile_force_maps

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


TactileAblationMode = Literal["live", "pressure_only", "zero"]


def matched_tactile_force_maps(
    env: ManagerBasedEnv,
    left_sensor_name: str,
    right_sensor_name: str,
    grid_shape: tuple[int, int] = (20, 25),
    taxel_area_m2: float = 1.18138624e-6,
    stress_scale: float = 1.0e-5,
    mode: TactileAblationMode = "live",
) -> torch.Tensor:
    """Return live, pressure-only, or zeroed maps with an identical shape.

    ``zero`` is a non-tactile matched control, not a tactile modality. The
    official sensors are still evaluated so this control cannot gain a
    throughput or scene-geometry advantage over the live branch.
    """

    flat_maps = tactile_force_maps(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        grid_shape=grid_shape,
        taxel_area_m2=taxel_area_m2,
        stress_scale=stress_scale,
    )
    if mode == "live":
        return flat_maps
    if mode == "zero":
        return torch.zeros_like(flat_maps)
    if mode == "pressure_only":
        maps = flat_maps.reshape(env.num_envs, 2, 3, *grid_shape).clone()
        maps[:, :, 1:] = 0.0
        return maps.reshape(env.num_envs, -1)
    raise ValueError(f"Unsupported matched tactile ablation mode: {mode}")
