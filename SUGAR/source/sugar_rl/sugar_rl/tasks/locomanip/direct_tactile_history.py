# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Temporal buffer for direct TacSL R15 pressure and signed shear fields."""

from __future__ import annotations

from typing import Any

import torch

from sugar_rl.tasks.locomanip.mdp.observations import (
    tactile_contact_velocity_maps,
    tactile_force_maps,
)
from sugar_rl.tasks.locomanip.direct_tactile_stress import (
    apply_configured_direct_tactile_stress,
    direct_tactile_stress_cache_key,
)


def explicit_zero_tactile_force_history(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int = 4,
    grid_shape: tuple[int, int] = (20, 25),
    taxel_area_m2: float = 1.18138624e-6,
    stress_scale: float = 1.0e-5,
) -> torch.Tensor:
    """Return the exact-width all-zero tactile control without sensor reads.

    This is an observation ablation, not a tactile model.  The unused sensor
    names and scaling arguments are accepted only so an existing IsaacLab
    observation term can switch functions without changing its declared
    parameter schema or actor width.  In particular, this function never
    indexes ``env.scene`` and never advances a tactile cache.
    """

    del left_sensor_name, right_sensor_name, taxel_area_m2, stress_scale
    if history_steps < 2:
        raise ValueError("explicit zero tactile history requires at least two frames")
    if len(grid_shape) != 2 or min(int(value) for value in grid_shape) <= 0:
        raise ValueError("explicit zero tactile grid must be two-dimensional and positive")
    width = int(history_steps) * 2 * 3 * int(grid_shape[0]) * int(grid_shape[1])
    return torch.zeros(
        (env.num_envs, width),
        dtype=torch.float32,
        device=env.device,
    )


def direct_tactile_force_history(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int = 4,
    grid_shape: tuple[int, int] = (20, 25),
    taxel_area_m2: float = 1.18138624e-6,
    stress_scale: float = 1.0e-5,
) -> torch.Tensor:
    """Return ``[env,history,hand,pressure/shear,row,col]`` flattened.

    The buffer is advanced at most once per simulator control step even if the
    policy, ICM, slip estimator, and logger request it independently. Reset
    environments are filled with the first post-reset direct sensor frame so no
    transition crosses an episode boundary.
    """

    if history_steps < 2:
        raise ValueError("direct tactile history requires at least two frames")
    current = tactile_force_maps(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        grid_shape=grid_shape,
        taxel_area_m2=taxel_area_m2,
        stress_scale=stress_scale,
    ).reshape(env.num_envs, 2, 3, *grid_shape)
    reset_mask = env.episode_length_buf == 0
    current = apply_configured_direct_tactile_stress(
        env,
        current,
        reset_mask=reset_mask,
    )
    if not torch.isfinite(current).all():
        raise RuntimeError("direct TacSL history received non-finite pressure/shear")

    cache: dict[tuple[Any, ...], dict[str, Any]] = getattr(
        env, "_sugar_direct_tactile_history_cache", {}
    )
    setattr(env, "_sugar_direct_tactile_history_cache", cache)
    key = (
        left_sensor_name,
        right_sensor_name,
        history_steps,
        tuple(grid_shape),
        float(taxel_area_m2),
        float(stress_scale),
        direct_tactile_stress_cache_key(env),
    )
    step = int(env.common_step_counter)
    entry = cache.get(key)
    if entry is None:
        history = current[:, None].repeat(1, history_steps, 1, 1, 1, 1)
        entry = {"step": step, "history": history}
        cache[key] = entry
    elif entry["step"] != step:
        history = entry["history"]
        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = current
        entry["step"] = step
    else:
        history = entry["history"]

    if reset_mask.any():
        # Boolean advanced-index assignment is routed through CUDA index_put.
        # Under torch.use_deterministic_algorithms it can mis-broadcast this
        # six-dimensional history buffer.  A broadcasted where is both
        # deterministic and preserves the exact reset semantics.
        reset_view = reset_mask.reshape(
            env.num_envs, *(1 for _ in range(history.ndim - 1))
        )
        history = torch.where(reset_view, current[:, None], history)
        entry["history"] = history
    return history.reshape(env.num_envs, -1)


def direct_tactile_contact_velocity_history(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int = 4,
    grid_shape: tuple[int, int] = (20, 25),
    velocity_scale: float = 1.0,
) -> torch.Tensor:
    """Return exact simulator-only TacSL contact-velocity history flattened.

    The unflattened layout is
    ``[env, history, hand, (normal,tangent_x,tangent_y), row, col]`` using the
    contacted-object-minus-sensor convention shared with KinematicTaxel.
    It is advanced and reset with the same causal clock contract as direct
    force history, but kept in a separate cache and log namespace because it
    is a simulator oracle rather than a hardware tactile measurement.
    """

    if history_steps < 2:
        raise ValueError("direct contact-velocity history requires at least two frames")
    current = tactile_contact_velocity_maps(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        grid_shape=grid_shape,
        velocity_scale=velocity_scale,
    ).reshape(env.num_envs, 2, 3, *grid_shape)
    if not torch.isfinite(current).all():
        raise RuntimeError("TacSL contact-velocity history received non-finite values")

    cache: dict[tuple[Any, ...], dict[str, Any]] = getattr(
        env, "_sugar_direct_tactile_velocity_history_cache", {}
    )
    setattr(env, "_sugar_direct_tactile_velocity_history_cache", cache)
    key = (
        left_sensor_name,
        right_sensor_name,
        history_steps,
        tuple(grid_shape),
        float(velocity_scale),
    )
    step = int(env.common_step_counter)
    reset_mask = env.episode_length_buf == 0
    entry = cache.get(key)
    if entry is None:
        history = current[:, None].repeat(1, history_steps, 1, 1, 1, 1)
        entry = {"step": step, "history": history}
        cache[key] = entry
    elif entry["step"] != step:
        history = entry["history"]
        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = current
        entry["step"] = step
    else:
        history = entry["history"]

    if reset_mask.any():
        reset_view = reset_mask.reshape(
            env.num_envs, *(1 for _ in range(history.ndim - 1))
        )
        history = torch.where(reset_view, current[:, None], history)
        entry["history"] = history
    return history.reshape(env.num_envs, -1)


def direct_tactile_force_velocity_history(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int = 4,
    grid_shape: tuple[int, int] = (20, 25),
    taxel_area_m2: float = 1.18138624e-6,
    stress_scale: float = 1.0e-5,
    velocity_scale: float = 1.0,
) -> torch.Tensor:
    """Return causal force plus contact-velocity spatial history.

    The unflattened layout is
    ``[env, history, hand, 6, row, col]`` with channels pressure, shear-X,
    shear-Y, normal relative speed, tangent-X speed and tangent-Y speed.  No
    channel is pooled, thresholded, interpolated or finite-differenced.  This
    temporal-first layout is the common logging/analysis contract; convolutional
    actors must call :func:`direct_tactile_force_velocity_actor_history` so a
    same-size reshape cannot interleave hands and history frames.
    """

    force = direct_tactile_force_history(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        history_steps=history_steps,
        grid_shape=grid_shape,
        taxel_area_m2=taxel_area_m2,
        stress_scale=stress_scale,
    ).reshape(env.num_envs, history_steps, 2, 3, *grid_shape)
    velocity = direct_tactile_contact_velocity_history(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        history_steps=history_steps,
        grid_shape=grid_shape,
        velocity_scale=velocity_scale,
    ).reshape(env.num_envs, history_steps, 2, 3, *grid_shape)
    combined = torch.cat((force, velocity), dim=3)
    return combined.reshape(env.num_envs, -1)


def direct_tactile_force_velocity_actor_history(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int = 4,
    grid_shape: tuple[int, int] = (20, 25),
    taxel_area_m2: float = 1.18138624e-6,
    stress_scale: float = 1.0e-5,
    velocity_scale: float = 1.0,
    zero_force: bool = False,
    zero_sim_contact_velocity: bool = False,
) -> torch.Tensor:
    """Return the explicit per-hand convolutional actor layout.

    The logical tensor is
    ``[env, hand, history, (pressure,shear_x,shear_y,sim_vn,sim_vx,sim_vy), row, col]``.
    Flattening it therefore matches a spatial encoder configured with
    ``history_steps * 6`` channels per hand.  ``sim_v*`` is the exact
    simulator-only contacted-object-minus-sensor callable; its name and the
    separate zeroing switch prevent it from being represented as a hardware
    measurement.  The two zeroing switches implement matched observation
    controls without changing tensor width or channel order.
    """

    combined = direct_tactile_force_velocity_history(
        env,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        history_steps=history_steps,
        grid_shape=grid_shape,
        taxel_area_m2=taxel_area_m2,
        stress_scale=stress_scale,
        velocity_scale=velocity_scale,
    ).reshape(env.num_envs, history_steps, 2, 6, *grid_shape)
    if zero_force or zero_sim_contact_velocity:
        combined = combined.clone()
        if zero_force:
            combined[:, :, :, :3].zero_()
        if zero_sim_contact_velocity:
            combined[:, :, :, 3:].zero_()
    actor_layout = combined.permute(0, 2, 1, 3, 4, 5).contiguous()
    return actor_layout.reshape(env.num_envs, -1)
