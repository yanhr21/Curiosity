# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Causal native-TacSL history used by the no-RGB CarryBox student.

The observation keeps the current whole-hand representation without contact
labels, object state, simulator contact velocity, thresholding, pooling, or
interpolation.  Its logical layout is
``[environment, history, hand, patch, channel, row, column]`` where channel is
exactly native normal force and signed local-X/Y shear.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1 import (
    ANATOMICAL_WHOLE_HAND_PATCH_SPECS,
)


NATIVE_TACTILE_PATCH_NAMES = tuple(
    specification.name for specification in ANATOMICAL_WHOLE_HAND_PATCH_SPECS
)
NATIVE_TACTILE_SENSOR_NAMES = tuple(
    tuple(f"{side}_{patch}_tactile" for patch in NATIVE_TACTILE_PATCH_NAMES)
    for side in ("left", "right")
)
NATIVE_TACTILE_GRID_SHAPE = (20, 25)
NATIVE_TACTILE_HISTORY_STEPS = 4
NATIVE_TACTILE_CHANNELS = (
    "normal_force_n",
    "signed_shear_x_n",
    "signed_shear_y_n",
)
# Fixed conditioning values from the canonical CarryBox visualization.  They
# change units only; no clipping or contact-dependent rescaling is applied.
NATIVE_TACTILE_NORMAL_SCALE_N = 0.5768324136734009
NATIVE_TACTILE_SHEAR_SCALE_N = 0.5144117593765258


def native_whole_hand_tactile_contract() -> dict[str, Any]:
    channels_per_hand = (
        NATIVE_TACTILE_HISTORY_STEPS
        * len(NATIVE_TACTILE_PATCH_NAMES)
        * len(NATIVE_TACTILE_CHANNELS)
    )
    return {
        "source": "54 physical IsaacLab/TacSL patch sensors",
        "logging_layout": [
            "history",
            "hand",
            "patch",
            "channel",
            "row",
            "column",
        ],
        "actor_layout": [
            "hand",
            "history",
            "patch",
            "channel",
            "row",
            "column",
        ],
        "hands": 2,
        "patches_per_hand": len(NATIVE_TACTILE_PATCH_NAMES),
        "history_steps": NATIVE_TACTILE_HISTORY_STEPS,
        "channels": list(NATIVE_TACTILE_CHANNELS),
        "grid_shape": list(NATIVE_TACTILE_GRID_SHAPE),
        "channels_per_hand_after_serialization": channels_per_hand,
        "flat_width": 2 * channels_per_hand * 20 * 25,
        "uses_rgb": False,
        "uses_object_state": False,
        "uses_contact_proxy": False,
        "uses_simulator_contact_velocity": False,
    }


def _validate(
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]],
    history_steps: int,
    grid_shape: tuple[int, int],
    normal_scale_n: float,
    shear_scale_n: float,
) -> None:
    if len(sensor_names_by_hand) != 2 or any(
        len(names) != 27 for names in sensor_names_by_hand
    ):
        raise ValueError("native whole-hand tactile requires 27 patches on each hand")
    if len(set(sensor_names_by_hand[0] + sensor_names_by_hand[1])) != 54:
        raise ValueError("native whole-hand tactile sensor names must be unique")
    if history_steps < 2:
        raise ValueError("native whole-hand tactile requires at least two frames")
    if tuple(int(value) for value in grid_shape) != NATIVE_TACTILE_GRID_SHAPE:
        raise ValueError(f"native whole-hand tactile grid must be {NATIVE_TACTILE_GRID_SHAPE}")
    for name, value in (
        ("normal_scale_n", normal_scale_n),
        ("shear_scale_n", shear_scale_n),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")


def _patch_force_map(
    env,
    sensor_name: str,
    grid_shape: tuple[int, int],
    normal_scale_n: float,
    shear_scale_n: float,
) -> torch.Tensor:
    data = env.scene.sensors[sensor_name].data
    taxels = int(grid_shape[0]) * int(grid_shape[1])
    normal = data.tactile_normal_force
    shear = data.tactile_shear_force
    if normal is None or tuple(normal.shape) != (env.num_envs, taxels):
        raise RuntimeError(
            f"unexpected normal force for {sensor_name}: "
            f"{None if normal is None else tuple(normal.shape)}"
        )
    if shear is None or tuple(shear.shape) != (env.num_envs, taxels, 2):
        raise RuntimeError(
            f"unexpected signed shear for {sensor_name}: "
            f"{None if shear is None else tuple(shear.shape)}"
        )
    scaled = torch.cat(
        (
            normal.unsqueeze(-1) / float(normal_scale_n),
            shear / float(shear_scale_n),
        ),
        dim=-1,
    )
    return torch.nan_to_num(scaled).transpose(1, 2).reshape(
        env.num_envs, len(NATIVE_TACTILE_CHANNELS), *grid_shape
    )


def _current_field(
    env,
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]],
    grid_shape: tuple[int, int],
    normal_scale_n: float,
    shear_scale_n: float,
) -> torch.Tensor:
    hands = []
    for names in sensor_names_by_hand:
        hands.append(
            torch.stack(
                [
                    _patch_force_map(
                        env,
                        name,
                        grid_shape,
                        normal_scale_n,
                        shear_scale_n,
                    )
                    for name in names
                ],
                dim=1,
            )
        )
    current = torch.stack(hands, dim=1)
    expected = (
        env.num_envs,
        2,
        27,
        3,
        int(grid_shape[0]),
        int(grid_shape[1]),
    )
    if tuple(current.shape) != expected or not torch.isfinite(current).all():
        raise RuntimeError(
            f"native whole-hand tactile field is invalid: {tuple(current.shape)}"
        )
    return current


def native_whole_hand_tactile_actor_history(
    env,
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]] = (
        NATIVE_TACTILE_SENSOR_NAMES
    ),
    history_steps: int = NATIVE_TACTILE_HISTORY_STEPS,
    grid_shape: tuple[int, int] = NATIVE_TACTILE_GRID_SHAPE,
    normal_scale_n: float = NATIVE_TACTILE_NORMAL_SCALE_N,
    shear_scale_n: float = NATIVE_TACTILE_SHEAR_SCALE_N,
) -> torch.Tensor:
    """Return actor layout ``[hand,history,patch,channel,row,column]``."""

    sensor_names_by_hand = tuple(
        tuple(names) for names in sensor_names_by_hand
    )
    _validate(
        sensor_names_by_hand,
        history_steps,
        grid_shape,
        normal_scale_n,
        shear_scale_n,
    )
    key = (
        sensor_names_by_hand,
        int(history_steps),
        tuple(int(value) for value in grid_shape),
        float(normal_scale_n),
        float(shear_scale_n),
    )
    cache = getattr(env, "_native_whole_hand_tactile_history_cache", {})
    setattr(env, "_native_whole_hand_tactile_history_cache", cache)
    step = int(env.common_step_counter)
    entry = cache.get(key)
    reset_mask = env.episode_length_buf == 0
    if entry is None or int(entry["step"]) != step:
        current = _current_field(
            env,
            sensor_names_by_hand,
            grid_shape,
            normal_scale_n,
            shear_scale_n,
        )
        if entry is None:
            history = current[:, None].repeat(
                1, history_steps, 1, 1, 1, 1, 1
            )
            entry = {"step": step, "history": history}
            cache[key] = entry
        else:
            history = entry["history"]
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = current
            entry["step"] = step
        if reset_mask.any():
            reset_view = reset_mask.reshape(
                env.num_envs, *(1 for _ in range(history.ndim - 1))
            )
            history = torch.where(reset_view, current[:, None], history)
            entry["history"] = history
    else:
        history = entry["history"]
    actor = history.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
    return actor.reshape(env.num_envs, -1)


def exact_zero_native_whole_hand_tactile_actor_history(
    env,
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]] = (
        NATIVE_TACTILE_SENSOR_NAMES
    ),
    history_steps: int = NATIVE_TACTILE_HISTORY_STEPS,
    grid_shape: tuple[int, int] = NATIVE_TACTILE_GRID_SHAPE,
    normal_scale_n: float = NATIVE_TACTILE_NORMAL_SCALE_N,
    shear_scale_n: float = NATIVE_TACTILE_SHEAR_SCALE_N,
) -> torch.Tensor:
    """Return the matched exact-zero arm without reading a tactile sensor."""

    _validate(
        sensor_names_by_hand,
        history_steps,
        grid_shape,
        normal_scale_n,
        shear_scale_n,
    )
    width = (
        2
        * int(history_steps)
        * 27
        * 3
        * int(grid_shape[0])
        * int(grid_shape[1])
    )
    return torch.zeros((env.num_envs, width), dtype=torch.float32, device=env.device)
