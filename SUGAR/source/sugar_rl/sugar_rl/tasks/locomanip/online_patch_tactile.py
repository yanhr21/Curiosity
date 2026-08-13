# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Online bilateral anatomical-patch observations from official TacSL fields.

Official R15 taxels remain the physical sampling backend.  This module reduces
the current taxels inside each physical elastomer to one policy token per
anatomical patch.  It never reads rigid ContactSensor data or object state.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1 import (
    ANATOMICAL_WHOLE_HAND_PATCH_SPECS,
)


PATCH_NAMES = tuple(spec.name for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS)
SENSOR_NAMES_BY_HAND = tuple(
    tuple(f"{side}_{name}_tactile" for name in PATCH_NAMES)
    for side in ("left", "right")
)
PATCH_AREAS_M2 = tuple(
    float(spec.width_m * spec.length_m)
    for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS
)
PATCH_HISTORY_STEPS = 4
BASE_PATCH_CHANNELS = (
    "contact",
    "normal_load_n",
    "mean_pressure_pa",
    "signed_shear_x_n",
    "signed_shear_y_n",
    "friction_utilization",
)
SLIP_PATCH_CHANNELS = (
    "slip_score",
    "incipient_slip",
    "gross_slip",
)
PATCH_CHANNELS = BASE_PATCH_CHANNELS + SLIP_PATCH_CHANNELS
PATCH_FEATURE_WIDTH = len(PATCH_CHANNELS)

if len(PATCH_NAMES) != 27 or len(PATCH_AREAS_M2) != 27:
    raise RuntimeError("online patch tactile requires exactly 27 patches per hand")


def online_patch_tactile_contract() -> dict[str, Any]:
    """Return the frozen policy-facing patch contract."""

    return {
        "source": "official IsaacLab VisuoTactileSensor/R15 taxel tensors",
        "policy_unit": "physical anatomical patch",
        "layout": ["history", "hand", "patch", "channel"],
        "history_steps": PATCH_HISTORY_STEPS,
        "hands": 2,
        "patches_per_hand": 27,
        "channels": list(PATCH_CHANNELS),
        "shape_without_batch": [PATCH_HISTORY_STEPS, 2, 27, PATCH_FEATURE_WIDTH],
        "flat_width": PATCH_HISTORY_STEPS * 2 * 27 * PATCH_FEATURE_WIDTH,
        "uses_taxels_as_policy_units": False,
        "uses_object_state": False,
        "uses_contact_proxy": False,
        "uses_simulator_contact_velocity": False,
        "online": True,
    }


def normalized_motion_phase(env, command_name: str = "motion") -> torch.Tensor:
    """Return the causal normalized reference phase as one actor feature."""

    command = env.command_manager.get_term(command_name)
    final_step = (
        command.motion.time_step_total_permotion[command.motion_id] - 1
    ).clamp_min(1)
    phase = command.time_steps.to(torch.float32) / final_step.to(torch.float32)
    return phase.clamp_(0.0, 1.0).unsqueeze(-1)


def reduce_patch_taxels(
    penetration_m: torch.Tensor,
    signed_normal_force_n: torch.Tensor,
    signed_shear_xy_n: torch.Tensor,
    patch_area_m2: torch.Tensor | float,
    friction_coefficient: torch.Tensor | float,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Reduce official taxels to six physical features for one patch.

    The leading dimensions are arbitrary and the final dimension is taxel.
    TacSL local-Z compression may have either sign because anatomical patch
    frames are geometry-fixed.  Normal load is therefore the summed magnitude
    of signed local-Z force on penetrating taxels; signed XY components retain
    their declared patch-local directions.
    """

    if penetration_m.shape != signed_normal_force_n.shape:
        raise ValueError(
            "penetration and normal tensors must have identical shapes, got "
            f"{tuple(penetration_m.shape)} and {tuple(signed_normal_force_n.shape)}"
        )
    if signed_shear_xy_n.shape != (*penetration_m.shape, 2):
        raise ValueError(
            "shear tensor must append a signed XY axis to the taxel shape, got "
            f"{tuple(signed_shear_xy_n.shape)}"
        )
    if penetration_m.ndim < 1 or penetration_m.shape[-1] < 1:
        raise ValueError("at least one taxel is required")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    penetration = torch.nan_to_num(penetration_m)
    normal = torch.nan_to_num(signed_normal_force_n)
    shear = torch.nan_to_num(signed_shear_xy_n)
    active = penetration > 0.0
    active_float = active.to(normal.dtype)

    contact = active.any(dim=-1).to(normal.dtype)
    normal_load = (normal.abs() * active_float).sum(dim=-1)
    signed_shear = (shear * active_float.unsqueeze(-1)).sum(dim=-2)

    area = torch.as_tensor(
        patch_area_m2,
        dtype=normal.dtype,
        device=normal.device,
    )
    mu = torch.as_tensor(
        friction_coefficient,
        dtype=normal.dtype,
        device=normal.device,
    )
    if torch.any(~torch.isfinite(area)) or torch.any(area <= 0.0):
        raise ValueError("patch area must be positive and finite")
    if torch.any(~torch.isfinite(mu)) or torch.any(mu <= 0.0):
        raise ValueError("friction coefficient must be positive and finite")

    mean_pressure = normal_load / area
    friction_utilization = torch.linalg.vector_norm(signed_shear, dim=-1) / (
        mu * normal_load + float(epsilon)
    )
    friction_utilization = torch.where(
        contact.bool(),
        friction_utilization,
        torch.zeros_like(friction_utilization),
    )
    return torch.stack(
        (
            contact,
            normal_load,
            mean_pressure,
            signed_shear[..., 0],
            signed_shear[..., 1],
            friction_utilization,
        ),
        dim=-1,
    )


def _sensor_data(env, sensor_name: str):
    if hasattr(env.scene, "sensors") and sensor_name in env.scene.sensors:
        return env.scene.sensors[sensor_name]
    return getattr(env.scene, sensor_name)


def current_whole_hand_patch_features(
    env,
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]] = (
        SENSOR_NAMES_BY_HAND
    ),
    patch_areas_m2: tuple[float, ...] = PATCH_AREAS_M2,
    friction_coefficient: float = 0.5,
) -> torch.Tensor:
    """Return live physical features with shape ``[B,2,27,6]``."""

    sensor_names_by_hand = tuple(tuple(names) for names in sensor_names_by_hand)
    if len(sensor_names_by_hand) != 2 or any(
        len(names) != 27 for names in sensor_names_by_hand
    ):
        raise ValueError("expected two hands with 27 sensor names each")
    if len(set(sensor_names_by_hand[0] + sensor_names_by_hand[1])) != 54:
        raise ValueError("all 54 sensor names must be unique")
    if len(patch_areas_m2) != 27:
        raise ValueError("expected 27 physical patch areas")

    hands = []
    for names in sensor_names_by_hand:
        patches = []
        for patch_index, sensor_name in enumerate(names):
            sensor = _sensor_data(env, sensor_name)
            data = sensor.data
            penetration = data.penetration_depth
            normal = data.tactile_normal_force
            shear = data.tactile_shear_force
            if penetration is None or normal is None or shear is None:
                raise RuntimeError(f"official TacSL data unavailable for {sensor_name}")
            if penetration.shape[0] != env.num_envs:
                raise RuntimeError(
                    f"{sensor_name} batch {penetration.shape[0]} != env batch {env.num_envs}"
                )
            sensor_mu = float(
                getattr(getattr(sensor, "cfg", None), "friction_coefficient", friction_coefficient)
            )
            patches.append(
                reduce_patch_taxels(
                    penetration,
                    normal,
                    shear,
                    patch_areas_m2[patch_index],
                    sensor_mu,
                )
            )
        hands.append(torch.stack(patches, dim=1))
    output = torch.stack(hands, dim=1)
    expected = (env.num_envs, 2, 27, len(BASE_PATCH_CHANNELS))
    if output.shape != expected or not torch.isfinite(output).all():
        raise RuntimeError(
            f"online whole-hand patch features are invalid: {tuple(output.shape)}"
        )
    return output


def _patch_history(
    env,
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]],
    patch_areas_m2: tuple[float, ...],
    friction_coefficient: float,
    history_steps: int,
) -> torch.Tensor:
    if history_steps != PATCH_HISTORY_STEPS:
        raise ValueError(f"patch history must have {PATCH_HISTORY_STEPS} steps")
    key = (
        tuple(tuple(names) for names in sensor_names_by_hand),
        tuple(float(value) for value in patch_areas_m2),
        float(friction_coefficient),
        int(history_steps),
    )
    cache = getattr(env, "_online_patch_tactile_history_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_online_patch_tactile_history_cache", cache)
    step = int(env.common_step_counter)
    entry = cache.get(key)
    if entry is None or int(entry["step"]) != step:
        current = current_whole_hand_patch_features(
            env,
            sensor_names_by_hand=sensor_names_by_hand,
            patch_areas_m2=patch_areas_m2,
            friction_coefficient=friction_coefficient,
        )
        if entry is None:
            history = current[:, None].repeat(1, history_steps, 1, 1, 1)
            entry = {"step": step, "history": history}
            cache[key] = entry
        else:
            history = entry["history"]
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = current
            entry["step"] = step
        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            history[reset_mask] = current[reset_mask, None].expand(
                -1, history_steps, -1, -1, -1
            )
        entry["history"] = history
    return entry["history"]


def online_patch_tactile_actor_history(
    env,
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]] = (
        SENSOR_NAMES_BY_HAND
    ),
    patch_areas_m2: tuple[float, ...] = PATCH_AREAS_M2,
    friction_coefficient: float = 0.5,
    history_steps: int = PATCH_HISTORY_STEPS,
) -> torch.Tensor:
    """Return ``[history,hand,patch,9]`` with zero slip fields for branch P."""

    history = _patch_history(
        env,
        sensor_names_by_hand,
        patch_areas_m2,
        friction_coefficient,
        history_steps,
    )
    slip_zeros = torch.zeros(
        (*history.shape[:-1], len(SLIP_PATCH_CHANNELS)),
        dtype=history.dtype,
        device=history.device,
    )
    actor = torch.cat((history, slip_zeros), dim=-1)
    return actor.reshape(env.num_envs, -1)


def _online_patch_slip_history(
    env,
    base_history: torch.Tensor,
) -> torch.Tensor:
    """Return causal slip history ``[B,4,2,27,3]`` once per control step."""

    from sugar_rl.tasks.locomanip.patch_slip import PatchSlipDetector

    step = int(env.common_step_counter)
    cache = getattr(env, "_online_patch_slip_history_cache", None)
    if cache is not None and int(cache["step"]) == step:
        return cache["history"]
    detector = getattr(env, "_online_patch_slip_detector", None)
    if detector is None:
        detector = PatchSlipDetector(env.num_envs, device=env.device)
        setattr(env, "_online_patch_slip_detector", detector)

    current = base_history[:, -1]
    timestamp_s = torch.full(
        (env.num_envs,),
        step * float(env.step_dt),
        dtype=torch.float32,
        device=env.device,
    )
    output = detector.update(
        contact=current[..., 0].bool(),
        normal_load_n=current[..., 1],
        mean_pressure_pa=current[..., 2],
        shear_xy_n=current[..., 3:5],
        friction_utilization=current[..., 5],
        timestamp_s=timestamp_s,
        reset_mask=env.episode_length_buf == 0,
    )
    current_slip = output.features()
    if cache is None:
        history = current_slip[:, None].repeat(
            1, PATCH_HISTORY_STEPS, 1, 1, 1
        )
        cache = {"step": step, "history": history}
        setattr(env, "_online_patch_slip_history_cache", cache)
    else:
        history = cache["history"]
        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = current_slip
        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            history[reset_mask] = current_slip[reset_mask, None].expand(
                -1, PATCH_HISTORY_STEPS, -1, -1, -1
            )
        cache["step"] = step
        cache["history"] = history
    env._online_patch_slip_diagnostics = {
        "state": output.state,
        "slip_score": output.slip_score,
        "incipient_slip": output.incipient_slip,
        "gross_slip": output.gross_slip,
        "timestamp_s": timestamp_s,
    }
    return cache["history"]


def online_patch_tactile_with_slip_actor_history(
    env,
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]] = (
        SENSOR_NAMES_BY_HAND
    ),
    patch_areas_m2: tuple[float, ...] = PATCH_AREAS_M2,
    friction_coefficient: float = 0.5,
    history_steps: int = PATCH_HISTORY_STEPS,
) -> torch.Tensor:
    """Return live base patch fields plus causal slip for branch PS."""

    base_history = _patch_history(
        env,
        sensor_names_by_hand,
        patch_areas_m2,
        friction_coefficient,
        history_steps,
    )
    slip_history = _online_patch_slip_history(env, base_history)
    actor = torch.cat((base_history, slip_history), dim=-1)
    return actor.reshape(env.num_envs, -1)


def exact_zero_online_patch_tactile_actor_history(
    env,
    sensor_names_by_hand: tuple[tuple[str, ...], tuple[str, ...]] = (
        SENSOR_NAMES_BY_HAND
    ),
    patch_areas_m2: tuple[float, ...] = PATCH_AREAS_M2,
    friction_coefficient: float = 0.5,
    history_steps: int = PATCH_HISTORY_STEPS,
) -> torch.Tensor:
    """Return matched exact zeros without touching ``env.scene.sensors``."""

    if len(sensor_names_by_hand) != 2 or any(
        len(names) != 27 for names in sensor_names_by_hand
    ):
        raise ValueError("expected two hands with 27 declared patch slots")
    if len(patch_areas_m2) != 27:
        raise ValueError("expected 27 physical patch areas")
    if friction_coefficient <= 0.0 or history_steps != PATCH_HISTORY_STEPS:
        raise ValueError("invalid exact-zero patch contract")
    width = history_steps * 2 * 27 * PATCH_FEATURE_WIDTH
    return torch.zeros((env.num_envs, width), dtype=torch.float32, device=env.device)
