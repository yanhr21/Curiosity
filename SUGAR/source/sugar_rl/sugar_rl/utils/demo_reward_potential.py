# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Policy-discount-matched shaping from a frozen demo-loss prediction.

The learned prediction is a potential/value estimate, not an instantaneous
observed reward.  A direct-TacSL failure closure masks the *state potential*;
the shaping reward is then the difference of consecutive masked potentials.
This placement is required for exact telescoping when imitation is disabled.

No ICM tensor, slip magnitude, task reward, success, privileged physics, or
future reference is accepted by this API.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


COMPONENT_NAMES = (
    "body",
    "box_position",
    "box_rotation_6d",
    "box_velocity",
)


def normalized_demo_mismatch(
    component_mse: torch.Tensor,
    target_scale: torch.Tensor,
    component_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> torch.Tensor:
    """Return the frozen nonnegative scalar mismatch estimate."""

    if component_mse.ndim < 1 or component_mse.shape[-1] != 4:
        raise ValueError("component_mse must end in four components")
    if target_scale.shape != (4,):
        raise ValueError("target_scale must have shape (4,)")
    if torch.any(component_mse < 0) or not torch.isfinite(
        component_mse
    ).all():
        raise ValueError("component mismatch must be finite and nonnegative")
    if torch.any(target_scale <= 0) or not torch.isfinite(
        target_scale
    ).all():
        raise ValueError("target scales must be finite and positive")
    weights = torch.as_tensor(
        component_weights,
        device=component_mse.device,
        dtype=component_mse.dtype,
    )
    if weights.shape != (4,) or torch.any(weights < 0):
        raise ValueError("component weights must be four nonnegative values")
    if not torch.isclose(
        weights.sum(),
        torch.ones((), device=weights.device, dtype=weights.dtype),
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise ValueError("component weights must sum exactly to one")
    return ((component_mse / target_scale) * weights).sum(dim=-1)


def demo_potential(
    component_mse: torch.Tensor,
    target_scale: torch.Tensor,
    component_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> torch.Tensor:
    """Higher potential means better predicted demo compatibility."""

    return -normalized_demo_mismatch(
        component_mse,
        target_scale,
        component_weights,
    )


def masked_demo_potential(
    component_mse: torch.Tensor,
    target_scale: torch.Tensor,
    imitation_active: torch.Tensor,
    *,
    eta: float = 1.0,
    component_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> torch.Tensor:
    """Return Psi(s)=eta*active(s)*Phi(s)."""

    if imitation_active.shape != component_mse.shape[:-1]:
        raise ValueError("imitation_active shape mismatch")
    if imitation_active.dtype is not torch.bool:
        raise TypeError("imitation_active must be bool")
    if not torch.isfinite(torch.tensor(float(eta))) or eta < 0.0:
        raise ValueError("eta must be finite and nonnegative")
    potential = demo_potential(
        component_mse,
        target_scale,
        component_weights,
    )
    return torch.where(
        imitation_active,
        potential * float(eta),
        torch.zeros_like(potential),
    )


def potential_difference_reward(
    current_component_mse: torch.Tensor,
    next_component_mse: torch.Tensor,
    target_scale: torch.Tensor,
    current_imitation_active: torch.Tensor,
    next_imitation_active: torch.Tensor,
    next_done: torch.Tensor,
    *,
    gamma: float,
    eta: float = 1.0,
    component_weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> dict[str, torch.Tensor]:
    """Return r=gamma*Psi(s')-Psi(s), with terminal Psi(s') fixed to zero."""

    if current_component_mse.shape != next_component_mse.shape:
        raise ValueError("current/next component shape mismatch")
    expected = current_component_mse.shape[:-1]
    for name, value in (
        ("current_imitation_active", current_imitation_active),
        ("next_imitation_active", next_imitation_active),
        ("next_done", next_done),
    ):
        if value.shape != expected or value.dtype is not torch.bool:
            raise ValueError(f"{name} must be bool with shape {expected}")
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")

    current_phi = demo_potential(
        current_component_mse,
        target_scale,
        component_weights,
    )
    next_phi = demo_potential(
        next_component_mse,
        target_scale,
        component_weights,
    )
    current_psi = torch.where(
        current_imitation_active,
        current_phi * float(eta),
        torch.zeros_like(current_phi),
    )
    next_active_nonterminal = next_imitation_active & ~next_done
    next_psi = torch.where(
        next_active_nonterminal,
        next_phi * float(eta),
        torch.zeros_like(next_phi),
    )
    reward = float(gamma) * next_psi - current_psi
    return {
        "reward": reward,
        "current_phi": current_phi,
        "next_phi": next_phi,
        "current_psi": current_psi,
        "next_psi": next_psi,
        "next_active_nonterminal": next_active_nonterminal,
    }
