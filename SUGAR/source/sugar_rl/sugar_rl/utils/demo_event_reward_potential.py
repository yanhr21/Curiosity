# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Uncertainty-aware internal feedback for the frozen contact/event predictor."""

from __future__ import annotations

from collections.abc import Sequence

import torch


EVENT_TARGET_COUNT = 13
DEFAULT_EVENT_WEIGHTS = (
    0.0625,
    0.0625,
    0.0625,
    0.0625,
    0.075,
    0.075,
    0.075,
    0.075,
    0.05,
    0.05,
    0.05,
    0.05,
    0.25,
)


def calibrated_event_risk(
    mean_log1p_scaled: torch.Tensor,
    log_variance_log1p_scaled: torch.Tensor,
    variance_multiplier: torch.Tensor,
    *,
    uncertainty_beta: float,
    target_weights: Sequence[float] = DEFAULT_EVENT_WEIGHTS,
    per_target_risk_clip: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Return weighted mean mismatch plus calibrated one-sided uncertainty.

    The predictor operates on ``log1p(target / target_scale)``.  Decoding in
    normalized target units makes the physical target scales cancel.  The
    uncertainty term uses the first-order log-normal delta approximation.
    """

    expected = mean_log1p_scaled.shape
    if expected != log_variance_log1p_scaled.shape or expected[-1] != EVENT_TARGET_COUNT:
        raise ValueError("event predictor tensors must share a final dimension of 13")
    if variance_multiplier.shape != (EVENT_TARGET_COUNT,):
        raise ValueError("variance multiplier must have shape (13,)")
    if not torch.isfinite(mean_log1p_scaled).all() or not torch.isfinite(
        log_variance_log1p_scaled
    ).all():
        raise ValueError("event predictor output is non-finite")
    if torch.any(variance_multiplier <= 0) or not torch.isfinite(
        variance_multiplier
    ).all():
        raise ValueError("variance multipliers must be finite and positive")
    if not 0.0 <= float(uncertainty_beta) <= 3.0:
        raise ValueError("uncertainty_beta must lie in [0, 3]")
    if not 0.0 < float(per_target_risk_clip) <= 20.0:
        raise ValueError("per_target_risk_clip must lie in (0, 20]")
    weights = torch.as_tensor(
        target_weights,
        dtype=mean_log1p_scaled.dtype,
        device=mean_log1p_scaled.device,
    )
    if weights.shape != (EVENT_TARGET_COUNT,) or torch.any(weights < 0):
        raise ValueError("target weights must be 13 nonnegative values")
    if not torch.isclose(
        weights.sum(),
        torch.ones((), dtype=weights.dtype, device=weights.device),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("target weights must sum to one")

    nonnegative_mean = torch.clamp(mean_log1p_scaled, min=0.0, max=8.0)
    normalized_mean = torch.expm1(nonnegative_mean)
    calibrated_std_log = torch.sqrt(
        torch.exp(torch.clamp(log_variance_log1p_scaled, min=-10.0, max=8.0))
        * variance_multiplier
    )
    normalized_std = torch.exp(nonnegative_mean) * calibrated_std_log
    per_target_risk = torch.clamp(
        normalized_mean + float(uncertainty_beta) * normalized_std,
        min=0.0,
        max=float(per_target_risk_clip),
    )
    risk = torch.sum(per_target_risk * weights, dim=-1)
    uncertainty = torch.sum(normalized_std * weights, dim=-1)
    return {
        "risk": risk,
        "normalized_mean": normalized_mean,
        "normalized_std": normalized_std,
        "weighted_uncertainty": uncertainty,
        "per_target_risk": per_target_risk,
    }


def compatibility_potential(risk: torch.Tensor) -> torch.Tensor:
    """Map nonnegative mismatch risk to a bounded higher-is-better potential."""

    if torch.any(risk < 0) or not torch.isfinite(risk).all():
        raise ValueError("risk must be finite and nonnegative")
    return torch.exp(-risk)


def event_internal_reward(
    next_potential: torch.Tensor,
    next_ready: torch.Tensor,
    done: torch.Tensor,
    failure_done: torch.Tensor,
    *,
    compatibility_baseline: float,
    eta: float,
    reward_clip: float,
) -> torch.Tensor:
    """Return dense causal compatibility feedback for the next observed state.

    Unlike potential-difference shaping, this signal intentionally changes the
    policy objective: sustained compatibility earns sustained positive
    feedback and incompatibility earns negative feedback.  Physical failure is
    assigned the negative clip; benign terminals and incomplete prefixes are
    exactly zero.
    """

    expected = next_potential.shape
    for name, value in (
        ("next_ready", next_ready),
        ("done", done),
        ("failure_done", failure_done),
    ):
        if value.shape != expected or value.dtype is not torch.bool:
            raise ValueError(f"{name} must be bool with shape {expected}")
    if torch.any(failure_done & ~done):
        raise ValueError("failure_done must be a subset of done")
    if not 0.0 <= float(compatibility_baseline) <= 1.0:
        raise ValueError("compatibility_baseline must lie in [0, 1]")
    if not 0.0 <= float(eta) <= 100.0:
        raise ValueError("eta must lie in [0, 100]")
    if not 0.0 < float(reward_clip) <= 100.0:
        raise ValueError("reward_clip must lie in (0, 100]")

    dense = torch.clamp(
        float(eta) * (next_potential - float(compatibility_baseline)),
        -float(reward_clip),
        float(reward_clip),
    )
    return torch.where(
        failure_done,
        torch.full_like(dense, -float(reward_clip)),
        torch.where(next_ready & ~done, dense, torch.zeros_like(dense)),
    )
