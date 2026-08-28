"""Causal reference-phase retiming for a frozen official Refiner."""

from __future__ import annotations

import torch


def bounded_phase_offset(
    raw_action: torch.Tensor,
    physical_t: torch.Tensor,
    *,
    handoff_step: int = 200,
    endpoint_step: int = 235,
    maximum_offset: int = 7,
) -> torch.Tensor:
    """Map one causal action per world to an integer reference-frame offset.

    The polynomial envelope is exactly zero at both fixed endpoints.  It lets a
    controller redistribute reference timing inside the recovery window without
    changing physical progress or the endpoint reference.
    """

    if raw_action.ndim == 2 and raw_action.shape[1] == 1:
        raw_action = raw_action[:, 0]
    if raw_action.ndim != 1 or raw_action.shape != physical_t.shape:
        raise ValueError("raw phase action must have one scalar per physical clock")
    if endpoint_step <= handoff_step:
        raise ValueError("phase endpoint must follow the handoff")
    progress = (
        (physical_t.to(torch.float32) - float(handoff_step))
        / float(endpoint_step - handoff_step)
    ).clamp(0.0, 1.0)
    envelope = 4.0 * progress * (1.0 - progress)
    continuous = float(maximum_offset) * envelope * torch.tanh(raw_action)
    offset = torch.round(continuous).to(dtype=physical_t.dtype)
    at_endpoint = (physical_t <= handoff_step) | (physical_t >= endpoint_step)
    return torch.where(at_endpoint, torch.zeros_like(offset), offset)


def policy_reference_phase(
    raw_action: torch.Tensor,
    physical_t: torch.Tensor,
    *,
    handoff_step: int = 200,
    endpoint_step: int = 235,
    maximum_offset: int = 7,
) -> torch.Tensor:
    """Return the retimed reference index while leaving ``physical_t`` untouched."""

    return physical_t + bounded_phase_offset(
        raw_action,
        physical_t,
        handoff_step=handoff_step,
        endpoint_step=endpoint_step,
        maximum_offset=maximum_offset,
    )
