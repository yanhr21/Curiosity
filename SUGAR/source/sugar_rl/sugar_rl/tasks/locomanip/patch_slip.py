# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Causal patch-level slip state for online IsaacLab observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NO_CONTACT = 0
STICK = 1
INCIPIENT = 2
GROSS = 3


@dataclass(frozen=True)
class PatchSlipOutput:
    state: torch.Tensor
    slip_score: torch.Tensor
    incipient_slip: torch.Tensor
    gross_slip: torch.Tensor

    def features(self) -> torch.Tensor:
        return torch.stack(
            (
                self.slip_score,
                self.incipient_slip.to(self.slip_score.dtype),
                self.gross_slip.to(self.slip_score.dtype),
            ),
            dim=-1,
        )


class PatchSlipDetector:
    """Stateful detector whose spatial unit is one physical hand patch.

    Thresholds are calibrated on a controlled official-R15 stick-to-slide
    trace.  The callable deliberately accepts no object state, contact-relative
    velocity, mass identifier, reward, or future sample.
    """

    def __init__(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        hands: int = 2,
        patches_per_hand: int = 27,
        incipient_friction_utilization: float = 0.60,
        gross_friction_utilization: float = 0.90,
        incipient_shear_rate_per_load_s: float = 0.5,
        gross_shear_rate_per_load_s: float = 3.0,
        incipient_pressure_drop_rate_s: float = 2.0,
        gross_pressure_drop_rate_s: float = 6.0,
        gross_evidence_steps: int = 2,
        contact_loss_min_load_n: float = 0.02,
        epsilon: float = 1.0e-8,
    ) -> None:
        if batch_size < 1 or hands != 2 or patches_per_hand != 27:
            raise ValueError("slip detector requires [batch,2,27] patch geometry")
        thresholds = (
            incipient_friction_utilization,
            gross_friction_utilization,
            incipient_shear_rate_per_load_s,
            gross_shear_rate_per_load_s,
            incipient_pressure_drop_rate_s,
            gross_pressure_drop_rate_s,
            contact_loss_min_load_n,
            epsilon,
        )
        if any(value <= 0.0 for value in thresholds):
            raise ValueError("all slip thresholds must be positive")
        if gross_friction_utilization <= incipient_friction_utilization:
            raise ValueError("gross friction threshold must exceed incipient")
        if gross_shear_rate_per_load_s <= incipient_shear_rate_per_load_s:
            raise ValueError("gross shear-rate threshold must exceed incipient")
        if gross_pressure_drop_rate_s <= incipient_pressure_drop_rate_s:
            raise ValueError("gross pressure-drop threshold must exceed incipient")
        if gross_evidence_steps < 2:
            raise ValueError("gross slip requires at least two causal evidence steps")

        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.shape = (self.batch_size, hands, patches_per_hand)
        self.incipient_friction_utilization = float(
            incipient_friction_utilization
        )
        self.gross_friction_utilization = float(gross_friction_utilization)
        self.incipient_shear_rate_per_load_s = float(
            incipient_shear_rate_per_load_s
        )
        self.gross_shear_rate_per_load_s = float(gross_shear_rate_per_load_s)
        self.incipient_pressure_drop_rate_s = float(
            incipient_pressure_drop_rate_s
        )
        self.gross_pressure_drop_rate_s = float(gross_pressure_drop_rate_s)
        self.gross_evidence_steps = int(gross_evidence_steps)
        self.contact_loss_min_load_n = float(contact_loss_min_load_n)
        self.epsilon = float(epsilon)

        self.initialized = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.previous_timestamp_s = torch.zeros(
            self.batch_size, dtype=torch.float32, device=self.device
        )
        self.previous_contact = torch.zeros(
            self.shape, dtype=torch.bool, device=self.device
        )
        self.previous_normal_load_n = torch.zeros(self.shape, device=self.device)
        self.previous_pressure_pa = torch.zeros(self.shape, device=self.device)
        self.previous_shear_xy_n = torch.zeros(
            (*self.shape, 2), device=self.device
        )
        self.state = torch.zeros(self.shape, dtype=torch.int64, device=self.device)
        self.gross_evidence_count = torch.zeros(
            self.shape, dtype=torch.int64, device=self.device
        )

    def reset(self, reset_mask: torch.Tensor) -> None:
        mask = torch.as_tensor(
            reset_mask, dtype=torch.bool, device=self.device
        ).reshape(self.batch_size)
        if not mask.any():
            return
        self.initialized[mask] = False
        self.previous_timestamp_s[mask] = 0.0
        self.previous_contact[mask] = False
        self.previous_normal_load_n[mask] = 0.0
        self.previous_pressure_pa[mask] = 0.0
        self.previous_shear_xy_n[mask] = 0.0
        self.state[mask] = NO_CONTACT
        self.gross_evidence_count[mask] = 0

    def _patch_tensor(
        self, value: torch.Tensor, name: str, *, dtype: torch.dtype
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype, device=self.device)
        if tuple(tensor.shape) != self.shape:
            raise ValueError(f"{name} must have shape {self.shape}, got {tuple(tensor.shape)}")
        if dtype.is_floating_point and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
        return tensor

    def update(
        self,
        contact: torch.Tensor,
        normal_load_n: torch.Tensor,
        mean_pressure_pa: torch.Tensor,
        shear_xy_n: torch.Tensor,
        friction_utilization: torch.Tensor,
        timestamp_s: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> PatchSlipOutput:
        """Update from only current and retained past patch tactile fields."""

        contact_now = self._patch_tensor(contact, "contact", dtype=torch.bool)
        normal_now = self._patch_tensor(
            normal_load_n, "normal_load_n", dtype=torch.float32
        )
        pressure_now = self._patch_tensor(
            mean_pressure_pa, "mean_pressure_pa", dtype=torch.float32
        )
        utilization_now = self._patch_tensor(
            friction_utilization,
            "friction_utilization",
            dtype=torch.float32,
        )
        shear_now = torch.as_tensor(
            shear_xy_n, dtype=torch.float32, device=self.device
        )
        if tuple(shear_now.shape) != (*self.shape, 2):
            raise ValueError(
                f"shear_xy_n must have shape {(*self.shape, 2)}, got {tuple(shear_now.shape)}"
            )
        if not torch.isfinite(shear_now).all():
            raise ValueError("shear_xy_n contains non-finite values")
        timestamp = torch.as_tensor(
            timestamp_s, dtype=torch.float32, device=self.device
        ).reshape(self.batch_size)
        if not torch.isfinite(timestamp).all():
            raise ValueError("timestamp_s contains non-finite values")

        self.reset(reset_mask)
        prior_initialized = self.initialized.clone()
        if prior_initialized.any():
            dt = timestamp - self.previous_timestamp_s
            if torch.any(dt[prior_initialized] <= 0.0):
                raise ValueError("timestamp_s must increase within each episode")
        dt = torch.where(
            prior_initialized,
            timestamp - self.previous_timestamp_s,
            torch.ones_like(timestamp),
        )[:, None, None]

        shear_delta = torch.linalg.vector_norm(
            shear_now - self.previous_shear_xy_n, dim=-1
        )
        reference_load = torch.maximum(normal_now, self.previous_normal_load_n)
        shear_rate = shear_delta / (dt * reference_load + self.epsilon)
        pressure_drop_rate = torch.clamp(
            (self.previous_pressure_pa - pressure_now)
            / (dt * self.previous_pressure_pa + self.epsilon),
            min=0.0,
        )
        temporal_valid = (
            prior_initialized[:, None, None]
            & self.previous_contact
            & contact_now
        )
        shear_rate = torch.where(
            temporal_valid, shear_rate, torch.zeros_like(shear_rate)
        )
        pressure_drop_rate = torch.where(
            temporal_valid,
            pressure_drop_rate,
            torch.zeros_like(pressure_drop_rate),
        )

        utilization_score = torch.clamp(
            utilization_now / self.gross_friction_utilization, min=0.0
        )
        shear_score = torch.clamp(
            shear_rate / self.gross_shear_rate_per_load_s, min=0.0
        )
        pressure_score = torch.clamp(
            pressure_drop_rate / self.gross_pressure_drop_rate_s, min=0.0
        )
        score = torch.maximum(
            utilization_score,
            torch.maximum(shear_score, pressure_score),
        )
        score = torch.where(contact_now, score, torch.zeros_like(score))
        contact_loss = (
            prior_initialized[:, None, None]
            & self.previous_contact
            & (~contact_now)
            & (self.previous_normal_load_n >= self.contact_loss_min_load_n)
        )
        score = torch.where(contact_loss, torch.ones_like(score), score).clamp_max_(2.0)

        incipient_evidence = (
            (utilization_now >= self.incipient_friction_utilization)
            | (shear_rate >= self.incipient_shear_rate_per_load_s)
            | (pressure_drop_rate >= self.incipient_pressure_drop_rate_s)
        ) & contact_now
        gross_candidate = (
            (
                (shear_rate >= self.gross_shear_rate_per_load_s)
                | (pressure_drop_rate >= self.gross_pressure_drop_rate_s)
            )
            & contact_now
        )
        self.gross_evidence_count = torch.where(
            gross_candidate,
            self.gross_evidence_count + 1,
            torch.zeros_like(self.gross_evidence_count),
        )
        newly_gross = self.gross_evidence_count >= self.gross_evidence_steps
        retained_gross = (self.state == GROSS) & incipient_evidence
        gross_evidence = newly_gross | retained_gross | contact_loss

        state = torch.full_like(self.state, NO_CONTACT)
        state[contact_now] = STICK
        state[incipient_evidence] = INCIPIENT
        state[gross_evidence] = GROSS
        self.state.copy_(state)

        self.previous_timestamp_s.copy_(timestamp)
        self.previous_contact.copy_(contact_now)
        self.previous_normal_load_n.copy_(normal_now)
        self.previous_pressure_pa.copy_(pressure_now)
        self.previous_shear_xy_n.copy_(shear_now)
        self.initialized[:] = True

        return PatchSlipOutput(
            state=state.clone(),
            slip_score=score,
            incipient_slip=state == INCIPIENT,
            gross_slip=state == GROSS,
        )
