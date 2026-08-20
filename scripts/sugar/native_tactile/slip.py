#!/usr/bin/env python3
"""Causal tactile-history-only slip evidence and hysteretic state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Sequence

import numpy as np

from scripts.sugar.native_tactile.universal import UniversalTactileFrame


class SlipState(IntEnum):
    """Ordinal tactile-only contact state."""

    NO_CONTACT = 0
    STICK = 1
    INCIPIENT = 2
    GROSS = 3


@dataclass(frozen=True)
class SlipDetectorConfig:
    """Physical thresholds for the deterministic slip detector."""

    minimum_normal_load_n: float = 1.0e-5
    incipient_friction_utilization: float = 0.65
    gross_friction_utilization: float = 0.9
    incipient_cop_speed_m_s: float = 0.005
    gross_cop_speed_m_s: float = 0.02
    incipient_footprint_rate_s: float = 2.0
    gross_footprint_rate_s: float = 8.0
    gross_normal_loss_rate_s: float = 5.0
    enter_frames: int = 2
    exit_frames: int = 3


@dataclass(frozen=True)
class SlipEvidence:
    """Continuous causal evidence and current state for every batch/patch."""

    normal_load_n: np.ndarray
    tangential_load_n: np.ndarray
    friction_utilization: np.ndarray
    center_of_pressure_xy_m: np.ndarray
    center_of_pressure_speed_m_s: np.ndarray
    footprint_change_rate_s: np.ndarray
    normal_loss_rate_s: np.ndarray
    contact_age_s: np.ndarray
    state: np.ndarray


def _numpy(value: Any) -> np.ndarray:
    if type(value).__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class TactileSlipDetector:
    """Detect slip from current/past tactile fields and source timestamps only.

    Simulator contact velocity, object pose, labels, reward, and outcome are
    intentionally absent from the interface. The continuous evidence remains
    available even when the hysteretic ordinal state does not change.
    """

    def __init__(
        self,
        patch_names: Sequence[str],
        *,
        friction_coefficient: float | Sequence[float] | None = None,
        config: SlipDetectorConfig | None = None,
    ) -> None:
        self.patch_names = tuple(patch_names)
        self.config = config or SlipDetectorConfig()
        if friction_coefficient is None:
            self.friction_coefficient = None
        elif np.isscalar(friction_coefficient):
            self.friction_coefficient = np.full(len(self.patch_names), float(friction_coefficient), dtype=np.float64)
        else:
            self.friction_coefficient = np.asarray(friction_coefficient, dtype=np.float64)
            if self.friction_coefficient.shape != (len(self.patch_names),):
                raise ValueError("Friction coefficient must be scalar or one value per patch.")
        self.reset()

    def reset(self) -> None:
        self._previous_pressure: np.ndarray | None = None
        self._previous_cop: np.ndarray | None = None
        self._previous_normal: np.ndarray | None = None
        self._contact_age: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._pending: np.ndarray | None = None
        self._pending_count: np.ndarray | None = None
        self._last_sequence = -1

    def update(self, frame: UniversalTactileFrame) -> SlipEvidence:
        """Advance the detector by one common tactile frame."""
        if frame.patch_names != self.patch_names:
            raise ValueError("Tactile frame patch order does not match the detector.")
        if frame.clock.sequence <= self._last_sequence:
            raise ValueError("Slip detector requires strictly increasing tactile sequence numbers.")

        normal_field = np.abs(_numpy(frame.normal_force_n).astype(np.float64, copy=False))
        shear_field = _numpy(frame.shear_force_xy_n).astype(np.float64, copy=False)
        active = _numpy(frame.active).astype(bool, copy=False)
        batch, patch_count, rows, columns = normal_field.shape
        normal_load = normal_field.sum(axis=(2, 3))
        tangential_load = np.linalg.norm(shear_field, axis=-1).sum(axis=(2, 3))
        contact = active.any(axis=(2, 3)) & (normal_load >= self.config.minimum_normal_load_n)

        if self.friction_coefficient is None:
            friction_utilization = np.full_like(normal_load, np.nan)
        else:
            denominator = np.maximum(
                normal_load * self.friction_coefficient[None, :],
                self.config.minimum_normal_load_n,
            )
            friction_utilization = tangential_load / denominator

        row = np.linspace(-0.5, 0.5, rows, dtype=np.float64)
        column = np.linspace(-0.5, 0.5, columns, dtype=np.float64)
        grid_column, grid_row = np.meshgrid(column, row)
        size = np.asarray(frame.patch_size_m, dtype=np.float64)
        coordinates = np.empty((patch_count, rows, columns, 2), dtype=np.float64)
        coordinates[..., 0] = grid_row[None] * size[:, None, None, 0]
        coordinates[..., 1] = grid_column[None] * size[:, None, None, 1]

        pressure_sum = np.maximum(normal_load[..., None, None], self.config.minimum_normal_load_n)
        normalized_pressure = normal_field / pressure_sum
        normalized_pressure[~contact] = 0.0
        cop = np.einsum("bphw,phwc->bpc", normalized_pressure, coordinates, optimize=True)

        shape = (batch, patch_count)
        if self._state is None or self._state.shape != shape:
            self._state = np.full(shape, SlipState.NO_CONTACT, dtype=np.int8)
            self._pending = self._state.copy()
            self._pending_count = np.zeros(shape, dtype=np.int32)
            self._contact_age = np.zeros(shape, dtype=np.float64)
            self._previous_pressure = normalized_pressure.copy()
            self._previous_cop = cop.copy()
            self._previous_normal = normal_load.copy()
            cop_speed = np.zeros(shape, dtype=np.float64)
            footprint_rate = np.zeros(shape, dtype=np.float64)
            normal_loss_rate = np.zeros(shape, dtype=np.float64)
        else:
            dt = float(frame.clock.dt_s)
            if dt <= 0.0:
                raise ValueError("Slip detector requires positive dt after the first frame.")
            previous_contact = self._previous_normal >= self.config.minimum_normal_load_n
            persistent = contact & previous_contact
            cop_speed = np.linalg.norm(cop - self._previous_cop, axis=-1) / dt
            cop_speed[~persistent] = 0.0
            footprint_rate = 0.5 * np.abs(normalized_pressure - self._previous_pressure).sum(axis=(2, 3)) / dt
            footprint_rate[~persistent] = 0.0
            normal_loss_rate = np.maximum(
                0.0,
                self._previous_normal - normal_load,
            ) / np.maximum(self._previous_normal, self.config.minimum_normal_load_n) / dt
            normal_loss_rate[~persistent] = 0.0

        dt_age = max(float(frame.clock.dt_s), 0.0)
        self._contact_age = np.where(contact, self._contact_age + dt_age, 0.0)

        utilization_incipient = np.nan_to_num(friction_utilization, nan=-np.inf) >= (
            self.config.incipient_friction_utilization
        )
        utilization_gross = np.nan_to_num(friction_utilization, nan=-np.inf) >= (
            self.config.gross_friction_utilization
        )
        temporal_incipient = (cop_speed >= self.config.incipient_cop_speed_m_s) | (
            footprint_rate >= self.config.incipient_footprint_rate_s
        )
        temporal_gross = (
            (cop_speed >= self.config.gross_cop_speed_m_s)
            | (footprint_rate >= self.config.gross_footprint_rate_s)
            | (normal_loss_rate >= self.config.gross_normal_loss_rate_s)
        )

        # A high static shear/normal ratio is a useful risk measure, but on a
        # curved gel it can also be produced by a stationary normal load whose
        # direction differs from the local taxel normal.  It therefore cannot
        # by itself prove slip.  State entry requires a causal redistribution
        # of the tactile field; friction utilization remains reported as
        # continuous evidence and gates the stronger ordinal transitions.
        candidate = np.full(shape, SlipState.STICK, dtype=np.int8)
        candidate[temporal_incipient & utilization_incipient] = SlipState.INCIPIENT
        candidate[temporal_gross & (utilization_gross | temporal_incipient)] = SlipState.GROSS
        candidate[~contact] = SlipState.NO_CONTACT

        for batch_index in range(batch):
            for patch_index in range(patch_count):
                if not contact[batch_index, patch_index]:
                    self._state[batch_index, patch_index] = SlipState.NO_CONTACT
                    self._pending[batch_index, patch_index] = SlipState.NO_CONTACT
                    self._pending_count[batch_index, patch_index] = 0
                    continue
                if self._state[batch_index, patch_index] == SlipState.NO_CONTACT:
                    self._state[batch_index, patch_index] = SlipState.STICK
                    self._pending[batch_index, patch_index] = SlipState.STICK
                    self._pending_count[batch_index, patch_index] = 0
                    continue
                target = candidate[batch_index, patch_index]
                current = self._state[batch_index, patch_index]
                if target == current:
                    self._pending[batch_index, patch_index] = target
                    self._pending_count[batch_index, patch_index] = 0
                    continue
                if self._pending[batch_index, patch_index] != target:
                    self._pending[batch_index, patch_index] = target
                    self._pending_count[batch_index, patch_index] = 1
                else:
                    self._pending_count[batch_index, patch_index] += 1
                required = self.config.enter_frames if target > current else self.config.exit_frames
                if self._pending_count[batch_index, patch_index] >= required:
                    self._state[batch_index, patch_index] = target
                    self._pending_count[batch_index, patch_index] = 0

        evidence = SlipEvidence(
            normal_load_n=normal_load,
            tangential_load_n=tangential_load,
            friction_utilization=friction_utilization,
            center_of_pressure_xy_m=cop,
            center_of_pressure_speed_m_s=cop_speed,
            footprint_change_rate_s=footprint_rate,
            normal_loss_rate_s=normal_loss_rate,
            contact_age_s=self._contact_age.copy(),
            state=self._state.copy(),
        )
        self._previous_pressure = normalized_pressure.copy()
        self._previous_cop = cop.copy()
        self._previous_normal = normal_load.copy()
        self._last_sequence = frame.clock.sequence
        return evidence
