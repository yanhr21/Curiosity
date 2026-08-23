# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Causal ordinal slip severity from direct TacSL spatial-temporal fields.

The estimator consumes only pressure/shear history.  Oracle labels, body/object
state, contact reports, material parameters, rewards, and outcomes are absent
from its runtime API.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


SPATIOTEMPORAL_FEATURE_NAMES = (
    "log_normal_load",
    "log_shear_magnitude_load",
    "integrated_utilization",
    "signed_shear_coherence",
    "active_fraction",
    "local_utilization_q50",
    "local_utilization_q75",
    "local_utilization_q90",
    "local_utilization_q95",
    "local_utilization_ge_050",
    "local_utilization_ge_075",
    "local_utilization_ge_090",
    "local_utilization_ge_098",
    "pressure_centroid_speed",
    "active_footprint_overlap",
    "pressure_shape_l1_change",
    "pressure_shape_cosine",
    "shear_shape_l1_change",
    "shear_shape_cosine",
    "normal_log_ratio",
    "utilization_delta",
    "contact_age_steps",
)


@dataclass(frozen=True)
class SpatiotemporalDirectTactileSlipCalibration:
    active_pressure_threshold_scaled: float
    minimum_normal_load_n: float
    nominal_sensor_friction_coefficient: float
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    slip_weight: tuple[float, ...]
    slip_bias: float
    slip_threshold: float
    gross_weight: tuple[float, ...]
    gross_bias: float
    gross_threshold: float

    def __post_init__(self) -> None:
        if self.active_pressure_threshold_scaled <= 0.0:
            raise ValueError("active pressure threshold must be positive")
        if self.minimum_normal_load_n <= 0.0:
            raise ValueError("minimum normal load must be positive")
        if self.nominal_sensor_friction_coefficient <= 0.0:
            raise ValueError("sensor friction coefficient must be positive")
        expected = len(SPATIOTEMPORAL_FEATURE_NAMES)
        for name in ("feature_mean", "feature_std", "slip_weight", "gross_weight"):
            values = getattr(self, name)
            if len(values) != expected:
                raise ValueError(f"{name} must contain {expected} entries")
            if not all(torch.isfinite(torch.tensor(values)).tolist()):
                raise ValueError(f"{name} contains non-finite values")
        if not all(value > 0.0 for value in self.feature_std):
            raise ValueError("feature standard deviations must be positive")
        for name in ("slip_threshold", "gross_threshold"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if not torch.isfinite(torch.tensor((self.slip_bias, self.gross_bias))).all():
            raise ValueError("classifier biases must be finite")

    @classmethod
    def from_artifact(
        cls, path: str | Path
    ) -> tuple["SpatiotemporalDirectTactileSlipCalibration", str]:
        resolved = Path(path).expanduser().resolve()
        raw = resolved.read_bytes()
        payload = json.loads(raw)
        if payload.get("protocol") != "sugar_direct_tacsl_spatiotemporal_v14_development_admission":
            raise ValueError(f"unexpected v14 calibration protocol: {resolved}")
        if payload.get("passed") is not True or payload.get("stage_e_development_admitted") is not True:
            raise ValueError(f"v14 development calibration was not admitted: {resolved}")
        values = payload.get("parameters", {})
        expected = {field.name for field in fields(cls)}
        if set(values) != expected:
            raise ValueError(
                f"v14 parameter mismatch: missing={sorted(expected - set(values))}, "
                f"unexpected={sorted(set(values) - expected)}"
            )
        for name in ("feature_mean", "feature_std", "slip_weight", "gross_weight"):
            values[name] = tuple(float(value) for value in values[name])
        return cls(**values), hashlib.sha256(raw).hexdigest()


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0e-12)


@torch.no_grad()
def direct_tactile_spatiotemporal_features(
    tactile_history: torch.Tensor,
    contact_age: torch.Tensor,
    active_pressure_threshold_scaled: float,
    minimum_normal_load_n: float,
    nominal_sensor_friction_coefficient: float,
    taxel_area_m2: float,
    stress_scale: float,
    control_dt: float,
) -> dict[str, torch.Tensor]:
    """Return per-palm current features from the last two tactile frames."""

    if tactile_history.ndim != 6 or tactile_history.shape[2:4] != (2, 3):
        raise ValueError(
            "v14 direct tactile history must be (env,history,2,3,row,col), got "
            f"{tuple(tactile_history.shape)}"
        )
    if tactile_history.shape[1] < 2 or tactile_history.shape[-2:] != (20, 25):
        raise ValueError("v14 requires at least two 20x25 direct R15 frames")
    if contact_age.shape != (tactile_history.shape[0], 2):
        raise ValueError("v14 contact-age state has the wrong shape")
    if taxel_area_m2 <= 0.0 or stress_scale <= 0.0 or control_dt <= 0.0:
        raise ValueError("taxel area, stress scale, and control dt must be positive")
    if not torch.isfinite(tactile_history).all():
        raise ValueError("v14 received non-finite direct TacSL fields")

    pressure = tactile_history[:, -2:, :, 0].clamp_min(0.0)
    shear = tactile_history[:, -2:, :, 1:3]
    active = pressure >= active_pressure_threshold_scaled
    active_count = active.sum(dim=(-2, -1))
    force_factor = taxel_area_m2 / stress_scale
    normal = pressure.sum(dim=(-2, -1)) * force_factor
    shear_norm = torch.linalg.vector_norm(shear, dim=3)
    shear_magnitude = (shear_norm * active).sum(dim=(-2, -1)) * force_factor
    utilization = shear_magnitude / (
        nominal_sensor_friction_coefficient * normal + 1.0e-8
    )
    signed_shear = (shear * active.unsqueeze(3)).sum(dim=(-2, -1))
    coherence = _safe_div(
        torch.linalg.vector_norm(signed_shear, dim=-1),
        (shear_norm * active).sum(dim=(-2, -1)),
    )
    local_utilization = shear_norm / (
        nominal_sensor_friction_coefficient * pressure + 1.0e-8
    )
    latest_local = torch.where(
        active[:, -1],
        local_utilization[:, -1],
        torch.full_like(local_utilization[:, -1], torch.nan),
    )
    quantiles = torch.nanquantile(
        latest_local.flatten(-2),
        torch.tensor(
            (0.50, 0.75, 0.90, 0.95),
            device=tactile_history.device,
            dtype=tactile_history.dtype,
        ),
        dim=-1,
    ).permute(1, 2, 0)
    latest_active = active[:, -1]
    latest_active_count = active_count[:, -1]
    fractions = torch.stack(
        tuple(
            _safe_div(
                ((local_utilization[:, -1] >= threshold) & latest_active).sum(
                    dim=(-2, -1)
                ),
                latest_active_count,
            )
            for threshold in (0.50, 0.75, 0.90, 0.98)
        ),
        dim=-1,
    )

    rows, cols = pressure.shape[-2:]
    row_grid = torch.arange(
        rows, device=pressure.device, dtype=pressure.dtype
    ).view(1, 1, 1, rows, 1)
    col_grid = torch.arange(
        cols, device=pressure.device, dtype=pressure.dtype
    ).view(1, 1, 1, 1, cols)
    pressure_sum = pressure.sum(dim=(-2, -1)).clamp_min(1.0e-12)
    centroid = torch.stack(
        (
            (pressure * row_grid).sum(dim=(-2, -1)) / pressure_sum,
            (pressure * col_grid).sum(dim=(-2, -1)) / pressure_sum,
        ),
        dim=-1,
    )
    centroid_speed = torch.linalg.vector_norm(
        centroid[:, -1] - centroid[:, -2], dim=-1
    ) / control_dt
    intersection = (active[:, -1] & active[:, -2]).sum(dim=(-2, -1))
    union = (active[:, -1] | active[:, -2]).sum(dim=(-2, -1))
    overlap = _safe_div(intersection, union)

    pressure_shape = pressure / pressure_sum.unsqueeze(-1).unsqueeze(-1)
    pressure_l1 = torch.abs(pressure_shape[:, -1] - pressure_shape[:, -2]).sum(
        dim=(-2, -1)
    )
    pressure_cos = F.cosine_similarity(
        pressure_shape[:, -1].flatten(2), pressure_shape[:, -2].flatten(2), dim=-1
    )
    shear_flat = (shear * active.unsqueeze(3)).flatten(3)
    shear_scale = torch.linalg.vector_norm(shear_flat, dim=-1).clamp_min(1.0e-12)
    shear_shape = shear_flat / shear_scale.unsqueeze(-1)
    shear_l1 = torch.abs(shear_shape[:, -1] - shear_shape[:, -2]).sum(dim=-1)
    shear_cos = F.cosine_similarity(
        shear_shape[:, -1], shear_shape[:, -2], dim=-1
    )
    normal_log_ratio = torch.log(
        (normal[:, -1] + 1.0e-8) / (normal[:, -2] + 1.0e-8)
    )
    utilization_delta = utilization[:, -1] - utilization[:, -2]
    contact = (latest_active_count >= 1) & (normal[:, -1] >= minimum_normal_load_n)

    features = torch.stack(
        (
            torch.log(normal[:, -1] + 1.0e-8),
            torch.log(shear_magnitude[:, -1] + 1.0e-8),
            utilization[:, -1],
            coherence[:, -1],
            latest_active_count.float() / float(rows * cols),
            quantiles[..., 0],
            quantiles[..., 1],
            quantiles[..., 2],
            quantiles[..., 3],
            fractions[..., 0],
            fractions[..., 1],
            fractions[..., 2],
            fractions[..., 3],
            centroid_speed,
            overlap,
            pressure_l1,
            pressure_cos,
            shear_l1,
            shear_cos,
            normal_log_ratio,
            utilization_delta,
            contact_age.float(),
        ),
        dim=-1,
    )
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    expected = (tactile_history.shape[0], 2, len(SPATIOTEMPORAL_FEATURE_NAMES))
    if features.shape != expected:
        raise RuntimeError(f"unexpected v14 feature shape: {features.shape}")
    return {
        "features": features,
        "contact": contact,
        "normal_load_n": normal[:, -1],
        "active_count": latest_active_count,
    }


class SpatiotemporalDirectTactileSlipEstimator:
    NO_CONTACT = 0
    STICK = 1
    INCIPIENT = 2
    GROSS = 3

    def __init__(
        self,
        calibration: SpatiotemporalDirectTactileSlipCalibration,
        num_envs: int,
        device: torch.device | str,
        taxel_area_m2: float,
        stress_scale: float,
        control_dt: float,
    ) -> None:
        self.calibration = calibration
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.taxel_area_m2 = float(taxel_area_m2)
        self.stress_scale = float(stress_scale)
        self.control_dt = float(control_dt)
        self.state = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.contact_age = torch.zeros_like(self.state)
        self.previous_contact = torch.zeros_like(self.state, dtype=torch.bool)
        self.feature_mean = torch.tensor(
            calibration.feature_mean, dtype=torch.float32, device=self.device
        )
        self.feature_std = torch.tensor(
            calibration.feature_std, dtype=torch.float32, device=self.device
        )
        self.slip_weight = torch.tensor(
            calibration.slip_weight, dtype=torch.float32, device=self.device
        )
        self.gross_weight = torch.tensor(
            calibration.gross_weight, dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def update(
        self, tactile_history: torch.Tensor, reset_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if tactile_history.shape[0] != self.num_envs:
            raise ValueError("v14 environment count changed")
        if reset_mask is not None:
            if reset_mask.shape != (self.num_envs,) or reset_mask.dtype != torch.bool:
                raise ValueError("v14 reset mask must be one-dimensional bool")
            self.state[reset_mask] = self.NO_CONTACT
            self.contact_age[reset_mask] = 0
            self.previous_contact[reset_mask] = False

        provisional = direct_tactile_spatiotemporal_features(
            tactile_history,
            self.contact_age,
            self.calibration.active_pressure_threshold_scaled,
            self.calibration.minimum_normal_load_n,
            self.calibration.nominal_sensor_friction_coefficient,
            self.taxel_area_m2,
            self.stress_scale,
            self.control_dt,
        )
        contact = provisional["contact"]
        next_age = torch.where(contact, self.contact_age + 1, torch.zeros_like(self.contact_age))
        metrics = direct_tactile_spatiotemporal_features(
            tactile_history,
            next_age,
            self.calibration.active_pressure_threshold_scaled,
            self.calibration.minimum_normal_load_n,
            self.calibration.nominal_sensor_friction_coefficient,
            self.taxel_area_m2,
            self.stress_scale,
            self.control_dt,
        )
        features = metrics["features"]
        normalized = (features - self.feature_mean) / self.feature_std
        slip_probability = torch.sigmoid(
            normalized @ self.slip_weight + self.calibration.slip_bias
        )
        gross_probability = torch.sigmoid(
            normalized @ self.gross_weight + self.calibration.gross_bias
        )
        temporal_ready = next_age >= 2
        slip = contact & temporal_ready & (
            slip_probability >= self.calibration.slip_threshold
        )
        gross = slip & (gross_probability >= self.calibration.gross_threshold)
        previous_state = self.state.clone()
        next_state = torch.full_like(self.state, self.NO_CONTACT)
        next_state = torch.where(contact, torch.full_like(next_state, self.STICK), next_state)
        next_state = torch.where(slip, torch.full_like(next_state, self.INCIPIENT), next_state)
        next_state = torch.where(gross, torch.full_like(next_state, self.GROSS), next_state)
        contact_transition = contact ^ self.previous_contact
        if reset_mask is not None:
            contact_transition[reset_mask] = False
        self.state = next_state
        self.contact_age = next_age
        self.previous_contact = contact
        return {
            "features": features,
            "normalized_features": normalized,
            "slip_probability": slip_probability,
            "gross_probability": gross_probability,
            "state": self.state.clone(),
            "incipient": self.state >= self.INCIPIENT,
            "gross": self.state == self.GROSS,
            "contact": contact,
            "contact_age": self.contact_age.clone(),
            "contact_transition": contact_transition,
            "slip_event_started": (previous_state < self.INCIPIENT)
            & (self.state >= self.INCIPIENT),
            "slip_event_ended": (previous_state >= self.INCIPIENT)
            & (self.state < self.INCIPIENT),
        }
