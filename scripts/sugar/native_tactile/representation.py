#!/usr/bin/env python3
"""Transparent representation of native IsaacLab visuo-tactile fields.

This module intentionally has no IsaacLab task, object-state, reward, contact
sensor, or outcome-label dependency. It consumes only fields exported by
``VisuoTactileSensorData`` and retains the raw signed channels unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TactileFeatureFrame:
    """Tactile-only summaries for one timestamp and all sensors."""

    total_normal_signed: np.ndarray
    total_normal_magnitude: np.ndarray
    total_shear_xy: np.ndarray
    total_shear_magnitude: np.ndarray
    active_taxel_count: np.ndarray
    contact_fraction: np.ndarray
    center_of_pressure_xy: np.ndarray
    pressure_covariance_xy: np.ndarray
    principal_direction_xy: np.ndarray
    spatial_spread: np.ndarray
    shear_to_normal_ratio: np.ndarray
    normal_change_l1: np.ndarray
    shear_change_l1: np.ndarray
    center_of_pressure_motion: np.ndarray
    contact_onset: np.ndarray
    contact_release: np.ndarray
    contact_persistence_frames: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return asdict(self)


class NativeTactileRepresentation:
    """Stateful causal feature extractor for a fixed native taxel grid.

    Args:
        sensor_count: Number of independent native sensors.
        grid_shape: Native taxel rows and columns.
        ratio_floor: Numerical denominator floor used only for the declared
            shear-to-normal magnitude ratio.

    The extractor never thresholds force magnitudes. Native contact activity
    follows the sensor's exact positive SDF penetration output. Consequently,
    exact inactive zeros stay visible and no task-specific contact threshold is
    introduced.
    """

    def __init__(
        self,
        sensor_count: int,
        grid_shape: tuple[int, int],
        ratio_floor: float = 1.0e-9,
    ) -> None:
        rows, columns = grid_shape
        if sensor_count <= 0 or rows <= 0 or columns <= 0:
            raise ValueError("sensor_count and grid dimensions must be positive")
        if not np.isfinite(ratio_floor) or ratio_floor <= 0.0:
            raise ValueError("ratio_floor must be finite and positive")
        self.sensor_count = int(sensor_count)
        self.grid_shape = (int(rows), int(columns))
        self.ratio_floor = float(ratio_floor)
        self._previous_normal: np.ndarray | None = None
        self._previous_shear: np.ndarray | None = None
        self._previous_cop: np.ndarray | None = None
        self._previous_contact = np.zeros(self.sensor_count, dtype=bool)
        self._persistence = np.zeros(self.sensor_count, dtype=np.int64)

    def reset(self) -> None:
        self._previous_normal = None
        self._previous_shear = None
        self._previous_cop = None
        self._previous_contact.fill(False)
        self._persistence.fill(0)

    def reshape_flat_taxels(
        self,
        penetration_depth: np.ndarray,
        normal_force: np.ndarray,
        signed_shear_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reshape native sensor-major flat taxels without reordering values."""

        taxels = self.grid_shape[0] * self.grid_shape[1]
        penetration = np.asarray(penetration_depth)
        normal = np.asarray(normal_force)
        shear = np.asarray(signed_shear_xy)
        if penetration.shape != (self.sensor_count, taxels):
            raise ValueError(
                f"penetration_depth shape {penetration.shape} != "
                f"{(self.sensor_count, taxels)}"
            )
        if normal.shape != (self.sensor_count, taxels):
            raise ValueError(
                f"normal_force shape {normal.shape} != "
                f"{(self.sensor_count, taxels)}"
            )
        if shear.shape != (self.sensor_count, taxels, 2):
            raise ValueError(
                f"signed_shear_xy shape {shear.shape} != "
                f"{(self.sensor_count, taxels, 2)}"
            )
        return (
            penetration.reshape(self.sensor_count, *self.grid_shape),
            normal.reshape(self.sensor_count, *self.grid_shape),
            shear.reshape(self.sensor_count, *self.grid_shape, 2),
        )

    def update(
        self,
        penetration_depth: np.ndarray,
        normal_force: np.ndarray,
        signed_shear_xy: np.ndarray,
    ) -> TactileFeatureFrame:
        """Validate native arrays and compute one causal feature frame."""

        penetration, normal, shear = self.reshape_flat_taxels(
            penetration_depth, normal_force, signed_shear_xy
        )
        for name, value in (
            ("penetration_depth", penetration),
            ("normal_force", normal),
            ("signed_shear_xy", shear),
        ):
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError(f"{name} must be numeric")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if np.any(penetration < 0.0):
            raise ValueError("native penetration_depth must be nonnegative")
        active = penetration > 0.0
        inactive = ~active
        if np.any(normal[inactive] != 0.0):
            raise ValueError("inactive native normal-force taxels must be exact zero")
        if np.any(shear[inactive] != 0.0):
            raise ValueError("inactive native shear-force taxels must be exact zero")
        has_contact = np.any(active, axis=(1, 2))
        active_count = np.count_nonzero(active, axis=(1, 2)).astype(np.int64)
        contact_fraction = active_count.astype(np.float64) / float(
            self.grid_shape[0] * self.grid_shape[1]
        )

        total_normal_signed = np.sum(normal, axis=(1, 2), dtype=np.float64)
        total_normal_magnitude = np.sum(
            np.abs(normal), axis=(1, 2), dtype=np.float64
        )
        total_shear_xy = np.sum(shear, axis=(1, 2), dtype=np.float64)
        shear_magnitude_map = np.linalg.norm(shear, axis=-1)
        total_shear_magnitude = np.sum(
            shear_magnitude_map, axis=(1, 2), dtype=np.float64
        )
        shear_to_normal_ratio = total_shear_magnitude / np.maximum(
            total_normal_magnitude, self.ratio_floor
        )

        y_coordinates = np.linspace(-1.0, 1.0, self.grid_shape[0])
        x_coordinates = np.linspace(-1.0, 1.0, self.grid_shape[1])
        grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
        coordinates = np.stack((grid_x, grid_y), axis=-1)
        cop = np.zeros((self.sensor_count, 2), dtype=np.float64)
        covariance = np.zeros((self.sensor_count, 2, 2), dtype=np.float64)
        principal = np.zeros((self.sensor_count, 2), dtype=np.float64)
        spread = np.zeros(self.sensor_count, dtype=np.float64)
        for sensor_index in range(self.sensor_count):
            # Preserve the official signed local-Z field and use an explicitly
            # derived pressure magnitude only for this spatial statistic.
            weights = np.abs(normal[sensor_index]).astype(
                np.float64, copy=False
            )
            weight_sum = float(np.sum(weights, dtype=np.float64))
            if weight_sum <= 0.0:
                continue
            cop[sensor_index] = np.sum(
                coordinates * weights[..., None], axis=(0, 1), dtype=np.float64
            ) / weight_sum
            centered = coordinates - cop[sensor_index]
            covariance[sensor_index] = np.einsum(
                "...i,...j,...->ij", centered, centered, weights, optimize=True
            ) / weight_sum
            eigenvalues, eigenvectors = np.linalg.eigh(covariance[sensor_index])
            principal[sensor_index] = eigenvectors[:, int(np.argmax(eigenvalues))]
            # Resolve the eigenvector's arbitrary sign deterministically.
            dominant = int(np.argmax(np.abs(principal[sensor_index])))
            if principal[sensor_index, dominant] < 0.0:
                principal[sensor_index] *= -1.0
            spread[sensor_index] = float(np.sqrt(np.trace(covariance[sensor_index])))

        if self._previous_normal is None:
            normal_change = np.zeros(self.sensor_count, dtype=np.float64)
            shear_change = np.zeros(self.sensor_count, dtype=np.float64)
            cop_motion = np.zeros(self.sensor_count, dtype=np.float64)
        else:
            normal_change = np.sum(
                np.abs(normal - self._previous_normal),
                axis=(1, 2),
                dtype=np.float64,
            )
            shear_change = np.sum(
                np.abs(shear - self._previous_shear),
                axis=(1, 2, 3),
                dtype=np.float64,
            )
            cop_motion = np.linalg.norm(cop - self._previous_cop, axis=-1)
            cop_motion[~(has_contact & self._previous_contact)] = 0.0

        onset = has_contact & ~self._previous_contact
        release = ~has_contact & self._previous_contact
        self._persistence = np.where(has_contact, self._persistence + 1, 0)
        features = TactileFeatureFrame(
            total_normal_signed=total_normal_signed,
            total_normal_magnitude=total_normal_magnitude,
            total_shear_xy=total_shear_xy,
            total_shear_magnitude=total_shear_magnitude,
            active_taxel_count=active_count,
            contact_fraction=contact_fraction,
            center_of_pressure_xy=cop,
            pressure_covariance_xy=covariance,
            principal_direction_xy=principal,
            spatial_spread=spread,
            shear_to_normal_ratio=shear_to_normal_ratio,
            normal_change_l1=normal_change,
            shear_change_l1=shear_change,
            center_of_pressure_motion=cop_motion,
            contact_onset=onset,
            contact_release=release,
            contact_persistence_frames=self._persistence.copy(),
        )
        self._previous_normal = normal.copy()
        self._previous_shear = shear.copy()
        self._previous_cop = cop.copy()
        self._previous_contact = has_contact.copy()
        return features
