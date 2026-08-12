#!/usr/bin/env python3
"""Common native tactile frame and direct Newton/IsaacLab adapters.

The adapters preserve simulator-native physical signals. IsaacLab data comes
only from official TacSL ``VisuoTactileSensorData`` fields. Newton data comes
only from ``newton.sensors.SensorTactile``, which rasterizes solved native
contact forces. No object state, rigid-contact proxy, or outcome label enters
the frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TactileClock:
    """Source simulation clock for one tactile frame."""

    sequence: int
    timestamp_s: float
    dt_s: float


@dataclass(frozen=True)
class CounterpartTactileField:
    """Unaggregated native taxel field for one declared contact object."""

    penetration_m: Any
    normal_force_n: Any
    shear_force_xy_n: Any
    active: Any


@dataclass(frozen=True)
class OpticalTactileFrame:
    """Optional native optical streams, retained patch by patch."""

    available: tuple[bool, ...]
    rgb: tuple[Any | None, ...]
    depth: tuple[Any | None, ...]
    clock: TactileClock | None


@dataclass(frozen=True)
class NewtonRawTactileSamples:
    """Unmodified Newton contact samples behind a derived patch raster."""

    contact_index: np.ndarray
    contact_kind: np.ndarray
    patch_index: np.ndarray
    counterpart_shape: np.ndarray
    counterpart_particle: np.ndarray
    sensor_is_shape0: np.ndarray
    point_world_m: np.ndarray
    point_patch_m: np.ndarray
    force_world_n: np.ndarray
    force_patch_n: np.ndarray
    native_wrench_body0: np.ndarray
    penetration_m: np.ndarray


@dataclass(frozen=True)
class UniversalTactileFrame:
    """Backend-neutral native tactile frame.

    Dense scalar tensors use ``[batch, patch, row, column]``. Signed shear
    uses ``[batch, patch, row, column, 2]`` and world positions/orientations
    append dimensions 3 and 4. The first local tangent axis (signed shear X)
    increases with row, the second local tangent axis (signed shear Y)
    increases with column, and local Z is the signed normal-force direction.
    ``patch_size_m[..., 0]`` is therefore the row/X extent and
    ``patch_size_m[..., 1]`` the column/Y extent. This is the released TacSL
    taxel order, not an image-space transpose.
    """

    backend: str
    clock: TactileClock
    patch_names: tuple[str, ...]
    patch_size_m: np.ndarray
    penetration_m: Any
    normal_force_n: Any
    shear_force_xy_n: Any
    active: Any
    taxel_position_w_m: Any | None
    taxel_orientation_w_xyzw: Any | None
    counterpart_fields: Mapping[str, CounterpartTactileField]
    optical: OpticalTactileFrame
    raw_samples: NewtonRawTactileSamples | None

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.penetration_m.shape[2]), int(self.penetration_m.shape[3]))

    @property
    def batch_size(self) -> int:
        return int(self.penetration_m.shape[0])


def _stack(values: Sequence[Any], axis: int) -> Any:
    first = values[0]
    if type(first).__module__.startswith("torch"):
        import torch

        return torch.stack(tuple(values), dim=axis)
    return np.stack(values, axis=axis)


def _sum(values: Sequence[Any]) -> Any:
    stacked = _stack(values, axis=0)
    if type(stacked).__module__.startswith("torch"):
        return stacked.sum(dim=0)
    return stacked.sum(axis=0)


def _maximum(values: Sequence[Any]) -> Any:
    stacked = _stack(values, axis=0)
    if type(stacked).__module__.startswith("torch"):
        return stacked.max(dim=0).values
    return stacked.max(axis=0)


class IsaacLabTacSLAdapter:
    """Adapt official TacSL data streams without changing force or taxel order.

    ``update`` accepts one mapping entry per declared SDF counterpart. Each
    value contains one official ``VisuoTactileSensorData`` object per physical
    patch, in ``patch_names`` order. The per-counterpart fields are retained;
    the main dense field is the explicit sum of signed forces and maximum of
    penetration across those named streams.
    """

    def __init__(
        self,
        patch_names: Sequence[str],
        *,
        grid_shape: tuple[int, int] = (20, 25),
        patch_size_m: tuple[float, float] | Sequence[tuple[float, float]] = (0.04, 0.03),
    ) -> None:
        if not patch_names:
            raise ValueError("At least one physical patch is required.")
        self.patch_names = tuple(patch_names)
        self.grid_shape = tuple(grid_shape)
        self.taxel_count = self.grid_shape[0] * self.grid_shape[1]
        if len(patch_size_m) == 2 and isinstance(patch_size_m[0], (int, float)):
            sizes = [tuple(patch_size_m)] * len(self.patch_names)
        else:
            sizes = [tuple(size) for size in patch_size_m]
        if len(sizes) != len(self.patch_names):
            raise ValueError("Patch size must be one pair or one pair per patch.")
        self.patch_size_m = np.asarray(sizes, dtype=np.float32)
        self._sequence = -1
        self._timestamp_s = 0.0
        self._has_timestamp = False
        self._optical_sequence = -1
        self._optical_timestamp_s = 0.0
        self._has_optical_timestamp = False
        self._optical_clock: TactileClock | None = None

    def reset(self) -> None:
        self._sequence = -1
        self._timestamp_s = 0.0
        self._has_timestamp = False
        self._optical_sequence = -1
        self._optical_timestamp_s = 0.0
        self._has_optical_timestamp = False
        self._optical_clock = None

    def _field(self, data_by_patch: Sequence[Any]) -> CounterpartTactileField:
        if len(data_by_patch) != len(self.patch_names):
            raise ValueError("Every counterpart must provide one official stream per patch.")
        for data in data_by_patch:
            if data.penetration_depth is None or data.tactile_normal_force is None or data.tactile_shear_force is None:
                raise ValueError("Official TacSL penetration, normal force, and shear force must all be available.")
            if data.penetration_depth.shape[-1] != self.taxel_count:
                raise ValueError("Official TacSL taxel count does not match the declared grid.")

        batch = int(data_by_patch[0].penetration_depth.shape[0])
        shape = (batch, len(self.patch_names), *self.grid_shape)
        penetration = _stack([data.penetration_depth for data in data_by_patch], axis=1).reshape(shape)
        normal = _stack([data.tactile_normal_force for data in data_by_patch], axis=1).reshape(shape)
        shear = _stack([data.tactile_shear_force for data in data_by_patch], axis=1).reshape((*shape, 2))
        return CounterpartTactileField(
            penetration_m=penetration,
            normal_force_n=normal,
            shear_force_xy_n=shear,
            active=penetration > 0.0,
        )

    def update(
        self,
        data_by_counterpart: Mapping[str, Sequence[Any]],
        *,
        timestamp_s: float,
        optical_timestamp_s: float | None = None,
    ) -> UniversalTactileFrame:
        """Create one frame from current official TacSL data objects."""
        if not data_by_counterpart:
            raise ValueError("At least one declared TacSL counterpart stream is required.")
        if self._has_timestamp and timestamp_s < self._timestamp_s:
            raise ValueError("Tactile source timestamps must be nondecreasing.")

        fields = {name: self._field(streams) for name, streams in data_by_counterpart.items()}
        penetration = _maximum([field.penetration_m for field in fields.values()])
        normal = _sum([field.normal_force_n for field in fields.values()])
        shear = _sum([field.shear_force_xy_n for field in fields.values()])

        first_streams = next(iter(data_by_counterpart.values()))
        positions = None
        orientations = None
        if all(data.tactile_points_pos_w is not None for data in first_streams):
            positions = _stack([data.tactile_points_pos_w for data in first_streams], axis=1).reshape(
                (*penetration.shape, 3)
            )
        if all(data.tactile_points_quat_w is not None for data in first_streams):
            # IsaacLab tensors use scalar-first wxyz; the common contract and
            # Warp/Newton use scalar-last xyzw. This is an order conversion of
            # the same native orientation, not a reconstructed pose.
            orientations_wxyz = _stack(
                [data.tactile_points_quat_w for data in first_streams], axis=1
            ).reshape((*penetration.shape, 4))
            orientations = orientations_wxyz[..., (1, 2, 3, 0)]

        optical_rgb: list[Any | None] = []
        optical_depth: list[Any | None] = []
        optical_available: list[bool] = []
        for patch_index in range(len(self.patch_names)):
            patch_rgb = None
            patch_depth = None
            for streams in data_by_counterpart.values():
                data = streams[patch_index]
                if patch_rgb is None and data.tactile_rgb_image is not None:
                    patch_rgb = data.tactile_rgb_image
                if patch_depth is None and data.tactile_depth_image is not None:
                    patch_depth = data.tactile_depth_image
            optical_rgb.append(patch_rgb)
            optical_depth.append(patch_depth)
            optical_available.append(patch_rgb is not None and patch_depth is not None)

        any_optical = any(optical_available)
        optical_clock = None
        if any_optical:
            if optical_timestamp_s is None:
                if self._optical_clock is None:
                    raise ValueError(
                        "The first available official RGB/depth frame requires an optical timestamp."
                    )
                optical_clock = self._optical_clock
            else:
                if self._has_optical_timestamp and optical_timestamp_s < self._optical_timestamp_s:
                    raise ValueError("Optical source timestamps must be nondecreasing.")
                if not self._has_optical_timestamp or optical_timestamp_s > self._optical_timestamp_s:
                    optical_dt_s = (
                        optical_timestamp_s - self._optical_timestamp_s
                        if self._has_optical_timestamp
                        else 0.0
                    )
                    self._optical_timestamp_s = float(optical_timestamp_s)
                    self._has_optical_timestamp = True
                    self._optical_sequence += 1
                    self._optical_clock = TactileClock(
                        self._optical_sequence,
                        self._optical_timestamp_s,
                        optical_dt_s,
                    )
                optical_clock = self._optical_clock

        dt_s = timestamp_s - self._timestamp_s if self._has_timestamp else 0.0
        self._timestamp_s = float(timestamp_s)
        self._has_timestamp = True
        self._sequence += 1
        clock = TactileClock(self._sequence, self._timestamp_s, dt_s)
        return UniversalTactileFrame(
            backend="isaaclab_tacsl",
            clock=clock,
            patch_names=self.patch_names,
            patch_size_m=self.patch_size_m.copy(),
            penetration_m=penetration,
            normal_force_n=normal,
            shear_force_xy_n=shear,
            active=penetration > 0.0,
            taxel_position_w_m=positions,
            taxel_orientation_w_xyzw=orientations,
            counterpart_fields=fields,
            optical=OpticalTactileFrame(
                available=tuple(optical_available),
                rgb=tuple(optical_rgb),
                depth=tuple(optical_depth),
                clock=optical_clock,
            ),
            raw_samples=None,
        )


def _quat_rotate_xyzw(quaternion: np.ndarray, points: np.ndarray) -> np.ndarray:
    q_xyz = quaternion[..., :3]
    q_w = quaternion[..., 3:4]
    q_xyz = np.broadcast_to(q_xyz, points.shape)
    q_w = np.broadcast_to(q_w, points.shape[:-1] + (1,))
    cross = np.cross(q_xyz, points)
    return points + 2.0 * (q_w * cross + np.cross(q_xyz, cross))


class NewtonTactileAdapter:
    """Adapt one updated ``newton.sensors.SensorTactile`` to the common frame."""

    def __init__(self, sensor: Any, patch_names: Sequence[str] | None = None) -> None:
        self.sensor = sensor
        if patch_names is None:
            patch_names = [f"patch_{index}" for index in range(sensor.patch_count)]
        if len(patch_names) != sensor.patch_count:
            raise ValueError("Patch names must match the Newton sensor patch count.")
        self.patch_names = tuple(patch_names)

    def frame(self) -> UniversalTactileFrame:
        """Read the current sensor outputs without recomputing contact physics."""
        sensor = self.sensor
        rows, columns = sensor.grid_shape
        patch_count = sensor.patch_count
        dense_shape = (1, patch_count, rows, columns)
        force = sensor.force.numpy().reshape((*dense_shape, 3))
        penetration = sensor.max_penetration.numpy().reshape(dense_shape)
        active = sensor.active.numpy().reshape(dense_shape).astype(bool, copy=False)
        patch_size = sensor.patch_size.numpy()

        row = np.linspace(-0.5, 0.5, rows, dtype=np.float32)
        column = np.linspace(-0.5, 0.5, columns, dtype=np.float32)
        grid_column, grid_row = np.meshgrid(column, row)
        unit_points = np.stack(
            (grid_row, grid_column, np.zeros_like(grid_row)), axis=-1
        )
        transforms = sensor.patch_transform_world.numpy()
        local_points = unit_points[None] * np.concatenate(
            (patch_size[:, None, None, :], np.ones((patch_count, 1, 1, 1), dtype=np.float32)), axis=-1
        )
        rotations = transforms[:, None, None, 3:7]
        positions = _quat_rotate_xyzw(rotations, local_points) + transforms[:, None, None, :3]
        orientations = np.broadcast_to(rotations, (patch_count, rows, columns, 4)).copy()

        raw_count = int(sensor.raw_count.numpy()[0])
        raw = NewtonRawTactileSamples(
            contact_index=sensor.raw_contact_index.numpy()[:raw_count].copy(),
            contact_kind=sensor.raw_contact_kind.numpy()[:raw_count].copy(),
            patch_index=sensor.raw_patch.numpy()[:raw_count].copy(),
            counterpart_shape=sensor.raw_counterpart_shape.numpy()[:raw_count].copy(),
            counterpart_particle=sensor.raw_counterpart_particle.numpy()[:raw_count].copy(),
            sensor_is_shape0=sensor.raw_sensor_is_shape0.numpy()[:raw_count].copy(),
            point_world_m=sensor.raw_point_world.numpy()[:raw_count].copy(),
            point_patch_m=sensor.raw_point_patch.numpy()[:raw_count].copy(),
            force_world_n=sensor.raw_force_world.numpy()[:raw_count].copy(),
            force_patch_n=sensor.raw_force_patch.numpy()[:raw_count].copy(),
            native_wrench_body0=sensor.raw_native_wrench_body0.numpy()[:raw_count].copy(),
            penetration_m=sensor.raw_penetration.numpy()[:raw_count].copy(),
        )
        return UniversalTactileFrame(
            backend="newton_native_contacts",
            clock=TactileClock(sensor.sequence, sensor.timestamp, sensor.dt),
            patch_names=self.patch_names,
            patch_size_m=patch_size.copy(),
            penetration_m=penetration,
            normal_force_n=force[..., 2],
            shear_force_xy_n=force[..., :2],
            active=active,
            taxel_position_w_m=positions[None],
            taxel_orientation_w_xyzw=orientations[None],
            counterpart_fields={},
            optical=OpticalTactileFrame(
                available=(False,) * patch_count,
                rgb=(None,) * patch_count,
                depth=(None,) * patch_count,
                clock=None,
            ),
            raw_samples=raw,
        )
