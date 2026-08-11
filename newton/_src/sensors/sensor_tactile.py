# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from ..sim import Contacts, Model, State
from ..sim.contacts import contact_surface_point, contact_surface_separation
from ..utils.selection import match_labels


@wp.kernel(enable_backward=False)
def _compute_patch_transforms(
    patch_shapes: wp.array[wp.int32],
    patch_transform_shape: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    # output
    patch_transform_world: wp.array[wp.transform],
):
    patch = wp.tid()
    shape = patch_shapes[patch]
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = wp.transform_multiply(body_q[body], X_ws)
    patch_transform_world[patch] = wp.transform_multiply(X_ws, patch_transform_shape[patch])


@wp.func
def _record_tactile_sample(
    contact_index: int,
    sensor_is_shape0: int,
    patch: int,
    counterpart: int,
    point_world: wp.vec3,
    penetration: float,
    native_wrench_body0: wp.spatial_vector,
    patch_transform_world: wp.array[wp.transform],
    patch_size: wp.array[wp.vec2],
    rows: int,
    columns: int,
    # output
    raw_count: wp.array[wp.int32],
    raw_contact_index: wp.array[wp.int32],
    raw_patch: wp.array[wp.int32],
    raw_counterpart_shape: wp.array[wp.int32],
    raw_sensor_is_shape0: wp.array[wp.int32],
    raw_point_world: wp.array[wp.vec3],
    raw_point_patch: wp.array[wp.vec3],
    raw_force_world: wp.array[wp.vec3],
    raw_force_patch: wp.array[wp.vec3],
    raw_native_wrench_body0: wp.array[wp.spatial_vector],
    raw_penetration: wp.array[float],
    force: wp.array2d[wp.vec3],
    max_penetration: wp.array2d[float],
    active: wp.array2d[wp.int32],
    total_force_world: wp.array[wp.vec3],
    total_force_patch: wp.array[wp.vec3],
    unmapped_force_patch: wp.array[wp.vec3],
):
    X_wp = patch_transform_world[patch]
    X_pw = wp.transform_inverse(X_wp)
    point_patch = wp.transform_point(X_pw, point_world)

    force_world = wp.spatial_top(native_wrench_body0)
    if sensor_is_shape0 == 0:
        force_world = -force_world
    force_patch = wp.quat_rotate_inv(X_wp.q, force_world)

    raw_index = wp.atomic_add(raw_count, 0, 1)
    raw_contact_index[raw_index] = contact_index
    raw_patch[raw_index] = patch
    raw_counterpart_shape[raw_index] = counterpart
    raw_sensor_is_shape0[raw_index] = sensor_is_shape0
    raw_point_world[raw_index] = point_world
    raw_point_patch[raw_index] = point_patch
    raw_force_world[raw_index] = force_world
    raw_force_patch[raw_index] = force_patch
    raw_native_wrench_body0[raw_index] = native_wrench_body0
    raw_penetration[raw_index] = penetration

    wp.atomic_add(total_force_world, patch, force_world)
    wp.atomic_add(total_force_patch, patch, force_patch)

    size = patch_size[patch]
    u = point_patch[0]
    v = point_patch[1]
    if u < -0.5 * size[0] or u > 0.5 * size[0] or v < -0.5 * size[1] or v > 0.5 * size[1]:
        wp.atomic_add(unmapped_force_patch, patch, force_patch)
        return

    row_f = wp.clamp((u / size[0] + 0.5) * float(rows - 1), 0.0, float(rows - 1))
    col_f = wp.clamp((v / size[1] + 0.5) * float(columns - 1), 0.0, float(columns - 1))
    col0 = int(wp.floor(col_f))
    row0 = int(wp.floor(row_f))
    col1 = wp.min(col0 + 1, columns - 1)
    row1 = wp.min(row0 + 1, rows - 1)
    col_alpha = col_f - float(col0)
    row_alpha = row_f - float(row0)

    w00 = (1.0 - row_alpha) * (1.0 - col_alpha)
    w01 = (1.0 - row_alpha) * col_alpha
    w10 = row_alpha * (1.0 - col_alpha)
    w11 = row_alpha * col_alpha

    index00 = row0 * columns + col0
    index01 = row0 * columns + col1
    index10 = row1 * columns + col0
    index11 = row1 * columns + col1

    if w00 > 0.0:
        wp.atomic_add(force, patch, index00, w00 * force_patch)
        wp.atomic_max(max_penetration, patch, index00, penetration)
        wp.atomic_max(active, patch, index00, 1)
    if w01 > 0.0:
        wp.atomic_add(force, patch, index01, w01 * force_patch)
        wp.atomic_max(max_penetration, patch, index01, penetration)
        wp.atomic_max(active, patch, index01, 1)
    if w10 > 0.0:
        wp.atomic_add(force, patch, index10, w10 * force_patch)
        wp.atomic_max(max_penetration, patch, index10, penetration)
        wp.atomic_max(active, patch, index10, 1)
    if w11 > 0.0:
        wp.atomic_add(force, patch, index11, w11 * force_patch)
        wp.atomic_max(max_penetration, patch, index11, penetration)
        wp.atomic_max(active, patch, index11, 1)


@wp.kernel(enable_backward=False)
def _rasterize_contacts(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    contact_wrench: wp.array[wp.spatial_vector],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_to_patch: wp.array[wp.int32],
    counterpart_allowed: wp.array[wp.int32],
    patch_transform_world: wp.array[wp.transform],
    patch_size: wp.array[wp.vec2],
    rows: int,
    columns: int,
    # output
    raw_count: wp.array[wp.int32],
    raw_contact_index: wp.array[wp.int32],
    raw_patch: wp.array[wp.int32],
    raw_counterpart_shape: wp.array[wp.int32],
    raw_sensor_is_shape0: wp.array[wp.int32],
    raw_point_world: wp.array[wp.vec3],
    raw_point_patch: wp.array[wp.vec3],
    raw_force_world: wp.array[wp.vec3],
    raw_force_patch: wp.array[wp.vec3],
    raw_native_wrench_body0: wp.array[wp.spatial_vector],
    raw_penetration: wp.array[float],
    force: wp.array2d[wp.vec3],
    max_penetration: wp.array2d[float],
    active: wp.array2d[wp.int32],
    total_force_world: wp.array[wp.vec3],
    total_force_patch: wp.array[wp.vec3],
    unmapped_force_patch: wp.array[wp.vec3],
):
    contact_index = wp.tid()
    if contact_index >= contact_count[0]:
        return

    shape0 = contact_shape0[contact_index]
    shape1 = contact_shape1[contact_index]
    patch0 = shape_to_patch[shape0]
    patch1 = shape_to_patch[shape1]

    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    X_wb0 = wp.transform_identity()
    X_wb1 = wp.transform_identity()
    if body0 >= 0:
        X_wb0 = body_q[body0]
    if body1 >= 0:
        X_wb1 = body_q[body1]

    point0_world = contact_surface_point(X_wb0, contact_point0[contact_index], contact_offset0[contact_index])
    point1_world = contact_surface_point(X_wb1, contact_point1[contact_index], contact_offset1[contact_index])
    separation = contact_surface_separation(
        point0_world,
        point1_world,
        contact_normal[contact_index],
        contact_margin0[contact_index],
        contact_margin1[contact_index],
    )
    penetration = wp.max(0.0, -separation)
    native_wrench = contact_wrench[contact_index]

    if patch0 >= 0 and counterpart_allowed[shape1] != 0:
        _record_tactile_sample(
            contact_index,
            1,
            patch0,
            shape1,
            point0_world,
            penetration,
            native_wrench,
            patch_transform_world,
            patch_size,
            rows,
            columns,
            raw_count,
            raw_contact_index,
            raw_patch,
            raw_counterpart_shape,
            raw_sensor_is_shape0,
            raw_point_world,
            raw_point_patch,
            raw_force_world,
            raw_force_patch,
            raw_native_wrench_body0,
            raw_penetration,
            force,
            max_penetration,
            active,
            total_force_world,
            total_force_patch,
            unmapped_force_patch,
        )
    if patch1 >= 0 and counterpart_allowed[shape0] != 0:
        _record_tactile_sample(
            contact_index,
            0,
            patch1,
            shape0,
            point1_world,
            penetration,
            native_wrench,
            patch_transform_world,
            patch_size,
            rows,
            columns,
            raw_count,
            raw_contact_index,
            raw_patch,
            raw_counterpart_shape,
            raw_sensor_is_shape0,
            raw_point_world,
            raw_point_patch,
            raw_force_world,
            raw_force_patch,
            raw_native_wrench_body0,
            raw_penetration,
            force,
            max_penetration,
            active,
            total_force_world,
            total_force_patch,
            unmapped_force_patch,
        )


class SensorTactile:
    """Rasterize native solved contact forces on geometry-fixed tactile patches.

    Each sensing shape defines one patch. The patch frame is fixed relative to
    the shape, with local X increasing across rows, local Y increasing across
    columns, and local Z defining the signed normal-force channel. Contacts are
    accumulated bilinearly on a metric surface grid without changing their
    solved force or sign.

    The dense :attr:`force` field is a spatial serialization of
    :attr:`raw_force_patch`, not a compliant tactile force model. Native Newton
    does not provide GelSight RGB or elastomer depth, so this sensor does not
    fabricate optical output.

    Construct the sensor before creating :class:`~newton.Contacts`, then call
    ``solver.update_contacts(contacts, state)`` before :meth:`update`.

    Args:
        model: Simulation model providing the sensing geometry.
        sensing_shapes: Shape indices or label patterns, one per tactile patch.
        grid_shape: Number of rows and columns for every patch.
        patch_size: Patch row extent along local X and column extent along local Y [m]. A
            single pair applies to every patch; otherwise pass one pair per
            sensing shape.
        patch_transform_shape: Patch-to-shape transforms [m, unitless
            quaternion]. Defaults to the shape frame for every patch.
        counterpart_shapes: Optional shape indices or label patterns to retain.
            By default, contacts with every counterpart are retained.

    Raises:
        ValueError: If the patch geometry or shape selection is invalid.
    """

    force: wp.array2d[wp.vec3]
    """Signed patch-local XY shear and Z normal force [N], shape ``(patch_count, rows * columns)``."""

    max_penetration: wp.array2d[float]
    """Maximum positive penetration [m], shape ``(patch_count, rows * columns)``."""

    active: wp.array2d[wp.int32]
    """Taxel activity mask, shape ``(patch_count, rows * columns)``."""

    patch_transform_world: wp.array[wp.transform]
    """Patch-to-world transforms [m, unitless quaternion], shape ``(patch_count,)``."""

    patch_size: wp.array[wp.vec2]
    """Patch width and height [m], shape ``(patch_count,)``."""

    patch_transform_shape: wp.array[wp.transform]
    """Patch-to-shape transforms [m, unitless quaternion], shape ``(patch_count,)``."""

    total_force_world: wp.array[wp.vec3]
    """Native contact force on each sensing shape [N], world frame, shape ``(patch_count,)``."""

    total_force_patch: wp.array[wp.vec3]
    """Native contact force on each sensing shape [N], patch frame, shape ``(patch_count,)``."""

    unmapped_force_patch: wp.array[wp.vec3]
    """Force outside the declared patch bounds [N], patch frame, shape ``(patch_count,)``."""

    raw_count: wp.array[wp.int32]
    """Number of valid raw sensor-side samples, shape ``(1,)``."""

    raw_contact_index: wp.array[wp.int32]
    """Source Newton contact index per raw sample, shape ``(raw_capacity,)``."""

    raw_patch: wp.array[wp.int32]
    """Patch index per raw sample, shape ``(raw_capacity,)``."""

    raw_counterpart_shape: wp.array[wp.int32]
    """Counterpart shape index per raw sample, shape ``(raw_capacity,)``."""

    raw_sensor_is_shape0: wp.array[wp.int32]
    """One when the sensing shape is native shape0, zero for native shape1."""

    raw_point_world: wp.array[wp.vec3]
    """Effective-surface contact position [m], world frame, shape ``(raw_capacity,)``."""

    raw_point_patch: wp.array[wp.vec3]
    """Effective-surface contact position [m], patch frame, shape ``(raw_capacity,)``."""

    raw_force_world: wp.array[wp.vec3]
    """Solved force on the sensing shape [N], world frame, shape ``(raw_capacity,)``."""

    raw_force_patch: wp.array[wp.vec3]
    """Solved signed XY-shear/Z-normal force [N], patch frame, shape ``(raw_capacity,)``."""

    raw_native_wrench_body0: wp.array[wp.spatial_vector]
    """Unmodified native wrench on body0 [N, N·m], world frame, shape ``(raw_capacity,)``."""

    raw_penetration: wp.array[float]
    """Positive native effective-surface penetration [m], shape ``(raw_capacity,)``."""

    def __init__(
        self,
        model: Model,
        *,
        sensing_shapes: str | list[str] | list[int],
        grid_shape: tuple[int, int] = (20, 25),
        patch_size: tuple[float, float] | Sequence[tuple[float, float]] = (0.04, 0.03),
        patch_transform_shape: Sequence[wp.transform] | None = None,
        counterpart_shapes: str | list[str] | list[int] | None = None,
    ):
        self.device = model.device
        self._model = model

        sensing_indices = match_labels(model.shape_label, sensing_shapes)
        if not sensing_indices:
            raise ValueError("No shapes matched `sensing_shapes`.")
        if len(set(sensing_indices)) != len(sensing_indices):
            raise ValueError("`sensing_shapes` contains duplicate shape indices.")
        if any(index < 0 or index >= model.shape_count for index in sensing_indices):
            raise IndexError("`sensing_shapes` contains an out-of-range shape index.")

        rows, columns = grid_shape
        if rows < 2 or columns < 2:
            raise ValueError("`grid_shape` must contain at least two rows and two columns.")

        patch_count = len(sensing_indices)
        if len(patch_size) == 2 and isinstance(patch_size[0], (int, float)):
            sizes = [tuple(patch_size)] * patch_count
        else:
            sizes = [tuple(size) for size in patch_size]
        if len(sizes) != patch_count:
            raise ValueError("`patch_size` must be one pair or one pair per sensing shape.")
        if any(len(size) != 2 or size[0] <= 0.0 or size[1] <= 0.0 for size in sizes):
            raise ValueError("Every patch size must contain two positive metric lengths.")

        if patch_transform_shape is None:
            transforms = [wp.transform_identity()] * patch_count
        else:
            transforms = list(patch_transform_shape)
            if len(transforms) != patch_count:
                raise ValueError("`patch_transform_shape` must contain one transform per sensing shape.")

        counterpart_allowed = np.ones(model.shape_count, dtype=np.int32)
        if counterpart_shapes is not None:
            counterpart_indices = match_labels(model.shape_label, counterpart_shapes)
            if not counterpart_indices:
                raise ValueError("No shapes matched `counterpart_shapes`.")
            if any(index < 0 or index >= model.shape_count for index in counterpart_indices):
                raise IndexError("`counterpart_shapes` contains an out-of-range shape index.")
            counterpart_allowed.fill(0)
            counterpart_allowed[counterpart_indices] = 1
        else:
            counterpart_indices = list(range(model.shape_count))

        shape_to_patch = np.full(model.shape_count, -1, dtype=np.int32)
        shape_to_patch[sensing_indices] = np.arange(patch_count, dtype=np.int32)

        self.sensing_indices = sensing_indices
        self.counterpart_indices = counterpart_indices
        self.grid_shape = (rows, columns)
        self.patch_count = patch_count
        self.taxel_count = rows * columns
        self.optical_available = False
        self.sequence = -1
        self.timestamp = 0.0
        self.dt = 0.0
        self._has_timestamp = False
        self._raw_capacity = 0

        model.request_contact_attributes("force")
        self._patch_shapes = wp.array(sensing_indices, dtype=wp.int32, device=self.device)
        self.patch_transform_shape = wp.array(transforms, dtype=wp.transform, device=self.device)
        self.patch_size = wp.array(sizes, dtype=wp.vec2, device=self.device)
        self._shape_to_patch = wp.array(shape_to_patch, dtype=wp.int32, device=self.device)
        self._counterpart_allowed = wp.array(counterpart_allowed, dtype=wp.int32, device=self.device)

        self.force = wp.zeros((patch_count, self.taxel_count), dtype=wp.vec3, device=self.device)
        self.max_penetration = wp.zeros((patch_count, self.taxel_count), dtype=float, device=self.device)
        self.active = wp.zeros((patch_count, self.taxel_count), dtype=wp.int32, device=self.device)
        self.patch_transform_world = wp.zeros(patch_count, dtype=wp.transform, device=self.device)
        self.total_force_world = wp.zeros(patch_count, dtype=wp.vec3, device=self.device)
        self.total_force_patch = wp.zeros(patch_count, dtype=wp.vec3, device=self.device)
        self.unmapped_force_patch = wp.zeros(patch_count, dtype=wp.vec3, device=self.device)
        self._allocate_raw(0)

    @property
    def dense_shape(self) -> tuple[int, int, int]:
        """Logical dense shape ``(patch, row, column)``."""
        return (self.patch_count, *self.grid_shape)

    def _allocate_raw(self, contact_capacity: int) -> None:
        raw_capacity = max(1, 2 * contact_capacity)
        self._raw_capacity = raw_capacity
        self.raw_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.raw_contact_index = wp.full(raw_capacity, -1, dtype=wp.int32, device=self.device)
        self.raw_patch = wp.full(raw_capacity, -1, dtype=wp.int32, device=self.device)
        self.raw_counterpart_shape = wp.full(raw_capacity, -1, dtype=wp.int32, device=self.device)
        self.raw_sensor_is_shape0 = wp.zeros(raw_capacity, dtype=wp.int32, device=self.device)
        self.raw_point_world = wp.zeros(raw_capacity, dtype=wp.vec3, device=self.device)
        self.raw_point_patch = wp.zeros(raw_capacity, dtype=wp.vec3, device=self.device)
        self.raw_force_world = wp.zeros(raw_capacity, dtype=wp.vec3, device=self.device)
        self.raw_force_patch = wp.zeros(raw_capacity, dtype=wp.vec3, device=self.device)
        self.raw_native_wrench_body0 = wp.zeros(raw_capacity, dtype=wp.spatial_vector, device=self.device)
        self.raw_penetration = wp.zeros(raw_capacity, dtype=float, device=self.device)

    def reset(self) -> None:
        """Clear readings and reset the source clock."""
        self.force.zero_()
        self.max_penetration.zero_()
        self.active.zero_()
        self.total_force_world.zero_()
        self.total_force_patch.zero_()
        self.unmapped_force_patch.zero_()
        self.raw_count.zero_()
        self.sequence = -1
        self.timestamp = 0.0
        self.dt = 0.0
        self._has_timestamp = False

    def update(self, state: State, contacts: Contacts, *, timestamp: float) -> None:
        """Update the tactile frame from current solved contacts.

        Args:
            state: Current simulation state providing body transforms.
            contacts: Current contacts after the solver contact update.
            timestamp: Source simulation timestamp [s].

        Raises:
            ValueError: If force data, device, state, or source time is invalid.
        """
        if state is None or state.body_q is None:
            raise ValueError("SensorTactile requires a state with `body_q`.")
        if contacts.force is None:
            raise ValueError(
                "SensorTactile requires `Contacts.force`; construct the sensor before creating the Contacts object."
            )
        if contacts.device != self.device:
            raise ValueError(f"Contacts device ({contacts.device}) does not match sensor device ({self.device}).")
        if self._has_timestamp and timestamp < self.timestamp:
            raise ValueError("Tactile source timestamps must be nondecreasing.")

        if 2 * contacts.rigid_contact_max > self._raw_capacity:
            self._allocate_raw(contacts.rigid_contact_max)

        self.force.zero_()
        self.max_penetration.zero_()
        self.active.zero_()
        self.total_force_world.zero_()
        self.total_force_patch.zero_()
        self.unmapped_force_patch.zero_()
        self.raw_count.zero_()

        wp.launch(
            _compute_patch_transforms,
            dim=self.patch_count,
            inputs=[
                self._patch_shapes,
                self.patch_transform_shape,
                self._model.shape_body,
                self._model.shape_transform,
                state.body_q,
            ],
            outputs=[self.patch_transform_world],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _rasterize_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.force,
                self._model.shape_body,
                state.body_q,
                self._shape_to_patch,
                self._counterpart_allowed,
                self.patch_transform_world,
                self.patch_size,
                self.grid_shape[0],
                self.grid_shape[1],
            ],
            outputs=[
                self.raw_count,
                self.raw_contact_index,
                self.raw_patch,
                self.raw_counterpart_shape,
                self.raw_sensor_is_shape0,
                self.raw_point_world,
                self.raw_point_patch,
                self.raw_force_world,
                self.raw_force_patch,
                self.raw_native_wrench_body0,
                self.raw_penetration,
                self.force,
                self.max_penetration,
                self.active,
                self.total_force_world,
                self.total_force_patch,
                self.unmapped_force_patch,
            ],
            device=self.device,
            record_tape=False,
        )

        self.dt = timestamp - self.timestamp if self._has_timestamp else 0.0
        self.timestamp = float(timestamp)
        self._has_timestamp = True
        self.sequence += 1
