# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-face tactile *field* over the hydroelastic contact surface.

:mod:`sugar_newton.tactile.reducer` answers "how hard is this link loaded" -- one number
per link per channel. That is what a policy consumes, but it is not what a skin looks
like. This module keeps the same measurements at their native resolution: one sample per
contact-surface triangle, positioned in a body frame, so the result is a continuous map
across the pads rather than a value per link.

Three channels, and they are not equally direct -- the difference is the point:

**Pressure** [Pa] is genuinely per-face. The hydroelastic surface carries a penetration
depth per face, and :func:`reducer.reduce_contact_surface_kernel` explains at length why
``kh * depth`` is *not* the pressure when ``use_mujoco_contacts=False`` (measured 67x too
large on the incline scene). The depth field supplies the shape, the solved normal load
supplies the magnitude::

    p_i = penetration_i * normal_load / sum_j(penetration_j * area_j)

which integrates over the patch to the load MuJoCo actually solved.

**Slip velocity** [m/s] is genuinely per-face, and is recorded as a *vector*
(``slip_vec``) as well as a magnitude. It is the relative surface velocity of the two
bodies evaluated at *that face's* centroid, ``v + omega x (p - com)``, projected onto the
face's tangent plane -- so it says which way the object is sliding across the skin, not
just how fast. Two faces on the same link report different slip whenever the link is
rotating, which is exactly the signal a per-link mean destroys.

**Tangential traction** [Pa] is *not* measured per face, and this module does not pretend
otherwise. Newton's contact buffer carries friction force per contact, and the pipeline
reduces a patch's surface to a handful of contacts, so there is no per-triangle friction
to read. What is done instead: the patch's measured friction load is distributed across
its faces in proportion to pressure, which for the traction field reduces to

    tau_i = p_i * (friction_load / normal_load)

i.e. the pressure map scaled by the patch's own realised friction ratio. The *shape* of
the traction map is therefore the shape of the pressure map; what it adds is where the
patch is actually shearing and by how much. Read it as an estimate.

Its *direction* (``traction_vec``) comes from the patch's measured tangential force
vector -- the one the solver reports on this patch's body, sign-corrected by the reducer
-- projected onto each face's tangent plane. So direction is per-patch and orientation is
per-face. It should point against the slip: friction opposes relative motion, and
comparing ``traction_vec`` with ``slip_vec`` is a free consistency check on both.

Face centroids are expressed in a chosen body frame (the palm, for a hand) so the map
holds still while the hand moves -- a skin readout, not a drifting point cloud.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.math.spatial import velocity_at_point


@wp.kernel
def extract_field_kernel(
    face_count: wp.array[wp.int32],
    surface_point: wp.array[wp.vec3f],
    surface_depth: wp.array[wp.float32],
    surface_shape_pair: wp.array[wp.vec2i],
    shape_to_patch: wp.array[wp.int32],
    shape_is_counterpart: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    patch_normal_load: wp.array[wp.float32],
    patch_friction_load: wp.array[wp.float32],
    patch_friction_vec: wp.array[wp.vec3],
    patch_depth_area_sum: wp.array[wp.float32],
    patch_frame_body: wp.array[wp.int32],
    patch_pad_offset: wp.array[wp.int32],
    pad_footprint: wp.array2d[wp.float32],
    pad_normal_sign: wp.array[wp.float32],
    capacity: int,
    # outputs
    out_total: wp.array[wp.int32],
    out_pos: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3],
    out_area: wp.array[wp.float32],
    out_depth: wp.array[wp.float32],
    out_pressure: wp.array[wp.float32],
    out_traction: wp.array[wp.float32],
    out_traction_vec: wp.array[wp.vec3],
    out_slip: wp.array[wp.float32],
    out_slip_vec: wp.array[wp.vec3],
    out_patch: wp.array[wp.int32],
    out_pad: wp.array[wp.int32],
):
    """One sample per penetrating contact-surface triangle, compacted by an atomic.

    ``out_total`` counts every qualifying face, including those past ``capacity`` that
    were not written, so the caller can report the truncation instead of hiding it.
    """
    face = wp.tid()
    if face >= face_count[0]:
        return

    pair = surface_shape_pair[face]
    shape_a, shape_b = pair[0], pair[1]
    if shape_a < 0 or shape_b < 0:
        return

    row = shape_to_patch[shape_a]
    patch_shape = shape_a
    other = shape_b
    if row < 0:
        row = shape_to_patch[shape_b]
        patch_shape = shape_b
        other = shape_a
    if row < 0 or shape_is_counterpart[other] == 0:
        return

    penetration = -surface_depth[face]
    if penetration <= 0.0:
        return  # outside the overlap: carries no load, same rule as the reducer
    denom = patch_depth_area_sum[row]
    if denom <= 0.0:
        return

    v0 = surface_point[3 * face + 0]
    v1 = surface_point[3 * face + 1]
    v2 = surface_point[3 * face + 2]
    cr = wp.cross(v1 - v0, v2 - v0)
    area = 0.5 * wp.length(cr)
    if area <= 0.0:
        return
    n = wp.normalize(cr)
    c = (v0 + v1 + v2) / 3.0

    normal_load = patch_normal_load[row]
    pressure = penetration * normal_load / denom
    traction = pressure * patch_friction_load[row] / wp.max(normal_load, 1.0e-9)

    # relative surface velocity at THIS face, not at the patch's reduced contact point
    b_patch = shape_body[patch_shape]
    b_other = shape_body[other]
    v_rel = wp.vec3(0.0, 0.0, 0.0)
    if b_patch >= 0:
        r = c - wp.transform_point(body_q[b_patch], body_com[b_patch])
        v_rel = v_rel + velocity_at_point(body_qd[b_patch], r)
    if b_other >= 0:
        r = c - wp.transform_point(body_q[b_other], body_com[b_other])
        v_rel = v_rel - velocity_at_point(body_qd[b_other], r)
    v_t = v_rel - wp.dot(v_rel, n) * n

    # Traction as a VECTOR. The magnitude is per-face (above); the direction is the
    # patch's own measured tangential force, projected onto this face's tangent plane so
    # it is a legal surface traction there. Direction is therefore per-patch and
    # orientation per-face -- Newton has no per-triangle friction to read, for the same
    # reason the magnitude has to be distributed rather than measured.
    t_vec = wp.vec3(0.0, 0.0, 0.0)
    ft = patch_friction_vec[row]
    ft_t = ft - wp.dot(ft, n) * n
    ft_len = wp.length(ft_t)
    if ft_len > 1.0e-12:
        t_vec = ft_t * (traction / ft_len)

    frame_body = patch_frame_body[row]
    if frame_body >= 0:
        inv = wp.transform_inverse(body_q[frame_body])
        rot = wp.transform_get_rotation(inv)
        c = wp.transform_point(inv, c)
        n = wp.quat_rotate(rot, n)
        v_t = wp.quat_rotate(rot, v_t)
        t_vec = wp.quat_rotate(rot, t_vec)

    # Which anatomical patch of the skin is this face on?
    #
    # Not "which shape did it hit" -- the collider is the whole hand, one shape, exactly
    # as the asset ships it. The 27 patches are a fixed partition of the palm-side
    # surface, so a face is assigned by testing where it landed. -1 means it is on skin
    # no patch covers, which is worth knowing rather than hiding.
    pad = int(-1)
    base = patch_pad_offset[row]
    if base >= 0 and pad_normal_sign[row] * c[1] > 0.0:
        n_pads = pad_footprint.shape[0]
        for k in range(n_pads):
            dx = c[0] - pad_footprint[k, 0]
            dz = c[2] - pad_footprint[k, 1]
            ca = wp.cos(pad_footprint[k, 4])
            sa = wp.sin(pad_footprint[k, 4])
            lx = ca * dx + sa * dz
            lz = -sa * dx + ca * dz
            if wp.abs(lx) <= pad_footprint[k, 2] and wp.abs(lz) <= pad_footprint[k, 3]:
                pad = base + k
                break

    idx = wp.atomic_add(out_total, 0, 1)
    if idx >= capacity:
        return
    out_pos[idx] = c
    out_normal[idx] = n
    out_area[idx] = area
    out_depth[idx] = penetration
    out_pressure[idx] = pressure
    out_traction[idx] = traction
    out_traction_vec[idx] = t_vec
    out_slip[idx] = wp.length(v_t)
    out_slip_vec[idx] = v_t
    out_patch[idx] = row
    out_pad[idx] = pad


class ContactField:
    """Per-face tactile field, sampled once per control step.

    Args:
        tactile: A :class:`~sugar_newton.tactile.reducer.PatchTactile` that has already
            been updated this step.  Its patch/counterpart tables and its solved
            ``normal_load`` / ``friction_load`` / depth-area sums are what turn the raw
            surface into pascals, so the two are deliberately not independent.
        frame_body: Body frame each patch's samples are expressed in -- one index, or one
            per patch.  A hand's own link is the natural choice: the patch layout is fixed
            in it, so the map is a canonical skin diagram rather than a moving point
            cloud.  With two hands, pass one body per patch and each hand's faces land in
            its own frame.  ``-1`` leaves samples in world coordinates.
        capacity: Maximum faces retained per step.

    Call after :meth:`PatchTactile.update`, with the same contact surface::

        surface = pipeline.hydroelastic_sdf.get_contact_surface()
        tactile.update(state, contacts, contact_surface=surface)
        field.update(state, surface)
    """

    def __init__(self, tactile, frame_body=-1, capacity: int = 200_000,
                 pad_footprint=None, pad_offset=None, palm_sign=None):
        self.tactile = tactile
        self.model = tactile.model
        n_rows = tactile.num_patches
        frames = np.full(n_rows, int(frame_body), dtype=np.int32) if np.isscalar(frame_body) \
            else np.asarray(frame_body, dtype=np.int32)
        if frames.shape != (n_rows,):
            raise ValueError(f"frame_body must be a scalar or one index per patch ({n_rows})")
        self.frame_body = frames
        self.capacity = int(capacity)
        self.count = 0
        self.total = 0
        self.overflow_steps = 0
        self.max_faces_seen = 0

        with wp.ScopedDevice(self.model.device):
            self._total = wp.zeros(1, dtype=wp.int32)
            self._frame_body = wp.array(frames, dtype=wp.int32)
            # The anatomical partition, optional: without it every face reports pad -1.
            fp = np.zeros((1, 5), dtype=np.float32) if pad_footprint is None \
                else np.ascontiguousarray(pad_footprint, dtype=np.float32)
            self._pad_footprint = wp.array2d(fp, dtype=wp.float32)
            self._pad_offset = wp.array(
                np.full(n_rows, -1, dtype=np.int32) if pad_offset is None
                else np.asarray(pad_offset, dtype=np.int32), dtype=wp.int32)
            self._palm_sign = wp.array(
                np.ones(n_rows, dtype=np.float32) if palm_sign is None
                else np.asarray(palm_sign, dtype=np.float32), dtype=wp.float32)
            self.pad = wp.zeros(self.capacity, dtype=wp.int32)
            self.n_pads = fp.shape[0]
            self.pos = wp.zeros(self.capacity, dtype=wp.vec3)
            self.normal = wp.zeros(self.capacity, dtype=wp.vec3)
            self.area = wp.zeros(self.capacity, dtype=wp.float32)
            self.depth = wp.zeros(self.capacity, dtype=wp.float32)
            self.pressure = wp.zeros(self.capacity, dtype=wp.float32)
            self.traction = wp.zeros(self.capacity, dtype=wp.float32)
            self.traction_vec = wp.zeros(self.capacity, dtype=wp.vec3)
            self.slip = wp.zeros(self.capacity, dtype=wp.float32)
            self.slip_vec = wp.zeros(self.capacity, dtype=wp.vec3)
            self.patch = wp.zeros(self.capacity, dtype=wp.int32)

    def update(self, state, contact_surface) -> None:
        self.count = self.total = 0
        if contact_surface is None:
            return
        t = self.tactile
        self._total.zero_()
        wp.launch(
            extract_field_kernel,
            dim=contact_surface.max_num_face_contacts,
            inputs=[
                contact_surface.face_contact_count,
                contact_surface.contact_surface_point,
                contact_surface.contact_surface_depth,
                contact_surface.contact_surface_shape_pair,
                t.shape_to_patch,
                t.shape_is_counterpart,
                self.model.shape_body,
                state.body_q,
                state.body_qd,
                self.model.body_com,
                t.normal_load,
                t.friction_load,
                t.friction_vec,
                t._depth_area_sum,
                self._frame_body,
                self._pad_offset,
                self._pad_footprint,
                self._palm_sign,
                self.capacity,
            ],
            outputs=[
                self._total, self.pos, self.normal, self.area, self.depth,
                self.pressure, self.traction, self.traction_vec,
                self.slip, self.slip_vec, self.patch, self.pad,
            ],
            device=self.model.device,
        )
        self.total = int(self._total.numpy()[0])
        self.max_faces_seen = max(self.max_faces_seen, self.total)
        if self.total > self.capacity:
            self.overflow_steps += 1
        self.count = min(self.total, self.capacity)

    def to_numpy(self, stride_to: int | None = None) -> dict:
        """Host copy of this step's samples.

        Args:
            stride_to: If given and the step holds more faces than this, keep an evenly
                spaced subset.  Every channel here is intensive (Pa, m/s), so a subset is
                still an unbiased picture of the field -- but it is no longer something
                to integrate, and the caller is expected to say how many were dropped.
        """
        n = self.count
        sel = slice(0, n)
        if stride_to is not None and n > stride_to:
            sel = np.linspace(0, n - 1, stride_to).astype(np.int64)
        # Slice the DEVICE array before ``numpy()``: the buffers are sized for the worst
        # case, and copying all of `capacity` back every frame when a handful of faces
        # are in contact dominates the step.
        #
        # The empty case spells its numpy dtype out. ``wp.float32`` is not a numpy dtype
        # and ``np.zeros(0, wp.float32)`` quietly produces an OBJECT array, which survives
        # ``np.concatenate`` and only fails much later, at ``np.load``, as
        # "Object arrays cannot be loaded when allow_pickle=False".
        def get(a, shape, dtype):
            return a[:n].numpy()[sel] if n else np.zeros((0, *shape), dtype)

        f32, i32 = np.float32, np.int32
        return {
            "pos": get(self.pos, (3,), f32), "normal": get(self.normal, (3,), f32),
            "area": get(self.area, (), f32), "depth": get(self.depth, (), f32),
            "pressure": get(self.pressure, (), f32), "traction": get(self.traction, (), f32),
            "traction_vec": get(self.traction_vec, (3,), f32),
            "slip": get(self.slip, (), f32), "slip_vec": get(self.slip_vec, (3,), f32),
            "patch": get(self.patch, (), i32), "pad": get(self.pad, (), i32),
        }
