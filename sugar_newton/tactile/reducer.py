# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-patch tactile reduction from Newton rigid contacts.

Plan 16 §4.  Every channel is read off the contact buffer the solver integrated,
not off a parallel read-only view of the scene.  This is the whole point of the
rewrite: Plan 15's tactile came from TacSL, which projected the *total* contact
force into a per-taxel frame and divided by a sensor-fixed ``mu = 0.5``, so its
"shear" channel leaked the normal force under stick and its "friction" channel
could not see the object's real friction at all.

Three properties this module holds to, each repairing a specific audit finding:

* **The normal/friction split uses the true contact normal.**
  ``f_n = (f . n) n``, ``f_t = f - f_n``.  Same decomposition Newton's own
  ``SensorContact`` performs (``newton/_src/sensors/sensor_contact.py:90-95``).
  Repairs audit #5.
* **Friction utilization divides by the per-contact friction the solver used**
  (``rigid_contact_friction``, written by the collision pipeline at
  ``newton/_src/sim/collide.py:1067`` and read by MuJoCo as ``friction_scale``
  at ``newton/_src/solvers/mujoco/kernels.py:460``), so it tracks domain
  randomization by construction.  Repairs audit #4.
* **Slip is a displacement, measured against a persistent contact anchor** that
  is carried across frames by ``rigid_contact_match_index``.  There is no
  threshold latch, no evidence counter and no reset mask, so audit #6 and #7 --
  the stuck GROSS latch and the silently swallowed reset that is the best single
  explanation for PS < P -- describe state that no longer exists here.

Units are physical throughout (N, m, m/s, Pa).  There is no channel-scale file:
Plan 15 baked p99.5 max-scaling into the encoder's persistent buffer and hence
into every checkpoint, with nothing binding a scale file to the channel
definitions that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from newton._src.math.spatial import velocity_at_point


# Channel order of the packed per-patch record.  Kept as a module constant so a
# checkpoint can be interpreted without the code that wrote it.
CHANNELS: tuple[str, ...] = (
    "contact_count",
    "normal_load",
    "friction_load",
    "friction_load_abs",
    "utilization_max",
    "utilization_mean",
    "slip_displacement",
    "slip_velocity",
    "gross_slip_fraction",
    "signed_normal_load",
)
NUM_CHANNELS = len(CHANNELS)

MATCH_NOT_FOUND = -1
MATCH_BROKEN = -2


@dataclass
class TactileConfig:
    """Configuration for :class:`PatchTactile`.

    Args:
        fallback_friction: Coefficient used when the contacts buffer carries no
            per-contact friction (``rigid_contact_friction is None``, which is
            the case when the pipeline runs without hydroelastic SDF).  This is
            a fallback, never a substitute: with hydroelastic enabled the real
            per-contact value is always preferred.  Plan 15's bug was making a
            constant like this the *only* path.
        min_normal_load: Contacts carrying less normal load than this [N] are
            excluded from utilization, slip and the load-weighted means.  A
            near-zero denominator makes utilization meaningless and is how a
            grazing contact produces a spurious slip reading.
        max_utilization: Utilization above this is reported through
            :attr:`PatchTactile.utilization_overflow` rather than silently
            clamped.  Under a converged solve Coulomb's condition bounds the
            ratio at 1, so a value materially above 1 is evidence of frame
            contamination -- exactly the defect that went unnoticed for the
            whole of Plan 15.
    """

    fallback_friction: float = 1.0
    min_normal_load: float = 1.0e-4
    max_utilization: float = 1.05


@wp.kernel
def propagate_anchors_kernel(
    num_contacts: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    match_index: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    prev_anchor: wp.array[wp.vec3],
    prev_anchor_valid: wp.array[wp.int32],
    prev_count: wp.array[wp.int32],
    # output
    anchor: wp.array[wp.vec3],
    anchor_valid: wp.array[wp.int32],
    anchor_age: wp.array[wp.int32],
    prev_anchor_age: wp.array[wp.int32],
):
    """Carry each contact's material anchor forward one frame.

    A contact that persists keeps the world position it first touched at; a new
    or broken contact re-anchors to where it is now.  The tangential drift of a
    surviving contact away from its anchor *is* material slip -- it needs no
    threshold and no detector state.
    """
    i = wp.tid()
    if i >= num_contacts[0]:
        return

    # World-space contact midpoint, symmetric in both shapes.  Same definition
    # the matcher itself uses (newton/_src/geometry/contact_match.py).
    b0 = shape_body[contact_shape0[i]]
    b1 = shape_body[contact_shape1[i]]
    w0 = contact_point0[i]
    if b0 >= 0:
        w0 = wp.transform_point(body_q[b0], w0)
    w1 = contact_point1[i]
    if b1 >= 0:
        w1 = wp.transform_point(body_q[b1], w1)
    midpoint = 0.5 * (w0 + w1)

    m = wp.int32(MATCH_NOT_FOUND)
    if match_index:
        m = match_index[i]

    if m >= 0 and m < prev_count[0] and prev_anchor_valid[m] == 1:
        anchor[i] = prev_anchor[m]
        anchor_age[i] = prev_anchor_age[m] + 1
    else:
        anchor[i] = midpoint
        anchor_age[i] = 0
    anchor_valid[i] = 1


@wp.kernel
def reduce_contacts_kernel(
    num_contacts: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_force: wp.array[wp.spatial_vector],
    contact_friction: wp.array[wp.float32],
    anchor: wp.array[wp.vec3],
    anchor_age: wp.array[wp.int32],
    shape_to_patch: wp.array[wp.int32],
    shape_is_counterpart: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    fallback_friction: wp.float32,
    min_normal_load: wp.float32,
    # output
    out_count: wp.array[wp.int32],
    out_normal_vec: wp.array[wp.vec3],
    out_friction_vec: wp.array[wp.vec3],
    out_normal_load: wp.array[wp.float32],
    out_signed_normal: wp.array[wp.float32],
    out_friction_abs: wp.array[wp.float32],
    out_util_max: wp.array[wp.float32],
    out_util_wsum: wp.array[wp.float32],
    out_slip_disp_wsum: wp.array[wp.float32],
    out_slip_vel_wsum: wp.array[wp.float32],
    out_reanchor_count: wp.array[wp.int32],
    out_weight: wp.array[wp.float32],
):
    """Reduce rigid contacts onto their patch rows.

    Parallelizes over contacts; every accumulation is atomic.  A patch is the
    unit -- never a contact point, and never a taxel (Plan 16 §8).
    """
    i = wp.tid()
    if i >= num_contacts[0]:
        return

    s0 = contact_shape0[i]
    s1 = contact_shape1[i]
    if s0 < 0 or s1 < 0:
        return

    p0 = shape_to_patch[s0]
    p1 = shape_to_patch[s1]

    # Identify which side is the patch and orient the force onto it.  Newton's
    # convention: contact_force is the force on shape0's body, and the normal
    # points from shape0 toward shape1.
    row = wp.int32(-1)
    sign = wp.float32(1.0)
    patch_shape = wp.int32(-1)
    other_shape = wp.int32(-1)
    if p0 >= 0 and shape_is_counterpart[s1] == 1:
        row = p0
        sign = 1.0
        patch_shape = s0
        other_shape = s1
    elif p1 >= 0 and shape_is_counterpart[s0] == 1:
        row = p1
        sign = -1.0
        patch_shape = s1
        other_shape = s0
    if row < 0:
        return

    f = sign * wp.spatial_top(contact_force[i])

    n = contact_normal[i]
    len_sq = wp.dot(n, n)
    if wp.abs(len_sq - 1.0) > 1.0e-4:
        if len_sq < 1.0e-12:
            return
        n = wp.normalize(n)

    fn_signed = wp.dot(f, n)
    f_t = f - fn_signed * n
    normal_load = wp.abs(fn_signed)
    friction_mag = wp.length(f_t)

    wp.atomic_add(out_count, row, 1)
    wp.atomic_add(out_normal_vec, row, fn_signed * n)
    wp.atomic_add(out_friction_vec, row, f_t)
    wp.atomic_add(out_normal_load, row, normal_load)
    wp.atomic_add(out_signed_normal, row, fn_signed)
    wp.atomic_add(out_friction_abs, row, friction_mag)
    # A contact that re-anchored this frame is one the matcher could not carry
    # forward.  Under stick that is rare; under gross slip it is every contact,
    # because sliding faster than contact_matching_pos_threshold per step breaks
    # the match by construction.  Without this counter the anchor drift of a
    # fully sliding patch reads exactly zero -- a silent zero of exactly the
    # kind the Plan 15 audit kept finding.
    if anchor_age[i] == 0:
        wp.atomic_add(out_reanchor_count, row, 1)

    # Below the load floor the Coulomb ratio has a meaningless denominator and
    # the anchor drift of a grazing contact is not slip.  Excluded outright
    # rather than clamped.
    if normal_load < min_normal_load:
        return

    mu = fallback_friction
    if contact_friction:
        mu_c = contact_friction[i]
        # 0.0 means "no friction was set" for this contact (contact_data.py:39).
        if mu_c > 0.0:
            mu = mu_c

    utilization = friction_mag / (mu * normal_load)
    wp.atomic_max(out_util_max, row, utilization)
    wp.atomic_add(out_util_wsum, row, utilization * normal_load)

    # --- slip displacement: tangential drift from the persistent anchor ---
    b_patch = shape_body[patch_shape]
    b_other = shape_body[other_shape]
    bp0 = shape_body[contact_shape0[i]]
    bp1 = shape_body[contact_shape1[i]]
    w0 = contact_point0[i]
    if bp0 >= 0:
        w0 = wp.transform_point(body_q[bp0], w0)
    w1 = contact_point1[i]
    if bp1 >= 0:
        w1 = wp.transform_point(body_q[bp1], w1)
    midpoint = 0.5 * (w0 + w1)

    drift = midpoint - anchor[i]
    drift_t = drift - wp.dot(drift, n) * n
    wp.atomic_add(out_slip_disp_wsum, row, wp.length(drift_t) * normal_load)

    # --- slip velocity: relative tangential velocity at the contact point ---
    # Computed from body states, independent of the anchor path, so the two
    # slip estimates can be cross-checked against each other.
    v_rel = wp.vec3(0.0, 0.0, 0.0)
    if b_patch >= 0:
        r = midpoint - wp.transform_point(body_q[b_patch], body_com[b_patch])
        v_rel = v_rel + velocity_at_point(body_qd[b_patch], r)
    if b_other >= 0:
        r = midpoint - wp.transform_point(body_q[b_other], body_com[b_other])
        v_rel = v_rel - velocity_at_point(body_qd[b_other], r)
    v_t = v_rel - wp.dot(v_rel, n) * n
    wp.atomic_add(out_slip_vel_wsum, row, wp.length(v_t) * normal_load)

    wp.atomic_add(out_weight, row, normal_load)


@wp.kernel
def finalize_kernel(
    out_normal_load: wp.array[wp.float32],
    out_friction_vec: wp.array[wp.vec3],
    out_util_wsum: wp.array[wp.float32],
    out_slip_disp_wsum: wp.array[wp.float32],
    out_slip_vel_wsum: wp.array[wp.float32],
    out_reanchor_count: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
    out_weight: wp.array[wp.float32],
    # output
    out_friction_load: wp.array[wp.float32],
    out_util_mean: wp.array[wp.float32],
    out_slip_disp: wp.array[wp.float32],
    out_slip_vel: wp.array[wp.float32],
    out_gross_slip: wp.array[wp.float32],
):
    """Turn load-weighted sums into load-weighted means."""
    row = wp.tid()
    n = out_count[row]
    if n > 0:
        out_gross_slip[row] = wp.float32(out_reanchor_count[row]) / wp.float32(n)
    else:
        out_gross_slip[row] = 0.0
    out_friction_load[row] = wp.length(out_friction_vec[row])
    w = out_weight[row]
    if w > 0.0:
        out_util_mean[row] = out_util_wsum[row] / w
        out_slip_disp[row] = out_slip_disp_wsum[row] / w
        out_slip_vel[row] = out_slip_vel_wsum[row] / w
    else:
        out_util_mean[row] = 0.0
        out_slip_disp[row] = 0.0
        out_slip_vel[row] = 0.0


class PatchTactile:
    """Reduces rigid contacts onto patch rows, once per control step.

    Args:
        model: The finalized :class:`newton.Model`.
        patch_shapes: Shape indices acting as tactile patches, in the row order
            the policy will see.
        counterpart_shapes: Shape indices whose contacts count.  ``None`` means
            every shape counts, which is almost never what you want -- Plan 15
            shipped a reward whose contact sensor matched 87 of 91 bodies.
        config: See :class:`TactileConfig`.

    Call order per control step::

        pipeline.collide(state, contacts)
        solver.step(...)
        solver.update_contacts(contacts, state)   # fills contacts.force
        tactile.update(state, contacts)
    """

    def __init__(
        self,
        model,
        patch_shapes: list[int],
        counterpart_shapes: list[int] | None,
        config: TactileConfig | None = None,
    ):
        self.model = model
        self.config = config or TactileConfig()
        self.patch_shapes = list(patch_shapes)
        self.num_patches = len(self.patch_shapes)
        if self.num_patches == 0:
            raise ValueError("PatchTactile needs at least one patch shape")
        self.device = model.device

        with wp.ScopedDevice(self.device):
            shape_to_patch = [-1] * model.shape_count
            for row, shape in enumerate(self.patch_shapes):
                if not 0 <= shape < model.shape_count:
                    raise IndexError(f"patch shape {shape} out of range")
                if shape_to_patch[shape] != -1:
                    raise ValueError(f"shape {shape} listed twice as a patch")
                shape_to_patch[shape] = row
            self.shape_to_patch = wp.array(shape_to_patch, dtype=wp.int32)

            if counterpart_shapes is None:
                is_counterpart = [1] * model.shape_count
            else:
                is_counterpart = [0] * model.shape_count
                for shape in counterpart_shapes:
                    if not 0 <= shape < model.shape_count:
                        raise IndexError(f"counterpart shape {shape} out of range")
                    is_counterpart[shape] = 1
            self.shape_is_counterpart = wp.array(is_counterpart, dtype=wp.int32)

            n = self.num_patches
            self.count = wp.zeros(n, dtype=wp.int32)
            self.normal_vec = wp.zeros(n, dtype=wp.vec3)
            self.friction_vec = wp.zeros(n, dtype=wp.vec3)
            self.normal_load = wp.zeros(n, dtype=wp.float32)
            self.signed_normal_load = wp.zeros(n, dtype=wp.float32)
            self.friction_load = wp.zeros(n, dtype=wp.float32)
            self.friction_load_abs = wp.zeros(n, dtype=wp.float32)
            self.utilization_max = wp.zeros(n, dtype=wp.float32)
            self.utilization_mean = wp.zeros(n, dtype=wp.float32)
            self.slip_displacement = wp.zeros(n, dtype=wp.float32)
            self.slip_velocity = wp.zeros(n, dtype=wp.float32)
            self.gross_slip_fraction = wp.zeros(n, dtype=wp.float32)
            self._reanchor_count = wp.zeros(n, dtype=wp.int32)
            self._util_wsum = wp.zeros(n, dtype=wp.float32)
            self._slip_disp_wsum = wp.zeros(n, dtype=wp.float32)
            self._slip_vel_wsum = wp.zeros(n, dtype=wp.float32)
            self._weight = wp.zeros(n, dtype=wp.float32)

            self._anchor_capacity = 0
            self._anchor = None
            self._anchor_valid = None
            self._anchor_age = None
            self._prev_anchor = None
            self._prev_anchor_valid = None
            self._prev_anchor_age = None
            self._prev_count = wp.zeros(1, dtype=wp.int32)

        self.match_index_available: bool | None = None
        """Set on the first :meth:`update`.  ``False`` means the pipeline was
        built without ``contact_matching``, so every contact re-anchors every
        frame and slip displacement is identically zero.  Slip velocity is
        unaffected."""

    def _ensure_anchor_capacity(self, capacity: int) -> None:
        if capacity <= self._anchor_capacity:
            return
        with wp.ScopedDevice(self.device):
            self._anchor = wp.zeros(capacity, dtype=wp.vec3)
            self._anchor_valid = wp.zeros(capacity, dtype=wp.int32)
            self._anchor_age = wp.zeros(capacity, dtype=wp.int32)
            self._prev_anchor = wp.zeros(capacity, dtype=wp.vec3)
            self._prev_anchor_valid = wp.zeros(capacity, dtype=wp.int32)
            self._prev_anchor_age = wp.zeros(capacity, dtype=wp.int32)
        self._anchor_capacity = capacity

    def reset(self) -> None:
        """Drop all anchor history.

        Note there is deliberately no per-environment reset mask and no cached
        step counter guarding this call.  Plan 15's equivalent
        (``_online_patch_slip_history``) returned early on a step-counter match
        *before* forwarding the reset to its detector, and ``env.reset()`` does
        not advance that counter -- so between evaluation batches the reset was
        silently swallowed and a GROSS latch survived the episode boundary
        (audit #7).  Here the anchors are the only cross-frame state and this
        method is the only thing that clears them.
        """
        if self._anchor_valid is not None:
            self._anchor_valid.zero_()
            self._prev_anchor_valid.zero_()
            self._anchor_age.zero_()
            self._prev_anchor_age.zero_()
        self._prev_count.zero_()

    def update(self, state, contacts) -> None:
        """Recompute every channel from the current contact buffer."""
        if contacts.force is None:
            raise ValueError(
                "PatchTactile requires contacts with the 'force' attribute. Call "
                "model.request_contact_attributes('force') before allocating "
                "the contacts buffer."
            )

        if self.match_index_available is None:
            self.match_index_available = contacts.rigid_contact_match_index is not None

        capacity = contacts.rigid_contact_max
        self._ensure_anchor_capacity(capacity)

        wp.launch(
            propagate_anchors_kernel,
            dim=capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_match_index,
                self.model.shape_body,
                state.body_q,
                self._prev_anchor,
                self._prev_anchor_valid,
                self._prev_count,
            ],
            outputs=[
                self._anchor,
                self._anchor_valid,
                self._anchor_age,
                self._prev_anchor_age,
            ],
            device=self.device,
        )

        for buf in (
            self.count,
            self.normal_vec,
            self.friction_vec,
            self.normal_load,
            self.signed_normal_load,
            self.friction_load_abs,
            self.utilization_max,
            self._util_wsum,
            self._slip_disp_wsum,
            self._slip_vel_wsum,
            self._reanchor_count,
            self._weight,
        ):
            buf.zero_()

        wp.launch(
            reduce_contacts_kernel,
            dim=capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.force,
                contacts.rigid_contact_friction,
                self._anchor,
                self._anchor_age,
                self.shape_to_patch,
                self.shape_is_counterpart,
                self.model.shape_body,
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.config.fallback_friction,
                self.config.min_normal_load,
            ],
            outputs=[
                self.count,
                self.normal_vec,
                self.friction_vec,
                self.normal_load,
                self.signed_normal_load,
                self.friction_load_abs,
                self.utilization_max,
                self._util_wsum,
                self._slip_disp_wsum,
                self._slip_vel_wsum,
                self._reanchor_count,
                self._weight,
            ],
            device=self.device,
        )

        wp.launch(
            finalize_kernel,
            dim=self.num_patches,
            inputs=[
                self.normal_load,
                self.friction_vec,
                self._util_wsum,
                self._slip_disp_wsum,
                self._slip_vel_wsum,
                self._reanchor_count,
                self.count,
                self._weight,
            ],
            outputs=[
                self.friction_load,
                self.utilization_mean,
                self.slip_displacement,
                self.slip_velocity,
                self.gross_slip_fraction,
            ],
            device=self.device,
        )

        # Anchors become the previous frame's anchors.  Contact indices are
        # stable only because a non-disabled contact_matching mode implies
        # deterministic=True, so match_index refers into this sorted ordering.
        self._anchor, self._prev_anchor = self._prev_anchor, self._anchor
        self._anchor_valid, self._prev_anchor_valid = (
            self._prev_anchor_valid,
            self._anchor_valid,
        )
        self._anchor_age, self._prev_anchor_age = self._prev_anchor_age, self._anchor_age
        wp.copy(self._prev_count, contacts.rigid_contact_count)

    @property
    def utilization_overflow(self) -> list[int]:
        """Patch rows whose utilization exceeds ``config.max_utilization``.

        Under a converged solve Coulomb bounds this ratio at 1.  A value
        materially above 1 is proof of frame contamination by construction --
        it is the signature of the Plan 15 defect, where off-centre contact
        made a static grasp read 0.622 with nothing moving.
        """
        util = self.utilization_max.numpy()
        return [int(i) for i in range(self.num_patches) if util[i] > self.config.max_utilization]

    def to_numpy(self) -> dict:
        """Return every channel as numpy, keyed by :data:`CHANNELS` names."""
        return {
            "contact_count": self.count.numpy(),
            "normal_load": self.normal_load.numpy(),
            "friction_load": self.friction_load.numpy(),
            "friction_load_abs": self.friction_load_abs.numpy(),
            "utilization_max": self.utilization_max.numpy(),
            "utilization_mean": self.utilization_mean.numpy(),
            "slip_displacement": self.slip_displacement.numpy(),
            "slip_velocity": self.slip_velocity.numpy(),
            "gross_slip_fraction": self.gross_slip_fraction.numpy(),
            "signed_normal_load": self.signed_normal_load.numpy(),
        }
