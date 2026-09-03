"""IsaacLab's `ArticulationData` / `RigidObjectData` surface, computed from a Newton state.

SUGAR's MDP terms read the simulator exclusively through `asset.data.<attr>`, so this module
is where the swap actually happens: everything above it (SUGAR's terms, IsaacLab's managers
and math) stays byte-for-byte identical, and everything below it is Newton.

Three conventions differ between the engines and are converted here, once:

======================  ==========================  =============================
quantity                Newton                      IsaacLab (what callers expect)
======================  ==========================  =============================
body orientation        ``body_q[3:7]`` as xyzw     ``root_quat_w`` as wxyz
body velocity           ``body_qd`` as (lin, ang)   separate ``*_lin_vel_w`` / ``*_ang_vel_w``
joint indexing          root free joint + object    actuated joints only
======================  ==========================  =============================

Newton exposes neither joint acceleration nor actuator torque, both of which SUGAR's
regularisation rewards need. They are reconstructed the same way IsaacLab does: acceleration
by finite-differencing joint velocity across the step, and torque from the implicit PD law
that the solver applies internally. Reconstruction rather than measurement means these two
are the least exact part of the swap, which matters because they are only ever consumed as
squared penalties.

Attributes are computed on demand and cached until `update()` invalidates them, so a term
that reads `root_pos_w` five times pays for it once. Anything not implemented raises
`AttributeError` rather than returning zeros: a silently-zero observation would train a
policy that looks fine and is wrong.
"""

from __future__ import annotations

import torch


class _Cached:
    """Descriptor caching a property for the lifetime of one simulation step."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.__doc__ = fn.__doc__

    def __get__(self, obj, _owner=None):
        if obj is None:
            return self
        cache = obj._cache
        if self.name not in cache:
            cache[self.name] = self.fn(obj)
        return cache[self.name]


def _quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a wxyz quaternion."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate `v` by wxyz quaternion `q`, both broadcastable to (..., 3/4)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate `v` into the frame described by wxyz quaternion `q`."""
    return _quat_apply(_quat_conj(q), v)


class RigidObjectData:
    """Root-body state of a single free body, in IsaacLab's layout.

    Reads the owning asset's slice of Newton's `body_q` / `body_qd` arrays. The asset is
    responsible for supplying the body index, so the box and the robot root share this code.
    """

    def __init__(self, asset, device: str):
        self._asset = asset
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}
        self.GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0], device=device).repeat(asset.num_instances, 1)
        self.FORWARD_VEC_B = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(asset.num_instances, 1)
        # Populated by the asset on spawn; SUGAR's reset events read these.
        self.default_root_state: torch.Tensor | None = None
        # IsaacLab's mass randomisation perturbs the *default* every time it runs, so that
        # repeated application does not compound. Both are read off the physx view at build
        # time, exactly as IsaacLab does, which is what keeps their layouts in step.
        self.default_mass: torch.Tensor | None = None
        self.default_inertia: torch.Tensor | None = None

    def update(self, dt: float) -> None:
        self._cache.clear()

    # ---- root pose ---------------------------------------------------------
    @_Cached
    def _body_q(self) -> torch.Tensor:
        return self._asset._root_body_q()

    @_Cached
    def _body_qd(self) -> torch.Tensor:
        return self._asset._root_body_qd()

    @_Cached
    def root_pos_w(self) -> torch.Tensor:
        return self._body_q[:, :3]

    @_Cached
    def root_quat_w(self) -> torch.Tensor:
        return _quat_xyzw_to_wxyz(self._body_q[:, 3:7])

    @_Cached
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._body_qd[:, :3]

    @_Cached
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._body_qd[:, 3:6]

    @_Cached
    def root_lin_vel_b(self) -> torch.Tensor:
        return quat_apply_inverse(self.root_quat_w, self.root_lin_vel_w)

    @_Cached
    def root_ang_vel_b(self) -> torch.Tensor:
        return quat_apply_inverse(self.root_quat_w, self.root_ang_vel_w)

    @_Cached
    def root_vel_w(self) -> torch.Tensor:
        return torch.cat([self.root_lin_vel_w, self.root_ang_vel_w], dim=-1)

    @_Cached
    def root_state_w(self) -> torch.Tensor:
        return torch.cat([self.root_pos_w, self.root_quat_w, self.root_vel_w], dim=-1)

    @_Cached
    def projected_gravity_b(self) -> torch.Tensor:
        return quat_apply_inverse(self.root_quat_w, self.GRAVITY_VEC_W)


class ArticulationData(RigidObjectData):
    """Root, joint and per-body state of an articulation, in IsaacLab's layout."""

    def __init__(self, asset, device: str):
        super().__init__(asset, device)
        self.default_joint_pos: torch.Tensor | None = None
        self.default_joint_vel: torch.Tensor | None = None
        self.soft_joint_pos_limits: torch.Tensor | None = None
        self.joint_pos_limits: torch.Tensor | None = None
        self.joint_stiffness: torch.Tensor | None = None
        self.joint_damping: torch.Tensor | None = None
        self.joint_effort_limits: torch.Tensor | None = None
        self._prev_joint_vel: torch.Tensor | None = None
        self._joint_acc: torch.Tensor | None = None

    def update(self, dt: float) -> None:
        """Advance the finite-differenced quantities, then invalidate the caches.

        Order matters: joint acceleration needs the velocity from before the step, so it is
        computed against the still-valid cache and only then is the cache dropped.
        """
        vel = self.joint_vel
        if self._prev_joint_vel is None:
            self._joint_acc = torch.zeros_like(vel)
        elif dt > 0.0:
            self._joint_acc = (vel - self._prev_joint_vel) / dt
        self._prev_joint_vel = vel.clone()
        self._cache.clear()

    # ---- joint state ------------------------------------------------------
    @_Cached
    def joint_pos(self) -> torch.Tensor:
        return self._asset._joint_pos()

    @_Cached
    def joint_vel(self) -> torch.Tensor:
        return self._asset._joint_vel()

    @property
    def joint_acc(self) -> torch.Tensor:
        """Finite-differenced joint acceleration; zero on the first step after a reset."""
        if self._joint_acc is None:
            self._joint_acc = torch.zeros_like(self.joint_pos)
        return self._joint_acc

    def reset_joint_acc(self, velocity: torch.Tensor, joint_ids=None, env_ids=None) -> None:
        """Drop the finite-difference history for joints whose velocity was written directly.

        A teleport is not a physical acceleration. IsaacLab zeroes `joint_acc` and reseeds
        `_previous_joint_vel` inside `write_joint_state_to_sim` for exactly this reason
        (`articulation.py:600`); differencing across the write instead charges the policy for
        the discontinuity. That path runs on every reset, so omitting this put a large
        spurious `joint_acc` penalty on the first step of every episode -- measured at up to
        688 rad/s^2 against IsaacLab's exact zero by the per-term diff in experiments/equiv.
        """
        ref = self.default_joint_vel if self.default_joint_vel is not None else self.joint_vel
        if self._joint_acc is None:
            self._joint_acc = torch.zeros_like(ref)
        if self._prev_joint_vel is None:
            self._prev_joint_vel = torch.zeros_like(ref)

        if joint_ids is None:
            if env_ids is None:
                self._joint_acc.zero_()
                self._prev_joint_vel.copy_(velocity)
            else:
                self._joint_acc[env_ids] = 0.0
                self._prev_joint_vel[env_ids] = velocity
        else:
            rows = slice(None) if env_ids is None else env_ids[:, None]
            self._joint_acc[rows, joint_ids] = 0.0
            self._prev_joint_vel[rows, joint_ids] = velocity

    @_Cached
    def computed_torque(self) -> torch.Tensor:
        """Torque the implicit PD law asks for, before the effort limit.

        Newton applies this inside the solver and does not report it, so it is recomputed
        from the same gains and targets the solver was given.
        """
        target = self._asset._joint_target()
        return self.joint_stiffness * (target - self.joint_pos) - self.joint_damping * self.joint_vel

    @_Cached
    def applied_torque(self) -> torch.Tensor:
        """`computed_torque` clipped to the effort limit, as the actuator would deliver it."""
        limit = self.joint_effort_limits
        return torch.clamp(self.computed_torque, -limit, limit)

    # ---- per-body state ---------------------------------------------------
    @_Cached
    def _bodies_q(self) -> torch.Tensor:
        return self._asset._bodies_body_q()

    @_Cached
    def _bodies_qd(self) -> torch.Tensor:
        return self._asset._bodies_body_qd()

    @_Cached
    def body_pos_w(self) -> torch.Tensor:
        return self._bodies_q[..., :3]

    @_Cached
    def body_quat_w(self) -> torch.Tensor:
        return _quat_xyzw_to_wxyz(self._bodies_q[..., 3:7])

    @_Cached
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._bodies_qd[..., :3]

    @_Cached
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._bodies_qd[..., 3:6]

    @_Cached
    def body_state_w(self) -> torch.Tensor:
        return torch.cat(
            [self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w], dim=-1
        )
