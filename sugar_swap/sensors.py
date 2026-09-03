"""`ContactSensor` over Newton's contact buffer.

SUGAR leans on this harder than on any other sensor: `force_matrix_w_history` appears in
sixteen places (hand/foot/object contact rewards and terminations) and `last_air_time` drives
the gait term whose absence was the leading suspect for the refiner refusing to stand.

Newton reports contacts as a flat, per-step list of shape pairs with forces. IsaacLab reports
them already reduced per body, and optionally per (body, filter-body) pair. This class does
that reduction, entirely on the GPU.

Staying on the GPU is the whole design constraint. The tactile visualiser reads the same
buffer with `.numpy()`, which is fine for one frame of video but would serialise the device
on every step of training. Here the contact count is compared against an `arange` on-device
so nothing is copied back, and the reduction is a `scatter_add`.

Two limits are worth stating plainly:

* Forces are summed per body, so a body touching two things reports the vector sum, matching
  IsaacLab's `net_forces_w`.
* `force_matrix_w` needs contacts attributed to a *pair* of bodies. Newton gives shape pairs,
  which map to bodies exactly, so this is faithful -- but only for shapes that belong to a
  filtered body, and self-contacts are excluded the same way IsaacLab excludes them.
"""

from __future__ import annotations

from typing import Any

import torch

from .lenient import LenientCfg


class ContactSensorCfg(LenientCfg):
    """Consumed: history length, air-time tracking, and the filter expressions."""

    prim_path: str = ""
    history_length: int = 0
    track_air_time: bool = False
    track_pose: bool = False
    filter_prim_paths_expr: list[str] | None = None
    update_period: float = 0.0
    force_threshold: float = 1.0
    debug_vis: bool = False


class ContactSensorData:
    """Container mirroring IsaacLab's `ContactSensorData` field names."""

    def __init__(self):
        self.net_forces_w: torch.Tensor | None = None
        self.net_forces_w_history: torch.Tensor | None = None
        self.force_matrix_w: torch.Tensor | None = None
        self.force_matrix_w_history: torch.Tensor | None = None
        self.last_air_time: torch.Tensor | None = None
        self.current_air_time: torch.Tensor | None = None
        self.last_contact_time: torch.Tensor | None = None
        self.current_contact_time: torch.Tensor | None = None
        self.pos_w: torch.Tensor | None = None
        self.quat_w: torch.Tensor | None = None


class ContactSensor:
    """Per-body contact forces reduced from Newton's contact list.

    Constructed by the scene, which resolves `prim_path` and `filter_prim_paths_expr` into
    body indices and hands over the shape-to-body mapping.
    """

    def __init__(
        self,
        cfg: ContactSensorCfg,
        scene: Any,
        name: str,
        body_indices: torch.Tensor,
        body_names: list[str],
        filter_body_indices: torch.Tensor | None = None,
    ):
        self.cfg = cfg
        self._scene = scene
        self.name = name
        self.device = scene.device
        self.body_names = body_names
        self._body_indices = body_indices
        self._filter_indices = filter_body_indices

        n_env = scene.num_envs
        n_body = len(body_names)
        # Both index maps arrive as (num_envs, n) global indices, so the per-environment
        # count is the trailing dimension -- `numel()` would count every environment's copy.
        n_filter = 0 if filter_body_indices is None else int(
            filter_body_indices.reshape(n_env, -1).shape[-1]
        )
        self.num_bodies = n_body
        self._n_filter = n_filter
        # IsaacLab's history includes the current step, so a length of 0 still stores one.
        self._history = max(int(cfg.history_length), 0) + 1

        dev = self.device
        self.data = ContactSensorData()
        self.data.net_forces_w = torch.zeros(n_env, n_body, 3, device=dev)
        self.data.net_forces_w_history = torch.zeros(n_env, self._history, n_body, 3, device=dev)
        if n_filter:
            self.data.force_matrix_w = torch.zeros(n_env, n_body, n_filter, 3, device=dev)
            self.data.force_matrix_w_history = torch.zeros(
                n_env, self._history, n_body, n_filter, 3, device=dev
            )
        if cfg.track_air_time:
            self.data.last_air_time = torch.zeros(n_env, n_body, device=dev)
            self.data.current_air_time = torch.zeros(n_env, n_body, device=dev)
            self.data.last_contact_time = torch.zeros(n_env, n_body, device=dev)
            self.data.current_contact_time = torch.zeros(n_env, n_body, device=dev)

        # Reverse lookup from a global body index to this sensor's slot, so the scatter can
        # be done with a single gather instead of a search per contact. One row longer than
        # the body count: `contact_digest` reports static geometry and invalid contact rows
        # as `total_bodies`, and that row must read back as "not mine".
        self._body_slot = torch.full((scene.total_bodies + 1,), -1, dtype=torch.long, device=dev)
        self._body_slot[body_indices.reshape(-1)] = (
            torch.arange(n_body, device=dev).repeat(scene.num_envs)
        )
        if n_filter:
            self._filter_slot = torch.full(
                (scene.total_bodies + 1,), -1, dtype=torch.long, device=dev
            )
            self._filter_slot[filter_body_indices.reshape(-1)] = (
                torch.arange(n_filter, device=dev).repeat(scene.num_envs)
            )
        # Which environment a global body index belongs to. The trailing sentinel is never
        # accumulated, since its slot is -1, so the environment it names does not matter.
        self._body_env = torch.zeros(scene.total_bodies + 1, dtype=torch.long, device=dev)
        self._body_env[: scene.total_bodies] = torch.arange(
            scene.num_envs, device=dev
        ).repeat_interleave(scene.total_bodies // scene.num_envs)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self.data.net_forces_w[idx] = 0.0
        self.data.net_forces_w_history[idx] = 0.0
        if self._n_filter:
            self.data.force_matrix_w[idx] = 0.0
            self.data.force_matrix_w_history[idx] = 0.0
        if self.cfg.track_air_time:
            self.data.current_air_time[idx] = 0.0
            self.data.current_contact_time[idx] = 0.0
            self.data.last_air_time[idx] = 0.0
            self.data.last_contact_time[idx] = 0.0

    def update(self, dt: float, force_recompute: bool = False) -> None:
        if self._scene.contacts is None:
            return

        body0, body1, forces, net_per_body = self._scene.contact_digest()

        # The shared reduction already summed every contact onto its body, so this sensor's
        # net force is a gather of its own rows.
        self.data.net_forces_w = net_per_body[self._body_indices]

        if self._n_filter:
            self._update_force_matrix(body0, body1, forces)

        self._push_history()
        if self.cfg.track_air_time:
            self._update_air_time(dt)

    def _update_force_matrix(self, body0, body1, forces) -> None:
        """Attribute each contact to a (sensor body, filter body) pair.

        Pairs that do not involve both a sensor body and a filter body land on a sink row
        that is then dropped, which keeps the scatter free of the data-dependent boolean
        masking that would sync the device.
        """
        n_body, n_filter = self.num_bodies, self._n_filter
        sink = self._scene.num_envs * n_body * n_filter
        matrix = torch.zeros(sink + 1, 3, device=self.device)
        for a, b, sign in ((body0, body1, 1.0), (body1, body0, -1.0)):
            slot = self._body_slot[a]
            fslot = self._filter_slot[b]
            flat = torch.where(
                (slot >= 0) & (fslot >= 0),
                (self._body_env[a] * n_body + slot) * n_filter + fslot,
                sink,
            )
            matrix.index_add_(0, flat, sign * forces)
        self.data.force_matrix_w = matrix[:sink].reshape(
            self._scene.num_envs, n_body, n_filter, 3
        )

    def _push_history(self) -> None:
        """Shift the ring buffer so index 0 is the current step, as IsaacLab does."""
        hist = self.data.net_forces_w_history
        if self._history > 1:
            hist[:, 1:] = hist[:, :-1].clone()
        hist[:, 0] = self.data.net_forces_w
        if self._n_filter:
            fhist = self.data.force_matrix_w_history
            if self._history > 1:
                fhist[:, 1:] = fhist[:, :-1].clone()
            fhist[:, 0] = self.data.force_matrix_w

    def _update_air_time(self, dt: float) -> None:
        in_contact = self.data.net_forces_w.norm(dim=-1) > self.cfg.force_threshold
        landed = in_contact & (self.data.current_air_time > 0.0)
        took_off = (~in_contact) & (self.data.current_contact_time > 0.0)
        self.data.last_air_time = torch.where(
            landed, self.data.current_air_time + dt, self.data.last_air_time
        )
        self.data.last_contact_time = torch.where(
            took_off, self.data.current_contact_time + dt, self.data.last_contact_time
        )
        self.data.current_air_time = torch.where(
            in_contact, torch.zeros_like(self.data.current_air_time), self.data.current_air_time + dt
        )
        self.data.current_contact_time = torch.where(
            in_contact,
            self.data.current_contact_time + dt,
            torch.zeros_like(self.data.current_contact_time),
        )

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Bodies that established contact within the last `dt` seconds.

        SUGAR's `feet_air_time_min_penalty` reads this every step, so the comparison is
        IsaacLab's: in contact, and in contact for no longer than one policy step.
        """
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "sugar_swap: contact sensor is not configured to track contact time; "
                "enable 'track_air_time' in the sensor configuration."
            )
        current = self.data.current_contact_time
        return (current > 0.0) * (current < (dt + abs_tol))

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Bodies that broke contact within the last `dt` seconds."""
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "sugar_swap: contact sensor is not configured to track contact time; "
                "enable 'track_air_time' in the sensor configuration."
            )
        current = self.data.current_air_time
        return (current > 0.0) * (current < (dt + abs_tol))

    def find_bodies(self, name_keys, preserve_order: bool = False):
        from isaaclab.utils.string import resolve_matching_names

        if isinstance(name_keys, str):
            name_keys = [name_keys]
        return resolve_matching_names(list(name_keys), list(self.body_names), preserve_order)


def _unimplemented(name: str) -> type:
    """A sensor class that resolves as a name but refuses to be instantiated.

    IsaacLab's action and observation term packages import every sensor type at module
    scope, so these names must exist for the terms SUGAR does use to be importable. Making
    construction raise keeps an accidental dependency visible instead of silently inert.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            f"sugar_swap: {name} is not implemented on the Newton backend."
        )

    return type(name, (), {"__init__": __init__, "__doc__": f"Unimplemented: {name}."})


def build():
    """Construct the `isaaclab.sensors` shadow module."""
    import types

    mod = types.ModuleType("isaaclab.sensors")
    mod.__path__ = []
    mod.ContactSensor = ContactSensor
    mod.ContactSensorCfg = ContactSensorCfg
    mod.ContactSensorData = ContactSensorData
    mod.SensorBase = ContactSensor
    mod.SensorBaseCfg = ContactSensorCfg

    for name in (
        "FrameTransformer",
        "RayCaster",
        "RayCasterCamera",
        "Camera",
        "TiledCamera",
        "Imu",
    ):
        setattr(mod, name, _unimplemented(name))
        setattr(mod, f"{name}Cfg", type(f"{name}Cfg", (LenientCfg,), {}))
    return {"isaaclab.sensors": mod}
