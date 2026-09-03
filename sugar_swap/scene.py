"""`InteractiveScene` over a single Newton model shared by every environment.

IsaacLab gives each environment its own set of USD prims and lets PhysX batch them. Newton
instead builds one `Model` containing every environment's bodies, so the scene's job is
bookkeeping: it holds the Newton `Model`/`State`/`Control`, exposes them as torch views
shaped `(num_envs, ...)`, and records which rows belong to which asset.

Every accessor below is part of the contract that `assets.py`, `sensors.py` and
`physx_view.py` are written against. Keeping the index maps here rather than in the assets is
what allows one Newton model, and therefore one vectorised solver step, for the whole batch.

Model construction is delegated to the builder in `sugar_newton.rl.carrybox_env`, which
already assembles the G1 and the box with the collider settings this project settled on
(convex hulls for speed, zero contact margin, corrected box mass). Reusing it keeps the
physics identical to the configuration that was benchmarked, and confines the swap to the
translation layer above it.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from .lenient import LenientCfg


class InteractiveSceneCfg(LenientCfg):
    """Consumed: `num_envs`. Spacing is irrelevant because Newton keeps envs co-located."""

    num_envs: int = 1
    env_spacing: float = 0.0
    replicate_physics: bool = True
    lazy_sensor_update: bool = True


class InteractiveScene:
    """Owns the Newton model and the asset/sensor index maps.

    Assets are created by `build_scene`, which resolves SUGAR's config entries against the
    bodies and joints the Newton builder produced.
    """

    def __init__(self, cfg: InteractiveSceneCfg, device: str):
        self.cfg = cfg
        self.device = device
        self.num_envs = int(cfg.num_envs)
        self.all_envs = torch.arange(self.num_envs, device=device)

        self.articulations: dict[str, Any] = {}
        self.rigid_objects: dict[str, Any] = {}
        self.sensors: dict[str, Any] = {}
        self.extras: dict[str, Any] = {}

        # Set by the builder.
        self.model = None
        self.control = None
        self.contacts = None
        self.solver = None
        self.pipeline = None
        self.physics_dt = 1.0 / 200.0
        # Newton's solver writes into a second state and the two are swapped, so `state`
        # tracks whichever buffer currently holds the live values.
        self.state_0 = None
        self.state_1 = None
        self.shape_body: torch.Tensor | None = None
        self.total_bodies = 0
        self._bodies_per_env = 0
        self._shapes_per_env = 0
        self._coord_to_dof: dict[int, int] = {}
        self._kinematics_dirty = False
        # Set by `builder.enable_cuda_graph`: one captured substep per state-buffer parity.
        self._physics_graphs: list | None = None
        self._graph_parity = 0
        self._env_origins = torch.zeros(self.num_envs, 3, device=device)
        self._digest: tuple | None = None

    # ---- IsaacLab scene protocol ------------------------------------------
    def __getitem__(self, key: str) -> Any:
        for table in (self.articulations, self.rigid_objects, self.sensors, self.extras):
            if key in table:
                return table[key]
        raise KeyError(
            f"sugar_swap: no scene entity named {key!r}; "
            f"have {sorted(self.keys())}"
        )

    def keys(self) -> list[str]:
        return [
            *self.articulations,
            *self.rigid_objects,
            *self.sensors,
            *self.extras,
        ]

    @property
    def env_origins(self) -> torch.Tensor:
        """Newton co-locates environments, so every origin is the world origin.

        SUGAR adds this to reference positions when building observations. Because all
        environments share the origin here, that addition is a no-op -- which is correct for
        this backend but would be wrong if environments were ever offset in a grid.
        """
        return self._env_origins

    @property
    def state(self):
        """The buffer holding the current values, after any number of swaps."""
        return self.state_0

    def swap_states(self) -> None:
        self.state_0, self.state_1 = self.state_1, self.state_0

    # ---- Newton state as torch views --------------------------------------
    def body_q(self) -> torch.Tensor:
        return wp.to_torch(self.state.body_q).view(self.num_envs, self._bodies_per_env, 7)

    def body_qd(self) -> torch.Tensor:
        return wp.to_torch(self.state.body_qd).view(self.num_envs, self._bodies_per_env, 6)

    def joint_q(self) -> torch.Tensor:
        return wp.to_torch(self.state.joint_q).view(self.num_envs, -1)

    def joint_qd(self) -> torch.Tensor:
        return wp.to_torch(self.state.joint_qd).view(self.num_envs, -1)

    def joint_target(self) -> torch.Tensor:
        return wp.to_torch(self.control.joint_target_q).view(self.num_envs, -1)

    # ---- index maps -------------------------------------------------------
    def joint_dof_of_coord(self, coord: int) -> int:
        """Velocity-array offset for the joint whose position starts at `coord`.

        Free joints occupy seven coordinates but six degrees of freedom, so position and
        velocity indices diverge after the first floating base.
        """
        return self._coord_to_dof[coord]

    def global_body_indices(self, asset: Any) -> torch.Tensor:
        """(num_envs, bodies_per_asset) indices into Newton's flat body arrays."""
        local = getattr(asset, "_body_indices", None)
        if local is None:
            local = torch.tensor([asset._body_index], device=self.device)
        local = local.reshape(-1)
        offsets = (self.all_envs * self._bodies_per_env).unsqueeze(-1)
        return offsets + local.unsqueeze(0)

    def global_shape_indices(self, asset: Any) -> torch.Tensor:
        """(num_envs, shapes_per_asset) indices into Newton's flat shape arrays."""
        local = self._asset_shapes[asset.name]
        offsets = (self.all_envs * self._shapes_per_env).unsqueeze(-1)
        return offsets + local.unsqueeze(0)

    def shapes_per_asset(self, asset: Any) -> int:
        return int(self._asset_shapes[asset.name].numel())

    def shapes_of_body(self, asset: Any, body_name: str) -> int:
        """Collision shapes attached to one named body, counted in a single environment.

        Environment 0's global body indices coincide with the local ones, so the head of
        the tiled shape-to-body map is the per-environment map.
        """
        local_map = self.shape_body[: self._shapes_per_env]
        slot = list(asset.body_names).index(body_name)
        target = getattr(asset, "_body_indices", None)
        target = asset._body_index if target is None else int(target.reshape(-1)[slot])
        return int((local_map == target).sum())

    def joint_limits(self, asset: Any) -> torch.Tensor:
        return self._joint_limits[asset.name]

    # ---- contacts ---------------------------------------------------------
    def invalidate_contacts(self) -> None:
        """Drop the cached reduction, because the solver replaced the contact set."""
        self._digest = None

    def contact_digest(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contact list resolved to bodies, plus the net force per body.

        Eight sensors read the same contact buffer every substep. Reducing it once here and
        letting each of them gather its own rows replaces eight scans with one; the per-body
        net force is what `net_forces_w` needs and is identical for every sensor, since each
        only ever restricts the same sum to its own bodies.

        The list is truncated to `rigid_contact_count` first. That costs one device-to-host
        sync per substep, but the allocated capacity exceeds the live contact count by
        orders of magnitude at 512 environments, and the alternative -- masking on-device
        and scanning the whole capacity once per sensor -- measured an order of magnitude
        slower than the physics solve it feeds.

        The returned body indices are in `[0, total_bodies]`, where the extra value marks
        "no body": either a culled pair, or a shape belonging to static geometry, which
        Newton labels with body `-1` and which no sensor can own.
        """
        if self._digest is not None:
            return self._digest

        contacts = self.contacts
        n = int(wp.to_torch(contacts.rigid_contact_count)[0])
        none = self.total_bodies
        if n == 0:
            empty_i = torch.zeros(0, dtype=torch.long, device=self.device)
            empty_f = torch.zeros(0, 3, device=self.device)
            self._digest = (empty_i, empty_i, empty_f, torch.zeros(none, 3, device=self.device))
            return self._digest

        forces = wp.to_torch(contacts.force)[:n, :3]
        shape0 = wp.to_torch(contacts.rigid_contact_shape0)[:n].long()
        shape1 = wp.to_torch(contacts.rigid_contact_shape1)[:n].long()

        n_shape = self.shape_body.numel()
        live = (shape0 >= 0) & (shape1 >= 0) & (shape0 < n_shape) & (shape1 < n_shape)
        forces = forces * live.unsqueeze(-1)
        # Clamped so the gather cannot read out of bounds; such rows carry zero force and
        # are routed to the sentinel, so the clamp target is never accumulated anywhere.
        raw0 = self.shape_body[shape0.clamp(0, n_shape - 1)]
        raw1 = self.shape_body[shape1.clamp(0, n_shape - 1)]
        # Each side is judged on its own, because a foot-on-ground contact has a real body
        # on one side only and its force still has to reach that body.
        body0 = torch.where(live & (raw0 >= 0), raw0, none)
        body1 = torch.where(live & (raw1 >= 0), raw1, none)

        net = torch.zeros(none + 1, 3, device=self.device)
        # Newton reports one force per pair, acting on shape0's body and reacting on shape1's.
        net.index_add_(0, body0, forces)
        net.index_add_(0, body1, -forces)

        self._digest = (body0, body1, forces, net[:none])
        return self._digest

    # ---- stepping ---------------------------------------------------------
    def mark_kinematics_dirty(self) -> None:
        """Record that joint coordinates were written and forward kinematics must re-run."""
        self._kinematics_dirty = True

    def flush_kinematics(self) -> None:
        """Recompute body transforms from joint coordinates, if a reset wrote them."""
        if not self._kinematics_dirty:
            return
        import newton

        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self._kinematics_dirty = False

    def update(self, dt: float) -> None:
        for asset in (*self.articulations.values(), *self.rigid_objects.values()):
            asset.update(dt)
        for sensor in self.sensors.values():
            sensor.update(dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        for asset in (*self.articulations.values(), *self.rigid_objects.values()):
            asset.reset(env_ids)
        for sensor in self.sensors.values():
            sensor.reset(env_ids)

    def write_data_to_sim(self) -> None:
        for asset in (*self.articulations.values(), *self.rigid_objects.values()):
            asset.write_data_to_sim()


def build():
    """Construct the `isaaclab.scene` shadow module."""
    import types

    mod = types.ModuleType("isaaclab.scene")
    mod.InteractiveScene = InteractiveScene
    mod.InteractiveSceneCfg = InteractiveSceneCfg
    return {"isaaclab.scene": mod}
