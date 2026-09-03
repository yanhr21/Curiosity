"""`root_physx_view` stand-in, so IsaacLab's domain-randomisation events run unmodified.

IsaacLab's `randomize_rigid_body_mass`, `randomize_rigid_body_com` and
`randomize_rigid_body_material` reach past the asset API straight into the PhysX tensor view:
they read a property with `get_masses()`, perturb it on the CPU, and push it back with
`set_masses()`. SUGAR uses all three, so rather than reimplement the randomisation logic
(and risk a different distribution) this class presents the same handful of methods over
Newton's model arrays.

TWO correctness traps are handled here, and both have the same failure mode: the
randomisation looks configured, runs without error, and changes nothing.

First, Newton's solver consumes `body_inv_mass` and `body_inv_inertia`, not `body_mass` and
`body_inertia`. Every setter updates the inverse alongside the forward array.

Second, `SolverMuJoCo` does not read `model.*` during a step at all. It converts the Newton
model into its own `mjw_model` once at construction and steps that, so a write to
`model.body_mass` lands in an array nothing is looking at until
`solver.notify_model_changed()` copies it across (`update_body_mass_ipos_kernel` for the
inertial properties). Every setter therefore notifies, which is also why the setters have to
reach the solver rather than just the model.

Shapes follow IsaacLab's convention -- leading dimension is the environment, properties are
flattened per body -- because that is what the event code indexes.
"""

from __future__ import annotations

import torch
import warp as wp

from newton import ModelFlags


def _torch(arr) -> torch.Tensor:
    return wp.to_torch(arr)


class _LinkPhysxView:
    """Shape count for one link, the only field IsaacLab reads off a per-link view."""

    def __init__(self, max_shapes: int):
        self.max_shapes = max_shapes


class PhysicsSimView:
    """`asset._physics_sim_view` stand-in for the material randomisation term.

    `randomize_rigid_body_material` addresses one body's slice of the flat per-shape
    material array, and PhysX only exposes shapes-per-link through a per-link view. Newton
    keeps the shape-to-body map on the model, so here it is a lookup rather than a view.
    """

    def __init__(self, asset, scene):
        self._asset = asset
        self._scene = scene

    def create_rigid_body_view(self, link_path: str) -> _LinkPhysxView:
        return _LinkPhysxView(self._scene.shapes_of_body(self._asset, link_path))


class _PhysxViewBase:
    """Common mass/inertia/COM access for a set of bodies owned by one asset."""

    def __init__(self, asset, scene):
        self._asset = asset
        self._scene = scene
        self._model = scene.model

    def _rows(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        """Global body rows for `env_ids`.

        IsaacLab's randomisation events do their sampling on the CPU and pass CPU index
        tensors back in, so the indices have to be moved before they touch a Warp array.
        """
        rows = self._body_rows
        return rows if env_ids is None else rows[env_ids.to(rows.device)]

    def _values(self, value: torch.Tensor, env_ids: torch.Tensor | None) -> torch.Tensor:
        """Move a setter's payload to the device and take the `env_ids` subset.

        The events pass the whole buffer they read out of `get_*` and expect the setter to
        apply only the selected rows, which is what PhysX's own setters do.
        """
        out = value.to(self._scene.device, dtype=torch.float32)
        if env_ids is not None and out.shape[0] == self.count:
            out = out[env_ids.to(out.device)]
        return out

    @property
    def count(self) -> int:
        return self._scene.num_envs

    @property
    def _body_rows(self) -> torch.Tensor:
        """Global body indices for this asset, shaped (num_envs, bodies_per_asset)."""
        return self._scene.global_body_indices(self._asset)

    def _notify(self, flags: ModelFlags) -> None:
        """Push a `model.*` write through to the solver's own copy.

        Called once per setter rather than batched at the end of an event: IsaacLab's terms
        are independent and any one of them can be the only one SUGAR enables, so the cost
        (a handful of kernel launches, at reset cadence) buys not having to reason about
        which combination ran.
        """
        solver = getattr(self._scene, "solver", None)
        if solver is not None:
            solver.notify_model_changed(flags)

    # ---- mass -------------------------------------------------------------
    def get_masses(self) -> torch.Tensor:
        return _torch(self._model.body_mass)[self._body_rows].cpu()

    def set_masses(self, masses: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        rows = self._rows(env_ids)
        value = self._values(masses, env_ids).reshape(rows.shape)
        _torch(self._model.body_mass)[rows] = value
        inv = torch.where(value > 0.0, 1.0 / value.clamp(min=1e-12), torch.zeros_like(value))
        _torch(self._model.body_inv_mass)[rows] = inv
        self._notify(ModelFlags.BODY_INERTIAL_PROPERTIES)

    # ---- inertia ----------------------------------------------------------
    def get_inertias(self) -> torch.Tensor:
        inertia = _torch(self._model.body_inertia)[self._body_rows]
        return inertia.reshape(*inertia.shape[:2], 9).cpu()

    def set_inertias(self, inertias: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        rows = self._rows(env_ids)
        mat = self._values(inertias, env_ids).reshape(*rows.shape, 3, 3)
        _torch(self._model.body_inertia)[rows] = mat
        _torch(self._model.body_inv_inertia)[rows] = torch.linalg.pinv(mat)
        self._notify(ModelFlags.BODY_INERTIAL_PROPERTIES)

    # ---- centre of mass ---------------------------------------------------
    def get_coms(self) -> torch.Tensor:
        return _torch(self._model.body_com)[self._body_rows].cpu()

    def set_coms(self, coms: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        rows = self._rows(env_ids)
        value = self._values(coms, env_ids).reshape(*rows.shape, -1)
        _torch(self._model.body_com)[rows] = value[..., :3]
        self._notify(ModelFlags.BODY_INERTIAL_PROPERTIES)

    # ---- friction ---------------------------------------------------------
    @property
    def max_shapes(self) -> int:
        return self._scene.shapes_per_asset(self._asset)

    @property
    def link_paths(self) -> list[list[str]]:
        """IsaacLab inspects these only to count shapes per link."""
        return [list(self._asset.body_names) for _ in range(self.count)]

    def get_material_properties(self) -> torch.Tensor:
        """(num_envs, num_shapes, 3) as static friction, dynamic friction, restitution.

        Newton carries a single friction coefficient per shape, so both friction columns
        report it and restitution reports zero. A randomisation that sets the two frictions
        apart therefore collapses to the dynamic one on write.
        """
        rows = self._scene.global_shape_indices(self._asset)
        mu = _torch(self._model.shape_material_mu)[rows]
        out = torch.zeros(*mu.shape, 3, device=mu.device, dtype=mu.dtype)
        out[..., 0] = mu
        out[..., 1] = mu
        return out.cpu()

    def set_material_properties(self, materials: torch.Tensor, env_ids: torch.Tensor | None = None):
        rows = self._scene.global_shape_indices(self._asset)
        rows = rows if env_ids is None else rows[env_ids.to(rows.device)]
        value = self._values(materials, env_ids)
        _torch(self._model.shape_material_mu)[rows] = value[..., 1]
        self._notify(ModelFlags.SHAPE_PROPERTIES)


class RigidBodyPhysxView(_PhysxViewBase):
    """Tensor view over a single free body.

    PhysX's rigid-body view drops the body axis that the articulation view carries, and
    IsaacLab's randomisation terms index accordingly -- `inertias[env_ids]` against a
    `(num_envs, 9)` buffer for a rigid object versus `(num_envs, num_bodies, 9)` for an
    articulation. Keeping that difference means the mass term runs verbatim on both.
    """

    def get_inertias(self) -> torch.Tensor:
        return super().get_inertias().squeeze(1)


class ArticulationPhysxView(_PhysxViewBase):
    """Tensor view over an articulation's links."""

    def get_dof_limits(self) -> torch.Tensor:
        return self._scene.joint_limits(self._asset).cpu()
