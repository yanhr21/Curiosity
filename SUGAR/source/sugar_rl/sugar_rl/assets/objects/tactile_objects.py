# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Official SUGAR objects adapted only to expose an IsaacLab SDF collision.

TacSL's official force-field implementation requires the contacted object's
collision mesh to use PhysX's SDF approximation.  The adapter below spawns the
unchanged official SUGAR USD, makes its source prim editable, and applies the
standard IsaacLab SDF schema before the normal scene cloning step.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd
from isaaclab.utils import configclass


@sim_utils.clone
def spawn_from_usd_with_sdf(
    prim_path: str,
    cfg: "SdfUsdFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the official USD and mark each collision mesh as an SDF mesh."""
    # Bypass only the stock cloning decorator.  The complete official USD
    # spawner still performs path validation, reference creation, transforms,
    # material binding, and physical-property overrides.
    root_prim = spawn_from_usd.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    sim_utils.make_uninstanceable(prim_path)

    collision_meshes: list[Usd.Prim] = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        current = prim
        while current and current.IsValid() and current.GetPath().HasPrefix(root_prim.GetPath()):
            if current.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                collision_meshes.append(prim)
                break
            current = current.GetParent()

    if not collision_meshes:
        if not cfg.add_collision_to_mesh_if_absent:
            raise RuntimeError(f"Object has no collision mesh below {prim_path}")
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            collision_meshes.append(prim)
        if not collision_meshes:
            raise RuntimeError(f"Object has no mesh that can become collision below {prim_path}")

    sdf_cfg = sim_utils.SDFMeshPropertiesCfg(
        sdf_margin=cfg.sdf_margin,
        sdf_narrow_band_thickness=cfg.sdf_narrow_band_thickness,
        sdf_resolution=cfg.sdf_resolution,
        sdf_subgrid_resolution=cfg.sdf_subgrid_resolution,
    )
    if cfg.solid_outer_shell_only:
        if len(collision_meshes) != 1:
            raise RuntimeError(
                "Solid-outer SDF conversion requires exactly one source "
                f"collision mesh, found {len(collision_meshes)} below {prim_path}"
            )
        source_prim = collision_meshes[0]
        source_mesh = UsdGeom.Mesh(source_prim)
        points = np.asarray(source_mesh.GetPointsAttr().Get(), dtype=np.float64)
        face_counts = np.asarray(
            source_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64
        )
        face_indices = np.asarray(
            source_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
        )
        if not np.all(face_counts == 3):
            raise RuntimeError(
                "Solid-outer SDF conversion requires a triangular source mesh"
            )
        source_topology = trimesh.Trimesh(
            vertices=points,
            faces=face_indices.reshape(-1, 3),
            process=False,
        )
        components = source_topology.split(only_watertight=False)
        positive = [
            component
            for component in components
            if component.is_watertight
            and component.is_winding_consistent
            and float(component.volume) > 0.0
        ]
        if len(positive) != 1:
            raise RuntimeError(
                "Expected exactly one watertight positive-volume exterior "
                f"component, found {len(positive)} of {len(components)}"
            )
        outer = positive[0]
        outer_bounds = np.asarray(outer.bounds, dtype=np.float64)
        for component in components:
            if component is outer:
                continue
            component_bounds = np.asarray(component.bounds, dtype=np.float64)
            if np.any(component_bounds[0] < outer_bounds[0]) or np.any(
                component_bounds[1] > outer_bounds[1]
            ):
                raise RuntimeError(
                    "Non-exterior component is not enclosed by the selected "
                    "positive-volume CarryBox shell"
                )

        # The official scanned CarryBox contains a positive-winding exterior
        # shell and a negative-winding interior shell. Cooking both as one
        # PhysX SDF makes the rigid object hollow: sufficiently compressed
        # taxels cross the wall midpoint and receive the opposite inner-shell
        # gradient. The original SUGAR convex-decomposition collision behaves
        # as a solid exterior object. Preserve the complete immutable mesh for
        # rendering, disable only its old collision API, and author one exact
        # connected component copied from that same official exterior surface
        # as the sole physical/TacSL SDF shape.
        source_collision = (
            UsdPhysics.CollisionAPI(source_prim)
            if source_prim.HasAPI(UsdPhysics.CollisionAPI)
            else UsdPhysics.CollisionAPI.Apply(source_prim)
        )
        source_collision.CreateCollisionEnabledAttr().Set(False)
        outer_path = (
            source_prim.GetParent().GetPath().AppendChild(
                "tacsl_solid_outer_sdf"
            )
        )
        outer_mesh = UsdGeom.Mesh.Define(root_prim.GetStage(), outer_path)
        outer_mesh.CreatePointsAttr(
            [tuple(float(value) for value in point) for point in outer.vertices]
        )
        outer_mesh.CreateFaceVertexCountsAttr(
            [3] * int(len(outer.faces))
        )
        outer_mesh.CreateFaceVertexIndicesAttr(
            np.asarray(outer.faces, dtype=np.int64).reshape(-1).tolist()
        )
        outer_mesh.CreateSubdivisionSchemeAttr().Set("none")
        outer_mesh.CreateOrientationAttr().Set(
            source_mesh.GetOrientationAttr().Get() or "rightHanded"
        )
        UsdGeom.Imageable(outer_mesh.GetPrim()).MakeInvisible()
        outer_prim = outer_mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(outer_prim).CreateCollisionEnabledAttr(
            True
        )
        outer_api = UsdPhysics.MeshCollisionAPI.Apply(outer_prim)
        outer_api.CreateApproximationAttr().Set("sdf")
        sim_utils.define_mesh_collision_properties(
            outer_path.pathString, sdf_cfg
        )
        for name, value in (
            (
                "curiosity:schema",
                "sugar_official_carrybox_solid_outer_sdf_v1",
            ),
            (
                "curiosity:sourceMesh",
                source_prim.GetPath().pathString,
            ),
        ):
            outer_prim.CreateAttribute(
                name, Sdf.ValueTypeNames.String
            ).Set(value)
        outer_prim.CreateAttribute(
            "curiosity:sourceComponentCount", Sdf.ValueTypeNames.Int
        ).Set(int(len(components)))
        outer_prim.CreateAttribute(
            "curiosity:solidOuterOnly", Sdf.ValueTypeNames.Bool
        ).Set(True)
        outer_prim.CreateAttribute(
            "curiosity:sdfResolution", Sdf.ValueTypeNames.Int
        ).Set(int(cfg.sdf_resolution))
        outer_prim.CreateAttribute(
            "curiosity:sdfSubgridResolution", Sdf.ValueTypeNames.Int
        ).Set(int(cfg.sdf_subgrid_resolution))
        collision_meshes = [outer_prim]

    for mesh_prim in collision_meshes:
        if not mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        else:
            mesh_api = UsdPhysics.MeshCollisionAPI(mesh_prim)
        mesh_api.CreateApproximationAttr().Set("sdf")
        sim_utils.define_mesh_collision_properties(mesh_prim.GetPath().pathString, sdf_cfg)

    return root_prim


@configclass
class SdfUsdFileCfg(UsdFileCfg):
    """USD spawner configuration with standard PhysX SDF cooking controls."""

    func = spawn_from_usd_with_sdf
    sdf_margin: float = 0.01
    sdf_narrow_band_thickness: float = 0.01
    sdf_resolution: int = 128
    sdf_subgrid_resolution: int = 6
    solid_outer_shell_only: bool = False
    add_collision_to_mesh_if_absent: bool = False


SMALLBOX_SDF_CFG = RigidObjectCfg(
    spawn=SdfUsdFileCfg(
        usd_path="descriptions/objects/small_box/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        solid_outer_shell_only=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)
"""The official SUGAR CarryBox object with only its collision approximation changed to SDF."""


BOTTLE_SDF_CFG = RigidObjectCfg(
    spawn=SdfUsdFileCfg(
        usd_path="descriptions/objects/bottle/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            angular_damping=0.2,
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.75),
        solid_outer_shell_only=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)
"""The official SUGAR PickBottle object with only its collision approximation changed to SDF."""
