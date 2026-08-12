#!/usr/bin/env python3
"""Export the actual IsaacLab anatomical-hand collision meshes for Newton."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import traceback


from isaaclab.app import AppLauncher


ROOT = Path(os.environ.get("CURIOSITY_ROOT", Path(__file__).resolve().parents[3])).resolve()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

os.environ.setdefault(
    "CURIOSITY_TACSL_CALIBRATION_DIR",
    str(ROOT / "experiments/sugar_reproduction/assets/official_tacsl/calibration"),
)
os.environ.setdefault("CURIOSITY_ANATOMICAL_TACSL_CAMERA_APERTURES", "both")
os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(ROOT / "SUGAR/descriptions/terrain/sugar_ground_plane.usda"),
)

simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1 import (  # noqa: E402
    ANATOMICAL_WHOLE_HAND_PATCH_SPECS,
    _surface_frame,
    anatomical_whole_hand_tacsl_robot_cfg,
)
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG  # noqa: E402


def _triangles(mesh: UsdGeom.Mesh) -> np.ndarray:
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += int(count)
        for corner in range(1, int(count) - 1):
            triangles.append((int(face[0]), int(face[corner]), int(face[corner + 1])))
    return np.asarray(triangles, dtype=np.int32)


def _enabled_collision(prim: Usd.Prim) -> bool:
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return False
    enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    return enabled is not False


def _point(matrix: Gf.Matrix4d, value: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    transformed = matrix.Transform(Gf.Vec3d(*(float(item) for item in value)))
    return np.asarray(transformed, dtype=np.float64)


def main() -> None:
    simulation = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args.device)
    )
    base_robot_cfg = UNITREE_G1_29DOF_MIMIC_CFG.replace(
        spawn=UNITREE_G1_29DOF_MIMIC_CFG.spawn.replace(
            asset_path=str(
                ROOT
                / "SUGAR/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
            )
        )
    )
    robot_cfg = anatomical_whole_hand_tacsl_robot_cfg(
        base_robot_cfg,
        "/World/Robot",
    )
    robot = Articulation(cfg=robot_cfg)
    stage = simulation.get_initial_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    patch_names: list[str] = []
    sides: list[str] = []
    sizes: list[tuple[float, float]] = []
    optical: list[bool] = []
    frame_origins: list[np.ndarray] = []
    frame_rotations: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    vertex_offsets = [0]
    triangle_offsets = [0]

    for side in ("left", "right"):
        hand_prim = stage.GetPrimAtPath(f"/World/Robot/{side}_rubber_hand")
        if not hand_prim.IsValid():
            raise RuntimeError(f"Missing spawned {side} rubber-hand body.")
        hand_world_inverse = cache.GetLocalToWorldTransform(hand_prim).GetInverse()
        for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS:
            body_name = f"{side}_anatomical_{spec.name}_elastomer"
            body_prim = stage.GetPrimAtPath(f"/World/Robot/{body_name}")
            if not body_prim.IsValid() or not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing physical anatomical body {body_name}.")
            body_world = cache.GetLocalToWorldTransform(body_prim)
            body_origin = _point(hand_world_inverse, _point(body_world, (0.0, 0.0, 0.0)))
            basis = np.column_stack(
                [
                    _point(hand_world_inverse, _point(body_world, axis)) - body_origin
                    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                ]
            )
            surface_origin, surface_basis = _surface_frame(side, spec)
            if not np.allclose(basis, surface_basis, atol=1.0e-5):
                raise RuntimeError(f"Spawned patch frame disagrees with source frame for {body_name}.")
            collision_prims = [
                prim
                for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies())
                if prim.IsA(UsdGeom.Mesh) and _enabled_collision(prim)
            ]
            if not collision_prims:
                raise RuntimeError(f"No enabled collision mesh below {body_name}.")
            patch_vertices: list[np.ndarray] = []
            patch_triangles: list[np.ndarray] = []
            patch_vertex_count = 0
            for prim in collision_prims:
                mesh = UsdGeom.Mesh(prim)
                mesh_world = cache.GetLocalToWorldTransform(prim)
                points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                local = np.stack(
                    [_point(hand_world_inverse, _point(mesh_world, point)) for point in points]
                ).astype(np.float32)
                patch_vertices.append(local)
                patch_triangles.append(_triangles(mesh) + patch_vertex_count)
                patch_vertex_count += len(local)
            combined_vertices = np.concatenate(patch_vertices)
            combined_triangles = np.concatenate(patch_triangles)
            patch_names.append(spec.name)
            sides.append(side)
            sizes.append((spec.width_m, spec.length_m))
            optical.append(spec.optical_r15)
            frame_origins.append(surface_origin.astype(np.float32))
            frame_rotations.append(surface_basis.astype(np.float32))
            vertices.append(combined_vertices)
            triangles.append(combined_triangles)
            vertex_offsets.append(vertex_offsets[-1] + len(combined_vertices))
            triangle_offsets.append(triangle_offsets[-1] + len(combined_triangles))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        patch_names=np.asarray(patch_names),
        sides=np.asarray(sides),
        patch_size_m=np.asarray(sizes, dtype=np.float32),
        optical_r15=np.asarray(optical, dtype=bool),
        patch_frame_origin_hand_m=np.stack(frame_origins),
        patch_frame_rotation_hand=np.stack(frame_rotations),
        vertices_hand_m=np.concatenate(vertices),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        triangles=np.concatenate(triangles),
        triangle_offsets=np.asarray(triangle_offsets, dtype=np.int64),
        source=np.asarray("IsaacLab anatomical whole-hand spawned enabled collision meshes"),
    )
    print(
        {
            "output": str(output),
            "patches": len(patch_names),
            "vertices": int(vertex_offsets[-1]),
            "triangles": int(triangle_offsets[-1]),
        },
        flush=True,
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
