# SPDX-License-Identifier: Apache-2.0
"""``SceneSpec`` → Newton ``ModelBuilder`` (dataset-agnostic).

Encodes the one verified asset→Newton recipe from ``example_panda_clock_metal.py``:
convex-hull collision + hydroelastic SDF + full mesh as visual, material from PBR + mass
(``density = mass / hull_volume``), static room boxes, optional robot with compliant tactile
pads. ``newton``/``warp`` are imported lazily so the adapters remain importable without the
``newton`` conda env.

Status: reference implementation — **not yet run in the live ``newton`` env**. The mesh
up-axis, bbox-fit mode, SDF resolution, and sensor wiring are calibration knobs (see
``claude_context/dataset_ingestion.md`` §4–§6).
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from .spec import ObjectSpec, SceneSpec

_ROT_Y_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # +90° about X


def _euler_deg_to_quat_xyzw(ex: float, ey: float, ez: float) -> tuple[float, float, float, float]:
    """Intrinsic Z·Y·X Euler (degrees) → quaternion ``(x, y, z, w)`` (Newton/Warp order)."""
    hx, hy, hz = (math.radians(a) * 0.5 for a in (ex, ey, ez))
    cx, sx, cy, sy, cz, sz = math.cos(hx), math.sin(hx), math.cos(hy), math.sin(hy), math.cos(hz), math.sin(hz)
    return (
        sx * cy * cz - cx * sy * sz,  # x
        cx * sy * cz + sx * cy * sz,  # y
        cx * cy * sz - sx * sy * cz,  # z
        cx * cy * cz + sx * sy * sz,  # w
    )


def load_object_geometry(obj: ObjectSpec, fit: str = "uniform") -> dict | None:
    """Load a mesh, orient to Z-up, fit to ``bbox_dims``, seat its base at the origin.

    Returns visual verts/faces, convex-hull verts/faces, and hull volume [m³] (for density),
    or ``None`` if the mesh is missing/empty. Mirrors ``load_clock_geometry`` but fits the AABB
    to the dataset's physical ``bbox_dims`` and seats the base at z=0 (SAGE objects rest on a
    support surface, so ``position.z`` is the base height — not the centre).

    Args:
        obj: The object spec.
        fit: ``"uniform"`` (scalar scale, preserves shape) or ``"stretch"`` (per-axis to match
            the exact target AABB — matches the dataset's collision-free footprint but distorts).
    """
    try:
        scene = trimesh.load(obj.mesh_path)
    except Exception:
        return None
    mesh = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    if mesh is None or len(mesh.vertices) == 0:
        return None
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if obj.up_axis == "y":
        v = (_ROT_Y_TO_Z @ v.T).T

    cur = v.max(axis=0) - v.min(axis=0)
    cur = np.where(cur < 1e-9, 1.0, cur)
    if obj.bbox_dims is not None:
        tgt = np.asarray(obj.bbox_dims, dtype=np.float64)
        if fit == "stretch":
            v *= tgt / cur
        else:  # uniform: match the largest extent, preserve shape
            v *= float(np.max(tgt / cur)) if np.all(tgt > 0) else 1.0

    # seat base at z=0, centre XY on origin; body xform then places it at obj.position
    mn, mx = v.min(axis=0), v.max(axis=0)
    v[:, 0] -= 0.5 * (mn[0] + mx[0])
    v[:, 1] -= 0.5 * (mn[1] + mx[1])
    v[:, 2] -= mn[2]

    faces = np.asarray(mesh.faces, dtype=np.int32)
    hull = trimesh.Trimesh(vertices=v, faces=faces).convex_hull
    return {
        "viz_v": v.astype(np.float32),
        "viz_f": faces.flatten(),
        "hull_v": np.asarray(hull.vertices, dtype=np.float32),
        "hull_f": np.asarray(hull.faces, dtype=np.int32).flatten(),
        "hull_volume": float(hull.volume),
    }


def load_glb_scene(glb_path: str) -> list[dict]:
    """Load a pre-assembled SAGE ``.glb`` into world-placed, Z-up meshes with authentic baked UVs.

    The dataset ships an assembled GLB (``_out/layout_<id>.glb``) that its *own* reference previews
    are rendered from, so its per-vertex UVs and baked textures are authoritative — we use them
    directly instead of hand-parsing PLY ``texcoord`` elements (which mismap). Each geometry
    instance is placed by its scene-graph transform, then rotated glTF Y-up -> Newton Z-up so the
    floor lands on z=0.

    Returns one dict per mesh: ``name``, ``category`` (``floor``/``wall``/``ceiling``/``furniture``),
    ``verts`` (float32 ``[N,3]``, Z-up world), ``faces`` (int32, flattened tris), ``uvs``
    (float32 ``[N,2]`` per-vertex or ``None``), ``texture`` (uint8 ``[H,W,3]`` RGB or ``None``).
    """
    scene = trimesh.load(glb_path, process=False)  # process=False keeps vertex<->UV correspondence
    geoms = scene.geometry if isinstance(scene, trimesh.Scene) else {"_": scene}
    out: list[dict] = []
    nodes = list(scene.graph.nodes_geometry) if isinstance(scene, trimesh.Scene) else [None]
    for node in nodes:
        if node is not None:
            xf, gname = scene.graph[node]
        else:
            xf, gname = np.eye(4), "_"
        g = geoms[gname]
        v = trimesh.transform_points(np.asarray(g.vertices, dtype=np.float64), xf) @ _ROT_Y_TO_Z.T
        vis = getattr(g, "visual", None)
        uv = getattr(vis, "uv", None)
        uv = np.asarray(uv, dtype=np.float32) if uv is not None and len(uv) == len(g.vertices) else None
        tex = None
        mat = getattr(vis, "material", None)
        img = None
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if img is not None:
            tex = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # SAGE shell meshes are named ``mesh_floor_*`` / ``mesh_wall_*`` (+ a ``*_ceiling``); anchor on
        # the prefix so furniture like ``mesh_room_..._floor_lamp`` isn't misread as floor/wall.
        n = gname.lower()
        if n.endswith("ceiling") or "_ceiling" in n:
            cat = "ceiling"
        elif n.startswith("mesh_floor"):
            cat = "floor"
        elif n.startswith("mesh_wall"):
            cat = "wall"
        else:
            cat = "furniture"
        out.append(
            {
                "name": gname,
                "category": cat,
                "verts": v.astype(np.float32),
                "faces": np.asarray(g.faces, dtype=np.int32).reshape(-1),
                "uvs": uv,
                "texture": tex,
            }
        )
    return out


def build_newton_scene(spec: SceneSpec, *, fit: str = "uniform", touch_sdf: bool = False):
    """Build a Newton ``ModelBuilder`` from a ``SceneSpec``.

    Args:
        spec: The ingested scene.
        fit: Mesh AABB fit mode (see :func:`load_object_geometry`).
        touch_sdf: If True, build a hydroelastic mesh SDF for every object (needed for dense
            tactile pressure); expensive — normally build lazily only for the touched object.

    Returns:
        The populated ``newton.ModelBuilder`` (call ``.finalize()`` then wire sensors — see
        ``tactile_video.py`` / ``context.md`` §4).
    """
    from dataclasses import replace

    import warp as wp

    import newton  # lazy: needs the `newton` conda env

    builder = newton.ModelBuilder()
    base_cfg = newton.ModelBuilder.ShapeConfig(kh=1e11, gap=0.01, mu_torsional=0.0, mu_rolling=0.0)
    builder.default_shape_cfg = base_cfg

    # ---- floor + walls (static, world body) ----
    builder.add_ground_plane()
    for w in spec.room.walls:
        ax, ay = w.start
        bx, by = w.end
        cx, cy = 0.5 * (ax + bx), 0.5 * (ay + by)
        length = math.hypot(bx - ax, by - ay)
        yaw = math.atan2(by - ay, bx - ax)
        q = _euler_deg_to_quat_xyzw(0.0, 0.0, math.degrees(yaw))
        xform = wp.transform((cx, cy, 0.5 * w.height), wp.quat(*q))
        builder.add_shape_box(
            body=-1, xform=xform, hx=0.5 * length, hy=0.5 * w.thickness, hz=0.5 * w.height, cfg=base_cfg
        )
    # doors: optional revolute panel — skip for a static room first (see dataset_ingestion.md §5).

    # ---- objects ----
    for obj in spec.objects:
        geo = load_object_geometry(obj, fit=fit)
        if geo is None:
            continue
        density = None
        if obj.mass is not None and geo["hull_volume"] > 1e-9:
            density = obj.mass / geo["hull_volume"]
        cfg = replace(
            base_cfg,
            kh=obj.material.rigid_kh,
            mu=obj.material.mu,
            restitution=obj.material.restitution,
            density=density if density is not None else (obj.material.density or 500.0),
            is_hydroelastic=touch_sdf,
        )
        q = _euler_deg_to_quat_xyzw(*obj.rotation_euler_deg)
        xform = wp.transform(obj.position, wp.quat(*q))
        hull_mesh = newton.Mesh(geo["hull_v"], geo["hull_f"])
        if obj.is_static:
            builder.add_shape_mesh(body=-1, mesh=hull_mesh, xform=xform, cfg=cfg)
        else:
            b = builder.add_body(xform=xform)
            builder.add_joint_free(child=b)  # dynamic 6-DoF free body
            builder.add_shape_mesh(body=b, mesh=hull_mesh, cfg=cfg)
        # visual full mesh + hydroelastic SDF for touched objects: build lazily at grasp time —
        # mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.01, 0.01), margin=cfg.gap) then
        # set newton.ShapeFlags.HYDROELASTIC (see example_panda_clock_metal.py:153-175).

    # ---- robot (optional) — reuse the panda_hydro/clock recipe for the tactile pads ----
    if spec.robot is not None:
        _attach_robot(builder, spec, base_cfg)

    # non-touch objects → convex hulls for cheap, robust collision
    # builder.approximate_meshes(method="convex_hull", keep_visual_shapes=True)  # after adding visuals
    return builder


def _attach_robot(builder, spec, base_cfg) -> None:
    """Add the robot URDF and mark its fingertip pads compliant + hydroelastic.

    Minimal wiring; the full verified pad/SDF setup (added pad mesh, scale-baked SDF, IK) lives
    in ``example_panda_clock_metal.py`` — fork that for a real tactile run.
    """
    import newton

    r = spec.robot
    builder.add_urdf(
        newton.utils.download_asset(r.asset) / r.urdf_subpath,
        xform=None,  # set to base_position via wp.transform in the real run
        enable_self_collisions=False,
        parse_visuals_as_colliders=True,
    )
    # mark tactile links hydroelastic with a compliant pad kh (broad contact patch) — see clock §153.


if __name__ == "__main__":
    import argparse

    from .adapters.sage import load_sage_scene

    ap = argparse.ArgumentParser(description="Ingest a SAGE scene and (optionally) build a Newton model.")
    ap.add_argument("layout_json")
    ap.add_argument("--build", action="store_true", help="also build the Newton ModelBuilder (needs `newton` env)")
    args = ap.parse_args()
    spec = load_sage_scene(args.layout_json, robot=None)
    print(spec.summary())
    if args.build:
        b = build_newton_scene(spec)
        print("built ModelBuilder:", b)
