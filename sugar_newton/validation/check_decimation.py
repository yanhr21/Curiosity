"""What does a cheaper box collider actually cost in geometric accuracy?

Collision is 94.5 % of the step at 16 worlds and its cost is driven by triangle pairs, so
the collider's triangle count is the lever. The question is whether the 100k triangles carry
shape or tessellation. Evidence that it is tessellation
(:mod:`sugar_newton.validation.check_geometry`): the mesh has 100k triangles of median
7.8 mm^2 tiling 1.2264 m^2, and 1.2264 m^2 is within 0.4 % of the exact surface area of an
open carton with this bounding box -- so the macro-geometry is five thin panels.

This measures the claim instead of asserting it. For each target triangle count it reports
the symmetric surface deviation against the original mesh, read against the 5 mm contact
margin (``default_shape_cfg.margin``) and the 3.2 mm wall thickness.

Surface deviation is a necessary check, not a sufficient one, and it is worth being explicit
about what it cannot see. A collider whose surface is within 1.6 mm of the original still
produces a measurably different contact set, because retessellation changes which triangles
generate contacts and how the solver splits the load between them:
:mod:`sugar_newton.validation.compare_contacts` measures that directly and finds the net
load per hand preserved to ~3 % while the per-contact force distribution moves 18 % and the
patch centroid moves up to 10 mm. Use this script for "is the shape still the shape" and that
one for "would a tactile channel notice".

Decimation is not convex hulling and not decomposition: it keeps the mesh non-convex and
keeps the open top, so the concavity the grasp relies on survives.
"""

from __future__ import annotations

import time

import numpy as np

from sugar_newton.validation.g1_carrybox_policy import load_box_mesh

MARGIN_M = 0.005          # ModelBuilder default_shape_cfg.margin used by the scene


def _o3d_mesh(verts: np.ndarray, tris: np.ndarray):
    import open3d as o3d

    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32)))


def deviation(ref, other, n_samples: int = 200_000) -> tuple[float, float, float]:
    """Symmetric surface deviation between two meshes, in metres."""
    import open3d as o3d

    out = []
    for a, b in ((ref, other), (other, ref)):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(b))
        pts = np.asarray(a.sample_points_uniformly(n_samples).points, dtype=np.float32)
        out.append(scene.compute_distance(o3d.core.Tensor(pts)).numpy())
    d = np.concatenate(out)
    return float(d.mean()), float(np.percentile(d, 99.0)), float(d.max())


def hand_collision_meshes() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The two rubber-hand collision meshes, straight out of the URDF."""
    import newton

    from sugar_newton.validation.g1_carrybox_policy import URDF

    b = newton.ModelBuilder()
    b.add_urdf(str(URDF), floating=True, collapse_fixed_joints=False,
               enable_self_collisions=False, joint_ordering="bfs")
    hands = {i: lbl.split("/")[-1] for i, lbl in enumerate(b.body_label)
             if lbl.split("/")[-1] in ("left_rubber_hand", "right_rubber_hand")}
    out = []
    for sh in range(b.shape_count):
        if b.shape_body[sh] not in hands:
            continue
        if not (b.shape_flags[sh] & int(newton.ShapeFlags.COLLIDE_SHAPES)):
            continue
        src = b.shape_source[sh]
        if src is None or getattr(src, "indices", None) is None:
            continue
        out.append((hands[b.shape_body[sh]],
                    np.asarray(src.vertices, dtype=np.float64),
                    np.asarray(src.indices).reshape(-1, 3)))
    return out


def sweep(name: str, verts: np.ndarray, tris: np.ndarray, targets) -> None:
    ref = _o3d_mesh(verts, tris)
    v_ref = ref.get_volume() if ref.is_watertight() else float("nan")
    print(f"{name}: {len(tris)} triangles, watertight={ref.is_watertight()}")
    print(f"{'target':>8} {'actual':>8} {'build_s':>8} {'mean_mm':>9} {'p99_mm':>8} "
          f"{'max_mm':>8} {'vol_err':>9}")

    for target in targets:
        if target >= len(tris):
            continue
        t0 = time.perf_counter()
        dec = ref.simplify_quadric_decimation(target_number_of_triangles=target)
        dec.remove_duplicated_vertices()
        dec.remove_degenerate_triangles()
        el = time.perf_counter() - t0
        mean, p99, mx = deviation(ref, dec)
        n = len(np.asarray(dec.triangles))
        if dec.is_watertight() and np.isfinite(v_ref):
            verr = f"{100 * abs(dec.get_volume() - v_ref) / v_ref:.2f}%"
        else:
            verr = "n/a"
        flag = "" if mx < MARGIN_M else "   <-- exceeds margin"
        print(f"{target:>8} {n:>8} {el:>8.2f} {mean * 1e3:>9.3f} {p99 * 1e3:>8.3f} "
              f"{mx * 1e3:>8.3f} {verr:>9}{flag}")
    print()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="box", choices=("box", "hands", "both"))
    args = ap.parse_args()

    print(f"contact margin {MARGIN_M * 1e3:.1f} mm -- deviation below this cannot change "
          f"which contacts are found\n")
    if args.part in ("box", "both"):
        verts, tris = load_box_mesh("small")
        sweep("box", verts, tris, (20000, 5000, 2000, 1000, 500, 200))
    if args.part in ("hands", "both"):
        for name, verts, tris in hand_collision_meshes():
            sweep(name, verts, tris, (3000, 2000, 1000, 500, 300, 200, 100))


if __name__ == "__main__":
    main()
