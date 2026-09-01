"""How much shape information is actually in the collision meshes?

Collision is 94.5 % of the step at 16 worlds (``sugar_newton.rl.bench_env --profile``), and
mesh-vs-mesh narrow phase is the expensive kind. Before reducing anything, this asks whether
the triangle count is carrying geometry or just tessellation: a shipping carton is a
thin-walled open box, and a few hundred triangles describe that exactly, so a 100k-triangle
collider would be paying for detail the shape does not have.

It also reports what a convex decomposition costs, because the asset itself authors
``physics:approximation = convexDecomposition`` -- Isaac collides the decomposition, not the
raw mesh, so matching it is a fidelity correction as well as a speed one.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from sugar_newton.validation.g1_carrybox_policy import (
    HAND_HULLS,
    URDF,
    load_box_mesh,
)


def describe(name: str, verts: np.ndarray, tris: np.ndarray) -> None:
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    ext = verts.max(0) - verts.min(0)
    print(f"{name}: {len(tris)} triangles, {len(verts)} vertices")
    print(f"  bbox {np.round(ext, 3)} m")
    print(f"  triangle area: median {np.median(area) * 1e6:.2f} mm^2, "
          f"max {area.max() * 1e6:.1f} mm^2, total {area.sum():.4f} m^2")
    # A tessellated flat panel has many coplanar triangles. Counting distinct face normals
    # says how many genuinely different surface orientations the collider actually has.
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    uniq = np.unique(np.round(n, 2), axis=0)
    print(f"  distinct face normals (0.01 rounding): {len(uniq)}"
          f"  -> {len(tris) / max(len(uniq), 1):.0f} triangles per orientation")


def main() -> None:
    wp.init()
    for which in ("small", "big"):
        verts, tris = load_box_mesh(which)
        describe(f"box[{which}]", verts.astype(np.float64), tris)
        print()

    hulls = np.load(HAND_HULLS)
    for side in ("left", "right"):
        describe(f"hand hull[{side}]", hulls[f"{side}_verts"].astype(np.float64),
                 hulls[f"{side}_tris"])
        print()

    # What a decomposition would cost. coacd is what Newton maps
    # physics:approximation=convexDecomposition onto (import_usd.py:3211-3226).
    try:
        import coacd
    except ImportError:
        print("coacd not importable; skipping decomposition")
        return
    verts, tris = load_box_mesh("small")
    for threshold in (0.05, 0.1):
        t0 = time.perf_counter()
        parts = coacd.run_coacd(
            coacd.Mesh(verts.astype(np.float64), tris), threshold=threshold)
        el = time.perf_counter() - t0
        tri_total = sum(len(p[1]) for p in parts)
        print(f"coacd threshold {threshold}: {len(parts)} hulls, "
              f"{tri_total} triangles total, {el:.1f} s to build")


if __name__ == "__main__":
    main()
