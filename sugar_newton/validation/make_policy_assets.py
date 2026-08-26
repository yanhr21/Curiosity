# SPDX-License-Identifier: BSD-3-Clause
"""Generate the two derived binaries :mod:`g1_carrybox_policy` needs.

Both outputs are ``*.npz`` and therefore gitignored (``.gitignore:59``), which is the
right convention -- they are derived from assets that are themselves not in the repo.
Run this once on a machine that has torch and scipy (the login-node conda envs do; the
Newton container deliberately does not) and the policy scene becomes runnable:

    python -m sugar_newton.validation.make_policy_assets

``tracker_actor.npz``
    SUGAR's official ``demo_ckpts/CarryBox/tracker.pt`` actor as plain arrays, so the
    container can evaluate the policy in NumPy instead of carrying a torch dependency for
    four Linear layers.

``hand_hulls.npz``
    Convex hulls of the two rubber-hand STLs. Used *only* by the ``--hull-hands``
    diagnostic ablation, which exists to test -- and, as it turned out, to refute -- the
    claim that Isaac's convex-hull hand colliders explain its stronger grasp. Nothing in
    the default path hulls anything.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUGAR = HERE.parents[1] / "SUGAR"
TRACKER_PT = SUGAR / "demo_ckpts/CarryBox/tracker.pt"
MESHES = SUGAR / "descriptions/robots/g1/meshes"


def export_actor(out: Path) -> None:
    import torch

    ck = torch.load(TRACKER_PT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    arrays = {k.replace(".", "_"): v.detach().numpy().astype(np.float32)
              for k, v in sd.items() if k.startswith("actor.") or k == "std"}
    np.savez(out, **arrays)
    print(f"wrote {out}  (iter {ck.get('iter')}, "
          f"actor {arrays['actor_0_weight'].shape[1]} -> {arrays['actor_6_weight'].shape[0]})")


def read_stl(path: Path) -> np.ndarray:
    """Binary STL to an (n, 3, 3) array of triangle vertices."""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return raw[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(np.float64)


def export_hand_hulls(out: Path) -> None:
    from scipy.spatial import ConvexHull

    arrays = {}
    for side in ("left", "right"):
        tri = read_stl(MESHES / f"{side}_rubber_hand.STL")
        verts_in = tri.reshape(-1, 3)
        hull = ConvexHull(verts_in)
        verts = hull.points[hull.vertices]
        remap = {int(o): i for i, o in enumerate(hull.vertices)}
        tris = np.array([[remap[int(i)] for i in s] for s in hull.simplices], dtype=np.int32)
        # scipy does not promise outward winding; fix it against the hull's own planes.
        for k, _ in enumerate(hull.simplices):
            a, b, c = verts[tris[k]]
            if np.dot(np.cross(b - a, c - a), hull.equations[k, :3]) < 0.0:
                tris[k] = tris[k][::-1]
        arrays[f"{side}_verts"] = verts.astype(np.float32)
        arrays[f"{side}_tris"] = tris

        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        mesh_vol = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        print(f"  {side}: mesh {len(tri)} tris, {mesh_vol * 1e6:.1f} cm^3  ->  "
              f"hull {len(tris)} tris, {hull.volume * 1e6:.1f} cm^3 "
              f"({hull.volume / mesh_vol:.2f}x)")
    np.savez(out, **arrays)
    print(f"wrote {out}")


def main() -> None:
    export_actor(HERE / "tracker_actor.npz")
    export_hand_hulls(HERE / "hand_hulls.npz")


if __name__ == "__main__":
    main()
