"""Report the built model's actual masses against the assets they came from.

``ModelBuilder.ShapeConfig.density`` defaults to 1000 kg/m^3, and ``add_shape_*`` ADDS the
shape's computed mass and inertia to the body whenever density > 0 (builder.py:6125-6126).
So ``add_body(mass=0.5)`` followed by ``add_shape_mesh(cfg=ShapeConfig(...))`` does not give
a 0.5 kg body -- it gives 0.5 kg plus the mesh's volume times 1000. This prints what the
model really contains so the question is settled by the model rather than by reading code.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from sugar_newton.validation.g1_carrybox_policy import (
    BOX_MASS,
    G1PolicyScene,
    load_box_mesh,
    load_clip,
)


def mesh_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh, by the divergence theorem."""
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    return float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def main() -> None:
    wp.init()
    clip = load_clip("data_000")
    scene = G1PolicyScene(clip, box="small", mu=1.0)
    m = scene.model
    mass = m.body_mass.numpy()
    inertia = m.body_inertia.numpy()
    labels = [l.split("/")[-1] for l in m.body_label]

    box_i = scene.box_body
    print(f"box body index {box_i} ({labels[box_i]})")
    print(f"  asset mass (SUGAR BOX_MASS)      {BOX_MASS['small']:.4f} kg")
    print(f"  model  mass (what we simulate)   {mass[box_i]:.4f} kg")
    print(f"  ratio                            {mass[box_i] / BOX_MASS['small']:.2f}x")
    verts, tris = load_box_mesh("small")
    vol = mesh_volume(verts.astype(np.float64), tris)
    print(f"  mesh volume                      {vol * 1e6:.1f} cm^3"
          f"  -> {vol * 1000.0:.4f} kg at density 1000")
    print(f"  box inertia diag                 {np.diag(inertia[box_i])}")

    robot = [i for i in range(m.body_count) if i != box_i]
    print(f"\nrobot: {len(robot)} bodies, total mass {mass[robot].sum():.3f} kg")
    print("  (Unitree G1 29-DOF with rubber hands is ~35 kg)")
    heavy = np.argsort(-mass[robot])[:6]
    print("  heaviest: " + "  ".join(
        f"{labels[robot[i]]} {mass[robot[i]]:.2f}" for i in heavy))
    hands = [i for i in range(m.body_count) if labels[i].endswith("_rubber_hand")]
    print("  hands: " + "  ".join(f"{labels[i]} {mass[i]:.3f} kg" for i in hands))


if __name__ == "__main__":
    main()
