# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Thin headless driver for example_panda_soft_rod.Example (the panda_hydro scene with the
# rigid pencil replaced by a soft FEM rod). Renders ONE scene frame with the SAME camera and
# crop as tactile_video.py, so the result can be compared directly to tactile_material_metal.mp4.
#
#   python render_soft_rod.py --preview-scene 150 --out soft_rod_scene.png
import pyglet  # noqa: E402

pyglet.options["headless"] = True

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton  # noqa: E402
import newton.examples  # noqa: E402
from example_panda_soft_rod import Example  # noqa: E402


def rod_bend(P):
    """(center, axis, max transverse deflection [m]) of the rod particle cloud."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    axis = vt[0]
    d = P - c
    transverse = d - np.outer(d @ axis, axis)
    return c, axis, float(np.linalg.norm(transverse, axis=1).max())


def main():
    import example_panda_soft_rod as M

    parser = Example.create_parser()
    parser.add_argument("--preview-scene", type=int, default=150, help="step to this frame, render the scene")
    parser.add_argument("--out", default="soft_rod_scene.png")
    parser.add_argument("--e-rod", type=float, default=None, help="override rod Young's modulus")
    parser.add_argument("--rho", type=float, default=None, help="override rod density")
    parser.add_argument("--iters", type=int, default=None, help="override VBD iterations/substep")
    parser.add_argument("--grip-close", type=float, default=None, help="override closed finger position [m]")
    parser.set_defaults(viewer="gl", headless=True)
    viewer, args = newton.examples.init(parser)

    if args.e_rod is not None:
        M.E_ROD = args.e_rod
    if args.rho is not None:
        M.ROD_DENSITY = args.rho
    if args.iters is not None:
        M.VBD_ITERS = args.iters
    if args.grip_close is not None:
        M.GRIP_CLOSE = args.grip_close

    ex = Example(viewer, args)
    # Match tactile_video.py exactly: same camera + hydroelastic overlay enabled.
    ex.show_isosurface = True
    try:
        ex.viewer.show_hydro_contact_surface = True
    except Exception:
        pass
    ex.viewer.set_camera(pos=wp.vec3(0.38, -0.12, 0.46), pitch=-26, yaw=-116)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = args.preview_scene
    print(f"# E_ROD={M.E_ROD:g} GRIP_CLOSE={M.GRIP_CLOSE} rod_particles={ex.model.particle_count - ex.rod_p0}")
    for f in range(n + 1):
        ex.step()
        ex.render()
        if f % 30 == 0 or f == n:
            P = ex.state_0.particle_q.numpy()[ex.rod_p0 :]
            cen, _, bend = rod_bend(P)
            lf, rf = ex.left_idx if hasattr(ex, "left_idx") else 12, 13
            bq = ex.state_0.body_q.numpy()
            gap = float(np.linalg.norm(bq[12][:3] - bq[13][:3]))
            print(
                f"f={f:3d} t={ex.sim_time:4.2f} rod_c=({cen[0] * 1000:+.0f},{cen[1] * 1000:+.0f},"
                f"{cen[2] * 1000:.0f})mm bend={bend * 1000:5.1f}mm finger_gap={gap * 1000:4.0f}mm"
            )
    scene = ex.viewer.get_frame().numpy()  # (H,W,3) uint8
    crop = scene[40:980, 220:1760]
    plt.imsave(args.out, crop)
    print(f"wrote {args.out}  (frame {n})")


if __name__ == "__main__":
    main()
