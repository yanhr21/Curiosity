# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Composite tactile video for the SOFT (FEM) pencil in the panda_hydro scene
# (example_panda_soft_rod.Example). Mirrors tactile_material_metal.mp4's layout AND its
# PAD-CENTRIC pad view: the heatmap window is anchored to the pad's fixed geometry (pad
# origin + thin-axis normal, gravity-up vertical) in the finger's local frame, and the rod
# projection MOVES within it as the contact slides — exactly like the rigid video. Soft
# contact (compression [mm]) comes from SolverVBD soft contacts, not a hydroelastic surface.
#
#   preview ONE composite frame:  python tactile_soft_rod_video.py --preview 240
#   full render:                  python tactile_soft_rod_video.py --frames 720
#   rigid-vs-soft timing only:    python tactile_soft_rod_video.py --time-only 60
import pyglet  # noqa: E402

pyglet.options["headless"] = True

import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

import newton  # noqa: E402
import newton.examples  # noqa: E402
import example_panda_soft_rod as M  # noqa: E402
from example_panda_soft_rod import Example  # noqa: E402

ROD_RADIUS = 0.005  # original pencil capsule radius (rest cross-section radius)
GRID = 180
PAD_HALF = 0.02  # fixed pad window half-size [m] (pad face is ~4x6 cm) — same as rigid


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rot(q, v):  # works for v shape (3,) or (n,3)
    xyz, w = q[:3], q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def to_local(world_pts, body_q):
    if len(world_pts) == 0:
        return world_pts.reshape(0, 3)
    return quat_rot(quat_conj(body_q[3:7]), world_pts - body_q[:3])


def rod_axis(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0]


def rod_bend(P):
    c, ax = rod_axis(P)
    d = P - c
    return float(np.linalg.norm(d - np.outer(d @ ax, ax), axis=1).max())


def main():
    parser = Example.create_parser()
    parser.add_argument("--preview", type=int, default=None, help="render ONE composite frame at this index")
    parser.add_argument("--frames", type=int, default=720)
    parser.add_argument("--outdir", default="soft_rod_frames")
    parser.add_argument("--mp4", default="tactile_material_rubber_soft.mp4")
    parser.add_argument("--out", default="soft_rod_preview.png")
    parser.add_argument("--time-only", type=int, default=None, help="just benchmark N frames")
    parser.set_defaults(viewer="gl", headless=True)
    viewer, args = newton.examples.init(parser)

    if args.time_only is not None:
        benchmark(viewer, args, args.time_only)
        return

    ex = Example(viewer, args)
    ex.show_isosurface = True
    try:
        ex.viewer.show_hydro_contact_surface = True
    except Exception:
        pass
    ex.viewer.set_camera(pos=wp.vec3(0.38, -0.12, 0.46), pitch=-26, yaw=-116)

    m = ex.model
    labels = m.body_key if hasattr(m, "body_key") else m.body_label
    shape_body = m.shape_body.numpy()
    lb = next(i for i, l in enumerate(labels) if l.endswith("leftfinger"))
    rb = next(i for i, l in enumerate(labels) if l.endswith("rightfinger"))
    shape_tf = m.shape_transform.numpy()
    shape_sc = m.shape_scale.numpy()
    # fixed pad frame per finger (pad origin + thin-axis normal), in the finger's local frame
    pad_info = {}
    for nm, fb in (("left", lb), ("right", rb)):
        fsh = set(np.where(shape_body == fb)[0].tolist())
        pad = max(fsh)  # pad mesh added after the URDF finger meshes -> highest index
        V = np.asarray(m.shape_source[pad].vertices) * shape_sc[pad]
        n_idx = int(np.argmin(V.max(0) - V.min(0)))  # thin axis = pad face normal
        n_hat = quat_rot(shape_tf[pad][3:7], np.eye(3)[n_idx])
        pad_info[nm] = {
            "pad": int(pad),
            "fb": fb,
            "pf": shape_tf[pad][:3].astype(float),
            "n_hat": n_hat / (np.linalg.norm(n_hat) + 1e-12),
        }
    ap_pt = (2 * M.ROD_PRAD) ** 2  # per-particle contact area [m^2]
    soft_ke = float(m.soft_contact_ke)
    g_world = np.array([0.0, 0.0, -9.81])
    g_world = g_world / (np.linalg.norm(g_world) + 1e-12)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "tactile", ["#00004d", "#0010c0", "#00c8ff", "#00ff66", "#ffe000", "#ff2a00"]
    )

    def measure():
        ex.collision_pipeline.collide(ex.state_0, ex.contacts)
        nsc = int(ex.contacts.soft_contact_count.numpy()[0])
        ssh = ex.contacts.soft_contact_shape.numpy()[:nsc] if nsc else np.zeros(0, int)
        spp = ex.contacts.soft_contact_particle.numpy()[:nsc] if nsc else np.zeros(0, int)
        P = ex.state_0.particle_q.numpy()[ex.rod_p0 :]
        cen, uax = rod_axis(P)
        d = P - cen
        radial = np.linalg.norm(d - np.outer(d @ uax, uax), axis=1)
        comp = np.clip(ROD_RADIUS - radial, 0.0, None)  # per-particle squash [m]
        bq = ex.state_0.body_q.numpy()
        e_world = np.array([cen - 0.07 * uax, cen + 0.07 * uax])  # rod endpoints (world)
        out = {}
        fa = 0.0
        peak = 0.0
        for nm, info in pad_info.items():
            pad, fb, pf, n_hat = info["pad"], info["fb"], info["pf"], info["n_hat"]
            mask = ssh == pad
            pids = np.unique(spp[mask]) if mask.any() else np.zeros(0, int)
            wpts = P[pids]
            cval = comp[pids]
            # pad-plane basis (finger-local), vertical aligned to gravity (recomputed/frame)
            g_local = quat_rot(quat_conj(bq[fb][3:7]), g_world)
            gp = g_local - np.dot(g_local, n_hat) * n_hat
            if np.linalg.norm(gp) < 1e-6:
                inplane = [i for i in range(3) if abs(n_hat[i]) < 0.9]
                v_hat = np.zeros(3)
                v_hat[inplane[0]] = 1.0
            else:
                v_hat = -gp / np.linalg.norm(gp)  # plot-up = opposite gravity
            u_hat = np.cross(n_hat, v_hat)
            u_hat /= np.linalg.norm(u_hat) + 1e-12
            B = np.stack([u_hat, v_hat], axis=1)  # (3,2) finger-local -> pad plane
            loc = to_local(wpts, bq[fb])
            uv = (loc - pf) @ B if len(loc) else np.zeros((0, 2))
            ke = to_local(e_world, bq[fb])
            ends = ((ke[0] - pf) @ B, (ke[1] - pf) @ B)
            gripN = float(soft_ke * cval.sum() * ap_pt)
            out[nm] = (uv, cval * 1000.0, gripN, ends)
            fa += len(pids) * ap_pt
            if len(cval):
                peak = max(peak, float(cval.max() * 1000.0))
        return out, fa * 1e6, peak, float(P[:, 2].mean()) * 1000.0, rod_bend(P) * 1000.0

    def pad_img(ax, uv, cc, ends, pmax):
        ext = (-PAD_HALF, PAD_HALF, -PAD_HALF, PAD_HALF)
        field = np.zeros((GRID, GRID))
        if len(uv) >= 4:
            gx = np.linspace(ext[0], ext[1], GRID)
            GX, GY = np.meshgrid(gx, gx)
            z = griddata(uv, cc, (GX, GY), method="linear", fill_value=0.0)
            field = np.nan_to_num(np.clip(z, 0.0, pmax))
        im = ax.imshow(field, origin="lower", extent=ext, cmap=cmap, vmin=0.0, vmax=pmax,
                       interpolation="bilinear", aspect="equal")
        # moving rod projection (cyan): centerline + rest-radius edges, in the FIXED pad frame
        q0, q1 = np.asarray(ends[0]), np.asarray(ends[1])
        dv = q1 - q0
        Ln = np.linalg.norm(dv)
        if Ln > 1e-9:
            perp = np.array([-dv[1], dv[0]]) / Ln * ROD_RADIUS
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color="cyan", lw=1.1, ls="--", zorder=7)
            for s in (1, -1):
                ax.plot([q0[0] + s * perp[0], q1[0] + s * perp[0]],
                        [q0[1] + s * perp[1], q1[1] + s * perp[1]], color="cyan", lw=1.3, zorder=7)
        # gravity reference (down = -v, since vertical axis is gravity-up)
        gx0 = ext[0] + 0.13 * (ext[1] - ext[0])
        gy0 = ext[3] - 0.10 * (ext[3] - ext[2])
        glen = 0.18 * (ext[3] - ext[2])
        ax.annotate("", xy=(gx0, gy0 - glen), xytext=(gx0, gy0), arrowprops=dict(arrowstyle="-|>", color="gold", lw=2.0))
        ax.text(gx0 + 0.03 * (ext[1] - ext[0]), gy0 - 0.5 * glen, "g", color="gold", fontsize=11, va="center", fontweight="bold")
        ax.text(0.98, 0.02, "cyan = rod", transform=ax.transAxes, ha="right", va="bottom", color="white",
                fontsize=7, path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.set_xlim(*ext[:2])
        ax.set_ylim(*ext[2:])
        ax.set_xlabel("tangential u [m]")
        ax.set_ylabel("tangential v  (gravity ↓) [m]")
        return im

    sceneraw = args.outdir + "_scene"
    os.makedirs(sceneraw, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    # ---- pass 1: simulate, cache scene + measurements (with timing) ----
    nframes = (args.preview + 1) if args.preview is not None else args.frames
    store = {"left": [], "right": []}
    gripL, gripR, area, mean_c, peak_c, rodz, rbend = [], [], [], [], [], [], []
    sim_t = []
    for f in range(nframes):
        t0 = time.perf_counter()
        ex.step()
        ex.render()
        sim_t.append(time.perf_counter() - t0)
        scene = ex.viewer.get_frame().numpy()
        if args.preview is None or f == args.preview:
            plt.imsave(os.path.join(sceneraw, f"s_{f:04d}.png"), scene[40:980, 220:1760])
        pads, fa, pk, rz, rbv = measure()
        store["left"].append(pads["left"])
        store["right"].append(pads["right"])
        gripL.append(pads["left"][2])
        gripR.append(pads["right"][2])
        area.append(fa)
        peak_c.append(pk)
        allc = np.concatenate([pads["left"][1], pads["right"][1]]) if (len(pads["left"][1]) + len(pads["right"][1])) else np.zeros(1)
        mean_c.append(float(allc[allc > 0].mean()) if (allc > 0).any() else 0.0)
        rodz.append(rz)
        rbend.append(rbv)

    sim_ms = 1000.0 * np.mean(sim_t)
    print(f"# SOFT sim+render: {sim_ms:.0f} ms/frame  ({1000.0 / sim_ms:.1f} fps)  over {nframes} frames")

    allcomp = [c for fr in store["left"] + store["right"] for c in fr[1] if c > 0]
    pmax = max(0.5, float(np.percentile(allcomp, 98)) if allcomp else 0.5)
    t = np.arange(nframes) / ex.fps

    def composite(f):
        fig = plt.figure(figsize=(11, 16.5))
        gs = fig.add_gridspec(5, 2, height_ratios=[1.35, 1.15, 0.55, 0.55, 0.6], hspace=0.5, wspace=0.16)
        ax_scene = fig.add_subplot(gs[0, :])
        ax_scene.imshow(plt.imread(os.path.join(sceneraw, f"s_{f:04d}.png")))
        ax_scene.axis("off")
        ax_scene.set_title("interaction scene (soft FEM rod + hydroelastic overlay)", fontsize=12)

        axL, axR = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        ims = []
        for ax, nm, gN in ((axL, "left", gripL[f]), (axR, "right", gripR[f])):
            uv, cc, _, ends = store[nm][f]
            ims.append(pad_img(ax, uv, cc, ends, pmax))
            ax.set_title(f"{nm} pad   grip≈{gN:.1f} N   (VBD compression)")
        cb = fig.colorbar(ims[0], ax=[axL, axR], fraction=0.046, pad=0.02)
        cb.set_label("rod compression  [mm]")

        axF = fig.add_subplot(gs[2, :])
        axF.plot(t, gripL, color="tab:red", label="grip left")
        axF.plot(t, gripR, color="tab:orange", label="grip right")
        axF.axvline(t[f], color="k", lw=1.2)
        axF.set_xlabel("time [s]")
        axF.set_ylabel("grip force ≈ kₑ·Σcompression·area [N]")
        axF.set_title("grip force vs time (compression proxy)")
        axF.legend(fontsize=8, ncol=2)
        axF.grid(alpha=0.3)

        axB = fig.add_subplot(gs[3, :])
        axB.plot(t, rbend, color="tab:green")
        axB.axvline(t[f], color="k", lw=1.2)
        axB.set_xlabel("time [s]")
        axB.set_ylabel("rod bend [mm]", color="tab:green")
        axB.tick_params(axis="y", labelcolor="tab:green")
        axB.grid(alpha=0.3)
        axB.set_title("soft-body deformation: rod bend & height")
        axB2 = axB.twinx()
        axB2.plot(t, rodz, color="tab:blue", lw=1.0)
        axB2.set_ylabel("rod mean height [mm]", color="tab:blue")
        axB2.tick_params(axis="y", labelcolor="tab:blue")

        axM = fig.add_subplot(gs[4, :])
        axM.plot(t, area, color="tab:purple")
        axM.set_xlabel("time [s]")
        axM.set_ylabel("contact area [mm²]", color="tab:purple")
        axM.tick_params(axis="y", labelcolor="tab:purple")
        axM.grid(alpha=0.3)
        axM.axvline(t[f], color="k", lw=1.2)
        axM2 = axM.twinx()
        axM2.plot(t, mean_c, color="tab:brown")
        axM2.set_ylabel("mean compression [mm]", color="tab:brown")
        axM2.tick_params(axis="y", labelcolor="tab:brown")
        axM.set_title(f"material signature — area & compression   ·   peak compression now: {peak_c[f]:.1f} mm")

        fig.suptitle(
            f"panda_hydro tactile — SOFT rod   frame {f}/{nframes}  t={t[f]:.2f}s   "
            f"[E={M.E_ROD:g} Pa, ρ={M.ROD_DENSITY:g} kg/m³, {M.VBD_ITERS} VBD it]",
            fontsize=13,
        )
        return fig

    if args.preview is not None:
        fig = composite(args.preview)
        fig.savefig(args.out, dpi=96, bbox_inches="tight")
        plt.close(fig)
        for fn in os.listdir(sceneraw):
            os.remove(os.path.join(sceneraw, fn))
        os.rmdir(sceneraw)
        print(f"wrote {args.out} (composite frame {args.preview})")
        return

    for f in range(nframes):
        fig = composite(f)
        fig.savefig(os.path.join(args.outdir, f"frame_{f:04d}.png"), dpi=96, bbox_inches="tight")
        plt.close(fig)

    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(args.outdir, "frame_%04d.png"),
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", args.mp4],
        check=True, capture_output=True,
    )
    for fn in os.listdir(sceneraw):
        os.remove(os.path.join(sceneraw, fn))
    os.rmdir(sceneraw)
    print(f"wrote {args.mp4}  ({nframes} frames, soft sim {sim_ms:.0f} ms/frame)")


def benchmark(viewer, args, n):
    soft = Example(viewer, args)
    soft.viewer.set_camera(pos=wp.vec3(0.38, -0.12, 0.46), pitch=-26, yaw=-116)
    for _ in range(5):
        soft.step()
        soft.render()
    t0 = time.perf_counter()
    for _ in range(n):
        soft.step()
        soft.render()
    soft_ms = 1000.0 * (time.perf_counter() - t0) / n
    print(f"# SOFT (MuJoCo arm + SolverVBD {M.VBD_ITERS} it, eager): {soft_ms:.1f} ms/frame  ({1000 / soft_ms:.1f} fps)")


if __name__ == "__main__":
    main()
