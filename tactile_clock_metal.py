# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Tactile measurement video for the metal-clock grasp (same composite as
# tactile_material_*.mp4, but the grasped object is the TRELLIS.2 clock rather
# than the capsule pen). Layout:
#   top:    headless GL render of the scene (arm + clock + hydroelastic patch)
#   middle: per-pad contact-pressure heatmap (pressure = kh*max(0,-depth)) + shear arrow
#   bottom: grip force vs time, shear-on-clock from each pad, and a material signature
#           (contact area / mean penetration / peak pressure)
#
# Usage: python tactile_clock_metal.py --frames 430 --mp4 tactile_material_clock_metal.mp4
#        python tactile_clock_metal.py --preview 200   # composite ONE frame, no video

import pyglet

pyglet.options["headless"] = True

import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

import newton  # noqa: E402
import newton.examples  # noqa: E402
from example_panda_clock_metal import Example  # noqa: E402
from newton.geometry import HydroelasticSDF  # noqa: E402
from newton.sensors import SensorContact  # noqa: E402

# patches: emit the hydroelastic contact surface, allocate per-contact 'force'
_oc = HydroelasticSDF.Config.__init__


def _cfg(self, *a, **k):
    _oc(self, *a, **k)
    self.output_contact_surface = True


HydroelasticSDF.Config.__init__ = _cfg

_op = newton.CollisionPipeline.contacts


def _pc(self, *a, **k):
    self.model.request_contact_attributes("force")
    return _op(self, *a, **k)


newton.CollisionPipeline.contacts = _pc

# Pressure proxy scale: pressure = KH * max(0, -depth). Metal is a rigid material,
# so KH is the rigid hydroelastic stiffness (matches METAL_KH in the clock scene).
KH = 1.0e12
GRID = 180


def tri_area(v):  # v: (n,3,3) world triangle verts -> (n,) areas [m^2]
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rot(q, v):
    xyz, w = q[:3], q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def to_local(world_pts, body_q):
    if len(world_pts) == 0:
        return world_pts.reshape(0, 3)
    return quat_rot(quat_conj(body_q[3:7]), world_pts - body_q[:3])


def main():
    parser = Example.create_parser()
    parser.add_argument("--frames", type=int, default=430)
    parser.add_argument("--outdir", default="tactile_clock_frames")
    parser.add_argument("--mp4", default="tactile_material_clock_metal.mp4")
    parser.add_argument("--preview", type=int, default=None, help="composite only this single frame (no video)")
    parser.set_defaults(viewer="gl", headless=True, scene="clock")
    viewer, args = newton.examples.init(parser)
    if args.preview is not None:
        args.frames = args.preview + 1

    # Pressure proxy = KH * penetration. The compliant pad is the sensing element and
    # the contact-limiting (softer) stiffness, so scale by the pad kh (not the rigid
    # clock kh) for physically sensible pressures.
    global KH
    KH = args.pad_kh

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

    def finger(sub):
        bset = [i for i, l in enumerate(labels) if sub in l]
        return bset[0], set(np.where(np.isin(shape_body, bset))[0].tolist())

    lb, lshapes = finger("leftfinger")
    rb, rshapes = finger("rightfinger")
    fingers = [("left", lb, lshapes), ("right", rb, rshapes)]

    # fixed pad sensor frame per finger (from the pad shape geometry): center = pad
    # origin, normal = pad's thin axis. Anchoring here (instead of the moving contact
    # points) keeps the window stable as the contact slides along the long pad.
    shape_tf = m.shape_transform.numpy()
    shape_sc = m.shape_scale.numpy()
    pad_info = {}
    for nm, fb, fsh in fingers:
        pad = max(fsh)  # pad mesh is added after the URDF finger meshes -> highest index
        V = np.asarray(m.shape_source[pad].vertices) * shape_sc[pad]
        n_idx = int(np.argmin(V.max(0) - V.min(0)))  # thin axis = pad face normal
        n_hat_f = quat_rot(shape_tf[pad][3:7], np.eye(3)[n_idx])
        pad_info[nm] = {
            "pad": int(pad),
            "pf": shape_tf[pad][:3].astype(float),
            "n_hat": n_hat_f / (np.linalg.norm(n_hat_f) + 1e-12),
        }

    sensor = SensorContact(m, sensing_bodies=["*leftfinger*", "*rightfinger*"], counterpart_bodies="object")
    # second sensor: shear ON the clock FROM each finger (world frame, no pad projection)
    sensor_obj = SensorContact(m, sensing_bodies="object", counterpart_bodies=["*leftfinger*", "*rightfinger*"])
    ci = sensor_obj.counterpart_indices[0]
    col_L, col_R = ci.index(lb), ci.index(rb)
    contacts = ex.contacts
    hsdf = ex.collision_pipeline.hydroelastic_sdf

    sceneraw = args.outdir + "_scene"
    os.makedirs(sceneraw, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    try:
        g_world = np.asarray(m.gravity, dtype=float).reshape(-1)[:3]
    except Exception:
        g_world = np.array([0.0, 0.0, -9.81])
    g_world = g_world / (np.linalg.norm(g_world) + 1e-12)

    # ---- pass 1: simulate, capture scene frames + tactile data ----
    store = {"left": [], "right": []}
    forces = {"normal": [[], []], "shear": [[], []], "shear_vec": [[], []], "rot": [[], []], "ftrans": [[], []]}
    obj_sL, obj_sR, obj_net = [], [], []  # shear ON clock from left/right pad + net (world)
    mat_area, mat_pen, mat_peak = [], [], []  # contact area [mm^2], mean penetration [mm], peak pressure [MPa]
    _t_pass1 = time.perf_counter()
    for f in range(args.frames):
        ex.step()
        ex.render()
        scene = ex.viewer.get_frame().numpy()  # (H,W,3) uint8, top-left origin
        plt.imsave(os.path.join(sceneraw, f"s_{f:04d}.png"), scene[40:980, 220:1760])

        ex.solver.update_contacts(contacts)
        sensor.update(ex.state_0, contacts)
        cs = hsdf.get_contact_surface()
        nf = int(cs.face_contact_count.numpy()[0])
        verts = cs.contact_surface_point.numpy()[: 3 * nf].reshape(nf, 3, 3) if nf else np.zeros((0, 3, 3))
        depth = cs.contact_surface_depth.numpy()[:nf] if nf else np.zeros((0,))
        sp = cs.contact_surface_shape_pair.numpy()[:nf] if nf else np.zeros((0, 2), dtype=int)
        bq = ex.state_0.body_q.numpy()
        tf = sensor.total_force.numpy()
        tfr = sensor.total_force_friction.numpy()
        sensor_obj.update(ex.state_0, contacts)
        fmf = sensor_obj.force_matrix_friction.numpy()[0]
        obj_sL.append(fmf[col_L].copy())
        obj_sR.append(fmf[col_R].copy())
        obj_net.append(sensor_obj.total_force_friction.numpy()[0].copy())
        fa, fpen = 0.0, []  # per-frame contact area + penetrations (both pads)
        for fi, (name, fbody, fsh) in enumerate(fingers):
            pad = pad_info[name]["pad"]  # only the pad shape (exclude finger collision mesh)
            mask = np.array([(a == pad or b == pad) for a, b in sp], dtype=bool) if nf else np.zeros(0, bool)
            if mask.any():
                fa += float(tri_area(verts[mask]).sum())
                fpen.append(np.clip(-depth[mask], 0.0, None))
            vw = verts[mask].reshape(-1, 3) if mask.any() else np.zeros((0, 3))
            vd = np.repeat(depth[mask], 3) if mask.any() else np.zeros((0,))
            store[name].append((to_local(vw, bq[fbody]), vd))
            forces["normal"][fi].append(float(np.linalg.norm(tf[fi] - tfr[fi])))
            forces["shear"][fi].append(float(np.linalg.norm(tfr[fi])))
            forces["shear_vec"][fi].append(quat_rot(quat_conj(bq[fbody][3:7]), tfr[fi]))
            forces["rot"][fi].append(bq[fbody][3:7].copy())
            forces["ftrans"][fi].append(bq[fbody][:3].copy())
        ap = np.concatenate(fpen) if fpen else np.zeros(1)
        mat_area.append(fa * 1e6)  # mm^2
        mat_pen.append(float(ap.mean()) * 1e3)  # mm
        mat_peak.append(float(ap.max()) * KH / 1e6)  # MPa

    _dur_pass1 = time.perf_counter() - _t_pass1

    # ---- gravity-aligned pad-plane projection ----
    proj = {"left": [], "right": []}
    pshear = {"left": [], "right": []}
    ext = {}
    pmax = 1.0
    PAD_HALF = 0.02  # fixed half-window [m] anchored on the pad (pad face is ~4x6 cm)
    for fi, (name, _, _) in enumerate(fingers):
        n_hat = pad_info[name]["n_hat"]  # fixed pad face normal (finger frame)
        center = pad_info[name]["pf"]  # fixed pad origin (finger frame)
        inplane = [i for i in range(3) if abs(n_hat[i]) < 0.9]
        alldep = (
            np.concatenate([s[1] for s in store[name] if len(s[1])])
            if any(len(s[1]) for s in store[name])
            else np.zeros(1)
        )
        press_all = np.clip(-alldep, 0.0, None) * KH / 1e6
        if (press_all > 0).any():
            pmax = max(pmax, float(np.percentile(press_all[press_all > 0], 99)))
        for frame in range(args.frames):
            loc, dep = store[name][frame]
            g_local = quat_rot(quat_conj(forces["rot"][fi][frame]), g_world)
            gp = g_local - np.dot(g_local, n_hat) * n_hat  # gravity projected into pad plane
            if np.linalg.norm(gp) < 1e-6:
                v_hat = np.zeros(3)
                v_hat[inplane[0]] = 1.0
            else:
                v_hat = -gp / np.linalg.norm(gp)  # plot-up = opposite of gravity
            u_hat = np.cross(n_hat, v_hat)
            u_hat /= np.linalg.norm(u_hat) + 1e-12
            B = np.stack([u_hat, v_hat], axis=1)  # (3,2) world->plane
            uv = (loc - center) @ B if len(loc) else np.zeros((0, 2))
            proj[name].append((uv, dep))
            sv = forces["shear_vec"][fi][frame]
            pshear[name].append((float(np.dot(sv, u_hat)), float(np.dot(sv, v_hat))))

    # Size the pad window to the ACTUAL contact (common to both pads for comparability),
    # so the patch fills the plot instead of sitting tiny in a fixed 4x4 cm box.
    radii = [np.abs(uv).max() for name in ("left", "right") for uv, _ in proj[name] if len(uv)]
    half = float(np.clip(1.4 * max(radii), 0.005, PAD_HALF)) if radii else PAD_HALF
    for name in ("left", "right"):
        ext[name] = (-half, half, -half, half)

    cmap = LinearSegmentedColormap.from_list(
        "tactile", ["#00004d", "#0010c0", "#00c8ff", "#00ff66", "#ffe000", "#ff2a00"]
    )

    def render_pad(ax, name, frame):
        e = ext[name]
        gx, gy = np.linspace(e[0], e[1], GRID), np.linspace(e[2], e[3], GRID)
        GX, GY = np.meshgrid(gx, gy)
        uv, dep = proj[name][frame]
        field = np.zeros((GRID, GRID))
        if len(uv) >= 4:
            press = np.clip(-dep, 0.0, None) * KH / 1e6  # MPa; depth<0 == penetration
            z = griddata(uv, press, (GX, GY), method="linear", fill_value=0.0)
            field = np.nan_to_num(np.clip(z, 0.0, pmax))
        return ax.imshow(
            field, origin="lower", extent=e, cmap=cmap, vmin=0.0, vmax=pmax, interpolation="bilinear", aspect="equal"
        )

    # ---- pass 2: composite (scene / pads / grip-force / shear-on-clock) ----
    t = np.arange(args.frames) / ex.fps
    up = -g_world
    sLarr, sRarr, netarr = np.array(obj_sL), np.array(obj_sR), np.array(obj_net)
    obj_sLz, obj_sRz, obj_netz = sLarr @ up, sRarr @ up, netarr @ up
    _nL, _nR = np.linalg.norm(sLarr, axis=1), np.linalg.norm(sRarr, axis=1)
    obj_cos = np.where((_nL > 1e-6) & (_nR > 1e-6), np.einsum("ij,ij->i", sLarr, sRarr) / (_nL * _nR + 1e-12), np.nan)

    frames_to_do = [args.preview] if args.preview is not None else range(args.frames)
    _t_pass2 = time.perf_counter()
    for f in frames_to_do:
        fig = plt.figure(figsize=(11, 16.5))
        gs = fig.add_gridspec(5, 2, height_ratios=[1.35, 1.15, 0.55, 0.55, 0.6], hspace=0.5, wspace=0.16)
        ax_scene = fig.add_subplot(gs[0, :])
        ax_scene.imshow(plt.imread(os.path.join(sceneraw, f"s_{f:04d}.png")))
        ax_scene.axis("off")
        ax_scene.set_title("interaction scene (green = hydroelastic contact patch)", fontsize=12)

        axL, axR = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        ims = []
        for ax, (name, _, _), fi in ((axL, fingers[0], 0), (axR, fingers[1], 1)):
            ims.append(render_pad(ax, name, f))
            e = ext[name]
            ax.set_title(f"{name} pad   Fn={forces['normal'][fi][f]:.1f} N   Ft={forces['shear'][fi][f]:.1f} N")
            ax.set_xlabel("tangential u [m]")
            ax.set_ylabel("tangential v  (gravity ↓) [m]")
            su, sv = pshear[name][f]
            n = (su * su + sv * sv) ** 0.5
            if n > 1e-6:
                a_u, a_v = su / n * (e[1] - e[0]) * 0.42, sv / n * (e[3] - e[2]) * 0.42
                arr = ax.arrow(
                    0.0,
                    0.0,
                    a_u,
                    a_v,
                    color="white",
                    width=0.00025,
                    head_width=0.0013,
                    length_includes_head=True,
                    zorder=8,
                )
                arr.set_path_effects([pe.withStroke(linewidth=2.0, foreground="black")])
            # gravity reference (always straight down) + legend
            gx0 = e[0] + 0.13 * (e[1] - e[0])
            gy0 = e[3] - 0.10 * (e[3] - e[2])
            glen = 0.18 * (e[3] - e[2])
            ax.annotate(
                "", xy=(gx0, gy0 - glen), xytext=(gx0, gy0), arrowprops=dict(arrowstyle="-|>", color="gold", lw=2.0)
            )
            ax.text(
                gx0 + 0.03 * (e[1] - e[0]),
                gy0 - 0.5 * glen,
                "g",
                color="gold",
                fontsize=11,
                va="center",
                fontweight="bold",
            )
            ax.text(
                0.98,
                0.02,
                "white = shear",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="white",
                fontsize=7,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )
            ax.set_xlim(e[0], e[1])
            ax.set_ylim(e[2], e[3])
        cb = fig.colorbar(ims[0], ax=[axL, axR], fraction=0.046, pad=0.02)
        cb.set_label("contact pressure ≈ kh·penetration  [MPa]")

        axF = fig.add_subplot(gs[2, :])
        axF.plot(t, forces["normal"][0], color="tab:red", label="Fn left")
        axF.plot(t, forces["normal"][1], color="tab:orange", label="Fn right")
        axF.plot(t, forces["shear"][0], color="tab:blue", ls="--", label="Ft left")
        axF.plot(t, forces["shear"][1], color="tab:cyan", ls="--", label="Ft right")
        axF.axvline(t[f], color="k", lw=1.2)
        axF.set_xlabel("time [s]")
        axF.set_ylabel("force [N]")
        axF.set_title("grip force vs time")
        axF.legend(fontsize=8, ncol=2)
        axF.grid(alpha=0.3)

        # shear force ON the clock from each pad (world frame, no pad-frame projection)
        axS = fig.add_subplot(gs[3, :])
        axS.plot(t, obj_sLz, color="tab:red", label="from left")
        axS.plot(t, obj_sRz, color="tab:blue", label="from right")
        axS.plot(t, obj_netz, color="black", lw=2, label="net (sum)")
        axS.axhline(0, color="gray", lw=0.6)
        axS.axvline(t[f], color="k", lw=1.2)
        axS.set_xlabel("time [s]")
        axS.set_ylabel("shear on clock · up  [N]")
        axS.legend(fontsize=8, ncol=3, loc="upper left")
        axS.grid(alpha=0.3)
        axS.set_title("shear on clock from each pad (world frame)   cos∠(L,R): +1 same dir, −1 opposite")
        axS2 = axS.twinx()
        axS2.plot(t, obj_cos, color="tab:green", lw=1.0)
        axS2.set_ylim(-1.15, 1.15)
        axS2.set_ylabel("cos∠(L,R)", color="tab:green")
        axS2.tick_params(axis="y", labelcolor="tab:green")
        axS2.axhline(0, color="tab:green", ls=":", lw=0.6)

        # material signature: contact area + mean penetration (compliance) + peak pressure
        axM = fig.add_subplot(gs[4, :])
        axM.plot(t, mat_area, color="tab:purple", label="contact area")
        axM.set_xlabel("time [s]")
        axM.set_ylabel("contact area [mm²]", color="tab:purple")
        axM.tick_params(axis="y", labelcolor="tab:purple")
        axM.grid(alpha=0.3)
        axM.axvline(t[f], color="k", lw=1.2)
        axM2 = axM.twinx()
        axM2.plot(t, mat_pen, color="tab:brown", label="mean penetration")
        axM2.set_ylabel("mean penetration [mm]", color="tab:brown")
        axM2.tick_params(axis="y", labelcolor="tab:brown")
        axM.set_title(f"material signature — area & compliance   ·   peak pressure now: {mat_peak[f]:.0f} MPa")

        mtxt = f"   [metal clock, μ=0.5, ρ derived from mass · compliant pad kh={KH:g}]"
        fig.suptitle(f"panda_hydro tactile — metal clock — frame {f}/{args.frames}  t={t[f]:.2f}s{mtxt}", fontsize=13)
        out = os.path.join(args.outdir, f"frame_{f:04d}.png")
        fig.savefig(out, dpi=96, bbox_inches="tight")
        plt.close(fig)

    _dur_pass2 = time.perf_counter() - _t_pass2
    n = args.frames
    n2 = 1 if args.preview is not None else n
    print(
        f"[PROFILE tactile] frames={n}  "
        f"pass1_sim+render={_dur_pass1:.1f}s ({_dur_pass1 / n * 1e3:.1f} ms/frame)  "
        f"pass2_composite={_dur_pass2:.1f}s ({_dur_pass2 / n2 * 1e3:.1f} ms/frame)",
        flush=True,
    )
    if args.preview is not None:
        print(f"wrote preview frame {args.preview}: {os.path.join(args.outdir, f'frame_{args.preview:04d}.png')}")
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        os.path.join(args.outdir, "frame_%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        args.mp4,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    for fn in os.listdir(sceneraw):
        os.remove(os.path.join(sceneraw, fn))
    os.rmdir(sceneraw)
    print(f"wrote {args.mp4}  ({args.frames} frames)")


if __name__ == "__main__":
    main()
