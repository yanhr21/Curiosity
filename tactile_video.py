# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Render a tactile video for the panda_hydro grasp:
#   top:    headless GL render of the interaction scene (arm + pen + contact iso-surface)
#   middle: per-finger continuous contact-pressure heatmap (pressure = kh*max(0,-depth),
#           so deepest penetration at the patch center is hottest) + shear-force arrow
#   bottom: normal / shear force-vs-time traces with a moving cursor
#
# Usage: python tactile_video.py --world-count 1 --scene pen --device cuda:0

import pyglet  # noqa: E402  -- must set headless before newton/pyglet import the display

pyglet.options["headless"] = True

import os  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

import newton  # noqa: E402
import newton.examples  # noqa: E402
from newton.geometry import HydroelasticSDF  # noqa: E402
from newton.sensors import SensorContact  # noqa: E402
from newton.examples.robot.example_robot_panda_hydro import Example, broadcast_ik_solution_kernel  # noqa: E402

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

KH = 1e11  # hydroelastic stiffness; pressure proxy = KH * max(0, -depth)  [depth<0 == penetrating]
GRID = 180


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rot(q, v):
    xyz, w = q[:3], q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def tmul(a, b):  # compose transforms a∘b (7-vectors)
    return np.concatenate([a[:3] + quat_rot(a[3:7], b[:3]), quat_mul(a[3:7], b[3:7])])


def to_local(world_pts, body_q):
    if len(world_pts) == 0:
        return world_pts.reshape(0, 3)
    return quat_rot(quat_conj(body_q[3:7]), world_pts - body_q[:3])


def install_push(ex, base, rate):
    """Override the example trajectory: hold the grasp and drive the EE straight
    down at ``rate`` m/frame so the pen is pushed into the cup bottom."""
    base_pos = base[:3].astype(float)
    rq = base[3:7].astype(float)
    st = {"i": 0}

    def push():
        i = st["i"]
        tp = wp.vec3(float(base_pos[0]), float(base_pos[1]), float(base_pos[2] - rate * i))
        ex.pos_obj.set_target_positions(wp.array([tp], dtype=wp.vec3))
        ex.rot_obj.set_target_rotations(
            wp.array([wp.vec4(float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3]))], dtype=wp.vec4))
        if ex.graph_ik is not None:
            wp.capture_launch(ex.graph_ik)
        else:
            ex.ik_solver.step(ex.joint_q_ik, ex.joint_q_ik, iterations=ex.ik_iters)
        wp.launch(broadcast_ik_solution_kernel, dim=ex.world_count,
                  inputs=[ex.joint_q_ik, ex.joint_targets_2d, 0.0])  # gripper stays closed
        wp.copy(ex.control.joint_target_q, ex.joint_targets_2d.flatten())
        st["i"] += 1

    ex.set_joint_targets = push


def main():
    parser = Example.create_parser()
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--outdir", default="tactile_frames")
    parser.add_argument("--mp4", default="tactile_panda.mp4")
    parser.add_argument("--push-after", type=int, default=None,
                        help="after this frame, hold grasp and push the gripper straight down")
    parser.add_argument("--push-frames", type=int, default=100)
    parser.add_argument("--push-rate", type=float, default=0.001)
    parser.set_defaults(viewer="gl", headless=True)
    viewer, args = newton.examples.init(parser)
    if args.push_after is not None:
        args.frames = args.push_after + 1 + args.push_frames
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

    # pen (capsule) geometry: length along local Z, scale = (radius, half_height, radius)
    obj = [i for i, l in enumerate(labels) if l.endswith("object")][0]
    obj_shape = int(np.where(shape_body == obj)[0][0])
    pen_st = m.shape_transform.numpy()[obj_shape]
    pen_ssc = m.shape_scale.numpy()[obj_shape]
    pen_r = float(pen_ssc[0])
    pen_e0s, pen_e1s = [], []

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
        pad_info[nm] = {"pad": int(pad), "pf": shape_tf[pad][:3].astype(float),
                        "n_hat": n_hat_f / (np.linalg.norm(n_hat_f) + 1e-12)}

    sensor = SensorContact(m, sensing_bodies=["*leftfinger*", "*rightfinger*"], counterpart_bodies="object")
    # second sensor: shear ON the pen FROM each finger (world frame, no pad projection)
    sensor_pen = SensorContact(m, sensing_bodies="object", counterpart_bodies=["*leftfinger*", "*rightfinger*"])
    ci = sensor_pen.counterpart_indices[0]
    col_L, col_R = ci.index(lb), ci.index(rb)
    contacts = ex.contacts
    hsdf = ex.collision_pipeline.hydroelastic_sdf

    sceneraw = args.outdir + "_scene"
    os.makedirs(sceneraw, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap

    # gravity direction (world), for aligning the pad plots
    try:
        g_world = np.asarray(m.gravity, dtype=float).reshape(-1)[:3]
    except Exception:
        g_world = np.array([0.0, 0.0, -9.81])
    g_world = g_world / (np.linalg.norm(g_world) + 1e-12)

    # ---- pass 1: simulate, capture scene frames + tactile data ----
    store = {"left": [], "right": []}
    forces = {"normal": [[], []], "shear": [[], []], "shear_vec": [[], []], "rot": [[], []], "ftrans": [[], []]}
    pen_sL, pen_sR, pen_net = [], [], []  # shear ON pen from left/right finger + net (world)
    for f in range(args.frames):
        if args.push_after is not None and f == args.push_after + 1:
            install_push(ex, ex.state_0.body_q.numpy()[ex.ee_index].copy(), args.push_rate)
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
        sensor_pen.update(ex.state_0, contacts)
        fmf = sensor_pen.force_matrix_friction.numpy()[0]
        pen_sL.append(fmf[col_L].copy())
        pen_sR.append(fmf[col_R].copy())
        pen_net.append(sensor_pen.total_force_friction.numpy()[0].copy())
        sw = tmul(bq[obj], pen_st)  # pen capsule world transform -> endpoints
        axw = quat_rot(sw[3:7], np.array([0.0, 0.0, 1.0]))
        pen_e0s.append(sw[:3] - float(pen_ssc[1]) * axw)
        pen_e1s.append(sw[:3] + float(pen_ssc[1]) * axw)
        for fi, (name, fbody, fsh) in enumerate(fingers):
            pad = pad_info[name]["pad"]  # only the pad shape (exclude finger collision mesh)
            mask = np.array([(a == pad or b == pad) for a, b in sp], dtype=bool) if nf else np.zeros(0, bool)
            vw = verts[mask].reshape(-1, 3) if mask.any() else np.zeros((0, 3))
            vd = np.repeat(depth[mask], 3) if mask.any() else np.zeros((0,))
            store[name].append((to_local(vw, bq[fbody]), vd))
            forces["normal"][fi].append(float(np.linalg.norm(tf[fi] - tfr[fi])))
            forces["shear"][fi].append(float(np.linalg.norm(tfr[fi])))
            forces["shear_vec"][fi].append(quat_rot(quat_conj(bq[fbody][3:7]), tfr[fi]))
            forces["rot"][fi].append(bq[fbody][3:7].copy())
            forces["ftrans"][fi].append(bq[fbody][:3].copy())

    # ---- gravity-aligned pad-plane projection ----
    # pad normal = local axis of least contact-point spread; in-plane "down" = world
    # gravity projected into the pad plane (recomputed per frame as the finger rotates),
    # so the plot's vertical axis is the gravity direction (down = bottom).
    proj = {"left": [], "right": []}
    pshear = {"left": [], "right": []}
    proj_pen = {"left": [], "right": []}
    ext = {}
    pmax = 1.0
    PAD_HALF = 0.02  # fixed half-window [m] anchored on the pad (pad face is ~4x6 cm)
    for fi, (name, _, _) in enumerate(fingers):
        n_hat = pad_info[name]["n_hat"]          # fixed pad face normal (finger frame)
        center = pad_info[name]["pf"]            # fixed pad origin (finger frame)
        inplane = [i for i in range(3) if abs(n_hat[i]) < 0.9]
        alldep = np.concatenate([s[1] for s in store[name] if len(s[1])]) if any(len(s[1]) for s in store[name]) else np.zeros(1)
        press_all = np.clip(-alldep, 0.0, None) * KH / 1e6
        if (press_all > 0).any():
            pmax = max(pmax, float(np.percentile(press_all[press_all > 0], 99)))
        for frame in range(args.frames):
            loc, dep = store[name][frame]
            g_local = quat_rot(quat_conj(forces["rot"][fi][frame]), g_world)
            gp = g_local - np.dot(g_local, n_hat) * n_hat  # gravity projected into pad plane
            if np.linalg.norm(gp) < 1e-6:
                v_hat = np.zeros(3); v_hat[inplane[0]] = 1.0
            else:
                v_hat = -gp / np.linalg.norm(gp)  # plot-up = opposite of gravity
            u_hat = np.cross(n_hat, v_hat)
            u_hat /= np.linalg.norm(u_hat) + 1e-12
            B = np.stack([u_hat, v_hat], axis=1)  # (3,2) world->plane
            uv = (loc - center) @ B if len(loc) else np.zeros((0, 2))
            proj[name].append((uv, dep))
            sv = forces["shear_vec"][fi][frame]
            pshear[name].append((float(np.dot(sv, u_hat)), float(np.dot(sv, v_hat))))
            # pen endpoints in this finger's local frame -> same uv plane
            fr, ftr = forces["rot"][fi][frame], forces["ftrans"][fi][frame]
            le0 = quat_rot(quat_conj(fr), pen_e0s[frame] - ftr)
            le1 = quat_rot(quat_conj(fr), pen_e1s[frame] - ftr)
            proj_pen[name].append(((le0 - center) @ B, (le1 - center) @ B))
        ext[name] = (-PAD_HALF, PAD_HALF, -PAD_HALF, PAD_HALF)

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
            # linear (not cubic) — cubic overshoots on near-collinear contact points
            z = griddata(uv, press, (GX, GY), method="linear", fill_value=0.0)
            field = np.nan_to_num(np.clip(z, 0.0, pmax))
        return ax.imshow(field, origin="lower", extent=e, cmap=cmap, vmin=0.0, vmax=pmax,
                         interpolation="bilinear", aspect="equal")

    # ---- pass 2: composite (scene / pads / grip-force / shear-on-pen) ----
    t = np.arange(args.frames) / ex.fps
    up = -g_world
    sLarr, sRarr, netarr = np.array(pen_sL), np.array(pen_sR), np.array(pen_net)
    pen_sLz, pen_sRz, pen_netz = sLarr @ up, sRarr @ up, netarr @ up
    _nL, _nR = np.linalg.norm(sLarr, axis=1), np.linalg.norm(sRarr, axis=1)
    pen_cos = np.where((_nL > 1e-6) & (_nR > 1e-6),
                       np.einsum("ij,ij->i", sLarr, sRarr) / (_nL * _nR + 1e-12), np.nan)
    for f in range(args.frames):
        fig = plt.figure(figsize=(11, 14.5))
        gs = fig.add_gridspec(4, 2, height_ratios=[1.4, 1.2, 0.62, 0.62], hspace=0.4, wspace=0.16)
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
            ax.set_xlabel("tangential u [m]"); ax.set_ylabel("tangential v  (gravity ↓) [m]")
            su, sv = pshear[name][f]
            n = (su * su + sv * sv) ** 0.5
            if n > 1e-6:
                a_u, a_v = su / n * (e[1] - e[0]) * 0.42, sv / n * (e[3] - e[2]) * 0.42
                arr = ax.arrow(0.0, 0.0, a_u, a_v, color="white", width=0.00025,
                               head_width=0.0013, length_includes_head=True, zorder=8)
                arr.set_path_effects([pe.withStroke(linewidth=2.0, foreground="black")])
            # projected pen outline (stadium) — does the pressure sit where the pen is?
            q0, q1 = proj_pen[name][f]
            dv = q1 - q0
            Ln = np.linalg.norm(dv)
            if Ln > 1e-9:
                dvn = dv / Ln
                perp = np.array([-dvn[1], dvn[0]]) * pen_r
                ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color="cyan", lw=1.1, ls="--", zorder=7)
                for s in (1, -1):
                    ax.plot([q0[0] + s * perp[0], q1[0] + s * perp[0]],
                            [q0[1] + s * perp[1], q1[1] + s * perp[1]], color="cyan", lw=1.3, zorder=7)
            for cc in (q0, q1):
                ax.add_patch(plt.Circle((cc[0], cc[1]), pen_r, fill=False, color="cyan", lw=1.3, zorder=7))
            # gravity reference (always straight down) + legend
            gx0 = e[0] + 0.13 * (e[1] - e[0])
            gy0 = e[3] - 0.10 * (e[3] - e[2])
            glen = 0.18 * (e[3] - e[2])
            ax.annotate("", xy=(gx0, gy0 - glen), xytext=(gx0, gy0),
                        arrowprops=dict(arrowstyle="-|>", color="gold", lw=2.0))
            ax.text(gx0 + 0.03 * (e[1] - e[0]), gy0 - 0.5 * glen, "g", color="gold",
                    fontsize=11, va="center", fontweight="bold")
            ax.text(0.98, 0.02, "white=shear  cyan=pen", transform=ax.transAxes, ha="right",
                    va="bottom", color="white", fontsize=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3])  # pen would expand the view
        cb = fig.colorbar(ims[0], ax=[axL, axR], fraction=0.046, pad=0.02)
        cb.set_label("contact pressure ≈ kh·penetration  [MPa]")

        axF = fig.add_subplot(gs[2, :])
        axF.plot(t, forces["normal"][0], color="tab:red", label="Fn left")
        axF.plot(t, forces["normal"][1], color="tab:orange", label="Fn right")
        axF.plot(t, forces["shear"][0], color="tab:blue", ls="--", label="Ft left")
        axF.plot(t, forces["shear"][1], color="tab:cyan", ls="--", label="Ft right")
        axF.axvline(t[f], color="k", lw=1.2)
        axF.set_xlabel("time [s]"); axF.set_ylabel("force [N]")
        axF.set_title("grip force vs time"); axF.legend(fontsize=8, ncol=2); axF.grid(alpha=0.3)

        # shear force ON the pen from each finger (world frame, no pad-frame projection)
        axS = fig.add_subplot(gs[3, :])
        axS.plot(t, pen_sLz, color="tab:red", label="from left")
        axS.plot(t, pen_sRz, color="tab:blue", label="from right")
        axS.plot(t, pen_netz, color="black", lw=2, label="net (sum)")
        axS.axhline(0, color="gray", lw=0.6)
        axS.axvline(t[f], color="k", lw=1.2)
        axS.set_xlabel("time [s]"); axS.set_ylabel("shear on pen · up  [N]")
        axS.legend(fontsize=8, ncol=3, loc="upper left"); axS.grid(alpha=0.3)
        axS.set_title("shear on pencil from each pad (world frame)   cos∠(L,R): +1 same dir, −1 opposite")
        axS2 = axS.twinx()
        axS2.plot(t, pen_cos, color="tab:green", lw=1.0)
        axS2.set_ylim(-1.15, 1.15); axS2.set_ylabel("cos∠(L,R)", color="tab:green")
        axS2.tick_params(axis="y", labelcolor="tab:green")
        axS2.axhline(0, color="tab:green", ls=":", lw=0.6)

        fig.suptitle(f"panda_hydro tactile — frame {f}/{args.frames}  t={t[f]:.2f}s", fontsize=13)
        fig.savefig(os.path.join(args.outdir, f"frame_{f:04d}.png"), dpi=96, bbox_inches="tight")
        plt.close(fig)

    cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(args.outdir, "frame_%04d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", args.mp4]
    subprocess.run(cmd, check=True, capture_output=True)
    # clean raw scene frames
    for fn in os.listdir(sceneraw):
        os.remove(os.path.join(sceneraw, fn))
    os.rmdir(sceneraw)
    print(f"wrote {args.mp4}  ({args.frames} frames)")


if __name__ == "__main__":
    main()
