# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Single-frame tactile diagnostic for a chosen frame:
#   - interaction scene render
#   - 2D gravity-aligned pressure heatmaps (left/right pad)
#   - 3D scatter of the contact points in WORLD coordinates, colorized by pressure,
#     with gravity (gray), pad-normal (green), and shear-force (white) vectors drawn
#   - grip force-vs-time trace
#
# Usage: python tactile_frame420.py --frame 420 --world-count 1 --scene pen --device cuda:0

import pyglet  # noqa: E402

pyglet.options["headless"] = True

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

import newton  # noqa: E402
import newton.examples  # noqa: E402
from newton.geometry import HydroelasticSDF  # noqa: E402
from newton.sensors import SensorContact  # noqa: E402
from newton.examples.robot.example_robot_panda_hydro import Example  # noqa: E402

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

KH = 1e11


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


def tmul(a, b):  # compose transforms a∘b (7-vectors px,py,pz,qx,qy,qz,qw)
    return np.concatenate([a[:3] + quat_rot(a[3:7], b[:3]), quat_mul(a[3:7], b[3:7])])


def main():
    parser = Example.create_parser()
    parser.add_argument("--frame", type=int, default=420)
    parser.set_defaults(viewer="gl", headless=True)
    viewer, args = newton.examples.init(parser)
    ex = Example(viewer, args)
    ex.show_isosurface = True
    try:
        ex.viewer.show_hydro_contact_surface = True
    except Exception:
        pass
    ex.viewer.set_camera(pos=wp.vec3(0.38, -0.12, 0.46), pitch=-26, yaw=-116)

    m = ex.model
    labels = m.body_key if hasattr(m, "body_key") else m.body_label
    sb = m.shape_body.numpy()

    def fget(sub):
        bl = [i for i, l in enumerate(labels) if sub in l]
        return bl[0], set(np.where(np.isin(sb, bl))[0].tolist())

    lb, lsh = fget("leftfinger")
    rb, rsh = fget("rightfinger")
    fingers = [("left", lb, lsh), ("right", rb, rsh)]
    obj = [i for i, l in enumerate(labels) if l.endswith("object")][0]
    obj_shape = int(np.where(sb == obj)[0][0])  # the pen capsule
    sensor = SensorContact(m, sensing_bodies=["*leftfinger*", "*rightfinger*"], counterpart_bodies="object")
    # second sensor: shear ON the pen FROM each finger (world frame, no pad projection)
    sensor_pen = SensorContact(m, sensing_bodies="object", counterpart_bodies=["*leftfinger*", "*rightfinger*"])
    ci = sensor_pen.counterpart_indices[0]
    col_L, col_R = ci.index(lb), ci.index(rb)
    contacts = ex.contacts
    hsdf = ex.collision_pipeline.hydroelastic_sdf
    gW = np.array([0, 0, -1.0])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = LinearSegmentedColormap.from_list(
        "tactile", ["#00004d", "#0010c0", "#00c8ff", "#00ff66", "#ffe000", "#ff2a00"])

    fn_hist = [[], []]
    ft_hist = [[], []]
    pen_sL, pen_sR, pen_net = [], [], []  # shear on pen from left/right finger + net (world)
    scene = None
    pad_data = {}
    for f in range(args.frame + 1):
        ex.step()
        ex.render()
        ex.solver.update_contacts(contacts)
        sensor.update(ex.state_0, contacts)
        tf = sensor.total_force.numpy()
        tfr = sensor.total_force_friction.numpy()
        for fi in range(2):
            fn_hist[fi].append(float(np.linalg.norm(tf[fi] - tfr[fi])))
            ft_hist[fi].append(float(np.linalg.norm(tfr[fi])))
        sensor_pen.update(ex.state_0, contacts)
        fmf = sensor_pen.force_matrix_friction.numpy()[0]
        pen_sL.append(fmf[col_L].copy())
        pen_sR.append(fmf[col_R].copy())
        pen_net.append(sensor_pen.total_force_friction.numpy()[0].copy())
        if f == args.frame:
            scene = ex.viewer.get_frame().numpy()
            cs = hsdf.get_contact_surface()
            nf = int(cs.face_contact_count.numpy()[0])
            verts = cs.contact_surface_point.numpy()[: 3 * nf].reshape(nf, 3, 3)
            depth = cs.contact_surface_depth.numpy()[:nf]
            sp = cs.contact_surface_shape_pair.numpy()[:nf]
            bq = ex.state_0.body_q.numpy()
            # pen capsule world endpoints (length along local Z; scale=(r, half_h, r))
            st = m.shape_transform.numpy()[obj_shape]
            ssc = m.shape_scale.numpy()[obj_shape]
            sw = tmul(bq[obj], st)
            axis_w = quat_rot(sw[3:7], np.array([0.0, 0.0, 1.0]))
            pen_r = float(ssc[0])
            pen_e0 = sw[:3] - float(ssc[1]) * axis_w
            pen_e1 = sw[:3] + float(ssc[1]) * axis_w
            for fi, (name, fbody, fsh) in enumerate(fingers):
                mask = np.array([(a in fsh or b in fsh) for a, b in sp], dtype=bool)
                cen_w = verts[mask].mean(axis=1)
                dep = depth[mask]
                q = bq[fbody][3:7]
                loc = quat_rot(quat_conj(q), cen_w - bq[fbody][:3])
                na = int(np.argmin(loc.var(axis=0)))
                n_hat = np.zeros(3); n_hat[na] = 1.0
                gl = quat_rot(quat_conj(q), gW)
                gp = gl - np.dot(gl, n_hat) * n_hat
                v_hat = -gp / (np.linalg.norm(gp) + 1e-12)
                u_hat = np.cross(n_hat, v_hat); u_hat /= np.linalg.norm(u_hat) + 1e-12
                pad_data[name] = dict(
                    cen_w=cen_w, dep=dep, center=cen_w.mean(0),
                    n_world=quat_rot(q, n_hat), v_world=quat_rot(q, v_hat), u_world=quat_rot(q, u_hat),
                    shear_world=tfr[fi], fn=fn_hist[fi][-1], ft=ft_hist[fi][-1],
                    loc=loc, u_hat=u_hat, v_hat=v_hat,
                )

    pmax = 1.0
    for name in pad_data:
        d = np.clip(-pad_data[name]["dep"], 0, None) * KH / 1e6
        if d.size and d.max() > 0:
            pmax = max(pmax, float(np.percentile(d[d > 0], 99)))

    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.3, 1.15, 1.25, 0.6], hspace=0.32, wspace=0.18)
    ax_scene = fig.add_subplot(gs[0, :])
    ax_scene.imshow(scene[40:980, 220:1760]); ax_scene.axis("off")
    ax_scene.set_title(f"interaction scene — frame {args.frame}  (green = hydroelastic contact patch)")

    # 2D gravity-aligned heatmaps
    for col, name in enumerate(["left", "right"]):
        ax = fig.add_subplot(gs[1, col])
        d = pad_data[name]
        uv = (d["loc"] - d["loc"].mean(0)) @ np.stack([d["u_hat"], d["v_hat"]], axis=1)
        press = np.clip(-d["dep"], 0, None) * KH / 1e6
        half = max(np.abs(uv).max() if uv.size else 0.006, 0.006) * 1.1
        gx = np.linspace(-half, half, 160)
        GX, GY = np.meshgrid(gx, gx)
        field = np.zeros_like(GX)
        if len(uv) >= 4:
            z = griddata(uv, press, (GX, GY), method="linear", fill_value=0.0)
            field = np.nan_to_num(np.clip(z, 0, pmax))
        im = ax.imshow(field, origin="lower", extent=(-half, half, -half, half), cmap=cmap,
                       vmin=0, vmax=pmax, interpolation="bilinear", aspect="equal")
        # discrete evaluation points: one per contact-surface face (centroid)
        ax.scatter(uv[:, 0], uv[:, 1], s=9, c="white", edgecolors="black", linewidths=0.3, zorder=8)
        print(f"{name} pad: {len(uv)} contact-surface evaluation points (faces)")
        su, sv = np.dot(d["shear_world"], d["u_world"]), np.dot(d["shear_world"], d["v_world"])
        nrm = (su * su + sv * sv) ** 0.5
        if nrm > 1e-6:
            a = np.array([su, sv]) / nrm * half * 0.42
            arr = ax.arrow(0, 0, a[0], a[1], color="white", width=0.00025, head_width=0.0013,
                           length_includes_head=True, zorder=6)
            arr.set_path_effects([pe.withStroke(linewidth=2.0, foreground="black")])
        ax.annotate("", xy=(-half * 0.75, -half * 0.55), xytext=(-half * 0.75, -half * 0.2),
                    arrowprops=dict(arrowstyle="-|>", color="gold", lw=2))
        ax.text(-half * 0.68, -half * 0.4, "g", color="gold", fontweight="bold")
        # project the pen capsule outline (stadium) onto this pad plane
        c0 = d["center"]
        a0 = np.array([np.dot(pen_e0 - c0, d["u_world"]), np.dot(pen_e0 - c0, d["v_world"])])
        a1 = np.array([np.dot(pen_e1 - c0, d["u_world"]), np.dot(pen_e1 - c0, d["v_world"])])
        dv = a1 - a0
        Ln = np.linalg.norm(dv)
        if Ln > 1e-9:
            dvn = dv / Ln
            pp = np.array([-dvn[1], dvn[0]]) * pen_r
            ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color="cyan", lw=1.3, ls="--", zorder=7)
            for s in (1, -1):
                ax.plot([a0[0] + s * pp[0], a1[0] + s * pp[0]], [a0[1] + s * pp[1], a1[1] + s * pp[1]],
                        color="cyan", lw=1.6, zorder=7)
        for cc in (a0, a1):
            ax.add_patch(plt.Circle((cc[0], cc[1]), pen_r, fill=False, color="cyan", lw=1.6, zorder=7))
        ax.plot([], [], color="cyan", label="pen outline")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)  # keep patch-scale view (pen would expand it)
        ax.set_title(f"{name} pad 2D (gravity ↓)  Fn={d['fn']:.1f} Ft={d['ft']:.1f} N  ·  {len(uv)} eval pts (white)")
        ax.set_xlabel("u [m]"); ax.set_ylabel("v [m]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("pressure [MPa]")

    # 3D scatter in WORLD coordinates, colorized by pressure
    for col, name in enumerate(["left", "right"]):
        ax = fig.add_subplot(gs[2, col], projection="3d")
        d = pad_data[name]
        P = d["cen_w"]; c = d["center"]
        press = np.clip(-d["dep"], 0, None) * KH / 1e6
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=press, cmap=cmap, vmin=0, vmax=pmax, s=18)
        L = 0.012
        # gravity (gray), pad normal (green), shear (white), up=v (gold)
        ax.quiver(*c, *(gW * L), color="gray", linewidth=2)
        ax.quiver(*c, *(d["n_world"] * L * 0.8), color="limegreen", linewidth=2)
        sh = d["shear_world"]; shn = sh / (np.linalg.norm(sh) + 1e-9)
        ax.quiver(*c, *(shn * L), color="white", linewidth=2)
        ax.text(*(c + gW * L), "g", color="gray")
        ax.text(*(c + d["n_world"] * L * 0.8), "n", color="green")
        ax.text(*(c + shn * L), "shear", color="black")
        ax.set_title(f"{name} pad 3D (world)  shear_z={sh[2]:+.2f} N")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (up)")
        rng = 0.018
        ax.set_xlim(c[0] - rng, c[0] + rng); ax.set_ylim(c[1] - rng, c[1] + rng); ax.set_zlim(c[2] - rng, c[2] + rng)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    t = np.arange(args.frame + 1) / ex.fps
    axF = fig.add_subplot(gs[3, 0])
    axF.plot(t, fn_hist[0], color="tab:red", label="Fn left")
    axF.plot(t, fn_hist[1], color="tab:orange", label="Fn right")
    axF.plot(t, ft_hist[0], color="tab:blue", ls="--", label="Ft left")
    axF.plot(t, ft_hist[1], color="tab:cyan", ls="--", label="Ft right")
    axF.axvline(t[-1], color="k", lw=1.2)
    axF.set_xlabel("time [s]"); axF.set_ylabel("force [N]"); axF.legend(fontsize=8, ncol=2); axF.grid(alpha=0.3)
    axF.set_title("grip force vs time")

    # shear force ON the pen from each finger (world frame, no pad-frame projection)
    up = -gW
    sL = np.array(pen_sL); sR = np.array(pen_sR); net = np.array(pen_net)
    sLz, sRz, netz = sL @ up, sR @ up, net @ up
    nL, nR = np.linalg.norm(sL, axis=1), np.linalg.norm(sR, axis=1)
    cosang = np.where((nL > 1e-6) & (nR > 1e-6), np.einsum("ij,ij->i", sL, sR) / (nL * nR + 1e-12), np.nan)
    axS = fig.add_subplot(gs[3, 1])
    axS.plot(t, sLz, color="tab:red", label="from left")
    axS.plot(t, sRz, color="tab:blue", label="from right")
    axS.plot(t, netz, color="black", lw=2, label="net (sum)")
    axS.axhline(0, color="gray", lw=0.6)
    axS.axvline(t[-1], color="k", lw=1.2)
    axS.set_xlabel("time [s]"); axS.set_ylabel("shear on pen · up  [N]")
    axS.legend(fontsize=7, ncol=2, loc="upper left"); axS.grid(alpha=0.3)
    axS.set_title("shear on pencil from each pad (world frame)")
    axS2 = axS.twinx()
    axS2.plot(t, cosang, color="tab:green", lw=1.0)
    axS2.set_ylim(-1.15, 1.15); axS2.set_ylabel("cos∠(L,R)", color="tab:green")
    axS2.tick_params(axis="y", labelcolor="tab:green")
    axS2.axhline(0, color="tab:green", ls=":", lw=0.6)

    fig.suptitle(f"panda_hydro tactile diagnostic — frame {args.frame}", fontsize=14)
    out = f"tactile_frame{args.frame}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
