# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Render a tactile "contact map" video for the panda_hydro grasp:
#   - per-finger contact-pressure heatmap (hydroelastic faces projected into the
#     pad's local plane, colored by pressure = kh * penetration_depth)
#   - aggregate shear (friction) force drawn as an arrow on each pad
#   - normal / shear force-vs-time traces with a moving cursor
#
# Usage: python tactile_video.py --viewer null --world-count 1 --scene pen --device cuda:0

import argparse
import os
import subprocess

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact
from newton.examples.robot.example_robot_panda_hydro import Example

# --- patches: emit contact surface + allocate per-contact force ---
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

KH = 1e11  # matches the example's hydroelastic stiffness; pressure proxy = KH * depth


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rot(q, v):
    xyz = q[:3]
    w = q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def to_local(world_pts, body_q):
    # inverse transform: local = R^T (p - t)
    t = body_q[:3]
    qc = quat_conj(body_q[3:7])
    return np.array([quat_rot(qc, p - t) for p in world_pts])


def main():
    parser = Example.create_parser()
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--outdir", default="tactile_frames")
    parser.add_argument("--mp4", default="tactile_panda.mp4")
    viewer, args = newton.examples.init(parser)
    ex_args = args
    ex = Example(viewer, ex_args)
    m = ex.model
    labels = m.body_key if hasattr(m, "body_key") else m.body_label
    shape_body = m.shape_body.numpy()

    def finger(sub):
        bset = [i for i, l in enumerate(labels) if sub in l]
        return bset[0], set(np.where(np.isin(shape_body, bset))[0].tolist())

    lb, lshapes = finger("leftfinger")
    rb, rshapes = finger("rightfinger")
    fingers = [("left", lb, lshapes), ("right", rb, rshapes)]

    sensor = SensorContact(m, sensing_bodies=["*leftfinger*", "*rightfinger*"], counterpart_bodies="object")
    contacts = ex.contacts
    hsdf = ex.collision_pipeline.hydroelastic_sdf

    # ---- pass 1: simulate and store per-frame tactile data ----
    store = {"left": [], "right": []}
    forces = {"normal": [[], []], "shear": [[], []], "shear_vec": [[], []]}
    for f in range(args.frames):
        ex.step()
        ex.solver.update_contacts(contacts)
        sensor.update(ex.state_0, contacts)
        cs = hsdf.get_contact_surface()
        nf = int(cs.face_contact_count.numpy()[0])
        pts = cs.contact_surface_point.numpy()[: 3 * nf].reshape(nf, 3, 3) if nf else np.zeros((0, 3, 3))
        depth = cs.contact_surface_depth.numpy()[:nf] if nf else np.zeros((0,))
        sp = cs.contact_surface_shape_pair.numpy()[:nf] if nf else np.zeros((0, 2), dtype=int)
        centroids = pts.mean(axis=1) if nf else np.zeros((0, 3))
        body_q = ex.state_0.body_q.numpy()
        tf = sensor.total_force.numpy()
        tfr = sensor.total_force_friction.numpy()
        for fi, (name, fb, fsh) in enumerate(fingers):
            mask = np.array([(a in fsh or b in fsh) for a, b in sp], dtype=bool) if nf else np.zeros(0, bool)
            loc = to_local(centroids[mask], body_q[fb]) if mask.any() else np.zeros((0, 3))
            store[name].append((loc, depth[mask] if nf else np.zeros(0)))
            forces["normal"][fi].append(float(np.linalg.norm(tf[fi] - tfr[fi])))
            forces["shear"][fi].append(float(np.linalg.norm(tfr[fi])))
            # shear vector in pad-local frame
            sv = quat_rot(quat_conj(body_q[fb][3:7]), tfr[fi])
            forces["shear_vec"][fi].append(sv)

    # ---- determine pad-plane axes (min-variance local axis = normal) ----
    axes_uv = {}
    for name, _, _ in fingers:
        allpts = np.concatenate([s[0] for s in store[name] if len(s[0])], axis=0) if any(len(s[0]) for s in store[name]) else np.zeros((1, 3))
        var = allpts.var(axis=0)
        normal_axis = int(np.argmin(var))
        uv = [i for i in range(3) if i != normal_axis]
        axes_uv[name] = uv
    # global extents + color scale
    ext = {}
    pmax = 0.0
    for name, _, _ in fingers:
        uv = axes_uv[name]
        pts = [s[0][:, uv] for s in store[name] if len(s[0])]
        d = [s[1] for s in store[name] if len(s[1])]
        if pts:
            allp = np.concatenate(pts); ext[name] = (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())
            pmax = max(pmax, float(np.percentile(np.concatenate(d), 98)) * KH)
        else:
            ext[name] = (-0.01, 0.01, -0.01, 0.01)
    pmax = max(pmax, 1.0)

    # ---- pass 2: render frames ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.outdir, exist_ok=True)
    t = np.arange(args.frames) / ex.fps
    for f in range(args.frames):
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
        for fi, (name, _, _) in enumerate(fingers):
            ax = axs[fi]
            uv = axes_uv[name]
            loc, dep = store[name][f]
            e = ext[name]
            ax.set_xlim(e[0] - 0.002, e[1] + 0.002); ax.set_ylim(e[2] - 0.002, e[3] + 0.002)
            ax.set_aspect("equal"); ax.set_title(f"{name} pad  |  Fn={forces['normal'][fi][f]:.1f} N  Ft={forces['shear'][fi][f]:.1f} N")
            ax.set_xlabel("pad u [m]"); ax.set_ylabel("pad v [m]")
            if len(loc):
                p = loc[:, uv]
                sc = ax.scatter(p[:, 0], p[:, 1], c=dep * KH / 1e6, s=60, cmap="inferno", vmin=0, vmax=pmax / 1e6, edgecolors="none")
                # shear arrow at patch centroid
                c = p.mean(axis=0)
                sv = forces["shear_vec"][fi][f][uv]
                n = np.linalg.norm(sv)
                if n > 1e-6:
                    a = sv / n * (e[1] - e[0]) * 0.4
                    ax.arrow(c[0], c[1], a[0], a[1], color="cyan", width=0.0004, head_width=0.0015, length_includes_head=True)
                if fi == 0:
                    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("pressure ≈ kh·depth [MPa]")
            else:
                ax.text(0.5, 0.5, "no contact", ha="center", va="center", transform=ax.transAxes, color="gray")

        ax = axs[2]
        ax.plot(t, forces["normal"][0], color="tab:red", label="Fn left")
        ax.plot(t, forces["normal"][1], color="tab:orange", label="Fn right")
        ax.plot(t, forces["shear"][0], color="tab:blue", ls="--", label="Ft left")
        ax.plot(t, forces["shear"][1], color="tab:cyan", ls="--", label="Ft right")
        ax.axvline(t[f], color="k", lw=1)
        ax.set_xlabel("time [s]"); ax.set_ylabel("force [N]"); ax.set_title("grip force vs time"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        fig.suptitle(f"panda_hydro tactile contact map — frame {f}/{args.frames}  t={t[f]:.2f}s", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(args.outdir, f"frame_{f:04d}.png"), dpi=110)
        plt.close(fig)

    # ---- encode ----
    cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(args.outdir, "frame_%04d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", args.mp4]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"wrote {args.mp4}  ({args.frames} frames)")


if __name__ == "__main__":
    main()
