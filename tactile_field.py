# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Tactile field readout for the G1-in-SAGE scene.
#
# Extracts the three signals a vision-tactile policy would train on, per frame:
#
#   pressure       hydroelastic contact surface -- a dense per-face field, not a
#                  single vector. p = kh * depth at each face centroid, from
#                  CollisionPipeline.hydroelastic_sdf.get_contact_surface().
#   normal force   per contact, contacts.force projected on the contact normal
#                  (the solved force, pulled back from MuJoCo).
#   slip velocity  tangential relative surface speed at each contact:
#                  v_p = v + w x (p - com), slip = v_rel - (v_rel.n) n.
#
# Each is split robot<->furniture vs robot<->floor, because they are different
# signals: furniture contact is touch, floor contact is locomotion.
#
#   uv run python tactile_field.py --scene <layout.json> --frames 150 \
#       --command 1.0,0.0,0.0 --spawn 3.0,0.15 --out renders/tactile.npz
###########################################################################

from __future__ import annotations

import os
import sys

import numpy as np


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate row-vectors ``v`` by xyzw quaternions ``q``."""
    u, w = q[:, :3], q[:, 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _point_velocity(bodies: np.ndarray, pts: np.ndarray, body_q: np.ndarray, body_qd: np.ndarray, com: np.ndarray):
    """World-space velocity of ``pts`` on ``bodies``; zero where body < 0 (static shape).

    Newton stores body_qd as (linear, angular) about the body COM, so
    ``v_p = v + w x (p - com_world)`` (see newton.math.velocity_at_point).
    """
    out = np.zeros_like(pts)
    m = bodies >= 0
    if m.any():
        b = bodies[m]
        com_w = body_q[b, :3] + _quat_rotate(body_q[b, 3:7], com[b])
        v, w = body_qd[b, :3], body_qd[b, 3:6]
        out[m] = v + np.cross(w, pts[m] - com_w)
    return out


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import newton.examples

    from example_g1_in_sage import _make_parser, build, policy_step

    parser = _make_parser()
    parser.add_argument("--out", default="renders/tactile.npz", help="npz trace of the per-frame fields")
    parser.add_argument("--snapshot", type=int, default=-1, help="frame whose full contact surface is saved (-1 = peak)")
    parser.set_defaults(viewer="null", record=None)
    viewer, args = newton.examples.init(parser)

    ex = build(args, viewer)
    model, contacts = ex.model, ex.contacts
    if ex.collision_pipeline is None or ex.collision_pipeline.hydroelastic_sdf is None:
        sys.exit("[tactile] no hydroelastic pipeline — run without --contacts mujoco")
    hsdf = ex.collision_pipeline.hydroelastic_sdf

    shape_body = model.shape_body.numpy()
    body_com = model.body_com.numpy()
    kh = model.shape_material_kh.numpy()
    robot = np.zeros(model.shape_count, dtype=bool)
    robot[: ex.n_robot_shapes] = True
    furniture = np.zeros(model.shape_count, dtype=bool)
    furniture[np.asarray(ex.obj_shapes, dtype=int)] = True
    print(f"[tactile] kh: robot {kh[: ex.n_robot_shapes].min():.2e}..{kh[: ex.n_robot_shapes].max():.2e}, "
          f"furniture {kh[np.asarray(ex.obj_shapes, dtype=int)].min():.2e}", flush=True)

    if getattr(args, "settle", 0) > 0:
        import torch

        saved = ex.command.clone()
        ex.command = torch.zeros_like(saved)
        for _ in range(args.settle):
            policy_step(ex)
        ex.command = saved
        print(f"[tactile] settled {args.settle} frames", flush=True)

    F = args.frames
    z = lambda: np.zeros(F)  # noqa: E731
    tr = {
        "faces_furn": z(), "faces_floor": z(),
        "press_peak_furn": z(), "press_mean_furn": z(), "press_peak_floor": z(),
        "area_furn": z(), "area_floor": z(),
        "fn_furn": z(), "fn_floor": z(),
        "slip_peak_furn": z(), "slip_mean_furn": z(), "slip_peak_floor": z(),
    }
    best = {"press": -1.0, "frame": -1, "pts": None, "press_face": None}

    for f in range(F):
        policy_step(ex)
        bq, bqd = ex.state_0.body_q.numpy(), ex.state_0.body_qd.numpy()

        # ---- pressure: hydroelastic contact surface (3 verts per face) ----
        cs = hsdf.get_contact_surface()
        nfaces = int(cs.face_contact_count.numpy()[0]) if cs is not None else 0
        if nfaces:
            pair = cs.contact_surface_shape_pair.numpy()[:nfaces]
            depth = cs.contact_surface_depth.numpy()[:nfaces].astype(np.float64)
            verts = cs.contact_surface_point.numpy()[: nfaces * 3].reshape(nfaces, 3, 3).astype(np.float64)
            a, b = pair[:, 0], pair[:, 1]
            # pressure balances across the surface, so either side's kh gives the same value
            press = np.abs(depth) * kh[a]
            e0, e1 = verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0]
            area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=-1)
            fm = (robot[a] | robot[b]) & (furniture[a] | furniture[b])
            gm = (robot[a] | robot[b]) & ((a == ex.ground_shape) | (b == ex.ground_shape))
            for tag, m in (("furn", fm), ("floor", gm)):
                tr[f"faces_{tag}"][f] = m.sum()
                tr[f"area_{tag}"][f] = area[m].sum()
                tr[f"press_peak_{tag}"][f] = press[m].max() if m.any() else 0.0
            tr["press_mean_furn"][f] = (
                (press[fm] * area[fm]).sum() / area[fm].sum() if fm.any() and area[fm].sum() > 0 else 0.0
            )
            if fm.any() and press[fm].max() > best["press"]:
                best.update(press=float(press[fm].max()), frame=f,
                            pts=verts[fm].mean(axis=1).copy(), press_face=press[fm].copy())

        # ---- normal force + slip velocity: per solved contact ----
        n = int(contacts.rigid_contact_count.numpy()[0])
        if n:
            s0 = contacts.rigid_contact_shape0.numpy()[:n]
            s1 = contacts.rigid_contact_shape1.numpy()[:n]
            nrm = contacts.rigid_contact_normal.numpy()[:n].astype(np.float64)
            b0 = np.where(s0 >= 0, shape_body[np.clip(s0, 0, None)], -1)
            b1 = np.where(s1 >= 0, shape_body[np.clip(s1, 0, None)], -1)
            p0 = contacts.rigid_contact_point0.numpy()[:n].astype(np.float64)
            p1 = contacts.rigid_contact_point1.numpy()[:n].astype(np.float64)
            w0, w1 = p0.copy(), p1.copy()
            for pts, bods, out in ((p0, b0, w0), (p1, b1, w1)):
                m = bods >= 0
                if m.any():
                    tf = bq[bods[m]]
                    out[m] = tf[:, :3] + _quat_rotate(tf[:, 3:7], pts[m])
            mid = 0.5 * (w0 + w1)

            fn = np.zeros(n)
            if getattr(contacts, "force", None) is not None:
                fvec = contacts.force.numpy()[:n, :3].astype(np.float64)
                fn = np.abs(np.einsum("ij,ij->i", fvec, nrm))

            v0 = _point_velocity(b0, mid, bq, bqd, body_com)
            v1 = _point_velocity(b1, mid, bq, bqd, body_com)
            vrel = v1 - v0
            slip = np.linalg.norm(vrel - np.einsum("ij,ij->i", vrel, nrm)[:, None] * nrm, axis=-1)

            fmask = (robot[s0] | robot[s1]) & (furniture[s0] | furniture[s1])
            gmask = (robot[s0] | robot[s1]) & ((s0 == ex.ground_shape) | (s1 == ex.ground_shape))
            for tag, m in (("furn", fmask), ("floor", gmask)):
                tr[f"fn_{tag}"][f] = fn[m].sum() if m.any() else 0.0
                tr[f"slip_peak_{tag}"][f] = slip[m].max() if m.any() else 0.0
            tr["slip_mean_furn"][f] = (
                (slip[fmask] * fn[fmask]).sum() / fn[fmask].sum() if fmask.any() and fn[fmask].sum() > 1e-9 else 0.0
            )

        if f % 25 == 0:
            print(
                f"[tactile] frame {f}/{F} faces={int(tr['faces_furn'][f])}f/{int(tr['faces_floor'][f])}g "
                f"p_furn={tr['press_peak_furn'][f]:.3g}Pa Fn_furn={tr['fn_furn'][f]:.1f}N "
                f"slip_furn={tr['slip_peak_furn'][f]:.3f}m/s",
                flush=True,
            )

    print("\n[tactile] ===== rollout summary (robot<->furniture) =====", flush=True)
    print(f"[tactile]   peak pressure     {tr['press_peak_furn'].max():.4g} Pa", flush=True)
    print(f"[tactile]   peak contact area {tr['area_furn'].max() * 1e4:.2f} cm^2", flush=True)
    print(f"[tactile]   peak normal force {tr['fn_furn'].max():.1f} N", flush=True)
    print(f"[tactile]   peak slip speed   {tr['slip_peak_furn'].max():.4f} m/s", flush=True)
    print(f"[tactile]   frames in contact {int((tr['faces_furn'] > 0).sum())}/{F}", flush=True)
    print("[tactile] ----- robot<->floor -----", flush=True)
    print(f"[tactile]   peak pressure     {tr['press_peak_floor'].max():.4g} Pa", flush=True)
    print(f"[tactile]   peak normal force {tr['fn_floor'].max():.1f} N", flush=True)
    print(f"[tactile]   peak slip speed   {tr['slip_peak_floor'].max():.4f} m/s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    extra = {}
    if best["pts"] is not None:
        extra = {"snap_points": best["pts"], "snap_pressure": best["press_face"], "snap_frame": best["frame"]}
        print(f"[tactile] snapshot: frame {best['frame']}, {len(best['pts'])} faces, peak {best['press']:.4g} Pa")
    np.savez(args.out, **tr, **extra)
    print(f"[tactile] wrote {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
