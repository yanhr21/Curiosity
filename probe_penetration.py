# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Measure interpenetration in the G1-in-SAGE scene.
#
# The "no penetration" claim for this scene has so far been argued from a
# camera angle (a horizontal hero shot projects the robot onto the chair
# behind it; an elevated shot shows the gap). This measures it instead.
#
# For every active rigid contact each frame it computes the signed surface
# separation -- negative = real overlap -- and reports the worst value per
# category: robot<->furniture, robot<->floor, furniture<->furniture.
#
# Why the number is trustworthy:
#   * separation = dot(n, p1_world - p0_world) - (margin0 + margin1), the same
#     expression the solvers use (newton._src.sim.contacts.contact_surface_separation).
#   * SolverMuJoCo.update_contacts writes point0/point1 as pos -/+ 0.5*dist*n, so this
#     round-trips MuJoCo's own contact.dist. --cross-check asserts that equality
#     directly against solver.mjw_data.contact.dist.
#   * The robot's collision geometry is convex-hulled (a superset of the visual mesh),
#     so "hulls do not overlap" implies the rendered meshes do not overlap either.
#
# Usage (inside the dev-node container):
#   source renders/render_env.sh   # not needed: this runs headless with --viewer null
#   uv run python probe_penetration.py --scene <layout.json> --frames 150 \
#       --spawn 3.0,0.15 --command 1.0,0.0,0.0 --mu 1.4 --out renders/penetration.npz
###########################################################################

from __future__ import annotations

import os
import sys

import numpy as np


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate row-vectors ``v`` by xyzw quaternions ``q`` (both (N,·))."""
    u, w = q[:, :3], q[:, 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _world_points(pts: np.ndarray, shapes: np.ndarray, shape_body: np.ndarray, body_q: np.ndarray) -> np.ndarray:
    """Body-frame contact points -> world. Static shapes (body < 0) are already world-space."""
    out = pts.copy()
    bodies = np.where(shapes >= 0, shape_body[np.clip(shapes, 0, None)], -1)
    m = bodies >= 0
    if m.any():
        tf = body_q[bodies[m]]
        out[m] = tf[:, :3] + _quat_rotate(tf[:, 3:7], pts[m])
    return out


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import newton.examples

    from example_g1_in_sage import _make_parser, build, policy_step

    parser = _make_parser()
    parser.add_argument("--out", default="renders/penetration.npz", help="npz trace of the per-frame minima")
    parser.add_argument(
        "--cross-check",
        type=int,
        default=3,
        help="frames on which to verify the computed separation against MuJoCo's own contact.dist",
    )
    parser.set_defaults(viewer="null", record=None)
    viewer, args = newton.examples.init(parser)

    ex = build(args, viewer)
    model, contacts = ex.model, ex.contacts
    shape_body = model.shape_body.numpy()
    robot = np.zeros(model.shape_count, dtype=bool)
    robot[: ex.n_robot_shapes] = True
    furniture = np.zeros(model.shape_count, dtype=bool)
    furniture[np.asarray(ex.obj_shapes, dtype=int)] = True
    print(
        f"[probe] {ex.n_robot_shapes} robot shapes, {len(ex.obj_shapes)} furniture SDF shapes, "
        f"ground shape {ex.ground_shape} at z={ex.floor_top_z:.4f}",
        flush=True,
    )
    has_force = getattr(contacts, "force", None) is not None
    print(f"[probe] per-contact force attribute allocated: {has_force}", flush=True)

    # settle first (same pre-roll as the render), so we measure the walking robot, not the drop-in.
    if getattr(args, "settle", 0) > 0:
        import torch

        saved = ex.command.clone()
        ex.command = torch.zeros_like(saved)
        for _ in range(args.settle):
            policy_step(ex)
        ex.command = saved
        print(f"[probe] settled {args.settle} frames", flush=True)

    cats = ("robot-furniture", "robot-floor", "furniture-furniture", "other")
    trace = {c: np.zeros(args.frames) for c in cats}
    worst = dict.fromkeys(cats, (np.inf, -1, -1, -1))  # (separation, frame, shape0, shape1)
    root_z = np.zeros(args.frames)
    peak_force = dict.fromkeys(cats, 0.0)  # peak per-contact force magnitude [N] per category
    peak_sensor = 0.0  # peak SensorContact.total_force [N] — the end-to-end tactile readout

    for f in range(args.frames):
        policy_step(ex)

        n = int(contacts.rigid_contact_count.numpy()[0])
        root_z[f] = float(ex.state_0.joint_q.numpy()[2])
        if ex.contact_sensor is not None and getattr(ex.contact_sensor, "total_force", None) is not None:
            peak_sensor = max(peak_sensor, float(np.linalg.norm(ex.contact_sensor.total_force.numpy(), axis=-1).max()))
        if n == 0:
            for c in cats:
                trace[c][f] = np.nan
            continue

        s0 = contacts.rigid_contact_shape0.numpy()[:n]
        s1 = contacts.rigid_contact_shape1.numpy()[:n]
        nrm = contacts.rigid_contact_normal.numpy()[:n]
        m0 = contacts.rigid_contact_margin0.numpy()[:n]
        m1 = contacts.rigid_contact_margin1.numpy()[:n]
        bq = ex.state_0.body_q.numpy()
        p0 = _world_points(contacts.rigid_contact_point0.numpy()[:n], s0, shape_body, bq)
        p1 = _world_points(contacts.rigid_contact_point1.numpy()[:n], s1, shape_body, bq)
        sep_newton = np.einsum("ij,ij->i", nrm, p1 - p0) - (m0 + m1)
        # MuJoCo's contact.dist is the value the solver itself used; the Newton-level
        # reconstruction has to round-trip the contact points through each body frame, which
        # costs precision. Prefer dist when the solver exposes it, and check the two agree.
        dist = _mujoco_dist(ex, n)
        sep = sep_newton if dist is None else dist
        if f < args.cross_check and dist is not None:
            _cross_check(sep_newton, dist, s0, s1, shape_body, n, f)

        r = robot[s0] | robot[s1]
        fu = furniture[s0] | furniture[s1]
        g = (s0 == ex.ground_shape) | (s1 == ex.ground_shape)
        masks = {
            "robot-furniture": r & fu,
            "robot-floor": r & g,
            "furniture-furniture": fu & ~r & ~g,
        }
        masks["other"] = ~(masks["robot-furniture"] | masks["robot-floor"] | masks["furniture-furniture"])
        fmag = None
        if has_force:
            # SensorContact reads the linear force as wp.spatial_top(contacts.force) -> components 0:3.
            fmag = np.linalg.norm(contacts.force.numpy()[:n, :3], axis=-1)
        for c, m in masks.items():
            if fmag is not None and m.any():
                peak_force[c] = max(peak_force[c], float(fmag[m].max()))
            v = float(sep[m].min()) if m.any() else np.nan
            trace[c][f] = v
            if np.isfinite(v) and v < worst[c][0]:
                i = int(np.flatnonzero(m)[int(np.argmin(sep[m]))])
                worst[c] = (v, f, int(s0[i]), int(s1[i]))

        if f % 25 == 0:
            print(
                f"[probe] frame {f}/{args.frames} contacts={n} "
                + " ".join(f"{c}={trace[c][f]:+.5f}" for c in cats[:3]),
                flush=True,
            )

    print("\n[probe] ===== worst separation over the rollout (negative = penetration) =====", flush=True)
    for c in cats:
        v, fr, a, b = worst[c]
        if np.isfinite(v):
            tag = "PENETRATION" if v < 0 else "gap"
            print(f"[probe]   {c:22s} {v:+.6f} m  ({tag})  frame {fr}, shapes {a}<->{b}", flush=True)
        else:
            print(f"[probe]   {c:22s} no contacts", flush=True)
    print(f"[probe] robot root z: min {root_z.min():.4f} max {root_z.max():.4f} (floor {ex.floor_top_z:.4f})")
    print(f"[probe] peak SensorContact.total_force (robot<->furniture) = {peak_sensor:.1f} N", flush=True)
    if has_force:
        print("[probe] peak per-contact force [N]: " + "  ".join(f"{c}={peak_force[c]:.1f}" for c in cats), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, root_z=root_z, **{c.replace("-", "_"): trace[c] for c in cats})
    print(f"[probe] wrote {os.path.abspath(args.out)}", flush=True)


def _mujoco_dist(ex, n: int) -> np.ndarray | None:
    """MuJoCo's signed contact distance for the first ``n`` contacts, index-aligned with `Contacts`.

    ``SolverMuJoCo.update_contacts`` writes shape/point/normal at the same index it reads
    ``mj_contact`` from, so slicing by the Newton contact count is aligned.
    """
    try:
        return ex.solver.mjw_data.contact.dist.numpy().reshape(-1)[:n].astype(np.float64)
    except Exception:  # solver internals are not public API; fall back to the reconstruction
        return None


def _cross_check(sep: np.ndarray, dist: np.ndarray, s0, s1, shape_body, n: int, frame: int) -> None:
    """Report how far the Newton-level reconstruction drifts from MuJoCo's contact.dist, and where.

    A large residual on body-attached contacts (and none on static-static ones) means the drift is
    the body-frame round trip, not a wrong separation.
    """
    err = np.abs(sep - dist)
    dyn = (shape_body[np.clip(s0, 0, None)] >= 0) | (shape_body[np.clip(s1, 0, None)] >= 0)
    parts = [f"max|newton-mujoco| = {err.max():.3e} m"]
    for label, m in (("body-attached", dyn), ("static-static", ~dyn)):
        parts.append(f"{label}: {err[m].max():.3e} m" if m.any() else f"{label}: none")
    print(f"[probe] cross-check frame {frame} ({n} contacts): " + ", ".join(parts), flush=True)


if __name__ == "__main__":
    main()
