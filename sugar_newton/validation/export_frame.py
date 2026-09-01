"""Dump one frame of the carry to inspectable 3D geometry, and audit the contact set.

Written to answer "is it really true that only two digits are touching the box?". A tactile
heat map cannot answer that on its own: it shows where force IS, never whether a silent
finger is resting on the box or waving 4 cm clear of it. So this also measures, per digit,
the actual surface-to-surface gap to the box in world space, which distinguishes

  * gap far above the ``--margin`` band  -> silence is correct, the finger is not touching
  * gap inside the band but no force      -> a real problem worth chasing

This is also what found the floating grasp: at the old 5 mm margin the loaded fingertips sat
4.5 mm off the box, because ``margin`` offsets the resting surface rather than only widening
detection. Pass ``--margin 0.005`` to reproduce that.

Outputs, all in world coordinates, at ``_out/frame_<n>/``:

    scene.ply             every collision mesh: hands blue, box tan, rest grey
    left_hand.ply         the two hand colliders on their own
    right_hand.ply
    box.ply
    contacts.ply          contact points, red = load-bearing, grey = zero-force candidate
    contacts_spheres.ply  the load-bearing ones as spheres, radius scaled by force
    contacts.csv          point, normal, force vector, magnitude, which digit

    python -m sugar_newton.validation.export_frame --frame 359
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import warp as wp

from sugar_newton.validation.g1_carrybox_policy import Actor, G1PolicyScene, load_clip
from sugar_newton.validation.hand_atlas import digit_bands, digit_mask

LOAD_N = 0.01           # above this a contact is carrying load, below it is a candidate


def quat_rotate(q, v):
    """Rotate ``v`` (N,3) by quaternion ``q`` (x, y, z, w)."""
    u, w = q[:3], q[3]
    t = 2.0 * np.cross(np.broadcast_to(u, v.shape), v)
    return v + w * t + np.cross(np.broadcast_to(u, t.shape), t)


def apply_xform(x, v):
    return quat_rotate(np.asarray(x[3:7], dtype=np.float64), v) + np.asarray(x[:3])


def shape_world(model, body_q, shape):
    """World-space (verts, tris) of one collision shape, or ``None`` if it has no mesh."""
    src = model.shape_source[shape]
    if src is None or getattr(src, "indices", None) is None:
        return None
    v = np.asarray(src.vertices, dtype=np.float64)
    t = np.asarray(src.indices).reshape(-1, 3)
    scale = getattr(model, "shape_scale", None)
    if scale is not None:
        v = v * np.asarray(scale.numpy()[shape], dtype=np.float64)
    v = apply_xform(model.shape_transform.numpy()[shape], v)
    body = int(model.shape_body.numpy()[shape])
    if body >= 0:
        v = apply_xform(body_q[body], v)
    return v, t


def write_mesh(path, chunks):
    """``chunks`` is a list of (verts, tris, rgb); writes one vertex-coloured PLY."""
    import open3d as o3d

    V, T, C, off = [], [], [], 0
    for v, t, c in chunks:
        V.append(v)
        T.append(t + off)
        C.append(np.tile(np.asarray(c, dtype=np.float64), (len(v), 1)))
        off += len(v)
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.concatenate(V))
    m.triangles = o3d.utility.Vector3iVector(np.concatenate(T))
    m.vertex_colors = o3d.utility.Vector3dVector(np.concatenate(C))
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data_000")
    ap.add_argument("--frame", type=int, default=359, help="clip frame to export")
    ap.add_argument("--start", type=int, default=60, help="frame the rollout resets to")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--box-tris", type=int, default=2000)
    ap.add_argument("--hand-tris", type=int, default=5000)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="collider surface thickness [m]")
    ap.add_argument("--outdir", default="sugar_newton/_out")
    args = ap.parse_args()

    wp.init()
    clip = load_clip(args.clip)
    dt = 1.0 / clip["fps"]
    scene = G1PolicyScene(clip, box_tris=args.box_tris, hand_tris=args.hand_tris,
                          margin=args.margin)
    actor = Actor()

    body_of = scene.model.shape_body.numpy()
    labels = [l.split("/")[-1] for l in scene.model.body_label]
    hands, box_shapes = {}, set()
    for s, b in enumerate(body_of):
        if b < 0:
            continue
        if labels[b] == "box":
            box_shapes.add(s)
        for side in ("left", "right"):
            if labels[b] == f"{side}_rubber_hand":
                hands.setdefault(side, set()).add(s)

    # Replay from the same reset the video uses, so the exported frame is the frame you saw.
    # make_loop_video labels the state after its k-th step as clip frame start + k, hence
    # the +1 here.
    scene.reset(args.start)
    steps = max(0, args.frame - args.start + 1)
    for k in range(steps):
        scene.apply(actor(scene.observe()))
        scene.step(dt, args.substeps, "step")
        if (k + 1) % 50 == 0:
            print(f"  stepped {k + 1}/{steps}", flush=True)
    scene.solver.update_contacts(scene.contacts, scene.state_0)
    print("  rollout done, writing geometry", flush=True)

    out = Path(args.outdir) / f"frame_{args.frame}"
    out.mkdir(parents=True, exist_ok=True)
    body_q = scene.state_0.body_q.numpy()

    chunks, hand_geo, box_geo = [], {}, []
    for s in range(len(body_of)):
        g = shape_world(scene.model, body_q, s)
        if g is None:
            continue
        side = next((k for k, v in hands.items() if s in v), None)
        if side is not None:
            hand_geo[side] = (s, *g)
            chunks.append((*g, (0.35, 0.55, 0.90)))
        elif s in box_shapes:
            box_geo.append(g)
            chunks.append((*g, (0.85, 0.72, 0.45)))
        else:
            chunks.append((*g, (0.72, 0.72, 0.74)))
    write_mesh(out / "scene.ply", chunks)
    for side, (_, v, t) in hand_geo.items():
        write_mesh(out / f"{side}_hand.ply", [(v, t, (0.35, 0.55, 0.90))])
    if box_geo:
        write_mesh(out / "box.ply", [(*g, (0.85, 0.72, 0.45)) for g in box_geo])
    print(f"  meshes written ({sum(len(t) for _, t, _ in chunks)} triangles)", flush=True)

    # ---- contacts, in world space ----
    import open3d as o3d

    c = scene.contacts
    n = int(c.rigid_contact_count.numpy()[0])
    s0 = c.rigid_contact_shape0.numpy()[:n]
    s1 = c.rigid_contact_shape1.numpy()[:n]
    p0 = c.rigid_contact_point0.numpy()[:n]
    p1 = c.rigid_contact_point1.numpy()[:n]
    nrm = c.rigid_contact_normal.numpy()[:n]
    f = c.force.numpy()[:n, :3] if c.force is not None else np.zeros((n, 3))

    print(f"  {n} contacts in the set, selecting hand-box pairs", flush=True)
    all_hand = set().union(*hands.values())
    rows, pts, cols = [], [], []
    for i in range(n):
        a, b = int(s0[i]), int(s1[i])
        hand_is_0 = a in all_hand and b in box_shapes
        hand_is_1 = b in all_hand and a in box_shapes
        if not (hand_is_0 or hand_is_1):
            continue
        shape = a if hand_is_0 else b
        side = next(k for k, v in hands.items() if shape in v)
        local = p0[i] if hand_is_0 else p1[i]
        world = apply_xform(body_q[int(body_of[shape])], local[None, :])[0]
        mag = float(np.linalg.norm(f[i]))
        pts.append(world)
        cols.append((0.90, 0.15, 0.10) if mag > LOAD_N else (0.62, 0.62, 0.66))
        rows.append(dict(side=side, x=world[0], y=world[1], z=world[2],
                         nx=nrm[i][0], ny=nrm[i][1], nz=nrm[i][2],
                         fx=f[i][0], fy=f[i][1], fz=f[i][2], mag=mag,
                         load_bearing=int(mag > LOAD_N)))

    if pts:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.array(pts))
        pc.colors = o3d.utility.Vector3dVector(np.array(cols))
        o3d.io.write_point_cloud(str(out / "contacts.ply"), pc)

        hot = [(p, r["mag"]) for p, r in zip(pts, rows) if r["mag"] > LOAD_N]
        if hot:
            top = max(m for _, m in hot)
            blob = o3d.geometry.TriangleMesh()
            for p, m in hot:
                s = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.002 + 0.006 * (m / top) ** 0.5, resolution=6)
                s.translate(p)
                s.paint_uniform_color((0.90, 0.15, 0.10))
                blob += s
            blob.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(out / "contacts_spheres.ply"), blob)

    with open(out / "contacts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["side"])
        w.writeheader()
        w.writerows(rows)

    # ---- the audit: per-digit gap to the box, against the contact set ----
    box_scene = o3d.t.geometry.RaycastingScene()
    for bv, bt in box_geo:
        bm = o3d.geometry.TriangleMesh()
        bm.vertices = o3d.utility.Vector3dVector(bv)
        bm.triangles = o3d.utility.Vector3iVector(bt)
        box_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(bm))

    margin_mm = args.margin * 1e3
    print(f"\nframe {args.frame}: {len(rows)} hand-box contacts reported, "
          f"{sum(r['load_bearing'] for r in rows)} load-bearing (>{LOAD_N} N), "
          f"collider margin {margin_mm:.1f} mm per shape")
    print("gap is SIGNED distance from the digit's nearest vertex to the box surface: "
          "negative means penetrating.")
    print(f"{'digit':16s} {'gap_mm':>8s} {'contacts':>9s} {'loaded':>7s} "
          f"{'sum|f|_N':>9s}  verdict")
    for side, (shape, v, _) in hand_geo.items():
        # digit_bands works in the collider's own frame, so band it there and carry the
        # mask across to the world-space copy, which has the same vertex order.
        local = np.asarray(scene.model.shape_source[shape].vertices, dtype=np.float64)
        for nm, z0, z1 in digit_bands(local):
            m = digit_mask(local, nm, z0, z1)
            if m.sum() == 0:
                continue
            d = box_scene.compute_signed_distance(
                o3d.core.Tensor(v[m].astype(np.float32))).numpy() * 1e3
            gap = float(d.min())
            near = [r for r in rows if r["side"] == side and _in_band(r, v, m)]
            nl = sum(r["load_bearing"] for r in near)
            fsum = sum(r["mag"] for r in near)
            # A digit sitting right at the margin generates candidate contacts but no force,
            # which is correct, not a miss -- so only flag digits comfortably inside it.
            if nl:
                verdict = "touching, carrying load"
            elif gap <= margin_mm - 1.0:
                verdict = f"INSIDE the margin by {margin_mm - gap:.1f} mm but silent -- check this"
            elif gap <= margin_mm + 1.0:
                verdict = "grazing the margin, zero force is expected"
            else:
                verdict = f"clear by {gap - margin_mm:.1f} mm past the margin"
            print(f"{side + ' ' + nm:16s} {gap:8.1f} {len(near):9d} {nl:7d} "
                  f"{fsum:9.1f}  {verdict}")

    # ---- is a reported contact actually where the two surfaces meet? ----
    # A contact can legitimately be reported across a visible gap if the shapes carry an
    # effective radius: rigid_contact_margin* is "effective radius + margin", and the solver
    # works on the inflated surface, not the drawn one. This checks the drawn geometry
    # against the pair of witness points the pipeline itself produced.
    hand_scene = {}
    for side, (_, hv, ht) in hand_geo.items():
        hm = o3d.geometry.TriangleMesh()
        hm.vertices = o3d.utility.Vector3dVector(hv)
        hm.triangles = o3d.utility.Vector3iVector(ht)
        s = o3d.t.geometry.RaycastingScene()
        s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hm))
        hand_scene[side] = s

    m0 = c.rigid_contact_margin0.numpy()[:n]
    m1 = c.rigid_contact_margin1.numpy()[:n]
    scale = getattr(scene.model, "shape_scale", None)
    print("\nper load-bearing contact: where the two witness points sit, and how far the "
          "collider is inflated")
    print(f"{'side':6s} {'|p0-p1|_mm':>10s} {'solver_sep_mm':>13s} {'to_hand_mm':>10s} "
          f"{'to_box_mm':>9s} {'margin0+1_mm':>12s} {'|f|_N':>8s}")
    for i in range(n):
        a, b = int(s0[i]), int(s1[i])
        hand_is_0 = a in all_hand and b in box_shapes
        hand_is_1 = b in all_hand and a in box_shapes
        if not (hand_is_0 or hand_is_1):
            continue
        mag = float(np.linalg.norm(f[i]))
        if mag <= LOAD_N:
            continue
        wa = apply_xform(body_q[int(body_of[a])], p0[i][None, :])[0]
        wb = apply_xform(body_q[int(body_of[b])], p1[i][None, :])[0]
        side = next(k for k, v in hands.items() if (a if hand_is_0 else b) in v)
        ph, pb = (wa, wb) if hand_is_0 else (wb, wa)
        dh = float(hand_scene[side].compute_distance(
            o3d.core.Tensor(ph[None, :].astype(np.float32))).numpy()[0]) * 1e3
        db = float(box_scene.compute_distance(
            o3d.core.Tensor(pb[None, :].astype(np.float32))).numpy()[0]) * 1e3
        # contacts.py:65 -- what the solver treats as penetration.
        sep = (float(np.dot(nrm[i], wb - wa)) - float(m0[i] + m1[i])) * 1e3
        print(f"{side:6s} {np.linalg.norm(wa - wb) * 1e3:10.2f} {sep:13.2f} {dh:10.2f} "
              f"{db:9.2f} {(m0[i] + m1[i]) * 1e3:12.2f} {mag:8.1f}")
    if scale is not None:
        sv = scale.numpy()
        odd = [s for s in list(all_hand) + list(box_shapes)
               if not np.allclose(sv[s], 1.0)]
        print(f"shape_scale away from 1 on hand/box shapes: "
              f"{ {s: sv[s].tolist() for s in odd} if odd else 'none'}")

    print(f"\nwrote {out}/  (scene.ply, {'left_hand.ply, right_hand.ply, ' if hand_geo else ''}"
          f"box.ply, contacts.ply, contacts_spheres.ply, contacts.csv)")


def _in_band(row, verts, mask):
    """Is this contact nearest to a vertex of the masked digit?"""
    p = np.array([row["x"], row["y"], row["z"]])
    d_in = np.linalg.norm(verts[mask] - p, axis=1).min()
    d_out = np.linalg.norm(verts[~mask] - p, axis=1).min() if (~mask).any() else np.inf
    return d_in <= d_out


if __name__ == "__main__":
    main()
