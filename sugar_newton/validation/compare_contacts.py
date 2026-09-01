"""Does decimating the box collider change the contacts a tactile sensor would read?

``check_decimation`` answers a geometric question -- the decimated surface is within 1.6 mm
of the original -- but a tactile channel does not read surfaces, it reads contact forces and
where they land. This measures that directly.

The trap this avoids: two rollouts with different colliders diverge, because contact-rich
G1 rollouts are chaotic (0.07 m spread at *identical* parameters). Comparing frame k of one
against frame k of the other would therefore measure divergence, not the collider. So both
colliders are instead probed at the SAME states: one reference rollout is recorded, and then
every recorded state is pushed into both scenes, collided and stepped, and the resulting
contact sets compared. Any difference is then attributable to the collider alone.

Contacts cannot be paired one-to-one across two different tessellations, so the comparison
is over the aggregates a tactile array actually resolves: how much force each hand carries,
where the contact patch sits, and how spread out it is.

    python -m sugar_newton.validation.compare_contacts --frames 360 --window 200 360
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton

LOAD_N = 0.01     # above this a contact is carrying load, not merely inside the margin

from sugar_newton.validation.g1_carrybox_policy import (
    Actor,
    G1PolicyScene,
    load_clip,
)


def hand_and_box_shapes(scene) -> tuple[dict[str, set[int]], set[int]]:
    """Shape indices for each rubber hand and for the box."""
    body_of = scene.model.shape_body.numpy()
    labels = [l.split("/")[-1] for l in scene.model.body_label]
    hands = {"left": set(), "right": set()}
    box: set[int] = set()
    for s, b in enumerate(body_of):
        if b < 0:
            continue
        name = labels[b]
        if name == "box":
            box.add(s)
        for side in ("left", "right"):
            if name == f"{side}_rubber_hand":
                hands[side].add(s)
    return hands, box


def _empty_side() -> dict:
    return {"n": 0, "n_load": 0, "force": 0.0, "net": np.zeros(3),
            "pts": np.zeros((0, 3)), "mag": np.zeros(0), "frac_shape0": np.nan,
            "centroid": np.full(3, np.nan), "spread": np.nan}


def read_contacts(scene, hands, box) -> dict:
    """Per-hand contact aggregates for the current contact set."""
    c = scene.contacts
    n = int(c.rigid_contact_count.numpy()[0])
    out = {}
    if n == 0:
        for side in hands:
            out[side] = _empty_side()
        return out

    s0 = c.rigid_contact_shape0.numpy()[:n]
    s1 = c.rigid_contact_shape1.numpy()[:n]
    # point0/point1 are BODY-frame points on shape0/shape1 respectively (contacts.py:210).
    # Which of the pair is the hand is not guaranteed, so pick per contact rather than
    # always reading point0 -- mixing the two would average hand-local and box-local
    # coordinates into a meaningless centroid.
    p0 = c.rigid_contact_point0.numpy()[:n]
    p1 = c.rigid_contact_point1.numpy()[:n]
    # contacts.force is a spatial vector; wp.spatial_top is the force (reducer.py:222).
    f = c.force.numpy()[:n, :3] if c.force is not None else np.zeros((n, 3))

    for side, shapes in hands.items():
        hand_is_0 = np.array([a in shapes and b in box for a, b in zip(s0, s1)], dtype=bool)
        hand_is_1 = np.array([b in shapes and a in box for a, b in zip(s0, s1)], dtype=bool)
        sel = hand_is_0 | hand_is_1
        k = int(sel.sum())
        if k == 0:
            out[side] = _empty_side()
            continue
        pts = np.where(hand_is_0[sel, None], p0[sel], p1[sel])
        mag = np.linalg.norm(f[sel], axis=1)
        out[side] = {
            "n": k,
            # Hand-local contact points and force magnitudes, for the tactile map.
            "pts": pts,
            "mag": mag,
            "frac_shape0": float(hand_is_0[sel].mean()) if k else np.nan,
            # Sum of magnitudes: what a pressure-sensitive array integrates, but NOT
            # tessellation-invariant -- a finer mesh splits one patch into many small
            # near-opposing contacts, inflating the sum without changing the load.
            "force": float(mag.sum()),
            # Vector sum: the net load this hand puts on the box. This one IS
            # tessellation-invariant (it has to hold up the same 0.5 kg either way), so
            # it is the metric that can actually falsify decimation.
            "net": f[sel].sum(0),
            # Newton reports every pair within the 5 mm margin, and ~96 % of those carry
            # exactly zero force -- they are proximity candidates, not touches. Counting
            # them makes the contact set look 20x richer than it is, so report the
            # load-bearing count separately and weight the patch by force.
            "n_load": int((mag > LOAD_N).sum()),
            "centroid": (pts * mag[:, None]).sum(0) / mag.sum() if mag.sum() > 1e-9
                        else np.full(3, np.nan),
            "spread": (float(np.linalg.norm(pts - pts.mean(0), axis=1) @ mag / mag.sum())
                       if mag.sum() > 1e-9 else np.nan),
        }
    return out


def probe(scene, states, vels, targets, dt, substeps, hands, box) -> list[dict]:
    """Push each recorded state into ``scene`` and read the contacts it produces."""
    sub = dt / substeps
    rows = []
    for q, qd, tgt in zip(states, vels, targets):
        scene.state_0.joint_q.assign(q)
        scene.state_0.joint_qd.assign(qd)
        newton.eval_fk(scene.model, scene.state_0.joint_q, scene.state_0.joint_qd,
                       scene.state_0)
        scene.control.joint_target_q.assign(tgt)
        scene.pipeline.collide(scene.state_0, scene.contacts)
        scene.state_0.clear_forces()
        scene.solver.step(scene.state_0, scene.state_1, scene.control, scene.contacts, sub)
        scene.solver.update_contacts(scene.contacts, scene.state_0)
        rows.append(read_contacts(scene, hands, box))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data_000")
    ap.add_argument("--frames", type=int, default=360)
    ap.add_argument("--window", type=int, nargs=2, default=(200, 360),
                    help="frame range to compare; default is the carry")
    ap.add_argument("--variants", type=int, nargs="+", default=(0, 0, 2000),
                    help="box_tris values to probe; each is compared against the first. "
                         "Repeat the first value to get the probe's own noise floor -- "
                         "two identical colliders must agree, or no difference measured "
                         "against them means anything.")
    ap.add_argument("--box-tris", type=int, default=0,
                    help="box collider budget, held fixed when --vary is hand")
    ap.add_argument("--hand-tris", type=int, default=0,
                    help="hand collider budget, held fixed when --vary is box")
    ap.add_argument("--vary", default="box", choices=("box", "hand"),
                    help="which collider --variants refers to; the other stays at its "
                         "--box-tris / --hand-tris base")
    ap.add_argument("--dump", default=None,
                    help="write per-contact hand-local points and force magnitudes to this "
                         ".npz, for plot_hand_tactile.py")
    ap.add_argument("--collision", default="mesh", choices=("mesh", "hydro"))
    ap.add_argument("--mu", type=float, default=1.0)
    args = ap.parse_args()

    wp.init()
    clip = load_clip(args.clip)
    dt, substeps = 1.0 / clip["fps"], 4

    # Reference rollout on the ORIGINAL collider, recording the states to replay.
    ref = G1PolicyScene(clip, mu=args.mu, collision=args.collision,
                        box_tris=args.box_tris if args.vary == "hand" else 0,
                        hand_tris=args.hand_tris if args.vary == "box" else 0)
    actor = Actor()
    ref.reset(0)
    states, vels, targets = [], [], []
    for _ in range(args.frames):
        a = actor(ref.observe())
        ref.apply(a)
        states.append(ref.state_0.joint_q.numpy().copy())
        vels.append(ref.state_0.joint_qd.numpy().copy())
        targets.append(ref.control.joint_target_q.numpy().copy())
        ref.step(dt, substeps, "step")
    lo, hi = args.window
    states, vels, targets = states[lo:hi], vels[lo:hi], targets[lo:hi]
    print(f"recorded {len(states)} states from frames {lo}-{hi} of {args.clip}\n")

    runs = []
    dump: dict[str, np.ndarray] = {}
    for vi, tris in enumerate(args.variants):
        kw = ({"box_tris": tris, "hand_tris": args.hand_tris} if args.vary == "box"
              else {"box_tris": args.box_tris, "hand_tris": tris})
        scene = G1PolicyScene(clip, mu=args.mu, collision=args.collision, **kw)
        hands, box = hand_and_box_shapes(scene)
        # Warm up so contact buffers and kernels are resident before the probe.
        scene.reset(0)
        for _ in range(3):
            scene.step(dt, substeps, "step")
        rows = probe(scene, states, vels, targets, dt, substeps, hands, box)
        label = (f"{args.vary} original" if tris == 0
                 else f"{args.vary} decimated {tris}")
        runs.append((tris, label, rows))
        if args.dump is not None:
            for side in ("left", "right"):
                key = f"v{vi}_{tris}_{side}"
                dump[key + "_pts"] = np.concatenate([r[side]["pts"] for r in rows])
                dump[key + "_mag"] = np.concatenate([r[side]["mag"] for r in rows])
                # Frame index per contact, so the dump can be animated and not only
                # time-integrated.
                dump[key + "_frame"] = np.concatenate(
                    [np.full(r[side]["n"], i, dtype=np.int32)
                     for i, r in enumerate(rows)])
            dump[f"v{vi}_{tris}_label"] = np.array(label)
        tot = [sum(r[s]["n"] for s in r) for r in rows]
        frc = [sum(r[s]["force"] for s in r) for r in rows]
        # frac_shape0 confirms the hand/box ordering is stable; if it is not 0 or 1 the
        # per-contact point frame really does alternate and must be picked per contact.
        f0 = np.nanmean([r[s]["frac_shape0"] for r in rows for s in ("left", "right")])
        ld = [sum(r[s]["n_load"] for s in r) for r in rows]
        print(f"{label:>22}: margin candidates/frame {np.mean(tot):6.1f}, of which "
              f"load-bearing {np.mean(ld):5.1f}  |  summed force {np.mean(frc):7.1f} N  "
              f"hand-is-shape0 {f0:.2f}")

    base_tris, base_label, a = runs[0]
    for tris, label, b in runs[1:]:
        tag = "NOISE FLOOR (identical colliders)" if tris == base_tris else label
        print(f"\n{tag} vs {base_label}")
        print(f"{'':>8} {'d_cand':>7} {'d_load':>7} {'|d|sum_%':>9} {'net_A_N':>8} "
              f"{'net_B_N':>8} {'|d|net_%':>9} {'d_centroid_mm':>14} {'d_spread_mm':>12} "
              f"{'corr':>7} {'L1_%':>6}")
        for side in ("left", "right"):
            dn, dfp, dc, ds, dnp, na, nb, dl = [], [], [], [], [], [], [], []
            for ra, rb in zip(a, b):
                xa, xb = ra[side], rb[side]
                if xa["n"] == 0 and xb["n"] == 0:
                    continue
                dn.append(xb["n"] - xa["n"])
                dl.append(xb["n_load"] - xa["n_load"])
                if xa["force"] > 1e-6:
                    dfp.append(100.0 * abs(xb["force"] - xa["force"]) / xa["force"])
                ma, mb = np.linalg.norm(xa["net"]), np.linalg.norm(xb["net"])
                na.append(ma)
                nb.append(mb)
                if ma > 1e-6:
                    dnp.append(100.0 * np.linalg.norm(xb["net"] - xa["net"]) / ma)
                if np.all(np.isfinite(xa["centroid"])) and np.all(np.isfinite(xb["centroid"])):
                    dc.append(1e3 * np.linalg.norm(xb["centroid"] - xa["centroid"]))
                    ds.append(1e3 * abs(xb["spread"] - xa["spread"]))
            fa = np.array([r[side]["force"] for r in a])
            fb = np.array([r[side]["force"] for r in b])
            ok = np.isfinite(fa) & np.isfinite(fb) & ((fa > 1e-6) | (fb > 1e-6))
            corr = (float(np.corrcoef(fa[ok], fb[ok])[0, 1]) if ok.sum() >= 3
                    else float("nan"))
            rel = (100.0 * float(np.abs(fb[ok] - fa[ok]).sum()
                                 / max(np.abs(fa[ok]).sum(), 1e-9)) if ok.sum() else
                   float("nan"))
            print(f"{side:>8} {np.mean(dn):>7.1f} {np.mean(dl):>7.1f} "
                  f"{np.mean(dfp) if dfp else np.nan:>9.1f} "
                  f"{np.mean(na):>8.2f} {np.mean(nb):>8.2f} "
                  f"{np.mean(dnp) if dnp else np.nan:>9.1f} "
                  f"{np.mean(dc) if dc else np.nan:>14.2f} "
                  f"{np.mean(ds) if ds else np.nan:>12.2f} "
                  f"{corr:>7.3f} {rel:>6.1f}")
    print("\nd_cand counts pairs inside the 5 mm margin, d_load only those carrying "
          f">{LOAD_N} N -- about 96 % of candidates carry zero force, so the two differ by "
          "20x.\nThe centroid and spread are FORCE-WEIGHTED, i.e. where the load is, not "
          "where the margin is.\nmeans over frames where either collider reported "
          "hand-box contact. "
          "sum_% is the sum of contact-force magnitudes (what a pressure array integrates, "
          "but tessellation-dependent);\nnet is the vector-summed load the hand puts on the "
          "box, which must agree if the physics is preserved. Read every difference against "
          "the noise-floor row, not against zero.")

    if args.dump is not None:
        np.savez_compressed(args.dump, **dump)
        print(f"\nwrote per-contact dump to {args.dump}")


if __name__ == "__main__":
    main()
