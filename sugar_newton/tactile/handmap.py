# SPDX-License-Identifier: BSD-3-Clause
"""Hand/box shape lookup, per-frame contact readback, and the 2D tactile film for videos.

Extracted from ``validation/make_loop_video.py`` so the training-time evaluation video in
``rl/video.py`` draws the *same* panels from the *same* numbers. Two implementations would
drift, and a tactile map that silently disagrees between the validator and the training
monitor is worse than no map.

Read this before trusting a number that comes out of here:

* **A reported contact is not a load.** ~96 % of the contacts Newton reports carry exactly
  zero force -- they are proximity candidates from the broad phase. ``load_bearing`` counts
  only those above ``LOAD_N``, and every centroid or spread must be force-weighted.
* **``point0`` is local to ``shape0`` and ``point1`` to ``shape1``.** Whichever of the pair
  is the hand decides which array the position comes from; taking ``point0``
  unconditionally puts half the contacts in the box's frame, which reads as a plausible but
  wrong heatmap rather than as an error.
"""

from __future__ import annotations


import numpy as np

from sugar_newton.validation.hand_atlas import CANVAS_TRIS, HandAtlas, load_hands, palm_sign

LOAD_N = 0.01           # a contact below this is reported by the broad phase but carries nothing
SIDES = ("left", "right")


def hand_shapes(model) -> tuple[dict[str, set[int]], set[int]]:
    """Map ``{"left"/"right": {shape indices}}`` and the box's shape indices, from labels."""
    body_of = model.shape_body.numpy()
    labels = [l.split("/")[-1] for l in model.body_label]
    hands: dict[str, set[int]] = {}
    box: set[int] = set()
    for shape, body in enumerate(body_of):
        if body < 0:
            continue
        if labels[body] == "box":
            box.add(shape)
        for side in SIDES:
            if labels[body] == f"{side}_rubber_hand":
                hands.setdefault(side, set()).add(shape)
    return hands, box


def read_frame(contacts, hands, box, sides) -> tuple[dict, float, int]:
    """Hand-local contact points and force magnitudes for one frame.

    Call after ``solver.update_contacts(contacts, state)``: MuJoCo replaces the whole
    contact set during the solve, so reading before it gives the pre-solve guesses.

    Returns ``({side: (points_hand_local, |f|)}, net_force_N, load_bearing_count)``.
    """
    per = {s: (np.zeros((0, 3)), np.zeros(0)) for s in sides}
    n = int(contacts.rigid_contact_count.numpy()[0])
    if not n:
        return per, 0.0, 0

    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    f = contacts.force.numpy()[:n, :3]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]

    net, load = 0.0, 0
    for side in sides:
        patch = hands[side]
        h0 = np.array([a in patch and b in box for a, b in zip(s0, s1)])
        h1 = np.array([b in patch and a in box for a, b in zip(s0, s1)])
        sel = h0 | h1
        if not sel.any():
            continue
        mag = np.linalg.norm(f[sel], axis=1)
        per[side] = (np.where(h0[sel, None], p0[sel], p1[sel]), mag)
        net += float(np.linalg.norm(f[sel].sum(0)))
        load += int((mag > LOAD_N).sum())
    return per, net, load


class TactileFilm:
    """Accumulate per-frame hand contacts, then render them as a normalised heatmap stack.

    Splatting and colour normalisation are deferred to :meth:`develop` because the scale has
    to come from the whole rollout: a per-frame ``vmax`` makes a light touch look identical
    to a full grasp, which defeats the point of showing the map at all.
    """

    def __init__(self, sides=SIDES, canvas: int = CANVAS_TRIS):
        self.sides = [s for s in sides]
        self.canvases = load_hands(canvas, tuple(self.sides))
        self.raw: dict[str, list] = {s: [] for s in self.sides}
        self.nets: list[float] = []
        self.loads: list[int] = []

    @property
    def canvas_tris(self) -> int:
        return len(self.canvases[self.sides[0]][3])

    def add(self, per: dict, net: float, load: int) -> None:
        for side in self.sides:
            self.raw[side].append(per[side])
        self.nets.append(net)
        self.loads.append(load)

    def develop(self, gamma: float = 0.5, pct: float = 99.0):
        """Returns ``(atlases, maps, norm)`` ready for :meth:`HandAtlas.draw`."""
        import matplotlib

        atlases, maps = {}, {}
        for side in self.sides:
            _, _, cv, ct = self.canvases[side]
            rows = self.raw[side]
            allp = np.concatenate([p for p, _ in rows]) if rows else np.zeros((0, 3))
            allm = np.concatenate([m for _, m in rows]) if rows else np.zeros(0)
            # palm_sign needs the accumulated contacts: the palm is the side the grasp
            # force actually lands on, which one frame may not yet show.
            atlases[side] = HandAtlas(cv, ct, palm_sign(cv, allp, allm))
            maps[side] = np.array([atlases[side].splat(p, m) for p, m in rows])

        stack = np.concatenate([m for m in maps.values()]) if maps else np.zeros(1)
        hot = stack[stack > 1e-6]
        vmax = float(np.percentile(hot, pct)) if hot.size else 1.0
        norm = matplotlib.colors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax or 1e-9)
        return atlases, maps, norm
