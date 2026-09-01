"""Flat, palm-facing 2D layout of the G1 rubber-hand collider, for tactile maps.

The mesh is ``meshes/{left,right}_rubber_hand.STL`` read straight out of
``g1_29dof_rev_1_0_with_rubber_hand.urdf`` -- the fixed five-finger rubber hand that ships
on the 29-dof G1, and the same geometry Newton collides against. Nothing here substitutes
a generic hand model.

Why flat rather than a 3D render: the hand is a slab in its own x-z plane (131 x 107 mm
across, 67 mm thick including the thumb), a grasp loads exactly one side of that slab, and
an orthographic projection down the palm normal therefore loses nothing while giving the
familiar hand outline you can actually locate a fingertip on.

Local frame of the STL, confirmed against the mesh bounds:
  +x  wrist -> fingertips        (131.8 mm)
  +-y palm normal, sign mirrored between the two hands   (66.6 mm)
  +z  across the fingers         (106.5 mm)
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.collections import PolyCollection

CANVAS_TRIS = 3000        # display/splat canvas; 0.03 mm from the original, so invisible
TAXEL_M = 0.004           # splat kernel sigma, a plausible tactile element pitch
DEAD = (0.82, 0.82, 0.84)


def decimated(verts, tris, target):
    from sugar_newton.validation.check_decimation import _o3d_mesh

    m = _o3d_mesh(verts, tris).simplify_quadric_decimation(
        target_number_of_triangles=target)
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    return np.asarray(m.vertices), np.asarray(m.triangles)


def palm_sign(verts, pts, mag) -> float:
    """Which side of local y the grasp loads, i.e. which way the palm faces.

    Read off the load, and cross-checked against two purely geometric tests that agree with
    it on both hands: the palm side is concave (12.9 mm mean gap to the convex hull, against
    1.2 mm on the back) and all four fingers curl toward it (32-42 mm of tip deflection off
    a straight fit to the finger base). Measured palm normal is -y for the left hand and +y
    for the right, and 100.0 % of the grasp force lands on that side, both hands.

    With fingers at +x and pinky->thumb at +z, that makes ``f x p == -n`` for the right hand
    and ``+n`` for the left, which is the correct chirality for each -- the STLs are not
    swapped.
    """
    if len(pts) == 0 or float(np.sum(mag)) <= 0.0:
        return 1.0
    s = float(np.sign(np.sum(mag * (pts[:, 1] - verts[:, 1].mean()))))
    return s if s != 0.0 else 1.0


def digit_bands(verts):
    """``[(name, z_lo, z_hi)]`` for the five digits, or ``[]`` if the mesh does not split.

    The four fingers separate cleanly in z once you look only at the fingertip band, so they
    are found by splitting sorted z at gaps; the thumb is the short digit beyond them.
    """
    lo, hi = verts.min(0), verts.max(0)
    span = hi - lo
    tip = verts[verts[:, 0] > lo[0] + 0.88 * span[0]]
    if len(tip) < 100:
        return []
    z = np.sort(tip[:, 2])
    cut = np.where(np.diff(z) > 0.005)[0]
    if len(cut) != 3:
        return []
    edges = [z[0] - 1e-4] + [0.5 * (z[i] + z[i + 1]) for i in cut] + [z[-1] + 1e-4]
    names = ["pinky", "ring", "middle", "index"]
    out = [(names[i], edges[i], edges[i + 1]) for i in range(4)]
    # The thumb never reaches the fingertip band, so it is whatever lies beyond the index.
    out.append(("thumb", edges[4] + 0.004, hi[2] + 1e-4))
    return out


def digit_mask(verts, name, z0, z1):
    lo, hi = verts.min(0), verts.max(0)
    m = (verts[:, 2] >= z0) & (verts[:, 2] < z1)
    if name == "thumb":
        m &= verts[:, 0] > lo[0] + 0.5 * (hi[0] - lo[0])
    return m


def digits(verts, sign: float = 1.0):
    """``[(name, screen_x_mm, screen_y_mm)]`` at each fingertip, for annotation."""
    out = []
    for nm, z0, z1 in digit_bands(verts):
        m = digit_mask(verts, nm, z0, z1)
        if m.sum() < 30:
            return []
        p = verts[m]
        t = p[p[:, 0] >= p[:, 0].max() - 0.006].mean(0)
        out.append((nm, float(t[2] * (1.0 if sign >= 0 else -1.0) * 1e3), float(t[0] * 1e3)))
    return out


class HandAtlas:
    """One hand, projected flat, with a cached splat tree and face draw order."""

    def __init__(self, verts, tris, sign: float = 1.0):
        self.verts, self.tris = verts, tris
        d = np.array([0.0, 1.0 if sign >= 0 else -1.0, 0.0])   # palm outward = toward viewer
        up = np.array([1.0, 0.0, 0.0])                         # fingertips to the top
        right = np.cross(up, d)                                # right x up == d, so the
        self.axes = (right, up, d)                             # view is not mirrored

        tri = verts[tris]
        self.cen = tri.mean(1)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.normal = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

        # Painter's algorithm: a PolyCollection has no depth buffer, so draw back to front
        # and let the near half of each finger overpaint the far half.
        self.order = np.argsort(self.cen @ d)
        self.poly = np.stack([tri @ right, tri @ up], -1)[self.order] * 1e3   # mm
        lit = d + 0.35 * up + 0.15 * right
        lit /= np.linalg.norm(lit)
        self.light = (np.clip(self.normal @ lit, 0.0, 1.0) * 0.45 + 0.55)[self.order]
        self._tree = None

    def splat(self, pts, mag, sigma: float = TAXEL_M) -> np.ndarray:
        """Spread contact force magnitudes over canvas faces with a Gaussian kernel.

        Only load-bearing contacts matter: ~94 % of what Newton reports as a contact carries
        exactly zero force, and those sit up to 100 mm off the surface (they are broadphase
        proximity candidates). Zero weight makes them no-ops anyway; skipping them up front
        just makes this ~15x cheaper.
        """
        from scipy.spatial import cKDTree

        out = np.zeros(len(self.tris))
        if len(pts) == 0:
            return out
        keep = np.asarray(mag) > 0.0
        if not keep.any():
            return out
        if self._tree is None:
            self._tree = cKDTree(self.cen)
        for p, m in zip(np.asarray(pts)[keep], np.asarray(mag)[keep]):
            near = self._tree.query_ball_point(p, 3.0 * sigma)
            if not near:
                continue
            w = np.exp(-0.5 * np.sum((self.cen[near] - p) ** 2, axis=1) / sigma ** 2)
            s = w.sum()
            if s > 0:
                out[near] += m * w / s
        return out

    def paint(self, pc, force, norm, cmap="inferno", dead=DEAD) -> None:
        c = matplotlib.colormaps[cmap](norm(force))[:, :3]
        if dead is not None:
            # Grey means "no force this frame", not "bottom of the scale". Without it an
            # untouched finger renders as colormap-zero black and reads as a dark reading.
            c[np.asarray(force) <= 1e-6] = dead
        pc.set_facecolors(np.clip(c[self.order] * self.light[:, None], 0.0, 1.0))

    def draw(self, ax, force, norm, cmap="inferno", dead=DEAD) -> PolyCollection:
        pc = PolyCollection(self.poly, linewidths=0.0)
        self.paint(pc, force, norm, cmap, dead)
        return self._add(ax, pc)

    def draw_shaded(self, ax, color=(0.45, 0.60, 0.85), lw=0.0) -> PolyCollection:
        cols = np.clip(np.array(color)[None, :] * self.light[:, None], 0.0, 1.0)
        pc = PolyCollection(self.poly, facecolors=cols, linewidths=lw,
                            edgecolors="k" if lw else "none")
        return self._add(ax, pc)

    def label_digits(self, ax, fontsize=8.5, color="#333333") -> None:
        for nm, sx, sy in digits(self.verts, self.axes[2][1]):
            ax.text(sx, sy + 5.0, nm, ha="center", va="bottom",
                    fontsize=fontsize, color=color)

    def _add(self, ax, pc: PolyCollection) -> PolyCollection:
        ax.add_collection(pc)
        flat = self.poly.reshape(-1, 2)
        lo, hi = flat.min(0), flat.max(0)
        pad = 0.04 * float((hi - lo).max())
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad + 0.08 * float(hi[1] - lo[1]))
        ax.set_aspect("equal")
        ax.set_axis_off()
        return pc


def load_hands(canvas: int = CANVAS_TRIS, sides=("left", "right")):
    """``{side: (full_verts, full_tris, canvas_verts, canvas_tris)}`` from the G1 URDF."""
    from sugar_newton.validation.check_decimation import hand_collision_meshes

    out = {}
    for name, v, t in hand_collision_meshes():
        side = name.split("_")[0]
        if side not in sides:
            continue
        cv, ct = decimated(v, t, canvas)
        out[side] = (v, t, cv, ct)
    return out
