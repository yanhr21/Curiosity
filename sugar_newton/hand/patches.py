# SPDX-License-Identifier: BSD-3-Clause
"""The frozen 27-patch-per-hand anatomical skin, ported to ``newton.ModelBuilder``.

Layout is copied verbatim from SUGAR's
``source/sugar_rl/sugar_rl/assets/robots/anatomical_whole_hand_tacsl_g1.py``
(``_palm_specs`` / ``_digit_specs``): twelve palm patches in row-major order, then
five digits thumb-to-little, each proximal/middle/distal. 27 per hand, 54 for the pair.

Only the *geometry* is ported. The TacSL sensing half is not -- no taxel grids, no
optical gel, no camera prims. A patch here is a rigid pad with a real collision
surface, and everything measured from it comes out of Newton's own contact buffer
through :class:`sugar_newton.tactile.PatchTactile`. That is the whole point of Plan 16
section 3: the sensing lives in our package, on Newton's contacts, where it can be
validated against statics.

Mesh frame (``left_rubber_hand.STL``, verified against the released mesh): **X** runs
wrist to digits, **Z** runs little to index, **Y** is the palm normal with the palm
surface at maximum Y. ``width_m`` is the X extent and ``length_m`` the Z extent;
``tangent_angle_deg`` rotates the pad about the surface normal.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

#: Skin thickness [m]. SUGAR's value: the smallest symmetric stand-off that keeps the
#: R15 gel outside both rubber-hand meshes with a 0.25 mm tolerance.
PATCH_THICKNESS_M = 0.0049
PATCH_DENSITY_KG_M3 = 1070.0

_MESH_DIR = (
    "/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew"
    "/robot_baby/Curiosity/SUGAR/descriptions/robots/g1/meshes"
)


@dataclass(frozen=True)
class AnatomicalPatchSpec:
    """One physical anatomical elastomer, in frozen within-hand order."""

    name: str
    center_x_m: float
    center_z_m: float
    tangent_angle_deg: float
    width_m: float
    length_m: float
    optical_r15: bool = False


# name, centre X, centre Z, tangent angle, width (X), length (Z), optical
_PALM = (
    ("palm_r0_c0", 0.0250, -0.0300, 0.0, 0.0145, 0.0175, False),
    ("palm_r0_c1", 0.0440, -0.0300, 0.0, 0.0145, 0.0185, False),
    ("palm_r0_c2", 0.0625, -0.0300, 0.0, 0.0145, 0.0175, False),
    ("palm_r1_c0", 0.0105, -0.0100, 0.0, 0.0240, 0.0215, False),
    ("palm_r1_c1", 0.0380, -0.0100, 0.0, 0.023977, 0.032001, True),
    ("palm_r1_c2", 0.0630, -0.0100, 0.0, 0.0240, 0.0165, False),
    ("palm_r2_c0", 0.0120, 0.0115, 0.0, 0.0180, 0.0225, False),
    ("palm_r2_c1", 0.0365, 0.0115, 0.0, 0.0180, 0.0245, False),
    ("palm_r2_c2", 0.0605, 0.0115, 0.0, 0.0180, 0.0225, False),
    ("palm_r3_c0", 0.0199775810, 0.0316210573, 0.410492336, 0.0213122924, 0.0311282942, False),
    ("palm_r3_c1", 0.0427826066, 0.0261319201, -1.328076665, 0.0090500865, 0.0130606976, False),
    ("palm_r3_c2", 0.0611802571, 0.0271055836, 0.327108434, 0.0122237847, 0.0215293069, False),
)

# digit, (proximal, middle, distal) centres, tangent angle, width, length
_DIGITS = (
    ("thumb", ((0.0466551565, 0.0425276585), (0.06493139, 0.05065043), (0.08293347, 0.05865136)),
     23.962489, 0.016, 0.0192),
    ("index", ((0.081, 0.027), (0.099, 0.027), (0.118, 0.027)), 0.0, 0.018, 0.017),
    ("middle", ((0.082, 0.0072), (0.1015, 0.0072), (0.122, 0.0072)), 0.0, 0.016, 0.017),
    ("ring", ((0.081, -0.0138), (0.099, -0.0138), (0.117, -0.0138)), 0.0, 0.014, 0.017),
    ("little", ((0.080, -0.0342), (0.096, -0.0342), (0.112, -0.0342)), 0.0, 0.017, 0.0145),
)
_SEGMENTS = ("proximal", "middle", "distal")


def _digit_specs() -> tuple[AnatomicalPatchSpec, ...]:
    return tuple(
        AnatomicalPatchSpec(
            name=f"{digit}_{segment}",
            center_x_m=centre[0],
            center_z_m=centre[1],
            tangent_angle_deg=angle,
            width_m=width,
            # SUGAR lengthens only the thumb proximal pad
            length_m=0.0198 if (digit == "thumb" and segment == "proximal") else length,
        )
        for digit, centres, angle, width, length in _DIGITS
        for segment, centre in zip(_SEGMENTS, centres, strict=True)
    )


PATCH_SPECS: tuple[AnatomicalPatchSpec, ...] = (
    tuple(AnatomicalPatchSpec(*v) for v in _PALM) + _digit_specs()
)
if len(PATCH_SPECS) != 27:  # the frozen topology is part of the contract
    raise RuntimeError(f"anatomical topology is {len(PATCH_SPECS)} patches, not 27")


def patch_names(sides: tuple[str, ...] = ("left", "right")) -> tuple[str, ...]:
    """The frozen left-then-right sensor ordering the policy will see."""
    return tuple(f"{side}_{spec.name}" for side in sides for spec in PATCH_SPECS)


def hand_mesh_path(side: str) -> str:
    return os.path.join(_MESH_DIR, f"{side}_rubber_hand.STL")


def load_hand_mesh(side: str):
    """Load the rubber-hand mesh. Watertight, so it takes an SDF directly."""
    import trimesh

    return trimesh.load(hand_mesh_path(side), force="mesh")


def _surface_y(vertices: np.ndarray, x: float, z: float, half_x: float, half_z: float) -> float:
    """Palm-side surface height under a pad footprint.

    The palm faces +Y, so the surface is the footprint's maximum Y. The search box is
    widened until it catches vertices -- the released mesh is very sparse over the flat
    palm centre (15 vertices within 6 mm of ``palm_r1_c1``), so a fixed window silently
    misses patches there.
    """
    for grow in (1.0, 1.5, 2.5, 4.0, 8.0):
        sel = (np.abs(vertices[:, 0] - x) <= half_x * grow) & (
            np.abs(vertices[:, 2] - z) <= half_z * grow
        )
        if sel.any():
            return float(vertices[sel, 1].max())
    raise ValueError(f"no hand-mesh vertices near patch centre ({x:.4f}, {z:.4f})")


def add_hand_patches(builder, body: int, side: str, cfg, mesh=None) -> list[int]:
    """Attach one hand's 27 pads to ``body`` and return their shape indices, in order.

    Each pad is a box standing proud of the palm surface by :data:`PATCH_THICKNESS_M`,
    so the pads -- not the hand shell -- are what touches anything.

    The caller must filter collisions between the returned shapes: they share a body,
    are permanently overlapping neighbours, and Newton's pipeline does not exclude
    same-body pairs the way MuJoCo's narrow phase does. 27 pads on one body is 351
    permanent constraints if this is skipped.
    """
    import warp as wp

    verts = np.asarray((mesh if mesh is not None else load_hand_mesh(side)).vertices, dtype=float)
    shapes: list[int] = []
    for spec in PATCH_SPECS:
        hx, hz = spec.width_m * 0.5, spec.length_m * 0.5
        y = _surface_y(verts, spec.center_x_m, spec.center_z_m, hx, hz)
        angle = math.radians(spec.tangent_angle_deg)
        shapes.append(
            builder.add_shape_box(
                body=body,
                xform=wp.transform(
                    wp.vec3(spec.center_x_m, y + PATCH_THICKNESS_M * 0.5, spec.center_z_m),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle),
                ),
                hx=hx,
                hy=PATCH_THICKNESS_M * 0.5,
                hz=hz,
                cfg=cfg,
                label=f"{side}_{spec.name}",
            )
        )
    return shapes
