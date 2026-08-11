# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical 27-patch-per-hand TacSL skin for the official SUGAR G1.

This module is deliberately independent of the withdrawn collision-neutral
repeated-R15 and merged whole-hand atlas implementations.  It adds fifty-two
finite-area force/shear elastomers plus two geometry-fixed official R15
elastomers.  Every patch is a distinct rigid body fixed to the existing
official rubber-hand body, owns a real collision surface, and is sampled by
the official IsaacLab v2.3.2 ``VisuoTactileSensor`` implementation.

The custom force-only elastomers contain no hand-written taxels.  Their
``20 x 25`` taxel grids are generated from their authored surface meshes by
the official sensor at runtime.  The two ``palm_r1_c1`` patches reference the
official R15 gel and camera-tip prims without reconstructing their optical
internals.

This is a sensorized physical variant of the official SUGAR robot, not the
untouched control and not a hardware or sim-to-real claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import trimesh
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_urdf


_WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
_HAND_MESH_DIR = _WORKSPACE_ROOT / "SUGAR/descriptions/robots/g1/meshes"
_DEFAULT_R15_USD = (
    _WORKSPACE_ROOT
    / "experiments/sugar_reproduction/assets/official_tacsl"
    / "gelsight_r15_finger/gelsight_r15_finger.usd"
)
_EXPECTED_SOURCE_SHA256 = {
    "left": "cff2221a690fa69303f61fce68f2d155c1517b52efb6ca9262dd56e0bc6e70fe",
    "right": "99533b778bca6246144fa511bb9d4e555e075c641f2a0251e04372869cd99d67",
}
_EXPECTED_R15_SHA256 = (
    "92139f53c8cff8d70ee7668dddc4912b6b15b549cf2eaacf0f85e635ae93aa43"
)

# Exact authored transforms and sampling center from the released R15 asset.
_R15_TIP_TRANSLATION = (0.0, -0.0025591400917619467, 0.06775999814271927)
_R15_TIP_ROTATION = (0.7071067690849304, 0.7071067690849304, 0.0, 0.0)
_R15_TAXEL_CENTER_IN_ELASTOMER = (4.95e-7, -0.00208185, 0.067759994)
_R15_CAMERA_TRANSLATION_IN_TIP = (0.0, 0.0, -0.018592857142857144)
_R15_CAMERA_ROTATION_IN_TIP = (0.0, 1.0, 0.0, 0.0)

# The complete sensorized skin is 4.9 mm thick.  That value is the smallest
# symmetric stand-off which keeps the full hash-bound official R15 visual gel
# outside both exact rubber-hand meshes with the 0.25-mm geometric tolerance.
# Custom patches extend the same distance from their exact attachment surface,
# so the optical module is not a lone protruding contact owner.
_PATCH_THICKNESS_M = 0.0049
_PATCH_DENSITY_KG_M3 = 1070.0
_R15_DECLARED_MASS_KG = 0.0066
_TIP_MASS_KG = 1.0e-6
_TIP_INERTIA_KG_M2 = (1.0e-12, 1.0e-12, 1.0e-12)

# Exact material values composed by the released IsaacLab v2.3.2 R15 spawner
# remain the defaults. The environment overrides exist only for explicitly
# named, hash-bound physical-contact diagnostics. They do not alter the
# independent TacSL SDF force-field equations.
def _nonnegative_finite_environment_parameter(
    name: str,
    default: float,
) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a nonnegative finite float") from error
    if not (0.0 <= value < float("inf")):
        raise ValueError(f"{name} must be a nonnegative finite float")
    return value


_COMPLIANT_STIFFNESS = _nonnegative_finite_environment_parameter(
    "CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_STIFFNESS",
    10.0,
)
_COMPLIANT_DAMPING = _nonnegative_finite_environment_parameter(
    "CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_DAMPING",
    1.0,
)
_STATIC_FRICTION = 0.5
_DYNAMIC_FRICTION = 0.5
_RESTITUTION = 0.0
_CUSTOM_TACTILE_MARGIN_M = 0.0004
# PhysX must contact the same object-facing surface that the official TacSL
# rays sample.  The visible elastomer remains 4.9 mm thick, but using that
# entire closed solid as the collider exposes its attachment and boundary
# walls through inter-patch seams.  Those faces have no taxels.  Keep only a
# watertight 0.1-mm layer immediately behind the exact outer surface as the
# physical collider, matching the front-surface replacement already used for
# the official R15 module.
_CUSTOM_FRONT_COLLISION_THICKNESS_M = 0.0001
_PATCH_SDF_RESOLUTION = 256
_PATCH_SDF_SUBGRID_RESOLUTION = 6
_PATCH_SDF_MARGIN_M = 0.001
_PATCH_SDF_NARROW_BAND_M = 0.001
# The 41 x 51 construction showed that every residual >0.25 mm hand-shell
# intrusion was an attachment-face chord through a high-curvature region:
# all authored vertices and every outer contact face were already outside.
# Doubling the intervals reduces that geometric chord error without changing
# the frozen 20 x 25 official TacSL grid, patch footprint, thickness, material,
# or load-bearing/contact ownership.
_CUSTOM_CONSTRUCTION_ROWS = 81
_CUSTOM_CONSTRUCTION_COLUMNS = 101
# The exact left-hand shell below palm_r3_c0 contains one sharp X-direction
# first-hit transition.  The 81-row backing misses the frozen 0.25-mm limit
# by 8.855 micrometers at one *inner* face center even though every vertex and
# every outer contact face is outside the shell.  Resolve only that physical
# footprint more densely in X; this does not change its footprint, thickness,
# material, 20x25 official TacSL sampler, or any force parameter.
_PALM_R3_C0_CONSTRUCTION_ROWS = 161


@dataclass(frozen=True)
class AnatomicalPatchSpec:
    """One physical anatomical elastomer in frozen within-hand order."""

    name: str
    center_x_m: float
    center_z_m: float
    tangent_angle_deg: float
    width_m: float
    length_m: float
    optical_r15: bool = False


def _palm_specs() -> tuple[AnatomicalPatchSpec, ...]:
    """Return twelve non-overlapping palm patches in row-major order."""

    # X runs from wrist to digits and Z runs little-to-index across the
    # official rubber-hand mesh.  The four Z bands have real 0.5--1.0 mm
    # contact gaps.  X extents are irregular because the released palm tapers
    # at both outside rows and because row 1 must leave a physical gap around
    # the exact 23.977 x 32.001 mm R15.  Values are written per patch so an
    # invalid common-row rectangle cannot silently overlap a neighbor.
    values = (
        # name, center X/Z, tangent angle, width, length, optical
        ("palm_r0_c0", 0.0250, -0.0300, 0.0, 0.0145, 0.0175, False),
        ("palm_r0_c1", 0.0440, -0.0300, 0.0, 0.0145, 0.0185, False),
        ("palm_r0_c2", 0.0625, -0.0300, 0.0, 0.0145, 0.0175, False),
        ("palm_r1_c0", 0.0105, -0.0100, 0.0, 0.0240, 0.0215, False),
        ("palm_r1_c1", 0.0380, -0.0100, 0.0, 0.023977, 0.032001, True),
        ("palm_r1_c2", 0.0630, -0.0100, 0.0, 0.0240, 0.0165, False),
        ("palm_r2_c0", 0.0120, 0.0115, 0.0, 0.0180, 0.0225, False),
        ("palm_r2_c1", 0.0365, 0.0115, 0.0, 0.0180, 0.0245, False),
        ("palm_r2_c2", 0.0605, 0.0115, 0.0, 0.0180, 0.0225, False),
        (
            "palm_r3_c0",
            0.0199775810,
            0.0316210573,
            0.410492336,
            0.0213122924,
            0.0311282942,
            False,
        ),
        (
            "palm_r3_c1",
            0.0427826066,
            0.0261319201,
            -1.328076665,
            0.0090500865,
            0.0130606976,
            False,
        ),
        (
            "palm_r3_c2",
            0.0611802571,
            0.0271055836,
            0.327108434,
            0.0122237847,
            0.0215293069,
            False,
        ),
    )
    return tuple(
        AnatomicalPatchSpec(
            name=name,
            center_x_m=center_x,
            center_z_m=center_z,
            tangent_angle_deg=angle,
            width_m=width,
            length_m=length,
            optical_r15=optical,
        )
        for (
            name,
            center_x,
            center_z,
            angle,
            width,
            length,
            optical,
        ) in values
    )


def _digit_specs() -> tuple[AnatomicalPatchSpec, ...]:
    """Return thumb through little, proximal-to-distal, in frozen order."""

    digit_contract = (
        (
            "thumb",
            (
                (0.0466551565, 0.0425276585),
                (0.06493139, 0.05065043),
                (0.08293347, 0.05865136),
            ),
            23.962489,
            0.016,
            0.0192,
        ),
        (
            "index",
            ((0.081, 0.027), (0.099, 0.027), (0.118, 0.027)),
            0.0,
            0.018,
            0.017,
        ),
        (
            "middle",
            ((0.082, 0.0072), (0.1015, 0.0072), (0.122, 0.0072)),
            0.0,
            0.016,
            0.017,
        ),
        (
            "ring",
            ((0.081, -0.0138), (0.099, -0.0138), (0.117, -0.0138)),
            0.0,
            0.014,
            0.017,
        ),
        (
            "little",
            ((0.080, -0.0342), (0.096, -0.0342), (0.112, -0.0342)),
            0.0,
            0.017,
            0.0145,
        ),
    )
    segment_names = ("proximal", "middle", "distal")
    return tuple(
        AnatomicalPatchSpec(
            name=f"{digit}_{segment}",
            center_x_m=center[0],
            center_z_m=center[1],
            tangent_angle_deg=angle,
            width_m=width,
            length_m=(
                0.0198
                if digit == "thumb" and segment == "proximal"
                else length
            ),
        )
        for digit, centers, angle, width, length in digit_contract
        for segment, center in zip(segment_names, centers, strict=True)
    )


ANATOMICAL_WHOLE_HAND_PATCH_SPECS: tuple[AnatomicalPatchSpec, ...] = (
    _palm_specs() + _digit_specs()
)
if len(ANATOMICAL_WHOLE_HAND_PATCH_SPECS) != 27:
    raise RuntimeError("Frozen anatomical whole-hand topology is not 27 patches")
if tuple(spec.name for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS).count(
    "palm_r1_c1"
) != 1:
    raise RuntimeError("Frozen anatomical topology must contain one palm_r1_c1")


def anatomical_whole_hand_sensor_names() -> tuple[str, ...]:
    """Return the frozen left-then-right 54-sensor ordering."""

    return tuple(
        f"{side}_{spec.name}_tactile"
        for side in ("left", "right")
        for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS
    )


def anatomical_whole_hand_optical_sensor_names() -> tuple[str, str]:
    """Return the symmetric geometry-fixed optical pair."""

    return ("left_palm_r1_c1_tactile", "right_palm_r1_c1_tactile")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@cache
def _exact_hand_surface(side: str) -> trimesh.Trimesh:
    """Load and hash-check one untouched official rubber-hand surface."""

    if side not in ("left", "right"):
        raise ValueError(f"Unsupported hand side: {side!r}")
    path = _HAND_MESH_DIR / f"{side}_rubber_hand.STL"
    if _sha256(path) != _EXPECTED_SOURCE_SHA256[side]:
        raise RuntimeError(f"Official {side} rubber-hand STL hash changed")
    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if (
        not isinstance(loaded, trimesh.Trimesh)
        or not loaded.is_watertight
        or not loaded.is_winding_consistent
    ):
        raise RuntimeError(f"Official {side} hand mesh is not an oriented solid")
    return loaded


def _surface_frame(
    side: str,
    spec: AnatomicalPatchSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact surface point and local-to-hand rotation matrix."""

    mesh = _exact_hand_surface(side)
    origin_y = -0.12 if side == "left" else 0.12
    direction_y = 1.0 if side == "left" else -1.0
    origin = np.array(
        [[spec.center_x_m, origin_y, spec.center_z_m]], dtype=np.float64
    )
    direction = np.array([[0.0, direction_y, 0.0]], dtype=np.float64)
    locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
        ray_origins=origin,
        ray_directions=direction,
        multiple_hits=True,
    )
    candidates = np.flatnonzero(ray_indices == 0)
    expected_y_sign = -1.0 if side == "left" else 1.0
    candidates = np.asarray(
        [
            candidate
            for candidate in candidates
            if expected_y_sign
            * mesh.face_normals[int(triangle_indices[candidate]), 1]
            >= 0.2
        ],
        dtype=np.int64,
    )
    if len(candidates) == 0:
        raise RuntimeError(
            f"No exact palmar hand-surface hit for {side}/{spec.name} at "
            f"xz=({spec.center_x_m}, {spec.center_z_m})"
        )
    distances = np.linalg.norm(locations[candidates] - origin[0], axis=1)
    selected = candidates[int(np.argmin(distances))]
    point = np.asarray(locations[selected], dtype=np.float64)
    surface_normal = np.array(
        mesh.face_normals[int(triangle_indices[selected])],
        dtype=np.float64,
        copy=True,
    )
    surface_normal /= np.linalg.norm(surface_normal)
    if expected_y_sign * surface_normal[1] < 0.2:
        raise RuntimeError(
            f"{side}/{spec.name} selected a non-palmar triangle: "
            f"normal={surface_normal.tolist()}"
        )

    # The carried box approaches both hands along their opposed global-Y
    # directions.  A common approach axis gives every patch a single-valued
    # exact palmar height field, including across curved digits, while the
    # per-taxel TacSL basis below still uses each sampled triangle's actual
    # normal.  Using the center triangle normal as the ray direction can hit a
    # different digit or the rear shell at distal patches.
    outward = np.array((0.0, expected_y_sign, 0.0), dtype=np.float64)
    theta = math.radians(spec.tangent_angle_deg)
    long_axis = np.array([math.cos(theta), 0.0, math.sin(theta)])
    local_y = -outward
    local_z = long_axis
    local_x = np.cross(local_y, local_z)
    local_x /= np.linalg.norm(local_x)
    local_z = np.cross(local_x, local_y)
    local_z /= np.linalg.norm(local_z)
    rotation = np.stack((local_x, local_y, local_z), axis=1)
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10)
    ):
        raise RuntimeError(f"Invalid right-handed frame for {side}/{spec.name}")
    return point, rotation


def _quat_wxyz_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a right-handed 3x3 local-to-parent matrix to WXYZ."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _matrix(
    translation: tuple[float, float, float] | np.ndarray,
    rotation_wxyz: tuple[float, float, float, float],
) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(
        Gf.Quatd(rotation_wxyz[0], Gf.Vec3d(*rotation_wxyz[1:]))
    )
    matrix.SetTranslateOnly(Gf.Vec3d(*translation))
    return matrix


def _define_fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    hand_path: str,
    patch_path: str,
    patch_in_hand: Gf.Matrix4d,
) -> None:
    translation = patch_in_hand.ExtractTranslation()
    rotation = patch_in_hand.ExtractRotationQuat()
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(patch_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*translation))
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*rotation.GetImaginary()))
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def _box_mesh(
    width_m: float,
    length_m: float,
    thickness_m: float,
) -> tuple[Vt.Vec3fArray, Vt.IntArray, Vt.IntArray]:
    """Return a closed pad whose contact face is local ``-Y``."""

    half_x = 0.5 * width_m
    half_z = 0.5 * length_m
    points = Vt.Vec3fArray(
        [
            Gf.Vec3f(-half_x, -thickness_m, -half_z),
            Gf.Vec3f(half_x, -thickness_m, -half_z),
            Gf.Vec3f(half_x, -thickness_m, half_z),
            Gf.Vec3f(-half_x, -thickness_m, half_z),
            Gf.Vec3f(-half_x, 0.0, -half_z),
            Gf.Vec3f(half_x, 0.0, -half_z),
            Gf.Vec3f(half_x, 0.0, half_z),
            Gf.Vec3f(-half_x, 0.0, half_z),
        ]
    )
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 5, 1),
        (0, 4, 5),
        (1, 6, 2),
        (1, 5, 6),
        (2, 7, 3),
        (2, 6, 7),
        (3, 4, 0),
        (3, 7, 4),
    )
    return (
        points,
        Vt.IntArray([3] * len(faces)),
        Vt.IntArray([index for face in faces for index in face]),
    )


def _custom_taxel_grid_xz(
    width_m: float,
    length_m: float,
) -> np.ndarray:
    """Reproduce the official TacSL 20 x 25 projected grid in local X-Z."""

    divisions = np.asarray((20, 25), dtype=np.int64)
    dimensions = np.asarray((width_m, length_m), dtype=np.float64)
    division_size = (
        dimensions - 2.0 * _CUSTOM_TACTILE_MARGIN_M
    ) / (divisions + 1)
    tactile_pitch = float(np.min(division_size))
    x = np.linspace(
        -0.5 * tactile_pitch * (divisions[0] + 1),
        0.5 * tactile_pitch * (divisions[0] + 1),
        divisions[0] + 2,
    )[1:-1]
    z = np.linspace(
        -0.5 * tactile_pitch * (divisions[1] + 1),
        0.5 * tactile_pitch * (divisions[1] + 1),
        divisions[1] + 2,
    )[1:-1]
    x_values, z_values = np.meshgrid(x, z, indexing="ij")
    return np.stack((x_values.reshape(-1), z_values.reshape(-1)), axis=1)


def _nonuniform_construction_axis(
    *,
    extent_m: float,
    count: int,
    phase: float,
    taxel_coordinates: np.ndarray,
) -> np.ndarray:
    """Return a dense monotone axis that cannot coincide with taxel lines."""

    axis = np.linspace(-0.5 * extent_m, 0.5 * extent_m, count)
    step = extent_m / float(count - 1)
    indices = np.arange(1, count - 1, dtype=np.float64)
    axis[1:-1] += (
        0.17
        * step
        * np.sin(indices * math.sqrt(2.0) + phase)
    )
    minimum_clearance = max(1.0e-7, 0.015 * step)
    for index in range(1, count - 1):
        nearest = float(
            np.min(np.abs(taxel_coordinates - axis[index]))
        )
        if nearest >= minimum_clearance:
            continue
        direction = -1.0 if (index % 2) else 1.0
        candidate = axis[index] + direction * 0.08 * step
        lower = axis[index - 1] + 0.35 * step
        upper = axis[index + 1] - 0.35 * step
        axis[index] = float(np.clip(candidate, lower, upper))
    if (
        np.any(np.diff(axis) <= 0.3 * step)
        or float(
            np.min(
                np.abs(
                    axis[1:-1, None]
                    - taxel_coordinates[None, :]
                )
            )
        )
        < 1.0e-7
    ):
        raise RuntimeError("Construction grid is not taxel-edge separated")
    return axis


def _point_line_distance_2d(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    vector = second - first
    numerator = np.abs(
        vector[0] * (first[1] - points[:, 1])
        - (first[0] - points[:, 0]) * vector[1]
    )
    return numerator / np.linalg.norm(vector)


def _prefer_ad_diagonal(
    *,
    row: int,
    column: int,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    taxel_grid_xz: np.ndarray,
) -> bool:
    """Choose the quad diagonal farther from every taxel in that cell."""

    x_low, x_high = x_grid[row : row + 2]
    z_low, z_high = z_grid[column : column + 2]
    inside = (
        (taxel_grid_xz[:, 0] > x_low)
        & (taxel_grid_xz[:, 0] < x_high)
        & (taxel_grid_xz[:, 1] > z_low)
        & (taxel_grid_xz[:, 1] < z_high)
    )
    if not np.any(inside):
        return (row + column) % 2 == 0
    points = taxel_grid_xz[inside]
    a = np.asarray((x_low, z_low))
    b = np.asarray((x_high, z_low))
    c = np.asarray((x_low, z_high))
    d = np.asarray((x_high, z_high))
    ad_clearance = float(_point_line_distance_2d(points, a, d).min())
    bc_clearance = float(_point_line_distance_2d(points, b, c).min())
    return ad_clearance >= bc_clearance


def _audit_custom_taxel_triangulation(
    mesh: trimesh.Trimesh,
    *,
    width_m: float,
    length_m: float,
) -> None:
    """Reject a custom surface whose official taxels touch triangle edges."""

    float_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    grid_xz = _custom_taxel_grid_xz(width_m, length_m)
    origins = np.column_stack(
        (
            grid_xz[:, 0],
            np.full(len(grid_xz), -1.0),
            grid_xz[:, 1],
        )
    )
    directions = np.zeros_like(origins)
    directions[:, 1] = 1.0
    triangle_ids, ray_ids, locations = (
        trimesh.ray.ray_triangle.RayMeshIntersector(
            float_mesh
        ).intersects_id(
            origins,
            directions,
            return_locations=True,
            multiple_hits=False,
        )
    )
    order = np.argsort(ray_ids)
    ray_ids = ray_ids[order]
    triangle_ids = triangle_ids[order]
    locations = locations[order]
    if (
        len(ray_ids) != 500
        or not np.array_equal(ray_ids, np.arange(500))
    ):
        raise RuntimeError(
            "Custom elastomer does not expose 500 official TacSL hits"
        )
    minimum_edge_clearance = float("inf")
    for location, triangle_id in zip(
        locations,
        triangle_ids,
        strict=True,
    ):
        triangle = float_mesh.triangles[int(triangle_id)][:, (0, 2)]
        point = location[None, (0, 2)]
        for corner in range(3):
            distance = float(
                _point_line_distance_2d(
                    point,
                    triangle[corner],
                    triangle[(corner + 1) % 3],
                )[0]
            )
            minimum_edge_clearance = min(
                minimum_edge_clearance,
                distance,
            )
    if minimum_edge_clearance < 1.0e-7:
        raise RuntimeError(
            "Official TacSL grid approaches a custom triangle edge by only "
            f"{minimum_edge_clearance:.12g} m"
        )


def _conformal_pad_mesh(
    *,
    side: str,
    spec: AnatomicalPatchSpec,
    point: np.ndarray,
    rotation: np.ndarray,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Build one visible elastomer and its exact front-surface collider.

    The load-bearing outer face and the attachment face are one deterministic
    height field derived from the hash-bound official hand STL.  Valid grid
    vertices are exact first palmar ray hits.  Where a rectangular physical
    patch deliberately overhangs a digit silhouette, the attachment height is
    extended from the nearest *exact* valid boundary vertex; no tactile point
    or force value is synthesized.  The official TacSL implementation later
    ray-samples the authored outer mesh independently.
    """

    hand_mesh = _exact_hand_surface(side)
    # The dense nonuniform construction grid reduces curved-fingertip chord
    # error below the frozen 0.25-mm hand-penetration limit.  Its lines and
    # per-cell diagonals are explicitly separated from the later official
    # 20 x 25 TacSL grid, removing ambiguous triangle-normal ownership.
    taxel_grid_xz = _custom_taxel_grid_xz(
        spec.width_m,
        spec.length_m,
    )
    construction_rows = (
        _PALM_R3_C0_CONSTRUCTION_ROWS
        if spec.name == "palm_r3_c0"
        else _CUSTOM_CONSTRUCTION_ROWS
    )
    x_grid = _nonuniform_construction_axis(
        extent_m=spec.width_m,
        count=construction_rows,
        phase=0.37,
        taxel_coordinates=taxel_grid_xz[:, 0],
    )
    z_grid = _nonuniform_construction_axis(
        extent_m=spec.length_m,
        count=_CUSTOM_CONSTRUCTION_COLUMNS,
        phase=1.11,
        taxel_coordinates=taxel_grid_xz[:, 1],
    )
    x_values, z_values = np.meshgrid(x_grid, z_grid, indexing="ij")
    planar_local = np.stack(
        (
            x_values.reshape(-1),
            np.zeros(x_values.size),
            z_values.reshape(-1),
        ),
        axis=1,
    )
    nominal_hand = point[None] + planar_local @ rotation.T
    outward = -rotation[:, 1]
    origins = nominal_hand + 0.08 * outward[None]
    directions = np.repeat((-outward)[None], len(origins), axis=0)
    locations, ray_indices, triangle_indices = (
        hand_mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=True,
        )
    )
    inner_y = np.empty(len(origins), dtype=np.float64)
    geometry_hit_mask = np.zeros(len(origins), dtype=bool)
    palmar_valid_mask = np.zeros(len(origins), dtype=bool)
    for ray_index in range(len(origins)):
        candidates = np.flatnonzero(ray_indices == ray_index)
        if len(candidates) == 0:
            continue
        distances = np.linalg.norm(
            locations[candidates] - origins[ray_index], axis=1
        )
        selected = candidates[int(np.argmin(distances))]
        inner_y[ray_index] = float(
            (locations[selected] - point) @ rotation[:, 1]
        )
        geometry_hit_mask[ray_index] = True
        palmar_candidates = np.asarray(
            [
                candidate
                for candidate in candidates
                if float(
                    np.dot(
                        hand_mesh.face_normals[
                            int(triangle_indices[candidate])
                        ],
                        outward,
                    )
                )
                >= 0.2
            ],
            dtype=np.int64,
        )
        palmar_valid_mask[ray_index] = len(palmar_candidates) > 0
    if not np.any(geometry_hit_mask):
        raise RuntimeError(
            f"{side}/{spec.name} conformal footprint has no attachment samples"
        )

    # Rectangular finite-area patches can overhang a curved finger silhouette.
    # The physical attachment follows the first exact hand-shell hit even when
    # that hit is a steep fingertip side face.  Palmar validity remains a
    # separate immutable mask; rejecting the steep real hit and copying a
    # neighboring palmar height would make the backing cut through the curved
    # finger.  Only rays that miss the entire closed official hand mesh use
    # the closest exact outer-envelope boundary height.  The later 500 TacSL
    # samples remain independent real samples of the authored outer elastomer.
    if not np.all(geometry_hit_mask):
        valid_coordinates = planar_local[geometry_hit_mask][:, (0, 2)]
        missing_coordinates = planar_local[~geometry_hit_mask][:, (0, 2)]
        squared_distance = np.sum(
            (
                missing_coordinates[:, None, :]
                - valid_coordinates[None, :, :]
            )
            ** 2,
            axis=2,
        )
        nearest = np.argmin(squared_distance, axis=1)
        inner_y[~geometry_hit_mask] = inner_y[geometry_hit_mask][nearest]

    # Fifty micrometers of geometric clearance prevents numerical coplanar
    # classification as embedding; the visible elastomer then extends exactly
    # 4.9 mm toward the object.  Its physical collision is authored separately
    # below as a thin layer behind only the exact object-facing outer surface.
    mount_clearance_m = 5.0e-5
    attachment_y = inner_y - mount_clearance_m
    contact_y = attachment_y - _PATCH_THICKNESS_M
    rows = len(x_grid)
    columns = len(z_grid)
    outer = planar_local.copy()
    inner = planar_local.copy()
    outer[:, 1] = contact_y
    inner[:, 1] = attachment_y
    vertices = np.concatenate((outer, inner), axis=0)
    layer_size = rows * columns
    outer_faces: list[tuple[int, int, int]] = []
    inner_faces: list[tuple[int, int, int]] = []

    def vertex(row: int, column: int) -> int:
        return row * columns + column

    for row in range(rows - 1):
        for column in range(columns - 1):
            a = vertex(row, column)
            b = vertex(row + 1, column)
            c = vertex(row, column + 1)
            d = vertex(row + 1, column + 1)
            # Outer winding points toward local -Y; inner winding +Y.  Select
            # the projected diagonal with the larger taxel clearance.
            if _prefer_ad_diagonal(
                row=row,
                column=column,
                x_grid=x_grid,
                z_grid=z_grid,
                taxel_grid_xz=taxel_grid_xz,
            ):
                outer_faces.extend(((a, b, d), (a, d, c)))
                inner_faces.extend(
                    (
                        (layer_size + a, layer_size + d, layer_size + b),
                        (layer_size + a, layer_size + c, layer_size + d),
                    )
                )
            else:
                outer_faces.extend(((a, b, c), (b, d, c)))
                inner_faces.extend(
                    (
                        (layer_size + a, layer_size + c, layer_size + b),
                        (layer_size + b, layer_size + c, layer_size + d),
                    )
                )

    boundary: list[int] = []
    boundary.extend(vertex(row, 0) for row in range(rows))
    boundary.extend(
        vertex(rows - 1, column) for column in range(1, columns)
    )
    boundary.extend(
        vertex(row, columns - 1)
        for row in range(rows - 2, -1, -1)
    )
    boundary.extend(
        vertex(0, column) for column in range(columns - 2, 0, -1)
    )
    side_faces: list[tuple[int, int, int]] = []
    for index, a in enumerate(boundary):
        b = boundary[(index + 1) % len(boundary)]
        side_faces.extend(
            (
                (a, layer_size + b, b),
                (a, layer_size + a, layer_size + b),
            )
        )

    visual_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(
            outer_faces + inner_faces + side_faces,
            dtype=np.int64,
        ),
        process=True,
        validate=True,
    )
    if (
        not visual_mesh.is_watertight
        or not visual_mesh.is_winding_consistent
        or not visual_mesh.is_volume
        or visual_mesh.volume <= 0.0
    ):
        raise RuntimeError(f"{side}/{spec.name} conformal pad is not a solid")
    _audit_custom_taxel_triangulation(
        visual_mesh,
        width_m=spec.width_m,
        length_m=spec.length_m,
    )

    # The front collider reuses the exact outer vertices and triangulation.
    # Its rear layer is shifted only toward the hand (local +Y), so every
    # object-facing collision triangle is bitwise identical to the surface
    # sampled by the 20 x 25 official TacSL rays.  The very shallow boundary
    # wall cannot create the millimetre-scale unsensed side/back contacts
    # produced by the former 4.9-mm collision solid.
    collision_rear = outer.copy()
    collision_rear[:, 1] += _CUSTOM_FRONT_COLLISION_THICKNESS_M
    collision_vertices = np.concatenate((outer, collision_rear), axis=0)
    collision_inner_faces = [
        (layer_size + a, layer_size + c, layer_size + b)
        for a, b, c in outer_faces
    ]
    collision_side_faces: list[tuple[int, int, int]] = []
    for index, a in enumerate(boundary):
        b = boundary[(index + 1) % len(boundary)]
        collision_side_faces.extend(
            (
                (a, layer_size + b, b),
                (a, layer_size + a, layer_size + b),
            )
        )
    collision_mesh = trimesh.Trimesh(
        vertices=collision_vertices,
        faces=np.asarray(
            outer_faces + collision_inner_faces + collision_side_faces,
            dtype=np.int64,
        ),
        process=True,
        validate=True,
    )
    if (
        not collision_mesh.is_watertight
        or not collision_mesh.is_winding_consistent
        or not collision_mesh.is_volume
        or collision_mesh.volume <= 0.0
    ):
        raise RuntimeError(
            f"{side}/{spec.name} front collision layer is not a solid"
        )

    visual_mesh.metadata["exact_hand_ray_valid_fraction"] = float(
        palmar_valid_mask.mean()
    )
    visual_mesh.metadata["exact_hand_geometry_hit_fraction"] = float(
        geometry_hit_mask.mean()
    )
    collision_mesh.metadata["front_collision_thickness_m"] = (
        _CUSTOM_FRONT_COLLISION_THICKNESS_M
    )
    return visual_mesh, collision_mesh


def _author_mesh(
    stage: Usd.Stage,
    path: str,
    source_mesh: trimesh.Trimesh,
    *,
    collision: bool,
    collision_approximation: str = "convexHull",
) -> UsdGeom.Mesh:
    points = Vt.Vec3fArray(
        [Gf.Vec3f(*vertex) for vertex in source_mesh.vertices]
    )
    counts = Vt.IntArray([3] * len(source_mesh.faces))
    indices = Vt.IntArray(source_mesh.faces.reshape(-1).tolist())
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(points)
    mesh.CreateFaceVertexCountsAttr().Set(counts)
    mesh.CreateFaceVertexIndicesAttr().Set(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(False)
    mesh.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))
    if collision:
        # A collision mesh is a PhysX representation, not a second render
        # surface.  Every anatomical patch has a separate visual gel/pad, so
        # keep the collision duplicate invisible to RTX while retaining its
        # exact authored geometry for SDF contact.
        mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(
            True
        )
        UsdPhysics.MeshCollisionAPI.Apply(
            mesh.GetPrim()
        ).CreateApproximationAttr().Set(collision_approximation)
        if collision_approximation == "sdf":
            sim_utils.define_mesh_collision_properties(
                str(mesh.GetPath()),
                sim_utils.SDFMeshPropertiesCfg(
                    sdf_margin=_PATCH_SDF_MARGIN_M,
                    sdf_narrow_band_thickness=_PATCH_SDF_NARROW_BAND_M,
                    sdf_resolution=_PATCH_SDF_RESOLUTION,
                    sdf_subgrid_resolution=_PATCH_SDF_SUBGRID_RESOLUTION,
                ),
            )
    else:
        mesh.CreateDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(0.12, 0.62, 0.92)])
        )
        mesh.CreateDisplayOpacityAttr().Set(Vt.FloatArray([0.82]))
    return mesh


def _mesh_topology_sha256(
    points: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
) -> str:
    """Hash one authored mesh topology without renderer-dependent metadata."""

    digest = hashlib.sha256()
    for array in (
        np.asarray(points, dtype="<f8"),
        np.asarray(counts, dtype="<i8"),
        np.asarray(indices, dtype="<i8"),
    ):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _collision_mesh_records(root_prim: Usd.Prim) -> tuple[dict[str, object], ...]:
    """Return immutable topology records for collision meshes below one hand."""

    records: list[dict[str, object]] = []
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        mesh = UsdGeom.Mesh(prim)
        if not mesh or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(
            mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64
        )
        indices = np.asarray(
            mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
        )
        records.append(
            {
                "path": str(prim.GetPath()),
                "enabled": bool(
                    UsdPhysics.CollisionAPI(
                        prim
                    ).GetCollisionEnabledAttr().Get()
                ),
                "topology_sha256": _mesh_topology_sha256(
                    points, counts, indices
                ),
            }
        )
    return tuple(sorted(records, key=lambda record: str(record["path"])))


def _disable_original_hand_collision_owner(
    *,
    stage: Usd.Stage,
    robot_path: str,
    side: str,
) -> dict[str, object]:
    """Remove the unsensorized shell as a parallel external contact owner.

    The 27 physical elastomers are the exterior load-bearing surfaces in this
    sensorized robot variant. Keeping the original URDF rubber-hand collider
    active behind/through those patches lets a box contact the hand link
    without intersecting any TacSL elastomer, which produces visually dense
    contact with sparse or zero taxels. Deactivating only the imported
    ``collisions`` subtree preserves the official hand body, mass, inertia,
    joints and render mesh while making patch ownership unambiguous.
    """

    hand_path = Sdf.Path(f"{robot_path}/{side}_rubber_hand")
    collision_root_path = hand_path.AppendChild("collisions")
    collision_root = stage.GetPrimAtPath(collision_root_path)
    if not collision_root.IsValid():
        raise RuntimeError(
            f"Missing original {side} rubber-hand collision root: "
            f"{collision_root_path}"
        )
    enabled_before = tuple(
        sorted(
            str(prim.GetPath())
            for prim in Usd.PrimRange(
                collision_root, Usd.TraverseInstanceProxies()
            )
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and bool(
                UsdPhysics.CollisionAPI(
                    prim
                ).GetCollisionEnabledAttr().Get()
            )
        )
    )
    if not enabled_before:
        raise RuntimeError(
            f"Original {side} rubber-hand collision owner was not found"
        )
    if not collision_root.SetActive(False):
        raise RuntimeError(
            f"Could not deactivate original hand collision root: "
            f"{collision_root_path}"
        )
    hand_prim = stage.GetPrimAtPath(hand_path)
    hand_prim.CreateAttribute(
        "curiosity:originalCollisionOwnerDisabled",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)
    hand_prim.CreateAttribute(
        "curiosity:disabledOriginalCollisionPaths",
        Sdf.ValueTypeNames.String,
    ).Set(json.dumps(enabled_before, separators=(",", ":")))
    return {
        "side": side,
        "collision_root": str(collision_root_path),
        "enabled_collision_owners_before": list(enabled_before),
        "original_collision_root_active_after": bool(
            collision_root.IsActive()
        ),
        "patches_are_only_exterior_contact_owners": True,
    }


def _replace_hand_visual_with_anatomical_r15_aperture(
    *,
    stage: Usd.Stage,
    robot_path: str,
    side: str,
    spec: AnatomicalPatchSpec,
    surface_point: np.ndarray,
    patch_rotation: np.ndarray,
) -> dict[str, object]:
    """Open one render-only window along the installed official R15 ray path.

    The complete source render mesh is copied out of its URDF instance and all
    faces outside the frozen center-R15 window are retained bit-for-bit.  No
    collision prim, API, material, filter, rigid body, or sensor is edited.
    """

    if not spec.optical_r15 or spec.name != "palm_r1_c1":
        raise ValueError("Anatomical R15 aperture requires palm_r1_c1")
    hand_path = Sdf.Path(f"{robot_path}/{side}_rubber_hand")
    hand_prim = stage.GetPrimAtPath(hand_path)
    if not hand_prim.IsValid():
        raise RuntimeError(f"Cannot find anatomical aperture hand: {hand_path}")

    collision_before = _collision_mesh_records(hand_prim)

    visual_meshes: list[Usd.Prim] = []
    for prim in Usd.PrimRange(hand_prim, Usd.TraverseInstanceProxies()):
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = (
            imageable.ComputePurpose()
            if imageable
            else UsdGeom.Tokens.default_
        )
        if purpose == UsdGeom.Tokens.guide or prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            continue
        if "/visuals/" in str(prim.GetPath()):
            visual_meshes.append(prim)
    if len(visual_meshes) != 1:
        raise RuntimeError(
            f"Expected one rubber-hand render mesh below {hand_path}, found "
            f"{[str(prim.GetPath()) for prim in visual_meshes]}"
        )
    source_prim = visual_meshes[0]
    source_mesh = UsdGeom.Mesh(source_prim)
    source_points = np.asarray(
        source_mesh.GetPointsAttr().Get(), dtype=np.float64
    )
    source_counts = np.asarray(
        source_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64
    )
    source_indices = np.asarray(
        source_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
    )
    if (
        source_points.size == 0
        or source_counts.size == 0
        or source_indices.size == 0
    ):
        raise RuntimeError(f"Empty rubber-hand render mesh: {source_prim.GetPath()}")

    mesh_in_hand, _ = UsdGeom.XformCache().ComputeRelativeTransform(
        source_prim, hand_prim
    )
    hand_points = np.asarray(
        [
            mesh_in_hand.Transform(Gf.Vec3d(*point))
            for point in source_points
        ],
        dtype=np.float64,
    )

    tip_path = (
        f"{robot_path}/{side}_anatomical_{spec.name}_tip"
    )
    camera_prim = stage.GetPrimAtPath(f"{tip_path}/cam")
    if not camera_prim.IsValid():
        raise RuntimeError(f"Cannot find exact official R15 camera below {tip_path}")

    # At spawn time the articulated hand link has not yet been initialized by
    # PhysX/Fabric.  Compute the exact authored camera-in-hand transform from
    # the same hash-bound R15 transforms used by the mount rather than reading
    # a pre-initialization world transform and mislabelling it as hand-local.
    patch_quaternion = _quat_wxyz_from_matrix(patch_rotation)
    target_center = (
        np.asarray(surface_point, dtype=np.float64)
        + np.asarray(patch_rotation[:, 1], dtype=np.float64)
        * (-_PATCH_THICKNESS_M)
    )
    base_translation = target_center - np.asarray(
        patch_rotation, dtype=np.float64
    ) @ np.asarray(_R15_TAXEL_CENTER_IN_ELASTOMER, dtype=np.float64)
    patch_in_hand = _matrix(base_translation, patch_quaternion)
    camera_in_hand = (
        _matrix(
            _R15_CAMERA_TRANSLATION_IN_TIP,
            _R15_CAMERA_ROTATION_IN_TIP,
        )
        * _matrix(_R15_TIP_TRANSLATION, _R15_TIP_ROTATION)
        * patch_in_hand
    )
    camera_center = np.asarray(
        camera_in_hand.ExtractTranslation(), dtype=np.float64
    )

    # The 23.8-by-29.0-mm window is entirely inside the exact physical
    # 23.977-by-32.001-mm center R15 footprint.  Coordinates use the current
    # anatomical patch frame, not a stale hand-axis approximation.
    half_local_x_m = 0.0119
    half_local_z_m = 0.0145
    if (
        2.0 * half_local_x_m >= spec.width_m
        or 2.0 * half_local_z_m >= spec.length_m
    ):
        raise RuntimeError("Frozen R15 aperture exceeds the physical footprint")
    outward = -np.asarray(patch_rotation[:, 1], dtype=np.float64)
    gel_center = (
        np.asarray(surface_point, dtype=np.float64)
        + outward * _PATCH_THICKNESS_M
    )
    camera_s = float(np.dot(camera_center - surface_point, outward))
    gel_s = float(np.dot(gel_center - surface_point, outward))
    lower_s = min(camera_s, gel_s) - 0.001
    upper_s = max(camera_s, gel_s) + 0.001

    retained_counts: list[int] = []
    retained_indices: list[int] = []
    removed_face_indices: list[int] = []
    offset = 0
    for face_index, count_value in enumerate(source_counts):
        count = int(count_value)
        face_indices = source_indices[offset : offset + count].astype(
            np.int64, copy=False
        )
        offset += count
        face_points = hand_points[face_indices]
        local_points = (
            face_points - np.asarray(surface_point, dtype=np.float64)
        ) @ np.asarray(patch_rotation, dtype=np.float64)
        overlaps_window = (
            float(local_points[:, 0].min()) <= half_local_x_m
            and float(local_points[:, 0].max()) >= -half_local_x_m
            and float(local_points[:, 2].min()) <= half_local_z_m
            and float(local_points[:, 2].max()) >= -half_local_z_m
        )
        mean_s = float(
            np.dot(face_points.mean(axis=0) - surface_point, outward)
        )
        lies_between_camera_and_gel = lower_s <= mean_s <= upper_s
        if overlaps_window and lies_between_camera_and_gel:
            removed_face_indices.append(face_index)
            continue
        retained_counts.append(count)
        retained_indices.extend(int(index) for index in face_indices)
    if not (
        10 <= len(removed_face_indices) < 0.5 * len(source_counts)
    ):
        raise RuntimeError(
            f"Invalid {side} anatomical R15 aperture face count: "
            f"{len(removed_face_indices)}/{len(source_counts)}"
        )

    source_visual_root = stage.GetPrimAtPath(hand_path.AppendChild("visuals"))
    if not source_visual_root.IsValid():
        raise RuntimeError(f"Missing instanced visual root below {hand_path}")
    UsdGeom.Imageable(source_visual_root).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )

    hand_points_f = Vt.Vec3fArray(
        [Gf.Vec3f(*[float(value) for value in point]) for point in hand_points]
    )
    aperture_mesh = UsdGeom.Mesh.Define(
        stage,
        hand_path.AppendChild("anatomical_r15_aperture_visual"),
    )
    aperture_mesh.CreatePointsAttr().Set(hand_points_f)
    aperture_mesh.CreateFaceVertexCountsAttr().Set(
        Vt.IntArray(retained_counts)
    )
    aperture_mesh.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray(retained_indices)
    )
    aperture_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    aperture_mesh.CreateDoubleSidedAttr().Set(
        source_mesh.GetDoubleSidedAttr().Get() or False
    )
    aperture_mesh.CreateExtentAttr().Set(
        UsdGeom.PointBased.ComputeExtent(hand_points_f)
    )
    bound_material, _ = UsdShade.MaterialBindingAPI(
        source_prim
    ).ComputeBoundMaterial()
    if bound_material:
        UsdShade.MaterialBindingAPI.Apply(
            aperture_mesh.GetPrim()
        ).Bind(bound_material)

    collision_after = _collision_mesh_records(hand_prim)
    if collision_after != collision_before:
        raise RuntimeError("Anatomical R15 aperture changed hand collision topology")

    retained_counts_array = np.asarray(retained_counts, dtype=np.int64)
    retained_indices_array = np.asarray(retained_indices, dtype=np.int64)
    source_topology_sha256 = _mesh_topology_sha256(
        hand_points, source_counts, source_indices
    )
    retained_topology_sha256 = _mesh_topology_sha256(
        hand_points, retained_counts_array, retained_indices_array
    )
    record: dict[str, object] = {
        "side": side,
        "patch": spec.name,
        "source_visual_path": str(source_prim.GetPath()),
        "source_hand_stl_sha256": _EXPECTED_SOURCE_SHA256[side],
        "source_topology_sha256": source_topology_sha256,
        "retained_topology_sha256": retained_topology_sha256,
        "source_face_count": int(len(source_counts)),
        "retained_face_count": int(len(retained_counts)),
        "removed_face_count": int(len(removed_face_indices)),
        "removed_face_indices_sha256": hashlib.sha256(
            np.asarray(removed_face_indices, dtype="<i8").tobytes()
        ).hexdigest(),
        "half_local_x_m": half_local_x_m,
        "half_local_z_m": half_local_z_m,
        "physical_width_m": spec.width_m,
        "physical_length_m": spec.length_m,
        "surface_point_hand_m": np.asarray(
            surface_point, dtype=np.float64
        ).tolist(),
        "patch_rotation_hand": np.asarray(
            patch_rotation, dtype=np.float64
        ).tolist(),
        "outward_hand": outward.tolist(),
        "camera_center_hand_m": camera_center.tolist(),
        "gel_center_hand_m": gel_center.tolist(),
        "camera_to_gel_axis_interval_m": [lower_s, upper_s],
        "collision_mesh_records": collision_after,
        "collision_mesh_unchanged": True,
        "rubber_hand_collision_mesh_count": len(collision_after),
        "whole_hand_hidden": False,
        "source_visual_replaced_by_retained_copy": True,
    }
    hand_prim.CreateAttribute(
        "curiosity:anatomicalR15ApertureRecord",
        Sdf.ValueTypeNames.String,
    ).Set(json.dumps(record, sort_keys=True, separators=(",", ":")))
    print("[ANATOMICAL-WHOLE-HAND-TACSL] camera_aperture", record, flush=True)
    return record


def _apply_declared_mass(
    prim: Usd.Prim,
    *,
    mass_kg: float,
    width_m: float,
    length_m: float,
    thickness_m: float,
) -> None:
    mass = UsdPhysics.MassAPI(prim)
    if not mass:
        mass = UsdPhysics.MassAPI.Apply(prim)
    inertia = (
        mass_kg * (thickness_m**2 + length_m**2) / 12.0,
        mass_kg * (width_m**2 + length_m**2) / 12.0,
        mass_kg * (width_m**2 + thickness_m**2) / 12.0,
    )
    mass.CreateMassAttr().Set(mass_kg)
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, -0.5 * thickness_m, 0.0))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))


def _define_reference(
    stage: Usd.Stage,
    prim_path: str,
    usd_path: str,
    source_prim_path: str,
) -> Usd.Prim:
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path, Sdf.Path(source_prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"Failed to reference {source_prim_path} at {prim_path}")
    return prim


def _make_tip_negligible(tip_prim: Usd.Prim) -> None:
    mass = UsdPhysics.MassAPI(tip_prim)
    if not mass:
        mass = UsdPhysics.MassAPI.Apply(tip_prim)
    mass.CreateMassAttr().Set(_TIP_MASS_KG)
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*_TIP_INERTIA_KG_M2))


def _replace_r15_collision_with_front_layer(
    *,
    stage: Usd.Stage,
    patch_prim: Usd.Prim,
    r15_usd: str,
    material_path: str,
) -> str:
    """Use the exact official R15 visual gel as its physical contact mesh.

    The released collider is an approximately eight-millimeter backing solid,
    whereas TacSL samples the separate curved visual gel.  Reusing that deep
    backing would both enter the hand and make the taxel surface differ from
    the PhysX contact surface.  Disable only the released backing collider and
    duplicate the exact referenced visual gel as the compliant collision
    solid.  The original visual, taxel generation, camera and mass are kept.
    """

    collision_prims = [
        prim
        for prim in Usd.PrimRange(
            patch_prim, Usd.TraverseInstanceProxies()
        )
        if prim.IsA(UsdGeom.Mesh)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and bool(
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        )
    ]
    if len(collision_prims) != 1:
        raise RuntimeError(
            "Official R15 must expose exactly one enabled collision mesh"
        )
    source_collision_prim = collision_prims[0]
    visual_prims = [
        prim
        for prim in Usd.PrimRange(
            patch_prim, Usd.TraverseInstanceProxies()
        )
        if prim.IsA(UsdGeom.Mesh)
        and not prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(visual_prims) != 1:
        raise RuntimeError(
            "Official R15 must expose exactly one visual gel mesh"
        )
    source_visual_prim = visual_prims[0]

    # The elastomer is referenced below the full R15 asset root, while the
    # released material lives at a sibling Looks prim.  A binding to that
    # sibling cannot compose through the partial reference.  Re-reference the
    # exact official Material into local scope and bind the referenced meshes,
    # matching the released whole-asset mount path without recreating a shader.
    # Keep the material as a sibling of the referenced rigid-body prim, as in
    # the official dual-palm mount.  Authoring it below the referenced body
    # would make the renderer traverse a material inside a physics body.
    local_material_path = (
        f"{patch_prim.GetParent().GetPath().pathString}/"
        f"{patch_prim.GetName()}_official_r15_material"
    )
    local_material_prim = stage.DefinePrim(local_material_path, "Material")
    local_material_prim.GetReferences().AddReference(
        r15_usd,
        Sdf.Path("/gelsight_r15_finger/Looks/material_0"),
    )
    render_material = UsdShade.Material(local_material_prim)
    if not render_material:
        raise RuntimeError("Exact official R15 material reference is invalid")
    for prim in Usd.PrimRange(
        patch_prim, Usd.TraverseInstanceProxies()
    ):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(render_material)
    bound_material, _ = UsdShade.MaterialBindingAPI(
        source_visual_prim
    ).ComputeBoundMaterial()
    if (
        not bound_material
        or bound_material.GetPath() != Sdf.Path(local_material_path)
    ):
        raise RuntimeError(
            "Official R15 visual gel did not resolve to its local exact "
            "material reference"
        )

    source = UsdGeom.Mesh(source_visual_prim)
    points = np.asarray(source.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(
        source.GetFaceVertexCountsAttr().Get(), dtype=np.int64
    )
    indices = np.asarray(
        source.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
    )
    faces: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts:
        polygon = indices[offset : offset + int(count)]
        for corner in range(1, int(count) - 1):
            faces.append(
                (
                    int(polygon[0]),
                    int(polygon[corner]),
                    int(polygon[corner + 1]),
                )
            )
        offset += int(count)

    relative, _ = UsdGeom.XformCache().ComputeRelativeTransform(
        source_visual_prim, patch_prim
    )
    points_patch = np.asarray(
        [
            relative.Transform(Gf.Vec3d(*point))
            for point in points
        ],
        dtype=np.float64,
    )
    exact_visual_gel = trimesh.Trimesh(
        vertices=points_patch,
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    trimesh.repair.fix_normals(exact_visual_gel, multibody=False)
    if exact_visual_gel.volume < 0.0:
        exact_visual_gel.invert()
    if (
        not exact_visual_gel.is_watertight
        or not exact_visual_gel.is_winding_consistent
        or not exact_visual_gel.is_volume
        or exact_visual_gel.volume <= 0.0
    ):
        raise RuntimeError("Exact official R15 visual gel is not a solid")

    UsdPhysics.CollisionAPI(
        source_collision_prim
    ).GetCollisionEnabledAttr().Set(False)
    collision = _author_mesh(
        stage,
        f"{patch_prim.GetPath().pathString}/"
        "collisions/exact_visual_gel/mesh",
        exact_visual_gel,
        collision=True,
        collision_approximation="sdf",
    )
    sim_utils.bind_physics_material(str(collision.GetPath()), material_path)
    patch_prim.CreateAttribute(
        "curiosity:officialR15RearCollisionDisabled",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)
    patch_prim.CreateAttribute(
        "curiosity:r15CollisionUsesExactVisualGel",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)
    patch_prim.CreateAttribute(
        "curiosity:officialR15MaterialSource",
        Sdf.ValueTypeNames.String,
    ).Set("/gelsight_r15_finger/Looks/material_0")
    patch_prim.CreateAttribute(
        "curiosity:officialR15LocalMaterialPath",
        Sdf.ValueTypeNames.String,
    ).Set(local_material_path)
    return collision.GetPath().pathString


def _mount_custom_patch(
    *,
    stage: Usd.Stage,
    robot_prim: Usd.Prim,
    robot_path: str,
    side: str,
    spec: AnatomicalPatchSpec,
    point: np.ndarray,
    rotation: np.ndarray,
    material_path: str,
) -> str:
    hand_path = f"{robot_path}/{side}_rubber_hand"
    hand_prim = stage.GetPrimAtPath(hand_path)
    patch_path = f"{robot_path}/{side}_anatomical_{spec.name}_elastomer"
    patch_prim = stage.DefinePrim(patch_path, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(patch_prim)
    patch_prim.CreateAttribute(
        "curiosity:anatomicalPatchName", Sdf.ValueTypeNames.String
    ).Set(f"{side}/{spec.name}")
    patch_prim.CreateAttribute(
        "curiosity:physicalLoadBearing", Sdf.ValueTypeNames.Bool
    ).Set(True)
    patch_prim.CreateAttribute(
        "curiosity:taxelGrid", Sdf.ValueTypeNames.Int2
    ).Set(Gf.Vec2i(20, 25))

    conformal_mesh, front_collision_mesh = _conformal_pad_mesh(
        side=side,
        spec=spec,
        point=point,
        rotation=rotation,
    )
    patch_prim.CreateAttribute(
        "curiosity:exactHandRayValidFraction",
        Sdf.ValueTypeNames.Double,
    ).Set(float(conformal_mesh.metadata["exact_hand_ray_valid_fraction"]))
    patch_prim.CreateAttribute(
        "curiosity:physicalThicknessM",
        Sdf.ValueTypeNames.Double,
    ).Set(_PATCH_THICKNESS_M)
    patch_prim.CreateAttribute(
        "curiosity:frontCollisionThicknessM",
        Sdf.ValueTypeNames.Double,
    ).Set(_CUSTOM_FRONT_COLLISION_THICKNESS_M)
    patch_prim.CreateAttribute(
        "curiosity:collisionUsesTaxelSampledFrontSurface",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)
    patch_prim.CreateAttribute(
        "curiosity:collisionApproximation",
        Sdf.ValueTypeNames.String,
    ).Set("sdf")
    visual = _author_mesh(
        stage,
        f"{patch_path}/visuals/contact_surface/mesh",
        conformal_mesh,
        collision=False,
    )
    collision = _author_mesh(
        stage,
        f"{patch_path}/collisions/load_bearing_surface/mesh",
        front_collision_mesh,
        collision=True,
        collision_approximation="sdf",
    )
    sim_utils.bind_physics_material(str(collision.GetPath()), material_path)
    _apply_declared_mass(
        patch_prim,
        mass_kg=float(conformal_mesh.volume * _PATCH_DENSITY_KG_M3),
        width_m=spec.width_m,
        length_m=spec.length_m,
        thickness_m=_PATCH_THICKNESS_M,
    )

    quaternion = _quat_wxyz_from_matrix(rotation)
    patch_in_hand = _matrix(point, quaternion)
    hand_in_robot, _ = UsdGeom.XformCache().ComputeRelativeTransform(
        hand_prim, robot_prim
    )
    UsdGeom.Xformable(patch_prim).MakeMatrixXform().Set(
        patch_in_hand * hand_in_robot
    )
    _define_fixed_joint(
        stage,
        f"{robot_path}/joints/{side}_anatomical_{spec.name}_joint",
        hand_path,
        patch_path,
        patch_in_hand,
    )
    UsdPhysics.FilteredPairsAPI.Apply(hand_prim).CreateFilteredPairsRel().AddTarget(
        Sdf.Path(patch_path)
    )
    return patch_path


def _mount_optical_r15(
    *,
    stage: Usd.Stage,
    robot_prim: Usd.Prim,
    robot_path: str,
    side: str,
    spec: AnatomicalPatchSpec,
    point: np.ndarray,
    rotation: np.ndarray,
    r15_usd: str,
    material_path: str,
) -> tuple[str, str]:
    hand_path = f"{robot_path}/{side}_rubber_hand"
    hand_prim = stage.GetPrimAtPath(hand_path)
    patch_path = f"{robot_path}/{side}_anatomical_{spec.name}_elastomer"
    tip_path = f"{robot_path}/{side}_anatomical_{spec.name}_tip"
    patch_prim = _define_reference(
        stage, patch_path, r15_usd, "/gelsight_r15_finger/elastomer"
    )
    patch_prim.CreateAttribute(
        "curiosity:anatomicalPatchName", Sdf.ValueTypeNames.String
    ).Set(f"{side}/{spec.name}")
    patch_prim.CreateAttribute(
        "curiosity:physicalLoadBearing", Sdf.ValueTypeNames.Bool
    ).Set(True)
    patch_prim.CreateAttribute(
        "curiosity:officialR15Optical", Sdf.ValueTypeNames.Bool
    ).Set(True)
    patch_prim.CreateAttribute(
        "curiosity:taxelSurfaceStandOffM",
        Sdf.ValueTypeNames.Double,
    ).Set(_PATCH_THICKNESS_M)
    patch_prim.CreateAttribute(
        "curiosity:collisionApproximation",
        Sdf.ValueTypeNames.String,
    ).Set("sdf")
    _apply_declared_mass(
        patch_prim,
        mass_kg=_R15_DECLARED_MASS_KG,
        width_m=spec.width_m,
        length_m=spec.length_m,
        thickness_m=0.008059,
    )
    _replace_r15_collision_with_front_layer(
        stage=stage,
        patch_prim=patch_prim,
        r15_usd=r15_usd,
        material_path=material_path,
    )

    quaternion = _quat_wxyz_from_matrix(rotation)
    target_center = point + rotation[:, 1] * (-_PATCH_THICKNESS_M)
    local_center = np.asarray(_R15_TAXEL_CENTER_IN_ELASTOMER)
    base_translation = target_center - rotation @ local_center
    patch_in_hand = _matrix(base_translation, quaternion)
    hand_in_robot, _ = UsdGeom.XformCache().ComputeRelativeTransform(
        hand_prim, robot_prim
    )
    UsdGeom.Xformable(patch_prim).MakeMatrixXform().Set(
        patch_in_hand * hand_in_robot
    )
    _define_fixed_joint(
        stage,
        f"{robot_path}/joints/{side}_anatomical_{spec.name}_joint",
        hand_path,
        patch_path,
        patch_in_hand,
    )

    tip_prim = _define_reference(
        stage, tip_path, r15_usd, "/gelsight_r15_finger/elastomer_tip"
    )
    _make_tip_negligible(tip_prim)
    tip_in_patch = _matrix(_R15_TIP_TRANSLATION, _R15_TIP_ROTATION)
    tip_in_hand = tip_in_patch * patch_in_hand
    UsdGeom.Xformable(tip_prim).MakeMatrixXform().Set(
        tip_in_hand * hand_in_robot
    )
    # Preserve the official R15 as one coherent assembly. Attaching elastomer
    # and tip independently to the hand creates two solver constraints whose
    # relative drift was measured at 1.08 mm / 0.172 degrees, breaking
    # force/RGB same-frame correspondence. The elastomer is attached to the
    # hand once; the camera tip is fixed directly to that elastomer using the
    # hash-bound official relative transform.
    _define_fixed_joint(
        stage,
        f"{robot_path}/joints/{side}_anatomical_{spec.name}_tip_joint",
        patch_path,
        tip_path,
        tip_in_patch,
    )
    filtered = UsdPhysics.FilteredPairsAPI.Apply(
        hand_prim
    ).CreateFilteredPairsRel()
    filtered.AddTarget(Sdf.Path(patch_path))
    filtered.AddTarget(Sdf.Path(tip_path))
    UsdPhysics.FilteredPairsAPI.Apply(
        patch_prim
    ).CreateFilteredPairsRel().AddTarget(Sdf.Path(tip_path))
    return patch_path, tip_path


@sim_utils.clone
def spawn_g1_with_anatomical_whole_hand_tacsl(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the official G1 and attach 27 physical patches to each hand."""

    robot_prim = spawn_from_urdf.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    r15_path = Path(
        os.environ.get("CURIOSITY_TACSL_R15_USD", _DEFAULT_R15_USD)
    ).expanduser().resolve()
    if not r15_path.is_file() or _sha256(r15_path) != _EXPECTED_R15_SHA256:
        raise RuntimeError(f"Official hash-bound R15 asset is unavailable: {r15_path}")

    stage = get_current_stage()
    collision_ownership_records = [
        _disable_original_hand_collision_owner(
            stage=stage,
            robot_path=prim_path,
            side=side,
        )
        for side in ("left", "right")
    ]
    material_path = f"{prim_path}/anatomical_whole_hand_compliant_material"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=_STATIC_FRICTION,
        dynamic_friction=_DYNAMIC_FRICTION,
        restitution=_RESTITUTION,
        friction_combine_mode="average",
        restitution_combine_mode="average",
        compliant_contact_stiffness=_COMPLIANT_STIFFNESS,
        compliant_contact_damping=_COMPLIANT_DAMPING,
    )
    material_cfg.func(material_path, material_cfg)

    aperture_mode = os.environ.get(
        "CURIOSITY_ANATOMICAL_TACSL_CAMERA_APERTURES", "both"
    )
    if aperture_mode not in ("none", "left", "right", "both"):
        raise ValueError(
            "CURIOSITY_ANATOMICAL_TACSL_CAMERA_APERTURES must be none, left, "
            f"right, or both, got {aperture_mode!r}"
        )
    aperture_sides = {
        "none": (),
        "left": ("left",),
        "right": ("right",),
        "both": ("left", "right"),
    }[aperture_mode]

    patch_paths: list[str] = []
    center_frames: dict[str, tuple[AnatomicalPatchSpec, np.ndarray, np.ndarray]] = {}
    for side in ("left", "right"):
        for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS:
            point, rotation = _surface_frame(side, spec)
            if spec.optical_r15:
                center_frames[side] = (spec, point.copy(), rotation.copy())
                patch_path, _ = _mount_optical_r15(
                    stage=stage,
                    robot_prim=robot_prim,
                    robot_path=prim_path,
                    side=side,
                    spec=spec,
                    point=point,
                    rotation=rotation,
                    r15_usd=str(r15_path),
                    material_path=material_path,
                )
            else:
                patch_path = _mount_custom_patch(
                    stage=stage,
                    robot_prim=robot_prim,
                    robot_path=prim_path,
                    side=side,
                    spec=spec,
                    point=point,
                    rotation=rotation,
                    material_path=material_path,
                )
            patch_paths.append(patch_path)

    aperture_records: list[dict[str, object]] = []
    for side in aperture_sides:
        spec, point, rotation = center_frames[side]
        aperture_records.append(
            _replace_hand_visual_with_anatomical_r15_aperture(
                stage=stage,
                robot_path=prim_path,
                side=side,
                spec=spec,
                surface_point=point,
                patch_rotation=rotation,
            )
        )

    # Fixed patches must not collide with each other.  This filters only
    # sensor-sensor self-collision; all 54 load-bearing surfaces remain
    # collidable with CarryBox and the external world.
    for patch_path in patch_paths:
        relationship = UsdPhysics.FilteredPairsAPI.Apply(
            stage.GetPrimAtPath(patch_path)
        ).CreateFilteredPairsRel()
        for other_path in patch_paths:
            if other_path != patch_path:
                relationship.AddTarget(Sdf.Path(other_path))

    print(
        "[ANATOMICAL-WHOLE-HAND-TACSL] mounted",
        {
            "patches_per_hand": len(ANATOMICAL_WHOLE_HAND_PATCH_SPECS),
            "total_physical_patches": len(patch_paths),
            "optical_patches": list(anatomical_whole_hand_optical_sensor_names()),
            "custom_patch_thickness_m": _PATCH_THICKNESS_M,
            "collision_neutral": False,
            "official_r15_sha256": _EXPECTED_R15_SHA256,
            "camera_aperture_mode": aperture_mode,
            "camera_aperture_sides": list(aperture_sides),
            "camera_aperture_record_count": len(aperture_records),
            "collision_ownership": collision_ownership_records,
        },
        flush=True,
    )
    return robot_prim


def anatomical_whole_hand_tacsl_robot_cfg(base_robot_cfg, prim_path: str):
    """Return the physical 27-patch sensorized official SUGAR robot config."""

    tactile_spawn = base_robot_cfg.spawn.replace(
        func=spawn_g1_with_anatomical_whole_hand_tacsl
    )
    return base_robot_cfg.replace(prim_path=prim_path, spawn=tactile_spawn)
