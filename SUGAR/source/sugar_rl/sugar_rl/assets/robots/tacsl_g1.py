# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Official TacSL R15 sensor faces mounted on the official SUGAR G1.

The stock SUGAR robot remains the source articulation.  This spawner wraps its
official URDF conversion and, before Isaac Lab clones the robot, references the
official IsaacLab v2.3.2 R15 elastomer and camera-tip prims as fixed links on
the two rubber hands.  No sensor geometry is reconstructed locally.

The original dual-R15 path is retained as a frozen compatibility control.  The
whole-hand audit path uses repeated *official* R15 elastomer faces: twelve
independently sampled palm regions and three regions on each of the five rigid
rubber digits.  The accepted force path keeps all 54 regions
collision-neutral. Two side-specific palm regions can opt into the official
compliant-contact GelSight camera path through explicit integration windows
for controlled diagnostics only. This is a simulated whole-hand tactile skin,
not a claim that 54 hardware R15 units fit on the physical G1 hand.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_urdf


_WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
_DEFAULT_R15_USD = (
    _WORKSPACE_ROOT
    / "experiments/sugar_reproduction/assets/official_tacsl"
    / "gelsight_r15_finger/gelsight_r15_finger.usd"
)


@dataclass(frozen=True)
class _PalmMount:
    hand_link: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class WholeHandPatchSpec:
    """One direct-TacSL region on the official rigid rubber-hand mesh."""

    name: str
    back_attachment_center_left_hand: tuple[float, float, float]
    digit_angle_deg: float


# The R15 active surface has local center (0, -1.0325 mm, 67.76 mm).
# These transforms put the active face on a palm-mounted standoff, rather than
# embedding the 26 mm-thick R15 housing inside the rubber hand.  A contact-frame
# SDF audit of official CarryBox data measured 45--52 mm from the first mounted
# face to the box.  The joint-frame corrections below close that gap; their
# signs follow the authored fixed-joint frame convention.  The asymmetric
# right-hand correction accounts for the CarryBox/reference-hand geometry and
# was fitted from direct SDF queries over all 500 taxels, not contact labels.
_PALM_MOUNTS = {
    "left": _PalmMount(
        hand_link="left_rubber_hand",
        translation=(0.075, 0.0031265, -0.05776),
        rotation=(1.0, 0.0, 0.0, 0.0),
    ),
    "right": _PalmMount(
        hand_link="right_rubber_hand",
        translation=(0.075, -0.02975, 0.07776),
        rotation=(0.0, 1.0, 0.0, 0.0),
    ),
}

# Official transform of /gelsight_r15_finger/elastomer_tip relative to the
# R15 base/elastomer frame.  It preserves the official camera placement.
_TIP_TRANSLATION = (0.0, -0.0025591400917619467, 0.06775999814271927)
_TIP_ROTATION = (0.7071067690849304, 0.7071067690849304, 0.0, 0.0)

# Authored transform of ``/elastomer_tip/cam`` in the official R15 USD.  The
# camera sits behind the gel and looks through the elastomer.  A palm-mounted
# integration therefore needs a real visual aperture in the rubber-hand shell;
# otherwise the hand is the first surface in every camera ray.
_CAMERA_TRANSLATION_IN_TIP = (0.0, 0.0, -0.018592857142857144)
_CAMERA_ROTATION_IN_TIP = (0.0, 1.0, 0.0, 0.0)

# ``elastomer_tip`` is a camera/coordinate frame in the official fixed-base
# R15 asset.  It has no collision mesh and authors zero mass, which makes
# PhysX fall back to a 1 kg rigid-body mass when it is imported as a link of
# the mobile G1 articulation.  Keep the official prim and camera transform,
# but make this non-physical frame dynamically negligible.  Contact remains
# entirely on the unmodified official elastomer collision body.
_VIRTUAL_TIP_MASS_KG = 1.0e-6
_VIRTUAL_TIP_DIAGONAL_INERTIA_KG_M2 = (1.0e-12, 1.0e-12, 1.0e-12)

# Exact compliant-contact parameters used by the official IsaacLab v2.3.2
# TacSL R15 sensor test. The camera branch renders object penetration relative
# to a no-contact depth baseline; omitting this physics material leaves the
# referenced elastomer rigid and can preserve SDF forces while producing no
# GelSight deformation image.
_R15_COMPLIANT_CONTACT_STIFFNESS = 10.0
_R15_COMPLIANT_CONTACT_DAMPING = 1.0

# Mean of all 500 official R15 taxel positions measured from the referenced
# v2.3.2 elastomer mesh.  Mount translations below are solved from the desired
# active-face center rather than from the bulky sensor base origin.
_R15_TAXEL_CENTER_IN_ELASTOMER = (4.95e-7, -0.00208185, 0.067759994)

# The official R15 elastomer spans local Y=-2.562...+5.497 mm.  Its active
# taxel mean is at local Y=-2.08185 mm, while the flat back attachment plane is
# local Y=+5.497 mm.  Whole-hand patches must attach that *back* plane to the
# rigid rubber-hand surface.  The previous implementation incorrectly put the
# active taxel mean on the hand surface, embedding the approximately 7.58-mm
# gel thickness inside the rigid hand.  Since the audit-only patches have no
# collision response, the underlying rigid hand then stopped the CarryBox
# before palm taxels could enter its SDF.
_R15_BACK_ATTACHMENT_PLANE_Y = 0.005497
_R15_ACTIVE_FACE_STANDOFF_M = (
    _R15_BACK_ATTACHMENT_PLANE_Y - _R15_TAXEL_CENTER_IN_ELASTOMER[1]
)


def _whole_hand_active_face_standoff_m() -> float:
    """Return the explicit whole-hand diagnostic sampling standoff.

    The default preserves the prior back-mounted R15 geometry.  A scan must
    opt in through the environment so that a diagnostic cannot silently
    change the archived mount.
    """

    raw = os.environ.get("CURIOSITY_TACSL_WHOLE_HAND_ACTIVE_STANDOFF_M")
    if raw is None:
        return _R15_ACTIVE_FACE_STANDOFF_M
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            "CURIOSITY_TACSL_WHOLE_HAND_ACTIVE_STANDOFF_M must be a finite "
            f"non-negative number, got {raw!r}"
        ) from error
    if not math.isfinite(value) or value < 0.0 or value > 0.02:
        raise ValueError(
            "CURIOSITY_TACSL_WHOLE_HAND_ACTIVE_STANDOFF_M must be in "
            f"[0, 0.02] m, got {raw!r}"
        )
    return value

# Back-attachment locations come from ray/minimum-surface queries on the
# unchanged official ``left_rubber_hand.STL``.  The right-hand mesh is its Y
# mirror.  After mounting, the official active surface protrudes by the
# measured R15 gel thickness rather than being embedded in the rigid hand.
# The 12 palm tiles span approximately 65 x 65 mm.  Three 20.5-mm-long R15
# taxel regions follow each digit from proximal to distal, so the force atlas
# does not silently interpolate across unsensed gaps.
WHOLE_HAND_TACSL_PATCH_SPECS: tuple[WholeHandPatchSpec, ...] = (
    WholeHandPatchSpec("palm_r0_c0", (0.018, -0.011798, -0.027), 0.0),
    WholeHandPatchSpec("palm_r0_c1", (0.039, -0.012603, -0.027), 0.0),
    WholeHandPatchSpec("palm_r0_c2", (0.060, -0.012288, -0.027), 0.0),
    WholeHandPatchSpec("palm_r1_c0", (0.018, -0.012970, -0.009), 0.0),
    WholeHandPatchSpec("palm_r1_c1", (0.039, -0.011652, -0.009), 0.0),
    WholeHandPatchSpec("palm_r1_c2", (0.060, -0.012605, -0.009), 0.0),
    WholeHandPatchSpec("palm_r2_c0", (0.018, -0.013493, 0.009), 0.0),
    WholeHandPatchSpec("palm_r2_c1", (0.039, -0.012136, 0.009), 0.0),
    WholeHandPatchSpec("palm_r2_c2", (0.060, -0.012948, 0.009), 0.0),
    WholeHandPatchSpec("palm_r3_c0", (0.018, -0.013337, 0.027), 0.0),
    WholeHandPatchSpec("palm_r3_c1", (0.039, -0.013465, 0.027), 0.0),
    WholeHandPatchSpec("palm_r3_c2", (0.060, -0.013220, 0.027), 0.0),
    WholeHandPatchSpec("thumb_s0", (0.052, -0.011299, 0.044), 23.962489),
    WholeHandPatchSpec("thumb_s1", (0.070, -0.027143, 0.052), 23.962489),
    WholeHandPatchSpec("thumb_s2", (0.088, -0.028742, 0.060), 23.962489),
    WholeHandPatchSpec("index_s0", (0.083, -0.011153, 0.029), 0.0),
    WholeHandPatchSpec("index_s1", (0.104, -0.029608, 0.029), 0.0),
    WholeHandPatchSpec("index_s2", (0.125, -0.046825, 0.029), 0.0),
    WholeHandPatchSpec("middle_s0", (0.083, -0.012907, 0.006), 0.0),
    WholeHandPatchSpec("middle_s1", (0.104, -0.023605, 0.006), 0.0),
    WholeHandPatchSpec("middle_s2", (0.125, -0.048159, 0.006), 0.0),
    WholeHandPatchSpec("ring_s0", (0.083, -0.010504, -0.015), 0.0),
    WholeHandPatchSpec("ring_s1", (0.104, -0.025774, -0.015), 0.0),
    WholeHandPatchSpec("ring_s2", (0.125, -0.046179, -0.015), 0.0),
    WholeHandPatchSpec("little_s0", (0.083, -0.014712, -0.035), 0.0),
    WholeHandPatchSpec("little_s1", (0.104, -0.043925, -0.035), 0.0),
    WholeHandPatchSpec("little_s2", (0.125, -0.044333, -0.035), 0.0),
)

# Camera-bearing regions are geometry-fixed by the immutable whole-hand
# standard. They may not move to whichever patch happens to carry more load in
# one trajectory.
WHOLE_HAND_TACSL_CAMERA_REGION_BY_SIDE = {
    "left": "palm_r1_c1",
    "right": "palm_r1_c1",
}


def whole_hand_tacsl_region_has_camera(side: str, region: str) -> bool:
    """Return whether this side/region has the selected official R15 camera."""

    if side not in WHOLE_HAND_TACSL_CAMERA_REGION_BY_SIDE:
        raise ValueError(f"Unknown whole-hand side: {side!r}")
    return region == WHOLE_HAND_TACSL_CAMERA_REGION_BY_SIDE[side]


def whole_hand_tacsl_sensor_names() -> tuple[str, ...]:
    """Return the stable left-then-right whole-hand sensor ordering."""

    return tuple(
        f"{side}_{spec.name}_tactile"
        for side in ("left", "right")
        for spec in WHOLE_HAND_TACSL_PATCH_SPECS
    )


def whole_hand_tacsl_camera_sensor_names() -> tuple[str, ...]:
    """Return the two side-specific, load-bearing optical palm regions."""

    return tuple(
        f"{side}_{WHOLE_HAND_TACSL_CAMERA_REGION_BY_SIDE[side]}_tactile"
        for side in ("left", "right")
    )


def _mount_translation_with_scan_offset(side: str, translation: tuple[float, float, float]):
    """Apply an explicit scan-only hand-frame translation offset when requested."""

    variable = f"CURIOSITY_TACSL_{side.upper()}_MOUNT_TRANSLATION_OFFSET"
    raw = os.environ.get(variable)
    if raw is None:
        return translation
    try:
        offset = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError(f"{variable} must contain three comma-separated floats, got {raw!r}") from error
    if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
        raise ValueError(f"{variable} must contain three finite floats, got {raw!r}")
    return tuple(base + delta for base, delta in zip(translation, offset, strict=True))


def _bind_official_r15_compliant_contact(elastomer_path: str) -> None:
    """Bind the exact v2.3.2 R15 compliant-contact material to one mount."""

    material_cfg = sim_utils.RigidBodyMaterialCfg(
        compliant_contact_stiffness=_R15_COMPLIANT_CONTACT_STIFFNESS,
        compliant_contact_damping=_R15_COMPLIANT_CONTACT_DAMPING,
    )
    material_path = f"{elastomer_path}/compliant_material"
    material_cfg.func(material_path, material_cfg)
    sim_utils.bind_physics_material(elastomer_path, material_path)


def _matrix(
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Quatd(rotation[0], Gf.Vec3d(*rotation[1:])))
    matrix.SetTranslateOnly(Gf.Vec3d(*translation))
    return matrix


def _define_reference(
    stage: Usd.Stage,
    prim_path: str,
    usd_path: str,
    source_prim_path: str,
    prim_type: str = "Xform",
) -> Usd.Prim:
    prim = stage.DefinePrim(prim_path, prim_type)
    prim.GetReferences().AddReference(usd_path, Sdf.Path(source_prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"Failed to reference official TacSL prim at {prim_path}")
    return prim


def _define_fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    hand_path: str,
    sensor_body_path: str,
    body_in_hand: Gf.Matrix4d,
) -> None:
    translation = body_in_hand.ExtractTranslation()
    rotation = body_in_hand.ExtractRotationQuat()
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(sensor_body_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*translation))
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*rotation.GetImaginary()))
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def _quat_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product for WXYZ quaternions."""

    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _whole_hand_patch_rotation(
    side: str,
    digit_angle_deg: float,
) -> tuple[float, float, float, float]:
    """Map the R15 long taxel axis along one rigid digit.

    On the left hand, local ``-Y`` is the outward/palmar sampling direction.
    The mirrored right hand uses local ``+Y``.  The R15 local Z axis is the
    20.5-mm taxel dimension and follows the digit in the hand XZ plane.
    """

    theta = math.radians(digit_angle_deg)
    if side == "left":
        phi = 0.5 * (0.5 * math.pi - theta)
        return (math.cos(phi), 0.0, math.sin(phi), 0.0)
    if side == "right":
        phi = 0.5 * (theta - 0.5 * math.pi)
        q_y = (math.cos(phi), 0.0, math.sin(phi), 0.0)
        return _quat_multiply(q_y, (0.0, 1.0, 0.0, 0.0))
    raise ValueError(f"Unsupported hand side: {side!r}")


def _make_tactile_skin_body_non_interacting(root_prim: Usd.Prim) -> None:
    """Keep a referenced R15 body poseable while removing dynamics changes.

    TacSL's direct force field is evaluated from taxel positions against the
    CarryBox SDF.  It does not require PhysX collision response.  Disabling
    these audit-only patch collisions prevents 54 overlapping R15 solids from
    changing the frozen official SUGAR rollout.
    """

    mass_api = UsdPhysics.MassAPI(root_prim)
    if not mass_api:
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr().Set(_VIRTUAL_TIP_MASS_KG)
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(*_VIRTUAL_TIP_DIAGONAL_INERTIA_KG_M2)
    )
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)


def _replace_hand_visual_with_camera_aperture(
    stage: Usd.Stage,
    robot_path: str,
    side: str,
) -> None:
    """Replace one instanced hand visual with a mesh containing an R15 window.

    The accepted SUGAR collision mesh and articulation are left untouched.
    Only the render mesh is copied out of its URDF instance and opened over the
    side-specific optical palm region.  This represents the camera/gel window
    required by an embedded R15 installation and prevents the opaque rubber
    shell from being mistaken for a static GelSight depth image.
    """

    region = WHOLE_HAND_TACSL_CAMERA_REGION_BY_SIDE[side]
    spec = next(item for item in WHOLE_HAND_TACSL_PATCH_SPECS if item.name == region)
    hand_path = Sdf.Path(f"{robot_path}/{side}_rubber_hand")
    hand_prim = stage.GetPrimAtPath(hand_path)
    if not hand_prim.IsValid():
        raise RuntimeError(f"Cannot find whole-hand aperture link: {hand_path}")

    visual_meshes: list[Usd.Prim] = []
    for prim in Usd.PrimRange(hand_prim, Usd.TraverseInstanceProxies()):
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.ComputePurpose() if imageable else UsdGeom.Tokens.default_
        if purpose == UsdGeom.Tokens.guide or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if "/visuals/" in str(prim.GetPath()):
            visual_meshes.append(prim)
    if len(visual_meshes) != 1:
        raise RuntimeError(
            f"Expected one render mesh below {hand_path}, found "
            f"{[str(prim.GetPath()) for prim in visual_meshes]}"
        )
    source_prim = visual_meshes[0]
    source_mesh = UsdGeom.Mesh(source_prim)
    points = source_mesh.GetPointsAttr().Get()
    counts = source_mesh.GetFaceVertexCountsAttr().Get()
    indices = source_mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        raise RuntimeError(f"Hand render mesh has empty topology: {source_prim.GetPath()}")

    # Convert all vertices to the hand-link frame before creating a direct,
    # non-instanced visual mesh.
    mesh_in_hand, _ = UsdGeom.XformCache().ComputeRelativeTransform(
        source_prim, hand_prim
    )
    hand_points = [mesh_in_hand.Transform(Gf.Vec3d(point)) for point in points]

    back_attachment_center = list(spec.back_attachment_center_left_hand)
    if side == "right":
        back_attachment_center[1] = -back_attachment_center[1]
    outward_sign = -1.0 if side == "left" else 1.0
    active_center = list(back_attachment_center)
    active_center[1] += outward_sign * _whole_hand_active_face_standoff_m()
    rotation = _whole_hand_patch_rotation(side, spec.digit_angle_deg)
    rotation_gf = Gf.Rotation(Gf.Quatd(rotation[0], Gf.Vec3d(*rotation[1:])))
    taxel_center_offset = rotation_gf.TransformDir(
        Gf.Vec3d(*_R15_TAXEL_CENTER_IN_ELASTOMER)
    )
    base_translation = tuple(
        active_center[index] - taxel_center_offset[index] for index in range(3)
    )
    mount_in_hand = _matrix(base_translation, rotation)
    camera_in_tip = _matrix(
        _CAMERA_TRANSLATION_IN_TIP, _CAMERA_ROTATION_IN_TIP
    )
    camera_in_hand = camera_in_tip * _matrix(_TIP_TRANSLATION, _TIP_ROTATION) * mount_in_hand
    camera_center = camera_in_hand.ExtractTranslation()

    # The official camera FOV at the rubber shell is smaller than the
    # 16.3-by-20.5-mm taxel face.  A 24-by-29-mm rectangular integration
    # window adds a conservative 3.8--4.3-mm border while remaining confined
    # to the selected palm tile.
    half_x = 0.012
    half_z = 0.0145
    center_x = active_center[0]
    center_z = active_center[2]
    retained_counts: list[int] = []
    retained_indices: list[int] = []
    offset = 0
    removed_faces = 0
    for count in counts:
        count = int(count)
        face_indices = [int(value) for value in indices[offset : offset + count]]
        offset += count
        face_points = [hand_points[index] for index in face_indices]
        min_x = min(point[0] for point in face_points)
        max_x = max(point[0] for point in face_points)
        min_z = min(point[2] for point in face_points)
        max_z = max(point[2] for point in face_points)
        mean_y = sum(point[1] for point in face_points) / count
        overlaps_window = (
            min_x <= center_x + half_x
            and max_x >= center_x - half_x
            and min_z <= center_z + half_z
            and max_z >= center_z - half_z
        )
        lies_between_camera_and_gel = (
            outward_sign * (mean_y - camera_center[1]) >= -0.001
        )
        if overlaps_window and lies_between_camera_and_gel:
            removed_faces += 1
            continue
        retained_counts.append(count)
        retained_indices.extend(face_indices)
    if removed_faces < 10:
        raise RuntimeError(
            f"R15 visual aperture removed only {removed_faces} faces on {side}; "
            "the hand-frame geometry contract is inconsistent"
        )

    source_visual_root = stage.GetPrimAtPath(
        hand_path.AppendChild("visuals")
    )
    if not source_visual_root.IsValid():
        raise RuntimeError(f"Cannot find instanced visual root below {hand_path}")
    UsdGeom.Imageable(source_visual_root).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )

    hand_points_f = Vt.Vec3fArray(
        [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in hand_points]
    )
    aperture_mesh = UsdGeom.Mesh.Define(
        stage, hand_path.AppendChild("whole_hand_tacsl_aperture_visual")
    )
    aperture_mesh.CreatePointsAttr().Set(hand_points_f)
    aperture_mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(retained_counts))
    aperture_mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(retained_indices))
    aperture_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    aperture_mesh.CreateDoubleSidedAttr().Set(source_mesh.GetDoubleSidedAttr().Get() or False)
    aperture_mesh.CreateExtentAttr().Set(
        UsdGeom.PointBased.ComputeExtent(hand_points_f)
    )
    bound_material, _ = UsdShade.MaterialBindingAPI(source_prim).ComputeBoundMaterial()
    if bound_material:
        UsdShade.MaterialBindingAPI.Apply(aperture_mesh.GetPrim()).Bind(bound_material)
    print(
        "[WHOLE-HAND-TACSL] camera_aperture",
        {
            "side": side,
            "region": region,
            "camera_center_hand_m": tuple(float(value) for value in camera_center),
            "removed_faces": removed_faces,
            "retained_faces": len(retained_counts),
            "collision_mesh_unchanged": True,
        },
    )


def _mount_official_whole_hand_patch(
    stage: Usd.Stage,
    robot_path: str,
    side: str,
    spec: WholeHandPatchSpec,
    usd_path: str,
    material: UsdShade.Material,
) -> None:
    """Mount one official R15 taxel surface at a mesh-audited hand region."""

    hand_link = f"{side}_rubber_hand"
    hand_path = f"{robot_path}/{hand_link}"
    hand_prim = stage.GetPrimAtPath(hand_path)
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not hand_prim.IsValid() or not robot_prim.IsValid():
        raise RuntimeError(f"Official SUGAR hand prim is missing: {hand_path}")

    back_attachment_center = list(spec.back_attachment_center_left_hand)
    if side == "right":
        back_attachment_center[1] = -back_attachment_center[1]
    # The official R15 front face points toward -hand-Y on the left and
    # +hand-Y on the mirrored right.  Attach the elastomer back to the hand
    # surface and place the active taxel mean one measured gel thickness
    # outward.
    outward_sign = -1.0 if side == "left" else 1.0
    active_center = back_attachment_center
    active_center[1] += outward_sign * _whole_hand_active_face_standoff_m()
    rotation = _whole_hand_patch_rotation(side, spec.digit_angle_deg)
    rotation_gf = Gf.Rotation(Gf.Quatd(rotation[0], Gf.Vec3d(*rotation[1:])))
    taxel_center_offset = rotation_gf.TransformDir(
        Gf.Vec3d(*_R15_TAXEL_CENTER_IN_ELASTOMER)
    )
    base_translation = tuple(
        active_center[index] - taxel_center_offset[index] for index in range(3)
    )
    mount_in_hand = _matrix(base_translation, rotation)

    patch_prefix = f"{side}_wholehand_{spec.name}"
    elastomer_path = f"{robot_path}/{patch_prefix}_elastomer"
    elastomer_prim = _define_reference(
        stage,
        elastomer_path,
        usd_path,
        "/gelsight_r15_finger/elastomer",
    )
    camera_enabled = whole_hand_tacsl_region_has_camera(side, spec.name)
    physical_optical_enabled = (
        os.environ.get("CURIOSITY_TACSL_WHOLE_HAND_PHYSICAL_OPTICAL", "0")
        == "1"
    )
    if camera_enabled and physical_optical_enabled:
        # The official GelSight depth/RGB path depends on the R15
        # compliant-contact collision.  Disabling collision is valid for the
        # direct SDF force field but makes the optical stream static.  Physical
        # optical contact is opt-in because the two colliders change CarryBox
        # dynamics; it is used by controlled presses, not by the accepted
        # collision-neutral force rollout.
        _bind_official_r15_compliant_contact(elastomer_path)
    else:
        _make_tactile_skin_body_non_interacting(elastomer_prim)
    if not camera_enabled:
        # Fifty-two dense force-only R15 regions otherwise sit inside the
        # selected GelSight cameras' optical frusta.  They do not deform
        # because they are collision-neutral, so their nearer visual meshes
        # permanently occlude the physical camera-bearing gel.  Hide only
        # these renderer meshes; the referenced official geometry, 500 taxel
        # positions, rigid-body pose, SDF query, pressure, and shear remain
        # unchanged and independently archived.
        for render_prim in Usd.PrimRange(
            elastomer_prim, Usd.TraverseInstanceProxies()
        ):
            imageable = UsdGeom.Imageable(render_prim)
            if imageable:
                imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    for prim in Usd.PrimRange(elastomer_prim, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    xform_cache = UsdGeom.XformCache()
    hand_in_robot, _ = xform_cache.ComputeRelativeTransform(hand_prim, robot_prim)
    UsdGeom.Xformable(elastomer_prim).MakeMatrixXform().Set(mount_in_hand * hand_in_robot)
    _define_fixed_joint(
        stage,
        f"{robot_path}/joints/{patch_prefix}_elastomer_joint",
        hand_path,
        elastomer_path,
        mount_in_hand,
    )

    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(hand_prim).CreateFilteredPairsRel()
    filtered_pairs.AddTarget(Sdf.Path(elastomer_path))

    if not camera_enabled:
        return

    tip_path = f"{robot_path}/{patch_prefix}_tip"
    tip_prim = _define_reference(
        stage,
        tip_path,
        usd_path,
        "/gelsight_r15_finger/elastomer_tip",
    )
    _make_tactile_skin_body_non_interacting(tip_prim)
    for prim in Usd.PrimRange(tip_prim, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    tip_in_elastomer = _matrix(_TIP_TRANSLATION, _TIP_ROTATION)
    tip_in_hand = tip_in_elastomer * mount_in_hand
    UsdGeom.Xformable(tip_prim).MakeMatrixXform().Set(tip_in_hand * hand_in_robot)
    _define_fixed_joint(
        stage,
        f"{robot_path}/joints/{patch_prefix}_tip_joint",
        hand_path,
        tip_path,
        tip_in_hand,
    )
    filtered_pairs.AddTarget(Sdf.Path(tip_path))


def _mount_official_r15(stage: Usd.Stage, robot_path: str, side: str, usd_path: str) -> None:
    mount = _PALM_MOUNTS[side]
    hand_path = f"{robot_path}/{mount.hand_link}"
    hand_prim = stage.GetPrimAtPath(hand_path)
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not hand_prim.IsValid() or not robot_prim.IsValid():
        raise RuntimeError(f"Official SUGAR hand prim is missing: {hand_path}")

    elastomer_path = f"{robot_path}/{side}_tacsl_r15_elastomer"
    tip_path = f"{robot_path}/{side}_tacsl_r15_tip"
    elastomer_prim = _define_reference(
        stage,
        elastomer_path,
        usd_path,
        "/gelsight_r15_finger/elastomer",
    )
    tip_prim = _define_reference(
        stage,
        tip_path,
        usd_path,
        "/gelsight_r15_finger/elastomer_tip",
    )
    tip_mass_api = UsdPhysics.MassAPI(tip_prim)
    if not tip_mass_api:
        tip_mass_api = UsdPhysics.MassAPI.Apply(tip_prim)
    tip_mass_api.CreateMassAttr().Set(_VIRTUAL_TIP_MASS_KG)
    tip_mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(*_VIRTUAL_TIP_DIAGONAL_INERTIA_KG_M2)
    )
    material_prim = _define_reference(
        stage,
        f"{robot_path}/{side}_tacsl_r15_material",
        usd_path,
        "/gelsight_r15_finger/Looks/material_0",
        prim_type="Material",
    )
    material = UsdShade.Material(material_prim)
    for sensor_root in (elastomer_prim, tip_prim):
        for prim in Usd.PrimRange(sensor_root):
            if prim.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    _bind_official_r15_compliant_contact(elastomer_path)

    mount_translation = _mount_translation_with_scan_offset(side, mount.translation)
    mount_in_hand = _matrix(mount_translation, mount.rotation)
    tip_in_elastomer = _matrix(_TIP_TRANSLATION, _TIP_ROTATION)
    tip_in_hand = tip_in_elastomer * mount_in_hand

    # Author initial poses that already satisfy the fixed-joint frames.  This
    # avoids a projection impulse on the first physics step.
    xform_cache = UsdGeom.XformCache()
    hand_in_robot, _ = xform_cache.ComputeRelativeTransform(hand_prim, robot_prim)
    UsdGeom.Xformable(elastomer_prim).MakeMatrixXform().Set(mount_in_hand * hand_in_robot)
    UsdGeom.Xformable(tip_prim).MakeMatrixXform().Set(tip_in_hand * hand_in_robot)

    _define_fixed_joint(
        stage,
        f"{robot_path}/joints/{side}_tacsl_r15_elastomer_joint",
        hand_path,
        elastomer_path,
        mount_in_hand,
    )
    _define_fixed_joint(
        stage,
        f"{robot_path}/joints/{side}_tacsl_r15_tip_joint",
        hand_path,
        tip_path,
        tip_in_hand,
    )

    # The palm and embedded sensor housing overlap intentionally.  Filter only
    # those self-pairs; the official elastomer remains collidable with CarryBox.
    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(hand_prim).CreateFilteredPairsRel()
    filtered_pairs.AddTarget(Sdf.Path(elastomer_path))
    filtered_pairs.AddTarget(Sdf.Path(tip_path))


@sim_utils.clone
def spawn_g1_with_official_dual_r15(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn official SUGAR G1, then add two official R15 sensor faces."""
    robot_prim = spawn_from_urdf.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    usd_path = str(Path(os.environ.get("CURIOSITY_TACSL_R15_USD", _DEFAULT_R15_USD)).resolve())
    if not Path(usd_path).is_file():
        raise FileNotFoundError(
            "Official IsaacLab v2.3.2 TacSL R15 asset is missing. "
            f"Expected: {usd_path}"
        )

    stage = get_current_stage()
    for side in ("left", "right"):
        _mount_official_r15(stage, prim_path, side, usd_path)
    return robot_prim


def dual_r15_robot_cfg(base_robot_cfg, prim_path: str):
    """Return a tactile research copy of the official SUGAR robot config."""
    tactile_spawn = base_robot_cfg.spawn.replace(func=spawn_g1_with_official_dual_r15)
    return base_robot_cfg.replace(prim_path=prim_path, spawn=tactile_spawn)


@sim_utils.clone
def spawn_g1_with_official_whole_hand_tacsl(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn official SUGAR G1 with 27 direct-TacSL regions per rigid hand."""

    robot_prim = spawn_from_urdf.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    # Diagnostic-only proof of camera occlusion.  When paired with the
    # single-patch switch below, hiding the selected rigid hand should restore
    # the official R15 optical deformation if (and only if) that hand shell is
    # what blocks the camera.  This is never enabled in a CarryBox run.
    hidden_hand = os.environ.get("CURIOSITY_TACSL_WHOLE_HAND_HIDE_RIGID_HAND")
    if hidden_hand is not None:
        if hidden_hand not in ("left", "right"):
            raise ValueError(
                "CURIOSITY_TACSL_WHOLE_HAND_HIDE_RIGID_HAND must be left or "
                f"right, got {hidden_hand!r}"
            )
        hand_prim = get_current_stage().GetPrimAtPath(
            f"{prim_path}/{hidden_hand}_rubber_hand"
        )
        if not hand_prim.IsValid():
            raise RuntimeError(f"Cannot find diagnostic hand prim: {hand_prim.GetPath()}")
        # Author only on the link root.  URDF-imported visual descendants are
        # instance proxies and cannot receive local property specs, while root
        # visibility is inherited by the complete hand hierarchy.
        UsdGeom.Imageable(hand_prim).CreateVisibilityAttr().Set(
            UsdGeom.Tokens.invisible
        )
    usd_path = str(Path(os.environ.get("CURIOSITY_TACSL_R15_USD", _DEFAULT_R15_USD)).resolve())
    if not Path(usd_path).is_file():
        raise FileNotFoundError(
            "Official IsaacLab v2.3.2 TacSL R15 asset is missing. "
            f"Expected: {usd_path}"
        )

    stage = get_current_stage()
    # Diagnostic-only isolation switch used by the direct press bench.  It
    # lets the selected official R15 patch be spawned without the other 53
    # referenced meshes so that camera occlusion can be distinguished from an
    # incorrect sensor-internal transform.  Normal CarryBox configurations do
    # not set this variable and therefore retain the complete 54-region skin.
    only_patch = os.environ.get("CURIOSITY_TACSL_WHOLE_HAND_ONLY_PATCH")
    if only_patch is not None:
        valid_only_patches = {
            f"{side}:{spec.name}"
            for side in ("left", "right")
            for spec in WHOLE_HAND_TACSL_PATCH_SPECS
        }
        if only_patch not in valid_only_patches:
            raise ValueError(
                "CURIOSITY_TACSL_WHOLE_HAND_ONLY_PATCH must be one of "
                f"{sorted(valid_only_patches)}, got {only_patch!r}"
            )
    material_prim = _define_reference(
        stage,
        f"{prim_path}/whole_hand_tacsl_material",
        usd_path,
        "/gelsight_r15_finger/Looks/material_0",
        prim_type="Material",
    )
    material = UsdShade.Material(material_prim)
    # Physical optical regions require a camera/gel window, but the accepted
    # CarryBox force path remains collision-neutral and therefore defaults to
    # no window. The direct press bench explicitly selects one side.
    aperture_mode = os.environ.get(
        "CURIOSITY_TACSL_WHOLE_HAND_CAMERA_APERTURES", "none"
    )
    if aperture_mode not in ("none", "left", "right", "both"):
        raise ValueError(
            "CURIOSITY_TACSL_WHOLE_HAND_CAMERA_APERTURES must be none, left, "
            f"right, or both, got {aperture_mode!r}"
        )
    aperture_sides = {
        "none": (),
        "left": ("left",),
        "right": ("right",),
        "both": ("left", "right"),
    }[aperture_mode]
    for aperture_side in aperture_sides:
        _replace_hand_visual_with_camera_aperture(
            stage, prim_path, aperture_side
        )
    for side in ("left", "right"):
        for spec in WHOLE_HAND_TACSL_PATCH_SPECS:
            if only_patch is not None and f"{side}:{spec.name}" != only_patch:
                continue
            _mount_official_whole_hand_patch(
                stage,
                prim_path,
                side,
                spec,
                usd_path,
                material,
            )
    return robot_prim


def whole_hand_tacsl_robot_cfg(base_robot_cfg, prim_path: str):
    """Return a collision-neutral whole-hand tactile copy of SUGAR's G1."""

    tactile_spawn = base_robot_cfg.spawn.replace(func=spawn_g1_with_official_whole_hand_tacsl)
    return base_robot_cfg.replace(prim_path=prim_path, spawn=tactile_spawn)
