# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-material glue for the gated latent-contact-dynamics branch.

The official SUGAR CarryBox, official SUGAR G1, and official IsaacLab R15
assets remain the geometry sources.  These wrappers only bind an explicit
``average``-combine physics material to the box and all four possible palm
interface bodies (two collidable R15 elastomers plus their two rubber-hand
parents).  A later startup event writes identical per-environment material
coefficients to both sides of each interface, making the effective PhysX
coefficient equal to the TacSL coefficient by construction.
"""

from __future__ import annotations

from pxr import Usd, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import bind_physics_material

from sugar_rl.assets.objects.tactile_objects import SMALLBOX_SDF_CFG, spawn_from_usd_with_sdf
from sugar_rl.assets.robots.tacsl_g1 import spawn_g1_with_official_dual_r15


_OFFICIAL_R15_COMPLIANT_CONTACT_STIFFNESS = 10.0
_OFFICIAL_R15_COMPLIANT_CONTACT_DAMPING = 1.0


def _bind_average_contact_material(
    prim_paths: tuple[str, ...],
    material_path: str,
    *,
    official_r15_compliant_contact: bool = False,
) -> None:
    material_kwargs = dict(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        friction_combine_mode="average",
        restitution_combine_mode="average",
    )
    if official_r15_compliant_contact:
        material_kwargs.update(
            compliant_contact_stiffness=(
                _OFFICIAL_R15_COMPLIANT_CONTACT_STIFFNESS
            ),
            compliant_contact_damping=_OFFICIAL_R15_COMPLIANT_CONTACT_DAMPING,
        )
    material_cfg = sim_utils.RigidBodyMaterialCfg(**material_kwargs)
    material_prim = material_cfg.func(material_path, material_cfg)
    material = UsdShade.Material(material_prim)
    stage = material_prim.GetStage()
    for prim_path in prim_paths:
        # Preserve the official helper for ordinary editable colliders.
        bind_physics_material(prim_path, material_path, stronger_than_descendants=True)

        # IsaacLab's nested helper deliberately skips instanced prims.  The
        # official R15 elastomer is exposed as an instance proxy by Isaac Sim
        # 5.1, so author the same physics-purpose binding on its editable
        # reference root as a stronger inherited binding.  This changes no
        # geometry or contact equation; it makes the declared average combine
        # mode resolvable by every descendant collision shape.
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Missing contact-interface prim: {prim_path}")
        binding_api = UsdShade.MaterialBindingAPI.Apply(root_prim)
        binding_api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )


@sim_utils.clone
def spawn_g1_with_coherent_dual_r15_material(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the existing official dual-R15 G1 and bind interface material."""

    robot_prim = spawn_g1_with_official_dual_r15.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    _bind_average_contact_material(
        (
            f"{prim_path}/left_tacsl_r15_elastomer",
            f"{prim_path}/right_tacsl_r15_elastomer",
        ),
        f"{prim_path}/latentContactCompliantPhysicsMaterial",
        official_r15_compliant_contact=True,
    )
    _bind_average_contact_material(
        (
            f"{prim_path}/left_rubber_hand",
            f"{prim_path}/right_rubber_hand",
        ),
        f"{prim_path}/latentContactPhysicsMaterial",
    )
    return robot_prim


@sim_utils.clone
def spawn_smallbox_sdf_with_coherent_material(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the unchanged SUGAR SDF CarryBox and bind interface material."""

    object_prim = spawn_from_usd_with_sdf.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    _bind_average_contact_material(
        (prim_path,),
        f"{prim_path}/latentContactPhysicsMaterial",
    )
    return object_prim


def coherent_dual_r15_robot_cfg(base_robot_cfg, prim_path: str):
    """Return the official G1/R15 config with only the material wrapper changed."""

    return base_robot_cfg.replace(
        prim_path=prim_path,
        spawn=base_robot_cfg.spawn.replace(func=spawn_g1_with_coherent_dual_r15_material),
    )


COHERENT_SMALLBOX_SDF_CFG = SMALLBOX_SDF_CFG.replace(
    spawn=SMALLBOX_SDF_CFG.spawn.replace(func=spawn_smallbox_sdf_with_coherent_material)
)
"""Official SUGAR SDF CarryBox with explicit average-combine contact material."""
