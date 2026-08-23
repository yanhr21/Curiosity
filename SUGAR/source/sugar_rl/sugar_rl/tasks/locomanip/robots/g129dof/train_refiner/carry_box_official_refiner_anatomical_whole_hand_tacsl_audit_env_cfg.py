# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous CarryBox audit scene for the physical anatomical-27 hands.

The additional object-centric ContactSensors expose independent raw box/pad
PhysX tuples for force and spatial audits only. They do not construct or
modify an official TacSL taxel value and do not enter policy or reward.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import sugar_rl.tasks.locomanip.mdp as mdp
from sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1 import (
    ANATOMICAL_WHOLE_HAND_PATCH_SPECS,
)

from .carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg import (
    OfficialRefinerAnatomicalWholeHandTacSLEnvCfg,
    OfficialRefinerAnatomicalWholeHandTacSLSceneCfg,
)
from .carry_box_refiner_env_cfg import EventCfg as OfficialRefinerEventCfg


RAW_PHYSX_CONTACT_AUDIT_CAPACITY_PER_PATCH = 256


def _pad_filter_paths(side: str) -> tuple[str, ...]:
    return tuple(
        f"{{ENV_REGEX_NS}}/Robot/{side}_anatomical_{spec.name}_elastomer"
        for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS
    )


def _box_contact_cfg(filter_exprs: tuple[str, ...]) -> ContactSensorCfg:
    """Use the supported one-box-to-many-filter ContactSensor direction."""

    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Obj",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=list(filter_exprs),
        track_pose=True,
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=(
            RAW_PHYSX_CONTACT_AUDIT_CAPACITY_PER_PATCH * len(filter_exprs)
        ),
    )


def _box_unfiltered_normal_cfg() -> ContactSensorCfg:
    """Independent total normal contact on the box, for coverage audit."""

    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Obj",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )
@configclass
class OfficialRefinerAnatomicalWholeHandTacSLAuditSceneCfg(
    OfficialRefinerAnatomicalWholeHandTacSLSceneCfg
):
    """The unchanged physical sensor scene plus raw audit-only contacts."""

    left_patch_box_contact = _box_contact_cfg(_pad_filter_paths("left"))
    right_patch_box_contact = _box_contact_cfg(_pad_filter_paths("right"))
    all_pads_box_contact = _box_contact_cfg(
        _pad_filter_paths("left") + _pad_filter_paths("right")
    )
    box_all_normal_contact = _box_unfiltered_normal_cfg()


@configclass
class OfficialRefinerAnatomicalWholeHandTacSLAuditEventCfg(
    OfficialRefinerEventCfg
):
    """Apply official robot randomization to the original G1 bodies only.

    The sensorized articulation has the original 35 G1 bodies plus 54
    elastomers and two camera-tip bodies.  Letting either sensor-body class
    enter this startup event both changes its declared material and consumes
    extra RNG draws before the official object material/mass events.
    """

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=(
                    r"^(?!left_anatomical_.*$)"
                    r"(?!right_anatomical_.*$).*$"
                ),
            ),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )


@configclass
class OfficialRefinerAnatomicalWholeHandTacSLAuditEnvCfg(
    OfficialRefinerAnatomicalWholeHandTacSLEnvCfg
):
    """One-environment continuous no-learning evidence config."""

    scene: OfficialRefinerAnatomicalWholeHandTacSLAuditSceneCfg = (
        OfficialRefinerAnatomicalWholeHandTacSLAuditSceneCfg(
            num_envs=1,
            env_spacing=2.5,
        )
    )
    events: OfficialRefinerAnatomicalWholeHandTacSLAuditEventCfg = (
        OfficialRefinerAnatomicalWholeHandTacSLAuditEventCfg()
    )
