# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Official PickBottle Tracker scene with physical anatomical TacSL hands."""

from __future__ import annotations

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from sugar_rl.assets.objects.tactile_objects import BOTTLE_SDF_CFG
from sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1 import (
    ANATOMICAL_WHOLE_HAND_PATCH_SPECS,
    anatomical_whole_hand_tacsl_robot_cfg,
)
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg import (
    _patch_sensor_cfg,
    _review_camera,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg import (
    _all_robot_box_contact_cfg,
    _box_contact_cfg,
)

from .pick_bottle_tracker_env_cfg import RobotPlayEnvCfg as OfficialPickBottlePlayEnvCfg
from .pick_bottle_tracker_env_cfg import RobotSceneCfg as OfficialPickBottleSceneCfg


@configclass
class PickBottleAnatomicalWholeHandTacSLSceneCfg(OfficialPickBottleSceneCfg):
    """The official bottle and full G1 with 27 physical patches per hand."""

    robot = anatomical_whole_hand_tacsl_robot_cfg(
        UNITREE_G1_29DOF_MIMIC_CFG,
        "{ENV_REGEX_NS}/Robot",
    )
    obj: RigidObjectCfg = BOTTLE_SDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Obj")
    world_camera: TiledCameraCfg = _review_camera(
        name="WorldCamera",
        position=(3.3, 3.3, 2.25),
        quaternion_wxyz=(
            0.3043649418,
            0.2319667899,
            0.5600173703,
            0.7348019703,
        ),
        width=1280,
        height=720,
    )
    left_hand_camera: TiledCameraCfg = _review_camera(
        name="LeftWholeHandReviewCamera",
        position=(1.8, 1.5, 1.3),
        quaternion_wxyz=(
            0.3407149664581719,
            0.22744536094729864,
            0.5064853014369819,
            0.7587190249646106,
        ),
        width=960,
        height=540,
    )
    right_hand_camera: TiledCameraCfg = _review_camera(
        name="RightWholeHandReviewCamera",
        position=(1.8, -0.9, 1.3),
        quaternion_wxyz=(
            0.7587190249646105,
            0.506485301436982,
            0.2274453609472987,
            0.34071496645817195,
        ),
        width=960,
        height=540,
    )
    left_patch_box_contact = _box_contact_cfg("left")
    right_patch_box_contact = _box_contact_cfg("right")
    all_robot_box_contact = _all_robot_box_contact_cfg()

    def __post_init__(self):
        for side in ("left", "right"):
            for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS:
                setattr(
                    self,
                    f"{side}_{spec.name}_tactile",
                    _patch_sensor_cfg(side, spec.name, spec.optical_r15),
                )


@configclass
class PickBottleAnatomicalWholeHandTacSLEnvCfg(OfficialPickBottlePlayEnvCfg):
    """Single-environment no-learning PickBottle tactile demonstration."""

    scene: PickBottleAnatomicalWholeHandTacSLSceneCfg = (
        PickBottleAnatomicalWholeHandTacSLSceneCfg(num_envs=1, env_spacing=2.5)
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.rerender_on_reset = True
        self.sim.physx.gpu_collision_stack_size = 2**28
