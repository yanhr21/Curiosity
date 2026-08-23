# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Official Refiner CarryBox with the physical 27-patch-per-hand TacSL skin.

The policy, critic, action, reference, reward, event, and termination
interfaces remain the official SUGAR Refiner interfaces.  The sensorized robot
is an explicit physical variant: its 54 compliant patches affect contact
dynamics, but none of their values enter the frozen policy or reward.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from sugar_rl.assets.objects.tactile_objects import SMALLBOX_SDF_CFG
from sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1 import (
    ANATOMICAL_WHOLE_HAND_PATCH_SPECS,
    anatomical_whole_hand_tacsl_robot_cfg,
)
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG
import sugar_rl.tasks.locomanip.mdp as mdp

from .carry_box_refiner_env_cfg import (
    ObservationsCfg as OfficialRefinerObservationsCfg,
)
from .carry_box_refiner_env_cfg import RewardsCfg as OfficialCarryBoxRewardsCfg
from .carry_box_refiner_env_cfg import RobotEnvCfg as OfficialRefinerEnvCfg
from .carry_box_refiner_env_cfg import RobotSceneCfg as OfficialRefinerSceneCfg


TACTILE_GRID_SHAPE = (20, 25)
PATCH_OBJECT_CONTACT_SENSOR_NAMES_BY_HAND = tuple(
    tuple(f"{side}_{spec.name}_object_contact" for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS)
    for side in ("left", "right")
)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[9]
_R15_CALIBRATION_ROOT = Path(
    os.environ.get(
        "CURIOSITY_TACSL_CALIBRATION_DIR",
        _WORKSPACE_ROOT
        / "experiments/sugar_reproduction/assets/official_tacsl/calibration",
    )
).resolve()
_R15_RENDER_CFG = GELSIGHT_R15_CFG.replace(
    base_data_path=str(_R15_CALIBRATION_ROOT)
)


def _positive_parameter(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive finite float") from error
    if not (value > 0.0 and value < float("inf")):
        raise ValueError(f"{name} must be a positive finite float")
    return value


def _nonnegative_parameter(name: str, default: float) -> float:
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


def _patch_sensor_cfg(
    side: str,
    patch_name: str,
    optical_r15: bool,
) -> VisuoTactileSensorCfg:
    patch_prefix = f"{side}_anatomical_{patch_name}"
    cfg = VisuoTactileSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{patch_prefix}_elastomer/tactile_sensor",
        update_period=0.02,
        history_length=0,
        debug_vis=False,
        render_cfg=_R15_RENDER_CFG,
        enable_camera_tactile=False,
        enable_force_field=True,
        tactile_array_size=TACTILE_GRID_SHAPE,
        tactile_margin=0.0001,
        tactile_fill_mesh_extents=True,
        tactile_contact_offset_m=_nonnegative_parameter(
            "CURIOSITY_ANATOMICAL_TACSL_CONTACT_OFFSET_M", 0.0003
        ),
        tactile_sampling_axis=1,
        tactile_tip_direction_sign=-1,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/Obj",
        normal_contact_stiffness=_positive_parameter(
            "CURIOSITY_ANATOMICAL_TACSL_NORMAL_STIFFNESS", 1.0
        ),
        friction_coefficient=_positive_parameter(
            "CURIOSITY_ANATOMICAL_TACSL_FRICTION_COEFFICIENT", 0.5
        ),
        tangential_stiffness=_positive_parameter(
            "CURIOSITY_ANATOMICAL_TACSL_TANGENTIAL_STIFFNESS", 0.1
        ),
        camera_cfg=None,
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )
    if not optical_r15:
        return cfg
    return cfg.replace(
        enable_camera_tactile=True,
        camera_cfg=TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{patch_prefix}_tip/cam",
            update_period=0.02,
            height=_R15_RENDER_CFG.image_height,
            width=_R15_RENDER_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


def _patch_object_contact_sensor_cfg(
    side: str,
    patch_name: str,
) -> ContactSensorCfg:
    """One supported one-pad-to-one-box filtered PhysX contact sensor."""

    return ContactSensorCfg(
        prim_path=(
            f"{{ENV_REGEX_NS}}/Robot/{side}_anatomical_"
            f"{patch_name}_elastomer"
        ),
        history_length=3,
        track_air_time=True,
        force_threshold=0.1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obj"],
    )


def _review_camera(
    *,
    name: str,
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
    width: int,
    height: int,
) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        update_period=0.02,
        offset=TiledCameraCfg.OffsetCfg(
            pos=position,
            rot=quaternion_wxyz,
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        width=width,
        height=height,
    )


@configclass
class OfficialRefinerAnatomicalWholeHandTacSLSceneCfg(OfficialRefinerSceneCfg):
    """One official CarryBox scene with 54 physical tactile patches."""

    robot = anatomical_whole_hand_tacsl_robot_cfg(
        UNITREE_G1_29DOF_MIMIC_CFG,
        "{ENV_REGEX_NS}/Robot",
    )
    obj: RigidObjectCfg = SMALLBOX_SDF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Obj",
        spawn=SMALLBOX_SDF_CFG.spawn.replace(activate_contact_sensors=True),
    )
    # IsaacLab does not support filtering several sensor bodies against one
    # object in a single ContactSensor. The inherited dead-hand sensors are
    # removed; __post_init__ creates 54 supported one-pad-to-one-box sensors.
    left_hand_forces = None
    right_hand_forces = None
    world_camera: TiledCameraCfg = _review_camera(
        name="WorldCamera",
        position=(3.6, 3.6, 2.4),
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

    def __post_init__(self):
        for side in ("left", "right"):
            for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS:
                setattr(
                    self,
                    f"{side}_{spec.name}_tactile",
                    _patch_sensor_cfg(side, spec.name, spec.optical_r15),
                )
                setattr(
                    self,
                    f"{side}_{spec.name}_object_contact",
                    _patch_object_contact_sensor_cfg(side, spec.name),
                )


@configclass
class AnatomicalCarryBoxRewardsCfg(OfficialCarryBoxRewardsCfg):
    """Official rewards with bilateral contact read from all live pads."""

    hoi_contact = RewTerm(
        func=mdp.hands_contact,
        weight=1.0,
        params={
            "left_hand_sensor_cfg": PATCH_OBJECT_CONTACT_SENSOR_NAMES_BY_HAND[0],
            "right_hand_sensor_cfg": PATCH_OBJECT_CONTACT_SENSOR_NAMES_BY_HAND[1],
            "command_name": "motion",
            "threshold": 0.1,
        },
    )


@configclass
class OfficialRefinerAnatomicalWholeHandTacSLEnvCfg(OfficialRefinerEnvCfg):
    """Single-environment evaluation config; tactile stays recorder-only."""

    scene: OfficialRefinerAnatomicalWholeHandTacSLSceneCfg = (
        OfficialRefinerAnatomicalWholeHandTacSLSceneCfg(
            num_envs=1,
            env_spacing=2.5,
        )
    )
    observations: OfficialRefinerObservationsCfg = OfficialRefinerObservationsCfg()
    rewards: AnatomicalCarryBoxRewardsCfg = AnatomicalCarryBoxRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.rerender_on_reset = True
        self.sim.physx.gpu_collision_stack_size = 2**28
