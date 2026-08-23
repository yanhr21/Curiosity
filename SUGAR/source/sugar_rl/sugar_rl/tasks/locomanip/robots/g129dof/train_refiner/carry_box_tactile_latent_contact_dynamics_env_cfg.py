# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Unregistered SUGAR/TacSL latent-contact-dynamics follow-up configs.

Activation is forbidden unless the optimizer-clean current-distribution
seed-42 branch is negative and the protocol's official-code preflight passes.
Keeping these classes out of task registration guarantees that merely adding
this file cannot alter the active training or frozen evaluation suite.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

import sugar_rl.tasks.locomanip.mdp as mdp
from sugar_rl.assets.latent_contact_dynamics import (
    COHERENT_SMALLBOX_SDF_CFG,
    coherent_dual_r15_robot_cfg,
)
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG
from sugar_rl.tasks.locomanip.latent_contact_dynamics_events import (
    apply_reference_contact_phase_lateral_pulse,
    apply_stratified_latent_contact_dynamics,
)
from sugar_rl.tasks.locomanip.latent_contact_visuotactile_sensor import (
    PerEnvironmentFrictionVisuoTactileSensor,
)

from .carry_box_tactile_ablation_env_cfg import _set_tactile_mode
from .carry_box_tactile_reference_only_env_cfg import (
    ReferenceOnlyObservationsCfg,
    ReferenceOnlyRobotEnvCfg,
)
from .carry_box_tactile_refiner_env_cfg import (
    RobotSceneCfg as TactileRobotSceneCfg,
    _palm_sensor_cfg,
)
from .carry_box_refiner_env_cfg import EventCfg as CarryBoxEventCfg


def _latent_palm_sensor_cfg(side: str) -> VisuoTactileSensorCfg:
    """Change only the class hook on the existing official R15 config."""

    return _palm_sensor_cfg(side).replace(class_type=PerEnvironmentFrictionVisuoTactileSensor)


@configclass
class LatentContactRobotSceneCfg(TactileRobotSceneCfg):
    robot = coherent_dual_r15_robot_cfg(UNITREE_G1_29DOF_MIMIC_CFG, "{ENV_REGEX_NS}/Robot")
    obj = COHERENT_SMALLBOX_SDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Obj")
    left_palm_tactile: VisuoTactileSensorCfg = _latent_palm_sensor_cfg("left")
    right_palm_tactile: VisuoTactileSensorCfg = _latent_palm_sensor_cfg("right")


@configclass
class LatentContactEventCfg(CarryBoxEventCfg):
    # Preserve official robot material randomization outside the two collidable
    # R15/palm bodies.  The coherent event owns all possible palm interfaces.
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=(
                    r"^(?!left_tacsl_r15_elastomer$)(?!right_tacsl_r15_elastomer$)"
                    r"(?!left_rubber_hand$)(?!right_rubber_hand$).*$"
                ),
            ),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    obj_physics_material = None
    obj_mass = None
    push_robot = None
    push_object = None

    latent_contact_dynamics = EventTerm(
        func=apply_stratified_latent_contact_dynamics,
        mode="startup",
        params={
            "object_cfg": SceneEntityCfg("obj"),
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_tacsl_r15_elastomer",
                    "right_tacsl_r15_elastomer",
                    "left_rubber_hand",
                    "right_rubber_hand",
                ],
            ),
            "sensor_names": ("left_palm_tactile", "right_palm_tactile"),
            "distribution_seed": 42017,
            "mass_scale_range": (0.5, 2.0),
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "com_y_range_m": (-0.04, 0.04),
            "pulse_magnitude_range_mps": (0.0, 0.8),
        },
    )
    contact_phase_lateral_pulse = EventTerm(
        func=apply_reference_contact_phase_lateral_pulse,
        mode="interval",
        interval_range_s=(0.4, 1.2),
        params={
            "asset_cfg": SceneEntityCfg("obj"),
            "command_name": "motion",
            "dynamics_term_name": "latent_contact_dynamics",
        },
    )


@configclass
class LatentContactReferenceOnlyRobotEnvCfg(ReferenceOnlyRobotEnvCfg):
    scene: LatentContactRobotSceneCfg = LatentContactRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ReferenceOnlyObservationsCfg = ReferenceOnlyObservationsCfg()
    events: LatentContactEventCfg = LatentContactEventCfg()


@configclass
class LatentContactReferenceOnlyZeroRobotEnvCfg(LatentContactReferenceOnlyRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_tactile_mode(self, "zero")


@configclass
class LatentContactReferenceOnlyPressureRobotEnvCfg(LatentContactReferenceOnlyRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_tactile_mode(self, "pressure_only")


class LatentContactReferenceOnlyRobotPlayEnvCfg(LatentContactReferenceOnlyRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


class LatentContactReferenceOnlyZeroRobotPlayEnvCfg(LatentContactReferenceOnlyZeroRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


class LatentContactReferenceOnlyPressureRobotPlayEnvCfg(LatentContactReferenceOnlyPressureRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
