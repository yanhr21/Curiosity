# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""No-RGB CarryBox policy using the current 54-patch native TacSL skin.

The actor keeps the official-width SUGAR reference-only observation contract:
robot state and the commanded motion plan are visible, while measured current
object state and RGB are not.  This permits an exact official Refiner warm
start before the tactile adapter is trained.  Physical-skin resets always
begin at the trajectory start.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import sugar_rl.tasks.locomanip.mdp as mdp
from sugar_rl.tasks.locomanip import rgb_policy_observations as rgb_obs
from sugar_rl.tasks.locomanip.native_whole_hand_tactile_history import (
    NATIVE_TACTILE_GRID_SHAPE,
    NATIVE_TACTILE_HISTORY_STEPS,
    NATIVE_TACTILE_NORMAL_SCALE_N,
    NATIVE_TACTILE_SENSOR_NAMES,
    NATIVE_TACTILE_SHEAR_SCALE_N,
    exact_zero_native_whole_hand_tactile_actor_history,
    native_whole_hand_tactile_actor_history,
)

from .base_refiner_env_cfg import BaseObservationsCfg
from .carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg import (
    OfficialRefinerAnatomicalWholeHandTacSLSceneCfg,
)
from .carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg import (
    OfficialRefinerAnatomicalWholeHandTacSLAuditEventCfg,
)
from .carry_box_refiner_env_cfg import RobotEnvCfg as OfficialCarryBoxRobotEnvCfg
from .carry_box_tactile_reference_only_env_cfg import ReferenceOnlyPolicyCfg


TACTILE_TERM_PARAMS = {
    "sensor_names_by_hand": NATIVE_TACTILE_SENSOR_NAMES,
    "history_steps": NATIVE_TACTILE_HISTORY_STEPS,
    "grid_shape": NATIVE_TACTILE_GRID_SHAPE,
    "normal_scale_n": NATIVE_TACTILE_NORMAL_SCALE_N,
    "shear_scale_n": NATIVE_TACTILE_SHEAR_SCALE_N,
}


@configclass
class ForceOnlyTrainingSceneCfg(OfficialRefinerAnatomicalWholeHandTacSLSceneCfg):
    """Identical physical skin with every RGB/depth renderer disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.world_camera = None
        self.left_hand_camera = None
        self.right_hand_camera = None
        for names in NATIVE_TACTILE_SENSOR_NAMES:
            for sensor_name in names:
                sensor_cfg = getattr(self, sensor_name)
                if sensor_cfg.enable_camera_tactile or sensor_cfg.camera_cfg is not None:
                    setattr(
                        self,
                        sensor_name,
                        sensor_cfg.replace(
                            enable_camera_tactile=False,
                            camera_cfg=None,
                        ),
                    )


@configclass
class NativeTactileCfg(ObsGroup):
    native_force_shear_history = ObsTerm(
        func=native_whole_hand_tactile_actor_history,
        params=TACTILE_TERM_PARAMS,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ExactZeroTactileCfg(ObsGroup):
    exact_zero_force_shear_history = ObsTerm(
        func=exact_zero_native_whole_hand_tactile_actor_history,
        params=TACTILE_TERM_PARAMS,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TrackerCommandPolicyCfg(ObsGroup):
    """Official Tracker history without contact proxy or measured box state."""

    # Deployable official Generator/Tracker prefix: 29 + 3 + 3 = 35.
    # The 36th contact_label output is intentionally excluded.
    ref_joint_pos = ObsTerm(func=mdp.joint_pos, params={"command_name": "motion"})
    ref_root_lin_vel_b = ObsTerm(
        func=mdp.root_lin_vel_b,
        params={"command_name": "motion"},
    )
    ref_root_ang_vel_b = ObsTerm(
        func=mdp.root_ang_vel_b,
        params={"command_name": "motion"},
    )
    # Preserve the released Tracker's five-frame proprioception/action
    # contract instead of collapsing it to a single frame.
    base_ang_vel_history = ObsTerm(func=mdp.base_ang_vel, history_length=5)
    joint_pos_history = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
    joint_vel_history = ObsTerm(func=mdp.joint_vel_rel, history_length=5)
    actions_history = ObsTerm(func=mdp.last_action, history_length=5)
    projected_gravity_history = ObsTerm(
        func=mdp.project_gravity,
        params={"command_name": "motion"},
        history_length=5,
    )
    # Additional deployable current proprioception and motion phase have zero
    # authority at the official Tracker warm start and are learned by BCPPO.
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    motion_phase = ObsTerm(
        func=rgb_obs.normalized_motion_phase,
        params={"command_name": "motion"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class NativeTactileObservationsCfg:
    policy: ReferenceOnlyPolicyCfg = ReferenceOnlyPolicyCfg()
    native_whole_hand_tactile_history: NativeTactileCfg = NativeTactileCfg()
    critic: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    teacher: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()


@configclass
class ExactZeroObservationsCfg:
    policy: ReferenceOnlyPolicyCfg = ReferenceOnlyPolicyCfg()
    native_whole_hand_tactile_history: ExactZeroTactileCfg = ExactZeroTactileCfg()
    critic: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    teacher: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()


@configclass
class TrackerCommandNativeTactileObservationsCfg:
    policy: TrackerCommandPolicyCfg = TrackerCommandPolicyCfg()
    native_whole_hand_tactile_history: NativeTactileCfg = NativeTactileCfg()
    critic: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    teacher: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()


@configclass
class TrackerCommandExactZeroObservationsCfg:
    policy: TrackerCommandPolicyCfg = TrackerCommandPolicyCfg()
    native_whole_hand_tactile_history: ExactZeroTactileCfg = ExactZeroTactileCfg()
    critic: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    teacher: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()


@configclass
class NativeTactileRobotEnvCfg(OfficialCarryBoxRobotEnvCfg):
    scene: ForceOnlyTrainingSceneCfg = ForceOnlyTrainingSceneCfg(
        num_envs=4,
        env_spacing=2.5,
    )
    observations: NativeTactileObservationsCfg = NativeTactileObservationsCfg()
    # Preserve the elastomer materials declared by the physical TacSL skin.
    # The official unsensorized event targets every robot body; on the
    # sensorized robot that would also randomize all 54 elastomer bodies.
    events: OfficialRefinerAnatomicalWholeHandTacSLAuditEventCfg = (
        OfficialRefinerAnatomicalWholeHandTacSLAuditEventCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rerender_on_reset = False
        self.sim.physx.gpu_collision_stack_size = 2**28
        # A mid-trajectory reference-state teleport is invalid once the hands
        # own physical elastomer collision bodies: it skips the preceding
        # contact dynamics and produces broad transient penetration. Every
        # episode therefore starts at official frame zero and reaches tactile
        # contact only through continuous simulation.
        self.commands.motion.init_with_ref = False
        self.commands.motion.start_init_env_ratio = 1.0


@configclass
class ExactZeroRobotEnvCfg(NativeTactileRobotEnvCfg):
    observations: ExactZeroObservationsCfg = ExactZeroObservationsCfg()


@configclass
class TrackerCommandNativeTactileRobotEnvCfg(NativeTactileRobotEnvCfg):
    observations: TrackerCommandNativeTactileObservationsCfg = (
        TrackerCommandNativeTactileObservationsCfg()
    )


@configclass
class TrackerCommandExactZeroRobotEnvCfg(NativeTactileRobotEnvCfg):
    observations: TrackerCommandExactZeroObservationsCfg = (
        TrackerCommandExactZeroObservationsCfg()
    )


class NativeTactileRobotPlayEnvCfg(NativeTactileRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


class ExactZeroRobotPlayEnvCfg(ExactZeroRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


class TrackerCommandNativeTactileRobotPlayEnvCfg(
    TrackerCommandNativeTactileRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


class TrackerCommandExactZeroRobotPlayEnvCfg(TrackerCommandExactZeroRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
