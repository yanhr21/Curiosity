# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Plan-15 CarryBox environment with online 54-patch tactile and mass jumps."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import sugar_rl.tasks.locomanip.mdp as mdp

from sugar_rl.tasks.locomanip.online_mass_jump import (
    MASS_FACTORS,
    NOMINAL_CARRYBOX_MASS_KG,
    reset_online_mass_jump,
    post_handoff_box_lift_reward,
    step_online_mass_jump,
)
from sugar_rl.tasks.locomanip.online_mass_jump_action import (
    OnlineMassJumpJointPositionAction,
)
from sugar_rl.tasks.locomanip.online_patch_tactile import (
    PATCH_AREAS_M2,
    PATCH_HISTORY_STEPS,
    SENSOR_NAMES_BY_HAND,
    exact_zero_online_patch_tactile_actor_history,
    normalized_motion_phase,
    online_patch_tactile_actor_history,
    online_patch_tactile_with_slip_actor_history,
)
from sugar_rl.tasks.locomanip.online_teacher_handoff import (
    online_teacher_handoff_training_mask,
    reset_online_teacher_handoff,
    step_online_teacher_handoff,
)

from .base_refiner_env_cfg import BaseActionsCfg, BaseObservationsCfg
from .carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg import (
    AnatomicalCarryBoxRewardsCfg,
    OfficialRefinerAnatomicalWholeHandTacSLSceneCfg,
)
from .carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg import (
    OfficialRefinerAnatomicalWholeHandTacSLAuditEventCfg,
    OfficialRefinerAnatomicalWholeHandTacSLAuditSceneCfg,
)
from .carry_box_refiner_env_cfg import RobotEnvCfg as OfficialCarryBoxRobotEnvCfg


MASS_JUMP_PARAMS = {
    "asset_name": "obj",
    "nominal_mass_kg": NOMINAL_CARRYBOX_MASS_KG,
    "mass_factors": MASS_FACTORS,
    "minimum_lift_m": 0.05,
    "stable_lift_frames": 10,
    "delay_frames": (10, 50),
    # Formal runs inherit the matched RSL-RL/env seed. The fixed-action
    # leakage collector overrides this explicitly with its frozen audit seed.
    "seed": None,
}
PATCH_TERM_PARAMS = {
    "sensor_names_by_hand": SENSOR_NAMES_BY_HAND,
    "patch_areas_m2": PATCH_AREAS_M2,
    "friction_coefficient": 0.5,
    "history_steps": PATCH_HISTORY_STEPS,
}
HANDOFF_PARAMS = {
    "asset_name": "obj",
    "minimum_lift_m": 0.05,
    "stable_lift_frames": 10,
}


@configclass
class ForceOnlyTrainingSceneCfg(OfficialRefinerAnatomicalWholeHandTacSLSceneCfg):
    """Official 54-patch physical skin with optical renderers disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.world_camera = None
        self.left_hand_camera = None
        self.right_hand_camera = None
        for names in SENSOR_NAMES_BY_HAND:
            for sensor_name in names:
                sensor_cfg = getattr(self, sensor_name)
                setattr(
                    self,
                    sensor_name,
                    sensor_cfg.replace(enable_camera_tactile=False, camera_cfg=None),
                )


@configclass
class ForceOnlyAuditSceneCfg(OfficialRefinerAnatomicalWholeHandTacSLAuditSceneCfg):
    """Force-only training geometry plus independent PhysX audit sensors."""

    def __post_init__(self):
        super().__post_init__()
        self.world_camera = None
        self.left_hand_camera = None
        self.right_hand_camera = None
        for names in SENSOR_NAMES_BY_HAND:
            for sensor_name in names:
                sensor_cfg = getattr(self, sensor_name)
                setattr(
                    self,
                    sensor_name,
                    sensor_cfg.replace(enable_camera_tactile=False, camera_cfg=None),
                )


@configclass
class TrackerCommandPolicyCfg(ObsGroup):
    """504-D deployable Tracker/proprio contract without measured object state."""

    ref_joint_pos = ObsTerm(func=mdp.joint_pos, params={"command_name": "motion"})
    ref_root_lin_vel_b = ObsTerm(
        func=mdp.root_lin_vel_b, params={"command_name": "motion"}
    )
    ref_root_ang_vel_b = ObsTerm(
        func=mdp.root_ang_vel_b, params={"command_name": "motion"}
    )
    base_ang_vel_history = ObsTerm(func=mdp.base_ang_vel, history_length=5)
    joint_pos_history = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
    joint_vel_history = ObsTerm(func=mdp.joint_vel_rel, history_length=5)
    actions_history = ObsTerm(func=mdp.last_action, history_length=5)
    projected_gravity_history = ObsTerm(
        func=mdp.project_gravity,
        params={"command_name": "motion"},
        history_length=5,
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    motion_phase = ObsTerm(
        func=normalized_motion_phase,
        params={"command_name": "motion"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class OnlineMassJumpActionsCfg(BaseActionsCfg):
    JointPositionAction = BaseActionsCfg().JointPositionAction.replace(
        class_type=OnlineMassJumpJointPositionAction
    )


@configclass
class LivePatchTactileCfg(ObsGroup):
    online_patch_history = ObsTerm(
        func=online_patch_tactile_actor_history,
        params=PATCH_TERM_PARAMS,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ExactZeroPatchTactileCfg(ObsGroup):
    exact_zero_patch_history = ObsTerm(
        func=exact_zero_online_patch_tactile_actor_history,
        params=PATCH_TERM_PARAMS,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LivePatchTactileWithSlipCfg(ObsGroup):
    online_patch_history = ObsTerm(
        func=online_patch_tactile_with_slip_actor_history,
        params=PATCH_TERM_PARAMS,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TrainingHandoffMaskCfg(ObsGroup):
    active = ObsTerm(func=online_teacher_handoff_training_mask)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LivePatchObservationsCfg:
    policy: TrackerCommandPolicyCfg = TrackerCommandPolicyCfg()
    online_patch_tactile_history: LivePatchTactileCfg = LivePatchTactileCfg()
    critic: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    teacher: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    training_handoff_mask: TrainingHandoffMaskCfg = TrainingHandoffMaskCfg()


@configclass
class ExactZeroPatchObservationsCfg:
    policy: TrackerCommandPolicyCfg = TrackerCommandPolicyCfg()
    online_patch_tactile_history: ExactZeroPatchTactileCfg = (
        ExactZeroPatchTactileCfg()
    )
    critic: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    teacher: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    training_handoff_mask: TrainingHandoffMaskCfg = TrainingHandoffMaskCfg()


@configclass
class LivePatchWithSlipObservationsCfg:
    policy: TrackerCommandPolicyCfg = TrackerCommandPolicyCfg()
    online_patch_tactile_history: LivePatchTactileWithSlipCfg = (
        LivePatchTactileWithSlipCfg()
    )
    critic: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    teacher: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()
    training_handoff_mask: TrainingHandoffMaskCfg = TrainingHandoffMaskCfg()


@configclass
class OnlineMassJumpEventCfg(OfficialRefinerAnatomicalWholeHandTacSLAuditEventCfg):
    """Matched post-physics jump; the jump diagnostics never enter observations."""

    # Disable the official startup mass randomization.  The reset event writes
    # one absolute nominal mass before every continuous episode.
    obj_mass = None
    # Keep both sides of the real PhysX box/pad contact at the same 0.5
    # coefficient used by the TacSL penalty law. The inherited robot-material
    # event also covered the 54 elastomer bodies, so fixing only the object was
    # still insufficient.
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 0.5),
            "dynamic_friction_range": (0.5, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    obj_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("obj", body_names=".*"),
            "static_friction_range": (0.5, 0.5),
            "dynamic_friction_range": (0.5, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    reset_teacher_handoff = EventTerm(
        func=reset_online_teacher_handoff,
        mode="reset",
        params=HANDOFF_PARAMS,
    )
    reset_mass_jump = EventTerm(
        func=reset_online_mass_jump,
        mode="reset",
        params=MASS_JUMP_PARAMS,
    )
    step_teacher_handoff = EventTerm(
        func=step_online_teacher_handoff,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=False,
        params=HANDOFF_PARAMS,
    )
    step_mass_jump = EventTerm(
        func=step_online_mass_jump,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=False,
        params=MASS_JUMP_PARAMS,
    )


@configclass
class OnlineMassRewardsCfg(AnatomicalCarryBoxRewardsCfg):
    """Official tracking terms plus an explicit post-handoff hold outcome."""

    post_handoff_box_lift = RewTerm(
        func=post_handoff_box_lift_reward,
        weight=1.0,
        params={"asset_name": "obj", "target_lift_m": 0.05},
    )


@configclass
class OnlinePatchMassRobotEnvCfg(OfficialCarryBoxRobotEnvCfg):
    scene: ForceOnlyTrainingSceneCfg = ForceOnlyTrainingSceneCfg(
        num_envs=4,
        env_spacing=2.5,
    )
    observations: LivePatchObservationsCfg = LivePatchObservationsCfg()
    actions: OnlineMassJumpActionsCfg = OnlineMassJumpActionsCfg()
    events: OnlineMassJumpEventCfg = OnlineMassJumpEventCfg()
    rewards: OnlineMassRewardsCfg = OnlineMassRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rerender_on_reset = False
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.events.push_robot = None
        self.events.push_object = None
        # Physical elastomer bodies cannot be teleported directly into
        # mid-contact states.  Reach every jump through continuous frame-zero
        # dynamics.
        self.commands.motion.init_with_ref = False
        self.commands.motion.start_init_env_ratio = 1.0


@configclass
class ExactZeroPatchMassRobotEnvCfg(OnlinePatchMassRobotEnvCfg):
    observations: ExactZeroPatchObservationsCfg = ExactZeroPatchObservationsCfg()


@configclass
class OnlinePatchSlipMassRobotEnvCfg(OnlinePatchMassRobotEnvCfg):
    observations: LivePatchWithSlipObservationsCfg = LivePatchWithSlipObservationsCfg()


@configclass
class Fixed3xOnlinePatchSlipMassRobotEnvCfg(OnlinePatchSlipMassRobotEnvCfg):
    """Single-condition serious overfit gate for the corrected PS path.

    This keeps the official Tracker warm start, frozen Refiner handoff, BCPPO,
    actor and rewards. Only the diagnostic condition is frozen: motion45 is
    selected by the launcher, the mass event is always 3x after 20 frames, and
    reset pose noise is disabled.
    """

    def __post_init__(self):
        super().__post_init__()
        for term_name in ("reset_mass_jump", "step_mass_jump"):
            params = getattr(self.events, term_name).params
            params["mass_factors"] = (3.0,)
            params["delay_frames"] = (20, 20)
        self.commands.motion.pose_range = {
            key: (0.0, 0.0)
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.commands.motion.joint_position_range = (0.0, 0.0)


@configclass
class OnlinePatchMassRobotPlayEnvCfg(OnlinePatchMassRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


@configclass
class ExactZeroPatchMassRobotPlayEnvCfg(ExactZeroPatchMassRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


@configclass
class OnlinePatchSlipMassRobotPlayEnvCfg(OnlinePatchSlipMassRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


@configclass
class Fixed3xOnlinePatchSlipMassRobotPlayEnvCfg(
    Fixed3xOnlinePatchSlipMassRobotEnvCfg
):
    """Frozen-review counterpart of the fixed-condition overfit config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


@configclass
class Fixed3xOnlinePatchSlipMassAuditPlayEnvCfg(
    Fixed3xOnlinePatchSlipMassRobotEnvCfg
):
    """Fixed overfit review with evaluator-only PhysX force sensors."""

    scene: ForceOnlyAuditSceneCfg = ForceOnlyAuditSceneCfg(
        num_envs=1,
        env_spacing=2.5,
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


@configclass
class OnlinePatchSlipMassAuditPlayEnvCfg(OnlinePatchSlipMassRobotPlayEnvCfg):
    """No-learning play config with audit-only normal and friction contacts."""

    scene: ForceOnlyAuditSceneCfg = ForceOnlyAuditSceneCfg(
        num_envs=1,
        env_spacing=2.5,
    )
