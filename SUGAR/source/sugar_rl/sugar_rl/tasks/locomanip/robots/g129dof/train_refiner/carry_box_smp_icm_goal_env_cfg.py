# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-based SUGAR G1 CarryBox scene for SMP/ICM strategy discovery.

This task preserves the official G1, SDF CarryBox, dual official R15 sensors,
actuator scaling, simulation frequency, domain randomization, and SUGAR reset
states. It removes every per-frame reference-tracking observation, reward, and
termination. The final reference box pose remains only an outcome goal.
"""

from __future__ import annotations

import sugar_rl.tasks.locomanip.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
from sugar_rl.tasks.locomanip import goal_carry_mdp as goal_mdp
from sugar_rl.tasks.locomanip.direct_tactile_history import (
    direct_tactile_force_history,
)
from sugar_rl.tasks.locomanip.goal_tactile_strategy import (
    anti_repeat_strategy_observation,
    repeated_failed_strategy_cost,
    v16_tactile_slip_cost,
    v16_tactile_slip_observation,
)

from .base_refiner_env_cfg import BaseEventCfg
from .carry_box_tactile_refiner_env_cfg import (
    R15_TAXEL_AREA_M2,
    TACTILE_GRID_SHAPE,
    RobotSceneCfg as TactileRobotSceneCfg,
)


MOTION_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

TACTILE_RUNTIME_PARAMS = {
    "left_sensor_name": "left_palm_tactile",
    "right_sensor_name": "right_palm_tactile",
    "history_steps": 4,
    "grid_shape": TACTILE_GRID_SHAPE,
    "taxel_area_m2": R15_TAXEL_AREA_M2,
    "stress_scale": 1.0e-5,
}


@configclass
class CommandsCfg:
    # Motion data seeds valid initial robot/object states and final box goals.
    # GoalCarryMotionCommand never advances or tracks the reference frames.
    motion = goal_mdp.GoalCarryMotionCommandCfg(
        asset_name="robot",
        obj_name="obj",
        obj_mesh_scale=1.0,
        motion_folder=None,
        teacher_motion_folder=None,
        use_generator=False,
        generator_checkpoint_path=None,
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        },
        joint_position_range=(-0.05, 0.05),
        body_names=MOTION_BODY_NAMES,
        key_body_names=[
            "torso_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
        future_frames=8,
        start_init_env_ratio=1.0,
        pool_warmup_steps=1000 * 24,
        rollout_start_distance=1_000_000,
        rollout_window_length=1_000_000,
        init_with_ref=False,
        lifted_height_threshold=0.10,
        success_position_tolerance=0.12,
        success_orientation_tolerance=0.45,
        success_linear_speed_tolerance=0.20,
        success_angular_speed_tolerance=0.40,
        success_stable_steps=20,
    )


@configclass
class ActionsCfg:
    # Preserve the official unbounded policy-unit action before per-joint
    # scale/offset; ICM standardizes a copy without changing robot execution.
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_.*",
            ".*_knee_.*",
            ".*_ankle_.*",
            "waist_.*",
            ".*_shoulder_.*",
            ".*_elbow_.*",
            ".*_wrist_.*",
        ],
        scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
    )


@configclass
class ObservationsCfg:
    @configclass
    class ICMVectorCfg(ObsGroup):
        # Exact order/dimensions match ICM_VECTOR_FIELD_DIMS.
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_height = ObsTerm(
            func=goal_mdp.base_height, params={"command_name": "motion"}
        )
        base_linear_velocity_body = ObsTerm(func=mdp.base_lin_vel)
        base_angular_velocity_body = ObsTerm(func=mdp.base_ang_vel)
        joint_position_relative = ObsTerm(func=mdp.joint_pos_rel)
        joint_velocity = ObsTerm(func=mdp.joint_vel_rel)
        previous_applied_action_policy_units = ObsTerm(
            func=goal_mdp.previous_applied_action_policy_units
        )
        box_position_body = ObsTerm(
            func=goal_mdp.box_position_body, params={"command_name": "motion"}
        )
        box_orientation_tangent_normal_body = ObsTerm(
            func=goal_mdp.box_orientation_tangent_normal_body,
            params={"command_name": "motion"},
        )
        box_linear_velocity_body = ObsTerm(
            func=goal_mdp.box_linear_velocity_body,
            params={"command_name": "motion"},
        )
        box_angular_velocity_body = ObsTerm(
            func=goal_mdp.box_angular_velocity_body,
            params={"command_name": "motion"},
        )
        goal_position_body = ObsTerm(
            func=goal_mdp.goal_position_body, params={"command_name": "motion"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ICMVectorCfg):
        goal_orientation_tangent_normal_body = ObsTerm(
            func=goal_mdp.goal_orientation_tangent_normal_body,
            params={"command_name": "motion"},
        )
        # Both are external actor-visible state. They are deliberately absent
        # from ICMVectorCfg, so neither slip outcome nor failed-strategy memory
        # can define the learned ICM intrinsic signal.
        v16_tactile_slip_belief = ObsTerm(
            func=v16_tactile_slip_observation,
            params=TACTILE_RUNTIME_PARAMS,
        )
        anti_repeat_strategy_state = ObsTerm(
            func=anti_repeat_strategy_observation,
            params=TACTILE_RUNTIME_PARAMS,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TactileHistoryCfg(ObsGroup):
        force_history = ObsTerm(
            func=direct_tactile_force_history,
            params={
                "left_sensor_name": "left_palm_tactile",
                "right_sensor_name": "right_palm_tactile",
                "history_steps": 4,
                "grid_shape": TACTILE_GRID_SHAPE,
                "taxel_area_m2": R15_TAXEL_AREA_M2,
                "stress_scale": 1.0e-5,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()
    icm_vector: ICMVectorCfg = ICMVectorCfg()
    tactile_history: TactileHistoryCfg = TactileHistoryCfg()


@configclass
class EventCfg(BaseEventCfg):
    # No reference-contact-triggered impulses in the discovery baseline.
    # Mass/material randomization inherited from official SUGAR remains live.
    pass


@configclass
class RewardsCfg:
    # Outcome-only task objectives. ICM is learned and logged separately; it is
    # not defined in this ManagerBased reward config.
    goal_position = RewTerm(
        func=goal_mdp.goal_position_reward,
        weight=1.0,
        params={"command_name": "motion", "std": 0.35},
    )
    goal_orientation = RewTerm(
        func=goal_mdp.goal_orientation_reward,
        weight=0.20,
        params={"command_name": "motion", "std": 0.60},
    )
    lift_fraction = RewTerm(
        func=goal_mdp.lift_fraction_reward,
        weight=0.50,
        params={"command_name": "motion", "target_height": 0.25},
    )
    goal_stability = RewTerm(
        func=goal_mdp.goal_stability_reward,
        weight=0.25,
        params={
            "command_name": "motion",
            "position_tolerance": 0.15,
            "speed_tolerance": 0.25,
        },
    )
    # External failure objectives, separately logged from original ICM
    # forward-prediction error and the frozen SMP/SDS prior.
    tactile_slip = RewTerm(
        func=v16_tactile_slip_cost,
        weight=-0.25,
        params=TACTILE_RUNTIME_PARAMS,
    )
    repeated_failed_strategy = RewTerm(
        func=repeated_failed_strategy_cost,
        weight=-0.50,
        params=TACTILE_RUNTIME_PARAMS,
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.10)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.10,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll.*"
            ),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_rubber_hand$)(?!right_rubber_hand$)"
                    r"(?!left_tacsl_r15_elastomer$)"
                    r"(?!right_tacsl_r15_elastomer$)"
                    r"(?!left_anatomical_.*$)"
                    r"(?!right_anatomical_.*$).+$"
                ],
            ),
            "threshold": 0.1,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=goal_mdp.goal_reached, params={"command_name": "motion"})
    unsafe_fall = DoneTerm(
        func=goal_mdp.unsafe_robot_fall,
        params={
            "command_name": "motion",
            "maximum_root_height_loss_m": 0.35,
        },
    )
    dropped_after_lift = DoneTerm(
        func=goal_mdp.dropped_after_lift,
        params={"command_name": "motion", "maximum_height_above_start": 0.04},
    )
    box_out_of_workspace = DoneTerm(
        func=goal_mdp.box_out_of_workspace,
        params={"asset_cfg": SceneEntityCfg("obj"), "maximum_distance": 2.5},
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: TactileRobotSceneCfg = TactileRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**27


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


class PureDiscoveryRobotEnvCfg(RobotEnvCfg):
    """No-result Stage-H phase: discovery plus external failure/safety limits.

    Goal fields stay actor-visible so discovery can later be evaluated against
    carrying outcomes, but success/lift/pose rewards do not train the policy in
    this phase.  Reaching the goal also does not terminate exploration.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.goal_position.weight = 0.0
        self.rewards.goal_orientation.weight = 0.0
        self.rewards.lift_fraction.weight = 0.0
        self.rewards.goal_stability.weight = 0.0
        self.terminations.success = None


class PureDiscoveryRobotPlayEnvCfg(PureDiscoveryRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
