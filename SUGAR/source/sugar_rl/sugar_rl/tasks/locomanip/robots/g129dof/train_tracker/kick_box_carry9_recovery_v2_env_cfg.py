"""Fixed-condition BIGBOX recovery after an online CarryBox skill prefix."""

from __future__ import annotations

from pathlib import Path
import os

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import sugar_rl.tasks.locomanip.mdp as mdp
import isaaclab.envs.mdp as isaac_mdp

from .kick_box_tracker_env_cfg import (
    ActionsCfg,
    EventCfg as KickEventCfg,
    RewardsCfg as KickRewardsCfg,
    RobotEnvCfg as KickRobotEnvCfg,
    RobotSceneCfg,
)
from ..inference.base_inference_env_cfg import BaseCommandsCfg
from .base_tracker_env_cfg import BaseObservationsCfg


_SUGAR_ROOT = Path(__file__).resolve().parents[8]


def synchronized_physical_invalid(
    env,
    asset_cfg: SceneEntityCfg,
    minimum_root_height_m: float,
    maximum_root_linear_speed_mps: float,
    maximum_root_angular_speed_rps: float,
    maximum_joint_speed_rps: float,
) -> torch.Tensor:
    """End the synchronized recovery batch before one outlier poisons PPO."""

    asset = env.scene[asset_cfg.name]
    root = asset.data.root_state_w
    invalid = (
        ~torch.isfinite(root).all(dim=-1)
        | ~torch.isfinite(asset.data.joint_vel).all(dim=-1)
        | (root[:, 2] < minimum_root_height_m)
        | (torch.linalg.vector_norm(root[:, 7:10], dim=-1) > maximum_root_linear_speed_mps)
        | (torch.linalg.vector_norm(root[:, 10:13], dim=-1) > maximum_root_angular_speed_rps)
        | (torch.amax(torch.abs(asset.data.joint_vel), dim=-1) > maximum_joint_speed_rps)
    )
    return torch.ones_like(invalid) & torch.any(invalid)


@configclass
class CommandsCfg(BaseCommandsCfg):
    def __post_init__(self):
        self.motion.motion_folder = str(_SUGAR_ROOT / "data/KickBox/data_021")
        self.motion.generator_checkpoint_path = str(
            _SUGAR_ROOT / "demo_ckpts/KickBox/generator.ckpt"
        )
        self.motion.eval_mode = True
        self.motion.eval_random_motion = False
        self.motion.start_init_env_ratio = 1.0
        self.motion.init_with_ref = False


@configclass
class ObservationsCfg(BaseObservationsCfg):
    @configclass
    class TrackerCfg(ObsGroup):
        generated_command = ObsTerm(
            func=mdp.generated_command, params={"command_name": "motion"}
        )
        base_ang_vel_history = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=5,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_pos_history = ObsTerm(
            func=mdp.joint_pos_rel,
            history_length=5,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_history = ObsTerm(
            func=mdp.joint_vel_rel,
            history_length=5,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions_history = ObsTerm(func=mdp.last_action, history_length=5)
        project_gravity = ObsTerm(
            func=mdp.project_gravity,
            params={"command_name": "motion"},
            history_length=5,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        obj_pos_b = ObsTerm(
            func=mdp.obj_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        obj_ori_b = ObsTerm(
            func=mdp.obj_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            # This first recovery run is a fixed-condition learnability
            # diagnostic and must match its frozen evaluator.  Domain
            # randomization belongs only after this endpoint is established.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: TrackerCfg = TrackerCfg()

    @configclass
    class TeacherCfg(TrackerCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    # The released Kick Tracker is the behavior anchor.  This is not a fake
    # Refiner: teacher and deployed actor both use the official 510-D Tracker
    # contract, while the critic keeps the inherited 890-D privileged input.
    teacher: TeacherCfg = TeacherCfg()


@configclass
class EventCfg(KickEventCfg):
    """Deterministic physics for the first fixed-condition overfit."""

    push_robot = None
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    obj_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("obj", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    obj_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("obj"),
            "operation": "scale",
            "mass_distribution_params": (1.0, 1.0),
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    add_joint_default_pos = None
    base_com = None


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    physical_invalid = DoneTerm(
        func=synchronized_physical_invalid,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_root_height_m": 0.45,
            "maximum_root_linear_speed_mps": 20.0,
            "maximum_root_angular_speed_rps": 30.0,
            "maximum_joint_speed_rps": 100.0,
        },
    )


@configclass
class RewardsCfg(KickRewardsCfg):
    physical_invalid_penalty = RewTerm(
        func=isaac_mdp.is_terminated_term,
        weight=-10.0,
        params={"term_keys": "physical_invalid"},
    )


@configclass
class RobotEnvCfg(KickRobotEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        if os.environ.get("SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY", "1") == "0":
            self.rewards.physical_invalid_penalty = None
        self.episode_length_s = 6.0
        self.is_finite_horizon = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
