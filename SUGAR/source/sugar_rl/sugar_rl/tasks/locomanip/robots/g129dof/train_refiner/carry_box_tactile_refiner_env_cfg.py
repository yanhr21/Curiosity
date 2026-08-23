# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""SUGAR CarryBox Refiner with official TacSL spatial force-field sensing.

This task inherits the accepted SUGAR control and changes only the CarryBox
collision approximation plus two official R15 elastomer/camera-tip assemblies
fixed to the inward palm faces and their policy observation group.  Camera
rendering stays disabled during force-field policy training; a separate
validation entry point enables and records the official GelSight RGB/depth
path.

Fidelity boundary
-----------------
The force field is computed with the official TacSL SDF penalty equations;
an audited R15 channel mapping exposes their non-negative scalar normal load
and signed friction-only two-axis shear. Each hand retains a 20 by 25 array.
The observation adapter converts forces to per-taxel stress and applies one
documented scalar conditioning factor; it does not threshold, pool, integrate,
or replace the spatial field with a contact label.  The accepted SUGAR policy
and critic observations, task commands, actions, rewards, terminations, events,
robot description, CarryBox geometry, and reference motion remain intact.

The sensor surfaces and camera transforms are referenced from the official
IsaacLab v2.3.2 R15 USD before the official G1 articulation is cloned, so every
environment receives the same fixed palm links without changing the 29-DoF
action space.  These outputs remain ``high-fidelity simulated tactile`` until
physical GelSight calibration validates footprint, load, shear/slip, latency,
noise, and image response.
"""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

import sugar_rl.tasks.locomanip.mdp as mdp
from sugar_rl.assets.objects.tactile_objects import SMALLBOX_SDF_CFG
from sugar_rl.assets.robots.tacsl_g1 import dual_r15_robot_cfg
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG

from .carry_box_refiner_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg as CarryBoxObservationsCfg,
    RewardsCfg,
    RobotEnvCfg as CarryBoxRobotEnvCfg,
    TerminationsCfg,
)
from .carry_box_refiner_env_cfg import RobotSceneCfg as CarryBoxRobotSceneCfg


TACTILE_GRID_SHAPE = (20, 25)
# R15 active area (320*0.0877 mm by 240*0.0877 mm) divided by 500 taxels.
R15_TAXEL_AREA_M2 = 1.18138624e-6
_WORKSPACE_ROOT = Path(__file__).resolve().parents[9]
_R15_CALIBRATION_ROOT = Path(
    os.environ.get(
        "CURIOSITY_TACSL_CALIBRATION_DIR",
        _WORKSPACE_ROOT / "experiments/sugar_reproduction/assets/official_tacsl/calibration",
    )
).resolve()
_R15_RENDER_CFG = GELSIGHT_R15_CFG.replace(base_data_path=str(_R15_CALIBRATION_ROOT))


def _palm_sensor_cfg(side: str) -> VisuoTactileSensorCfg:
    return VisuoTactileSensorCfg(
        # SensorBase treats the final component as the sensor name and samples
        # the parent rigid body's visual mesh as the elastomer surface.
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_tacsl_r15_elastomer/tactile_sensor",
        update_period=0.02,
        history_length=0,
        debug_vis=False,
        render_cfg=_R15_RENDER_CFG,
        enable_camera_tactile=False,
        enable_force_field=True,
        tactile_array_size=TACTILE_GRID_SHAPE,
        tactile_margin=0.003,
        contact_object_prim_path_expr="{ENV_REGEX_NS}/Obj",
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        camera_cfg=None,
        trimesh_vis_tactile_points=False,
        visualize_sdf_closest_pts=False,
    )


def _palm_camera_sensor_cfg(side: str) -> VisuoTactileSensorCfg:
    """Enable the official R15 depth camera and TAXIM renderer for recording."""
    return _palm_sensor_cfg(side).replace(
        enable_camera_tactile=True,
        camera_cfg=TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_tacsl_r15_tip/cam",
            update_period=0.02,
            height=_R15_RENDER_CFG.image_height,
            width=_R15_RENDER_CFG.image_width,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
    )


@configclass
class RobotSceneCfg(CarryBoxRobotSceneCfg):
    robot = dual_r15_robot_cfg(UNITREE_G1_29DOF_MIMIC_CFG, "{ENV_REGEX_NS}/Robot")
    obj: RigidObjectCfg = SMALLBOX_SDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Obj")
    left_palm_tactile: VisuoTactileSensorCfg = _palm_sensor_cfg("left")
    right_palm_tactile: VisuoTactileSensorCfg = _palm_sensor_cfg("right")


@configclass
class RobotLeftCameraSceneCfg(RobotSceneCfg):
    """Dual-R15 scene recording the official left 320x240 RGB/depth stream."""

    left_palm_tactile: VisuoTactileSensorCfg = _palm_camera_sensor_cfg("left")


@configclass
class RobotRightCameraSceneCfg(RobotSceneCfg):
    """Dual-R15 scene recording the official right 320x240 RGB/depth stream."""

    right_palm_tactile: VisuoTactileSensorCfg = _palm_camera_sensor_cfg("right")


@configclass
class ObservationsCfg(CarryBoxObservationsCfg):
    @configclass
    class TactileCfg(ObsGroup):
        force_maps = ObsTerm(
            func=mdp.tactile_force_maps,
            params={
                "left_sensor_name": "left_palm_tactile",
                "right_sensor_name": "right_palm_tactile",
                "grid_shape": TACTILE_GRID_SHAPE,
                "taxel_area_m2": R15_TAXEL_AREA_M2,
                "stress_scale": 1.0e-5,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    tactile: TactileCfg = TactileCfg()


@configclass
class RobotEnvCfg(CarryBoxRobotEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Two SDF CarryBox contacts across 512+ cloned environments exceed the
        # v2.3.0 default 64 MiB GPU collision stack by less than 1 MiB during
        # scene startup.  Give this tactile task 128 MiB so PhysX does not drop
        # any startup contacts; all other solver settings stay at the accepted
        # SUGAR control values.
        self.sim.physx.gpu_collision_stack_size = 2**27
        # A new tactile branch has no serialized refiner state-pool even though
        # its policy is warm-started from model_10000.pt.  During the official
        # pool warmup, initialize free environments from real SUGAR trajectory
        # frames so the branch immediately sees intended hand-object contact
        # instead of training its tactile encoder only on empty fields.
        self.commands.motion.init_with_ref = True
        # ``init_with_ref`` activates the same reference-to-pool curriculum
        # used by official SUGAR Tracker.  Refiner leaves these two fields at
        # ``None`` because its normal path starts directly from the pool; make
        # the schedule explicit here so the tactile branch remains valid after
        # the 24,000-step warmup boundary instead of comparing an integer to a
        # missing endpoint.  Empty pool entries still fall back to real SUGAR
        # trajectory frames in the official command implementation.
        self.commands.motion.pool_minref_steps = 5000 * 24
        self.commands.motion.pool_minref_ratio = 0.33


@configclass
class RobotLeftCameraEnvCfg(RobotEnvCfg):
    """Finite left-camera validation; camera images are not policy inputs."""

    scene: RobotLeftCameraSceneCfg = RobotLeftCameraSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.rerender_on_reset = True


@configclass
class RobotRightCameraEnvCfg(RobotEnvCfg):
    """Finite right-camera validation; camera images are not policy inputs."""

    scene: RobotRightCameraSceneCfg = RobotRightCameraSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.rerender_on_reset = True


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


class RobotRolloutPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.commands.motion.rollout_traj = True
        self.events.push_robot = None
        self.events.push_object = None


# Mount audit notes (kept beside the task so Hydra snapshots the exact
# research geometry): both sensor surfaces are the official R15 elastomer
# subprims, referenced without mesh conversion.  Their active-area centers are
# x=75 mm on palm-mounted normal standoffs.  The left surface faces inward along
# -Y; the right surface faces inward along +Y and is shifted within the plane to
# cover the measured CarryBox contact patch.  Each official camera tip is
# attached with the transform authored in the R15 USD.  The
# resulting articulation still exposes exactly the original 29 actuated G1
# joints; the four sensor links are fixed and add no policy action dimensions.
