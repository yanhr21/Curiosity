# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-task composition for coherent per-environment CarryBox dynamics.

This keeps the goal-based SUGAR/SMP/ICM observation, reward, termination, and
action contracts unchanged.  It swaps only the already audited official-asset
material wrapper, per-environment official TacSL friction adapter, and
stratified mass/friction/COM startup event into the goal task.

The legacy reference-contact pulse is deliberately absent: GoalCarryMotionCommand
does not advance a reference trajectory, so a reference-phase intervention
would not be a valid event source for this task.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from sugar_rl.tasks.locomanip.latent_contact_dynamics_events import (
    apply_stratified_latent_contact_dynamics,
)

from .carry_box_smp_icm_goal_env_cfg import (
    PureDiscoveryRobotEnvCfg,
    RobotEnvCfg,
)
from .carry_box_tactile_latent_contact_dynamics_env_cfg import (
    LatentContactEventCfg,
    LatentContactRobotSceneCfg,
)


@configclass
class GoalCoherentLatentEventCfg(LatentContactEventCfg):
    """Coherent startup tuple without the invalid reference-phase pulse."""

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
            "pulse_magnitude_range_mps": (0.0, 0.0),
        },
    )
    contact_phase_lateral_pulse = None


@configclass
class GoalCoherentLatentRobotEnvCfg(RobotEnvCfg):
    scene: LatentContactRobotSceneCfg = LatentContactRobotSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    events: GoalCoherentLatentEventCfg = GoalCoherentLatentEventCfg()


class GoalCoherentLatentRobotPlayEnvCfg(GoalCoherentLatentRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class GoalCoherentLatentPureDiscoveryRobotEnvCfg(PureDiscoveryRobotEnvCfg):
    scene: LatentContactRobotSceneCfg = LatentContactRobotSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    events: GoalCoherentLatentEventCfg = GoalCoherentLatentEventCfg()


class GoalCoherentLatentPureDiscoveryRobotPlayEnvCfg(
    GoalCoherentLatentPureDiscoveryRobotEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
