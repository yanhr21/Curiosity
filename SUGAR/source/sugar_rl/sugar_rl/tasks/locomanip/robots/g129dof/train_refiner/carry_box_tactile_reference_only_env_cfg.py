# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Unregistered reference-only actor configs for the next tactile branch.

The policy receives the reference motion plan in place of measured box state.
The critic retains the official exact-state privileged observation group.  The
file is deliberately not registered until the active exact-state final suite
has completed, so it cannot change that suite's task or source provenance.
"""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from sugar_rl.tasks.locomanip.reference_only_observations import (
    reference_plan_obj_ang_vel_b,
    reference_plan_obj_lin_vel_b,
    reference_plan_obj_motion_ori_future,
    reference_plan_obj_motion_pos_future,
    reference_plan_obj_ori_b,
    reference_plan_obj_pos_b,
)

from .base_refiner_env_cfg import BaseObservationsCfg
from .carry_box_tactile_ablation_env_cfg import _set_tactile_mode
from .carry_box_tactile_refiner_env_cfg import ObservationsCfg as ExactTactileObservationsCfg
from .carry_box_tactile_refiner_env_cfg import RobotEnvCfg as ExactTactileRobotEnvCfg


@configclass
class ReferenceOnlyPolicyCfg(BaseObservationsCfg.PrivilegedCfg):
    """Official-width actor group with no measured box-state feedback."""

    ref_obj_pos_b_future = ObsTerm(
        func=reference_plan_obj_motion_pos_future,
        params={"command_name": "motion"},
    )
    ref_obj_ori_b_future = ObsTerm(
        func=reference_plan_obj_motion_ori_future,
        params={"command_name": "motion"},
    )
    obj_pos_b = ObsTerm(func=reference_plan_obj_pos_b, params={"command_name": "motion"})
    obj_ori_b = ObsTerm(func=reference_plan_obj_ori_b, params={"command_name": "motion"})
    obj_lin_vel_b = ObsTerm(
        func=reference_plan_obj_lin_vel_b,
        params={"command_name": "motion"},
    )
    obj_ang_vel_b = ObsTerm(
        func=reference_plan_obj_ang_vel_b,
        params={"command_name": "motion"},
    )


@configclass
class ReferenceOnlyObservationsCfg(ExactTactileObservationsCfg):
    policy: ReferenceOnlyPolicyCfg = ReferenceOnlyPolicyCfg()
    critic: BaseObservationsCfg.PrivilegedCfg = BaseObservationsCfg.PrivilegedCfg()


@configclass
class ReferenceOnlyRobotEnvCfg(ExactTactileRobotEnvCfg):
    observations: ReferenceOnlyObservationsCfg = ReferenceOnlyObservationsCfg()


@configclass
class ReferenceOnlyZeroRobotEnvCfg(ReferenceOnlyRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_tactile_mode(self, "zero")


@configclass
class ReferenceOnlyPressureRobotEnvCfg(ReferenceOnlyRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_tactile_mode(self, "pressure_only")


class ReferenceOnlyRobotPlayEnvCfg(ReferenceOnlyRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9

class ReferenceOnlyZeroRobotPlayEnvCfg(ReferenceOnlyZeroRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


class ReferenceOnlyPressureRobotPlayEnvCfg(ReferenceOnlyPressureRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
