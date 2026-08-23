# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Causally matched TacSL observation ablations for SUGAR CarryBox."""

from __future__ import annotations

from isaaclab.utils import configclass

from sugar_rl.tasks.locomanip.tactile_ablation_observations import matched_tactile_force_maps

from .carry_box_tactile_refiner_env_cfg import RobotEnvCfg as LiveRobotEnvCfg


def _set_tactile_mode(cfg: LiveRobotEnvCfg, mode: str) -> None:
    term = cfg.observations.tactile.force_maps
    term.func = matched_tactile_force_maps
    term.params = dict(term.params)
    term.params["mode"] = mode


@configclass
class ZeroTactileRobotEnvCfg(LiveRobotEnvCfg):
    """Matched non-tactile control: official sensors run, actor maps are zero."""

    def __post_init__(self):
        super().__post_init__()
        _set_tactile_mode(self, "zero")


class ZeroTactileRobotPlayEnvCfg(ZeroTactileRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9


@configclass
class PressureOnlyRobotEnvCfg(LiveRobotEnvCfg):
    """Pressure-only control with both signed shear channels masked."""

    def __post_init__(self):
        super().__post_init__()
        _set_tactile_mode(self, "pressure_only")


class PressureOnlyRobotPlayEnvCfg(PressureOnlyRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
