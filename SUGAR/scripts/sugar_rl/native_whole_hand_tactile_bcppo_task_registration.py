#!/usr/bin/env python3
"""Process-local registration for the matched no-RGB tactile experiment."""

from __future__ import annotations

import gymnasium as gym


_CFG_MODULE = (
    "sugar_rl.tasks.locomanip.robots.g129dof.train_refiner."
    "carry_box_native_whole_hand_tactile_bcppo_env_cfg"
)
_AGENT_CFG = (
    "sugar_rl.tasks.locomanip.agents."
    "rsl_rl_native_whole_hand_tactile_bcppo_cfg:"
    "NativeWholeHandTactileBCPPORunnerCfg"
)
_BOUNDED_AGENT_CFG = (
    "sugar_rl.tasks.locomanip.agents."
    "rsl_rl_native_whole_hand_tactile_bcppo_cfg:"
    "BoundedNativeWholeHandTactileBCPPORunnerCfg"
)
_ACTION_RESIDUAL_AGENT_CFG = (
    "sugar_rl.tasks.locomanip.agents."
    "rsl_rl_native_whole_hand_tactile_bcppo_cfg:"
    "ActionResidualNativeWholeHandTactileBCPPORunnerCfg"
)
_TRACKER_COMMAND_AGENT_CFG = (
    "sugar_rl.tasks.locomanip.agents."
    "rsl_rl_native_whole_hand_tactile_bcppo_cfg:"
    "TrackerCommandNativeWholeHandTactileBCPPORunnerCfg"
)
_TRACKER_COMMAND_PREFLIGHT_AGENT_CFG = (
    "sugar_rl.tasks.locomanip.agents."
    "rsl_rl_native_whole_hand_tactile_bcppo_cfg:"
    "TrackerCommandNativeWholeHandTactilePreflightBCPPORunnerCfg"
)
TASKS = {
    "Sugar-G129dof-CarryBox-NativeWholeHand-TrackerCommand-Preflight-TacSL-BCPPO": (
        "TrackerCommandNativeTactileRobotEnvCfg",
        "TrackerCommandNativeTactileRobotPlayEnvCfg",
        _TRACKER_COMMAND_PREFLIGHT_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-NativeWholeHand-TrackerCommand-Preflight-Zero-BCPPO": (
        "TrackerCommandExactZeroRobotEnvCfg",
        "TrackerCommandExactZeroRobotPlayEnvCfg",
        _TRACKER_COMMAND_PREFLIGHT_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-NativeWholeHand-TrackerCommand-TacSL-BCPPO": (
        "TrackerCommandNativeTactileRobotEnvCfg",
        "TrackerCommandNativeTactileRobotPlayEnvCfg",
        _TRACKER_COMMAND_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-NativeWholeHand-TrackerCommand-Zero-BCPPO": (
        "TrackerCommandExactZeroRobotEnvCfg",
        "TrackerCommandExactZeroRobotPlayEnvCfg",
        _TRACKER_COMMAND_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-NativeWholeHand-ProprioTaskTacSL-BCPPO": (
        "NativeTactileRobotEnvCfg",
        "NativeTactileRobotPlayEnvCfg",
        _AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-NativeWholeHand-ProprioTaskZero-BCPPO": (
        "ExactZeroRobotEnvCfg",
        "ExactZeroRobotPlayEnvCfg",
        _AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-BoundedNativeWholeHand-ProprioTaskTacSL-BCPPO": (
        "NativeTactileRobotEnvCfg",
        "NativeTactileRobotPlayEnvCfg",
        _BOUNDED_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-BoundedNativeWholeHand-ProprioTaskZero-BCPPO": (
        "ExactZeroRobotEnvCfg",
        "ExactZeroRobotPlayEnvCfg",
        _BOUNDED_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-ActionResidualNativeWholeHand-ProprioTaskTacSL-BCPPO": (
        "NativeTactileRobotEnvCfg",
        "NativeTactileRobotPlayEnvCfg",
        _ACTION_RESIDUAL_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-ActionResidualNativeWholeHand-ProprioTaskZero-BCPPO": (
        "ExactZeroRobotEnvCfg",
        "ExactZeroRobotPlayEnvCfg",
        _ACTION_RESIDUAL_AGENT_CFG,
    ),
}


def register_native_whole_hand_tactile_bcppo_tasks() -> None:
    for task_id, (env_cfg, play_cfg, agent_cfg) in TASKS.items():
        if task_id in gym.registry:
            raise RuntimeError(f"duplicate process-local task registration: {task_id}")
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_CFG_MODULE}:{env_cfg}",
                "play_env_cfg_entry_point": f"{_CFG_MODULE}:{play_cfg}",
                "rsl_rl_cfg_entry_point": agent_cfg,
            },
        )
