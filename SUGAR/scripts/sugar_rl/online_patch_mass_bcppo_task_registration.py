#!/usr/bin/env python3
"""Process-local task registration for Plan-15 matched Z/P/PS training."""

from __future__ import annotations

import gymnasium as gym


_CFG_MODULE = (
    "sugar_rl.tasks.locomanip.robots.g129dof.train_refiner."
    "carry_box_online_patch_tactile_mass_env_cfg"
)
_AGENT_MODULE = (
    "sugar_rl.tasks.locomanip.agents."
    "rsl_rl_online_patch_mass_bcppo_cfg"
)
_FORMAL_AGENT_CFG = f"{_AGENT_MODULE}:OnlinePatchMassBCPPORunnerCfg"
_PREFLIGHT_AGENT_CFG = f"{_AGENT_MODULE}:OnlinePatchMassPreflightBCPPORunnerCfg"
_OVERFIT_AGENT_CFG = f"{_AGENT_MODULE}:OnlinePatchMassOverfitBCPPORunnerCfg"

TASKS = {
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-Z-Preflight-BCPPO": (
        "ExactZeroPatchMassRobotEnvCfg",
        "ExactZeroPatchMassRobotPlayEnvCfg",
        _PREFLIGHT_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-P-Preflight-BCPPO": (
        "OnlinePatchMassRobotEnvCfg",
        "OnlinePatchMassRobotPlayEnvCfg",
        _PREFLIGHT_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-Preflight-BCPPO": (
        "OnlinePatchSlipMassRobotEnvCfg",
        "OnlinePatchSlipMassRobotPlayEnvCfg",
        _PREFLIGHT_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-Overfit-BCPPO": (
        "Fixed3xOnlinePatchSlipMassRobotEnvCfg",
        "Fixed3xOnlinePatchSlipMassRobotPlayEnvCfg",
        _OVERFIT_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-Overfit-Audit-BCPPO": (
        "Fixed3xOnlinePatchSlipMassRobotEnvCfg",
        "Fixed3xOnlinePatchSlipMassAuditPlayEnvCfg",
        _OVERFIT_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-Z-BCPPO": (
        "ExactZeroPatchMassRobotEnvCfg",
        "ExactZeroPatchMassRobotPlayEnvCfg",
        _FORMAL_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-P-BCPPO": (
        "OnlinePatchMassRobotEnvCfg",
        "OnlinePatchMassRobotPlayEnvCfg",
        _FORMAL_AGENT_CFG,
    ),
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-BCPPO": (
        "OnlinePatchSlipMassRobotEnvCfg",
        "OnlinePatchSlipMassRobotPlayEnvCfg",
        _FORMAL_AGENT_CFG,
    ),
}


def register_online_patch_mass_bcppo_tasks() -> None:
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
