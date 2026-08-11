#!/usr/bin/env python3
"""Guarded registration for the continuous anatomical-27 audit task."""

from __future__ import annotations

import os

import gymnasium as gym


TASK_ID = (
    "Sugar-G129dof-CarryBox-Official-Refiner-Anatomical27-"
    "WholeHand-TacSL-Audit"
)
ENABLE_ENV = "CURIOSITY_ENABLE_ANATOMICAL27_WHOLE_HAND_TACSL_AUDIT"


def register_official_refiner_anatomical_whole_hand_tacsl_audit_task() -> None:
    if os.environ.get(ENABLE_ENV) != "1":
        raise RuntimeError(f"Continuous audit task requires {ENABLE_ENV}=1")
    if TASK_ID in gym.registry:
        raise RuntimeError(
            f"Refusing duplicate process-local task registration: {TASK_ID}"
        )
    entry_point = (
        "sugar_rl.tasks.locomanip.robots.g129dof.train_refiner."
        "carry_box_official_refiner_anatomical_whole_hand_tacsl_"
        "audit_env_cfg:"
        "OfficialRefinerAnatomicalWholeHandTacSLAuditEnvCfg"
    )
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": entry_point,
            "play_env_cfg_entry_point": entry_point,
            "rsl_rl_cfg_entry_point": (
                "sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:"
                "BasePPORunnerCfg"
            ),
        },
    )
