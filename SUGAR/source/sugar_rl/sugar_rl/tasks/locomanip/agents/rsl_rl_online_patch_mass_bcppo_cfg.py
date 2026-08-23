# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Matched BCPPO configuration for Plan-15 online patch tactile arms."""

from __future__ import annotations

import json
import math
import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .rsl_rl_bcppo_cfg import BCPPOCfg, BCPPORunnerCfg


def _patch_channel_scales() -> list[float]:
    """Read the live-sweep scales injected by the Plan-15 launcher."""

    raw = os.environ.get("SUGAR_ONLINE_PATCH_CHANNEL_SCALES")
    if raw is None:
        # Keep ordinary task discovery importable.  The actor rejects this
        # sentinel if somebody bypasses the Plan-15 launcher.
        return [float("nan")] * 9
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError("Plan-15 requires exactly nine patch channel scales")
    scales = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("Plan-15 patch channel scales must be positive and finite")
    return scales


@configclass
class OnlinePatchTactileActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "OnlinePatchTactileActorCritic"
    tactile_obs_group: str = "online_patch_tactile_history"
    tactile_grid_shape: tuple[int, int] = (20, 25)
    tactile_num_hands: int = 2
    tactile_channels_per_hand: int = 3
    tactile_encoder_channels: list[int] = [32, 64, 64]
    tactile_embedding_dim: int = 128
    patch_channel_scales: list[float] = _patch_channel_scales()
    warm_start_tactile_gain: float = 0.01


@configclass
class OnlinePatchMassBCPPORunnerCfg(BCPPORunnerCfg):
    """The unchanged serious SUGAR schedule shared by Z, P, and PS."""

    experiment_name = "sugar_carrybox_online_patch_mass_bcppo"
    # BCPPO gives the actor no task-reward PPO before update 1000. Complete
    # the 1000-update PPO ramp and retain 1000 steady full-PPO updates.
    max_iterations = 3000
    save_interval = 250
    obs_groups = {
        "policy": ["policy", "online_patch_tactile_history"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    policy = OnlinePatchTactileActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        warm_start_tactile_gain=0.01,
    )
    algorithm = BCPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Keep the released Refiner behavior anchored after PPO reaches full
        # authority.  A zero floor caused the live-handoff student to forget
        # the pickup/hold behavior between updates 2000 and 2999 even though
        # PPO remained active.  This repository-native BCPPO setting is shared
        # unchanged by Z, P, and PS.
        stage3_distill_weight_floor=0.25,
        training_mask_obs_group="training_handoff_mask",
        # The first 751 updates retain the released SUGAR full-trajectory
        # distillation that produced the first strictly valid checkpoint.
        # Thereafter, teacher KL is deployment-aligned to student-controlled
        # post-handoff transitions so the pickup prefix cannot erase it.
        distill_mask_start_step=751,
        actor_hold_start_step=751,
        actor_hold_end_step=1000,
        behavior_anchor_checkpoint=os.environ.get(
            "SUGAR_PLAN15_BEHAVIOR_ANCHOR_CHECKPOINT"
        ),
        behavior_anchor_coef=float(
            os.environ.get("SUGAR_PLAN15_BEHAVIOR_ANCHOR_COEF", "0.0")
        ),
        behavior_anchor_start_step=1001,
        stage3_tactile_only_actor=True,
    )


@configclass
class OnlinePatchMassPreflightBCPPORunnerCfg(OnlinePatchMassBCPPORunnerCfg):
    """One update long enough to reach grasp, mass jump, and post-jump signal."""

    experiment_name = "sugar_carrybox_online_patch_mass_preflight"
    num_steps_per_env = 360
    max_iterations = 1
    save_interval = 1


@configclass
class OnlinePatchMassOverfitBCPPORunnerCfg(OnlinePatchMassBCPPORunnerCfg):
    """Same serious BCPPO schedule, bounded after the first 500 PPO updates."""

    experiment_name = "sugar_carrybox_online_patch_mass_corrected_overfit"
    max_iterations = 1500
    save_interval = 250
