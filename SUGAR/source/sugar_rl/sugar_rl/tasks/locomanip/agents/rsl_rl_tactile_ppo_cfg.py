# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the SUGAR spatial tactile branch."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class TactileActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "TactileActorCritic"
    tactile_obs_group: str = "tactile"
    tactile_grid_shape: tuple[int, int] = (20, 25)
    tactile_num_hands: int = 2
    tactile_channels_per_hand: int = 3
    tactile_encoder_channels: list[int] = [32, 64, 64]
    tactile_embedding_dim: int = 128
    warm_start_tactile_gain: float = 0.01


@configclass
class TactileRefinerPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "sugar_g129dof_carrybox_tactile_refiner"
    obs_groups = {"policy": ["policy", "tactile"], "critic": ["critic", "tactile"]}
    policy = TactileActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
