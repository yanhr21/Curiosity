# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Official SUGAR PPO configuration for the unregistered reference-only branch."""

from isaaclab.utils import configclass

from .rsl_rl_tactile_ppo_cfg import TactileActorCriticCfg, TactileRefinerPPORunnerCfg


@configclass
class ReferenceOnlyTactileActorCriticCfg(TactileActorCriticCfg):
    class_name: str = "ReferenceOnlyTactileActorCritic"


@configclass
class ReferenceOnlyTactileRefinerPPORunnerCfg(TactileRefinerPPORunnerCfg):
    experiment_name = "sugar_g129dof_carrybox_reference_only_tactile_refiner"
    obs_groups = {"policy": ["policy", "tactile"], "critic": ["critic"]}
    policy = ReferenceOnlyTactileActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
