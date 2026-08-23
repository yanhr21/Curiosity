# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-H SUGAR + official-SMP-style policy optimizer configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class OfficialSMPTactileActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "OfficialSMPTactileActorCritic"
    tactile_obs_group: str = "tactile_history"
    tactile_grid_shape: tuple[int, int] = (20, 25)
    tactile_num_hands: int = 2
    tactile_channels_per_hand: int = 12
    tactile_encoder_channels: list[int] = [32, 64, 64]
    tactile_embedding_dim: int = 128
    warm_start_tactile_gain: float = 0.01


@configclass
class OfficialSMPPolicyOptimizerCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "OfficialSMPPolicyOptimizerAdapter"
    critic_num_learning_epochs: int = 2
    critic_num_mini_batches: int = 16
    normalized_advantage_clip: float = 4.0
    action_bound_weight: float = 10.0


@configclass
class SMPICMPureDiscoveryRunnerCfg(RslRlOnPolicyRunnerCfg):
    """No-result discovery phase; ICM/SMP mixing is done by the Stage-H runner."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 32
    max_iterations = 1
    save_interval = 1
    experiment_name = "sugar_smp_icm_pure_discovery"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy", "tactile_history"],
        "critic": ["critic", "tactile_history"],
    }
    policy = OfficialSMPTactileActorCriticCfg(
        init_noise_std=0.05,
        noise_std_type="scalar",
        actor_hidden_dims=[1024, 1024],
        critic_hidden_dims=[1024, 1024],
        activation="relu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = OfficialSMPPolicyOptimizerCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=0.0,
        normalize_advantage_per_mini_batch=False,
    )


@configclass
class SugarNativeTactileActorCriticCfg(RslRlPpoActorCriticCfg):
    """Official SUGAR BasePPO network with the admitted spatial R15 branch."""

    class_name: str = "SugarNativeTactileActorCritic"
    tactile_obs_group: str = "tactile_history"
    tactile_grid_shape: tuple[int, int] = (20, 25)
    tactile_num_hands: int = 2
    tactile_channels_per_hand: int = 12
    tactile_encoder_channels: list[int] = [32, 64, 64]
    tactile_embedding_dim: int = 128
    warm_start_tactile_gain: float = 0.01


@configclass
class SugarNativeCuriosityPPOCfg(RslRlPpoAlgorithmCfg):
    """Unchanged upstream RSL PPO with read-only audit instrumentation."""

    class_name: str = "SugarNativeCuriosityPPO"


@configclass
class SMPICMSugarNativeRunnerCfg(RslRlOnPolicyRunnerCfg):
    """HN0/HN1 policy optimizer on native, unbounded SUGAR policy units."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1
    save_interval = 1
    experiment_name = "sugar_smp_icm_sugar_native_pure_discovery"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy", "tactile_history"],
        "critic": ["critic", "tactile_history"],
    }
    policy = SugarNativeTactileActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = SugarNativeCuriosityPPOCfg(
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
        normalize_advantage_per_mini_batch=False,
    )


@configclass
class SugarNativeTactileFloorLrPPOCfg(RslRlPpoAlgorithmCfg):
    """Named upstream RSL PPO study initialized at its adaptive LR floor."""

    class_name: str = "SugarNativeTactileFloorLrPPO"


@configclass
class SMPICMSugarNativeFloorLrRunnerCfg(RslRlOnPolicyRunnerCfg):
    """HLR0/HLR1: native actions and unchanged PPO, starting at LR 1e-5."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1
    save_interval = 1
    experiment_name = "sugar_smp_icm_sugar_native_tactile_floor_lr"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy", "tactile_history"],
        "critic": ["critic", "tactile_history"],
    }
    policy = SugarNativeTactileActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = SugarNativeTactileFloorLrPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


@configclass
class SugarNativeZeroPreservingTactileActorCriticCfg(
    SugarNativeTactileActorCriticCfg
):
    class_name: str = "SugarNativeZeroPreservingTactileActorCritic"


@configclass
class SugarNativeZeroPreservingTactileFloorLrPPOCfg(
    RslRlPpoAlgorithmCfg
):
    class_name: str = "SugarNativeZeroPreservingTactileFloorLrPPO"


@configclass
class SMPICMSugarNativeZeroPreservingFloorLrRunnerCfg(
    RslRlOnPolicyRunnerCfg
):
    """ZLR0/ZLR1: causal zero-taxel invariant plus the admitted floor LR."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1
    save_interval = 1
    experiment_name = (
        "sugar_smp_icm_sugar_native_zero_preserving_tactile_floor_lr"
    )
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy", "tactile_history"],
        "critic": ["critic", "tactile_history"],
    }
    policy = SugarNativeZeroPreservingTactileActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = SugarNativeZeroPreservingTactileFloorLrPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


@configclass
class SugarNativeZeroPreservingTactileFixedLowLrPPOCfg(
    RslRlPpoAlgorithmCfg
):
    class_name: str = "SugarNativeZeroPreservingTactileFixedLowLrPPO"


@configclass
class SMPICMSugarNativeZeroPreservingFixedLowLrRunnerCfg(
    RslRlOnPolicyRunnerCfg
):
    """ZF0/ZF1: zero-preserving tactile policy and fixed upstream LR."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1
    save_interval = 1
    experiment_name = (
        "sugar_smp_icm_sugar_native_zero_preserving_tactile_fixed_low_lr"
    )
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy", "tactile_history"],
        "critic": ["critic", "tactile_history"],
    }
    policy = SugarNativeZeroPreservingTactileActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = SugarNativeZeroPreservingTactileFixedLowLrPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )
