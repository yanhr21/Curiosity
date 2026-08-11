# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Official SUGAR BCPPO with the existing serious spatial tactile encoder."""

from isaaclab.utils import configclass

from .rsl_rl_bcppo_cfg import BCPPOCfg, BCPPORunnerCfg
from .rsl_rl_reference_only_tactile_ppo_cfg import (
    ReferenceOnlyTactileActorCriticCfg,
)


@configclass
class NativeWholeHandTactileActorCriticCfg(ReferenceOnlyTactileActorCriticCfg):
    class_name: str = "ReferenceOnlyTactileActorCritic"
    tactile_obs_group: str = "native_whole_hand_tactile_history"
    tactile_grid_shape: tuple[int, int] = (20, 25)
    tactile_num_hands: int = 2
    # four frames * 27 physical patches * three native force/shear channels
    tactile_channels_per_hand: int = 324
    tactile_encoder_channels: list[int] = [32, 64, 64]
    tactile_embedding_dim: int = 128


@configclass
class NativeTactileTrainingBCPPOCfg(BCPPOCfg):
    class_name: str = "NativeTactileTrainingBCPPO"
    contact_balanced_distillation: bool = False


@configclass
class TrackerCommandNativeWholeHandTactileActorCriticCfg(
    NativeWholeHandTactileActorCriticCfg
):
    class_name: str = "TrackerCommandTactileActorCritic"


@configclass
class TrackerCommandNativeWholeHandTactileBCPPORunnerCfg(BCPPORunnerCfg):
    """Official 24-step BCPPO schedule for the deployable no-RGB actor."""

    experiment_name = "sugar_carrybox_tracker_command_native_tactile_bcppo"
    save_interval = 16
    obs_groups = {
        "policy": ["policy", "native_whole_hand_tactile_history"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    policy = TrackerCommandNativeWholeHandTactileActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # Released SUGAR Tracker and Refiner checkpoints do not use empirical
        # observation normalization. Preserve that official input scale.
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = NativeTactileTrainingBCPPOCfg(
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
    )


@configclass
class TrackerCommandNativeWholeHandTactilePreflightBCPPORunnerCfg(
    TrackerCommandNativeWholeHandTactileBCPPORunnerCfg
):
    """One update whose continuous rollout spans the first grasp contact."""

    experiment_name = "sugar_carrybox_tracker_command_native_tactile_preflight"
    # Contact begins around control step 243 on the frame-zero CarryBox route.
    num_steps_per_env = 288
    save_interval = 1


@configclass
class NativeWholeHandTactileBCPPORunnerCfg(BCPPORunnerCfg):
    experiment_name = "sugar_carrybox_native_whole_hand_tactile_bcppo"
    save_interval = 64
    obs_groups = {
        "policy": ["policy", "native_whole_hand_tactile_history"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    policy = NativeWholeHandTactileActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # The released SUGAR Refiner does not use empirical observation
        # normalization.  Keeping both paths disabled preserves its exact
        # zero-tactile mapping after the official warm start.
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = NativeTactileTrainingBCPPOCfg(
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
    )


@configclass
class BoundedNativeWholeHandTactileActorCriticCfg(
    NativeWholeHandTactileActorCriticCfg
):
    # Declared from the update-63 authority curve and active-frame
    # preactivation telemetry before bounded training.
    tactile_preactivation_cap: float = 0.15


@configclass
class BoundedNativeWholeHandTactileBCPPORunnerCfg(
    NativeWholeHandTactileBCPPORunnerCfg
):
    experiment_name = "sugar_carrybox_bounded_native_whole_hand_tactile_bcppo"
    policy = BoundedNativeWholeHandTactileActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        tactile_preactivation_cap=0.15,
    )
    algorithm = NativeTactileTrainingBCPPOCfg(
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
        contact_balanced_distillation=True,
    )


@configclass
class ActionResidualNativeWholeHandTactileActorCriticCfg(
    BoundedNativeWholeHandTactileActorCriticCfg
):
    # The first bounded run still reached a 0.573 same-state action change and
    # worsened contact-state teacher alignment.  Keep the hidden cap and add a
    # direct normalized-action bound declared before this route is trained.
    tactile_action_residual_cap: float = 0.1


@configclass
class ActionResidualNativeWholeHandTactileBCPPORunnerCfg(
    NativeWholeHandTactileBCPPORunnerCfg
):
    experiment_name = "sugar_carrybox_action_residual_native_whole_hand_tactile_bcppo"
    policy = ActionResidualNativeWholeHandTactileActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        tactile_preactivation_cap=0.15,
        tactile_action_residual_cap=0.1,
    )
    # The first contact-balanced run amplified a 4/48 supported rollout by
    # 12x and its distillation loss later rose to 10.05.  Return to the exact
    # official all-sample mean; zero tactile rows already give the adapter an
    # exact-zero gradient.
    algorithm = NativeTactileTrainingBCPPOCfg(
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
        contact_balanced_distillation=False,
    )
