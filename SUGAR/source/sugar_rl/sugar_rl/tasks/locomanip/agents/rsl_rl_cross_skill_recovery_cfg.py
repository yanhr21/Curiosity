"""Official-topology PPO configuration for the Carry9-to-Kick recovery overfit."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .rsl_rl_bcppo_cfg import BCPPOCfg, BCPPORunnerCfg


@configclass
class CrossSkillRecoveryRunnerCfg(BCPPORunnerCfg):
    experiment_name = "sugar_kickbox_carry9_recovery_overfit"
    # RSL-RL iterations are zero-indexed: 65 iterations produce model_64.pt.
    # The attempted extension past this endpoint developed rare timeout-only
    # outliers and is not part of the admitted fixed diagnostic.
    max_iterations = 65
    save_interval = 64
    num_steps_per_env = 24
    # The wrapper generates every recovery start state by stepping all PhysX
    # environments through the same released-skill prefix.  Random episode
    # clocks would make only a subset auto-reset and bypass that contract.
    init_at_random_ep_len = False
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = BCPPOCfg(
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
        # Keep the short recovery overfit at its declared 1e-5 rate.  With a
        # zero-error warm start, adaptive KL would multiply the rate once per
        # minibatch before PPO receives its first nonzero actor gradient.
        desired_kl=None,
        max_grad_norm=1.0,
        stage3_distill_weight_floor=0.25,
        bc_only_steps=0,
        critic_warmup_steps=0,
        full_ppo_warmup_steps=1,
        teacher_mean_only=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.05
