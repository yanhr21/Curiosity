from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BCPPOCfg(RslRlPpoAlgorithmCfg):
    class_name = "BCPPO"
    teacher_ckpt = None
    # Preserve official SUGAR behavior by default. Research ablations must set
    # a nonzero value through a separately named runner/task.
    stage3_distill_weight_floor = 0.0
    training_mask_obs_group = None
    distill_mask_start_step = 0
    actor_hold_start_step = -1
    actor_hold_end_step = -1
    behavior_anchor_checkpoint = None
    behavior_anchor_coef = 0.0
    behavior_anchor_start_step = 0
    stage3_tactile_only_actor = False
    bc_only_steps = 500
    critic_warmup_steps = 1000
    full_ppo_warmup_steps = 2000
    teacher_mean_only = False
    minimum_action_std = None
@configclass
class BCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30001
    save_interval = 1000
    experiment_name = ""  # same as task name
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"], "teacher": ["teacher"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
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
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
