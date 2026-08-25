from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUGAR = ROOT / "SUGAR"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_recovery_task_uses_online_official_skill_prefix() -> None:
    source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert "_load_released_tracker_actor" in source
    assert "GeneratorWrapper.load" in source
    assert "alignment_action = self.kick_actor(policy)" in source
    assert "carry_action = self.carry_actor(carry_observation)" in source
    assert "state_teleport\": False" in source
    assert "offline_replay\": False" in source
    assert "ppo_prefix_transitions\": 0" in source
    assert "prefix_frame_callback" in source


def test_recovery_task_keeps_official_tracker_geometry() -> None:
    source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert "TRACKER_OBSERVATION_DIM = 510" in source
    assert "TRACKER_ACTION_DIM = 29" in source
    assert "hidden_dims != [512, 256, 128]" in source


def test_recovery_config_has_timeout_and_released_tracker_teacher_contract() -> None:
    source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_tracker/kick_box_carry9_recovery_v2_env_cfg.py"
    )
    assert "class TeacherCfg(TrackerCfg)" in source
    assert "self.enable_corruption = False" in source
    assert source.count("self.enable_corruption = False") >= 2
    assert "time_out = DoneTerm(func=mdp.time_out, time_out=True)" in source
    assert "physical_invalid = DoneTerm(" in source
    assert "synchronized_physical_invalid" in source
    assert "physical_invalid_penalty = RewTerm(" in source
    assert "weight=-10.0" in source
    assert "SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY" in source
    assert "self.episode_length_s = 6.0" in source


def test_warm_start_is_strict_and_does_not_resume_iteration() -> None:
    source = _read(SUGAR / "scripts/sugar_rl/train.py")
    assert "actor_critic_warm_start_checkpoint_path" in source
    assert "runner.alg.policy.load_state_dict(source_state, strict=True)" in source
    assert '"iteration_resumed": False' in source
    assert "SUGAR_ACTOR_CRITIC_WARM_START_EXPLORATION_STD" in source
    assert "source_exploration_std" in source
    assert "active_exploration_std" in source


def test_runner_does_not_randomize_synchronized_episode_clocks() -> None:
    source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/tasks/locomanip/agents/"
        "rsl_rl_cross_skill_recovery_cfg.py"
    )
    assert "init_at_random_ep_len = False" in source
    train_source = _read(SUGAR / "scripts/sugar_rl/train.py")
    assert 'getattr(agent_cfg, "init_at_random_ep_len", True)' in train_source
    assert "teacher_mean_only=True" in source
    assert "stage3_distill_weight_floor=0.25" in source
    assert "desired_kl=None" in source
    assert "minimum_action_std=0.005" in source


def test_synchronized_timeout_reinstalls_prefix_from_public_reset() -> None:
    source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert "super().reset()" in source
    assert "episode-boundary reset did not restore clock zero" in source
    assert "observations = self._install_prefix()" in source


def test_reproduction_pipeline_is_continuous_and_has_no_manual_gate() -> None:
    source = _read(
        ROOT / "scripts/sugar/demo_following/run_cross_skill_recovery_pipeline.sh"
    )
    assert "--max_iterations 65" in source
    assert "--teacher_ckpt" in source
    assert "model_pre_update.pt" in source
    assert "model_64.pt" in source
    assert 'OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"' in source
    assert 'test -s "$OUTPUT_ROOT/train/model_64.pt"' in source
    assert 'NUM_ENVS="${NUM_ENVS_OVERRIDE:-1024}"' in source
    assert "SUGAR_CROSS_SKILL_RECOVERY_REWARD_CLIP=10.0" in source
    assert "evaluate_cross_skill_recovery.py" in source
    assert "render_cross_skill_recovery_world.py" in source
    assert "read -" not in source
    assert "approval" not in source.lower()


def test_prefix_frontier_is_predeclared_and_uses_frozen_policy() -> None:
    source = _read(
        ROOT / "scripts/sugar/demo_following/run_cross_skill_prefix_frontier.sh"
    )
    assert "PREFIX_STEPS=(9 17 25 33 41 49 57 65 73 81 89 97)" in source
    assert "model_pre_update.pt" in source
    assert "--carry-prefix-steps" in source
    assert "train.py" not in source
    assert 'OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"' in source
    assert 'test -s "$result_dir/RESULT.json"' in source
    summary = _read(
        ROOT
        / "scripts/sugar/demo_following/summarize_cross_skill_prefix_frontier.py"
    )
    assert "ADMITTED_SAFE_KICK_SUCCESSES = 10" in summary
    assert "MINIMUM_UPRIGHT_ROOT_HEIGHT_M = 0.65" in summary
    assert '"frontier_found": selected is not None' in summary
    assert '"selected_recovery_boundary": recovery_boundary' in summary
    assert "largest physical-fall count" in summary


def test_recovery_claim_requires_a_physical_outcome_improvement() -> None:
    source = _read(
        ROOT / "scripts/sugar/demo_following/summarize_cross_skill_recovery_pair.py"
    )
    assert '"physical_recovery_improves"' in source
    assert 'delta["physical_fall_count"] < 0' in source
    assert 'delta.get("safe_kick_success_count"' in source


def test_conditional_tinymdm_reward_is_official_causal_and_online() -> None:
    source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_conditional_tinymdm_reward.py"
    )
    assert "from learning.tinymdm.tinymdm_model import TinyMDMModel" in source
    assert "from envs.amp_env import compute_disc_obs" in source
    assert "self.base_env.scene[\"robot\"].data" in source
    assert "self.base_env.scene[\"obj\"].data" in source
    assert "WINDOW_SIZE, FEATURE_DIM" in source
    assert "ESM_SDS_loss" in source
    assert 'reward_mode not in ("occupancy", "progress")' in source
    assert "previous_normalized - current_normalized" in source
    assert "progress_uses_matched_diffusion_noise" in source
    assert '"future_or_outcome_labels_used": False' in source
    for forbidden in ("physical_fall", "safe_kick", "future_frame", "trace.npz"):
        assert forbidden not in source


def test_train_entrypoint_requires_complete_conditional_reward_configuration() -> None:
    source = _read(SUGAR / "scripts/sugar_rl/train.py")
    assert 'SUGAR_CONDITIONAL_TINYMDM_REWARD") == "1"' in source
    assert "SUGAR_CONDITIONAL_TINYMDM_CONFIG" in source
    assert "SUGAR_CONDITIONAL_TINYMDM_CHECKPOINT" in source
    assert "SUGAR_CONDITIONAL_TINYMDM_CALIBRATION" in source
    assert "SUGAR_CONDITIONAL_TINYMDM_CLASS_ID" in source
    assert "conditional TinyMDM reward is missing" in source


def test_online_conditional_reward_preserves_task_reward_and_is_optional() -> None:
    source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert "self.conditional_tinymdm_reward = None" in source
    assert "observe_current_state()" in source
    assert "prepare_reward()" in source
    assert "self.conditional_tinymdm_task_reward_weight * rewards" in source
    assert "self.conditional_tinymdm_smp_reward_weight * conditional_reward" in source
    assert "conditional TinyMDM reward configuration is incomplete" in source


def test_conditional_smp_matched_pair_is_serial_and_has_no_human_gate() -> None:
    source = _read(ROOT / "scripts/sugar/smp/run_conditional_smp_recovery_matched_pair.sh")
    assert "correct_kick:1 wrong_carry:0" in source
    assert "TRAIN_SEED=171632" in source
    assert "EVAL_SEED=181632" in source
    assert "--max_iterations 65" in source
    assert "SUGAR_CONDITIONAL_TINYMDM_TASK_WEIGHT=0.5" in source
    assert "SUGAR_CONDITIONAL_TINYMDM_SMP_WEIGHT=0.5" in source
    assert "SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY=1" in source
    assert "RESUME_MATCHED_PAIR" in source
    assert "Reusing complete matched endpoint" in source
    assert "read -" not in source
    assert "approval" not in source.lower()


def test_conditional_smp_render_labels_camera_rollout_as_its_own_seed() -> None:
    source = _read(ROOT / "scripts/sugar/smp/render_conditional_smp_recovery_pair.sh")
    assert "EVAL_SEED" in source
    assert "Correct ${REWARD_MODE} reward: Kick condition" in source
    assert "Wrong ${REWARD_MODE} reward: Carry condition" in source
    assert "correct_kick_vs_wrong_carry_seed" in source
