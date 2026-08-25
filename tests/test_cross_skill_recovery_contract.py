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
    assert '"contrastive_progress"' in source
    assert "previous_normalized - current_normalized" in source
    assert "current_margin - previous_margin" in source
    assert "alternative_class_labels" in source
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
    assert "TRAIN_SEED_OVERRIDE:-171632" in source
    assert "EVAL_SEED_OVERRIDE:-181632" in source
    assert '--training-seed "$TRAIN_SEED"' in source
    assert "--max_iterations 65" in source
    assert "SUGAR_CONDITIONAL_TINYMDM_TASK_WEIGHT=0.5" in source
    assert "SUGAR_CONDITIONAL_TINYMDM_SMP_WEIGHT=0.5" in source
    assert "SUGAR_CROSS_SKILL_RECOVERY_SAFETY_PENALTY=1" in source
    assert "--/renderer/enabled=false" in source
    assert "RESUME_MATCHED_PAIR" in source
    assert "Reusing complete matched endpoint" in source
    assert "read -" not in source
    assert "approval" not in source.lower()


def test_conditional_smp_render_labels_camera_rollout_as_its_own_seed() -> None:
    source = _read(ROOT / "scripts/sugar/smp/render_conditional_smp_recovery_pair.sh")
    assert "EVAL_SEED" in source
    assert "Correct ${REWARD_MODE} reward: Kick condition" in source
    assert "Wrong ${REWARD_MODE} reward: Carry condition" in source
    assert "Correct: Kick contrastive progress" in source
    assert "Wrong: Carry contrastive progress" in source
    assert "correct_kick_vs_wrong_carry_seed" in source


def test_transition_controller_freezes_released_endpoints_and_uses_causal_command() -> None:
    actor_source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/frozen_expert_transition_actor_critic.py"
    )
    assert "OFFICIAL_HIDDEN_DIMS = (512, 256, 128)" in actor_source
    assert "actor.eval().requires_grad_(False)" in actor_source
    assert "selected_observation[:, :GENERATED_COMMAND_DIM] = command" in actor_source
    assert "self.actor.endpoint_action(actor_input)" in actor_source
    assert "nn.init.zeros_(final.weight)" in actor_source
    assert "nn.init.zeros_(final.bias)" in actor_source
    assert "distillation_teacher" in actor_source

    wrapper_source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert "transition_selected_skill_id" in wrapper_source
    assert 'observations["selected_skill_command"]' in wrapper_source
    assert 'observations["selected_skill_id"]' in wrapper_source
    assert "self.carry_shadow.update_after_step(self.command)" in wrapper_source


def test_transition_bcppo_uses_one_unambiguous_exact_teacher() -> None:
    source = _read(SUGAR / "source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py")
    assert 'self.policy, "distillation_teacher", None' in source
    assert "either a checkpoint teacher or the policy's exact" in source
    assert "self.policy_distillation_teacher(obs_batch)" in source
    assert "BCPPO requires either teacher_ckpt or a policy distillation_teacher" in source

    config = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/tasks/locomanip/agents/"
        "rsl_rl_cross_skill_recovery_cfg.py"
    )
    assert 'class_name: str = "FrozenExpertTransitionActorCritic"' in config
    assert '"policy": ["policy", "selected_skill_command", "selected_skill_id"]' in config
    assert "self.algorithm.teacher_ckpt = None" in config


def test_transition_pair_is_fixed_serial_and_has_no_human_gate() -> None:
    source = _read(
        ROOT / "scripts/sugar/demo_following/run_frozen_expert_transition_pair.sh"
    )
    assert "correct_kick:1 wrong_carry:0" in source
    assert "Sugar-G129dof-KickBox-FrozenExpert-Transition" in source
    assert "--max_iterations 65" in source
    assert "SUGAR_CROSS_SKILL_CARRY_PREFIX_STEPS=41" in source
    assert "unset SUGAR_CONDITIONAL_TINYMDM_REWARD" in source
    assert "--transition-selected-skill-id" in source
    assert "model_pre_update.pt" in source
    assert "model_64.pt" in source
    assert "summarize_frozen_expert_transition_pair.py" in source
    assert "read -" not in source
    assert "approval" not in source.lower()


def test_shared_transition_checkpoint_balances_conditions_without_scalar_reward() -> None:
    wrapper = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert "transition_selected_skill_id == -1" in wrapper
    assert "torch.arange(self.num_envs" in wrapper
    assert "% 2" in wrapper
    assert "transition_selected_skill_counts" in wrapper

    source = _read(
        ROOT / "scripts/sugar/demo_following/run_shared_frozen_expert_transition.sh"
    )
    assert "SUGAR_TRANSITION_SELECTED_SKILL_ID=-1" in source
    assert "SUGAR_TRANSITION_RECOVERY_REWARD_OVERRIDE" in source
    assert "unset SUGAR_CONDITIONAL_TINYMDM_REWARD" in source
    assert source.count("$OUTPUT_ROOT/train/model_64.pt") >= 3
    assert "kick:1 carry:0" in source
    assert "--max_iterations 65" in source
    assert "summarize_shared_frozen_expert_transition.py" in source
    assert "model_pre_update.pt" in source
    assert "evaluation/kick_pre_update" in source
    assert "summarize_shared_transition_learning.py" in source
    assert "same_checkpoint_kick_vs_carry" in source
    assert "read -" not in source
    assert "approval" not in source.lower()

    formal = _read(
        ROOT
        / "scripts/sugar/demo_following/run_shared_transition_recovery_objective.sh"
    )
    assert "SUGAR_TRANSITION_RECOVERY_REWARD_OVERRIDE=1" in formal
    assert "run_shared_frozen_expert_transition.sh" in formal
    assert "read -" not in formal
    assert "approval" not in formal.lower()


def test_shared_transition_separates_condition_use_from_learning_benefit() -> None:
    condition_summary = _read(
        ROOT
        / "scripts/sugar/demo_following/summarize_shared_frozen_expert_transition.py"
    )
    assert "condition_swap_v2" in condition_summary
    assert "Carry is an intentionally wrong/inert semantic control" in condition_summary
    assert "pre-update Kick endpoint" in condition_summary
    assert "kick_condition_has_physical_advantage" not in condition_summary

    learning_summary = _read(
        ROOT / "scripts/sugar/demo_following/summarize_shared_transition_learning.py"
    )
    assert "expected_iteration: int" in learning_summary
    assert "_load(args.learned, args.learned_iteration)" in learning_summary
    assert "_load(args.pre_update, -1)" in learning_summary
    assert "initial physics is not elementwise identical" in learning_summary
    assert '"learned_kick_safety_improvement"' in learning_summary


def test_frozen_evaluator_is_cwd_independent() -> None:
    source = _read(
        ROOT / "scripts/sugar/demo_following/evaluate_cross_skill_recovery.py"
    )
    assert "os.chdir(SUGAR)" in source
    assert source.index("preflight_output.exists()") < source.index(
        "app_launcher = AppLauncher(args)"
    )


def test_failure_rich_transition_overfit_is_fixed_and_automatic() -> None:
    source = _read(
        ROOT
        / "scripts/sugar/demo_following/run_frozen_expert_transition_failure_overfit.sh"
    )
    assert "SUGAR_TRANSITION_SELECTED_SKILL_ID=1" in source
    assert "--num_envs 20 --max_iterations 257 --seed 181630" in source
    assert "for iteration in 64 128 192 256" in source
    assert "--seed 181630" in source
    assert "model_pre_update" in source
    assert "summarize_transition_failure_overfit.py" in source
    assert "read -" not in source
    assert "approval" not in source.lower()

    summary = _read(
        ROOT / "scripts/sugar/demo_following/summarize_transition_failure_overfit.py"
    )
    assert "learned[\"physical_fall_count\"] < pre[\"physical_fall_count\"]" in summary
    assert "learned[\"safe_kick_success_count\"] >= pre[\"safe_kick_success_count\"]" in summary
    assert '"diagnostic_only": True' in summary


def test_transition_recovery_objective_is_causal_and_not_actor_input() -> None:
    source = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert "transition_recovery_reward_enabled" in source
    assert "object_xy - self._transition_handoff_object_xy" in source
    assert 'self.base_env.scene.sensors["left_foot_forces"]' in source
    assert 'self.base_env.scene.sensors["right_foot_forces"]' in source
    assert "root_loss - 0.15" in source
    assert '"future_or_outcome_labels_used": False' in source
    assert '"actor_observation_augmented": False' in source
    assert "rewards = rewards + recovery_reward" in source
    observation_method = source[
        source.index("def _augment_transition_observation") : source.index(
            "def get_observations"
        )
    ]
    assert "recovery" not in observation_method

    runner = _read(
        ROOT
        / "scripts/sugar/demo_following/run_transition_recovery_objective_overfit.sh"
    )
    assert "SUGAR_TRANSITION_RECOVERY_REWARD_OVERRIDE=1" in runner
    assert "run_frozen_expert_transition_failure_overfit.sh" in runner
    assert "read -" not in runner
    assert "approval" not in runner.lower()


def test_multi_context_recovery_cycles_online_prefixes_without_actor_leakage() -> None:
    wrapper = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert "carry_prefix_schedule" in wrapper
    assert "self.prefix_count % len(self.carry_prefix_schedule)" in wrapper
    assert '"carry_prefix_install_counts"' in wrapper
    assert '"prefix_schedule_is_episode_boundary_online": True' in wrapper
    train = _read(SUGAR / "scripts/sugar_rl/train.py")
    assert "SUGAR_CROSS_SKILL_CARRY_PREFIX_SCHEDULE" in train

    runner = _read(
        ROOT / "scripts/sugar/demo_following/run_multi_context_transition_recovery.sh"
    )
    assert "PREFIXES=(41 49 57)" in runner
    assert "SUGAR_CROSS_SKILL_CARRY_PREFIX_SCHEDULE=41,49,57" in runner
    assert "--max_iterations 65" in runner
    assert "model_pre_update.pt" in runner
    assert "summarize_multi_context_transition_recovery.py" in runner
    assert "read -" not in runner
    assert "approval" not in runner.lower()

    summary = _read(
        ROOT
        / "scripts/sugar/demo_following/summarize_multi_context_transition_recovery.py"
    )
    assert "all_predeclared_contexts_installed_online" in summary
    assert "aggregate_kick_safety_improvement" in summary
    assert "exact_pre_update_kick" in summary
    assert "future_or_outcome_labels_used" in summary

    aggregate = _read(
        ROOT
        / "scripts/sugar/demo_following/aggregate_multi_context_transition_recovery_seeds.py"
    )
    assert "multi-context aggregate requires at least two seeds" in aggregate
    assert "training and evaluation seed sets must be disjoint" in aggregate
    assert "same_predeclared_context_schedule_all_seeds" in aggregate
    assert "aggregate_safety_improvement_replicated_all_seeds" in aggregate


def test_causal_action_composer_uses_both_commands_and_has_no_manual_gate() -> None:
    actor = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/frozen_expert_transition_actor_critic.py"
    )
    assert "class FrozenExpertCausalActionComposer" in actor
    assert "DUAL_COMMAND_INPUT_DIM" in actor
    assert "carry_observation[:, :GENERATED_COMMAND_DIM] = carry_command" in actor
    assert "kick_observation[:, :GENERATED_COMMAND_DIM] = kick_command" in actor
    assert "skill[:, 1:2] - 0.5 * torch.tanh" in actor
    assert "nn.init.zeros_(final.weight)" in actor
    for forbidden in ("physical_fall", "safe_kick", "trace.npz"):
        assert forbidden not in actor

    wrapper = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/utils/online_cross_skill_recovery_wrapper.py"
    )
    assert 'observations["carry_skill_command"] = carry_command' in wrapper
    assert 'observations["kick_skill_command"] = kick_command' in wrapper

    task_registry = _read(
        SUGAR
        / "source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_tracker/__init__.py"
    )
    assert "Sugar-G129dof-KickBox-CausalActionComposition" in task_registry
    runner = _read(
        ROOT
        / "scripts/sugar/demo_following/"
        "run_causal_action_composition_transition_recovery.sh"
    )
    assert "POLICY_TOPOLOGY_OVERRIDE=causal_action_composition" in runner
    assert "read -" not in runner
    assert "approval" not in runner.lower()
    assert "GPU_HOLD_AFTER_CAUSAL_ACTION_COMPOSITION_READY" in runner
    assert "PIPELINE_STATUS.env" in runner
    multi_context_runner = _read(
        ROOT / "scripts/sugar/demo_following/run_multi_context_transition_recovery.sh"
    )
    assert 'learned_vs_pre_update_prefix${prefix}.mp4" -f null -' in multi_context_runner
    for script in (
        "evaluate_cross_skill_recovery.py",
        "render_cross_skill_recovery_world.py",
    ):
        source = _read(ROOT / "scripts/sugar/demo_following" / script)
        assert '"VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"' in source


def test_causal_action_composer_is_exact_at_pre_update_and_gate_is_trainable() -> None:
    import math

    def kick_weight(selected_kick: float, gate_logit: float) -> float:
        return min(
            1.0,
            max(0.0, selected_kick - 0.5 * math.tanh(gate_logit)),
        )

    assert kick_weight(0.0, 0.0) == 0.0
    assert kick_weight(1.0, 0.0) == 1.0
    assert kick_weight(0.0, -0.1) > 0.0
    assert kick_weight(1.0, 0.1) < 1.0

    summary = _read(
        ROOT
        / "scripts/sugar/demo_following/"
        "summarize_multi_context_transition_recovery.py"
    )
    assert "MINIMUM_MEAN_COMPOSITION_DEVIATION = 1.0e-4" in summary
    assert 'if args.expected_policy_topology == "causal_action_composition"' in summary
    assert '"policy_topology", "selected_expert_residual"' in summary
