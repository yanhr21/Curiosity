from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_pad_contacts_are_not_penalized_and_reward_sensors_watch_live_pads():
    carry_cfg = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_refiner/carry_box_refiner_env_cfg.py"
    )
    tactile_scene = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_refiner/carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py"
    )
    rewards = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/rewards.py"
    )
    assert "(?!left_anatomical_.*$)" in carry_cfg
    assert "(?!right_anatomical_.*$)" in carry_cfg
    assert "PATCH_OBJECT_CONTACT_SENSOR_NAMES_BY_HAND" in tactile_scene
    assert "_patch_object_contact_sensor_cfg(side, spec.name)" in tactile_scene
    assert "left_hand_forces = None" in tactile_scene
    assert "right_hand_forces = None" in tactile_scene
    assert "activate_contact_sensors=True" in tactile_scene
    assert ".amax(dim=(1, 2, 3))" in rewards
    assert "torch.stack(maxima, dim=1).amax(dim=1)" in rewards


def test_demo_following_goal_reward_does_not_penalize_tactile_surfaces():
    goal_cfg = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_refiner/carry_box_smp_icm_goal_env_cfg.py"
    )
    assert "(?!left_tacsl_r15_elastomer$)" in goal_cfg
    assert "(?!right_tacsl_r15_elastomer$)" in goal_cfg
    assert "(?!left_anatomical_.*$)" in goal_cfg
    assert "(?!right_anatomical_.*$)" in goal_cfg

    evaluator = source(
        "scripts/sugar/demo_following/evaluate_matched_fixed_teacher.py"
    )
    assert '"--teacher-only-zero-residual"' in evaluator
    assert '"teacher_only_residual_exact_zero"' in evaluator
    assert '"sugar_plan11_correct_teacher_zero_residual_gate_v1"' in evaluator
    assert '"maximum_robot_root_height_loss_m"' in evaluator
    assert '"physical_robot_fall"' in evaluator
    assert "NoTactileGoalRobotEnvCfg()" in evaluator
    assert "active_no_tactile_scene" in evaluator
    assert "apply_no_tactile_training_physics(base_env, proof)" in evaluator
    assert '"demo_control_has_no_tactile_scene"' in evaluator
    assert '"sugar_demo_no_tactile_scene_v1"' in evaluator


def test_demo_following_fall_termination_uses_physical_robot_root():
    goal_mdp = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/goal_carry_mdp.py"
    )
    goal_cfg = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_refiner/carry_box_smp_icm_goal_env_cfg.py"
    )
    fall = goal_mdp[goal_mdp.index("def unsafe_robot_fall(") :]
    assert "command.robot.data.root_pos_w[:, 2]" in fall
    assert "command.robot_anchor_pos_w" not in fall.split("def dropped_after_lift", 1)[0]
    assert '"maximum_root_height_loss_m": 0.35' in goal_cfg
    assert '"minimum_root_up_z"' not in goal_cfg


def test_next_demo_following_design_holds_teacher_fixed_and_changes_only_demo():
    namespace = runpy.run_path(
        str(ROOT / "scripts/sugar/demo_following/run_matched_state_predictor.py")
    )
    design = namespace["DESIGNS"]["same_teacher_reward_only"]
    correct = design["arms"]["correct"]
    unrelated = design["arms"]["unrelated"]
    assert correct["teacher"] == unrelated["teacher"]
    assert correct["demo_config"] != unrelated["demo_config"]
    assert correct["protocol_arm"] == "same_teacher_correct_reward"
    assert unrelated["protocol_arm"] == "same_teacher_unrelated_reward"

    renderer = source(
        "scripts/sugar/demo_following/render_demo_and_actual.py"
    )
    assert "output_frames_for_reference" in renderer
    assert "output_frames_for_actual" in renderer
    assert '"actual_fully_displayed"' in renderer
    assert '"both_reference_and_actual_trajectories_fully_displayed"' in renderer


def test_plan15_has_direct_hold_reward_and_matched_physics_friction():
    config = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_refiner/carry_box_online_patch_tactile_mass_env_cfg.py"
    )
    assert "post_handoff_box_lift = RewTerm" in config
    assert '"static_friction_range": (0.5, 0.5)' in config
    assert '"dynamic_friction_range": (0.5, 0.5)' in config
    assert "robot_physics_material = EventTerm" in config
    assert "Fixed3xOnlinePatchSlipMassRobotEnvCfg" in config
    assert "Fixed3xOnlinePatchSlipMassRobotPlayEnvCfg" in config
    assert "Fixed3xOnlinePatchSlipMassAuditPlayEnvCfg" in config


def test_tacsl_separates_normal_geometry_from_friction_shear():
    sensor = source(
        "IsaacLab/source/isaaclab_contrib/isaaclab_contrib/sensors/"
        "tacsl_sensor/visuotactile_sensor.py"
    )
    force_block = sensor[sensor.index("# Keep the two physical sources separate") :]
    assert "friction_force_tactile = math_utils.quat_apply_inverse" in force_block
    assert "tactile_normal_force[:] = torch.where" in force_block
    assert "tactile_friction_force_magnitude[:] = torch.where" in force_block
    assert "tactile_force_world = fc_world + ft_world" not in force_block


def test_anatomical_taxels_cover_each_patch_extent_instead_of_a_center_crop():
    sensor = source(
        "IsaacLab/source/isaaclab_contrib/isaaclab_contrib/sensors/"
        "tacsl_sensor/visuotactile_sensor.py"
    )
    config = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_refiner/carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg.py"
    )
    assert "if self.cfg.tactile_fill_mesh_extents:" in sensor
    assert "mesh_bounds[0, axis_i] + active_margin" in sensor
    assert "mesh_bounds[1, axis_i] - active_margin" in sensor
    assert "tactile_grid_effective_margin_m" in sensor
    assert "tactile_fill_mesh_extents=True" in config
    assert "tactile_margin=0.0001" in config
    assert "contact_offset_m - sdf_values[env_ids]" in sensor
    assert "CURIOSITY_ANATOMICAL_TACSL_CONTACT_OFFSET_M" in config
    assert "0.0003" in config


def test_formal_paths_match_motion45_and_reject_continued_outcome_scoring():
    trainer = source("SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py")
    evaluator = source("SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py")
    launcher = source("scripts/sugar/native_tactile/run_plan15_frozen_seed.sh")
    comparator = source(
        "SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py"
    )
    assert '"data/CarryBox/data_045"' in trainer
    assert "SUGAR/data/CarryBox/data_045" in launcher
    assert "--physical-outcome-view" not in launcher
    assert "CORRECTED_SCALE_FILE" in launcher
    assert "leakage_sweep_v1/patch_channel_scales.json" not in launcher
    assert 'summary.get("evaluation_view") != "strict_sugar_reference"' in comparator
    assert "exact_seed_sign_flip_pvalue" in comparator
    assert "apply_holm_familywise_correction" in comparator
    assert '/etc/vulkan/icd.d/nvidia_icd.json' in evaluator
    assert '"actor_receives_live_patch_observation": args.branch in {"P", "PS"}' in evaluator
    assert '"actor_receives_causal_slip_observation": args.branch == "PS"' in evaluator
    assert '"evaluator_reads_tacsl": tacsl_feature_read_calls > 0' in evaluator
    assert '"evaluator_tacsl_feature_read_calls": tacsl_feature_read_calls' in evaluator
    assert '"evaluator_slip_detector_update_calls": slip_detector_update_calls' in evaluator
    assert '"num_envs": int(args.num_envs)' in evaluator
    assert 'PLAN15_PIPELINE_LOCK_HELD' in evaluator
    assert 'requires a retained Slurm allocation' in evaluator
    assert 'refusing Plan-15 evaluation on login node' in evaluator
    assert 'if args.branch == "Z":' in evaluator
    assert '"exact_zero_control"' in evaluator
    assert '"evaluation_recomputed_patch_slip_labels_feed_actor": False' in evaluator
    assert "FIXED_OVERFIT_TASK" in evaluator
    assert '"--fixed-3x-overfit-gate"' in evaluator
    assert '"--audit-contact-forces"' in evaluator
    assert '"physx_box_normal_force_from_all_pads_w"' in evaluator
    assert '"physx_box_normal_force_per_pad_w"' in evaluator


def test_teacher_prefix_is_excluded_from_plan15_distillation_credit():
    algorithm = source(
        "SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py"
    )
    reduction = algorithm[
        algorithm.index("def _reduce_distill_loss") : algorithm.index(
            "def update", algorithm.index("def _reduce_distill_loss")
        )
    ]
    assert "self.training_mask_obs_group is None" in reduction
    assert "self.update_step < self.distill_mask_start_step" in reduction
    assert "active_weight.sum().clamp_min(" in reduction
    assert "per_sample_loss * active_weight" in reduction
    assert "del obs_batch" not in reduction
    assert 'loss_dict["distill_post_handoff_only"]' in algorithm
    assert "if actor_hold_active:" in algorithm
    assert '"actor_hold_active": float(actor_hold_active)' in algorithm
    assert "loss = critic_alpha * self.value_loss_coef * value_loss" in algorithm
    stage3 = algorithm.index("# Stage 3: Full PPO + Distill")
    hold = algorithm.index("if actor_hold_active:", stage3)
    assert algorithm.index("surrogate_loss * alpha", stage3) < hold
    hold_block = algorithm[hold : algorithm.index("# Symmetry loss", hold)]
    assert "surrogate_loss" not in hold_block
    assert "anchor_kl_per_sample" in algorithm
    assert "self.behavior_anchor_policy.act_inference" in algorithm
    assert "anchor_kl_per_sample * active_weight" in algorithm
    assert "self.behavior_anchor_coef * behavior_anchor_loss" in algorithm
    assert "parameter.grad = None" in algorithm
    assert "active_count.item() > 0" in algorithm
    assert "_reset_actor_optimizer_state" in algorithm
    assert "value.zero_()" in algorithm
    assert '"actor_optimizer_active_fraction"' in algorithm
    assert '"actor_optimizer_state_reset"' in algorithm
    assert "if tactile_only_active:" in algorithm
    assert "parameter.grad = None" in algorithm
    assert '"base_actor_optimizer_active_fraction"' in algorithm
    assert '"tactile_actor_optimizer_active_fraction"' in algorithm

    trainer = source("SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py")
    assert "SUGAR_PLAN15_MASKED_DISTILL_STABILITY_DIAGNOSTIC" in trainer
    assert 'Path(resume).name != "model_750.pt"' in trainer
    assert "SUGAR_PLAN15_ANCHORED_PPO_ENDPOINT" in trainer
    assert "diagnostic endpoint must lie in [1001, 1251]" in trainer
    assert "SUGAR_PLAN15_ANCHORED_PPO_STABILITY_DIAGNOSTIC" in trainer
    assert 'Path(resume).name != "model_1000.pt"' in trainer
    assert 'Path(anchor).name != "model_750.pt"' in trainer

    config = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/agents/"
        "rsl_rl_online_patch_mass_bcppo_cfg.py"
    )
    assert "distill_mask_start_step=751" in config
    assert "actor_hold_start_step=751" in config
    assert "actor_hold_end_step=1000" in config
    assert "SUGAR_PLAN15_BEHAVIOR_ANCHOR_CHECKPOINT" in config
    assert "behavior_anchor_start_step=1001" in config
    assert "stage3_tactile_only_actor=True" in config


def test_old_force_semantics_scale_files_are_rejected():
    trainer = source("SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py")
    evaluator = source("SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py")
    schema = "plan15_live_patch_channel_scales_v3_extent_offset_calibrated"
    assert schema in trainer
    assert schema in evaluator


def test_formal_paths_freeze_validated_force_calibration_and_audit_both_components():
    trainer = source("SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py")
    evaluator = source("SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py")
    gate = source("scripts/sugar/native_tactile/run_plan15_corrected_gate.sh")
    collector = source("scripts/sugar/native_tactile/preflight_online_patch_mass_jump.py")
    audit_scene = source(
        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_refiner/carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg.py"
    )
    offset = 'CONTACT_OFFSET_M"] = "0.0003"'
    normal = "7294.8755"
    tangential = 'TANGENTIAL_STIFFNESS"] = "9"'
    assert offset in trainer and offset in evaluator
    assert normal in trainer and normal in evaluator and normal in gate
    assert tangential in trainer and tangential in evaluator
    assert "TANGENTIAL_STIFFNESS=9" in gate
    assert 'data.friction_forces_w' in collector
    assert '"physx_box_normal_force_from_pads_w"' in collector
    assert '"physx_box_friction_force_from_pads_w"' in collector
    assert '"physx_box_friction_force_from_all_pads_w"' in collector
    assert '"physx_box_all_contact_normal_force_w"' in collector
    assert 'prim_path="{ENV_REGEX_NS}/Obj"' in audit_scene
    assert 'filter_prim_paths_expr=list(filter_exprs)' in audit_scene
    assert 'prim_path="{ENV_REGEX_NS}/Robot/.*"' not in audit_scene


def test_corrected_rerun_is_serial_and_fresh():
    gate = source("scripts/sugar/native_tactile/run_plan15_corrected_gate.sh")
    overfit = source("scripts/sugar/native_tactile/run_plan15_corrected_overfit.sh")
    review = source(
        "scripts/sugar/native_tactile/run_plan15_corrected_overfit_review.sh"
    )
    formal = source("scripts/sugar/native_tactile/run_plan15_corrected_formal_seed.sh")
    collector = source("scripts/sugar/native_tactile/preflight_online_patch_mass_jump.py")
    assert "SUGAR/data/CarryBox/data_045" in gate
    assert "for branch in Z P PS" in gate
    assert "formal training is not auto-started" in gate
    assert 'binary["precision"] < 0.8' in gate
    assert 'binary["recall"] < 0.8' in gate
    assert "Patch-PS-Overfit-BCPPO" in overfit
    assert '.plan15_training.lock' in overfit
    assert 'flock -n 9' in overfit
    assert '.plan15_training.lock' in formal
    assert 'flock -n 9' in formal
    assert '--num_envs 4' in overfit
    assert '--resume_checkpoint_path' in overfit
    assert 'resume checkpoint must be a direct child of OUTPUT_ROOT' in overfit
    assert "model_1499.pt" in overfit
    assert "--fixed-3x-overfit-gate" in review
    assert "--mass-factor 3.0" in review
    assert "--motion-id 0" in review
    assert "fresh corrected formal output already exists" in formal
    assert "model_2999.pt" in formal
    trainer = source("SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py")
    assert 'total_iteration_budget = "1251"' in trainer
    assert 'resume_name == "model_1250.pt"' in trainer
    assert 'resume_name == "model_2000.pt"' in trainer
    assert 'total_iteration_budget = "2501"' in trainer
    assert 'resume_name == "model_2500.pt"' in trainer
    assert 'total_iteration_budget = "3000"' in trainer
    assert 'expected_checkpoint="$output_root/model_1250.pt"' in formal
    assert 'expected_checkpoint="$output_root/model_2000.pt"' in formal
    assert 'expected_checkpoint="$output_root/model_2500.pt"' in formal
    assert 'expected_checkpoint="$output_root/model_2999.pt"' in formal
    assert "--resume_checkpoint_path" in formal
    assert (
        "formal resource-boundary resume requires model_1250.pt, model_2000.pt or model_2500.pt"
        in formal
    )
    assert "resume checkpoint must be a direct child of OUTPUT_ROOT" in formal
    assert 'default=ROOT / "SUGAR/data/CarryBox/data_045"' in collector
    assert "validate_sensor_read_contract" in source(
        "scripts/sugar/native_tactile/run_plan15_frozen_seed.sh"
    )
    frozen_seed = source(
        "scripts/sugar/native_tactile/run_plan15_frozen_seed.sh"
    )
    assert '.plan15_training.lock' in frozen_seed
    assert 'flock -n 9' in frozen_seed
    assert 'PLAN15_PIPELINE_LOCK_HELD=1' in frozen_seed
    evaluator = source("SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py")
    assert "os._exit(1)" in evaluator


def test_training_logs_the_actual_post_handoff_sample_budget():
    bcppo = source("SUGAR/source/sugar_rl/sugar_rl/utils/rsl_rl_bcppo.py")
    assert 'loss_dict["post_handoff_transitions"]' in bcppo
    assert 'loss_dict["post_handoff_fraction"]' in bcppo
