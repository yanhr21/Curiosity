from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/sugar/demo_following/train_official_tracker_router.py"
EVALUATOR = ROOT / "scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py"
RENDERER = ROOT / "scripts/sugar/demo_following/render_official_tracker_router.py"
JOINT_RUNNER = ROOT / "scripts/sugar/demo_following/run_joint_generator_tracker_router_eval.sh"
FACTORIAL_RUNNER = ROOT / "scripts/sugar/demo_following/run_carry_skill_asset_motion_factorial.sh"
FACTORIAL_SUMMARY = ROOT / "scripts/sugar/demo_following/summarize_carry_skill_asset_motion_factorial.py"
GEOMETRY_MASS_RUNNER = ROOT / "scripts/sugar/demo_following/run_carry_skill_geometry_mass_factorial.sh"
GEOMETRY_MASS_SUMMARY = ROOT / "scripts/sugar/demo_following/summarize_carry_skill_geometry_mass_factorial.py"
CONTEXT_GOAL_RUNNER = ROOT / "scripts/sugar/demo_following/run_carry_skill_context_goal_factorial.sh"
CONTEXT_GOAL_SUMMARY = ROOT / "scripts/sugar/demo_following/summarize_carry_skill_context_goal_factorial.py"
SAFE_RUNNER = ROOT / "scripts/sugar/demo_following/run_official_skill_safe_fallback_pair.sh"
SAFE_SUMMARY = ROOT / "scripts/sugar/demo_following/summarize_official_skill_safe_fallback.py"
COMPATIBILITY_RUNNER = ROOT / "scripts/sugar/demo_following/run_official_generator_compatibility_audit.sh"
COMPATIBILITY_SUMMARY = ROOT / "scripts/sugar/demo_following/summarize_official_generator_compatibility.py"


def _function_source(path: Path, class_name: str, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child) or ""
    raise AssertionError(f"missing {class_name}.{function_name}")


def test_router_deployment_reads_only_current_state_and_causal_condition() -> None:
    source = _function_source(
        TRAINER, "DemoConditionedOfficialTrackerRouter", "act_inference"
    )
    assert 'observation["policy"]' in source
    assert 'observation["demo_conditioning"]' in source
    for forbidden in ("teacher", "future", "target", "task_id", "motion_id"):
        assert forbidden not in source


def test_released_tracker_experts_are_loaded_exactly_and_frozen() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    assert 'source.items()' in source
    assert 'name.startswith("actor.")' in source
    assert 'load_state_dict(state, strict=True)' in source
    assert 'requires_grad_(False)' in source
    assert '"official_experts_frozen_and_exact": expert_delta == 0.0' in source
    assert 'optimizer = torch.optim.Adam(policy.router.parameters()' in source


def test_frozen_evaluator_rejects_falls_and_raw_action_explosion() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "RELEASED_TRACKER_RAW_ACTION_LIMIT = 25.0" in source
    assert 'int(task_success_count) >= 10' in source
    assert 'int(aggregate["physical_fall_count"]) <= 2' in source
    assert '"raw_student_actions_within_released_tracker_envelope"' in source
    assert '"router_selects_requested_expert"' in source
    assert '"shared_checkpoint": str(checkpoint_path)' in source


def test_joint_route_switches_official_generator_only_after_common_prefix() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    prefix = source.index("prefix_action = official_policy(obs)")
    post_prefix = source.index('post_prefix = {')
    route = source.index("selected_generator = GeneratorWrapper.load")
    assert prefix < post_prefix < route
    assert 'routed_generator_skill = SELECTED_SKILL[args.selected_demo_option]' in source
    assert 'command._fill_generator_obs_buffer(all_env_ids)' in source
    assert 'command._call_generator(all_env_ids)' in source
    assert '"selected_demo_routes_complete_generator_tracker_pair"' in source
    assert '"selected_skill_behavioral_gate"' in source


def test_safe_fallback_is_online_causal_and_uses_only_released_pairs() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    runner = SAFE_RUNNER.read_text(encoding="utf-8")
    summary = SAFE_SUMMARY.read_text(encoding="utf-8")
    shadow = _function_source(EVALUATOR, "_CausalShadowGenerator", "update_after_step")
    assert "command._get_generator_obs()" in shadow
    assert "future" not in shadow
    assert "student_action" in evaluator
    assert "teacher_action" in evaluator
    assert "RELEASED_TRACKER_RAW_ACTION_LIMIT" in evaluator
    assert "torch.where(safe.unsqueeze(-1), student_action, teacher_action)" in evaluator
    for cell in (
        "carry_kick_direct",
        "carry_kick_safe",
        "kick_carry_direct",
        "kick_carry_safe",
    ):
        assert cell in runner
        assert cell in summary
    assert '"kick_carry_safe_retains_domain_kick_behavior"' in summary


def test_generator_compatibility_audit_uses_released_normalizer_only() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    summary = COMPATIBILITY_SUMMARY.read_text(encoding="utf-8")
    assert "self.generator.policy.normalizer.normalize(raw)" in evaluator
    assert "selected_generator_outside_train_range_fraction" in evaluator
    assert '"early_window_frames": 100' in summary
    assert '"policy_gate_supported": natural_range_separates' in summary


def test_asset_motion_factorial_changes_only_declared_physical_asset() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    runner = FACTORIAL_RUNNER.read_text(encoding="utf-8")
    summary = FACTORIAL_SUMMARY.read_text(encoding="utf-8")
    assert 'choices=("domain", "small", "big")' in evaluator
    assert 'env_cfg.scene.obj = object_cfg.replace' in evaluator
    assert '"object_mass_kg"' in evaluator
    for cell in (
        "carry_motion_small_asset",
        "carry_motion_big_asset",
        "kick_motion_small_asset",
        "kick_motion_big_asset",
    ):
        assert cell in runner
        assert cell in summary
    assert '"prefix_action"' in summary
    assert '"matches_nominal_1p5_ratio"' in summary


def test_geometry_mass_factorial_scales_mass_and_inertia_together() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    runner = GEOMETRY_MASS_RUNNER.read_text(encoding="utf-8")
    summary = GEOMETRY_MASS_SUMMARY.read_text(encoding="utf-8")
    assert 'obj_view.set_masses(original_object_masses * mass_inertia_ratio' in evaluator
    assert 'obj_view.set_inertias(original_object_inertias * mass_inertia_ratio' in evaluator
    for cell in (
        "small_geometry_small_mass",
        "small_geometry_big_mass",
        "big_geometry_small_mass",
        "big_geometry_big_mass",
    ):
        assert cell in runner
        assert cell in summary
    assert '"big_over_small_mass_ratio_is_1p5"' in summary


def test_context_goal_factorial_installs_goal_before_common_prefix() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    runner = CONTEXT_GOAL_RUNNER.read_text(encoding="utf-8")
    summary = CONTEXT_GOAL_SUMMARY.read_text(encoding="utf-8")
    install = evaluator.index("command.obj_target_pos_w[:] = target_position")
    prefix = evaluator.index("prefix_action = official_policy(obs)")
    assert install < prefix
    assert 'command._call_generator(all_env_ids)' in evaluator[install:prefix]
    for cell in (
        "carry_context_carry_goal",
        "carry_context_kick_goal",
        "kick_context_carry_goal",
        "kick_context_kick_goal",
    ):
        assert cell in runner
        assert cell in summary
    assert '"carry_goal_is_exact_across_initialization_contexts"' in summary


def test_renderer_requires_one_checkpoint_exact_pairs_and_decodable_video() -> None:
    source = RENDERER.read_text(encoding="utf-8")
    assert '"one_shared_checkpoint"' in source
    assert '"carry_initial_state_exact_match"' in source
    assert '"kick_initial_state_exact_match"' in source
    assert '"carry_kick_route_action_explosion_rejected"' in source
    assert '"all_videos_h264_yuv420p"' in source
    assert '"carry_domain_full_kick_route_passes"' in source
    assert '"kick_domain_full_carry_route_rejection_is_recorded"' in source
    assert '"sugar_official_generator_tracker_router_exact_trace_video_v1"' in source


def test_new_pipeline_has_no_human_authorization_gate() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            TRAINER,
            EVALUATOR,
            RENDERER,
            JOINT_RUNNER,
            FACTORIAL_RUNNER,
            FACTORIAL_SUMMARY,
            GEOMETRY_MASS_RUNNER,
            GEOMETRY_MASS_SUMMARY,
            CONTEXT_GOAL_RUNNER,
            CONTEXT_GOAL_SUMMARY,
            SAFE_RUNNER,
            SAFE_SUMMARY,
            COMPATIBILITY_RUNNER,
            COMPATIBILITY_SUMMARY,
        )
    ).lower()
    for forbidden in ("approval flag", "authorization flag", "sentinel file"):
        assert forbidden not in combined
