from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/sugar/demo_following/train_official_tracker_router.py"
EVALUATOR = ROOT / "scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py"
RENDERER = ROOT / "scripts/sugar/demo_following/render_official_tracker_router.py"


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


def test_renderer_requires_one_checkpoint_exact_pairs_and_decodable_video() -> None:
    source = RENDERER.read_text(encoding="utf-8")
    assert '"one_shared_checkpoint"' in source
    assert '"carry_initial_state_exact_match"' in source
    assert '"kick_initial_state_exact_match"' in source
    assert '"carry_kick_route_action_explosion_rejected"' in source
    assert '"all_videos_h264_yuv420p"' in source


def test_new_pipeline_has_no_human_authorization_gate() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (TRAINER, EVALUATOR, RENDERER)
    ).lower()
    for forbidden in ("approval flag", "authorization flag", "sentinel file"):
        assert forbidden not in combined
