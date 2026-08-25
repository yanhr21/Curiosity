from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/sugar/demo_following/build_official_transition_risk_dataset.py"
TRAINER = ROOT / "scripts/sugar/demo_following/train_official_transition_risk_transformer.py"
RUNNER = ROOT / "scripts/sugar/demo_following/run_official_transition_risk_training.sh"
EVALUATOR = ROOT / "scripts/sugar/demo_following/evaluate_official_transition_risk_checkpoint.py"
ANCHOR9_AUDIT = ROOT / "scripts/sugar/demo_following/audit_official_transition_risk_anchor9.py"
ONLINE_EVALUATOR = ROOT / "scripts/sugar/demo_following/evaluate_demo_conditioned_tracker.py"
ONLINE_RUNNER = ROOT / "scripts/sugar/demo_following/run_online_transition_risk_fallback.sh"


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(source, child) or ""
    raise AssertionError(f"missing {class_name}.{method_name}")


def test_deployed_transition_predictor_is_past_only() -> None:
    source = _method_source(TRAINER, "CausalTransitionRiskTransformer", "forward")
    assert "causal_prefix" in source
    for forbidden in ("future", "outcome", "success", "risk_target", "profile"):
        assert forbidden not in source


def test_transition_predictor_retains_serious_transformer_scale() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    assert "d_model = 384" in source
    assert "nhead=8" in source
    assert "dim_feedforward=1536" in source
    assert "num_layers=6" in source
    assert "10_000_000 <= parameter_count <= 15_000_000" in source


def test_dataset_split_is_profile_and_seed_disjoint() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    assert "range(0, 14)" in source
    assert "range(14, 20)" in source
    assert "seed171622" in source
    assert '"train_validation_profile_disjoint": True' in source
    assert '"test_seed_and_context_disjoint": True' in source
    assert '"future_outcome_absent_from_input": True' in source


def test_training_runs_serious_overfit_before_formal() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    overfit = source.index("--mode overfit")
    formal = source.index("--mode formal")
    assert overfit < formal
    assert "--overfit-steps 500" in source
    for forbidden in ("approval", "authorization", "sentinel", "confirm"):
        assert forbidden not in source.lower()


def test_frozen_threshold_is_selected_on_validation_deployment_window() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert 'rows["validation"]' in source
    assert "None, True" in source
    assert 'threshold = float(validation_early["threshold"])' in source
    assert 'rows["test"]' in source
    assert '"threshold_selected_on_validation_first50_only": True' in source


def test_earliest_decision_audit_is_validation_only_anchor9() -> None:
    source = ANCHOR9_AUDIT.read_text(encoding="utf-8")
    assert "ANCHOR = 9" in source
    validation = source.index('_anchor_metrics(rows["validation"], None)')
    test = source.index('_anchor_metrics(rows["test"], threshold)')
    assert validation < test
    assert '"threshold_selected_on_validation_anchor9_only": True' in source
    assert '"matched_online_anchor9_fallback"' in source


def test_online_transition_gate_is_early_causal_and_latched() -> None:
    source = ONLINE_EVALUATOR.read_text(encoding="utf-8")
    runner = ONLINE_RUNNER.read_text(encoding="utf-8")
    assert "TRANSITION_RISK_DECISION_FRAME = 49" in source
    assert "if rollout_step <= TRANSITION_RISK_DECISION_FRAME" in source
    assert "transition_history" in source
    assert "transition_latched_fallback |= " in source
    assert "transition_probability_sum / transition_probability_count" in source
    assert "teacher_action" in source and "student_action" in source
    assert "shared_policy.experts[selected_expert_index]" in source
    assert "legacy_demo_event_scorer_absent_from_fallback_decision" in source
    assert "if not args.causal_transition_risk_fallback" in source
    assert "--causal-transition-risk-fallback" in runner
    for forbidden in ("approval", "authorization", "sentinel", "confirm"):
        assert forbidden not in runner.lower()
