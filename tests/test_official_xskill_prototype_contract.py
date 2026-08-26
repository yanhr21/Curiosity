from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/sugar/demo_following/train_evaluate_official_xskill_prototypes.py"
PREPARE = ROOT / "scripts/sugar/demo_following/prepare_official_xskill_runtime.sh"
RUNNER = ROOT / "scripts/sugar/demo_following/run_official_xskill_audit_then_hold.sh"
RENDERER = ROOT / "scripts/sugar/demo_following/render_official_xskill_prototype_evidence.py"


def _function(name: str):
    module = ast.parse(PATH.read_text(encoding="utf-8"))
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_adapter_imports_released_model_and_exact_training_step() -> None:
    text = PATH.read_text(encoding="utf-8")
    assert "from xskill.model.core import Model" in text
    assert "model.training_step(move_batch(batch, device), batch_index)" in text
    assert "model.encoder_q.get_state_representation" in text
    assert "model.encoder_q.get_traj_representation" in text
    assert "nn.functional.normalize(raw, dim=1, p=2)" in text
    assert "model.encoder_q.prototypes(normalized)" in text
    assert "class Model" not in text
    assert "FINAL_EPOCH = 79" in text
    assert "BATCH_SIZE = 28" in text


def test_architecture_and_loss_contract_match_released_simulation_config() -> None:
    text = PATH.read_text(encoding="utf-8")
    for required in (
        "n_layer=8",
        "heads=4",
        "dim_feedforward=512",
        "nmb_prototypes=128",
        "sinkhorn_iterations=3",
        "epsilon=0.03",
        "swav_loss_coef=0.5",
        "positive_window=4",
        "negative_window=12",
        "n_negative_samples=16",
        '"random_crop_112_112"',
        '"color_jitter"',
        '"grayscale"',
        '"gaussian_blur"',
    ):
        assert required in text


def test_gate_requires_temporal_progress_task_identity_and_order() -> None:
    text = PATH.read_text(encoding="utf-8")
    required = (
        "raw_test_temporal_mae_relative_improvement_at_least_5pct",
        "raw_test_kendalls_tau_improvement_at_least_0p05",
        "raw_valid_task_reference_accuracy_at_least_0p75",
        "raw_test_task_reference_accuracy_at_least_0p75",
        "raw_valid_ordered_reference_accuracy_at_least_0p75",
        "raw_test_ordered_reference_accuracy_at_least_0p75",
    )
    for criterion in required:
        assert criterion in text
    assert '"passed": bool(all(criteria.values()))' in text
    assert 'REFERENCE_IDS = {"CarryBox": 45, "KickBox": 21}' in text


def test_dtw_cost_prefers_an_exact_ordered_sequence() -> None:
    function = _function("dtw_cost")
    namespace = {"np": np, "cdist": __import__("scipy.spatial.distance", fromlist=["cdist"]).cdist}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PATH), "exec"), namespace)
    sequence = np.arange(12, dtype=np.float64).reshape(6, 2)
    assert namespace["dtw_cost"](sequence, sequence) == 0.0
    assert namespace["dtw_cost"](sequence, sequence[::-1]) > 0.0


def test_runner_pins_source_and_never_admits_a_policy_on_weak_evidence() -> None:
    prepare = PREPARE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    commit = "b748071daeb031d6b42a8dcb88c38c52297e20af"
    assert commit in prepare
    assert commit in runner
    assert "absl-py==2.0.0" in prepare
    assert 'DEPS="$ROOT/experiments/runtime_assets/official_xskill_pydeps"' in runner
    assert "official_xskill_audit requires" not in runner.lower()
    assert "XSKILL_AUTOMATIC_DECISION" in runner
    assert "GPU_HOLD_AFTER_XSKILL_AUDIT_READY" in runner
    assert "approval" not in runner.lower()
    assert "policy" not in runner.lower()


def test_renderer_writes_four_separate_h264_evidence_videos() -> None:
    text = RENDERER.read_text(encoding="utf-8")
    assert '"libx264"' in text
    assert '"yuv420p"' in text
    assert '"+faststart"' in text
    assert "ffmpeg-linux-x86_64-v7.0.2" in text
    assert '"representation only - no policy or action"' in text
    assert '("CarryBox", 45, "train", "reference")' in text
    assert '("KickBox", 21, "train", "reference")' in text
    assert '("CarryBox", 99, "test", "heldout")' in text
    assert '("KickBox", 89, "test", "heldout")' in text
    assert '(.videos | length) == 4' in RUNNER.read_text(encoding="utf-8")
