from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/sugar/demo_following/evaluate_official_roboclip_video_reward.py"
RUNNER = ROOT / "scripts/sugar/demo_following/run_official_roboclip_audit_then_hold.sh"


def _function(name: str):
    module = ast.parse(PATH.read_text(encoding="utf-8"))
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_uniform_frame_sampling_is_exact_and_monotonic() -> None:
    function = _function("uniform_frame_indices")
    namespace = {"np": np, "INPUT_FRAMES": 32}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PATH), "exec"), namespace)
    indices = namespace["uniform_frame_indices"](64)
    assert indices.shape == (32,)
    assert indices[0] == 0
    assert indices[-1] == 63
    assert np.all(np.diff(indices) > 0)


def test_adapter_uses_only_released_roboclip_s3d_reward() -> None:
    text = PATH.read_text(encoding="utf-8")
    assert "module.S3D(str(dictionary), 512)" in text
    assert "load_state_dict(payload, strict=True)" in text
    assert 'model(batch)["video_embedding"]' in text
    assert "np.dot(first, second)" in text
    assert "optimizer" not in text.lower()
    assert "train(" not in text
    assert "INPUT_FRAMES = 32" in text
    assert "INPUT_SIZE = 224" in text


def test_gate_requires_reference_identity_and_temporal_order() -> None:
    text = PATH.read_text(encoding="utf-8")
    required = (
        "official_dot_valid_reference_accuracy_at_least_0p75",
        "official_dot_test_reference_accuracy_at_least_0p75",
        "cosine_valid_reference_accuracy_at_least_0p75",
        "cosine_test_reference_accuracy_at_least_0p75",
        "official_dot_valid_order_accuracy_at_least_0p75",
        "official_dot_test_order_accuracy_at_least_0p75",
    )
    for criterion in required:
        assert criterion in text
    assert '"passed": bool(all(criteria.values()))' in text
    assert 'REFERENCE_IDS = {"CarryBox": 45, "KickBox": 21}' in text


def test_runner_pins_official_assets_and_has_no_policy_stage() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "2d3f779033f1f3adf307a64080742e158caafe67" in text
    assert "b8cd0bbfd16fe41629d1b15e0cf384d75f56101a" in text
    assert "evaluate_official_roboclip_video_reward.py" in text
    assert "ROBOCLIP_AUTOMATIC_DECISION" in text
    assert "GPU_HOLD_AFTER_ROBOCLIP_AUDIT_READY" in text
    assert "train_" not in text
    assert "approval" not in text.lower()
