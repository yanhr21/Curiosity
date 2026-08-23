from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SUGAR/source/sugar_rl"))
SOURCE = ROOT / "scripts/sugar/demo_reward/demo_conditioned_causal_predictor_v1.py"
SPEC = importlib.util.spec_from_file_location("demo_event_predictor", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_serious_event_predictor_shapes_parameters_and_demo_dependence():
    model = MODULE.DemoConditionedCausalEventPredictorV2(
        policy_dim=510,
        policy_history_steps=10,
        demo_windows=32,
        demo_window_steps=10,
        demo_feature_dim=132,
        d_model=384,
        nhead=8,
        num_layers=6,
        dim_feedforward=1536,
        dropout=0.0,
        state_mean=torch.zeros(510),
        state_std=torch.ones(510),
        demo_mean=torch.zeros(132),
        demo_std=torch.ones(132),
        target_scale=torch.ones(len(MODULE.EVENT_TARGET_NAMES)),
    ).eval()
    parameter_count = MODULE.count_trainable_parameters(model)
    assert 10_000_000 <= parameter_count <= 15_000_000
    policy = torch.randn(2, 10, 510)
    demo = torch.randn(2, 32, 10, 132)
    with torch.inference_mode():
        full = model(policy_prefix=policy, selected_demo_condition=demo)
        zero = model(
            policy_prefix=policy,
            selected_demo_condition=demo,
            zero_demo=True,
        )
    assert full["mean_log1p_scaled"].shape == (2, 13)
    assert full["log_variance_log1p_scaled"].shape == (2, 13)
    assert full["representation"].shape == (2, 384)
    assert not torch.equal(full["mean_log1p_scaled"], zero["mean_log1p_scaled"])
