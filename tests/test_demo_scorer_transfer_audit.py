from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "scripts/sugar/demo_following/audit_phase_event_scorer_transfer.py"
)
SPEC = importlib.util.spec_from_file_location("demo_scorer_transfer_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_causal_phase_step_matches_runtime_and_reference_offset() -> None:
    phase, steps = MODULE.causal_phase_step(
        np.asarray([0, 197], dtype=np.int64),
        np.asarray([False, False]),
    )
    assert np.allclose(
        phase,
        np.asarray([2 / 650, 199 / 650], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(steps, np.asarray([1, 198]))


def test_causal_phase_step_resets_after_done() -> None:
    phase, steps = MODULE.causal_phase_step(
        np.asarray([31, 228], dtype=np.int64),
        np.asarray([True, True]),
    )
    assert np.allclose(
        phase,
        np.asarray([1 / 650, 1 / 650], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(steps, np.asarray([0, 0]))
    phase, steps = MODULE.causal_phase_step(steps, np.asarray([False, False]))
    assert np.allclose(
        phase,
        np.asarray([2 / 650, 2 / 650], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_first_episode_transition_mask_includes_terminal_transition() -> None:
    done = np.asarray(
        [
            [False, False, False],
            [True, False, False],
            [False, False, False],
            [False, True, False],
        ]
    )
    mask = MODULE.first_episode_transition_mask(done)
    assert np.array_equal(
        mask,
        np.asarray(
            [
                [True, True, True],
                [True, True, True],
                [False, True, True],
                [False, True, True],
            ]
        ),
    )


def test_source_phase_variant_is_explicit_and_backward_compatible() -> None:
    assert MODULE.infer_source_phase_variant({}) == "reset_zero"
    assert MODULE.infer_source_phase_variant(
        {"phase_initialization": {"mode": "reset-zero-diagnostic"}}
    ) == "reset_zero"
    assert MODULE.infer_source_phase_variant(
        {"phase_initialization": {"mode": "reference-aware"}}
    ) == "reference_aware"


def test_runtime_reproduction_selects_declared_phase_variant() -> None:
    trace = {
        "demo_correct_risk": np.asarray([[3.0]], dtype=np.float32),
        "demo_correct_reward": np.asarray([[4.0]], dtype=np.float32),
        "demo_correct_ready": np.asarray([[True]]),
        "demo_correct_phase": np.asarray([[0.3]], dtype=np.float32),
        "demo_correct_weighted_uncertainty": np.asarray(
            [[5.0]], dtype=np.float32
        ),
    }
    scores = {
        "risk": np.asarray([[[30.0]], [[3.0]]], dtype=np.float32),
        "reward": np.asarray([[[40.0]], [[4.0]]], dtype=np.float32),
        "ready": np.asarray([[[False]], [[True]]]),
        "phase": np.asarray([[[0.0]], [[0.3]]], dtype=np.float32),
        "uncertainty": np.asarray([[[50.0]], [[5.0]]], dtype=np.float32),
    }
    assert MODULE.runtime_reproduction(
        trace,
        scores,
        "correct",
        phase_variant_index=1,
    )["passed"] is True
    assert MODULE.runtime_reproduction(
        trace,
        scores,
        "correct",
        phase_variant_index=0,
    )["passed"] is False
