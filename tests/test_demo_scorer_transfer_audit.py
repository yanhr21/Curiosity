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
