from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/sugar/demo_following/assess_corrected_phase_frozen_carry_gate.py"
SPEC = importlib.util.spec_from_file_location("corrected_phase_carry_gate", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_first_episode_mask_keeps_terminal_transition() -> None:
    done = np.zeros((MODULE.STEPS, MODULE.NUM_ENVS), dtype=bool)
    done[10, 0] = True
    mask = MODULE.first_episode_mask(done)
    assert mask[:11, 0].all()
    assert not mask[11:, 0].any()
    assert mask[:, 1:].all()


def test_semantic_blocks_require_reference_phase_and_profile_preference() -> None:
    shape = (MODULE.STEPS, MODULE.NUM_ENVS)
    phase = np.full(
        shape,
        np.float32((MODULE.REFERENCE_FRAME + 2) / MODULE.PHASE_HORIZON_STEPS),
        dtype=np.float32,
    )
    trace = {
        "done": np.zeros(shape, dtype=bool),
        "demo_correct_phase": phase,
        "demo_unrelated_phase": phase.copy(),
        "demo_correct_ready": np.ones(shape, dtype=bool),
        "demo_unrelated_ready": np.ones(shape, dtype=bool),
        "demo_correct_risk": np.full(shape, 0.25, dtype=np.float32),
        "demo_unrelated_risk": np.full(shape, 0.75, dtype=np.float32),
    }
    blocks = MODULE.semantic_blocks(trace)
    assert set(blocks) == {"update_0032", "update_0064"}
    assert all(
        block["carry_preferred_profile_count"] == MODULE.PROFILES_PER_UPDATE
        and block["carry_preferred_frame_fraction"] == 1.0
        and np.isclose(block["mean_kick_minus_carry_risk"], 0.5)
        for block in blocks.values()
    )
