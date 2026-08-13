from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "plan15_evaluate_online_patch_slip",
    ROOT / "scripts/sugar/native_tactile/evaluate_online_patch_slip.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_oracle_state_respects_contact_and_two_speed_thresholds():
    contact = np.array([False, True, True, True])
    speed = np.array([1.0, 0.001, 0.006, 0.03])
    state = module.oracle_state(
        contact,
        speed,
        incipient_speed_m_s=0.005,
        gross_speed_m_s=0.02,
    )
    assert state.tolist() == [0, 1, 2, 3]


def test_onset_delay_is_causal_and_counts_missed_events():
    oracle = np.zeros((8, 2, 27), dtype=bool)
    predicted = np.zeros_like(oracle)
    oracle[2:5, 0, 0] = True
    predicted[4:5, 0, 0] = True
    oracle[5:7, 1, 3] = True
    delays, missed = module.onset_delays(oracle, predicted)
    assert delays == [2]
    assert missed == 1
