from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/sugar/demo_following/audit_official_motiongpt_vqvae_instance_latent.py"
SPEC = importlib.util.spec_from_file_location("official_vqvae_gate", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_phase_window_has_exact_predeclared_length() -> None:
    body = np.arange(700 * 2 * 3, dtype=np.float32).reshape(700, 2, 3)
    for phase in MODULE.PHASES:
        result = MODULE.phase_window(body, phase)
        assert result.shape == (MODULE.SOURCE_WINDOW_FRAMES, 2, 3)


def test_wrong_demo_maps_are_deterministic_and_different() -> None:
    assert MODULE.same_task_wrong_key("CarryBox", 45) == ("CarryBox", 46)
    assert MODULE.cross_task_key("CarryBox", 45) == ("KickBox", 45)
    assert MODULE.same_task_wrong_key("KickBox", 98) == ("KickBox", 0)
    assert MODULE.cross_task_key("KickBox", 21) == ("CarryBox", 21)


def test_continuous_cosine_distance_is_clock_aligned() -> None:
    actual = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    same = actual.copy()
    opposite = -actual
    np.testing.assert_allclose(MODULE.continuous_cosine_distance(actual, same), 0.0)
    np.testing.assert_allclose(MODULE.continuous_cosine_distance(actual, opposite), 2.0)
