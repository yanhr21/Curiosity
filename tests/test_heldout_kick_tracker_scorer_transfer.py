from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "scripts/sugar/demo_following/audit_heldout_kick_tracker_scorer_transfer.py"
)
SPEC = importlib.util.spec_from_file_location("heldout_kick_tracker_transfer", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_shard_provenance_requires_released_generator_tracker_pair(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "TRACE.npz"
    trace_path.touch()
    payload = {
        "protocol": "sugar_official_tracker_actual_contact_event_canary_v1",
        "passed": True,
        "task_family": "KickBox",
        "tracker_checkpoint": str(MODULE.OFFICIAL_TRACKER),
        "generator_checkpoint": str(MODULE.OFFICIAL_GENERATOR),
        "checks": {
            "physical_force_vectors_recorded": True,
            "contact_is_exact_threshold_of_force": True,
        },
    }
    (tmp_path / "RESULT.json").write_text(json.dumps(payload), encoding="utf-8")
    assert MODULE.validate_shard_result(trace_path) == payload

    payload["generator_checkpoint"] = str(tmp_path / "replacement.ckpt")
    (tmp_path / "RESULT.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        MODULE.validate_shard_result(trace_path)
    except RuntimeError as error:
        assert "unverified official Kick shard" in str(error)
    else:
        raise AssertionError("non-official generator was accepted")


def test_semantic_summary_keeps_deployed_and_diagnostic_clocks_separate() -> None:
    steps = MODULE.TRACE_STEPS - 1
    profiles = len(MODULE.TEST_MOTION_IDS)
    shape = (len(MODULE.PHASE_VARIANTS), steps, profiles)
    data = {
        "motion_frame": np.broadcast_to(
            np.arange(MODULE.TRACE_STEPS, dtype=np.int64)[:, None],
            (MODULE.TRACE_STEPS, profiles),
        ),
        "reference_steps": np.full(profiles, MODULE.TRACE_STEPS, dtype=np.int64),
    }
    carry = np.zeros(shape, dtype=np.float32)
    kick = np.zeros(shape, dtype=np.float32)
    kick[0] = -0.25
    kick[1] = 0.5
    ready = np.ones(shape, dtype=bool)
    summary = MODULE.summarize_semantics(
        data,
        {
            "correct": {"risk": carry, "ready": ready},
            "unrelated": {"risk": kick, "ready": ready.copy()},
        },
    )
    assert summary["deployed_fixed_650"]["mean_kick_minus_carry_risk"] == -0.25
    assert summary["deployed_fixed_650"]["kick_preferred_profile_count"] == profiles
    assert (
        summary["source_duration_diagnostic"]["mean_kick_minus_carry_risk"]
        == 0.5
    )
    assert summary["source_duration_diagnostic"]["kick_preferred_profile_count"] == 0


def test_behavior_summary_requires_observed_kick_interaction() -> None:
    steps = MODULE.TRACE_STEPS
    profiles = len(MODULE.TEST_MOTION_IDS)
    contact = np.zeros((steps, profiles, 4), dtype=bool)
    contact[20, :5, 2] = True
    object_state = np.zeros((steps, profiles, 13), dtype=np.float32)
    object_state[40, :6, 0] = 0.02
    summary = MODULE.behavior_summary(
        {
            "contact": contact,
            "object_root_state_w": object_state,
            "lift_height_m": np.zeros((steps, profiles), dtype=np.float32),
            "reference_steps": np.full(profiles, steps, dtype=np.int64),
        }
    )
    assert summary["profiles_with_foot_contact"] == 5
    assert summary["profiles_moving_object_at_least_1cm"] == 6
