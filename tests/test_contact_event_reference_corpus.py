from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    import importlib.util

    path = ROOT / "scripts/sugar/demo_reward/audit_contact_event_reference_corpus.py"
    spec = importlib.util.spec_from_file_location("contact_event_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_true_runs_are_inclusive_and_preserve_duration():
    module = _load_module()
    mask = np.asarray([False, True, True, False, True], dtype=bool)
    assert module.true_runs(mask) == [(1, 2), (4, 4)]


def test_reference_audit_keeps_proxy_claim_boundary():
    module = _load_module()
    data_root = ROOT / "SUGAR/data"
    if not (data_root / "CarryBox/data_000/robot_50hz.npz").is_file():
        pytest.skip("official SUGAR demo assets are local-only")
    result, records = module.audit(data_root)
    assert records
    assert result["passed"] is True
    assert result["source_contract"]["contact_labels_are_not_tactile_force"] is True
    assert result["source_contract"]["physical_contact_force_target_present"] is False
    assert result["automatic_next_branch"].startswith("collect_actual_rollout")
