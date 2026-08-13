from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/patch_slip.py"
)
SPEC = importlib.util.spec_from_file_location("plan15_patch_slip", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
import sys

sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fields(batch: int = 1):
    shape = (batch, 2, 27)
    return {
        "contact": torch.ones(shape, dtype=torch.bool),
        "normal_load_n": torch.ones(shape),
        "mean_pressure_pa": torch.full(shape, 1000.0),
        "shear_xy_n": torch.zeros((*shape, 2)),
        "friction_utilization": torch.full(shape, 0.1),
    }


def update(detector, values, timestamp: float, reset: bool = False):
    return detector.update(
        **values,
        timestamp_s=torch.tensor([timestamp]),
        reset_mask=torch.tensor([reset]),
    )


def test_callable_signature_has_only_causal_patch_inputs():
    names = tuple(inspect.signature(MODULE.PatchSlipDetector.update).parameters)
    assert names == (
        "self",
        "contact",
        "normal_load_n",
        "mean_pressure_pa",
        "shear_xy_n",
        "friction_utilization",
        "timestamp_s",
        "reset_mask",
    )


def test_first_contact_is_stick_not_false_slip():
    detector = MODULE.PatchSlipDetector(1, device="cpu")
    output = update(detector, fields(), 0.0, reset=True)
    assert torch.all(output.state == MODULE.STICK)
    assert not output.incipient_slip.any()
    assert not output.gross_slip.any()
    assert torch.all(output.slip_score < 0.2)


def test_high_friction_utilization_causes_gross_slip():
    detector = MODULE.PatchSlipDetector(1, device="cpu")
    values = fields()
    update(detector, values, 0.0, reset=True)
    values["friction_utilization"][:] = 0.95
    output = update(detector, values, 0.02)
    assert torch.all(output.state == MODULE.GROSS)
    assert output.gross_slip.all()
    assert torch.all(output.slip_score >= 1.0)


def test_pressure_drop_and_contact_loss_are_causal_slip_evidence():
    detector = MODULE.PatchSlipDetector(1, device="cpu")
    values = fields()
    update(detector, values, 0.0, reset=True)
    values["mean_pressure_pa"][:] = 800.0
    output = update(detector, values, 0.02)
    assert output.gross_slip.all()

    values["contact"][:] = False
    output = update(detector, values, 0.04)
    assert output.gross_slip.all()
    assert torch.all(output.state == MODULE.GROSS)


def test_reset_clears_history_and_allows_new_episode_clock():
    detector = MODULE.PatchSlipDetector(1, device="cpu")
    values = fields()
    update(detector, values, 10.0, reset=True)
    values["friction_utilization"][:] = 0.1
    output = update(detector, values, 0.0, reset=True)
    assert torch.all(output.state == MODULE.STICK)
    assert not output.gross_slip.any()


def test_timestamp_must_increase_without_reset():
    detector = MODULE.PatchSlipDetector(1, device="cpu")
    values = fields()
    update(detector, values, 0.02, reset=True)
    with pytest.raises(ValueError, match="timestamp_s must increase"):
        update(detector, values, 0.02)


def test_rejects_non_patch_geometry():
    with pytest.raises(ValueError, match=r"\[batch,2,27\]"):
        MODULE.PatchSlipDetector(1, hands=1, device="cpu")
