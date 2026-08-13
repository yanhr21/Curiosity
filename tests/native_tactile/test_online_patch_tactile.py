from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ANATOMICAL_MODULE = "sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1"
anatomical = ModuleType(ANATOMICAL_MODULE)
anatomical.ANATOMICAL_WHOLE_HAND_PATCH_SPECS = tuple(
    SimpleNamespace(name=f"patch_{index:02d}", width_m=0.01, length_m=0.02)
    for index in range(27)
)
sys.modules.setdefault(ANATOMICAL_MODULE, anatomical)
spec = importlib.util.spec_from_file_location(
    "plan15_online_patch_tactile",
    ROOT / "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_patch_tactile.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
PATCH_FEATURE_WIDTH = module.PATCH_FEATURE_WIDTH
exact_zero_online_patch_tactile_actor_history = (
    module.exact_zero_online_patch_tactile_actor_history
)
online_patch_tactile_contract = module.online_patch_tactile_contract
reduce_patch_taxels = module.reduce_patch_taxels


def test_patch_contract_uses_54_patch_tokens_not_taxels():
    contract = online_patch_tactile_contract()
    assert contract["shape_without_batch"] == [4, 2, 27, 9]
    assert contract["flat_width"] == 1944
    assert contract["policy_unit"] == "physical anatomical patch"
    assert contract["uses_taxels_as_policy_units"] is False
    assert PATCH_FEATURE_WIDTH == 9


def test_reduce_patch_taxels_preserves_signed_shear_and_corrects_normal_sign():
    penetration = torch.tensor([[0.001, 0.002, 0.0, -0.1]])
    normal = torch.tensor([[-2.0, 3.0, 99.0, 99.0]])
    shear = torch.tensor([[[1.0, -2.0], [-0.5, 3.0], [7.0, 7.0], [8.0, 8.0]]])
    features = reduce_patch_taxels(
        penetration,
        normal,
        shear,
        patch_area_m2=0.01,
        friction_coefficient=0.5,
    )
    assert features.shape == (1, 6)
    torch.testing.assert_close(features[:, 0], torch.tensor([1.0]))
    torch.testing.assert_close(features[:, 1], torch.tensor([5.0]))
    torch.testing.assert_close(features[:, 2], torch.tensor([500.0]))
    torch.testing.assert_close(features[:, 3:5], torch.tensor([[0.5, 1.0]]))
    torch.testing.assert_close(
        features[:, 5],
        torch.tensor([(0.5**2 + 1.0**2) ** 0.5 / 2.5]),
    )


def test_reduce_patch_taxels_returns_zero_without_penetration():
    features = reduce_patch_taxels(
        torch.zeros(2, 4),
        torch.full((2, 4), 3.0),
        torch.full((2, 4, 2), 2.0),
        patch_area_m2=0.01,
        friction_coefficient=0.5,
    )
    torch.testing.assert_close(features, torch.zeros(2, 6))


def test_exact_zero_does_not_touch_scene_sensors():
    class ForbiddenScene:
        @property
        def sensors(self):
            raise AssertionError("exact-zero observation read a sensor")

    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene=ForbiddenScene(),
    )
    output = exact_zero_online_patch_tactile_actor_history(env)
    assert output.shape == (3, 1944)
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize("area", [0.0, -0.1, float("nan")])
def test_patch_reducer_rejects_invalid_area(area):
    with pytest.raises(ValueError, match="patch area"):
        reduce_patch_taxels(
            torch.ones(1, 2),
            torch.ones(1, 2),
            torch.ones(1, 2, 2),
            patch_area_m2=area,
            friction_coefficient=0.5,
        )
