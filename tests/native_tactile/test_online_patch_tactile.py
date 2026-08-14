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
current_whole_hand_patch_oracle_tangential_speed = (
    module.current_whole_hand_patch_oracle_tangential_speed
)
current_whole_hand_patch_timestamps_s = (
    module.current_whole_hand_patch_timestamps_s
)


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
    diagnostics = env._online_patch_tactile_runtime_diagnostics
    assert diagnostics["zero_observation_calls"] == 1
    assert diagnostics["zero_env_samples"] == 3
    assert diagnostics["patch_sensor_reads"] == 0


@pytest.mark.parametrize("branch", ["Z", "P", "PS"])
def test_live_preflight_report_enforces_each_branch_path(branch):
    env = SimpleNamespace(device="cpu")
    diagnostics = module._runtime_diagnostics(env)
    if branch == "Z":
        diagnostics["zero_observation_calls"] = 2
        diagnostics["zero_env_samples"] = 8
    else:
        diagnostics["live_feature_updates"] = 2
        diagnostics["live_env_samples"] = 8
        diagnostics["patch_sensor_reads"] = 108
        diagnostics["bilateral_contact_env_samples"] += 3
        diagnostics["maximum_normal_load_n"] += 1.5
        diagnostics["maximum_active_patches_per_hand"][:] = torch.tensor([4, 5])
    if branch == "PS":
        diagnostics["slip_updates"] = 2
    env._online_mass_jump_controller = SimpleNamespace(
        cumulative_jump_events=torch.tensor([1, 1, 0, 0]),
        cumulative_mass_changes=torch.tensor([0, 1, 0, 0]),
        cumulative_factor_events=torch.tensor(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        ),
    )
    env._online_teacher_handoff_controller = SimpleNamespace(
        cumulative_handoffs=torch.tensor([1, 1, 0, 0])
    )
    env._online_teacher_handoff_wrapper = SimpleNamespace(
        cumulative_teacher_control_steps=torch.tensor([100, 100, 100, 100]),
        cumulative_policy_control_steps=torch.tensor([20, 20, 20, 20]),
    )
    env._online_handoff_bcppo_mask_report = {
        "active_policy_transitions": 80,
        "masked_teacher_transitions": 400,
        "total_transitions": 480,
    }
    report = module.online_patch_preflight_runtime_report(env, branch)
    assert report["overall_pass"] is True
    assert report["mass_runtime"]["jump_events"] == 2
    assert report["mass_runtime"]["mass_changes"] == 1


def test_training_path_preflight_does_not_fake_a_mass_event_before_lift():
    env = SimpleNamespace(device="cpu")
    diagnostics = module._runtime_diagnostics(env)
    diagnostics["zero_observation_calls"] = 2
    diagnostics["zero_env_samples"] = 2
    env._online_mass_jump_controller = SimpleNamespace(
        cumulative_jump_events=torch.zeros(1, dtype=torch.long),
        cumulative_mass_changes=torch.zeros(1, dtype=torch.long),
        cumulative_factor_events=torch.zeros(1, 5, dtype=torch.long),
    )
    env._online_teacher_handoff_controller = SimpleNamespace(
        cumulative_handoffs=torch.zeros(1, dtype=torch.long)
    )
    env._online_teacher_handoff_wrapper = SimpleNamespace(
        cumulative_teacher_control_steps=torch.ones(1, dtype=torch.long),
        cumulative_policy_control_steps=torch.zeros(1, dtype=torch.long),
    )
    env._online_handoff_bcppo_mask_report = {
        "active_policy_transitions": 0,
        "masked_teacher_transitions": 2,
        "total_transitions": 2,
    }
    report = module.online_patch_preflight_runtime_report(env, "Z")
    assert report["checks"]["mass_event_seen"] is False
    assert report["checks"]["physical_mass_change_seen"] is False
    assert report["overall_pass"] is False


def test_live_branch_preflight_requires_contact_after_handoff():
    env = SimpleNamespace(device="cpu")
    diagnostics = module._runtime_diagnostics(env)
    diagnostics["live_feature_updates"] = 3
    diagnostics["live_env_samples"] = 3
    diagnostics["patch_sensor_reads"] = 162
    env._online_teacher_handoff_controller = SimpleNamespace(
        cumulative_handoffs=torch.ones(1, dtype=torch.long)
    )
    env._online_teacher_handoff_wrapper = SimpleNamespace(
        cumulative_teacher_control_steps=torch.ones(1, dtype=torch.long),
        cumulative_policy_control_steps=torch.ones(1, dtype=torch.long),
    )
    env._online_handoff_bcppo_mask_report = {
        "active_policy_transitions": 1,
        "masked_teacher_transitions": 1,
        "total_transitions": 2,
    }
    report = module.online_patch_preflight_runtime_report(env, "P")
    assert report["checks"]["live_branch_observed_bilateral_contact"] is False
    assert report["checks"]["live_branch_observed_nonzero_load"] is False
    assert report["overall_pass"] is False


def test_evaluation_oracle_reduces_max_active_taxel_speed_per_patch():
    names = tuple(tuple(names) for names in module.SENSOR_NAMES_BY_HAND)
    sensors = {}
    for hand_names in names:
        for sensor_name in hand_names:
            penetration = torch.zeros(2, 4)
            velocity = torch.full((2, 4, 3), 99.0)
            penetration[:, 1] = 0.001
            velocity[:, 1] = torch.tensor([3.0, 4.0, 0.0])
            sensors[sensor_name] = SimpleNamespace(
                data=SimpleNamespace(
                    penetration_depth=penetration,
                    tactile_relative_tangential_velocity_w=velocity,
                )
            )
    env = SimpleNamespace(num_envs=2, scene=SimpleNamespace(sensors=sensors))
    output = current_whole_hand_patch_oracle_tangential_speed(env, names)
    assert output.shape == (2, 2, 27)
    torch.testing.assert_close(output, torch.full_like(output, 5.0))


def test_collection_timestamps_preserve_all_54_official_sensor_clocks():
    names = tuple(tuple(names) for names in module.SENSOR_NAMES_BY_HAND)
    sensors = {}
    expected = torch.tensor([0.24, 0.48])
    for hand_names in names:
        for sensor_name in hand_names:
            sensors[sensor_name] = SimpleNamespace(
                data=SimpleNamespace(),
                _timestamp_last_update=expected.clone(),
            )
    env = SimpleNamespace(num_envs=2, scene=SimpleNamespace(sensors=sensors))
    output = current_whole_hand_patch_timestamps_s(env, names)
    assert output.shape == (2, 2, 27)
    torch.testing.assert_close(output, expected[:, None, None].expand_as(output))


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
