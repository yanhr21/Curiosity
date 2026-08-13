from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "plan15_online_mass_jump",
    ROOT / "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_mass_jump.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
MassJumpConfig = module.MassJumpConfig
OnlineMassJumpController = module.OnlineMassJumpController


class FakeRigidBodyView:
    def __init__(self, masses: torch.Tensor, inertias: torch.Tensor):
        self._masses = masses.clone()
        self._inertias = inertias.clone()

    def get_masses(self):
        return self._masses.clone()

    def set_masses(self, masses, indices):
        ids = indices.long()
        self._masses[ids] = masses[ids]

    def get_inertias(self):
        return self._inertias.clone()

    def set_inertias(self, inertias, indices):
        ids = indices.long()
        self._inertias[ids] = inertias[ids]


def fake_env(num_envs=2):
    default_mass = torch.full((num_envs, 1), 0.5)
    default_inertia = torch.eye(3).reshape(1, 9).repeat(num_envs, 1)
    view = FakeRigidBodyView(default_mass, default_inertia)
    obj = SimpleNamespace(
        data=SimpleNamespace(
            default_mass=default_mass,
            default_inertia=default_inertia,
            root_pos_w=torch.zeros(num_envs, 3),
        ),
        root_physx_view=view,
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"obj": obj},
    )


def test_mass_jump_scales_mass_and_inertia_without_touching_pose():
    env = fake_env(1)
    config = MassJumpConfig(
        nominal_mass_kg=0.3023375869,
        mass_factors=(3.0,),
        minimum_lift_m=0.05,
        stable_bilateral_frames=2,
        delay_frames=(1, 1),
    )
    controller = OnlineMassJumpController(env, "obj", config)
    controller.reset()
    nominal_inertia_scale = config.nominal_mass_kg / 0.5
    torch.testing.assert_close(
        env.scene["obj"].root_physx_view.get_inertias(),
        torch.eye(3).reshape(1, 9) * nominal_inertia_scale,
    )
    original_pose = env.scene["obj"].data.root_pos_w.clone()
    controller.advance(torch.tensor([True]), control_step=1)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.06
    assert controller.advance(torch.tensor([True]), control_step=2).numel() == 0
    jumped = controller.advance(torch.tensor([True]), control_step=3)
    assert jumped.tolist() == [0]
    expected_mass = torch.tensor([[config.nominal_mass_kg * 3.0]])
    torch.testing.assert_close(
        env.scene["obj"].root_physx_view.get_masses(), expected_mass
    )
    torch.testing.assert_close(
        env.scene["obj"].root_physx_view.get_inertias(),
        torch.eye(3).reshape(1, 9) * (config.nominal_mass_kg * 3.0 / 0.5),
    )
    torch.testing.assert_close(
        env.scene["obj"].data.root_pos_w[:, :2], original_pose[:, :2]
    )
    diagnostics = controller.diagnostics()
    assert diagnostics["jump_step"].tolist() == [3]
    torch.testing.assert_close(
        diagnostics["mass_readback_kg"], expected_mass.flatten()
    )


def test_mass_jump_requires_consecutive_lifted_bilateral_frames():
    env = fake_env(1)
    config = MassJumpConfig(
        mass_factors=(1.5,),
        stable_bilateral_frames=3,
        delay_frames=(2, 2),
    )
    controller = OnlineMassJumpController(env, "obj", config)
    controller.reset()
    controller.advance(torch.tensor([True]), control_step=0)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.06
    controller.advance(torch.tensor([True]), control_step=1)
    controller.advance(torch.tensor([False]), control_step=2)
    controller.advance(torch.tensor([True]), control_step=3)
    controller.advance(torch.tensor([True]), control_step=4)
    controller.advance(torch.tensor([True]), control_step=5)
    assert controller.diagnostics()["qualified"].item() is True
    assert controller.advance(torch.tensor([True]), control_step=6).tolist() == [0]


def test_factor_one_is_a_true_no_jump_episode():
    env = fake_env(1)
    config = MassJumpConfig(
        mass_factors=(1.0,),
        stable_bilateral_frames=1,
        delay_frames=(0, 0),
    )
    controller = OnlineMassJumpController(env, "obj", config)
    controller.reset()
    controller.advance(torch.tensor([True]), control_step=0)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.1
    controller.advance(torch.tensor([True]), control_step=1)
    diagnostics = controller.diagnostics()
    assert diagnostics["target_factor"].tolist() == [1.0]
    assert diagnostics["jump_applied"].item() is False
    assert diagnostics["jump_step"].tolist() == [-1]
