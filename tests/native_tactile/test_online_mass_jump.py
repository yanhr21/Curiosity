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
        cfg=SimpleNamespace(seed=150814),
        scene={"obj": obj},
    )


def test_mass_jump_scales_mass_and_inertia_without_touching_pose():
    env = fake_env(1)
    config = MassJumpConfig(
        nominal_mass_kg=0.3023375869,
        mass_factors=(3.0,),
        minimum_lift_m=0.05,
        stable_lift_frames=2,
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
    controller.advance(control_step=1)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.06
    assert controller.advance(control_step=2).numel() == 0
    jumped = controller.advance(control_step=3)
    assert jumped.tolist() == [0]
    # Scheduling occurs after the nominal physics step.  The write is delayed
    # to the next action boundary so the new mass affects physics before the
    # actor can observe its tactile consequence.
    torch.testing.assert_close(
        env.scene["obj"].root_physx_view.get_masses(),
        torch.tensor([[config.nominal_mass_kg]]),
    )
    assert controller.diagnostics()["pending"].item() is True
    applied = controller.apply_pending(control_step=3)
    assert applied.tolist() == [0]
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
    assert diagnostics["pending_step"].tolist() == [3]
    assert diagnostics["pending"].item() is False
    torch.testing.assert_close(
        diagnostics["mass_readback_kg"], expected_mass.flatten()
    )
    torch.testing.assert_close(
        diagnostics["inertia_readback_kg_m2"],
        env.scene["obj"].root_physx_view.get_inertias(),
    )
    assert diagnostics["cumulative_jump_events"].tolist() == [1]
    assert diagnostics["cumulative_mass_changes"].tolist() == [1]
    assert diagnostics["cumulative_factor_events"].tolist() == [[1]]


def test_mass_jump_requires_consecutive_lifted_frames_without_tactile_read():
    env = fake_env(1)
    config = MassJumpConfig(
        mass_factors=(1.5,),
        stable_lift_frames=3,
        delay_frames=(2, 2),
    )
    controller = OnlineMassJumpController(env, "obj", config)
    controller.reset()
    controller.advance(control_step=0)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.06
    controller.advance(control_step=1)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.0
    controller.advance(control_step=2)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.06
    controller.advance(control_step=3)
    controller.advance(control_step=4)
    controller.advance(control_step=5)
    assert controller.diagnostics()["qualified"].item() is True
    assert controller.advance(control_step=6).tolist() == [0]


def test_factor_one_uses_matched_event_clock_without_changing_mass():
    env = fake_env(1)
    config = MassJumpConfig(
        mass_factors=(1.0,),
        stable_lift_frames=1,
        delay_frames=(0, 0),
    )
    controller = OnlineMassJumpController(env, "obj", config)
    controller.reset()
    controller.advance(control_step=0)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.1
    scheduled = controller.advance(control_step=1)
    assert scheduled.tolist() == [0]
    before_mass = env.scene["obj"].root_physx_view.get_masses()
    applied = controller.apply_pending(control_step=1)
    assert applied.tolist() == [0]
    diagnostics = controller.diagnostics()
    assert diagnostics["target_factor"].tolist() == [1.0]
    assert diagnostics["jump_applied"].item() is True
    assert diagnostics["mass_changed"].item() is False
    assert diagnostics["jump_step"].tolist() == [1]
    torch.testing.assert_close(
        env.scene["obj"].root_physx_view.get_masses(), before_mass
    )


def test_mass_delay_starts_only_after_live_teacher_handoff():
    env = fake_env(1)
    env._online_teacher_handoff_controller = SimpleNamespace(
        handoff_active=torch.tensor([False])
    )
    config = MassJumpConfig(
        mass_factors=(3.0,),
        stable_lift_frames=1,
        delay_frames=(2, 2),
    )
    controller = OnlineMassJumpController(env, "obj", config)
    controller.reset()
    controller.advance(control_step=0)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.1
    for step in range(1, 5):
        assert controller.advance(control_step=step).numel() == 0
    assert controller.qualified.item() is False
    env._online_teacher_handoff_controller.handoff_active[:] = True
    assert controller.advance(control_step=5).numel() == 0
    assert controller.qualified.item() is True
    assert controller.advance(control_step=6).tolist() == [0]


def test_training_assignment_cycles_all_mass_factors_per_env():
    env = fake_env(4)
    config = MassJumpConfig()
    controller = OnlineMassJumpController(env, "obj", config)
    assigned = []
    for _ in range(len(config.mass_factors)):
        controller.reset()
        assigned.append(controller.diagnostics()["target_factor"].clone())
    assigned = torch.stack(assigned, dim=0)
    expected = sorted(config.mass_factors)
    for env_index in range(env.num_envs):
        assert sorted(assigned[:, env_index].tolist()) == expected
