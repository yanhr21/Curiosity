from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "plan15_online_teacher_handoff",
    ROOT
    / "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/online_teacher_handoff.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
OnlineTeacherHandoffController = module.OnlineTeacherHandoffController
TeacherHandoffConfig = module.TeacherHandoffConfig
online_teacher_handoff_training_mask = (
    module.online_teacher_handoff_training_mask
)


def fake_env(num_envs: int = 2):
    obj = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=torch.zeros(num_envs, 3))
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"obj": obj},
    )


def test_handoff_requires_consecutive_live_lift_and_preserves_other_env():
    env = fake_env()
    controller = OnlineTeacherHandoffController(
        env,
        "obj",
        TeacherHandoffConfig(minimum_lift_m=0.05, stable_lift_frames=3),
    )
    env._online_teacher_handoff_controller = controller
    controller.reset()
    controller.advance(control_step=0)
    env.scene["obj"].data.root_pos_w[0, 2] = 0.06
    controller.advance(control_step=1)
    env.scene["obj"].data.root_pos_w[0, 2] = 0.0
    controller.advance(control_step=2)
    env.scene["obj"].data.root_pos_w[0, 2] = 0.06
    controller.advance(control_step=3)
    controller.advance(control_step=4)
    assert controller.advance(control_step=5).tolist() == [0]
    assert controller.handoff_active.tolist() == [True, False]
    assert controller.handoff_step.tolist() == [5, -1]
    torch.testing.assert_close(
        online_teacher_handoff_training_mask(env),
        torch.tensor([[1.0], [0.0]]),
    )


def test_reset_clears_only_requested_handoff_state():
    env = fake_env()
    controller = OnlineTeacherHandoffController(
        env,
        "obj",
        TeacherHandoffConfig(minimum_lift_m=0.05, stable_lift_frames=1),
    )
    controller.reset()
    controller.advance(control_step=0)
    env.scene["obj"].data.root_pos_w[:, 2] = 0.06
    assert controller.advance(control_step=1).tolist() == [0, 1]
    controller.reset(torch.tensor([0]))
    assert controller.handoff_active.tolist() == [False, True]
    assert controller.cumulative_handoffs.tolist() == [1, 1]
