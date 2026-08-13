from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn


# The lightweight unit-test environment carries a newer incompatible RSL-RL
# package layout.  Expose the exact Sequential API used by the project version
# so this test exercises Plan-15 glue without importing Isaac Sim.
class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dims, activation="elu"):
        super().__init__()
        activation_module = nn.ELU() if activation == "elu" else nn.ReLU()
        widths = [input_dim, *hidden_dims, output_dim]
        index = 0
        for layer_index in range(len(widths) - 1):
            self.add_module(str(index), nn.Linear(widths[layer_index], widths[layer_index + 1]))
            index += 1
            if layer_index < len(widths) - 2:
                self.add_module(str(index), activation_module)
                index += 1


class EmpiricalNormalization(nn.Identity):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def update(self, value):
        return None


rsl_rl = types.ModuleType("rsl_rl")
networks = types.ModuleType("rsl_rl.networks")
networks.MLP = MLP
networks.EmpiricalNormalization = EmpiricalNormalization
rsl_rl.networks = networks
sys.modules["rsl_rl"] = rsl_rl
sys.modules["rsl_rl.networks"] = networks

from sugar_rl.utils.online_patch_tactile_actor_critic import (
    OnlinePatchTactileActorCritic,
)


SCALES = [1.0, 10.0, 10000.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]


def model_and_obs(batch: int = 2):
    obs = {
        "policy": torch.zeros(batch, 504),
        "online_patch_tactile_history": torch.zeros(batch, 1944),
        "critic": torch.zeros(batch, 890),
        "teacher": torch.zeros(batch, 890),
    }
    groups = {
        "policy": ["policy", "online_patch_tactile_history"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    model = OnlinePatchTactileActorCritic(
        obs,
        groups,
        29,
        patch_channel_scales=SCALES,
    )
    return model, obs


def released_tracker_state():
    actor = MLP(510, 29, [512, 256, 128], "elu")
    critic = MLP(890, 1, [512, 256, 128], "elu")
    state = {f"actor.{key}": value for key, value in actor.state_dict().items()}
    state.update(
        {f"critic.{key}": value for key, value in critic.state_dict().items()}
    )
    state["std"] = torch.ones(29)
    return state


def test_serious_actor_contract_and_live_forward():
    torch.manual_seed(14)
    model, obs = model_and_obs()
    assert model.actor[0].in_features == 504 + 128 == 632
    assert model.actor[0].out_features == 512
    assert model.actor[2].out_features == 256
    assert model.actor[4].out_features == 128
    obs["online_patch_tactile_history"][:, 0] = 1.0
    action = model.act_inference(obs)
    value = model.evaluate(obs)
    assert action.shape == (2, 29)
    assert value.shape == (2, 1)
    assert torch.isfinite(action).all() and torch.isfinite(value).all()


def test_released_tracker_warm_start_is_exact_for_zero_patch():
    torch.manual_seed(15)
    model, _ = model_and_obs()
    report = model.load_sugar_warm_start(released_tracker_state())
    assert report["target_actor_input_width"] == 632
    assert report["zero_patch_embedding_abs_max"] == 0.0
    assert report["actor_zero_patch_max_abs_error"] <= 1.0e-6
    assert report["critic_max_abs_error"] <= 1.0e-6
    assert report["actor_receives_excluded_source_values"] is False


def test_full_student_finetune_keeps_serious_components_trainable():
    model, _ = model_and_obs()
    report = model.configure_tactile_actor_finetune()
    assert report["mode"] == "full_sugar_student_with_anatomical_patch_transformer"
    assert all(parameter.requires_grad for parameter in model.actor.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.actor_tactile_encoder.parameters()
    )
