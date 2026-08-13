from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "SUGAR/source/sugar_rl/sugar_rl/utils/patch_tactile_encoder.py"
SPEC = importlib.util.spec_from_file_location("plan15_patch_encoder", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def encoder():
    return MODULE.AnatomicalPatchTactileEncoder(
        [1.0, 10.0, 10000.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_frozen_patch_transformer_contract():
    model = encoder()
    assert model.expected_flat_dim == 4 * 2 * 27 * 9 == 1944
    assert model.output_dim == 128
    assert len(model.transformer.layers) == 3
    assert model.transformer.layers[0].self_attn.num_heads == 4
    contract = model.architecture_contract()
    assert contract["policy_unit"] == "physical_anatomical_patch"
    assert contract["taxel_policy_dimension"] is False


def test_exact_zero_input_maps_bitwise_to_zero_in_train_and_eval():
    torch.manual_seed(7)
    model = encoder()
    zeros = torch.zeros(3, 1944)
    model.train()
    assert torch.equal(model(zeros), torch.zeros(3, 128))
    model.eval()
    assert torch.equal(model(zeros), torch.zeros(3, 128))


def test_live_patch_signal_produces_embedding_and_gradients():
    torch.manual_seed(8)
    model = encoder()
    tactile = torch.zeros(2, 4, 2, 27, 9)
    tactile[:, :, 0, 12, 0] = 1.0
    tactile[:, :, 0, 12, 1] = 2.0
    tactile[:, :, 0, 12, 2] = 4000.0
    tactile[:, :, 0, 12, 3] = 0.5
    tactile[:, :, 0, 12, 5] = 0.5
    output = model(tactile.reshape(2, -1))
    assert output.shape == (2, 128)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) > 0
    output.square().mean().backward()
    assert model.patch_projection.weight.grad is not None
    assert torch.count_nonzero(model.patch_projection.weight.grad) > 0
    assert model.transformer.layers[0].self_attn.in_proj_weight.grad is not None
    assert torch.count_nonzero(
        model.transformer.layers[0].self_attn.in_proj_weight.grad
    ) > 0


def test_channel_scales_are_frozen_state_not_trainable_parameters():
    model = encoder()
    assert "channel_scales" in dict(model.named_buffers())
    assert "channel_scales" not in dict(model.named_parameters())


def test_rejects_taxel_or_wrong_patch_geometry():
    model = encoder()
    with pytest.raises(ValueError, match="shape mismatch"):
        model(torch.zeros(1, 2 * 27 * 20 * 25))
    with pytest.raises(ValueError, match="geometry is frozen"):
        MODULE.AnatomicalPatchTactileEncoder(
            [1.0] * 9, patches_per_hand=26
        )
