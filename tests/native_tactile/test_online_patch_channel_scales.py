from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "plan15_fit_online_patch_channel_scales",
    ROOT / "scripts/sugar/native_tactile/fit_online_patch_channel_scales.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
fit_channel_scales = module.fit_channel_scales


def live_like_arrays():
    patch = np.zeros((8, 2, 27, 6), dtype=np.float32)
    slip = np.zeros((8, 2, 27, 3), dtype=np.float32)
    patch[:, :, :3, 0] = 1.0
    patch[:, :, :3, 1] = np.linspace(1.0, 8.0, 8)[:, None, None]
    patch[:, :, :3, 2] = np.linspace(100.0, 800.0, 8)[:, None, None]
    patch[:, :, :3, 3] = -2.0
    patch[:, :, :3, 4] = 4.0
    patch[:, :, :3, 5] = 0.75
    slip[-3:, :, :3, 0] = 0.6
    slip[-2:, :, :3, 1] = 1.0
    slip[-1:, :, :3, 2] = 1.0
    return patch, slip


def test_live_scale_fit_preserves_nine_patch_channels_and_shared_xy_scale():
    patch, slip = live_like_arrays()
    scales = fit_channel_scales(patch, slip, quantile=0.99)
    assert len(scales) == 9
    assert scales[0] == scales[7] == scales[8] == 1.0
    assert scales[3] == scales[4] == 4.0
    assert all(np.isfinite(scales)) and all(value > 0.0 for value in scales)


def test_scale_fit_rejects_missing_live_slip_signal():
    patch, slip = live_like_arrays()
    slip[..., 0] = 0.0
    with pytest.raises(ValueError, match="no nonzero samples"):
        fit_channel_scales(patch, slip)
