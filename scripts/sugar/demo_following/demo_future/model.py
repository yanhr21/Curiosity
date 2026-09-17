"""Multi-horizon task heads on the intact, pretrained project predictor.

This is an auxiliary-supervision adapter, not a replacement motion/world model
and not an implementation of official Zero-WAM IFP modules.
"""
from __future__ import annotations

import copy
import torch
from torch import nn

from .audit_frozen import CHECKPOINT, PARENT, model_from_normalization
from .data import TARGET_NAMES


class FutureSupervisionAdapter(nn.Module):
    def __init__(self, device, *, detach_auxiliary, normalization_path=None):
        super().__init__()
        self.base = model_from_normalization(PARENT / "NORMALIZATION.npz", device)
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self.base.load_state_dict(saved["model_state_dict"], strict=True)
        if normalization_path is not None:
            import numpy as np
            with np.load(normalization_path, allow_pickle=False) as z:
                for name in ("state_mean", "state_std", "demo_mean", "demo_std", "target_scale"):
                    buffer = getattr(self.base, name)
                    value = torch.as_tensor(z[name], device=buffer.device)
                    if value.shape != buffer.shape or not torch.isfinite(value).all():
                        raise ValueError(f"Invalid new normalization: {name}")
                    buffer.copy_(value)
        self.future_heads = nn.ModuleList([copy.deepcopy(self.base.target_heads) for _ in range(3)])
        self.detach_auxiliary = bool(detach_auxiliary)

    def forward(self, *, policy_prefix, selected_demo_condition, selected_demo_phase, zero_demo=False):
        result = self.base(policy_prefix=policy_prefix, selected_demo_condition=selected_demo_condition,
                           selected_demo_phase=selected_demo_phase, zero_demo=zero_demo)
        representation = result["representation"]
        auxiliary = representation.detach() if self.detach_auxiliary else representation
        means, variances = [result["mean_log1p_scaled"]], [result["log_variance_log1p_scaled"]]
        for heads in self.future_heads:
            raw = torch.stack([heads[name](auxiliary) for name in TARGET_NAMES], dim=1)
            means.append(raw[..., 0])
            variances.append(raw[..., 1].clamp(-10, 8))
        return {"mean": torch.stack(means, dim=1), "log_variance": torch.stack(variances, dim=1),
                "representation": representation}

    def losses(self, output, target):
        encoded = torch.log1p(target / self.base.target_scale)
        nll = self.base.gaussian_nll(output["mean"], output["log_variance"], encoded)
        return nll.mean(dim=(0, 2))
