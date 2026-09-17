"""Current-geometry skip adapter on the existing full causal predictor.

Official robot FK supplies geometry only; the complete project Transformer and
all event/future heads remain. This is not an official Zero-WAM implementation.
"""
from __future__ import annotations

import torch
from torch import nn

from .model import FutureSupervisionAdapter
from .data import TARGET_NAMES, CONTINUOUS_SLICES, OFFSETS
from .probe_observable_geometry import ObservableGeometryAdapter


class GeometryResidualAdapter(FutureSupervisionAdapter):
    uses_clock_rate = True

    def __init__(self, device, *, normalization_path, initialization_endpoint, add_geometry_baseline):
        super().__init__(device, detach_auxiliary=False, normalization_path=normalization_path)
        saved = torch.load(initialization_endpoint, map_location="cpu", weights_only=False)
        self.load_state_dict(saved["model"], strict=True)
        self.clock_rate_projection = nn.Linear(1, self.base.d_model).to(device)
        nn.init.zeros_(self.clock_rate_projection.weight)
        nn.init.zeros_(self.clock_rate_projection.bias)
        # Same mean-row initialization in both arms; event/variance rows retained.
        with torch.no_grad():
            for heads in (self.base.target_heads, *self.future_heads):
                for name in TARGET_NAMES[:4]:
                    heads[name].weight[0].zero_()
                    heads[name].bias[0].zero_()
        self.geometry = ObservableGeometryAdapter(device)
        self.add_geometry_baseline = bool(add_geometry_baseline)

    @torch.no_grad()
    def analytical_geometry(self, policy_prefix, selected_demo_condition, selected_demo_phase, selected_demo_phase_rate, zero_demo):
        geometry = self.geometry(policy_prefix[:, -1])
        demo = selected_demo_condition
        if zero_demo:
            demo = self.base.demo_mean.expand_as(demo)
        # Rate is fixed causal episode-clock metadata, not predicted/future state.
        # Recover integer clock ticks to reproduce original reference rounding.
        rate = selected_demo_phase_rate.double()
        ticks = torch.round(selected_demo_phase.double() / rate)
        offsets = torch.tensor(OFFSETS, device=rate.device, dtype=torch.float64)
        phases = ((ticks[:, None] + offsets[None]) * rate[:, None]).float()
        alignment = torch.round(phases * 31).long().clamp(0, 31)
        chosen = demo[torch.arange(len(demo), device=demo.device)[:, None], alignment, :, :120]
        square = (chosen - geometry[:, None, None, :]).square()
        raw = torch.stack([square[..., a:b].mean(dim=(-2, -1)) for a, b in CONTINUOUS_SLICES], dim=-1)
        return torch.log1p(raw / self.base.target_scale[:4])

    def forward(self, *, policy_prefix, selected_demo_condition, selected_demo_phase, selected_demo_phase_rate, zero_demo=False):
        if selected_demo_phase_rate.shape != selected_demo_phase.shape or not torch.isfinite(selected_demo_phase_rate).all() or torch.any(selected_demo_phase_rate <= 0):
            raise ValueError("Invalid causal episode-clock rate")
        base = self.base(policy_prefix=policy_prefix, selected_demo_condition=selected_demo_condition,
                         selected_demo_phase=selected_demo_phase, zero_demo=zero_demo)
        # Rate is in normalized phase per second, identical in both arms.
        representation = base["representation"] + self.clock_rate_projection(selected_demo_phase_rate.float()[:, None] / 0.02)
        raw = torch.stack([torch.stack([heads[name](representation) for name in TARGET_NAMES], dim=1)
                           for heads in (self.base.target_heads, *self.future_heads)], dim=1)
        mean, variance = raw[..., 0], raw[..., 1].clamp(-10, 8)
        baseline = self.analytical_geometry(policy_prefix, selected_demo_condition, selected_demo_phase,
                                            selected_demo_phase_rate, zero_demo)
        if self.add_geometry_baseline:
            mean = torch.cat((mean[..., :4] + baseline, mean[..., 4:]), dim=-1)
        return {"mean": mean, "log_variance": variance, "representation": representation,
                "analytical_geometry": baseline}
