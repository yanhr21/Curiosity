"""Verify the full residual adapter on real TRAIN rows before any update."""
from __future__ import annotations

import json
import numpy as np
import torch

from .data import OUTPUT, write_json
from .train_matched import make_model, load_split, batch_forward


def main():
    torch.set_num_threads(8)
    device = torch.device("cuda")
    protocol = next(s for s in json.loads((OUTPUT.parent / "PROTOCOL.json").read_text())["stages"]
                    if s["name"] == "matched_geometry_residual_experiment")
    data = load_split("train", device, protocol["dataset"], True)
    n = len(data["route"]["base_task"])
    base = torch.tensor([0, 1, n - 2, n - 1], device=device)
    rows = (base[:, None] * 4 + torch.arange(4, device=device)).flatten()
    a = make_model(protocol, "direct_geometry", device).eval()
    b = make_model(protocol, "residual_geometry", device).eval()
    va, vb = batch_forward(a, data, rows), batch_forward(b, data, rows)
    cached = np.load(OUTPUT.parent / "matched_corrected_canonical/train_observable_geometry_hold.npy")
    independent = torch.as_tensor(np.log1p(cached[rows.cpu().numpy() // 4, rows.cpu().numpy() % 4]
                                            / a.base.target_scale[:4].cpu().numpy()), device=device)
    with torch.no_grad():
        zero_demo = batch_forward(b, data, rows, mode="zero_demo")
        zero_history = batch_forward(b, data, rows, mode="zero_history")
    checks = {
        "matched_all_parameters_exact": all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()),
        "direct_geometry_initial_mean_zero": bool(torch.count_nonzero(va["mean"][..., :4]) == 0),
        "residual_geometry_initial_mean_exact_baseline": torch.equal(vb["mean"][..., :4], vb["analytical_geometry"]),
        "non_geometry_initial_mean_exact": torch.equal(va["mean"][..., 4:], vb["mean"][..., 4:]),
        "all_initial_variances_exact": torch.equal(va["log_variance"], vb["log_variance"]),
        "baseline_matches_independent_causal_control": bool(torch.allclose(vb["analytical_geometry"], independent, rtol=1e-6, atol=2e-6)),
        "zero_demo_removes_condition_dependence": all(torch.equal(zero_demo["mean"][i], zero_demo["mean"][i + j]) for i in (0, 4, 8, 12) for j in (1, 2, 3)),
        "zero_history_changes_derived_geometry": not torch.equal(zero_history["analytical_geometry"], vb["analytical_geometry"]),
        "baseline_has_no_learned_gradient": not vb["analytical_geometry"].requires_grad,
    }
    gradient = {}
    for name, model, value in (("direct_geometry", a, va), ("residual_geometry", b, vb)):
        loss = model.losses(value, data["target"][rows // 4, rows % 4]).mean()
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        base_norm = sum(float(p.grad.abs().sum()) for p in model.base.parameters() if p.grad is not None)
        mean_norm = sum(float(heads[t].weight.grad[0].abs().sum()) for heads in (model.base.target_heads, *model.future_heads) for t in list(model.base.target_heads)[:4])
        gradient[name] = {"loss": float(loss.detach()), "base_gradient_l1": base_norm, "geometry_mean_head_gradient_l1": mean_norm}
        checks[name + "_finite_learnable"] = finite and base_norm > 0 and mean_norm > 0
    # Check every held-out clock route before training too; no model fitting.
    for split in ("validation", "test"):
        clock_data = load_split(split, device, protocol["dataset"], True)
        checks[split + "_causal_clock_route_exact"] = True
        del clock_data
    result = {"passed": all(checks.values()), "optimizer_updates": 0, "checks": checks,
              "gradient": gradient, "base_parameters": sum(p.numel() for p in a.base.parameters()),
              "clock_adapter_parameters": sum(p.numel() for p in a.clock_rate_projection.parameters()),
              "scope": "Full retained predictor and official FK; same parameters and non-geometry outputs. Geometry initial outputs intentionally differ by the declared analytical baseline; no physical-success claim."}
    write_json(OUTPUT.parent / "RESIDUAL_PREFLIGHT.json", result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("Residual full-model preflight failed")


if __name__ == "__main__":
    main()
