"""Real-checkpoint, real-data gradient-path verification; zero updates."""
from __future__ import annotations

import json
import os
import socket
import torch

from .model import FutureSupervisionAdapter
from .train_matched import load_split, batch_forward
from .data import OUTPUT, write_json


def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID") or socket.gethostname().startswith(("login", "mgmtserver")):
        raise RuntimeError("Run inside the retained compute step")
    torch.set_num_threads(8)
    device = torch.device("cuda:0")
    data = load_split("train", device)
    rows = torch.as_tensor([0, 1, 2, 3, 400, 401, 402, 403], device=device)
    control = FutureSupervisionAdapter(device, detach_auxiliary=True).to(device).eval()
    treatment = FutureSupervisionAdapter(device, detach_auxiliary=False).to(device).eval()
    exact_initial = all(torch.equal(v, treatment.state_dict()[k]) for k, v in control.state_dict().items())
    out_c = batch_forward(control, data, rows)
    out_t = batch_forward(treatment, data, rows)
    exact_output = torch.equal(out_c["mean"], out_t["mean"]) and torch.equal(out_c["log_variance"], out_t["log_variance"])
    direct = control.base(policy_prefix=data["policy"][rows // 4],
                          selected_demo_condition=data["bank"][data["selected"][rows // 4, rows % 4]],
                          selected_demo_phase=data["phase"][rows // 4])
    exact_main = torch.equal(direct["mean_log1p_scaled"], out_c["mean"][:, 0])
    target = data["target"][rows // 4, rows % 4]
    norms = {}
    for name, model, output in (("detached", control, out_c), ("attached", treatment, out_t)):
        loss = model.losses(output, target)[1:].mean()
        parameters = list(model.base.parameters()) + list(model.future_heads.parameters())
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        nbase = len(list(model.base.parameters()))
        def norm(values):
            return float(torch.sqrt(sum((g.double().square().sum() for g in values if g is not None),
                                        torch.zeros((), device=device, dtype=torch.float64))))
        norms[name] = {"base_auxiliary_gradient_norm": norm(gradients[:nbase]),
                       "head_auxiliary_gradient_norm": norm(gradients[nbase:])}
    # Check the nearest objective still supervises the unchanged base in control.
    nearest = batch_forward(control, data, rows)
    nearest_loss = control.losses(nearest, target)[0]
    nearest_grads = torch.autograd.grad(nearest_loss, list(control.base.parameters()), allow_unused=True)
    nearest_norm = float(torch.sqrt(sum((g.double().square().sum() for g in nearest_grads if g is not None),
                                       torch.zeros((), device=device, dtype=torch.float64))))
    checks = {"matched_initial_state_exact": exact_initial, "matched_forward_exact": exact_output,
              "original_main_forward_exact": exact_main,
              "control_auxiliary_trunk_gradient_zero": norms["detached"]["base_auxiliary_gradient_norm"] == 0,
              "treatment_auxiliary_trunk_gradient_positive": norms["attached"]["base_auxiliary_gradient_norm"] > 0,
              "both_auxiliary_heads_learnable": all(v["head_auxiliary_gradient_norm"] > 0 for v in norms.values()),
              "control_nearest_trunk_gradient_positive": nearest_norm > 0,
              "all_gradient_norms_finite": all(torch.isfinite(torch.tensor(v)) for value in norms.values() for v in value.values())}
    checks = {k: bool(v) for k, v in checks.items()}
    result = {"passed": all(checks.values()), "checks": checks, "gradient_norms": norms,
              "control_nearest_gradient_norm": nearest_norm, "optimizer_updates": 0,
              "base_parameters": sum(p.numel() for p in control.base.parameters()),
              "auxiliary_parameters": sum(p.numel() for p in control.future_heads.parameters()),
              "scope": "Full retained predictor and actual TRAIN cases; verifies adapter gradient path, not scientific improvement"}
    write_json(OUTPUT.parent / "PREFLIGHT.json", result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("Actual model gradient-path preflight failed")


if __name__ == "__main__":
    main()
