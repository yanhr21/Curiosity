"""Nearest/future gradient interaction on fixed actual TRAIN batches."""
from __future__ import annotations

import json
import os
import socket
import torch

from .data import OUTPUT, write_json
from .model import FutureSupervisionAdapter
from .train_matched import load_split, batch_forward


def cosine_summary(a, b):
    device = next(x.device for x in a if x is not None)
    aa = torch.zeros((), device=device, dtype=torch.float64)
    bb, dot = aa.clone(), aa.clone()
    for x, y in zip(a, b):
        if x is not None:
            aa += x.double().square().sum()
        if y is not None:
            bb += y.double().square().sum()
        if x is not None and y is not None:
            dot += (x.double() * y.double()).sum()
    return {"nearest_norm": float(aa.sqrt()), "future_norm": float(bb.sqrt()),
            "cosine": float(dot / (aa * bb).sqrt().clamp_min(1e-30))}


def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID") or socket.gethostname().startswith(("login", "mgmtserver")):
        raise RuntimeError("Run inside the retained compute step")
    run = OUTPUT.parent / "matched_future_supervision"
    readback = json.loads((run / "READBACK.json").read_text())
    if readback["next_action"] != "read_only_nearest_vs_future_gradient_conflict_probe_on_actual_TRAIN_batches":
        raise RuntimeError("Endpoint directs a different diagnostic")
    torch.set_num_threads(8)
    device = torch.device("cuda:0")
    data = load_split("train", device)
    model = FutureSupervisionAdapter(device, detach_auxiliary=False).to(device)
    endpoint = torch.load(run / "aux_attached/endpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(endpoint["model"], strict=True)
    del endpoint
    model.eval()
    named = list(model.base.named_parameters())
    parameters = [value for _, value in named]
    groups = {"all_base": list(range(len(named)))}
    for name in ("demo_projection", "state_projection", "transformer", "readout", "target_heads"):
        groups[name] = [i for i, (key, _) in enumerate(named) if key.startswith(name + ".")]
    generator = torch.Generator().manual_seed(271913)
    batches = []
    for index in range(8):
        base = torch.randperm(len(data["policy"]), generator=generator)[:32].to(device)
        for conditions, condition_ids in (("original_three", (0, 1, 2)), ("including_reversed", (0, 1, 2, 3))):
            rows = (base[:, None] * 4 + torch.as_tensor(condition_ids, device=device)).reshape(-1)
            output = batch_forward(model, data, rows)
            losses = model.losses(output, data["target"][rows // 4, rows % 4])
            near = torch.autograd.grad(losses[0], parameters, retain_graph=True, allow_unused=True)
            far = torch.autograd.grad(losses[1:].mean(), parameters, allow_unused=True)
            batches.append({"batch": index, "conditions": conditions, "base_rows": base.cpu().tolist(),
                            "horizon_nll": losses.detach().tolist(),
                            "groups": {group: cosine_summary([near[i] for i in ids], [far[i] for i in ids])
                                       for group, ids in groups.items()}})
    summaries = {}
    for conditions in ("original_three", "including_reversed"):
        rows = [x for x in batches if x["conditions"] == conditions]
        summaries[conditions] = {
            group: {"negative_cosine_batches": sum(x["groups"][group]["cosine"] < 0 for x in rows),
                    "mean_cosine": sum(x["groups"][group]["cosine"] for x in rows) / len(rows)}
            for group in groups
        }
    result = {"execution_completed": True, "optimizer_updates": 0, "checkpoint_step": 4640,
              "fixed_training_base_batches": 8, "summaries": summaries, "batches": batches,
              "scope": "Local gradient interaction at the retained endpoint, not proof of a unique failure cause or policy benefit"}
    write_json(run / "GRADIENT_CONFLICT_PROBE.json", result)
    print(json.dumps({"summaries": summaries, "optimizer_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
