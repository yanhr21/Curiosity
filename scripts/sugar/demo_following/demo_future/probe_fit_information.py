"""Separate TRAIN fit from held-out error and privileged present-state evidence."""
from __future__ import annotations

import json
import numpy as np
import torch

from .data import OUTPUT, write_json
from .model import FutureSupervisionAdapter
from .train_matched import load_split, batch_forward


@torch.no_grad()
def main():
    run = OUTPUT.parent / "matched_corrected_canonical"
    result = json.loads((run / "RESULT.json").read_text())
    if not result["execution_completed"]:
        raise RuntimeError("Canonical matched endpoints are incomplete")
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    from pathlib import Path
    dataset = Path(protocol["dataset"])
    device = torch.device("cuda")
    torch.set_num_threads(8)
    model = FutureSupervisionAdapter(device, detach_auxiliary=False, normalization_path=protocol["normalization_path"]).to(device)
    saved = torch.load(run / "aux_attached/endpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    del saved
    model.eval()
    data = load_split("train", device, dataset)
    n = len(data["route"]["base_task"])
    prediction = np.empty((n, 4, 4, 13), dtype=np.float32)
    for start in range(0, n * 4, 128):
        rows = torch.arange(start, min(start + 128, n * 4), device=device)
        value = batch_forward(model, data, rows)["mean"]
        if not torch.isfinite(value).all():
            raise RuntimeError("Nonfinite TRAIN prediction")
        prediction.reshape(n * 4, 4, 13)[start:start + len(rows)] = value.cpu().numpy()
    np.save(run / "TRAIN_attached_predictions.npy", prediction, allow_pickle=False)
    del model, data
    with np.load(protocol["normalization_path"], allow_pickle=False) as z:
        scale = z["target_scale"]
    channels = [0, 1, 2, 3, 4, 5, 6, 7, 12]
    groups = {"geometry": [0, 1, 2, 3], "contact": [4, 5, 6, 7], "regime": [8], "all_nine": list(range(9))}
    summary = {}
    for split in ("train", "validation", "test"):
        if split != "train":
            with np.load(run / "aux_attached" / f"{split}_predictions.npz", allow_pickle=False) as z:
                prediction = z["full"]
        with np.load(dataset / split / "routing.npz", allow_pickle=False) as z:
            task, source = z["base_task"], z["base_source_motion_id"]
        truth = np.log1p(np.load(dataset / split / "target_mismatch.npy")[..., channels] / scale[channels])
        held = np.log1p(np.load(dataset / split / "current_hold_nine_channels.npy") / scale[channels])
        summary[split] = {}
        for task_id in (0, 1):
            mask = task == task_id
            macro = lambda value: np.mean([value[mask & (source == s)].mean(axis=0) for s in np.unique(source[mask])], axis=0)
            row = {}
            for name, ids in groups.items():
                errors = {}
                for label, value in (("model", prediction[..., channels]), ("privileged_current_hold", held)):
                    error = np.abs(value[:, :3, :, ids] - truth[:, :3, :, ids]).mean(axis=(1, 3))
                    errors[label] = macro(error)
                row[name] = {**{k: v.tolist() for k, v in errors.items()},
                             "model_over_current_hold": (errors["model"] / np.maximum(errors["privileged_current_hold"], 1e-12)).tolist()}
            summary[split][str(task_id)] = row
    report = {"execution_completed": True, "optimizer_updates": 0, "training_samples_scored": n * 4,
              "splits": summary, "scope": "Fixed completed attached endpoint; same native normalization and nine targets. Current hold uses extra measured body/contact states, so outperforming the model is evidence about information/fit, not a valid121-D deployed-policy baseline. TRAIN versus held-out motions differ in context; this diagnostic does not uniquely identify a cause.",
              "next_action": "Audit whether the current geometry/contact information can be recovered from the unchanged causal121-D interface using official robot kinematics and observed fields; keep any unavailable signal evaluation-only. Do not extend the three failed matched settings."}
    write_json(run / "FIT_INFORMATION_PROBE.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
