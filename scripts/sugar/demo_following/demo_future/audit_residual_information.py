"""Frozen residual fit, loss-metric and reference-specific information audit."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from .data import OUTPUT, write_json, TARGET_NAMES
from .train_matched import make_model, load_split, batch_forward


@torch.no_grad()
def main():
    run = OUTPUT.parent / "matched_geometry_residual"
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    endpoint = json.loads((run / "RESULT.json").read_text())
    if not endpoint["execution_completed"]:
        raise RuntimeError("Residual endpoint incomplete")
    dataset = Path(protocol["dataset"])
    torch.set_num_threads(8)
    device = torch.device("cuda")
    model = make_model(protocol, "residual_geometry", device).eval()
    checkpoint = torch.load(run / "residual_geometry/endpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    data = load_split("train", device, dataset, True)
    n = len(data["route"]["base_task"])
    learned = np.empty((n, 4, 4, 4), dtype=np.float32)
    for start in range(0, n * 4, 128):
        rows = torch.arange(start, min(start + 128, n * 4), device=device)
        base, condition = rows // 4, rows % 4
        output = model.base(policy_prefix=data["policy"][base],
                            selected_demo_condition=data["bank"][data["selected"][base, condition]],
                            selected_demo_phase=data["phase"][base])
        representation = output["representation"] + model.clock_rate_projection(data["phase_rate"][base].float()[:, None] / 0.02)
        correction = torch.stack([torch.stack([heads[t](representation)[:, 0] for t in TARGET_NAMES[:4]], dim=-1)
                                  for heads in (model.base.target_heads, *model.future_heads)], dim=1)
        if start == 0:
            full = batch_forward(model, data, rows)
            if not torch.allclose(correction, full["mean"][..., :4] - full["analytical_geometry"], rtol=1e-5, atol=2e-6):
                raise RuntimeError("Direct frozen correction readback differs from complete forward")
        if not torch.isfinite(correction).all():
            raise RuntimeError("Nonfinite correction")
        learned.reshape(n * 4, 4, 4)[start:start + len(rows)] = correction.cpu().numpy()
    np.save(run / "TRAIN_geometry_correction.npy", learned, allow_pickle=False)
    scale = model.base.target_scale[:4].cpu().numpy()
    del model, data
    details = {}
    for split in ("train", "validation", "test"):
        with np.load(dataset / split / "routing.npz", allow_pickle=False) as z:
            route = {k: z[k] for k in z.files}
        truth = np.log1p(np.load(dataset / split / "target_mismatch.npy")[..., :4] / scale)
        if split == "train":
            analytical = np.log1p(np.load(OUTPUT.parent / f"matched_corrected_canonical/{split}_observable_geometry_hold.npy") / scale)
            true_residual = truth - analytical
            train_mean = true_residual[:, :3].mean(axis=(0, 1))
            train_median = np.median(true_residual[:, :3], axis=(0, 1))
            np.savez(run / "TRAIN_CONSTANT_RESIDUALS.npz", mean=train_mean, median=train_median)
        else:
            analytical = np.load(run / "residual_geometry" / f"{split}_analytical_geometry.npy")
            with np.load(run / "residual_geometry" / f"{split}_predictions.npz", allow_pickle=False) as z:
                learned = z["full"][..., :4] - analytical
            true_residual = truth - analytical
        task, source = route["base_task"], route["base_source_motion_id"]
        details[split] = {}
        for task_id in (0, 1):
            mask = task == task_id
            macro = lambda v: np.mean([v[mask & (source == s)].mean(axis=0) for s in np.unique(source[mask])], axis=0)
            metrics = {}
            zero_mae = macro(np.abs(true_residual[:, :3]).mean(axis=(1, 3)))
            zero_mse = macro(np.square(true_residual[:, :3]).mean(axis=(1, 3)))
            for label, prediction in (("learned", learned), ("train_constant_mean", train_mean), ("train_constant_median", train_median)):
                error = prediction - true_residual
                metrics[label] = {"mae_over_zero": (macro(np.abs(error[:, :3]).mean(axis=(1, 3))) / np.maximum(zero_mae, 1e-12)).tolist(),
                                  "mse_over_zero": (macro(np.square(error[:, :3]).mean(axis=(1, 3))) / np.maximum(zero_mse, 1e-12)).tolist()}
            common = learned[:, :2].mean(axis=1)
            total_energy = macro(np.square(learned[:, :2]).mean(axis=(1, 3)))
            common_energy = macro(np.square(common).mean(axis=-1))
            p_gap = learned[:, 1] - learned[:, 0]
            t_gap = true_residual[:, 1] - true_residual[:, 0]
            gap_energy = macro(np.square(p_gap).mean(axis=-1)) / np.maximum(macro(np.square(t_gap).mean(axis=-1)), 1e-12)
            correlations = []
            for horizon in range(4):
                p = p_gap[mask, horizon].astype(np.float64).flatten()
                t = t_gap[mask, horizon].astype(np.float64).flatten()
                p, t = p - p.mean(), t - t.mean()
                denominator = np.linalg.norm(p) * np.linalg.norm(t)
                correlations.append(float(np.dot(p, t) / denominator) if denominator > 1e-12 else None)
            details[split][str(task_id)] = {"metrics": metrics,
                                          "same_task_common_correction_energy_fraction": (common_energy / np.maximum(total_energy, 1e-12)).tolist(),
                                          "predicted_over_true_same_task_residual_gap_energy": gap_energy.tolist(),
                                          "same_task_residual_gap_pooled_correlation": correlations}
        print(json.dumps({"split_complete": split}), flush=True)
    common_dominates = all(min(row["same_task_common_correction_energy_fraction"]) > .9 for split in details.values() for row in split.values())
    report = {"execution_completed": True, "optimizer_updates": 0, "matched_endpoint_passed": endpoint["passed"],
              "details": details, "reference_common_energy_above90pct_every_group": common_dominates,
              "constant_control_scope": "Only16 global TRAIN-derived mean/median residual values per control, with no test fit, task identity or extra observation; diagnostic statistics, not a replacement model. Metrics use original three conditions and equal motion weights.",
              "data_contract": {"actual_futures_per_base_history_and_horizon": 1, "scoring_reference_conditions": 4,
                                "proof": "data.py creates one actual_future(trace,env,anchor+offset), then mismatch broadcasts it across four selected-demo windows. All conditions retain one original executed-action trajectory. Previous h0 and corrected-channel reproduction checks cover the materialized labels.",
                                "interpretation": "These are selected-reference compatibility labels. Alternate scoring demos do not supply counterfactual executed futures under alternate intended demos. No conditional-control/different-future claim is supported by this corpus construction."},
              "scope": "Frozen full-model information diagnostic; MAE versus MSE is reported without changing the original failed scientific decision. Correlation/constant-fit statistics do not establish dynamics or physical following.",
              "next_action": "Audit shared future-feature supervision and availability of matched original-history/alternate-demo actual futures before more scalar mismatch-head optimization; preserve the analytical selected-demo geometry control."}
    write_json(run / "RESIDUAL_INFORMATION_AUDIT.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
