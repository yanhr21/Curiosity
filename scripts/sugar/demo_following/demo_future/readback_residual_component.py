"""Isolate learned geometry corrections from their analytical current baseline."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import OUTPUT, write_json


def main():
    run = OUTPUT.parent / "matched_geometry_residual"
    endpoint = json.loads((run / "RESULT.json").read_text())
    if not endpoint["execution_completed"]:
        raise RuntimeError("Residual endpoints incomplete")
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    dataset = Path(protocol["dataset"])
    with np.load(protocol["normalization_path"], allow_pickle=False) as z:
        scale = z["target_scale"]
    figures = run / "residual_component_figures"
    figures.mkdir(exist_ok=False)
    details, figure_records = {}, []
    for split in ("validation", "test"):
        with np.load(dataset / split / "routing.npz", allow_pickle=False) as z:
            route = {k: z[k] for k in z.files}
        with np.load(run / "residual_geometry" / f"{split}_predictions.npz", allow_pickle=False) as z:
            prediction = z["full"]
        analytical = np.load(run / "residual_geometry" / f"{split}_analytical_geometry.npy")
        truth = np.log1p(np.load(dataset / split / "target_mismatch.npy") / scale)
        ablated = np.concatenate((analytical, prediction[..., 4:]), axis=-1)
        task, source = route["base_task"], route["base_source_motion_id"]
        details[split] = {}
        for task_id, name in ((0, "CarryBox"), (1, "KickBox")):
            mask = task == task_id
            macro = lambda v: np.mean([v[mask & (source == s)].mean(axis=0) for s in np.unique(source[mask])], axis=0)
            scores = {}
            for label, values in (("full_residual", prediction), ("same_event_predictions_without_geometry_correction", ablated)):
                geometrical = np.abs(values[:, :3, :, :4] - truth[:, :3, :, :4]).mean(axis=(1, 3))
                gap = (values[:, 1] - values[:, 0]).mean(axis=-1)
                true_gap = (truth[:, 1] - truth[:, 0]).mean(axis=-1)
                geom_gap = (values[:, 1, :, :4] - values[:, 0, :, :4]).mean(axis=-1)
                true_geom_gap = (truth[:, 1, :, :4] - truth[:, 0, :, :4]).mean(axis=-1)
                scores[label] = {"geometry_mae": macro(geometrical), "all13_same_task_gap_mae": macro(np.abs(gap - true_gap)),
                                 "geometry_same_task_gap_mae": macro(np.abs(geom_gap - true_geom_gap))}
            full, control = scores.values()
            ratios = {k: (full[k] / np.maximum(control[k], 1e-12)).tolist() for k in full}
            details[split][name] = {"per_horizon": {k: {a: b.tolist() for a, b in v.items()} for k, v in scores.items()},
                                  "full_over_without_geometry_correction": ratios}
            if split == "validation":
                selected_source = int(np.min(source[mask]))
                selected = mask & (source == selected_source)
                time = route["base_anchor_frame"][selected] * 0.02
                fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
                for h, ax in enumerate(axes):
                    for values, label, color in ((truth[..., :4], "Actual future geometry gap", "#222222"),
                                                  (analytical, "Analytical current-geometry hold", "#bd7024"),
                                                  (prediction[..., :4], "Hold plus learned residual", "#2169a6")):
                        gap = (values[selected, 1, h] - values[selected, 0, h]).mean(axis=-1)
                        ax.plot(time, gap, label=label, color=color, linewidth=1.5)
                    ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8)
                    ax.set_ylabel(f"{(0.2,0.4,0.8,1.4)[h]:.1f}s\nGeometry gap")
                    ax.grid(alpha=0.15)
                axes[0].legend(fontsize=8)
                axes[-1].set_xlabel("Causal history endpoint (seconds)")
                fig.suptitle(f"{name} source{selected_source:03d}: selected-demo geometry contrast\n"
                             "Four geometry targets only; identical known episode clock; predictor diagnostic", fontsize=11)
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                path = figures / f"{name}_source{selected_source:03d}.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                figure_records.append({"task": name, "source": selected_source, "path": str(path)})
    report = {"execution_completed": True, "optimizer_updates": 0,
              "matched_endpoint_passed": endpoint["passed"], "details": details, "figures": figure_records,
              "scope": "Read-only component ablation of the exact same residual checkpoint. Event predictions unchanged; only learned geometry correction removed. Both use the same causal clock rate and nominal-calibration official FK. No altered endpoint threshold, new training, or physical-success claim.",
              "next_action": endpoint["next_action"]}
    write_json(run / "RESIDUAL_COMPONENT_READBACK.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "figures"}), flush=True)


if __name__ == "__main__":
    main()
