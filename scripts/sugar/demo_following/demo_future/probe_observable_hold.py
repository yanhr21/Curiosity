"""Causal121-D geometry-only hold control through official robot FK.

Nominal joint calibration makes this approximate; it does not assert the failed
1e-3 precise reconstruction check passed. No contact/regime fields are inferred.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import numpy as np
import torch

from .data import OUTPUT, PARENT, CONTINUOUS_SLICES, write_json
from .probe_observable_geometry import ObservableGeometryAdapter


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-mode", choices=("known-episode-clock", "current-phase-only"), default="known-episode-clock")
    args = parser.parse_args()
    suffix = "_current_phase_only" if args.clock_mode == "current-phase-only" else ""
    run = OUTPUT.parent / "matched_corrected_canonical"
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    dataset = Path(protocol["dataset"])
    device = torch.device("cuda")
    torch.set_num_threads(8)
    adapter = ObservableGeometryAdapter(device)
    with np.load(protocol["normalization_path"], allow_pickle=False) as z:
        scale = z["target_scale"][:4]
    summary = {}
    for split in ("train", "validation", "test"):
        with np.load(dataset / split / "routing.npz", allow_pickle=False) as z:
            route = {k: z[k] for k in z.files}
        policy = np.load(PARENT / split / "policy_prefix.npy", mmap_mode="r")
        current = policy[route["parent_base_row"], -1]
        bank = torch.as_tensor(np.load(dataset / split / "demo_bank.npy"), device=device)
        n = len(current)
        held = np.empty((n, 4, 4, 4), dtype=np.float32)
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            geometry = adapter(torch.as_tensor(current[start:stop], device=device))
            selected = torch.as_tensor(route["selected_demo"][start:stop], device=device, dtype=torch.long)
            phases = route["horizon_phase"][start:stop]
            if args.clock_mode == "current-phase-only":
                phases = np.repeat(phases[:, :1], 4, axis=1)
            alignment = torch.as_tensor(np.rint(phases * np.float32(31)), device=device, dtype=torch.long)
            chosen = bank[selected[:, :, None], alignment[:, None, :], :, :120]
            square = (chosen - geometry[:, None, None, None, :]).square()
            held[start:stop] = torch.stack([square[..., a:b].mean(dim=(-2, -1)) for a, b in CONTINUOUS_SLICES], dim=-1).cpu().numpy()
        np.save(run / f"{split}_observable_geometry_hold{suffix}.npy", held, allow_pickle=False)
        truth = np.log1p(np.load(dataset / split / "target_mismatch.npy")[..., :4] / scale)
        held = np.log1p(held / scale)
        if split == "train":
            prediction = np.load(run / "TRAIN_attached_predictions.npy")[..., :4]
        else:
            with np.load(run / "aux_attached" / f"{split}_predictions.npz", allow_pickle=False) as z:
                prediction = z["full"][..., :4]
        privileged = np.log1p(np.load(dataset / split / "current_hold_nine_channels.npy")[..., :4] / scale)
        task, source = route["base_task"], route["base_source_motion_id"]
        summary[split] = {}
        for task_id in (0, 1):
            mask = task == task_id
            macro = lambda value: np.mean([value[mask & (source == s)].mean(axis=0) for s in np.unique(source[mask])], axis=0)
            scores = {label: macro(np.abs(value[:, :3] - truth[:, :3]).mean(axis=(1, 3)))
                      for label, value in (("model", prediction), ("observable_geometry_hold", held), ("privileged_geometry_hold", privileged))}
            summary[split][str(task_id)] = {**{k: v.tolist() for k, v in scores.items()},
                                           "observable_over_model": (scores["observable_geometry_hold"] / np.maximum(scores["model"], 1e-12)).tolist(),
                                           "observable_over_privileged": (scores["observable_geometry_hold"] / np.maximum(scores["privileged_geometry_hold"], 1e-12)).tolist()}
    all_better = all(max(row["observable_over_model"]) < 1 for v in summary.values() for row in v.values())
    report = {"execution_completed": True, "optimizer_updates": 0, "splits": summary,
              "clock_mode": args.clock_mode,
              "clock_fairness": "current-phase-only uses exactly the model's scalar current phase for every horizon, with no episode duration or future phase. known-episode-clock also uses the configured source duration to advance phase; this is causal runtime metadata but was not explicitly supplied to the neural predictor. The h0 comparison uses identical phase in either mode.",
              "observable_hold_better_than_model_all_splits_tasks_horizons": all_better,
              "input": "Only last121-D causal observation, known selected numeric reference/clock and fixed official robot constants. Uses no actual body pose, force, contact, reset-relative regime or future state to construct the hold control.",
              "calibration_limit": "Official source randomizes default joint positions by[-0.01,0.01]rad at startup. Original199 traces do not archive those calibration values. Nominal-calibration FK fails the1e-3 precise reconstruction audit; this control explicitly remains approximate. No unknown calibration is inferred from target poses.",
              "scope": "Four continuous geometry mismatch channels only, same normalization/conditions/motion weighting as endpoint. This is an analytical causal control using official kinematics, not a learned replacement model, synthetic training corpus or physical success result.",
              "next_action": "If the observable control outperforms the network, preserve the recoverable current-geometry path explicitly and test future residual prediction with the same full predictor; do not continue unanchored score-head training. Contact/regime information requires a separate causal interface audit."}
    write_json(OUTPUT.parent / ("OBSERVABLE_GEOMETRY_HOLD_PROBE" + suffix.upper() + ".json"), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
