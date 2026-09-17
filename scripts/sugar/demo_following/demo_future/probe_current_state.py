"""Privileged current-state hold diagnostic, never a deployed world model.

Uses current measured body/object/contact state only, plus the prescribed demo
and clock. Remaining-event-duration targets are excluded because their current
value contains future information. This is an information/interface diagnostic.
"""
from __future__ import annotations

import json
import numpy as np

from .data import ROOT, OUTPUT, PARENT, CORPUS, TASKS, actual_entries, mismatch, write_json
from build_actual_contact_event_predictor_dataset import quaternion_wxyz_to_rotation6d

CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 12)


def main():
    entries = {(TASKS.index(e["task"]), e["source_id"]): e for e in actual_entries(CORPUS)}
    with np.load(PARENT / "NORMALIZATION.npz", allow_pickle=False) as z:
        scale = z["target_scale"]
    results = {}
    for split in ("validation", "test"):
        directory = OUTPUT / split
        with np.load(directory / "routing.npz", allow_pickle=False) as z:
            route = {k: z[k] for k in z.files}
        bank = np.load(directory / "demo_bank.npy")
        truth = np.log1p(np.load(directory / "target_mismatch.npy") / scale)
        baseline = np.empty_like(truth)
        cache = None
        trace = None
        for row, (task, source, anchor) in enumerate(zip(route["base_task"], route["base_source_motion_id"], route["base_anchor_frame"])):
            entry = entries[(int(task), int(source))]
            if cache != entry["trace"]:
                cache = entry["trace"]
                with np.load(cache, allow_pickle=False) as z:
                    trace = {k: z[k] for k in ("object_root_state_w", "robot_body_position_w", "contact", "motion_regime")}
            env = entry["env"]
            obj = trace["object_root_state_w"][anchor, env]
            body = trace["robot_body_position_w"][anchor, env] - obj[:3]
            continuous = np.concatenate((body.reshape(-1), np.zeros(3, dtype=np.float32),
                                         quaternion_wxyz_to_rotation6d(obj[3:7]), obj[7:13]))
            held = {"continuous": np.repeat(continuous[None], 10, axis=0),
                    "contact": np.repeat(trace["contact"][anchor, env][None].astype(np.float32), 10, axis=0),
                    "duration": np.zeros((10, 4), dtype=np.float32),
                    "regime": np.repeat(trace["motion_regime"][anchor, env], 10)}
            for horizon in range(4):
                alignment = int(np.rint(route["horizon_phase"][row, horizon] * np.float32(31)))
                raw = mismatch(held, bank[route["selected_demo"][row], alignment])
                baseline[row, :, horizon] = np.log1p(raw / scale)
        models = {"privileged_current_state_hold": baseline}
        for run, arm in (("matched_future_supervision", "aux_attached"),
                         ("matched_same_task_gap", "pair_supervised")):
            with np.load(OUTPUT.parent / run / arm / f"{split}_predictions.npz", allow_pickle=False) as z:
                models[arm] = z["full"]
        task_results = {}
        task_ids, source_ids = route["base_task"], route["base_source_motion_id"]
        for task in (0, 1):
            task_mask = task_ids == task
            sources = np.unique(source_ids[task_mask])
            scores = {}
            selected_truth = truth[..., list(CHANNELS)]
            truth_gap = selected_truth[:, 1] - selected_truth[:, 0]
            for name, prediction in models.items():
                pred = prediction[..., list(CHANNELS)]
                errors = np.abs(pred[:, :3] - selected_truth[:, :3]).mean(axis=(1, 3))
                gap_errors = np.abs((pred[:, 1] - pred[:, 0]) - truth_gap).mean(axis=-1)
                macro = lambda x: np.mean([x[task_mask & (source_ids == sid)].mean(axis=0) for sid in sources], axis=0).tolist()
                scores[name] = {"per_horizon_mae": macro(errors), "per_horizon_same_task_gap_mae": macro(gap_errors)}
            task_results[str(task)] = scores
        results[split] = task_results
        np.save(OUTPUT.parent / f"{split}_privileged_current_hold.npy", baseline[..., list(CHANNELS)], allow_pickle=False)
    result = {"execution_completed": True, "optimizer_updates": 0, "channels": list(CHANNELS),
              "splits": results, "scope": "Privileged current measured body/object/contact hold control; not deployable from the 121-D actor prefix as implemented, not a learned world model or physical rollout",
              "excluded": "All four remaining-event-duration targets, to prevent future-duration leakage",
              "causality": "Only trace[anchor] read for control values; known demo and declared clock supply reference future. No actual future state is used by the hold control."}
    write_json(OUTPUT.parent / "CURRENT_STATE_INFORMATION_PROBE.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
