"""Project yaw-invariant task-label adapter using official SMP heading math.

This is not the official 216-D SMP representation. Each 10-frame task window
uses its own first-frame pelvis heading. All 132 existing feature channels and
13 targets retain their capacities; the four continuous mismatch meanings change.
"""
from __future__ import annotations

import json
import sys
import numpy as np
import torch

from .data import ROOT, OUTPUT, PARENT, CORPUS, TASKS, actual_entries, actual_future, window_features, reverse_reference, mismatch, write_json
from .repair_reference_roles import REPAIRED, source_names, corrected_reference
from build_actual_contact_event_predictor_dataset import quaternion_wxyz_to_rotation6d

sys.path.insert(0, str(ROOT / "MimicKit/mimickit"))
from util import torch_util

CANONICAL = OUTPUT.parent / "dataset_heading_local_v3"
CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7, 12)


def canonicalize(continuous, anchor_wxyz):
    value = np.asarray(continuous, dtype=np.float32)
    q = torch.as_tensor(np.asarray(anchor_wxyz, dtype=np.float32)[..., [1, 2, 3, 0]].copy())
    q = q / q.norm(dim=-1, keepdim=True)
    heading = torch_util.calc_heading_quat_inv(q)
    vectors = torch.as_tensor(value.reshape(*value.shape[:-1], 40, 3).copy())
    heading = heading[..., None, None, :].expand(*vectors.shape[:-1], 4)
    return torch_util.quat_rotate(heading, vectors).numpy().reshape(value.shape)


def reference_bank(task, source, names):
    reference = corrected_reference(task, source, names)
    with np.load(ROOT / "SUGAR/data" / task / f"data_{source:03d}/robot_50hz.npz", allow_pickle=False) as z:
        pelvis_q = z["body_quat_w"][:len(reference["contact"]), names.index("pelvis")]
    result = []
    for ref, q in ((reference, pelvis_q), (reverse_reference(reference), pelvis_q[::-1])):
        bank = window_features(ref)
        starts = np.rint(np.linspace(0, len(q) - 10, 32)).astype(np.int64)
        bank[..., :120] = canonicalize(bank[..., :120], q[starts])
        result.append(bank)
    return result


def main():
    if not json.loads((REPAIRED / "MANIFEST.json").read_text())["execution_completed"]:
        raise RuntimeError("Reference role repair is incomplete")
    CANONICAL.mkdir(exist_ok=False)
    names = source_names()
    entries = {(TASKS.index(e["task"]), e["source_id"]): e for e in actual_entries(CORPUS)}
    summaries, invariance = {}, []
    for split in ("train", "validation", "test"):
        directory = CANONICAL / split
        directory.mkdir()
        with np.load(REPAIRED / split / "routing.npz", allow_pickle=False) as z:
            route = {k: z[k] for k in z.files}
        repaired_bank = np.load(REPAIRED / split / "demo_bank.npy")
        half = len(repaired_bank) // 2
        banks = [reference_bank(TASKS[int(t)], int(s), names) for t, s in
                 zip(route["demo_task"][:half], route["demo_source_motion_id"][:half])]
        bank = np.stack([v[0] for v in banks] + [v[1] for v in banks])
        if not np.array_equal(bank[..., 120:], repaired_bank[..., 120:]):
            raise RuntimeError("Canonicalization changed discrete reference labels")
        old = np.load(REPAIRED / split / "target_mismatch.npy")
        targets, held_targets = np.empty_like(old), np.empty_like(old)
        cache, trace = None, None
        checked = set()
        for row, (task, source, anchor) in enumerate(zip(route["base_task"], route["base_source_motion_id"], route["base_anchor_frame"])):
            entry = entries[(int(task), int(source))]
            if entry["trace"] != cache:
                cache = entry["trace"]
                with np.load(cache, allow_pickle=False) as z:
                    trace = {k: z[k] for k in ("object_root_state_w", "robot_body_position_w", "robot_root_state_w", "contact", "contact_event_remaining_frames", "motion_regime")}
            env, anchor = entry["env"], int(anchor)
            obj = trace["object_root_state_w"][anchor, env]
            body = trace["robot_body_position_w"][anchor, env] - obj[:3]
            continuous = np.concatenate((body.reshape(-1), np.zeros(3, dtype=np.float32), quaternion_wxyz_to_rotation6d(obj[3:7]), obj[7:13]))
            held = {"continuous": canonicalize(np.repeat(continuous[None], 10, axis=0), trace["robot_root_state_w"][anchor, env, 3:7]),
                    "contact": np.repeat(trace["contact"][anchor, env][None].astype(np.float32), 10, axis=0),
                    "duration": np.zeros((10, 4), dtype=np.float32),
                    "regime": np.repeat(trace["motion_regime"][anchor, env], 10)}
            for horizon, offset in enumerate((0, 10, 30, 60)):
                actual = actual_future(trace, env, anchor + offset)
                q = trace["robot_root_state_w"][anchor + offset + 1, env, 3:7]
                raw = actual["continuous"]
                actual["continuous"] = canonicalize(raw, q)
                if int(task) not in checked and horizon == 0:
                    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
                    yaw = torch.tensor([0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=torch.float32)
                    rotated_q = torch_util.quat_mul(yaw, torch.as_tensor(q[[1, 2, 3, 0]].copy())).numpy()[[3, 0, 1, 2]]
                    rotated = (raw.reshape(10, 40, 3) @ rotation.T).reshape(10, 120)
                    error = float(np.abs(canonicalize(rotated, rotated_q) - actual["continuous"]).max())
                    if error >= 1e-5:
                        raise RuntimeError("Canonical target geometry is not yaw-invariant")
                    invariance.append({"split": split, "task": TASKS[int(task)], "source": int(source), "anchor": anchor, "max_absolute_change": error})
                    checked.add(int(task))
                alignment = int(np.rint(route["horizon_phase"][row, horizon] * np.float32(31)))
                demo = bank[route["selected_demo"][row], alignment]
                targets[row, :, horizon] = mismatch(actual, demo)
                held_targets[row, :, horizon] = mismatch(held, demo)
        if not np.array_equal(targets[..., 4:], old[..., 4:]):
            raise RuntimeError("Canonicalization changed discrete mismatch labels")
        np.save(directory / "demo_bank.npy", bank, allow_pickle=False)
        np.save(directory / "target_mismatch.npy", targets, allow_pickle=False)
        np.save(directory / "current_hold_nine_channels.npy", held_targets[..., list(CHANNELS)], allow_pickle=False)
        np.savez(directory / "routing.npz", **route)
        if split == "train":
            with np.load(PARENT / "NORMALIZATION.npz", allow_pickle=False) as z:
                normalization = {k: z[k] for k in z.files}
            normalization["demo_mean"] = bank.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
            normalization["demo_std"] = np.maximum(bank.std(axis=(0, 1, 2), dtype=np.float64), 1e-6).astype(np.float32)
            scale = np.maximum(np.median(targets.reshape(-1, 13), axis=0), 1e-6)
            scale[4:] = np.maximum(scale[4:], 0.1)
            normalization["target_scale"] = scale.astype(np.float32)
            np.savez(CANONICAL / "NORMALIZATION.npz", **normalization)
        scores = {}
        for task in (0, 1):
            mask = route["base_task"] == task
            truth = np.log1p(targets[..., list(CHANNELS)] / scale[list(CHANNELS)])
            prediction = np.log1p(held_targets[..., list(CHANNELS)] / scale[list(CHANNELS)])
            errors = np.abs(prediction[:, :3] - truth[:, :3]).mean(axis=(1, 3))
            gap = np.abs((prediction[:, 1] - prediction[:, 0]) - (truth[:, 1] - truth[:, 0])).mean(axis=-1)
            macro = lambda v: np.mean([v[mask & (route["base_source_motion_id"] == s)].mean(axis=0)
                                      for s in np.unique(route["base_source_motion_id"][mask])], axis=0).tolist()
            scores[TASKS[task]] = {"current_hold_per_horizon_mae": macro(errors), "current_hold_per_horizon_gap_mae": macro(gap)}
        summaries[split] = {"base_histories": len(targets), "discrete_targets_exactly_preserved": True, "current_hold": scores}
        print(json.dumps({"split_complete": split}), flush=True)
    result = {"execution_completed": True, "optimizer_updates": 0, "geometry_checks": invariance,
              "splits": summaries, "reference_repair": str(REPAIRED.relative_to(ROOT)),
              "coordinate_contract": "Independent first-frame pelvis heading for each reference/actual ten-frame window. Rotate all body-relative-object, within-window object displacement, rotation6d columns and world velocity vectors with official MimicKit calc_heading_quat_inv/quat_rotate. This project task-feature adapter is not official final-torso-heading216-D SMP features.",
              "semantics_limit": "Invariant to common global yaw/translation; relative local motion and events retained, absolute world route/heading adherence is not measured by this label.",
              "causality": "Future actual states and their window heading are labels only. Current hold uses only trace[anchor] and known numeric reference/clock; all remaining-duration channels excluded. Extra current body poses make it privileged, not a deployed121-D baseline.",
              "normalization": "New demo/target normalization uses TRAIN only; original TRAIN-derived state normalization retained. Target scale is per-channel median across TRAIN conditions/horizons with1e-6 floor and0.1 for event channels. Old calibration does not transfer.",
              "next_action": "Run matched full-predictor adaptation with corrected canonical inputs/labels and new normalization, retaining detached versus attached future-gradient control; evaluate input ablations, same-task contrasts and current-state baseline before any policy launch."}
    write_json(CANONICAL / "MANIFEST.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
