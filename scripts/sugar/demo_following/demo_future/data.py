"""Outcome-grounded multi-horizon labels; no learned or simulated substitute.

Reuses the existing PhysX corpus, numeric reference builder and 13 mismatch
definitions. The fourth reference condition is a time-reversal diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/sugar/demo_reward"))
from build_actual_contact_event_predictor_dataset import (
    TASKS, CONTINUOUS_SLICES, TARGET_NAMES, actual_entries, actual_future,
    event_remaining, load_reference, window_features,
)

BASE = ROOT / "experiments/demo_following/contact_event_reward_redesign_v1"
PARENT = BASE / "phase_aware_goal_core_dataset_v1"
CORPUS = BASE / "deployable_goal_core_corpus_v1"
OUTPUT = ROOT / "experiments/demo_following/demo_future_smp_v1/dataset"
OFFSETS = (0, 10, 30, 60)
CONDITIONS = ("correct", "same_task_alternate", "cross_task_alternate", "reversed")


def reverse_reference(reference):
    """Reverse the full reference before windowing, recomputing event duration.

    Velocity changes sign; relative within-window displacements are recomputed
    by the original window builder. This does not assert physical feasibility.
    Regime is reversed as the original lifted/moving annotation, preserving its
    original floor reference instead of rebasing height at the last frame.
    """
    result = {key: np.array(value[::-1], copy=True) for key, value in reference.items()}
    result["object_velocity"] *= -1
    result["duration"] = event_remaining(result["contact"])
    return result


def mismatch(actual, chosen):
    square = np.square(chosen[..., :120] - actual["continuous"])
    continuous = [square[..., a:b].mean(axis=(-2, -1)) for a, b in CONTINUOUS_SLICES]
    return np.concatenate((
        np.stack(continuous, axis=-1),
        np.abs(chosen[..., 120:124] - actual["contact"]).mean(axis=-2),
        np.abs(chosen[..., 124:128] - actual["duration"]).mean(axis=-2),
        np.mean(np.argmax(chosen[..., 128:132], axis=-1) != actual["regime"], axis=-1)[..., None],
    ), axis=-1).astype(np.float32)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def build(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = actual_entries(CORPUS)
    entry_map = {(TASKS.index(e["task"]), e["source_id"]): e for e in entries}
    if len(entries) != 199 or len(entry_map) != 199:
        raise ValueError("Expected the unchanged 199-motion PhysX corpus")
    summary = {}
    split_keys = {}
    for split in ("train", "validation", "test"):
        parent = PARENT / split
        with np.load(parent / "routing.npz", allow_pickle=False) as archive:
            route = {key: archive[key] for key in archive.files}
        bank = np.load(parent / "demo_bank.npy", allow_pickle=False)
        keys = list(zip(route["demo_task"].tolist(), route["demo_source_motion_id"].tolist()))
        split_keys[split] = set(keys)
        references = {key: load_reference(TASKS[key[0]], key[1], ROOT / "SUGAR/data") for key in keys}
        # Reuse the exact trained demo bank; independently verify its source.
        for row, key in enumerate(keys):
            if not np.array_equal(bank[row], window_features(references[key])):
                raise ValueError(f"{split}: original demo bank no longer matches reference {key}")
        reverse_bank = np.stack([window_features(reverse_reference(references[key])) for key in keys])
        demo_bank = np.concatenate((bank, reverse_bank))
        lengths = np.array([len(references[(int(t), int(s))]["object_position"])
                            for t, s in zip(route["base_task"], route["base_source_motion_id"])])
        keep = np.flatnonzero(route["base_anchor_frame"] + max(OFFSETS) + 10 < np.minimum(700, lengths))
        n = len(keep)
        if not n:
            raise ValueError(f"No common multi-horizon support in {split}")
        parent_pairs = np.arange(len(route["pair_base_row"])).reshape(-1, 3)[keep]
        if not np.array_equal(route["pair_base_row"][parent_pairs], np.repeat(keep[:, None], 3, axis=1)):
            raise ValueError("Parent pair routing differs from the established three conditions")
        selected = np.column_stack((route["pair_selected_demo_row"][parent_pairs],
                                    route["pair_selected_demo_row"][parent_pairs[:, 0]] + len(keys)))
        targets = np.empty((n, 4, len(OFFSETS), len(TARGET_NAMES)), dtype=np.float32)
        phases = np.empty((n, len(OFFSETS)), dtype=np.float32)
        current_trace = None
        trace = None
        for row, old in enumerate(keep):
            key = (int(route["base_task"][old]), int(route["base_source_motion_id"][old]))
            entry = entry_map[key]
            if entry["trace"] != current_trace:
                current_trace = entry["trace"]
                with np.load(current_trace, allow_pickle=False) as z:
                    names = ("object_root_state_w", "robot_body_position_w", "contact",
                             "contact_event_remaining_frames", "motion_regime", "done",
                             "reset_before_frame", "source_motion_id", "control_dt_s")
                    trace = {name: z[name] for name in names}
            anchor, env = int(route["base_anchor_frame"][old]), int(entry["env"])
            span = slice(anchor - 9, anchor + max(OFFSETS) + 11)
            if trace["done"][span, env].any() or trace["reset_before_frame"][span, env].any():
                raise ValueError("A history/target spans a reset or terminal transition")
            if not np.all(trace["source_motion_id"][span, env] == key[1]):
                raise ValueError("Source changed inside a history/target span")
            if not np.allclose(trace["control_dt_s"], 0.02, rtol=0, atol=1e-8):
                raise ValueError("Expected original 50 Hz PhysX control clock")
            for horizon, offset in enumerate(OFFSETS):
                # Same deterministic episode clock as the parent; no future
                # matching, optimized alignment, or future actor input.
                phase = np.float32((anchor + offset + 1) / (lengths[old] - 10))
                phases[row, horizon] = phase
                alignment = int(np.rint(phase * np.float32(31)))
                actual = actual_future(trace, env, anchor + offset)
                targets[row, :, horizon] = mismatch(actual, demo_bank[selected[row], alignment])
            if row % 2000 == 0:
                print(json.dumps({"split": split, "rows": row, "total": n}), flush=True)
        old_targets = np.load(parent / "target_mismatch.npy", mmap_mode="r")[parent_pairs]
        h0_error = np.max(np.abs(targets[:, :3, 0] - old_targets))
        if not np.allclose(targets[:, :3, 0], old_targets, rtol=1e-5, atol=2e-6):
            raise ValueError(f"Original h0 labels changed: max difference {h0_error}")
        if not np.array_equal(phases[:, 0], route["base_normalized_demo_phase"][keep]):
            raise ValueError("Original causal clock changed")
        if not np.isfinite(targets).all() or np.any(targets < 0):
            raise ValueError("Nonfinite or negative mismatch target")
        out = output / split
        out.mkdir()
        np.save(out / "target_mismatch.npy", targets, allow_pickle=False)
        np.save(out / "demo_bank.npy", demo_bank, allow_pickle=False)
        np.savez(out / "routing.npz", parent_base_row=keep, selected_demo=selected,
                 horizon_phase=phases, base_task=route["base_task"][keep],
                 base_source_motion_id=route["base_source_motion_id"][keep],
                 base_anchor_frame=route["base_anchor_frame"][keep],
                 demo_task=np.tile(route["demo_task"], 2),
                 demo_source_motion_id=np.tile(route["demo_source_motion_id"], 2))
        summary[split] = {"motion_count": len(keys), "base_rows": n,
                          "parent_rows": len(lengths), "h0_max_absolute_reproduction_error": float(h0_error),
                          "target_shape": list(targets.shape)}
    disjoint = not any(split_keys[a] & split_keys[b] for a, b in
                       (("train", "validation"), ("train", "test"), ("validation", "test")))
    if not disjoint:
        raise ValueError("Motion split overlap")
    result = {"protocol": "sugar_demo_future_labels_v1", "execution_completed": True,
              "passed": True, "parent_dataset": str(PARENT), "corpus": str(CORPUS),
              "future_window_steps": 10, "future_start_offsets": list(OFFSETS),
              "future_end_seconds": [0.02 * (offset + 10) for offset in OFFSETS],
              "condition_names": list(CONDITIONS), "target_names": list(TARGET_NAMES),
              "motion_disjoint": disjoint, "splits": summary,
              "scope": "Actual PhysX future mismatch labels, not counterfactual robot outcomes or policy success",
              "causal_clock": "Original source duration fixed for the episode; current phase inherited exactly. Future phase used only for labels.",
              "reversed_reference": "Full timeline reversed; velocity sign and remaining contact duration recomputed. Diagnostic, not a feasibility assertion.",
              "actions": "Original executed actions preserved in corpus; no joint-position substitution"}
    write_json(output / "MANIFEST.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
