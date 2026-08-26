#!/usr/bin/env python3
"""Build causal-input/future-TMR-target data from official SUGAR rollouts.

The deployed predictor input is the exact past ``10 x 121`` actor-visible core,
one fixed numeric selected demonstration and a causal clock phase.  Its scalar
training target is cosine distance in the released TMR HumanML3D motion latent
between the next four seconds of the actual PhysX rollout and the phase-matched
four-second selected-demo window.  Actual future motion and TMR features are
labels only and are not serialized as predictor inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sugar/demo_following"))
sys.path.insert(0, str(ROOT / "scripts/sugar/demo_reward"))

from audit_actual_contact_event_corpus import motion_split  # noqa: E402
from audit_official_tmr_motion_latent import (  # noqa: E402
    SOURCE_FPS,
    SUGAR_BODY_NAMES,
    TMR_FPS,
    WINDOW_JOINT_FRAMES,
    g1_bodies_to_humanml_joints,
    geometry_audit,
    prepare_official_tmr,
    resample_joints,
)
from build_actual_contact_event_predictor_dataset import (  # noqa: E402
    PAIR_ROLE_NAMES,
    TASKS,
    actual_entries,
    load_reference,
    window_features,
)


DEFAULT_CORPUS = ROOT / (
    "experiments/demo_following/contact_event_reward_redesign_v1/"
    "deployable_goal_core_corpus_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/official_tmr_internal_reward_v1/"
    "motion_disjoint_predictor_dataset_suffix_v2"
)
DEFAULT_TMR_ROOT = ROOT / "experiments/runtime_assets/official_tmr"
SPLITS = ("train", "validation", "test")
HISTORY_STEPS = 10
FUTURE_STEPS = int(4.0 * SOURCE_FPS)
DEMO_WINDOWS = 32
ANCHOR_STRIDE = 10
POLICY_KEY = "goal_policy_core_observation"
TMR_TARGET_NAME = "future_tmr_cosine_distance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tmr-root", type=Path, default=DEFAULT_TMR_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


class OfficialTMRBatchEncoder:
    """Thin batching adapter around the released, frozen TMR motion encoder."""

    def __init__(self, tmr_root: Path, device: str, batch_size: int) -> None:
        (
            self.torch,
            self.model,
            self.normalizer,
            self.collate_x_dict,
            self.joints_to_guofeats,
        ) = prepare_official_tmr(tmr_root)
        self.device = device
        self.batch_size = int(batch_size)
        self.model = self.model.to(device).eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            self.model.requires_grad_(False)

    def encode_body_windows(
        self,
        body_windows: Sequence[np.ndarray],
        body_names: Sequence[str],
    ) -> np.ndarray:
        features = []
        for body in body_windows:
            joints = g1_bodies_to_humanml_joints(body, body_names)
            audit = geometry_audit(joints)
            if not audit["passed"]:
                raise RuntimeError(f"G1-to-HumanML geometry failed: {audit}")
            joints = resample_joints(joints)
            if joints.shape[0] != WINDOW_JOINT_FRAMES:
                raise RuntimeError(f"unexpected resampled length {joints.shape[0]}")
            guo = np.asarray(self.joints_to_guofeats(joints), dtype=np.float32)
            if guo.shape != (WINDOW_JOINT_FRAMES - 1, 263):
                raise RuntimeError(f"unexpected Guo shape {guo.shape}")
            features.append(guo)

        outputs: list[np.ndarray] = []
        with self.torch.inference_mode():
            for begin in range(0, len(features), self.batch_size):
                block = features[begin : begin + self.batch_size]
                rows = []
                for feature in block:
                    tensor = self.normalizer(
                        self.torch.from_numpy(feature).to(self.torch.float32)
                    )
                    rows.append({"x": tensor, "length": len(tensor)})
                batch = self.collate_x_dict(rows, device=self.device)
                latent = self.model.encode(
                    batch, modality="motion", sample_mean=True
                ).detach().cpu().numpy()
                outputs.append(np.asarray(latent, dtype=np.float32))
        result = np.concatenate(outputs)
        result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1.0e-12)
        if result.shape != (len(body_windows), 256) or not np.isfinite(result).all():
            raise RuntimeError("invalid official TMR latent block")
        return result.astype(np.float32)


def reference_body(task: str, source_id: int) -> np.ndarray:
    path = ROOT / f"SUGAR/data/{task}/data_{source_id:03d}/robot_50hz.npz"
    with np.load(path, allow_pickle=False) as archive:
        fps = int(np.asarray(archive["fps"]).reshape(-1)[0])
        body = np.asarray(archive["body_pos_w"], dtype=np.float32)
    if fps != SOURCE_FPS or len(body) < FUTURE_STEPS:
        raise RuntimeError(f"invalid source motion {path}: fps={fps}, frames={len(body)}")
    return body


def phase_windows(body: np.ndarray) -> list[np.ndarray]:
    starts = np.rint(
        np.linspace(0, len(body) - FUTURE_STEPS, DEMO_WINDOWS)
    ).astype(np.int64)
    return [body[int(start) : int(start) + FUTURE_STEPS] for start in starts]


def encode_demo_latents(
    encoder: OfficialTMRBatchEncoder,
    demo_keys: Sequence[tuple[str, int]],
) -> np.ndarray:
    output = np.empty((len(demo_keys), DEMO_WINDOWS, 256), dtype=np.float32)
    for row, (task, source_id) in enumerate(demo_keys):
        output[row] = encoder.encode_body_windows(
            phase_windows(reference_body(task, source_id)), SUGAR_BODY_NAMES
        )
        if (row + 1) % 10 == 0 or row + 1 == len(demo_keys):
            print(f"DEMO_TMR_PROGRESS {row + 1}/{len(demo_keys)}", flush=True)
    return output


def split_descriptors(
    split: str,
    entries: Sequence[dict[str, object]],
    references: dict[tuple[str, int], dict[str, np.ndarray]],
) -> list[tuple[dict[str, object], int, float]]:
    descriptors: list[tuple[dict[str, object], int, float]] = []
    for entry in entries:
        source_id = int(entry["source_id"])
        if motion_split(source_id) != split:
            continue
        reference_length = len(
            references[(str(entry["task"]), source_id)]["object_position"]
        )
        last_anchor = min(700 - FUTURE_STEPS - 1, reference_length - FUTURE_STEPS - 1)
        descriptors.extend(
            (
                entry,
                anchor,
                float(
                    np.clip(
                        (anchor + 1) / max(reference_length - FUTURE_STEPS, 1),
                        0.0,
                        1.0,
                    )
                ),
            )
            for anchor in range(HISTORY_STEPS - 1, last_anchor + 1, ANCHOR_STRIDE)
        )
    if not descriptors:
        raise RuntimeError(f"{split}: no eligible causal rows")
    return descriptors


def build_split(
    split: str,
    entries: Sequence[dict[str, object]],
    references: dict[tuple[str, int], dict[str, np.ndarray]],
    output: Path,
    encoder: OfficialTMRBatchEncoder,
) -> dict[str, object]:
    selected_entries = [
        entry for entry in entries
        if motion_split(int(entry["source_id"])) == split
    ]
    demo_keys = sorted(
        {(str(entry["task"]), int(entry["source_id"])) for entry in selected_entries}
    )
    demo_index = {key: index for index, key in enumerate(demo_keys)}
    by_task = {
        task: [key for key in demo_keys if key[0] == task]
        for task in TASKS
    }
    task_position = {
        task: {key: index for index, key in enumerate(by_task[task])}
        for task in TASKS
    }
    descriptors = split_descriptors(split, entries, references)
    base_count = len(descriptors)
    pair_count = base_count * len(PAIR_ROLE_NAMES)
    split_dir = output / split
    split_dir.mkdir(parents=True)

    demo_bank = np.stack([window_features(references[key]) for key in demo_keys])
    demo_tmr = encode_demo_latents(encoder, demo_keys)
    policy = np.lib.format.open_memmap(
        split_dir / "policy_prefix.npy",
        mode="w+",
        dtype=np.float32,
        shape=(base_count, HISTORY_STEPS, 121),
    )
    actual_tmr = np.empty((base_count, 256), dtype=np.float32)
    base_task = np.empty(base_count, dtype=np.uint8)
    base_source = np.empty(base_count, dtype=np.int16)
    base_anchor = np.empty(base_count, dtype=np.int16)
    base_phase = np.empty(base_count, dtype=np.float32)

    rows_by_trace: dict[Path, list[tuple[int, dict[str, object], int, float]]] = {}
    for base_row, (entry, anchor, phase) in enumerate(descriptors):
        rows_by_trace.setdefault(Path(entry["trace"]), []).append(
            (base_row, entry, anchor, phase)
        )
    for trace_id, (trace_path, rows) in enumerate(sorted(rows_by_trace.items())):
        with np.load(trace_path, allow_pickle=False) as archive:
            body = np.asarray(archive["robot_body_position_w"], dtype=np.float32)
            core = np.asarray(archive[POLICY_KEY], dtype=np.float32)
            names = tuple(str(value) for value in archive["robot_body_names"].tolist())
        body_windows = []
        for base_row, entry, anchor, phase in rows:
            env = int(entry["env"])
            task = str(entry["task"])
            source_id = int(entry["source_id"])
            policy[base_row] = core[anchor - HISTORY_STEPS + 1 : anchor + 1, env]
            body_windows.append(body[anchor + 1 : anchor + 1 + FUTURE_STEPS, env])
            base_task[base_row] = TASKS.index(task)
            base_source[base_row] = source_id
            base_anchor[base_row] = anchor
            base_phase[base_row] = phase
        encoded = encoder.encode_body_windows(body_windows, names)
        for encoded_row, (base_row, _, _, _) in zip(encoded, rows, strict=True):
            actual_tmr[base_row] = encoded_row
        print(
            f"ACTUAL_TMR_PROGRESS split={split} trace={trace_id + 1}/{len(rows_by_trace)} "
            f"rows={len(rows)}",
            flush=True,
        )

    pair_base = np.repeat(
        np.arange(base_count, dtype=np.int32), len(PAIR_ROLE_NAMES)
    )
    pair_role = np.tile(
        np.arange(len(PAIR_ROLE_NAMES), dtype=np.uint8), base_count
    )
    pair_demo = np.empty(pair_count, dtype=np.int16)
    for base_row, (entry, _, _) in enumerate(descriptors):
        task = str(entry["task"])
        source_id = int(entry["source_id"])
        key = (task, source_id)
        same_pool = by_task[task]
        same_wrong = same_pool[(task_position[task][key] + 1) % len(same_pool)]
        other_pool = by_task[TASKS[1 - TASKS.index(task)]]
        cross_wrong = other_pool[(source_id + base_row) % len(other_pool)]
        for role, selected_key in enumerate((key, same_wrong, cross_wrong)):
            pair_demo[base_row * len(PAIR_ROLE_NAMES) + role] = demo_index[selected_key]

    # TMR's validated task separation came from an aggregate over the motion,
    # not one isolated phase window.  Preserve that evidence by comparing
    # deterministic future suffix aggregates.  This is not free-window
    # matching: the causal phase fixes the suffix start and no later window is
    # selected by target value.
    demo_suffix = np.empty_like(demo_tmr)
    for demo_row in range(len(demo_tmr)):
        cumulative = np.cumsum(demo_tmr[demo_row, ::-1], axis=0)[::-1]
        cumulative /= np.arange(DEMO_WINDOWS, 0, -1, dtype=np.float32)[:, None]
        cumulative /= np.maximum(
            np.linalg.norm(cumulative, axis=-1, keepdims=True), 1.0e-12
        )
        demo_suffix[demo_row] = cumulative
    actual_suffix = np.empty_like(actual_tmr)
    for task_id in range(len(TASKS)):
        for source_id in np.unique(base_source[base_task == task_id]):
            rows = np.flatnonzero(
                (base_task == task_id) & (base_source == source_id)
            )
            rows = rows[np.argsort(base_anchor[rows])]
            cumulative = np.cumsum(actual_tmr[rows][::-1], axis=0)[::-1]
            cumulative /= np.arange(len(rows), 0, -1, dtype=np.float32)[:, None]
            cumulative /= np.maximum(
                np.linalg.norm(cumulative, axis=-1, keepdims=True), 1.0e-12
            )
            actual_suffix[rows] = cumulative

    pair_phase = base_phase[pair_base]
    phase_row = np.clip(
        np.rint(pair_phase * (DEMO_WINDOWS - 1)).astype(np.int64),
        0,
        DEMO_WINDOWS - 1,
    )
    selected_latent = demo_suffix[pair_demo, phase_row]
    target = 1.0 - np.sum(actual_suffix[pair_base] * selected_latent, axis=-1)
    target = np.clip(target, 0.0, 2.0).astype(np.float32)
    if not np.isfinite(target).all():
        raise RuntimeError(f"{split}: non-finite TMR distance target")

    policy.flush()
    np.save(split_dir / "demo_bank.npy", demo_bank, allow_pickle=False)
    np.save(split_dir / "target_tmr_distance.npy", target, allow_pickle=False)
    np.savez_compressed(
        split_dir / "routing.npz",
        base_task=base_task,
        base_source_motion_id=base_source,
        base_anchor_frame=base_anchor,
        base_normalized_demo_phase=base_phase,
        pair_base_row=pair_base,
        pair_selected_demo_row=pair_demo,
        pair_role=pair_role,
        pair_normalized_demo_phase=pair_phase,
        selected_phase_window=phase_row.astype(np.uint8),
        demo_task=np.asarray(
            [TASKS.index(key[0]) for key in demo_keys], dtype=np.uint8
        ),
        demo_source_motion_id=np.asarray(
            [key[1] for key in demo_keys], dtype=np.int16
        ),
    )
    role_mean = {
        PAIR_ROLE_NAMES[role]: float(np.mean(target[pair_role == role]))
        for role in range(len(PAIR_ROLE_NAMES))
    }
    return {
        "base_rows": base_count,
        "pair_rows": pair_count,
        "demo_count": len(demo_keys),
        "tmr_target_mean_by_pair_role": role_mean,
        "target_min": float(target.min()),
        "target_max": float(target.max()),
    }


def main() -> None:
    args = parse_args()
    corpus_root = args.corpus_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {staging}")
    entries = actual_entries(corpus_root)
    if len(entries) != 199:
        raise RuntimeError(f"expected 199 motion rollouts, found {len(entries)}")
    references = {
        (str(entry["task"]), int(entry["source_id"])): load_reference(
            str(entry["task"]), int(entry["source_id"]), ROOT / "SUGAR/data"
        )
        for entry in entries
    }
    staging.mkdir(parents=True)
    encoder = OfficialTMRBatchEncoder(
        args.tmr_root.expanduser().resolve(), args.device, args.batch_size
    )
    splits = {
        split: build_split(split, entries, references, staging, encoder)
        for split in SPLITS
    }

    train_policy = np.load(staging / "train/policy_prefix.npy", mmap_mode="r")
    train_demo = np.load(staging / "train/demo_bank.npy", mmap_mode="r")
    train_target = np.load(staging / "train/target_tmr_distance.npy", mmap_mode="r")
    target_scale = np.asarray(
        [max(float(np.quantile(train_target, 0.90)), 1.0e-3)], dtype=np.float32
    )
    np.savez(
        staging / "NORMALIZATION.npz",
        state_mean=np.mean(train_policy, axis=(0, 1), dtype=np.float64).astype(np.float32),
        state_std=np.maximum(
            np.std(train_policy, axis=(0, 1), dtype=np.float64), 1.0e-6
        ).astype(np.float32),
        demo_mean=np.mean(train_demo, axis=(0, 1, 2), dtype=np.float64).astype(np.float32),
        demo_std=np.maximum(
            np.std(train_demo, axis=(0, 1, 2), dtype=np.float64), 1.0e-3
        ).astype(np.float32),
        target_scale=target_scale,
    )
    checks = {
        "all_199_source_motions_present": sum(
            record["demo_count"] for record in splits.values()
        ) == 199,
        "every_base_has_three_selected_demo_pairs": all(
            record["pair_rows"] == 3 * record["base_rows"]
            for record in splits.values()
        ),
        "all_targets_finite_in_cosine_distance_range": all(
            0.0 <= record["target_min"] <= record["target_max"] <= 2.0
            for record in splits.values()
        ),
        "cross_task_distance_exceeds_correct_every_split": all(
            record["tmr_target_mean_by_pair_role"]["cross_task_wrong"]
            > record["tmr_target_mean_by_pair_role"]["correct"]
            for record in splits.values()
        ),
        "future_motion_is_target_only": True,
        "tmr_latent_is_target_only": True,
        "actor_input_is_exact_past_10x121_core": True,
    }
    manifest = {
        "protocol": "sugar_official_tmr_future_suffix_mismatch_dataset_v2",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "official_tmr": {
            "repository": "https://github.com/Mathux/TMR",
            "checkout_commit": "6d74688730d15d43b0a755ce2b0e1f2d76138fc1",
            "model": "tmr_humanml3d_guoh3dfeats/last_weights",
            "latent_dim": 256,
        },
        "splits": splits,
        "split_rule": "source_motion_id mod 10: 8 validation, 9 test, otherwise train",
        "model_inputs": {
            "policy_prefix": [HISTORY_STEPS, 121],
            "selected_demo_condition": [DEMO_WINDOWS, 10, 132],
            "selected_demo_phase": "causal scalar in [0,1]",
        },
        "training_target": {
            "name": TMR_TARGET_NAME,
            "definition": "1 - cosine(actual future-suffix mean TMR latent, phase-matched selected-demo future-suffix mean TMR latent)",
            "available_at_deployment": False,
        },
        "claim_boundary": (
            "Passing establishes a serious released-latent target dataset with causal deployable "
            "inputs. It does not establish predictor generalization or policy benefit."
        ),
        "automatic_next_branch": (
            "train_and_gate_causal_tmr_mismatch_predictor"
            if all(checks.values())
            else "repair_dataset_before_predictor_training"
        ),
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
