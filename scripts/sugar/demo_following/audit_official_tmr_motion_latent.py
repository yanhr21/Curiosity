#!/usr/bin/env python3
"""Audit the released TMR motion latent on SUGAR demos and PhysX rollouts.

This is a representation feasibility test, not a policy reward experiment.  It
uses the released HumanML3D TMR motion encoder and its released normalization
statistics without fitting an encoder or classifier.  A deterministic geometry
adapter maps the G1 rigid-body centers to the 22-joint HumanML3D topology before
the official ``joints_to_guofeats`` conversion is called.

The decision rule is fixed before inference: cosine-nearest class prototypes
formed from motion-disjoint source demonstrations.  The held-out source motions
and official IsaacLab/PhysX rollouts are never used to fit the prototypes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TMR_ROOT = ROOT / "experiments/runtime_assets/official_tmr"
DEFAULT_OUTPUT = ROOT / "experiments/demo_following/official_tmr_semantic_gate_v1"
DEFAULT_ROUTER_ROOT = ROOT / (
    "experiments/demo_following/official_tracker_router_v1/seed161610/"
    "frozen_eval_joint_final"
)

SOURCE_FPS = 50
TMR_FPS = 20
WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 2.0
WINDOW_JOINT_FRAMES = int(WINDOW_SECONDS * TMR_FPS)
STRIDE_JOINT_FRAMES = int(STRIDE_SECONDS * TMR_FPS)

# The serialized robot_50hz.npz order is the same 35-body order recorded by
# the official runtime traces.  It differs from the old SMP schema around the
# fixed head/logo links, so this script owns and checks the observed order.
SUGAR_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "pelvis_contour_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "head_link",
    "left_shoulder_pitch_link",
    "logo_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_rubber_hand",
    "right_rubber_hand",
)

HUMANML_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

HUMANML_EDGES = (
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
)

TRAIN_IDS = tuple(range(0, 20))
TEST_IDS = tuple(range(20, 40))
NAMED_QUERIES = (("carry45", "CarryBox", 45), ("kick21", "KickBox", 21))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmr-root", type=Path, default=DEFAULT_TMR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--router-root", type=Path, default=DEFAULT_ROUTER_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _indices(names: Sequence[str]) -> dict[str, int]:
    if len(names) != len(set(names)):
        raise ValueError("body names must be unique")
    return {str(name): index for index, name in enumerate(names)}


def g1_bodies_to_humanml_joints(
    body_position_w: np.ndarray,
    body_names: Sequence[str],
) -> np.ndarray:
    """Map G1 body centers to an explicit, non-degenerate 22-joint skeleton.

    IsaacLab is Z-up.  TMR's HumanML3D converter is Y-up, so the final proper
    rotation is ``(x, y, z) -> (x, z, -y)``.  The G1 has no HumanML3D spine or
    neck chain.  Those joints are deterministic interpolants between the pelvis
    and shoulder midpoint; the head extends along that same measured axis.
    """

    body = np.asarray(body_position_w, dtype=np.float32)
    if body.ndim != 3 or body.shape[-1] != 3:
        raise ValueError(f"expected [T,B,3] body positions, got {body.shape}")
    if body.shape[1] != len(body_names):
        raise ValueError("body position/name count mismatch")
    ids = _indices(body_names)
    required = {
        "pelvis", "left_hip_roll_link", "right_hip_roll_link",
        "left_knee_link", "right_knee_link",
        "left_ankle_pitch_link", "right_ankle_pitch_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_shoulder_pitch_link", "right_shoulder_pitch_link",
        "left_shoulder_roll_link", "right_shoulder_roll_link",
        "left_elbow_link", "right_elbow_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    }
    missing = sorted(required.difference(ids))
    if missing:
        raise ValueError(f"missing G1 bodies: {missing}")

    out = np.empty((len(body), 22, 3), dtype=np.float32)
    pelvis = body[:, ids["pelvis"]]
    left_shoulder = body[:, ids["left_shoulder_roll_link"]]
    right_shoulder = body[:, ids["right_shoulder_roll_link"]]
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    torso_axis = shoulder_mid - pelvis
    torso_norm = np.linalg.norm(torso_axis, axis=-1, keepdims=True)
    if np.any(torso_norm < 0.1):
        raise ValueError("degenerate pelvis-to-shoulder axis")
    torso_unit = torso_axis / torso_norm

    out[:, 0] = pelvis
    out[:, 1] = body[:, ids["left_hip_roll_link"]]
    out[:, 2] = body[:, ids["right_hip_roll_link"]]
    out[:, 3] = pelvis + 0.25 * torso_axis
    out[:, 4] = body[:, ids["left_knee_link"]]
    out[:, 5] = body[:, ids["right_knee_link"]]
    out[:, 6] = pelvis + 0.50 * torso_axis
    out[:, 7] = body[:, ids["left_ankle_pitch_link"]]
    out[:, 8] = body[:, ids["right_ankle_pitch_link"]]
    out[:, 9] = pelvis + 0.75 * torso_axis
    out[:, 10] = body[:, ids["left_ankle_roll_link"]]
    out[:, 11] = body[:, ids["right_ankle_roll_link"]]
    out[:, 12] = shoulder_mid
    out[:, 13] = 0.5 * (out[:, 9] + left_shoulder)
    out[:, 14] = 0.5 * (out[:, 9] + right_shoulder)
    out[:, 15] = shoulder_mid + 0.20 * torso_unit
    out[:, 16] = left_shoulder
    out[:, 17] = right_shoulder
    out[:, 18] = body[:, ids["left_elbow_link"]]
    out[:, 19] = body[:, ids["right_elbow_link"]]
    out[:, 20] = body[:, ids["left_wrist_yaw_link"]]
    out[:, 21] = body[:, ids["right_wrist_yaw_link"]]

    # Proper rotation with determinant +1: Isaac Z-up -> HumanML3D Y-up.
    x, y, z = np.moveaxis(out, -1, 0)
    out_y_up = np.stack((x, z, -y), axis=-1).astype(np.float32)
    if not np.isfinite(out_y_up).all():
        raise ValueError("non-finite adapted joints")
    return out_y_up


def geometry_audit(joints: np.ndarray) -> dict[str, float | bool]:
    lengths = np.stack(
        [np.linalg.norm(joints[:, child] - joints[:, parent], axis=-1)
         for parent, child in HUMANML_EDGES],
        axis=-1,
    )
    across = (
        joints[:, 2] - joints[:, 1]
        + joints[:, 17] - joints[:, 16]
    )
    facing_norm = np.linalg.norm(across, axis=-1)
    result = {
        "finite": bool(np.isfinite(joints).all()),
        "min_bone_length_m": float(lengths.min()),
        "median_bone_length_m": float(np.median(lengths)),
        "min_facing_axis_norm_m": float(facing_norm.min()),
    }
    result["passed"] = bool(
        result["finite"]
        and result["min_bone_length_m"] > 1.0e-4
        and result["min_facing_axis_norm_m"] > 1.0e-3
    )
    return result


def resample_joints(joints: np.ndarray) -> np.ndarray:
    result = resample_poly(joints, TMR_FPS, SOURCE_FPS, axis=0)
    return np.asarray(result, dtype=np.float32)


def windows(joints: np.ndarray) -> list[np.ndarray]:
    if len(joints) < WINDOW_JOINT_FRAMES:
        raise ValueError(
            f"motion has {len(joints)} frames; requires {WINDOW_JOINT_FRAMES}"
        )
    starts = list(
        range(0, len(joints) - WINDOW_JOINT_FRAMES + 1, STRIDE_JOINT_FRAMES)
    )
    if starts[-1] != len(joints) - WINDOW_JOINT_FRAMES:
        starts.append(len(joints) - WINDOW_JOINT_FRAMES)
    return [joints[start : start + WINDOW_JOINT_FRAMES] for start in starts]


def load_source_motion(task: str, motion_id: int) -> tuple[np.ndarray, tuple[str, ...]]:
    path = ROOT / f"SUGAR/data/{task}/data_{motion_id:03d}/robot_50hz.npz"
    with np.load(path, allow_pickle=False) as archive:
        fps = int(np.asarray(archive["fps"]).reshape(-1)[0])
        body = np.asarray(archive["body_pos_w"], dtype=np.float32)
    if fps != SOURCE_FPS:
        raise ValueError(f"{path} has unexpected fps={fps}")
    return body, SUGAR_BODY_NAMES


def first_episode(body: np.ndarray, done: np.ndarray, env_id: int) -> np.ndarray:
    env_done = np.asarray(done[:, env_id], dtype=bool)
    ends = np.flatnonzero(env_done)
    end = int(ends[0] + 1) if len(ends) else len(body)
    return np.asarray(body[:end, env_id], dtype=np.float32)


def load_rollout_samples(path: Path) -> list[tuple[str, np.ndarray, tuple[str, ...]]]:
    with np.load(path, allow_pickle=False) as archive:
        body = np.asarray(archive["robot_body_position_w"], dtype=np.float32)
        done = np.asarray(archive["done"], dtype=bool)
        names = tuple(str(value) for value in archive["robot_body_names"].tolist())
    if body.ndim != 4 or done.shape != body.shape[:2]:
        raise ValueError(f"invalid rollout shape at {path}")
    return [
        (f"{path.parent.name}_env{env_id:02d}", first_episode(body, done, env_id), names)
        for env_id in range(body.shape[1])
    ]


def prepare_official_tmr(tmr_root: Path):
    tmr_root = tmr_root.resolve()
    deps = tmr_root / "runtime_deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    sys.path.insert(0, str(tmr_root))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from hydra.utils import instantiate
    from src.config import read_config
    from src.data.collate import collate_x_dict
    from src.guofeats.motion_representation import joints_to_guofeats
    from src.load import load_model_from_cfg

    run_dir = tmr_root / "models/tmr_humanml3d_guoh3dfeats"
    cfg = read_config(str(run_dir))
    model = load_model_from_cfg(cfg, "last", device="cpu", eval_mode=True)
    normalizer_cfg = cfg.data.motion_loader.normalizer
    normalizer_cfg.base_dir = str(tmr_root / "stats/humanml3d/guoh3dfeats")
    normalizer = instantiate(normalizer_cfg)
    return torch, model, normalizer, collate_x_dict, joints_to_guofeats


def encode_samples(
    samples: Sequence[tuple[str, np.ndarray, Sequence[str]]],
    *,
    tmr_root: Path,
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | bool]], int]:
    torch, model, normalizer, collate_x_dict, joints_to_guofeats = prepare_official_tmr(
        tmr_root
    )
    model = model.to(device).eval()

    feature_items: list[tuple[str, np.ndarray]] = []
    audits: dict[str, dict[str, float | bool]] = {}
    for name, body, body_names in samples:
        adapted = g1_bodies_to_humanml_joints(body, body_names)
        audits[name] = geometry_audit(adapted)
        if not audits[name]["passed"]:
            raise RuntimeError(f"geometry audit failed for {name}: {audits[name]}")
        adapted = resample_joints(adapted)
        for window_id, joint_window in enumerate(windows(adapted)):
            guo = np.asarray(joints_to_guofeats(joint_window), dtype=np.float32)
            if guo.shape != (WINDOW_JOINT_FRAMES - 1, 263):
                raise RuntimeError(f"unexpected Guo feature shape {guo.shape}")
            if not np.isfinite(guo).all():
                raise RuntimeError(f"non-finite Guo features for {name}")
            feature_items.append((name, guo))

    per_sample: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in samples}
    with torch.inference_mode():
        for start in range(0, len(feature_items), batch_size):
            batch_items = feature_items[start : start + batch_size]
            x_dicts = []
            for _, feature in batch_items:
                tensor = normalizer(torch.from_numpy(feature).to(torch.float32))
                x_dicts.append({"x": tensor, "length": len(tensor)})
            batch = collate_x_dict(x_dicts, device=device)
            latent = model.encode(
                batch, modality="motion", sample_mean=True
            ).detach().cpu().numpy()
            for (name, _), vector in zip(batch_items, latent, strict=True):
                per_sample[name].append(np.asarray(vector, dtype=np.float32))

    aggregated: dict[str, np.ndarray] = {}
    for name, vectors in per_sample.items():
        matrix = np.stack(vectors)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1.0e-12)
        vector = matrix.mean(axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
        if vector.shape != (256,) or not np.isfinite(vector).all():
            raise RuntimeError(f"invalid TMR latent for {name}")
        aggregated[name] = vector.astype(np.float32)
    return aggregated, audits, len(feature_items)


def normalized_mean(vectors: Iterable[np.ndarray]) -> np.ndarray:
    value = np.stack(list(vectors)).mean(axis=0)
    return value / max(float(np.linalg.norm(value)), 1.0e-12)


def score_records(
    latents: dict[str, np.ndarray],
    labels: dict[str, str],
    splits: dict[str, str],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    prototypes = {
        label: normalized_mean(
            latents[name]
            for name in sorted(latents)
            if labels[name] == label and splits[name] == "train_source"
        )
        for label in ("carry", "kick")
    }
    rows: list[dict[str, object]] = []
    for name in sorted(latents):
        carry_similarity = float(latents[name] @ prototypes["carry"])
        kick_similarity = float(latents[name] @ prototypes["kick"])
        predicted = "carry" if carry_similarity >= kick_similarity else "kick"
        rows.append(
            {
                "name": name,
                "split": splits[name],
                "true_label": labels[name],
                "predicted_label": predicted,
                "carry_cosine": carry_similarity,
                "kick_cosine": kick_similarity,
                "carry_minus_kick": carry_similarity - kick_similarity,
                "correct": predicted == labels[name],
            }
        )
    return rows, prototypes


def accuracy(rows: Sequence[dict[str, object]], split: str) -> float:
    selected = [bool(row["correct"]) for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"no rows for split {split}")
    return float(np.mean(selected))


def render_latent_plot(
    latents: dict[str, np.ndarray],
    labels: dict[str, str],
    splits: dict[str, str],
    output: Path,
) -> None:
    train_names = [name for name in latents if splits[name] == "train_source"]
    matrix = np.stack([latents[name] for name in train_names])
    center = matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(matrix - center, full_matrices=False)
    basis = vh[:2].T

    style = {
        ("train_source", "carry"): ("#1f77b4", "o", 34, 0.35),
        ("train_source", "kick"): ("#d62728", "o", 34, 0.35),
        ("heldout_source", "carry"): ("#1f77b4", "^", 52, 0.75),
        ("heldout_source", "kick"): ("#d62728", "^", 52, 0.75),
        ("named_source", "carry"): ("#0b4f8a", "*", 180, 1.0),
        ("named_source", "kick"): ("#8e1717", "*", 180, 1.0),
        ("physical_router", "carry"): ("#17becf", "s", 58, 0.8),
        ("physical_router", "kick"): ("#ff7f0e", "s", 58, 0.8),
    }
    fig, ax = plt.subplots(figsize=(9.5, 6.0), facecolor="white")
    for split in ("train_source", "heldout_source", "named_source", "physical_router"):
        for label in ("carry", "kick"):
            names = [
                name for name in latents
                if splits[name] == split and labels[name] == label
            ]
            if not names:
                continue
            points = (np.stack([latents[name] for name in names]) - center) @ basis
            color, marker, size, alpha = style[(split, label)]
            ax.scatter(
                points[:, 0], points[:, 1], c=color, marker=marker, s=size,
                alpha=alpha, label=f"{split}: {label}", edgecolors="none",
            )
    ax.axhline(0.0, color="#dddddd", linewidth=0.8)
    ax.axvline(0.0, color="#dddddd", linewidth=0.8)
    ax.set_title("Released TMR motion latent: SUGAR source demos and PhysX rollouts")
    ax.set_xlabel("training-source PCA axis 1")
    ax.set_ylabel("training-source PCA axis 2")
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[str, np.ndarray, Sequence[str]]] = []
    labels: dict[str, str] = {}
    splits: dict[str, str] = {}
    for task, label in (("CarryBox", "carry"), ("KickBox", "kick")):
        for split, motion_ids in (("train_source", TRAIN_IDS), ("heldout_source", TEST_IDS)):
            for motion_id in motion_ids:
                name = f"source_{label}_{motion_id:03d}"
                body, names = load_source_motion(task, motion_id)
                samples.append((name, body, names))
                labels[name] = label
                splits[name] = split
    for name, task, motion_id in NAMED_QUERIES:
        label = "carry" if task == "CarryBox" else "kick"
        body, names = load_source_motion(task, motion_id)
        samples.append((name, body, names))
        labels[name] = label
        splits[name] = "named_source"

    rollout_specs = (
        ("router_carry45", "carry", args.router_root / "carry_carry45/TRACE.npz"),
        ("router_kick21", "kick", args.router_root / "kick_kick21/TRACE.npz"),
    )
    for prefix, label, path in rollout_specs:
        for raw_name, body, names in load_rollout_samples(path):
            name = f"{prefix}_{raw_name.rsplit('_env', 1)[-1]}"
            samples.append((name, body, names))
            labels[name] = label
            splits[name] = "physical_router"

    latents, audits, encoded_window_count = encode_samples(
        samples,
        tmr_root=args.tmr_root,
        device=args.device,
        batch_size=args.batch_size,
    )
    rows, prototypes = score_records(latents, labels, splits)

    heldout_accuracy = accuracy(rows, "heldout_source")
    physical_accuracy = accuracy(rows, "physical_router")
    named = {row["name"]: row for row in rows if row["split"] == "named_source"}
    result = {
        "protocol": "official_tmr_sugar_motion_semantic_gate_v1",
        "official_repository": "https://github.com/Mathux/TMR",
        "official_checkout_commit": "6d74688730d15d43b0a755ce2b0e1f2d76138fc1",
        "official_model": "tmr_humanml3d_guoh3dfeats/last_weights",
        "latent_dim": 256,
        "adapter": {
            "source_fps": SOURCE_FPS,
            "tmr_fps": TMR_FPS,
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "coordinate_map": "Isaac (x,y,z) -> HumanML3D (x,z,-y)",
            "policy": "G1 body centers -> explicit HumanML3D 22-joint topology -> official joints_to_guofeats",
            "geometry_audits_passed": bool(all(value["passed"] for value in audits.values())),
        },
        "data": {
            "train_source_ids_per_task": list(TRAIN_IDS),
            "heldout_source_ids_per_task": list(TEST_IDS),
            "physical_rollouts_per_task": 20,
            "encoded_window_count": encoded_window_count,
        },
        "metrics": {
            "heldout_source_accuracy": heldout_accuracy,
            "physical_router_accuracy": physical_accuracy,
            "carry45_carry_minus_kick": named["carry45"]["carry_minus_kick"],
            "kick21_carry_minus_kick": named["kick21"]["carry_minus_kick"],
            "prototype_cosine": float(prototypes["carry"] @ prototypes["kick"]),
        },
    }
    result["gate"] = {
        "finite_and_geometry_valid": result["adapter"]["geometry_audits_passed"],
        "heldout_source_accuracy_at_least_0p80": heldout_accuracy >= 0.80,
        "physical_router_accuracy_at_least_0p80": physical_accuracy >= 0.80,
        "named_carry45_and_kick21_correct": bool(
            named["carry45"]["correct"] and named["kick21"]["correct"]
        ),
    }
    result["gate"]["passed"] = bool(all(result["gate"].values()))
    result["interpretation"] = (
        "Passing permits a separate matched online-reward design; it does not establish "
        "policy benefit or object-interaction understanding."
        if result["gate"]["passed"]
        else "Do not integrate this TMR latent into policy training; the frozen semantic gate failed."
    )

    with (args.output / "SCORES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "RESULT.json").open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    np.savez_compressed(
        args.output / "LATENTS.npz",
        names=np.asarray(sorted(latents), dtype="U80"),
        latent=np.stack([latents[name] for name in sorted(latents)]),
        carry_prototype=prototypes["carry"],
        kick_prototype=prototypes["kick"],
    )
    render_latent_plot(
        latents, labels, splits, args.output / "tmr_latent_source_and_physx.png"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
