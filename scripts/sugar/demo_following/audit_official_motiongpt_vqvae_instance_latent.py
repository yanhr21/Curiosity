#!/usr/bin/env python3
"""Gate the released MotionGPT HumanML3D VQ-VAE for selected-demo distance.

This is a frozen-representation audit, not policy training.  The official
MotionGPT/T2M-GPT VQ-VAE receives exact normalized 263-D HumanML3D features.
For real IsaacLab/PhysX Tracker rollouts, it compares each future motion window
against three phase-matched references: its own source demonstration, a
different demonstration of the same task, and a demonstration from the other
task.  Neither a classifier nor a learned metric is fitted.

The predeclared gate requires the continuous official encoder distance to rank
the correct selected demo ahead of the cross-task demo in every motion-disjoint
split, with a paired win rate above 60 percent, while the released decoder must
beat the normalized zero reconstruction baseline.  Discrete-code mismatch is
reported independently and is not substituted when the continuous gate fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sugar/demo_following"))
sys.path.insert(0, str(ROOT / "scripts/sugar/demo_reward"))

from audit_actual_contact_event_corpus import motion_split  # noqa: E402
from audit_official_tmr_motion_latent import (  # noqa: E402
    SUGAR_BODY_NAMES,
    g1_bodies_to_humanml_joints,
    geometry_audit,
    resample_joints,
)
from build_actual_contact_event_predictor_dataset import actual_entries  # noqa: E402


DEFAULT_CORPUS = ROOT / (
    "experiments/demo_following/contact_event_reward_redesign_v1/"
    "deployable_goal_core_corpus_v1"
)
DEFAULT_OFFICIAL_ROOT = ROOT / "experiments/runtime_assets/official_motiongpt_qiqi"
DEFAULT_TMR_ROOT = ROOT / "experiments/runtime_assets/official_tmr"
DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/official_motiongpt_vqvae_instance_gate_v2"
)

TASKS = ("CarryBox", "KickBox")
PHASES = (0.25, 0.50, 0.75)
SOURCE_WINDOW_FRAMES = 161  # 50 Hz -> 65 joints at 20 Hz -> 64 Guo features.
FEATURE_FRAMES = 64
FEATURE_DIM = 263
CONTINUOUS_WIN_RATE_THRESHOLD = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--tmr-root", type=Path, default=DEFAULT_TMR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_reference_body(task: str, source_id: int) -> np.ndarray:
    path = ROOT / f"SUGAR/data/{task}/data_{source_id:03d}/robot_50hz.npz"
    with np.load(path, allow_pickle=False) as archive:
        fps = int(np.asarray(archive["fps"]).reshape(-1)[0])
        body = np.asarray(archive["body_pos_w"], dtype=np.float32)
    if fps != 50 or len(body) < SOURCE_WINDOW_FRAMES:
        raise RuntimeError(f"invalid source motion {path}: fps={fps}, frames={len(body)}")
    return body


def phase_window(body: np.ndarray, phase: float) -> np.ndarray:
    if len(body) < SOURCE_WINDOW_FRAMES:
        raise RuntimeError(f"motion is too short: {len(body)}")
    start = int(round(float(phase) * (len(body) - SOURCE_WINDOW_FRAMES)))
    return np.asarray(body[start : start + SOURCE_WINDOW_FRAMES], dtype=np.float32)


def cross_task_key(task: str, source_id: int) -> tuple[str, int]:
    if task == "CarryBox":
        return "KickBox", source_id % 99
    return "CarryBox", source_id % 100


def same_task_wrong_key(task: str, source_id: int) -> tuple[str, int]:
    count = 100 if task == "CarryBox" else 99
    return task, (source_id + 1) % count


def guo_features(
    body_windows: Sequence[np.ndarray],
    body_names: Sequence[str],
    joints_to_guofeats,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    result: list[np.ndarray] = []
    audits: list[dict[str, object]] = []
    for body in body_windows:
        joints = g1_bodies_to_humanml_joints(body, body_names)
        audit = geometry_audit(joints)
        audits.append(audit)
        if not audit["passed"]:
            raise RuntimeError(f"geometry audit failed: {audit}")
        joints = resample_joints(joints)
        if joints.shape != (FEATURE_FRAMES + 1, 22, 3):
            raise RuntimeError(f"unexpected resampled joint shape {joints.shape}")
        feature = np.asarray(joints_to_guofeats(joints), dtype=np.float32)
        if feature.shape != (FEATURE_FRAMES, FEATURE_DIM):
            raise RuntimeError(f"unexpected HumanML3D feature shape {feature.shape}")
        if not np.isfinite(feature).all():
            raise RuntimeError("non-finite HumanML3D features")
        result.append(feature)
    return np.stack(result), audits


class OfficialMotionGPTVQVAE:
    """Thin inference-only wrapper around the released MotionGPT VQ-VAE."""

    def __init__(
        self,
        official_root: Path,
        device: str,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        sys.path.insert(0, str(official_root.resolve()))
        from models.vqvae import HumanVQVAE

        model_args = SimpleNamespace(
            dataname="t2m", quantizer="ema_reset", mu=0.99, beta=1.0
        )
        # The released implementation initializes its EMA codebook on CUDA.
        # Instantiation therefore intentionally occurs only inside a GPU job.
        with torch.cuda.device(self.device):
            model = HumanVQVAE(
                model_args,
                nb_code=512,
                code_dim=512,
                output_emb_width=512,
                down_t=2,
                stride_t=2,
                width=512,
                depth=3,
                dilation_growth_rate=3,
                activation="relu",
                norm=None,
            )
        checkpoint = torch.load(
            official_root / "checkpoints/pretrained_vqvae/t2m.pth",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["net"], strict=True)
        self.model = model.to(self.device).eval().requires_grad_(False)
        stats = (
            official_root
            / "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
        )
        self.mean = torch.from_numpy(np.load(stats / "mean.npy")).float()
        self.std = torch.from_numpy(np.load(stats / "std.npy")).float()
        if self.mean.shape != (FEATURE_DIM,) or self.std.shape != (FEATURE_DIM,):
            raise RuntimeError("official HumanML3D normalization statistics have wrong shape")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise RuntimeError("non-finite official normalization statistics")
        if torch.any(self.std <= 0):
            raise RuntimeError("non-positive official normalization scale")

    def encode(
        self,
        features: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        torch = self.torch
        continuous: list[np.ndarray] = []
        codes: list[np.ndarray] = []
        reconstruction_error = 0.0
        zero_error = 0.0
        value_count = 0
        mean = self.mean.to(self.device)
        std = self.std.to(self.device)
        with torch.inference_mode():
            for begin in range(0, len(features), batch_size):
                x = torch.from_numpy(features[begin : begin + batch_size]).to(
                    self.device, torch.float32
                )
                x = (x - mean) / std
                x_in = self.model.vqvae.preprocess(x)
                encoded = self.model.vqvae.encoder(x_in)
                flat = self.model.vqvae.postprocess(encoded).contiguous()
                index = self.model.vqvae.quantizer.quantize(flat.reshape(-1, flat.shape[-1]))
                index = index.view(len(x), -1)
                decoded, _, _ = self.model(x)
                reconstruction_error += float(torch.abs(decoded - x).sum().item())
                zero_error += float(torch.abs(x).sum().item())
                value_count += x.numel()
                continuous.append(flat.cpu().numpy().astype(np.float32))
                codes.append(index.cpu().numpy().astype(np.int16))
        continuous_array = np.concatenate(continuous)
        code_array = np.concatenate(codes)
        if continuous_array.shape[:2] != code_array.shape:
            raise RuntimeError(
                f"encoder/code clock mismatch {continuous_array.shape} vs {code_array.shape}"
            )
        if not np.isfinite(continuous_array).all():
            raise RuntimeError("non-finite official VQ-VAE encoder output")
        return continuous_array, code_array, {
            "normalized_reconstruction_mae": reconstruction_error / value_count,
            "normalized_zero_baseline_mae": zero_error / value_count,
            "reconstruction_to_zero_ratio": reconstruction_error / max(zero_error, 1.0e-12),
        }


def continuous_cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1.0e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1.0e-12)
    return (1.0 - np.sum(left_norm * right_norm, axis=-1)).mean(axis=-1)


def summarize(rows: list[dict[str, object]], split: str) -> dict[str, float]:
    chosen = [row for row in rows if row["split"] == split]
    correct = np.asarray([row["continuous_correct"] for row in chosen], dtype=np.float64)
    same = np.asarray([row["continuous_same_task_wrong"] for row in chosen], dtype=np.float64)
    cross = np.asarray([row["continuous_cross_task_wrong"] for row in chosen], dtype=np.float64)
    code_correct = np.asarray([row["code_correct"] for row in chosen], dtype=np.float64)
    code_cross = np.asarray([row["code_cross_task_wrong"] for row in chosen], dtype=np.float64)
    return {
        "count": int(len(chosen)),
        "continuous_correct_mean": float(correct.mean()),
        "continuous_same_task_wrong_mean": float(same.mean()),
        "continuous_cross_task_wrong_mean": float(cross.mean()),
        "continuous_cross_margin_mean": float((cross - correct).mean()),
        "continuous_cross_paired_win_rate": float(np.mean(correct < cross)),
        "code_correct_mean": float(code_correct.mean()),
        "code_cross_task_wrong_mean": float(code_cross.mean()),
        "code_cross_margin_mean": float((code_cross - code_correct).mean()),
        "code_cross_paired_win_rate": float(np.mean(code_correct < code_cross)),
    }


def render(rows: list[dict[str, object]], output: Path) -> None:
    splits = ("train", "validation", "test")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor="white")
    x = np.arange(len(splits))
    width = 0.25
    for axis, prefix, title in (
        (axes[0], "continuous", "Official encoder cosine distance"),
        (axes[1], "code", "Official discrete-code mismatch"),
    ):
        for offset, role, label, color in (
            (-width, "correct", "own demo", "#2b6cb0"),
            (0.0, "same_task_wrong", "same-task wrong", "#dd6b20"),
            (width, "cross_task_wrong", "cross-task wrong", "#c53030"),
        ):
            means = [
                np.mean([float(row[f"{prefix}_{role}"]) for row in rows if row["split"] == split])
                for split in splits
            ]
            axis.bar(x + offset, means, width, label=label, color=color)
        axis.set_xticks(x, splits)
        axis.set_title(title)
        axis.set_ylabel("distance (lower is closer)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Released MotionGPT VQ-VAE on real SUGAR PhysX futures")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tmr_root = args.tmr_root.resolve()
    sys.path.insert(0, str(tmr_root / "runtime_deps"))
    sys.path.insert(0, str(tmr_root))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from src.guofeats.motion_representation import joints_to_guofeats

    entries = actual_entries(args.corpus_root)
    if len(entries) != 199:
        raise RuntimeError(f"expected 199 exact physical motion entries, got {len(entries)}")

    trace_cache: dict[Path, tuple[np.ndarray, tuple[str, ...]]] = {}
    reference_cache: dict[tuple[str, int], np.ndarray] = {}
    actual_windows: list[np.ndarray] = []
    correct_windows: list[np.ndarray] = []
    same_windows: list[np.ndarray] = []
    cross_windows: list[np.ndarray] = []
    descriptors: list[dict[str, object]] = []
    all_audits: list[dict[str, object]] = []
    for entry in entries:
        task = str(entry["task"])
        source_id = int(entry["source_id"])
        trace_path = Path(entry["trace"])
        env = int(entry["env"])
        if trace_path not in trace_cache:
            with np.load(trace_path, allow_pickle=False) as archive:
                trace_cache[trace_path] = (
                    np.asarray(archive["robot_body_position_w"], dtype=np.float32),
                    tuple(str(value) for value in archive["robot_body_names"].tolist()),
                )
        body, body_names = trace_cache[trace_path]
        if body_names != SUGAR_BODY_NAMES:
            raise RuntimeError(f"unexpected physical body order at {trace_path}")
        own_key = (task, source_id)
        same_key = same_task_wrong_key(task, source_id)
        cross_key = cross_task_key(task, source_id)
        for key in (own_key, same_key, cross_key):
            if key not in reference_cache:
                reference_cache[key] = load_reference_body(*key)
        for phase in PHASES:
            actual_windows.append(phase_window(body[:, env], phase))
            correct_windows.append(phase_window(reference_cache[own_key], phase))
            same_windows.append(phase_window(reference_cache[same_key], phase))
            cross_windows.append(phase_window(reference_cache[cross_key], phase))
            descriptors.append(
                {
                    "task": task,
                    "source_id": source_id,
                    "split": motion_split(source_id),
                    "phase": phase,
                    "same_task_wrong": f"{same_key[0]}:{same_key[1]}",
                    "cross_task_wrong": f"{cross_key[0]}:{cross_key[1]}",
                }
            )

    feature_blocks = []
    for label, windows_block in (
        ("actual", actual_windows),
        ("correct", correct_windows),
        ("same", same_windows),
        ("cross", cross_windows),
    ):
        features, audits = guo_features(windows_block, SUGAR_BODY_NAMES, joints_to_guofeats)
        feature_blocks.append(features)
        all_audits.extend(audits)
        print(f"HUMANML_FEATURES_READY {label} {features.shape}", flush=True)
    all_features = np.concatenate(feature_blocks)

    official = OfficialMotionGPTVQVAE(args.official_root, args.device)
    continuous, codes, reconstruction = official.encode(all_features, args.batch_size)
    count = len(descriptors)
    actual_c, correct_c, same_c, cross_c = np.split(continuous, (count, 2 * count, 3 * count))
    actual_q, correct_q, same_q, cross_q = np.split(codes, (count, 2 * count, 3 * count))
    distances = {
        "continuous_correct": continuous_cosine_distance(actual_c, correct_c),
        "continuous_same_task_wrong": continuous_cosine_distance(actual_c, same_c),
        "continuous_cross_task_wrong": continuous_cosine_distance(actual_c, cross_c),
        "code_correct": np.mean(actual_q != correct_q, axis=-1),
        "code_same_task_wrong": np.mean(actual_q != same_q, axis=-1),
        "code_cross_task_wrong": np.mean(actual_q != cross_q, axis=-1),
    }
    rows: list[dict[str, object]] = []
    for index, descriptor in enumerate(descriptors):
        row = dict(descriptor)
        row.update({name: float(value[index]) for name, value in distances.items()})
        rows.append(row)

    summaries = {split: summarize(rows, split) for split in ("train", "validation", "test")}
    named_checks = {}
    for task, source_id in (("CarryBox", 45), ("KickBox", 21)):
        chosen = [row for row in rows if row["task"] == task and row["source_id"] == source_id]
        named_checks[f"{task}{source_id}_correct_closer_than_cross"] = bool(
            np.mean([row["continuous_correct"] for row in chosen])
            < np.mean([row["continuous_cross_task_wrong"] for row in chosen])
        )

    gate = {
        "official_checkpoint_loaded_strictly": True,
        "all_geometry_audits_passed": bool(all(audit["passed"] for audit in all_audits)),
        "decoder_beats_normalized_zero_baseline": reconstruction["reconstruction_to_zero_ratio"] < 1.0,
        "correct_mean_closer_than_cross_in_every_split": all(
            summary["continuous_cross_margin_mean"] > 0.0 for summary in summaries.values()
        ),
        "correct_paired_win_rate_above_0p60_in_every_split": all(
            summary["continuous_cross_paired_win_rate"] > CONTINUOUS_WIN_RATE_THRESHOLD
            for summary in summaries.values()
        ),
        "named_carry45_and_kick21_correct": bool(all(named_checks.values())),
    }
    gate["passed"] = bool(all(gate.values()))
    result = {
        "protocol": "official_motiongpt_vqvae_selected_demo_instance_gate_v2",
        "official_repository": "https://github.com/qiqiApink/MotionGPT",
        "official_checkout_commit": "a1c939b34b8f4e73ba25326e5f934b46f25896e1",
        "official_checkpoint": "checkpoints/pretrained_vqvae/t2m.pth",
        "normalization": "exact official MotionGPT evaluator archive VQ-VAE mean.npy/std.npy",
        "data": {
            "physical_motion_count": len(entries),
            "phases": list(PHASES),
            "window_source_frames_at_50hz": SOURCE_WINDOW_FRAMES,
            "humanml_feature_frames_at_20hz": FEATURE_FRAMES,
            "motion_disjoint_split_rule": "source_motion_id mod10: 8 validation, 9 test, else train",
        },
        "reconstruction": reconstruction,
        "splits": summaries,
        "named": named_checks,
        "gate": gate,
        "interpretation": (
            "The released reconstruction VQ-VAE is eligible for a causal selected-demo mismatch target."
            if gate["passed"]
            else "Do not train a predictor or policy reward from this VQ-VAE distance; the frozen selected-demo gate failed."
        ),
    }
    with (args.output / "SCORES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "RESULT.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    render(rows, args.output / "official_vqvae_selected_demo_distances.png")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
