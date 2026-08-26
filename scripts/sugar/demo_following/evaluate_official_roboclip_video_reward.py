#!/usr/bin/env python3
"""Audit the released RoboCLIP video reward on clean SUGAR demonstrations.

This is a frozen-representation audit.  It loads the exact released S3D
HowTo100M model used by RoboCLIP and never trains a predictor or policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch


TASKS = ("CarryBox", "KickBox")
REFERENCE_IDS = {"CarryBox": 45, "KickBox": 21}
INPUT_FRAMES = 32
INPUT_SIZE = 224


def numeric_frame_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png"), key=lambda path: int(path.stem))
    if len(paths) != 64 or [int(path.stem) for path in paths] != list(range(64)):
        raise RuntimeError(f"{directory}: expected exact frames 0..63")
    return paths


def uniform_frame_indices(frame_count: int, output_count: int = INPUT_FRAMES) -> np.ndarray:
    if frame_count < output_count or output_count <= 0:
        raise ValueError("frame_count must be at least output_count > 0")
    indices = np.rint(np.linspace(0, frame_count - 1, output_count)).astype(np.int64)
    if len(np.unique(indices)) != output_count:
        raise RuntimeError("uniform sampling produced duplicate frame indices")
    return indices


def load_video_tensor(directory: Path) -> torch.Tensor:
    paths = numeric_frame_paths(directory)
    indices = uniform_frame_indices(len(paths))
    frames = []
    for index in indices:
        frame = cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to decode {paths[int(index)]}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    array = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()


def load_official_s3d(
    source: Path, weights: Path, dictionary: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, object]]:
    spec = importlib.util.spec_from_file_location("official_roboclip_s3dg", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official S3D source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.S3D(str(dictionary), 512)
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"non-strict S3D load: {incompatible}")
    model.eval().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model, {
        "parameter_count": int(parameter_count),
        "embedding_dimension": 512,
        "strict_state_dict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "weights_bytes": weights.stat().st_size,
        "dictionary_bytes": dictionary.stat().st_size,
    }


def embed_directories(
    model: torch.nn.Module,
    directories: list[Path],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    original: dict[str, np.ndarray] = {}
    reversed_order: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(directories), batch_size):
            batch_dirs = directories[start : start + batch_size]
            batch = torch.stack([load_video_tensor(path) for path in batch_dirs]).to(device)
            forward = model(batch)["video_embedding"]
            reverse = model(torch.flip(batch, dims=(2,)))["video_embedding"]
            for directory, embedding, reverse_embedding in zip(
                batch_dirs, forward.cpu().numpy(), reverse.cpu().numpy(), strict=True
            ):
                key = str(directory.resolve())
                original[key] = embedding.astype(np.float64)
                reversed_order[key] = reverse_embedding.astype(np.float64)
    return original, reversed_order


def similarity(first: np.ndarray, second: np.ndarray, mode: str) -> float:
    if mode == "dot":
        return float(np.dot(first, second))
    if mode == "cosine":
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 0:
            raise RuntimeError("zero-norm RoboCLIP embedding")
        return float(np.dot(first, second) / denominator)
    raise ValueError(mode)


def reference_selection_metrics(
    corpus: Path,
    split: str,
    embeddings: dict[str, np.ndarray],
    mode: str,
) -> dict[str, object]:
    references = {
        task: embeddings[str((corpus / "train" / task / str(REFERENCE_IDS[task])).resolve())]
        for task in TASKS
    }
    rows = []
    for task in TASKS:
        unrelated_task = TASKS[1 - TASKS.index(task)]
        for directory in sorted(
            (corpus / split / task).iterdir(), key=lambda path: int(path.name)
        ):
            query = embeddings[str(directory.resolve())]
            correct_score = similarity(query, references[task], mode)
            unrelated_score = similarity(query, references[unrelated_task], mode)
            rows.append(
                {
                    "task": task,
                    "motion_id": int(directory.name),
                    "correct_score": correct_score,
                    "unrelated_score": unrelated_score,
                    "margin": correct_score - unrelated_score,
                    "correct_reference_wins": bool(correct_score > unrelated_score),
                }
            )
    return {
        "accuracy": float(np.mean([row["correct_reference_wins"] for row in rows])),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
        "median_margin": float(np.median([row["margin"] for row in rows])),
        "queries": len(rows),
        "rows": rows,
    }


def order_sensitivity_metrics(
    corpus: Path,
    split: str,
    original: dict[str, np.ndarray],
    reversed_order: dict[str, np.ndarray],
    mode: str,
) -> dict[str, object]:
    reference_paths = {
        task: corpus / "train" / task / str(REFERENCE_IDS[task]) for task in TASKS
    }
    rows = []
    for task in TASKS:
        reference_key = str(reference_paths[task].resolve())
        for directory in sorted(
            (corpus / split / task).iterdir(), key=lambda path: int(path.name)
        ):
            query = original[str(directory.resolve())]
            ordered_score = similarity(query, original[reference_key], mode)
            reversed_score = similarity(query, reversed_order[reference_key], mode)
            rows.append(
                {
                    "task": task,
                    "motion_id": int(directory.name),
                    "ordered_score": ordered_score,
                    "reversed_score": reversed_score,
                    "margin": ordered_score - reversed_score,
                    "ordered_demo_wins": bool(ordered_score > reversed_score),
                }
            )
    return {
        "accuracy": float(np.mean([row["ordered_demo_wins"] for row in rows])),
        "mean_margin": float(np.mean([row["margin"] for row in rows])),
        "median_margin": float(np.median([row["margin"] for row in rows])),
        "queries": len(rows),
        "rows": rows,
    }


def corpus_directories(corpus: Path) -> list[Path]:
    directories = []
    expected = {"train": (80, 80), "valid": (10, 10), "test": (10, 9)}
    for split, counts in expected.items():
        for task, count in zip(TASKS, counts, strict=True):
            task_dirs = sorted(
                (corpus / split / task).iterdir(), key=lambda path: int(path.name)
            )
            if len(task_dirs) != count:
                raise RuntimeError(f"{split}/{task}: expected {count}, found {len(task_dirs)}")
            directories.extend(task_dirs)
    return directories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    corpus = args.corpus.expanduser().resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, model_audit = load_official_s3d(
        args.official_source.expanduser().resolve(),
        args.weights.expanduser().resolve(),
        args.dictionary.expanduser().resolve(),
        device,
    )
    directories = corpus_directories(corpus)
    original, reversed_order = embed_directories(
        model, directories, device, args.batch_size
    )
    if not all(np.isfinite(value).all() for value in original.values()):
        raise RuntimeError("non-finite official RoboCLIP embedding")

    metrics = {}
    for mode in ("dot", "cosine"):
        metrics[mode] = {
            split: {
                "reference_selection": reference_selection_metrics(
                    corpus, split, original, mode
                ),
                "order_sensitivity": order_sensitivity_metrics(
                    corpus, split, original, reversed_order, mode
                ),
            }
            for split in ("valid", "test")
        }
    criteria = {
        "official_dot_valid_reference_accuracy_at_least_0p75": bool(
            metrics["dot"]["valid"]["reference_selection"]["accuracy"] >= 0.75
        ),
        "official_dot_test_reference_accuracy_at_least_0p75": bool(
            metrics["dot"]["test"]["reference_selection"]["accuracy"] >= 0.75
        ),
        "cosine_valid_reference_accuracy_at_least_0p75": bool(
            metrics["cosine"]["valid"]["reference_selection"]["accuracy"] >= 0.75
        ),
        "cosine_test_reference_accuracy_at_least_0p75": bool(
            metrics["cosine"]["test"]["reference_selection"]["accuracy"] >= 0.75
        ),
        "official_dot_valid_order_accuracy_at_least_0p75": bool(
            metrics["dot"]["valid"]["order_sensitivity"]["accuracy"] >= 0.75
        ),
        "official_dot_test_order_accuracy_at_least_0p75": bool(
            metrics["dot"]["test"]["order_sensitivity"]["accuracy"] >= 0.75
        ),
    }
    result = {
        "protocol": "official_roboclip_sugar_selected_demo_v1",
        "scope": (
            "frozen released RoboCLIP video reward on source-ID-disjoint clean SUGAR RGB; "
            "representation admission only, not policy following"
        ),
        "official_components": {
            "roboclip_commit": "2d3f779033f1f3adf307a64080742e158caafe67",
            "s3d_commit": "b8cd0bbfd16fe41629d1b15e0cf384d75f56101a",
            "reward": "raw dot product of released 512-D video embeddings",
            "input": "32 uniformly sampled RGB frames at 224x224 in [0,1]",
        },
        "corpus": str(corpus),
        "model_audit": model_audit,
        "metrics": metrics,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        mode: {
            split: {
                "reference_accuracy": metrics[mode][split]["reference_selection"]["accuracy"],
                "order_accuracy": metrics[mode][split]["order_sensitivity"]["accuracy"],
            }
            for split in ("valid", "test")
        }
        for mode in ("dot", "cosine")
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"ROBOCLIP_REPRESENTATION_RESULT passed={result['passed']} output={args.output}")


if __name__ == "__main__":
    main()
