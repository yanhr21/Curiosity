#!/usr/bin/env python3
"""Evaluate released XIRL/TCC embeddings on motion-disjoint SUGAR videos.

The trained model is compared with the official raw-ImageNet ResNet18
baseline.  This evaluates temporal retrieval and CarryBox-vs-KickBox reference
selection; it does not claim selected-motion instance identification or policy
improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
import torch

from configs.xmagical.pretraining.imagenet import get_config as get_imagenet_config
from utils import load_config_from_dir
from xirl import factory
from xirl.types import SequenceType


TASKS = ("CarryBox", "KickBox")
REFERENCE_IDS = {"CarryBox": 45, "KickBox": 21}


def embed_corpus(model, config, corpus: Path, split: str, device: torch.device):
    config.data.root = str(corpus)
    config.data.downstream_action_class = TASKS
    config.data.max_vids_per_class = -1
    datasets = factory.dataset_from_config(config, True, split, debug=True)
    embedded: dict[str, dict[int, np.ndarray]] = {}
    model.eval().to(device)
    with torch.no_grad():
        for task, dataset in datasets.items():
            task_embs: dict[int, np.ndarray] = {}
            for index in range(len(dataset)):
                sample = dataset[(0, index)]
                motion_id = int(Path(sample[SequenceType.VIDEO_NAME]).name)
                frames = sample[SequenceType.FRAMES].unsqueeze(0).to(device)
                task_embs[motion_id] = model.infer(frames).embs.numpy()
            embedded[task] = task_embs
    return embedded


def temporal_metrics(task_embs: dict[int, np.ndarray]) -> dict[str, float]:
    errors: list[float] = []
    taus: list[float] = []
    items = sorted(task_embs.items())
    for query_id, query in items:
        for candidate_id, candidate in items:
            if query_id == candidate_id:
                continue
            distances = cdist(query, candidate, "sqeuclidean")
            nearest = np.argmin(distances, axis=1)
            expected = np.linspace(0, len(candidate) - 1, len(query))
            errors.append(float(np.mean(np.abs(nearest - expected)) / (len(candidate) - 1)))
            tau = kendalltau(np.arange(len(nearest)), nearest).correlation
            if np.isfinite(tau):
                taus.append(float(tau))
    if not errors or not taus:
        raise RuntimeError("not enough held-out videos for temporal retrieval")
    return {
        "normalized_temporal_mae": float(np.mean(errors)),
        "kendalls_tau": float(np.mean(taus)),
        "ordered_video_pairs": len(errors),
    }


def alignment_cost(first: np.ndarray, second: np.ndarray) -> float:
    distances = cdist(first, second, "sqeuclidean")
    return float(0.5 * (np.min(distances, axis=0).mean() + np.min(distances, axis=1).mean()))


def reference_selection_metrics(
    queries: dict[str, dict[int, np.ndarray]],
    references: dict[str, dict[int, np.ndarray]],
) -> dict[str, object]:
    reference_embs = {
        task: references[task][REFERENCE_IDS[task]] for task in TASKS
    }
    rows = []
    for task in TASKS:
        unrelated_task = TASKS[1 - TASKS.index(task)]
        for motion_id, query in sorted(queries[task].items()):
            correct_cost = alignment_cost(query, reference_embs[task])
            unrelated_cost = alignment_cost(query, reference_embs[unrelated_task])
            rows.append(
                {
                    "task": task,
                    "motion_id": motion_id,
                    "correct_reference_motion_id": REFERENCE_IDS[task],
                    "unrelated_reference_task": unrelated_task,
                    "unrelated_reference_motion_id": REFERENCE_IDS[unrelated_task],
                    "correct_cost": correct_cost,
                    "unrelated_cost": unrelated_cost,
                    "correct_reference_wins": bool(correct_cost < unrelated_cost),
                }
            )
    return {
        "accuracy": float(np.mean([row["correct_reference_wins"] for row in rows])),
        "queries": len(rows),
        "rows": rows,
    }


def load_trained_model(run_dir: Path, device: torch.device):
    config = load_config_from_dir(str(run_dir))
    model = factory.model_from_config(config)
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("*.ckpt"), key=lambda path: int(path.stem)
    )
    if not checkpoints:
        raise FileNotFoundError(f"no official XIRL checkpoint under {run_dir}")
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return config, model, checkpoints[-1]


def load_raw_imagenet_model(corpus: Path):
    config = get_imagenet_config()
    config.data.root = str(corpus)
    config.data.downstream_action_class = TASKS
    config.frame_sampler.all_sampler.stride = 1
    return config, factory.model_from_config(config)


def evaluate_model(model, config, corpus: Path, device: torch.device):
    train = embed_corpus(model, config, corpus, "train", device)
    valid = embed_corpus(model, config, corpus, "valid", device)
    test = embed_corpus(model, config, corpus, "test", device)
    return {
        "valid_temporal": {task: temporal_metrics(valid[task]) for task in TASKS},
        "test_temporal": {task: temporal_metrics(test[task]) for task in TASKS},
        "valid_reference_selection": reference_selection_metrics(valid, train),
        "test_reference_selection": reference_selection_metrics(test, train),
    }


def aggregate_temporal(result: dict[str, object], split: str, metric: str) -> float:
    return float(np.mean([result[f"{split}_temporal"][task][metric] for task in TASKS]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    corpus = args.corpus.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    trained_config, trained_model, checkpoint = load_trained_model(run_dir, device)
    raw_config, raw_model = load_raw_imagenet_model(corpus)
    trained = evaluate_model(trained_model, trained_config, corpus, device)
    raw = evaluate_model(raw_model, raw_config, corpus, device)

    trained_test_mae = aggregate_temporal(trained, "test", "normalized_temporal_mae")
    raw_test_mae = aggregate_temporal(raw, "test", "normalized_temporal_mae")
    trained_test_tau = aggregate_temporal(trained, "test", "kendalls_tau")
    raw_test_tau = aggregate_temporal(raw, "test", "kendalls_tau")
    criteria = {
        "test_temporal_mae_relative_improvement_at_least_5pct": bool(
            trained_test_mae <= 0.95 * raw_test_mae
        ),
        "test_kendalls_tau_improvement_at_least_0p05": bool(
            trained_test_tau >= raw_test_tau + 0.05
        ),
        "test_correct_task_reference_accuracy_at_least_0p75": bool(
            trained["test_reference_selection"]["accuracy"] >= 0.75
        ),
    }
    result = {
        "protocol": "official_xirl_tcc_sugar_motion_disjoint_v1",
        "scope": (
            "visual temporal progress and CarryBox-vs-KickBox semantic reference selection; "
            "not selected-motion instance identity and not policy improvement"
        ),
        "official_components": (
            "released Google Research XIRL ResNet18-linear 32-D encoder, TCC loss, "
            "same-class sampler and official raw-ImageNet baseline"
        ),
        "checkpoint": str(checkpoint),
        "trained": trained,
        "raw_imagenet": raw,
        "summary": {
            "trained_test_normalized_temporal_mae": trained_test_mae,
            "raw_test_normalized_temporal_mae": raw_test_mae,
            "trained_test_kendalls_tau": trained_test_tau,
            "raw_test_kendalls_tau": raw_test_tau,
            "trained_test_reference_selection_accuracy": trained[
                "test_reference_selection"
            ]["accuracy"],
            "raw_test_reference_selection_accuracy": raw[
                "test_reference_selection"
            ]["accuracy"],
        },
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"XIRL_TEMPORAL_RESULT passed={result['passed']} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
