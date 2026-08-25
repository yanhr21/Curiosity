#!/usr/bin/env python3
"""Train and test motion-disjoint official TinyMDM Carry/Kick task priors.

The model, normalizer, diffusion objective, EMA, sampler, and ESM/SDS energy
are the pinned official MimicKit implementation.  This file only constructs a
task-wide SUGAR G1+box dataset and orchestrates the official model.  Entire
motions, rather than overlapping windows, define train/validation/test splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_selected_demo_tinymdm import (  # noqa: E402
    MIMICKIT_PYTHON,
    PINNED_MIMICKIT_COMMIT,
    SINGLE_CLIP_CONFIG,
    atomic_json,
    atomic_npy,
    atomic_torch,
    build_feature_windows,
    fix_seed,
    load_clip,
    load_prior,
    raw_energy,
    require_compute_gpu,
    runtime_config,
    save_samples,
    validation_loss,
)
from sugar_g1_box_schema import FEATURE_DIM, WINDOW_SIZE  # noqa: E402


OUTPUT_ROOT = ROOT / "experiments/demo_following/taskwide_smp_v1"
TASK_ROOTS = {
    "carry": ROOT / "SUGAR/data/CarryBox",
    "kick": ROOT / "SUGAR/data/KickBox",
}
TRAIN_SEEDS = {"carry": 160825, "kick": 160826}


def motion_id(path: Path) -> int:
    try:
        return int(path.name.removeprefix("data_"))
    except ValueError as error:
        raise ValueError(f"invalid motion directory {path}") from error


def split_for_motion(identifier: int) -> str:
    remainder = identifier % 10
    if remainder == 8:
        return "validation"
    if remainder == 9:
        return "test"
    return "train"


def load_record_array(record: dict[str, Any]) -> np.ndarray:
    array = np.load(record["path"], allow_pickle=False)
    if list(array.shape) != record["shape"] or array.dtype != np.float32:
        raise ValueError(f"dataset drift: {record['path']}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite dataset: {record['path']}")
    return array


def prepare_dataset(output_root: Path, device: torch.device) -> dict[str, Any]:
    dataset_root = output_root / "dataset"
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["arrays"].values():
            if not Path(record["path"]).is_file() or not Path(
                record["motion_ids_path"]
            ).is_file():
                raise FileNotFoundError(record)
        return manifest
    if dataset_root.exists():
        raise FileExistsError(f"incomplete task-wide dataset: {dataset_root}")
    dataset_root.mkdir(parents=True)
    arrays: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for task, task_root in TASK_ROOTS.items():
        grouped: dict[str, list[np.ndarray]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        grouped_ids: dict[str, list[np.ndarray]] = {name: [] for name in grouped}
        paths = sorted(task_root.glob("data_*"), key=motion_id)
        for index, path in enumerate(paths, start=1):
            identifier = motion_id(path)
            split = split_for_motion(identifier)
            robot, obj = load_clip(path)
            features = build_feature_windows(robot, obj, device)
            # Validation/test are whole-motion held out; stride five controls
            # scoring cost without mixing their windows into training.
            if split != "train":
                features = features[::5]
            grouped[split].append(features)
            grouped_ids[split].append(
                np.full(features.shape[0], identifier, dtype=np.int32)
            )
            print(
                json.dumps(
                    {
                        "prepare_task": task,
                        "motion": identifier,
                        "split": split,
                        "windows": int(features.shape[0]),
                        "progress": f"{index}/{len(paths)}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        split_counts[task] = {}
        for split in grouped:
            values = np.concatenate(grouped[split], axis=0).astype(
                np.float32, copy=False
            )
            ids = np.concatenate(grouped_ids[split], axis=0)
            array_path = dataset_root / f"{task}_{split}.npy"
            ids_path = dataset_root / f"{task}_{split}_motion_ids.npy"
            atomic_npy(array_path, values)
            atomic_npy(ids_path, ids)
            arrays[f"{task}_{split}"] = {
                "path": str(array_path),
                "motion_ids_path": str(ids_path),
                "shape": list(values.shape),
                "motion_ids": sorted(set(ids.tolist())),
                "motion_count": len(set(ids.tolist())),
            }
            split_counts[task][split] = len(set(ids.tolist()))
    if split_counts != {
        "carry": {"train": 80, "validation": 10, "test": 10},
        "kick": {"train": 80, "validation": 10, "test": 9},
    }:
        raise RuntimeError(f"unexpected motion-disjoint split {split_counts}")
    manifest = {
        "protocol": "sugar_taskwide_tinymdm_dataset_v1",
        "mimickit_commit": PINNED_MIMICKIT_COMMIT,
        "representation": "official MimicKit compute_disc_obs plus 15-D box feature",
        "feature_geometry": [WINDOW_SIZE, FEATURE_DIM],
        "split_rule": (
            "entire motion id modulo 10: 8=validation, 9=test, all others=train; "
            "validation/test window start stride=5"
        ),
        "arrays": arrays,
        "motion_counts": split_counts,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def train_task_prior(
    output_root: Path,
    manifest: dict[str, Any],
    task: str,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = output_root / "priors" / task
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("passed") is not True or result.get("completed_iterations") != 50_000:
            raise RuntimeError(f"invalid existing task prior: {output_dir}")
        return result
    if output_dir.exists():
        raise FileExistsError(f"incomplete task prior: {output_dir}")
    output_dir.mkdir(parents=True)
    config = runtime_config(output_dir)
    config.update(SINGLE_CLIP_CONFIG)
    # runtime_config writes the same official recipe; persist after the explicit
    # update so the checkpoint always carries its exact configuration.
    config["env_config"] = str(output_dir / "env_config.yaml")
    (output_dir / "diffusion_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    train = torch.as_tensor(
        load_record_array(manifest["arrays"][f"{task}_train"]), device=device
    )
    validation = torch.as_tensor(
        load_record_array(manifest["arrays"][f"{task}_validation"]), device=device
    )
    seed = TRAIN_SEEDS[task]
    fix_seed(seed)
    sys.path.insert(0, str(MIMICKIT_PYTHON))
    from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: PLC0415

    model = TinyMDMModel(config, device).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != 2_836_096:
        raise RuntimeError(f"official TinyMDM parameter count {parameter_count}")
    stat_generator = torch.Generator(device=device).manual_seed(seed + 1)
    stat_indices = torch.randint(
        train.shape[0],
        (int(config["num_samples_stat"]),),
        generator=stat_generator,
        device=device,
    )
    model.update_normalizer(train.index_select(0, stat_indices))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["lr"]),
    )
    sample_generator = torch.Generator(device=device).manual_seed(seed + 2)
    batch_size = int(config["batch_size"])
    output_iter = int(config["output_iter"])
    gradient_clip = float(config["grad_clip_norm"])
    model.train()
    loss_sum = 0.0
    log_path = output_dir / "train_metrics.jsonl"
    for iteration in range(1, int(config["num_iterations"]) + 1):
        indices = torch.randint(
            train.shape[0],
            (batch_size,),
            generator=sample_generator,
            device=device,
        )
        normalized = model.normalize(train.index_select(0, indices)).reshape(
            batch_size, -1
        )
        loss = model(normalized)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip
        )
        optimizer.step()
        if model.model_ema is not False:
            model.ema_dmodel.update()
        loss_sum += float(loss.item())
        if iteration % output_iter == 0:
            metrics = {
                "iteration": iteration,
                "train_loss_mean": loss_sum / output_iter,
                "motion_disjoint_validation_loss": validation_loss(
                    model, validation, seed + 3
                ),
                "gradient_norm_before_clip": float(gradient_norm),
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps({"task": task, **metrics}, sort_keys=True), flush=True)
            loss_sum = 0.0
            model.train()
    model.eval()
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("non-finite task-wide TinyMDM endpoint")
    atomic_torch(output_dir / "model.pt", model.state_dict())
    save_samples(model, output_dir / "feature_samples.npy", device)
    result = {
        "protocol": "sugar_taskwide_official_tinymdm_v1",
        "passed": True,
        "task": task,
        "seed": seed,
        "mimickit_commit": PINNED_MIMICKIT_COMMIT,
        "official_recipe": "MimicKit TinyMDM DiT, EMA, diffusion and ESM/SDS unchanged",
        "parameter_count": parameter_count,
        "train_windows": int(train.shape[0]),
        "train_motion_count": int(
            manifest["arrays"][f"{task}_train"]["motion_count"]
        ),
        "validation_motion_count": int(
            manifest["arrays"][f"{task}_validation"]["motion_count"]
        ),
        "completed_iterations": int(config["num_iterations"]),
        "model": str(output_dir / "model.pt"),
        "config": str(output_dir / "diffusion_config.yaml"),
    }
    atomic_json(result_path, result)
    del train, validation, model
    torch.cuda.empty_cache()
    return result


def per_motion_means(values: np.ndarray, ids: np.ndarray) -> dict[int, float]:
    return {
        int(identifier): float(values[ids == identifier].mean())
        for identifier in sorted(set(ids.tolist()))
    }


def pairwise_probability_lower(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(left[:, None] < right[None, :]))


def score_motion_disjoint(
    output_root: Path, manifest: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    score_root = output_root / "motion_disjoint_score"
    result_path = score_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if score_root.exists():
        raise FileExistsError(f"incomplete task score: {score_root}")
    score_root.mkdir(parents=True)
    arrays = {
        task: load_record_array(manifest["arrays"][f"{task}_test"])
        for task in TASK_ROOTS
    }
    ids = {
        task: np.load(
            manifest["arrays"][f"{task}_test"]["motion_ids_path"],
            allow_pickle=False,
        )
        for task in TASK_ROOTS
    }
    energies: dict[str, dict[str, np.ndarray]] = {}
    for prior_task in TASK_ROOTS:
        model = load_prior(output_root / "priors" / prior_task, device)
        energies[prior_task] = {
            data_task: raw_energy(model, values, device)
            for data_task, values in arrays.items()
        }
        del model
        torch.cuda.empty_cache()
    prior_specific = {
        "carry_prior_pairwise_own_lower": pairwise_probability_lower(
            energies["carry"]["carry"], energies["carry"]["kick"]
        ),
        "kick_prior_pairwise_own_lower": pairwise_probability_lower(
            energies["kick"]["kick"], energies["kick"]["carry"]
        ),
    }
    margins = {
        task: energies["kick"][task] - energies["carry"][task]
        for task in TASK_ROOTS
    }
    motion_margin = {
        task: per_motion_means(margins[task], ids[task]) for task in TASK_ROOTS
    }
    carry_correct = sum(value > 0.0 for value in motion_margin["carry"].values())
    kick_correct = sum(value < 0.0 for value in motion_margin["kick"].values())
    motion_total = len(motion_margin["carry"]) + len(motion_margin["kick"])
    motion_correct = carry_correct + kick_correct
    prior_gate = all(value >= 0.75 for value in prior_specific.values())
    classification_gate = motion_correct / motion_total >= 0.80
    result = {
        "protocol": "sugar_taskwide_official_tinymdm_motion_disjoint_score_v1",
        "passed": bool(prior_gate and classification_gate),
        "prior_specific_preferences": prior_specific,
        "task_energy_margin_kick_minus_carry": {
            task: {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "preferred_window_fraction_correct": float(
                    np.mean(values > 0.0) if task == "carry" else np.mean(values < 0.0)
                ),
            }
            for task, values in margins.items()
        },
        "motion_level": {
            "carry_correct": carry_correct,
            "carry_total": len(motion_margin["carry"]),
            "kick_correct": kick_correct,
            "kick_total": len(motion_margin["kick"]),
            "overall_accuracy": motion_correct / motion_total,
            "margins": {
                task: {str(key): value for key, value in by_id.items()}
                for task, by_id in motion_margin.items()
            },
        },
        "checks": {
            "entire_motion_disjoint_split": True,
            "official_tinymdm_and_esm_sds": True,
            "prior_specific_pairwise_gate": prior_gate,
            "motion_level_classification_gate": classification_gate,
        },
    }
    for prior_task, by_data in energies.items():
        for data_task, values in by_data.items():
            atomic_npy(score_root / f"{prior_task}__{data_task}.npy", values)
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "score", "all"))
    parser.add_argument("--task", choices=tuple(TASK_ROOTS))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_relative_to((ROOT / "experiments").resolve()):
        raise ValueError("outputs must stay inside ignored experiments/")
    device = torch.device(args.device)
    require_compute_gpu(device)
    manifest = prepare_dataset(output_root, device)
    if args.command == "prepare":
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "train":
        if args.task is None:
            raise ValueError("train requires --task")
        print(
            json.dumps(
                train_task_prior(output_root, manifest, args.task, device),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "score":
        score_motion_disjoint(output_root, manifest, device)
    else:
        for task in TASK_ROOTS:
            train_task_prior(output_root, manifest, task, device)
        score_motion_disjoint(output_root, manifest, device)


if __name__ == "__main__":
    main()
