#!/usr/bin/env python3
"""Train one official CondTinyMDM on motion-disjoint Carry/Kick motions.

The shared checkpoint and shared normalizer make Carry-vs-Kick ESM/SDS energy
directly comparable.  Class labels select the semantic condition; the model is
MimicKit's released ``CondTinyStableMotionDiTModel``, not a local surrogate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from audit_cross_skill_recovery_tinymdm import (  # noqa: E402
    load_feature_complete_trace,
    outcome_labels,
    pairwise_probability_lower,
    summarize_arm,
)
from run_selected_demo_tinymdm import (  # noqa: E402
    DIFFUSION_STEPS,
    MIMICKIT_PYTHON,
    PINNED_MIMICKIT_COMMIT,
    SCORE_SEEDS,
    SINGLE_CLIP_CONFIG,
    atomic_json,
    atomic_npy,
    atomic_torch,
    fix_seed,
    require_compute_gpu,
)
from sugar_g1_box_schema import FEATURE_DIM, WINDOW_SIZE  # noqa: E402


DATASET_ROOT = ROOT / "experiments/demo_following/taskwide_smp_v1/dataset"
OUTPUT_ROOT = ROOT / "experiments/demo_following/conditional_taskwide_smp_v1"
TRAIN_SEED = 160827
CLASS_IDS = {"carry": 0, "kick": 1}
EXPECTED_PARAMETER_COUNT = 2_836_864


def load_dataset(manifest: dict[str, Any], task: str, split: str) -> np.ndarray:
    record = manifest["arrays"][f"{task}_{split}"]
    array = np.load(record["path"], allow_pickle=False)
    if list(array.shape) != record["shape"] or array.dtype != np.float32:
        raise ValueError(f"dataset drift: {record['path']}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite dataset: {record['path']}")
    return array


def conditional_config(output_dir: Path) -> dict[str, Any]:
    env_config = {
        "global_obs": False,
        "root_height_obs": True,
        "pose_termination": False,
        "enable_phase_obs": False,
        "enable_tar_obs": False,
        "disc_dof_vel_obs": False,
        "num_disc_obs_steps": WINDOW_SIZE,
    }
    env_path = output_dir / "env_config.yaml"
    env_path.write_text(yaml.safe_dump(env_config, sort_keys=False), encoding="utf-8")
    config = dict(SINGLE_CLIP_CONFIG)
    config.update(
        {
            "arch_name": "CondDiT",
            "num_class": 2,
            "cfg_dropout": 0.1,
            "env_config": str(env_path),
        }
    )
    (output_dir / "diffusion_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return config


@torch.no_grad()
def conditional_validation_loss(
    model: Any,
    windows: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
) -> float:
    training = model.training
    model.eval()
    devices = [windows.device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        normalized = model.normalize(windows).reshape(windows.shape[0], -1)
        value = float(model(normalized, class_labels=labels).item())
    model.train(training)
    return value


def train_shared_prior(
    output_root: Path, manifest: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    output_dir = output_root / "prior"
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("passed") is not True or result.get("completed_iterations") != 50_000:
            raise RuntimeError(f"invalid conditional prior: {output_dir}")
        return result
    if output_dir.exists():
        raise FileExistsError(f"incomplete conditional prior: {output_dir}")
    output_dir.mkdir(parents=True)
    config = conditional_config(output_dir)
    train_arrays = [load_dataset(manifest, task, "train") for task in CLASS_IDS]
    validation_arrays = [
        load_dataset(manifest, task, "validation") for task in CLASS_IDS
    ]
    train_np = np.concatenate(train_arrays, axis=0)
    validation_np = np.concatenate(validation_arrays, axis=0)
    train_labels_np = np.concatenate(
        [
            np.full(array.shape[0], CLASS_IDS[task], dtype=np.int64)
            for task, array in zip(CLASS_IDS, train_arrays, strict=True)
        ]
    )
    validation_labels_np = np.concatenate(
        [
            np.full(array.shape[0], CLASS_IDS[task], dtype=np.int64)
            for task, array in zip(CLASS_IDS, validation_arrays, strict=True)
        ]
    )
    train = torch.as_tensor(train_np, device=device)
    train_labels = torch.as_tensor(train_labels_np, device=device)
    validation = torch.as_tensor(validation_np, device=device)
    validation_labels = torch.as_tensor(validation_labels_np, device=device)
    del train_np, validation_np, train_arrays, validation_arrays

    fix_seed(TRAIN_SEED)
    sys.path.insert(0, str(MIMICKIT_PYTHON))
    from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: PLC0415

    model = TinyMDMModel(config, device).to(device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"official conditional TinyMDM parameters {parameter_count}")
    stat_generator = torch.Generator(device=device).manual_seed(TRAIN_SEED + 1)
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
    sample_generator = torch.Generator(device=device).manual_seed(TRAIN_SEED + 2)
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
        loss = model(
            normalized, class_labels=train_labels.index_select(0, indices)
        )
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
                "motion_disjoint_validation_loss": conditional_validation_loss(
                    model,
                    validation,
                    validation_labels,
                    TRAIN_SEED + 3,
                ),
                "gradient_norm_before_clip": float(gradient_norm),
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)
            loss_sum = 0.0
            model.train()
    model.eval()
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("non-finite conditional TinyMDM endpoint")
    atomic_torch(output_dir / "model.pt", model.state_dict())
    result = {
        "protocol": "sugar_conditional_taskwide_official_tinymdm_v1",
        "passed": True,
        "seed": TRAIN_SEED,
        "mimickit_commit": PINNED_MIMICKIT_COMMIT,
        "official_architecture": "MimicKit CondTinyStableMotionDiTModel",
        "class_ids": CLASS_IDS,
        "cfg_dropout": 0.1,
        "parameter_count": parameter_count,
        "shared_normalizer": True,
        "train_windows": int(train.shape[0]),
        "train_motion_count": 160,
        "validation_motion_count": 20,
        "completed_iterations": int(config["num_iterations"]),
        "model": str(output_dir / "model.pt"),
        "config": str(output_dir / "diffusion_config.yaml"),
    }
    atomic_json(result_path, result)
    del train, train_labels, validation, validation_labels, model
    torch.cuda.empty_cache()
    return result


def load_shared_prior(output_root: Path, device: torch.device) -> Any:
    output_dir = output_root / "prior"
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    if (
        result.get("protocol") != "sugar_conditional_taskwide_official_tinymdm_v1"
        or result.get("passed") is not True
        or result.get("completed_iterations") != 50_000
    ):
        raise RuntimeError("unadmitted conditional task-wide prior")
    config = yaml.safe_load(
        (output_dir / "diffusion_config.yaml").read_text(encoding="utf-8")
    )
    config["env_config"] = str(output_dir / "env_config.yaml")
    sys.path.insert(0, str(MIMICKIT_PYTHON))
    from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: PLC0415

    model = TinyMDMModel(config, device).to(device)
    model.load_state_dict(
        torch.load(output_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def conditional_raw_energy(
    model: Any,
    array: np.ndarray,
    class_id: int,
    device: torch.device,
    chunk_size: int = 256,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, array.shape[0], chunk_size):
        windows = torch.as_tensor(array[start : start + chunk_size], device=device)
        normalized = model.normalize(windows).reshape(windows.shape[0], -1)
        labels = torch.full(
            (windows.shape[0],), class_id, dtype=torch.long, device=device
        )
        repeats = []
        for seed in SCORE_SEEDS:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            energy = model.ESM_SDS_loss(
                normalized,
                t_lst=list(DIFFUSION_STEPS),
                class_labels=labels,
            ).mean(dim=-1)
            repeats.append(energy.cpu().numpy())
        outputs.append(np.mean(np.stack(repeats), axis=0).astype(np.float32))
    result = np.concatenate(outputs)
    if result.shape != (array.shape[0],) or not np.isfinite(result).all():
        raise RuntimeError("invalid conditional ESM/SDS energy")
    return result


def motion_ids(manifest: dict[str, Any], task: str) -> np.ndarray:
    return np.load(
        manifest["arrays"][f"{task}_test"]["motion_ids_path"],
        allow_pickle=False,
    )


def score_motion_disjoint(
    output_root: Path,
    manifest: dict[str, Any],
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    output = output_root / "motion_disjoint_score"
    result_path = output / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    margins: dict[str, np.ndarray] = {}
    by_motion: dict[str, dict[str, float]] = {}
    energies: dict[str, dict[str, np.ndarray]] = {}
    correct = 0
    total = 0
    for task in CLASS_IDS:
        windows = load_dataset(manifest, task, "test")
        task_energies = {
            condition: conditional_raw_energy(
                model, windows, class_id, device
            )
            for condition, class_id in CLASS_IDS.items()
        }
        energies[task] = task_energies
        margin = task_energies["kick"] - task_energies["carry"]
        margins[task] = margin
        ids = motion_ids(manifest, task)
        by_motion[task] = {
            str(identifier): float(margin[ids == identifier].mean())
            for identifier in sorted(set(ids.tolist()))
        }
        expected_positive = task == "carry"
        correct += sum((value > 0.0) == expected_positive for value in by_motion[task].values())
        total += len(by_motion[task])
        atomic_npy(output / f"{task}_carry_condition_energy.npy", task_energies["carry"])
        atomic_npy(output / f"{task}_kick_condition_energy.npy", task_energies["kick"])
    accuracy = correct / total
    result = {
        "protocol": "sugar_conditional_taskwide_tinymdm_motion_disjoint_score_v1",
        "passed": accuracy >= 0.80,
        "motion_level_accuracy": accuracy,
        "motion_level_correct": correct,
        "motion_level_total": total,
        "motion_margins_kick_minus_carry": by_motion,
        "window_level": {
            task: {
                "margin_mean": float(values.mean()),
                "correct_condition_preferred_fraction": float(
                    np.mean(values > 0.0) if task == "carry" else np.mean(values < 0.0)
                ),
            }
            for task, values in margins.items()
        },
        "checks": {
            "one_shared_checkpoint": True,
            "one_shared_normalizer": True,
            "matched_noise_for_both_conditions": True,
            "entire_motion_disjoint_test": True,
            "official_mimickit_cond_dit_and_esm_sds": True,
        },
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def causal_safety_audit(
    energy: np.ndarray,
    raw_labels: dict[str, np.ndarray],
    outcomes: dict[str, np.ndarray],
) -> dict[str, Any]:
    safe = outcomes["safe_kick"]
    fall = outcomes["physical_fall"]
    early_profile = energy[:50].mean(axis=0)
    early_probability = pairwise_probability_lower(
        early_profile[safe], early_profile[fall]
    )
    root = raw_labels["robot_root_state_w"]
    root_loss = root[0, :, 2][None, :] - root[:, :, 2]
    prefall: list[dict[str, Any]] = []
    for profile in np.flatnonzero(fall):
        onset = int(np.flatnonzero(root_loss[:, profile] >= 0.35)[0])
        last_window = onset - WINDOW_SIZE
        if last_window < 0:
            prefall.append(
                {"profile": int(profile), "fall_onset_step": onset, "available": False}
            )
            continue
        first_window = max(0, last_window - 19)
        fall_value = float(energy[first_window : last_window + 1, profile].mean())
        safe_values = energy[first_window : last_window + 1, safe].mean(axis=0)
        prefall.append(
            {
                "profile": int(profile),
                "fall_onset_step": onset,
                "window_start": first_window,
                "window_end": last_window,
                "fall_energy": fall_value,
                "safe_energy_mean_same_clock": float(safe_values.mean()),
                "safe_lower_probability": float(np.mean(safe_values < fall_value)),
                "available": True,
            }
        )
    available = [row for row in prefall if row["available"]]
    prefall_gate = bool(available) and all(
        float(row["safe_lower_probability"]) >= 0.65 for row in available
    )
    return {
        "early_first_50_pairwise_probability_safe_energy_lower_than_fall": early_probability,
        "prefall_last_20_windows": prefall,
        "prefall_all_events_rank_above_65pct_safe": prefall_gate,
    }


def score_recovery(
    output_root: Path,
    model: Any,
    device: torch.device,
    arms: dict[str, Path],
    semantic_passed: bool,
) -> dict[str, Any]:
    output = output_root / "recovery_score"
    result_path = output / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    summaries: dict[str, Any] = {}
    causal_gates: list[bool] = []
    for arm, trace in arms.items():
        windows, metadata, raw_labels = load_feature_complete_trace(trace, device)
        flat = windows.transpose(1, 0, 2, 3).reshape(-1, WINDOW_SIZE, FEATURE_DIM)
        conditional: dict[str, np.ndarray] = {}
        for condition, class_id in CLASS_IDS.items():
            values = conditional_raw_energy(model, flat, class_id, device)
            conditional[condition] = values.reshape(
                windows.shape[1], windows.shape[0]
            ).T
        outcomes = outcome_labels(raw_labels)
        safety = causal_safety_audit(conditional["kick"], raw_labels, outcomes)
        causal_gates.append(bool(safety["prefall_all_events_rank_above_65pct_safe"]))
        summaries[arm] = {
            "feature_contract": metadata,
            **summarize_arm(
                conditional["carry"], conditional["kick"], outcomes
            ),
            "causal_safety_audit": safety,
        }
        atomic_npy(output / f"{arm}_carry_condition_energy.npy", conditional["carry"])
        atomic_npy(output / f"{arm}_kick_condition_energy.npy", conditional["kick"])
    signal_gate = semantic_passed and all(causal_gates)
    result = {
        "protocol": "sugar_conditional_taskwide_tinymdm_recovery_audit_v1",
        "arms": summaries,
        "checks": {
            "motion_disjoint_semantic_gate_passed": semantic_passed,
            "causal_prefall_ranking_gate_all_arms": all(causal_gates),
            "one_shared_checkpoint_and_normalizer": True,
            "outcome_labels_excluded_from_model_inputs": True,
        },
        "state_aware_controller_diagnostic_supported": signal_gate,
        "decision": (
            "conditional_prior_signal_admitted_for_matched_controller_diagnostic"
            if signal_gate
            else "conditional_prior_signal_rejected_before_controller_training"
        ),
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if not output_root.is_relative_to((ROOT / "experiments").resolve()):
        raise ValueError("output must stay under ignored experiments/")
    device = torch.device(args.device)
    require_compute_gpu(device)
    manifest = json.loads(
        (args.dataset_root.expanduser().resolve() / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    train_shared_prior(output_root, manifest, device)
    model = load_shared_prior(output_root, device)
    semantic = score_motion_disjoint(output_root, manifest, model, device)
    trace_root = (
        ROOT
        / "experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1/traces"
    )
    score_recovery(
        output_root,
        model,
        device,
        {
            "released_baseline": trace_root / "released_baseline/trace.npz",
            "unconstrained_update64": trace_root / "unconstrained_update64/trace.npz",
            "safety_update64": trace_root / "safety_update64/trace.npz",
        },
        bool(semantic["passed"]),
    )


if __name__ == "__main__":
    main()
