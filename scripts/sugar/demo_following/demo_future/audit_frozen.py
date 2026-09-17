"""Audit frozen serious components before adding future supervision.

No optimization or simulator runs. Tests causal history dependence, selected
reference dependence and target distinguishability on motion-disjoint data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from .data import ROOT, BASE, PARENT, OUTPUT, CONDITIONS, OFFSETS, write_json

sys.path.insert(0, str(ROOT / "scripts/sugar/demo_reward"))
sys.path.insert(0, str(ROOT / "scripts/sugar/smp"))
sys.path.insert(0, str(ROOT / "MimicKit/mimickit"))
from train_actual_contact_event_predictor import model_from_normalization
from run_conditional_taskwide_tinymdm import load_shared_prior, conditional_raw_energy
# Import official dependencies before main, so the existing import-only
# bootstrap cannot replay evaluation after a partial package import failure.
from learning.tinymdm.tinymdm_model import TinyMDMModel

CHECKPOINT = BASE / "phase_aware_event_predictor_formal_seed271303_v1/best.pt"
MODES = ("full", "zero_demo", "other_motion_history", "zero_history")


def history_partners(task, source, phase):
    partners = np.empty(len(task), dtype=np.int64)
    for task_id in np.unique(task):
        ids = np.unique(source[task == task_id])
        if len(ids) < 2:
            raise ValueError("History swap requires a second same-task motion")
        for index, source_id in enumerate(ids):
            rows = np.flatnonzero((task == task_id) & (source == source_id))
            pool = np.flatnonzero((task == task_id) & (source == ids[(index + 1) % len(ids)]))
            partners[rows] = pool[np.argmin(np.abs(phase[rows, None] - phase[pool]), axis=1)]
    if np.any(source == source[partners]) or np.any(task != task[partners]):
        raise ValueError("Invalid same-task different-motion history pairing")
    return partners


def motion_macro(values, task, source):
    rows = []
    for task_id, source_id in sorted(set(zip(task.tolist(), source.tolist()))):
        mask = (task == task_id) & (source == source_id)
        rows.append({"task": int(task_id), "source": int(source_id),
                     "mean": float(np.mean(values[mask]))})
    return {"mean": float(np.mean([x["mean"] for x in rows])), "motions": rows}


def condition_margin_summary(target, prediction, task, source):
    result = {}
    for condition in (1, 2, 3):
        truth_gap = target[:, condition].mean(axis=-1) - target[:, 0].mean(axis=-1)
        informative = np.abs(truth_gap) >= 0.01
        row = {"informative_rows": int(informative.sum()), "total_rows": len(truth_gap),
               "oracle_correct_lower_fraction": float(np.mean(truth_gap > 0.01)),
               "oracle_gap_motion_macro": motion_macro(truth_gap, task, source)}
        if prediction is not None:
            gap = prediction[:, condition].mean(axis=-1) - prediction[:, 0].mean(axis=-1)
            agree = (np.sign(gap) == np.sign(truth_gap)).astype(np.float32)
            row["gap_absolute_error_motion_macro"] = motion_macro(np.abs(gap - truth_gap), task, source)
            row["sign_agreement_informative"] = (
                motion_macro(agree[informative], task[informative], source[informative])
                if informative.any() else None)
        result[CONDITIONS[condition]] = row
    return result


@torch.no_grad()
def evaluate_split(model, split, output, device):
    directory = OUTPUT / split
    with np.load(directory / "routing.npz", allow_pickle=False) as z:
        route = {k: z[k] for k in z.files}
    policy = np.load(PARENT / split / "policy_prefix.npy", mmap_mode="r")
    policy = np.array(policy[route["parent_base_row"]], copy=True)
    bank = torch.as_tensor(np.load(directory / "demo_bank.npy"), device=device)
    targets = np.load(directory / "target_mismatch.npy")
    scale = model.target_scale.cpu().numpy()
    transformed = np.log1p(targets / scale)
    task, source = route["base_task"], route["base_source_motion_id"]
    phase = route["horizon_phase"][:, 0]
    partner = history_partners(task, source, phase)
    predictions = {}
    n = len(policy)
    for mode in MODES:
        prediction = np.empty((n, 4, 13), dtype=np.float32)
        for begin in range(0, n * 4, 128):
            rows = np.arange(begin, min(begin + 128, n * 4))
            base, condition = rows // 4, rows % 4
            history = policy[partner[base] if mode == "other_motion_history" else base]
            history_tensor = torch.as_tensor(history, device=device)
            if mode == "zero_history":
                history_tensor = model.state_mean.expand_as(history_tensor)
            demo = bank[torch.as_tensor(route["selected_demo"][base, condition], device=device, dtype=torch.long)]
            values = model(policy_prefix=history_tensor,
                           selected_demo_condition=demo,
                           selected_demo_phase=torch.as_tensor(phase[base], device=device),
                           zero_demo=mode == "zero_demo")["mean_log1p_scaled"]
            if not torch.isfinite(values).all():
                raise RuntimeError("Nonfinite frozen prediction")
            prediction[base, condition] = values.cpu().numpy()
        predictions[mode] = prediction
        print(json.dumps({"split": split, "mode_completed": mode, "base_rows": n}), flush=True)
    statistics = {}
    for mode, pred in predictions.items():
        # Original checkpoint was not trained on reversed references: report
        # those separately, keeping the original three-condition metric intact.
        error = np.abs(pred - transformed[:, :, 0])
        statistics[mode] = {
            "original_conditions_mae": motion_macro(error[:, :3].mean(axis=(1, 2)), task, source),
            "reversed_mae": motion_macro(error[:, 3].mean(axis=1), task, source),
            "condition_margins": condition_margin_summary(transformed[:, :, 0], pred, task, source),
        }
    label_information = {
        str(offset): condition_margin_summary(transformed[:, :, h], None, task, source)
        for h, offset in enumerate(OFFSETS)
    }
    future_change = {
        str(offset): motion_macro(np.abs(transformed[:, :, h] - transformed[:, :, 0]).mean(axis=(1, 2)), task, source)
        for h, offset in enumerate(OFFSETS) if h > 0
    }
    np.savez(output / f"{split}_predictions.npz", **predictions, history_partner=partner)
    return {"base_rows": n, "motion_count": len(set(zip(task.tolist(), source.tolist()))),
            "history_swap_max_phase_difference": float(np.max(np.abs(phase - phase[partner]))),
            "modes": statistics, "oracle_label_information": label_information,
            "future_target_change_from_h0": future_change}


@torch.no_grad()
def audit_official_prior(device):
    prior_root = ROOT / "experiments/demo_following/conditional_taskwide_smp_v1"
    prior = load_shared_prior(prior_root, device)
    if not isinstance(prior, TinyMDMModel):
        raise TypeError("Expected the official MimicKit model")
    prior.requires_grad_(False)
    sample = np.load(ROOT / "experiments/demo_following/taskwide_smp_v1/dataset/carry_validation.npy", mmap_mode="r")[:8]
    # Only exercise the released scoring API, not a new latent interpretation.
    energies = {str(c): conditional_raw_energy(prior, np.array(sample, copy=True), c, device).tolist()
                for c in (0, 1)}
    if not all(np.isfinite(value).all() for value in energies.values()):
        raise RuntimeError("Official prior returned nonfinite energies")
    return {"strict_checkpoint_loaded": True, "class": type(prior).__module__ + "." + type(prior).__name__,
            "trainable_parameters": sum(p.numel() for p in prior.parameters() if p.requires_grad),
            "parameter_count_including_ema": sum(p.numel() for p in prior.parameters()),
            "conditioning": "Carry/Kick class only; no selected-demo latent API",
            "eight_validation_window_energies": energies,
            "scope": "API and checkpoint compatibility, not renewed generalization or policy evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT.parent / "frozen_audit")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID") or socket.gethostname().startswith(("login", "mgmtserver")):
        raise RuntimeError("Run inside the retained compute step")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest = json.loads((OUTPUT / "MANIFEST.json").read_text())
    if not manifest["passed"]:
        raise RuntimeError("Dataset outcome checks failed")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(271912)
    device = torch.device("cuda:0")
    model = model_from_normalization(PARENT / "NORMALIZATION.npz", device)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 11_386_010:
        raise RuntimeError("Existing full predictor architecture changed")
    model.eval().requires_grad_(False)
    prior = audit_official_prior(device)
    write_json(args.output / "OFFICIAL_COMPONENTS.json", prior)
    results = {split: evaluate_split(model, split, args.output, device) for split in ("validation", "test")}
    checks = {}
    for split, result in results.items():
        values = result["modes"]
        full = values["full"]["original_conditions_mae"]["mean"]
        checks[split + "_uses_demo"] = values["zero_demo"]["original_conditions_mae"]["mean"] > full * 1.02
        checks[split + "_uses_causal_history"] = values["other_motion_history"]["original_conditions_mae"]["mean"] > full * 1.02
    result = {"protocol": "sugar_frozen_demo_future_audit_v1", "execution_completed": True,
              "optimizer_updates": 0, "parameter_count": parameter_count,
              "checkpoint": str(CHECKPOINT), "official_prior": prior,
              "checks": checks, "passed": all(checks.values()), "splits": results,
              "next_action": ("matched_original_vs_multihorizon_predictor_experiment" if all(checks.values())
                              else "inspect_demo_only_or_history_shortcut_before_predictor_training"),
              "scope": "Frozen predictor and label audit; no counterfactual rollout or policy-following success"}
    write_json(args.output / "RESULT.json", result)
    print(json.dumps({"checks": checks, "next_action": result["next_action"]}), flush=True)


if __name__ == "__main__":
    main()
