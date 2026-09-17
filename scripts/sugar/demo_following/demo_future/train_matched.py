"""Matched auxiliary-gradient experiment on the existing full predictor.

Both arms retain identical auxiliary capacity and labels. Only whether the
auxiliary gradient reaches the original trunk differs. Official SMP stays
frozen for later policy use and is not reinterpreted as a demo metric.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from .data import OUTPUT, PARENT, OFFSETS, write_json
from .audit_frozen import history_partners, motion_macro, condition_margin_summary
from .model import FutureSupervisionAdapter
from .residual_model import GeometryResidualAdapter

SEED = 271912


def load_split(split, device, dataset=OUTPUT, include_clock_rate=False):
    directory = Path(dataset) / split
    with np.load(directory / "routing.npz", allow_pickle=False) as z:
        route = {k: z[k] for k in z.files}
    policy = np.load(PARENT / split / "policy_prefix.npy", mmap_mode="r")
    result = {
        "route": route,
        "policy": torch.as_tensor(np.array(policy[route["parent_base_row"]], copy=True), device=device),
        "bank": torch.as_tensor(np.load(directory / "demo_bank.npy"), device=device),
        "target": torch.as_tensor(np.load(directory / "target_mismatch.npy"), device=device),
        "selected": torch.as_tensor(route["selected_demo"], dtype=torch.long, device=device),
        "phase": torch.as_tensor(route["horizon_phase"][:, 0], device=device),
    }
    if include_clock_rate:
        manifest = json.loads((OUTPUT.parent / "dataset_reference_roles_v2/MANIFEST.json").read_text())
        lengths = {(0 if r["task"] == "CarryBox" else 1, r["source"]): r["frames"] for r in manifest["source_geometry"]["records"]}
        duration = np.array([lengths[(int(t), int(s))] - 10 for t, s in zip(route["base_task"], route["base_source_motion_id"])], dtype=np.float64)
        phase = ((route["base_anchor_frame"][:, None] + np.array(OFFSETS)[None] + 1) / duration[:, None]).astype(np.float32)
        if not np.array_equal(phase, route["horizon_phase"]):
            raise RuntimeError("Known source clock does not reproduce declared phase routing")
        rate = torch.as_tensor(1 / duration, device=device, dtype=torch.float64)
        ticks = torch.round(result["phase"].double() / rate)
        reconstructed = ((ticks[:, None] + torch.tensor(OFFSETS, device=device)[None]) * rate[:, None]).float()
        if not np.array_equal(torch.round(reconstructed * 31).cpu().numpy(), np.rint(route["horizon_phase"] * np.float32(31))):
            raise RuntimeError("Causal phase/rate rounding changes reference windows")
        result["phase_rate"] = rate
    return result


def make_model(protocol, arm, device):
    if protocol.get("model_adapter") == "geometry_residual":
        return GeometryResidualAdapter(device, normalization_path=protocol["normalization_path"],
                                       initialization_endpoint=protocol["initialization_endpoint"],
                                       add_geometry_baseline=arm == "residual_geometry").to(device)
    return FutureSupervisionAdapter(device, detach_auxiliary=arm == "aux_detached",
                                    normalization_path=protocol.get("normalization_path")).to(device)


def batch_forward(model, data, rows, *, mode="full", partners=None):
    base, condition = rows // 4, rows % 4
    history_rows = partners[base] if mode == "other_motion_history" else base
    policy = data["policy"][history_rows]
    if mode == "zero_history":
        policy = model.base.state_mean.expand_as(policy)
    extra = {"selected_demo_phase_rate": data["phase_rate"][base]} if getattr(model, "uses_clock_rate", False) else {}
    return model(policy_prefix=policy,
                 selected_demo_condition=data["bank"][data["selected"][base, condition]],
                 selected_demo_phase=data["phase"][base], zero_demo=mode == "zero_demo", **extra)


@torch.no_grad()
def evaluate(model, split, output, device, dataset=OUTPUT):
    data = load_split(split, device, dataset, getattr(model, "uses_clock_rate", False))
    route = data["route"]
    task, source = route["base_task"], route["base_source_motion_id"]
    partners = torch.as_tensor(history_partners(task, source, route["horizon_phase"][:, 0]), device=device)
    n = len(task)
    truth = torch.log1p(data["target"] / model.base.target_scale).cpu().numpy()
    model.eval()
    predictions = {}
    metrics = {}
    analytical = np.empty((*truth.shape[:3], 4), dtype=np.float32) if getattr(model, "uses_clock_rate", False) else None
    for mode in ("full", "zero_demo", "other_motion_history", "zero_history"):
        pred = np.empty_like(truth)
        for begin in range(0, n * 4, 128):
            rows = torch.arange(begin, min(begin + 128, n * 4), device=device)
            forward = batch_forward(model, data, rows, mode=mode, partners=partners)
            values = forward["mean"]
            if analytical is not None and mode == "full":
                analytical.reshape(n * 4, 4, 4)[begin:begin + len(rows)] = forward["analytical_geometry"].cpu().numpy()
            if not torch.isfinite(values).all():
                raise RuntimeError("Nonfinite endpoint predictions")
            pred.reshape(n * 4, 4, 13)[begin:begin + len(rows)] = values.cpu().numpy()
        error = np.abs(pred - truth)
        mode_result = {}
        for task_id in (0, 1):
            mask = task == task_id
            mode_result[str(task_id)] = {
                "h0_mae": motion_macro(error[mask, :3, 0].mean(axis=(1, 2)), task[mask], source[mask])["mean"],
                "future_mae": motion_macro(error[mask, :3, 1:].mean(axis=(1, 2, 3)), task[mask], source[mask])["mean"],
                "reversed_future_mae": motion_macro(error[mask, 3, 1:].mean(axis=(1, 2)), task[mask], source[mask])["mean"],
                "horizons": {str(offset): condition_margin_summary(truth[mask, :, h], pred[mask, :, h], task[mask], source[mask])
                             for h, offset in enumerate(OFFSETS)},
            }
        metrics[mode] = mode_result
        predictions[mode] = pred
        print(json.dumps({"evaluation_split": split, "mode": mode, "complete": True}), flush=True)
    np.savez(output / f"{split}_predictions.npz", **predictions)
    if analytical is not None:
        np.save(output / f"{split}_analytical_geometry.npy", analytical, allow_pickle=False)
        audit = {}
        for task_id in (0, 1):
            mask = task == task_id
            macro = lambda v: np.mean([v[mask & (source == s)].mean(axis=0) for s in np.unique(source[mask])], axis=0).tolist()
            audit[str(task_id)] = {name: macro(np.abs(value[:, :3] - truth[:, :3, :, :4]).mean(axis=(1, 3)))
                                   for name, value in (("model", predictions["full"][..., :4]), ("analytical_geometry", analytical))}
        metrics["observable_geometry_control"] = audit
    hold_path = Path(dataset) / split / "current_hold_nine_channels.npy"
    if hold_path.exists():
        channels = [0, 1, 2, 3, 4, 5, 6, 7, 12]
        scale = model.base.target_scale.cpu().numpy()[channels]
        held = np.log1p(np.load(hold_path) / scale)
        selected_truth = truth[..., channels]
        audit = {}
        for task_id in (0, 1):
            mask = task == task_id
            macro = lambda v: np.mean([v[mask & (source == s)].mean(axis=0) for s in np.unique(source[mask])], axis=0).tolist()
            audit[str(task_id)] = {name: macro(np.abs(value[:, :3] - selected_truth[:, :3]).mean(axis=(1, 3)))
                                   for name, value in (("model", predictions["full"][..., channels]), ("privileged_current_hold", held))}
        metrics["current_state_information_control"] = audit
    return metrics


def train_arm(arm, root, protocol, device):
    out = root / arm
    out.mkdir(exist_ok=False)
    torch.manual_seed(SEED)
    model = make_model(protocol, arm, device)
    resume = None
    if protocol.get("parent_endpoint"):
        resume = torch.load(protocol["parent_endpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
    initial_path = root / "initial.pt"
    if not initial_path.exists():
        torch.save(model.state_dict(), initial_path)
    else:
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        if not all(torch.equal(value.cpu(), initial[name]) for name, value in model.state_dict().items()):
            raise RuntimeError("Matched initial weights differ")
    base_params = list(model.base.parameters())
    aux_params = list(model.future_heads.parameters())
    if hasattr(model, "clock_rate_projection"):
        aux_params += list(model.clock_rate_projection.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    parent_steps = 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        parent_steps = int(resume["steps"])
        # Directly verify actual restored optimizer tensors and clocks.
        restored = optimizer.state_dict()
        for key, state in resume["optimizer"]["state"].items():
            for name, value in state.items():
                actual = restored["state"][key][name]
                if torch.is_tensor(value) and not torch.equal(value, actual.cpu()):
                    raise RuntimeError("Optimizer restoration changed a moment/clock")
        if restored["param_groups"] != resume["optimizer"]["param_groups"]:
            raise RuntimeError("Optimizer restoration changed groups")
        write_json(out / "RESUME.json", {"parent_steps": parent_steps, "optimizer_moments_and_clocks_exact": True,
                                        "initial_state_identical_between_arms": True, "optimizer_updates_added": 0})
        del resume
    data = load_split("train", device, protocol.get("dataset", OUTPUT), getattr(model, "uses_clock_rate", False))
    paired = bool(protocol.get("paired_condition_batches", False))
    examples = data["target"].shape[0] if paired else data["target"].shape[0] * 4
    teacher = None
    if paired:
        teacher = FutureSupervisionAdapter(device, detach_auxiliary=False).to(device)
        saved_teacher = torch.load(protocol["nearest_teacher_endpoint"], map_location="cpu", weights_only=False)
        teacher.load_state_dict(saved_teacher["model"], strict=True)
        teacher.eval().requires_grad_(False)
        del saved_teacher
        encoded_train = torch.log1p(data["target"] / model.base.target_scale)
        gap_scale = (encoded_train[:, 1, 1:] - encoded_train[:, 0, 1:]).std(dim=0).clamp_min(0.05)
        del encoded_train
        np.save(out / "TRAIN_GAP_SCALE.npy", gap_scale.cpu().numpy(), allow_pickle=False)
    step = 0
    started = time.monotonic()
    model.train()
    with (out / "TRAIN_TRACE.jsonl").open("x") as trace:
        for epoch in range(protocol["epochs"]):
            generator = torch.Generator().manual_seed(SEED + epoch)
            order = torch.randperm(examples, generator=generator)
            batch_units = protocol["batch_size"] // 4 if paired else protocol["batch_size"]
            for begin in range(0, examples, batch_units):
                rows = order[begin:begin + batch_units].to(device)
                if paired:
                    rows = (rows[:, None] * 4 + torch.arange(4, device=device)).reshape(-1)
                torch.manual_seed(SEED + 10000 + parent_steps + step)
                optimizer.zero_grad(set_to_none=True)
                output = batch_forward(model, data, rows)
                losses = model.losses(output, data["target"][rows // 4, rows % 4])
                loss = losses[0] + losses[1:].mean()
                extra_metrics = {}
                if paired:
                    with torch.no_grad():
                        teacher_mean = batch_forward(teacher, data, rows)["mean"][:, 0]
                    anchor_loss = torch.square(output["mean"][:, 0] - teacher_mean).mean()
                    predicted = output["mean"].reshape(-1, 4, 4, 13)
                    truth = torch.log1p(data["target"][rows // 4, rows % 4] / model.base.target_scale).reshape(-1, 4, 4, 13)
                    gap_error = ((predicted[:, 1, 1:] - predicted[:, 0, 1:])
                                 - (truth[:, 1, 1:] - truth[:, 0, 1:])) / gap_scale
                    pair_loss = torch.square(gap_error).mean()
                    pair_weight = 1.0 if arm == "pair_supervised" else 0.0
                    loss = loss + protocol["nearest_anchor_weight"] * anchor_loss + pair_weight * pair_loss
                    extra_metrics = {"nearest_anchor_mse": float(anchor_loss.detach()),
                                     "same_task_standardized_gap_mse": float(pair_loss.detach()),
                                     "pair_weight": pair_weight}
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training objective")
                loss.backward()
                base_norm = torch.nn.utils.clip_grad_norm_(base_params, 1.0, error_if_nonfinite=True)
                aux_norm = torch.nn.utils.clip_grad_norm_(aux_params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                # This model is small enough for a direct finite-parameter check.
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise RuntimeError("Nonfinite parameter after optimizer update")
                row = {"step": step, "global_step": parent_steps + step, "epoch": epoch, "samples": len(rows),
                       "loss": float(loss.detach()), "horizon_nll": losses.detach().tolist(),
                       "base_gradient_norm": float(base_norm), "aux_gradient_norm": float(aux_norm),
                       "optimizer_applied": True, "elapsed_seconds": time.monotonic() - started,
                       **extra_metrics}
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                if step % 100 == 0:
                    print(json.dumps({"arm": arm, **row}), flush=True)
                step += 1
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "steps": parent_steps + step, "arm": arm, "protocol": protocol}, out / "endpoint.pt")
    del data
    evaluation = {split: evaluate(model, split, out, device, protocol.get("dataset", OUTPUT)) for split in ("validation", "test")}
    result = {"execution_completed": True, "arm": arm, "optimizer_steps": step,
              "base_parameter_count": sum(p.numel() for p in base_params),
              "auxiliary_parameter_count": sum(p.numel() for p in aux_params),
              "evaluation": evaluation, "elapsed_seconds": time.monotonic() - started}
    write_json(out / "RESULT.json", result)
    return result


def compare(detached, attached):
    checks, ratios = {}, {}
    for split in ("validation", "test"):
        for task in ("0", "1"):
            prefix = split + "_task" + task
            control = detached["evaluation"][split]["full"][task]
            treat = attached["evaluation"][split]["full"][task]
            future_ratio = treat["future_mae"] / max(control["future_mae"], 1e-12)
            h0_ratio = treat["h0_mae"] / max(control["h0_mae"], 1e-12)
            def margin(value):
                return np.mean([value["horizons"][str(offset)]["same_task_alternate"]["gap_absolute_error_motion_macro"]["mean"]
                                for offset in OFFSETS[1:]])
            margin_ratio = float(margin(treat) / max(margin(control), 1e-12))
            ratios[prefix] = {"future_mae": future_ratio, "h0_mae": h0_ratio, "same_task_margin_error": margin_ratio}
            checks[prefix + "_future_improved"] = future_ratio <= 0.95
            checks[prefix + "_h0_retained"] = h0_ratio <= 1.02
            checks[prefix + "_same_task_margin_improved"] = margin_ratio <= 0.95
            for mode in ("zero_demo", "other_motion_history"):
                value = attached["evaluation"][split][mode][task]["future_mae"]
                checks[prefix + "_" + mode + "_degrades"] = value > treat["future_mae"] * 1.02
    return checks, ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT.parent / "matched_future_supervision")
    parser.add_argument("--experiment", choices=("auxiliary", "paired", "canonical", "geometry-residual"), default="auxiliary")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID") or socket.gethostname().startswith(("login", "mgmtserver")):
        raise RuntimeError("Run inside the retained compute step")
    audit = json.loads((OUTPUT.parent / "frozen_audit/RESULT.json").read_text())
    if not audit["passed"]:
        raise RuntimeError("Frozen audit directs a shortcut diagnostic before matched training")
    preflight = json.loads((OUTPUT.parent / "PREFLIGHT.json").read_text())
    if not preflight["passed"] or preflight["optimizer_updates"] != 0:
        raise RuntimeError("Actual model gradient-path preflight failed")
    full_protocol = json.loads((OUTPUT.parent / "PROTOCOL.json").read_text())
    stage_name = {"auxiliary": "matched_auxiliary_gradient_experiment", "paired": "matched_same_task_gap_experiment", "canonical": "matched_corrected_canonical_experiment", "geometry-residual": "matched_geometry_residual_experiment"}[args.experiment]
    protocol = next(x for x in full_protocol["stages"] if x["name"] == stage_name)
    if args.experiment == "geometry-residual":
        residual_preflight = json.loads((OUTPUT.parent / "RESIDUAL_PREFLIGHT.json").read_text())
        if not residual_preflight["passed"] or residual_preflight["optimizer_updates"] != 0:
            raise RuntimeError("Residual full-model preflight failed")
    if args.experiment == "canonical":
        manifest = json.loads((Path(protocol["dataset"]) / "MANIFEST.json").read_text())
        if not manifest["execution_completed"] or not all(r["max_absolute_change"] < 1e-5 for r in manifest["geometry_checks"]):
            raise RuntimeError("Corrected canonical target audit is incomplete")
        # Real corrected TRAIN examples and the intact models, zero updates.
        torch.manual_seed(SEED)
        a = FutureSupervisionAdapter("cuda", detach_auxiliary=True, normalization_path=protocol["normalization_path"]).cuda().eval()
        b = FutureSupervisionAdapter("cuda", detach_auxiliary=False, normalization_path=protocol["normalization_path"]).cuda().eval()
        if not all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()):
            raise RuntimeError("Canonical initial states differ")
        examples = load_split("train", "cuda", protocol["dataset"])
        rows = torch.arange(16, device="cuda")
        va, vb = batch_forward(a, examples, rows), batch_forward(b, examples, rows)
        if not torch.equal(va["mean"], vb["mean"]) or not torch.isfinite(a.losses(va, examples["target"][rows // 4, rows % 4])).all():
            raise RuntimeError("Canonical full-model forward preflight failed")
        a.losses(va, examples["target"][rows // 4, rows % 4])[1:].sum().backward()
        b.losses(vb, examples["target"][rows // 4, rows % 4])[1:].sum().backward()
        norm = lambda model: sum(float(p.grad.detach().abs().sum()) for p in model.base.parameters() if p.grad is not None)
        if norm(a) != 0 or norm(b) <= 0:
            raise RuntimeError("Canonical full-model auxiliary gradient routing failed")
        print(json.dumps({"canonical_preflight_passed": True, "optimizer_updates": 0, "control_auxiliary_base_gradient_l1": norm(a), "treatment_auxiliary_base_gradient_l1": norm(b)}), flush=True)
        del a, b, va, vb, examples
    if args.experiment == "paired":
        diagnostic = json.loads((OUTPUT.parent / "matched_future_supervision/GRADIENT_CONFLICT_PROBE.json").read_text())
        if not diagnostic["execution_completed"] or diagnostic["optimizer_updates"] != 0:
            raise RuntimeError("Prior endpoint diagnostic incomplete")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "PROTOCOL.json", protocol)
    torch.set_num_threads(8)
    device = torch.device("cuda:0")
    arms = tuple(protocol["arms"]) if args.experiment == "geometry-residual" else (("pair_control", "pair_supervised") if args.experiment == "paired" else ("aux_detached", "aux_attached"))
    results = {arm: train_arm(arm, args.output, protocol, device) for arm in arms}
    checks, ratios = compare(results[arms[0]], results[arms[1]])
    if args.experiment == "geometry-residual":
        for split in ("validation", "test"):
            for task in ("0", "1"):
                value = results[arms[1]]["evaluation"][split]["observable_geometry_control"][task]
                ratio = np.array(value["model"]) / np.maximum(value["analytical_geometry"], 1e-12)
                ratios[split + "_task" + task]["geometry_over_analytical"] = ratio.tolist()
                checks[split + "_task" + task + "_geometry_h0_retained"] = bool(ratio[0] <= 1.02)
                for i in range(1, 4):
                    checks[split + "_task" + task + "_geometry_horizon" + str(i) + "_improved"] = bool(ratio[i] <= 0.95)
    passed = all(checks.values())
    result = {"execution_completed": True, "passed": passed, "checks": checks, "ratios": ratios,
              "next_action": protocol["if_pass"] if passed else protocol["if_fail"],
              "scope": "Exploratory matched predictor-supervision experiment; no policy, video-generation, or physics-success claim"}
    write_json(args.output / "RESULT.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
