"""Full paper model on one H200: GPU parameters, CPU AdamW, eight serial samples.

The immutable eight-slot schedule is retained verbatim. Its ``rank`` field is
a logical schedule slot here, never a claim of eight physical GPUs. Runtime
evidence records world_size=1, accumulation=8 and each physical execution slot.
No reduced model, alternate objective, or digest checks. The user-requested
fixed-noise overfit diagnostic has an isolated output and never starts formal training.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import defaultdict
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload, FullyShardedDataParallel as FSDP, FullStateDictConfig,
    MixedPrecision, ShardingStrategy, StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from .artifacts import emit_json_best_effort, write_json_atomic
from .config import PaperZeroWAMConfig, repaired_overfit_config
from .data import ScheduledSamples, read_jsonl
from .model import IFPHead, PaperMoTLayer, PaperZeroWAM
from .results import refresh_results_document_best_effort
from .train import (
    TRACE_SAMPLE_FIELDS, branch_name, decay_parameter_groups,
    expected_overfit_step_evidence,
    formal_training_decision, learning_rate, overfit_execution_decision,
    publish_runtime_milestone, retained_trace_prefix,
)


EXECUTION = {
    "world_size": 1, "packed_samples_per_rank": 1,
    "global_packed_samples": 8, "gradient_accumulation_steps": 8,
    "logical_schedule_slots": 8, "optimizer": "torch.optim.AdamW",
    "parameter_offload": "none", "optimizer_state_device": "cpu",
    "optimizer_master_parameters": "fp32_cpu_exact_gpu_copy_readback",
    "gradient_accumulation": "explicit_cpu_buffers_per_microbatch",
    "backward_loss_divisor": 8,
    "optimizer_execution": "native_adamw_parameter_serial_same_global_gradient",
    "update_audit_snapshot": "one_parameter_at_a_time",
}

CHECKPOINT_INTERVAL = 35  # Eight recovery points per unchanged 280-update epoch.
CPU_ACCUMULATION_IMPLEMENTATION = "matching_flat_storage_add_else_parameterwise_v1"

# Full-width parameter count; used for the memory plan below.
FULL_PARAMETER_COUNT = 10_680_751_069


@contextlib.contextmanager
def full_state_dict_context(model, **kwargs):
    """FSDP FULL_STATE_DICT when wrapped; a no-op for a bare module."""
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, *(
            [FullStateDictConfig(**kwargs)] if kwargs else []
        )):
            yield
    else:
        yield


def resolve_execution_plan(device_index: int = 0) -> dict:
    """Pick GPU parameter dtype from the card actually present.

    The original code hard-required an H200 and then kept parameters and
    gradients in FP32 with NO_SHARD, which needs ~80 GB before a single
    activation.  That fits a 141 GB H200 and nothing else.  An 80 GB card
    (H800/A100-80G/H100) needs BF16 resident parameters; the FP32 master
    weights and the AdamW moments already live on the CPU, so optimizer
    precision is unchanged -- only the resident forward/backward copy differs.
    """

    name = torch.cuda.get_device_name(device_index)
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    total_gib = total_bytes / 2**30
    fp32_resident_gib = FULL_PARAMETER_COUNT * 8 / 2**30  # params + grads
    bf16_resident_gib = FULL_PARAMETER_COUNT * 4 / 2**30
    # Leave headroom for activations, fragmentation and the flash workspace.
    if total_gib >= fp32_resident_gib + 24.0:
        param_dtype, resident = torch.float32, fp32_resident_gib
    elif total_gib >= bf16_resident_gib + 16.0:
        param_dtype, resident = torch.bfloat16, bf16_resident_gib
    else:
        raise RuntimeError(
            f"{name} has {total_gib:.1f} GiB; the full-width model needs at least "
            f"{bf16_resident_gib + 16.0:.1f} GiB resident. Use more GPUs or a larger card."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{name} does not support BF16 autocast")
    return {
        "device_name": name,
        "total_memory_gib": round(total_gib, 2),
        "gpu_parameter_dtype": str(param_dtype).replace("torch.", ""),
        "resident_parameter_gradient_gib": round(resident, 2),
        "optimizer_master_dtype": "float32_cpu",
        "_param_dtype": param_dtype,
    }


def requested_training_end(mode, stop_after_step):
    if stop_after_step is None:
        return 32 if mode == "overfit" else 4200
    if mode != "formal" or type(stop_after_step) is not int or stop_after_step != 700:
        raise ValueError("user-requested early endpoint must be formal step 700")
    return 700


def publish_requested_stop(root, config):
    checkpoint = (root / "latest_checkpoint").resolve()
    state = json.loads((checkpoint / "STATE.json").read_text())
    if single_gpu_checkpoint_step(state, config) != 700:
        raise ValueError("requested stop requires the complete step-700 checkpoint")
    records = read_jsonl(root / "TRAIN_TRACE.jsonl")
    if ([row["optimizer_step"] for row in records] != list(range(700))
            or not all(row.get("optimizer_applied") is True and row.get("amp_scaler_skipped") is False
                       and row.get("all_trainable_parameters_finite") is True
                       and row.get("gpu_parameter_master_readback_exact") is True for row in records)):
        raise ValueError("requested stop lacks exactly 700 complete real updates")
    path = root / "STEP700_STOP_RESULT.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if (existing.get("checkpoint_state") != state or existing.get("training_stopped") is not True
                or existing.get("completed_optimizer_steps") != 700):
            raise ValueError("existing requested-stop evidence belongs to another boundary")
        return
    write_json_atomic(path, {
        "protocol": "paper_zero_wam_user_step700_native_stop_v1",
        "reason": "user_requested_stop_at_700_then_evaluate",
        "training_stopped": True, "completed_optimizer_steps": 700,
        "formal_execution_complete": False, "original_formal_optimizer_budget": 4200,
        "automatic_training_continuation": False, "allocation_release_requested": False,
        "checkpoint_directory": str(checkpoint), "checkpoint_state": state,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "process_id": os.getpid(),
        "hash_checks": False,
    })


def allocation_deadline():
    output = subprocess.check_output(
        ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "--oneliner"],
        text=True, timeout=30)
    end = next((field.split("=", 1)[1] for field in output.split()
                if field.startswith("EndTime=")), None)
    if end is None or end in ("Unknown", "None", "N/A"):
        raise RuntimeError("retained allocation has no resolved end time")
    return datetime.fromisoformat(end).timestamp()


def deadline_checkpoint_needed(now, deadline, step_seconds):
    return deadline - now <= max(1200.0, 2.0 * step_seconds)


def next_checkpoint_directory(root):
    latest = root / "latest_checkpoint"
    if not latest.exists() and not latest.is_symlink():
        return root / "checkpoint_slot0"
    current = latest.resolve()
    if (not latest.is_symlink() or current.parent != root.resolve()
            or current.name not in ("checkpoint_slot0", "checkpoint_slot1")):
        raise ValueError("latest checkpoint is not a local rotating-slot symlink")
    return root / ("checkpoint_slot1" if current.name == "checkpoint_slot0" else "checkpoint_slot0")


def scalar(value):
    return float(value.detach().float().item())


def squared_norm(value):
    total = 0.0
    for part in value.detach().reshape(-1).split(1_048_576):
        total += float(part.double().square().sum())
    return total


def branch_norms(named_values):
    totals = dict.fromkeys(("video", "action", "ifp"), 0.0)
    for name, value in named_values:
        totals[branch_name(name)] += squared_norm(value)
    return {key: math.sqrt(value) for key, value in totals.items()}


def require_positive_finite(values, label):
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise FloatingPointError(f"inactive/non-finite {label}: {values}")


def matching_cpu_accumulator(entries, host, gradients, storage_members):
    """Return a flat destination only for identical, nonoverlapping layouts."""
    previous = gradients.get(entries[0][0])
    if previous is None or previous.device.type != "cpu" or previous.dtype != host.dtype:
        return None
    storage = previous.untyped_storage()
    identity = (previous.dtype, storage.data_ptr())
    if (storage.nbytes() != host.untyped_storage().nbytes()
            or storage_members.get(identity) != {name for name, _ in entries}):
        return None
    intervals = []
    for name, value in entries:
        retained = gradients.get(name)
        if (retained is None or not value.is_contiguous() or not retained.is_contiguous()
                or retained.device.type != "cpu" or retained.dtype != host.dtype
                or retained.shape != value.shape or retained.stride() != value.stride()
                or retained.storage_offset() != value.storage_offset()
                or retained.untyped_storage().data_ptr() != storage.data_ptr()):
            return None
        intervals.append((value.storage_offset(), value.storage_offset() + value.numel()))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        return None
    return torch.empty(0, dtype=host.dtype, device="cpu").set_(
        storage, 0, host.shape, (1,))


def accumulate_cpu_gradients(named_parameters, gradients, timing=None):
    """Copy each shared FSDP gradient storage once, retaining exact tensor views.

    This avoids a synchronized device-to-host transfer for every original
    parameter. Process one storage group at a time so temporary CPU memory is
    bounded by one FSDP block, not another complete model-gradient copy.
    When the retained CPU layout is identical, sum its complete flat storage
    once. Missing parameters, changed layouts and noncontiguous/overlapping
    views retain the original per-parameter addition path.
    Optional wall-clock counters observe the existing synchronous transfer and
    host addition separately; they introduce no tensor operation or CUDA fence.
    """
    transfer_seconds = 0.0
    addition_seconds = 0.0
    storage_members = defaultdict(set)
    for name, value in gradients.items():
        storage_members[(value.dtype, value.untyped_storage().data_ptr())].add(name)
    groups = {}
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach()
        storage = value.untyped_storage()
        key = (value.device, value.dtype, storage.data_ptr())
        groups.setdefault(key, []).append((name, value))
    for entries in groups.values():
        first = entries[0][1]
        storage = first.untyped_storage()
        count = storage.nbytes() // first.element_size()
        flat = torch.empty(0, dtype=first.dtype, device=first.device).set_(
            storage, 0, (count,), (1,))
        phase_started = time.perf_counter() if timing is not None else None
        # Accumulate in FP32 on the host even when the resident gradient is
        # BF16: summing eight microbatches in 8-bit mantissa would lose real
        # signal, and the AdamW master weights are FP32 anyway.
        host = flat.to(device="cpu", dtype=torch.float32, copy=True)
        if timing is not None:
            transfer_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
        accumulator = matching_cpu_accumulator(entries, host, gradients, storage_members)
        if accumulator is not None:
            accumulator.add_(host)
        else:
            for name, value in entries:
                view = host.as_strided(value.shape, value.stride(), value.storage_offset())
                if name in gradients:
                    gradients[name].add_(view)
                else:
                    gradients[name] = view
            del view
        if timing is not None:
            addition_seconds += time.perf_counter() - phase_started
        del flat, host, accumulator
    if timing is not None:
        for key, seconds in (("cpu_gradient_device_to_host_copy", transfer_seconds),
                             ("cpu_gradient_layout_check_and_addition", addition_seconds)):
            timing[key] = timing.get(key, 0.0) + seconds
    return len(groups)


def native_adamw_streamed_step(optimizer, named_parameters, gradients, clip_scale):
    """One global update using unmodified native AdamW on independent tensors.

    Global clipping is already computed over all eight samples. AdamW has no
    cross-parameter state: serial native calls preserve each tensor's exact
    step/moments/weight decay, while only one pre-update audit copy is live.
    The original optimizer groups are restored for checkpoint compatibility.
    No forward/backward or gradient recomputation occurs inside this boundary.
    """
    parameters = list(named_parameters)
    names = {id(parameter): name for name, parameter in parameters}
    original_groups = optimizer.param_groups
    update_sums = dict.fromkeys(("video", "action", "ifp"), 0.0)
    calls = 0
    try:
        for group in original_groups:
            for parameter in group["params"]:
                name = names[id(parameter)]
                if name not in gradients:
                    continue
                if parameter.device.type != "cpu":
                    raise RuntimeError("optimizer parameters must remain CPU-offloaded")
                parameter.grad = gradients.pop(name).mul_(clip_scale)
                before = parameter.detach().clone()
                optimizer.param_groups = [{**group, "params": [parameter]}]
                optimizer.step()
                calls += 1
                update_sums[branch_name(name)] += squared_norm(parameter.detach() - before)
                parameter.grad = None
                del before
    finally:
        optimizer.param_groups = original_groups
    finite = all(bool(torch.isfinite(parameter).all()) for _, parameter in parameters)
    return {key: math.sqrt(value) for key, value in update_sums.items()}, finite, calls


def prompt_gate(model, datasets, device, seed, loss_repeats=8):
    """Same eight full-width cases/conditions as the distributed overfit gate.

    ``loss_repeats`` averages each (case, condition) loss over that many
    independent (noise, t) draws.  With a single draw the gate ratio is a
    high-variance point estimate -- the same checkpoint measured video 1.2382
    at t=0.767 and 0.0487 at t=0.963 -- so prompt margins and endpoint ratios
    were dominated by which flow time happened to be sampled.  The draw seeds
    are derived from ``seed``, keeping the gate exactly reproducible.  The
    one-step connectivity probe stays single-draw: it only needs to show that
    swapping the prompt changes the generated future at all.
    """
    model.eval()
    names = ("loss", "video_loss", "action_loss", "ifp_loss")
    interventions = ("wrong_task", "reversed", "same_task_alternate")
    task_values = {task: defaultdict(list) for task in ("CarryBox", "KickBox")}
    connectivity = {task: defaultdict(list) for task in task_values}
    # FSDP CPU offload materializes parameter views during this gate. Those
    # views must remain normal tensors when the same model resumes training.
    with torch.no_grad():
        for logical_slot, dataset in enumerate(datasets):
            generated = {}
            for condition in ("matched", *interventions):
                batch = dataset.conditioned_sample(0, device, condition)
                task = batch["meta"]["task"]
                totals = {key: 0.0 for key in names}
                for repeat in range(loss_repeats):
                    draw_seed = seed + 1_000_003 * repeat
                    torch.manual_seed(draw_seed)
                    torch.cuda.manual_seed_all(draw_seed)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        losses = model(batch)
                    for key in names:
                        totals[key] += scalar(losses[key])
                    del losses
                task_values[task][condition].append(
                    {key: totals[key] / loss_repeats for key in names})
                torch.manual_seed(seed + 1)
                torch.cuda.manual_seed_all(seed + 1)
                probe = dict(batch, inference=True, video_inference_steps=1,
                             action_inference_steps=1, video_guidance_scale=1.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(probe)
                generated[condition] = {key: value.detach().cpu() for key, value in prediction.items()}
                del prediction, probe, batch
            for condition in interventions:
                connectivity[task][condition].append({
                    "predicted_future_mse_vs_matched": float((
                        generated[condition]["predicted_video_latents"].double()
                        - generated["matched"]["predicted_video_latents"].double()
                    ).square().mean()),
                    "predicted_action_mse_vs_matched": float((
                        generated[condition]["predicted_normalized_actions"].double()
                        - generated["matched"]["predicted_normalized_actions"].double()
                    ).square().mean()),
                })
            emit_json_best_effort({"stage": "overfit_prompt_gate", "logical_slot": logical_slot})
    def mean_rows(rows):
        return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    task_losses = {task: {condition: mean_rows(rows) for condition, rows in values.items()}
                   for task, values in task_values.items()}
    if any(len(rows) != 4 for values in task_values.values() for rows in values.values()):
        raise ValueError("overfit prompt gate requires four cases per task")
    conditions = {condition: mean_rows([task_losses[task][condition] for task in task_losses])
                  for condition in ("matched", *interventions)}
    margins = {task: {condition: {
        key: values[condition][key] - values["matched"][key] for key in names
    } for condition in interventions} for task, values in task_losses.items()}
    path = {task: {condition: mean_rows(rows) for condition, rows in values.items()}
            for task, values in connectivity.items()}
    prompt_passed = all(margins[task][condition][key] > 0
        for task in margins for condition in interventions for key in ("video_loss", "ifp_loss"))
    isolated = all(abs(margins[task][condition]["action_loss"]) <= 1e-7 * max(
        1.0, abs(task_losses[task]["matched"]["action_loss"]))
        for task in margins for condition in interventions)
    connected = all(value > 1e-12 for task in path.values()
                    for condition in task.values() for value in condition.values())
    return {"passed": prompt_passed and isolated and connected,
            "losses": conditions, "task_losses": task_losses,
            "task_margins_vs_matched": margins, "task_generated_connectivity": path,
            "prompt_loss_passed": prompt_passed, "teacher_action_isolated": isolated,
            "generated_future_to_action_path_active": connected,
            "interventions": list(interventions), "condition_count": 4,
            "loss_draws_per_condition": loss_repeats,
            "loss_forward_count": 32 * loss_repeats,
            "one_step_connectivity_forward_count": 32,
            "optimizer_updates_added": 0, "execution": EXECUTION}


def task_records(observations, config):
    result = {}
    for task in ("CarryBox", "KickBox"):
        rows = [row for row in observations if row["task"] == task]
        if len(rows) != 4:
            raise ValueError("each applied update must contain four samples per task")
        record = {"count": 4.0}
        for branch in ("video", "action"):
            count = sum(row["elements"][branch] for row in rows)
            numerator = sum(row["raw"][branch] * row["elements"][branch] for row in rows)
            record.update({f"{branch}_loss": numerator / count,
                           f"{branch}_loss_sum": numerator,
                           f"{branch}_loss_elements": count})
        counts = [sum(row["elements"]["ifp"][head] for row in rows) for head in range(4)]
        sums = [sum(row["raw"]["ifp"][head] * row["elements"]["ifp"][head]
                    for row in rows) for head in range(4)]
        means = [numerator / count if count else 0.0 for numerator, count in zip(sums, counts)]
        record.update(ifp_head_losses=means, ifp_loss_sums=sums, ifp_loss_elements=counts,
                      ifp_loss=sum(w * value for w, value in zip(config.ifp_weights, means)))
        result[task] = record
    return result


def save_checkpoint(model, optimizer, root, step, config):
    directory = next_checkpoint_directory(root)
    directory.mkdir(exist_ok=True)
    with full_state_dict_context(model, offload_to_cpu=True, rank0_only=True):
        state = model.state_dict()
    temporary = directory / "model_rank00.tmp"
    torch.save({"step": step, "model": state}, temporary)
    temporary.replace(directory / "model_rank00.pt")
    del state
    temporary = directory / "optimizer_rank00.tmp"
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, temporary)
    temporary.replace(directory / "optimizer_rank00.pt")
    write_json_atomic(directory / "STATE.json", {
        "protocol": "paper_zero_wam_single_gpu_checkpoint_v2",
        "completed_optimizer_steps": step, "architecture_parameter_count": config.expected_parameter_count,
        "completed_epochs": step // 280, "steps_in_current_epoch": step % 280,
        "optimizer_boundary_complete": True,
        "execution": EXECUTION, "model_and_schedule_config": config.as_dict(), "hash_checks": False,
        "cpu_accumulation_implementation": CPU_ACCUMULATION_IMPLEMENTATION,
    })
    temporary = root / ".latest_checkpoint.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(directory.name, target_is_directory=True)
    temporary.replace(root / "latest_checkpoint")


def single_gpu_checkpoint_step(state, config):
    if (state.get("execution") != EXECUTION or state.get("model_and_schedule_config") != json.loads(
            json.dumps(config.as_dict())) or state.get("protocol") != "paper_zero_wam_single_gpu_checkpoint_v2"
            or state.get("hash_checks") is not False
            or state.get("architecture_parameter_count") != config.expected_parameter_count):
        raise ValueError("single-GPU checkpoint execution/config mismatch")
    step = int(state["completed_optimizer_steps"])
    if (type(state["completed_optimizer_steps"]) is not int or not 0 < step <= 4200
            or state.get("optimizer_boundary_complete") is not True
            or state.get("completed_epochs") != step // 280
            or state.get("steps_in_current_epoch") != step % 280):
        raise ValueError("checkpoint is not a complete exact optimizer boundary")
    return step


def load_checkpoint(model, optimizer, root, config, master_pairs):
    directory = root / "latest_checkpoint"
    if not directory.exists() and not directory.is_symlink():
        return 0
    state = json.loads((directory / "STATE.json").read_text())
    step = single_gpu_checkpoint_step(state, config)
    payload = torch.load(directory / "model_rank00.pt", map_location="cpu", weights_only=False)
    if payload["step"] != step:
        raise ValueError("model checkpoint step mismatch")
    with full_state_dict_context(model):
        model.load_state_dict(payload["model"], strict=True)
    del payload
    with torch.no_grad():
        for _, gpu_parameter, master in master_pairs:
            master.copy_(gpu_parameter.detach().cpu())
    payload = torch.load(directory / "optimizer_rank00.pt", map_location="cpu", weights_only=False)
    if payload["step"] != step:
        raise ValueError("optimizer checkpoint step mismatch")
    optimizer.load_state_dict(payload["optimizer"])
    return step


def prepare_training_trace(log, completed_steps):
    """Retain committed updates, archiving any infrastructure-interrupted tail."""
    retained = retained_trace_prefix(log, completed_steps) if completed_steps else []
    prefix = "".join(retained)
    previous = log.read_text(encoding="utf-8") if log.exists() else ""
    if previous and previous != prefix:
        archive = log.with_name(f"{log.stem}.interrupted_{time.time_ns()}.jsonl")
        with archive.open("x", encoding="utf-8") as stream:
            stream.write(previous)
        write_json_atomic(archive.with_suffix(".record.json"), {
            "reason": "infrastructure_recovery_to_committed_optimizer_boundary",
            "complete_prior_trace": str(archive), "retained_optimizer_steps": completed_steps,
            "uncheckpointed_tail_reexecuted": True, "hash_checks": False,
        })
    log.write_text(prefix, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("overfit", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-terminal", action="store_true")
    parser.add_argument("--stop-after-step", type=int, choices=[700])
    parser.add_argument("--fixed-noise-overfit-diagnostic", action="store_true")
    parser.add_argument("--repaired-overfit", action="store_true")
    parser.add_argument("--resampled-noise-overfit", action="store_true",
                        help="resample noise and flow time every step/slot: the "
                             "generative overfit, whose renders are meaningful")
    parser.add_argument("--resume-repaired-preupdate", action="store_true")
    parser.add_argument("--save-overfit-optimizer", action="store_true",
                        help="retain endpoint AdamW state for the continuing overfit investigation")
    args = parser.parse_args()
    if args.save_overfit_optimizer and (args.mode != "overfit" or not args.fixed_noise_overfit_diagnostic):
        raise ValueError("optimizer retention is only supported for an isolated overfit diagnostic")
    config = repaired_overfit_config() if args.repaired_overfit else PaperZeroWAMConfig()
    if args.repaired_overfit and (args.mode != "overfit" or not args.fixed_noise_overfit_diagnostic):
        raise ValueError("correction is admitted only for the user-requested isolated overfit")
    from .overfit_diagnostic import (
        diagnostic_contract, matched_loss_probe, reduction_result,
        render_training_cases, training_noise_seed, validate_diagnostic_request,
        verify_first_forward,
        restored_conditioning_gradient_decision, validate_preupdate_recovery,
    )
    validate_diagnostic_request(args.fixed_noise_overfit_diagnostic, args.mode,
                                args.output_dir, config, args.resampled_noise_overfit)
    if args.resume_repaired_preupdate:
        if not args.repaired_overfit:
            raise ValueError("pre-update recovery requires the isolated repaired diagnostic")
        validate_preupdate_recovery(args.output_dir, config)
    if args.fixed_noise_overfit_diagnostic and args.output_dir.exists() and not args.resume_repaired_preupdate:
        raise RuntimeError("isolated diagnostic output already exists; refusing overwrite/replay")
    execution_end = requested_training_end(args.mode, args.stop_after_step)
    if args.stop_after_step is not None:
        existing_state = args.output_dir / "latest_checkpoint" / "STATE.json"
        if existing_state.exists():
            existing_step = single_gpu_checkpoint_step(json.loads(existing_state.read_text()), PaperZeroWAMConfig())
            if existing_step > execution_end:
                raise ValueError("checkpoint is beyond the requested stop boundary")
            if existing_step == execution_end:
                publish_requested_stop(args.output_dir.resolve(), PaperZeroWAMConfig())
                emit_json_best_effort({"requested_stop_reused": True, "optimizer_updates_added": 0})
                return
    terminal_name = "OVERFIT_RESULT.json" if args.mode == "overfit" else "FORMAL_TRAINING_RESULT.json"
    if args.verify_terminal or (args.output_dir / terminal_name).is_file():
        root = args.output_dir.resolve()
        name = "OVERFIT_RESULT.json" if args.mode == "overfit" else "FORMAL_TRAINING_RESULT.json"
        result = json.loads((root / name).read_text())
        schedule = config.resolved(config.output_root) / "schedule/TRAIN_SCHEDULE.jsonl"
        log = root / "TRAIN_TRACE.jsonl"
        if args.mode == "overfit":
            decision = overfit_execution_decision(log, schedule, 32, config,
                                                 execution_world_size=1, accumulation_steps=8)
            steps = 32
        else:
            formal = formal_training_decision(log, schedule, config,
                                             execution_world_size=1, accumulation_steps=8)
            if result.get("formal_decision") != formal or result.get("passed") is not formal["passed"]:
                raise ValueError("formal scientific terminal differs from complete trace")
            checks = {key: value for key, value in formal["checks"].items() if key != "both_tasks_progress"}
            decision = {"checks": checks, "passed": all(checks.values())}
            steps = 4200
        if (not decision["passed"] or result.get("execution_decision") != decision
                or result.get("protocol") != f"paper_zero_wam_sugar_{args.mode}_v1"
                or result.get("execution_completed") is not True
                or result.get("optimizer_steps") != steps
                or result.get("architecture_parameter_count") != config.expected_parameter_count
                or result.get("batch") != EXECUTION or result.get("hash_checks") is not False
                or not all(row.get("gpu_parameter_master_readback_exact") is True
                           and int(row.get("optimizer_native_parameter_calls", 0)) > 0
                           for row in read_jsonl(log))):
            raise ValueError("terminal is not an exact complete single-GPU execution")
        emit_json_best_effort({"terminal_reused": str(root / name), "optimizer_updates_added": 0})
        return
    if int(os.environ.get("WORLD_SIZE", "0")) != 1:
        raise RuntimeError("single-GPU trainer requires exactly one rank")
    if not os.environ.get("SLURM_STEP_ID") and not os.environ.get(
        "PZW_ALLOW_NON_SLURM"
    ):
        raise RuntimeError(
            "launch inside a retained srun step, or set PZW_ALLOW_NON_SLURM=1 "
            "when the scheduler is not present"
        )
    torch.cuda.set_device(0)
    execution_plan = resolve_execution_plan(0)
    gpu_param_dtype = execution_plan.pop("_param_dtype")
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    device = torch.device("cuda", 0)
    config.validate()
    if args.repaired_overfit:
        from .data import latent_materialization_admission_is_exact
        cache_root = config.resolved(config.latent_cache)
        audit = json.loads((cache_root / "OFFICIAL_FORWARD_AUDIT.json").read_text())
        original_stats = config.resolved(PaperZeroWAMConfig().latent_cache) / "ACTION_QUANTILES.json"
        action_stats_unchanged = (json.loads(original_stats.read_text()) ==
                                 json.loads((cache_root / "ACTION_QUANTILES.json").read_text()))
        if (not latent_materialization_admission_is_exact(cache_root, repaired=True)
                or not action_stats_unchanged
                or audit.get("passed") is not True or audit.get("official_layers") != 30
                or audit.get("mask_forward_backward_passed") is not True
                or audit.get("official_parameters") != 4_999_787_712
                or audit.get("inherited_text_projection_parameters") != 22_026_240):
            raise RuntimeError("repaired overfit requires completed official-forward and data checks")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    terminal = root / ("OVERFIT_RESULT.json" if args.mode == "overfit" else "FORMAL_TRAINING_RESULT.json")
    if terminal.exists():
        raise RuntimeError("completed scientific result is terminal; refusing replay")
    previous_config = root / "CONFIG.json"
    initial_path = root / "INITIAL_PROMPT_GATE.json"
    reuse_initial = args.mode == "overfit" and initial_path.exists() and previous_config.exists()
    if reuse_initial and json.loads(previous_config.read_text()).get(
            "model_and_schedule_config") != json.loads(json.dumps(config.as_dict())):
        raise ValueError("initial gate recovery requires identical model/data/noise configuration")
    write_json_atomic(root / "CONFIG.json", {
        "model_and_schedule_config": config.as_dict(), "execution": EXECUTION, "hash_checks": False,
        "cpu_accumulation_implementation": CPU_ACCUMULATION_IMPLEMENTATION,
    })
    diagnostic = (diagnostic_contract(config, args.resampled_noise_overfit)
                  if args.fixed_noise_overfit_diagnostic else None)
    if diagnostic:
        write_json_atomic(root / "DIAGNOSTIC_CONTRACT.json", diagnostic)
    started = time.perf_counter()
    def milestone(name, **evidence):
        publish_runtime_milestone(root, mode=args.mode, rank=0, world_size=1,
            device=device, milestone=name, started_at=started, execution=EXECUTION, **evidence)
    milestone("distributed_initialized")
    torch.manual_seed(config.noise_seed)
    torch.cuda.manual_seed_all(config.noise_seed)
    model = PaperZeroWAM.from_wan_pretrained(config, dtype=gpu_param_dtype)
    if model.parameter_count != config.expected_parameter_count:
        raise ValueError("full model parameter count changed")
    if gpu_param_dtype is not torch.float32:
        execution_plan.update(model.upcast_precision_critical_modules())
    milestone("full_width_model_constructed",
              architecture_parameter_count=model.parameter_count,
              execution_plan=execution_plan)
    # FSDP with NO_SHARD on a single rank performs no sharding, no gradient
    # reduction and no parameter offload -- it is a pass-through wrapper here.
    # It does, however, require every parameter inside a flattened unit to
    # share one dtype, which is incompatible with keeping Wan's FP32 numerical
    # boundary (norms / modulation / heads) under BF16 residency.  Use the
    # bare module in that case; the checkpoint helpers below handle both.
    fsdp_wrapped = gpu_param_dtype is torch.float32
    if fsdp_wrapped:
        model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({PaperMoTLayer, IFPHead}),
            sharding_strategy=ShardingStrategy.NO_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=MixedPrecision(param_dtype=None, reduce_dtype=torch.bfloat16),
            device_id=device, use_orig_params=True, limit_all_gathers=True)
    else:
        model = model.to(device)
    execution_plan["fsdp_no_shard_wrapper"] = fsdp_wrapped
    # FP32 CPU master weights and AdamW moments are unchanged regardless of the
    # resident GPU dtype, so optimizer arithmetic keeps full precision.
    master_pairs = [(name, parameter, torch.nn.Parameter(parameter.detach().float().cpu(),
                                                       requires_grad=parameter.requires_grad))
                    for name, parameter in model.named_parameters()]
    optimizer = torch.optim.AdamW(
        decay_parameter_groups(((n, m) for n, _, m in master_pairs), config),
        lr=config.peak_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
        weight_decay=config.weight_decay, foreach=False)
    milestone("gpu_resident_model_and_cpu_adamw_constructed")
    schedule = config.resolved(config.output_root) / "schedule/TRAIN_SCHEDULE.jsonl"
    datasets = [ScheduledSamples(schedule, config.resolved(config.latent_cache), slot, config,
                                 args.mode, 32) for slot in range(8)]
    # Eight logical slots share immutable CPU data; they are not eight workers.
    for dataset in datasets[1:]:
        dataset._latent_cpu_cache = datasets[0]._latent_cpu_cache
        dataset._action_cpu_cache = datasets[0]._action_cpu_cache
        dataset._action_shard_cpu_cache = datasets[0]._action_shard_cpu_cache
    for dataset in datasets:
        dataset._scheduled_action_selector_count = 160 if args.mode == "formal" else 8
    total = 32 if args.mode == "overfit" else 4200
    deadline = allocation_deadline() if args.mode == "formal" else None
    start = load_checkpoint(model, optimizer, root, config, master_pairs) if args.mode == "formal" else 0
    log = root / "TRAIN_TRACE.jsonl"
    prepare_training_trace(log, start)
    milestone("schedule_and_data_admitted", scheduled_optimizer_steps=total)
    # A fixed-noise diagnostic is *defined* by its single (noise, t), so its
    # gate stays single-draw and keeps the step-0 equivalence check.  Any run
    # that resamples during training needs an averaged gate to get a ratio that
    # reflects the model instead of one sampled flow time.
    gate_draws = 8 if args.resampled_noise_overfit else 1
    initial = (json.loads(initial_path.read_text()) if reuse_initial else
               prompt_gate(model, datasets, device, config.noise_seed + 9_999_991,
                           gate_draws)
               if args.mode == "overfit" else None)
    if initial is not None and not reuse_initial:
        write_json_atomic(root / "INITIAL_PROMPT_GATE.json", initial)
    if reuse_initial:
        milestone("identical_initial_gate_reused_after_infrastructure_failure")
    unseen_initial = None
    if diagnostic:
        if args.resume_repaired_preupdate:
            unseen_initial = json.loads((root / "UNSEEN_NOISE_INITIAL.json").read_text())
            milestone("identical_preupdate_probes_reused_zero_optimizer_updates")
        else:
            unseen_initial = matched_loss_probe(model, datasets, device, diagnostic["unseen_noise_seed"], gate_draws)
            write_json_atomic(root / "UNSEEN_NOISE_INITIAL.json", unseen_initial)
    model.train()
    first_forward_check = None
    for step in range(start, execution_end):
        step_started = time.perf_counter()
        timing = dict(data_loading=0.0, forward_backward=0.0, cpu_gradient_transfer_accumulation=0.0)
        metas = []
        for slot, dataset in enumerate(datasets):
            meta = {key: dataset.rows[step][key] for key in TRACE_SAMPLE_FIELDS}
            meta["prompt_enabled"] = args.mode == "overfit" or meta["prompt_enabled"]
            metas.append(meta)
        evidence = expected_overfit_step_evidence(metas, config)
        observations = []
        gradients = {}
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        lr = learning_rate(step, total, config, args.mode)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for slot, dataset in enumerate(datasets):
            phase_started = time.perf_counter()
            batch = dataset.sample(step, device)
            batch["loss_scales"] = evidence["loss_rank_evidence"][slot]["scales"]
            step_noise_seed = training_noise_seed(
                config, step, slot,
                bool(diagnostic) and not args.resampled_noise_overfit)
            torch.manual_seed(step_noise_seed)
            torch.cuda.manual_seed_all(step_noise_seed)
            timing["data_loading"] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(batch)
            if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
                raise FloatingPointError(f"non-finite loss at step {step} slot {slot}")
            (losses["loss"] / 8.0).backward()
            torch.cuda.synchronize()
            timing["forward_backward"] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            # Own the exact eight-microbatch sum on CPU. FSDP original
            # parameters are views into a small number of flat gradient
            # storages; transfer each block once, without changing values.
            gradient_storage_transfers = accumulate_cpu_gradients(model.named_parameters(), gradients, timing)
            timing["cpu_gradient_transfer_accumulation"] += time.perf_counter() - phase_started
            observations.append({"task": metas[slot]["task"],
                "elements": evidence["loss_rank_evidence"][slot]["local_elements"],
                "raw": {"video": scalar(losses["video_loss_unscaled"]),
                        "action": scalar(losses["action_loss_unscaled"]),
                        "ifp": [scalar(losses[f"ifp_head_{head}_loss_unscaled"]) for head in range(4)]},
                "losses": {key: scalar(losses[key]) for key in
                           ("loss", "video_loss", "action_loss", "ifp_loss", "ifp_active_heads")}})
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            del losses, batch
            emit_json_best_effort({"optimizer_step_pending": step, "microbatch_completed": slot + 1,
                                  "gradient_accumulation_steps": 8, "optimizer_applied": False,
                                  "gradient_storage_transfers": gradient_storage_transfers})
        if diagnostic and step == 0 and not args.resampled_noise_overfit:
            first_aggregate = {key: sum(row["losses"][key] for row in observations) / 8
                               for key in observations[0]["losses"]}
            first_forward_check = verify_first_forward(
                initial, first_aggregate,
                initial.get("loss_draws_per_condition", 1))
            write_json_atomic(root / "TRAIN_EVAL_FORWARD_EQUIVALENCE.json", first_forward_check)
        phase_started = time.perf_counter()
        norms = branch_norms(gradients.items())
        require_positive_finite(norms, "gradient")
        if args.repaired_overfit and step in (0, 1):
            cross_sums = defaultdict(float)
            for name, module in model.named_modules():
                if name.endswith(".cross_attn"):
                    cross_sums[name.removesuffix(".cross_attn")] = 0.0
            text_sum = 0.0
            for name, value in gradients.items():
                if ".cross_attn." in name:
                    cross_sums[name.split(".cross_attn.")[0]] += squared_norm(value)
                if "video_text_embedding." in name:
                    text_sum += squared_norm(value)
            restored_decision = restored_conditioning_gradient_decision(
                cross_sums, text_sum, step, config.zero_init_action_head)
            gradient_result_name = ("RESTORED_MODULE_GRADIENTS_INITIAL.json" if step == 0
                                    else "RESTORED_MODULE_GRADIENTS.json")
            write_json_atomic(root / gradient_result_name, restored_decision)
            if not restored_decision["passed"]:
                raise RuntimeError("restored video/action/IFP conditioning has inactive or invalid gradients")
        clip_scale = min(1.0, config.gradient_clip_norm / (math.sqrt(sum(v*v for v in norms.values())) + 1e-6))
        global_gradient_norm = math.sqrt(sum(v * v for v in norms.values()))
        milestone("global_gradient_ready", optimizer_step_pending=step)
        timing["gradient_norm_and_clip_audit"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        updates, finite, native_calls = native_adamw_streamed_step(
            optimizer, ((name, master) for name, _, master in master_pairs), gradients, clip_scale)
        timing["native_cpu_adamw_and_update_audit"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        with torch.no_grad():
            for _, parameter, master in master_pairs:
                # master is FP32 on CPU; the resident parameter may be BF16.
                # Verify the copy landed exactly in the *resident* dtype, i.e.
                # compare against the master rounded to that dtype.
                parameter.copy_(master)
                if not torch.equal(parameter, master.to(device=parameter.device,
                                                        dtype=parameter.dtype)):
                    raise RuntimeError("updated GPU parameter differs from native AdamW master")
        timing["gpu_master_copy_and_exact_readback"] = time.perf_counter() - phase_started
        require_positive_finite(updates, "update")
        if not finite:
            raise FloatingPointError(f"non-finite trainable parameter after optimizer step {step}")
        optimizer.zero_grad(set_to_none=True)
        del gradients
        aggregate = {key: sum(row["losses"][key] for row in observations) / 8 for key in observations[0]["losses"]}
        epoch = 0 if args.mode == "overfit" else step // 280
        timing["step_before_trace_and_checkpoint"] = time.perf_counter() - step_started
        record = {**evidence, "mode": args.mode, "optimizer_step": step, "resumed_from_step": start,
            "epoch": epoch, "epoch_step": step if args.mode == "overfit" else step % 280,
            "learning_rate": lr, "world_size": 1, "packed_samples_per_rank": 1,
            "global_packed_sample_batch": 8, "gradient_accumulation_steps": 8,
            "main_pass_count_per_sample": 2, "losses": aggregate,
            "task_losses": task_records(observations, config), "samples": metas,
            "physical_execution_map": [{"logical_schedule_slot": slot, "physical_rank": 0,
                                        "microbatch_index": slot} for slot in range(8)],
            "gradient_norms": norms, "parameter_update_norms": updates,
            "global_gradient_norm": global_gradient_norm,
            "gradient_clip_scale": clip_scale,
            "gradient_clip_fired": clip_scale < 1.0,
            "gpu_parameter_dtype": execution_plan["gpu_parameter_dtype"],
            "all_trainable_parameters_finite": finite, "optimizer_applied": True,
            "optimizer_native_parameter_calls": native_calls,
            "gpu_parameter_master_readback_exact": True,
            "cpu_accumulation_implementation": CPU_ACCUMULATION_IMPLEMENTATION,
            "timing_seconds": timing,
            "amp_scaler_skipped": False, "elapsed_seconds": time.perf_counter() - started}
        if diagnostic:
            record["diagnostic_noise_seeds"] = [
                training_noise_seed(config, step, slot,
                                    not args.resampled_noise_overfit)
                for slot in range(8)]
            record["noise_resampled_every_step"] = bool(args.resampled_noise_overfit)
            record["fixed_noise_overfit_diagnostic"] = True
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        emit_json_best_effort(record)
        milestone("optimizer_update_applied", completed_optimizer_steps=step + 1)
        stopping_for_deadline = deadline is not None and deadline_checkpoint_needed(
            time.time(), deadline, time.perf_counter() - step_started)
        if args.mode == "formal" and ((step + 1) % CHECKPOINT_INTERVAL == 0 or stopping_for_deadline):
            milestone("checkpoint_write_started", completed_optimizer_steps=step + 1)
            checkpoint_started = time.perf_counter()
            save_checkpoint(model, optimizer, root, step + 1, config)
            write_json_atomic(root / "FORMAL_PROGRESS.json", {
                "optimizer_steps_completed": step + 1, "optimizer_steps_planned": 4200,
                "complete_epochs": (step + 1) // 280, "epochs_planned": 15,
                "batch": EXECUTION, "latest_losses": aggregate, "hash_checks": False,
                "checkpoint_complete": True, "formal_execution_complete": step + 1 == total,
                "recovery_checkpoint_interval": CHECKPOINT_INTERVAL,
                "checkpoint_write_seconds": time.perf_counter() - checkpoint_started,
                "latest_step_timing_seconds": timing,
            })
            milestone("checkpoint_write_completed", completed_optimizer_steps=step + 1)
            refresh_results_document_best_effort()
        if args.stop_after_step is not None and step + 1 == execution_end:
            dist.destroy_process_group()
            publish_requested_stop(root, config)
            emit_json_best_effort({"training_stopped_at_user_requested_step": execution_end,
                                   "formal_execution_complete": False})
            return
        if stopping_for_deadline and step + 1 < total:
            write_json_atomic(root / "INFRASTRUCTURE_PAUSE.json", {
                "reason": "retained_allocation_near_end", "scientific_terminal": False,
                "completed_optimizer_steps": step + 1, "planned_optimizer_steps": total,
                "resume_checkpoint": str((root / "latest_checkpoint").resolve()),
                "allocation_end_timestamp": deadline, "hash_checks": False,
                "allocation_release_requested": False,
            })
            dist.destroy_process_group()
            raise SystemExit(75)  # End only this child; keep the retained compute shell alive.
    prompt = (prompt_gate(model, datasets, device, config.noise_seed + 9_999_991,
                          gate_draws) if args.mode == "overfit" else None)
    records = read_jsonl(log)
    formal = None
    if args.mode == "overfit":
        decision = overfit_execution_decision(log, schedule, 32, config,
            execution_world_size=1, accumulation_steps=8)
        ratios = {key: prompt["losses"]["matched"][key] / initial["losses"]["matched"][key]
                  for key in ("video_loss", "action_loss", "ifp_loss")}
        passed = prompt["passed"] and all(math.isfinite(v) and v <= 0.5 for v in ratios.values())
    else:
        formal = formal_training_decision(log, schedule, config,
            execution_world_size=1, accumulation_steps=8)
        checks = {key: value for key, value in formal["checks"].items() if key != "both_tasks_progress"}
        decision = {"checks": checks, "passed": all(checks.values())}
        passed = formal["passed"]
        ratios = {key: records[-1]["losses"][key] / records[0]["losses"][key]
                  for key in ("video_loss", "action_loss", "ifp_loss")}
    if not decision["passed"]:
        raise RuntimeError("single-GPU run failed its execution trace contract")
    result = {"protocol": f"paper_zero_wam_sugar_{args.mode}_v1", "passed": passed,
        "execution_completed": True, "execution_decision": decision, "formal_decision": formal,
        "architecture_parameter_count": config.expected_parameter_count, "optimizer_steps": total,
        "first_losses": records[0]["losses"], "last_losses": records[-1]["losses"],
        "last_to_first_loss_ratios": ratios, "initial_prompt_gate": initial, "prompt_gate": prompt,
        "batch": EXECUTION, "hash_checks": False, "elapsed_seconds": time.perf_counter() - started}
    if diagnostic:
        unseen_final = matched_loss_probe(model, datasets, device, diagnostic["unseen_noise_seed"], gate_draws)
        write_json_atomic(root / "UNSEEN_NOISE_FINAL.json", unseen_final)
        diagnostic_result = {
            "contract": diagnostic, "execution_completed": True,
            "execution_decision": decision, "train_eval_forward_equivalence": first_forward_check,
            "fixed_noise_fitting": reduction_result(initial, prompt),
            "unseen_noise_fitting": reduction_result(unseen_initial, unseen_final),
            "prompt_gate_passed": prompt["passed"],
            "original_combined_overfit_criterion_passed": passed,
            "automatic_formal_training": False, "hash_checks": False,
        }
        write_json_atomic(root / "FITTING_RESULT.json", diagnostic_result)
        # Preserve the endpoint model for inspection; no optimizer/continuation
        # checkpoint and no mutation of the original overfit or formal artifacts.
        with full_state_dict_context(model, offload_to_cpu=True, rank0_only=True):
            torch.save({"model": model.state_dict(), "step": 32,
                        "diagnostic_contract": diagnostic}, root / "diagnostic_model_step32.pt")
        if args.save_overfit_optimizer:
            torch.save({"step": total, "optimizer": optimizer.state_dict(),
                        "parameter_names": [name for name, _, _ in master_pairs],
                        "model_and_schedule_config": config.as_dict(),
                        "execution": EXECUTION, "diagnostic_contract": diagnostic},
                       root / "diagnostic_optimizer_step32.pt")
        # Publish training evidence before rendering so a renderer failure cannot
        # be misreported as a missing/failed optimizer result or retrigger training.
        write_json_atomic(terminal, result)
        rendered = render_training_cases(model, datasets, config, device, root / "training_videos")
        diagnostic_result["rendering"] = rendered
        diagnostic_result["requested_workflow_completed"] = True
        write_json_atomic(root / "DIAGNOSTIC_RESULT.json", diagnostic_result)
    dist.destroy_process_group()
    write_json_atomic(terminal, result)
    refresh_results_document_best_effort()


if __name__ == "__main__":
    main()
