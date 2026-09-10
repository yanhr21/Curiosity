#!/usr/bin/env python3
"""Frozen motion-disjoint prompt-dependence evaluation for paper Zero-WAM."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from .artifacts import emit_json_best_effort, write_json_atomic, write_text_atomic
from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .data import (
    ActionNormalizer,
    load_executed_actions,
    load_validated_latent_payload,
    read_jsonl,
)
from .model import IFPHead, PaperMoTLayer, PaperZeroWAM
from .results import refresh_results_document_best_effort
from .train import validated_checkpoint_step


# These are the old fixed raw-10-Hz phase anchors mapped onto the causal
# stride-four Wan latent clock.  The final two-latent target always remains in
# the 36-frame encoded robot trajectory.
LATENT_ANCHORS = (1, 5, 8, 12, 15, 19, 22, 26, 29, 33)
CONDITIONS = ("matched", "reversed", "same_task_alternate", "wrong_task", "masked")
INTERVENTIONS = CONDITIONS[1:]
LOSS_NAMES = ("video_loss", "action_loss")
CONNECTIVITY_NAMES = (
    "predicted_future_mse_vs_matched",
    "predicted_action_velocity_mse_vs_matched",
)


def evaluation_scope(single_gpu, interim_step=None):
    """The user-requested step-700 evaluation is not formal completion."""
    if interim_step is None:
        return 4200, "paper_zero_wam_motion_disjoint_prompt_gate_v2"
    if not single_gpu or type(interim_step) is not int or interim_step != 700:
        raise ValueError("the only interim evaluation is single-GPU step 700")
    return 700, "paper_zero_wam_step700_interim_prompt_evaluation_v1"


def heldout_groups(config):
    rows = read_jsonl(config.resolved(config.manifest))
    by_key = {(row["split"], row["task"], int(row["source_motion_id"])): row for row in rows}
    heldout = sorted((row for row in rows if row["split"] in ("validation", "test")),
                     key=lambda row: (row["split"], row["task"], int(row["source_motion_id"])))
    groups = [{"row": row, "anchor": anchor, "group_index": index * 10 + ordinal}
              for index, row in enumerate(heldout) for ordinal, anchor in enumerate(LATENT_ANCHORS)]
    if len(groups) != 390:
        raise ValueError(f"expected 390 frozen held-out groups, found {len(groups)}")
    return by_key, groups


def validate_score_prefix(records, groups, config):
    """Validate complete groups in immutable source/anchor/noise order, not outcomes."""
    if len(records) > len(groups):
        raise ValueError("score journal has extra groups")
    for record, group in zip(records, groups):
        row = group["row"]
        expected = dict(group_index=group["group_index"], split=row["split"], task=row["task"],
                        source_motion_id=int(row["source_motion_id"]), latent_anchor=group["anchor"],
                        matched_noise_seed=config.noise_seed + 10_000_019 + group["group_index"])
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("score journal differs from frozen source/anchor/noise prefix")
        if (set(record.get("losses", {})) != set(CONDITIONS)
                or any(set(values) != set(LOSS_NAMES) for values in record["losses"].values())
                or set(record.get("connectivity_vs_matched", {})) != set(INTERVENTIONS)
                or any(set(values) != set(CONNECTIVITY_NAMES)
                       for values in record["connectivity_vs_matched"].values())):
            raise ValueError("score journal contains a partial condition group")
        # Non-finite numeric outcomes are preserved, never erased and re-scored
        # until favorable. The scientific reducer must close those outcomes.
        numbers = [value for field in ("losses", "connectivity_vs_matched")
                   for values in record[field].values() for value in values.values()]
        if any(type(value) not in (int, float) for value in numbers):
            raise ValueError("score journal contains nonnumeric outcomes")
    return len(records)


def setup(single_gpu: bool = False) -> tuple[int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != (1 if single_gpu else 8):
        raise ValueError("held-out world size does not match selected execution backend")
    torch.cuda.set_device(local_rank)
    # Rank-local checkpoint/cache I/O can be skewed on shared storage.  Keep a
    # bounded watchdog above that legitimate wait; the one-day Slurm limit is
    # still the outer evaluation bound.
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    return rank, world_size, torch.device("cuda", local_rank)


def cache_path(root: Path, row: dict[str, Any]) -> Path:
    return root / row["split"] / row["task"] / f'{int(row["source_motion_id"]):03d}.pt'


def load_actions(
    row: dict[str, Any],
    normalizer: ActionNormalizer,
    device: torch.device,
    action_cache: dict[tuple[Path, int], torch.Tensor],
) -> torch.Tensor:
    path = Path(row["action_target"]["trace_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    key = (path.resolve(), int(row["action_target"]["environment_index"]))
    if key not in action_cache:
        shard = load_executed_actions(key[0], key[1])
        actions = torch.from_numpy(shard.copy())
        if tuple(actions.shape) != (700, 29) or not bool(
            torch.isfinite(actions).all()
        ):
            raise ValueError(
                f"held-out action geometry/finiteness mismatch: {key} {tuple(actions.shape)}"
            )
        normalized = normalizer(actions)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(f"held-out normalized actions are non-finite: {key}")
        action_cache[key] = normalized
    return action_cache[key].to(device=device, dtype=torch.bfloat16)


def cached_latent_payload(
    row: dict[str, Any],
    cache_root: Path,
    latent_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    path = cache_path(cache_root, row)
    if path not in latent_cache:
        latent_cache[path] = load_validated_latent_payload(
            path,
            split=str(row["split"]),
            task=str(row["task"]),
            source_motion_id=int(row["source_motion_id"]),
        )
    return latent_cache[path]


def prompt_for_condition(
    condition: str,
    target_row: dict[str, Any],
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]],
    cache_root: Path,
    device: torch.device,
    latent_cache: dict[Path, dict[str, Any]],
) -> torch.Tensor:
    if condition in ("matched", "reversed", "masked"):
        prompt_row = target_row
    else:
        spec = target_row["fixed_counterfactual_prompts"][condition]
        prompt_row = rows_by_key[(target_row["split"], spec["task"], int(spec["source_motion_id"]))]
    payload = cached_latent_payload(prompt_row, cache_root, latent_cache)
    key = "reversed_prompt_latents" if condition == "reversed" else "prompt_latents"
    prompt = payload[key].to(device=device, dtype=torch.bfloat16)
    if condition == "masked":
        prompt = torch.zeros_like(prompt)
    return prompt


def build_batch(
    group: dict[str, Any],
    condition: str,
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]],
    cache_root: Path,
    normalizer: ActionNormalizer,
    device: torch.device,
    latent_cache: dict[Path, dict[str, Any]],
    action_cache: dict[tuple[Path, int], torch.Tensor],
) -> dict[str, Any]:
    row = group["row"]
    anchor = int(group["anchor"])
    payload = cached_latent_payload(row, cache_root, latent_cache)
    robot = payload["robot_latents"].to(device=device, dtype=torch.bfloat16)
    actions = load_actions(row, normalizer, device, action_cache)
    target = robot[:, anchor + 1 : anchor + 3]
    action_start = anchor * 20
    zero_future = torch.zeros_like(target).unsqueeze(0)
    return {
        "prompt_latents": prompt_for_condition(
            condition, row, rows_by_key, cache_root, device, latent_cache
        ).unsqueeze(0),
        "robot_history_latents": robot[:, : anchor + 1].unsqueeze(0),
        "video_target_latents": target.unsqueeze(0),
        "action_history": actions[:action_start].unsqueeze(0),
        "action_target": actions[action_start : action_start + 40].unsqueeze(0),
        "ifp_target_latents": [zero_future.clone() for _ in range(4)],
        "ifp_valid": [False, False, False, False],
        "ifp_latent_starts": [anchor + 2, anchor + 6, anchor + 10, anchor + 14],
        # Match the training/CFG dropout path exactly for the masked condition;
        # zero-valued latents alone would still expose patch-projection biases
        # and positional states through attention.
        "prompt_enabled": condition != "masked",
    }


def exact_sign_pvalue(margins: list[float]) -> float:
    nonzero = [value for value in margins if value != 0.0]
    positives = sum(value > 0.0 for value in nonzero)
    count = len(nonzero)
    if count == 0:
        return 1.0
    return sum(math.comb(count, index) for index in range(positives, count + 1)) / 2**count


def holm_decisions(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    decisions: dict[str, bool] = {}
    still_rejecting = True
    total = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        still_rejecting = still_rejecting and pvalue <= alpha / (total - index)
        decisions[name] = still_rejecting
    return decisions


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    actual_group_indices = [int(record["group_index"]) for record in records]
    expected_group_indices = list(range(390))
    expected_noise_seed = PaperZeroWAMConfig().noise_seed
    if actual_group_indices != expected_group_indices:
        raise ValueError(
            "held-out records must be the unique ordered group-index partition 0..389"
        )
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["task"], record["source_motion_id"])].append(record)
    if len(grouped) != 39 or any(len(rows) != 10 for rows in grouped.values()):
        raise ValueError("held-out result must contain ten anchors for each of 39 motions")
    if any(
        {int(row["latent_anchor"]) for row in rows} != set(LATENT_ANCHORS)
        for rows in grouped.values()
    ):
        raise ValueError("every held-out motion must contain each frozen anchor exactly once")
    if any(
        int(record["matched_noise_seed"])
        != expected_noise_seed + 10_000_019 + int(record["group_index"])
        for record in records
    ):
        raise ValueError("held-out matched-noise seed mapping changed")
    connectivity_checks = {
        f"{record['group_index']}/{intervention}/{name}": float(
            record["connectivity_vs_matched"][intervention][name]
        )
        > 1.0e-12
        for record in records
        for intervention in INTERVENTIONS
        for name in CONNECTIVITY_NAMES
    }

    motion_margins: list[dict[str, Any]] = []
    for (split, task, source_motion_id), motion_rows in sorted(grouped.items()):
        margins: dict[str, dict[str, float]] = {}
        for intervention in INTERVENTIONS:
            margins[intervention] = {}
            for loss_name in LOSS_NAMES:
                margins[intervention][loss_name] = float(
                    np.mean(
                        [
                            row["losses"][intervention][loss_name]
                            - row["losses"]["matched"][loss_name]
                            for row in motion_rows
                        ]
                    )
                )
        motion_margins.append(
            {
                "split": split,
                "task": task,
                "source_motion_id": source_motion_id,
                "margins": margins,
            }
        )

    directional: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for split in ("validation", "test"):
        expected = 20 if split == "validation" else 19
        minimum_wins = 18
        directional[split] = {}
        split_rows = [row for row in motion_margins if row["split"] == split]
        if len(split_rows) != expected:
            raise ValueError(f"{split} motion count mismatch: {len(split_rows)}")
        for intervention in INTERVENTIONS:
            directional[split][intervention] = {}
            for loss_name in LOSS_NAMES:
                values = [row["margins"][intervention][loss_name] for row in split_rows]
                key = f"{split}/{intervention}/{loss_name}"
                pvalue = exact_sign_pvalue(values)
                pvalues[key] = pvalue
                directional[split][intervention][loss_name] = {
                    "motion_count": expected,
                    "positive_motion_count": sum(value > 0.0 for value in values),
                    "mean_margin": float(np.mean(values)),
                    "median_margin": float(np.median(values)),
                    "one_sided_exact_sign_pvalue": pvalue,
                    "minimum_positive_motions": minimum_wins,
                }

    task_checks: dict[str, Any] = {}
    for split in ("validation", "test"):
        task_checks[split] = {}
        for task in ("CarryBox", "KickBox"):
            rows = [
                row for row in motion_margins
                if row["split"] == split and row["task"] == task
            ]
            expected_task_motions = (
                10 if split == "validation" or task == "CarryBox" else 9
            )
            if len(rows) != expected_task_motions:
                raise ValueError(
                    f"{split}/{task} motion count mismatch: "
                    f"expected {expected_task_motions}, found {len(rows)}"
                )
            task_checks[split][task] = {}
            for intervention in INTERVENTIONS:
                task_checks[split][task][intervention] = {}
                for loss_name in LOSS_NAMES:
                    values = [row["margins"][intervention][loss_name] for row in rows]
                    task_checks[split][task][intervention][loss_name] = {
                        "mean_margin": float(np.mean(values)),
                        "win_rate": sum(value > 0.0 for value in values) / len(values),
                        "passed": float(np.mean(values)) > 0.0
                        and sum(value > 0.0 for value in values) / len(values) > 0.5,
                    }

    holm = holm_decisions(pvalues)
    # Keep fully qualified names in the machine-readable decision.
    count_checks = {
        f"{split}/{intervention}/{loss_name}": (
            directional[split][intervention][loss_name]["positive_motion_count"]
            >= directional[split][intervention][loss_name]["minimum_positive_motions"]
        )
        for split in directional
        for intervention in directional[split]
        for loss_name in directional[split][intervention]
    }
    all_task_checks = all(
        value["passed"]
        for split_values in task_checks.values()
        for task_values in split_values.values()
        for intervention_values in task_values.values()
        for value in intervention_values.values()
    )
    checks = {
        "all_scores_and_connectivity_finite": all(
            math.isfinite(float(value)) for record in records
            for field in ("losses", "connectivity_vs_matched")
            for values in record[field].values() for value in values.values()
        ),
        "exact_390_matched_noise_groups": len(records) == 390,
        "unique_complete_group_index_partition": actual_group_indices
        == expected_group_indices,
        "exact_ten_anchor_partition_per_motion": all(
            {int(row["latent_anchor"]) for row in rows} == set(LATENT_ANCHORS)
            for rows in grouped.values()
        ),
        "exact_1950_condition_scores": len(records) * len(CONDITIONS) == 1950,
        "every_prompt_swap_changes_predicted_future_and_action": (
            len(connectivity_checks) == 390 * 4 * 2
            and all(connectivity_checks.values())
        ),
        "all_directional_motion_counts_pass": all(count_checks.values()),
        "all_16_holm_tests_pass": len(holm) == 16 and all(holm.values()),
        "carry_and_kick_each_positive": all_task_checks,
    }
    return {
        "execution_completed": (
            checks["exact_390_matched_noise_groups"]
            and checks["unique_complete_group_index_partition"]
            and checks["exact_ten_anchor_partition_per_motion"]
            and checks["exact_1950_condition_scores"]
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "directional_motion_results": directional,
        "holm_decisions": holm,
        "task_directional_results": task_checks,
        "motion_margins": motion_margins,
        "connectivity_check_count": len(connectivity_checks),
        "failed_connectivity_checks": sorted(
            name for name, passed in connectivity_checks.items() if not passed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--interim-step", type=int, choices=[700])
    args = parser.parse_args()
    expected_step, result_protocol = evaluation_scope(args.single_gpu, args.interim_step)
    if args.interim_step is not None and args.output_dir.name != "heldout_prompt_gate_step700":
        raise ValueError("interim scores must use the isolated heldout_prompt_gate_step700 directory")
    config = PaperZeroWAMConfig()
    rank, world_size, device = setup(args.single_gpu)
    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    rows_by_key, groups = heldout_groups(config)
    retained_records = []
    journal_resumed = False
    checkpoint_state = json.loads((args.checkpoint / "STATE.json").read_text(encoding="utf-8"))
    binding = None
    if args.single_gpu:
        from .train_single_gpu import single_gpu_checkpoint_step
        if single_gpu_checkpoint_step(checkpoint_state, config) != expected_step:
            raise ValueError(f"evaluation requires exact checkpoint step {expected_step}")
        binding = dict(protocol="paper_zero_wam_single_gpu_score_journal_v1",
                       checkpoint_directory=str(args.checkpoint.resolve()), checkpoint_state=checkpoint_state,
                       latent_anchors=list(LATENT_ANCHORS), conditions=list(CONDITIONS), hash_checks=False)
        if args.interim_step is not None:
            binding.update(evaluation_scope="user_requested_step700_interim",
                           formal_execution_complete=False, physical_continuation_allowed=False)
        binding_path = output_dir / "SCORE_JOURNAL_RUN.json"
        journal_path = output_dir / "SCORE_JOURNAL.jsonl"
        if binding_path.exists():
            journal_resumed = True
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("held-out journal belongs to another checkpoint or protocol")
        elif journal_path.exists():
            raise ValueError("score journal has no checkpoint binding")
        else:
            write_json_atomic(binding_path, binding)
        terminal_path = output_dir / "HELDOUT_PROMPT_RESULT.json"
        if terminal_path.exists():
            existing = json.loads(terminal_path.read_text())
            records = read_jsonl(output_dir / "HELDOUT_PROMPT_SCORES.jsonl")
            validate_score_prefix(records, groups, config)
            decision = aggregate(records)
            if (existing.get("protocol") != result_protocol
                    or existing.get("score_journal_binding") != binding
                    or existing.get("decision") != decision
                    or existing.get("passed") is not decision["passed"]
                    or existing.get("execution_completed") is not True
                    or existing.get("checkpoint_step") != expected_step
                    or existing.get("group_count") != 390 or existing.get("score_instance_count") != 1950
                    or existing.get("architecture_parameter_count") != config.expected_parameter_count):
                raise ValueError("completed held-out terminal differs from full frozen scores")
            dist.destroy_process_group()
            emit_json_best_effort({"terminal_reused": str(terminal_path), "model_score_calls_added": 0})
            return
        if journal_path.exists():
            retained_records = read_jsonl(journal_path)
            validate_score_prefix(retained_records, groups, config)

    model = PaperZeroWAM.from_wan_pretrained(config, dtype=torch.float32)
    full_parameter_count = model.parameter_count
    if full_parameter_count != config.expected_parameter_count:
        raise RuntimeError("held-out model parameter contract changed")
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({PaperMoTLayer, IFPHead}),
        sharding_strategy=(ShardingStrategy.NO_SHARD if args.single_gpu else ShardingStrategy.FULL_SHARD),
        mixed_precision=MixedPrecision(
            param_dtype=None, reduce_dtype=torch.bfloat16, buffer_dtype=None
        ),
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    if args.single_gpu:
        from .train_single_gpu import single_gpu_checkpoint_step
        checkpoint_step = single_gpu_checkpoint_step(checkpoint_state, config)
    else:
        checkpoint_step = validated_checkpoint_step(checkpoint_state, "formal", config)
    checkpoint_path = args.checkpoint / f"model_rank{rank:02d}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint_step != expected_step
        or int(checkpoint.get("step", -1)) != checkpoint_step
    ):
        raise ValueError(f"held-out evaluation requires exact checkpoint step {expected_step}")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT if args.single_gpu else StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()
    dist.barrier()

    cache_root = config.resolved(config.latent_cache)
    normalizer = ActionNormalizer(cache_root / "ACTION_QUANTILES.json")
    latent_cache: dict[Path, dict[str, Any]] = {}
    action_cache: dict[tuple[Path, int], torch.Tensor] = {}
    local_records: list[dict[str, Any]] = list(retained_records)
    slots = math.ceil(len(groups) / world_size)
    first_slot = len(retained_records) if args.single_gpu else 0
    for slot in range(first_slot, slots):
        group_index = slot * world_size + rank
        padded = group_index >= len(groups)
        group = groups[0] if padded else groups[group_index]
        row = group["row"]
        condition_losses: dict[str, dict[str, float]] = {}
        condition_predictions: dict[str, dict[str, torch.Tensor]] = {}
        score_seed = config.noise_seed + 10_000_019 + int(group["group_index"])
        with torch.inference_mode():
            for condition in CONDITIONS:
                batch = build_batch(
                    group,
                    condition,
                    rows_by_key,
                    cache_root,
                    normalizer,
                    device,
                    latent_cache,
                    action_cache,
                )
                batch["prompt_score"] = True
                # Reset after every condition-specific cache/load path so the
                # model receives the same RNG state even when one payload was
                # a cache miss and another was already resident.
                torch.manual_seed(score_seed)
                torch.cuda.manual_seed_all(score_seed)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scores = model(batch)
                condition_losses[condition] = {
                    loss_name: float(scores[loss_name].detach().float().item())
                    for loss_name in LOSS_NAMES
                }
                condition_predictions[condition] = {
                    "predicted_future_latents": scores[
                        "predicted_future_latents"
                    ].detach(),
                    "predicted_action_velocity": scores[
                        "predicted_action_velocity"
                    ].detach(),
                }
        connectivity_vs_matched = {
            condition: {
                "predicted_future_mse_vs_matched": float(
                    (
                        condition_predictions[condition]["predicted_future_latents"].float()
                        - condition_predictions["matched"]["predicted_future_latents"].float()
                    )
                    .square()
                    .mean()
                    .item()
                ),
                "predicted_action_velocity_mse_vs_matched": float(
                    (
                        condition_predictions[condition]["predicted_action_velocity"].float()
                        - condition_predictions["matched"]["predicted_action_velocity"].float()
                    )
                    .square()
                    .mean()
                    .item()
                ),
            }
            for condition in INTERVENTIONS
        }
        if not padded:
            local_records.append(
                {
                    "group_index": int(group["group_index"]),
                    "split": row["split"],
                    "task": row["task"],
                    "source_motion_id": int(row["source_motion_id"]),
                    "latent_anchor": int(group["anchor"]),
                    "matched_noise_seed": score_seed,
                    "losses": condition_losses,
                    "connectivity_vs_matched": connectivity_vs_matched,
                }
            )
        if rank == 0 and (slot + 1) % 5 == 0:
            emit_json_best_effort(
                {"completed_slots": slot + 1, "total_slots": slots}
            )
        if args.single_gpu:
            validate_score_prefix(local_records, groups, config)
            write_text_atomic(journal_path, "".join(json.dumps(record, sort_keys=True) + "\n"
                                                     for record in local_records))

    gathered: list[list[dict[str, Any]] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    records: list[dict[str, Any]] | None = None
    decision: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    if rank == 0:
        assert gathered is not None
        records = sorted(
            [record for rank_records in gathered if rank_records for record in rank_records],
            key=lambda record: int(record["group_index"]),
        )
        decision = aggregate(records)
        result = {
            "protocol": result_protocol,
            "execution_completed": decision["execution_completed"],
            "passed": decision["passed"],
            "checkpoint_step": checkpoint_step,
            "architecture_parameter_count": full_parameter_count,
            "latent_anchors": list(LATENT_ANCHORS),
            "condition_count": len(CONDITIONS),
            "group_count": len(records),
            "score_instance_count": len(records) * len(CONDITIONS),
            "execution_world_size": world_size,
            "executed_model_score_calls": None if journal_resumed else slots * world_size * len(CONDITIONS),
            "retained_complete_score_groups": first_slot,
            "this_process_model_score_calls": (slots - first_slot) * world_size * len(CONDITIONS),
            "interrupted_uncommitted_score_calls": "unknown" if journal_resumed else 0,
            "score_journal_binding": binding,
            "padding_score_calls": (slots * world_size - len(records)) * len(CONDITIONS),
            "decision": decision,
            "hash_checks": False,
            "claim_boundary": (
                "Motion-disjoint fixed-noise next-video flow and generated-future-conditioned "
                "action-flow dependence on selected SUGAR prompts; not closed-loop physical "
                "success or open-ended task following."
            ),
        }
        if args.interim_step is not None:
            result.update(
                evaluation_scope="user_requested_step700_interim",
                formal_execution_complete=False,
                original_formal_optimizer_budget=config.optimizer_steps,
                physical_continuation_allowed=False,
                automatic_training_continuation=False,
                claim_boundary="Step-700 interim diagnostic. " + result["claim_boundary"],
            )
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        assert records is not None and decision is not None and result is not None
        write_text_atomic(
            output_dir / "HELDOUT_PROMPT_SCORES.jsonl",
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        )
        # Publish the terminal only after the complete score log exists and no
        # distributed tail remains.  A missing terminal therefore remains a
        # truthful, retryable infrastructure failure.
        write_json_atomic(output_dir / "HELDOUT_PROMPT_RESULT.json", result)
        refresh_results_document_best_effort()
        emit_json_best_effort(
            {"passed": result["passed"], "checks": decision["checks"]}
        )


if __name__ == "__main__":
    main()
