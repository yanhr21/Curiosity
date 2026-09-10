#!/usr/bin/env python3
"""Materialize the frozen full-coverage BPP target/prompt schedule.

This is deterministic index-selection glue around the released BPP data path.  It
does not implement a model, tokenizer, sampler item loader, collation function,
forward pass, loss, optimizer, or evaluation metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "sugar_bpp_exact_training_schedule_v1"
CONTRACT_PROTOCOL = "sugar_bpp_exact_training_schedule_contract_v1"
SENSOR_PROTOCOL = "sugar_bpp_sensorimotor_corpus_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = base / path
    if local.exists():
        return local.resolve()
    return (ROOT / path).resolve()


def stable_order(items: Iterable[Any], namespace: str, seed: int) -> list[Any]:
    def key(item: Any) -> tuple[str, bytes]:
        encoded = canonical_bytes(item)
        digest = hashlib.sha256(
            f"{seed}|{namespace}|".encode("utf-8") + encoded
        ).hexdigest()
        return digest, encoded

    return sorted(items, key=key)


def identity(row: dict[str, Any]) -> tuple[str, int]:
    behavior_identity = row.get("behavior_identity")
    if not isinstance(behavior_identity, dict):
        raise ValueError("sensorimotor row lacks behavior_identity")
    return str(row["task_family"]), int(behavior_identity["source_motion_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensorimotor-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            ROOT
            / "scripts/sugar/demo_following/config/"
            "bpp_exact_training_schedule_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.sensorimotor_result.resolve()
    contract_path = args.contract.resolve()
    output = args.output.resolve()
    contract = read_json(contract_path)
    if contract.get("protocol") != CONTRACT_PROTOCOL:
        raise ValueError("training schedule contract protocol drift")
    sensor_result = read_json(result_path)
    if sensor_result.get("protocol") != SENSOR_PROTOCOL or sensor_result.get("passed") is not True:
        raise ValueError("sensorimotor admission is absent or failed")
    artifacts = sensor_result.get("artifacts")
    if not isinstance(artifacts, dict) or "sensorimotor_manifest" not in artifacts:
        raise ValueError("sensorimotor result lacks manifest artifact")
    manifest_path = resolve_from(result_path.parent, str(artifacts["sensorimotor_manifest"]))
    manifest_rows = read_jsonl(manifest_path)
    if len(manifest_rows) != 995:
        raise ValueError(f"expected 995 admitted trajectories, found {len(manifest_rows)}")

    train_rows = [row for row in manifest_rows if row.get("split") == "train"]
    heldout_rows = [row for row in manifest_rows if row.get("split") != "train"]
    if len(train_rows) != 800 or len(heldout_rows) != 195:
        raise ValueError("sensorimotor split count drift")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[identity(row)].append(row)
    if len(grouped) != 160:
        raise ValueError(f"expected 160 train behavior identities, found {len(grouped)}")
    task_identity_counts = Counter(task for task, _ in grouped)
    if task_identity_counts != Counter({"CarryBox": 80, "KickBox": 80}):
        raise ValueError(f"train task identity count drift: {task_identity_counts}")

    normalized: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for source_identity, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["admitted_variant_rank"]))
        ranks = [int(row["admitted_variant_rank"]) for row in rows]
        if ranks != list(range(5)):
            raise ValueError(f"admitted variant rank drift for {source_identity}: {ranks}")
        candidate_ids = [int(row["candidate_variant_id"]) for row in rows]
        if candidate_ids != sorted(candidate_ids) or len(set(candidate_ids)) != 5:
            raise ValueError(f"candidate variant order drift for {source_identity}")
        for row in rows:
            target = row.get("bpp_target_samples")
            if not isinstance(target, dict):
                raise ValueError("sensorimotor row lacks target geometry")
            anchors = target.get("sample_anchor_action_indices_50hz")
            if anchors != list(range(700)) or target.get("sample_count") != 700:
                raise ValueError(f"target anchor geometry drift for {source_identity}")
        normalized[source_identity] = rows

    epochs = int(contract["official_recipe"]["epochs"])
    schedule_seed = int(contract["schedule_seed"])
    microbatch_size = int(contract["single_h200_equivalence"]["microbatch_size"])
    accumulation = int(
        contract["single_h200_equivalence"]["gradient_accumulation_steps"]
    )
    if epochs != 7 or microbatch_size != 100 or accumulation != 4:
        raise ValueError("frozen epoch/batch/accumulation geometry drift")

    output.mkdir(parents=True, exist_ok=False)
    schedule_path = output / "BPP_EXACT_TRAINING_SCHEDULE.jsonl"
    epoch_summaries: list[dict[str, Any]] = []
    global_microbatch_index = 0
    global_seen: set[tuple[int, str, int, int, int]] = set()
    with schedule_path.open("w", encoding="utf-8") as schedule_stream:
        for epoch in range(epochs):
            pending: list[dict[str, Any]] = []
            prompt_counts: Counter[tuple[str, int, int]] = Counter()
            epoch_targets: set[tuple[str, int, int, int]] = set()
            for source_identity in sorted(normalized):
                task, source_motion_id = source_identity
                variants = normalized[source_identity]
                for target_rank, target_row in enumerate(variants):
                    anchors = stable_order(
                        range(700),
                        (
                            f"target-anchor|epoch={epoch}|task={task}|"
                            f"source={source_motion_id}|target_rank={target_rank}"
                        ),
                        schedule_seed,
                    )
                    for block_rank in range(7):
                        target_anchors = anchors[
                            block_rank * microbatch_size : (block_rank + 1) * microbatch_size
                        ]
                        if len(target_anchors) != microbatch_size:
                            raise AssertionError("target block geometry drift")
                        prompt_rank = (
                            target_rank + 1 + ((block_rank + epoch) % 4)
                        ) % 5
                        if prompt_rank == target_rank:
                            raise AssertionError("frozen prompt formula produced a self-pair")
                        prompt_row = variants[prompt_rank]
                        for anchor in target_anchors:
                            target_key = (task, source_motion_id, target_rank, int(anchor))
                            if target_key in epoch_targets:
                                raise AssertionError(f"duplicate target row: {target_key}")
                            epoch_targets.add(target_key)
                            global_seen.add((epoch, *target_key))
                        prompt_counts[(task, source_motion_id, prompt_rank)] += 1
                        pending.append({
                            "protocol": PROTOCOL,
                            "epoch": epoch,
                            "task_label": task,
                            "source_motion_id": source_motion_id,
                            "target_variant_rank": target_rank,
                            "target_variant_id": int(target_row["candidate_variant_id"]),
                            "target_sensorimotor_row_sha256": canonical_sha256(target_row),
                            "target_anchor_block_rank": block_rank,
                            "target_anchor_indices": [int(anchor) for anchor in target_anchors],
                            "prompt_variant_rank": prompt_rank,
                            "prompt_variant_id": int(prompt_row["candidate_variant_id"]),
                            "prompt_sensorimotor_row_sha256": canonical_sha256(prompt_row),
                        })

            if len(pending) != 5600 or len(epoch_targets) != 560000:
                raise AssertionError("epoch target coverage geometry drift")
            expected_prompt_keys = {
                (task, source_motion_id, prompt_rank)
                for task, source_motion_id in normalized
                for prompt_rank in range(5)
            }
            if set(prompt_counts) != expected_prompt_keys or set(prompt_counts.values()) != {7}:
                raise AssertionError("per-identity prompt balance drift")
            ordered = stable_order(pending, f"epoch-microbatch-order|epoch={epoch}", schedule_seed)
            for epoch_microbatch_index, row in enumerate(ordered):
                row["epoch_microbatch_index"] = epoch_microbatch_index
                row["global_microbatch_index"] = global_microbatch_index
                row["epoch_accumulation_boundary_index"] = epoch_microbatch_index // accumulation
                row["global_accumulation_boundary_index"] = (
                    global_microbatch_index // accumulation
                )
                schedule_stream.write(json.dumps(row, sort_keys=True) + "\n")
                global_microbatch_index += 1
            epoch_summaries.append({
                "epoch": epoch,
                "microbatch_count": len(ordered),
                "accumulation_boundary_count": len(ordered) // accumulation,
                "target_row_count": len(epoch_targets),
                "unique_target_row_count": len(epoch_targets),
                "prompt_uses_per_variant_per_identity": sorted(set(prompt_counts.values())),
                "target_set_sha256": canonical_sha256(sorted(epoch_targets)),
            })

    checks = {
        "sensorimotor_admission_passed": True,
        "train_trajectory_count_exact_800": len(train_rows) == 800,
        "train_identity_count_exact_160": len(normalized) == 160,
        "five_variants_per_identity": all(len(rows) == 5 for rows in normalized.values()),
        "seven_complete_epochs": epochs == 7,
        "microbatch_size_exact_100": microbatch_size == 100,
        "gradient_accumulation_exact_4": accumulation == 4,
        "microbatch_count_exact_39200": global_microbatch_index == 39200,
        "target_exposures_exact_3920000": len(global_seen) == 3920000,
        "every_epoch_has_560000_unique_target_rows": all(
            summary["unique_target_row_count"] == 560000
            for summary in epoch_summaries
        ),
        "every_prompt_variant_used_seven_times_per_identity_per_epoch": all(
            summary["prompt_uses_per_variant_per_identity"] == [7]
            for summary in epoch_summaries
        ),
        "self_pair_excluded_by_construction": True,
        "heldout_rows_excluded": all(row.get("split") != "train" for row in heldout_rows),
    }
    if not all(checks.values()):
        raise AssertionError(f"schedule audit failed: {checks}")
    result = {
        "protocol": PROTOCOL,
        "passed": True,
        "checks": checks,
        "contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
        },
        "sensorimotor_admission": {
            "result_path": str(result_path),
            "result_sha256": file_sha256(result_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
        },
        "schedule": {
            "path": schedule_path.name,
            "sha256": file_sha256(schedule_path),
            "epochs": epochs,
            "microbatches": global_microbatch_index,
            "accumulation_boundaries": global_microbatch_index // accumulation,
            "target_sample_exposures": len(global_seen),
        },
        "epoch_summaries": epoch_summaries,
        "claim_boundary": (
            "This materializes exact target/prompt index selection for the official BPP "
            "loader. It is not evidence that the ReplayBuffer consumed the indices, that "
            "an optimizer update ran, or that a trained policy follows prompts."
        ),
    }
    result_path_out = output / "RESULT.json"
    result_path_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
