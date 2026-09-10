#!/usr/bin/env python3
"""Admit five real SUGAR BPP demonstrations per source-motion identity.

This is a data/evidence audit, not a policy. It consumes the ten frozen
Generator+Tracker candidate variants, independently recomputes physical task
success, and selects the first five safe candidates by ascending variant ID.
No BPP feature, loss, checkpoint, or output participates in selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "sugar_bpp_five_variant_action_admission_v1"
CONTRACT_PROTOCOL = "sugar_bpp_five_variant_corpus_contract_v1"
TASK_COUNTS = {"CarryBox": 100, "KickBox": 99}
EXPECTED_CANDIDATES = 10
ADMITTED_PER_MOTION = 5
EXPECTED_STEPS = 700
EXPECTED_CANDIDATE_TRAJECTORIES = 1990
EXPECTED_ADMITTED_TRAJECTORIES = 995
LIFT_THRESHOLD_M = 0.05
FALL_HEIGHT_LOSS_M = 0.35
FALL_ROOT_TILT_DEG = 60.0
KICK_NET_DISPLACEMENT_M = 0.05
KICK_CONTACT_COUPLED_PATH_M = 0.01
KICK_POST_CONTACT_PATH_M = 0.03
STARTUP_ARRAYS = (
    "startup_robot_material_properties",
    "startup_object_material_properties",
    "startup_object_masses",
    "startup_robot_coms",
    "startup_robot_default_joint_pos",
)
TRACE_FINGERPRINT_ARRAYS = (
    "robot_root_state_before_w",
    "robot_joint_pos_before",
    "robot_joint_vel_before",
    "object_root_state_before_w",
    "policy_observation_before",
    "goal_policy_core_observation_before",
    "generator_command_before",
    "executed_action",
    "robot_root_state_w",
    "robot_joint_pos",
    "robot_joint_vel",
    "object_root_state_w",
    "contact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--action-corpus-root", type=Path, required=True)
    parser.add_argument("--action-audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def arrays_sha256(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def split_for_motion(motion_id: int) -> str:
    remainder = motion_id % 10
    if remainder == 8:
        return "validation"
    if remainder == 9:
        return "test"
    return "train"


def planar_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left[:2]) - np.asarray(right[:2])))


def summarize_physics(
    robot_root_state: np.ndarray,
    object_root_state: np.ndarray,
    contact: np.ndarray,
) -> dict[str, Any]:
    if robot_root_state.shape != (EXPECTED_STEPS, 13):
        raise ValueError(f"robot root shape drift: {robot_root_state.shape}")
    if object_root_state.shape != (EXPECTED_STEPS, 13):
        raise ValueError(f"object root shape drift: {object_root_state.shape}")
    if contact.shape != (EXPECTED_STEPS, 4):
        raise ValueError(f"contact shape drift: {contact.shape}")
    objects = object_root_state[:, :3].astype(np.float64)
    heights = robot_root_state[:, 2].astype(np.float64)
    quaternions = robot_root_state[:, 3:7].astype(np.float64)
    bilateral = contact[:, 0].astype(bool) & contact[:, 1].astype(bool)
    foot = contact[:, 2].astype(bool) | contact[:, 3].astype(bool)
    baseline_z = float(np.median(objects[:25, 2]))
    maximum_lift = float(np.max(objects[:, 2] - baseline_z))
    root_height_loss = float(heights[0] - np.min(heights))
    maximum_tilt = 0.0
    for quaternion in quaternions:
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-3:
            raise ValueError(f"invalid robot root quaternion norm: {norm}")
        _, x, y, _ = quaternion.tolist()
        up_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
        maximum_tilt = max(maximum_tilt, math.degrees(math.acos(up_z)))
    physical_fall = (
        root_height_loss >= FALL_HEIGHT_LOSS_M
        or maximum_tilt >= FALL_ROOT_TILT_DEG
    )
    planar_net = planar_distance(objects[-1], objects[0])
    step_paths = np.asarray(
        [planar_distance(objects[index], objects[index - 1]) for index in range(1, len(objects))],
        dtype=np.float64,
    )
    contact_coupled = float(
        sum(
            step_paths[index - 1]
            for index in range(1, len(objects))
            if foot[index - 1] or foot[index]
        )
    )
    if np.any(foot):
        first_contact = int(np.flatnonzero(foot)[0])
        post_contact = float(np.sum(step_paths[max(first_contact - 1, 0):]))
    else:
        post_contact = 0.0
    carry = bool(np.any(bilateral) and maximum_lift >= LIFT_THRESHOLD_M)
    kick = bool(
        np.any(foot)
        and planar_net >= KICK_NET_DISPLACEMENT_M
        and contact_coupled >= KICK_CONTACT_COUPLED_PATH_M
        and post_contact >= KICK_POST_CONTACT_PATH_M
    )
    return {
        "maximum_lift_m": maximum_lift,
        "bilateral_contact_frames": int(np.count_nonzero(bilateral)),
        "foot_contact_frames": int(np.count_nonzero(foot)),
        "planar_object_net_displacement_m": planar_net,
        "contact_coupled_planar_path_m": contact_coupled,
        "post_first_contact_planar_path_m": post_contact,
        "maximum_robot_root_height_loss_m": root_height_loss,
        "maximum_robot_root_tilt_deg": maximum_tilt,
        "physical_fall": bool(physical_fall),
        "safe_carry_success": bool(carry and not physical_fall),
        "safe_kick_success": bool(kick and not physical_fall),
    }


def expected_seed(contract: dict[str, Any], variant_id: int, task: str, shard: int) -> int:
    row = contract["variants"][variant_id]
    if int(row["variant_id"]) != variant_id:
        raise ValueError("contract variant ordering drift")
    key = "carry_shard_seeds" if task == "CarryBox" else "kick_shard_seeds"
    return int(row[key][shard])


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    action_root = args.action_corpus_root.resolve()
    audit_root = args.action_audit_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit: {output}")
    contract = read_json(contract_path)
    if (
        contract.get("protocol") != CONTRACT_PROTOCOL
        or contract.get("candidate_variant_count_per_source_motion") != EXPECTED_CANDIDATES
        or contract.get("admitted_variant_count_per_source_motion") != ADMITTED_PER_MOTION
        or len(contract.get("variants", [])) != EXPECTED_CANDIDATES
    ):
        raise ValueError("frozen BPP variant contract drift")

    candidate_rows: list[dict[str, Any]] = []
    audit_hashes: list[dict[str, Any]] = []
    trace_hash_cache: dict[Path, str] = {}
    checkpoint_identities: dict[str, set[str]] = defaultdict(set)

    for variant_id in range(EXPECTED_CANDIDATES):
        variant_audit = audit_root / f"candidate_variant_{variant_id}"
        audit_result_path = variant_audit / "RESULT.json"
        audit_manifest_path = variant_audit / "ACTION_MANIFEST.jsonl"
        audit_result = read_json(audit_result_path)
        if (
            audit_result.get("passed") is not True
            or audit_result.get("trajectory_count") != 199
            or audit_result.get("transition_count") != 139300
        ):
            raise ValueError(f"candidate action audit failed: {audit_result_path}")
        audit_hashes.append({
            "variant_id": variant_id,
            "result_path": relative(audit_result_path),
            "result_sha256": file_sha256(audit_result_path),
            "manifest_path": relative(audit_manifest_path),
            "manifest_sha256": file_sha256(audit_manifest_path),
        })
        for identity, record in audit_result["released_experts"].items():
            checkpoint_identities[str(identity)].add(str(record["sha256"]))

        manifest_rows = read_jsonl(audit_manifest_path)
        grouped_by_trace: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        for row in manifest_rows:
            trace_path = ROOT / str(row["action_target"]["trace_path"])
            grouped_by_trace[trace_path.resolve()].append(row)

        for trace_path, trace_rows in sorted(grouped_by_trace.items(), key=lambda item: str(item[0])):
            result_path = trace_path.parent / "RESULT.json"
            collector_result = read_json(result_path)
            task = str(collector_result["task_family"])
            shard_match = re.search(r"shard(\d+)", trace_path.parent.name)
            if task not in TASK_COUNTS or shard_match is None:
                raise ValueError(f"candidate shard identity drift: {trace_path}")
            shard = int(shard_match.group(1))
            seed = expected_seed(contract, variant_id, task, shard)
            if (
                collector_result.get("passed") is not True
                or collector_result.get("bpp_variant_id") != variant_id
                or collector_result.get("seed") != seed
                or collector_result.get("steps") != EXPECTED_STEPS
            ):
                raise ValueError(f"collector variant/seed contract failed: {result_path}")
            if trace_path not in trace_hash_cache:
                trace_hash_cache[trace_path] = file_sha256(trace_path)

            with np.load(trace_path, allow_pickle=False) as archive:
                envs = int(np.asarray(archive["executed_action"]).shape[1])
                causal_core = np.asarray(
                    archive["goal_policy_core_observation_before"]
                )
                if causal_core.shape != (EXPECTED_STEPS, envs, 121):
                    raise ValueError(
                        "causal deployable-core geometry drift: "
                        f"{causal_core.shape}"
                    )
                if not np.isfinite(causal_core).all():
                    raise ValueError("causal deployable core contains non-finite values")
                startup_values: dict[str, np.ndarray] = {}
                for name in STARTUP_ARRAYS:
                    if name not in archive.files:
                        raise ValueError(f"missing startup readback {name}: {trace_path}")
                    value = np.asarray(archive[name])
                    if value.shape[0] != envs or not np.isfinite(value).all():
                        raise ValueError(f"invalid startup readback {name}: {value.shape}")
                    expected = collector_result.get("startup_profile", {}).get(name, {})
                    if expected.get("sha256") != array_sha256(value):
                        raise ValueError(f"startup readback hash mismatch {name}: {trace_path}")
                    startup_values[name] = value
                if len(trace_rows) != envs:
                    raise ValueError(f"manifest/trace env mismatch: {trace_path}")

                for row in trace_rows:
                    source_id = int(row["source_motion_id"])
                    env_index = int(row["action_target"]["environment_index"])
                    if not 0 <= env_index < envs:
                        raise ValueError(f"invalid environment index: {row}")
                    profile_hash = arrays_sha256(
                        [startup_values[name][env_index] for name in STARTUP_ARRAYS]
                    )
                    trace_values = []
                    for name in TRACE_FINGERPRINT_ARRAYS:
                        if name not in archive.files:
                            raise ValueError(f"missing trace fingerprint array {name}")
                        value = np.asarray(archive[name])
                        if value.shape[0] != EXPECTED_STEPS or value.shape[1] != envs:
                            raise ValueError(f"trace geometry drift {name}: {value.shape}")
                        trace_values.append(value[:, env_index])
                    trace_fingerprint = arrays_sha256(trace_values)
                    action_fingerprint = array_sha256(
                        np.asarray(archive["executed_action"])[:, env_index]
                    )
                    physics = summarize_physics(
                        np.asarray(archive["robot_root_state_w"])[:, env_index],
                        np.asarray(archive["object_root_state_w"])[:, env_index],
                        np.asarray(archive["contact"])[:, env_index],
                    )
                    safe_success = (
                        physics["safe_carry_success"]
                        if task == "CarryBox"
                        else physics["safe_kick_success"]
                    )
                    candidate_rows.append({
                        "task": task,
                        "source_motion_id": source_id,
                        "split": split_for_motion(source_id),
                        "candidate_variant_id": variant_id,
                        "seed": seed,
                        "shard": shard,
                        "trace_path": relative(trace_path),
                        "trace_file_sha256": trace_hash_cache[trace_path],
                        "environment_index": env_index,
                        "startup_profile_sha256": profile_hash,
                        "full_trace_sha256": trace_fingerprint,
                        "executed_action_sha256": action_fingerprint,
                        "physics": physics,
                        "safe_task_success": bool(safe_success),
                    })

    expected_identities = {
        (task, source_id)
        for task, count in TASK_COUNTS.items()
        for source_id in range(count)
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(str(row["task"]), int(row["source_motion_id"]))].append(row)

    admitted_rows: list[dict[str, Any]] = []
    motion_records: list[dict[str, Any]] = []
    every_motion_has_ten = True
    every_motion_has_five_safe = True
    admitted_profiles_unique = True
    admitted_traces_unique = True
    admitted_actions_unique = True
    for identity in sorted(expected_identities):
        rows = sorted(grouped.get(identity, []), key=lambda row: int(row["candidate_variant_id"]))
        candidate_ids = [int(row["candidate_variant_id"]) for row in rows]
        ten_exact = candidate_ids == list(range(EXPECTED_CANDIDATES))
        every_motion_has_ten &= ten_exact
        safe_rows = [row for row in rows if row["safe_task_success"]]
        selected = safe_rows[:ADMITTED_PER_MOTION]
        enough = len(selected) == ADMITTED_PER_MOTION
        every_motion_has_five_safe &= enough
        profile_unique = len({row["startup_profile_sha256"] for row in selected}) == len(selected)
        trace_unique = len({row["full_trace_sha256"] for row in selected}) == len(selected)
        action_unique = len({row["executed_action_sha256"] for row in selected}) == len(selected)
        admitted_profiles_unique &= profile_unique
        admitted_traces_unique &= trace_unique
        admitted_actions_unique &= action_unique
        for rank, row in enumerate(selected):
            admitted = dict(row)
            admitted["admitted_variant_rank"] = rank
            admitted_rows.append(admitted)
        motion_records.append({
            "task": identity[0],
            "source_motion_id": identity[1],
            "split": split_for_motion(identity[1]),
            "candidate_variant_ids": candidate_ids,
            "safe_candidate_variant_ids": [
                int(row["candidate_variant_id"]) for row in safe_rows
            ],
            "admitted_candidate_variant_ids": [
                int(row["candidate_variant_id"]) for row in selected
            ],
            "ten_candidates_exact": ten_exact,
            "at_least_five_safe_successes": enough,
            "admitted_startup_profiles_unique": profile_unique,
            "admitted_full_traces_unique": trace_unique,
            "admitted_actions_unique": action_unique,
        })

    split_counts = Counter(str(row["split"]) for row in admitted_rows)
    task_counts = Counter(str(row["task"]) for row in admitted_rows)
    source_counts = Counter(
        (str(row["task"]), int(row["source_motion_id"]))
        for row in admitted_rows
    )
    checks = {
        "frozen_contract_exact": True,
        "ten_candidate_audits_pass": len(audit_hashes) == EXPECTED_CANDIDATES,
        "candidate_trajectory_count_exact_1990": (
            len(candidate_rows) == EXPECTED_CANDIDATE_TRAJECTORIES
        ),
        "all_199_source_motion_identities_present": set(grouped) == expected_identities,
        "ten_candidates_per_source_motion_exact": every_motion_has_ten,
        "at_least_five_safe_task_successes_per_source_motion": every_motion_has_five_safe,
        "first_five_safe_candidates_selected_automatically": all(
            record["admitted_candidate_variant_ids"]
            == record["safe_candidate_variant_ids"][:ADMITTED_PER_MOTION]
            for record in motion_records
        ),
        "admitted_startup_profiles_unique_within_source_motion": admitted_profiles_unique,
        "admitted_full_traces_unique_within_source_motion": admitted_traces_unique,
        "admitted_executed_actions_unique_within_source_motion": admitted_actions_unique,
        "admitted_trajectory_count_exact_995": len(admitted_rows) == EXPECTED_ADMITTED_TRAJECTORIES,
        "five_admitted_variants_per_source_motion": all(
            source_counts[identity] == ADMITTED_PER_MOTION for identity in expected_identities
        ),
        "admitted_split_counts_exact": dict(split_counts)
        == {"train": 800, "validation": 100, "test": 95},
        "admitted_task_counts_exact": dict(task_counts)
        == {"CarryBox": 500, "KickBox": 495},
        "released_checkpoint_identity_constant": all(
            len(values) == 1 for values in checkpoint_identities.values()
        ),
        "selection_independent_of_bpp_model": True,
    }
    passed = all(checks.values())
    output.mkdir(parents=True, exist_ok=False)
    candidate_path = output / "BPP_CANDIDATE_VARIANTS.jsonl"
    admitted_path = output / "BPP_ADMITTED_VARIANTS.jsonl"
    motion_path = output / "BPP_MOTION_ADMISSION.jsonl"
    for path, rows in (
        (candidate_path, candidate_rows),
        (admitted_path, admitted_rows),
        (motion_path, motion_records),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "protocol": PROTOCOL,
        "passed": passed,
        "checks": checks,
        "contract": {
            "path": relative(contract_path),
            "sha256": file_sha256(contract_path),
        },
        "candidate_trajectory_count": len(candidate_rows),
        "admitted_trajectory_count": len(admitted_rows),
        "admitted_transition_count": len(admitted_rows) * EXPECTED_STEPS,
        "admitted_five_action_interval_count": len(admitted_rows) * (EXPECTED_STEPS // 5),
        "train_five_action_interval_count": split_counts["train"] * (EXPECTED_STEPS // 5),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "candidate_action_audits": audit_hashes,
        "released_checkpoint_sha256": {
            key: sorted(values) for key, values in sorted(checkpoint_identities.items())
        },
        "artifacts": {
            "candidate_variants": candidate_path.name,
            "admitted_variants": admitted_path.name,
            "motion_admission": motion_path.name,
        },
        "selection_rule": (
            "For each immutable source-motion identity, sort the ten frozen candidates "
            "by candidate_variant_id and admit the first five that independently pass "
            "the exact safe physical task criterion."
        ),
        "claim_boundary": (
            "This admits same-embodiment sensorimotor demonstrations for official BPP "
            "adaptation. It is not BPP training, prompt dependence, or policy success."
        ),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
