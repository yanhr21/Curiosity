#!/usr/bin/env python3
"""Freeze one common BPP dual-camera contract from Carry/Kick visibility gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_TRAJECTORIES = 995
ENV_SPACING_M = 30.0
FAR_CLIP_M = 20.0
# The d435 origin is rigidly inside the G1 body.  Two metres is a deliberately
# conservative planar root-to-camera envelope, larger than the full robot's
# horizontal body radius, and is combined with every admitted root trace below.
CONSERVATIVE_ROOT_TO_D435_XY_M = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--carry-result", type=Path, required=True)
    parser.add_argument("--kick-result", type=Path, required=True)
    parser.add_argument("--action-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def candidate_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["candidate_id"]): row
        for row in result["eye_in_hand_candidates"]
    }


def admitted_max_centered_root_radius(
    manifest_rows: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        path = (ROOT / str(row["trace_path"])).resolve()
        grouped.setdefault(path, []).append(row)
    maximum = 0.0
    trace_records = []
    for path, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        with np.load(path, allow_pickle=False) as archive:
            before = np.asarray(archive["robot_root_state_before_w"])
            terminal = np.asarray(archive["robot_root_state_w"])[-1:]
            obj_initial = np.asarray(archive["object_root_state_before_w"])[0]
            robot_initial = before[0]
            origins = np.asarray(archive["environment_origin_w"])
            if before.shape[0] != 700 or before.shape[2:] != (13,):
                raise ValueError(f"root trace geometry drift: {path} {before.shape}")
            if origins.shape != (before.shape[1], 3):
                raise ValueError(f"origin geometry drift: {path}")
            centered_midpoint = 0.5 * (
                robot_initial[:, :2] + obj_initial[:, :2]
            ) - origins[:, :2]
            for row in rows:
                env_index = int(row["environment_index"])
                if not 0 <= env_index < before.shape[1]:
                    raise ValueError(f"invalid admitted environment index: {row}")
                all_root = np.concatenate(
                    [before[:, env_index], terminal[:, env_index]], axis=0
                )
                centered_xy = (
                    all_root[:, :2]
                    - origins[env_index, :2]
                    - centered_midpoint[env_index]
                )
                if not np.isfinite(centered_xy).all():
                    raise ValueError(f"non-finite centered root path: {path}")
                maximum = max(
                    maximum,
                    float(np.max(np.linalg.norm(centered_xy, axis=-1))),
                )
        trace_records.append({"path": relative(path), "sha256": file_sha256(path)})
    return maximum, trace_records


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate_contract.resolve()
    carry_path = args.carry_result.resolve()
    kick_path = args.kick_result.resolve()
    admission_dir = args.action_admission.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable camera contract: {output}")
    if not output.is_relative_to((ROOT / "experiments").resolve()):
        raise ValueError("camera contract evidence must remain under ignored experiments/")

    candidate_contract = read_json(candidate_path)
    carry = read_json(carry_path)
    kick = read_json(kick_path)
    expected_candidate_hash = file_sha256(candidate_path)
    for expected_task, result in (("CarryBox", carry), ("KickBox", kick)):
        if (
            result.get("protocol")
            != "sugar_bpp_dual_camera_single_task_visibility_v1"
            or result.get("passed") is not True
            or result.get("task") != expected_task
            or result.get("source_motion_id") != 0
            or result.get("candidate_contract", {}).get("sha256")
            != expected_candidate_hash
            or result.get("agentview", {}).get("passed") is not True
        ):
            raise ValueError(f"failed or drifting {expected_task} visibility result")

    carry_candidates = candidate_map(carry)
    kick_candidates = candidate_map(kick)
    configured_ids = [
        int(row["candidate_id"])
        for row in candidate_contract["eye_in_hand_candidates"]
    ]
    joint_passes = [
        candidate_id
        for candidate_id in configured_ids
        if carry_candidates[candidate_id]["passed"]
        and kick_candidates[candidate_id]["passed"]
    ]
    if not joint_passes:
        raise RuntimeError("no predeclared eye camera passes both Carry and Kick")
    ranking = []
    for candidate_id in joint_passes:
        carry_row = carry_candidates[candidate_id]
        kick_row = kick_candidates[candidate_id]
        ranking.append(
            (
                min(
                    float(carry_row["object_pixel_median"]),
                    float(kick_row["object_pixel_median"]),
                ),
                min(
                    float(carry_row["robot_pixel_median"]),
                    float(kick_row["robot_pixel_median"]),
                ),
                -candidate_id,
                candidate_id,
            )
        )
    ranking.sort(reverse=True)
    selected_id = int(ranking[0][3])
    selected_spec = next(
        row
        for row in candidate_contract["eye_in_hand_candidates"]
        if int(row["candidate_id"]) == selected_id
    )

    admission_result_path = admission_dir / "RESULT.json"
    admission_result = read_json(admission_result_path)
    if (
        admission_result.get("protocol")
        != "sugar_bpp_five_variant_action_admission_v1"
        or admission_result.get("passed") is not True
        or admission_result.get("admitted_trajectory_count")
        != EXPECTED_TRAJECTORIES
    ):
        raise ValueError("action admission is absent or failed")
    admission_manifest_path = admission_dir / str(
        admission_result["artifacts"]["admitted_variants"]
    )
    manifest_rows = read_jsonl(admission_manifest_path)
    if len(manifest_rows) != EXPECTED_TRAJECTORIES:
        raise ValueError("admitted action manifest row count drift")
    maximum_root_radius, trace_records = admitted_max_centered_root_radius(
        manifest_rows
    )
    external_radius = float(np.hypot(3.6, 3.6))
    eye_radius_bound = maximum_root_radius + CONSERVATIVE_ROOT_TO_D435_XY_M
    maximum_camera_radius = max(external_radius, eye_radius_bound)
    neighbor_lower_bound = ENV_SPACING_M - maximum_camera_radius
    if neighbor_lower_bound <= FAR_CLIP_M:
        raise RuntimeError(
            "full admitted camera isolation fails far clip: "
            f"lower_bound={neighbor_lower_bound}"
        )

    contract = {
        "protocol": "sugar_bpp_dual_camera_render_contract_v1",
        "passed_visibility_audit": True,
        "frozen_on": "2026-09-02",
        "camera_keys": ["agentview_rgb", "eye_in_hand_rgb"],
        "saved_resolution": candidate_contract["saved_resolution"],
        "rtx_resolution": candidate_contract["rtx_resolution"],
        "saved_fps": candidate_contract["saved_fps"],
        "camera_far_clip_m": FAR_CLIP_M,
        "environment_spacing_m": ENV_SPACING_M,
        "selected_eye_candidate_id": selected_id,
        "selection_rule": candidate_contract["selection_rule"],
        "joint_passing_candidate_ids": joint_passes,
        "cameras": {
            "agentview_rgb": candidate_contract["agentview_rgb"],
            "eye_in_hand_rgb": selected_spec,
        },
        "visibility_evidence": {
            "candidate_contract": {
                "path": relative(candidate_path),
                "sha256": expected_candidate_hash,
            },
            "carry": {"path": relative(carry_path), "sha256": file_sha256(carry_path)},
            "kick": {"path": relative(kick_path), "sha256": file_sha256(kick_path)},
            "selected_candidate_metrics": {
                "CarryBox": carry_candidates[selected_id],
                "KickBox": kick_candidates[selected_id],
            },
        },
        "full_admitted_isolation_evidence": {
            "action_admission_result": {
                "path": relative(admission_result_path),
                "sha256": file_sha256(admission_result_path),
            },
            "action_admission_manifest": {
                "path": relative(admission_manifest_path),
                "sha256": file_sha256(admission_manifest_path),
                "rows": len(manifest_rows),
            },
            "unique_trace_files": trace_records,
            "maximum_centered_robot_root_xy_radius_m": maximum_root_radius,
            "conservative_root_to_d435_xy_envelope_m": (
                CONSERVATIVE_ROOT_TO_D435_XY_M
            ),
            "maximum_camera_xy_radius_bound_m": maximum_camera_radius,
        },
        "neighbor_isolation_lower_bound_m": neighbor_lower_bound,
        "claim_boundary": (
            "A model-independent, two-task physical-state visibility and complete-"
            "corpus camera-isolation pass. It is not BPP training or prompt following."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
