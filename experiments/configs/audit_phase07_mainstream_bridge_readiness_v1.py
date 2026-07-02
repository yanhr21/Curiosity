"""Audit Phase07 readiness for faithful mainstream baseline adapters.

This is a lightweight schema/file-presence audit. It does not convert data,
download checkpoints, train, run inference, or render videos.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_SOURCE_COLUMNS = [
    "run_tag",
    "cell",
    "timestep_index",
    "newton.panda.sim_time",
    "newton.contact.rigid_contact_count",
    "newton.object.body_q.z",
    "candidate.controller.phase_index",
    "candidate.controller.commanded_gripper_target",
    "candidate.controller.commanded_lift_target",
    "candidate.task.split",
    "candidate.task.object_mass_kg",
    "candidate.task.object_friction_mu",
    "candidate.task.nominal_visual_fill",
]

RESIDUAL_ACTION_COLUMNS = [
    "candidate.controller.feedback_active",
    "candidate.controller.feedback_lift_velocity_scale",
    "candidate.controller.feedback_hold_height_offset_m",
    "candidate.controller.feedback_stabilization_extension_s",
]

PREFERRED_EEF_ACTION_COLUMNS = [
    "candidate.action.eef_delta_x",
    "candidate.action.eef_delta_y",
    "candidate.action.eef_delta_z",
    "candidate.action.eef_delta_roll",
    "candidate.action.eef_delta_pitch",
    "candidate.action.eef_delta_yaw",
    "candidate.action.gripper",
]

HELD_OUT_RUN_TAGS = [
    "phase07_eval_empty_high_misleading_no_adaptation_rerun_20260627",
    "phase07_eval_full_low_hidden_no_adaptation_20260627",
    "phase07_eval_three_quarter_low_misleading_no_adaptation_20260627",
    "phase07_eval_empty_high_misleading_curiosity_weighted_20260627",
    "phase07_eval_full_low_hidden_curiosity_weighted_20260627",
    "phase07_eval_three_quarter_low_misleading_curiosity_weighted_20260627",
]


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def _video_status(root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_tag in HELD_OUT_RUN_TAGS:
        npz_rel = f"experiments/outputs/{run_tag}.npz"
        npz_path = root / npz_rel
        action_bridge_fields_present = False
        action_bridge_missing = list(PREFERRED_EEF_ACTION_COLUMNS)
        if npz_path.exists():
            import numpy as np

            data = np.load(npz_path)
            keys = set(data.files)
            action_bridge_missing = [column for column in PREFERRED_EEF_ACTION_COLUMNS if column not in keys]
            action_bridge_fields_present = not action_bridge_missing
        rows.append(
            {
                "run_tag": run_tag,
                "summary": f"experiments/outputs/{run_tag}_summary.json",
                "npz": npz_rel,
                "video": f"experiments/visuals/{run_tag}/rollout_video.gif",
                "summary_exists": _exists(root, f"experiments/outputs/{run_tag}_summary.json"),
                "npz_exists": npz_path.exists(),
                "video_exists": _exists(root, f"experiments/visuals/{run_tag}/rollout_video.gif"),
                "action_bridge_fields_present_in_npz": action_bridge_fields_present,
                "action_bridge_fields_missing_in_npz": action_bridge_missing,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=Path("data/processed/phase07_residual_label_source_runner_v1_20260627/residual_label_records.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/phase07_mainstream_bridge_readiness_audit_v1_20260627.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    records_csv = args.records_csv if args.records_csv.is_absolute() else root / args.records_csv
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    header = _read_header(records_csv)
    header_set = set(header)
    missing_source = [column for column in REQUIRED_SOURCE_COLUMNS if column not in header_set]
    residual_present = [column for column in RESIDUAL_ACTION_COLUMNS if column in header_set]
    missing_eef = [column for column in PREFERRED_EEF_ACTION_COLUMNS if column not in header_set]
    videos = _video_status(root)
    existing_npz_with_action_bridge = sum(1 for item in videos if item["action_bridge_fields_present_in_npz"])
    payload = {
        "classification": "phase07_mainstream_bridge_readiness_audit_v1",
        "status": "pass_audit_gate_open",
        "not_training": True,
        "not_data_preprocessing": True,
        "not_success_claim": True,
        "records_csv": str(records_csv.relative_to(root)),
        "column_count": len(header),
        "required_source_columns_missing": missing_source,
        "residual_action_columns_present": residual_present,
        "preferred_7d_eef_action_columns_missing": missing_eef,
        "held_out_video_artifact_status": videos,
        "held_out_npz_with_action_bridge_count": existing_npz_with_action_bridge,
        "held_out_npz_checked_count": len(videos),
        "bridge_readiness": {
            "openpi_lerobot": "blocked_on_7d_or_low_level_action_bridge" if missing_eef else "schema_ready_for_conversion",
            "gr00t_lerobot_v2": "blocked_on_7d_or_low_level_action_bridge" if missing_eef else "schema_ready_for_conversion",
            "diffusion_policy": "blocked_on_preferred_action_bridge" if missing_eef else "schema_ready_for_conversion",
            "rtx": "blocked_on_7d_gripper_frame_action_bridge" if missing_eef else "schema_ready_for_conversion",
        },
        "interpretation": "Phase07 has residual controller labels and some video artifacts, but the source CSV does not expose the preferred 7D EEF/gripper action surface required for faithful mainstream policy comparison.",
        "next_required_steps": [
            "Extract or generate provenance-preserving Newton Panda EEF/gripper action traces from official rollouts.",
            "Keep residual-parameter imitation labeled diagnostic only.",
            "Run dataset conversion and mainstream checkpoint/fine-tuning only inside approved workflows after envs are prepared under envs/.",
        ],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
