#!/usr/bin/env python3
"""Collect the current G1 carry pipeline artifact status.

This script is read-only.  It records whether the active render, contact,
gauntlet, audit, and recommendation artifacts exist, and extracts their JSON
status fields when available.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/public/home/yanhongru/Curiosity")
DEFAULT_ARTIFACTS = {
    "render_summary": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3/g1_replay_render_summary.json",
    "render_check": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3/g1_replay_showcase_check.json",
    "fallback_render_summary": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_960x540/g1_replay_render_summary.json",
    "fallback_render_check": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_960x540/g1_replay_showcase_check.json",
    "fallback_abs_render_summary": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_abs_960x540/g1_replay_render_summary.json",
    "fallback_abs_render_check": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_abs_960x540/g1_replay_showcase_check.json",
    "fallback_direct_render_summary": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_direct_960x540/g1_replay_render_summary.json",
    "fallback_direct_render_check": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_direct_960x540/g1_replay_showcase_check.json",
    "fallback_ext_render_summary": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_ext_960x540/g1_replay_render_summary.json",
    "fallback_ext_render_check": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_fallback_ext_960x540/g1_replay_showcase_check.json",
    "presentation_fallback_summary": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_presentation_fallback_gif/g1_replay_presentation_fallback_summary.json",
    "contact_summary": ROOT
    / "experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_gpu_contact_next_chestpad_terminal_contact/agile_low_cradle_freebox_walk/core_world_g1_box_scene_summary.json",
    "contact_check": ROOT
    / "experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_gpu_contact_next_chestpad_terminal_contact/agile_low_cradle_freebox_walk/check.json",
    "contact_comparison_pending": ROOT
    / "experiments/reports/2026-07-07_g1_contact_followup_comparison_pending.json",
    "contact_comparison_after_168802": ROOT
    / "experiments/reports/2026-07-07_g1_contact_followup_comparison_after_168802.json",
    "contact_rescue_comparison": ROOT
    / "experiments/reports/2026-07-07_g1_contact_rescue_comparison_after_run.json",
    "contact_rescue_abs_comparison": ROOT
    / "experiments/reports/2026-07-07_g1_contact_rescue_abs_comparison_after_run.json",
    "contact_rescue_direct_comparison": ROOT
    / "experiments/reports/2026-07-07_g1_contact_rescue_direct_comparison_after_run.json",
    "balance_rescue_comparison": ROOT
    / "experiments/reports/2026-07-07_g1_balance_rescue_comparison_after_run.json",
    "late_recovery_comparison": ROOT
    / "experiments/reports/2026-07-07_g1_late_recovery_comparison_after_run.json",
    "target_window_arrest_comparison": ROOT
    / "experiments/reports/2026-07-07_g1_target_window_arrest_comparison_after_run.json",
    "box_progress_controller_comparison": ROOT
    / "experiments/reports/2026-07-07_g1_box_progress_controller_comparison_after_run.json",
    "box_progress_retention_comparison": ROOT
    / "experiments/reports/2026-07-07_g1_box_progress_retention_comparison_after_run.json",
    "prismatic_reference_summary": ROOT
    / (
        "experiments/outputs/core_world_prismatic_carrier_stand/"
        "20260707_prismatic_reference_probe_adaptive_10kg_mid/"
        "core_world_prismatic_carrier_stand_summary.json"
    ),
    "prismatic_reference_check": ROOT
    / (
        "experiments/outputs/core_world_prismatic_carrier_stand/"
        "20260707_prismatic_reference_probe_adaptive_10kg_mid/reference_check.json"
    ),
    "prismatic_reference_visual_summary": ROOT
    / (
        "experiments/visuals/prismatic_reference_showcase/"
        "20260707_prismatic_reference_probe_adaptive_10kg_mid/"
        "prismatic_reference_presentation_fallback_summary.json"
    ),
    "posture_gauntlet_summary": ROOT
    / "experiments/outputs/core_world_g1_posture_gauntlet/20260707_g1_posture_gauntlet_after_contact/g1_posture_gauntlet_summary.json",
    "completion_audit_current": ROOT
    / "experiments/reports/2026-07-07_g1_carry_completion_audit_current.json",
    "completion_audit_after_gauntlet": ROOT
    / "experiments/reports/2026-07-07_g1_carry_completion_audit_after_gauntlet.json",
    "next_actions_current": ROOT
    / "experiments/reports/2026-07-07_g1_next_carry_actions_current.json",
    "next_actions_after_audit": ROOT
    / "experiments/reports/2026-07-07_g1_next_carry_actions_after_audit.json",
}
TRACKED_SLURM_JOB_IDS = (
    "168801",
    "168802",
    "168849",
    "168850",
    "168851",
    "168882",
    "168883",
    "168895",
    "168896",
    "168900",
    "168972",
    "168986",
    "168995",
    "168997",
    "169004",
    "169006",
    "169008",
)


def _collect_slurm_jobs() -> list[dict[str, Any]]:
    """Return a lightweight squeue snapshot for active pipeline jobs."""
    command = [
        "squeue",
        "-h",
        "-j",
        ",".join(TRACKED_SLURM_JOB_IDS),
        "-o",
        "%i|%j|%T|%M|%l|%P|%R|%S",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [{"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}]
    if result.returncode != 0:
        return [{"status": "unavailable", "error": result.stderr.strip()}]
    jobs: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 8:
            continue
        jobs.append(
            {
                "job_id": parts[0],
                "name": parts[1],
                "state": parts[2],
                "elapsed": parts[3],
                "time_limit": parts[4],
                "partition": parts[5],
                "node_or_reason": parts[6],
                "start_time": parts[7],
            }
        )
    known = {job.get("job_id") for job in jobs}
    for job_id in TRACKED_SLURM_JOB_IDS:
        if job_id not in known:
            jobs.append({"job_id": job_id, "state": "not_in_squeue"})
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _read_status(path: Path) -> tuple[str | None, dict[str, Any] | None]:
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return "invalid_json", {"error": f"{type(exc).__name__}: {exc}"}
    return str(data.get("status")) if data.get("status") is not None else None, data


def _artifact_record(label: str, path: Path) -> dict[str, Any]:
    status, data = _read_status(path)
    record: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.is_file(),
        "status": status,
    }
    if data is None:
        return record
    for key in (
        "success_claim",
        "case_count",
        "present_case_count",
        "passing_case_count",
        "passed_case_count",
        "failed_case_count",
        "frame_count",
        "captured_frames",
        "completion_failures",
    ):
        if key in data:
            record[key] = data[key]
    return record


def main() -> int:
    args = parse_args()
    artifacts = [_artifact_record(label, path) for label, path in DEFAULT_ARTIFACTS.items()]
    missing = [item["label"] for item in artifacts if not item["exists"]]
    failing = [
        item["label"]
        for item in artifacts
        if item["exists"] and item.get("status") not in (None, "pass")
    ]
    report = {
        "scene_type": "core_world_g1_active_pipeline_status",
        "success_claim": "status_summary_only_not_final_carrying_success",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not missing and not failing else "incomplete",
        "tracked_slurm_jobs": _collect_slurm_jobs(),
        "missing_artifacts": missing,
        "failing_artifacts": failing,
        "artifacts": artifacts,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
