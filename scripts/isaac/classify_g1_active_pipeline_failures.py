#!/usr/bin/env python3
"""Classify failures in the active G1 carrying pipeline.

This is read-only and intentionally shallow: it inspects expected artifacts and
logs, then emits a JSON failure category useful for choosing the next branch.
It does not run Isaac or reinterpret failed experiments as success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/public/home/yanhongru/Curiosity")
DEFAULT_PATHS = {
    "render_log": ROOT / "logs/core_world_g1_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3_srun.log",
    "contact_log": ROOT / "logs/core_world_g1_low_cradle/20260707_g1_lowcarry_contact_next_gpu_srun.log",
    "render_check": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3/g1_replay_showcase_check.json",
    "render_summary": ROOT
    / "experiments/visuals/g1_replay_showcase/20260707_g1_lowcarry_168398_replay_render_gpu_q3/g1_replay_render_summary.json",
    "contact_check": ROOT
    / "experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_gpu_contact_next_chestpad_terminal_contact/agile_low_cradle_freebox_walk/check.json",
    "contact_summary": ROOT
    / "experiments/outputs/core_world_g1_agile_policy_low_cradle/20260707_gpu_contact_next_chestpad_terminal_contact/agile_low_cradle_freebox_walk/core_world_g1_box_scene_summary.json",
    "gauntlet_summary": ROOT
    / "experiments/outputs/core_world_g1_posture_gauntlet/20260707_g1_posture_gauntlet_after_contact/g1_posture_gauntlet_summary.json",
}


ERROR_PATTERNS = (
    ("queued", ("queued and waiting for resources",)),
    ("timeout", ("TIMEOUT", "DUE TO TIME LIMIT", "time limit")),
    ("isaac_startup_failure", ("Failed to acquire interface", "ModuleNotFoundError", "Extension", "Traceback")),
    ("render_capture_failure", ("renderer_capture", "capture", "no frames", "frame count")),
    ("physx_or_articulation_failure", ("Invalid physics simulation view", "PhysX", "articulation")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _read_text_tail(path: Path, max_chars: int = 12000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"status": "invalid_json", "failures": [f"{type(exc).__name__}: {exc}"]}


def _classify_log(path: Path) -> dict[str, Any]:
    text = _read_text_tail(path)
    if not text:
        return {
            "path": str(path),
            "exists": path.is_file(),
            "category": "missing_log",
            "matched_patterns": [],
        }
    matched: list[str] = []
    lower_text = text.lower()
    for category, patterns in ERROR_PATTERNS:
        if any(pattern.lower() in lower_text for pattern in patterns):
            matched.append(category)
    category = matched[0] if matched else "log_present_no_known_error"
    return {
        "path": str(path),
        "exists": True,
        "category": category,
        "matched_patterns": matched,
        "tail_excerpt": text[-1200:],
    }


def _classify_json(label: str, path: Path) -> dict[str, Any]:
    data = _load_json(path)
    if data is None:
        return {"label": label, "path": str(path), "exists": False, "category": "missing_artifact"}
    status = data.get("status")
    failures = [str(item) for item in data.get("failures", [])]
    if status == "pass":
        category = "pass"
    elif any("missing" in failure.lower() for failure in failures):
        category = "missing_dependency"
    elif any("fall" in failure.lower() or "drop" in failure.lower() for failure in failures):
        category = "control_fall_or_drop"
    elif any("target" in failure.lower() or "travel" in failure.lower() for failure in failures):
        category = "target_progress_or_window_failure"
    elif any("tilt" in failure.lower() or "relative" in failure.lower() or "lateral" in failure.lower() for failure in failures):
        category = "balance_or_retention_quality_failure"
    else:
        category = "json_status_failure" if status not in (None, "pass") else "json_present_no_status"
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "status": status,
        "category": category,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    log_reports = {
        "render_log": _classify_log(DEFAULT_PATHS["render_log"]),
        "contact_log": _classify_log(DEFAULT_PATHS["contact_log"]),
    }
    json_reports = {
        label: _classify_json(label, path)
        for label, path in DEFAULT_PATHS.items()
        if label not in ("render_log", "contact_log")
    }
    categories = [item["category"] for item in log_reports.values()] + [
        item["category"] for item in json_reports.values()
    ]
    report = {
        "scene_type": "core_world_g1_active_pipeline_failure_classification",
        "success_claim": "failure_classification_only_not_final_carrying_success",
        "status": "pass" if all(category == "pass" for category in categories if category != "log_present_no_known_error") else "incomplete_or_failed",
        "categories": categories,
        "logs": log_reports,
        "artifacts": json_reports,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
