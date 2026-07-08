#!/usr/bin/env python3
"""Audit whether the current G1 box-carrying work satisfies the full goal.

This script is intentionally conservative.  It only reads existing JSON files
and returns nonzero unless the current evidence proves the broad goal:
walking/balanced free-box carrying, contact/posture variants, load variation,
and no rollout shortcut writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/public/home/yanhongru/Curiosity")
DEFAULT_BASELINE_SUMMARY = ROOT / (
    "experiments/outputs/core_world_g1_agile_policy_low_cradle/"
    "20260706_g1_lowcarry_168398_replay_record_retry2/"
    "agile_low_cradle_freebox_walk/core_world_g1_box_scene_summary.json"
)
DEFAULT_CONTACT_REPORT = ROOT / "experiments/reports/2026-07-07_g1_contact_followup_comparison_after_168802.json"
DEFAULT_CONTACT_PENDING_REPORT = ROOT / "experiments/reports/2026-07-07_g1_contact_followup_comparison_pending.json"
DEFAULT_GAUNTLET_SUMMARY = ROOT / (
    "experiments/outputs/core_world_g1_posture_gauntlet/"
    "20260707_g1_posture_gauntlet_after_contact/g1_posture_gauntlet_summary.json"
)
DEFAULT_SHOWCASE_CHECK = ROOT / (
    "experiments/visuals/g1_replay_showcase/"
    "20260707_g1_lowcarry_168398_replay_render_gpu_q3/"
    "g1_replay_showcase_check.json"
)


REQUIRED_GAUNTLET_CASE_HINTS = (
    "lowcarry_base",
    "chestpad_terminal",
    "boxtilt_diagnostic",
    "lowcarry_lightbox",
    "lowcarry_heavybox",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--contact-report", type=Path, default=DEFAULT_CONTACT_REPORT)
    parser.add_argument("--contact-pending-report", type=Path, default=DEFAULT_CONTACT_PENDING_REPORT)
    parser.add_argument("--gauntlet-summary", type=Path, default=DEFAULT_GAUNTLET_SUMMARY)
    parser.add_argument("--showcase-check", type=Path, default=DEFAULT_SHOWCASE_CHECK)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-box-travel", type=float, default=2.0)
    parser.add_argument("--max-box-travel", type=float, default=2.35)
    parser.add_argument("--min-target-window-final-hold-end-streak", type=int, default=40)
    parser.add_argument("--max-final-relative-error", type=float, default=0.25)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _check_file(path: Path, label: str, failures: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        failures.append(f"{label} missing: {path}")
        return None
    return _load_json(path)


def _as_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _baseline_audit(summary: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    if summary is None:
        return {"status": "missing", "failures": ["baseline summary missing"]}
    fall = _as_int(summary.get("fall_events"))
    drop = _as_int(summary.get("box_drop_events"))
    root_pose = _as_int(summary.get("root_pose_write_count_rollout"))
    root_velocity = _as_int(summary.get("root_velocity_write_count_rollout"))
    box_pose = _as_int(summary.get("box_pose_write_count_rollout"))
    box_travel = _as_float(summary.get("final_box_target_directed_travel_m"))
    final_rel = _as_float(summary.get("final_box_robot_relative_offset_error_m"), 999.0)
    final_hold_end = _as_int(summary.get("target_window_both_final_hold_streak_at_end_steps"))
    if summary.get("status") != "pass":
        failures.append(f"status {summary.get('status')} != pass")
    if fall != 0 or drop != 0:
        failures.append(f"fall/drop {fall}/{drop} != 0/0")
    if root_pose or root_velocity or box_pose:
        failures.append(f"shortcut writes root_pose/root_velocity/box_pose={root_pose}/{root_velocity}/{box_pose}")
    if box_travel < float(args.min_box_travel) or box_travel > float(args.max_box_travel):
        failures.append(f"final_box_target_directed_travel_m {box_travel} outside [{args.min_box_travel}, {args.max_box_travel}]")
    if final_rel > float(args.max_final_relative_error):
        failures.append(f"final_box_robot_relative_offset_error_m {final_rel} > {args.max_final_relative_error}")
    if final_hold_end < int(args.min_target_window_final_hold_end_streak):
        failures.append(
            "target_window_both_final_hold_streak_at_end_steps "
            f"{final_hold_end} < {args.min_target_window_final_hold_end_streak}"
        )
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "final_box_target_directed_travel_m": box_travel,
        "final_relative_error_m": final_rel,
        "target_window_both_final_hold_streak_at_end_steps": final_hold_end,
        "fall_events": fall,
        "box_drop_events": drop,
        "shortcut_writes": {
            "root_pose": root_pose,
            "root_velocity": root_velocity,
            "box_pose": box_pose,
        },
    }


def _contact_audit(report: dict[str, Any] | None) -> dict[str, Any]:
    failures: list[str] = []
    if report is None:
        return {"status": "missing", "failures": ["contact comparison report missing"]}
    terminal_cases = [
        case
        for case in report.get("cases", [])
        if "terminal" in str(case.get("label", "")) or "terminal" in str(case.get("stamp", ""))
    ]
    if not terminal_cases:
        failures.append("terminal-contact case missing from contact report")
    for case in terminal_cases:
        if case.get("status") != "pass":
            failures.append(f"terminal-contact case {case.get('label') or case.get('stamp')} status {case.get('status')} != pass")
            failures.extend(str(item) for item in case.get("failures", []))
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "terminal_cases": terminal_cases,
    }


def _gauntlet_audit(report: dict[str, Any] | None) -> dict[str, Any]:
    failures: list[str] = []
    if report is None:
        return {"status": "missing", "failures": ["posture gauntlet summary missing"]}
    cases = list(report.get("cases", []))
    if report.get("status") != "pass":
        failures.append(f"gauntlet status {report.get('status')} != pass")
    if int(report.get("case_count") or len(cases)) < len(REQUIRED_GAUNTLET_CASE_HINTS):
        failures.append(f"gauntlet case_count {report.get('case_count')} < {len(REQUIRED_GAUNTLET_CASE_HINTS)}")
    case_blob = "\n".join(str(case.get("case_dir", "")) for case in cases)
    for hint in REQUIRED_GAUNTLET_CASE_HINTS:
        if hint not in case_blob:
            failures.append(f"required gauntlet case hint missing: {hint}")
    for case in cases:
        if case.get("passed") is not True:
            failures.append(f"gauntlet case failed: {case.get('case_dir')}")
            failures.extend(str(item) for item in case.get("failures", []))
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "case_count": report.get("case_count", len(cases)),
        "passed_case_count": report.get("passed_case_count"),
    }


def _showcase_audit(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "status": "missing",
            "failures": ["showcase render check missing"],
            "required_for_goal_completion": False,
        }
    failures = list(report.get("failures") or [])
    if report.get("status") != "pass":
        failures.append(f"showcase check status {report.get('status')} != pass")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "required_for_goal_completion": False,
        "frame_count": report.get("frame_count"),
        "mp4_exists": report.get("mp4_exists"),
        "annotated_mp4_exists": report.get("annotated_mp4_exists"),
    }


def main() -> int:
    args = parse_args()
    missing_failures: list[str] = []
    baseline = _check_file(args.baseline_summary, "baseline summary", missing_failures)
    contact_path = args.contact_report if args.contact_report.is_file() else args.contact_pending_report
    contact = _check_file(contact_path, "contact comparison", missing_failures)
    gauntlet = _check_file(args.gauntlet_summary, "posture gauntlet summary", missing_failures)
    showcase = _load_json(args.showcase_check) if args.showcase_check.is_file() else None

    audits = {
        "baseline_lowcarry": _baseline_audit(baseline, args),
        "terminal_contact": _contact_audit(contact),
        "posture_load_gauntlet": _gauntlet_audit(gauntlet),
        "showcase_render": _showcase_audit(showcase),
    }
    completion_failures = list(missing_failures)
    for name, section in audits.items():
        if name == "showcase_render":
            continue
        if section.get("status") != "pass":
            completion_failures.append(f"{name}: {section.get('status')}")
            completion_failures.extend(f"{name}: {item}" for item in section.get("failures", []))

    report = {
        "scene_type": "core_world_g1_carry_completion_audit",
        "success_claim": "completion_audit_only_not_final_carrying_success",
        "status": "pass" if not completion_failures else "fail",
        "completion_failures": completion_failures,
        "audits": audits,
        "inputs": {
            "baseline_summary": str(args.baseline_summary),
            "contact_report_used": str(contact_path),
            "gauntlet_summary": str(args.gauntlet_summary),
            "showcase_check": str(args.showcase_check),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
