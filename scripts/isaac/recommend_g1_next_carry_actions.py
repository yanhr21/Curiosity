#!/usr/bin/env python3
"""Recommend next G1 carrying actions from existing audit/comparison reports.

The recommender is deliberately narrow: it reads JSON reports, classifies the
current failure mode, and emits a ranked action list.  It never claims success
and never launches Isaac.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/public/home/yanhongru/Curiosity")
DEFAULT_COMPLETION_AUDIT = ROOT / "experiments/reports/2026-07-07_g1_carry_completion_audit_current.json"
DEFAULT_CONTACT_AFTER = ROOT / "experiments/reports/2026-07-07_g1_contact_followup_comparison_after_168802.json"
DEFAULT_CONTACT_PENDING = ROOT / "experiments/reports/2026-07-07_g1_contact_followup_comparison_pending.json"
DEFAULT_CONTACT_RESCUE = ROOT / "experiments/reports/2026-07-07_g1_contact_rescue_direct_comparison_after_run.json"
DEFAULT_BALANCE_RESCUE = ROOT / "experiments/reports/2026-07-07_g1_balance_rescue_comparison_after_run.json"
DEFAULT_LATE_RECOVERY = ROOT / "experiments/reports/2026-07-07_g1_late_recovery_comparison_after_run.json"
DEFAULT_TARGET_WINDOW_ARREST = ROOT / (
    "experiments/reports/2026-07-07_g1_target_window_arrest_comparison_after_run.json"
)
DEFAULT_BOX_PROGRESS = ROOT / (
    "experiments/reports/2026-07-07_g1_box_progress_controller_comparison_after_run.json"
)
DEFAULT_GAUNTLET = ROOT / (
    "experiments/outputs/core_world_g1_posture_gauntlet/"
    "20260707_g1_posture_gauntlet_after_contact/g1_posture_gauntlet_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-audit", type=Path, default=DEFAULT_COMPLETION_AUDIT)
    parser.add_argument("--contact-after", type=Path, default=DEFAULT_CONTACT_AFTER)
    parser.add_argument("--contact-pending", type=Path, default=DEFAULT_CONTACT_PENDING)
    parser.add_argument("--contact-rescue", type=Path, default=DEFAULT_CONTACT_RESCUE)
    parser.add_argument("--balance-rescue", type=Path, default=DEFAULT_BALANCE_RESCUE)
    parser.add_argument("--late-recovery", type=Path, default=DEFAULT_LATE_RECOVERY)
    parser.add_argument("--target-window-arrest", type=Path, default=DEFAULT_TARGET_WINDOW_ARREST)
    parser.add_argument("--box-progress", type=Path, default=DEFAULT_BOX_PROGRESS)
    parser.add_argument("--gauntlet-summary", type=Path, default=DEFAULT_GAUNTLET)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _case_failures(report: dict[str, Any] | None, needle: str) -> list[str]:
    if report is None:
        return [f"{needle} report missing"]
    failures: list[str] = []
    for case in report.get("cases", []):
        blob = f"{case.get('label', '')} {case.get('stamp', '')} {case.get('case_dir', '')}"
        if needle in blob:
            failures.extend(str(item) for item in case.get("failures", []))
            if case.get("status") != "pass" and not case.get("failures"):
                failures.append(f"case status {case.get('status')} != pass")
    if not failures:
        failures.append(f"{needle} case not found or no failures reported")
    return failures


def _append_action(actions: list[dict[str, Any]], priority: int, action: str, reason: str, evidence: list[str]) -> None:
    actions.append(
        {
            "priority": int(priority),
            "action": action,
            "reason": reason,
            "evidence": evidence,
        }
    )


def _terminal_rescue_failed(report: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if report is None:
        return False, []
    cases = [
        case
        for case in report.get("cases", [])
        if "terminal" in f"{case.get('label', '')} {case.get('stamp', '')} {case.get('case_dir', '')}"
        and "baseline" not in str(case.get("label", ""))
    ]
    if not cases:
        return False, ["contact rescue report has no terminal rescue cases"]
    failed_cases = [case for case in cases if case.get("status") != "pass"]
    evidence = [
        "{label}: status={status}, fall_events={fall}, box_drop_events={drop}".format(
            label=case.get("label", case.get("stamp", "unknown")),
            status=case.get("status"),
            fall=case.get("fall_events"),
            drop=case.get("box_drop_events"),
        )
        for case in cases
    ]
    return len(failed_cases) == len(cases), evidence


def _balance_rescue_failed_after_progress(report: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if report is None:
        return False, []
    cases = [
        case
        for case in report.get("cases", [])
        if str(case.get("label", "")).startswith("nopad_")
        or "balance_rescue" in str(case.get("stamp", ""))
    ]
    if not cases:
        return False, ["balance rescue report has no non-pad rescue cases"]
    failed_cases = [case for case in cases if case.get("status") != "pass"]
    progressed_failed_cases = [
        case
        for case in failed_cases
        if float(case.get("final_robot_target_directed_travel_m") or 0.0) >= 2.0
        and float(case.get("final_box_target_directed_travel_m") or 0.0) >= 2.0
        and (int(case.get("fall_events") or 0) > 0 or int(case.get("box_drop_events") or 0) > 0)
    ]
    evidence = [
        (
            "{label}: status={status}, robot_travel={robot:.3f}, box_travel={box:.3f}, "
            "fall_events={fall}, box_drop_events={drop}, end_streak={streak}, max_tilt={tilt:.3f}, "
            "max_box_tilt={box_tilt:.3f}"
        ).format(
            label=case.get("label", case.get("stamp", "unknown")),
            status=case.get("status"),
            robot=float(case.get("final_robot_target_directed_travel_m") or 0.0),
            box=float(case.get("final_box_target_directed_travel_m") or 0.0),
            fall=int(case.get("fall_events") or 0),
            drop=int(case.get("box_drop_events") or 0),
            streak=int(case.get("target_window_both_streak_at_end_steps") or 0),
            tilt=float(case.get("max_tilt_rad") or 0.0),
            box_tilt=float(case.get("max_box_tilt_rad") or 0.0),
        )
        for case in cases
    ]
    return len(progressed_failed_cases) > 0 and len(failed_cases) == len(cases), evidence


def _late_recovery_failed(report: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if report is None:
        return False, []
    cases = [
        case
        for case in report.get("cases", [])
        if str(case.get("label", "")).startswith("nopad_late_")
    ]
    if not cases:
        return False, ["late recovery report has no late-recovery cases"]
    failed_cases = [case for case in cases if case.get("status") != "pass"]
    evidence = [
        (
            "{label}: status={status}, robot_travel={robot:.3f}, box_travel={box:.3f}, "
            "fall_events={fall}, box_drop_events={drop}, rel={rel:.3f}, end_streak={streak}, "
            "max_tilt={tilt:.3f}, max_box_tilt={box_tilt:.3f}"
        ).format(
            label=case.get("label", case.get("stamp", "unknown")),
            status=case.get("status"),
            robot=float(case.get("final_robot_target_directed_travel_m") or 0.0),
            box=float(case.get("final_box_target_directed_travel_m") or 0.0),
            fall=int(case.get("fall_events") or 0),
            drop=int(case.get("box_drop_events") or 0),
            rel=float(case.get("final_relative_error_m") or 0.0),
            streak=int(case.get("target_window_both_streak_at_end_steps") or 0),
            tilt=float(case.get("max_tilt_rad") or 0.0),
            box_tilt=float(case.get("max_box_tilt_rad") or 0.0),
        )
        for case in cases
    ]
    return len(failed_cases) == len(cases), evidence


def _target_window_arrest_failed(report: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if report is None:
        return False, []
    cases = [
        case
        for case in report.get("cases", [])
        if str(case.get("label", "")).startswith("load05_window_")
    ]
    if not cases:
        return False, ["target-window arrest report has no load05 arrest cases"]
    failed_cases = [case for case in cases if case.get("status") != "pass"]
    evidence = [
        (
            "{label}: status={status}, robot_travel={robot:.3f}, box_travel={box:.3f}, "
            "fall_events={fall}, box_drop_events={drop}, rel={rel:.3f}, end_streak={streak}, "
            "max_tilt={tilt:.3f}, max_box_tilt={box_tilt:.3f}"
        ).format(
            label=case.get("label", case.get("stamp", "unknown")),
            status=case.get("status"),
            robot=float(case.get("final_robot_target_directed_travel_m") or 0.0),
            box=float(case.get("final_box_target_directed_travel_m") or 0.0),
            fall=int(case.get("fall_events") or 0),
            drop=int(case.get("box_drop_events") or 0),
            rel=float(case.get("final_relative_error_m") or 0.0),
            streak=int(case.get("target_window_both_streak_at_end_steps") or 0),
            tilt=float(case.get("max_tilt_rad") or 0.0),
            box_tilt=float(case.get("max_box_tilt_rad") or 0.0),
        )
        for case in cases
    ]
    return len(failed_cases) == len(cases), evidence


def _box_progress_controller_failed(report: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if report is None:
        return False, []
    cases = [
        case
        for case in report.get("cases", [])
        if str(case.get("label", "")).startswith("load05_box_progress_")
    ]
    if not cases:
        return False, ["box-progress report has no load05 box-progress cases"]
    failed_cases = [case for case in cases if case.get("status") != "pass"]
    evidence = [
        (
            "{label}: status={status}, robot_travel={robot:.3f}, box_travel={box:.3f}, "
            "fall_events={fall}, box_drop_events={drop}, rel={rel:.3f}, end_streak={streak}, "
            "max_tilt={tilt:.3f}, max_box_tilt={box_tilt:.3f}"
        ).format(
            label=case.get("label", case.get("stamp", "unknown")),
            status=case.get("status"),
            robot=float(case.get("final_robot_target_directed_travel_m") or 0.0),
            box=float(case.get("final_box_target_directed_travel_m") or 0.0),
            fall=int(case.get("fall_events") or 0),
            drop=int(case.get("box_drop_events") or 0),
            rel=float(case.get("final_relative_error_m") or 0.0),
            streak=int(case.get("target_window_both_streak_at_end_steps") or 0),
            tilt=float(case.get("max_tilt_rad") or 0.0),
            box_tilt=float(case.get("max_box_tilt_rad") or 0.0),
        )
        for case in cases
    ]
    return len(failed_cases) == len(cases), evidence


def main() -> int:
    args = parse_args()
    completion = _load(args.completion_audit)
    contact = _load(args.contact_after) or _load(args.contact_pending)
    contact_rescue = _load(args.contact_rescue)
    balance_rescue = _load(args.balance_rescue)
    late_recovery = _load(args.late_recovery)
    target_window_arrest = _load(args.target_window_arrest)
    box_progress = _load(args.box_progress)
    gauntlet = _load(args.gauntlet_summary)
    actions: list[dict[str, Any]] = []
    observations: list[str] = []
    terminal_rescue_all_failed, terminal_rescue_evidence = _terminal_rescue_failed(contact_rescue)
    balance_rescue_failed_after_progress, balance_rescue_evidence = _balance_rescue_failed_after_progress(
        balance_rescue
    )
    late_recovery_all_failed, late_recovery_evidence = _late_recovery_failed(late_recovery)
    target_window_arrest_all_failed, target_window_arrest_evidence = _target_window_arrest_failed(
        target_window_arrest
    )
    box_progress_all_failed, box_progress_evidence = _box_progress_controller_failed(box_progress)

    if contact_rescue is not None:
        observations.append(f"contact rescue status: {contact_rescue.get('status')}")
        observations.extend(terminal_rescue_evidence)
    if balance_rescue is not None:
        observations.append(f"balance rescue status: {balance_rescue.get('status')}")
        observations.extend(balance_rescue_evidence)
    if late_recovery is not None:
        observations.append(f"late recovery status: {late_recovery.get('status')}")
        observations.extend(late_recovery_evidence)
    if target_window_arrest is not None:
        observations.append(f"target-window arrest status: {target_window_arrest.get('status')}")
        observations.extend(target_window_arrest_evidence)
    if box_progress is not None:
        observations.append(f"box-progress controller status: {box_progress.get('status')}")
        observations.extend(box_progress_evidence)

    if completion is None:
        observations.append(f"completion audit missing: {args.completion_audit}")
    else:
        observations.append(f"completion audit status: {completion.get('status')}")
        for failure in completion.get("completion_failures", []):
            observations.append(str(failure))

    if contact is None:
        _append_action(
            actions,
            10,
            "wait_for_or_generate_contact_comparison",
            "No contact comparison report is available, so terminal-contact progress/retention cannot be judged.",
            [str(args.contact_after), str(args.contact_pending)],
        )
    else:
        terminal_failures = _case_failures(contact, "terminal")
        if any("missing summary" in failure for failure in terminal_failures):
            _append_action(
                actions,
                10,
                "wait_for_168802_terminal_contact_result",
                "Terminal chest-pad case has not produced a summary yet; duplicate submissions would confuse evidence.",
                terminal_failures,
            )
        elif box_progress_all_failed:
            _append_action(
                actions,
                8,
                "replace_agile_command_wrapper_after_box_progress_failure",
                "The box-progress closed-loop command controller also failed on 0.5 kg cases. This is stronger evidence that wrapper-level command shaping around the Agile policy is not enough; move to a different locomotion/retention formulation or train/use a load-aware controller-backed policy.",
                box_progress_evidence
                + target_window_arrest_evidence[:4]
                + late_recovery_evidence[:4],
            )
        elif target_window_arrest_all_failed:
            _append_action(
                actions,
                9,
                "stop_microtuning_open_loop_agile_wrapper_for_load05",
                "The 0.5 kg target-window arrest cases also failed before target-window hold. This points to an inadequate open-loop command wrapper under load variation; the next step should replace the locomotion/retention formulation or add a controller-backed load-aware policy, rather than keep tuning final-hold scalars.",
                target_window_arrest_evidence
                + late_recovery_evidence[:4]
                + balance_rescue_evidence[:4],
            )
        elif late_recovery_all_failed:
            _append_action(
                actions,
                11,
                "replace_late_rescue_with_target_window_overshoot_arrest",
                "Both late-recovery branches failed; one overran the target window and fell/dropped, while the other under-traveled and dropped. The next diagnostic should gate/shape command authority at target-window entry and evaluate retention, not add another late rescue posture.",
                late_recovery_evidence + balance_rescue_evidence[:4] + terminal_failures[:4],
            )
        elif balance_rescue_failed_after_progress:
            _append_action(
                actions,
                12,
                "target_late_pitch_drop_stabilization_after_progress",
                "Non-pad rescue restored target-directed progress but still failed from late target-window falls/drops; the next diagnostic should focus on small late rescue/freeze stabilization after progress, not another chest-pad geometry tweak.",
                balance_rescue_evidence + terminal_failures[:6],
            )
        elif terminal_rescue_all_failed:
            _append_action(
                actions,
                15,
                "run_nopad_balance_rescue_followup",
                "Terminal chest-pad rescue variants all failed; switch the next targeted diagnostic to non-pad final-window balance/freeze stabilization.",
                terminal_rescue_evidence + terminal_failures[:8],
            )
        elif any("final_box_target_directed_travel" in failure or "target_window" in failure for failure in terminal_failures):
            _append_action(
                actions,
                20,
                "tune_delayed_chestpad_for_target_progress",
                "Terminal chest-pad is present but did not preserve target-window progress.",
                terminal_failures,
            )
        elif any("fall_events" in failure or "box_drop_events" in failure or "relative" in failure for failure in terminal_failures):
            _append_action(
                actions,
                20,
                "tune_terminal_retention_and_balance",
                "Terminal chest-pad reached progress gates but failed retention or balance gates.",
                terminal_failures,
            )

    if gauntlet is None:
        _append_action(
            actions,
            30,
            "wait_for_posture_load_gauntlet",
            "The broad posture/load verification has not run yet.",
            [str(args.gauntlet_summary)],
        )
    else:
        failed_cases = [case for case in gauntlet.get("cases", []) if case.get("passed") is not True]
        if not failed_cases:
            _append_action(
                actions,
                80,
                "broaden_unknown_load_probe_and_selector",
                "Current gauntlet cases passed; next gap is active probing and autonomous posture selection robustness.",
                [f"gauntlet case_count={gauntlet.get('case_count')}"],
            )
        for case in failed_cases:
            case_dir = str(case.get("case_dir", ""))
            failures = [str(item) for item in case.get("failures", [])]
            if "boxtilt" in case_dir:
                _append_action(
                    actions,
                    40,
                    "stabilize_distinct_boxtilt_or_replace_with_valid_second_posture",
                    "A distinct non-lowcarry posture failed strict gates.",
                    [case_dir, *failures],
                )
            elif "lightbox" in case_dir or "heavybox" in case_dir:
                _append_action(
                    actions,
                    50,
                    "add_load_adaptive_stop_hold_or_probe_conditioning",
                    "Held-out load case failed strict gates; low-carry is not load-robust.",
                    [case_dir, *failures],
                )
            elif "chestpad" in case_dir:
                _append_action(
                    actions,
                    35,
                    "continue_terminal_chestpad_stabilization",
                    "Chest-pad posture/contact case failed strict gates.",
                    [case_dir, *failures],
                )
            else:
                _append_action(
                    actions,
                    60,
                    "inspect_failed_gauntlet_case",
                    "A gauntlet case failed and needs case-specific analysis.",
                    [case_dir, *failures],
                )

    actions.sort(key=lambda item: item["priority"])
    report = {
        "scene_type": "core_world_g1_next_carry_action_recommendation",
        "success_claim": "planning_only_not_final_carrying_success",
        "status": "pass" if actions else "no_action_needed",
        "observations": observations,
        "actions": actions,
        "inputs": {
            "completion_audit": str(args.completion_audit),
            "contact_report_used": str(args.contact_after if args.contact_after.is_file() else args.contact_pending),
            "contact_rescue": str(args.contact_rescue),
            "balance_rescue": str(args.balance_rescue),
            "late_recovery": str(args.late_recovery),
            "target_window_arrest": str(args.target_window_arrest),
            "box_progress": str(args.box_progress),
            "gauntlet_summary": str(args.gauntlet_summary),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
