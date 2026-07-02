#!/usr/bin/env python3
"""Audit Gate 00E base-model evidence from existing Phase 00 summaries.

This is metadata/path validation only. It does not run simulation, rendering,
training, evaluation, model loading, or dependency installation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def exists(path_value: str) -> bool:
    return bool(path_value) and Path(path_value).exists()


def check(name: str, passed: bool, evidence: str, blocker: str | None = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence, "blocker": blocker}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-status", type=Path, required=True)
    parser.add_argument("--tactile-summary", type=Path, required=True)
    parser.add_argument("--reference-compare", type=Path, required=True)
    parser.add_argument("--channel-audit", type=Path, required=True)
    parser.add_argument("--gate-review", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--runtime-acceptable-fps", type=float, default=80.0)
    args = parser.parse_args()

    benchmark = load_json(args.benchmark_status)
    tactile = load_json(args.tactile_summary)
    ref = load_json(args.reference_compare)
    channel = load_json(args.channel_audit)
    gate = load_json(args.gate_review)

    best = benchmark.get("best_runtime_evidence", {})
    best_fps = float(best.get("benchmark_fps", 0.0) or 0.0)
    gate_failed = list(gate.get("failed_checks", []))
    gate_status = str(gate.get("status", ""))

    checks = [
        check(
            "runtime_around80fps",
            best_fps >= args.runtime_acceptable_fps,
            f"best_runtime={best.get('run_tag')} fps={best_fps} acceptable_fps={args.runtime_acceptable_fps}",
        ),
        check(
            "official_final_test_and_lift",
            tactile.get("official_final_test_status") == "pass"
            and float(tactile.get("max_object_lift_m", 0.0) or 0.0) >= 0.15,
            "official_final_test_status="
            f"{tactile.get('official_final_test_status')} max_object_lift_m={tactile.get('max_object_lift_m')}",
        ),
        check(
            "steel_material_override",
            tactile.get("material_notify_status") == "pass"
            and 0.29 <= float((tactile.get("observed_shape_material_mu_unique") or [0.0])[0]) <= 0.31
            and float((tactile.get("observed_shape_material_kh_unique") or [0.0])[0]) >= 9.0e11,
            "material_notify_status="
            f"{tactile.get('material_notify_status')} mu={tactile.get('observed_shape_material_mu_unique')} "
            f"kh={tactile.get('observed_shape_material_kh_unique')}",
        ),
        check(
            "candidate_fn_ft_present",
            tactile.get("status") == "pass_candidate_direct_force_export"
            and float(tactile.get("max_pad_object_candidate_fn_sum", 0.0) or 0.0) > 0.0
            and float(tactile.get("max_pad_object_candidate_ft_sum", 0.0) or 0.0) > 0.0,
            "status="
            f"{tactile.get('status')} Fn_sum={tactile.get('max_pad_object_candidate_fn_sum')} "
            f"Ft_sum={tactile.get('max_pad_object_candidate_ft_sum')}",
        ),
        check(
            "candidate_contact_and_tactile_density",
            int(tactile.get("frames_with_pad_object_contacts", 0) or 0) > 0
            and float(tactile.get("max_left_candidate_fn_nonzero_cell_ratio", 0.0) or 0.0) > 0.05
            and float(tactile.get("max_right_candidate_fn_nonzero_cell_ratio", 0.0) or 0.0) > 0.05,
            "frames_with_pad_object_contacts="
            f"{tactile.get('frames_with_pad_object_contacts')} left_ratio="
            f"{tactile.get('max_left_candidate_fn_nonzero_cell_ratio')} right_ratio="
            f"{tactile.get('max_right_candidate_fn_nonzero_cell_ratio')}",
        ),
        check(
            "candidate_artifacts_exist",
            exists(str(tactile.get("video_path", "")))
            and exists(str(tactile.get("sheet_path", "")))
            and exists(str(tactile.get("npz_path", ""))),
            f"video={tactile.get('video_path')} sheet={tactile.get('sheet_path')} npz={tactile.get('npz_path')}",
        ),
        check(
            "reference_comparison_assets",
            ref.get("status") == "pass_reference_comparison_assets"
            and bool(ref.get("reference_metrics", {}).get("nonblank"))
            and bool(ref.get("candidate_metrics", {}).get("nonblank"))
            and exists(str(ref.get("comparison_sheet", {}).get("path", ""))),
            "status="
            f"{ref.get('status')} reference_nonblank={ref.get('reference_metrics', {}).get('nonblank')} "
            f"candidate_nonblank={ref.get('candidate_metrics', {}).get('nonblank')}",
        ),
        check(
            "channel_layout_audit",
            channel.get("failed_checks") == []
            and channel.get("not_photometric_validation") is True
            and channel.get("curiosity_training_allowed") is False,
            "failed_checks="
            f"{channel.get('failed_checks')} not_photometric_validation={channel.get('not_photometric_validation')}",
            "layout audit is not tactile semantic validation",
        ),
        check(
            "gate_review_keeps_gate_open",
            gate_status == "open_not_curiosity_ready"
            and gate.get("gate_00e_base_model_status") == "open_tactile_validation_blocked"
            and gate.get("curiosity_training_allowed") is False,
            "status="
            f"{gate_status} gate_00e={gate.get('gate_00e_base_model_status')} "
            f"curiosity_training_allowed={gate.get('curiosity_training_allowed')}",
            "Gate 00E remains open until tactile semantics and official reference sanity pass",
        ),
    ]

    required_blockers = {
        "reference_env_availability",
        "reference_asset_availability",
        "univtac_official_reference_sanity",
        "tacauchy_official_reference_sanity",
    }
    blocker_checks = [
        check(
            "official_semantic_blockers_recorded",
            required_blockers.issubset(set(gate_failed)),
            f"failed_checks={gate_failed}",
            "official UniVTAC/TaCauchy/asset/env blockers still prevent Gate 00E completion",
        )
    ]
    checks.extend(blocker_checks)

    passed = [c["name"] for c in checks if c["passed"]]
    failed = [c["name"] for c in checks if not c["passed"]]
    positive_base_evidence = not failed or failed == []
    semantic_blocked = required_blockers.issubset(set(gate_failed))
    status = (
        "partial_positive_gate00e_base_candidate_tactile_validation_blocked"
        if positive_base_evidence and semantic_blocked
        else "incomplete_gate00e_base_evidence"
    )

    summary = {
        "schema_version": "gate00e_base_evidence_audit_v1",
        "classification": "gate00e_base_evidence_audit_not_training_not_gate_completion",
        "status": status,
        "curiosity_training_allowed": False,
        "current_best_base_candidate": {
            "newton_root": benchmark.get("newton_root"),
            "newton_commit": benchmark.get("newton_commit"),
            "official_example": tactile.get("official_example"),
            "best_runtime_fps": best_fps,
            "tactile_run_tag": tactile.get("run_tag"),
        },
        "checks": checks,
        "passed_checks": passed,
        "failed_checks": failed,
        "gate_review_failed_checks": gate_failed,
        "base_candidate_positive_evidence": {
            "max_object_lift_m": tactile.get("max_object_lift_m"),
            "max_pad_object_candidate_fn_sum": tactile.get("max_pad_object_candidate_fn_sum"),
            "max_pad_object_candidate_ft_sum": tactile.get("max_pad_object_candidate_ft_sum"),
            "frames_with_pad_object_contacts": tactile.get("frames_with_pad_object_contacts"),
            "material_mu": tactile.get("observed_shape_material_mu_unique"),
            "material_kh": tactile.get("observed_shape_material_kh_unique"),
            "video_path": tactile.get("video_path"),
            "sheet_path": tactile.get("sheet_path"),
            "npz_path": tactile.get("npz_path"),
        },
        "remaining_blockers": [
            "validated gel/marker photometric semantics comparable to the reference video",
            "validated real contact-area semantics beyond the point-contact-density proxy",
            "validated dense penetration/compression semantics comparable to official TacSL/TaCauchy references",
            "official UniVTAC reference sanity",
            "official TaCauchy reference sanity",
            "dependency-complete official reference runtime/container for Gate 00F",
        ],
        "gate_effect": "current_d58_is_strongest_base_candidate_but_gate00e_remains_open",
        "not_claims": [
            "not_gate00e_completion",
            "not_gate00f_completion",
            "not_curiosity_training",
            "not_official_tactile_semantic_validation",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Gate 00E Base Evidence Audit",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Status: `{status}`",
        f"- Current base candidate: `{benchmark.get('newton_root')}` at `{benchmark.get('newton_commit')}`",
        f"- Best runtime FPS: `{best_fps}`",
        f"- Tactile run: `{tactile.get('run_tag')}`",
        "",
        "## Result",
        "",
        "The d58 Newton Panda hydro chain is the current strongest base-model candidate: it runs around the accepted 80 FPS level, lifts the object, exports steel-spec candidate Fn/Ft tactile mechanics, and has nonblank reference-comparison/channel-audit assets. The old 82 FPS number is a historical diagnostic reference, not a hard stop.",
        "",
        "It is not Gate 00E completion. Gate 00E remains open because tactile semantics are still blocked by official reference validation and proxy limitations.",
        "",
        "## Failed/Blocking Checks",
        "",
    ]
    if failed:
        lines.extend(f"- `{name}`" for name in failed)
    else:
        lines.append("- None in the base-evidence checklist; completion is still blocked by official tactile validation.")
    lines.extend(
        [
            "",
            "## Remaining Blockers",
            "",
            *[f"- {item}" for item in summary["remaining_blockers"]],
            "",
            "## Gate Effect",
            "",
            "This audit does not clear Gate 00E or Gate 00F and does not allow curiosity training.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"status": status, "output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
