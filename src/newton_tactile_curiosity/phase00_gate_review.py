#!/usr/bin/env python3
"""Phase 00 Gate 00D/00E/00F evidence review.

This script reads existing Phase 00 summaries and produces a strict gate
decision. It is a review/report generator, not training and not a simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verdict(name: str, passed: bool, evidence: str, blocker: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "evidence": evidence,
        "blocker": blocker,
    }


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def review(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    bench = load_json(args.benchmark_summary)
    candidate = load_json(args.candidate_summary)
    refcmp = load_json(args.reference_compare_summary)
    alignment = load_json(args.alignment_summary)
    channel_audit = load_json(args.channel_audit_summary) if args.channel_audit_summary else None
    semantic_matrix = load_json(args.semantic_reference_matrix) if args.semantic_reference_matrix else None
    semantic_bridge_spec = load_json(args.semantic_bridge_spec) if args.semantic_bridge_spec else None
    reference_env_availability = (
        load_json(args.reference_env_availability_summary) if args.reference_env_availability_summary else None
    )
    reference_asset_availability = (
        load_json(args.reference_asset_availability_summary) if args.reference_asset_availability_summary else None
    )
    reference_asset_reuse_plan = (
        load_json(args.reference_asset_reuse_plan) if args.reference_asset_reuse_plan else None
    )
    univtac_sanity = load_json(args.univtac_sanity_summary) if args.univtac_sanity_summary else None
    tacauchy_sanity = load_json(args.tacauchy_sanity_summary) if args.tacauchy_sanity_summary else None
    isaaclab_tacsl_sanity = (
        load_json(args.isaaclab_tacsl_sanity_summary) if args.isaaclab_tacsl_sanity_summary else None
    )

    ref_gate = refcmp.get("gate_checklist", {})
    remaining_gaps = list(ref_gate.get("remaining_reference_video_gaps", []))
    candidate_channels = list(ref_gate.get("candidate_current_channels", []))

    bench_fps = float(bench.get("benchmark_fps", 0.0))
    lift = float(candidate.get("max_object_lift_m", 0.0))
    fn_sum = float(candidate.get("max_pad_object_candidate_fn_sum", 0.0))
    ft_sum = float(candidate.get("max_pad_object_candidate_ft_sum", 0.0))
    marker_left = float(candidate.get("max_left_candidate_marker_flow_norm", 0.0))
    marker_right = float(candidate.get("max_right_candidate_marker_flow_norm", 0.0))
    area_left = float(candidate.get("max_left_candidate_contact_area_proxy_cell_ratio", 0.0))
    area_right = float(candidate.get("max_right_candidate_contact_area_proxy_cell_ratio", 0.0))
    normal_left = float(candidate.get("max_left_candidate_normal_yz_norm", 0.0))
    normal_right = float(candidate.get("max_right_candidate_normal_yz_norm", 0.0))
    force_align = alignment.get("force_metrics_best", {})
    friction_align = alignment.get("friction_metrics_best", {})
    force_rmse = float(force_align.get("relative_rmse", 1.0))
    friction_rmse = float(friction_align.get("relative_rmse", 1.0))
    force_cos = float(force_align.get("mean_cosine", 0.0))
    friction_cos = float(friction_align.get("mean_cosine", 0.0))

    checks = [
        verdict(
            "official_newton_runtime_around80_fps",
            bench.get("status") == "pass" and bench_fps >= args.runtime_acceptable_fps,
            (
                f"{bench.get('run_tag')} benchmark_fps={bench_fps} "
                f"runtime_acceptable_fps={args.runtime_acceptable_fps}"
            ),
        ),
        verdict(
            "base_grasp_lift_final_test",
            candidate.get("official_final_test_status") == "pass" and lift >= args.lift_threshold_m,
            f"{candidate.get('run_tag')} official_final_test_status={candidate.get('official_final_test_status')} max_object_lift_m={lift}",
        ),
        verdict(
            "steel_spec_material",
            candidate.get("material_notify_status") == "pass"
            and 0.299 <= min(candidate.get("observed_shape_material_mu_unique", [0.0])) <= 0.301
            and max(candidate.get("observed_shape_material_kh_unique", [0.0])) >= 9.0e11,
            (
                f"material_notify_status={candidate.get('material_notify_status')} "
                f"mu={candidate.get('observed_shape_material_mu_unique')} "
                f"kh={candidate.get('observed_shape_material_kh_unique')}"
            ),
        ),
        verdict(
            "candidate_direct_fn_ft",
            candidate.get("status") == "pass_candidate_direct_force_export" and fn_sum > 0.0 and ft_sum > 0.0,
            f"status={candidate.get('status')} Fn_sum={fn_sum} Ft_sum={ft_sum}",
        ),
        verdict(
            "sensorcontact_alignment",
            alignment.get("status") == "pass_candidate_sensor_alignment"
            and force_rmse < 1.0e-5
            and friction_rmse < 1.0e-5
            and force_cos >= 0.999
            and friction_cos >= 0.999,
            (
                f"{alignment.get('run_tag')} status={alignment.get('status')} "
                f"force_relative_rmse={force_rmse} friction_relative_rmse={friction_rmse} "
                f"force_mean_cosine={force_cos} friction_mean_cosine={friction_cos}"
            ),
        ),
        verdict(
            "normal_and_area_proxy_overlay",
            bool(candidate.get("normal_area_overlay")) and area_left > 0.0 and area_right > 0.0 and normal_left > 0.0 and normal_right > 0.0,
            (
                f"normal_area_overlay={bool(candidate.get('normal_area_overlay'))} "
                f"area_cell_ratio={area_left}/{area_right} normal_yz_norm={normal_left}/{normal_right}"
            ),
            "area remains a point-contact-density proxy, not validated real sensor area",
        ),
        verdict(
            "candidate_gel_marker_render",
            bool(candidate.get("candidate_gel_marker_render")) and marker_left > 0.0 and marker_right > 0.0,
            f"marker_flow_norm={marker_left}/{marker_right}",
            "candidate rendering is derived from force fields, not validated Taccel/hardware photometry",
        ),
        verdict(
            "reference_comparison_assets",
            refcmp.get("status") == "pass_reference_comparison_assets"
            and bool(refcmp.get("reference_metrics", {}).get("nonblank"))
            and bool(refcmp.get("candidate_metrics", {}).get("nonblank")),
            (
                f"{refcmp.get('run_tag')} status={refcmp.get('status')} "
                f"reference_nonblank={refcmp.get('reference_metrics', {}).get('nonblank')} "
                f"candidate_nonblank={refcmp.get('candidate_metrics', {}).get('nonblank')}"
            ),
        ),
    ]
    if channel_audit is not None:
        checks.append(
            verdict(
                "channel_semantic_layout_audit",
                channel_audit.get("status") == "pass_channel_audit_open_validation"
                and not channel_audit.get("failed_checks"),
                (
                    f"{channel_audit.get('run_tag')} status={channel_audit.get('status')} "
                    f"failed_checks={channel_audit.get('failed_checks')}"
                ),
                "layout/channel audit is not photometric or physical semantic validation",
            )
        )

    semantic_refs = {item.get("name"): item for item in semantic_matrix.get("references", [])} if semantic_matrix else {}
    semantic_bridge = semantic_matrix.get("active_candidate_to_reference_bridge", {}) if semantic_matrix else {}
    bridge_items = semantic_bridge_spec.get("bridge_items", []) if semantic_bridge_spec else []
    bridge_item_fields = {item.get("candidate_field") for item in bridge_items}
    univtac_env_present = bool(
        reference_env_availability
        and reference_env_availability.get("univtac", {}).get("candidate_environment_present")
    )
    tacauchy_env_present = bool(
        reference_env_availability
        and reference_env_availability.get("tacauchy", {}).get("candidate_environment_present")
    )
    tacauchy_assets_ready = bool(
        reference_asset_availability
        and reference_asset_availability.get("tacauchy", {}).get("summary") == "required_assets_present"
    )
    univtac_assets_observed = bool(
        reference_asset_availability
        and reference_asset_availability.get("univtac_bundled_tacex", {}).get("summary")
        == "key_assets_present_for_bundled_tacex_reference"
    )
    asset_reuse_plan_available = bool(
        reference_asset_reuse_plan
        and reference_asset_reuse_plan.get("classification")
        in {"planned_local_asset_reuse_not_executed", "approved_local_asset_reuse_executed"}
    )
    required_bridge_fields = {
        "candidate.newton_mjw.Fn",
        "candidate.newton_mjw.Ft",
        "candidate.newton_mjw.marker_flow",
        "candidate.newton_mjw.area_proxy",
        "candidate.newton_mjw.contact_normal",
        "candidate.newton_mjw.penetration_or_compression",
        "candidate.newton_mjw.scene_rgb",
    }
    checks.extend(
        [
            verdict(
                "semantic_reference_matrix_available",
                semantic_matrix is not None
                and "UniVTAC" in semantic_refs
                and "TaCauchy" in semantic_refs
                and "OfficialIsaacLabTacSL" in semantic_refs
                and bool(semantic_bridge),
                (
                    f"semantic_matrix={rel(args.semantic_reference_matrix, root) if args.semantic_reference_matrix else None} "
                    f"references={sorted(semantic_refs)} bridge_keys={sorted(semantic_bridge)}"
                ),
                "matrix is a source/schema bridge only, not compute-side semantic validation",
            ),
            verdict(
                "semantic_bridge_spec_available",
                semantic_bridge_spec is not None
                and semantic_bridge_spec.get("status") == "draft_blocked_by_official_reference_envs"
                and required_bridge_fields.issubset(bridge_item_fields),
                (
                    f"semantic_bridge_spec={rel(args.semantic_bridge_spec, root) if args.semantic_bridge_spec else None} "
                    f"status={semantic_bridge_spec.get('status') if semantic_bridge_spec else None} "
                    f"bridge_item_fields={sorted(field for field in bridge_item_fields if field)}"
                ),
                "bridge spec is source/document-level mapping only, not compute-side semantic validation",
            ),
            verdict(
                "reference_env_availability",
                univtac_env_present and tacauchy_env_present,
                (
                    f"summary={rel(args.reference_env_availability_summary, root) if args.reference_env_availability_summary else None} "
                    f"univtac_present={univtac_env_present} tacauchy_present={tacauchy_env_present}"
                ),
                "approved local UniVTAC/TaCauchy Python environments are not both present",
            ),
            verdict(
                "reference_asset_availability",
                univtac_assets_observed and tacauchy_assets_ready,
                (
                    f"summary={rel(args.reference_asset_availability_summary, root) if args.reference_asset_availability_summary else None} "
                    f"univtac_assets_observed={univtac_assets_observed} "
                    f"tacauchy_assets_ready={tacauchy_assets_ready}"
                ),
                "TaCauchy required official tactile assets are incomplete or missing",
            ),
            verdict(
                "reference_asset_reuse_plan_available",
                asset_reuse_plan_available,
                (
                    f"summary={rel(args.reference_asset_reuse_plan, root) if args.reference_asset_reuse_plan else None} "
                    f"classification={reference_asset_reuse_plan.get('classification') if reference_asset_reuse_plan else None}"
                ),
                "asset reuse plan is a candidate plan only and does not satisfy official asset availability",
            ),
            verdict(
                "univtac_official_reference_sanity",
                univtac_sanity is not None and univtac_sanity.get("status") == "pass_official_schema_probe",
                (
                    f"summary={rel(args.univtac_sanity_summary, root) if args.univtac_sanity_summary else None} "
                    f"status={univtac_sanity.get('status') if univtac_sanity else None}"
                ),
                "UniVTAC official compute-side sanity has not passed",
            ),
            verdict(
                "tacauchy_official_reference_sanity",
                tacauchy_sanity is not None and tacauchy_sanity.get("status") == "pass_official_schema_probe",
                (
                    f"summary={rel(args.tacauchy_sanity_summary, root) if args.tacauchy_sanity_summary else None} "
                    f"status={tacauchy_sanity.get('status') if tacauchy_sanity else None}"
                ),
                "TaCauchy official compute-side sanity has not passed",
            ),
            verdict(
                "isaaclab_tacsl_official_reference_sanity",
                isaaclab_tacsl_sanity is not None
                and isaaclab_tacsl_sanity.get("status") == "pass_official_isaaclab_tacsl_demo_exited_zero",
                (
                    f"summary={rel(args.isaaclab_tacsl_sanity_summary, root) if args.isaaclab_tacsl_sanity_summary else None} "
                    f"status={isaaclab_tacsl_sanity.get('status') if isaaclab_tacsl_sanity else None}"
                ),
                "Official IsaacLab TacSL compute-side sanity has not passed",
            ),
        ]
    )

    hard_blockers = [
        "validated gel/marker photometric semantics comparable to the reference video",
        "validated photometric/deformation marker tracking on the pad surface",
        "validated real contact-area semantics beyond the current point-contact-density proxy",
        "reference-video channel-by-channel semantic matching beyond frame-level visual metrics",
    ]
    present_hard_blockers = [gap for gap in remaining_gaps if gap in hard_blockers]
    if channel_audit is not None and channel_audit.get("status") == "pass_channel_audit_open_validation":
        present_hard_blockers = [
            gap
            for gap in present_hard_blockers
            if gap != "reference-video channel-by-channel semantic matching beyond frame-level visual metrics"
        ]
        present_hard_blockers.append("validated channel-level semantic equivalence beyond current layout audit")
    if semantic_matrix is None:
        present_hard_blockers.append("missing official tactile semantic reference matrix")
    if semantic_bridge_spec is None:
        present_hard_blockers.append("missing detailed tactile semantic bridge spec")
    if not univtac_env_present or not tacauchy_env_present:
        present_hard_blockers.append("approved UniVTAC/TaCauchy reference environments not both present")
    if not tacauchy_assets_ready:
        present_hard_blockers.append("TaCauchy official tactile assets incomplete or missing")
    if univtac_sanity is None or univtac_sanity.get("status") != "pass_official_schema_probe":
        present_hard_blockers.append("UniVTAC official reference sanity not passed")
    if tacauchy_sanity is None or tacauchy_sanity.get("status") != "pass_official_schema_probe":
        present_hard_blockers.append("TaCauchy official reference sanity not passed")
    if (
        isaaclab_tacsl_sanity is None
        or isaaclab_tacsl_sanity.get("status") != "pass_official_isaaclab_tacsl_demo_exited_zero"
    ):
        present_hard_blockers.append("Official IsaacLab TacSL reference sanity not passed")

    passed_names = [item["name"] for item in checks if item["passed"]]
    failed_names = [item["name"] for item in checks if not item["passed"]]
    semantic_failed = {
        "semantic_reference_matrix_available",
        "semantic_bridge_spec_available",
        "reference_env_availability",
        "reference_asset_availability",
        "reference_asset_reuse_plan_available",
        "univtac_official_reference_sanity",
        "tacauchy_official_reference_sanity",
        "isaaclab_tacsl_official_reference_sanity",
    }.intersection(failed_names)
    nonsemantic_failed = [name for name in failed_names if name not in semantic_failed]
    gate_00d_status = "open_positive_candidate" if not nonsemantic_failed else "open_failed_required_check"
    gate_00e_status = "open_positive_base_candidate" if not nonsemantic_failed else "open_failed_required_check"
    gate_00f_status = "pass_official_semantic_reference_sanity" if not semantic_failed else "open_official_semantic_validation_blocked"
    if present_hard_blockers:
        gate_00d_status = "open_reference_semantics_blocked"
        gate_00e_status = "open_tactile_validation_blocked"
        gate_00f_status = "open_official_semantic_validation_blocked"

    summary = {
        "classification": "phase00_gate_00d_00e_00f_review_v3",
        "run_tag": args.run_tag,
        "status": "open_not_curiosity_ready",
        "not_training_result": True,
        "not_curiosity_success": True,
        "curiosity_training_allowed": False,
        "reason_curiosity_training_not_allowed": "Gate 00D/00E/00F remain open on validated tactile semantics, official reference sanity, and final channel-level reference match.",
        "gate_00d_reference_diagnostic_status": gate_00d_status,
        "gate_00e_base_model_status": gate_00e_status,
        "gate_00f_official_semantic_validation_status": gate_00f_status,
        "checks": checks,
        "passed_checks": passed_names,
        "failed_checks": failed_names,
        "candidate_current_channels": candidate_channels,
        "remaining_reference_video_gaps": remaining_gaps,
        "hard_blockers": present_hard_blockers,
        "evidence_paths": {
            "benchmark_summary": rel(args.benchmark_summary, root),
            "candidate_summary": rel(args.candidate_summary, root),
            "candidate_video": candidate.get("video_path"),
            "candidate_sheet": candidate.get("sheet_path"),
            "reference_compare_summary": rel(args.reference_compare_summary, root),
            "reference_compare_sheet": refcmp.get("comparison_sheet", {}).get("path"),
            "alignment_summary": rel(args.alignment_summary, root),
            "channel_audit_summary": rel(args.channel_audit_summary, root) if args.channel_audit_summary else None,
            "semantic_reference_matrix": rel(args.semantic_reference_matrix, root) if args.semantic_reference_matrix else None,
            "semantic_bridge_spec": rel(args.semantic_bridge_spec, root) if args.semantic_bridge_spec else None,
            "reference_env_availability_summary": rel(args.reference_env_availability_summary, root)
            if args.reference_env_availability_summary
            else None,
            "reference_asset_availability_summary": rel(args.reference_asset_availability_summary, root)
            if args.reference_asset_availability_summary
            else None,
            "reference_asset_reuse_plan": rel(args.reference_asset_reuse_plan, root)
            if args.reference_asset_reuse_plan
            else None,
            "univtac_sanity_summary": rel(args.univtac_sanity_summary, root) if args.univtac_sanity_summary else None,
            "tacauchy_sanity_summary": rel(args.tacauchy_sanity_summary, root) if args.tacauchy_sanity_summary else None,
            "isaaclab_tacsl_sanity_summary": rel(args.isaaclab_tacsl_sanity_summary, root)
            if args.isaaclab_tacsl_sanity_summary
            else None,
        },
    }
    return summary


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Phase 00 Gate 00D/00E/00F Review",
        "",
        f"- run_tag: `{summary['run_tag']}`",
        f"- status: `{summary['status']}`",
        f"- Gate 00D: `{summary['gate_00d_reference_diagnostic_status']}`",
        f"- Gate 00E: `{summary['gate_00e_base_model_status']}`",
        f"- Gate 00F: `{summary['gate_00f_official_semantic_validation_status']}`",
        f"- curiosity_training_allowed: `{summary['curiosity_training_allowed']}`",
        "",
        "## Passed Checks",
        "",
    ]
    lines.extend(f"- `{name}`" for name in summary["passed_checks"])
    lines.extend(["", "## Failed Checks", ""])
    if summary["failed_checks"]:
        lines.extend(f"- `{name}`" for name in summary["failed_checks"])
    else:
        lines.append("- none")
    lines.extend(["", "## Hard Blockers", ""])
    if summary["hard_blockers"]:
        lines.extend(f"- {gap}" for gap in summary["hard_blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Evidence", ""])
    for key, value in summary["evidence_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "This review is intentionally conservative. Candidate force-derived tactile visuals do not close the gate until tactile semantics are validated against the reference standard.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-compare-summary", type=Path, required=True)
    parser.add_argument("--alignment-summary", type=Path, required=True)
    parser.add_argument("--channel-audit-summary", type=Path, default=None)
    parser.add_argument(
        "--semantic-reference-matrix",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/experiments/configs/phase00/ref_tactile/semantic_validation_reference_matrix_v1.json"
        ),
    )
    parser.add_argument(
        "--semantic-bridge-spec",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/experiments/configs/phase00/ref_tactile/semantic_bridge_spec_v1.json"
        ),
    )
    parser.add_argument(
        "--reference-env-availability-summary",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json"
        ),
    )
    parser.add_argument(
        "--reference-asset-availability-summary",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/experiments/configs/phase00/ref_tactile/envprep/reference_asset_availability_v1.json"
        ),
    )
    parser.add_argument(
        "--reference-asset-reuse-plan",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/experiments/configs/phase00/ref_tactile/envprep/reference_asset_reuse_plan_v1.json"
        ),
    )
    parser.add_argument("--univtac-sanity-summary", type=Path, default=None)
    parser.add_argument("--tacauchy-sanity-summary", type=Path, default=None)
    parser.add_argument("--isaaclab-tacsl-sanity-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--lift-threshold-m", type=float, default=0.15)
    parser.add_argument("--runtime-acceptable-fps", type=float, default=80.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary = review(args)
    summary_path = args.output_dir / "phase00_gate_review_summary.json"
    report_path = args.report_dir / "phase00_gate_review.md"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary, report_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
