#!/usr/bin/env python3
"""Preflight for dense tactile closed-loop curiosity training.

This validates evidence contracts only. It does not train a model, build a
dataset, or claim curiosity success.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


REQUIRED_BASE_MEMBERS = {
    "left_candidate_fn_map.npy",
    "right_candidate_fn_map.npy",
    "left_candidate_ft_map.npy",
    "right_candidate_ft_map.npy",
    "left_candidate_shear_y_map.npy",
    "left_candidate_shear_z_map.npy",
    "right_candidate_shear_y_map.npy",
    "right_candidate_shear_z_map.npy",
    "left_candidate_contact_area_proxy_map.npy",
    "right_candidate_contact_area_proxy_map.npy",
    "left_candidate_center_of_pressure_proxy_yz.npy",
    "right_candidate_center_of_pressure_proxy_yz.npy",
    "left_candidate_marker_flow_y_map.npy",
    "left_candidate_marker_flow_z_map.npy",
    "right_candidate_marker_flow_y_map.npy",
    "right_candidate_marker_flow_z_map.npy",
    "object_z.npy",
    "pad_object_candidate_fn_sum.npy",
    "pad_object_candidate_ft_sum.npy",
    "pad_object_contact_count.npy",
}

REQUIRED_HYDRO_MEMBERS = {
    "left_pressure_map.npy",
    "right_pressure_map.npy",
    "left_deform_proxy_map.npy",
    "right_deform_proxy_map.npy",
    "left_calibrated_view_deform_proxy_map.npy",
    "right_calibrated_view_deform_proxy_map.npy",
    "left_shear_vector_y_map.npy",
    "left_shear_vector_z_map.npy",
    "right_shear_vector_y_map.npy",
    "right_shear_vector_z_map.npy",
    "contact_area_sum.npy",
    "max_penetration.npy",
    "fn_proxy.npy",
    "ft_capacity_proxy.npy",
    "stress_proxy.npy",
    "object_acceleration.npy",
    "object_z_acceleration.npy",
}

FORBIDDEN_OLD_FIELDS = {
    "newton.panda.rigid_contact_count",
    "candidate.modality.contact_available_mask",
    "contact_count_as_tactile",
    "offline_learning_progress_sample_reweighting_only",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path, failures: list[str]) -> bool:
    if path.exists() and path.is_file() and path.stat().st_size > 0:
        return True
    failures.append(f"missing_or_empty_file:{path}")
    return False


def zip_members(path: Path, failures: list[str]) -> set[str]:
    if not require_file(path, failures):
        return set()
    try:
        with zipfile.ZipFile(path) as zf:
            return set(zf.namelist())
    except Exception as exc:  # pragma: no cover - defensive evidence path
        failures.append(f"cannot_read_npz:{path}:{type(exc).__name__}:{exc}")
        return set()


def check_required_members(label: str, members: set[str], required: set[str], failures: list[str]) -> None:
    missing = sorted(required - members)
    for name in missing:
        failures.append(f"missing_{label}_npz_member:{name}")


def path_from_manifest(root: Path, manifest: dict[str, Any], dotted: str) -> Path:
    cur: Any = manifest
    for part in dotted.split("."):
        cur = cur[part]
    return root / str(cur)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--design-contract", type=Path, required=True)
    parser.add_argument("--baseline-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    root = args.root.resolve()
    failures: list[str] = []
    warnings: list[str] = []

    manifest = read_json(args.evidence_manifest)
    design = read_json(args.design_contract)
    baseline = read_json(args.baseline_contract)

    base_summary = path_from_manifest(root, manifest, "base_export.summary")
    base_npz = path_from_manifest(root, manifest, "base_export.npz")
    base_video = path_from_manifest(root, manifest, "base_export.video")
    base_sheet = path_from_manifest(root, manifest, "base_export.sheet")
    hydro_summary = path_from_manifest(root, manifest, "hydro_compression.summary")
    hydro_npz = path_from_manifest(root, manifest, "hydro_compression.npz")
    hydro_video = path_from_manifest(root, manifest, "hydro_compression.video")
    ref_summary = path_from_manifest(root, manifest, "reference_compare.summary")
    ref_sheet = path_from_manifest(root, manifest, "reference_compare.comparison_sheet")

    for path in [base_summary, base_npz, base_video, base_sheet, hydro_summary, hydro_npz, hydro_video, ref_summary, ref_sheet]:
        require_file(path, failures)

    base = read_json(base_summary) if base_summary.exists() else {}
    hydro = read_json(hydro_summary) if hydro_summary.exists() else {}
    ref = read_json(ref_summary) if ref_summary.exists() else {}

    if base.get("direct_tactile_claim_allowed") is not False:
        failures.append("base_direct_tactile_claim_boundary_missing")
    if base.get("not_training_result") is not True or base.get("not_curiosity_success") is not True:
        failures.append("base_nonclaim_flags_missing")
    if float(base.get("max_pad_object_candidate_fn_sum", 0.0)) <= 0.0:
        failures.append("base_candidate_fn_not_positive")
    if float(base.get("max_pad_object_candidate_ft_sum", 0.0)) <= 0.0:
        failures.append("base_candidate_ft_not_positive")
    if int(base.get("left_candidate_center_of_pressure_proxy_valid_frames", 0)) <= 0:
        failures.append("base_left_cop_proxy_missing")
    if int(base.get("right_candidate_center_of_pressure_proxy_valid_frames", 0)) <= 0:
        failures.append("base_right_cop_proxy_missing")

    if hydro.get("not_training_result") is not True or hydro.get("not_curiosity_success") is not True:
        failures.append("hydro_nonclaim_flags_missing")
    if hydro.get("scene_camera_nonblank") is not True:
        failures.append("hydro_scene_camera_blank")
    if float(hydro.get("max_left_deform_proxy_map", 0.0)) <= 0.0:
        failures.append("hydro_left_deform_proxy_not_positive")
    if float(hydro.get("max_right_deform_proxy_map", 0.0)) <= 0.0:
        failures.append("hydro_right_deform_proxy_not_positive")
    if float(hydro.get("max_contact_area_sum_m2", 0.0)) <= 0.0:
        failures.append("hydro_contact_area_proxy_not_positive")

    if ref.get("status") != "pass_reference_comparison_assets":
        failures.append("reference_comparison_not_passed")
    checklist = ref.get("gate_checklist", {})
    if checklist.get("curiosity_training_allowed") is not False:
        failures.append("reference_compare_training_boundary_missing")

    base_members = zip_members(base_npz, failures)
    hydro_members = zip_members(hydro_npz, failures)
    check_required_members("base", base_members, REQUIRED_BASE_MEMBERS, failures)
    check_required_members("hydro", hydro_members, REQUIRED_HYDRO_MEMBERS, failures)

    joined_contracts = json.dumps({"design": design, "baseline": baseline, "manifest": manifest}, sort_keys=True)
    for forbidden in FORBIDDEN_OLD_FIELDS:
        if forbidden in joined_contracts:
            failures.append(f"forbidden_old_field_present:{forbidden}")

    if design.get("intrinsic_reward", {}).get("must_affect_policy_optimization") is not True:
        failures.append("intrinsic_reward_not_required_to_affect_policy")
    if design.get("intrinsic_reward", {}).get("sample_reweighting_only_allowed") is not False:
        failures.append("sample_reweighting_not_explicitly_forbidden")
    design_ablations = set(design.get("modality_mask_training", {}).get("ablations", []))
    if not {"vision_only", "vision_only_ablation"} & design_ablations:
        failures.append("vision_only_ablation_missing")
    if "tactile_only_masked_vision" not in baseline.get("baseline_set", []):
        failures.append("tactile_only_baseline_missing")
    if "strongest_baseline_comparison" not in baseline.get("metrics", []):
        failures.append("strongest_baseline_metric_missing")

    if not baseline.get("success_claim_condition", "").startswith("harder held-out tasks beat strongest baseline"):
        failures.append("success_condition_not_strict_enough")

    status = "pass_preflight_training_contract_ready" if not failures else "fail_preflight"
    summary = {
        "classification": "phase01_dense_closed_loop_training_preflight_v1",
        "run_tag": args.run_tag,
        "status": status,
        "not_training_result": True,
        "not_curiosity_success": True,
        "evidence_manifest": str(args.evidence_manifest),
        "design_contract": str(args.design_contract),
        "baseline_contract": str(args.baseline_contract),
        "base_npz_member_count": len(base_members),
        "hydro_npz_member_count": len(hydro_members),
        "failures": failures,
        "warnings": warnings,
        "next_required_step": "real closed-loop training in tmux-held Slurm allocation with attempt ledger" if not failures else "fix preflight failures before training",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "dense_training_preflight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    report_path = args.report_dir / "dense_training_preflight.md"
    report_lines = [
        "# Dense Closed-Loop Training Preflight",
        "",
        f"- run_tag: `{args.run_tag}`",
        f"- status: `{status}`",
        f"- failures: `{len(failures)}`",
        f"- base NPZ members: `{len(base_members)}`",
        f"- hydro NPZ members: `{len(hydro_members)}`",
        "",
        "This is not training, not a checkpoint, and not curiosity success.",
        "",
    ]
    if failures:
        report_lines.append("## Failures")
        report_lines.extend(f"- `{item}`" for item in failures)
        report_lines.append("")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
