#!/usr/bin/env python3
"""Validate the Phase 01 Newton lift-and-hold task specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/public/home/yanhongru/Curiosity")
SPEC = ROOT / "experiments/configs/lift_hold_variable_mass_cup_task_v1.json"
OUT = ROOT / "experiments/outputs/lift_hold_variable_mass_cup_task_v1_validation.json"

FORBIDDEN_EXACT_KEYS = {
    "observation.state",
    "action",
    "action_abs",
    "observation.tactile_f6",
}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def flatten_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            values.append(str(key))
            values.extend(flatten_values(child))
    elif isinstance(value, list):
        for item in value:
            values.extend(flatten_values(item))
    else:
        values.append(str(value))
    return values


def main() -> int:
    failures: list[str] = []
    if not SPEC.exists() or SPEC.stat().st_size <= 0:
        failures.append(f"missing_spec:{rel(SPEC)}")
        spec: dict[str, Any] = {}
    else:
        spec = read_json(SPEC)

    if spec.get("status") != "design_ready_visual_gate_pending":
        failures.append(f"status:{spec.get('status')}")
    if spec.get("generated_trex_fields") != []:
        failures.append("generated_trex_fields_not_empty")
    if spec.get("schema_promotion") != "blocked":
        failures.append(f"schema_promotion:{spec.get('schema_promotion')}")
    if spec.get("no_model_or_training") is not True:
        failures.append("no_model_or_training_not_true")

    official = spec.get("official_source_basis", {})
    for key in ["scene_entry_point", "cup_asset", "cup_mesh", "cup_texture"]:
        path = resolve(str(official.get(key, "")))
        if not path.exists() or path.stat().st_size <= 0:
            failures.append(f"missing_official_source:{key}:{official.get(key)}")
    if official.get("first_supported_scene") not in {"cube", "pen"}:
        failures.append(f"first_supported_scene:{official.get('first_supported_scene')}")

    task = spec.get("task", {})
    if task.get("name") != "lift_and_hold_under_object_property_uncertainty":
        failures.append(f"task_name:{task.get('name')}")
    if float(task.get("episode_length_s", 0.0)) <= 0.0:
        failures.append(f"episode_length_s:{task.get('episode_length_s')}")
    for required_phase in ["approach", "close_gripper", "lift", "hold"]:
        if required_phase not in task.get("phases", []):
            failures.append(f"missing_phase:{required_phase}")

    grid = spec.get("object_parameter_grid", {})
    if len(grid.get("fill_mass_levels", [])) < 3:
        failures.append("fill_mass_levels_lt_3")
    if len(grid.get("friction_levels", [])) < 3:
        failures.append("friction_levels_lt_3")
    for item in grid.get("fill_mass_levels", []):
        if float(item.get("expected_total_mass_kg", 0.0)) <= 0.0:
            failures.append(f"bad_mass_level:{item}")
    for item in grid.get("friction_levels", []):
        if float(item.get("static_mu", 0.0)) <= 0.0 or float(item.get("dynamic_mu", 0.0)) <= 0.0:
            failures.append(f"bad_friction_level:{item}")
    pose = grid.get("pose_randomization", {})
    if float(pose.get("xy_radius_m", 0.0)) <= 0.0:
        failures.append("pose_xy_radius_not_positive")

    observations = spec.get("observations", {})
    namespaces = set(observations.get("required_namespaces", []))
    required_namespaces = {"newton.panda.*", "newton.object.*", "newton.contact.*", "newton.camera.*"}
    missing_namespaces = sorted(required_namespaces - namespaces)
    if missing_namespaces:
        failures.append(f"missing_namespaces:{missing_namespaces}")
    required_signals = set(observations.get("required_signals", []))
    for signal in ["object_body_q", "rigid_contact_count", "camera_color_rgba", "camera_depth"]:
        if signal not in required_signals:
            failures.append(f"missing_required_signal:{signal}")
    if observations.get("forbidden_promotions") is None:
        failures.append("missing_forbidden_promotions")

    metrics = spec.get("metrics", {})
    for section in ["success", "failure", "adaptation"]:
        if section not in metrics:
            failures.append(f"missing_metric_section:{section}")
    if float(metrics.get("success", {}).get("lift_height_m_min", 0.0)) <= 0.0:
        failures.append("success_lift_height_not_positive")
    if float(metrics.get("failure", {}).get("over_force_contact_proxy_threshold", 0.0)) <= 0.0:
        failures.append("over_force_threshold_not_positive")
    for term in [
        "object_motion_prediction_error",
        "contact_prediction_error",
        "bounded_useful_change",
        "safety_penalty",
        "no_op_penalty",
        "learning_progress",
    ]:
        if term not in metrics.get("curiosity_diagnostic_terms", []):
            failures.append(f"missing_curiosity_term:{term}")

    visual_gate = spec.get("visual_gate", {})
    if visual_gate.get("status") != "pending_compute_run":
        failures.append(f"visual_gate_status:{visual_gate.get('status')}")
    if visual_gate.get("must_run_on_compute_node") is not True:
        failures.append("visual_gate_not_compute_node")
    if visual_gate.get("requires_fresh_official_newton_sanity") is not True:
        failures.append("visual_gate_missing_fresh_sanity")
    if visual_gate.get("requires_manual_browser_inspection") is not True:
        failures.append("visual_gate_missing_manual_inspection")
    if len(visual_gate.get("direct_image_paths_required", [])) < 3:
        failures.append("direct_image_paths_lt_3")

    forbidden_exact_found = sorted(FORBIDDEN_EXACT_KEYS & set(flatten_values(spec)))
    allowed_forbidden_list = set(observations.get("forbidden_promotions", []))
    unexpected_forbidden = sorted(set(forbidden_exact_found) - allowed_forbidden_list)
    if unexpected_forbidden:
        failures.append(f"unexpected_forbidden_exact_values:{unexpected_forbidden}")

    result = {
        "classification": "lift_hold_variable_mass_cup_task_v1_validation",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "spec": rel(SPEC),
        "official_scene_entry_point": official.get("scene_entry_point"),
        "cup_asset": official.get("cup_asset"),
        "fill_mass_level_count": len(grid.get("fill_mass_levels", [])),
        "friction_level_count": len(grid.get("friction_levels", [])),
        "required_signal_count": len(observations.get("required_signals", [])),
        "curiosity_term_count": len(metrics.get("curiosity_diagnostic_terms", [])),
        "visual_gate_status": visual_gate.get("status"),
        "generated_trex_fields": spec.get("generated_trex_fields"),
        "schema_promotion": spec.get("schema_promotion"),
        "no_model_or_training": spec.get("no_model_or_training"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
