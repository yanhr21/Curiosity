"""Readiness validation for selecting the next real acquisition route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _exists(root: Path, value: str | None) -> bool:
    if not value:
        return False
    return (root / value).exists()


def validate_acquisition_route(
    *,
    source_audit_path: Path,
    sim_spec_path: Path,
    root: Path,
) -> dict[str, Any]:
    source_audit = _load_json(source_audit_path)
    sim_spec = _load_json(sim_spec_path)

    route_decision = source_audit.get("decision", {})
    selected = route_decision.get("selected_next_route")
    official_route = route_decision.get("official_inlab_raw_route")

    candidate_assets = sim_spec.get("candidate_assets", {})
    taccel = candidate_assets.get("taccel", {}) if isinstance(candidate_assets, dict) else {}
    newton = candidate_assets.get("newton", {}) if isinstance(candidate_assets, dict) else {}
    asset_checks = {
        "taccel.left_allegro_urdf": _exists(root, taccel.get("left_allegro_urdf")),
        "taccel.right_allegro_urdf": _exists(root, taccel.get("right_allegro_urdf")),
        "taccel.tactile_fabrication": _exists(root, taccel.get("tactile_fabrication")),
        "newton.allegro_asset_cache": _exists(root, newton.get("allegro_asset_cache")),
        "newton.allegro_asset_source": _exists(root, newton.get("allegro_asset_source")),
    }

    required_outputs = sim_spec.get("required_real_outputs", {})
    required_keys = [
        "bimanual_state_62",
        "bimanual_action_chunk_16x62",
        "action_abs_62",
        "camera_triplet",
        "tactile_f6_10x6",
        "tactile_deform_10_streams",
    ]
    missing_required_output_specs = [
        key for key in required_keys if key not in required_outputs
    ] if isinstance(required_outputs, dict) else required_keys

    first_milestone = sim_spec.get("first_executable_milestone", {})
    deliverables = first_milestone.get("deliverables", []) if isinstance(first_milestone, dict) else []
    forbidden = first_milestone.get("forbidden", []) if isinstance(first_milestone, dict) else []
    required_forbidden_terms = ["padding", "renaming", "training"]
    missing_forbidden_terms = [
        term
        for term in required_forbidden_terms
        if not any(term in str(item) for item in forbidden)
    ]

    failures: list[str] = []
    if official_route != "not_executable_from_current_public_or_local_sources":
        failures.append("official in-lab raw route is not recorded as currently unavailable")
    if selected != "future_bimanual_tactile_sim_source_spec":
        failures.append("selected route is not future_bimanual_tactile_sim_source_spec")
    if not all(asset_checks.values()):
        failures.append("one or more candidate official assets are missing")
    if missing_required_output_specs:
        failures.append("bimanual sim source spec is missing required output sections")
    if len(deliverables) < 4:
        failures.append("first executable milestone does not list enough deliverables")
    if missing_forbidden_terms:
        failures.append("first executable milestone lacks forbidden-operation guard terms")

    return {
        "status": "pass" if not failures else "blocked",
        "classification": "next_real_acquisition_route_selection_readiness",
        "source_audit": str(source_audit_path),
        "sim_spec": str(sim_spec_path),
        "official_inlab_raw_route": official_route,
        "selected_next_route": selected,
        "asset_checks": asset_checks,
        "missing_required_output_specs": missing_required_output_specs,
        "missing_forbidden_terms": missing_forbidden_terms,
        "failures": failures,
        "next_action": (
            "Run the first executable compute-node bimanual asset/sensor inventory milestone."
            if not failures
            else "Fix route-selection/spec evidence before executing acquisition work."
        ),
    }
