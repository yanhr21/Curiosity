"""Build a namespace-preserving Sharpa + f_tac source-composition manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_PREFIXES = ("observation.", "action", "action_abs")
ALLOWED_PREFIXES = ("trex_sharpa.", "taccel.ftac.", "blocked.")


@dataclass(frozen=True)
class SharpaFtacManifestPaths:
    composition_readiness: Path
    contract: Path
    output: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _project_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _validate_fields(fields: list[str]) -> dict[str, Any]:
    forbidden = [field for field in fields if field.startswith(FORBIDDEN_PREFIXES)]
    unexpected = [field for field in fields if not field.startswith(ALLOWED_PREFIXES)]
    return {
        "status": "pass" if not forbidden and not unexpected else "fail",
        "field_count": len(fields),
        "forbidden_trex_fields": forbidden,
        "unexpected_namespace_fields": unexpected,
    }


def build_sharpa_ftac_manifest(
    paths: SharpaFtacManifestPaths,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    readiness = _load_json(paths.composition_readiness)
    contract = _load_json(paths.contract)
    if readiness.get("status") != "pass":
        raise ValueError("source-composition readiness must be status=pass")
    if readiness.get("schema_promotion", {}).get("status") != "blocked":
        raise ValueError("schema promotion must remain blocked for this manifest")

    state = readiness.get("state_action_source", {})
    tactile = readiness.get("tactile_source_candidate", {})
    if not isinstance(state, dict) or not isinstance(tactile, dict):
        raise ValueError("readiness output is missing source sections")

    fields = [
        "trex_sharpa.state_action_semantics",
        "trex_sharpa.left_hand_joint_order_22dof",
        "trex_sharpa.right_hand_joint_order_22dof",
        "trex_sharpa.visual_joint_inventory",
        "taccel.ftac.patch_inventory_17_per_hand",
        "taccel.ftac.distal_streams_per_hand",
        "taccel.ftac.visual_tactile_inventory",
        "taccel.ftac.deform_candidate_not_f6",
        "blocked.calibrated_tactile_f6_10x6",
        "blocked.strict_trex_episode_export",
    ]
    field_validation = _validate_fields(fields)

    manifest = {
        "status": "pass" if field_validation["status"] == "pass" else "fail",
        "classification": "trex_sharpa_plus_taccel_ftac_source_composition_manifest_not_trex_schema",
        "note": (
            "Namespace-preserving source-composition manifest. It composes "
            "evidence and intended roles only; it does not create T-Rex "
            "observation/action/F6/deform tensors."
        ),
        "source_contract": _project_path(paths.contract, root),
        "source_readiness": _project_path(paths.composition_readiness, root),
        "dataset_fields": fields,
        "field_groups": {
            "trex_sharpa": [field for field in fields if field.startswith("trex_sharpa.")],
            "taccel_ftac": [field for field in fields if field.startswith("taccel.ftac.")],
            "blocked": [field for field in fields if field.startswith("blocked.")],
        },
        "state_action_source": {
            "source": state.get("source"),
            "evidence_summary": _project_path(Path(state.get("evidence_summary", "")), root),
            "left_hand_dof": state.get("left_hand_dof"),
            "right_hand_dof": state.get("right_hand_dof"),
            "joint_order_status": state.get("joint_order_status"),
            "role": "state_action_semantics_only",
        },
        "tactile_source_candidate": {
            "source": tactile.get("source"),
            "evidence_summary": _project_path(Path(tactile.get("evidence_summary", "")), root),
            "visual_validation": _project_path(Path(tactile.get("visual_validation", "")), root),
            "patch_count_per_hand": tactile.get("patch_count_per_hand"),
            "distal_streams_per_hand": tactile.get("distal_streams_per_hand"),
            "hand_state_status": tactile.get("hand_state_status"),
            "direct_eef_dim_if_used": tactile.get("direct_eef_dim_if_used"),
            "role": "tactile_deform_candidate_not_state_action_not_f6",
        },
        "trex_schema_status": {
            "status": "not_trex_schema",
            "schema_promotion": "blocked",
            "blocked_required_fields": [
                "observation.state",
                "action",
                "action_abs",
                "observation.images.head",
                "observation.images.wrist_right",
                "observation.images.wrist_left",
                "observation.tactile_f6",
                "observation.tactile_deform.*",
            ],
            "forbidden_policy": (
                "Do not train, collate, or run official T-Rex model sanity from "
                "this manifest. It is source-composition metadata only."
            ),
        },
        "contract_forbidden_summary": contract.get("next_executable_step", {}).get("must_not_do", []),
    }
    validation = {
        **field_validation,
        "classification": manifest["classification"],
        "source_readiness_status": readiness.get("status"),
        "schema_promotion": manifest["trex_schema_status"]["schema_promotion"],
        "note": "Validation checks namespaces and blocker semantics only; it does not create tensors.",
    }
    return manifest, validation
