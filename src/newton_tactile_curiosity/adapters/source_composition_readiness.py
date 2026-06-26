"""Readiness checks for composing Sharpa state/action with f_tac tactile evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPECTED_DISTAL_STREAMS = ["thumb_3", "index_3", "middle_3", "ring_3", "little_3"]
FORBIDDEN_TERMS = ["pad", "rename", "F6", "15-DOF", "48D"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_source_composition_contract(*, contract_path: Path, root: Path) -> dict[str, Any]:
    contract = _load_json(contract_path)
    state_spec = contract.get("state_action_source", {})
    tactile_spec = contract.get("tactile_source_candidate", {})
    unresolved = contract.get("unresolved_required_fields", {})
    next_step = contract.get("next_executable_step", {})

    if not isinstance(state_spec, dict) or not isinstance(tactile_spec, dict):
        raise ValueError("contract must contain state_action_source and tactile_source_candidate objects")

    state_summary_path = _resolve(root, state_spec.get("evidence_summary", ""))
    tactile_summary_path = _resolve(root, tactile_spec.get("evidence_summary", ""))
    tactile_visual_path = _resolve(root, tactile_spec.get("visual_validation", ""))

    failures: list[str] = []
    if not state_summary_path.exists():
        failures.append("missing Sharpa state/action source evidence summary")
    if not tactile_summary_path.exists():
        failures.append("missing f_tac tactile-source evidence summary")
    if not tactile_visual_path.exists():
        failures.append("missing f_tac visual validation evidence")

    state_summary = _load_json(state_summary_path) if state_summary_path.exists() else {}
    tactile_summary = _load_json(tactile_summary_path) if tactile_summary_path.exists() else {}
    tactile_visual = _load_json(tactile_visual_path) if tactile_visual_path.exists() else {}

    joint_validation = state_summary.get("joint_order_validation", {})
    if state_summary.get("status") != "pass":
        failures.append("Sharpa source inventory status is not pass")
    if not isinstance(joint_validation, dict) or joint_validation.get("status") != "pass":
        failures.append("Sharpa joint-order validation is not pass")
    if joint_validation.get("left_count") != 22 or joint_validation.get("right_count") != 22:
        failures.append("Sharpa hand DOF count is not 22 per side")
    if not joint_validation.get("left_matches_schema") or not joint_validation.get("right_matches_schema"):
        failures.append("Sharpa joint order does not match official T-Rex schema")

    tactile_inventory = tactile_summary.get("tactile_source_inventory", {})
    distal_streams = tactile_inventory.get("distal_fingertip_streams_per_hand", [])
    if tactile_summary.get("status") != "pass":
        failures.append("f_tac tactile source inventory status is not pass")
    if tactile_visual.get("status") != "pass":
        failures.append("f_tac visual validation status is not pass")
    if tactile_inventory.get("total_patch_count_per_hand") != 17:
        failures.append("f_tac tactile patch count is not 17 per hand")
    if distal_streams != EXPECTED_DISTAL_STREAMS:
        failures.append("f_tac distal fingertip stream list does not match expected five fingers")

    hand_state_contract = tactile_summary.get("hand_state_contract", {})
    if hand_state_contract.get("actuated_joint_count") != 15:
        failures.append("f_tac hand-state incompatibility is not recorded as 15 DOF")
    if hand_state_contract.get("inferred_eef_dim_if_used_directly") != 48:
        failures.append("f_tac direct 48D EEF incompatibility is not recorded")
    if hand_state_contract.get("status") != "blocked_for_trex_state_action":
        failures.append("f_tac state/action incompatibility is not blocked")

    if not isinstance(unresolved, dict) or "calibrated_tactile_f6_10x6" not in unresolved:
        failures.append("calibrated [10,6] F6 unresolved field is not explicit")
    elif unresolved["calibrated_tactile_f6_10x6"].get("status") != "blocked":
        failures.append("calibrated [10,6] F6 is not marked blocked")

    forbidden_text = " ".join(str(item) for item in tactile_spec.get("forbidden_use", []))
    forbidden_text += " " + " ".join(str(item) for item in next_step.get("must_not_do", []))
    missing_forbidden_terms = [term for term in FORBIDDEN_TERMS if term not in forbidden_text]
    if missing_forbidden_terms:
        failures.append("composition contract is missing one or more forbidden-operation guard terms")

    status = "pass" if not failures else "blocked"
    return {
        "status": status,
        "classification": "source_composition_contract_readiness",
        "contract": str(contract_path),
        "state_action_source": {
            "source": state_spec.get("name"),
            "evidence_summary": str(state_summary_path),
            "status": state_summary.get("status"),
            "left_hand_dof": joint_validation.get("left_count"),
            "right_hand_dof": joint_validation.get("right_count"),
            "joint_order_status": joint_validation.get("status"),
        },
        "tactile_source_candidate": {
            "source": tactile_spec.get("name"),
            "evidence_summary": str(tactile_summary_path),
            "visual_validation": str(tactile_visual_path),
            "status": tactile_summary.get("status"),
            "patch_count_per_hand": tactile_inventory.get("total_patch_count_per_hand"),
            "distal_streams_per_hand": distal_streams,
            "hand_state_status": hand_state_contract.get("status"),
            "direct_eef_dim_if_used": hand_state_contract.get("inferred_eef_dim_if_used_directly"),
        },
        "schema_promotion": {
            "status": "blocked",
            "reason": "This validates a source-composition plan only. Calibrated [10,6] F6 and strict T-Rex episode inventory are still unresolved.",
        },
        "missing_forbidden_terms": missing_forbidden_terms,
        "failures": failures,
        "next_action": (
            "Build a namespace-preserving composed-source manifest/spec with strict inventory rejection for missing F6."
            if status == "pass"
            else "Fix evidence or contract guards before composing sources."
        ),
    }
