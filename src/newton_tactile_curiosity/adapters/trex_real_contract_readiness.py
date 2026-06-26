"""Readiness checks for real Newton/Taccel to T-Rex field acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PASS_STATUSES = {"present_real_contract", "accepted_exact_equivalent"}
FORBIDDEN_DATASET_PREFIXES = ("observation.", "action", "action_abs")
ALLOWED_COMPAT_PREFIXES = ("newton.", "taccel.marker.")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate_real_contract_readiness(
    *,
    contract_path: Path,
    compat_manifest_path: Path,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    manifest = _load_json(compat_manifest_path)
    required = contract.get("required_trex_fields")
    if not isinstance(required, dict):
        raise ValueError("contract.required_trex_fields must be an object")
    dataset_fields = manifest.get("dataset_fields")
    if not isinstance(dataset_fields, list) or not all(isinstance(item, str) for item in dataset_fields):
        raise ValueError("manifest.dataset_fields must be a list of strings")

    field_results: list[dict[str, Any]] = []
    missing_or_unaccepted: list[str] = []
    for key, spec in required.items():
        if not isinstance(spec, dict):
            raise ValueError(f"contract field {key} must be an object")
        source_fields = spec.get("current_compat_sources", [])
        if not isinstance(source_fields, list):
            source_fields = []
        source_hits = [field for field in source_fields if field in dataset_fields]
        status = spec.get("current_status", "missing_real_contract")
        accepted = status in PASS_STATUSES
        if not accepted:
            missing_or_unaccepted.append(key)
        field_results.append(
            {
                "name": key,
                "status": status,
                "accepted_as_real_trex_contract": accepted,
                "required_shape": spec.get("required_shape"),
                "required_source_type": spec.get("required_source_type"),
                "compat_source_hits": source_hits,
                "reason": spec.get("reason"),
            }
        )

    forbidden_dataset_fields = [
        field for field in dataset_fields if field.startswith(FORBIDDEN_DATASET_PREFIXES)
    ]
    unexpected_compat_fields = [
        field for field in dataset_fields if not field.startswith(ALLOWED_COMPAT_PREFIXES)
    ]
    status = (
        "pass"
        if not missing_or_unaccepted and not forbidden_dataset_fields and not unexpected_compat_fields
        else "blocked"
    )
    return {
        "status": status,
        "classification": "real_trex_adapter_contract_readiness",
        "note": (
            "This is a readiness audit only. It must remain blocked until every "
            "T-Rex-required field is backed by a real accepted source."
        ),
        "contract": str(contract_path),
        "compat_manifest": str(compat_manifest_path),
        "required_field_count": len(required),
        "accepted_field_count": sum(1 for item in field_results if item["accepted_as_real_trex_contract"]),
        "blocked_field_count": len(missing_or_unaccepted),
        "blocked_fields": missing_or_unaccepted,
        "forbidden_dataset_fields": forbidden_dataset_fields,
        "unexpected_compat_fields": unexpected_compat_fields,
        "field_results": field_results,
        "next_action": (
            "Acquire or generate real sources for blocked T-Rex-required fields; "
            "do not promote compatibility namespaces by renaming or padding."
            if status != "pass"
            else "Run strict schema exporter and official T-Rex loader/model sanity next."
        ),
    }
