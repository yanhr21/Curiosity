"""Combine compatibility manifests without creating T-Rex schema fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_PREFIXES = ("observation.", "action", "action_abs")
ALLOWED_PREFIXES = ("newton.", "taccel.marker.")


@dataclass(frozen=True)
class MultimodalManifestPaths:
    newton_visual_manifest: Path
    taccel_marker_manifest: Path
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
    unexpected = [
        field
        for field in fields
        if not field.startswith(ALLOWED_PREFIXES)
    ]
    status = "pass" if not forbidden and not unexpected else "fail"
    return {
        "status": status,
        "field_count": len(fields),
        "forbidden_trex_fields": forbidden,
        "unexpected_namespace_fields": unexpected,
    }


def build_multimodal_manifest(
    paths: MultimodalManifestPaths,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    newton_manifest = _load_json(paths.newton_visual_manifest)
    taccel_manifest = _load_json(paths.taccel_marker_manifest)
    if newton_manifest.get("status") != "pass":
        raise ValueError("Newton visual manifest must have status=pass")
    if taccel_manifest.get("status") != "pass":
        raise ValueError("Taccel marker manifest must have status=pass")

    newton_fields = newton_manifest.get("dataset_fields")
    taccel_fields = taccel_manifest.get("dataset_fields")
    if not isinstance(newton_fields, list) or not all(isinstance(item, str) for item in newton_fields):
        raise ValueError("Newton manifest dataset_fields must be a list of strings")
    if not isinstance(taccel_fields, list) or not all(isinstance(item, str) for item in taccel_fields):
        raise ValueError("Taccel manifest dataset_fields must be a list of strings")

    combined_fields = sorted([*newton_fields, *taccel_fields])
    field_validation = _validate_fields(combined_fields)
    manifest = {
        "status": "pass" if field_validation["status"] == "pass" else "fail",
        "classification": "newton_visual_plus_taccel_marker_metadata_compat_not_trex_schema",
        "note": (
            "Metadata-only multimodal compatibility manifest. It references "
            "real Newton visual/rollout fields and real Taccel marker tactile "
            "fields, but does not create T-Rex observation/action/tactile keys."
        ),
        "source_manifests": {
            "newton_visual": _project_path(paths.newton_visual_manifest, root),
            "taccel_marker": _project_path(paths.taccel_marker_manifest, root),
        },
        "dataset_fields": combined_fields,
        "field_groups": {
            "newton": sorted(newton_fields),
            "taccel_marker": sorted(taccel_fields),
        },
        "newton_summary": {
            "scene": newton_manifest.get("scene"),
            "num_steps": newton_manifest.get("num_steps"),
            "camera_names": newton_manifest.get("camera_names"),
            "sample_count": len(newton_manifest.get("sample_records", [])),
        },
        "taccel_marker_summary": {
            "sensor_links": taccel_manifest.get("sensor_links"),
            "saved_frame_count": len(taccel_manifest.get("saved_frames", [])),
            "quality": taccel_manifest.get("quality"),
        },
        "trex_schema_status": {
            "status": "not_trex_schema",
            "policy": (
                "This manifest is metadata-only. Do not train or run T-Rex "
                "adapter code from it until a real T-Rex schema exporter passes."
            ),
            "forbidden_fields": [
                "observation.*",
                "action",
                "action_abs",
                "observation.tactile_f6",
                "observation.tactile_deform.*",
            ],
        },
    }
    validation = {
        **field_validation,
        "classification": manifest["classification"],
        "newton_field_count": len(newton_fields),
        "taccel_marker_field_count": len(taccel_fields),
        "source_newton_status": newton_manifest.get("status"),
        "source_taccel_status": taccel_manifest.get("status"),
        "note": "Validation checks metadata namespaces only; it does not create tensors.",
    }
    return manifest, validation
