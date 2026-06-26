"""Manifest helpers for Taccel marker-derived tactile compatibility data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_TREX_PREFIXES = (
    "observation.",
    "action",
    "action_abs",
)


@dataclass(frozen=True)
class MarkerCompatManifestPaths:
    summary: Path
    quality: Path
    raw_npz: Path
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


def _summarize_npz(path: Path) -> dict[str, dict[str, Any]]:
    import numpy as np

    out: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            finite = np.isfinite(value) if value.dtype.kind in "fiu" else None
            out[key] = {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "finite_count": int(finite.sum()) if finite is not None else None,
                "nan_count": int(np.isnan(value).sum()) if value.dtype.kind == "f" else 0,
            }
    return out


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    dataset_fields = manifest.get("dataset_fields", [])
    forbidden = [
        field
        for field in dataset_fields
        if isinstance(field, str) and field.startswith(FORBIDDEN_TREX_PREFIXES)
    ]
    non_taccel = [
        field
        for field in dataset_fields
        if isinstance(field, str) and not field.startswith("taccel.marker.")
    ]
    required = {
        "taccel.marker.frame",
        "taccel.marker.current",
        "taccel.marker.rest",
        "taccel.marker.flow",
        "taccel.marker.counts",
    }
    missing = sorted(required.difference(dataset_fields))
    status = "pass" if not forbidden and not non_taccel and not missing else "fail"
    return {
        "status": status,
        "classification": manifest.get("classification"),
        "dataset_field_count": len(dataset_fields),
        "forbidden_trex_fields": forbidden,
        "non_taccel_marker_fields": non_taccel,
        "missing_required_marker_fields": missing,
        "quality_status": manifest.get("quality", {}).get("status"),
        "note": "Validation checks manifest metadata only; it does not create T-Rex tensors.",
    }


def build_marker_compat_manifest(
    paths: MarkerCompatManifestPaths,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_json(paths.summary)
    quality = _load_json(paths.quality)
    if summary.get("status") != "pass":
        raise ValueError("source summary must have status=pass")
    if quality.get("status") != "pass":
        raise ValueError("source quality must have status=pass")
    if not paths.raw_npz.is_file():
        raise FileNotFoundError(paths.raw_npz)

    arrays = _summarize_npz(paths.raw_npz)
    required_npz_keys = {"frame", "marker_current", "marker_rest", "marker_flow", "marker_counts"}
    missing_npz = sorted(required_npz_keys.difference(arrays))
    if missing_npz:
        raise ValueError(f"raw npz missing keys: {missing_npz}")

    field_map = {
        "taccel.marker.frame": "frame",
        "taccel.marker.current": "marker_current",
        "taccel.marker.rest": "marker_rest",
        "taccel.marker.flow": "marker_flow",
        "taccel.marker.counts": "marker_counts",
    }
    manifest = {
        "status": "pass",
        "classification": "taccel_marker_derived_tactile_compat_manifest_not_trex_schema",
        "note": (
            "Manifest-only compatibility export for official Taccel marker "
            "signals. It preserves taccel.marker.* fields and does not create "
            "T-Rex tactile_f6 or tactile_deform tensors."
        ),
        "source_artifacts": {
            "summary": _project_path(paths.summary, root),
            "quality": _project_path(paths.quality, root),
            "raw_npz": _project_path(paths.raw_npz, root),
            "frame_browser": _project_path(
                root / "experiments/visuals/taccel_marker_flow_tacman_20260626/marker_flow/frame_browser.html",
                root,
            ),
            "contact_sheet": _project_path(
                root / "experiments/visuals/taccel_marker_flow_tacman_20260626/marker_flow/contact_sheet.png",
                root,
            ),
        },
        "dataset_fields": sorted(field_map),
        "field_map": field_map,
        "arrays": arrays,
        "sensor_links": summary.get("sensor_links"),
        "saved_frames": summary.get("saved_frames"),
        "quality": {
            key: quality.get(key)
            for key in [
                "status",
                "nonuniform_frame_count",
                "max_displacement_m",
                "max_active_marker_count_1e-5",
            ]
        },
        "trex_schema_status": {
            "status": "not_trex_schema",
            "missing_by_design": summary.get("trex_missing_by_design", []),
            "policy": (
                "Do not rename taccel.marker.* fields into T-Rex tactile keys "
                "until a real contract and validation path exist."
            ),
        },
    }
    validation = _validate_manifest(manifest)
    return manifest, validation
