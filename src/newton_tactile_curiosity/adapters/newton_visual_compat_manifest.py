"""Manifest helpers for real Newton visual compatibility artifacts.

This module records existing Newton rollout/camera artifacts without converting
them into T-Rex tensors. It deliberately preserves `newton.*` field names and
only links visual frames back to real rollout steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TREX_DATA_PREFIXES = (
    "observation.",
    "action",
    "action_abs",
)


@dataclass(frozen=True)
class CompatManifestPaths:
    summary: Path
    npz: Path
    preview_dir: Path
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

    arrays: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            arrays[key] = {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "nonzero": int(np.count_nonzero(value)) if value.size else 0,
            }
    return arrays


def _load_step_values(path: Path) -> list[int]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if "newton.panda.step" not in data.files:
            raise ValueError(f"{path} does not contain newton.panda.step")
        values = data["newton.panda.step"]
        if values.ndim != 1:
            raise ValueError("newton.panda.step must be a 1D array")
        return [int(item) for item in values.tolist()]


def _frame_records(
    *,
    summary: dict[str, Any],
    npz_path: Path,
    preview_dir: Path,
    root: Path,
) -> list[dict[str, Any]]:
    sample_steps = summary.get("sample_steps")
    camera_names = summary.get("camera_names")
    if not isinstance(sample_steps, list) or not all(isinstance(item, int) for item in sample_steps):
        raise ValueError("summary.sample_steps must be a list of integers")
    if not isinstance(camera_names, list) or not all(isinstance(item, str) for item in camera_names):
        raise ValueError("summary.camera_names must be a list of strings")

    step_values = _load_step_values(npz_path)
    records: list[dict[str, Any]] = []
    for sample_index, rollout_index in enumerate(sample_steps):
        if rollout_index < 0 or rollout_index >= len(step_values):
            raise ValueError(f"sample step {rollout_index} is outside newton.panda.step")
        frame_path = preview_dir / f"frame_{rollout_index:04d}.png"
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        records.append(
            {
                "sample_index": sample_index,
                "requested_sample_step": rollout_index,
                "rollout_index": rollout_index,
                "newton_panda_step_value": step_values[rollout_index],
                "frame": _project_path(frame_path, root),
                "frame_size_bytes": frame_path.stat().st_size,
                "camera_panels": camera_names,
                "source_arrays": {
                    "rgba": "newton.camera.color_rgba",
                    "depth": "newton.camera.depth",
                    "object_z": "newton.camera.object_z",
                    "rollout_step": "newton.panda.step",
                },
            }
        )
    return records


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    arrays = manifest.get("newton_arrays", {})
    sample_records = manifest.get("sample_records", [])
    dataset_fields = manifest.get("dataset_fields", [])

    array_keys = sorted(arrays.keys()) if isinstance(arrays, dict) else []
    non_newton_arrays = [key for key in array_keys if not key.startswith("newton.")]
    forbidden_dataset_fields = [
        key
        for key in dataset_fields
        if isinstance(key, str) and key.startswith(TREX_DATA_PREFIXES)
    ]
    missing_frame_records = [
        record
        for record in sample_records
        if not isinstance(record, dict) or not record.get("frame")
    ]

    status = "pass"
    failures: list[str] = []
    if non_newton_arrays:
        status = "fail"
        failures.append("manifest contains non-newton array keys")
    if forbidden_dataset_fields:
        status = "fail"
        failures.append("manifest exposes T-Rex data fields")
    if missing_frame_records:
        status = "fail"
        failures.append("sample records are missing frame paths")
    if not sample_records:
        status = "fail"
        failures.append("manifest has no sample records")

    return {
        "status": status,
        "classification": manifest.get("classification"),
        "frame_record_count": len(sample_records) if isinstance(sample_records, list) else 0,
        "newton_array_count": len(array_keys),
        "non_newton_arrays": non_newton_arrays,
        "forbidden_dataset_fields": forbidden_dataset_fields,
        "failures": failures,
        "note": "Validation checks manifest metadata only; it does not decode images or create tensors.",
    }


def build_visual_compat_manifest(
    paths: CompatManifestPaths,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_json(paths.summary)
    if summary.get("status") != "pass":
        raise ValueError("source summary must have status=pass")
    if not paths.npz.is_file():
        raise FileNotFoundError(paths.npz)
    if not paths.preview_dir.is_dir():
        raise FileNotFoundError(paths.preview_dir)

    arrays = _summarize_npz(paths.npz)
    frame_records = _frame_records(
        summary=summary,
        npz_path=paths.npz,
        preview_dir=paths.preview_dir,
        root=root,
    )
    dataset_fields = sorted(arrays.keys())

    manifest = {
        "status": "pass",
        "classification": "newton_panda_hydro_visual_compat_manifest_not_trex_schema",
        "note": (
            "Manifest-only compatibility export. It references real Newton "
            "artifacts, preserves newton.* keys, and does not create T-Rex "
            "observation/action/tactile tensors."
        ),
        "source_artifacts": {
            "summary": _project_path(paths.summary, root),
            "npz": _project_path(paths.npz, root),
            "preview_dir": _project_path(paths.preview_dir, root),
            "frame_browser": _project_path(
                paths.preview_dir / "frame_browser.html",
                root,
            ),
            "contact_sheet": _project_path(
                paths.preview_dir / "contact_sheet.png",
                root,
            ),
        },
        "scene": summary.get("scene"),
        "num_steps": summary.get("num_steps"),
        "sample_steps": summary.get("sample_steps"),
        "camera_names": summary.get("camera_names"),
        "dataset_fields": dataset_fields,
        "newton_arrays": arrays,
        "sample_records": frame_records,
        "trex_schema_status": {
            "status": "not_trex_schema",
            "missing_by_design": summary.get("trex_missing_by_design", []),
            "policy": (
                "Do not rename these Newton compatibility fields into T-Rex "
                "keys until the corresponding real T-Rex-contract fields exist."
            ),
        },
    }
    validation = _validate_manifest(manifest)
    return manifest, validation
