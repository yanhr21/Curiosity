"""Fail-loud inventory helpers for Newton/Taccel to T-Rex sample export.

This module inspects existing artifact files and reports whether real fields
needed by the T-Rex contract are present. It does not create tensors, pad
missing fields, train models, or implement any T-Rex component.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from newton_tactile_curiosity.data_schemas.trex_contract import (
    DEFAULT_TREX_CONTRACT,
    TRexTensorContract,
)


MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov"}


@dataclass(frozen=True)
class FieldInventory:
    name: str
    expected_shape: list[int | str] | None
    status: str
    source: str | None
    observed_shape: list[int] | None
    detail: str


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    if isinstance(value, list):
        if not value:
            return [0]
        inner = _shape_of(value[0])
        if inner is None:
            return [len(value)]
        return [len(value), *inner]
    return None


def _flatten_json_keys(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            out[full] = child
            out.update(_flatten_json_keys(child, full))
        return out
    return {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {"__root__": data}
    return data


def _collect_json_candidates(paths: list[Path]) -> dict[str, tuple[Path, Any]]:
    found: dict[str, tuple[Path, Any]] = {}
    for path in paths:
        if path.suffix.lower() != ".json" or not path.is_file():
            continue
        try:
            data = _load_json(path)
        except Exception as exc:  # noqa: BLE001 - inventory must report unreadable files.
            found[f"__unreadable_json__.{path.name}"] = (path, str(exc))
            continue
        for key, value in _flatten_json_keys(data).items():
            found.setdefault(key, (path, value))
    return found


def _collect_npz_candidates(paths: list[Path]) -> dict[str, tuple[Path, Any]]:
    found: dict[str, tuple[Path, Any]] = {}
    for path in paths:
        if path.suffix.lower() != ".npz" or not path.is_file():
            continue
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    found.setdefault(key, (path, data[key]))
        except Exception as exc:  # noqa: BLE001 - inventory must report unreadable files.
            found[f"__unreadable_npz__.{path.name}"] = (path, str(exc))
    return found


def _collect_media(paths: list[Path]) -> dict[str, Path]:
    media: dict[str, Path] = {}
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        stem = path.stem.lower()
        if "head" in stem:
            media.setdefault("observation.images.head", path)
        if "wrist_right" in stem or "right_wrist" in stem:
            media.setdefault("observation.images.wrist_right", path)
        if "wrist_left" in stem or "left_wrist" in stem:
            media.setdefault("observation.images.wrist_left", path)
    return media


def _expected_fields(contract: TRexTensorContract) -> dict[str, list[int | str] | None]:
    fields: dict[str, list[int | str] | None] = {
        contract.state_key: [contract.action_dim],
        contract.action_key: [contract.action_chunk, contract.action_dim],
        contract.action_abs_key: [contract.action_dim],
        contract.tactile_f6_key: [contract.tactile_fingers, contract.tactile_channels],
        contract.slow_camera_key: ["T", 3, "H", "W"],
        contract.wrist_right_camera_key: [3, "H", "W"],
        contract.wrist_left_camera_key: [3, "H", "W"],
    }
    for key in contract.tactile_deform_prefixes:
        fields[key] = [1, "H", "W"]
    return fields


def _shape_matches(observed: list[int] | None, expected: list[int | str] | None) -> bool:
    if observed is None or expected is None:
        return False
    if len(observed) != len(expected):
        return False
    for obs, exp in zip(observed, expected, strict=True):
        if isinstance(exp, int) and obs != exp:
            return False
    return True


def _find_field(
    key: str,
    expected_shape: list[int | str] | None,
    *,
    json_keys: dict[str, tuple[Path, Any]],
    npz_keys: dict[str, tuple[Path, Any]],
    media: dict[str, Path],
) -> FieldInventory:
    if key in npz_keys:
        path, value = npz_keys[key]
        observed = _shape_of(value)
        ok = _shape_matches(observed, expected_shape)
        return FieldInventory(
            name=key,
            expected_shape=expected_shape,
            status="present" if ok else "incompatible",
            source=str(path),
            observed_shape=observed,
            detail="exact NPZ key found" if ok else "exact NPZ key found with incompatible shape",
        )
    if key in json_keys:
        path, value = json_keys[key]
        observed = _shape_of(value)
        ok = _shape_matches(observed, expected_shape)
        return FieldInventory(
            name=key,
            expected_shape=expected_shape,
            status="present" if ok else "incompatible",
            source=str(path),
            observed_shape=observed,
            detail="exact JSON key found" if ok else "exact JSON key found with incompatible shape",
        )
    if key in media:
        return FieldInventory(
            name=key,
            expected_shape=expected_shape,
            status="media_present_unvalidated",
            source=str(media[key]),
            observed_shape=None,
            detail="media file name suggests this camera key; decode/shape validation still required",
        )
    return FieldInventory(
        name=key,
        expected_shape=expected_shape,
        status="missing",
        source=None,
        observed_shape=None,
        detail="no exact real field found; exporter must fail instead of padding/faking",
    )


def discover_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(path for path in item.rglob("*") if path.is_file())
        else:
            paths.append(item)
    return sorted(set(paths))


def build_inventory(
    inputs: list[Path],
    *,
    contract: TRexTensorContract = DEFAULT_TREX_CONTRACT,
) -> dict[str, Any]:
    paths = discover_paths(inputs)
    json_keys = _collect_json_candidates(paths)
    npz_keys = _collect_npz_candidates(paths)
    media = _collect_media(paths)

    fields = [
        _find_field(
            key,
            expected,
            json_keys=json_keys,
            npz_keys=npz_keys,
            media=media,
        )
        for key, expected in _expected_fields(contract).items()
    ]
    missing = [field.name for field in fields if field.status == "missing"]
    incompatible = [field.name for field in fields if field.status == "incompatible"]
    unvalidated = [field.name for field in fields if field.status == "media_present_unvalidated"]
    status = "pass" if not missing and not incompatible and not unvalidated else "blocked"

    return {
        "status": status,
        "note": (
            "Inventory only. No tensors are created and no missing fields are "
            "padded or faked."
        ),
        "contract": contract.as_json(),
        "input_count": len(inputs),
        "scanned_file_count": len(paths),
        "fields": [asdict(field) for field in fields],
        "missing_fields": missing,
        "incompatible_fields": incompatible,
        "media_fields_requiring_decode_validation": unvalidated,
        "next_action": (
            "run strict exporter only after all required fields are present and "
            "validated from real artifacts"
            if status != "pass"
            else "candidate inventory passes; run strict LeRobot/export validator next"
        ),
    }
