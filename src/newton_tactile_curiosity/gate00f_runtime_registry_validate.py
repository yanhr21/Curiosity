#!/usr/bin/env python3
"""Validate the Gate 00F reference runtime registry.

This is metadata/path validation only. It does not import target runtimes,
start containers, run simulators, load models, convert data, or install
dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_TARGETS = ("univtac", "tacauchy", "isaaclab_tacsl")
ACCEPTED_STATUS = "dependency_complete_registered"
ALLOWED_KINDS = {"python_env", "container"}
ALLOWED_CONTAINER_RUNTIMES = {"docker", "singularity", "apptainer", "enroot", "sif", "sqsh", "tar"}
ALLOWED_RESOLUTION_PATHS = {
    "reuse_existing_dependency_complete_env",
    "reuse_existing_prebuilt_container",
    "compliant_external_env_prep_then_register",
}
CONTAINER_ARTIFACT_SUFFIXES = (".sif", ".sqsh", ".tar", ".tar.gz", ".img")
IMAGE_ID_RE = re.compile(r"^(sha256:)?[0-9a-fA-F]{12,}$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def validate(registry_path: Path) -> dict[str, Any]:
    registry = load_json(registry_path)
    failures: list[str] = []
    warnings: list[str] = []
    target_results: dict[str, Any] = {}
    root = registry_path.resolve().parents[4] if len(registry_path.resolve().parents) > 4 else Path.cwd()

    excluded = [str(x) for x in as_list(registry.get("excluded_path_fragments"))]
    targets = registry.get("targets")
    if not isinstance(targets, dict):
        targets = {}
        failures.append("registry.targets must be an object")

    for name in REQUIRED_TARGETS:
        entry = targets.get(name)
        if not isinstance(entry, dict):
            failures.append(f"{name}: missing target entry")
            target_results[name] = {"status": "missing_entry"}
            continue

        kind = str(entry.get("kind", ""))
        status = str(entry.get("status", ""))
        path = str(entry.get("path", ""))
        artifact_path = str(entry.get("artifact_path", ""))
        image_ref = str(entry.get("image_ref", ""))
        image_id = str(entry.get("image_id", ""))
        container_python = str(entry.get("container_python", "python3"))
        container_runtime = str(entry.get("container_runtime", ""))
        resolution_path = str(entry.get("resolution_path", ""))
        expected_modules = [str(x) for x in as_list(entry.get("expected_modules"))]
        provenance = [str(x) for x in as_list(entry.get("provenance"))]

        entry_failures: list[str] = []
        entry_warnings: list[str] = []
        if kind not in ALLOWED_KINDS:
            entry_failures.append(f"kind must be one of {sorted(ALLOWED_KINDS)}, observed {kind}")
        if status != ACCEPTED_STATUS:
            entry_failures.append(f"status must be {ACCEPTED_STATUS}, observed {status}")
        if resolution_path not in ALLOWED_RESOLUTION_PATHS:
            entry_failures.append(
                f"resolution_path must be one of {sorted(ALLOWED_RESOLUTION_PATHS)}, observed {resolution_path}"
            )
        if not expected_modules:
            entry_failures.append("expected_modules must be nonempty")
        if not provenance:
            entry_failures.append("provenance must be nonempty")
        provenance_results: list[dict[str, Any]] = []
        for provenance_item in provenance:
            provenance_path = Path(provenance_item)
            if "<" in provenance_item or ">" in provenance_item:
                entry_failures.append(f"provenance contains placeholder brackets: {provenance_item}")
            if not provenance_path.is_absolute():
                provenance_path = root / provenance_path
            provenance_exists = provenance_path.exists()
            if not provenance_exists:
                entry_failures.append(f"provenance path does not exist: {provenance_item}")
            provenance_results.append(
                {
                    "path": provenance_item,
                    "resolved_path": str(provenance_path),
                    "exists": provenance_exists,
                }
            )
        if kind == "python_env" and not path:
            entry_failures.append("path must be nonempty")
        if kind == "container":
            if container_runtime not in ALLOWED_CONTAINER_RUNTIMES:
                entry_failures.append(
                    "container_runtime must be one of "
                    f"{sorted(ALLOWED_CONTAINER_RUNTIMES)}, observed {container_runtime}"
                )
            if "<" in container_python or ">" in container_python:
                entry_failures.append(f"container_python contains placeholder brackets: {container_python}")
            if not container_python:
                entry_failures.append("container_python must be nonempty for container entries")
            if not (artifact_path or image_id):
                entry_failures.append(
                    "container entry must provide artifact_path or local image_id; "
                    "image_ref alone is only an acquisition candidate"
                )
            if image_id:
                compact_image_id = image_id.removeprefix("sha256:")
                if not IMAGE_ID_RE.match(image_id):
                    entry_failures.append(
                        "image_id must look like a local immutable image digest or ID "
                        "(sha256:<hex> or >=12 hex chars), not a tag or image ref"
                    )
                if image_ref and image_id == image_ref:
                    entry_failures.append("image_id must not be identical to image_ref")
                if "/" in compact_image_id or ":" in compact_image_id:
                    entry_failures.append("image_id must not contain registry/tag separators after sha256 prefix")
        for field_name, field_value in {
            "path": path,
            "artifact_path": artifact_path,
            "image_ref": image_ref,
            "image_id": image_id,
        }.items():
            for fragment in excluded:
                if fragment and field_value.startswith(fragment):
                    entry_failures.append(f"{field_name} is inside excluded resource zone: {fragment}")

        path_exists = False
        path_executable = False
        artifact_exists = False
        if path:
            p = Path(path)
            path_exists = p.exists()
            path_executable = p.exists() and p.is_file() and p.stat().st_mode & 0o111 != 0
            if kind == "python_env" and not path_executable:
                entry_failures.append(f"python_env path is not executable: {path}")
        else:
            p = None
        if artifact_path:
            artifact = Path(artifact_path)
            artifact_exists = artifact.exists()
            if kind == "container" and not artifact_exists:
                entry_failures.append(f"container artifact_path does not exist: {artifact_path}")
            if kind == "container" and artifact_exists and not artifact.is_file():
                entry_failures.append(f"container artifact_path must be a file: {artifact_path}")
            if kind == "container" and artifact_path and not artifact_path.endswith(CONTAINER_ARTIFACT_SUFFIXES):
                entry_failures.append(
                    "container artifact_path must end with one of "
                    f"{CONTAINER_ARTIFACT_SUFFIXES}, observed {artifact_path}"
                )

        source_path = str(entry.get("official_source", ""))
        source_exists = bool(source_path and Path(source_path).exists())
        if source_path and not source_exists:
            entry_warnings.append(f"official_source does not exist: {source_path}")

        failures.extend(f"{name}: {msg}" for msg in entry_failures)
        warnings.extend(f"{name}: {msg}" for msg in entry_warnings)
        target_results[name] = {
            "kind": kind,
            "status": status,
            "path": path,
            "artifact_path": artifact_path,
            "image_ref": image_ref,
            "image_id": image_id,
            "container_python": container_python if kind == "container" else "",
            "container_runtime": container_runtime,
            "resolution_path": resolution_path,
            "path_exists": path_exists,
            "path_executable": bool(path_executable),
            "artifact_exists": artifact_exists,
            "official_source": source_path,
            "official_source_exists": source_exists,
            "expected_modules": expected_modules,
            "provenance": provenance,
            "provenance_results": provenance_results,
            "failures": entry_failures,
            "warnings": entry_warnings,
        }

    status = "pass_gate00f_runtime_registry" if not failures else "fail_gate00f_runtime_registry"
    return {
        "classification": "gate00f_runtime_registry_validation_not_training_not_gate_completion",
        "registry": str(registry_path),
        "status": status,
        "required_targets": list(REQUIRED_TARGETS),
        "accepted_status": ACCEPTED_STATUS,
        "allowed_kinds": sorted(ALLOWED_KINDS),
        "allowed_container_runtimes": sorted(ALLOWED_CONTAINER_RUNTIMES),
        "allowed_resolution_paths": sorted(ALLOWED_RESOLUTION_PATHS),
        "target_results": target_results,
        "failures": failures,
        "warnings": warnings,
        "gate_effect": "registry_pass_required_before_runtime_preflight_but_not_sufficient_for_gate00f",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = validate(args.registry)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["status"].startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
