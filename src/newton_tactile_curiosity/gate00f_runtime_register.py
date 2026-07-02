#!/usr/bin/env python3
"""Register a Gate 00F runtime candidate into a copied registry JSON.

This is metadata editing only. It does not pull images, build images, start
containers, import Isaac/TacSL/TacEx modules, run simulators, or install
dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TARGETS = ("univtac", "tacauchy", "isaaclab_tacsl")
KINDS = ("python_env", "container")
ACCEPTED_STATUS = "dependency_complete_registered"
ALLOWED_CONTAINER_RUNTIMES = ("docker", "singularity", "apptainer", "enroot", "sif", "sqsh", "tar")
ALLOWED_RESOLUTION_PATHS = (
    "reuse_existing_dependency_complete_env",
    "reuse_existing_prebuilt_container",
    "compliant_external_env_prep_then_register",
)
CONTAINER_ARTIFACT_SUFFIXES = (".sif", ".sqsh", ".tar", ".tar.gz", ".img")
IMAGE_ID_RE = re.compile(r"^(sha256:)?[0-9a-fA-F]{12,}$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def reject_placeholder(value: str, field: str) -> None:
    if "<" in value or ">" in value:
        raise ValueError(f"{field} contains placeholder brackets: {value}")


def reject_excluded_paths(registry: dict[str, Any], values: dict[str, str]) -> None:
    excluded = [str(x) for x in as_list(registry.get("excluded_path_fragments"))]
    for field, value in values.items():
        if not value:
            continue
        reject_placeholder(value, field)
        for fragment in excluded:
            if fragment and value.startswith(fragment):
                raise ValueError(f"{field} points into excluded resource zone: {fragment}")


def validate_container_local_reference(artifact_path: str, image_id: str, image_ref: str) -> None:
    if image_id:
        compact_image_id = image_id.removeprefix("sha256:")
        if not IMAGE_ID_RE.match(image_id):
            raise ValueError(
                "image_id must look like a local immutable image digest or ID "
                "(sha256:<hex> or >=12 hex chars), not a tag or image ref"
            )
        if image_ref and image_id == image_ref:
            raise ValueError("image_id must not be identical to image_ref")
        if "/" in compact_image_id or ":" in compact_image_id:
            raise ValueError("image_id must not contain registry/tag separators after sha256 prefix")
    if artifact_path:
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise ValueError(f"container artifact_path does not exist: {artifact_path}")
        if not artifact.is_file():
            raise ValueError(f"container artifact_path must be a file: {artifact_path}")
        if not artifact_path.endswith(CONTAINER_ARTIFACT_SUFFIXES):
            raise ValueError(
                "container artifact_path must end with one of "
                f"{CONTAINER_ARTIFACT_SUFFIXES}, observed {artifact_path}"
            )


def register(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_json(args.registry)
    targets = registry.setdefault("targets", {})
    if not isinstance(targets, dict):
        raise ValueError("registry.targets must be an object")

    expected_modules = args.expected_module or []
    provenance = args.provenance or []
    if not expected_modules:
        raise ValueError("--expected-module is required at least once")
    if not provenance:
        raise ValueError("--provenance is required at least once")
    root = args.registry.resolve().parents[4] if len(args.registry.resolve().parents) > 4 else Path.cwd()
    for provenance_item in provenance:
        reject_placeholder(provenance_item, "provenance")
        provenance_path = Path(provenance_item)
        if not provenance_path.is_absolute():
            provenance_path = root / provenance_path
        if not provenance_path.exists():
            raise ValueError(f"provenance path does not exist: {provenance_item}")

    path = str(args.path or "")
    artifact_path = str(args.artifact_path or "")
    image_id = str(args.image_id or "")
    image_ref = str(args.image_ref or "")
    official_source = str(args.official_source or "")
    container_python = str(args.container_python or "python3")

    reject_excluded_paths(
        registry,
        {
            "path": path,
            "artifact_path": artifact_path,
            "image_id": image_id,
            "image_ref": image_ref,
            "official_source": official_source,
        },
    )

    if args.kind == "python_env":
        if not path:
            raise ValueError("python_env registration requires --path")
        python_path = Path(path)
        if not python_path.exists() or not python_path.is_file() or python_path.stat().st_mode & 0o111 == 0:
            raise ValueError(f"python_env path must be an executable file: {path}")
        if artifact_path or image_id:
            raise ValueError("python_env registration must not include --artifact-path or --image-id")
    if args.kind == "container":
        if not args.container_runtime:
            raise ValueError("container registration requires --container-runtime")
        reject_placeholder(container_python, "container_python")
        if not container_python:
            raise ValueError("container registration requires a nonempty --container-python")
        if not (artifact_path or image_id):
            raise ValueError("container registration requires --artifact-path or --image-id")
        validate_container_local_reference(artifact_path=artifact_path, image_id=image_id, image_ref=image_ref)
        if not args.container_provenance_summary:
            raise ValueError("container registration requires --container-provenance-summary")
        container_provenance_summary = Path(args.container_provenance_summary)
        if not container_provenance_summary.exists():
            raise ValueError(f"container_provenance_summary does not exist: {container_provenance_summary}")
        provenance_summary = load_json(container_provenance_summary)
        if provenance_summary.get("status") != "pass_gate00f_container_provenance":
            raise ValueError(
                "container_provenance_summary status must be "
                f"pass_gate00f_container_provenance, observed {provenance_summary.get('status')}"
            )
        if provenance_summary.get("target") != args.target:
            raise ValueError(
                "container_provenance_summary target mismatch: "
                f"expected {args.target}, observed {provenance_summary.get('target')}"
            )
        provenance_summary_str = str(container_provenance_summary)
        if provenance_summary_str not in provenance:
            provenance.append(provenance_summary_str)
        if artifact_path and not Path(artifact_path).exists():
            raise ValueError(f"container artifact_path does not exist: {artifact_path}")
        if path:
            raise ValueError("container registration must not include --path")

    official_source_path = Path(official_source)
    if not official_source_path.exists():
        raise ValueError(f"official_source must exist: {official_source}")

    entry: dict[str, Any] = {
        "kind": args.kind,
        "status": ACCEPTED_STATUS,
        "path": path,
        "resolution_path": args.resolution_path,
        "official_source": official_source,
        "expected_modules": expected_modules,
        "provenance": provenance,
        "notes": args.notes,
    }
    if args.kind == "container":
        entry.update(
            {
                "container_runtime": args.container_runtime,
                "artifact_path": artifact_path,
                "image_id": image_id,
                "image_ref": image_ref,
                "container_python": container_python,
            }
        )

    targets[args.target] = entry
    registry["classification"] = "runtime_registry_with_registered_candidate_not_training_not_gate_completion"
    registry["gate_effect"] = "registry_candidate_must_still_pass_validator_and_runtime_preflight"
    return registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--path", default="")
    parser.add_argument("--artifact-path", default="")
    parser.add_argument("--image-id", default="")
    parser.add_argument("--image-ref", default="")
    parser.add_argument("--container-runtime", choices=ALLOWED_CONTAINER_RUNTIMES, default="")
    parser.add_argument("--container-python", default="python3")
    parser.add_argument("--container-provenance-summary", default="")
    parser.add_argument("--resolution-path", choices=ALLOWED_RESOLUTION_PATHS, required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--expected-module", action="append", default=[])
    parser.add_argument("--provenance", action="append", default=[])
    parser.add_argument("--notes", default="registered runtime candidate; validator and runtime preflight still required")
    args = parser.parse_args()

    updated = register(args)
    write_json(args.output, updated)
    print(json.dumps({"status": "wrote_runtime_registry_candidate", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
