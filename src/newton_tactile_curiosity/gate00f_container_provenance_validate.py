#!/usr/bin/env python3
"""Validate a Gate 00F container provenance packet.

This checks metadata and local evidence paths only. It does not pull images,
build images, run containers, import simulator modules, install dependencies,
or start official sanity tasks.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ALLOWED_TARGETS = {"univtac", "tacauchy", "isaaclab_tacsl"}
ALLOWED_CONTAINER_RUNTIMES = {"docker", "singularity", "apptainer", "enroot", "sif", "sqsh", "tar"}
REQUIRED_SHARED_FIELDS = (
    "target",
    "container_runtime",
    "official_source",
    "official_source_commit",
    "expected_modules",
    "provenance",
)
CONTAINER_ARTIFACT_SUFFIXES = (".sif", ".sqsh", ".tar", ".tar.gz", ".img")
IMAGE_ID_RE = re.compile(r"^(sha256:)?[0-9a-fA-F]{12,}$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def has_placeholder(value: str) -> bool:
    return "<" in value or ">" in value


def validate(packet_path: Path, root: Path) -> dict[str, Any]:
    packet = load_json(packet_path)
    failures: list[str] = []
    warnings: list[str] = []

    for field in REQUIRED_SHARED_FIELDS:
        if field not in packet:
            failures.append(f"missing required field: {field}")

    target = str(packet.get("target", ""))
    container_runtime = str(packet.get("container_runtime", ""))
    official_source = str(packet.get("official_source", ""))
    official_source_commit = str(packet.get("official_source_commit", ""))
    image_ref = str(packet.get("image_ref", ""))
    image_id = str(packet.get("image_id", ""))
    artifact_path = str(packet.get("artifact_path", ""))
    build_context = str(packet.get("build_context", ""))
    dockerfile = str(packet.get("dockerfile", ""))
    compose_file = str(packet.get("compose_file", ""))
    expected_modules = [str(x) for x in as_list(packet.get("expected_modules"))]
    provenance = [str(x) for x in as_list(packet.get("provenance"))]
    evidence_paths = [str(x) for x in as_list(packet.get("evidence_paths"))]
    declared_non_claims = [str(x) for x in as_list(packet.get("declared_non_claims"))]
    excluded_fragments = [str(x) for x in as_list(packet.get("excluded_path_fragments"))]

    if target not in ALLOWED_TARGETS:
        failures.append(f"target must be one of {sorted(ALLOWED_TARGETS)}, observed {target}")
    if container_runtime not in ALLOWED_CONTAINER_RUNTIMES:
        failures.append(
            f"container_runtime must be one of {sorted(ALLOWED_CONTAINER_RUNTIMES)}, observed {container_runtime}"
        )
    if not expected_modules:
        failures.append("expected_modules must be nonempty")
    if not provenance:
        failures.append("provenance must be nonempty")
    if not (image_id or artifact_path):
        failures.append("packet must include local image_id or existing artifact_path; image_ref alone is not enough")
    if image_id:
        compact_image_id = image_id.removeprefix("sha256:")
        if not IMAGE_ID_RE.match(image_id):
            failures.append(
                "image_id must look like a local immutable image digest or ID "
                "(sha256:<hex> or >=12 hex chars), not a tag or image ref"
            )
        if image_ref and image_id == image_ref:
            failures.append("image_id must not be identical to image_ref")
        if "/" in compact_image_id or ":" in compact_image_id:
            failures.append("image_id must not contain registry/tag separators after sha256 prefix")

    checked_paths: dict[str, Any] = {}
    path_fields = {
        "official_source": official_source,
        "artifact_path": artifact_path,
        "build_context": build_context,
        "dockerfile": dockerfile,
        "compose_file": compose_file,
    }
    for field, value in path_fields.items():
        if not value:
            checked_paths[field] = {"path": "", "exists": False}
            continue
        if has_placeholder(value):
            failures.append(f"{field} contains placeholder brackets: {value}")
        for fragment in excluded_fragments:
            if fragment and value.startswith(fragment):
                failures.append(f"{field} points into excluded resource zone: {fragment}")
        p = resolve(root, value)
        exists = p.exists()
        checked_paths[field] = {"path": value, "resolved_path": str(p), "exists": exists}
        if field in {"official_source", "artifact_path"} and not exists:
            failures.append(f"{field} does not exist: {value}")
        if field == "artifact_path" and exists and not p.is_file():
            failures.append(f"artifact_path must be a file: {value}")
        if field == "artifact_path" and value and not value.endswith(CONTAINER_ARTIFACT_SUFFIXES):
            failures.append(
                "artifact_path must end with one of "
                f"{CONTAINER_ARTIFACT_SUFFIXES}, observed {value}"
            )
        if field in {"build_context", "dockerfile", "compose_file"} and value and not exists:
            failures.append(f"{field} does not exist: {value}")

    if official_source:
        observed_commit = git_head(resolve(root, official_source))
        checked_paths["official_source"]["observed_commit"] = observed_commit
        if not observed_commit:
            failures.append(f"official_source is not a readable git repo: {official_source}")
        elif official_source_commit and observed_commit != official_source_commit:
            failures.append(
                "official_source_commit mismatch: "
                f"declared {official_source_commit}, observed {observed_commit}"
            )

    provenance_results: list[dict[str, Any]] = []
    for item in provenance + evidence_paths:
        if has_placeholder(item):
            failures.append(f"evidence/provenance contains placeholder brackets: {item}")
        p = resolve(root, item)
        exists = p.exists()
        provenance_results.append({"path": item, "resolved_path": str(p), "exists": exists})
        if not exists:
            failures.append(f"evidence/provenance path does not exist: {item}")

    if "not_gate_completion" not in declared_non_claims:
        warnings.append("declared_non_claims should include not_gate_completion")
    if "not_runtime_preflight" not in declared_non_claims:
        warnings.append("declared_non_claims should include not_runtime_preflight")

    status = "pass_gate00f_container_provenance" if not failures else "fail_gate00f_container_provenance"
    return {
        "classification": "gate00f_container_provenance_validation_not_runtime_not_gate_completion",
        "packet": str(packet_path),
        "status": status,
        "target": target,
        "container_runtime": container_runtime,
        "image_ref": image_ref,
        "image_id_present": bool(image_id),
        "artifact_path": artifact_path,
        "expected_modules": expected_modules,
        "checked_paths": checked_paths,
        "provenance_results": provenance_results,
        "failures": failures,
        "warnings": warnings,
        "gate_effect": "container_provenance_pass_is_required_before_registry_registration_but_not_sufficient_for_gate00f",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = validate(args.packet, args.root.resolve())
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["status"].startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
