#!/usr/bin/env python3
"""Run the metadata-only Gate 00F runtime intake chain.

The chain is:
1. validate a container provenance packet;
2. register the runtime into a copied candidate registry;
3. validate the copied candidate registry.

It does not pull/build/run containers, import simulator modules, install
dependencies, run simulation, render, train, evaluate, or start Slurm jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gate00f_container_provenance_validate as provenance_validate
import gate00f_runtime_register as runtime_register
import gate00f_runtime_registry_validate as registry_validate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_list(value: Any) -> list[str]:
    return [str(x) for x in value] if isinstance(value, list) else []


def build_register_args(args: argparse.Namespace, packet: dict[str, Any], provenance_summary_path: Path) -> SimpleNamespace:
    provenance = as_list(packet.get("provenance"))
    evidence_paths = as_list(packet.get("evidence_paths"))
    combined_provenance = []
    for item in provenance + evidence_paths + [str(provenance_summary_path)]:
        if item and item not in combined_provenance:
            combined_provenance.append(item)

    return SimpleNamespace(
        registry=args.base_registry,
        output=args.output_registry,
        target=str(packet.get("target", "")),
        kind="container",
        path="",
        artifact_path=str(packet.get("artifact_path", "")),
        image_id=str(packet.get("image_id", "")),
        image_ref=str(packet.get("image_ref", "")),
        container_runtime=str(packet.get("container_runtime", "")),
        container_provenance_summary=str(provenance_summary_path),
        resolution_path=args.resolution_path,
        official_source=str(args.root / str(packet.get("official_source", "")))
        if not Path(str(packet.get("official_source", ""))).is_absolute()
        else str(packet.get("official_source", "")),
        expected_module=as_list(packet.get("expected_modules")),
        provenance=combined_provenance,
        notes=args.notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--provenance-summary", type=Path, required=True)
    parser.add_argument("--registry-validation-summary", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--resolution-path",
        choices=runtime_register.ALLOWED_RESOLUTION_PATHS,
        default="reuse_existing_prebuilt_container",
    )
    parser.add_argument(
        "--notes",
        default="registered by gate00f_runtime_intake_chain; runtime preflight and Gate00F bundle still required",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    packet = load_json(args.packet)

    provenance_summary = provenance_validate.validate(args.packet, root)
    write_json(args.provenance_summary, provenance_summary)
    if provenance_summary["status"] != "pass_gate00f_container_provenance":
        summary = {
            "classification": "gate00f_runtime_intake_chain_not_runtime_not_gate_completion",
            "status": "fail_container_provenance",
            "packet": str(args.packet),
            "provenance_summary": str(args.provenance_summary),
            "output_registry": str(args.output_registry),
            "registry_validation_summary": str(args.registry_validation_summary),
            "failures": provenance_summary.get("failures", []),
            "gate_effect": "intake_stopped_before_registry_registration",
        }
        write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    register_args = build_register_args(args, packet, args.provenance_summary)
    try:
        candidate_registry = runtime_register.register(register_args)
    except Exception as exc:
        summary = {
            "classification": "gate00f_runtime_intake_chain_not_runtime_not_gate_completion",
            "status": "fail_runtime_registration",
            "packet": str(args.packet),
            "provenance_summary": str(args.provenance_summary),
            "output_registry": str(args.output_registry),
            "registry_validation_summary": str(args.registry_validation_summary),
            "failures": [str(exc)],
            "gate_effect": "intake_stopped_before_registry_validation",
        }
        write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    runtime_register.write_json(args.output_registry, candidate_registry)
    registry_summary = registry_validate.validate(args.output_registry)
    write_json(args.registry_validation_summary, registry_summary)

    status = (
        "pass_runtime_intake_chain_registry_ready"
        if registry_summary["status"] == "pass_gate00f_runtime_registry"
        else "fail_runtime_registry_validation"
    )
    summary = {
        "classification": "gate00f_runtime_intake_chain_not_runtime_not_gate_completion",
        "status": status,
        "packet": str(args.packet),
        "provenance_summary": str(args.provenance_summary),
        "output_registry": str(args.output_registry),
        "registry_validation_summary": str(args.registry_validation_summary),
        "registry_status": registry_summary["status"],
        "registry_failures": registry_summary.get("failures", []),
        "gate_effect": "registry_ready_is_required_before_runtime_preflight_but_not_sufficient_for_gate00f",
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
