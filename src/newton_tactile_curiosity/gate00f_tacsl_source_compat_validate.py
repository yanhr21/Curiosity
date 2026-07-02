#!/usr/bin/env python3
"""Static compatibility checks for the Gate 00F IsaacLab TacSL source.

This validates source files and metadata only. It does not import Isaac Lab,
start Isaac Sim, launch containers, run simulation, load models, or install
dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_DATA_FIELDS = [
    "tactile_depth_image",
    "tactile_rgb_image",
    "tactile_points_pos_w",
    "tactile_points_quat_w",
    "penetration_depth",
    "tactile_normal_force",
    "tactile_shear_force",
]
REQUIRED_CLI_FLAGS = [
    "--use_tactile_rgb",
    "--use_tactile_ff",
    "--normal_contact_stiffness",
    "--tangential_stiffness",
    "--friction_coefficient",
    "--contact_object_type",
    "--save_viz",
    "--enable_cameras",
]
REQUIRED_IMPORT_STRINGS = [
    "from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg",
    "from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData",
    "from isaaclab_assets.sensors import GELSIGHT_R15_CFG",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_isaacsim_versions(readme: str) -> list[str]:
    versions = sorted(set(re.findall(r"Isaac Sim\s+([0-9]+(?:\.[0-9]+){1,2})", readme)))
    badge_versions = re.findall(r"IsaacSim-([0-9]+(?:\.[0-9]+){1,2})", readme)
    return sorted(set(versions + badge_versions))


def validate(repo: Path, expected_version: str | None = None, expected_image_ref: str | None = None) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    version_file = repo / "VERSION"
    readme_file = repo / "README.md"
    demo_file = repo / "scripts/demos/sensors/tacsl_sensor.py"
    data_file = repo / "source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor_data.py"
    cfg_file = repo / "source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor_cfg.py"
    init_file = repo / "source/isaaclab_contrib/isaaclab_contrib/sensors/tacsl_sensor/__init__.py"

    required_files = [version_file, readme_file, demo_file, data_file, cfg_file, init_file]
    for path in required_files:
        if not path.exists():
            failures.append(f"missing required file: {path}")

    version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else ""
    if expected_version and version != expected_version:
        failures.append(f"VERSION expected {expected_version}, observed {version}")

    readme = read_text(readme_file) if readme_file.exists() else ""
    isaacsim_versions = parse_isaacsim_versions(readme)
    if not isaacsim_versions:
        warnings.append("no Isaac Sim compatibility versions parsed from README")

    demo = read_text(demo_file) if demo_file.exists() else ""
    data = read_text(data_file) if data_file.exists() else ""
    cfg = read_text(cfg_file) if cfg_file.exists() else ""
    init_text = read_text(init_file) if init_file.exists() else ""

    missing_fields = [field for field in REQUIRED_DATA_FIELDS if field not in data]
    if missing_fields:
        failures.append(f"missing TacSL data fields: {missing_fields}")

    missing_flags = [flag for flag in REQUIRED_CLI_FLAGS if flag not in demo]
    if missing_flags:
        failures.append(f"missing TacSL demo CLI flags: {missing_flags}")

    missing_imports = [text for text in REQUIRED_IMPORT_STRINGS if text not in demo]
    if missing_imports:
        failures.append(f"missing TacSL demo imports: {missing_imports}")

    required_cfg_tokens = [
        "enable_camera_tactile",
        "enable_force_field",
        "normal_contact_stiffness",
        "tangential_stiffness",
        "friction_coefficient",
        "tactile_array_size",
        "GELSIGHT_R15_CFG",
    ]
    missing_cfg_tokens = [token for token in required_cfg_tokens if token not in (cfg + demo)]
    if missing_cfg_tokens:
        failures.append(f"missing TacSL config tokens: {missing_cfg_tokens}")

    init_exports = ["VisuoTactileSensor", "VisuoTactileSensorCfg", "VisuoTactileSensorData"]
    missing_exports = [token for token in init_exports if token not in init_text]
    if missing_exports:
        failures.append(f"missing TacSL __init__ exports: {missing_exports}")

    status = "pass_tacsl_source_compat" if not failures else "fail_tacsl_source_compat"
    return {
        "classification": "tacsl_source_compat_validation_not_runtime_not_gate_completion",
        "repo": str(repo),
        "status": status,
        "expected_version": expected_version or "",
        "observed_version": version,
        "expected_image_ref": expected_image_ref or "",
        "parsed_isaacsim_versions": isaacsim_versions,
        "required_data_fields": REQUIRED_DATA_FIELDS,
        "required_cli_flags": REQUIRED_CLI_FLAGS,
        "required_import_strings": REQUIRED_IMPORT_STRINGS,
        "failures": failures,
        "warnings": warnings,
        "gate_effect": "source_compat_only_does_not_register_runtime_or_clear_gate00f",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("external/IsaacLab_official"))
    parser.add_argument("--expected-version", default=None)
    parser.add_argument("--expected-image-ref", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = validate(args.repo, args.expected_version, args.expected_image_ref)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["status"].startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
