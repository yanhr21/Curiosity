#!/usr/bin/env python3
"""Check ANYmal payload-mass locomotion diagnostic summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check an ANYmal payload-carry diagnostic summary.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--min-payload-mass", type=float, default=None)
    parser.add_argument("--max-payload-mass", type=float, default=None)
    parser.add_argument("--min-travel-xy", type=float, default=None)
    parser.add_argument("--min-base-z", type=float, default=None)
    parser.add_argument("--max-tilt-xy", type=float, default=None)
    parser.add_argument("--max-fall-events", type=int, default=0)
    parser.add_argument("--max-done-events", type=int, default=0)
    parser.add_argument("--require-diagnostic-claim", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    failures: list[str] = []

    if int(summary.get("completed_steps") or 0) < int(args.min_steps):
        failures.append(f"completed_steps {summary.get('completed_steps')} < {args.min_steps}")
    if args.require_diagnostic_claim:
        claim = summary.get("claim_level")
        expected = "payload_mass_locomotion_diagnostic_not_grasp_or_contact_carry_success"
        if claim != expected:
            failures.append(f"claim_level {claim!r} != {expected!r}")
    if args.min_payload_mass is not None and float(summary.get("payload_mass_kg") or -999.0) < args.min_payload_mass:
        failures.append(f"payload_mass_kg {summary.get('payload_mass_kg')} < {args.min_payload_mass}")
    if args.max_payload_mass is not None and float(summary.get("payload_mass_kg") or 999.0) > args.max_payload_mass:
        failures.append(f"payload_mass_kg {summary.get('payload_mass_kg')} > {args.max_payload_mass}")
    if args.min_travel_xy is not None and float(summary.get("max_travel_xy_m") or 0.0) < args.min_travel_xy:
        failures.append(f"max_travel_xy_m {summary.get('max_travel_xy_m')} < {args.min_travel_xy}")
    if args.min_base_z is not None:
        value = summary.get("min_base_z_m")
        if value is None or float(value) < args.min_base_z:
            failures.append(f"min_base_z_m {value} < {args.min_base_z}")
    if args.max_tilt_xy is not None and float(summary.get("max_tilt_xy") or 999.0) > args.max_tilt_xy:
        failures.append(f"max_tilt_xy {summary.get('max_tilt_xy')} > {args.max_tilt_xy}")
    if int(summary.get("fall_events") or 0) > args.max_fall_events:
        failures.append(f"fall_events {summary.get('fall_events')} > {args.max_fall_events}")
    if int(summary.get("done_events") or 0) > args.max_done_events:
        failures.append(f"done_events {summary.get('done_events')} > {args.max_done_events}")

    report = {
        "summary_path": str(args.summary),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "completed_steps": summary.get("completed_steps"),
        "payload_mass_kg": summary.get("payload_mass_kg"),
        "max_travel_xy_m": summary.get("max_travel_xy_m"),
        "min_base_z_m": summary.get("min_base_z_m"),
        "max_tilt_xy": summary.get("max_tilt_xy"),
        "fall_events": summary.get("fall_events"),
        "done_events": summary.get("done_events"),
        "claim_level": summary.get("claim_level"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
