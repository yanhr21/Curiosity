#!/usr/bin/env python3
"""Collect the serial paired Plan-15 mass sweep, then analyze live leakage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COLLECTOR = ROOT / "scripts/sugar/native_tactile/preflight_online_patch_mass_jump.py"
ANALYZER = ROOT / "scripts/sugar/native_tactile/analyze_online_mass_leakage.py"
SCALE_FITTER = ROOT / "scripts/sugar/native_tactile/fit_online_patch_channel_scales.py"
SLIP_EVALUATOR = ROOT / "scripts/sugar/native_tactile/evaluate_online_patch_slip.py"
FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)
DEFAULT_SEEDS = (150814, 150815, 150816)


def factor_name(factor: float) -> str:
    return f"factor_{factor:g}x".replace(".", "p")


def run(command: list[str]) -> None:
    print("[Plan15]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--motion-id", type=int, default=45)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--jump-delay-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS
    if args.max_steps < 1 or args.jump_delay_frames < 0:
        raise ValueError("max steps must be positive and jump delay nonnegative")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    traces: dict[float, list[Path]] = {factor: [] for factor in FACTORS}

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir()
        nominal_root = seed_root / factor_name(1.0)
        common = [
            sys.executable,
            str(COLLECTOR),
            "--motion-id",
            str(args.motion_id),
            "--seed",
            str(seed),
            "--max-steps",
            str(args.max_steps),
            "--delay-frames",
            str(args.jump_delay_frames),
            str(args.jump_delay_frames),
            "--headless",
            "--device",
            args.device,
        ]
        run(
            common
            + [
                "--mass-factor",
                "1.0",
                "--output-root",
                str(nominal_root),
            ]
        )
        nominal_trace = nominal_root / "online_mass_jump_trace.npz"
        traces[1.0].append(nominal_trace)

        for factor in FACTORS[1:]:
            factor_root = seed_root / factor_name(factor)
            run(
                common
                + [
                    "--mass-factor",
                    str(factor),
                    "--action-trace",
                    str(nominal_trace),
                    "--output-root",
                    str(factor_root),
                ]
            )
            traces[factor].append(factor_root / "online_mass_jump_trace.npz")

    leakage_output = output_root / "leakage_audit.json"
    analyzer_command = [sys.executable, str(ANALYZER)]
    for factor in FACTORS:
        for trace in traces[factor]:
            analyzer_command.extend(("--trace", f"{factor}={trace}"))
    analyzer_command.extend(("--output", str(leakage_output)))
    run(analyzer_command)

    slip_output = output_root / "slip_evaluation.json"
    slip_command = [sys.executable, str(SLIP_EVALUATOR)]
    for factor in FACTORS:
        for trace in traces[factor]:
            slip_command.extend(("--trace", str(trace)))
    slip_command.extend(("--output", str(slip_output)))
    run(slip_command)

    scale_output = output_root / "patch_channel_scales.json"
    scale_command = [sys.executable, str(SCALE_FITTER)]
    for factor in FACTORS:
        for trace in traces[factor]:
            scale_command.extend(("--trace", str(trace)))
    scale_command.extend(("--output", str(scale_output)))
    run(scale_command)

    manifest = {
        "schema": "plan15_paired_online_mass_leakage_sweep_v1",
        "semantics": "online IsaacLab collection; fixed nominal action replay",
        "motion_id": int(args.motion_id),
        "seeds": list(seeds),
        "mass_factors": list(FACTORS),
        "fixed_jump_delay_frames": int(args.jump_delay_frames),
        "traces": {
            str(factor): [str(path) for path in traces[factor]]
            for factor in FACTORS
        },
        "leakage_audit": str(leakage_output),
        "slip_evaluation": str(slip_output),
        "patch_channel_scales": str(scale_output),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
