#!/usr/bin/env python3
"""Render one Carry45 reference-versus-actual video for every factorial cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from render_frozen_trace_behavior import (
    ROOT,
    first_episode,
    load_npz,
    load_reference,
    render_pair,
)


FACTORIALS = {
    "asset_motion": {
        "protocol": "sugar_carry_skill_asset_motion_factorial_v1",
        "cells": (
            ("carry_motion_small_asset", "CARRY CTX / SMALL ASSET"),
            ("carry_motion_big_asset", "CARRY CTX / BIG ASSET"),
            ("kick_motion_small_asset", "KICK CTX / SMALL ASSET"),
            ("kick_motion_big_asset", "KICK CTX / BIG ASSET"),
        ),
    },
    "geometry_mass": {
        "protocol": "sugar_carry_skill_geometry_mass_factorial_v1",
        "cells": (
            ("small_geometry_small_mass", "SMALL GEOM / SMALL MASS"),
            ("small_geometry_big_mass", "SMALL GEOM / BIG MASS"),
            ("big_geometry_small_mass", "BIG GEOM / SMALL MASS"),
            ("big_geometry_big_mass", "BIG GEOM / BIG MASS"),
        ),
    },
    "context_goal": {
        "protocol": "sugar_carry_skill_context_goal_factorial_v1",
        "cells": (
            ("carry_context_carry_goal", "CARRY CTX / CARRY GOAL"),
            ("carry_context_kick_goal", "CARRY CTX / KICK GOAL"),
            ("kick_context_carry_goal", "KICK CTX / CARRY GOAL"),
            ("kick_context_kick_goal", "KICK CTX / KICK GOAL"),
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial", choices=tuple(FACTORIALS), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-env", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    if experiments not in input_root.parents or experiments not in output.parents:
        raise ValueError("factorial traces and videos must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    if args.source_env != 0:
        raise ValueError("factorial evidence uses fixed profile 0")
    specification = FACTORIALS[args.factorial]
    summary = json.loads(
        (input_root / "FACTORIAL_RESULT.json").read_text(encoding="utf-8")
    )
    if summary.get("protocol") != specification["protocol"] or not summary.get("passed"):
        raise RuntimeError("factorial structural proof is not admitted")

    output.mkdir(parents=True, exist_ok=False)
    reference = load_reference(ROOT / "SUGAR/data/CarryBox/data_045")
    videos = []
    for index, (cell, short_label) in enumerate(specification["cells"], start=1):
        directory = input_root / cell
        result = json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
        trace = load_npz(directory / "TRACE.npz")
        aggregate = result["aggregate"]
        admitted = bool(summary["admission_pattern"][cell])
        status = "PASS" if admitted else "REJECT"
        actual_label = f"{short_label} | {status}"
        record = render_pair(
            reference,
            first_episode(trace, args.source_env),
            trace["robot_body_names"],
            "INPUT DEMO: CARRYBOX MOTION 45",
            output / f"{index:02d}_{cell}.mp4",
            actual_label,
        )
        record.update(
            {
                "cell": cell,
                "behaviorally_admitted": admitted,
                "carry_success_count": int(aggregate["carry_success_count"]),
                "physical_fall_count": int(aggregate["physical_fall_count"]),
                "maximum_abs_raw_action": float(
                    result["maximum_abs_raw_student_action"]
                ),
            }
        )
        videos.append(record)
    checks = {
        "factorial_structural_proof_passed": bool(summary["passed"]),
        "all_four_cells_rendered": len(videos) == 4,
        "all_videos_h264_yuv420p": all(v["decode"]["passed"] for v in videos),
        "all_reference_and_actual_sequences_fully_displayed": all(
            v["reference_fully_displayed"] and v["actual_fully_displayed"]
            for v in videos
        ),
    }
    proof = {
        "protocol": "sugar_skill_transition_factorial_exact_trace_video_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "factorial": args.factorial,
        "factorial_result": str(input_root / "FACTORIAL_RESULT.json"),
        "source_env": args.source_env,
        "rendering_semantics": (
            "Left is the official Carry45 reference. Right is exact recorded PhysX "
            "body centers and object pose for one frozen profile; no physics replay. "
            "PASS/REJECT is the aggregate 20-profile behavior decision."
        ),
        "videos": videos,
    }
    (output / "RENDER_PROOF.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(proof, indent=2, sort_keys=True))
    if not proof["passed"]:
        raise RuntimeError("factorial video proof failed")


if __name__ == "__main__":
    main()
