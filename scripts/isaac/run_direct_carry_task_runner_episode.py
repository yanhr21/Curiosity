#!/usr/bin/env python3
"""Run one direct Isaac carry task episode through DirectCarryTaskRunner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from direct_carry_task_runner import DirectCarryAction, DirectCarryReset, DirectCarryTaskRunner
from direct_carry_task_shell_backend import DirectCarryShellBackend


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run direct carry task runner on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one direct Isaac carry task-runner episode.")
    parser.add_argument("--root-dir", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--box-seed", type=int, required=True)
    parser.add_argument("--carry-posture", choices=("front_mid", "low_front", "chest_high"), default="front_mid")
    parser.add_argument("--target-x", type=float, default=0.64)
    parser.add_argument("--steps", type=int, default=3580)
    parser.add_argument(
        "--support-mode",
        choices=("alternating_anchor_feet", "alternating_placement_feet"),
        default="alternating_anchor_feet",
    )
    parser.add_argument("--gait-speed-scale", type=float, default=1.0)
    parser.add_argument("--feedback-step-x-gain", type=float, default=0.015)
    parser.add_argument("--feedback-step-x-limit", type=float, default=0.008)
    parser.add_argument("--feedback-step-tilt-gain", type=float, default=0.05)
    parser.add_argument("--feedback-step-tilt-limit", type=float, default=0.005)
    parser.add_argument("--support-foot-double-support-fraction", type=float, default=0.18)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--probe-amplitude-x", type=float, default=0.0)
    parser.add_argument("--probe-amplitude-z", type=float, default=0.0)
    parser.add_argument("--reference-video-id", default=None)
    return parser.parse_args()


def main() -> int:
    _refuse_login_node()
    args = parse_args()
    backend = DirectCarryShellBackend(
        root_dir=args.root_dir,
        stamp=args.stamp,
        steps=args.steps,
        support_mode=args.support_mode,
    )
    runner = DirectCarryTaskRunner(backend)
    reset = DirectCarryReset(
        box_seed=args.box_seed,
        morphology_config="scaffold_support_feet_v1",
        target_distance_x_m=args.target_x,
        reference_video_id=args.reference_video_id,
    )
    action = DirectCarryAction(
        carry_posture=args.carry_posture,
        feedback_step_x_gain=args.feedback_step_x_gain,
        feedback_step_x_limit_m=args.feedback_step_x_limit,
        feedback_step_tilt_gain=args.feedback_step_tilt_gain,
        feedback_step_tilt_limit_m=args.feedback_step_tilt_limit,
        support_foot_double_support_fraction=args.support_foot_double_support_fraction,
        gait_speed_scale=args.gait_speed_scale,
        probe_steps=args.probe_steps,
        probe_amplitude_x_m=args.probe_amplitude_x,
        probe_amplitude_z_m=args.probe_amplitude_z,
    )
    runner.reset(reset)
    summary = runner.run_episode(action)
    row = runner.export_episode_row(
        source_summary=str(backend.summary_path),
        episode_id=f"{args.stamp}:{args.carry_posture}",
        summary=summary,
    )
    output_dir = backend.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "direct_carry_task_runner_episode.jsonl"
    report_path = output_dir / "direct_carry_task_runner_report.json"
    row_path.write_text(json.dumps(row, sort_keys=True) + "\n")
    row_passed = bool(row.get("gates", {}).get("passed"))
    backend_returncode = summary.get("task_runner_backend_returncode")
    backend_capabilities = row.get("backend_capabilities", {})
    report = {
        "status": "pass" if row_passed and backend_returncode == 0 else summary.get("status", "unknown"),
        "stamp": args.stamp,
        "summary_path": str(backend.summary_path),
        "episode_row_path": str(row_path),
        "backend_log": summary.get("task_runner_backend_log", str(backend.log_path)),
        "backend_id": backend_capabilities.get("backend_id"),
        "backend_family": backend_capabilities.get("backend_family"),
        "scaffold_backend": backend_capabilities.get("scaffold_backend"),
        "trainable_policy_backend": backend_capabilities.get("trainable_policy_backend"),
        "reward_proxy": row.get("metrics", {}).get("reward_proxy"),
        "limitation_label": row.get("limitation_label"),
        "success_claim": row.get("success_claim"),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if summary.get("task_runner_backend_returncode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
