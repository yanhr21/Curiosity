#!/usr/bin/env python3
"""Direct Core API G1 + box Isaac scene diagnostic.

This script deliberately avoids IsaacLab InteractiveScene and external policy
servers.  It is a scene/bootstrap diagnostic only: load the local G1 USD, spawn
a physical box and target marker, optionally fixed-joint the box to a robot body,
and step the scene while recording basic stability metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        raise RuntimeError("Cannot synchronously run Replicator capture while an asyncio loop is already running.")
    return loop.run_until_complete(coro)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Core API G1 + box scene diagnostic.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--g1-usd", type=Path, default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd"))
    parser.add_argument("--box-mass", type=float, default=2.0)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.45, 0.30, 0.30), metavar=("X", "Y", "Z"))
    parser.add_argument("--box-position", type=float, nargs=3, default=(0.55, 0.0, 0.95), metavar=("X", "Y", "Z"))
    parser.add_argument("--target-xy", type=float, nargs=2, default=(1.2, 0.0), metavar=("X", "Y"))
    parser.add_argument("--box-support-mode", choices=("none", "table"), default="none")
    parser.add_argument("--box-support-size", type=float, nargs=3, default=(0.75, 0.55, 0.65), metavar=("X", "Y", "Z"))
    parser.add_argument("--box-support-top-clearance", type=float, default=0.0)
    parser.add_argument("--box-support-release-step", type=int, default=-1)
    parser.add_argument("--disable-carry-box-spawn", action="store_true")
    parser.add_argument("--disable-box-collision", action="store_true")
    parser.add_argument("--g1-root-position", type=float, nargs=3, default=(0.0, 0.0, 0.78), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--g1-root-orientation-wxyz",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
    )
    parser.add_argument("--disable-setup-root-pose", action="store_true")
    parser.add_argument("--disable-setup-joint-state-write", action="store_true")
    parser.add_argument("--disable-usd-pelvis-xform", action="store_true")
    parser.add_argument("--disable-stand-joint-targets", action="store_true")
    parser.add_argument("--apply-arena-stand-gains", action="store_true")
    parser.add_argument("--stand-drive-preset", choices=("arena", "isaaclab29dof"), default="arena")
    parser.add_argument("--stand-gain-scale", type=float, default=1.0)
    parser.add_argument("--stand-force-scale", type=float, default=1.0)
    parser.add_argument("--stand-hip-pitch", type=float, default=None)
    parser.add_argument("--stand-knee", type=float, default=None)
    parser.add_argument("--stand-ankle-pitch", type=float, default=None)
    parser.add_argument("--stand-hip-roll", type=float, default=None)
    parser.add_argument("--stand-ankle-roll", type=float, default=None)
    parser.add_argument("--gait-mode", choices=("stand", "open_loop_march", "staged_march", "targeted_creep", "agile_policy"), default="stand")
    parser.add_argument("--gait-amplitude", type=float, default=0.0)
    parser.add_argument("--gait-frequency-hz", type=float, default=0.7)
    parser.add_argument("--gait-start-step", type=int, default=0)
    parser.add_argument("--gait-stop-step", type=int, default=-1)
    parser.add_argument("--creep-hip-pitch-offset", type=float, default=0.12)
    parser.add_argument("--creep-knee-offset", type=float, default=0.04)
    parser.add_argument("--creep-ankle-pitch-offset", type=float, default=-0.06)
    parser.add_argument("--creep-waist-pitch-offset", type=float, default=0.04)
    parser.add_argument("--creep-stance-push-scale", type=float, default=0.18)
    parser.add_argument("--creep-lift-scale", type=float, default=0.50)
    parser.add_argument("--creep-ankle-lift-scale", type=float, default=-0.30)
    parser.add_argument("--creep-decel-box-travel-start", type=float, default=-1.0)
    parser.add_argument("--creep-decel-box-travel-end", type=float, default=-1.0)
    parser.add_argument("--creep-decel-robot-travel-start", type=float, default=-1.0)
    parser.add_argument("--creep-decel-robot-travel-end", type=float, default=-1.0)
    parser.add_argument("--creep-min-amplitude-scale", type=float, default=0.0)
    parser.add_argument("--creep-min-push-scale", type=float, default=0.0)
    parser.add_argument("--creep-min-bias-scale", type=float, default=0.0)
    parser.add_argument("--creep-pitch-brake-threshold", type=float, default=999.0)
    parser.add_argument("--creep-pitch-brake-rate-threshold", type=float, default=999.0)
    parser.add_argument("--creep-pitch-brake-amplitude-scale", type=float, default=0.0)
    parser.add_argument("--creep-pitch-brake-push-scale", type=float, default=0.0)
    parser.add_argument("--creep-pitch-brake-bias-scale", type=float, default=0.0)
    parser.add_argument("--creep-pitch-brake-latch", action="store_true")
    parser.add_argument("--creep-pitch-brake-positive-only", action="store_true")
    parser.add_argument("--creep-reverse-brake-box-travel", type=float, default=-1.0)
    parser.add_argument("--creep-reverse-brake-robot-travel", type=float, default=-1.0)
    parser.add_argument("--creep-reverse-brake-pitch-threshold", type=float, default=999.0)
    parser.add_argument("--creep-reverse-brake-positive-pitch-only", action="store_true")
    parser.add_argument("--creep-reverse-brake-duration-steps", type=int, default=-1)
    parser.add_argument("--creep-reverse-brake-amplitude-scale", type=float, default=1.0)
    parser.add_argument("--creep-reverse-brake-stance-push-scale", type=float, default=-0.20)
    parser.add_argument("--creep-reverse-brake-lift-scale", type=float, default=0.35)
    parser.add_argument("--creep-reverse-brake-hip-pitch-offset", type=float, default=0.02)
    parser.add_argument("--creep-reverse-brake-knee-offset", type=float, default=0.02)
    parser.add_argument("--creep-reverse-brake-ankle-pitch-offset", type=float, default=0.02)
    parser.add_argument("--creep-reverse-brake-waist-pitch-offset", type=float, default=-0.04)
    parser.add_argument("--gait-ramp-down-start-step", type=int, default=-1)
    parser.add_argument("--gait-ramp-down-end-step", type=int, default=-1)
    parser.add_argument("--gait-min-amplitude-scale", type=float, default=0.0)
    parser.add_argument("--recovery-pitch-threshold", type=float, default=999.0)
    parser.add_argument("--recovery-pitch-rate-threshold", type=float, default=999.0)
    parser.add_argument("--recovery-hip-pitch-offset", type=float, default=0.0)
    parser.add_argument("--recovery-knee-offset", type=float, default=0.0)
    parser.add_argument("--recovery-ankle-pitch-offset", type=float, default=0.0)
    parser.add_argument("--recovery-waist-pitch-offset", type=float, default=0.0)
    parser.add_argument("--terminal-hold-start-step", type=int, default=-1)
    parser.add_argument("--terminal-hold-box-target-travel", type=float, default=-1.0)
    parser.add_argument("--terminal-hold-robot-target-travel", type=float, default=-1.0)
    parser.add_argument("--terminal-hold-pitch-threshold", type=float, default=999.0)
    parser.add_argument("--terminal-hold-pitch-rate-threshold", type=float, default=999.0)
    parser.add_argument("--terminal-hold-hip-pitch-offset", type=float, default=0.0)
    parser.add_argument("--terminal-hold-knee-offset", type=float, default=0.0)
    parser.add_argument("--terminal-hold-ankle-pitch-offset", type=float, default=0.0)
    parser.add_argument("--terminal-hold-waist-pitch-offset", type=float, default=0.0)
    parser.add_argument("--terminal-drive-gain-scale", type=float, default=-1.0)
    parser.add_argument("--terminal-drive-force-scale", type=float, default=-1.0)
    parser.add_argument("--balance-feedback-controller", action="store_true")
    parser.add_argument("--balance-pitch-gain", type=float, default=0.0)
    parser.add_argument("--balance-roll-gain", type=float, default=0.0)
    parser.add_argument("--balance-pitch-rate-gain", type=float, default=0.0)
    parser.add_argument("--balance-roll-rate-gain", type=float, default=0.0)
    parser.add_argument("--balance-adjustment-limit", type=float, default=0.25)
    parser.add_argument("--balance-feedback-base", choices=("stand", "command"), default="stand")
    parser.add_argument("--balance-start-on-agile-hold", action="store_true")
    parser.add_argument("--balance-roll-left-ankle-scale", type=float, default=1.0)
    parser.add_argument("--balance-roll-right-ankle-scale", type=float, default=1.0)
    parser.add_argument("--balance-roll-left-hip-scale", type=float, default=-0.5)
    parser.add_argument("--balance-roll-right-hip-scale", type=float, default=-0.5)
    parser.add_argument("--balance-pitch-target", type=float, default=0.0)
    parser.add_argument("--balance-roll-target", type=float, default=0.0)
    parser.add_argument("--balance-roll-target-from-lateral", action="store_true")
    parser.add_argument(
        "--balance-roll-target-lateral-source",
        choices=("robot", "box", "average"),
        default="robot",
    )
    parser.add_argument("--balance-roll-target-lateral-gain", type=float, default=0.0)
    parser.add_argument("--balance-roll-target-lateral-limit", type=float, default=0.0)
    parser.add_argument("--balance-roll-target-lateral-deadband", type=float, default=0.0)
    parser.add_argument("--balance-roll-target-lateral-sign", type=float, default=1.0)
    parser.add_argument("--balance-roll-target-lateral-start-after-hold-steps", type=int, default=0)
    parser.add_argument("--balance-roll-target-lateral-ramp-steps", type=int, default=0)
    parser.add_argument("--balance-roll-target-lateral-max-tilt", type=float, default=999.0)
    parser.add_argument("--balance-roll-target-lateral-max-box-tilt", type=float, default=999.0)
    parser.add_argument("--balance-target-start-step", type=int, default=0)
    parser.add_argument("--balance-target-end-step", type=int, default=-1)
    parser.add_argument("--balance-target-pulse-period-steps", type=int, default=0)
    parser.add_argument("--balance-target-pulse-width-steps", type=int, default=0)
    parser.add_argument("--balance-target-pulse-phase-step", type=int, default=0)
    parser.add_argument("--balance-pitch-sign", type=float, default=-1.0)
    parser.add_argument("--balance-roll-sign", type=float, default=-1.0)
    parser.add_argument("--balance-start-step", type=int, default=0)
    parser.add_argument("--balance-pitch-activation-threshold", type=float, default=0.0)
    parser.add_argument("--balance-roll-activation-threshold", type=float, default=0.0)
    parser.add_argument("--balance-pitch-rate-activation-threshold", type=float, default=0.0)
    parser.add_argument("--balance-roll-rate-activation-threshold", type=float, default=0.0)
    parser.add_argument("--diagnostic-root-drive", choices=("none", "smooth_x"), default="none")
    parser.add_argument("--diagnostic-root-drive-start-step", type=int, default=0)
    parser.add_argument("--diagnostic-root-drive-stop-step", type=int, default=-1)
    parser.add_argument("--diagnostic-root-drive-speed", type=float, default=0.0)
    parser.add_argument("--diagnostic-root-drive-ramp-steps", type=int, default=120)
    parser.add_argument("--target-window-center", type=float, default=-1.0)
    parser.add_argument("--target-window-halfwidth", type=float, default=-1.0)
    parser.add_argument("--arm-pose-mode", choices=("none", "right_front_reach", "both_front_reach", "manual"), default="none")
    parser.add_argument("--arm-pose-start-step", type=int, default=0)
    parser.add_argument("--arm-pose-ramp-steps", type=int, default=120)
    parser.add_argument("--right-shoulder-pitch", type=float, default=None)
    parser.add_argument("--right-shoulder-roll", type=float, default=None)
    parser.add_argument("--right-shoulder-yaw", type=float, default=None)
    parser.add_argument("--right-elbow", type=float, default=None)
    parser.add_argument("--right-wrist-roll", type=float, default=None)
    parser.add_argument("--right-wrist-pitch", type=float, default=None)
    parser.add_argument("--right-wrist-yaw", type=float, default=None)
    parser.add_argument("--left-shoulder-pitch", type=float, default=None)
    parser.add_argument("--left-shoulder-roll", type=float, default=None)
    parser.add_argument("--left-shoulder-yaw", type=float, default=None)
    parser.add_argument("--left-elbow", type=float, default=None)
    parser.add_argument("--left-wrist-roll", type=float, default=None)
    parser.add_argument("--left-wrist-pitch", type=float, default=None)
    parser.add_argument("--left-wrist-yaw", type=float, default=None)
    parser.add_argument("--box-retention-posture-controller", action="store_true")
    parser.add_argument("--box-retention-rel-start", type=float, default=0.10)
    parser.add_argument("--box-retention-rel-stop", type=float, default=0.28)
    parser.add_argument("--box-retention-tilt-start", type=float, default=0.20)
    parser.add_argument("--box-retention-tilt-stop", type=float, default=0.55)
    parser.add_argument("--box-retention-hip-pitch-offset", type=float, default=-0.04)
    parser.add_argument("--box-retention-knee-offset", type=float, default=0.12)
    parser.add_argument("--box-retention-ankle-pitch-offset", type=float, default=-0.06)
    parser.add_argument("--box-retention-waist-pitch-offset", type=float, default=-0.03)
    parser.add_argument("--box-retention-shoulder-pitch-offset", type=float, default=-0.10)
    parser.add_argument("--box-retention-elbow-offset", type=float, default=0.16)
    parser.add_argument("--box-retention-wrist-pitch-offset", type=float, default=-0.04)
    parser.add_argument("--policy-start-step", type=int, default=40)
    parser.add_argument("--policy-control-decimation", type=int, default=4)
    parser.add_argument("--agile-command", type=float, nargs=3, default=(0.25, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
    parser.add_argument("--agile-height-command", type=float, default=0.72)
    parser.add_argument("--agile-command-stop-step", type=int, default=-1)
    parser.add_argument("--agile-command-stop-box-target-travel", type=float, default=-1.0)
    parser.add_argument("--agile-command-stop-robot-target-travel", type=float, default=-1.0)
    parser.add_argument("--agile-command-stop-target-window", action="store_true")
    parser.add_argument("--agile-command-stop-target-window-min-step", type=int, default=-1)
    parser.add_argument("--agile-command-hold-scale", type=float, default=0.0)
    parser.add_argument("--agile-command-hold-adaptive-scale", action="store_true")
    parser.add_argument("--agile-command-hold-adaptive-min-scale", type=float, default=0.0)
    parser.add_argument("--agile-command-hold-adaptive-max-scale", type=float, default=1.0)
    parser.add_argument("--agile-command-hold-adaptive-tilt-start", type=float, default=0.20)
    parser.add_argument("--agile-command-hold-adaptive-tilt-stop", type=float, default=0.65)
    parser.add_argument("--agile-command-hold-adaptive-rate-start", type=float, default=2.0)
    parser.add_argument("--agile-command-hold-adaptive-rate-stop", type=float, default=8.0)
    parser.add_argument("--agile-command-hold-adaptive-rel-start", type=float, default=0.16)
    parser.add_argument("--agile-command-hold-adaptive-rel-stop", type=float, default=0.35)
    parser.add_argument("--agile-command-hold-adaptive-box-tilt", action="store_true")
    parser.add_argument("--agile-command-hold-adaptive-box-tilt-start", type=float, default=0.16)
    parser.add_argument("--agile-command-hold-adaptive-box-tilt-stop", type=float, default=0.45)
    parser.add_argument("--agile-command-hold-adaptive-box-tilt-rate-start", type=float, default=2.0)
    parser.add_argument("--agile-command-hold-adaptive-box-tilt-rate-stop", type=float, default=8.0)
    parser.add_argument("--agile-command-hold-adaptive-scale-smoothing", type=float, default=0.15)
    parser.add_argument("--agile-command-hold-lateral-correction", action="store_true")
    parser.add_argument("--agile-command-hold-lateral-gain", type=float, default=0.0)
    parser.add_argument("--agile-command-hold-lateral-limit", type=float, default=0.05)
    parser.add_argument("--agile-command-hold-lateral-sign", type=float, default=1.0)
    parser.add_argument("--agile-command-hold-lateral-terminal-only", action="store_true")
    parser.add_argument("--agile-command-hold-lateral-error-start", type=float, default=0.0)
    parser.add_argument("--agile-command-hold-lateral-use-excess-error", action="store_true")
    parser.add_argument("--agile-command-hold-lateral-max-tilt", type=float, default=999.0)
    parser.add_argument("--agile-command-hold-lateral-max-box-tilt", type=float, default=999.0)
    parser.add_argument("--agile-command-hold-yaw-correction", action="store_true")
    parser.add_argument("--agile-command-hold-yaw-gain", type=float, default=0.0)
    parser.add_argument("--agile-command-hold-yaw-limit", type=float, default=0.20)
    parser.add_argument("--agile-command-hold-yaw-sign", type=float, default=1.0)
    parser.add_argument("--agile-command-box-progress-controller", action="store_true")
    parser.add_argument("--agile-command-box-progress-start-step", type=int, default=0)
    parser.add_argument("--agile-command-box-progress-target", type=float, default=-1.0)
    parser.add_argument("--agile-command-box-progress-deadband", type=float, default=0.05)
    parser.add_argument("--agile-command-box-progress-gain", type=float, default=0.08)
    parser.add_argument("--agile-command-box-progress-max-forward", type=float, default=0.10)
    parser.add_argument("--agile-command-box-progress-max-reverse", type=float, default=0.03)
    parser.add_argument("--agile-command-box-progress-max-tilt", type=float, default=999.0)
    parser.add_argument("--agile-command-box-progress-max-box-tilt", type=float, default=999.0)
    parser.add_argument("--agile-command-box-progress-scale-on-hold", action="store_true")
    parser.add_argument("--agile-command-box-lateral-controller", action="store_true")
    parser.add_argument("--agile-command-box-lateral-deadband", type=float, default=0.08)
    parser.add_argument("--agile-command-box-lateral-gain", type=float, default=0.02)
    parser.add_argument("--agile-command-box-lateral-limit", type=float, default=0.004)
    parser.add_argument("--agile-command-box-lateral-sign", type=float, default=1.0)
    parser.add_argument("--agile-command-box-lateral-scale-on-hold", action="store_true")
    parser.add_argument("--agile-command-hold-terminal-box-target-travel", type=float, default=-1.0)
    parser.add_argument("--agile-command-hold-terminal-min-robot-target-travel", type=float, default=-1.0)
    parser.add_argument("--agile-command-hold-terminal-min-step", type=int, default=-1)
    parser.add_argument("--agile-command-hold-terminal-scale", type=float, default=0.0)
    parser.add_argument("--agile-command-hold-terminal-latch", action="store_true")
    parser.add_argument("--agile-command-hold-final-box-target-travel", type=float, default=-1.0)
    parser.add_argument("--agile-command-hold-final-min-robot-target-travel", type=float, default=-1.0)
    parser.add_argument("--agile-command-hold-final-min-step", type=int, default=-1)
    parser.add_argument("--agile-command-hold-final-scale", type=float, default=-1.0)
    parser.add_argument("--agile-command-hold-final-latch", action="store_true")
    parser.add_argument("--agile-command-hold-final-zero-corrections", action="store_true")
    parser.add_argument("--agile-command-hold-final-reset-policy-state", action="store_true")
    parser.add_argument("--agile-command-hold-final-brake-command-x", type=float, default=0.0)
    parser.add_argument("--agile-command-hold-final-brake-delay-steps", type=int, default=0)
    parser.add_argument("--agile-command-hold-final-brake-steps", type=int, default=0)
    parser.add_argument("--agile-command-hold-final-freeze-in-target-window", action="store_true")
    parser.add_argument("--agile-command-hold-final-freeze-max-tilt", type=float, default=0.25)
    parser.add_argument("--agile-command-hold-final-freeze-max-box-tilt", type=float, default=0.35)
    parser.add_argument("--agile-command-hold-rescue-overrides-final-freeze", action="store_true")
    parser.add_argument("--agile-command-hold-stand-overrides-final-freeze", action="store_true")
    parser.add_argument("--agile-command-hold-final-stand", action="store_true")
    parser.add_argument("--agile-command-hold-final-stand-delay-steps", type=int, default=0)
    parser.add_argument(
        "--agile-command-hold-mode",
        choices=("policy_command", "stand_targets", "policy_then_stand"),
        default="policy_command",
    )
    parser.add_argument("--agile-command-hold-stand-blend-rate", type=float, default=0.04)
    parser.add_argument("--agile-command-hold-policy-then-stand-delay-steps", type=int, default=80)
    parser.add_argument("--agile-command-hold-stand-hip-pitch", type=float, default=None)
    parser.add_argument("--agile-command-hold-stand-knee", type=float, default=None)
    parser.add_argument("--agile-command-hold-stand-ankle-pitch", type=float, default=None)
    parser.add_argument("--agile-command-hold-stand-hip-roll", type=float, default=None)
    parser.add_argument("--agile-command-hold-stand-ankle-roll", type=float, default=None)
    parser.add_argument("--agile-command-hold-stand-waist-pitch", type=float, default=None)
    parser.add_argument("--agile-command-hold-rescue-enable", action="store_true")
    parser.add_argument("--agile-command-hold-rescue-forward-pitch-threshold", type=float, default=-999.0)
    parser.add_argument("--agile-command-hold-rescue-abs-roll-threshold", type=float, default=999.0)
    parser.add_argument("--agile-command-hold-rescue-blend-rate", type=float, default=0.04)
    parser.add_argument("--agile-command-hold-rescue-hip-pitch", type=float, default=None)
    parser.add_argument("--agile-command-hold-rescue-knee", type=float, default=None)
    parser.add_argument("--agile-command-hold-rescue-ankle-pitch", type=float, default=None)
    parser.add_argument("--agile-command-hold-rescue-hip-roll", type=float, default=None)
    parser.add_argument("--agile-command-hold-rescue-ankle-roll", type=float, default=None)
    parser.add_argument("--agile-command-hold-rescue-waist-pitch", type=float, default=None)
    parser.add_argument("--agile-command-hold-reset-policy-state", action="store_true")
    parser.add_argument("--agile-policy-backend", choices=("onnx", "torch_checkpoint"), default="torch_checkpoint")
    parser.add_argument(
        "--agile-config",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/"
            "isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/g1_agile.yaml"
        ),
    )
    parser.add_argument(
        "--agile-onnx",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/external/WBC-AGILE/agile/data/policy/"
            "velocity_height_g1/unitree_g1_velocity_height_recurrent_student.onnx"
        ),
    )
    parser.add_argument(
        "--agile-torch-checkpoint",
        type=Path,
        default=Path(
            "/public/home/yanhongru/Curiosity/external/WBC-AGILE/agile/data/policy/"
            "velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt"
        ),
    )
    parser.add_argument("--attach-box", choices=("none", "fixed_torso"), default="none")
    parser.add_argument("--attach-body-path", default="/World/G1/torso_link")
    parser.add_argument("--attach-local-pos0", type=float, nargs=3, default=(0.24, 0.0, 0.08), metavar=("X", "Y", "Z"))
    parser.add_argument("--torso-cradle", choices=("none", "front_tray"), default="none")
    parser.add_argument("--cradle-deck-size", type=float, nargs=3, default=(0.34, 0.42, 0.035), metavar=("X", "Y", "Z"))
    parser.add_argument("--cradle-deck-local-pos0", type=float, nargs=3, default=(0.30, 0.0, -0.02), metavar=("X", "Y", "Z"))
    parser.add_argument("--cradle-side-rail-height", type=float, default=0.12)
    parser.add_argument("--cradle-end-stop-height", type=float, default=0.16)
    parser.add_argument("--cradle-rail-thickness", type=float, default=0.025)
    parser.add_argument("--cradle-mass-scale", type=float, default=1.0)
    parser.add_argument("--cradle-top-lid", action="store_true")
    parser.add_argument("--cradle-top-lid-local-z", type=float, default=0.12)
    parser.add_argument("--cradle-top-lid-thickness", type=float, default=0.018)
    parser.add_argument("--cradle-top-lid-x-scale", type=float, default=1.0)
    parser.add_argument("--cradle-top-lid-y-scale", type=float, default=1.0)
    parser.add_argument("--cradle-top-lid-enable-on-hold", action="store_true")
    parser.add_argument("--cradle-chest-pad", action="store_true")
    parser.add_argument("--cradle-chest-pad-local-pos0", type=float, nargs=3, default=(0.10, 0.0, 0.08))
    parser.add_argument("--cradle-chest-pad-size", type=float, nargs=3, default=(0.035, 0.34, 0.20))
    parser.add_argument("--cradle-chest-pad-mass-scale", type=float, default=1.0)
    parser.add_argument("--cradle-chest-pad-spawn-on-trigger", action="store_true")
    parser.add_argument("--cradle-chest-pad-enable-on-hold", action="store_true")
    parser.add_argument("--cradle-chest-pad-enable-on-terminal-hold", action="store_true")
    parser.add_argument("--cradle-chest-pad-enable-on-final-hold", action="store_true")
    parser.add_argument("--cradle-chest-pad-enable-on-target-window", action="store_true")
    parser.add_argument("--cradle-chest-pad-target-window-min-step", type=int, default=-1)
    parser.add_argument("--cradle-chest-pad-enable-on-box-tilt", action="store_true")
    parser.add_argument("--cradle-chest-pad-box-tilt-threshold", type=float, default=999.0)
    parser.add_argument("--cradle-chest-pad-box-tilt-min-step", type=int, default=-1)
    parser.add_argument("--disable-cradle-collision", action="store_true")
    parser.add_argument("--probe-mode", choices=("none", "front_bumper"), default="none")
    parser.add_argument("--probe-start-step", type=int, default=0)
    parser.add_argument("--probe-end-step", type=int, default=-1)
    parser.add_argument("--probe-collision-window", action="store_true")
    parser.add_argument("--probe-pad-size", type=float, nargs=3, default=(0.05, 0.36, 0.18), metavar=("X", "Y", "Z"))
    parser.add_argument("--probe-pad-local-pos0", type=float, nargs=3, default=(0.50, 0.0, 0.02), metavar=("X", "Y", "Z"))
    parser.add_argument("--probe-pad-mass", type=float, default=0.2)
    parser.add_argument("--disable-probe-pad-collision", action="store_true")
    parser.add_argument("--grasp-mode", choices=("none", "staged_fixed_torso", "staged_fixed_body"), default="none")
    parser.add_argument("--grasp-body-path", default="/World/G1/torso_link")
    parser.add_argument("--grasp-enable-step", type=int, default=120)
    parser.add_argument("--grasp-lift-offset-z", type=float, default=0.0)
    parser.add_argument("--require-box-no-drop", action="store_true")
    parser.add_argument("--fall-z", type=float, default=0.45)
    parser.add_argument("--drop-z", type=float, default=0.20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--capture-rgb", action="store_true")
    parser.add_argument("--capture-rgb-every-n-steps", type=int, default=10)
    parser.add_argument("--capture-rgb-resolution", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--capture-rgb-rt-subframes", type=int, default=4)
    parser.add_argument("--capture-camera-position", type=float, nargs=3, default=(1.8, -2.4, 1.25), metavar=("X", "Y", "Z"))
    parser.add_argument("--capture-camera-look-at", type=float, nargs=3, default=(-0.45, 0.0, 0.82), metavar=("X", "Y", "Z"))
    parser.add_argument("--record-replay-csv", action="store_true")
    parser.add_argument("--record-replay-every-n-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/core_world_g1_box_scene"))
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROGRESS] AppLauncher started", flush=True)

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


G1_PATH = "/World/G1"
G1_ARTICULATION_PATH = "/World/G1/pelvis"
BOX_PATH = "/World/CarryBox"
BOX_SUPPORT_TABLE_PATH = "/World/CarryBoxSupportTable"
TARGET_XY = (float(args_cli.target_xy[0]), float(args_cli.target_xy[1]))
G1_STAND_JOINT_TARGETS = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
}
G1_STAND_DRIVE_GAINS = {
    "left_hip_pitch_joint": (150.0, 2.0, 88.0),
    "left_hip_roll_joint": (150.0, 2.0, 88.0),
    "left_hip_yaw_joint": (150.0, 2.0, 88.0),
    "left_knee_joint": (300.0, 4.0, 139.0),
    "left_ankle_pitch_joint": (40.0, 2.0, 50.0),
    "left_ankle_roll_joint": (40.0, 2.0, 50.0),
    "right_hip_pitch_joint": (150.0, 2.0, 88.0),
    "right_hip_roll_joint": (150.0, 2.0, 88.0),
    "right_hip_yaw_joint": (150.0, 2.0, 88.0),
    "right_knee_joint": (300.0, 4.0, 139.0),
    "right_ankle_pitch_joint": (40.0, 2.0, 50.0),
    "right_ankle_roll_joint": (40.0, 2.0, 50.0),
    "waist_yaw_joint": (250.0, 5.0, 88.0),
    "waist_roll_joint": (250.0, 5.0, 50.0),
    "waist_pitch_joint": (250.0, 5.0, 50.0),
    "left_shoulder_pitch_joint": (100.0, 5.0, 25.0),
    "left_shoulder_roll_joint": (100.0, 5.0, 25.0),
    "left_shoulder_yaw_joint": (40.0, 2.0, 25.0),
    "left_elbow_joint": (40.0, 2.0, 25.0),
    "right_shoulder_pitch_joint": (100.0, 5.0, 25.0),
    "right_shoulder_roll_joint": (100.0, 5.0, 25.0),
    "right_shoulder_yaw_joint": (40.0, 2.0, 25.0),
    "right_elbow_joint": (40.0, 2.0, 25.0),
}
G1_ISAACLAB_29DOF_DRIVE_GAINS = {
    "left_hip_pitch_joint": (100.0, 2.5, 88.0),
    "left_hip_roll_joint": (100.0, 2.5, 88.0),
    "left_hip_yaw_joint": (100.0, 2.5, 88.0),
    "left_knee_joint": (200.0, 5.0, 139.0),
    "left_ankle_pitch_joint": (20.0, 0.2, 50.0),
    "left_ankle_roll_joint": (20.0, 0.1, 50.0),
    "right_hip_pitch_joint": (100.0, 2.5, 88.0),
    "right_hip_roll_joint": (100.0, 2.5, 88.0),
    "right_hip_yaw_joint": (100.0, 2.5, 88.0),
    "right_knee_joint": (200.0, 5.0, 139.0),
    "right_ankle_pitch_joint": (20.0, 0.2, 50.0),
    "right_ankle_roll_joint": (20.0, 0.1, 50.0),
    "waist_yaw_joint": (5000.0, 5.0, 88.0),
    "waist_roll_joint": (5000.0, 5.0, 50.0),
    "waist_pitch_joint": (5000.0, 5.0, 50.0),
    "left_shoulder_pitch_joint": (3000.0, 10.0, 300.0),
    "left_shoulder_roll_joint": (3000.0, 10.0, 300.0),
    "left_shoulder_yaw_joint": (3000.0, 10.0, 300.0),
    "left_elbow_joint": (3000.0, 10.0, 300.0),
    "left_wrist_roll_joint": (3000.0, 10.0, 300.0),
    "left_wrist_pitch_joint": (3000.0, 10.0, 300.0),
    "left_wrist_yaw_joint": (3000.0, 10.0, 300.0),
    "right_shoulder_pitch_joint": (3000.0, 10.0, 300.0),
    "right_shoulder_roll_joint": (3000.0, 10.0, 300.0),
    "right_shoulder_yaw_joint": (3000.0, 10.0, 300.0),
    "right_elbow_joint": (3000.0, 10.0, 300.0),
    "right_wrist_roll_joint": (3000.0, 10.0, 300.0),
    "right_wrist_pitch_joint": (3000.0, 10.0, 300.0),
    "right_wrist_yaw_joint": (3000.0, 10.0, 300.0),
}


def _stand_joint_targets() -> dict[str, float]:
    targets = dict(G1_STAND_JOINT_TARGETS)
    paired_overrides = {
        "hip_pitch": args_cli.stand_hip_pitch,
        "knee": args_cli.stand_knee,
        "ankle_pitch": args_cli.stand_ankle_pitch,
        "hip_roll": args_cli.stand_hip_roll,
        "ankle_roll": args_cli.stand_ankle_roll,
    }
    for joint_suffix, value in paired_overrides.items():
        if value is None:
            continue
        for side in ("left", "right"):
            joint_name = f"{side}_{joint_suffix}_joint"
            if joint_name in targets:
                targets[joint_name] = float(value)
    return targets


class AgileOnnxJointPolicy:
    """Official WBC-AGILE ONNX adapter for the Core API G1 scene."""

    def __init__(self, config_path: Path, model_path: Path):
        import onnxruntime as ort
        import yaml

        self.config_path = Path(config_path)
        self.model_path = Path(model_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"AGILE config not found: {self.config_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"AGILE ONNX model not found: {self.model_path}")
        self.config = yaml.safe_load(self.config_path.read_text())
        self.input_joint_names = list(self.config["onnx_input_joint_names"])
        self.controlled_joint_names = list(self.config["controlled_joint_names"])
        stand_targets = _stand_joint_targets()
        self.default_q = np.array(
            [stand_targets.get(name, 0.0) for name in self.input_joint_names],
            dtype=np.float32,
        ).reshape(1, -1)
        self.action_clip = tuple(float(v) for v in self.config.get("action_clip", (-6.0, 6.0)))
        self.action_scale = np.array(self.config["action_scale"], dtype=np.float32).reshape(1, -1)
        self.action_offset = np.array(self.config["action_offset"], dtype=np.float32).reshape(1, -1)
        self.joint_vel_scale = float(self.config.get("joint_vel_scale", 0.1))
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.h_state = np.zeros((1, 1, 256), dtype=np.float32)
        self.c_state = np.zeros((1, 1, 256), dtype=np.float32)
        self.last_action = np.zeros((1, len(self.controlled_joint_names)), dtype=np.float32)
        self.last_raw_action_norm = 0.0
        print(
            "[PROGRESS] AGILE ONNX policy loaded "
            f"model={self.model_path} inputs={self.input_names} outputs={self.output_names}",
            flush=True,
        )

    def reset_state(self) -> bool:
        self.h_state.fill(0.0)
        self.c_state.fill(0.0)
        self.last_action.fill(0.0)
        self.last_raw_action_norm = 0.0
        return True

    def _collect(self, values: np.ndarray, joint_names: list[str], names: list[str], fill: float = 0.0) -> np.ndarray:
        out = np.full((len(names),), float(fill), dtype=np.float32)
        for idx, name in enumerate(names):
            if name in joint_names and values.size > joint_names.index(name):
                out[idx] = float(values[joint_names.index(name)])
        return out

    def infer(
        self,
        command: tuple[float, float, float],
        height_command: float,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_names: list[str],
        projected_gravity_b: np.ndarray,
        root_ang_vel_b: np.ndarray | None = None,
    ) -> dict[str, float]:
        q = self._collect(joint_positions, joint_names, self.input_joint_names).reshape(1, -1)
        dq = self._collect(joint_velocities, joint_names, self.input_joint_names).reshape(1, -1)
        cmd = np.array([[float(command[0]), float(command[1]), float(command[2]), float(height_command)]], dtype=np.float32)
        ang_vel = np.zeros((1, 3), dtype=np.float32) if root_ang_vel_b is None else np.asarray(root_ang_vel_b, dtype=np.float32).reshape(1, 3)
        gravity = np.asarray(projected_gravity_b, dtype=np.float32).reshape(1, 3)
        obs = np.concatenate(
            [cmd, ang_vel, gravity, q - self.default_q, dq * self.joint_vel_scale, self.last_action],
            axis=1,
        ).astype(np.float32)
        outputs = self.session.run(self.output_names, {"obs": obs, "h_in": self.h_state, "c_in": self.c_state})
        result = dict(zip(self.output_names, outputs))
        raw = np.asarray(result["actions"], dtype=np.float32)
        self.h_state = np.asarray(result["h_out"], dtype=np.float32)
        self.c_state = np.asarray(result["c_out"], dtype=np.float32)
        self.last_action = raw
        self.last_raw_action_norm = float(np.linalg.norm(raw))
        target = np.clip(raw, self.action_clip[0], self.action_clip[1]) * self.action_scale + self.action_offset
        return {name: float(target[0, idx]) for idx, name in enumerate(self.controlled_joint_names)}


class AgileTorchCheckpointJointPolicy:
    """Official WBC-AGILE PyTorch checkpoint adapter for the Core API G1 scene."""

    def __init__(self, agile_config_path: Path, io_config_path: Path, checkpoint_path: Path):
        import torch
        import yaml

        policy_path = Path("/public/home/yanhongru/Curiosity/external/WBC-AGILE/agile/sim2mujoco/policy.py")
        spec = importlib.util.spec_from_file_location("wbc_agile_policy_direct", policy_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load WBC-AGILE policy module from {policy_path}")
        policy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy_module)
        PolicyWrapper = policy_module.PolicyWrapper
        CheckpointPolicyWrapper = policy_module.CheckpointPolicyWrapper

        self.agile_config_path = Path(agile_config_path)
        self.io_config_path = Path(io_config_path)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.agile_config_path.is_file():
            raise FileNotFoundError(f"AGILE config not found: {self.agile_config_path}")
        if not self.io_config_path.is_file():
            raise FileNotFoundError(f"AGILE IO config not found: {self.io_config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"AGILE torch checkpoint not found: {self.checkpoint_path}")
        self.config = yaml.safe_load(self.agile_config_path.read_text())
        io_config = yaml.safe_load(self.io_config_path.read_text())
        self.input_joint_names = list(self.config["onnx_input_joint_names"])
        self.controlled_joint_names = list(self.config["controlled_joint_names"])
        stand_targets = _stand_joint_targets()
        self.default_q = np.array(
            [stand_targets.get(name, 0.0) for name in self.input_joint_names],
            dtype=np.float32,
        ).reshape(1, -1)
        self.action_clip = tuple(float(v) for v in self.config.get("action_clip", (-6.0, 6.0)))
        self.action_scale = np.array(self.config["action_scale"], dtype=np.float32).reshape(1, -1)
        self.action_offset = np.array(self.config["action_offset"], dtype=np.float32).reshape(1, -1)
        self.joint_vel_scale = float(self.config.get("joint_vel_scale", 0.1))
        self.device = torch.device("cpu")
        if self.checkpoint_path.name.endswith("_checkpoint.pt"):
            self.policy = CheckpointPolicyWrapper.from_checkpoint(self.checkpoint_path, io_config, self.device)
        else:
            self.policy = PolicyWrapper.from_config(self.checkpoint_path, io_config, self.device)
        self.last_action = np.zeros((1, len(self.controlled_joint_names)), dtype=np.float32)
        self.last_raw_action_norm = 0.0
        print(
            "[PROGRESS] AGILE torch checkpoint policy loaded "
            f"checkpoint={self.checkpoint_path}",
            flush=True,
        )

    def reset_state(self) -> bool:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.last_action.fill(0.0)
        self.last_raw_action_norm = 0.0
        return True

    def _collect(self, values: np.ndarray, joint_names: list[str], names: list[str], fill: float = 0.0) -> np.ndarray:
        out = np.full((len(names),), float(fill), dtype=np.float32)
        for idx, name in enumerate(names):
            if name in joint_names and values.size > joint_names.index(name):
                out[idx] = float(values[joint_names.index(name)])
        return out

    def infer(
        self,
        command: tuple[float, float, float],
        height_command: float,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_names: list[str],
        projected_gravity_b: np.ndarray,
        root_ang_vel_b: np.ndarray | None = None,
    ) -> dict[str, float]:
        import torch

        q = self._collect(joint_positions, joint_names, self.input_joint_names).reshape(1, -1)
        dq = self._collect(joint_velocities, joint_names, self.input_joint_names).reshape(1, -1)
        cmd = np.array([[float(command[0]), float(command[1]), float(command[2]), float(height_command)]], dtype=np.float32)
        ang_vel = np.zeros((1, 3), dtype=np.float32) if root_ang_vel_b is None else np.asarray(root_ang_vel_b, dtype=np.float32).reshape(1, 3)
        gravity = np.asarray(projected_gravity_b, dtype=np.float32).reshape(1, 3)
        obs = np.concatenate(
            [cmd, ang_vel, gravity, q - self.default_q, dq * self.joint_vel_scale, self.last_action],
            axis=1,
        ).astype(np.float32)
        obs_t = torch.from_numpy(obs[0]).to(self.device)
        raw_t = self.policy(obs_t)
        raw = raw_t.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
        if raw.shape[1] != len(self.controlled_joint_names):
            raise RuntimeError(
                f"AGILE torch checkpoint output dim {raw.shape[1]} != controlled joints {len(self.controlled_joint_names)}"
            )
        self.last_action = raw
        self.last_raw_action_norm = float(np.linalg.norm(raw))
        target = np.clip(raw, self.action_clip[0], self.action_clip[1]) * self.action_scale + self.action_offset
        return {name: float(target[0, idx]) for idx, name in enumerate(self.controlled_joint_names)}


def _patch_core_api_simulation_manager_compat() -> None:
    if not hasattr(SimulationManager, "_backend"):
        SimulationManager._backend = "numpy"
    if not hasattr(SimulationManager, "get_backend"):
        SimulationManager.get_backend = classmethod(lambda cls: getattr(cls, "_backend", "numpy"))
    if not hasattr(SimulationManager, "_get_backend_utils"):
        def _get_backend_utils(cls):
            import isaacsim.core.utils.numpy as np_utils

            return np_utils

        SimulationManager._get_backend_utils = classmethod(_get_backend_utils)
    if not hasattr(SimulationManager, "get_physics_sim_device"):
        SimulationManager.get_physics_sim_device = classmethod(lambda cls: args_cli.device)
    if not hasattr(SimulationManager, "get_physics_dt"):
        SimulationManager.get_physics_dt = classmethod(lambda cls: 0.005)


def _set_xform(prim: Usd.Prim, translation: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    xform_api.SetScale(Gf.Vec3f(*[float(v) for v in scale]))


def _define_material(stage: Usd.Stage, path: str, static_friction: float, dynamic_friction: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(float(static_friction))
    physics_material.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    physics_material.CreateRestitutionAttr().Set(0.0)
    return material


def _spawn_box(
    stage: Usd.Stage,
    path: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    translation: tuple[float, float, float],
    *,
    rigid: bool = True,
    collision: bool = True,
    material: UsdShade.Material | None = None,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if material is not None and collision:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass_api.CreateMassAttr(float(mass))


def _set_collision_enabled(stage: Usd.Stage, path: str, enabled: bool) -> bool:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return False
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    attr = collision_api.GetCollisionEnabledAttr()
    if not attr.IsValid():
        attr = collision_api.CreateCollisionEnabledAttr()
    attr.Set(bool(enabled))
    return True


def _fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos1]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def _spawn_fixed_torso_box(
    stage: Usd.Stage,
    path: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    attach_body_path: str,
    local_pos0: tuple[float, float, float],
    material: UsdShade.Material | None,
    initial_translation: tuple[float, float, float] | None = None,
    collision: bool = True,
) -> str:
    _spawn_box(
        stage,
        path,
        size,
        mass,
        color,
        initial_translation if initial_translation is not None else local_pos0,
        rigid=True,
        collision=collision,
        material=material,
    )
    joint_path = f"{path}/FixedJointToG1"
    _fixed_joint(stage, joint_path, attach_body_path, path, local_pos0, (0.0, 0.0, 0.0))
    return joint_path


def _spawn_front_cradle_chest_pad(
    stage: Usd.Stage,
    material: UsdShade.Material | None,
    *,
    collision: bool,
) -> str:
    if stage.GetPrimAtPath("/World/G1FrontCradle_chest_pad").IsValid():
        return "/World/G1FrontCradle_chest_pad/FixedJointToG1"
    if not stage.GetPrimAtPath(args_cli.attach_body_path).IsValid():
        raise RuntimeError(f"Attach body path not found after G1 reference: {args_cli.attach_body_path}")
    attach_pose = _usd_world_pose_wxyz(stage, str(args_cli.attach_body_path))
    attach_world = (0.0, 0.0, 0.8) if attach_pose is None else (
        float(attach_pose[0]),
        float(attach_pose[1]),
        float(attach_pose[2]),
    )
    chest_pad_local = tuple(float(v) for v in args_cli.cradle_chest_pad_local_pos0)
    chest_pad_size = tuple(max(0.001, float(v)) for v in args_cli.cradle_chest_pad_size)
    initial_world = (
        attach_world[0] + chest_pad_local[0],
        attach_world[1] + chest_pad_local[1],
        attach_world[2] + chest_pad_local[2],
    )
    mass_scale = max(0.0, float(args_cli.cradle_mass_scale))
    chest_mass_scale = max(0.0, float(args_cli.cradle_chest_pad_mass_scale))
    return _spawn_fixed_torso_box(
        stage,
        "/World/G1FrontCradle_chest_pad",
        chest_pad_size,
        0.20 * mass_scale * chest_mass_scale,
        (0.30, 0.30, 0.30),
        str(args_cli.attach_body_path),
        chest_pad_local,
        material,
        initial_world,
        collision,
    )


def _spawn_front_torso_cradle(stage: Usd.Stage, material: UsdShade.Material | None) -> dict[str, str]:
    if not stage.GetPrimAtPath(args_cli.attach_body_path).IsValid():
        raise RuntimeError(f"Attach body path not found after G1 reference: {args_cli.attach_body_path}")
    attach_pose = _usd_world_pose_wxyz(stage, str(args_cli.attach_body_path))
    attach_world = (0.0, 0.0, 0.8) if attach_pose is None else (float(attach_pose[0]), float(attach_pose[1]), float(attach_pose[2]))

    def approx_world(local: tuple[float, float, float]) -> tuple[float, float, float]:
        return (attach_world[0] + local[0], attach_world[1] + local[1], attach_world[2] + local[2])

    deck_size = tuple(float(v) for v in args_cli.cradle_deck_size)
    deck_pos = tuple(float(v) for v in args_cli.cradle_deck_local_pos0)
    rail_t = float(args_cli.cradle_rail_thickness)
    side_h = float(args_cli.cradle_side_rail_height)
    stop_h = float(args_cli.cradle_end_stop_height)
    mass_scale = max(0.0, float(args_cli.cradle_mass_scale))
    collision = not bool(args_cli.disable_cradle_collision)
    x_len, y_len, z_len = deck_size
    pieces: dict[str, str] = {}
    pieces["deck"] = _spawn_fixed_torso_box(
        stage,
        "/World/G1FrontCradleDeck",
        deck_size,
        0.8 * mass_scale,
        (0.18, 0.18, 0.18),
        str(args_cli.attach_body_path),
        deck_pos,
        material,
        approx_world(deck_pos),
        collision,
    )
    for name, y_sign in (("left_rail", 1.0), ("right_rail", -1.0)):
        local = (deck_pos[0], deck_pos[1] + y_sign * (0.5 * y_len + 0.5 * rail_t), deck_pos[2] + 0.5 * side_h)
        pieces[name] = _spawn_fixed_torso_box(
            stage,
            f"/World/G1FrontCradle_{name}",
            (x_len, rail_t, side_h),
            0.25 * mass_scale,
            (0.20, 0.20, 0.20),
            str(args_cli.attach_body_path),
            local,
            material,
            approx_world(local),
            collision,
        )
    for name, x_sign in (("front_stop", 1.0), ("rear_stop", -1.0)):
        local = (deck_pos[0] + x_sign * (0.5 * x_len + 0.5 * rail_t), deck_pos[1], deck_pos[2] + 0.5 * stop_h)
        pieces[name] = _spawn_fixed_torso_box(
            stage,
            f"/World/G1FrontCradle_{name}",
            (rail_t, y_len + 2.0 * rail_t, stop_h),
            0.3 * mass_scale,
            (0.22, 0.22, 0.22),
            str(args_cli.attach_body_path),
            local,
            material,
            approx_world(local),
            collision,
        )
    if bool(args_cli.cradle_top_lid):
        lid_t = max(0.001, float(args_cli.cradle_top_lid_thickness))
        lid_path = "/World/G1FrontCradle_top_lid"
        lid_local = (deck_pos[0], deck_pos[1], float(args_cli.cradle_top_lid_local_z))
        lid_size = (
            max(0.001, x_len * max(0.0, float(args_cli.cradle_top_lid_x_scale))),
            max(0.001, y_len * max(0.0, float(args_cli.cradle_top_lid_y_scale))),
            lid_t,
        )
        pieces["top_lid"] = _spawn_fixed_torso_box(
            stage,
            lid_path,
            lid_size,
            0.18 * mass_scale,
            (0.26, 0.26, 0.26),
            str(args_cli.attach_body_path),
            lid_local,
            material,
            approx_world(lid_local),
            collision,
        )
        if bool(args_cli.cradle_top_lid_enable_on_hold):
            _set_collision_enabled(stage, lid_path, False)
    if bool(args_cli.cradle_chest_pad) and not bool(args_cli.cradle_chest_pad_spawn_on_trigger):
        chest_pad_path = "/World/G1FrontCradle_chest_pad"
        pieces["chest_pad"] = _spawn_front_cradle_chest_pad(stage, material, collision=collision)
        if (
            bool(args_cli.cradle_chest_pad_enable_on_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_terminal_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_final_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_target_window)
            or bool(args_cli.cradle_chest_pad_enable_on_box_tilt)
        ):
            _set_collision_enabled(stage, chest_pad_path, False)
    return pieces


def _spawn_front_probe_bumper(stage: Usd.Stage, material: UsdShade.Material | None) -> str:
    if not stage.GetPrimAtPath(args_cli.attach_body_path).IsValid():
        raise RuntimeError(f"Attach body path not found after G1 reference: {args_cli.attach_body_path}")
    attach_pose = _usd_world_pose_wxyz(stage, str(args_cli.attach_body_path))
    attach_world = (0.0, 0.0, 0.8) if attach_pose is None else (float(attach_pose[0]), float(attach_pose[1]), float(attach_pose[2]))
    local = tuple(float(v) for v in args_cli.probe_pad_local_pos0)
    initial_world = (attach_world[0] + local[0], attach_world[1] + local[1], attach_world[2] + local[2])
    collision_enabled = not bool(args_cli.disable_probe_pad_collision)
    if bool(args_cli.probe_collision_window):
        collision_enabled = False
    return _spawn_fixed_torso_box(
        stage,
        "/World/G1FrontProbePad",
        tuple(float(v) for v in args_cli.probe_pad_size),
        float(args_cli.probe_pad_mass),
        (0.72, 0.18, 0.14),
        str(args_cli.attach_body_path),
        local,
        material,
        initial_world,
        collision_enabled,
    )


def _pose_wxyz(prim: SingleArticulation | SingleRigidPrim) -> list[float]:
    pos, quat = prim.get_world_pose()
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def _usd_world_pose_wxyz(stage: Usd.Stage, prim_path: str) -> list[float] | None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    imag = quat.GetImaginary()
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(quat.GetReal()),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
    ]


def _quat_rotate_inverse_wxyz(qw: float, qx: float, qy: float, qz: float, vec_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    # Rotate a world-frame vector into a body frame using q^-1 * v * q.
    vx, vy, vz = (float(v) for v in vec_xyz)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1.0e-12:
        return (vx, vy, vz)
    qw, qx, qy, qz = (float(qw) / norm, float(qx) / norm, float(qy) / norm, float(qz) / norm)
    ix, iy, iz, iw = -qx, -qy, -qz, qw
    tx = 2.0 * (iy * vz - iz * vy)
    ty = 2.0 * (iz * vx - ix * vz)
    tz = 2.0 * (ix * vy - iy * vx)
    return (
        vx + iw * tx + (iy * tz - iz * ty),
        vy + iw * ty + (iz * tx - ix * tz),
        vz + iw * tz + (ix * ty - iy * tx),
    )


def _world_delta_as_body_local(body_pose_wxyz: list[float], target_pose_wxyz: list[float], lift_offset_z: float) -> tuple[float, float, float]:
    world_delta = (
        float(target_pose_wxyz[0]) - float(body_pose_wxyz[0]),
        float(target_pose_wxyz[1]) - float(body_pose_wxyz[1]),
        float(target_pose_wxyz[2]) - float(body_pose_wxyz[2]) + float(lift_offset_z),
    )
    return _quat_rotate_inverse_wxyz(
        float(body_pose_wxyz[3]),
        float(body_pose_wxyz[4]),
        float(body_pose_wxyz[5]),
        float(body_pose_wxyz[6]),
        world_delta,
    )


def _quat_to_roll_pitch(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _quat_wxyz_to_euler_deg(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    roll, pitch = _quat_to_roll_pitch(qw, qx, qy, qz)
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _projected_gravity_body_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    # R_world_body columns are body axes in world.  Project world gravity into body frame with R^T g.
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _root_angular_velocity_body(robot: SingleArticulation, pose_wxyz: list[float]) -> tuple[np.ndarray, str | None]:
    try:
        ang_vel_w = np.array(robot.get_angular_velocity(), dtype=float).reshape(-1)
    except Exception as exc:
        return np.zeros(3, dtype=np.float32), f"{type(exc).__name__}: {exc}"
    if ang_vel_w.size < 3:
        return np.zeros(3, dtype=np.float32), f"unexpected angular velocity shape {ang_vel_w.shape}"
    ang_vel_b = _quat_rotate_inverse_wxyz(
        float(pose_wxyz[3]),
        float(pose_wxyz[4]),
        float(pose_wxyz[5]),
        float(pose_wxyz[6]),
        (float(ang_vel_w[0]), float(ang_vel_w[1]), float(ang_vel_w[2])),
    )
    return np.array(ang_vel_b, dtype=np.float32), None


def _target_direction_xy(initial_xy: tuple[float, float]) -> tuple[float, float]:
    dx = float(TARGET_XY[0]) - float(initial_xy[0])
    dy = float(TARGET_XY[1]) - float(initial_xy[1])
    norm = math.hypot(dx, dy)
    if norm < 1.0e-9:
        return (1.0, 0.0)
    return (dx / norm, dy / norm)


def _project_xy_delta(
    pose: list[float],
    initial_pose: list[float],
    direction_xy: tuple[float, float],
) -> float:
    return (
        (float(pose[0]) - float(initial_pose[0])) * float(direction_xy[0])
        + (float(pose[1]) - float(initial_pose[1])) * float(direction_xy[1])
    )


def _lateral_xy_delta(
    pose: list[float],
    initial_pose: list[float],
    direction_xy: tuple[float, float],
) -> float:
    # Positive is the left-hand normal of the target direction in world XY.
    normal_x = -float(direction_xy[1])
    normal_y = float(direction_xy[0])
    return (float(pose[0]) - float(initial_pose[0])) * normal_x + (
        float(pose[1]) - float(initial_pose[1])
    ) * normal_y


def _ramp01(value: float, start: float, stop: float) -> float:
    if float(stop) <= float(start):
        return 1.0 if float(value) >= float(stop) else 0.0
    return max(0.0, min(1.0, (float(value) - float(start)) / (float(stop) - float(start))))


def _set_stand_drive_gains(stage: Usd.Stage, gain_scale: float, force_scale: float) -> dict[str, dict[str, float]]:
    applied: dict[str, dict[str, float]] = {}
    gain_table = G1_ISAACLAB_29DOF_DRIVE_GAINS if str(args_cli.stand_drive_preset) == "isaaclab29dof" else G1_STAND_DRIVE_GAINS
    stand_targets = _stand_joint_targets()
    for prim in stage.Traverse():
        joint_name = prim.GetName()
        if joint_name not in gain_table:
            continue
        stiffness, damping, max_force = gain_table[joint_name]
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(stiffness) * float(gain_scale))
        drive.CreateDampingAttr().Set(float(damping) * float(gain_scale))
        drive.CreateMaxForceAttr().Set(float(max_force) * float(force_scale))
        drive.CreateTargetPositionAttr().Set(float(stand_targets.get(joint_name, 0.0)))
        drive.CreateTargetVelocityAttr().Set(0.0)
        applied[joint_name] = {
            "stiffness": float(stiffness) * float(gain_scale),
            "damping": float(damping) * float(gain_scale),
            "max_force": float(max_force) * float(force_scale),
            "target_position": float(stand_targets.get(joint_name, 0.0)),
        }
    return applied


def _apply_stand_drive_gains(stage: Usd.Stage) -> dict[str, dict[str, float]]:
    return _set_stand_drive_gains(stage, float(args_cli.stand_gain_scale), float(args_cli.stand_force_scale))


def _command_joint_position(
    command: np.ndarray,
    joint_names: list[str],
    joint_name: str,
    value: float,
) -> None:
    if joint_name in joint_names:
        command[joint_names.index(joint_name)] = float(value)


def _apply_hold_stand_overrides(command: np.ndarray, joint_names: list[str]) -> dict[str, float]:
    applied: dict[str, float] = {}
    paired_overrides = {
        "hip_pitch": args_cli.agile_command_hold_stand_hip_pitch,
        "knee": args_cli.agile_command_hold_stand_knee,
        "ankle_pitch": args_cli.agile_command_hold_stand_ankle_pitch,
        "hip_roll": args_cli.agile_command_hold_stand_hip_roll,
        "ankle_roll": args_cli.agile_command_hold_stand_ankle_roll,
    }
    for joint_suffix, value in paired_overrides.items():
        if value is None:
            continue
        for side in ("left", "right"):
            joint_name = f"{side}_{joint_suffix}_joint"
            if joint_name in joint_names:
                command[joint_names.index(joint_name)] = float(value)
                applied[joint_name] = float(value)
    if args_cli.agile_command_hold_stand_waist_pitch is not None and "waist_pitch_joint" in joint_names:
        command[joint_names.index("waist_pitch_joint")] = float(args_cli.agile_command_hold_stand_waist_pitch)
        applied["waist_pitch_joint"] = float(args_cli.agile_command_hold_stand_waist_pitch)
    return applied


def _apply_hold_rescue_overrides(command: np.ndarray, joint_names: list[str]) -> dict[str, float]:
    applied: dict[str, float] = {}
    paired_overrides = {
        "hip_pitch": args_cli.agile_command_hold_rescue_hip_pitch,
        "knee": args_cli.agile_command_hold_rescue_knee,
        "ankle_pitch": args_cli.agile_command_hold_rescue_ankle_pitch,
        "hip_roll": args_cli.agile_command_hold_rescue_hip_roll,
        "ankle_roll": args_cli.agile_command_hold_rescue_ankle_roll,
    }
    for joint_suffix, value in paired_overrides.items():
        if value is None:
            continue
        for side in ("left", "right"):
            joint_name = f"{side}_{joint_suffix}_joint"
            if joint_name in joint_names:
                command[joint_names.index(joint_name)] = float(value)
                applied[joint_name] = float(value)
    if args_cli.agile_command_hold_rescue_waist_pitch is not None and "waist_pitch_joint" in joint_names:
        command[joint_names.index("waist_pitch_joint")] = float(args_cli.agile_command_hold_rescue_waist_pitch)
        applied["waist_pitch_joint"] = float(args_cli.agile_command_hold_rescue_waist_pitch)
    return applied


def _arm_pose_targets() -> dict[str, float]:
    targets: dict[str, float] = {}
    if str(args_cli.arm_pose_mode) == "right_front_reach":
        targets.update(
            {
                "right_shoulder_pitch_joint": -0.95,
                "right_shoulder_roll_joint": -0.35,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 1.10,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": -0.25,
                "right_wrist_yaw_joint": 0.0,
            }
        )
    elif str(args_cli.arm_pose_mode) == "both_front_reach":
        targets.update(
            {
                "left_shoulder_pitch_joint": -0.90,
                "left_shoulder_roll_joint": 0.30,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 1.05,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": -0.20,
                "left_wrist_yaw_joint": 0.0,
                "right_shoulder_pitch_joint": -0.90,
                "right_shoulder_roll_joint": -0.30,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 1.05,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": -0.20,
                "right_wrist_yaw_joint": 0.0,
            }
        )
    manual = {
        "right_shoulder_pitch_joint": args_cli.right_shoulder_pitch,
        "right_shoulder_roll_joint": args_cli.right_shoulder_roll,
        "right_shoulder_yaw_joint": args_cli.right_shoulder_yaw,
        "right_elbow_joint": args_cli.right_elbow,
        "right_wrist_roll_joint": args_cli.right_wrist_roll,
        "right_wrist_pitch_joint": args_cli.right_wrist_pitch,
        "right_wrist_yaw_joint": args_cli.right_wrist_yaw,
        "left_shoulder_pitch_joint": args_cli.left_shoulder_pitch,
        "left_shoulder_roll_joint": args_cli.left_shoulder_roll,
        "left_shoulder_yaw_joint": args_cli.left_shoulder_yaw,
        "left_elbow_joint": args_cli.left_elbow,
        "left_wrist_roll_joint": args_cli.left_wrist_roll,
        "left_wrist_pitch_joint": args_cli.left_wrist_pitch,
        "left_wrist_yaw_joint": args_cli.left_wrist_yaw,
    }
    for joint_name, value in manual.items():
        if value is not None:
            targets[joint_name] = float(value)
    return targets


def _apply_arm_pose_targets(
    command: np.ndarray,
    joint_names: list[str],
    step: int,
    targets: dict[str, float],
) -> bool:
    if not targets or int(step) < int(args_cli.arm_pose_start_step):
        return False
    ramp_steps = max(1, int(args_cli.arm_pose_ramp_steps))
    alpha = min(1.0, max(0.0, (int(step) - int(args_cli.arm_pose_start_step) + 1) / float(ramp_steps)))
    stand_targets = _stand_joint_targets()
    for joint_name, target in targets.items():
        base = stand_targets.get(joint_name, 0.0)
        _command_joint_position(command, joint_names, joint_name, float(base) + alpha * (float(target) - float(base)))
    return True


def _apply_box_retention_posture_feedback(
    command: np.ndarray,
    joint_names: list[str],
    box_tilt: float,
    box_robot_rel_error: float,
) -> float:
    if not bool(args_cli.box_retention_posture_controller):
        return 0.0
    risk = max(
        _ramp01(
            float(box_robot_rel_error),
            float(args_cli.box_retention_rel_start),
            float(args_cli.box_retention_rel_stop),
        ),
        _ramp01(
            float(box_tilt),
            float(args_cli.box_retention_tilt_start),
            float(args_cli.box_retention_tilt_stop),
        ),
    )
    if risk <= 0.0:
        return 0.0
    stand_targets = _stand_joint_targets()
    for side in ("left", "right"):
        _command_joint_position(
            command,
            joint_names,
            f"{side}_hip_pitch_joint",
            stand_targets.get(f"{side}_hip_pitch_joint", -0.10)
            + risk * float(args_cli.box_retention_hip_pitch_offset),
        )
        _command_joint_position(
            command,
            joint_names,
            f"{side}_knee_joint",
            stand_targets.get(f"{side}_knee_joint", 0.30)
            + risk * float(args_cli.box_retention_knee_offset),
        )
        _command_joint_position(
            command,
            joint_names,
            f"{side}_ankle_pitch_joint",
            stand_targets.get(f"{side}_ankle_pitch_joint", -0.20)
            + risk * float(args_cli.box_retention_ankle_pitch_offset),
        )
        shoulder_roll_bias = 0.02 if side == "left" else -0.02
        _command_joint_position(
            command,
            joint_names,
            f"{side}_shoulder_pitch_joint",
            stand_targets.get(f"{side}_shoulder_pitch_joint", 0.0)
            + risk * float(args_cli.box_retention_shoulder_pitch_offset),
        )
        _command_joint_position(
            command,
            joint_names,
            f"{side}_shoulder_roll_joint",
            stand_targets.get(f"{side}_shoulder_roll_joint", 0.0) + risk * shoulder_roll_bias,
        )
        _command_joint_position(
            command,
            joint_names,
            f"{side}_elbow_joint",
            stand_targets.get(f"{side}_elbow_joint", 0.0)
            + risk * float(args_cli.box_retention_elbow_offset),
        )
        _command_joint_position(
            command,
            joint_names,
            f"{side}_wrist_pitch_joint",
            stand_targets.get(f"{side}_wrist_pitch_joint", 0.0)
            + risk * float(args_cli.box_retention_wrist_pitch_offset),
        )
    _command_joint_position(
        command,
        joint_names,
        "waist_pitch_joint",
        stand_targets.get("waist_pitch_joint", 0.0)
        + risk * float(args_cli.box_retention_waist_pitch_offset),
    )
    return float(risk)


def _apply_symmetric_pitch_offsets(
    command: np.ndarray,
    joint_names: list[str],
    stand_targets: dict[str, float],
    hip_pitch_offset: float,
    knee_offset: float,
    ankle_pitch_offset: float,
    waist_pitch_offset: float,
) -> None:
    for side in ("left", "right"):
        _command_joint_position(
            command,
            joint_names,
            f"{side}_hip_pitch_joint",
            stand_targets.get(f"{side}_hip_pitch_joint", -0.10) + float(hip_pitch_offset),
        )
        _command_joint_position(
            command,
            joint_names,
            f"{side}_knee_joint",
            stand_targets.get(f"{side}_knee_joint", 0.30) + float(knee_offset),
        )
        _command_joint_position(
            command,
            joint_names,
            f"{side}_ankle_pitch_joint",
            stand_targets.get(f"{side}_ankle_pitch_joint", -0.20) + float(ankle_pitch_offset),
        )
    _command_joint_position(
        command,
        joint_names,
        "waist_pitch_joint",
        stand_targets.get("waist_pitch_joint", 0.0) + float(waist_pitch_offset),
    )


def _gait_joint_positions(
    base_positions: np.ndarray,
    joint_names: list[str],
    time_s: float,
    step: int,
    pitch: float,
    pitch_rate: float,
    robot_target_directed: float = 0.0,
    box_target_directed: float = 0.0,
    creep_pitch_brake_latched: bool = False,
    creep_reverse_brake_active: bool = False,
    terminal_hold_active: bool = False,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    command = np.array(base_positions, dtype=float, copy=True)
    diag: dict[str, float | bool] = {
        "creep_decel_active": False,
        "creep_pitch_brake_active": False,
        "creep_reverse_brake_active": False,
        "creep_amplitude_scale": 1.0,
        "creep_push_scale": 1.0,
        "creep_bias_scale": 1.0,
    }
    stand_targets = _stand_joint_targets()
    if terminal_hold_active:
        _apply_symmetric_pitch_offsets(
            command,
            joint_names,
            stand_targets,
            float(args_cli.terminal_hold_hip_pitch_offset),
            float(args_cli.terminal_hold_knee_offset),
            float(args_cli.terminal_hold_ankle_pitch_offset),
            float(args_cli.terminal_hold_waist_pitch_offset),
        )
        return command, diag
    if int(step) < int(args_cli.gait_start_step):
        return command, diag
    if int(args_cli.gait_stop_step) >= 0 and int(step) >= int(args_cli.gait_stop_step):
        return command, diag
    if str(args_cli.gait_mode) == "stand" or float(args_cli.gait_amplitude) <= 0.0:
        return command, diag
    if str(args_cli.gait_mode) == "targeted_creep":
        def _travel_scale(value: float, start: float, end: float, min_scale: float) -> float:
            if float(start) < 0.0 or float(end) <= float(start):
                return 1.0
            if float(value) <= float(start):
                return 1.0
            progress = min(1.0, max(0.0, (float(value) - float(start)) / float(end - start)))
            return 1.0 - progress * (1.0 - max(0.0, min(1.0, float(min_scale))))

        amp_scale = min(
            _travel_scale(
                float(box_target_directed),
                float(args_cli.creep_decel_box_travel_start),
                float(args_cli.creep_decel_box_travel_end),
                float(args_cli.creep_min_amplitude_scale),
            ),
            _travel_scale(
                float(robot_target_directed),
                float(args_cli.creep_decel_robot_travel_start),
                float(args_cli.creep_decel_robot_travel_end),
                float(args_cli.creep_min_amplitude_scale),
            ),
        )
        push_scale = min(
            _travel_scale(
                float(box_target_directed),
                float(args_cli.creep_decel_box_travel_start),
                float(args_cli.creep_decel_box_travel_end),
                float(args_cli.creep_min_push_scale),
            ),
            _travel_scale(
                float(robot_target_directed),
                float(args_cli.creep_decel_robot_travel_start),
                float(args_cli.creep_decel_robot_travel_end),
                float(args_cli.creep_min_push_scale),
            ),
        )
        bias_scale = min(
            _travel_scale(
                float(box_target_directed),
                float(args_cli.creep_decel_box_travel_start),
                float(args_cli.creep_decel_box_travel_end),
                float(args_cli.creep_min_bias_scale),
            ),
            _travel_scale(
                float(robot_target_directed),
                float(args_cli.creep_decel_robot_travel_start),
                float(args_cli.creep_decel_robot_travel_end),
                float(args_cli.creep_min_bias_scale),
            ),
        )
        pitch_brake_value = float(pitch) if bool(args_cli.creep_pitch_brake_positive_only) else abs(float(pitch))
        if (
            bool(creep_pitch_brake_latched)
            or float(pitch_brake_value) >= float(args_cli.creep_pitch_brake_threshold)
            or abs(float(pitch_rate)) >= float(args_cli.creep_pitch_brake_rate_threshold)
        ):
            amp_scale = min(amp_scale, max(0.0, min(1.0, float(args_cli.creep_pitch_brake_amplitude_scale))))
            push_scale = min(push_scale, max(0.0, min(1.0, float(args_cli.creep_pitch_brake_push_scale))))
            bias_scale = min(bias_scale, max(0.0, min(1.0, float(args_cli.creep_pitch_brake_bias_scale))))
            diag["creep_pitch_brake_active"] = True
        if bool(creep_reverse_brake_active):
            amp_scale = float(args_cli.creep_reverse_brake_amplitude_scale)
            push_scale = float(args_cli.creep_reverse_brake_stance_push_scale) / max(
                1e-6,
                abs(float(args_cli.creep_stance_push_scale)),
            )
            bias_scale = 1.0
            diag["creep_reverse_brake_active"] = True
        diag["creep_decel_active"] = bool(amp_scale < 0.999 or push_scale < 0.999 or bias_scale < 0.999)
        diag["creep_amplitude_scale"] = float(amp_scale)
        diag["creep_push_scale"] = float(push_scale)
        diag["creep_bias_scale"] = float(bias_scale)
        phase = 2.0 * math.pi * float(args_cli.gait_frequency_hz) * float(time_s)
        swing = math.sin(phase)
        left = float(args_cli.gait_amplitude) * float(amp_scale) * swing
        right = -left
        left_lift = max(0.0, left)
        right_lift = max(0.0, right)
        left_push = max(0.0, -left)
        right_push = max(0.0, -right)
        if bool(creep_reverse_brake_active):
            hip_bias = float(args_cli.creep_reverse_brake_hip_pitch_offset)
            knee_bias = float(args_cli.creep_reverse_brake_knee_offset)
            ankle_bias = float(args_cli.creep_reverse_brake_ankle_pitch_offset)
            waist_bias = float(args_cli.creep_reverse_brake_waist_pitch_offset)
            stance_push = float(args_cli.creep_reverse_brake_stance_push_scale)
            lift_scale = float(args_cli.creep_reverse_brake_lift_scale) * abs(float(amp_scale))
        else:
            hip_bias = float(args_cli.creep_hip_pitch_offset) * float(bias_scale)
            knee_bias = float(args_cli.creep_knee_offset) * float(bias_scale)
            ankle_bias = float(args_cli.creep_ankle_pitch_offset) * float(bias_scale)
            waist_bias = float(args_cli.creep_waist_pitch_offset) * float(bias_scale)
            stance_push = float(args_cli.creep_stance_push_scale) * float(push_scale)
            lift_scale = float(args_cli.creep_lift_scale) * float(amp_scale)
        ankle_lift_scale = float(args_cli.creep_ankle_lift_scale)
        _command_joint_position(
            command,
            joint_names,
            "left_hip_pitch_joint",
            stand_targets.get("left_hip_pitch_joint", -0.10) + hip_bias + stance_push * left_push - 0.20 * left_lift,
        )
        _command_joint_position(
            command,
            joint_names,
            "right_hip_pitch_joint",
            stand_targets.get("right_hip_pitch_joint", -0.10) + hip_bias + stance_push * right_push - 0.20 * right_lift,
        )
        _command_joint_position(
            command,
            joint_names,
            "left_knee_joint",
            stand_targets.get("left_knee_joint", 0.30) + knee_bias + lift_scale * left_lift,
        )
        _command_joint_position(
            command,
            joint_names,
            "right_knee_joint",
            stand_targets.get("right_knee_joint", 0.30) + knee_bias + lift_scale * right_lift,
        )
        _command_joint_position(
            command,
            joint_names,
            "left_ankle_pitch_joint",
            stand_targets.get("left_ankle_pitch_joint", -0.20) + ankle_bias + ankle_lift_scale * left_lift,
        )
        _command_joint_position(
            command,
            joint_names,
            "right_ankle_pitch_joint",
            stand_targets.get("right_ankle_pitch_joint", -0.20) + ankle_bias + ankle_lift_scale * right_lift,
        )
        _command_joint_position(
            command,
            joint_names,
            "waist_pitch_joint",
            stand_targets.get("waist_pitch_joint", 0.0) + waist_bias,
        )
        return command, diag
    amplitude_scale = 1.0
    if str(args_cli.gait_mode) == "staged_march":
        ramp_start = int(args_cli.gait_ramp_down_start_step)
        ramp_end = int(args_cli.gait_ramp_down_end_step)
        min_scale = max(0.0, min(1.0, float(args_cli.gait_min_amplitude_scale)))
        if ramp_start >= 0 and ramp_end > ramp_start and int(step) >= ramp_start:
            progress = min(1.0, max(0.0, (float(step) - float(ramp_start)) / float(ramp_end - ramp_start)))
            amplitude_scale = 1.0 - progress * (1.0 - min_scale)
        elif ramp_start >= 0 and ramp_end >= 0 and int(step) >= ramp_end:
            amplitude_scale = min_scale
        recovery_active = (
            abs(float(pitch)) >= float(args_cli.recovery_pitch_threshold)
            or abs(float(pitch_rate)) >= float(args_cli.recovery_pitch_rate_threshold)
        )
        if recovery_active:
            amplitude_scale = min(amplitude_scale, max(0.0, min(1.0, float(args_cli.gait_min_amplitude_scale))))
    else:
        recovery_active = False
    phase = 2.0 * math.pi * float(args_cli.gait_frequency_hz) * float(time_s)
    swing = math.sin(phase)
    left = float(args_cli.gait_amplitude) * amplitude_scale * swing
    right = -left
    left_lift = max(0.0, left)
    right_lift = max(0.0, right)
    _command_joint_position(command, joint_names, "left_hip_pitch_joint", stand_targets.get("left_hip_pitch_joint", -0.10) + 0.35 * left)
    _command_joint_position(command, joint_names, "right_hip_pitch_joint", stand_targets.get("right_hip_pitch_joint", -0.10) + 0.35 * right)
    _command_joint_position(command, joint_names, "left_knee_joint", stand_targets.get("left_knee_joint", 0.30) + 0.45 * left_lift)
    _command_joint_position(command, joint_names, "right_knee_joint", stand_targets.get("right_knee_joint", 0.30) + 0.45 * right_lift)
    _command_joint_position(command, joint_names, "left_ankle_pitch_joint", stand_targets.get("left_ankle_pitch_joint", -0.20) - 0.25 * left_lift)
    _command_joint_position(command, joint_names, "right_ankle_pitch_joint", stand_targets.get("right_ankle_pitch_joint", -0.20) - 0.25 * right_lift)
    _command_joint_position(command, joint_names, "left_shoulder_pitch_joint", stand_targets.get("left_shoulder_pitch_joint", 0.0) - 0.15 * right)
    _command_joint_position(command, joint_names, "right_shoulder_pitch_joint", stand_targets.get("right_shoulder_pitch_joint", 0.0) - 0.15 * left)
    if recovery_active:
        for side in ("left", "right"):
            if f"{side}_hip_pitch_joint" in joint_names:
                command[joint_names.index(f"{side}_hip_pitch_joint")] += float(args_cli.recovery_hip_pitch_offset)
            if f"{side}_knee_joint" in joint_names:
                command[joint_names.index(f"{side}_knee_joint")] += float(args_cli.recovery_knee_offset)
            if f"{side}_ankle_pitch_joint" in joint_names:
                command[joint_names.index(f"{side}_ankle_pitch_joint")] += float(args_cli.recovery_ankle_pitch_offset)
        if "waist_pitch_joint" in joint_names:
            command[joint_names.index("waist_pitch_joint")] = (
                stand_targets.get("waist_pitch_joint", 0.0) + float(args_cli.recovery_waist_pitch_offset)
            )
    return command, diag


def _apply_balance_feedback(
    command: np.ndarray,
    joint_names: list[str],
    step: int,
    roll: float,
    pitch: float,
    roll_rate: float,
    pitch_rate: float,
    pitch_target_override: float | None = None,
    roll_target_override: float | None = None,
) -> tuple[np.ndarray, bool]:
    if not bool(args_cli.balance_feedback_controller):
        return command, False
    if int(step) < int(args_cli.balance_start_step):
        return command, False
    target_active = _balance_target_active(step)
    pitch_target = (
        float(pitch_target_override)
        if pitch_target_override is not None
        else (float(args_cli.balance_pitch_target) if target_active else 0.0)
    )
    roll_target = (
        float(roll_target_override)
        if roll_target_override is not None
        else (float(args_cli.balance_roll_target) if target_active else 0.0)
    )
    pitch_error = float(pitch) - pitch_target
    roll_error = float(roll) - roll_target
    active = (
        abs(float(pitch_error)) >= float(args_cli.balance_pitch_activation_threshold)
        or abs(float(roll_error)) >= float(args_cli.balance_roll_activation_threshold)
        or abs(float(pitch_rate)) >= float(args_cli.balance_pitch_rate_activation_threshold)
        or abs(float(roll_rate)) >= float(args_cli.balance_roll_rate_activation_threshold)
    )
    if not active:
        return command, False
    adjusted = np.array(command, dtype=float, copy=True)
    stand_targets = _stand_joint_targets()
    limit = abs(float(args_cli.balance_adjustment_limit))
    feedback_base = str(args_cli.balance_feedback_base)

    def base_joint(joint_name: str, fallback: float) -> float:
        if feedback_base == "command" and joint_name in joint_names:
            return float(command[joint_names.index(joint_name)])
        return float(stand_targets.get(joint_name, fallback))

    pitch_adjust = max(
        -limit,
        min(
            limit,
            float(args_cli.balance_pitch_sign)
            * (
                float(args_cli.balance_pitch_gain) * float(pitch_error)
                + float(args_cli.balance_pitch_rate_gain) * float(pitch_rate)
            ),
        ),
    )
    roll_adjust = max(
        -limit,
        min(
            limit,
            float(args_cli.balance_roll_sign)
            * (
                float(args_cli.balance_roll_gain) * float(roll_error)
                + float(args_cli.balance_roll_rate_gain) * float(roll_rate)
            ),
        ),
    )
    for side in ("left", "right"):
        _command_joint_position(
            adjusted,
            joint_names,
            f"{side}_ankle_pitch_joint",
            base_joint(f"{side}_ankle_pitch_joint", -0.20) + pitch_adjust,
        )
        _command_joint_position(
            adjusted,
            joint_names,
            f"{side}_hip_pitch_joint",
            base_joint(f"{side}_hip_pitch_joint", -0.10) - 0.5 * pitch_adjust,
        )
    _command_joint_position(
        adjusted,
        joint_names,
        "left_ankle_roll_joint",
        base_joint("left_ankle_roll_joint", 0.0)
        + float(args_cli.balance_roll_left_ankle_scale) * roll_adjust,
    )
    _command_joint_position(
        adjusted,
        joint_names,
        "right_ankle_roll_joint",
        base_joint("right_ankle_roll_joint", 0.0)
        + float(args_cli.balance_roll_right_ankle_scale) * roll_adjust,
    )
    _command_joint_position(
        adjusted,
        joint_names,
        "left_hip_roll_joint",
        base_joint("left_hip_roll_joint", 0.0)
        + float(args_cli.balance_roll_left_hip_scale) * roll_adjust,
    )
    _command_joint_position(
        adjusted,
        joint_names,
        "right_hip_roll_joint",
        base_joint("right_hip_roll_joint", 0.0)
        + float(args_cli.balance_roll_right_hip_scale) * roll_adjust,
    )
    return adjusted, True


def _balance_target_active(step: int) -> bool:
    active = int(step) >= int(args_cli.balance_target_start_step) and (
        int(args_cli.balance_target_end_step) < 0 or int(step) < int(args_cli.balance_target_end_step)
    )
    period = int(args_cli.balance_target_pulse_period_steps)
    width = int(args_cli.balance_target_pulse_width_steps)
    if not active or period <= 0 or width <= 0:
        return active
    phase_step = int(step) - int(args_cli.balance_target_pulse_phase_step)
    if phase_step < 0:
        return False
    return int(phase_step % period) < min(width, period)


def design_scene(stage: Usd.Stage) -> tuple[str | None, dict[str, str], str | None, dict[str, dict[str, float]]]:
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = _define_material(stage, "/World/Looks/CarrySceneFriction", 1.2, 0.9)
    _spawn_box(stage, "/World/Ground", (5.0, 4.0, 0.05), 1.0, (0.31, 0.33, 0.33), (0.0, 0.0, -0.025), rigid=False, material=material)
    _spawn_box(stage, "/World/CarryTarget", (0.50, 0.35, 0.02), 1.0, (0.10, 0.42, 0.82), (TARGET_XY[0], TARGET_XY[1], 0.01), rigid=False)

    g1_prim = stage.DefinePrim(G1_PATH, "Xform")
    g1_prim.GetReferences().AddReference(str(args_cli.g1_usd))
    UsdGeom.XformCommonAPI(g1_prim).SetTranslate(Gf.Vec3d(*[float(v) for v in args_cli.g1_root_position]))
    root_euler_deg = _quat_wxyz_to_euler_deg(
        float(args_cli.g1_root_orientation_wxyz[0]),
        float(args_cli.g1_root_orientation_wxyz[1]),
        float(args_cli.g1_root_orientation_wxyz[2]),
        float(args_cli.g1_root_orientation_wxyz[3]),
    )
    UsdGeom.XformCommonAPI(g1_prim).SetRotate(Gf.Vec3f(*[float(v) for v in root_euler_deg]))
    pelvis_prim = stage.GetPrimAtPath(G1_ARTICULATION_PATH)
    if pelvis_prim.IsValid() and not bool(args_cli.disable_usd_pelvis_xform):
        UsdGeom.XformCommonAPI(pelvis_prim).SetTranslate(Gf.Vec3d(*[float(v) for v in args_cli.g1_root_position]))
        UsdGeom.XformCommonAPI(pelvis_prim).SetRotate(Gf.Vec3f(*[float(v) for v in root_euler_deg]))

    if str(args_cli.box_support_mode) == "table" and not bool(args_cli.disable_carry_box_spawn):
        box_pos = tuple(float(v) for v in args_cli.box_position)
        box_size = tuple(float(v) for v in args_cli.box_size)
        support_size = tuple(float(v) for v in args_cli.box_support_size)
        support_top_z = float(box_pos[2]) - 0.5 * float(box_size[2]) - float(args_cli.box_support_top_clearance)
        support_center = (
            float(box_pos[0]),
            float(box_pos[1]),
            float(support_top_z) - 0.5 * float(support_size[2]),
        )
        _spawn_box(
            stage,
            BOX_SUPPORT_TABLE_PATH,
            support_size,
            1.0,
            (0.25, 0.27, 0.30),
            support_center,
            rigid=False,
            collision=True,
            material=material,
        )
    if not bool(args_cli.disable_carry_box_spawn):
        _spawn_box(
            stage,
            BOX_PATH,
            tuple(float(v) for v in args_cli.box_size),
            float(args_cli.box_mass),
            (0.58, 0.43, 0.24),
            tuple(float(v) for v in args_cli.box_position),
            collision=not bool(args_cli.disable_box_collision),
            material=material,
        )
    payload_joint_path = None
    cradle_piece_joints: dict[str, str] = {}
    if str(args_cli.attach_box) == "fixed_torso":
        if bool(args_cli.disable_carry_box_spawn):
            raise RuntimeError("attach_box=fixed_torso requires carry-box spawn to be enabled")
        if not stage.GetPrimAtPath(args_cli.attach_body_path).IsValid():
            raise RuntimeError(f"Attach body path not found after G1 reference: {args_cli.attach_body_path}")
        payload_joint_path = f"{BOX_PATH}/FixedJointToG1"
        _fixed_joint(stage, payload_joint_path, str(args_cli.attach_body_path), BOX_PATH, tuple(float(v) for v in args_cli.attach_local_pos0), (0.0, 0.0, 0.0))
    if str(args_cli.torso_cradle) == "front_tray":
        cradle_piece_joints = _spawn_front_torso_cradle(stage, material)
    probe_joint_path = None
    if str(args_cli.probe_mode) == "front_bumper":
        probe_joint_path = _spawn_front_probe_bumper(stage, material)
    applied_drive_gains = _apply_stand_drive_gains(stage) if bool(args_cli.apply_arena_stand_gains) else {}
    return payload_joint_path, cradle_piece_joints, probe_joint_path, applied_drive_gains


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args_cli.output_dir / "core_world_g1_box_scene_summary.json"
    csv_path = args_cli.output_dir / "core_world_g1_box_scene_state.csv"
    replay_csv_path = args_cli.output_dir / "core_world_g1_box_scene_replay.csv"
    bootstrap_summary = {
        "scene_type": "core_world_g1_box_scene",
        "success_claim": "g1_usd_core_api_scene_diagnostic_not_walking_or_carrying_success",
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "device": args_cli.device,
        "record_replay_csv": bool(args_cli.record_replay_csv),
        "record_replay_csv_path": str(replay_csv_path),
        "record_replay_every_n_steps": int(args_cli.record_replay_every_n_steps),
        "stand_force_scale": float(args_cli.stand_force_scale),
        "gait_mode": str(args_cli.gait_mode),
        "gait_amplitude": float(args_cli.gait_amplitude),
        "gait_frequency_hz": float(args_cli.gait_frequency_hz),
        "creep_hip_pitch_offset": float(args_cli.creep_hip_pitch_offset),
        "creep_knee_offset": float(args_cli.creep_knee_offset),
        "creep_ankle_pitch_offset": float(args_cli.creep_ankle_pitch_offset),
        "creep_waist_pitch_offset": float(args_cli.creep_waist_pitch_offset),
        "creep_stance_push_scale": float(args_cli.creep_stance_push_scale),
        "creep_lift_scale": float(args_cli.creep_lift_scale),
        "creep_ankle_lift_scale": float(args_cli.creep_ankle_lift_scale),
        "creep_decel_box_travel_start_m": float(args_cli.creep_decel_box_travel_start),
        "creep_decel_box_travel_end_m": float(args_cli.creep_decel_box_travel_end),
        "creep_decel_robot_travel_start_m": float(args_cli.creep_decel_robot_travel_start),
        "creep_decel_robot_travel_end_m": float(args_cli.creep_decel_robot_travel_end),
        "creep_min_amplitude_scale": float(args_cli.creep_min_amplitude_scale),
        "creep_min_push_scale": float(args_cli.creep_min_push_scale),
        "creep_min_bias_scale": float(args_cli.creep_min_bias_scale),
        "creep_pitch_brake_threshold_rad": float(args_cli.creep_pitch_brake_threshold),
        "creep_pitch_brake_rate_threshold_radps": float(args_cli.creep_pitch_brake_rate_threshold),
        "creep_pitch_brake_amplitude_scale": float(args_cli.creep_pitch_brake_amplitude_scale),
        "creep_pitch_brake_push_scale": float(args_cli.creep_pitch_brake_push_scale),
        "creep_pitch_brake_bias_scale": float(args_cli.creep_pitch_brake_bias_scale),
        "creep_pitch_brake_latch_enabled": bool(args_cli.creep_pitch_brake_latch),
        "creep_pitch_brake_positive_only": bool(args_cli.creep_pitch_brake_positive_only),
        "creep_pitch_brake_latched_step": None,
        "creep_reverse_brake_box_travel_m": float(args_cli.creep_reverse_brake_box_travel),
        "creep_reverse_brake_robot_travel_m": float(args_cli.creep_reverse_brake_robot_travel),
        "creep_reverse_brake_pitch_threshold_rad": float(args_cli.creep_reverse_brake_pitch_threshold),
        "creep_reverse_brake_positive_pitch_only": bool(args_cli.creep_reverse_brake_positive_pitch_only),
        "creep_reverse_brake_duration_steps": int(args_cli.creep_reverse_brake_duration_steps),
        "creep_reverse_brake_amplitude_scale": float(args_cli.creep_reverse_brake_amplitude_scale),
        "creep_reverse_brake_stance_push_scale": float(args_cli.creep_reverse_brake_stance_push_scale),
        "creep_reverse_brake_lift_scale": float(args_cli.creep_reverse_brake_lift_scale),
        "creep_reverse_brake_hip_pitch_offset": float(args_cli.creep_reverse_brake_hip_pitch_offset),
        "creep_reverse_brake_knee_offset": float(args_cli.creep_reverse_brake_knee_offset),
        "creep_reverse_brake_ankle_pitch_offset": float(args_cli.creep_reverse_brake_ankle_pitch_offset),
        "creep_reverse_brake_waist_pitch_offset": float(args_cli.creep_reverse_brake_waist_pitch_offset),
        "creep_reverse_brake_latched_step": None,
        "creep_reverse_brake_first_reason": None,
        "creep_reverse_brake_active_steps": 0,
        "creep_reverse_brake_first_active_step": None,
        "creep_decel_active_steps": 0,
        "creep_decel_first_active_step": None,
        "creep_pitch_brake_active_steps": 0,
        "creep_pitch_brake_first_active_step": None,
        "min_creep_amplitude_scale": 1.0,
        "min_creep_push_scale": 1.0,
        "min_creep_bias_scale": 1.0,
        "gait_ramp_down_start_step": int(args_cli.gait_ramp_down_start_step),
        "gait_ramp_down_end_step": int(args_cli.gait_ramp_down_end_step),
        "gait_min_amplitude_scale": float(args_cli.gait_min_amplitude_scale),
        "recovery_pitch_threshold": float(args_cli.recovery_pitch_threshold),
        "recovery_pitch_rate_threshold": float(args_cli.recovery_pitch_rate_threshold),
        "recovery_hip_pitch_offset": float(args_cli.recovery_hip_pitch_offset),
        "recovery_knee_offset": float(args_cli.recovery_knee_offset),
        "recovery_ankle_pitch_offset": float(args_cli.recovery_ankle_pitch_offset),
        "recovery_waist_pitch_offset": float(args_cli.recovery_waist_pitch_offset),
        "terminal_hold_start_step": int(args_cli.terminal_hold_start_step),
        "terminal_hold_box_target_travel": float(args_cli.terminal_hold_box_target_travel),
        "terminal_hold_robot_target_travel": float(args_cli.terminal_hold_robot_target_travel),
        "terminal_hold_pitch_threshold": float(args_cli.terminal_hold_pitch_threshold),
        "terminal_hold_pitch_rate_threshold": float(args_cli.terminal_hold_pitch_rate_threshold),
        "terminal_hold_hip_pitch_offset": float(args_cli.terminal_hold_hip_pitch_offset),
        "terminal_hold_knee_offset": float(args_cli.terminal_hold_knee_offset),
        "terminal_hold_ankle_pitch_offset": float(args_cli.terminal_hold_ankle_pitch_offset),
        "terminal_hold_waist_pitch_offset": float(args_cli.terminal_hold_waist_pitch_offset),
        "terminal_drive_gain_scale": float(args_cli.terminal_drive_gain_scale),
        "terminal_drive_force_scale": float(args_cli.terminal_drive_force_scale),
        "terminal_hold_active_steps": 0,
        "terminal_hold_first_active_step": None,
        "terminal_hold_first_reason": None,
        "agile_policy_backend": str(args_cli.agile_policy_backend),
        "attach_box": str(args_cli.attach_box),
        "torso_cradle": str(args_cli.torso_cradle),
        "probe_mode": str(args_cli.probe_mode),
        "probe_start_step": int(args_cli.probe_start_step),
        "probe_end_step": int(args_cli.probe_end_step),
        "probe_collision_window_enabled": bool(args_cli.probe_collision_window),
        "probe_pad_size_m": [float(v) for v in args_cli.probe_pad_size],
        "probe_pad_local_pos0_m": [float(v) for v in args_cli.probe_pad_local_pos0],
        "probe_pad_mass_kg": float(args_cli.probe_pad_mass),
        "probe_pad_collision_enabled": (
            not bool(args_cli.disable_probe_pad_collision)
            and not bool(args_cli.probe_collision_window)
        ),
        "grasp_mode": str(args_cli.grasp_mode),
        "grasp_body_path": str(args_cli.grasp_body_path),
        "grasp_enable_step": int(args_cli.grasp_enable_step),
        "grasp_lift_offset_z_m": float(args_cli.grasp_lift_offset_z),
        "require_box_no_drop": bool(args_cli.require_box_no_drop),
        "carry_box_spawned": not bool(args_cli.disable_carry_box_spawn),
        "box_support_mode": str(args_cli.box_support_mode),
        "box_support_size_m": [float(v) for v in args_cli.box_support_size],
        "box_support_top_clearance_m": float(args_cli.box_support_top_clearance),
        "box_support_release_step": int(args_cli.box_support_release_step),
        "balance_feedback_controller_enabled": bool(args_cli.balance_feedback_controller),
        "balance_pitch_gain": float(args_cli.balance_pitch_gain),
        "balance_roll_gain": float(args_cli.balance_roll_gain),
        "balance_pitch_rate_gain": float(args_cli.balance_pitch_rate_gain),
        "balance_roll_rate_gain": float(args_cli.balance_roll_rate_gain),
        "balance_adjustment_limit": float(args_cli.balance_adjustment_limit),
        "balance_feedback_base": str(args_cli.balance_feedback_base),
        "balance_start_on_agile_hold": bool(args_cli.balance_start_on_agile_hold),
        "balance_roll_left_ankle_scale": float(args_cli.balance_roll_left_ankle_scale),
        "balance_roll_right_ankle_scale": float(args_cli.balance_roll_right_ankle_scale),
        "balance_roll_left_hip_scale": float(args_cli.balance_roll_left_hip_scale),
        "balance_roll_right_hip_scale": float(args_cli.balance_roll_right_hip_scale),
        "balance_pitch_target": float(args_cli.balance_pitch_target),
        "balance_roll_target": float(args_cli.balance_roll_target),
        "balance_roll_target_from_lateral": bool(args_cli.balance_roll_target_from_lateral),
        "balance_roll_target_lateral_source": str(args_cli.balance_roll_target_lateral_source),
        "balance_roll_target_lateral_gain": float(args_cli.balance_roll_target_lateral_gain),
        "balance_roll_target_lateral_limit": float(args_cli.balance_roll_target_lateral_limit),
        "balance_roll_target_lateral_deadband": float(args_cli.balance_roll_target_lateral_deadband),
        "balance_roll_target_lateral_sign": float(args_cli.balance_roll_target_lateral_sign),
        "balance_roll_target_lateral_start_after_hold_steps": int(
            args_cli.balance_roll_target_lateral_start_after_hold_steps
        ),
        "balance_roll_target_lateral_ramp_steps": int(args_cli.balance_roll_target_lateral_ramp_steps),
        "balance_roll_target_lateral_max_tilt": float(args_cli.balance_roll_target_lateral_max_tilt),
        "balance_roll_target_lateral_max_box_tilt": float(
            args_cli.balance_roll_target_lateral_max_box_tilt
        ),
        "balance_target_start_step": int(args_cli.balance_target_start_step),
        "balance_target_end_step": int(args_cli.balance_target_end_step),
        "balance_target_pulse_period_steps": int(args_cli.balance_target_pulse_period_steps),
        "balance_target_pulse_width_steps": int(args_cli.balance_target_pulse_width_steps),
        "balance_target_pulse_phase_step": int(args_cli.balance_target_pulse_phase_step),
        "balance_pitch_sign": float(args_cli.balance_pitch_sign),
        "balance_roll_sign": float(args_cli.balance_roll_sign),
        "balance_start_step": int(args_cli.balance_start_step),
        "balance_pitch_activation_threshold": float(args_cli.balance_pitch_activation_threshold),
        "balance_roll_activation_threshold": float(args_cli.balance_roll_activation_threshold),
        "balance_pitch_rate_activation_threshold": float(args_cli.balance_pitch_rate_activation_threshold),
        "balance_roll_rate_activation_threshold": float(args_cli.balance_roll_rate_activation_threshold),
        "diagnostic_root_drive": str(args_cli.diagnostic_root_drive),
        "diagnostic_root_drive_start_step": int(args_cli.diagnostic_root_drive_start_step),
        "diagnostic_root_drive_stop_step": int(args_cli.diagnostic_root_drive_stop_step),
        "diagnostic_root_drive_speed_mps": float(args_cli.diagnostic_root_drive_speed),
        "diagnostic_root_drive_ramp_steps": int(args_cli.diagnostic_root_drive_ramp_steps),
        "arm_pose_mode": str(args_cli.arm_pose_mode),
        "arm_pose_start_step": int(args_cli.arm_pose_start_step),
        "arm_pose_ramp_steps": int(args_cli.arm_pose_ramp_steps),
        "g1_root_position_requested_m": [float(v) for v in args_cli.g1_root_position],
        "g1_root_orientation_requested_wxyz": [float(v) for v in args_cli.g1_root_orientation_wxyz],
        "disable_usd_pelvis_xform": bool(args_cli.disable_usd_pelvis_xform),
        "init_stage": "argument_validation",
        "status": "fail",
        "error": None,
        "failures": [],
    }

    def _write_bootstrap_failure(stage_name: str, exc: BaseException) -> Path:
        bootstrap_summary["init_stage"] = stage_name
        bootstrap_summary["error"] = f"{type(exc).__name__}: {exc}"
        bootstrap_summary["failures"] = [str(bootstrap_summary["error"])]
        summary_path.write_text(json.dumps(bootstrap_summary, indent=2, sort_keys=True) + "\n")
        print(f"[ERROR] {bootstrap_summary['error']}", flush=True)
        print(f"[INFO] Summary written to: {summary_path}", flush=True)
        return summary_path

    if not args_cli.g1_usd.is_file():
        return _write_bootstrap_failure("argument_validation", FileNotFoundError(f"G1 USD not found: {args_cli.g1_usd}"))
    if str(args_cli.gait_mode) == "agile_policy":
        if not args_cli.agile_config.is_file():
            return _write_bootstrap_failure("argument_validation", FileNotFoundError(f"AGILE config not found: {args_cli.agile_config}"))
        if str(args_cli.agile_policy_backend) == "onnx" and not args_cli.agile_onnx.is_file():
            return _write_bootstrap_failure("argument_validation", FileNotFoundError(f"AGILE ONNX not found: {args_cli.agile_onnx}"))
        if str(args_cli.agile_policy_backend) == "torch_checkpoint" and not args_cli.agile_torch_checkpoint.is_file():
            return _write_bootstrap_failure(
                "argument_validation",
                FileNotFoundError(f"AGILE torch checkpoint not found: {args_cli.agile_torch_checkpoint}"),
            )

    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    try:
        bootstrap_summary["init_stage"] = "create_stage"
        create_new_stage()
        stage = get_current_stage()
        bootstrap_summary["init_stage"] = "design_scene"
        payload_joint_path, cradle_piece_joints, probe_joint_path, applied_drive_gains = design_scene(stage)
        usd_g1_pose_after_design = _usd_world_pose_wxyz(stage, G1_ARTICULATION_PATH)
        bootstrap_summary["init_stage"] = "create_world"
        world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
        robot = SingleArticulation(prim_path=G1_ARTICULATION_PATH, name="g1")
        box = None
        if not bool(args_cli.disable_carry_box_spawn):
            box = SingleRigidPrim(prim_path=BOX_PATH, name="carry_box")
        active_grasp_body_path = None
        grasp_body = None
        grasp_body_wrapper_initialized = False
        grasp_body_wrapper_error = None
        if str(args_cli.grasp_mode) == "staged_fixed_torso":
            active_grasp_body_path = str(args_cli.attach_body_path)
        elif str(args_cli.grasp_mode) == "staged_fixed_body":
            active_grasp_body_path = str(args_cli.grasp_body_path)
        if active_grasp_body_path is not None:
            if not stage.GetPrimAtPath(active_grasp_body_path).IsValid():
                raise RuntimeError(f"grasp body path not found after G1 reference: {active_grasp_body_path}")
            grasp_body = SingleRigidPrim(prim_path=active_grasp_body_path, name="grasp_body")
        print("[PROGRESS] Core World and prim wrappers created", flush=True)
        bootstrap_summary["init_stage"] = "world_reset"
        world.reset()
        print("[PROGRESS] Core World reset completed", flush=True)
        bootstrap_summary["init_stage"] = "wrapper_initialize"
        robot.initialize()
        if box is not None:
            box.initialize()
            print("[PROGRESS] G1 and box wrappers initialized", flush=True)
        else:
            print("[PROGRESS] G1 wrapper initialized; carry box disabled for stand diagnostic", flush=True)
        if grasp_body is not None:
            try:
                grasp_body.initialize()
                grasp_body_wrapper_initialized = True
                print(f"[PROGRESS] Grasp body wrapper initialized: {active_grasp_body_path}", flush=True)
            except Exception as exc:
                grasp_body_wrapper_error = f"{type(exc).__name__}: {exc}"
                grasp_body = None
                print(
                    "[WARN] grasp body wrapper initialization failed; "
                    f"will use USD world pose fallback for {active_grasp_body_path}: {grasp_body_wrapper_error}",
                    flush=True,
                )
    except BaseException as exc:
        return _write_bootstrap_failure(str(bootstrap_summary.get("init_stage", "bootstrap")), exc)

    joint_positions = np.array(robot.get_joint_positions(), dtype=float)
    joint_velocities = np.zeros_like(joint_positions)
    joint_names = list(getattr(robot, "dof_names", []))
    arm_pose_targets = _arm_pose_targets()
    print(f"[PROGRESS] G1 articulation initialized with {len(joint_names)} joints", flush=True)
    agile_policy = None
    if str(args_cli.gait_mode) == "agile_policy":
        if str(args_cli.agile_policy_backend) == "onnx":
            print("[PROGRESS] Loading AGILE ONNX policy", flush=True)
            agile_policy = AgileOnnxJointPolicy(args_cli.agile_config, args_cli.agile_onnx)
        else:
            print("[PROGRESS] Loading AGILE torch checkpoint policy", flush=True)
            agile_policy = AgileTorchCheckpointJointPolicy(
                args_cli.agile_config,
                Path("/public/home/yanhongru/Curiosity/external/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml"),
                args_cli.agile_torch_checkpoint,
            )
    policy_joint_targets = np.array(joint_positions, dtype=float, copy=True)
    policy_inference_count = 0
    applied_stand_joint_targets: dict[str, float] = {}
    if not bool(args_cli.disable_stand_joint_targets):
        for joint_name, target in _stand_joint_targets().items():
            if joint_name in joint_names:
                joint_positions[joint_names.index(joint_name)] = float(target)
                applied_stand_joint_targets[joint_name] = float(target)
    stand_hold_joint_targets = np.array(joint_positions, dtype=float, copy=True)
    applied_hold_stand_joint_targets = _apply_hold_stand_overrides(stand_hold_joint_targets, joint_names)
    hold_rescue_joint_targets = np.array(stand_hold_joint_targets, dtype=float, copy=True)
    applied_hold_rescue_joint_targets = _apply_hold_rescue_overrides(hold_rescue_joint_targets, joint_names)
    root_pose_write_count_setup = 0
    joint_state_write_count_setup = 0
    joint_state_write_error = None
    if not bool(args_cli.disable_setup_root_pose):
        robot.set_world_pose(
            position=np.array([float(v) for v in args_cli.g1_root_position], dtype=float),
            orientation=np.array([float(v) for v in args_cli.g1_root_orientation_wxyz], dtype=float),
        )
        root_pose_write_count_setup += 1
        if joint_positions.size and not bool(args_cli.disable_setup_joint_state_write):
            try:
                robot.set_joint_positions(joint_positions.tolist())
                robot.set_joint_velocities(np.zeros_like(joint_positions).tolist())
                joint_state_write_count_setup += 1
            except Exception as exc:
                joint_state_write_error = f"{type(exc).__name__}: {exc}"
                print(f"[WARN] setup joint state write failed: {joint_state_write_error}", flush=True)
        if joint_positions.size:
            robot.apply_action(ArticulationAction(joint_positions=joint_positions.tolist()))
        world.step(render=False)
    initial_robot = _pose_wxyz(robot)
    initial_box = _pose_wxyz(box) if box is not None else None
    robot_target_direction_xy = _target_direction_xy((float(initial_robot[0]), float(initial_robot[1])))
    box_target_direction_xy = (
        _target_direction_xy((float(initial_box[0]), float(initial_box[1])))
        if initial_box is not None
        else robot_target_direction_xy
    )
    initial_box_robot_rel = None
    if initial_box is not None:
        initial_box_robot_rel = (
            float(initial_box[0]) - float(initial_robot[0]),
            float(initial_box[1]) - float(initial_robot[1]),
            float(initial_box[2]) - float(initial_robot[2]),
        )
    joint_count = int(joint_positions.size)
    summary = {
        "scene_type": "core_world_g1_box_scene",
        "success_claim": "g1_usd_core_api_scene_diagnostic_not_walking_or_carrying_success",
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "device": args_cli.device,
        "render_requested": bool(args_cli.render),
        "record_replay_csv": bool(args_cli.record_replay_csv),
        "record_replay_csv_path": str(replay_csv_path),
        "record_replay_every_n_steps": int(args_cli.record_replay_every_n_steps),
        "capture_rgb_enabled": bool(args_cli.capture_rgb),
        "capture_rgb_output_dir": str(args_cli.output_dir / "rgb_frames"),
        "capture_rgb_every_n_steps": int(args_cli.capture_rgb_every_n_steps),
        "capture_rgb_resolution": [int(v) for v in args_cli.capture_rgb_resolution],
        "capture_rgb_camera_position": [float(v) for v in args_cli.capture_camera_position],
        "capture_rgb_camera_look_at": [float(v) for v in args_cli.capture_camera_look_at],
        "capture_rgb_frame_count": 0,
        "capture_rgb_error": None,
        "g1_usd": str(args_cli.g1_usd),
        "g1_reference_path": G1_PATH,
        "g1_articulation_path": G1_ARTICULATION_PATH,
        "g1_root_position_requested_m": [float(v) for v in args_cli.g1_root_position],
        "g1_root_orientation_requested_wxyz": [float(v) for v in args_cli.g1_root_orientation_wxyz],
        "usd_g1_articulation_pose_after_design_wxyz": usd_g1_pose_after_design,
        "stand_joint_targets_enabled": not bool(args_cli.disable_stand_joint_targets),
        "stand_joint_target_overrides": {
            "stand_hip_pitch": args_cli.stand_hip_pitch,
            "stand_knee": args_cli.stand_knee,
            "stand_ankle_pitch": args_cli.stand_ankle_pitch,
            "stand_hip_roll": args_cli.stand_hip_roll,
            "stand_ankle_roll": args_cli.stand_ankle_roll,
        },
        "applied_stand_joint_targets": applied_stand_joint_targets,
        "stand_drive_gains_enabled": bool(args_cli.apply_arena_stand_gains),
        "stand_drive_preset": str(args_cli.stand_drive_preset),
        "stand_gain_scale": float(args_cli.stand_gain_scale),
        "stand_force_scale": float(args_cli.stand_force_scale),
        "applied_stand_drive_gains": applied_drive_gains,
        "applied_stand_drive_gain_count": len(applied_drive_gains),
        "gait_mode": str(args_cli.gait_mode),
        "gait_amplitude": float(args_cli.gait_amplitude),
        "gait_frequency_hz": float(args_cli.gait_frequency_hz),
        "gait_start_step": int(args_cli.gait_start_step),
        "gait_stop_step": int(args_cli.gait_stop_step),
        "creep_hip_pitch_offset": float(args_cli.creep_hip_pitch_offset),
        "creep_knee_offset": float(args_cli.creep_knee_offset),
        "creep_ankle_pitch_offset": float(args_cli.creep_ankle_pitch_offset),
        "creep_waist_pitch_offset": float(args_cli.creep_waist_pitch_offset),
        "creep_stance_push_scale": float(args_cli.creep_stance_push_scale),
        "creep_lift_scale": float(args_cli.creep_lift_scale),
        "creep_ankle_lift_scale": float(args_cli.creep_ankle_lift_scale),
        "creep_decel_box_travel_start_m": float(args_cli.creep_decel_box_travel_start),
        "creep_decel_box_travel_end_m": float(args_cli.creep_decel_box_travel_end),
        "creep_decel_robot_travel_start_m": float(args_cli.creep_decel_robot_travel_start),
        "creep_decel_robot_travel_end_m": float(args_cli.creep_decel_robot_travel_end),
        "creep_min_amplitude_scale": float(args_cli.creep_min_amplitude_scale),
        "creep_min_push_scale": float(args_cli.creep_min_push_scale),
        "creep_min_bias_scale": float(args_cli.creep_min_bias_scale),
        "creep_pitch_brake_threshold_rad": float(args_cli.creep_pitch_brake_threshold),
        "creep_pitch_brake_rate_threshold_radps": float(args_cli.creep_pitch_brake_rate_threshold),
        "creep_pitch_brake_amplitude_scale": float(args_cli.creep_pitch_brake_amplitude_scale),
        "creep_pitch_brake_push_scale": float(args_cli.creep_pitch_brake_push_scale),
        "creep_pitch_brake_bias_scale": float(args_cli.creep_pitch_brake_bias_scale),
        "creep_pitch_brake_latch_enabled": bool(args_cli.creep_pitch_brake_latch),
        "creep_pitch_brake_positive_only": bool(args_cli.creep_pitch_brake_positive_only),
        "creep_pitch_brake_latched_step": None,
        "creep_reverse_brake_box_travel_m": float(args_cli.creep_reverse_brake_box_travel),
        "creep_reverse_brake_robot_travel_m": float(args_cli.creep_reverse_brake_robot_travel),
        "creep_reverse_brake_pitch_threshold_rad": float(args_cli.creep_reverse_brake_pitch_threshold),
        "creep_reverse_brake_positive_pitch_only": bool(args_cli.creep_reverse_brake_positive_pitch_only),
        "creep_reverse_brake_duration_steps": int(args_cli.creep_reverse_brake_duration_steps),
        "creep_reverse_brake_amplitude_scale": float(args_cli.creep_reverse_brake_amplitude_scale),
        "creep_reverse_brake_stance_push_scale": float(args_cli.creep_reverse_brake_stance_push_scale),
        "creep_reverse_brake_lift_scale": float(args_cli.creep_reverse_brake_lift_scale),
        "creep_reverse_brake_hip_pitch_offset": float(args_cli.creep_reverse_brake_hip_pitch_offset),
        "creep_reverse_brake_knee_offset": float(args_cli.creep_reverse_brake_knee_offset),
        "creep_reverse_brake_ankle_pitch_offset": float(args_cli.creep_reverse_brake_ankle_pitch_offset),
        "creep_reverse_brake_waist_pitch_offset": float(args_cli.creep_reverse_brake_waist_pitch_offset),
        "creep_reverse_brake_latched_step": None,
        "creep_reverse_brake_first_reason": None,
        "creep_reverse_brake_active_steps": 0,
        "creep_reverse_brake_first_active_step": None,
        "creep_decel_active_steps": 0,
        "creep_decel_first_active_step": None,
        "creep_pitch_brake_active_steps": 0,
        "creep_pitch_brake_first_active_step": None,
        "min_creep_amplitude_scale": 1.0,
        "min_creep_push_scale": 1.0,
        "min_creep_bias_scale": 1.0,
        "gait_ramp_down_start_step": int(args_cli.gait_ramp_down_start_step),
        "gait_ramp_down_end_step": int(args_cli.gait_ramp_down_end_step),
        "gait_min_amplitude_scale": float(args_cli.gait_min_amplitude_scale),
        "recovery_pitch_threshold": float(args_cli.recovery_pitch_threshold),
        "recovery_pitch_rate_threshold": float(args_cli.recovery_pitch_rate_threshold),
        "recovery_hip_pitch_offset": float(args_cli.recovery_hip_pitch_offset),
        "recovery_knee_offset": float(args_cli.recovery_knee_offset),
        "recovery_ankle_pitch_offset": float(args_cli.recovery_ankle_pitch_offset),
        "recovery_waist_pitch_offset": float(args_cli.recovery_waist_pitch_offset),
        "recovery_active_steps": 0,
        "recovery_first_active_step": None,
        "terminal_hold_start_step": int(args_cli.terminal_hold_start_step),
        "terminal_hold_box_target_travel": float(args_cli.terminal_hold_box_target_travel),
        "terminal_hold_robot_target_travel": float(args_cli.terminal_hold_robot_target_travel),
        "terminal_hold_pitch_threshold": float(args_cli.terminal_hold_pitch_threshold),
        "terminal_hold_pitch_rate_threshold": float(args_cli.terminal_hold_pitch_rate_threshold),
        "terminal_hold_hip_pitch_offset": float(args_cli.terminal_hold_hip_pitch_offset),
        "terminal_hold_knee_offset": float(args_cli.terminal_hold_knee_offset),
        "terminal_hold_ankle_pitch_offset": float(args_cli.terminal_hold_ankle_pitch_offset),
        "terminal_hold_waist_pitch_offset": float(args_cli.terminal_hold_waist_pitch_offset),
        "terminal_drive_gain_scale": float(args_cli.terminal_drive_gain_scale),
        "terminal_drive_force_scale": float(args_cli.terminal_drive_force_scale),
        "terminal_hold_active_steps": 0,
        "terminal_hold_first_active_step": None,
        "terminal_hold_first_reason": None,
        "terminal_drive_gain_applied_step": None,
        "terminal_applied_drive_gain_count": 0,
        "terminal_applied_stand_drive_gains": {},
        "balance_feedback_controller_enabled": bool(args_cli.balance_feedback_controller),
        "balance_pitch_gain": float(args_cli.balance_pitch_gain),
        "balance_roll_gain": float(args_cli.balance_roll_gain),
        "balance_pitch_rate_gain": float(args_cli.balance_pitch_rate_gain),
        "balance_roll_rate_gain": float(args_cli.balance_roll_rate_gain),
        "balance_adjustment_limit": float(args_cli.balance_adjustment_limit),
        "balance_feedback_base": str(args_cli.balance_feedback_base),
        "balance_start_on_agile_hold": bool(args_cli.balance_start_on_agile_hold),
        "balance_roll_left_ankle_scale": float(args_cli.balance_roll_left_ankle_scale),
        "balance_roll_right_ankle_scale": float(args_cli.balance_roll_right_ankle_scale),
        "balance_roll_left_hip_scale": float(args_cli.balance_roll_left_hip_scale),
        "balance_roll_right_hip_scale": float(args_cli.balance_roll_right_hip_scale),
        "balance_pitch_target": float(args_cli.balance_pitch_target),
        "balance_roll_target": float(args_cli.balance_roll_target),
        "balance_roll_target_from_lateral": bool(args_cli.balance_roll_target_from_lateral),
        "balance_roll_target_lateral_source": str(args_cli.balance_roll_target_lateral_source),
        "balance_roll_target_lateral_gain": float(args_cli.balance_roll_target_lateral_gain),
        "balance_roll_target_lateral_limit": float(args_cli.balance_roll_target_lateral_limit),
        "balance_roll_target_lateral_deadband": float(args_cli.balance_roll_target_lateral_deadband),
        "balance_roll_target_lateral_sign": float(args_cli.balance_roll_target_lateral_sign),
        "balance_roll_target_lateral_start_after_hold_steps": int(
            args_cli.balance_roll_target_lateral_start_after_hold_steps
        ),
        "balance_roll_target_lateral_ramp_steps": int(args_cli.balance_roll_target_lateral_ramp_steps),
        "balance_roll_target_lateral_max_tilt": float(args_cli.balance_roll_target_lateral_max_tilt),
        "balance_roll_target_lateral_max_box_tilt": float(
            args_cli.balance_roll_target_lateral_max_box_tilt
        ),
        "balance_roll_target_lateral_active_steps": 0,
        "balance_roll_target_lateral_first_active_step": None,
        "balance_roll_target_lateral_last_error_m": 0.0,
        "balance_roll_target_lateral_last_target_rad": 0.0,
        "balance_roll_target_lateral_max_abs_target_rad": 0.0,
        "balance_roll_target_lateral_suppressed_by_hold_delay_steps": 0,
        "balance_roll_target_lateral_suppressed_by_tilt_steps": 0,
        "balance_target_start_step": int(args_cli.balance_target_start_step),
        "balance_target_end_step": int(args_cli.balance_target_end_step),
        "balance_target_pulse_period_steps": int(args_cli.balance_target_pulse_period_steps),
        "balance_target_pulse_width_steps": int(args_cli.balance_target_pulse_width_steps),
        "balance_target_pulse_phase_step": int(args_cli.balance_target_pulse_phase_step),
        "balance_pitch_sign": float(args_cli.balance_pitch_sign),
        "balance_roll_sign": float(args_cli.balance_roll_sign),
        "balance_start_step": int(args_cli.balance_start_step),
        "balance_pitch_activation_threshold": float(args_cli.balance_pitch_activation_threshold),
        "balance_roll_activation_threshold": float(args_cli.balance_roll_activation_threshold),
        "balance_pitch_rate_activation_threshold": float(args_cli.balance_pitch_rate_activation_threshold),
        "balance_roll_rate_activation_threshold": float(args_cli.balance_roll_rate_activation_threshold),
        "balance_feedback_active_steps": 0,
        "balance_feedback_first_active_step": None,
        "balance_target_active_steps": 0,
        "balance_target_first_active_step": None,
        "diagnostic_root_drive": str(args_cli.diagnostic_root_drive),
        "diagnostic_root_drive_start_step": int(args_cli.diagnostic_root_drive_start_step),
        "diagnostic_root_drive_stop_step": int(args_cli.diagnostic_root_drive_stop_step),
        "diagnostic_root_drive_speed_mps": float(args_cli.diagnostic_root_drive_speed),
        "diagnostic_root_drive_ramp_steps": int(args_cli.diagnostic_root_drive_ramp_steps),
        "diagnostic_root_drive_active_steps": 0,
        "diagnostic_root_drive_final_commanded_xy_m": [0.0, 0.0],
        "arm_pose_mode": str(args_cli.arm_pose_mode),
        "arm_pose_start_step": int(args_cli.arm_pose_start_step),
        "arm_pose_ramp_steps": int(args_cli.arm_pose_ramp_steps),
        "arm_pose_targets": {k: float(v) for k, v in arm_pose_targets.items()},
        "box_retention_posture_controller_enabled": bool(args_cli.box_retention_posture_controller),
        "box_retention_rel_start_m": float(args_cli.box_retention_rel_start),
        "box_retention_rel_stop_m": float(args_cli.box_retention_rel_stop),
        "box_retention_tilt_start_rad": float(args_cli.box_retention_tilt_start),
        "box_retention_tilt_stop_rad": float(args_cli.box_retention_tilt_stop),
        "box_retention_active_steps": 0,
        "box_retention_first_active_step": None,
        "box_retention_max_risk": 0.0,
        "box_retention_last_risk": 0.0,
        "arm_pose_active_steps": 0,
        "arm_pose_first_active_step": None,
        "max_abs_roll_rad": 0.0,
        "max_abs_pitch_rad": 0.0,
        "final_roll_rad": None,
        "final_pitch_rad": None,
        "policy_start_step": int(args_cli.policy_start_step),
        "policy_control_decimation": int(args_cli.policy_control_decimation),
        "agile_command_xyz_yaw": [float(v) for v in args_cli.agile_command],
        "agile_height_command": float(args_cli.agile_height_command),
        "agile_command_stop_step": int(args_cli.agile_command_stop_step),
        "agile_command_stop_box_target_travel_m": float(args_cli.agile_command_stop_box_target_travel),
        "agile_command_stop_robot_target_travel_m": float(args_cli.agile_command_stop_robot_target_travel),
        "agile_command_stop_target_window_enabled": bool(args_cli.agile_command_stop_target_window),
        "agile_command_stop_target_window_min_step": int(
            args_cli.agile_command_stop_target_window_min_step
        ),
        "agile_command_stop_target_window_latched_step": None,
        "agile_command_hold_scale": float(args_cli.agile_command_hold_scale),
        "agile_command_hold_adaptive_scale_enabled": bool(args_cli.agile_command_hold_adaptive_scale),
        "agile_command_hold_adaptive_min_scale": float(args_cli.agile_command_hold_adaptive_min_scale),
        "agile_command_hold_adaptive_max_scale": float(args_cli.agile_command_hold_adaptive_max_scale),
        "agile_command_hold_adaptive_tilt_start": float(args_cli.agile_command_hold_adaptive_tilt_start),
        "agile_command_hold_adaptive_tilt_stop": float(args_cli.agile_command_hold_adaptive_tilt_stop),
        "agile_command_hold_adaptive_rate_start": float(args_cli.agile_command_hold_adaptive_rate_start),
        "agile_command_hold_adaptive_rate_stop": float(args_cli.agile_command_hold_adaptive_rate_stop),
        "agile_command_hold_adaptive_rel_start": float(args_cli.agile_command_hold_adaptive_rel_start),
        "agile_command_hold_adaptive_rel_stop": float(args_cli.agile_command_hold_adaptive_rel_stop),
        "agile_command_hold_adaptive_box_tilt_enabled": bool(args_cli.agile_command_hold_adaptive_box_tilt),
        "agile_command_hold_adaptive_box_tilt_start": float(args_cli.agile_command_hold_adaptive_box_tilt_start),
        "agile_command_hold_adaptive_box_tilt_stop": float(args_cli.agile_command_hold_adaptive_box_tilt_stop),
        "agile_command_hold_adaptive_box_tilt_rate_start": float(
            args_cli.agile_command_hold_adaptive_box_tilt_rate_start
        ),
        "agile_command_hold_adaptive_box_tilt_rate_stop": float(
            args_cli.agile_command_hold_adaptive_box_tilt_rate_stop
        ),
        "agile_command_hold_adaptive_scale_smoothing": float(
            args_cli.agile_command_hold_adaptive_scale_smoothing
        ),
        "agile_command_hold_lateral_correction_enabled": bool(args_cli.agile_command_hold_lateral_correction),
        "agile_command_hold_lateral_gain": float(args_cli.agile_command_hold_lateral_gain),
        "agile_command_hold_lateral_limit": float(args_cli.agile_command_hold_lateral_limit),
        "agile_command_hold_lateral_sign": float(args_cli.agile_command_hold_lateral_sign),
        "agile_command_hold_lateral_terminal_only": bool(args_cli.agile_command_hold_lateral_terminal_only),
        "agile_command_hold_lateral_error_start_m": float(args_cli.agile_command_hold_lateral_error_start),
        "agile_command_hold_lateral_use_excess_error": bool(
            args_cli.agile_command_hold_lateral_use_excess_error
        ),
        "agile_command_hold_lateral_max_tilt_rad": float(args_cli.agile_command_hold_lateral_max_tilt),
        "agile_command_hold_lateral_max_box_tilt_rad": float(
            args_cli.agile_command_hold_lateral_max_box_tilt
        ),
        "agile_command_hold_lateral_suppressed_by_tilt_steps": 0,
        "agile_command_hold_yaw_correction_enabled": bool(args_cli.agile_command_hold_yaw_correction),
        "agile_command_hold_yaw_gain": float(args_cli.agile_command_hold_yaw_gain),
        "agile_command_hold_yaw_limit": float(args_cli.agile_command_hold_yaw_limit),
        "agile_command_hold_yaw_sign": float(args_cli.agile_command_hold_yaw_sign),
        "agile_command_hold_terminal_box_target_travel_m": float(
            args_cli.agile_command_hold_terminal_box_target_travel
        ),
        "agile_command_hold_terminal_min_robot_target_travel_m": float(
            args_cli.agile_command_hold_terminal_min_robot_target_travel
        ),
        "agile_command_hold_terminal_min_step": int(args_cli.agile_command_hold_terminal_min_step),
        "agile_command_hold_terminal_scale": float(args_cli.agile_command_hold_terminal_scale),
        "agile_command_hold_terminal_latch_enabled": bool(args_cli.agile_command_hold_terminal_latch),
        "agile_command_hold_terminal_latched": False,
        "agile_command_hold_terminal_latched_step": None,
        "agile_command_hold_terminal_active_steps": 0,
        "agile_command_hold_terminal_first_active_step": None,
        "agile_command_hold_terminal_last_reason": None,
        "agile_command_hold_final_box_target_travel_m": float(
            args_cli.agile_command_hold_final_box_target_travel
        ),
        "agile_command_hold_final_min_robot_target_travel_m": float(
            args_cli.agile_command_hold_final_min_robot_target_travel
        ),
        "agile_command_hold_final_min_step": int(args_cli.agile_command_hold_final_min_step),
        "agile_command_hold_final_scale": float(args_cli.agile_command_hold_final_scale),
        "agile_command_hold_final_latch_enabled": bool(args_cli.agile_command_hold_final_latch),
        "agile_command_hold_final_zero_corrections_enabled": bool(
            args_cli.agile_command_hold_final_zero_corrections
        ),
        "agile_command_hold_final_reset_policy_state": bool(
            args_cli.agile_command_hold_final_reset_policy_state
        ),
        "agile_command_hold_final_policy_state_reset_count": 0,
        "agile_command_hold_final_last_policy_state_reset_error": None,
        "agile_command_hold_final_lateral_suppressed_steps": 0,
        "agile_command_hold_final_yaw_suppressed_steps": 0,
        "agile_command_hold_final_brake_command_x": float(
            args_cli.agile_command_hold_final_brake_command_x
        ),
        "agile_command_hold_final_brake_delay_steps": int(
            args_cli.agile_command_hold_final_brake_delay_steps
        ),
        "agile_command_hold_final_brake_steps": int(args_cli.agile_command_hold_final_brake_steps),
        "agile_command_hold_final_brake_active_steps": 0,
        "agile_command_hold_final_brake_first_active_step": None,
        "agile_command_hold_final_brake_last_active_step": None,
        "agile_command_hold_final_brake_max_abs_command_x": 0.0,
        "agile_command_hold_final_freeze_in_target_window": bool(
            args_cli.agile_command_hold_final_freeze_in_target_window
        ),
        "agile_command_hold_final_freeze_max_tilt_rad": float(
            args_cli.agile_command_hold_final_freeze_max_tilt
        ),
        "agile_command_hold_final_freeze_max_box_tilt_rad": float(
            args_cli.agile_command_hold_final_freeze_max_box_tilt
        ),
        "agile_command_hold_final_freeze_latched": False,
        "agile_command_hold_final_freeze_latched_step": None,
        "agile_command_hold_final_freeze_active_steps": 0,
        "agile_command_hold_final_freeze_first_active_step": None,
        "agile_command_hold_rescue_overrides_final_freeze": bool(
            args_cli.agile_command_hold_rescue_overrides_final_freeze
        ),
        "agile_command_hold_rescue_override_freeze_active_steps": 0,
        "agile_command_hold_rescue_override_freeze_first_active_step": None,
        "agile_command_hold_stand_overrides_final_freeze": bool(
            args_cli.agile_command_hold_stand_overrides_final_freeze
        ),
        "agile_command_hold_stand_override_freeze_active_steps": 0,
        "agile_command_hold_stand_override_freeze_first_active_step": None,
        "agile_command_hold_final_max_abs_command_x": 0.0,
        "agile_command_hold_final_max_abs_command_y": 0.0,
        "agile_command_hold_final_max_abs_command_yaw": 0.0,
        "agile_command_hold_final_last_command_xyz_yaw": None,
        "agile_command_hold_final_stand_enabled": bool(args_cli.agile_command_hold_final_stand),
        "agile_command_hold_final_stand_delay_steps": int(args_cli.agile_command_hold_final_stand_delay_steps),
        "agile_command_hold_final_latched": False,
        "agile_command_hold_final_latched_step": None,
        "agile_command_hold_final_active_steps": 0,
        "agile_command_hold_final_first_active_step": None,
        "agile_command_hold_final_last_reason": None,
        "agile_command_hold_final_stand_active_steps": 0,
        "agile_command_hold_final_stand_first_active_step": None,
        "agile_command_hold_adaptive_active_steps": 0,
        "agile_command_hold_adaptive_first_active_step": None,
        "agile_command_hold_adaptive_min_observed_scale": None,
        "agile_command_hold_adaptive_max_observed_scale": None,
        "agile_command_hold_adaptive_last_risk": 0.0,
        "agile_command_hold_lateral_active_steps": 0,
        "agile_command_hold_lateral_first_active_step": None,
        "agile_command_hold_lateral_max_abs_command": 0.0,
        "agile_command_hold_lateral_last_error_m": 0.0,
        "agile_command_hold_yaw_active_steps": 0,
        "agile_command_hold_yaw_first_active_step": None,
        "agile_command_hold_yaw_max_abs_command": 0.0,
        "agile_command_hold_yaw_last_error_m": 0.0,
        "agile_command_box_progress_controller_enabled": bool(
            args_cli.agile_command_box_progress_controller
        ),
        "agile_command_box_progress_start_step": int(args_cli.agile_command_box_progress_start_step),
        "agile_command_box_progress_target_m": float(args_cli.agile_command_box_progress_target),
        "agile_command_box_progress_deadband_m": float(args_cli.agile_command_box_progress_deadband),
        "agile_command_box_progress_gain": float(args_cli.agile_command_box_progress_gain),
        "agile_command_box_progress_max_forward": float(args_cli.agile_command_box_progress_max_forward),
        "agile_command_box_progress_max_reverse": float(args_cli.agile_command_box_progress_max_reverse),
        "agile_command_box_progress_max_tilt_rad": float(args_cli.agile_command_box_progress_max_tilt),
        "agile_command_box_progress_max_box_tilt_rad": float(
            args_cli.agile_command_box_progress_max_box_tilt
        ),
        "agile_command_box_progress_scale_on_hold": bool(
            args_cli.agile_command_box_progress_scale_on_hold
        ),
        "agile_command_box_progress_active_steps": 0,
        "agile_command_box_progress_first_active_step": None,
        "agile_command_box_progress_last_error_m": 0.0,
        "agile_command_box_progress_last_command_x": 0.0,
        "agile_command_box_progress_max_abs_command_x": 0.0,
        "agile_command_box_progress_hold_scaled_steps": 0,
        "agile_command_box_progress_tilt_suppressed_steps": 0,
        "agile_command_box_lateral_controller_enabled": bool(args_cli.agile_command_box_lateral_controller),
        "agile_command_box_lateral_deadband_m": float(args_cli.agile_command_box_lateral_deadband),
        "agile_command_box_lateral_gain": float(args_cli.agile_command_box_lateral_gain),
        "agile_command_box_lateral_limit": float(args_cli.agile_command_box_lateral_limit),
        "agile_command_box_lateral_sign": float(args_cli.agile_command_box_lateral_sign),
        "agile_command_box_lateral_scale_on_hold": bool(args_cli.agile_command_box_lateral_scale_on_hold),
        "agile_command_box_lateral_active_steps": 0,
        "agile_command_box_lateral_first_active_step": None,
        "agile_command_box_lateral_last_error_m": 0.0,
        "agile_command_box_lateral_last_command_y": 0.0,
        "agile_command_box_lateral_max_abs_command_y": 0.0,
        "agile_command_box_lateral_hold_scaled_steps": 0,
        "agile_command_hold_mode": str(args_cli.agile_command_hold_mode),
        "agile_command_hold_stand_blend_rate": float(args_cli.agile_command_hold_stand_blend_rate),
        "agile_command_hold_policy_then_stand_delay_steps": int(
            args_cli.agile_command_hold_policy_then_stand_delay_steps
        ),
        "agile_command_hold_stand_target_overrides": {
            "hip_pitch": args_cli.agile_command_hold_stand_hip_pitch,
            "knee": args_cli.agile_command_hold_stand_knee,
            "ankle_pitch": args_cli.agile_command_hold_stand_ankle_pitch,
            "hip_roll": args_cli.agile_command_hold_stand_hip_roll,
            "ankle_roll": args_cli.agile_command_hold_stand_ankle_roll,
            "waist_pitch": args_cli.agile_command_hold_stand_waist_pitch,
        },
        "agile_command_hold_applied_stand_joint_targets": applied_hold_stand_joint_targets,
        "agile_command_hold_rescue_enabled": bool(args_cli.agile_command_hold_rescue_enable),
        "agile_command_hold_rescue_forward_pitch_threshold_rad": float(
            args_cli.agile_command_hold_rescue_forward_pitch_threshold
        ),
        "agile_command_hold_rescue_abs_roll_threshold_rad": float(
            args_cli.agile_command_hold_rescue_abs_roll_threshold
        ),
        "agile_command_hold_rescue_blend_rate": float(args_cli.agile_command_hold_rescue_blend_rate),
        "agile_command_hold_rescue_target_overrides": {
            "hip_pitch": args_cli.agile_command_hold_rescue_hip_pitch,
            "knee": args_cli.agile_command_hold_rescue_knee,
            "ankle_pitch": args_cli.agile_command_hold_rescue_ankle_pitch,
            "hip_roll": args_cli.agile_command_hold_rescue_hip_roll,
            "ankle_roll": args_cli.agile_command_hold_rescue_ankle_roll,
            "waist_pitch": args_cli.agile_command_hold_rescue_waist_pitch,
        },
        "agile_command_hold_applied_rescue_joint_targets": applied_hold_rescue_joint_targets,
        "agile_command_hold_rescue_active": False,
        "agile_command_hold_rescue_first_active_step": None,
        "agile_command_hold_rescue_first_reason": None,
        "agile_command_hold_rescue_active_steps": 0,
        "agile_command_hold_reset_policy_state": bool(args_cli.agile_command_hold_reset_policy_state),
        "agile_command_hold_active": False,
        "agile_command_hold_first_active_step": None,
        "agile_command_hold_first_reason": None,
        "agile_command_hold_active_steps": 0,
        "agile_command_hold_stand_target_active_steps": 0,
        "agile_command_hold_policy_state_reset_count": 0,
        "agile_command_hold_last_policy_state_reset_error": None,
        "agile_last_command_xyz_yaw": [float(v) for v in args_cli.agile_command],
        "agile_policy_backend": str(args_cli.agile_policy_backend),
        "agile_config": str(args_cli.agile_config) if agile_policy is not None else None,
        "agile_onnx": str(args_cli.agile_onnx) if agile_policy is not None else None,
        "agile_torch_checkpoint": str(args_cli.agile_torch_checkpoint) if agile_policy is not None else None,
        "policy_inference_count": 0,
        "max_policy_raw_action_norm": 0.0,
        "agile_root_ang_vel_source": "core_api_get_angular_velocity_rotated_to_body" if agile_policy is not None else None,
        "agile_root_ang_vel_read_failures": 0,
        "agile_last_root_ang_vel_read_error": None,
        "max_agile_root_ang_vel_norm": 0.0,
        "joint_count": joint_count,
        "attach_box": str(args_cli.attach_box),
        "torso_cradle": str(args_cli.torso_cradle),
        "require_box_no_drop": bool(args_cli.require_box_no_drop),
        "carry_box_spawned": not bool(args_cli.disable_carry_box_spawn),
        "attach_body_path": str(args_cli.attach_body_path),
        "attach_local_pos0_m": [float(v) for v in args_cli.attach_local_pos0],
        "payload_joint_path": payload_joint_path,
        "cradle_piece_joints": cradle_piece_joints,
        "cradle_piece_count": len(cradle_piece_joints),
        "probe_mode": str(args_cli.probe_mode),
        "probe_joint_path": probe_joint_path,
        "probe_start_step": int(args_cli.probe_start_step),
        "probe_end_step": int(args_cli.probe_end_step),
        "probe_collision_window_enabled": bool(args_cli.probe_collision_window),
        "probe_pad_size_m": [float(v) for v in args_cli.probe_pad_size],
        "probe_pad_local_pos0_m": [float(v) for v in args_cli.probe_pad_local_pos0],
        "probe_pad_mass_kg": float(args_cli.probe_pad_mass),
        "probe_pad_collision_enabled": (
            not bool(args_cli.disable_probe_pad_collision)
            and not bool(args_cli.probe_collision_window)
        ),
        "probe_collision_enabled_initial": (
            not bool(args_cli.disable_probe_pad_collision)
            and not bool(args_cli.probe_collision_window)
        ),
        "probe_collision_enabled_step": None,
        "probe_collision_disabled_step": None,
        "probe_collision_update_count": 0,
        "probe_collision_update_error": None,
        "probe_active_steps": 0,
        "probe_reference_step": None,
        "probe_reference_box_pose_wxyz": None,
        "probe_reference_robot_pose_wxyz": None,
        "max_probe_box_travel_xy_m": 0.0,
        "final_probe_box_travel_xy_m": 0.0,
        "max_probe_box_target_directed_travel_m": 0.0,
        "final_probe_box_target_directed_travel_m": 0.0,
        "probe_box_displacement_xyz_m": None,
        "probe_box_moved": False,
        "grasp_mode": str(args_cli.grasp_mode),
        "grasp_body_path": str(args_cli.grasp_body_path),
        "active_grasp_body_path": active_grasp_body_path,
        "grasp_body_wrapper_initialized": bool(grasp_body_wrapper_initialized),
        "grasp_body_wrapper_error": grasp_body_wrapper_error,
        "grasp_enable_step": int(args_cli.grasp_enable_step),
        "grasp_lift_offset_z_m": float(args_cli.grasp_lift_offset_z),
        "grasp_attached": False,
        "grasp_attach_step": None,
        "grasp_joint_path": None,
        "grasp_local_pos0_m": None,
        "grasp_box_pose_at_attach_wxyz": None,
        "grasp_robot_pose_at_attach_wxyz": None,
        "grasp_body_pose_at_attach_wxyz": None,
        "grasp_body_box_world_delta_at_attach_m": None,
        "grasp_body_box_world_distance_at_attach_m": None,
        "grasp_box_z_at_attach_m": None,
        "max_post_grasp_box_z_delta_m": 0.0,
        "final_post_grasp_box_z_delta_m": 0.0,
        "min_post_grasp_box_z_m": None,
        "cradle_deck_size_m": [float(v) for v in args_cli.cradle_deck_size],
        "cradle_deck_local_pos0_m": [float(v) for v in args_cli.cradle_deck_local_pos0],
        "cradle_side_rail_height_m": float(args_cli.cradle_side_rail_height),
        "cradle_end_stop_height_m": float(args_cli.cradle_end_stop_height),
        "cradle_rail_thickness_m": float(args_cli.cradle_rail_thickness),
        "cradle_mass_scale": float(args_cli.cradle_mass_scale),
        "cradle_top_lid_enabled": bool(args_cli.cradle_top_lid),
        "cradle_top_lid_local_z_m": float(args_cli.cradle_top_lid_local_z),
        "cradle_top_lid_thickness_m": float(args_cli.cradle_top_lid_thickness),
        "cradle_top_lid_x_scale": float(args_cli.cradle_top_lid_x_scale),
        "cradle_top_lid_y_scale": float(args_cli.cradle_top_lid_y_scale),
        "cradle_top_lid_enable_on_hold": bool(args_cli.cradle_top_lid_enable_on_hold),
        "cradle_top_lid_collision_enabled_initial": bool(args_cli.cradle_top_lid) and not bool(args_cli.cradle_top_lid_enable_on_hold),
        "cradle_top_lid_collision_enabled_step": None,
        "cradle_top_lid_collision_update_count": 0,
        "cradle_top_lid_collision_update_error": None,
        "cradle_chest_pad_enabled": bool(args_cli.cradle_chest_pad),
        "cradle_chest_pad_local_pos0_m": [float(v) for v in args_cli.cradle_chest_pad_local_pos0],
        "cradle_chest_pad_size_m": [float(v) for v in args_cli.cradle_chest_pad_size],
        "cradle_chest_pad_mass_scale": float(args_cli.cradle_chest_pad_mass_scale),
        "cradle_chest_pad_spawn_on_trigger": bool(args_cli.cradle_chest_pad_spawn_on_trigger),
        "cradle_chest_pad_enable_on_hold": bool(args_cli.cradle_chest_pad_enable_on_hold),
        "cradle_chest_pad_enable_on_terminal_hold": bool(args_cli.cradle_chest_pad_enable_on_terminal_hold),
        "cradle_chest_pad_enable_on_final_hold": bool(args_cli.cradle_chest_pad_enable_on_final_hold),
        "cradle_chest_pad_enable_on_target_window": bool(args_cli.cradle_chest_pad_enable_on_target_window),
        "cradle_chest_pad_target_window_min_step": int(args_cli.cradle_chest_pad_target_window_min_step),
        "cradle_chest_pad_enable_on_box_tilt": bool(args_cli.cradle_chest_pad_enable_on_box_tilt),
        "cradle_chest_pad_box_tilt_threshold_rad": float(args_cli.cradle_chest_pad_box_tilt_threshold),
        "cradle_chest_pad_box_tilt_min_step": int(args_cli.cradle_chest_pad_box_tilt_min_step),
        "cradle_chest_pad_collision_enabled_initial": bool(args_cli.cradle_chest_pad)
        and not (
            bool(args_cli.cradle_chest_pad_spawn_on_trigger)
            or bool(args_cli.cradle_chest_pad_enable_on_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_terminal_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_final_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_target_window)
            or bool(args_cli.cradle_chest_pad_enable_on_box_tilt)
        ),
        "cradle_chest_pad_collision_enabled_step": None,
        "cradle_chest_pad_collision_enabled_reason": None,
        "cradle_chest_pad_spawned_step": None,
        "cradle_chest_pad_spawn_error": None,
        "cradle_chest_pad_collision_update_count": 0,
        "cradle_chest_pad_collision_update_error": None,
        "cradle_collision_enabled": not bool(args_cli.disable_cradle_collision),
        "box_mass_kg": float(args_cli.box_mass),
        "box_size_m": [float(v) for v in args_cli.box_size],
        "box_position_requested_m": [float(v) for v in args_cli.box_position],
        "target_xy_m": [float(v) for v in TARGET_XY],
        "box_support_mode": str(args_cli.box_support_mode),
        "box_support_size_m": [float(v) for v in args_cli.box_support_size],
        "box_support_top_clearance_m": float(args_cli.box_support_top_clearance),
        "box_support_release_step": int(args_cli.box_support_release_step),
        "box_support_released": False,
        "box_support_actual_release_step": None,
        "box_support_release_path": BOX_SUPPORT_TABLE_PATH if str(args_cli.box_support_mode) == "table" else None,
        "box_collision_enabled": not bool(args_cli.disable_box_collision),
        "root_pose_write_count_setup": root_pose_write_count_setup,
        "joint_state_write_count_setup": joint_state_write_count_setup,
        "joint_state_write_error": joint_state_write_error,
        "root_pose_write_count_rollout": 0,
        "root_velocity_write_count_rollout": 0,
        "box_pose_write_count_rollout": 0,
        "fall_events": 0,
        "box_drop_events": 0,
        "first_fall_step": None,
        "first_fall_time_s": None,
        "first_box_drop_step": None,
        "first_box_drop_time_s": None,
        "agile_command_hold_final_fall_events": 0,
        "agile_command_hold_final_box_drop_events": 0,
        "agile_command_hold_final_first_fall_step": None,
        "agile_command_hold_final_first_box_drop_step": None,
        "agile_command_hold_final_stand_fall_events": 0,
        "agile_command_hold_final_stand_box_drop_events": 0,
        "agile_command_hold_final_stand_first_fall_step": None,
        "agile_command_hold_final_stand_first_box_drop_step": None,
        "max_robot_travel_xy_m": 0.0,
        "max_box_travel_xy_m": 0.0,
        "max_robot_target_directed_travel_m": 0.0,
        "max_box_target_directed_travel_m": 0.0,
        "final_robot_travel_xy_m": 0.0,
        "final_box_travel_xy_m": 0.0,
        "final_robot_target_directed_travel_m": 0.0,
        "final_box_target_directed_travel_m": 0.0,
        "final_robot_delta_xy_m": [0.0, 0.0],
        "final_box_delta_xy_m": [0.0, 0.0] if initial_box is not None else None,
        "max_abs_robot_target_lateral_error_m": 0.0,
        "max_abs_box_target_lateral_error_m": 0.0,
        "final_robot_target_lateral_error_m": 0.0,
        "final_box_target_lateral_error_m": None,
        "target_window_center_m": float(args_cli.target_window_center),
        "target_window_halfwidth_m": float(args_cli.target_window_halfwidth),
        "target_window_enabled": bool(
            float(args_cli.target_window_center) >= 0.0 and float(args_cli.target_window_halfwidth) >= 0.0
        ),
        "target_window_robot_stable_steps": 0,
        "target_window_box_stable_steps": 0,
        "target_window_both_stable_steps": 0,
        "target_window_robot_longest_streak_steps": 0,
        "target_window_box_longest_streak_steps": 0,
        "target_window_both_longest_streak_steps": 0,
        "target_window_both_streak_at_end_steps": 0,
        "target_window_robot_first_stable_step": None,
        "target_window_box_first_stable_step": None,
        "target_window_both_first_stable_step": None,
        "target_window_both_final_hold_stable_steps": 0,
        "target_window_both_final_hold_longest_streak_steps": 0,
        "target_window_both_final_hold_streak_at_end_steps": 0,
        "target_window_both_final_hold_first_stable_step": None,
        "target_window_both_final_stand_stable_steps": 0,
        "target_window_both_final_stand_longest_streak_steps": 0,
        "target_window_both_final_stand_streak_at_end_steps": 0,
        "target_window_both_final_stand_first_stable_step": None,
        "target_window_both_stable_at_final_step": False,
        "target_window_both_final_hold_stable_at_final_step": False,
        "target_window_both_final_stand_stable_at_final_step": False,
        "robot_target_direction_xy": [float(v) for v in robot_target_direction_xy],
        "box_target_direction_xy": [float(v) for v in box_target_direction_xy],
        "max_box_robot_relative_offset_error_m": 0.0,
        "final_box_robot_relative_offset_error_m": None,
        "min_robot_z_m": float(initial_robot[2]),
        "min_box_z_m": float(initial_box[2]) if initial_box is not None else None,
        "max_tilt_rad": 0.0,
        "max_box_tilt_rad": 0.0,
        "agile_command_hold_final_min_robot_z_m": None,
        "agile_command_hold_final_min_box_z_m": None,
        "agile_command_hold_final_max_tilt_rad": 0.0,
        "agile_command_hold_final_max_box_tilt_rad": 0.0,
        "agile_command_hold_final_stand_min_robot_z_m": None,
        "agile_command_hold_final_stand_min_box_z_m": None,
        "agile_command_hold_final_stand_max_tilt_rad": 0.0,
        "agile_command_hold_final_stand_max_box_tilt_rad": 0.0,
        "max_abs_box_roll_rad": 0.0,
        "max_abs_box_pitch_rad": 0.0,
        "final_box_roll_rad": None,
        "final_box_pitch_rad": None,
        "error": None,
        "init_stage": "rollout",
        "status": "unknown",
        "failures": [],
    }

    capture_rep = None
    capture_writer = None
    capture_render_product = None

    try:
        if bool(args_cli.capture_rgb):
            try:
                print("[PROGRESS] enabling omni.replicator.core for RGB capture", flush=True)
                enable_extension("omni.replicator.core")
                for _ in range(5):
                    simulation_app.update()
                import omni.replicator.core as rep  # noqa: PLC0415

                rep.orchestrator.set_capture_on_play(False)
                capture_dir = args_cli.output_dir / "rgb_frames"
                capture_dir.mkdir(parents=True, exist_ok=True)
                backend = rep.backends.get("DiskBackend")
                backend.initialize(output_dir=str(capture_dir))
                camera = rep.functional.create.camera(
                    position=tuple(float(v) for v in args_cli.capture_camera_position),
                    look_at=tuple(float(v) for v in args_cli.capture_camera_look_at),
                    parent="/World",
                    name="G1ShowcaseCamera",
                )
                capture_render_product = rep.create.render_product(
                    camera,
                    tuple(int(v) for v in args_cli.capture_rgb_resolution),
                    name="G1ShowcaseRenderProduct",
                )
                capture_writer = rep.writers.get("BasicWriter")
                capture_writer.initialize(backend=backend, rgb=True)
                capture_writer.attach(capture_render_product)
                capture_rep = rep
                print(f"[INFO] RGB capture enabled: {capture_dir}", flush=True)
            except Exception as exc:
                summary["capture_rgb_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[WARN] RGB capture setup failed: {summary['capture_rgb_error']}", flush=True)
        prev_feedback_roll = None
        prev_feedback_pitch = None
        prev_box_feedback_roll = None
        prev_box_feedback_pitch = None
        box_feedback_tilt = 0.0
        box_feedback_tilt_rate = 0.0
        prev_robot_target_directed = 0.0
        prev_box_target_directed = 0.0
        prev_box_robot_rel_error = 0.0
        target_window_robot_streak = 0
        target_window_box_streak = 0
        target_window_both_streak = 0
        target_window_both_final_hold_streak = 0
        target_window_both_final_stand_streak = 0
        probe_reference_box_pose = None
        probe_reference_robot_pose = None
        probe_collision_active = bool(
            str(args_cli.probe_mode) != "none"
            and not bool(args_cli.disable_probe_pad_collision)
            and not bool(args_cli.probe_collision_window)
        )
        grasp_attached = False
        grasp_box_z_at_attach = None
        box_support_released = False
        terminal_hold_active = False
        terminal_hold_first_reason = None
        terminal_drive_applied = False
        agile_command_hold_active = False
        agile_command_hold_first_reason = None
        agile_command_hold_rescue_active = False
        agile_command_hold_rescue_first_reason = None
        agile_command_hold_terminal_latched = False
        agile_command_hold_final_latched = False
        agile_command_hold_adaptive_scale_value = float(args_cli.agile_command_hold_scale)
        agile_hold_policy_state_reset_done = False
        agile_final_policy_state_reset_done = False
        final_frozen_policy_joint_targets = None
        top_lid_hold_collision_enabled = bool(args_cli.cradle_top_lid) and not bool(args_cli.cradle_top_lid_enable_on_hold)
        chest_pad_hold_collision_enabled = bool(args_cli.cradle_chest_pad) and not (
            bool(args_cli.cradle_chest_pad_spawn_on_trigger)
            or bool(args_cli.cradle_chest_pad_enable_on_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_terminal_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_final_hold)
            or bool(args_cli.cradle_chest_pad_enable_on_target_window)
            or bool(args_cli.cradle_chest_pad_enable_on_box_tilt)
        )
        creep_pitch_brake_latched = False
        creep_reverse_brake_latched = False
        creep_reverse_brake_latched_step = None
        diagnostic_root_drive_position = None
        diagnostic_root_drive_orientation = None
        diagnostic_root_drive_direction = np.array(robot_target_direction_xy, dtype=float)
        with ExitStack() as file_stack:
            f = file_stack.enter_context(csv_path.open("w", newline=""))
            writer = csv.writer(f)
            replay_writer = None
            if bool(args_cli.record_replay_csv):
                replay_file = file_stack.enter_context(replay_csv_path.open("w", newline=""))
                replay_writer = csv.writer(replay_file)
                replay_writer.writerow([
                    "step",
                    "time_s",
                    "robot_x",
                    "robot_y",
                    "robot_z",
                    "robot_qw",
                    "robot_qx",
                    "robot_qy",
                    "robot_qz",
                    "box_x",
                    "box_y",
                    "box_z",
                    "box_qw",
                    "box_qx",
                    "box_qy",
                    "box_qz",
                    "joint_names_json",
                    "joint_positions_json",
                ])
            writer.writerow([
                "step",
                "time_s",
                "robot_x",
                "robot_y",
                "robot_z",
                "box_x",
                "box_y",
                "box_z",
                "robot_travel_xy_m",
                "box_travel_xy_m",
                "robot_target_lateral_error_m",
                "box_target_lateral_error_m",
                "roll",
                "pitch",
                "tilt",
                "box_roll",
                "box_pitch",
                "box_tilt",
                "fall",
                "drop",
            ])
            for step in range(int(args_cli.steps)):
                time_s = step * 0.005
                current_final_hold_scale_active = False
                current_final_stand_active = False
                if joint_count:
                    pose_for_feedback = _pose_wxyz(robot)
                    feedback_roll, feedback_pitch = _quat_to_roll_pitch(
                        float(pose_for_feedback[3]),
                        float(pose_for_feedback[4]),
                        float(pose_for_feedback[5]),
                        float(pose_for_feedback[6]),
                    )
                    feedback_roll_rate = 0.0 if prev_feedback_roll is None else (float(feedback_roll) - float(prev_feedback_roll)) / 0.005
                    feedback_pitch_rate = 0.0 if prev_feedback_pitch is None else (float(feedback_pitch) - float(prev_feedback_pitch)) / 0.005
                    prev_feedback_roll = float(feedback_roll)
                    prev_feedback_pitch = float(feedback_pitch)
                    box_pose_for_feedback = _pose_wxyz(box) if box is not None else None
                    if box_pose_for_feedback is not None:
                        box_feedback_roll, box_feedback_pitch = _quat_to_roll_pitch(
                            float(box_pose_for_feedback[3]),
                            float(box_pose_for_feedback[4]),
                            float(box_pose_for_feedback[5]),
                            float(box_pose_for_feedback[6]),
                        )
                        box_feedback_roll_rate = (
                            0.0
                            if prev_box_feedback_roll is None
                            else (float(box_feedback_roll) - float(prev_box_feedback_roll)) / 0.005
                        )
                        box_feedback_pitch_rate = (
                            0.0
                            if prev_box_feedback_pitch is None
                            else (float(box_feedback_pitch) - float(prev_box_feedback_pitch)) / 0.005
                        )
                        box_feedback_tilt = max(abs(float(box_feedback_roll)), abs(float(box_feedback_pitch)))
                        box_feedback_tilt_rate = max(
                            abs(float(box_feedback_roll_rate)),
                            abs(float(box_feedback_pitch_rate)),
                        )
                        prev_box_feedback_roll = float(box_feedback_roll)
                        prev_box_feedback_pitch = float(box_feedback_pitch)
                    else:
                        box_feedback_tilt = 0.0
                        box_feedback_tilt_rate = 0.0
                    if agile_policy is not None and step >= int(args_cli.policy_start_step):
                        if not agile_command_hold_active:
                            agile_hold_reasons = []
                            if int(args_cli.agile_command_stop_step) >= 0 and int(step) >= int(args_cli.agile_command_stop_step):
                                agile_hold_reasons.append("stop_step")
                            if (
                                float(args_cli.agile_command_stop_box_target_travel) >= 0.0
                                and float(prev_box_target_directed) >= float(args_cli.agile_command_stop_box_target_travel)
                            ):
                                agile_hold_reasons.append("box_target_travel")
                            if (
                                float(args_cli.agile_command_stop_robot_target_travel) >= 0.0
                                and float(prev_robot_target_directed) >= float(args_cli.agile_command_stop_robot_target_travel)
                            ):
                                agile_hold_reasons.append("robot_target_travel")
                            if (
                                bool(args_cli.agile_command_stop_target_window)
                                and bool(summary.get("target_window_enabled"))
                                and (
                                    int(args_cli.agile_command_stop_target_window_min_step) < 0
                                    or int(step) >= int(args_cli.agile_command_stop_target_window_min_step)
                                )
                            ):
                                target_center = float(args_cli.target_window_center)
                                target_halfwidth = float(args_cli.target_window_halfwidth)
                                if (
                                    abs(float(prev_robot_target_directed) - target_center) <= target_halfwidth
                                    and abs(float(prev_box_target_directed) - target_center) <= target_halfwidth
                                ):
                                    agile_hold_reasons.append("target_window")
                            if agile_hold_reasons:
                                agile_command_hold_active = True
                                agile_command_hold_first_reason = ",".join(agile_hold_reasons)
                                summary["agile_command_hold_active"] = True
                                summary["agile_command_hold_first_active_step"] = int(step)
                                summary["agile_command_hold_first_reason"] = agile_command_hold_first_reason
                                if (
                                    "target_window" in agile_hold_reasons
                                    and summary["agile_command_stop_target_window_latched_step"] is None
                                ):
                                    summary["agile_command_stop_target_window_latched_step"] = int(step)
                                if (
                                    bool(args_cli.cradle_top_lid)
                                    and bool(args_cli.cradle_top_lid_enable_on_hold)
                                    and not top_lid_hold_collision_enabled
                                ):
                                    try:
                                        if not _set_collision_enabled(stage, "/World/G1FrontCradle_top_lid", True):
                                            raise RuntimeError("top lid prim not found")
                                        summary["cradle_top_lid_collision_enabled_step"] = int(step)
                                        summary["cradle_top_lid_collision_update_count"] = (
                                            int(summary["cradle_top_lid_collision_update_count"]) + 1
                                        )
                                        top_lid_hold_collision_enabled = True
                                    except Exception as exc:
                                        summary["cradle_top_lid_collision_update_error"] = f"{type(exc).__name__}: {exc}"
                                if (
                                    bool(args_cli.cradle_chest_pad)
                                    and bool(args_cli.cradle_chest_pad_enable_on_hold)
                                    and not chest_pad_hold_collision_enabled
                                ):
                                    try:
                                        if not _set_collision_enabled(stage, "/World/G1FrontCradle_chest_pad", True):
                                            raise RuntimeError("chest pad prim not found")
                                        summary["cradle_chest_pad_collision_enabled_step"] = int(step)
                                        summary["cradle_chest_pad_collision_update_count"] = (
                                            int(summary["cradle_chest_pad_collision_update_count"]) + 1
                                        )
                                        chest_pad_hold_collision_enabled = True
                                    except Exception as exc:
                                        summary["cradle_chest_pad_collision_update_error"] = f"{type(exc).__name__}: {exc}"
                                if bool(args_cli.agile_command_hold_reset_policy_state) and not agile_hold_policy_state_reset_done:
                                    try:
                                        agile_policy.reset_state()
                                        summary["agile_command_hold_policy_state_reset_count"] = (
                                            int(summary["agile_command_hold_policy_state_reset_count"]) + 1
                                        )
                                    except Exception as exc:
                                        summary["agile_command_hold_last_policy_state_reset_error"] = f"{type(exc).__name__}: {exc}"
                                    agile_hold_policy_state_reset_done = True
                        if agile_command_hold_active:
                            summary["agile_command_hold_active_steps"] = int(summary["agile_command_hold_active_steps"]) + 1
                        if (
                            bool(args_cli.cradle_chest_pad)
                            and not chest_pad_hold_collision_enabled
                            and (
                                bool(args_cli.cradle_chest_pad_enable_on_target_window)
                                or bool(args_cli.cradle_chest_pad_enable_on_box_tilt)
                            )
                        ):
                            chest_pad_reasons = []
                            if (
                                bool(args_cli.cradle_chest_pad_enable_on_target_window)
                                and bool(summary.get("target_window_enabled"))
                                and (
                                    int(args_cli.cradle_chest_pad_target_window_min_step) < 0
                                    or int(step) >= int(args_cli.cradle_chest_pad_target_window_min_step)
                                )
                            ):
                                target_center = float(args_cli.target_window_center)
                                target_halfwidth = float(args_cli.target_window_halfwidth)
                                if (
                                    abs(float(prev_robot_target_directed) - target_center) <= target_halfwidth
                                    and abs(float(prev_box_target_directed) - target_center) <= target_halfwidth
                                ):
                                    chest_pad_reasons.append("target_window")
                            if (
                                bool(args_cli.cradle_chest_pad_enable_on_box_tilt)
                                and (
                                    int(args_cli.cradle_chest_pad_box_tilt_min_step) < 0
                                    or int(step) >= int(args_cli.cradle_chest_pad_box_tilt_min_step)
                                )
                                and float(box_feedback_tilt) >= float(args_cli.cradle_chest_pad_box_tilt_threshold)
                            ):
                                chest_pad_reasons.append("box_tilt")
                            if chest_pad_reasons:
                                try:
                                    if (
                                        bool(args_cli.cradle_chest_pad_spawn_on_trigger)
                                        and not stage.GetPrimAtPath("/World/G1FrontCradle_chest_pad").IsValid()
                                    ):
                                        _spawn_front_cradle_chest_pad(stage, None, collision=True)
                                        summary["cradle_chest_pad_spawned_step"] = int(step)
                                    elif not _set_collision_enabled(stage, "/World/G1FrontCradle_chest_pad", True):
                                        raise RuntimeError("chest pad prim not found")
                                    summary["cradle_chest_pad_collision_enabled_step"] = int(step)
                                    summary["cradle_chest_pad_collision_enabled_reason"] = ",".join(chest_pad_reasons)
                                    summary["cradle_chest_pad_collision_update_count"] = (
                                        int(summary["cradle_chest_pad_collision_update_count"]) + 1
                                    )
                                    chest_pad_hold_collision_enabled = True
                                except Exception as exc:
                                    error_text = f"{type(exc).__name__}: {exc}"
                                    summary["cradle_chest_pad_collision_update_error"] = error_text
                                    if bool(args_cli.cradle_chest_pad_spawn_on_trigger):
                                        summary["cradle_chest_pad_spawn_error"] = error_text
                        command_scale = float(args_cli.agile_command_hold_scale) if agile_command_hold_active else 1.0
                        adaptive_risk = 0.0
                        if agile_command_hold_active and bool(args_cli.agile_command_hold_adaptive_scale):
                            tilt_now = max(abs(float(feedback_roll)), abs(float(feedback_pitch)))
                            rate_now = max(abs(float(feedback_roll_rate)), abs(float(feedback_pitch_rate)))
                            adaptive_risk = max(
                                _ramp01(
                                    tilt_now,
                                    float(args_cli.agile_command_hold_adaptive_tilt_start),
                                    float(args_cli.agile_command_hold_adaptive_tilt_stop),
                                ),
                                _ramp01(
                                    rate_now,
                                    float(args_cli.agile_command_hold_adaptive_rate_start),
                                    float(args_cli.agile_command_hold_adaptive_rate_stop),
                                ),
                                _ramp01(
                                    float(prev_box_robot_rel_error),
                                    float(args_cli.agile_command_hold_adaptive_rel_start),
                                    float(args_cli.agile_command_hold_adaptive_rel_stop),
                                ),
                            )
                            if bool(args_cli.agile_command_hold_adaptive_box_tilt):
                                adaptive_risk = max(
                                    adaptive_risk,
                                    _ramp01(
                                        float(box_feedback_tilt),
                                        float(args_cli.agile_command_hold_adaptive_box_tilt_start),
                                        float(args_cli.agile_command_hold_adaptive_box_tilt_stop),
                                    ),
                                    _ramp01(
                                        float(box_feedback_tilt_rate),
                                        float(args_cli.agile_command_hold_adaptive_box_tilt_rate_start),
                                        float(args_cli.agile_command_hold_adaptive_box_tilt_rate_stop),
                                    ),
                                )
                            min_scale = float(args_cli.agile_command_hold_adaptive_min_scale)
                            max_scale = float(args_cli.agile_command_hold_adaptive_max_scale)
                            target_scale = max_scale - adaptive_risk * (max_scale - min_scale)
                            smoothing = max(
                                0.0,
                                min(1.0, float(args_cli.agile_command_hold_adaptive_scale_smoothing)),
                            )
                            agile_command_hold_adaptive_scale_value = (
                                (1.0 - smoothing) * float(agile_command_hold_adaptive_scale_value)
                                + smoothing * float(target_scale)
                            )
                            command_scale = float(agile_command_hold_adaptive_scale_value)
                            summary["agile_command_hold_adaptive_active_steps"] = (
                                int(summary["agile_command_hold_adaptive_active_steps"]) + 1
                            )
                            if summary["agile_command_hold_adaptive_first_active_step"] is None:
                                summary["agile_command_hold_adaptive_first_active_step"] = int(step)
                            min_observed = summary["agile_command_hold_adaptive_min_observed_scale"]
                            max_observed = summary["agile_command_hold_adaptive_max_observed_scale"]
                            summary["agile_command_hold_adaptive_min_observed_scale"] = (
                                command_scale if min_observed is None else min(float(min_observed), command_scale)
                            )
                            summary["agile_command_hold_adaptive_max_observed_scale"] = (
                                command_scale if max_observed is None else max(float(max_observed), command_scale)
                            )
                            summary["agile_command_hold_adaptive_last_risk"] = float(adaptive_risk)
                        terminal_threshold_reached = (
                            agile_command_hold_active
                            and float(args_cli.agile_command_hold_terminal_box_target_travel) >= 0.0
                            and float(prev_box_target_directed)
                            >= float(args_cli.agile_command_hold_terminal_box_target_travel)
                            and (
                                float(args_cli.agile_command_hold_terminal_min_robot_target_travel) < 0.0
                                or float(prev_robot_target_directed)
                                >= float(args_cli.agile_command_hold_terminal_min_robot_target_travel)
                            )
                            and (
                                int(args_cli.agile_command_hold_terminal_min_step) < 0
                                or int(step) >= int(args_cli.agile_command_hold_terminal_min_step)
                            )
                        )
                        if terminal_threshold_reached and bool(args_cli.agile_command_hold_terminal_latch):
                            agile_command_hold_terminal_latched = True
                            summary["agile_command_hold_terminal_latched"] = True
                            if summary["agile_command_hold_terminal_latched_step"] is None:
                                summary["agile_command_hold_terminal_latched_step"] = int(step)
                        terminal_hold_scale_active = bool(terminal_threshold_reached) or bool(
                            agile_command_hold_terminal_latched
                        )
                        if agile_command_hold_active and terminal_hold_scale_active:
                            if (
                                bool(args_cli.cradle_chest_pad)
                                and bool(args_cli.cradle_chest_pad_enable_on_terminal_hold)
                                and not chest_pad_hold_collision_enabled
                            ):
                                try:
                                    if not _set_collision_enabled(stage, "/World/G1FrontCradle_chest_pad", True):
                                        raise RuntimeError("chest pad prim not found")
                                    summary["cradle_chest_pad_collision_enabled_step"] = int(step)
                                    summary["cradle_chest_pad_collision_update_count"] = (
                                        int(summary["cradle_chest_pad_collision_update_count"]) + 1
                                    )
                                    chest_pad_hold_collision_enabled = True
                                except Exception as exc:
                                    summary["cradle_chest_pad_collision_update_error"] = f"{type(exc).__name__}: {exc}"
                            command_scale = min(
                                float(command_scale),
                                max(0.0, float(args_cli.agile_command_hold_terminal_scale)),
                            )
                            summary["agile_command_hold_terminal_active_steps"] = (
                                int(summary["agile_command_hold_terminal_active_steps"]) + 1
                            )
                            if summary["agile_command_hold_terminal_first_active_step"] is None:
                                summary["agile_command_hold_terminal_first_active_step"] = int(step)
                            summary["agile_command_hold_terminal_last_reason"] = (
                                "box_target_travel_latched"
                                if bool(agile_command_hold_terminal_latched)
                                and not bool(terminal_threshold_reached)
                                else "box_target_travel"
                            )
                        final_threshold_reached = (
                            agile_command_hold_active
                            and float(args_cli.agile_command_hold_final_box_target_travel) >= 0.0
                            and float(args_cli.agile_command_hold_final_scale) >= 0.0
                            and float(prev_box_target_directed)
                            >= float(args_cli.agile_command_hold_final_box_target_travel)
                            and (
                                float(args_cli.agile_command_hold_final_min_robot_target_travel) < 0.0
                                or float(prev_robot_target_directed)
                                >= float(args_cli.agile_command_hold_final_min_robot_target_travel)
                            )
                            and (
                                int(args_cli.agile_command_hold_final_min_step) < 0
                                or int(step) >= int(args_cli.agile_command_hold_final_min_step)
                            )
                        )
                        if final_threshold_reached and bool(args_cli.agile_command_hold_final_latch):
                            agile_command_hold_final_latched = True
                            summary["agile_command_hold_final_latched"] = True
                            if summary["agile_command_hold_final_latched_step"] is None:
                                summary["agile_command_hold_final_latched_step"] = int(step)
                        final_hold_scale_active = bool(final_threshold_reached) or bool(
                            agile_command_hold_final_latched
                        )
                        current_final_hold_scale_active = bool(final_hold_scale_active)
                        if (
                            agile_command_hold_active
                            and final_hold_scale_active
                            and float(args_cli.agile_command_hold_final_scale) >= 0.0
                        ):
                            if (
                                bool(args_cli.cradle_chest_pad)
                                and bool(args_cli.cradle_chest_pad_enable_on_final_hold)
                                and not chest_pad_hold_collision_enabled
                            ):
                                try:
                                    if not _set_collision_enabled(stage, "/World/G1FrontCradle_chest_pad", True):
                                        raise RuntimeError("chest pad prim not found")
                                    summary["cradle_chest_pad_collision_enabled_step"] = int(step)
                                    summary["cradle_chest_pad_collision_update_count"] = (
                                        int(summary["cradle_chest_pad_collision_update_count"]) + 1
                                    )
                                    chest_pad_hold_collision_enabled = True
                                except Exception as exc:
                                    summary["cradle_chest_pad_collision_update_error"] = f"{type(exc).__name__}: {exc}"
                            command_scale = min(
                                float(command_scale),
                                max(0.0, float(args_cli.agile_command_hold_final_scale)),
                            )
                            summary["agile_command_hold_final_active_steps"] = (
                                int(summary["agile_command_hold_final_active_steps"]) + 1
                            )
                            if summary["agile_command_hold_final_first_active_step"] is None:
                                summary["agile_command_hold_final_first_active_step"] = int(step)
                            summary["agile_command_hold_final_last_reason"] = (
                                "box_target_travel_latched"
                                if bool(agile_command_hold_final_latched)
                                and not bool(final_threshold_reached)
                                else "box_target_travel"
                            )
                            if (
                                bool(args_cli.agile_command_hold_final_reset_policy_state)
                                and not agile_final_policy_state_reset_done
                                and agile_policy is not None
                            ):
                                try:
                                    agile_policy.reset_state()
                                    summary["agile_command_hold_final_policy_state_reset_count"] = (
                                        int(summary["agile_command_hold_final_policy_state_reset_count"]) + 1
                                    )
                                except Exception as exc:
                                    summary["agile_command_hold_final_last_policy_state_reset_error"] = (
                                        f"{type(exc).__name__}: {exc}"
                                    )
                                agile_final_policy_state_reset_done = True
                        applied_agile_command_list = [float(v) * command_scale for v in args_cli.agile_command]
                        if (
                            bool(args_cli.agile_command_box_progress_controller)
                            and int(step) >= int(args_cli.agile_command_box_progress_start_step)
                            and box_pose_for_feedback is not None
                            and initial_box is not None
                        ):
                            progress_target = (
                                float(args_cli.agile_command_box_progress_target)
                                if float(args_cli.agile_command_box_progress_target) >= 0.0
                                else float(args_cli.target_window_center)
                            )
                            if progress_target >= 0.0:
                                feedback_box_progress = _project_xy_delta(
                                    box_pose_for_feedback,
                                    initial_box,
                                    box_target_direction_xy,
                                )
                                progress_error = float(progress_target) - float(feedback_box_progress)
                                abs_error = max(
                                    0.0,
                                    abs(float(progress_error))
                                    - max(0.0, float(args_cli.agile_command_box_progress_deadband)),
                                )
                                progress_command_x = math.copysign(
                                    abs_error * float(args_cli.agile_command_box_progress_gain),
                                    float(progress_error),
                                )
                                progress_command_x = max(
                                    -abs(float(args_cli.agile_command_box_progress_max_reverse)),
                                    min(
                                        abs(float(args_cli.agile_command_box_progress_max_forward)),
                                        float(progress_command_x),
                                    ),
                                )
                                progress_tilt_allowed = (
                                    max(abs(float(feedback_roll)), abs(float(feedback_pitch)))
                                    <= max(0.0, float(args_cli.agile_command_box_progress_max_tilt))
                                    and float(box_feedback_tilt)
                                    <= max(0.0, float(args_cli.agile_command_box_progress_max_box_tilt))
                                )
                                if not progress_tilt_allowed and progress_command_x > 0.0:
                                    progress_command_x = 0.0
                                    summary["agile_command_box_progress_tilt_suppressed_steps"] = (
                                        int(summary["agile_command_box_progress_tilt_suppressed_steps"]) + 1
                                    )
                                if (
                                    bool(args_cli.agile_command_box_progress_scale_on_hold)
                                    and bool(agile_command_hold_active)
                                ):
                                    progress_command_x *= float(command_scale)
                                    summary["agile_command_box_progress_hold_scaled_steps"] = (
                                        int(summary["agile_command_box_progress_hold_scaled_steps"]) + 1
                                    )
                                applied_agile_command_list[0] = float(progress_command_x)
                                summary["agile_command_box_progress_active_steps"] = (
                                    int(summary["agile_command_box_progress_active_steps"]) + 1
                                )
                                if summary["agile_command_box_progress_first_active_step"] is None:
                                    summary["agile_command_box_progress_first_active_step"] = int(step)
                                summary["agile_command_box_progress_last_error_m"] = float(progress_error)
                                summary["agile_command_box_progress_last_command_x"] = float(progress_command_x)
                                summary["agile_command_box_progress_max_abs_command_x"] = max(
                                    float(summary["agile_command_box_progress_max_abs_command_x"]),
                                    abs(float(progress_command_x)),
                                )
                        if (
                            bool(args_cli.agile_command_box_lateral_controller)
                            and int(step) >= int(args_cli.agile_command_box_progress_start_step)
                            and box_pose_for_feedback is not None
                            and initial_box is not None
                        ):
                            box_lateral_error = _lateral_xy_delta(
                                box_pose_for_feedback,
                                initial_box,
                                box_target_direction_xy,
                            )
                            lateral_abs_error = max(
                                0.0,
                                abs(float(box_lateral_error))
                                - max(0.0, float(args_cli.agile_command_box_lateral_deadband)),
                            )
                            if lateral_abs_error > 0.0:
                                box_lateral_command = max(
                                    -abs(float(args_cli.agile_command_box_lateral_limit)),
                                    min(
                                        abs(float(args_cli.agile_command_box_lateral_limit)),
                                        -float(args_cli.agile_command_box_lateral_sign)
                                        * float(args_cli.agile_command_box_lateral_gain)
                                        * math.copysign(float(lateral_abs_error), float(box_lateral_error)),
                                    ),
                                )
                                if (
                                    bool(args_cli.agile_command_box_lateral_scale_on_hold)
                                    and bool(agile_command_hold_active)
                                ):
                                    box_lateral_command *= float(command_scale)
                                    summary["agile_command_box_lateral_hold_scaled_steps"] = (
                                        int(summary["agile_command_box_lateral_hold_scaled_steps"]) + 1
                                    )
                                applied_agile_command_list[1] += float(box_lateral_command)
                                summary["agile_command_box_lateral_active_steps"] = (
                                    int(summary["agile_command_box_lateral_active_steps"]) + 1
                                )
                                if summary["agile_command_box_lateral_first_active_step"] is None:
                                    summary["agile_command_box_lateral_first_active_step"] = int(step)
                                summary["agile_command_box_lateral_last_error_m"] = float(box_lateral_error)
                                summary["agile_command_box_lateral_last_command_y"] = float(box_lateral_command)
                                summary["agile_command_box_lateral_max_abs_command_y"] = max(
                                    float(summary["agile_command_box_lateral_max_abs_command_y"]),
                                    abs(float(box_lateral_command)),
                                )
                        hold_path_lateral_error = None
                        if agile_command_hold_active and (
                            bool(args_cli.agile_command_hold_lateral_correction)
                            or bool(args_cli.agile_command_hold_yaw_correction)
                        ):
                            hold_path_lateral_error = _lateral_xy_delta(
                                list(pose_for_feedback),
                                list(initial_robot),
                                robot_target_direction_xy,
                            )
                        lateral_correction_allowed = bool(args_cli.agile_command_hold_lateral_correction) and (
                            not bool(args_cli.agile_command_hold_lateral_terminal_only)
                            or bool(terminal_hold_scale_active)
                        )
                        if (
                            lateral_correction_allowed
                            and bool(args_cli.agile_command_hold_final_zero_corrections)
                            and bool(final_hold_scale_active)
                        ):
                            lateral_correction_allowed = False
                            summary["agile_command_hold_final_lateral_suppressed_steps"] = (
                                int(summary["agile_command_hold_final_lateral_suppressed_steps"]) + 1
                            )
                        if agile_command_hold_active and lateral_correction_allowed:
                            lateral_error = _lateral_xy_delta(
                                list(pose_for_feedback),
                                list(initial_robot),
                                robot_target_direction_xy,
                            ) if hold_path_lateral_error is None else float(hold_path_lateral_error)
                            lateral_correction_allowed = abs(float(lateral_error)) >= max(
                                0.0,
                                float(args_cli.agile_command_hold_lateral_error_start),
                            )
                            if lateral_correction_allowed:
                                lateral_robot_tilt = max(abs(float(feedback_roll)), abs(float(feedback_pitch)))
                                lateral_tilt_allowed = (
                                    float(lateral_robot_tilt)
                                    <= max(0.0, float(args_cli.agile_command_hold_lateral_max_tilt))
                                    and float(box_feedback_tilt)
                                    <= max(0.0, float(args_cli.agile_command_hold_lateral_max_box_tilt))
                                )
                                if not lateral_tilt_allowed:
                                    lateral_correction_allowed = False
                                    summary["agile_command_hold_lateral_suppressed_by_tilt_steps"] = (
                                        int(summary["agile_command_hold_lateral_suppressed_by_tilt_steps"]) + 1
                                    )
                        if agile_command_hold_active and lateral_correction_allowed:
                            lateral_command = max(
                                -abs(float(args_cli.agile_command_hold_lateral_limit)),
                                min(
                                    abs(float(args_cli.agile_command_hold_lateral_limit)),
                                    -float(args_cli.agile_command_hold_lateral_sign)
                                    * float(args_cli.agile_command_hold_lateral_gain)
                                    * (
                                        math.copysign(
                                            max(
                                                0.0,
                                                abs(float(lateral_error))
                                                - max(
                                                    0.0,
                                                    float(args_cli.agile_command_hold_lateral_error_start),
                                                ),
                                            ),
                                            float(lateral_error),
                                        )
                                        if bool(args_cli.agile_command_hold_lateral_use_excess_error)
                                        else float(lateral_error)
                                    ),
                                ),
                            )
                            applied_agile_command_list[1] += float(lateral_command)
                            summary["agile_command_hold_lateral_active_steps"] = (
                                int(summary["agile_command_hold_lateral_active_steps"]) + 1
                            )
                            if summary["agile_command_hold_lateral_first_active_step"] is None:
                                summary["agile_command_hold_lateral_first_active_step"] = int(step)
                            summary["agile_command_hold_lateral_max_abs_command"] = max(
                                float(summary["agile_command_hold_lateral_max_abs_command"]),
                                abs(float(lateral_command)),
                            )
                            summary["agile_command_hold_lateral_last_error_m"] = float(lateral_error)
                        yaw_correction_allowed = bool(args_cli.agile_command_hold_yaw_correction)
                        if (
                            yaw_correction_allowed
                            and bool(args_cli.agile_command_hold_final_zero_corrections)
                            and bool(final_hold_scale_active)
                        ):
                            yaw_correction_allowed = False
                            summary["agile_command_hold_final_yaw_suppressed_steps"] = (
                                int(summary["agile_command_hold_final_yaw_suppressed_steps"]) + 1
                            )
                        if agile_command_hold_active and yaw_correction_allowed:
                            yaw_error = _lateral_xy_delta(
                                list(pose_for_feedback),
                                list(initial_robot),
                                robot_target_direction_xy,
                            ) if hold_path_lateral_error is None else float(hold_path_lateral_error)
                            yaw_command = max(
                                -abs(float(args_cli.agile_command_hold_yaw_limit)),
                                min(
                                    abs(float(args_cli.agile_command_hold_yaw_limit)),
                                    -float(args_cli.agile_command_hold_yaw_sign)
                                    * float(args_cli.agile_command_hold_yaw_gain)
                                    * float(yaw_error),
                                ),
                            )
                            applied_agile_command_list[2] += float(yaw_command)
                            summary["agile_command_hold_yaw_active_steps"] = (
                                int(summary["agile_command_hold_yaw_active_steps"]) + 1
                            )
                            if summary["agile_command_hold_yaw_first_active_step"] is None:
                                summary["agile_command_hold_yaw_first_active_step"] = int(step)
                            summary["agile_command_hold_yaw_max_abs_command"] = max(
                                float(summary["agile_command_hold_yaw_max_abs_command"]),
                                abs(float(yaw_command)),
                            )
                            summary["agile_command_hold_yaw_last_error_m"] = float(yaw_error)
                        final_brake_active = False
                        final_hold_age_for_brake = None
                        if final_hold_scale_active and summary["agile_command_hold_final_first_active_step"] is not None:
                            final_hold_age_for_brake = (
                                int(step) - int(summary["agile_command_hold_final_first_active_step"])
                            )
                        if (
                            agile_command_hold_active
                            and final_hold_scale_active
                            and final_hold_age_for_brake is not None
                            and int(args_cli.agile_command_hold_final_brake_steps) > 0
                            and abs(float(args_cli.agile_command_hold_final_brake_command_x)) > 0.0
                        ):
                            brake_delay = max(0, int(args_cli.agile_command_hold_final_brake_delay_steps))
                            brake_steps = max(0, int(args_cli.agile_command_hold_final_brake_steps))
                            final_brake_active = (
                                int(final_hold_age_for_brake) >= brake_delay
                                and int(final_hold_age_for_brake) < brake_delay + brake_steps
                            )
                        if final_brake_active:
                            brake_command_x = float(args_cli.agile_command_hold_final_brake_command_x)
                            applied_agile_command_list[0] += brake_command_x
                            summary["agile_command_hold_final_brake_active_steps"] = (
                                int(summary["agile_command_hold_final_brake_active_steps"]) + 1
                            )
                            if summary["agile_command_hold_final_brake_first_active_step"] is None:
                                summary["agile_command_hold_final_brake_first_active_step"] = int(step)
                            summary["agile_command_hold_final_brake_last_active_step"] = int(step)
                            summary["agile_command_hold_final_brake_max_abs_command_x"] = max(
                                float(summary["agile_command_hold_final_brake_max_abs_command_x"]),
                                abs(float(brake_command_x)),
                            )
                        applied_agile_command = tuple(applied_agile_command_list)
                        summary["agile_last_command_xyz_yaw"] = [float(v) for v in applied_agile_command]
                        if bool(final_hold_scale_active):
                            summary["agile_command_hold_final_max_abs_command_x"] = max(
                                float(summary["agile_command_hold_final_max_abs_command_x"]),
                                abs(float(applied_agile_command[0])),
                            )
                            summary["agile_command_hold_final_max_abs_command_y"] = max(
                                float(summary["agile_command_hold_final_max_abs_command_y"]),
                                abs(float(applied_agile_command[1])),
                            )
                            summary["agile_command_hold_final_max_abs_command_yaw"] = max(
                                float(summary["agile_command_hold_final_max_abs_command_yaw"]),
                                abs(float(applied_agile_command[2])),
                            )
                            summary["agile_command_hold_final_last_command_xyz_yaw"] = [
                                float(v) for v in applied_agile_command
                            ]
                        agile_hold_age_steps = None
                        if agile_command_hold_active and summary["agile_command_hold_first_active_step"] is not None:
                            agile_hold_age_steps = int(step) - int(summary["agile_command_hold_first_active_step"])
                        hold_mode = str(args_cli.agile_command_hold_mode)
                        final_hold_age_steps = None
                        if final_hold_scale_active and summary["agile_command_hold_final_first_active_step"] is not None:
                            final_hold_age_steps = int(step) - int(summary["agile_command_hold_final_first_active_step"])
                        final_stand_active = (
                            agile_command_hold_active
                            and final_hold_scale_active
                            and bool(args_cli.agile_command_hold_final_stand)
                            and final_hold_age_steps is not None
                            and final_hold_age_steps >= max(
                                0, int(args_cli.agile_command_hold_final_stand_delay_steps)
                            )
                        )
                        current_final_stand_active = bool(final_stand_active)
                        if (
                            agile_command_hold_active
                            and final_hold_scale_active
                            and bool(args_cli.agile_command_hold_final_freeze_in_target_window)
                            and final_frozen_policy_joint_targets is None
                            and bool(summary.get("target_window_enabled"))
                        ):
                            window_center = float(args_cli.target_window_center)
                            window_halfwidth = float(args_cli.target_window_halfwidth)
                            prev_robot_in_window = abs(float(prev_robot_target_directed) - window_center) <= window_halfwidth
                            prev_box_in_window = abs(float(prev_box_target_directed) - window_center) <= window_halfwidth
                            freeze_tilt_allowed = (
                                max(abs(float(feedback_roll)), abs(float(feedback_pitch)))
                                <= max(0.0, float(args_cli.agile_command_hold_final_freeze_max_tilt))
                                and float(box_feedback_tilt)
                                <= max(0.0, float(args_cli.agile_command_hold_final_freeze_max_box_tilt))
                            )
                            if prev_robot_in_window and prev_box_in_window and freeze_tilt_allowed:
                                final_frozen_policy_joint_targets = np.asarray(policy_joint_targets, dtype=float).copy()
                                summary["agile_command_hold_final_freeze_latched"] = True
                                summary["agile_command_hold_final_freeze_latched_step"] = int(step)
                        policy_then_stand_active = (
                            agile_command_hold_active
                            and hold_mode == "policy_then_stand"
                            and agile_hold_age_steps is not None
                            and agile_hold_age_steps >= max(
                                0, int(args_cli.agile_command_hold_policy_then_stand_delay_steps)
                            )
                        )
                        if (
                            agile_command_hold_active
                            and bool(args_cli.agile_command_hold_rescue_enable)
                            and not agile_command_hold_rescue_active
                        ):
                            rescue_reasons = []
                            forward_pitch_threshold = float(args_cli.agile_command_hold_rescue_forward_pitch_threshold)
                            if forward_pitch_threshold > -998.0 and float(feedback_pitch) <= forward_pitch_threshold:
                                rescue_reasons.append("forward_pitch")
                            if abs(float(feedback_roll)) >= float(args_cli.agile_command_hold_rescue_abs_roll_threshold):
                                rescue_reasons.append("abs_roll")
                            if rescue_reasons:
                                agile_command_hold_rescue_active = True
                                agile_command_hold_rescue_first_reason = ",".join(rescue_reasons)
                                summary["agile_command_hold_rescue_active"] = True
                                summary["agile_command_hold_rescue_first_active_step"] = int(step)
                                summary["agile_command_hold_rescue_first_reason"] = agile_command_hold_rescue_first_reason
                        if agile_command_hold_rescue_active:
                            summary["agile_command_hold_rescue_active_steps"] = (
                                int(summary["agile_command_hold_rescue_active_steps"]) + 1
                            )
                        if final_stand_active:
                            summary["agile_command_hold_final_stand_active_steps"] = (
                                int(summary["agile_command_hold_final_stand_active_steps"]) + 1
                            )
                            if summary["agile_command_hold_final_stand_first_active_step"] is None:
                                summary["agile_command_hold_final_stand_first_active_step"] = int(step)
                        final_freeze_active = final_frozen_policy_joint_targets is not None
                        rescue_overrides_freeze_active = (
                            agile_command_hold_active
                            and final_freeze_active
                            and agile_command_hold_rescue_active
                            and bool(args_cli.agile_command_hold_rescue_overrides_final_freeze)
                        )
                        stand_overrides_freeze_active = (
                            agile_command_hold_active
                            and final_freeze_active
                            and final_stand_active
                            and bool(args_cli.agile_command_hold_stand_overrides_final_freeze)
                            and not rescue_overrides_freeze_active
                        )
                        if (
                            agile_command_hold_active
                            and final_freeze_active
                            and not rescue_overrides_freeze_active
                            and not stand_overrides_freeze_active
                        ):
                            policy_joint_targets = np.asarray(final_frozen_policy_joint_targets, dtype=float).copy()
                            summary["agile_command_hold_final_freeze_active_steps"] = (
                                int(summary["agile_command_hold_final_freeze_active_steps"]) + 1
                            )
                            if summary["agile_command_hold_final_freeze_first_active_step"] is None:
                                summary["agile_command_hold_final_freeze_first_active_step"] = int(step)
                        elif agile_command_hold_active and (
                            hold_mode == "stand_targets"
                            or policy_then_stand_active
                            or final_stand_active
                            or agile_command_hold_rescue_active
                        ):
                            if rescue_overrides_freeze_active:
                                summary["agile_command_hold_rescue_override_freeze_active_steps"] = (
                                    int(summary["agile_command_hold_rescue_override_freeze_active_steps"]) + 1
                                )
                                if summary["agile_command_hold_rescue_override_freeze_first_active_step"] is None:
                                    summary["agile_command_hold_rescue_override_freeze_first_active_step"] = int(step)
                            if stand_overrides_freeze_active:
                                summary["agile_command_hold_stand_override_freeze_active_steps"] = (
                                    int(summary["agile_command_hold_stand_override_freeze_active_steps"]) + 1
                                )
                                if summary["agile_command_hold_stand_override_freeze_first_active_step"] is None:
                                    summary["agile_command_hold_stand_override_freeze_first_active_step"] = int(step)
                            blend_rate = min(
                                1.0,
                                max(
                                    0.0,
                                    float(args_cli.agile_command_hold_rescue_blend_rate)
                                    if agile_command_hold_rescue_active
                                    else float(args_cli.agile_command_hold_stand_blend_rate),
                                ),
                            )
                            hold_target = (
                                np.asarray(hold_rescue_joint_targets, dtype=float)
                                if agile_command_hold_rescue_active
                                else np.asarray(stand_hold_joint_targets, dtype=float)
                            )
                            if stand_overrides_freeze_active:
                                hold_target = np.asarray(stand_hold_joint_targets, dtype=float)
                            policy_joint_targets = (
                                (1.0 - blend_rate) * np.asarray(policy_joint_targets, dtype=float)
                                + blend_rate * hold_target
                            )
                            summary["agile_command_hold_stand_target_active_steps"] = (
                                int(summary["agile_command_hold_stand_target_active_steps"]) + 1
                            )
                        elif (step - int(args_cli.policy_start_step)) % max(1, int(args_cli.policy_control_decimation)) == 0:
                            try:
                                joint_velocities = np.array(robot.get_joint_velocities(), dtype=float)
                            except Exception:
                                joint_velocities = np.zeros_like(joint_positions)
                            robot_pose_for_policy = _pose_wxyz(robot)
                            projected_gravity = _projected_gravity_body_from_quat_wxyz(
                                float(robot_pose_for_policy[3]),
                                float(robot_pose_for_policy[4]),
                                float(robot_pose_for_policy[5]),
                                float(robot_pose_for_policy[6]),
                            )
                            root_ang_vel_b, root_ang_vel_error = _root_angular_velocity_body(robot, robot_pose_for_policy)
                            if root_ang_vel_error is not None:
                                summary["agile_root_ang_vel_read_failures"] = int(summary.get("agile_root_ang_vel_read_failures") or 0) + 1
                                summary["agile_last_root_ang_vel_read_error"] = str(root_ang_vel_error)
                            summary["max_agile_root_ang_vel_norm"] = max(
                                float(summary["max_agile_root_ang_vel_norm"]),
                                float(np.linalg.norm(root_ang_vel_b)),
                            )
                            target_by_name = agile_policy.infer(
                                applied_agile_command,
                                float(args_cli.agile_height_command),
                                np.array(robot.get_joint_positions(), dtype=float),
                                joint_velocities,
                                joint_names,
                                projected_gravity,
                                root_ang_vel_b,
                            )
                            for name, value in target_by_name.items():
                                _command_joint_position(policy_joint_targets, joint_names, name, value)
                            policy_inference_count += 1
                            summary["policy_inference_count"] = policy_inference_count
                            summary["max_policy_raw_action_norm"] = max(
                                float(summary["max_policy_raw_action_norm"]),
                                float(agile_policy.last_raw_action_norm),
                            )
                        command_positions = policy_joint_targets
                    else:
                        if not terminal_hold_active:
                            terminal_reasons = []
                            if int(args_cli.terminal_hold_start_step) >= 0 and int(step) >= int(args_cli.terminal_hold_start_step):
                                terminal_reasons.append("start_step")
                            if (
                                float(args_cli.terminal_hold_box_target_travel) >= 0.0
                                and float(prev_box_target_directed) >= float(args_cli.terminal_hold_box_target_travel)
                            ):
                                terminal_reasons.append("box_target_travel")
                            if (
                                float(args_cli.terminal_hold_robot_target_travel) >= 0.0
                                and float(prev_robot_target_directed) >= float(args_cli.terminal_hold_robot_target_travel)
                            ):
                                terminal_reasons.append("robot_target_travel")
                            if abs(float(feedback_pitch)) >= float(args_cli.terminal_hold_pitch_threshold):
                                terminal_reasons.append("pitch")
                            if abs(float(feedback_pitch_rate)) >= float(args_cli.terminal_hold_pitch_rate_threshold):
                                terminal_reasons.append("pitch_rate")
                            if terminal_reasons:
                                terminal_hold_active = True
                                terminal_hold_first_reason = ",".join(terminal_reasons)
                                summary["terminal_hold_first_active_step"] = int(step)
                                summary["terminal_hold_first_reason"] = terminal_hold_first_reason
                                if (
                                    not terminal_drive_applied
                                    and float(args_cli.terminal_drive_gain_scale) > 0.0
                                    and float(args_cli.terminal_drive_force_scale) > 0.0
                                ):
                                    terminal_drive_gains = _set_stand_drive_gains(
                                        stage,
                                        float(args_cli.terminal_drive_gain_scale),
                                        float(args_cli.terminal_drive_force_scale),
                                    )
                                    summary["terminal_drive_gain_applied_step"] = int(step)
                                    summary["terminal_applied_drive_gain_count"] = len(terminal_drive_gains)
                                    summary["terminal_applied_stand_drive_gains"] = terminal_drive_gains
                                    terminal_drive_applied = True
                        if terminal_hold_active:
                            summary["terminal_hold_active_steps"] = int(summary["terminal_hold_active_steps"]) + 1
                        if (
                            bool(args_cli.creep_pitch_brake_latch)
                            and str(args_cli.gait_mode) == "targeted_creep"
                            and not creep_pitch_brake_latched
                            and (
                                (
                                    float(feedback_pitch)
                                    if bool(args_cli.creep_pitch_brake_positive_only)
                                    else abs(float(feedback_pitch))
                                )
                                >= float(args_cli.creep_pitch_brake_threshold)
                                or abs(float(feedback_pitch_rate)) >= float(args_cli.creep_pitch_brake_rate_threshold)
                            )
                        ):
                            creep_pitch_brake_latched = True
                            summary["creep_pitch_brake_latched_step"] = int(step)
                        if str(args_cli.gait_mode) == "targeted_creep" and not creep_reverse_brake_latched:
                            reverse_reasons = []
                            if (
                                float(args_cli.creep_reverse_brake_box_travel) >= 0.0
                                and float(prev_box_target_directed) >= float(args_cli.creep_reverse_brake_box_travel)
                            ):
                                reverse_reasons.append("box_target_travel")
                            if (
                                float(args_cli.creep_reverse_brake_robot_travel) >= 0.0
                                and float(prev_robot_target_directed) >= float(args_cli.creep_reverse_brake_robot_travel)
                            ):
                                reverse_reasons.append("robot_target_travel")
                            reverse_pitch_value = (
                                float(feedback_pitch)
                                if bool(args_cli.creep_reverse_brake_positive_pitch_only)
                                else abs(float(feedback_pitch))
                            )
                            if float(reverse_pitch_value) >= float(args_cli.creep_reverse_brake_pitch_threshold):
                                reverse_reasons.append("pitch")
                            if reverse_reasons:
                                creep_reverse_brake_latched = True
                                creep_reverse_brake_latched_step = int(step)
                                summary["creep_reverse_brake_latched_step"] = int(step)
                                summary["creep_reverse_brake_first_reason"] = ",".join(reverse_reasons)
                        creep_reverse_brake_active = bool(creep_reverse_brake_latched)
                        if (
                            creep_reverse_brake_active
                            and creep_reverse_brake_latched_step is not None
                            and int(args_cli.creep_reverse_brake_duration_steps) >= 0
                            and int(step) - int(creep_reverse_brake_latched_step) >= int(args_cli.creep_reverse_brake_duration_steps)
                        ):
                            creep_reverse_brake_active = False
                        command_positions, gait_diag = _gait_joint_positions(
                            joint_positions,
                            joint_names,
                            time_s,
                            step,
                            feedback_pitch,
                            feedback_pitch_rate,
                            prev_robot_target_directed,
                            prev_box_target_directed,
                            creep_pitch_brake_latched,
                            creep_reverse_brake_active,
                            terminal_hold_active,
                        )
                        if creep_reverse_brake_active:
                            summary["creep_reverse_brake_active_steps"] = int(summary["creep_reverse_brake_active_steps"]) + 1
                            if summary["creep_reverse_brake_first_active_step"] is None:
                                summary["creep_reverse_brake_first_active_step"] = int(step)
                        if bool(gait_diag.get("creep_decel_active")):
                            summary["creep_decel_active_steps"] = int(summary["creep_decel_active_steps"]) + 1
                            if summary["creep_decel_first_active_step"] is None:
                                summary["creep_decel_first_active_step"] = int(step)
                        if bool(gait_diag.get("creep_pitch_brake_active")):
                            summary["creep_pitch_brake_active_steps"] = int(summary["creep_pitch_brake_active_steps"]) + 1
                            if summary["creep_pitch_brake_first_active_step"] is None:
                                summary["creep_pitch_brake_first_active_step"] = int(step)
                        summary["min_creep_amplitude_scale"] = min(
                            float(summary["min_creep_amplitude_scale"]),
                            float(
                                gait_diag["creep_amplitude_scale"]
                                if gait_diag.get("creep_amplitude_scale") is not None
                                else 1.0
                            ),
                        )
                        summary["min_creep_push_scale"] = min(
                            float(summary["min_creep_push_scale"]),
                            float(gait_diag["creep_push_scale"] if gait_diag.get("creep_push_scale") is not None else 1.0),
                        )
                        summary["min_creep_bias_scale"] = min(
                            float(summary["min_creep_bias_scale"]),
                            float(gait_diag["creep_bias_scale"] if gait_diag.get("creep_bias_scale") is not None else 1.0),
                        )
                        recovery_active = (
                            str(args_cli.gait_mode) == "staged_march"
                            and (
                                abs(float(feedback_pitch)) >= float(args_cli.recovery_pitch_threshold)
                                or abs(float(feedback_pitch_rate)) >= float(args_cli.recovery_pitch_rate_threshold)
                            )
                        )
                        if recovery_active:
                            summary["recovery_active_steps"] = int(summary["recovery_active_steps"]) + 1
                            if summary["recovery_first_active_step"] is None:
                                summary["recovery_first_active_step"] = int(step)
                    arm_pose_active = _apply_arm_pose_targets(command_positions, joint_names, step, arm_pose_targets)
                    if arm_pose_active:
                        summary["arm_pose_active_steps"] = int(summary["arm_pose_active_steps"]) + 1
                        if summary["arm_pose_first_active_step"] is None:
                            summary["arm_pose_first_active_step"] = int(step)
                    feedback_box_robot_rel_error = float(prev_box_robot_rel_error)
                    if box_pose_for_feedback is not None and initial_box_robot_rel is not None:
                        feedback_rel_now = (
                            float(box_pose_for_feedback[0]) - float(pose_for_feedback[0]),
                            float(box_pose_for_feedback[1]) - float(pose_for_feedback[1]),
                            float(box_pose_for_feedback[2]) - float(pose_for_feedback[2]),
                        )
                        feedback_box_robot_rel_error = float(
                            math.sqrt(
                                (feedback_rel_now[0] - initial_box_robot_rel[0]) ** 2
                                + (feedback_rel_now[1] - initial_box_robot_rel[1]) ** 2
                                + (feedback_rel_now[2] - initial_box_robot_rel[2]) ** 2
                            )
                        )
                    retention_risk = _apply_box_retention_posture_feedback(
                        command_positions,
                        joint_names,
                        float(box_feedback_tilt),
                        float(feedback_box_robot_rel_error),
                    )
                    if retention_risk > 0.0:
                        summary["box_retention_active_steps"] = int(summary["box_retention_active_steps"]) + 1
                        if summary["box_retention_first_active_step"] is None:
                            summary["box_retention_first_active_step"] = int(step)
                        summary["box_retention_last_risk"] = float(retention_risk)
                        summary["box_retention_max_risk"] = max(
                            float(summary["box_retention_max_risk"]),
                            float(retention_risk),
                        )
                    balance_allowed = not bool(args_cli.balance_start_on_agile_hold) or bool(agile_command_hold_active)
                    if balance_allowed:
                        dynamic_balance_roll_target = None
                        if bool(args_cli.balance_roll_target_from_lateral):
                            robot_lateral_error = _lateral_xy_delta(
                                list(pose_for_feedback),
                                list(initial_robot),
                                robot_target_direction_xy,
                            )
                            box_lateral_error = robot_lateral_error
                            if box_pose_for_feedback is not None and initial_box is not None:
                                box_lateral_error = _lateral_xy_delta(
                                    list(box_pose_for_feedback),
                                    list(initial_box),
                                    box_target_direction_xy,
                                )
                            lateral_source = str(args_cli.balance_roll_target_lateral_source)
                            if lateral_source == "box":
                                balance_lateral_error = float(box_lateral_error)
                            elif lateral_source == "average":
                                balance_lateral_error = 0.5 * (
                                    float(robot_lateral_error) + float(box_lateral_error)
                                )
                            else:
                                balance_lateral_error = float(robot_lateral_error)
                            deadband = max(0.0, float(args_cli.balance_roll_target_lateral_deadband))
                            lateral_excess = max(0.0, abs(float(balance_lateral_error)) - deadband)
                            target_limit = abs(float(args_cli.balance_roll_target_lateral_limit))
                            hold_delay = max(
                                0,
                                int(args_cli.balance_roll_target_lateral_start_after_hold_steps),
                            )
                            hold_first_step = summary.get("agile_command_hold_first_active_step")
                            hold_elapsed = (
                                int(step) - int(hold_first_step)
                                if hold_first_step is not None
                                else 0
                            )
                            lateral_target_allowed = True
                            if hold_delay > 0 and (not agile_command_hold_active or hold_elapsed < hold_delay):
                                lateral_target_allowed = False
                                summary["balance_roll_target_lateral_suppressed_by_hold_delay_steps"] = (
                                    int(summary["balance_roll_target_lateral_suppressed_by_hold_delay_steps"]) + 1
                                )
                            lateral_tilt = max(abs(float(feedback_roll)), abs(float(feedback_pitch)))
                            if (
                                float(lateral_tilt) > max(0.0, float(args_cli.balance_roll_target_lateral_max_tilt))
                                or float(box_feedback_tilt)
                                > max(0.0, float(args_cli.balance_roll_target_lateral_max_box_tilt))
                            ):
                                lateral_target_allowed = False
                                summary["balance_roll_target_lateral_suppressed_by_tilt_steps"] = (
                                    int(summary["balance_roll_target_lateral_suppressed_by_tilt_steps"]) + 1
                                )
                            ramp_steps = max(0, int(args_cli.balance_roll_target_lateral_ramp_steps))
                            ramp_scale = 1.0
                            if ramp_steps > 0:
                                ramp_scale = max(0.0, min(1.0, float(hold_elapsed) / float(ramp_steps)))
                            dynamic_balance_roll_target = max(
                                -target_limit,
                                min(
                                    target_limit,
                                    float(args_cli.balance_roll_target_lateral_sign)
                                    * float(args_cli.balance_roll_target_lateral_gain)
                                    * math.copysign(float(lateral_excess), float(balance_lateral_error)),
                                ),
                            )
                            dynamic_balance_roll_target *= float(ramp_scale)
                            if not lateral_target_allowed:
                                dynamic_balance_roll_target = 0.0
                            if abs(float(dynamic_balance_roll_target)) > 0.0:
                                summary["balance_roll_target_lateral_active_steps"] = (
                                    int(summary["balance_roll_target_lateral_active_steps"]) + 1
                                )
                                if summary["balance_roll_target_lateral_first_active_step"] is None:
                                    summary["balance_roll_target_lateral_first_active_step"] = int(step)
                            summary["balance_roll_target_lateral_last_error_m"] = float(balance_lateral_error)
                            summary["balance_roll_target_lateral_last_target_rad"] = float(dynamic_balance_roll_target)
                            summary["balance_roll_target_lateral_max_abs_target_rad"] = max(
                                float(summary["balance_roll_target_lateral_max_abs_target_rad"]),
                                abs(float(dynamic_balance_roll_target)),
                            )
                        command_positions, balance_active = _apply_balance_feedback(
                            command_positions,
                            joint_names,
                            step,
                            feedback_roll,
                            feedback_pitch,
                            feedback_roll_rate,
                            feedback_pitch_rate,
                            roll_target_override=dynamic_balance_roll_target,
                        )
                    else:
                        balance_active = False
                    if _balance_target_active(step):
                        summary["balance_target_active_steps"] = int(summary["balance_target_active_steps"]) + 1
                        if summary["balance_target_first_active_step"] is None:
                            summary["balance_target_first_active_step"] = int(step)
                    if balance_active:
                        summary["balance_feedback_active_steps"] = int(summary["balance_feedback_active_steps"]) + 1
                        if summary["balance_feedback_first_active_step"] is None:
                            summary["balance_feedback_first_active_step"] = int(step)
                    robot.apply_action(ArticulationAction(joint_positions=command_positions.tolist()))
                if str(args_cli.diagnostic_root_drive) == "smooth_x" and float(args_cli.diagnostic_root_drive_speed) > 0.0:
                    drive_start = int(args_cli.diagnostic_root_drive_start_step)
                    drive_stop = int(args_cli.diagnostic_root_drive_stop_step)
                    if int(step) >= drive_start and (drive_stop < 0 or int(step) < drive_stop):
                        if diagnostic_root_drive_position is None or diagnostic_root_drive_orientation is None:
                            current_pose = _pose_wxyz(robot)
                            diagnostic_root_drive_position = np.array(current_pose[:3], dtype=float)
                            diagnostic_root_drive_orientation = np.array(current_pose[3:7], dtype=float)
                        ramp_steps = max(1, int(args_cli.diagnostic_root_drive_ramp_steps))
                        ramp_alpha = min(1.0, max(0.0, (int(step) - drive_start + 1) / float(ramp_steps)))
                        distance = float(args_cli.diagnostic_root_drive_speed) * ramp_alpha * 0.005
                        diagnostic_root_drive_position[0] += float(diagnostic_root_drive_direction[0]) * distance
                        diagnostic_root_drive_position[1] += float(diagnostic_root_drive_direction[1]) * distance
                        robot.set_world_pose(
                            position=diagnostic_root_drive_position,
                            orientation=diagnostic_root_drive_orientation,
                        )
                        summary["root_pose_write_count_rollout"] = int(summary["root_pose_write_count_rollout"]) + 1
                        summary["diagnostic_root_drive_active_steps"] = int(summary["diagnostic_root_drive_active_steps"]) + 1
                        summary["diagnostic_root_drive_final_commanded_xy_m"] = [
                            float(diagnostic_root_drive_position[0]) - float(initial_robot[0]),
                            float(diagnostic_root_drive_position[1]) - float(initial_robot[1]),
                        ]
                if (
                    str(args_cli.box_support_mode) == "table"
                    and int(args_cli.box_support_release_step) >= 0
                    and not box_support_released
                    and step >= int(args_cli.box_support_release_step)
                ):
                    if stage.GetPrimAtPath(BOX_SUPPORT_TABLE_PATH).IsValid():
                        stage.RemovePrim(BOX_SUPPORT_TABLE_PATH)
                    box_support_released = True
                    summary["box_support_released"] = True
                    summary["box_support_actual_release_step"] = int(step)
                    print(f"[EVENT] removed box support table step={step} path={BOX_SUPPORT_TABLE_PATH}", flush=True)
                world.step(render=bool(args_cli.render) or bool(args_cli.capture_rgb))
                if (
                    capture_rep is not None
                    and int(args_cli.capture_rgb_every_n_steps) > 0
                    and int(step) % int(args_cli.capture_rgb_every_n_steps) == 0
                ):
                    try:
                        _run_async(
                            capture_rep.orchestrator.step_async(
                                rt_subframes=max(1, int(args_cli.capture_rgb_rt_subframes)),
                                delta_time=0.0,
                            )
                        )
                        summary["capture_rgb_frame_count"] = int(summary["capture_rgb_frame_count"]) + 1
                    except Exception as exc:
                        summary["capture_rgb_error"] = f"{type(exc).__name__}: {exc}"
                        print(f"[WARN] RGB capture step failed: {summary['capture_rgb_error']}", flush=True)
                        capture_rep = None
                robot_pose = _pose_wxyz(robot)
                box_pose = _pose_wxyz(box) if box is not None else None
                if (
                    str(args_cli.grasp_mode) in ("staged_fixed_torso", "staged_fixed_body")
                    and not grasp_attached
                    and box_pose is not None
                    and step >= int(args_cli.grasp_enable_step)
                ):
                    if active_grasp_body_path is None:
                        raise RuntimeError(f"grasp mode {args_cli.grasp_mode} has no active grasp body path")
                    if grasp_body is not None:
                        grasp_body_pose = _pose_wxyz(grasp_body)
                    else:
                        grasp_body_pose = _usd_world_pose_wxyz(stage, active_grasp_body_path)
                    if grasp_body_pose is None:
                        raise RuntimeError(f"could not read grasp body pose: {active_grasp_body_path}")
                    grasp_joint_path = f"{BOX_PATH}/StagedFixedBodyGraspJoint"
                    body_box_world_delta = (
                        float(box_pose[0]) - float(grasp_body_pose[0]),
                        float(box_pose[1]) - float(grasp_body_pose[1]),
                        float(box_pose[2]) - float(grasp_body_pose[2]),
                    )
                    body_box_world_distance = math.sqrt(sum(float(v) * float(v) for v in body_box_world_delta))
                    local_pos0 = _world_delta_as_body_local(
                        grasp_body_pose,
                        box_pose,
                        float(args_cli.grasp_lift_offset_z),
                    )
                    _fixed_joint(
                        stage,
                        grasp_joint_path,
                        active_grasp_body_path,
                        BOX_PATH,
                        local_pos0,
                        (0.0, 0.0, 0.0),
                    )
                    grasp_attached = True
                    grasp_box_z_at_attach = float(box_pose[2])
                    summary["grasp_attached"] = True
                    summary["grasp_attach_step"] = int(step)
                    summary["grasp_joint_path"] = grasp_joint_path
                    summary["grasp_local_pos0_m"] = [float(v) for v in local_pos0]
                    summary["grasp_box_pose_at_attach_wxyz"] = list(box_pose)
                    summary["grasp_robot_pose_at_attach_wxyz"] = list(robot_pose)
                    summary["grasp_body_pose_at_attach_wxyz"] = list(grasp_body_pose)
                    summary["grasp_body_box_world_delta_at_attach_m"] = [float(v) for v in body_box_world_delta]
                    summary["grasp_body_box_world_distance_at_attach_m"] = float(body_box_world_distance)
                    summary["grasp_box_z_at_attach_m"] = float(grasp_box_z_at_attach)
                    summary["min_post_grasp_box_z_m"] = float(box_pose[2])
                    print(
                        "[EVENT] staged fixed-body grasp attach "
                        f"step={step} body={active_grasp_body_path} local_pos0={local_pos0}",
                        flush=True,
                    )
                if (
                    str(args_cli.probe_mode) != "none"
                    and box_pose is not None
                    and step >= int(args_cli.probe_start_step)
                    and probe_reference_box_pose is None
                ):
                    probe_reference_box_pose = list(box_pose)
                    probe_reference_robot_pose = list(robot_pose)
                    summary["probe_reference_step"] = int(step)
                    summary["probe_reference_box_pose_wxyz"] = list(box_pose)
                    summary["probe_reference_robot_pose_wxyz"] = list(robot_pose)
                if (
                    str(args_cli.probe_mode) != "none"
                    and bool(args_cli.probe_collision_window)
                    and not bool(args_cli.disable_probe_pad_collision)
                    and not probe_collision_active
                    and step >= int(args_cli.probe_start_step)
                    and (int(args_cli.probe_end_step) < 0 or step < int(args_cli.probe_end_step))
                ):
                    try:
                        if not _set_collision_enabled(stage, "/World/G1FrontProbePad", True):
                            raise RuntimeError("probe pad prim not found")
                        probe_collision_active = True
                        summary["probe_pad_collision_enabled"] = True
                        summary["probe_collision_enabled_step"] = int(step)
                        summary["probe_collision_update_count"] = (
                            int(summary["probe_collision_update_count"]) + 1
                        )
                    except Exception as exc:
                        summary["probe_collision_update_error"] = f"{type(exc).__name__}: {exc}"
                if (
                    str(args_cli.probe_mode) != "none"
                    and bool(args_cli.probe_collision_window)
                    and probe_collision_active
                    and int(args_cli.probe_end_step) >= 0
                    and step >= int(args_cli.probe_end_step)
                ):
                    try:
                        if not _set_collision_enabled(stage, "/World/G1FrontProbePad", False):
                            raise RuntimeError("probe pad prim not found")
                        probe_collision_active = False
                        summary["probe_pad_collision_enabled"] = False
                        summary["probe_collision_disabled_step"] = int(step)
                        summary["probe_collision_update_count"] = (
                            int(summary["probe_collision_update_count"]) + 1
                        )
                    except Exception as exc:
                        summary["probe_collision_update_error"] = f"{type(exc).__name__}: {exc}"
                roll, pitch = _quat_to_roll_pitch(float(robot_pose[3]), float(robot_pose[4]), float(robot_pose[5]), float(robot_pose[6]))
                tilt = float(max(abs(roll), abs(pitch)))
                box_roll = 0.0
                box_pitch = 0.0
                box_tilt = 0.0
                if box_pose is not None:
                    box_roll, box_pitch = _quat_to_roll_pitch(
                        float(box_pose[3]),
                        float(box_pose[4]),
                        float(box_pose[5]),
                        float(box_pose[6]),
                    )
                    box_tilt = float(max(abs(box_roll), abs(box_pitch)))
                robot_travel = float(math.hypot(float(robot_pose[0]) - float(initial_robot[0]), float(robot_pose[1]) - float(initial_robot[1])))
                robot_target_directed = _project_xy_delta(robot_pose, initial_robot, robot_target_direction_xy)
                box_travel = (
                    float(math.hypot(float(box_pose[0]) - float(initial_box[0]), float(box_pose[1]) - float(initial_box[1])))
                    if box_pose is not None and initial_box is not None
                    else 0.0
                )
                box_target_directed = (
                    _project_xy_delta(box_pose, initial_box, box_target_direction_xy)
                    if box_pose is not None and initial_box is not None
                    else 0.0
                )
                robot_target_lateral_error = _lateral_xy_delta(
                    robot_pose,
                    initial_robot,
                    robot_target_direction_xy,
                )
                box_target_lateral_error = (
                    _lateral_xy_delta(box_pose, initial_box, box_target_direction_xy)
                    if box_pose is not None and initial_box is not None
                    else 0.0
                )
                box_robot_rel_error = 0.0
                if box_pose is not None and initial_box_robot_rel is not None:
                    rel_now = (
                        float(box_pose[0]) - float(robot_pose[0]),
                        float(box_pose[1]) - float(robot_pose[1]),
                        float(box_pose[2]) - float(robot_pose[2]),
                    )
                    box_robot_rel_error = float(
                        math.sqrt(
                            (rel_now[0] - initial_box_robot_rel[0]) ** 2
                            + (rel_now[1] - initial_box_robot_rel[1]) ** 2
                            + (rel_now[2] - initial_box_robot_rel[2]) ** 2
                        )
                    )
                probe_window_active = probe_reference_box_pose is not None and (
                    int(args_cli.probe_end_step) < 0 or step < int(args_cli.probe_end_step)
                )
                if probe_window_active and box_pose is not None:
                    summary["probe_active_steps"] = int(summary["probe_active_steps"]) + 1
                    probe_box_travel = float(
                        math.hypot(
                            float(box_pose[0]) - float(probe_reference_box_pose[0]),
                            float(box_pose[1]) - float(probe_reference_box_pose[1]),
                        )
                    )
                    probe_box_target_directed = _project_xy_delta(
                        box_pose,
                        probe_reference_box_pose,
                        box_target_direction_xy,
                    )
                    summary["max_probe_box_travel_xy_m"] = max(
                        float(summary["max_probe_box_travel_xy_m"]),
                        probe_box_travel,
                    )
                    summary["max_probe_box_target_directed_travel_m"] = max(
                        float(summary["max_probe_box_target_directed_travel_m"]),
                        float(probe_box_target_directed),
                    )
                    summary["final_probe_box_travel_xy_m"] = probe_box_travel
                    summary["final_probe_box_target_directed_travel_m"] = float(probe_box_target_directed)
                    displacement = [
                        float(box_pose[0]) - float(probe_reference_box_pose[0]),
                        float(box_pose[1]) - float(probe_reference_box_pose[1]),
                        float(box_pose[2]) - float(probe_reference_box_pose[2]),
                    ]
                    summary["probe_box_displacement_xyz_m"] = displacement
                    summary["probe_box_moved"] = bool(probe_box_travel > 0.002)
                fall = int(float(robot_pose[2]) < float(args_cli.fall_z) or tilt > 0.85)
                drop = int(float(box_pose[2]) < float(args_cli.drop_z)) if box_pose is not None else 0
                summary["completed_steps"] = step + 1
                summary["fall_events"] += fall
                summary["box_drop_events"] += drop
                if bool(current_final_hold_scale_active):
                    summary["agile_command_hold_final_fall_events"] = (
                        int(summary["agile_command_hold_final_fall_events"]) + int(fall)
                    )
                    summary["agile_command_hold_final_box_drop_events"] = (
                        int(summary["agile_command_hold_final_box_drop_events"]) + int(drop)
                    )
                    if fall and summary["agile_command_hold_final_first_fall_step"] is None:
                        summary["agile_command_hold_final_first_fall_step"] = int(step)
                    if drop and summary["agile_command_hold_final_first_box_drop_step"] is None:
                        summary["agile_command_hold_final_first_box_drop_step"] = int(step)
                if bool(current_final_stand_active):
                    summary["agile_command_hold_final_stand_fall_events"] = (
                        int(summary["agile_command_hold_final_stand_fall_events"]) + int(fall)
                    )
                    summary["agile_command_hold_final_stand_box_drop_events"] = (
                        int(summary["agile_command_hold_final_stand_box_drop_events"]) + int(drop)
                    )
                    if fall and summary["agile_command_hold_final_stand_first_fall_step"] is None:
                        summary["agile_command_hold_final_stand_first_fall_step"] = int(step)
                    if drop and summary["agile_command_hold_final_stand_first_box_drop_step"] is None:
                        summary["agile_command_hold_final_stand_first_box_drop_step"] = int(step)
                if fall and summary["first_fall_step"] is None:
                    summary["first_fall_step"] = int(step)
                    summary["first_fall_time_s"] = float(time_s)
                if drop and summary["first_box_drop_step"] is None:
                    summary["first_box_drop_step"] = int(step)
                    summary["first_box_drop_time_s"] = float(time_s)
                summary["max_robot_travel_xy_m"] = max(float(summary["max_robot_travel_xy_m"]), robot_travel)
                summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), box_travel)
                summary["max_robot_target_directed_travel_m"] = max(
                    float(summary["max_robot_target_directed_travel_m"]),
                    float(robot_target_directed),
                )
                summary["max_box_target_directed_travel_m"] = max(
                    float(summary["max_box_target_directed_travel_m"]),
                    float(box_target_directed),
                )
                summary["final_robot_travel_xy_m"] = robot_travel
                summary["final_box_travel_xy_m"] = box_travel
                summary["final_robot_target_directed_travel_m"] = float(robot_target_directed)
                summary["final_box_target_directed_travel_m"] = float(box_target_directed)
                summary["max_abs_robot_target_lateral_error_m"] = max(
                    float(summary["max_abs_robot_target_lateral_error_m"]),
                    abs(float(robot_target_lateral_error)),
                )
                summary["max_abs_box_target_lateral_error_m"] = max(
                    float(summary["max_abs_box_target_lateral_error_m"]),
                    abs(float(box_target_lateral_error)),
                )
                summary["final_robot_target_lateral_error_m"] = float(robot_target_lateral_error)
                summary["final_box_target_lateral_error_m"] = (
                    float(box_target_lateral_error) if box_pose is not None else None
                )
                if bool(summary["target_window_enabled"]):
                    target_center = float(args_cli.target_window_center)
                    target_halfwidth = float(args_cli.target_window_halfwidth)
                    stable_step = not bool(fall) and not bool(drop)
                    robot_in_window = (
                        stable_step
                        and abs(float(robot_target_directed) - target_center) <= target_halfwidth
                    )
                    box_in_window = (
                        stable_step
                        and box_pose is not None
                        and abs(float(box_target_directed) - target_center) <= target_halfwidth
                    )
                    both_in_window = bool(robot_in_window and box_in_window)

                    if robot_in_window:
                        target_window_robot_streak += 1
                        summary["target_window_robot_stable_steps"] = (
                            int(summary["target_window_robot_stable_steps"]) + 1
                        )
                        if summary["target_window_robot_first_stable_step"] is None:
                            summary["target_window_robot_first_stable_step"] = int(step)
                    else:
                        target_window_robot_streak = 0
                    if box_in_window:
                        target_window_box_streak += 1
                        summary["target_window_box_stable_steps"] = (
                            int(summary["target_window_box_stable_steps"]) + 1
                        )
                        if summary["target_window_box_first_stable_step"] is None:
                            summary["target_window_box_first_stable_step"] = int(step)
                    else:
                        target_window_box_streak = 0
                    if both_in_window:
                        target_window_both_streak += 1
                        summary["target_window_both_stable_steps"] = (
                            int(summary["target_window_both_stable_steps"]) + 1
                        )
                        if summary["target_window_both_first_stable_step"] is None:
                            summary["target_window_both_first_stable_step"] = int(step)
                    else:
                        target_window_both_streak = 0
                    if both_in_window and bool(current_final_hold_scale_active):
                        target_window_both_final_hold_streak += 1
                        summary["target_window_both_final_hold_stable_steps"] = (
                            int(summary["target_window_both_final_hold_stable_steps"]) + 1
                        )
                        if summary["target_window_both_final_hold_first_stable_step"] is None:
                            summary["target_window_both_final_hold_first_stable_step"] = int(step)
                    else:
                        target_window_both_final_hold_streak = 0
                    if both_in_window and bool(current_final_stand_active):
                        target_window_both_final_stand_streak += 1
                        summary["target_window_both_final_stand_stable_steps"] = (
                            int(summary["target_window_both_final_stand_stable_steps"]) + 1
                        )
                        if summary["target_window_both_final_stand_first_stable_step"] is None:
                            summary["target_window_both_final_stand_first_stable_step"] = int(step)
                    else:
                        target_window_both_final_stand_streak = 0

                    summary["target_window_robot_longest_streak_steps"] = max(
                        int(summary["target_window_robot_longest_streak_steps"]),
                        int(target_window_robot_streak),
                    )
                    summary["target_window_box_longest_streak_steps"] = max(
                        int(summary["target_window_box_longest_streak_steps"]),
                        int(target_window_box_streak),
                    )
                    summary["target_window_both_longest_streak_steps"] = max(
                        int(summary["target_window_both_longest_streak_steps"]),
                        int(target_window_both_streak),
                    )
                    summary["target_window_both_streak_at_end_steps"] = int(target_window_both_streak)
                    summary["target_window_both_final_hold_longest_streak_steps"] = max(
                        int(summary["target_window_both_final_hold_longest_streak_steps"]),
                        int(target_window_both_final_hold_streak),
                    )
                    summary["target_window_both_final_hold_streak_at_end_steps"] = int(
                        target_window_both_final_hold_streak
                    )
                    summary["target_window_both_final_stand_longest_streak_steps"] = max(
                        int(summary["target_window_both_final_stand_longest_streak_steps"]),
                        int(target_window_both_final_stand_streak),
                    )
                    summary["target_window_both_final_stand_streak_at_end_steps"] = int(
                        target_window_both_final_stand_streak
                    )
                    summary["target_window_both_stable_at_final_step"] = bool(both_in_window)
                    summary["target_window_both_final_hold_stable_at_final_step"] = bool(
                        both_in_window and bool(current_final_hold_scale_active)
                    )
                    summary["target_window_both_final_stand_stable_at_final_step"] = bool(
                        both_in_window and bool(current_final_stand_active)
                    )
                prev_robot_target_directed = float(robot_target_directed)
                prev_box_target_directed = float(box_target_directed)
                summary["final_robot_delta_xy_m"] = [
                    float(robot_pose[0]) - float(initial_robot[0]),
                    float(robot_pose[1]) - float(initial_robot[1]),
                ]
                summary["final_box_delta_xy_m"] = (
                    [
                        float(box_pose[0]) - float(initial_box[0]),
                        float(box_pose[1]) - float(initial_box[1]),
                    ]
                    if box_pose is not None and initial_box is not None
                    else None
                )
                summary["max_box_robot_relative_offset_error_m"] = max(
                    float(summary["max_box_robot_relative_offset_error_m"]),
                    box_robot_rel_error,
                )
                summary["final_box_robot_relative_offset_error_m"] = box_robot_rel_error if box_pose is not None else None
                prev_box_robot_rel_error = float(box_robot_rel_error)
                summary["min_robot_z_m"] = min(float(summary["min_robot_z_m"]), float(robot_pose[2]))
                if box_pose is not None:
                    summary["min_box_z_m"] = min(float(summary["min_box_z_m"]), float(box_pose[2]))
                    if grasp_attached and grasp_box_z_at_attach is not None:
                        box_z_delta = float(box_pose[2]) - float(grasp_box_z_at_attach)
                        summary["max_post_grasp_box_z_delta_m"] = max(
                            float(summary["max_post_grasp_box_z_delta_m"]),
                            box_z_delta,
                        )
                        summary["final_post_grasp_box_z_delta_m"] = box_z_delta
                        summary["min_post_grasp_box_z_m"] = min(
                            float(summary["min_post_grasp_box_z_m"]),
                            float(box_pose[2]),
                        )
                summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), tilt)
                summary["max_abs_roll_rad"] = max(float(summary["max_abs_roll_rad"]), abs(float(roll)))
                summary["max_abs_pitch_rad"] = max(float(summary["max_abs_pitch_rad"]), abs(float(pitch)))
                summary["final_roll_rad"] = float(roll)
                summary["final_pitch_rad"] = float(pitch)
                if bool(current_final_hold_scale_active):
                    final_min_robot_z = summary["agile_command_hold_final_min_robot_z_m"]
                    summary["agile_command_hold_final_min_robot_z_m"] = (
                        float(robot_pose[2])
                        if final_min_robot_z is None
                        else min(float(final_min_robot_z), float(robot_pose[2]))
                    )
                    summary["agile_command_hold_final_max_tilt_rad"] = max(
                        float(summary["agile_command_hold_final_max_tilt_rad"]),
                        float(tilt),
                    )
                if bool(current_final_stand_active):
                    final_stand_min_robot_z = summary["agile_command_hold_final_stand_min_robot_z_m"]
                    summary["agile_command_hold_final_stand_min_robot_z_m"] = (
                        float(robot_pose[2])
                        if final_stand_min_robot_z is None
                        else min(float(final_stand_min_robot_z), float(robot_pose[2]))
                    )
                    summary["agile_command_hold_final_stand_max_tilt_rad"] = max(
                        float(summary["agile_command_hold_final_stand_max_tilt_rad"]),
                        float(tilt),
                    )
                if box_pose is not None:
                    summary["max_box_tilt_rad"] = max(float(summary["max_box_tilt_rad"]), float(box_tilt))
                    if bool(current_final_hold_scale_active):
                        final_min_box_z = summary["agile_command_hold_final_min_box_z_m"]
                        summary["agile_command_hold_final_min_box_z_m"] = (
                            float(box_pose[2])
                            if final_min_box_z is None
                            else min(float(final_min_box_z), float(box_pose[2]))
                        )
                        summary["agile_command_hold_final_max_box_tilt_rad"] = max(
                            float(summary["agile_command_hold_final_max_box_tilt_rad"]),
                            float(box_tilt),
                        )
                    if bool(current_final_stand_active):
                        final_stand_min_box_z = summary["agile_command_hold_final_stand_min_box_z_m"]
                        summary["agile_command_hold_final_stand_min_box_z_m"] = (
                            float(box_pose[2])
                            if final_stand_min_box_z is None
                            else min(float(final_stand_min_box_z), float(box_pose[2]))
                        )
                        summary["agile_command_hold_final_stand_max_box_tilt_rad"] = max(
                            float(summary["agile_command_hold_final_stand_max_box_tilt_rad"]),
                            float(box_tilt),
                        )
                    summary["max_abs_box_roll_rad"] = max(
                        float(summary["max_abs_box_roll_rad"]),
                        abs(float(box_roll)),
                    )
                    summary["max_abs_box_pitch_rad"] = max(
                        float(summary["max_abs_box_pitch_rad"]),
                        abs(float(box_pitch)),
                    )
                    summary["final_box_roll_rad"] = float(box_roll)
                    summary["final_box_pitch_rad"] = float(box_pitch)
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    box_x = box_pose[0] if box_pose is not None else ""
                    box_y = box_pose[1] if box_pose is not None else ""
                    box_z = box_pose[2] if box_pose is not None else ""
                    box_text = (
                        f"box=({box_pose[0]:.3f},{box_pose[1]:.3f},{box_pose[2]:.3f}) "
                        if box_pose is not None
                        else "box=(disabled) "
                    )
                    writer.writerow([
                        step,
                        time_s,
                        robot_pose[0],
                        robot_pose[1],
                        robot_pose[2],
                        box_x,
                        box_y,
                        box_z,
                        robot_travel,
                        box_travel,
                        robot_target_lateral_error,
                        box_target_lateral_error if box_pose is not None else "",
                        roll,
                        pitch,
                        tilt,
                        box_roll if box_pose is not None else "",
                        box_pitch if box_pose is not None else "",
                        box_tilt if box_pose is not None else "",
                        fall,
                        drop,
                    ])
                    if (
                        replay_writer is not None
                        and int(args_cli.record_replay_every_n_steps) > 0
                        and (step % int(args_cli.record_replay_every_n_steps) == 0 or step == int(args_cli.steps) - 1)
                    ):
                        try:
                            current_joint_positions = np.array(robot.get_joint_positions(), dtype=float).reshape(-1).tolist()
                        except Exception:
                            current_joint_positions = []
                        replay_writer.writerow([
                            step,
                            time_s,
                            robot_pose[0],
                            robot_pose[1],
                            robot_pose[2],
                            robot_pose[3],
                            robot_pose[4],
                            robot_pose[5],
                            robot_pose[6],
                            box_x,
                            box_y,
                            box_z,
                            box_pose[3] if box_pose is not None else "",
                            box_pose[4] if box_pose is not None else "",
                            box_pose[5] if box_pose is not None else "",
                            box_pose[6] if box_pose is not None else "",
                            json.dumps(joint_names),
                            json.dumps([float(v) for v in current_joint_positions]),
                        ])
                    print(
                        "[STATE] "
                        f"step={step} robot=({robot_pose[0]:.3f},{robot_pose[1]:.3f},{robot_pose[2]:.3f}) "
                        + box_text
                        +
                        f"roll={roll:.4f} pitch={pitch:.4f} tilt={tilt:.4f} "
                        f"fall={fall} drop={drop}",
                        flush=True,
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)

    if capture_writer is not None:
        try:
            if capture_rep is not None:
                _run_async(capture_rep.orchestrator.wait_until_complete_async())
            capture_writer.detach()
            if capture_render_product is not None:
                capture_render_product.destroy()
        except Exception as exc:
            summary["capture_rgb_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] RGB capture cleanup failed: {summary['capture_rgb_error']}", flush=True)

    failures = []
    if summary["error"] is not None:
        failures.append(str(summary["error"]))
    if int(summary["completed_steps"]) < int(args_cli.steps):
        failures.append(f"completed_steps {summary['completed_steps']} < requested {args_cli.steps}")
    if int(summary["fall_events"]) > 0:
        failures.append(f"fall_events {summary['fall_events']} > 0")
    if (str(args_cli.attach_box) != "none" or bool(args_cli.require_box_no_drop)) and int(summary["box_drop_events"]) > 0:
        failures.append(f"box_drop_events {summary['box_drop_events']} > 0")
    summary["failures"] = failures
    summary["status"] = "pass" if not failures else "fail"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run_scene()
    finally:
        simulation_app.close()
