#!/usr/bin/env python3
"""Launch one Plan-15 Z/P/PS arm with live-sweep patch normalization."""

from __future__ import annotations

import builtins
import json
import math
import os
import runpy
import sys
from pathlib import Path

import rsl_rl.runners.on_policy_runner as on_policy_runner_module

from online_patch_mass_bcppo_task_registration import (
    TASKS,
    register_online_patch_mass_bcppo_tasks,
)
from sugar_rl.utils.online_patch_tactile_actor_critic import (
    OnlinePatchTactileActorCritic,
)


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_TRACKER = ROOT / "SUGAR/demo_ckpts/CarryBox/tracker.pt"
OFFICIAL_REFINER = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/"
    "baseline/ckpts/refiner_model10000.pt"
)
FORMAL_SEEDS = (151014, 151015, 151016)


def _option_value(argv: list[str], option: str) -> str | None:
    if option in argv:
        index = argv.index(option)
        if index + 1 >= len(argv):
            raise ValueError(f"{option} requires a value")
        return argv[index + 1]
    prefix = option + "="
    matches = [value[len(prefix) :] for value in argv if value.startswith(prefix)]
    if len(matches) > 1:
        raise ValueError(f"{option} was provided more than once")
    return matches[0] if matches else None


def _inject_official_training_contract(argv: list[str]) -> list[str]:
    if "--help" in argv or "-h" in argv:
        return argv
    task = _option_value(argv, "--task")
    if task not in TASKS:
        raise ValueError("Plan-15 launcher requires one registered Z/P/PS task")
    seed_text = _option_value(argv, "--seed")
    if seed_text is None:
        raise ValueError("Plan-15 requires an explicit matched --seed")
    seed = int(seed_text)
    preflight = "-Preflight-" in task
    if not preflight and seed not in FORMAL_SEEDS:
        raise ValueError(f"formal Plan-15 seed must be one of {FORMAL_SEEDS}")
    os.environ["SUGAR_TOTAL_ITERATION_BUDGET"] = "1" if preflight else "3000"
    os.environ["SUGAR_INIT_AT_RANDOM_EP_LEN"] = "0"
    if preflight:
        for branch in ("PS", "P", "Z"):
            if f"-Patch-{branch}-Preflight-" in task:
                os.environ["SUGAR_PLAN15_PREFLIGHT_BRANCH"] = branch
                break
        else:
            raise ValueError(f"cannot identify Plan-15 preflight branch: {task}")

    output = list(argv)
    teacher = _option_value(output, "--teacher_ckpt")
    if teacher is None:
        output.extend(("--teacher_ckpt", str(OFFICIAL_REFINER)))
    elif Path(teacher).expanduser().resolve() != OFFICIAL_REFINER.resolve():
        raise ValueError("Plan-15 teacher must be the official frozen Refiner")

    resume = _option_value(output, "--resume_checkpoint_path")
    warm_start = _option_value(output, "--warm_start_checkpoint_path")
    if resume is None:
        if warm_start is None:
            output.extend(
                ("--warm_start_checkpoint_path", str(OFFICIAL_TRACKER))
            )
        elif Path(warm_start).expanduser().resolve() != OFFICIAL_TRACKER.resolve():
            raise ValueError("Plan-15 warm start must be the official Tracker")
    elif warm_start is not None:
        raise ValueError("resume and warm start are mutually exclusive")
    for checkpoint in (OFFICIAL_REFINER, OFFICIAL_TRACKER):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    return output


def _consume_scale_file(argv: list[str]) -> list[str]:
    option = "--patch-scale-file"
    if option not in argv:
        if "--help" in argv or "-h" in argv:
            return argv
        raise ValueError(
            "Plan-15 training requires --patch-scale-file from the live mass sweep"
        )
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise ValueError("--patch-scale-file requires a JSON path")
    path = Path(argv[index + 1]).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    values = payload.get("patch_channel_scales")
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError("patch scale JSON must contain nine patch_channel_scales")
    scales = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("patch channel scales must be positive and finite")
    os.environ["SUGAR_ONLINE_PATCH_CHANNEL_SCALES"] = json.dumps(scales)
    return argv[:index] + argv[index + 2 :]


sys.argv = _consume_scale_file(sys.argv)
sys.argv = _inject_official_training_contract(sys.argv)
register_online_patch_mass_bcppo_tasks()
setattr(builtins, "OnlinePatchTactileActorCritic", OnlinePatchTactileActorCritic)
setattr(
    on_policy_runner_module,
    "OnlinePatchTactileActorCritic",
    OnlinePatchTactileActorCritic,
)
runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
