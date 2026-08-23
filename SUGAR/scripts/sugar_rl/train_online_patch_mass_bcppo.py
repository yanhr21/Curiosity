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

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_TRACKER = ROOT / "SUGAR/demo_ckpts/CarryBox/tracker.pt"
OFFICIAL_REFINER = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/"
    "baseline/ckpts/refiner_model10000.pt"
)
FORMAL_SEEDS = (151014, 151015, 151016)
CORRECTED_SCALE_SCHEMA = (
    "plan15_live_patch_channel_scales_v3_extent_offset_calibrated"
)

# Keep the formal trainer on the same local IsaacLab assets as the admitted
# frozen evaluator.  Falling back to the remote default ground USD can create
# an empty terrain prim when the Nucleus asset is unavailable, and rerunning
# the URDF importer is unnecessary because this exact G1 USD already exists.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_STIFFNESS", "100")
os.environ.setdefault("CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_DAMPING", "20")
# CarryBox-specific Newton-unit calibration against the independent per-pad
# PhysX audit stream.  It uses full-extent anatomical grids and a 0.3-mm
# compliant-layer offset; formal training must not inherit shell overrides.
os.environ["CURIOSITY_ANATOMICAL_TACSL_CONTACT_OFFSET_M"] = "0.0003"
os.environ["CURIOSITY_ANATOMICAL_TACSL_NORMAL_STIFFNESS"] = "7294.8755"
os.environ["CURIOSITY_ANATOMICAL_TACSL_TANGENTIAL_STIFFNESS"] = "9"
os.environ.setdefault("CURIOSITY_ANATOMICAL_TACSL_FRICTION_COEFFICIENT", "0.5")
os.environ.setdefault(
    "CURIOSITY_TACSL_CALIBRATION_DIR",
    str(ROOT / "experiments/sugar_reproduction/assets/official_tacsl/calibration"),
)
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("CURIOSITY_ENABLE_ANATOMICAL27_WHOLE_HAND_TACSL_AUDIT", "1")
os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(ROOT / "SUGAR/descriptions/terrain/sugar_ground_plane.usda"),
)
_CACHED_G1_USD = (
    ROOT
    / "experiments/online_patch_tactile_mass_adaptation/runtime_assets"
    / "g1_29dof_preconverted_isaacsim510"
    / "g1_29dof_rev_1_0_with_rubber_hand.usd"
)
if _CACHED_G1_USD.is_file():
    os.environ.setdefault("CURIOSITY_G1_PRECONVERTED_USD", str(_CACHED_G1_USD))

# Task modules read the debug-visualization and tactile-physics environment
# contract at import time, so these imports must remain below the block above.
import rsl_rl.runners.on_policy_runner as on_policy_runner_module  # noqa: E402

from online_patch_mass_bcppo_task_registration import (  # noqa: E402
    TASKS,
    register_online_patch_mass_bcppo_tasks,
)
from sugar_rl.utils.online_patch_tactile_actor_critic import (  # noqa: E402
    OnlinePatchTactileActorCritic,
)


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
    overfit = "-Overfit-" in task
    resume = _option_value(argv, "--resume_checkpoint_path")
    stability_diagnostic = os.environ.get(
        "SUGAR_PLAN15_MASKED_DISTILL_STABILITY_DIAGNOSTIC", "0"
    ) == "1"
    anchored_ppo_diagnostic = os.environ.get(
        "SUGAR_PLAN15_ANCHORED_PPO_STABILITY_DIAGNOSTIC", "0"
    ) == "1"
    if stability_diagnostic and anchored_ppo_diagnostic:
        raise ValueError("select only one Plan-15 stability diagnostic")
    if not preflight and not overfit and seed not in FORMAL_SEEDS:
        raise ValueError(f"formal Plan-15 seed must be one of {FORMAL_SEEDS}")
    if overfit and seed != FORMAL_SEEDS[0]:
        raise ValueError("corrected overfit gate uses fixed seed 151014")
    if anchored_ppo_diagnostic:
        if preflight or overfit or seed != FORMAL_SEEDS[0]:
            raise ValueError(
                "anchored-PPO stability diagnostic requires the formal "
                "seed 151014 task"
            )
        if resume is None or Path(resume).name != "model_1000.pt":
            raise ValueError(
                "anchored-PPO stability diagnostic must resume model_1000.pt"
            )
        anchor = os.environ.get("SUGAR_PLAN15_BEHAVIOR_ANCHOR_CHECKPOINT")
        coefficient = float(
            os.environ.get("SUGAR_PLAN15_BEHAVIOR_ANCHOR_COEF", "0.0")
        )
        if anchor is None or Path(anchor).name != "model_750.pt":
            raise ValueError(
                "anchored-PPO stability diagnostic requires model_750.pt "
                "as its frozen behavior anchor"
            )
        if coefficient <= 0.0:
            raise ValueError(
                "anchored-PPO stability diagnostic requires a positive anchor coefficient"
            )
        diagnostic_endpoint = int(
            os.environ.get("SUGAR_PLAN15_ANCHORED_PPO_ENDPOINT", "1251")
        )
        if diagnostic_endpoint < 1001 or diagnostic_endpoint > 1251:
            raise ValueError(
                "anchored-PPO diagnostic endpoint must lie in [1001, 1251]"
            )
        total_iteration_budget = str(diagnostic_endpoint)
    elif stability_diagnostic:
        if preflight or overfit or seed != FORMAL_SEEDS[0]:
            raise ValueError(
                "masked-distill stability diagnostic requires the formal "
                "seed 151014 task"
            )
        if resume is None or Path(resume).name != "model_750.pt":
            raise ValueError(
                "masked-distill stability diagnostic must resume model_750.pt"
            )
        total_iteration_budget = "1251"
    elif preflight:
        total_iteration_budget = "1"
    elif overfit:
        total_iteration_budget = "1500"
    elif resume is None:
        total_iteration_budget = "1251"
    else:
        resume_name = Path(resume).name
        if resume_name == "model_1250.pt":
            total_iteration_budget = "2001"
        elif resume_name == "model_2000.pt":
            total_iteration_budget = "2501"
        elif resume_name == "model_2500.pt":
            total_iteration_budget = "3000"
        else:
            raise ValueError(
                "formal Plan-15 resume checkpoint must be model_1250.pt "
                "model_2000.pt or model_2500.pt"
            )
    os.environ["SUGAR_TOTAL_ITERATION_BUDGET"] = total_iteration_budget
    os.environ["SUGAR_INIT_AT_RANDOM_EP_LEN"] = "0"
    os.environ["SUGAR_PLAN15_LIVE_HANDOFF"] = "1"
    os.environ["SUGAR_PLAN15_HANDOFF_TEACHER_CKPT"] = str(OFFICIAL_REFINER)
    if preflight:
        for branch in ("PS", "P", "Z"):
            if f"-Patch-{branch}-Preflight-" in task:
                os.environ["SUGAR_PLAN15_PREFLIGHT_BRANCH"] = branch
                break
        else:
            raise ValueError(f"cannot identify Plan-15 preflight branch: {task}")
    output = list(argv)
    if _option_value(output, "--motion_folder") is None:
        # One local motion means all four training environments use the exact
        # CarryBox clip used by frozen evaluation. Loading the 100-motion
        # folder with start_init_env_ratio=1 selected only local IDs 0--3.
        output.extend(("--motion_folder", "data/CarryBox/data_045"))
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
    if payload.get("schema") != CORRECTED_SCALE_SCHEMA:
        raise ValueError(
            "Plan-15 requires scales collected after the corrected "
            f"normal/shear/friction semantics ({CORRECTED_SCALE_SCHEMA})"
        )
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
(ROOT / "experiments/sugar_reproduction/logs").mkdir(parents=True, exist_ok=True)
(ROOT / "experiments/sugar_reproduction/logs/sugar_hydra.log").touch(exist_ok=True)
os.chdir(ROOT / "SUGAR")
runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
