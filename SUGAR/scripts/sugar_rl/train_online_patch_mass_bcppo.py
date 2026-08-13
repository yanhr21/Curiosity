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
    register_online_patch_mass_bcppo_tasks,
)
from sugar_rl.utils.online_patch_tactile_actor_critic import (
    OnlinePatchTactileActorCritic,
)


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
register_online_patch_mass_bcppo_tasks()
setattr(builtins, "OnlinePatchTactileActorCritic", OnlinePatchTactileActorCritic)
setattr(
    on_policy_runner_module,
    "OnlinePatchTactileActorCritic",
    OnlinePatchTactileActorCritic,
)
runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
