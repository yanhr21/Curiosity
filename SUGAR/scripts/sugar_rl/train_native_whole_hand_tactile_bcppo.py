#!/usr/bin/env python3
"""Run the matched native whole-hand tactile arms through SUGAR training."""

from __future__ import annotations

import builtins
import runpy
from pathlib import Path

import rsl_rl.algorithms
import rsl_rl.runners.on_policy_runner as on_policy_runner_module

from native_whole_hand_tactile_bcppo_task_registration import (
    register_native_whole_hand_tactile_bcppo_tasks,
)
from sugar_rl.utils.reference_only_tactile_actor_critic import (
    ReferenceOnlyTactileActorCritic,
)
from sugar_rl.utils.tracker_command_tactile_actor_critic import (
    TrackerCommandTactileActorCritic,
)
from sugar_rl.utils.native_tactile_training_bcppo import (
    NativeTactileTrainingBCPPO,
)


register_native_whole_hand_tactile_bcppo_tasks()
setattr(builtins, "ReferenceOnlyTactileActorCritic", ReferenceOnlyTactileActorCritic)
setattr(
    on_policy_runner_module,
    "ReferenceOnlyTactileActorCritic",
    ReferenceOnlyTactileActorCritic,
)
setattr(builtins, "TrackerCommandTactileActorCritic", TrackerCommandTactileActorCritic)
setattr(
    on_policy_runner_module,
    "TrackerCommandTactileActorCritic",
    TrackerCommandTactileActorCritic,
)
setattr(builtins, "NativeTactileTrainingBCPPO", NativeTactileTrainingBCPPO)
setattr(rsl_rl.algorithms, "NativeTactileTrainingBCPPO", NativeTactileTrainingBCPPO)
setattr(
    on_policy_runner_module,
    "NativeTactileTrainingBCPPO",
    NativeTactileTrainingBCPPO,
)
runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
