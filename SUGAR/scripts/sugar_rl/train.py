# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import math
import os
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

# The full100 research task is deliberately absent from the package-level Gym
# registry.  Register it only inside an explicitly admitted training process,
# and do so before argparse freezes its task choices.
if os.environ.get("CURIOSITY_ENABLE_RGB_FULL100_STAGE0") == "1":
    from rgb_full100_task_registration import register_rgb_full100_stage0_task

    register_rgb_full100_stage0_task()

tasks = []
for task_spec in gym.registry.values():
    if "Sugar" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--log_dir", type=str, default=None, help="Exact absolute or relative path to the logging directory.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--load_from_distillation", action="store_true", default=False,
    help="Load student weights from a distillation checkpoint (converts student->actor mapping)."
)
parser.add_argument(
    "--distill_experiment", type=str,
    default="unitree_g1_29dofwithhand_mimic_locomanip_distill",
    help="Path or experiment name of the distillation model. "
         "Use this to load distillation checkpoint from a different task's log directory. "
         "If not set, will auto-derive by replacing 'controller' with 'distill' in current experiment name."
)
parser.add_argument("--motion_folder", type=str, default=None, help="Path to motion folder for the environment.")
parser.add_argument("--teacher_motion_folder", type=str, default=None, help="Path to teacher motion folder for the environment.")
parser.add_argument("--teacher_ckpt", type=str, default=None, help="Path to teacher checkpoint for PPO-BC algorithm.")
parser.add_argument(
    "--resume_checkpoint_path",
    type=str,
    default=None,
    help="Exact checkpoint path for resuming an interrupted run without regex-based run discovery.",
)
parser.add_argument(
    "--warm_start_checkpoint_path",
    type=str,
    default=None,
    help=(
        "Official SUGAR checkpoint used to initialize a compatible research branch without "
        "resuming its optimizer or iteration counter."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()
hydra_args.append("hydra.run.dir=.")
hydra_args.append("hydra.output_subdir=null")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import json
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.
import rsl_rl.runners.on_policy_runner as on_policy_runner_module

import rsl_rl.algorithms
import builtins
from sugar_rl.utils.rsl_rl_bcppo import BCPPO
from sugar_rl.utils.tactile_actor_critic import TactileActorCritic
setattr(builtins, "BCPPO", BCPPO)
setattr(rsl_rl.algorithms, "BCPPO", BCPPO)
# OnPolicyRunner resolves the configured policy class in its own module.
setattr(on_policy_runner_module, "TactileActorCritic", TactileActorCritic)

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import sugar_rl.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    if args_cli.resume_checkpoint_path is not None and args_cli.warm_start_checkpoint_path is not None:
        raise ValueError("Choose either --resume_checkpoint_path or --warm_start_checkpoint_path, not both")
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.resume_checkpoint_path is not None:
        agent_cfg.resume = True
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    if getattr(args_cli, "motion_folder", None) is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            env_cfg.commands.motion.motion_folder = args_cli.motion_folder

    if getattr(args_cli, "teacher_motion_folder", None) is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            env_cfg.commands.motion.teacher_motion_folder = args_cli.teacher_motion_folder

    if getattr(args_cli, "teacher_ckpt", None) is not None:
        if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "teacher_ckpt"):
            agent_cfg.algorithm.teacher_ckpt = args_cli.teacher_ckpt

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
        print("[INFO] Disabling Fabric; using USD I/O operations for this compatibility run")

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.log_dir is not None:
        log_dir = os.path.abspath(args_cli.log_dir)
        print(f"[INFO] Command line requested exact log_dir: {log_dir}")
    else:
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # This way, the Ray Tune workflow can extract experiment name.
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if args_cli.load_from_distillation:
            # Determine distillation log directory (similar to --teacher_experiment in distill.py)
            if args_cli.distill_experiment is not None:
                if os.path.isabs(args_cli.distill_experiment):
                    distill_log_path = args_cli.distill_experiment
                else:
                    distill_log_path = os.path.join("logs", "rsl_rl", args_cli.distill_experiment)
                    distill_log_path = os.path.abspath(distill_log_path)
            else:
                # Auto-derive: replace 'controller' with 'distill' in the experiment name
                distill_experiment_name = agent_cfg.experiment_name.replace("controller", "distill")
                distill_log_path = os.path.join("logs", "rsl_rl", distill_experiment_name)
                distill_log_path = os.path.abspath(distill_log_path)
            print(f"[INFO] Loading distillation checkpoint from directory: {distill_log_path}")
            resume_path = get_checkpoint_path(distill_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        elif args_cli.resume_checkpoint_path is not None:
            resume_path = os.path.abspath(args_cli.resume_checkpoint_path)
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"Explicit resume checkpoint does not exist: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap around the environment for RSL-RL.  Formal full100 runs can opt in
    # to a transparent, fail-closed rollout telemetry wrapper; all ordinary
    # SUGAR runs retain the upstream wrapper exactly.
    if os.environ.get("SUGAR_RGB_TELEMETRY_OUTPUT"):
        from sugar_rl.utils.rgb_training_telemetry import (
            RGBTrainingTelemetryVecEnvWrapper,
        )

        env = RGBTrainingTelemetryVecEnvWrapper(
            env, clip_actions=agent_cfg.clip_actions
        )
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    if args_cli.warm_start_checkpoint_path is not None:
        warm_start_path = os.path.abspath(args_cli.warm_start_checkpoint_path)
        if not os.path.isfile(warm_start_path):
            raise FileNotFoundError(f"Warm-start checkpoint does not exist: {warm_start_path}")
        if not hasattr(runner.alg.policy, "load_sugar_warm_start"):
            raise TypeError(
                f"Policy {type(runner.alg.policy).__name__} does not support an official SUGAR warm start"
            )
        print(f"[INFO]: Warm-starting tactile branch from official SUGAR checkpoint: {warm_start_path}")
        checkpoint = torch.load(warm_start_path, map_location=agent_cfg.device, weights_only=False)
        warm_start_report = runner.alg.policy.load_sugar_warm_start(checkpoint["model_state_dict"])
        if not hasattr(runner.alg.policy, "configure_tactile_actor_finetune"):
            raise TypeError(
                f"Policy {type(runner.alg.policy).__name__} does not expose the tactile finetune gate"
            )
        warm_start_report["actor_finetune"] = (
            runner.alg.policy.configure_tactile_actor_finetune()
        )
        # The tactile adapter has different optimizer parameters, so the
        # official optimizer state cannot be loaded. Preserve its converged
        # learning-rate scalar instead of silently restarting the adapter at
        # the configuration's 20x larger initial rate.
        source_optimizer = checkpoint.get("optimizer_state_dict")
        if not isinstance(source_optimizer, dict):
            raise KeyError("Official SUGAR warm start is missing optimizer_state_dict")
        source_lrs = {
            float(group["lr"])
            for group in source_optimizer.get("param_groups", [])
            if "lr" in group
        }
        if len(source_lrs) != 1:
            raise RuntimeError(
                "Official SUGAR warm start requires one optimizer learning rate; "
                f"got {sorted(source_lrs)}"
            )
        source_learning_rate = source_lrs.pop()
        if not math.isfinite(source_learning_rate) or source_learning_rate <= 0.0:
            raise RuntimeError(
                f"Invalid official warm-start learning rate: {source_learning_rate}"
            )
        configured_learning_rate = float(runner.alg.learning_rate)
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = source_learning_rate
        runner.alg.learning_rate = source_learning_rate
        warm_start_report.update(
            {
                "source_checkpoint": warm_start_path,
                "source_iteration": checkpoint.get("iter"),
                "optimizer_loaded": False,
                "configured_learning_rate": configured_learning_rate,
                "source_optimizer_learning_rate": source_learning_rate,
                "active_learning_rate": float(runner.alg.learning_rate),
                "learning_rate_semantics": (
                    "official checkpoint scalar retained; optimizer moments not loaded"
                ),
                "iteration_resumed": False,
            }
        )
        os.makedirs(log_dir, exist_ok=True)
        pre_update_checkpoint = os.path.join(log_dir, "model_pre_update.pt")
        torch.save(
            {
                "model_state_dict": runner.alg.policy.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": -1,
                "infos": {
                    "checkpoint_semantics": "official SUGAR warm start before any tactile PPO update",
                    "source_checkpoint": warm_start_path,
                    "source_iteration": checkpoint.get("iter"),
                },
            },
            pre_update_checkpoint,
        )
        warm_start_report.update(
            {
                "pre_update_checkpoint": pre_update_checkpoint,
                "pre_update_checkpoint_semantics": (
                    "official SUGAR warm start after zero-tactile equivalence audit and before PPO"
                ),
            }
        )
        with open(os.path.join(log_dir, "sugar_warm_start.json"), "w", encoding="utf-8") as file:
            json.dump(warm_start_report, file, indent=2, sort_keys=True)
        print(f"[INFO]: SUGAR tactile warm-start report: {warm_start_report}")
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.load_from_distillation:
            # Load student weights from distillation checkpoint and map to actor
            checkpoint = torch.load(resume_path, weights_only=False)
            distill_state_dict = checkpoint["model_state_dict"]
            
            # Convert student -> actor mapping
            actor_critic_state_dict = {}
            for key, value in distill_state_dict.items():
                if key.startswith("student."):
                    new_key = key.replace("student.", "actor.")
                    actor_critic_state_dict[new_key] = value
                elif key.startswith("student_obs_normalizer."):
                    new_key = key.replace("student_obs_normalizer.", "actor_obs_normalizer.")
                    actor_critic_state_dict[new_key] = value
                elif key in ["std", "log_std"]:
                    actor_critic_state_dict[key] = value
                # Skip teacher-related parameters
            
            # Load only actor part, critic remains randomly initialized
            runner.alg.policy.load_state_dict(actor_critic_state_dict, strict=False)
            print(f"[INFO]: Loaded student weights from distillation checkpoint (actor only, critic randomly initialized)")
        else:
            # load previously trained model
            runner.load(resume_path)
            # RSL-RL restores Adam's parameter groups but does not restore the
            # separate adaptive-KL controller scalar.  Leaving that scalar at
            # the configured 1e-3 while the checkpoint optimizer is at 1e-5
            # makes the first resumed mini-batch overwrite the restored LR and
            # can destroy a learned policy before KL adaptation reacts.  Keep
            # resume semantics faithful by synchronizing the controller to the
            # exact optimizer LR loaded from the checkpoint.
            optimizer_lrs = {
                float(group["lr"]) for group in runner.alg.optimizer.param_groups
            }
            if len(optimizer_lrs) != 1:
                raise RuntimeError(
                    "Resume requires one shared optimizer learning rate; got "
                    f"{sorted(optimizer_lrs)}"
                )
            restored_learning_rate = optimizer_lrs.pop()
            if not math.isfinite(restored_learning_rate) or restored_learning_rate <= 0.0:
                raise RuntimeError(
                    f"Invalid restored optimizer learning rate: {restored_learning_rate}"
                )
            configured_learning_rate = float(runner.alg.learning_rate)
            runner.alg.learning_rate = restored_learning_rate
            resume_optimizer_sync = {
                "protocol": "rsl_rl_adaptive_kl_scalar_matches_loaded_optimizer_v1",
                "resume_checkpoint": resume_path,
                "configured_algorithm_learning_rate": configured_learning_rate,
                "restored_optimizer_learning_rate": restored_learning_rate,
                "synchronized_algorithm_learning_rate": float(runner.alg.learning_rate),
                "overall_pass": float(runner.alg.learning_rate) == restored_learning_rate,
            }
            os.makedirs(log_dir, exist_ok=True)
            with open(
                os.path.join(log_dir, "resume_optimizer_sync.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(resume_optimizer_sync, file, indent=2, sort_keys=True)
            print(
                "[INFO]: Synchronized resumed adaptive-KL learning rate: "
                f"{resume_optimizer_sync}",
                flush=True,
            )
            if isinstance(runner.alg, BCPPO):
                # The upstream RSL-RL checkpoint stores the completed runner
                # iteration but not SUGAR BCPPO's independent curriculum
                # counter.  Reconstructing it from the checkpoint iteration
                # preserves the official 500/1000/2000 BC-to-PPO boundaries
                # instead of silently restarting pure behavior cloning.
                configured_update_step = int(runner.alg.update_step)
                restored_update_step = int(runner.current_learning_iteration) + 1
                runner.alg.update_step = restored_update_step
                stage3_distill_weight_floor = float(
                    runner.alg.stage3_distill_weight_floor
                )
                if not 0.0 <= stage3_distill_weight_floor <= 1.0:
                    raise RuntimeError(
                        "Invalid resumed Stage-3 distillation floor: "
                        f"{stage3_distill_weight_floor}"
                    )
                bcppo_stage_sync = {
                    "protocol": (
                        "modified_sugar_bcppo_persistent_distill_resume_v1"
                        if stage3_distill_weight_floor > 0.0
                        else "official_sugar_bcppo_update_step_from_checkpoint_iteration_v1"
                    ),
                    "resume_checkpoint": resume_path,
                    "checkpoint_iteration": int(runner.current_learning_iteration),
                    "configured_update_step": configured_update_step,
                    "restored_update_step": int(runner.alg.update_step),
                    "bc_only_steps": int(runner.alg.bc_only_steps),
                    "critic_warmup_steps": int(runner.alg.critic_warmup_steps),
                    "full_ppo_warmup_steps": int(runner.alg.full_ppo_warmup_steps),
                    "stage3_distill_weight_floor": stage3_distill_weight_floor,
                    "modified_persistent_distillation": (
                        stage3_distill_weight_floor > 0.0
                    ),
                    "overall_pass": (
                        int(runner.alg.update_step) == restored_update_step
                        and 0.0 <= stage3_distill_weight_floor <= 1.0
                    ),
                }
                with open(
                    os.path.join(log_dir, "resume_bcppo_stage_sync.json"),
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(bcppo_stage_sync, file, indent=2, sort_keys=True)
                print(
                    f"[INFO]: Restored SUGAR BCPPO stage state: {bcppo_stage_sync}",
                    flush=True,
                )
            # RSL-RL stores the label of the completed checkpoint iteration.
            # Its default load path starts the next learn() loop at that same
            # label, producing one extra optimizer update with a repeated file
            # name.  Advance only the runner label after reconstructing BCPPO's
            # independent stage counter above.
            checkpoint_iteration = int(runner.current_learning_iteration)
            runner.current_learning_iteration = checkpoint_iteration + 1
            resume_iteration_sync = {
                "protocol": "rsl_rl_resume_at_next_iteration_v1",
                "resume_checkpoint": resume_path,
                "checkpoint_iteration": checkpoint_iteration,
                "next_learning_iteration": int(runner.current_learning_iteration),
                "overall_pass": (
                    int(runner.current_learning_iteration)
                    == checkpoint_iteration + 1
                ),
            }
            with open(
                os.path.join(log_dir, "resume_iteration_sync.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(resume_iteration_sync, file, indent=2, sort_keys=True)
            print(
                f"[INFO]: Advanced resume to the next iteration: {resume_iteration_sync}",
                flush=True,
            )

    if (
        args_cli.warm_start_checkpoint_path is None
        and args_cli.resume_checkpoint_path is not None
        and hasattr(runner.alg.policy, "configure_tactile_actor_finetune")
    ):
        resumed_finetune_report = runner.alg.policy.configure_tactile_actor_finetune()
        resumed_finetune_report.update(
            {
                "resume_checkpoint": os.path.abspath(args_cli.resume_checkpoint_path),
                "semantics": "reinstalled tactile-only actor gradient gate after checkpoint load",
            }
        )
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "tactile_finetune_resume.json"), "w", encoding="utf-8") as file:
            json.dump(resumed_finetune_report, file, indent=2, sort_keys=True)

    prelearn_checkpoint = os.environ.get("SUGAR_PRELEARN_CHECKPOINT")
    if prelearn_checkpoint:
        prelearn_checkpoint = os.path.abspath(prelearn_checkpoint)
        if os.path.dirname(prelearn_checkpoint) != os.path.abspath(log_dir):
            raise ValueError(
                "SUGAR_PRELEARN_CHECKPOINT must be a direct child of log_dir"
            )
        if os.path.exists(prelearn_checkpoint):
            raise FileExistsError(prelearn_checkpoint)
        os.makedirs(log_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": runner.alg.policy.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": -1,
                "infos": {
                    "semantics": "policy and optimizer immediately before learn()",
                    "seed": int(agent_cfg.seed),
                },
            },
            prelearn_checkpoint,
        )
        print(f"[INFO]: Saved pre-learning checkpoint: {prelearn_checkpoint}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if hasattr(env, "finalize_telemetry"):
        env.finalize_telemetry()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
