# SPDX-License-Identifier: BSD-3-Clause
"""Train the CarryBox tracker on Newton with SUGAR's own BCPPO, logging to Weights & Biases.

Nothing about the algorithm is reimplemented. This imports ``BCPPO`` from SUGAR and runs it
inside ``rsl_rl``'s ``OnPolicyRunner``, with the hyperparameters read out of SUGAR's
``BCPPORunnerCfg``. The only local code in the loop is
:class:`~sugar_newton.rl.vec_env.CarryBoxVecEnv`, which presents the Newton environment in
the shape rsl_rl expects.

``BCPPO`` is a three-stage curriculum around the frozen refiner as teacher::

    stage 1   step < 500          loss = distill                    (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value    (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

so the teacher checkpoint is not optional -- ``BCPPO.__init__`` asserts on a missing one,
and without it stages 1-2 have no loss at all.

Usage::

    python -m sugar_newton.rl.train_bcppo \
        --num-envs 512 --max-iterations 30001 \
        --teacher-ckpt experiments/.../ckpts/refiner_model10000.pt \
        --wandb-project sugar_newton --run-name carrybox_bcppo_$(date +%m%d_%H%M)

The runner writes wandb through rsl_rl's own ``WandbSummaryWriter`` (``logger: wandb``),
so the run name is the log directory's basename and the project comes from
``wandb_project`` in the runner config. Credentials follow the convention used elsewhere in
this workspace: ``WANDB_API_KEY`` from the environment, else ``~/.netrc`` for
``api.wandb.ai``. The key is never printed.
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
from pathlib import Path

import torch
import warp as wp

HERE = Path(__file__).resolve().parent
SUGAR_SRC = HERE.parents[1] / "SUGAR" / "source" / "sugar_rl"
DEFAULT_TEACHER = (HERE.parents[1] / "experiments/sugar_reproduction/outputs/final"
                   / "official_sugar/baseline/ckpts/refiner_model10000.pt")


def sugar_bcppo():
    """Import SUGAR's BCPPO and register it where rsl_rl's runner will find it.

    ``OnPolicyRunner._construct_algorithm`` resolves the algorithm with
    ``eval(alg_cfg["class_name"])``, so the class has to be reachable by bare name. This is
    SUGAR's own mechanism, copied from ``scripts/sugar_rl/train.py:147-150``, not a new one.
    ``sugar_rl/utils/__init__.py`` is empty, so the module imports without pulling in
    IsaacLab.
    """
    sys.path.insert(0, str(SUGAR_SRC))
    import rsl_rl.algorithms
    from sugar_rl.utils.rsl_rl_bcppo import BCPPO

    builtins.BCPPO = BCPPO
    rsl_rl.algorithms.BCPPO = BCPPO
    return BCPPO


def runner_cfg(args) -> dict:
    """BCPPORunnerCfg, transcribed. Values are SUGAR's; only logging is added."""
    return {
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "experiment_name": args.run_name,
        "empirical_normalization": False,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"], "teacher": ["teacher"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.5,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "BCPPO",
            "teacher_ckpt": args.teacher_ckpt,
            "stage3_distill_weight_floor": 0.0,
            "training_mask_obs_group": None,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "logger": args.logger,
        "wandb_project": args.wandb_project,
    }


def ensure_wandb_credentials() -> None:
    """Fail here rather than several minutes into a run with logging silently off."""
    if os.environ.get("WANDB_API_KEY"):
        return
    try:
        import netrc

        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth and auth[2]:
            os.environ["WANDB_API_KEY"] = auth[2]
            return
    except Exception:
        pass
    raise SystemExit("wandb: set WANDB_API_KEY or add api.wandb.ai to ~/.netrc "
                     "(or pass --logger tensorboard)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--max-iterations", type=int, default=30001)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--episode-length", type=int, default=300)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--teacher-ckpt", default=str(DEFAULT_TEACHER))
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--run-name", default="carrybox_bcppo")
    ap.add_argument("--log-root", default="logs/newton_bcppo")
    ap.add_argument("--logger", default="wandb", choices=("wandb", "tensorboard"))
    ap.add_argument("--wandb-project", default="sugar_newton")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.logger == "wandb":
        ensure_wandb_credentials()
    if not Path(args.teacher_ckpt).is_file():
        raise SystemExit(f"teacher checkpoint not found: {args.teacher_ckpt}\n"
                         "BCPPO's stages 1-2 have no loss without it.")

    sugar_bcppo()
    from rsl_rl.runners import OnPolicyRunner

    from sugar_newton.rl.vec_env import OBS_DIMS, make

    wp.init()
    torch.manual_seed(args.seed)

    env = make(args.num_envs, clip_names=args.clips, episode_length=args.episode_length,
               substeps=args.substeps, mu=args.mu, device=args.device, seed=args.seed)
    print(f"[env] {args.num_envs} worlds, {len(env.env.clip_names)} clips, "
          f"obs {OBS_DIMS}, act {env.num_actions}")

    log_dir = Path(args.log_root) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg(args), log_dir=str(log_dir), device=args.device)
    print(f"[alg] {type(runner.alg).__name__}  teacher={args.teacher_ckpt}")
    if args.resume:
        runner.load(args.resume)
        print(f"[alg] resumed from {args.resume}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
