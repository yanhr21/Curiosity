"""Preflight for `sugar_swap.train`: everything that can be checked without training.

Two stages, because they need different machines:

    python slurm/swap_train_check.py                # login node, no GPU: stages 1-5
    python slurm/swap_train_check.py --with-runner  # GPU: also builds the env and the runner

Stages 1-5 are config and plumbing: the argparse surface, both registry entry points, the
runner-config dict `OnPolicyRunner` will actually receive, the evaluation config's two
hard-won properties, and an offline `wandb.init`. Stage 6 builds a small environment and the
runner and checks the policy's input width against the refiner's 890-D privileged
observation -- the number the hand-written port got wrong.

Nothing here trains, and nothing here writes into a real run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}{f': {detail}' if detail else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-runner", action="store_true",
                    help="also build the env and the rsl_rl runner (needs a GPU)")
    ap.add_argument("--with-video", action="store_true",
                    help="also build the evaluation env and record one short rollout "
                         "(needs a GPU and hardware EGL; implies --with-runner)")
    ap.add_argument("--envs", type=int, default=4, help="envs for the stage-7 build")
    ap.add_argument("--video-frames", type=int, default=30,
                    help="frames for the stage-8 rollout; keep it short")
    opts = ap.parse_args()
    if opts.with_video:
        opts.with_runner = True

    from sugar_swap import train as T

    # ---- 1. argparse surface ----------------------------------------------------------
    print("\n=== 1. argparse ===")
    parser = T.build_parser()
    defaults = parser.parse_args([])
    check("--max-iterations defaults to None (SUGAR's 30001 is used)",
          defaults.max_iterations is None)
    check("--save-interval defaults to 25", defaults.save_interval == 25, str(defaults.save_interval))
    check("--ddp-verify defaults to 0 (off)", defaults.ddp_verify == 0, str(defaults.ddp_verify))
    check("--eval-minutes defaults to 20", defaults.eval_minutes == 20.0, str(defaults.eval_minutes))
    check("--logger defaults to wandb", defaults.logger == "wandb")
    check("default task is the refiner", defaults.task == "Sugar-G129dof-CarryBox-Refiner",
          defaults.task)
    check("default motion folder exists", Path(defaults.motion_folder).is_dir(),
          defaults.motion_folder)

    over = parser.parse_args([
        "--num-envs", "512", "--max-iterations", "30001", "--seed", "7",
        "--save-interval", "25", "--resume", "logs/x/model_25.pt",
        "--logger", "tensorboard", "--eval-minutes", "5", "--dry-run",
    ])
    check("overrides parse", (over.num_envs, over.max_iterations, over.seed, over.save_interval,
                              over.resume, over.logger, over.eval_minutes, over.dry_run)
          == (512, 30001, 7, 25, "logs/x/model_25.pt", "tensorboard", 5.0, True))
    for flag in ("--num_envs", "--max_iterations"):
        rc = 0
        try:
            parser.parse_args([flag, "1"])
        except SystemExit:
            rc = 2
        check(f"{flag} (underscore form) is rejected, not silently ignored", rc == 2)

    # ---- 2. bootstrap -----------------------------------------------------------------
    print("\n=== 2. bootstrap ===")
    T.install_swap()
    import isaaclab.assets  # noqa: F401
    import isaaclab.envs

    check("bootstrap.install() succeeded and isaaclab.envs is the shadow",
          "sugar_swap" in isaaclab.envs.ManagerBasedRLEnv.__module__,
          isaaclab.envs.ManagerBasedRLEnv.__module__)
    check("DirectRLEnv shim present for isaaclab_rl", hasattr(isaaclab.envs, "DirectRLEnv"))
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: F401

    check("isaaclab_rl.rsl_rl imports (RslRlVecEnvWrapper available)", True)
    leaked = [m for m in sys.modules if m.startswith(("isaacsim", "omni.physx"))]
    check("Isaac Sim never booted", not leaked, str(leaked))

    # ---- 3. registry --------------------------------------------------------------------
    print("\n=== 3. registry entry points ===")
    import gymnasium as gym

    spec = gym.registry[T.TASK]
    for key in (T.ENV_CFG_KEY, T.AGENT_CFG_KEY, T.EVAL_CFG_KEY):
        check(f"{key} registered", key in spec.kwargs, str(spec.kwargs.get(key)))

    args = parser.parse_args(["--num-envs", "512"])
    args.motion_folder = Path(args.motion_folder)
    agent_cfg = T.load_agent_cfg(T.TASK, args)
    env_cfg = T.load_env_cfg(T.TASK, T.ENV_CFG_KEY, args, num_envs=args.num_envs,
                             seed=agent_cfg.seed)

    check("agent cfg is SUGAR's BasePPORunnerCfg",
          type(agent_cfg).__name__ == "BasePPORunnerCfg",
          f"{type(agent_cfg).__module__}:{type(agent_cfg).__name__}")
    check("env cfg is SUGAR's RobotEnvCfg",
          type(env_cfg).__name__ == "RobotEnvCfg",
          f"{type(env_cfg).__module__}:{type(env_cfg).__name__}")
    check("refiner is plain PPO, not BCPPO", agent_cfg.algorithm.class_name == "PPO",
          agent_cfg.algorithm.class_name)
    check("policy is ActorCritic", agent_cfg.policy.class_name == "ActorCritic")
    check("num_steps_per_env is SUGAR's 24", agent_cfg.num_steps_per_env == 24)
    check("max_iterations is SUGAR's 30001 when not overridden",
          agent_cfg.max_iterations == 30001, str(agent_cfg.max_iterations))
    check("seed is SUGAR's 42 when not overridden", agent_cfg.seed == 42, str(agent_cfg.seed))
    check("policy = critic group (privileged actor, no teacher)",
          type(env_cfg.observations.policy) is type(env_cfg.observations.critic)
          and set(vars(env_cfg.observations)) - {"_"} >= {"policy", "critic"}
          and "teacher" not in vars(env_cfg.observations))
    check("21 reward terms",
          sum(1 for n, t in vars(env_cfg.rewards).items()
              if not n.startswith("_") and t is not None) == 21)
    check("6 termination terms",
          sum(1 for n, t in vars(env_cfg.terminations).items()
              if not n.startswith("_") and t is not None) == 6)
    check("feet_air_time weight is +5.0 (the term the hand port omitted)",
          env_cfg.rewards.feet_air_time.weight == 5.0, str(env_cfg.rewards.feet_air_time.weight))
    check("50 Hz control (dt 0.005, decimation 4)",
          (env_cfg.sim.dt, env_cfg.decimation) == (0.005, 4))
    check("--num-envs override applied", env_cfg.scene.num_envs == 512)

    # ---- 3b. --num-envs is a TOTAL under DDP ---------------------------------------------
    # The failure this guards is silent and expensive: --num-envs read per-rank would make
    # the default 8-GPU launch an 8x larger batch than SUGAR's, i.e. a different experiment
    # with the same command line and the same log directory.
    print("\n=== 3b. --num-envs is a total across ranks ===")
    ranks = 8
    default_args = parser.parse_args([])
    default_args.motion_folder = Path(default_args.motion_folder)
    ddp_default = T.load_env_cfg(T.TASK, T.ENV_CFG_KEY, default_args, num_envs=None,
                                 seed=42, world_size=ranks)
    check(f"no --num-envs on {ranks} ranks splits SUGAR's registered 4096 total",
          ddp_default.scene.num_envs * ranks == 4096,
          f"{ddp_default.scene.num_envs} per rank x {ranks} = "
          f"{ddp_default.scene.num_envs * ranks}")
    ddp_explicit = T.load_env_cfg(T.TASK, T.ENV_CFG_KEY, default_args, num_envs=4096,
                                  seed=42, world_size=ranks)
    check("--num-envs 4096 on 8 ranks is 512 per rank, NOT 4096 per rank",
          ddp_explicit.scene.num_envs == 512, str(ddp_explicit.scene.num_envs))
    check("single process is unchanged: --num-envs 512 stays 512",
          T.load_env_cfg(T.TASK, T.ENV_CFG_KEY, default_args, num_envs=512,
                         seed=42).scene.num_envs == 512)
    indivisible = 0
    try:
        T.load_env_cfg(T.TASK, T.ENV_CFG_KEY, default_args, num_envs=4095, seed=42,
                       world_size=ranks)
    except SystemExit:
        indivisible = 1
    check("a total that does not divide the rank count raises rather than skewing ranks",
          indivisible == 1)
    check("ranks are read from torchrun's environment, (0, 0, 1) without it",
          T.ddp_ranks() == (0, 0, 1), str(T.ddp_ranks()))
    check("--device is left alone in a single process (no CUDA touched)",
          T.pin_device(0, 1) == "")

    print("\n  SUGAR's PPO hyperparameters, as loaded:")
    alg = agent_cfg.algorithm
    for name in ("class_name", "learning_rate", "schedule", "desired_kl", "gamma", "lam",
                 "entropy_coef", "clip_param", "value_loss_coef", "use_clipped_value_loss",
                 "num_learning_epochs", "num_mini_batches", "max_grad_norm"):
        print(f"    algorithm.{name:32s} {getattr(alg, name)}")
    for name in ("class_name", "init_noise_std", "actor_hidden_dims", "critic_hidden_dims",
                 "activation"):
        print(f"    policy.{name:35s} {getattr(agent_cfg.policy, name)}")
    for name in ("num_steps_per_env", "max_iterations", "save_interval",
                 "empirical_normalization", "clip_actions", "experiment_name"):
        print(f"    runner.{name:35s} {getattr(agent_cfg, name)}")

    # ---- 4. runner config dict ----------------------------------------------------------
    print("\n=== 4. runner config dict (what OnPolicyRunner receives) ===")
    d = agent_cfg.to_dict()
    check("to_dict() carries the algorithm class", d["algorithm"]["class_name"] == "PPO")
    check("obs_groups arrives as {} for rsl_rl to resolve from the env",
          d["obs_groups"] == {}, repr(d["obs_groups"]))
    check("empirical_normalization is False", d["empirical_normalization"] is False)
    check("no teacher_ckpt anywhere (this is not BCPPO)",
          "teacher_ckpt" not in json.dumps(d, default=repr))
    print(json.dumps(d, indent=4, default=repr))

    # ---- 5. evaluation config -----------------------------------------------------------
    print("\n=== 5. evaluation config ===")
    eval_cfg, tracking = T.eval_env_cfg(T.TASK, args, seed=agent_cfg.seed)
    check("one environment", eval_cfg.scene.num_envs == 1)
    check("episode_length_s is the play config's 1e9", eval_cfg.episode_length_s == 1e9)
    check("eval_mode pins motion/start (commands.py:254)",
          eval_cfg.commands.motion.eval_mode is True)
    check("eval_random_motion stays off, so motion_id = env_id % num_motion = 0",
          eval_cfg.commands.motion.eval_random_motion is False)
    check("five reference-tracking terminations removed",
          sorted(tracking) == ["anchor_ori", "anchor_pos", "ee_body_pos", "obj_ori", "obj_pos"],
          str(sorted(tracking)))
    remaining = [n for n, t in vars(eval_cfg.terminations).items()
                 if not n.startswith("_") and t is not None]
    check("only the trajectory_complete time-out is left",
          remaining == ["trajectory_complete"], str(remaining))
    check("random pushes off, so successive evaluations are comparable",
          eval_cfg.events.push_robot is None and eval_cfg.events.push_object is None)
    check("training config is untouched by the evaluation config",
          env_cfg.commands.motion.eval_mode is False
          and env_cfg.terminations.anchor_pos is not None
          and env_cfg.events.push_robot is not None)

    # ---- 6. wandb, offline ---------------------------------------------------------------
    print("\n=== 6. wandb (offline) ===")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = tmp
        os.environ["WANDB_SILENT"] = "true"
        from sugar_newton.rl.run_dir import META_NAME, bind_wandb_run

        run_dir = Path(tmp) / "carrybox_refiner_swap"
        first = bind_wandb_run(run_dir, project="sugar_newton", stage="refiner_swap")
        check("run id minted and persisted", bool(first) and (run_dir / META_NAME).is_file(),
              str(first))
        os.environ.pop("WANDB_RUN_ID", None)
        second = bind_wandb_run(run_dir, project="sugar_newton", stage="refiner_swap")
        check("a second leg reuses the same run id (chained jobs are one curve)",
              second == first, f"{first} -> {second}")
        check("WANDB_RESUME=allow is exported", os.environ.get("WANDB_RESUME") == "allow")

        import wandb

        run = wandb.init(project="sugar_newton", name=run_dir.name, dir=tmp)
        check("wandb.init() honours the persisted id", run.id == first, f"{run.id} vs {first}")
        wandb.log({"preflight/ok": 1.0}, step=0)
        wandb.finish()

    # ---- 7. env + runner (GPU) ------------------------------------------------------------
    if opts.with_runner:
        print(f"\n=== 7. env + runner ({opts.envs} envs) ===")
        import torch

        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        build_args = parser.parse_args(["--num-envs", str(opts.envs)])
        build_args.motion_folder = Path(build_args.motion_folder)
        agent_cfg = T.load_agent_cfg(T.TASK, build_args)
        build_cfg = T.load_env_cfg(T.TASK, T.ENV_CFG_KEY, build_args,
                                   num_envs=opts.envs, seed=agent_cfg.seed)
        env = ManagerBasedRLEnv(build_cfg, device=build_args.device)
        if not hasattr(env, "unwrapped"):
            env.unwrapped = env
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        obs = vec_env.get_observations()
        for group, value in obs.items():
            print(f"    obs[{group}] {tuple(value.shape)}")
        check("policy group is the 890-D privileged observation",
              obs["policy"].shape[-1] == 890, str(obs["policy"].shape[-1]))
        check("critic group is the same 890-D vector",
              obs["critic"].shape[-1] == 890, str(obs["critic"].shape[-1]))

        with tempfile.TemporaryDirectory() as tmp:
            agent_cfg.logger = "tensorboard"
            runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=tmp,
                                    device=agent_cfg.device)
        check("rsl_rl resolved obs_groups from the env",
              runner.cfg["obs_groups"] == {"policy": ["policy"], "critic": ["critic"]},
              str(runner.cfg["obs_groups"]))
        actor_in = runner.alg.policy.actor[0].in_features
        actor_out = runner.alg.policy.actor[-1].out_features
        critic_in = runner.alg.policy.critic[0].in_features
        check("actor input is 890", actor_in == 890, str(actor_in))
        check("critic input is 890", critic_in == 890, str(critic_in))
        check("actor output matches the action manager's total_action_dim",
              actor_out == vec_env.num_actions == env.action_manager.total_action_dim,
              f"{actor_out} / {vec_env.num_actions}")
        check("critic outputs a scalar",
              runner.alg.policy.critic[-1].out_features == 1)
        check("algorithm is PPO", type(runner.alg).__name__ == "PPO",
              type(runner.alg).__name__)
        check("no empirical normalization on either branch",
              isinstance(runner.alg.policy.actor_obs_normalizer, torch.nn.Identity)
              and isinstance(runner.alg.policy.critic_obs_normalizer, torch.nn.Identity))
        check("a fresh run is asked for the full budget",
              int(agent_cfg.max_iterations) - int(runner.current_learning_iteration)
              == 30001)

        # The resume arithmetic, which is where "silently retrains from scratch each leg"
        # lives. rsl_rl stores the COMPLETED iteration, so the label has to be advanced.
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "model_50.pt")
            runner.logger_type = "tensorboard"     # `save` consults it before uploading
            runner.current_learning_iteration = 50
            runner.save(ckpt)
            runner.current_learning_iteration = 0
            runner.load(ckpt)
            check("load() restores the completed iteration, not the next one",
                  runner.current_learning_iteration == 50,
                  str(runner.current_learning_iteration))
            runner.current_learning_iteration += 1
            todo = int(agent_cfg.max_iterations) - int(runner.current_learning_iteration)
            check("a resumed leg trains the remainder from the NEXT iteration",
                  todo == 29950, f"{todo} (30001 - 51)")
            runner.current_learning_iteration = 30001
            check("a finished run asks for nothing rather than one duplicate iteration",
                  int(agent_cfg.max_iterations) - 30001 <= 0)

        # ---- 8. evaluation env + one recorded rollout -----------------------------------
        if opts.with_video:
            print(f"\n=== 8. evaluation video ({opts.video_frames} frames) ===")
            with tempfile.TemporaryDirectory() as tmp:
                video_args = parser.parse_args([
                    "--log-root", tmp, "--run-name", "preflight",
                    "--video-frames", str(opts.video_frames), "--logger", "tensorboard",
                ])
                video_args.motion_folder = Path(video_args.motion_folder)
                recorder = T.build_recorder(video_args, T.TASK, agent_cfg.seed)
                path, stats = recorder.record(runner.alg.policy, 0)
                check("_ensure() built the evaluation env and a GL context",
                      recorder.viewer is not None)
                if recorder.env is not None:
                    ev = recorder.env
                    live = [n for n, t in vars(ev._env.cfg.terminations).items()
                            if not n.startswith("_") and t is not None]
                    check("evaluation env has only the time-out termination",
                          live == ["trajectory_complete"], str(live))
                    check("start state pinned to motion 0, frame 0",
                          (int(ev._command.motion_id[0]), int(ev._command.time_steps[0]))[0] == 0)
                    check("drift is still reported although it no longer terminates",
                          ev._tracking is not None
                          and len(ev._tracking.active_terms) == 5,
                          str(sorted(ev._tracking.active_terms)) if ev._tracking else "none")
                check("a video file was written", path is not None and Path(path).is_file(),
                      f"{path} "
                      f"({Path(path).stat().st_size // 1024 if path else 0} KiB)")
                check("the rollout played past the drift point rather than resetting",
                      stats.get("video/frames", 0) == opts.video_frames,
                      f"{stats.get('video/frames')} frames")
                check("tactile panels rendered (load-bearing contact count present)",
                      "video/load_bearing_contacts" in stats,
                      "scene-only fallback" if "video/load_bearing_contacts" not in stats
                      else f"{stats['video/load_bearing_contacts']:.1f} mean")
                print("    stats: " + json.dumps(
                    {k: round(float(v), 4) for k, v in stats.items()}, indent=6))
        else:
            print("\n=== 8. evaluation video: SKIPPED (pass --with-video on a GPU) ===")

        env.close()
    else:
        print("\n=== 7. env + runner: SKIPPED (pass --with-runner on a GPU) ===")
        print("=== 8. evaluation video: SKIPPED ===")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: " + "; ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
