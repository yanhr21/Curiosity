# SPDX-License-Identifier: BSD-3-Clause
"""Train (or fine-tune) the CarryBox tracker on Newton.

    # smoke test: does a step run and is the reward finite?
    python -m sugar_newton.rl.train --num-envs 8 --iterations 3 --clips data_000

    # fine-tune from SUGAR's official tracker, which is the intended use
    python -m sugar_newton.rl.train --num-envs 512 --iterations 2000 \
        --warm-start SUGAR/demo_ckpts/CarryBox/tracker.pt --out runs/finetune

Checkpoints are written in the same layout SUGAR uses (``model_state_dict`` with
``actor.*``/``critic.*`` and ``std``), so anything trained here can be replayed by
:mod:`sugar_newton.validation.g1_carrybox_policy` after
``make_policy_assets``-style export, and compared against the Isaac baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl.carrybox_env import CarryBoxEnv, N_DOF, OBS_DIM
from sugar_newton.rl.ppo import PPO, ActorCritic, PPOConfig, RolloutBuffer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip directory names; default is every data_* under data/CarryBox")
    ap.add_argument("--episode-length", type=int, default=300)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--warm-start", default="", help="SUGAR tracker.pt to start the actor from")
    ap.add_argument("--out", default="runs/carrybox")
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wp.init()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = CarryBoxEnv(num_envs=args.num_envs, clip_names=args.clips,
                      episode_length=args.episode_length, substeps=args.substeps,
                      mu=args.mu, device=args.device, seed=args.seed)
    print(f"[env] {args.num_envs} worlds, {len(env.clip_names)} clips, "
          f"{env.model.body_count} bodies, obs {OBS_DIM}, act {N_DOF}")

    cfg = PPOConfig()
    device = torch.device(args.device)
    policy = ActorCritic(OBS_DIM, N_DOF, cfg).to(device)
    if args.warm_start:
        policy.load_sugar_actor(args.warm_start)
    algo = PPO(policy, cfg, device)
    buf = RolloutBuffer(cfg.num_steps_per_env, args.num_envs, OBS_DIM, N_DOF, device)

    obs = env.observe()
    ep_return = torch.zeros(args.num_envs, device=device)
    ep_len = torch.zeros(args.num_envs, device=device)
    recent: list[tuple[float, float]] = []
    log_path = out / "log.jsonl"
    t_start = time.perf_counter()

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(cfg.num_steps_per_env):
                act, logp, val = policy.act(obs)
                next_obs, rew, done, _ = env.step(act)
                buf.add(obs, act, logp, val, rew, done)
                ep_return += rew
                ep_len += 1
                fin = done.nonzero(as_tuple=False).flatten()
                if fin.numel():
                    recent += list(zip(ep_return[fin].tolist(), ep_len[fin].tolist()))
                    ep_return[fin] = 0.0
                    ep_len[fin] = 0.0
                obs = next_obs
            last_val = policy.critic(obs).squeeze(-1)
        buf.finish(last_val, cfg.gamma, cfg.lam)
        t_collect = time.perf_counter() - t0

        stats = algo.update(buf)
        recent = recent[-200:]
        ret = sum(r for r, _ in recent) / max(len(recent), 1)
        length = sum(l for _, l in recent) / max(len(recent), 1)
        steps = it * cfg.num_steps_per_env * args.num_envs
        fps = cfg.num_steps_per_env * args.num_envs / max(t_collect, 1e-9)

        row = {"iter": it, "steps": steps, "return": ret, "ep_len": length,
               "fps": round(fps), "diverged": env.num_diverged,
               "elapsed": round(time.perf_counter() - t_start, 1), **stats}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if it % 10 == 0 or it <= 3:
            print(f"it {it:5d}  steps {steps/1e6:6.2f}M  return {ret:8.3f}  "
                  f"ep_len {length:6.1f}  kl {stats['kl']:.4f}  lr {stats['lr']:.2e}  "
                  f"{fps:7.0f} env-steps/s"
                  + (f"  diverged {env.num_diverged}" if env.num_diverged else ""))

        if it % args.save_interval == 0 or it == args.iterations:
            sd = {f"actor.{k}": v for k, v in policy.actor.state_dict().items()}
            sd |= {f"critic.{k}": v for k, v in policy.critic.state_dict().items()}
            sd["std"] = policy.std.data
            torch.save({"model_state_dict": sd, "iter": it}, out / f"model_{it}.pt")
            print(f"  saved {out / f'model_{it}.pt'}")


if __name__ == "__main__":
    main()
