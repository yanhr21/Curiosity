# SPDX-License-Identifier: BSD-3-Clause
"""Measure how one PPO collection (``num_steps_per_env`` steps) scales with worlds per GPU.

This exists to answer one question: is an 8-GPU run worth it? Two configurations give the
*same* 12288-sample batch that BCPPO is tuned for --

    1 GPU  x 512 worlds     (what we run today)
    8 GPUs x  64 worlds     (DDP, same batch, 8x fewer worlds each)

and they are only equivalent in wall clock if per-iteration cost is roughly linear in worlds.
It is not: stepping 64x more worlds costs far less than 64x, because a small world count
leaves the GPU idle. So the sub-linearity that makes 512 worlds efficient is exactly what
makes splitting the batch across GPUs a *smaller* win than 8x. Measure it rather than assume.

    python -m sugar_newton.rl.bench_scaling --envs 64,128,256,512

Reports env-steps/s and the projected 3000-iteration wall clock for each split. Physics
dominates, so this times ``env.step`` only -- the policy forward and the DDP all-reduce are
not included, and the all-reduce in particular makes the real 8-GPU number worse than the
projection here. Treat the projection as an upper bound on the speedup.
"""

from __future__ import annotations

import argparse
import time

import torch
import warp as wp

STEPS_PER_ITER = 24     # runner_cfg num_steps_per_env; one PPO collection
TARGET_ITERS = 3000     # the full BCPPO curriculum


def bench(num_envs: int, args, warmup: int = 2, trials: int = 3) -> dict:
    """Time ``STEPS_PER_ITER`` env steps at ``num_envs`` worlds, after warmup."""
    from sugar_newton.rl.vec_env import make

    env = make(num_envs, clip_names=args.clips, episode_length=args.episode_length,
               substeps=args.substeps, mu=args.mu, device=args.device, seed=0,
               box_tris=args.box_tris, hand_tris=args.hand_tris, margin=args.margin)
    act = torch.zeros(num_envs, env.num_actions, device=args.device)

    def collection() -> float:
        # Synchronise on both sides: warp kernels and torch ops share the stream, and a
        # missing sync here silently measures launch time instead of execution time.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS_PER_ITER):
            env.step(act)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    env.reset()
    for _ in range(warmup):                      # CUDA graph capture lands in the first pass
        collection()
    times = sorted(collection() for _ in range(trials))
    dt = times[len(times) // 2]                  # median, so one preempted trial cannot skew it

    steps = STEPS_PER_ITER * num_envs
    del env
    torch.cuda.empty_cache()
    return {"envs": num_envs, "iter_s": dt, "steps": steps, "rate": steps / dt}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", default="64,128,256,512",
                   help="comma-separated worlds-per-GPU counts to time")
    p.add_argument("--batch", type=int, default=512 * STEPS_PER_ITER,
                   help="total samples per iteration to hold fixed across splits")
    p.add_argument("--clips", nargs="*", default=None)
    p.add_argument("--episode-length", type=int, default=300)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--box-tris", type=int, default=2000)
    p.add_argument("--hand-tris", type=int, default=0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    wp.init()
    counts = [int(c) for c in args.envs.split(",") if c.strip()]

    rows = []
    for n in counts:
        try:
            rows.append(bench(n, args))
            r = rows[-1]
            print(f"[bench] {n:4d} worlds: {r['iter_s']:6.2f} s / {STEPS_PER_ITER} steps "
                  f"= {r['rate']:7.1f} env-steps/s", flush=True)
        except Exception as exc:                 # OOM at the top end is a result, not a crash
            print(f"[bench] {n:4d} worlds: FAILED {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        return
    print(f"\n{'worlds/GPU':>11} {'GPUs':>5} {'s/iter':>8} {'env-steps/s':>12} "
          f"{'3000 iters':>11}  vs 1x512")
    base = next((r for r in rows if r["envs"] == 512), rows[-1])
    for r in rows:
        gpus = max(1, args.batch // r["steps"])  # GPUs needed to keep the batch fixed
        hours = r["iter_s"] * TARGET_ITERS / 3600
        print(f"{r['envs']:>11} {gpus:>5} {r['iter_s']:>8.2f} {r['rate']:>12.1f} "
              f"{hours:>10.1f} h  {base['iter_s'] / r['iter_s']:>5.2f}x")
    print("\nSpeedup column assumes perfect DDP overlap; the all-reduce makes it optimistic.")


if __name__ == "__main__":
    main()
