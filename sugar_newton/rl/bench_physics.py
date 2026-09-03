# SPDX-License-Identifier: BSD-3-Clause
"""Attribute our ~150x per-GPU throughput deficit against SUGAR/PhysX to individual settings.

:mod:`sugar_newton.rl.bench_scaling` answers "how does cost scale with worlds"; this answers
a different question with the same timing discipline: *how much of the gap is the physics we
asked for, rather than Newton itself?* Measured against SUGAR's released recipe -- 4096 envs
x 30001 iterations x 24 steps on ONE GPU in under a day, i.e. ~34,100 env-steps/s per GPU --
where this env collects 234.

The gap is not one thing, so one lumped "Newton is slow" number is useless for deciding what
to change. Each row below moves exactly ONE setting away from today's configuration, so the
cost lands on the setting that caused it:

    baseline        what training runs now: triangle-mesh colliders, box decimated to 2000
                    triangles, hands at the asset's full 45748 / 43852, solver 100/50
    hands-5000      hand colliders decimated; the configuration the tactile channel was
                    actually validated on (validation.g1_carrybox_policy.decimate_hand_colliders)
    convex          collision="convex": every collider becomes a <=64-vertex convex hull and
                    leaves the triangle narrow phase entirely. This is what PhysX does -- it
                    CANNOT collide two dynamic triangle meshes at all -- so it is the
                    matched-geometry control, not a proposal
    iters-8/4       SUGAR's own solver budget (assets/robots/unitree.py:43-44), against the
                    100/50 this env asks for
    physx-matched   all three, plus self collisions on, which SUGAR enables and we disable

Rows are driven by SUGAR's released 510-D tracker, not zero actions, because contact cost is
set by the actual grasp: zero actions never close the hands and would understate every row
that contact dominates. ``--policy zero`` is available for cross-checking against
``bench_scaling``, which uses it.

    python -m sugar_newton.rl.bench_physics --envs 512
    python -m sugar_newton.rl.bench_physics --lift        # does the tracker still lift?

Timing covers the whole collection -- observation, policy forward and physics -- because that
is what a training iteration pays and what the 234 figure it is compared against measured.
The policy is a small MLP and is sub-millisecond at 512 worlds; it is not what separates these
rows.

``--lift`` is the other half of the answer, and it is not optional: a configuration that is
fast but drops the box is not a comparison point. It rolls the tracker out in one world per
configuration and reports peak box lift against ``data_000``'s 0.628 m reference.
"""

from __future__ import annotations

import argparse
import time

import torch
import warp as wp

from sugar_newton.rl.carrybox_env import N_DOF, OBS_DIM

STEPS_PER_ITER = 24         # runner_cfg num_steps_per_env; one PPO collection
SUGAR_ITERS = 30001         # SUGAR's released tracker recipe
SUGAR_ENVS = 4096
SUGAR_RATE = 34_100         # env-steps/s per GPU implied by that recipe finishing in a day

# Deltas from the baseline, one setting each, so a row's cost is attributable. The baseline
# itself is empty: it is whatever the env defaults to, which is the point of comparison.
CONFIGS: dict[str, dict] = {
    "baseline": {},
    "hands-5000": {"hand_tris": 5000},
    "convex": {"collision": "convex"},
    "iters-8/4": {"iterations": 8, "ls_iterations": 4},
    "physx-matched": {"collision": "convex", "iterations": 8, "ls_iterations": 4,
                      "self_collisions": True},
}


def make_env(num_envs: int, overrides: dict, args):
    from sugar_newton.rl.carrybox_env import CarryBoxEnv

    kwargs = dict(clip_names=args.clips, episode_length=args.episode_length,
                  substeps=args.substeps, mu=args.mu, device=args.device, seed=0,
                  box_tris=args.box_tris, margin=args.margin)
    kwargs.update(overrides)
    return CarryBoxEnv(num_envs=num_envs, **kwargs)


def load_policy(args, device):
    """SUGAR's released tracker, or None for zero actions."""
    if args.policy == "zero":
        return None
    # Reuse the loader that already reads this checkpoint rather than a second copy of the
    # layer-shape walk; the NPZ is written by validation.make_policy_assets.
    from sugar_newton.rl.check_teacher import TRACKER_NPZ, load_tracker

    net, n_in = load_tracker(args.tracker or TRACKER_NPZ, device)
    if n_in != OBS_DIM:
        raise RuntimeError(f"tracker expects {n_in}-D, env observes {OBS_DIM}-D")
    return net


@torch.no_grad()
def bench(num_envs: int, overrides: dict, args, warmup: int = 2, trials: int = 3) -> dict:
    """Time ``STEPS_PER_ITER`` steps at ``num_envs`` worlds under ``overrides``, after warmup."""
    env = make_env(num_envs, overrides, args)
    net = load_policy(args, args.device)
    zero = torch.zeros(num_envs, N_DOF, device=args.device)

    def collection() -> float:
        # Synchronise on both sides: warp kernels and torch ops share the stream, and a
        # missing sync here silently measures launch time instead of execution time.
        obs = env.observe()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS_PER_ITER):
            obs, _, _, _ = env.step(zero if net is None else net(obs))
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    env.reset()
    for _ in range(warmup):                      # CUDA graph capture lands in the first pass
        collection()
    diverged0 = env.num_diverged
    times = sorted(collection() for _ in range(trials))
    dt = times[len(times) // 2]                  # median, so one preempted trial cannot skew it

    out = {"iter_s": dt, "rate": STEPS_PER_ITER * num_envs / dt,
           "ms_step": 1000.0 * dt / STEPS_PER_ITER,
           "diverged": env.num_diverged - diverged0,
           "convex_tris": env.convex_tris}
    del env, net
    torch.cuda.empty_cache()
    return out


def lift(overrides: dict, args) -> dict:
    """Roll the tracker out in one world and report whether the box still comes up.

    ``episode_length`` is left huge and tracking termination off so the rollout is not cut by
    the 0.3 m bound -- drift is reported instead of ending the measurement, which is what
    makes lift comparable across configurations that drift at different frames.
    """
    from sugar_newton.rl.check_teacher import rollout

    env = make_env(1, {**overrides, "episode_length": 10 ** 9,
                       "track_termination": False}, args)
    net = load_policy(args, args.device)
    if net is None:
        raise RuntimeError("--lift needs the tracker; --policy zero cannot lift the box")
    out = rollout(env, net, "policy", args.frames)
    del env, net
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", type=int, default=512,
                   help="worlds per GPU; 512 is the measured best per-GPU point")
    p.add_argument("--rows", default="",
                   help=f"comma-separated subset of {','.join(CONFIGS)}")
    p.add_argument("--lift", action="store_true",
                   help="fidelity instead of speed: tracker lift per configuration, 1 world")
    p.add_argument("--frames", type=int, default=400, help="--lift rollout length")
    p.add_argument("--policy", choices=("tracker", "zero"), default="tracker")
    p.add_argument("--tracker", default="", help="override the tracker NPZ path")
    p.add_argument("--clips", nargs="*", default=None)
    p.add_argument("--episode-length", type=int, default=300)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--box-tris", type=int, default=2000)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    wp.init()
    names = [n for n in (args.rows.split(",") if args.rows else CONFIGS) if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown rows {unknown}; known: {', '.join(CONFIGS)}")

    if args.lift:
        rows = {}
        for name in names:
            try:
                rows[name] = lift(CONFIGS[name], args)
                r = rows[name]
                print(f"[lift] {name:14} {r['box_lift']:.3f} m of "
                      f"{r['box_lift_reference']:.3f} m, drift frame {r['drift_step']}",
                      flush=True)
            except Exception as exc:
                print(f"[lift] {name:14} FAILED {type(exc).__name__}: {exc}", flush=True)
        print(f"\n{'configuration':16} {'lift m':>8} {'ref m':>7} {'drift':>7} "
              f"{'|a|':>6} {'sat':>6}")
        for name, r in rows.items():
            print(f"{name:16} {r['box_lift']:>8.3f} {r['box_lift_reference']:>7.3f} "
                  f"{r['drift_step']:>7} {r['action_mag']:>6.2f} "
                  f"{100 * r['action_saturated']:>5.0f}%")
        print("\nA configuration that is fast and cannot hold the box is not a comparison"
              "\npoint. drift is the frame the 0.3 m tracking bound is first exceeded.")
        return

    rows = {}
    for name in names:
        try:
            rows[name] = bench(args.envs, CONFIGS[name], args)
            r = rows[name]
            extra = ""
            if r["convex_tris"]:
                extra = f", colliders {r['convex_tris'][0]} -> {r['convex_tris'][1]} tris"
            print(f"[bench] {name:14} {r['ms_step']:8.1f} ms/step "
                  f"{r['rate']:9.1f} env-steps/s{extra}", flush=True)
        except Exception as exc:                 # OOM is a result, not a crash
            print(f"[bench] {name:14} FAILED {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        return
    base = rows.get("baseline") or next(iter(rows.values()))
    recipe = SUGAR_ITERS * STEPS_PER_ITER * SUGAR_ENVS
    print(f"\n{args.envs} worlds/GPU, {args.policy} policy, "
          f"box_tris={args.box_tris}, substeps={args.substeps}, mu={args.mu}")
    print(f"\n{'configuration':16} {'ms/step':>9} {'env-steps/s':>12} {'vs base':>8} "
          f"{'30001-iter recipe':>18} {'div':>5}")
    for name, r in rows.items():
        print(f"{name:16} {r['ms_step']:>9.1f} {r['rate']:>12.1f} "
              f"{r['rate'] / base['rate']:>7.2f}x {recipe / r['rate'] / 86400:>15.1f} d "
              f"{r['diverged']:>5}")
    print(f"{'SUGAR (PhysX)':16} {'':>9} {SUGAR_RATE:>12.1f} "
          f"{SUGAR_RATE / base['rate']:>7.2f}x {recipe / SUGAR_RATE / 86400:>15.1f} d")
    print(f"\nRecipe column is SUGAR's own: {SUGAR_ENVS} envs x {SUGAR_ITERS} iterations x "
          f"{STEPS_PER_ITER} steps\n({recipe / 1e9:.2f}e9 env-steps) at that row's rate, on"
          f" one GPU. 'div' counts diverged worlds\nduring the timed collections -- a row"
          f" that is fast because it exploded is not fast.")


if __name__ == "__main__":
    main()
