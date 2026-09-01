"""Aggregate throughput of the vectorised CarryBox environment.

The playback path in ``validation/g1_carrybox_policy.py`` runs one world and answers a
different question: whether the physics is right. Training speed is a property of the
batch, since one ``solver.step`` advances every world, so it has to be measured here.

Two things this reports that a bare timer would not:

* **Contacts used against the limits.** ``njmax``/``nconmax`` truncate silently, and every
  throughput number taken while they were clipping was measuring the wrong simulation. A
  fast row with contacts pinned at the limit is not a result.
* **Steps under load, not at reset.** The grasp is where the contact count peaks, so the
  timed window starts after the warm-up has walked the robot into the box.

    python -m sugar_newton.rl.bench_env --envs 1 4 16 64 --steps 60
"""

from __future__ import annotations

import argparse
import time

import torch
import warp as wp

from sugar_newton.rl.carrybox_env import CarryBoxEnv, N_DOF


def _instrument(env) -> dict:
    """Wrap collide and solver.step with synchronising timers.

    Wrapping rather than re-implementing the step keeps this honest: the phases measured are
    literally the calls the env makes, so the split cannot drift from the code it describes.
    The cost is that each phase pays a ``wp.synchronize``, which serialises work the GPU
    would otherwise overlap -- so the instrumented total runs slower than the real step, and
    only the *proportions* should be read, never the absolute instrumented time.
    """
    acc = {"collide": 0.0, "solve": 0.0, "calls": 0}
    raw_collide, raw_step = env.pipeline.collide, env.solver.step

    def timed_collide(*a, **k):
        wp.synchronize()
        t = time.perf_counter()
        r = raw_collide(*a, **k)
        wp.synchronize()
        acc["collide"] += time.perf_counter() - t
        return r

    def timed_step(*a, **k):
        wp.synchronize()
        t = time.perf_counter()
        r = raw_step(*a, **k)
        wp.synchronize()
        acc["solve"] += time.perf_counter() - t
        acc["calls"] += 1
        return r

    env.pipeline.collide, env.solver.step = timed_collide, timed_step
    return acc


def bench(num_envs: int, steps: int, warmup: int, profile: bool = False,
          **env_kwargs) -> dict:
    t_build = time.perf_counter()
    env = CarryBoxEnv(num_envs=num_envs, **env_kwargs)
    wp.synchronize()
    build_s = time.perf_counter() - t_build
    env.reset()
    act = torch.zeros(num_envs, N_DOF, device=env.device)

    for _ in range(warmup):
        env.step(act)
    wp.synchronize()

    peak_contacts = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act)
        peak_contacts = max(peak_contacts, int(
            env.contacts.rigid_contact_count.numpy()[0]))
    wp.synchronize()
    el = time.perf_counter() - t0

    out = {
        "envs": num_envs,
        "build_s": build_s,
        "step_ms": 1e3 * el / steps,
        "aggregate": num_envs * steps / el,
        "contacts": peak_contacts,
        "limit": env.contacts.rigid_contact_max,
    }

    if profile:
        acc = _instrument(env)
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(act)
        wp.synchronize()
        inst = time.perf_counter() - t0
        out["prof"] = {
            "collide": acc["collide"] / inst,
            "solve": acc["solve"] / inst,
            "other": max(0.0, 1.0 - (acc["collide"] + acc["solve"]) / inst),
            "inst_step_ms": 1e3 * inst / steps,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--collision", default="mesh", choices=("mesh", "hydro"))
    ap.add_argument("--contact-refresh", default="step", choices=("step", "substep"))
    ap.add_argument("--clips", nargs="+", default=["data_000"])
    # Left as None rather than 0 so the env's own defaults show through; a literal 0 here
    # would silently benchmark the undecimated collider instead of what training runs.
    ap.add_argument("--box-tris", type=int, default=None,
                    help="decimate the box collider to about this many triangles "
                         "(default: the env's 2000; pass 0 for the asset's 100k)")
    ap.add_argument("--hand-tris", type=int, default=None,
                    help="decimate each rubber-hand collider to about this many triangles "
                         "(default: the env's 0, i.e. the asset's 45k/44k)")
    ap.add_argument("--profile", action="store_true",
                    help="also attribute the step to collide / solve / everything else")
    args = ap.parse_args()

    wp.init()
    tris = {k: v for k, v in (("box_tris", args.box_tris),
                              ("hand_tris", args.hand_tris)) if v is not None}
    print(f"{'envs':>6} {'build_s':>9} {'step_ms':>9} {'env-steps/s':>13} "
          f"{'contacts':>10} {'limit':>8}")
    for n in args.envs:
        try:
            r = bench(n, args.steps, args.warmup, profile=args.profile,
                      clip_names=args.clips, collision=args.collision,
                      contact_refresh=args.contact_refresh, **tris)
        except Exception as exc:                      # OOM is the expected failure at the top
            print(f"{n:>6}  FAILED: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        flag = "  <-- AT LIMIT" if r["contacts"] >= r["limit"] else ""
        print(f"{r['envs']:>6} {r['build_s']:>9.1f} {r['step_ms']:>9.1f} "
              f"{r['aggregate']:>13.1f} {r['contacts']:>10} {r['limit']:>8}{flag}")
        if p := r.get("prof"):
            print(f"       collide {100 * p['collide']:5.1f} %   "
                  f"solve {100 * p['solve']:5.1f} %   "
                  f"obs+reward+python {100 * p['other']:5.1f} %   "
                  f"(instrumented step {p['inst_step_ms']:.1f} ms)")


if __name__ == "__main__":
    main()
