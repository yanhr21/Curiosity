# SPDX-License-Identifier: BSD-3-Clause
"""Report peak triangle-pair usage against the deterministic-packing ceiling.

``carrybox_env`` warns above ~41 worlds that "contacts will be dropped", but that warning is
a static estimate -- ``want = 25_000 * num_envs`` -- calibrated when the box collided as its
full ~100k-triangle mesh. With ``box_tris=2000`` the real demand is far lower, so the warning
may fire on a configuration that never actually drops a contact. Guessing either way is bad:
believing a false warning caps the world count for no reason, and ignoring a true one
silently degrades exactly the grasp contacts this project exists to get right.

So read the counter Newton already maintains:

    python -m sugar_newton.rl.probe_tripairs --envs 512 --steps 120

``NarrowPhase.triangle_pairs_count`` is the live count and ``triangle_pairs.shape[0]`` the
capacity; Newton's own ``verify_narrow_phase_buffers`` kernel (on by default) prints
"Triangle pair buffer overflowed" when the former exceeds the latter, so a clean run with
headroom here means the warning is spurious for this geometry.
"""

from __future__ import annotations

import argparse

import torch
import warp as wp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=120)
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
    from sugar_newton.rl.vec_env import make

    env = make(args.envs, clip_names=args.clips, episode_length=args.episode_length,
               substeps=args.substeps, mu=args.mu, device=args.device, seed=0,
               box_tris=args.box_tris, hand_tris=args.hand_tris, margin=args.margin)
    inner = env.env if hasattr(env, "env") else env
    np_ = inner.pipeline.narrow_phase

    cap = int(np_.triangle_pairs.shape[0])
    act = torch.zeros(args.envs, env.num_actions, device=args.device)
    env.reset()

    peak, total = 0, 0
    for i in range(args.steps):
        env.step(act)
        n = int(np_.triangle_pairs_count.numpy()[0])
        peak, total = max(peak, n), total + n
        if i % 20 == 0:
            print(f"  step {i:4d}: {n:>9,} pairs  ({100 * n / cap:5.1f} % of capacity)",
                  flush=True)

    per_world = peak / args.envs
    print(f"\nworlds            : {args.envs}")
    print(f"capacity          : {cap:,}  (2**20 deterministic-packing ceiling)")
    print(f"peak used         : {peak:,}  ({100 * peak / cap:.1f} % of capacity)")
    print(f"mean used         : {total // args.steps:,}")
    print(f"peak per world    : {per_world:,.0f}   (the static estimate assumes 25,000)")
    print(f"headroom          : {cap / max(peak, 1):.1f}x")
    print(f"max worlds at this rate: {int(cap / max(per_world, 1e-9)):,}")
    print("\nNo 'Triangle pair buffer overflowed' line above means nothing was dropped.")


if __name__ == "__main__":
    main()
