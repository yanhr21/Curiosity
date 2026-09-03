"""End-to-end probe for sugar_swap: build SUGAR's refiner env on Newton, reset, step.

Run inside the dev-node container:

    bash slurm/devrun.sh "source env/activate.sh && python slurm/_swap_probe.py"
    bash slurm/devrun.sh "source env/activate.sh && python slurm/_swap_probe.py --bench"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (
    REPO,
    REPO / "IsaacLab" / "source" / "isaaclab",
    REPO / "IsaacLab" / "source" / "isaaclab_tasks",
    REPO / "SUGAR" / "source" / "sugar_rl",
    REPO / "SUGAR" / "source" / "sugar_il",
):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(REPO))

from sugar_swap import bootstrap  # noqa: E402

bootstrap.install()

import torch  # noqa: E402

import sugar_rl.tasks  # noqa: E402,F401
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

TASK = "Sugar-G129dof-CarryBox-Refiner"


def make_env(num_envs: int):
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    # SUGAR leaves this None in the config and supplies it as `--motion_folder` from its own
    # launchers, which run with SUGAR/ as the working directory.
    cfg.commands.motion.motion_folder = str(REPO / "SUGAR" / "data" / "CarryBox")
    return ManagerBasedRLEnv(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    n = 512 if args.bench else args.envs

    t0 = time.time()
    env = make_env(n)
    print(f"[1] env constructed in {time.time() - t0:.1f}s", flush=True)
    print(f"    sensors: {sorted(env.scene.sensors)}", flush=True)
    for k, s in sorted(env.scene.sensors.items()):
        print(f"      {k}: {s.num_bodies} bodies, {s._n_filter} filters", flush=True)
    print(f"    num_actions={env.num_actions} device={env.device}", flush=True)

    obs, _ = env.reset()
    print("[2] reset() observation groups:", flush=True)
    for g, v in obs.items():
        print(f"      {g}: {tuple(v.shape)}", flush=True)

    h0 = env.scene["robot"].data.root_pos_w[:, 2].clone()
    print(f"[5] root height at reset: {h0.tolist()[:8]}", flush=True)

    if args.bench:
        for _ in range(20):
            env.step(torch.zeros(n, env.num_actions, device=env.device))
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(100):
            env.step(torch.zeros(n, env.num_actions, device=env.device))
        torch.cuda.synchronize()
        wall = time.time() - t
        print(f"[T] {n} envs, 100 steps, {wall:.3f}s -> {n * 100 / wall:.0f} env-steps/s", flush=True)
        return 0

    for i in range(args.steps):
        obs, rew, term, timeout, extras = env.step(
            torch.zeros(n, env.num_actions, device=env.device)
        )
    print(f"[3] {args.steps} steps ok", flush=True)
    print(f"    reward: {rew.tolist()}", flush=True)
    print(f"    finite: {bool(torch.isfinite(rew).all())}", flush=True)
    print(f"    terminated={term.tolist()} timeout={timeout.tolist()}", flush=True)

    print("[4] per-term reward breakdown:", flush=True)
    ep = extras.get("log", {})
    nonzero, zero = [], []
    for k, v in env.reward_manager._episode_sums.items():
        val = float(v.mean())
        (nonzero if abs(val) > 0 else zero).append((k, val))
    for k, val in nonzero:
        print(f"      NONZERO {k}: {val:+.6g}", flush=True)
    for k, val in zero:
        print(f"      zero    {k}: {val:+.6g}", flush=True)
    print(f"    {len(nonzero)} non-zero of {len(nonzero) + len(zero)}", flush=True)
    if ep:
        print(f"    extras['log'] keys: {sorted(ep)[:5]} ...", flush=True)

    h1 = env.scene["robot"].data.root_pos_w[:, 2]
    print(f"[5] root height after {args.steps} steps: {h1.tolist()[:8]}", flush=True)
    print(f"    delta: {(h1 - h0).tolist()[:8]}", flush=True)

    # The shared reduction in scene.contact_digest() is compared against a direct
    # per-contact sum, because sixteen SUGAR terms read these forces and a wrong body
    # attribution would be invisible in the reward totals.
    import warp as wp

    c = env.scene.contacts
    nc = int(wp.to_torch(c.rigid_contact_count)[0])
    s0 = wp.to_torch(c.rigid_contact_shape0)[:nc].long()
    s1 = wp.to_torch(c.rigid_contact_shape1)[:nc].long()
    f = wp.to_torch(c.force)[:nc, :3]
    sb = env.scene.shape_body
    ref = torch.zeros(env.scene.total_bodies, 3, device=env.device)
    for i in range(nc):
        b0, b1 = int(sb[s0[i]]), int(sb[s1[i]])
        if b0 >= 0:
            ref[b0] += f[i]
        if b1 >= 0:
            ref[b1] -= f[i]
    worst = 0.0
    for k, s in sorted(env.scene.sensors.items()):
        got = s.data.net_forces_w
        want = ref[s._body_indices]
        worst = max(worst, float((got - want).abs().max()))
    print(f"[V] shared-reduction vs brute force over {nc} contacts: max abs diff {worst:.3g}", flush=True)

    print("[C] contact sensor diagnostics:", flush=True)
    for k, s in sorted(env.scene.sensors.items()):
        net = s.data.net_forces_w.norm(dim=-1)
        line = f"      {k}: |net| max={float(net.max()):.3g} nnz={int((net > 1e-6).sum())}"
        if s.data.force_matrix_w is not None:
            fm = s.data.force_matrix_w.norm(dim=-1)
            line += f" |filter| max={float(fm.max()):.3g} nnz={int((fm > 1e-6).sum())}"
        if s.cfg.track_air_time:
            line += f" air_max={float(s.data.current_air_time.max()):.3g}"
            line += f" contact_max={float(s.data.current_contact_time.max()):.3g}"
        print(line, flush=True)

    leaked = [m for m in sys.modules if m.startswith(("isaacsim", "omni.physx"))]
    print(f"[6] isaacsim/omni.physx modules: {leaked}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
