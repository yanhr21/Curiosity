"""Where the sugar_swap step time goes at 512 envs.

Splits one policy step into the Newton solve, the contact readback, the sensor reduction and
the manager work, so an optimisation is aimed at whatever actually dominates.

    bash slurm/devrun.sh "source env/activate.sh && python slurm/_swap_profile.py"
"""

from __future__ import annotations

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

from sugar_swap import bootstrap  # noqa: E402

bootstrap.install()

import torch  # noqa: E402
import warp as wp  # noqa: E402

import sugar_rl.tasks  # noqa: E402,F401
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

N = 512
REPS = 30


def sync():
    wp.synchronize()
    torch.cuda.synchronize()


def timeit(label: str, fn, reps: int = REPS) -> float:
    for _ in range(3):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    el = (time.perf_counter() - t) / reps
    print(f"  {label:<34s} {el * 1e3:8.2f} ms/policy-step", flush=True)
    return el


def main() -> int:
    cfg = load_cfg_from_registry("Sugar-G129dof-CarryBox-Refiner", "env_cfg_entry_point")
    cfg.scene.num_envs = N
    cfg.commands.motion.motion_folder = str(REPO / "SUGAR" / "data" / "CarryBox")
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    act = torch.zeros(N, env.num_actions, device=env.device)

    scene = env.scene
    dec = env.cfg.decimation
    dt = env.physics_dt

    for _ in range(20):
        env.step(act)

    live = int(wp.to_torch(scene.contacts.rigid_contact_count)[0])
    print(
        f"[profile] {N} envs, decimation={dec}, sim dt={dt}, "
        f"contact capacity={scene.contacts.rigid_contact_max} live={live}",
        flush=True,
    )

    def collide_only():
        for _ in range(dec):
            scene.state_0.clear_forces()
            scene.pipeline.collide(scene.state_0, scene.contacts)

    def solve_only():
        for _ in range(dec):
            scene.state_0.clear_forces()
            scene.pipeline.collide(scene.state_0, scene.contacts)
            scene.solver.step(
                scene.state_0, scene.state_1, scene.control, scene.contacts, dt
            )
            scene.swap_states()

    def solve_readback():
        solve_only()
        for _ in range(dec):
            scene.solver.update_contacts(scene.contacts, scene.state_0)

    def sensors_only():
        for _ in range(dec):
            for s in scene.sensors.values():
                s.update(dt)

    def assets_only():
        for _ in range(dec):
            for a in (*scene.articulations.values(), *scene.rigid_objects.values()):
                a.update(dt)

    def obs_only():
        env.observation_manager.compute()

    def rew_only():
        env.reward_manager.compute(dt=env.step_dt)

    def term_only():
        env.termination_manager.compute()

    def cmd_only():
        env.command_manager.compute(dt=env.step_dt)

    def full_step():
        env.step(act)

    timeit("collide x4", collide_only)
    timeit("collide+solve x4", solve_only)
    timeit("collide+solve+update_contacts x4", solve_readback)
    timeit("sensor update x4 (8 sensors)", sensors_only)
    timeit("asset update x4", assets_only)
    timeit("observation_manager.compute", obs_only)
    timeit("reward_manager.compute", rew_only)
    timeit("termination_manager.compute", term_only)
    timeit("command_manager.compute", cmd_only)
    el = timeit("FULL env.step", full_step)
    print(f"  -> {N / el:.0f} env-steps/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
