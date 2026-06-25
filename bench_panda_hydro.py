# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Throwaway benchmark: FPS vs collision quality for the panda_hydro example.
#
# Drives the real `example_robot_panda_hydro.Example` (Franka pick-and-place with
# hydroelastic SDF contacts + MuJoCo "newton" solver), overriding the solver
# `iterations`/`ls_iterations`, and measures:
#   - FPS: median isolated per-frame wall time (sync-before / sync-after), so the
#     comparison across settings is apples-to-apples.
#   - Penetration: signed-distance overlap from the contact buffers, tracked over
#     the grasp/lift/place trajectory (max + mean of penetrating contacts).
#
# Prints one CSV line so a sweep can fan out across processes.

import argparse
import time

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
from bench_fps_vs_collision import penetration_stats
from newton.examples.robot.example_robot_panda_hydro import Example


def main():
    base = Example.create_parser()
    base.add_argument("--iters", type=int, default=15)
    base.add_argument("--ls-iters", type=int, default=100)
    base.add_argument("--impratio", type=float, default=None)  # None => keep example's 1000
    base.add_argument("--kh", type=float, default=None)  # None => keep example's 1e11
    base.add_argument("--substeps", type=int, default=None)  # None => keep example's 10
    base.add_argument("--warmup", type=int, default=60)
    base.add_argument("--sample-frames", type=int, default=260)
    args = base.parse_args()

    if args.device:
        wp.set_device(args.device)

    # Override the solver's iteration counts / impratio wherever the example constructs it.
    orig_solver = newton.solvers.SolverMuJoCo

    def patched_solver(*a, **kw):
        kw["iterations"] = args.iters
        kw["ls_iterations"] = args.ls_iters
        if args.impratio is not None:
            kw["impratio"] = args.impratio
        return orig_solver(*a, **kw)

    newton.solvers.SolverMuJoCo = patched_solver

    # Override hydroelastic contact stiffness kh on every ShapeConfig the example builds.
    if args.kh is not None:
        SC = newton.ModelBuilder.ShapeConfig
        orig_sc_init = SC.__init__

        def sc_init(self, *a, **kw):
            orig_sc_init(self, *a, **kw)
            self.kh = args.kh

        SC.__init__ = sc_init

    viewer = newton.viewer.ViewerNull(num_frames=args.warmup + args.sample_frames)

    if args.substeps is not None:
        # Skip the in-__init__ graph capture and capture once afterward with the
        # new substeps. Re-capturing a mujoco_warp graph triggers capture-time
        # allocation -> CUDA illegal memory access.
        real_capture = Example.capture
        Example.capture = lambda self: setattr(self, "graph", None)
        ex = Example(viewer, args)
        Example.capture = real_capture
        ex.sim_substeps = args.substeps
        ex.sim_dt = ex.frame_dt / args.substeps
        ex.capture()
    else:
        ex = Example(viewer, args)

    frame_times = []
    pen_max = 0.0
    pen_means = []
    frac_list = []
    max_contacts = 0

    # Grasp-success + blow-up tracking (catches that penetration depth misses).
    obj_idx = [w * ex.bodies_per_world + ex.object_body_local for w in range(ex.world_count)]
    z0 = ex.object_pos[2]
    lift_per_world_max = np.full(ex.world_count, -1e9)
    blew = False

    total = args.warmup + args.sample_frames
    for f in range(total):
        wp.synchronize_device()
        t0 = time.perf_counter()
        ex.step()
        wp.synchronize_device()
        frame_times.append(time.perf_counter() - t0)
        if f >= args.warmup:
            n, mx, mean, frac = penetration_stats(ex.model, ex.contacts, ex.state_0)
            max_contacts = max(max_contacts, n)
            pen_max = max(pen_max, mx)
            if n > 0:
                pen_means.append(mean)
                frac_list.append(frac)
            bq = ex.state_0.body_q.numpy()
            pos = bq[:, :3]
            if not np.all(np.isfinite(pos)) or np.abs(pos).max() > 50.0:
                blew = True
            lift_per_world_max = np.maximum(lift_per_world_max, bq[obj_idx, 2] - z0)

    steady = sorted(frame_times[args.warmup :])
    median_ft = steady[len(steady) // 2]
    fps = 1.0 / median_ft
    steps_per_s = fps * ex.sim_substeps * ex.world_count
    mean_pen = float(np.mean(pen_means)) if pen_means else 0.0
    frac_pen = float(np.mean(frac_list)) if frac_list else 0.0

    eff_impratio = 1000.0 if args.impratio is None else args.impratio
    eff_kh = 1e11 if args.kh is None else args.kh
    lift_best = float(lift_per_world_max.max()) * 1000.0
    lift_worst = float(lift_per_world_max.min()) * 1000.0
    print(
        f"mujoco-hydro,{args.iters},{args.ls_iters},{ex.sim_substeps},{eff_impratio:g},{eff_kh:g},"
        f"{ex.world_count},{ex.model.body_count},"
        f"{max_contacts},{fps:.1f},{steps_per_s:.0f},"
        f"{pen_max * 1000:.3f},{mean_pen * 1000:.3f},{frac_pen:.3f},"
        f"{lift_best:.1f},{lift_worst:.1f},{int(blew)}"
    )


if __name__ == "__main__":
    main()
