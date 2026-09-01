# SPDX-License-Identifier: BSD-3-Clause
"""Speed against physical accuracy for the tactile Allegro scene.

Speed alone is easy and worthless: dropping substeps and solver iterations makes the
clock happy and the contact soft, and a soft contact is exactly the thing that shows up
as fingers sunk into the cube and as tactile readings that were never real. So every
configuration is reported with what it COSTS as well as what it buys:

    fps            steps per second of wall clock (target: 15)
    pen            penetration depth, p99 and max [mm] -- the accuracy that matters most
    load           mean total normal load [N]
    slip           p99 slip velocity [mm/s]
    faces          mean contact-surface faces; the physics is only meaningful if > 0
    held           whether the cube was still in the hand at the end

Read the table for the cheapest row that keeps penetration low. A row with ``held=no``
is timing an empty solve and means nothing, whatever its fps says.

Two speed numbers, because they answer different questions: **sim** is stepping only,
which is what a training loop pays; **+readback** adds the per-step host copy of the
channels and the per-face field, which is what recording a video pays and which a policy
never does.

    python -m sugar_newton.validation.allegro_bench --ke 1e3 1e5 --substeps 4 8
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np
import warp as wp

from sugar_newton.validation.allegro_tactile import AllegroTactileScene

FPS = 50.0


def time_steps(scene, dt: float, substeps: int, n: int, readback: bool,
               field_max: int) -> tuple[float, float]:
    """Steps per second (warm) and the mean live contact count over the window.

    The contact count is reported because hydroelastic cost scales with it, so a
    configuration that DROPPED the cube benchmarks beautifully and means nothing. Only
    collected on the readback pass -- sampling it during the sim pass would be the very
    host round trip that pass exists to exclude.
    """
    wp.synchronize_device()
    faces = 0.0
    t0 = time.perf_counter()
    for _ in range(n):
        scene.step(dt, substeps=substeps)
        if readback:
            scene.tactile.to_numpy()
            if scene.field is not None:
                scene.field.to_numpy(stride_to=field_max)
                faces += scene.field.total
    wp.synchronize_device()
    el = time.perf_counter() - t0
    return (n / el if el > 0 else 0.0), faces / n


def verify_graph(scene, dt: float, substeps: int, frames: int) -> tuple[float, float]:
    """Does the captured graph produce the same trajectory as the plain loop?

    Worth asking rather than assuming: :meth:`AllegroTactileScene.rock_wrist` writes
    ``joint_X_p`` and calls ``notify_model_changed`` every step, from OUTSIDE the graph.
    That is safe only while the solver keeps updating the same buffers -- if it ever
    reallocated them, the graph would replay against stale memory and the physics would
    be quietly wrong, not loudly broken. Compares the cube's pose either way.
    """
    cube_body = int(scene.model.shape_body.numpy()[scene.cube_shapes[0]])

    def run(use_graph: bool) -> np.ndarray:
        scene.reset()
        scene.graph = None
        if use_graph:
            scene.capture(dt, substeps)
        out = np.zeros((frames, 7))
        for i in range(frames):
            scene.step(dt, substeps=substeps)
            out[i] = scene.state_0.body_q.numpy()[cube_body]
        return out

    plain, graphed = run(False), run(True)
    dp = np.linalg.norm(plain[:, :3] - graphed[:, :3], axis=1)
    dq = np.degrees(2.0 * np.arccos(np.clip(np.abs((plain[:, 3:] * graphed[:, 3:]).sum(1)), -1, 1)))
    return float(dp.max()), float(dq.max())


def accuracy(scene, dt: float, substeps: int, n: int) -> dict:
    """Penetration, load and slip over an untimed window, plus "is the cube still here".

    Untimed on purpose: reading these back every step is a host round trip that would
    show up in the fps column as a cost the training loop does not actually pay.
    """
    palm = scene.palm_body
    cube_body = int(scene.model.shape_body.numpy()[scene.cube_shapes[0]])
    pen, load, slip, live, reach = [], [], [], [], []
    for _ in range(n):
        scene.step(dt, substeps=substeps)
        ch = scene.tactile.to_numpy()
        pen.append(float(scene.tactile._peak_depth.numpy().max()))
        load.append(float(ch["normal_load"].sum()))
        slip.append(float(ch["slip_velocity"].max()))
        live.append(int((ch["contact_count"] > 0).sum()))
        bq = scene.state_0.body_q.numpy()
        reach.append(float(np.linalg.norm(bq[cube_body, :3] - bq[palm, :3])))
    pen, load, slip = np.array(pen), np.array(load), np.array(slip)
    return dict(
        pen_p99=float(np.percentile(pen, 99)) * 1e3,
        pen_max=float(pen.max()) * 1e3,
        load=float(load.mean()),
        slip_p99=float(np.percentile(slip, 99)) * 1e3,
        live=float(np.mean(live)),
        held=bool(reach[-1] < 0.15 and np.mean(live) > 0.5),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=150, help="timed steps per configuration")
    ap.add_argument("--target-fps", type=float, default=15.0,
                    help="the bar a configuration has to clear before accuracy is compared")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--profile-frames", type=int, default=40)
    ap.add_argument("--kh", type=float, default=1.0e10)
    ap.add_argument("--substeps", type=int, nargs="+", default=[16])
    ap.add_argument("--sdf-res", type=int, nargs="+", default=[48])
    ap.add_argument("--buffer-fraction", type=float, nargs="+", default=[1.0])
    ap.add_argument("--buffer-mult-iso", type=int, nargs="+", default=[2])
    ap.add_argument("--field", type=int, nargs="+", default=[1],
                    help="1 = also build the per-face field, 0 = per-patch channels only")
    ap.add_argument("--graph", type=int, nargs="+", default=[0],
                    help="1 = capture the substep loop as a CUDA graph")
    ap.add_argument("--ke", type=float, nargs="+", default=[1.0e3],
                    help="contact stiffness [N/m] -- the penetration knob")
    ap.add_argument("--kd", type=float, nargs="+", default=[1.0e2])
    ap.add_argument("--iterations", type=int, nargs="+", default=[100])
    ap.add_argument("--ls-iterations", type=int, nargs="+", default=[50])
    ap.add_argument("--cone", nargs="+", default=["elliptic"])
    ap.add_argument("--settle-frames", type=int, default=90,
                    help="steps run before timing, so the cube is in the hand")
    ap.add_argument("--accuracy-frames", type=int, default=120,
                    help="untimed steps used to measure penetration and loads")
    ap.add_argument("--verify-graph", type=int, default=0,
                    help="check the captured graph reproduces the plain loop's trajectory")
    ap.add_argument("--field-max", type=int, default=6000)
    ap.add_argument("--grasp", type=float, default=0.34)
    ap.add_argument("--amplitude", type=float, default=0.08)
    ap.add_argument("--rate", type=float, default=1.4)
    ap.add_argument("--rock", type=float, default=0.25)
    ap.add_argument("--swing-mode", default="sym")
    args = ap.parse_args()

    wp.init()
    if not wp.get_device().is_cuda:
        print("ERROR: hydroelastic SDF is CUDA-only.")
        return 2
    dt = 1.0 / FPS
    print(f"{wp.get_device()}  real time = {FPS:.0f} steps/s\n", flush=True)

    header = (f"{'ke':>7} {'it':>4} {'ls':>3} {'sub':>4} {'gr':>3} | {'fps':>6} {'xRT':>6} | "
              f"{'+rb':>6} | {'pen99':>6} {'penmx':>6} | {'load':>6} {'slip99':>7} "
              f"{'links':>6} {'held':>5}")
    print(header)
    print("-" * len(header), flush=True)

    rows = []
    for sdf, bf, iso, fld, it, ls, cone, ke, kd in itertools.product(
        args.sdf_res, args.buffer_fraction, args.buffer_mult_iso, args.field,
        args.iterations, args.ls_iterations, args.cone, args.ke, args.kd,
    ):
        # Everything here is baked in at build time, so each combination costs a rebuild.
        scene = AllegroTactileScene(kh=args.kh, sdf_res=sdf, buffer_fraction=bf,
                                    buffer_mult_iso=iso, want_field=bool(fld),
                                    iterations=it, ls_iterations=ls, cone=cone,
                                    ke=ke, kd=kd)
        scene.grasp, scene.amplitude = args.grasp, args.amplitude
        scene.rate, scene.rock, scene.swing_mode = args.rate, args.rock, args.swing_mode
        for sub, gr in itertools.product(args.substeps, args.graph):
            if gr and sub % 2:
                continue
            scene.reset()
            # Settle FIRST: hydroelastic cost scales with the contact set, and timing a
            # scene that has thrown its cube on the floor measures an empty solve.
            for _ in range(args.settle_frames):
                scene.step(dt, substeps=sub)
            if gr:
                scene.capture(dt, sub)
            for _ in range(args.warmup):
                scene.step(dt, substeps=sub)
            sim, _ = time_steps(scene, dt, sub, args.frames, False, args.field_max)
            rb, faces = time_steps(scene, dt, sub, args.frames, True, args.field_max)
            acc = accuracy(scene, dt, sub, args.accuracy_frames)
            row = dict(ke=ke, kd=kd, it=it, ls=ls, cone=cone, sub=sub, graph=gr,
                       fps=sim, rb=rb, faces=faces, **acc)
            rows.append(row)
            print(f"{ke:>7.0e} {it:>4} {ls:>3} {sub:>4} {gr:>3} | {sim:>6.1f} "
                  f"{sim / FPS:>5.2f}x | {rb:>6.1f} | {acc['pen_p99']:>6.2f} "
                  f"{acc['pen_max']:>6.2f} | {acc['load']:>6.2f} {acc['slip_p99']:>7.1f} "
                  f"{acc['live']:>6.1f} {'yes' if acc['held'] else 'NO':>5}", flush=True)

        # phase breakdown for this build, at the last substep count tried
        scene.reset()
        scene.graph = None
        for _ in range(args.settle_frames):
            scene.step(dt, substeps=args.substeps[-1])
        scene.timings.clear()
        scene.profile = True
        for _ in range(args.profile_frames):
            scene.step(dt, substeps=args.substeps[-1])
        scene.profile = False
        total = sum(scene.timings.values())
        parts = "  ".join(f"{k}={1e3 * v / args.profile_frames:.1f}ms"
                          f"({100 * v / total:.0f}%)" for k, v in scene.timings.items())
        print(f"     phases (sdf={sdf}, sub={args.substeps[-1]}, synced): {parts}", flush=True)

    ok = [r for r in rows if r["held"] and r["fps"] >= args.target_fps]
    if ok:
        best = min(ok, key=lambda r: r["pen_p99"])
        print(f"\ncheapest penetration at >= {args.target_fps:g} fps: "
              f"--ke {best['ke']:.0e} --kd {best['kd']:.0e} --iterations {best['it']} "
              f"--ls-iterations {best['ls']} --cone {best['cone']} --substeps {best['sub']}"
              + (" --graph" if best["graph"] else "") +
              f"  ->  {best['fps']:.1f} fps, penetration {best['pen_p99']:.2f} mm (p99), "
              f"{best['live']:.1f} links loaded")
    else:
        held = [r for r in rows if r["held"]]
        if not held:
            print("\nno configuration kept the cube in the hand -- every row is an empty solve")
        else:
            fastest = max(held, key=lambda r: r["fps"])
            print(f"\nnothing reached {args.target_fps:g} fps while holding the cube; "
                  f"fastest valid row was {fastest['fps']:.1f} fps at "
                  f"--substeps {fastest['sub']} --iterations {fastest['it']}")

    if args.verify_graph and 1 in args.graph:
        sub = max(s for s in args.substeps if s % 2 == 0)
        scene = AllegroTactileScene(kh=args.kh, sdf_res=args.sdf_res[0],
                                    iterations=args.iterations[0],
                                    ls_iterations=args.ls_iterations[0], cone=args.cone[0])
        scene.grasp, scene.amplitude = args.grasp, args.amplitude
        scene.rate, scene.rock, scene.swing_mode = args.rate, args.rock, args.swing_mode
        dp, dq = verify_graph(scene, dt, sub, 120)
        print(f"\ngraph vs plain loop over 120 steps at --substeps {sub}: "
              f"cube position differs by {dp * 1e3:.3f} mm, orientation by {dq:.3f} deg",
              flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
