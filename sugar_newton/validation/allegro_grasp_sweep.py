# SPDX-License-Identifier: BSD-3-Clause
"""Find a drive setting that actually manipulates the cube instead of dropping it.

In-hand manipulation has a narrow window. Too little swing and the grasp only breathes
in place -- the first version of this scene, which the tactile field showed as almost
static. Too much and the digits open faster than the cube can follow and it is thrown:
measured 1.15 m of travel and 178 deg of rotation in 3 s, with contact on exactly one
frame out of 150.

So the setting is chosen by measurement, not by taste. Each candidate is scored on

    held      -- the cube stays in the hand (drift from its settled pose stays small)
    contact   -- fraction of post-settle frames with any patch loaded
    rotation  -- how far the cube is actually turned, post-settle, in degrees
    slip      -- steady-state peak slip velocity, the channel the motion exists to excite

and the report is the whole table, so a rejected row can be read as easily as the
winner. The model is built once and :meth:`AllegroTactileScene.reset` runs between
candidates -- rebuilding costs 18 mesh SDFs.

**Treat this as a screen, not a certificate.** ``reset`` restores the states, the control
and the anchor history, but the MuJoCo solver object is reused, and its warm-start and
contact caches are not reset with it. A setting that survived 300 frames in its own
process (grasp 0.34 / cube-drop 0.05) was scored as dropped here when it ran fifth in a
row. Re-run the winner standalone before building anything on it.

    python -m sugar_newton.validation.allegro_grasp_sweep --frames 200
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import warp as wp

from sugar_newton.validation.allegro_tactile import AllegroTactileScene


def run_one(scene, frames: int, dt: float, kh: float, mu: float, grasp: float, spread,
            amplitude: float, rate: float, settle: float, mode: str,
            hold: tuple, rock: float, thumb, drop: float, close_time,
            cdrop: float) -> dict:
    scene.reset()
    scene.grasp, scene.amplitude, scene.rate, scene.settle = grasp, amplitude, rate, settle
    scene.spread = spread
    scene.swing_mode = mode
    scene.hold_digits = hold
    scene.rock = rock
    scene.thumb_oppose = thumb
    scene.drop = drop
    scene.close_time = close_time
    n = len(scene.patch_shapes)
    cube_body = int(scene.model.shape_body.numpy()[scene.cube_shapes[0]])

    palm = scene.palm_body
    cube_q = np.zeros((frames, 7), dtype=np.float64)
    palm_p = np.zeros((frames, 3), dtype=np.float64)
    live = np.zeros(frames, dtype=np.int32)
    load = np.zeros(frames, dtype=np.float64)
    slip = np.zeros(frames, dtype=np.float64)
    depth = np.zeros(frames, dtype=np.float64)
    for i in range(frames):
        scene.step(dt, substeps=16)
        ch = scene.tactile.to_numpy()
        live[i] = int((ch["contact_count"] > 0).sum())
        load[i] = float(ch["normal_load"].sum())
        slip[i] = float(ch["slip_velocity"].max())
        depth[i] = float(scene.tactile._peak_depth.numpy().max())
        bq = scene.state_0.body_q.numpy()
        cube_q[i] = bq[cube_body]
        palm_p[i] = bq[palm, :3] if palm >= 0 else 0.0

    s0 = min(int((settle + 0.5) / dt), frames - 1)   # post-settle window
    tail = slice(s0, None)
    ref = cube_q[s0]
    drift = np.linalg.norm(cube_q[tail, :3] - ref[:3], axis=1)
    dots = np.abs(cube_q[tail, 3:] @ ref[3:])
    ang = np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))
    # "held" has to be measured against the hand, not against the cube's own past: a
    # cube lying motionless on the floor has zero drift too, and the first version of
    # this sweep scored that as a perfect hold.
    reach = np.linalg.norm(cube_q[tail, :3] - palm_p[tail], axis=1)
    return dict(
        kh=kh, mu=mu, cube_drop=cdrop, hold="+".join(hold) or "none", rock=rock, thumb=thumb, grasp=grasp, spread=spread, amplitude=amplitude, rate=rate, mode=mode,
        held=bool(reach.max() < 0.15 and live[tail].mean() > 0.5),
        reach_mm=float(reach.max() * 1e3),
        drift_mm=float(drift.max() * 1e3),
        contact=float((live[tail] > 0).mean()),
        live=float(live[tail].mean()),
        rot_deg=float(ang.max()),
        load_N=float(load[tail].mean()),
        slip_mms=float(np.percentile(slip[tail], 99) * 1e3),
        pen_mm=float(np.percentile(depth[tail], 99) * 1e3),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--kh", type=float, nargs="+", default=[1.0e10])
    ap.add_argument("--cube-drop", type=float, nargs="+", default=[0.05],
                    help="cube start heights above its USD rest pose [m]")
    ap.add_argument("--mu", type=float, nargs="+", default=[-1.0],
                    help="friction override per build; -1 keeps the USD's")
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--drop", type=float, nargs="+", default=[0.0])
    ap.add_argument("--close-time", type=float, nargs="+", default=[-1.0],
                    help="-1 means close all the way from --drop to --settle")
    ap.add_argument("--grasp", type=float, nargs="+", default=[0.40])
    ap.add_argument("--spread", type=float, nargs="+", default=[-1.0],
                    help="abduction target [rad]; -1 means follow --grasp")
    ap.add_argument("--amplitude", type=float, nargs="+", default=[0.04, 0.08, 0.14])
    ap.add_argument("--rate", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--swing-mode", nargs="+", default=["close"], choices=("close", "sym"))
    ap.add_argument("--rock", type=float, nargs="+", default=[0.10])
    ap.add_argument("--thumb-oppose", type=float, nargs="+", default=[-1.0],
                    help="thumb opposition targets [rad]; -1 follows --spread")
    ap.add_argument("--hold-digits", nargs="+", default=["none"],
                    help="digit sets that clamp instead of swinging, e.g. thumb thumb+ring")
    args = ap.parse_args()

    wp.init()
    if not wp.get_device().is_cuda:
        print("ERROR: hydroelastic SDF is CUDA-only.")
        return 2

    dt = 1.0 / args.fps
    rows = []
    for kh, mu, cdrop in itertools.product(args.kh, args.mu, args.cube_drop):
        # kh, mu and the cube's start pose are baked in, so they cost a rebuild.
        scene = AllegroTactileScene(kh=kh, mu=None if mu < 0 else mu, cube_drop=cdrop)
        print(f"kh={kh:.0e} mu={mu:.2f} cube_drop={cdrop:.3f}  flexion dofs={int(scene.is_flex.sum())}/{scene.n_dof} across "
              f"{scene.n_digits} digits", flush=True)
        for l, f, d, lo, hi, q0 in zip(scene.dof_labels, scene.is_flex, scene.digit,
                                       scene.limit_lo, scene.limit_hi, scene.q_init):
            print(f"    {l:18s} {'flex' if f else 'sprd'} digit={d} "
                  f"limits=[{lo:+.3f},{hi:+.3f}] q0={q0:+.3f}", flush=True)
        for grasp, spr, amp, rate, mode, holds, rock, thb, drop, ct in itertools.product(
            args.grasp, args.spread, args.amplitude, args.rate, args.swing_mode,
            args.hold_digits, args.rock, args.thumb_oppose, args.drop, args.close_time,
        ):
            spread = None if spr < 0 else spr
            hold = tuple(x for x in holds.split("+") if x and x != "none")
            thumb = None if thb < 0 else thb
            close_time = None if ct < 0 else ct
            r = run_one(scene, args.frames, dt, kh, mu, grasp, spread, amp, rate,
                        args.settle, mode, hold, rock, thumb, drop, close_time, cdrop)
            rows.append(r)
            print(f"  kh={kh:.0e} cdrop={cdrop:.3f} grasp={grasp:.2f} spread={spr:.2f} amp={amp:.2f} "
                  f"rate={rate:.1f} {mode} rock={rock:.2f} thumb={thb:5.2f} "
                  f"drop={drop:.1f} close={ct:5.2f} -> "
                  f"held={str(r['held']):5s} reach={r['reach_mm']:6.1f}mm "
                  f"drift={r['drift_mm']:6.1f}mm contact={r['contact']:.2f} "
                  f"live={r['live']:4.1f} rot={r['rot_deg']:6.1f}deg "
                  f"load={r['load_N']:6.2f}N slip={r['slip_mms']:7.1f}mm/s "
                  f"pen={r['pen_mm']:.2f}mm", flush=True)

    held = [r for r in rows if r["held"]]
    print()
    if not held:
        print("no candidate held the cube -- widen --grasp or shrink --amplitude")
        return 1
    best = max(held, key=lambda r: r["rot_deg"] * r["contact"])
    print(f"best held candidate: --kh {best['kh']:.0e} --mu {best['mu']} --grasp {best['grasp']} "
          f"--spread {best['spread']} --amplitude {best['amplitude']} --rate {best['rate']} "
          f"--swing-mode {best['mode']} --hold-digits {best['hold']} --rock {best['rock']} "
          f"--cube-drop {best['cube_drop']} "
          f"--thumb-oppose {best['thumb']}")
    print(f"  cube turned {best['rot_deg']:.1f} deg, {best['live']:.1f} links loaded, "
          f"{best['load_N']:.2f} N, slip {best['slip_mms']:.1f} mm/s, "
          f"penetration {best['pen_mm']:.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
