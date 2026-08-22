# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Render the stick-to-slip transition, driven by the friction channel.

Pass 1 of the two-pass video pipeline (the composite runs on the login node,
where matplotlib and ffmpeg live -- see ``compose_friction_video.py``).

The scene is the incline held at a fixed sub-critical angle while the material
friction is swept **down** through the critical value.  The block therefore
starts stuck and ends sliding without anything else in the scene changing, so
what the video shows is the tactile channels responding to friction alone:

    utilization = tan(theta) / mu     climbs as mu falls
    slip                              stays ~0, then departs at utilization ~ 1
    gross slip fraction               0 -> 1 as the matcher stops carrying contacts

That is a direct picture of the thing Plan 15 could not measure: its
utilization divided by a fixed 0.5 and so was flat across exactly this sweep.

Requires CUDA -- hydroelastic SDF construction uses ``wp.Volume.allocate_by_tiles``
and ``wp.Texture3D``.  Run inside the CUDA container::

    python -m sugar_newton.validation.render_friction --out <dir> --frames 600
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import warp as wp

from newton import ModelFlags
from sugar_newton.validation.incline import InclineScene, GRAVITY, SLIDER_MASS


CHANNELS = (
    "normal_load",
    "friction_load",
    "utilization_mean",
    "utilization_max",
    "slip_displacement",
    "slip_velocity",
    "gross_slip_fraction",
    "contact_count",
    "contact_area",
    "peak_pressure",
)


def set_friction(scene: InclineScene, mu: float) -> None:
    """Set both shapes' material friction and tell the solver it changed.

    ``shape_material_mu`` is the value MuJoCo resolves the contact pair from
    (by elementwise max, ``kernels.py:165``); without the notify the solver
    keeps the friction it cached when the model was converted.
    """
    mus = scene.model.shape_material_mu.numpy()
    mus[scene.ramp_shape] = mu
    mus[scene.patch_shape] = mu
    scene.model.shape_material_mu.assign(mus)
    scene.solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory (on Lustre)")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--theta", type=float, default=20.0)
    ap.add_argument("--mu-start", type=float, default=0.75)
    ap.add_argument("--mu-end", type=float, default=0.15)
    ap.add_argument("--settle", type=int, default=150)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)
    ap.add_argument("--hydroelastic", action="store_true")
    ap.add_argument("--no-viewer", action="store_true", help="numbers only, no PNGs")
    ap.add_argument("--ramp-half-x", type=float, default=1.2,
                    help="short enough that the 10 cm block is not lost on an 8 m slab")
    # Auto-framed on the block by default: --cam-offset is applied to the slider's own
    # start position, so the 10 cm subject stays centred whatever the ramp geometry is.
    # The previous fixed pair sat ~1 m out aimed at a point the block was not on, which
    # is why the block read as a speck in the corner.
    ap.add_argument("--cam", type=float, nargs=3, default=None, help="absolute camera position; overrides --cam-offset")
    ap.add_argument("--cam-offset", type=float, nargs=3, default=(0.34, -0.40, 0.20),
                    help="camera position relative to the slider's start position [m]")
    ap.add_argument("--look-at", type=float, nargs=3, default=None,
                    help="aim point; default is the slider's own start position")
    ap.add_argument("--pitch", type=float, default=-18.0)
    args = ap.parse_args()

    wp.init()
    dev = wp.get_device()
    print(f"device={dev}  cuda={dev.is_cuda}", flush=True)
    if args.hydroelastic and not dev.is_cuda:
        print("ERROR: --hydroelastic needs CUDA (Volume.allocate_by_tiles, Texture3D)")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame_dir = out / "frames"
    frame_dir.mkdir(exist_ok=True)

    scene = InclineScene(
        args.theta,
        mu=args.mu_start,
        hydroelastic=args.hydroelastic,
        ramp_half_x=args.ramp_half_x,
        fallback_friction=99.0,  # absurd: a fall-through would be obvious in the trace
    )
    print(
        f"scene ready  theta={args.theta}  hydroelastic={args.hydroelastic}  "
        f"per_contact_scale={scene.contacts.rigid_contact_friction is not None}",
        flush=True,
    )

    viewer = None
    if not args.no_viewer:
        try:
            # Two headless GL paths, and they are mutually exclusive:
            #   G1_XVFB=1  -> windowed GL context on an Xvfb display (software mesa)
            #   otherwise  -> pyglet EGL headless
            # This container has no usable EGL device (libEGL: failed to open
            # /dev/dri/renderD135), so forcing headless=True fails with
            # NoSuchConfigException. render_env.sh exports G1_XVFB=1.
            xvfb = os.environ.get("G1_XVFB") == "1"
            import pyglet

            if not xvfb:
                pyglet.options["headless"] = True
            from newton.viewer import ViewerGL

            viewer = ViewerGL(headless=not xvfb)
            print(f"viewer path: {'xvfb/glx' if xvfb else 'egl headless'}", flush=True)
            viewer.set_model(scene.model)
            # Frame the block, not the slab. yaw is atan2 in the XY plane toward
            # the look-at point; pitch is given directly (same convention as
            # example_g1_in_sage.py:406).
            if hasattr(viewer, "set_camera"):
                target = scene.slider_position()
                if args.look_at is not None:
                    target = np.asarray(args.look_at, dtype=float)
                if args.cam is not None:
                    px, py, pz = args.cam
                else:
                    px, py, pz = (target + np.asarray(args.cam_offset, dtype=float)).tolist()
                tx, ty = float(target[0]), float(target[1])
                yaw = math.degrees(math.atan2(ty - py, tx - px))
                viewer.set_camera(wp.vec3(px, py, pz), args.pitch, yaw)
                print(f"camera pos=({px},{py},{pz}) pitch={args.pitch} yaw={yaw:.1f}", flush=True)
            print("viewer: ViewerGL headless ready", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"viewer unavailable ({type(exc).__name__}: {exc}); numbers only", flush=True)
            viewer = None

    set_friction(scene, args.mu_start)
    for _ in range(args.settle):
        scene.step(args.dt)

    trace = {k: [] for k in CHANNELS}
    trace["mu"] = []
    trace["expected_utilization"] = []
    trace["block_x"] = []
    tan_theta = math.tan(math.radians(args.theta))

    for i in range(args.frames):
        frac = i / max(1, args.frames - 1)
        mu = args.mu_start + (args.mu_end - args.mu_start) * frac
        set_friction(scene, mu)
        scene.step(args.dt)

        ch = scene.tactile.to_numpy()
        for k in CHANNELS:
            trace[k].append(float(ch[k][0]))
        trace["mu"].append(mu)
        trace["expected_utilization"].append(tan_theta / mu)
        trace["block_x"].append(float(scene.slider_position()[0]))

        if viewer is not None:
            try:
                viewer.begin_frame(i * args.dt)
                viewer.log_state(scene.state_0)
                viewer.log_contacts(scene.contacts, scene.state_0)
                viewer.end_frame()
                img = viewer.get_frame().numpy()
                from PIL import Image

                Image.fromarray(img[..., :3].astype(np.uint8)).save(
                    frame_dir / f"f{i:05d}.png"
                )
            except Exception as exc:  # noqa: BLE001
                if i == 0:
                    print(f"frame capture failed ({type(exc).__name__}: {exc})", flush=True)
                viewer = None

        if i % 100 == 0:
            print(
                f"  frame {i:4d}  mu={mu:.3f}  util={ch['utilization_mean'][0]:.3f} "
                f"slip_v={ch['slip_velocity'][0]:.3e}  gross={ch['gross_slip_fraction'][0]:.2f}",
                flush=True,
            )

    npz = out / "trace.npz"
    np.savez(
        npz,
        theta=args.theta,
        mass=SLIDER_MASS,
        gravity=GRAVITY,
        dt=args.dt,
        hydroelastic=args.hydroelastic,
        **{k: np.asarray(v, dtype=np.float64) for k, v in trace.items()},
    )
    n_png = len(list(frame_dir.glob("f*.png")))
    print(f"WROTE {npz}  frames={n_png}", flush=True)

    # A one-line verdict so the log alone says whether the sweep did anything.
    util = np.asarray(trace["utilization_mean"])
    gross = np.asarray(trace["gross_slip_fraction"])
    print(
        f"VERDICT utilization {util.min():.3f} -> {util.max():.3f} "
        f"(span {util.max() - util.min():.3f}); gross slip {gross.min():.2f} -> {gross.max():.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
