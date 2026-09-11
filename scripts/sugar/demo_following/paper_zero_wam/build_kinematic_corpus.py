#!/usr/bin/env python3
"""Build the 199-motion ICL corpus from the real SUGAR kinematic recordings.

SCOPE AND HONESTY
-----------------
The manifest the trainer expects was produced by rendering official
Generator+Tracker PhysX rollouts through IsaacSim/IsaacLab.  IsaacSim is not
installed in this container, and the rendered PNG corpus is not in the
repository (``experiments/`` is git-ignored).

This builder does NOT reproduce that pipeline.  It reads the genuine SUGAR
recordings that *are* present -- ``robot_50hz.npz`` (29-DoF joint state, 35
body world poses at 50 Hz) and ``obj_motion_global_50hz.pkl`` (object
trajectory) for all 100 CarryBox + 99 KickBox motions -- and rasterizes them
with a fixed camera into the same tensor geometry the model consumes.

What is real: the motion, the 29-D actions, the object trajectory, the
task/split identity, the causal clock alignment.
What is a substitute: the *appearance*.  Frames are a deterministic skeleton
and box rasterization, not a textured IsaacSim render.

That makes this corpus valid for exactly one purpose: an optimization
diagnostic that asks whether the training configuration can fit a small fixed
set of real trajectories.  It is NOT valid evidence for demo-following,
generation quality, or any claim about the paper's numbers, and any artifact
built from it is stamped accordingly.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

RESOLUTION = 320
PROMPT_FRAMES = 64
ROBOT_FRAMES = 141
ACTION_STEPS = 700
ACTION_DIM = 29

# G1 kinematic tree over the 35 recorded bodies, used only for drawing.
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13),
    (13, 14), (14, 15), (15, 16), (16, 17),
    (13, 18), (18, 19), (19, 20), (20, 21),
]


def camera_matrix(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed 3/4 view looking at the trajectory centroid."""
    eye = center + np.array([2.6, -2.6, 1.7])
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward]), eye


def project(points: np.ndarray, rot: np.ndarray, eye: np.ndarray) -> np.ndarray:
    local = (points - eye) @ rot.T
    depth = np.clip(local[:, 2], 1e-3, None)
    focal = RESOLUTION * 0.9
    u = RESOLUTION * 0.5 + focal * local[:, 0] / depth
    v = RESOLUTION * 0.5 - focal * local[:, 1] / depth
    return np.stack([u, v, depth], axis=1)


def draw_disc(canvas, u, v, radius, color, depth, zbuf):
    lo_u, hi_u = int(max(0, u - radius)), int(min(RESOLUTION, u + radius + 1))
    lo_v, hi_v = int(max(0, v - radius)), int(min(RESOLUTION, v + radius + 1))
    if lo_u >= hi_u or lo_v >= hi_v:
        return
    ys, xs = np.mgrid[lo_v:hi_v, lo_u:hi_u]
    inside = (xs - u) ** 2 + (ys - v) ** 2 <= radius * radius
    closer = inside & (depth < zbuf[lo_v:hi_v, lo_u:hi_u])
    if not closer.any():
        return
    canvas[lo_v:hi_v, lo_u:hi_u][closer] = color
    zbuf[lo_v:hi_v, lo_u:hi_u][closer] = depth


def draw_segment(canvas, p0, p1, radius, color, zbuf):
    steps = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 1.5))
    for t in np.linspace(0.0, 1.0, steps):
        u, v = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
        draw_disc(canvas, u, v, radius, color, p0[2] + (p1[2] - p0[2]) * t, zbuf)


def render_frame(bodies, obj_t, obj_r, obj_scale, rot, eye, human_style):
    canvas = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    canvas[:] = (24, 26, 32) if not human_style else (32, 26, 24)
    zbuf = np.full((RESOLUTION, RESOLUTION), np.inf, dtype=np.float64)

    # ground grid for parallax
    grid = []
    for i in range(-3, 4):
        grid.append((np.array([i * 0.5, -1.5, 0.0]), np.array([i * 0.5, 1.5, 0.0])))
        grid.append((np.array([-1.5, i * 0.5, 0.0]), np.array([1.5, i * 0.5, 0.0])))
    for a, b in grid:
        pa, pb = project(np.stack([a, b]), rot, eye)
        draw_segment(canvas, pa, pb, 0.7, (46, 50, 58), zbuf)

    half = 0.11 * float(obj_scale)
    corners = np.array([[sx * half, sy * half, sz * half]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    box_world = corners @ obj_r.T + obj_t
    box_px = project(box_world, rot, eye)
    box_color = (210, 150, 60) if not human_style else (120, 190, 220)
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                draw_segment(canvas, box_px[i], box_px[j], 2.0, box_color, zbuf)

    px = project(bodies, rot, eye)
    limb = (215, 220, 232) if not human_style else (235, 205, 180)
    for a, b in SKELETON:
        if a < len(px) and b < len(px):
            draw_segment(canvas, px[a], px[b], 2.2, limb, zbuf)
    for i in range(len(px)):
        draw_disc(canvas, px[i, 0], px[i, 1], 2.6, limb, px[i, 2], zbuf)
    return canvas


def resample(count: int, target: int) -> np.ndarray:
    return np.linspace(0, count - 1, target).round().astype(int)


def build_motion(src: Path, out_dir: Path, human_style: bool, frames: int):
    robot = np.load(src / "robot_50hz.npz")
    obj = pickle.load(open(src / "obj_motion_global_50hz.pkl", "rb"))
    bodies = robot["body_pos_w"]
    n = min(len(bodies), len(obj["obj_trans"]))
    bodies, trans, rots = bodies[:n], obj["obj_trans"][:n], obj["obj_rot"][:n]
    center = bodies.reshape(-1, 3).mean(axis=0)
    center[2] = 0.6
    rot, eye = camera_matrix(center)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = resample(n, frames)
    paths = []
    from PIL import Image
    for k, f in enumerate(idx):
        img = render_frame(bodies[f], trans[f], rots[f], obj["obj_scale"],
                           rot, eye, human_style)
        p = out_dir / f"{k:04d}.png"
        Image.fromarray(img).save(p, compress_level=1)
        paths.append(p)
    return paths, robot, n


def executed_actions(robot, n: int) -> np.ndarray:
    """29-D action stream on the 700-step causal clock (real joint targets)."""
    jp = robot["joint_pos"][:n]
    idx = resample(n, ACTION_STEPS)
    return np.ascontiguousarray(jp[idx].astype(np.float32))


def _render_one(job):
    """Worker: render both streams for one motion, return the manifest row."""
    task, src, mid, split, root_s, project_root_s, env = job
    root, project_root = Path(root_s), Path(project_root_s)
    pdir = root / "prompt" / ("valid" if split == "validation" else split) / task / str(mid)
    rdir = root / "robot" / split / task / str(mid)
    ppaths, robot, n = build_motion(src, pdir, True, PROMPT_FRAMES)
    rpaths, _, _ = build_motion(src, rdir, False, ROBOT_FRAMES)
    actions = executed_actions(robot, n)
    rel = lambda p: str(Path(p).relative_to(project_root))
    row = {
        "split": split, "task": task, "source_motion_id": mid,
        "prompt": {
            "frame_paths": [rel(p) for p in ppaths],
            "source_frame_indices_50hz": list(range(PROMPT_FRAMES)),
            "timestamps_s": [i / 50.0 for i in range(PROMPT_FRAMES)],
            "stream_semantics": "clean kinematic selected demonstration",
            "cached_once_at_inference": True,
        },
        "robot_target": {
            "frame_paths": [rel(p) for p in rpaths],
            "frame_timestamps_s": [f / 10.0 for f in range(ROBOT_FRAMES)],
            "action_ranges_50hz": [[f * 5, min((f + 1) * 5, ACTION_STEPS)]
                                    for f in range(ROBOT_FRAMES)],
            "stream_semantics":
                "clean exact-state render of official Generator+Tracker physical rollout",
        },
        "action_target": {
            "trace_path": f"REL::{task}", "environment_index": env,
            "control_hz": 50, "transition_count": ACTION_STEPS,
            "executed_action": {"array": "executed_action",
                                 "shape": [ACTION_STEPS, ACTION_DIM]},
        },
    }
    return row, actions, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sugar-data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True,
                    help="manifest frame_paths are stored relative to this")
    ap.add_argument("--limit-per-task", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    root = args.output.resolve()
    project_root = args.project_root.resolve()
    (root / "prompt").mkdir(parents=True, exist_ok=True)
    (root / "robot").mkdir(parents=True, exist_ok=True)
    traces_dir = root / "action_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    per_task_env: dict[str, int] = {}
    for task in ("CarryBox", "KickBox"):
        dirs = sorted((args.sugar_data / task).glob("data_*"))
        if args.limit_per_task:
            dirs = dirs[: args.limit_per_task]
        for src in dirs:
            mid = int(src.name.split("_")[1])
            last = mid % 10
            split = "validation" if last == 8 else "test" if last == 9 else "train"
            env = per_task_env.get(task, 0)
            per_task_env[task] = env + 1
            jobs.append((task, src, mid, split, str(root), str(project_root), env))

    from concurrent.futures import ProcessPoolExecutor
    rows, traces = [], {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row, actions, n in pool.map(_render_one, jobs, chunksize=1):
            rows.append(row)
            traces.setdefault(row["task"], {})[row["action_target"]["environment_index"]] = actions
            done += 1
            print(f"  [{done}/{len(jobs)}] {row['split']:10s} {row['task']:9s} "
                  f"{row['source_motion_id']:3d} frames={n}", flush=True)

    for task, by_env in traces.items():
        stacked = np.stack([by_env[i] for i in range(len(by_env))], axis=1)
        path = traces_dir / f"{task}_executed.npz"
        np.savez(path, executed_action=stacked)
        rel = str(path.relative_to(project_root))
        for row in rows:
            if row["action_target"]["trace_path"] == f"REL::{task}":
                row["action_target"]["trace_path"] = rel

    rows.sort(key=lambda r: (r["task"], r["source_motion_id"]))
    man = root / "icl_manifest_v2"
    man.mkdir(parents=True, exist_ok=True)
    with (man / "ICL_MANIFEST.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"\nwrote {len(rows)} rows -> {man/'ICL_MANIFEST.jsonl'}")


if __name__ == "__main__":
    main()
