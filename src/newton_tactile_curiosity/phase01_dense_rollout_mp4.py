#!/usr/bin/env python3
"""MP4-only rollout visualization for Phase01 dense closed-loop checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp
from PIL import Image, ImageDraw

from newton_tactile_curiosity.phase00_sync_hydro_diagnostic import SurfaceNullViewer, write_mp4_video
from newton_tactile_curiosity.phase01_dense_closed_loop_probe import (
    ACTION_NAMES,
    FEATURE_NAMES,
    PARAM_NAMES,
    ClosedLoopPolicy,
    DenseFeatureExtractor,
)


def load_best_params(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["best_params"], dtype=np.float32)


def shifted_vec3(vec: Any, dx: float, dy: float, dz: float) -> Any:
    return wp.vec3(float(vec[0]) + dx, float(vec[1]) + dy, float(vec[2]) + dz)


def rollout_trace(args: argparse.Namespace, params: np.ndarray, method: str) -> dict[str, Any]:
    from newton.examples.robot.example_robot_panda_hydro import Example
    import newton

    wp.set_device(args.device)
    viewer = SurfaceNullViewer(num_frames=args.num_frames)
    example = Example(viewer, SimpleNamespace(scene=args.scene, test=True, world_count=1))
    if args.override_mu is not None:
        example.model.shape_material_mu.fill_(float(args.override_mu))
    if args.override_kh is not None:
        example.model.shape_material_kh.fill_(float(args.override_kh))
    if args.override_mu is not None or args.override_kh is not None:
        example.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        wp.synchronize()

    base_set_joint_targets = example.set_joint_targets
    policy = ClosedLoopPolicy(params)
    latest_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)

    def closed_loop_set_joint_targets() -> None:
        cartesian_action = latest_action[1:4].copy()
        apply_cartesian = bool(np.abs(cartesian_action).sum() > 0.0)
        original_positions: list[Any] = []
        if apply_cartesian:
            dz = float(cartesian_action[0])
            dy = float(cartesian_action[1] + cartesian_action[2])
            original_positions = [waypoint[0] for waypoint in example.waypoints]
            for waypoint, original in zip(example.waypoints, original_positions, strict=False):
                waypoint[0] = shifted_vec3(original, 0.0, dy, dz)
        try:
            base_set_joint_targets()
        finally:
            if apply_cartesian:
                for waypoint, original in zip(example.waypoints, original_positions, strict=False):
                    waypoint[0] = original
        if float(np.abs(latest_action).sum()) <= 0.0:
            return
        targets = example.control.joint_target_q.numpy().reshape((example.world_count, -1)).astype(np.float32)
        grip_progress = np.clip(1.0 - targets[:, 7] / 0.06 + float(latest_action[0]), 0.0, 1.0)
        targets[:, 7] = 0.06 * (1.0 - grip_progress)
        targets[:, 8] = 0.06 * (1.0 - grip_progress)
        wp.copy(example.control.joint_target_q, wp.array(targets.reshape(-1), dtype=wp.float32))

    example.set_joint_targets = closed_loop_set_joint_targets
    extractor = DenseFeatureExtractor(example, args.map_size)
    features_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, float]] = []
    initial_z: float | None = None
    hold_frames = 0
    max_z = -1.0e9
    for frame in range(args.num_frames):
        example.step()
        wp.synchronize()
        features, meta = extractor.read(example)
        if initial_z is None:
            initial_z = float(features[0])
        max_z = max(max_z, float(features[0]))
        if float(features[0]) - initial_z > args.hold_lift_threshold:
            hold_frames += 1
        action = policy.update(frame, features)
        latest_action = action
        features_rows.append(features.copy())
        action_rows.append(action.copy())
        meta_rows.append({key: float(value) for key, value in meta.items() if isinstance(value, (int, float, np.number))})
    viewer.close()
    features_arr = np.stack(features_rows, axis=0)
    actions_arr = np.stack(action_rows, axis=0)
    initial = float(features_arr[0, 0])
    final_z = float(features_arr[-1, 0])
    max_lift = float(max_z - initial)
    drop_after_lift = float(max(0.0, max_z - final_z))
    return {
        "method": method,
        "features": features_arr,
        "actions": actions_arr,
        "meta": meta_rows,
        "initial_z": initial,
        "max_lift": max_lift,
        "hold_frames": int(hold_frames),
        "drop_after_lift": drop_after_lift,
    }


def plot_series(draw: ImageDraw.ImageDraw, values: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(180, 180, 180))
    vals = np.asarray(values, dtype=np.float32)
    if vals.size < 2:
        return
    lo = float(vals.min())
    hi = float(vals.max())
    if abs(hi - lo) < 1.0e-6:
        hi = lo + 1.0
    pts = []
    for i, value in enumerate(vals):
        x = x0 + int((x1 - x0) * i / max(1, vals.size - 1))
        y = y1 - int((y1 - y0) * (float(value) - lo) / (hi - lo))
        pts.append((x, y))
    draw.line(pts, fill=color, width=2)


def bar(draw: ImageDraw.ImageDraw, label: str, value: float, vmax: float, xy: tuple[int, int], width: int, color: tuple[int, int, int]) -> None:
    x, y = xy
    draw.text((x, y), f"{label}: {value:.3g}", fill=(25, 25, 25))
    frac = max(0.0, min(1.0, value / max(vmax, 1.0e-6)))
    draw.rectangle((x, y + 18, x + width, y + 32), outline=(170, 170, 170))
    draw.rectangle((x, y + 18, x + int(width * frac), y + 32), fill=color)


def render_frames(args: argparse.Namespace, base: dict[str, Any], checkpoint: dict[str, Any]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    traces = [base, checkpoint]
    z0 = min(float(trace["initial_z"]) for trace in traces)
    for idx in range(args.num_frames):
        canvas = Image.new("RGB", (1280, 720), (242, 242, 238))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 16), f"{args.run_tag} | frame {idx:03d} | MP4 rollout evidence", fill=(20, 20, 20))
        for col, trace in enumerate(traces):
            x = 30 + col * 620
            features = trace["features"]
            actions = trace["actions"]
            method = trace["method"]
            lift = float(features[idx, 0] - z0)
            total_fn = float(features[idx, 2] * 80.0)
            total_ft = float(features[idx, 3] * 40.0)
            left_fn = float(features[idx, 4] * 40.0)
            right_fn = float(features[idx, 5] * 40.0)
            slip = float(features[idx, 11])
            hold = lift > args.hold_lift_threshold
            color = (28, 104, 180) if method.startswith("base") else (24, 136, 78)
            draw.rectangle((x, 54, x + 585, 680), outline=(150, 150, 150))
            draw.text((x + 12, 66), method, fill=color)
            draw.text(
                (x + 12, 92),
                f"max_lift={trace['max_lift']:.3f}m hold={trace['hold_frames']} drop={trace['drop_after_lift']:.3f}m",
                fill=(35, 35, 35),
            )
            draw.text((x + 12, 120), f"current lift={lift:.3f}m  hold_now={int(hold)}", fill=(35, 35, 35))
            plot_series(draw, features[: idx + 1, 0] - z0, (x + 12, 152, x + 560, 278), color)
            draw.text((x + 12, 282), "object lift over time", fill=(60, 60, 60))
            bar(draw, "Fn total", total_fn, 80.0, (x + 12, 315), 250, (70, 120, 210))
            bar(draw, "Ft total", total_ft, 40.0, (x + 300, 315), 250, (190, 100, 70))
            bar(draw, "left Fn", left_fn, 40.0, (x + 12, 365), 250, (80, 150, 170))
            bar(draw, "right Fn", right_fn, 40.0, (x + 300, 365), 250, (80, 150, 170))
            bar(draw, "slip proxy", slip, 1.0, (x + 12, 415), 250, (170, 80, 120))
            action = actions[idx]
            for aidx, name in enumerate(ACTION_NAMES):
                bar(draw, name, float(abs(action[aidx])), 0.05 if aidx else 0.3, (x + 12, 470 + 42 * aidx), 350, color)
        frames.append(np.asarray(canvas, dtype=np.uint8))
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=["cube", "pen"], default="pen")
    parser.add_argument("--override-mu", type=float, default=0.05)
    parser.add_argument("--override-kh", type=float, default=1.0e12)
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-lift-threshold", type=float, default=0.08)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    checkpoint_params = load_best_params(args.checkpoint)
    base = rollout_trace(args, np.zeros(len(PARAM_NAMES), dtype=np.float32), "base_zero_action")
    checkpoint = rollout_trace(args, checkpoint_params, "checkpoint_policy")
    frames = render_frames(args, base, checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.visual_dir / "base_vs_checkpoint_rollout.mp4"
    write_mp4_video(video_path, frames, args.fps)
    summary = {
        "classification": "phase01_dense_rollout_mp4_v1",
        "run_tag": args.run_tag,
        "status": "complete_mp4_rollout_visualization",
        "video_path": str(video_path),
        "video_format": "mp4",
        "avi_generated": False,
        "num_frames": int(args.num_frames),
        "fps": int(args.fps),
        "scene": args.scene,
        "override_mu": float(args.override_mu),
        "base_max_lift": float(base["max_lift"]),
        "base_hold_frames": int(base["hold_frames"]),
        "checkpoint_max_lift": float(checkpoint["max_lift"]),
        "checkpoint_hold_frames": int(checkpoint["hold_frames"]),
        "checkpoint": str(args.checkpoint),
        "not_final_curiosity_success": True,
    }
    summary_path = args.output_dir / "dense_rollout_mp4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
