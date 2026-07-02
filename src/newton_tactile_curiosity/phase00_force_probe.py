#!/usr/bin/env python3
"""Probe direct solver contact forces for official Newton Panda hydro.

This is a diagnostic only. It does not train a policy and does not replace the
hydroelastic proxy fields. The only question is whether the official solver
path can populate ``Contacts.force`` with usable nonzero force vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton_tactile_curiosity.phase00_sync_hydro_diagnostic import SurfaceNullViewer, classify_shape


def body_label(model, body_idx: int) -> str:
    if body_idx < 0:
        return "world"
    if body_idx < len(model.body_label):
        return model.body_label[body_idx]
    return f"body_{body_idx}"


def force_linear(spatial: np.ndarray) -> np.ndarray:
    """Return linear part from Warp spatial_vector numpy representation."""
    arr = np.asarray(spatial)
    if arr.ndim == 1 and arr.shape[0] >= 6:
        return arr[:3].astype(np.float32)
    return arr[..., :3].astype(np.float32)


def run(args: argparse.Namespace) -> dict:
    from newton.examples.robot.example_robot_panda_hydro import Example

    wp.set_device(args.device)
    viewer = SurfaceNullViewer(num_frames=args.num_frames)
    example = Example(viewer, SimpleNamespace(scene=args.scene, test=True, world_count=1))

    # The official example creates contacts before any extended attributes are
    # requested. Recreate the compatible contacts buffer with the official
    # model/pipeline after requesting force, and disable the captured graph so
    # the current contacts object is used by simulate().
    example.model.request_contact_attributes("force")
    example.contacts = example.collision_pipeline.contacts()
    example.graph = None

    shape_body = example.model.shape_body.numpy()
    shape_classes = [classify_shape(i, shape_body, example.model) for i in range(example.model.shape_count)]

    force_norm_sum = np.zeros(args.num_frames, dtype=np.float32)
    normal_component_sum = np.zeros(args.num_frames, dtype=np.float32)
    tangential_component_sum = np.zeros(args.num_frames, dtype=np.float32)
    contact_count = np.zeros(args.num_frames, dtype=np.int32)
    update_errors: list[str] = []
    max_sample: dict | None = None

    for frame in range(args.num_frames):
        example.step()
        wp.synchronize()
        try:
            example.solver.update_contacts(example.contacts, example.state_0)
            wp.synchronize()
        except Exception as exc:  # noqa: BLE001
            update_errors.append(f"frame {frame}: {type(exc).__name__}: {exc}")
            continue

        n = int(example.contacts.rigid_contact_count.numpy()[0])
        contact_count[frame] = n
        if n <= 0 or example.contacts.force is None:
            continue

        shapes0 = example.contacts.rigid_contact_shape0.numpy()[:n].astype(np.int32)
        shapes1 = example.contacts.rigid_contact_shape1.numpy()[:n].astype(np.int32)
        normals = example.contacts.rigid_contact_normal.numpy()[:n].astype(np.float32)
        forces = force_linear(example.contacts.force.numpy()[:n])
        fn_signed = np.einsum("ij,ij->i", forces, normals)
        fn_abs = np.abs(fn_signed)
        tangential = forces - fn_signed[:, None] * normals
        ft = np.linalg.norm(tangential, axis=1)

        force_norm_sum[frame] = float(np.linalg.norm(forces, axis=1).sum(initial=0.0))
        normal_component_sum[frame] = float(fn_abs.sum(initial=0.0))
        tangential_component_sum[frame] = float(ft.sum(initial=0.0))

        if force_norm_sum[frame] > 0.0 and max_sample is None:
            idx = int(np.argmax(np.linalg.norm(forces, axis=1)))
            max_sample = {
                "frame": frame,
                "shape0": int(shapes0[idx]),
                "shape1": int(shapes1[idx]),
                "shape0_class": shape_classes[int(shapes0[idx])] if int(shapes0[idx]) >= 0 else "invalid",
                "shape1_class": shape_classes[int(shapes1[idx])] if int(shapes1[idx]) >= 0 else "invalid",
                "shape0_body": body_label(example.model, int(shape_body[int(shapes0[idx])])),
                "shape1_body": body_label(example.model, int(shape_body[int(shapes1[idx])])),
                "normal": normals[idx].tolist(),
                "force_linear": forces[idx].tolist(),
                "normal_component_abs": float(fn_abs[idx]),
                "tangential_component": float(ft[idx]),
            }

    viewer.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "force_probe_timeseries.npz"
    np.savez_compressed(
        npz_path,
        force_norm_sum=force_norm_sum,
        normal_component_sum=normal_component_sum,
        tangential_component_sum=tangential_component_sum,
        contact_count=contact_count,
    )
    summary = {
        "classification": "phase00_official_newton_hydro_direct_force_probe_v1",
        "run_tag": args.run_tag,
        "status": "pass_nonzero_force" if float(force_norm_sum.max(initial=0.0)) > 0.0 else "blocked_zero_or_unavailable_force",
        "not_training_result": True,
        "not_curiosity_success": True,
        "official_example": "newton.examples.robot.example_robot_panda_hydro",
        "probe_method": "request Contacts.force, recreate official collision-pipeline contacts, step official example, call SolverMuJoCo.update_contacts after each frame",
        "num_frames": args.num_frames,
        "max_contact_count_after_update": int(contact_count.max(initial=0)),
        "max_force_norm_sum": float(force_norm_sum.max(initial=0.0)),
        "max_normal_component_sum_abs": float(normal_component_sum.max(initial=0.0)),
        "max_tangential_component_sum": float(tangential_component_sum.max(initial=0.0)),
        "nonzero_force_frames": int((force_norm_sum > 0.0).sum()),
        "update_error_count": len(update_errors),
        "update_errors_first": update_errors[:5],
        "max_sample": max_sample,
        "npz_path": str(npz_path),
    }
    summary_path = args.output_dir / "force_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "force_probe.md"
    report_path.write_text(
        "# Phase 00 Direct Force Probe\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{summary['status']}`\n"
        f"- max force norm sum: `{summary['max_force_norm_sum']}`\n"
        f"- max normal component sum abs: `{summary['max_normal_component_sum_abs']}`\n"
        f"- max tangential component sum: `{summary['max_tangential_component_sum']}`\n"
        f"- nonzero force frames: `{summary['nonzero_force_frames']}`\n"
        f"- update errors: `{summary['update_error_count']}`\n"
        f"- summary: `{summary_path}`\n\n"
        "This is a direct-force availability diagnostic only, not training and not curiosity success.\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=["cube", "pen"], default="cube")
    parser.add_argument("--num-frames", type=int, default=240)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass_nonzero_force" else 1


if __name__ == "__main__":
    raise SystemExit(main())
