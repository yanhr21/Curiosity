#!/usr/bin/env python3
"""Probe official SensorContact force/friction on a Panda MuJoCo-contact variant.

This is a diagnostic only. It does not replace the active Newton hydro base.
The goal is to test whether the official SensorContact path can provide direct
force/friction signals for a related Panda grasp variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact

from newton_tactile_curiosity.phase00_sync_hydro_diagnostic import SurfaceNullViewer


def vec_norm_series(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=np.float32), axis=-1)


def run(args: argparse.Namespace) -> dict:
    from newton.examples.robot.example_robot_panda_hydro import Example

    wp.set_device(args.device)
    viewer = SurfaceNullViewer(num_frames=args.num_frames)
    example = Example(viewer, SimpleNamespace(scene=args.scene, test=False, world_count=1))

    if args.override_mu is not None:
        example.model.shape_material_mu.fill_(float(args.override_mu))
    if args.override_kh is not None:
        example.model.shape_material_kh.fill_(float(args.override_kh))
    wp.synchronize()

    left_body = next((label for label in example.model.body_label if "leftfinger" in label.lower()), None)
    right_body = next((label for label in example.model.body_label if "rightfinger" in label.lower()), None)
    object_body = next((label for label in example.model.body_label if label.endswith("object")), None)
    if left_body is None or right_body is None or object_body is None:
        raise RuntimeError("missing left/right/object body labels for SensorContact probe")

    sensor = SensorContact(
        example.model,
        sensing_bodies=[left_body, right_body],
        counterpart_bodies=[object_body],
        measure_total=True,
        verbose=False,
    )
    solver = newton.solvers.SolverMuJoCo(
        example.model,
        use_mujoco_contacts=True,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        njmax=500,
        nconmax=500,
        iterations=15,
        ls_iterations=100,
        impratio=1000.0,
    )
    contacts = newton.Contacts(
        solver.get_max_contact_count(),
        0,
        requested_attributes=example.model.get_requested_contact_attributes(),
    )

    labels = list(example.model.body_label)
    object_idx = example.object_body_local
    object_pos = np.zeros((args.num_frames, 3), dtype=np.float32)
    total_force = np.zeros((args.num_frames, 2, 3), dtype=np.float32)
    total_friction = np.zeros((args.num_frames, 2, 3), dtype=np.float32)
    matrix_force = np.zeros((args.num_frames, 2, 1, 3), dtype=np.float32)
    matrix_friction = np.zeros((args.num_frames, 2, 1, 3), dtype=np.float32)
    contact_count = np.zeros(args.num_frames, dtype=np.int32)
    update_errors: list[str] = []

    initial_z = float(example.object_pos[2])

    for frame in range(args.num_frames):
        example.set_joint_targets()
        example.state_0.clear_forces()
        example.state_1.clear_forces()
        for _ in range(example.sim_substeps):
            solver.step(example.state_0, example.state_1, example.control, None, example.sim_dt)
            example.state_0, example.state_1 = example.state_1, example.state_0
        example.sim_time += example.frame_dt
        wp.synchronize()
        body_q = example.state_0.body_q.numpy().astype(np.float32)
        object_pos[frame] = body_q[object_idx, :3]
        try:
            solver.update_contacts(contacts, example.state_0)
            sensor.update(example.state_0, contacts)
            wp.synchronize()
        except Exception as exc:  # noqa: BLE001
            update_errors.append(f"frame {frame}: {type(exc).__name__}: {exc}")
            continue
        contact_count[frame] = int(contacts.rigid_contact_count.numpy()[0])
        total_force[frame] = sensor.total_force.numpy().astype(np.float32)
        total_friction[frame] = sensor.total_force_friction.numpy().astype(np.float32)
        if sensor.force_matrix is not None:
            matrix_force[frame] = sensor.force_matrix.numpy().astype(np.float32)
        if sensor.force_matrix_friction is not None:
            matrix_friction[frame] = sensor.force_matrix_friction.numpy().astype(np.float32)

    viewer.close()

    object_z = object_pos[:, 2]
    lift_threshold_m = 0.15
    lifted_mask = object_z >= initial_z + lift_threshold_m
    lift_success = bool(lifted_mask.any())
    first_lift_frame = int(np.argmax(lifted_mask)) if lift_success else None
    hold_frames = int((object_z[first_lift_frame:] >= initial_z + lift_threshold_m).sum()) if lift_success else 0
    force_norm = vec_norm_series(total_force)
    friction_norm = vec_norm_series(total_friction)
    matrix_force_norm = vec_norm_series(matrix_force)
    matrix_friction_norm = vec_norm_series(matrix_friction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "mujoco_sensor_probe_timeseries.npz"
    np.savez_compressed(
        npz_path,
        object_pos=object_pos,
        object_z=object_z,
        contact_count=contact_count,
        total_force=total_force,
        total_friction=total_friction,
        matrix_force=matrix_force,
        matrix_friction=matrix_friction,
    )
    shape_mu = example.model.shape_material_mu.numpy()
    shape_kh = example.model.shape_material_kh.numpy()
    summary = {
        "classification": "phase00_mujoco_contact_sensor_probe_v1",
        "run_tag": args.run_tag,
        "status": "pass_nonzero_friction" if float(friction_norm.max(initial=0.0)) > 0.0 else "blocked_zero_or_unavailable_friction",
        "not_training_result": True,
        "not_curiosity_success": True,
        "not_active_base_replacement": True,
        "variant": "official Panda scene/waypoints with SolverMuJoCo(use_mujoco_contacts=True), no Newton hydro collision pipeline",
        "num_frames": args.num_frames,
        "scene": args.scene,
        "material_label": args.material_label,
        "requested_override_mu": args.override_mu,
        "requested_override_kh": args.override_kh,
        "observed_shape_material_mu_unique": sorted({float(v) for v in shape_mu.tolist()}),
        "observed_shape_material_kh_unique": sorted({float(v) for v in shape_kh.tolist()}),
        "sensing_bodies": [left_body, right_body],
        "counterpart_bodies": [object_body],
        "max_contact_count": int(contact_count.max(initial=0)),
        "max_object_lift_m": float(object_z.max(initial=object_z[0]) - object_z[0]),
        "lift_success_threshold_m": lift_threshold_m,
        "lift_success": lift_success,
        "first_lift_frame": first_lift_frame,
        "hold_frames_above_lift_threshold": hold_frames,
        "max_total_force_norm": float(force_norm.max(initial=0.0)),
        "max_total_friction_norm": float(friction_norm.max(initial=0.0)),
        "max_matrix_force_norm": float(matrix_force_norm.max(initial=0.0)),
        "max_matrix_friction_norm": float(matrix_friction_norm.max(initial=0.0)),
        "nonzero_total_force_frames": int((force_norm.max(axis=1) > 0.0).sum()),
        "nonzero_total_friction_frames": int((friction_norm.max(axis=1) > 0.0).sum()),
        "update_error_count": len(update_errors),
        "update_errors_first": update_errors[:5],
        "npz_path": str(npz_path),
        "body_labels": labels,
    }
    summary_path = args.output_dir / "mujoco_sensor_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "mujoco_sensor_probe.md"
    report_path.write_text(
        "# Phase 00 MuJoCo SensorContact Probe\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{summary['status']}`\n"
        f"- variant: `{summary['variant']}`\n"
        f"- max total force norm: `{summary['max_total_force_norm']}`\n"
        f"- max total friction norm: `{summary['max_total_friction_norm']}`\n"
        f"- max object lift m: `{summary['max_object_lift_m']}`\n"
        f"- lift success: `{summary['lift_success']}`\n"
        f"- source arrays: `{npz_path}`\n"
        f"- summary: `{summary_path}`\n\n"
        "This is a direct SensorContact diagnostic on a MuJoCo-contact variant. "
        "It is not the active hydro base and not curiosity success.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=["cube", "pen"], default="cube")
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--material-label", default="official_default")
    parser.add_argument("--override-mu", type=float, default=None)
    parser.add_argument("--override-kh", type=float, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    summary = run(args)
    return 0 if summary["status"] == "pass_nonzero_friction" else 1


if __name__ == "__main__":
    raise SystemExit(main())
