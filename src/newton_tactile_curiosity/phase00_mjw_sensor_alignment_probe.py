#!/usr/bin/env python3
"""Compare candidate MJWarp EFC force mapping against official SensorContact.

This diagnostic runs a compatible MuJoCo-contact Panda grasp variant where the
official ``SensorContact`` path is known to work. It then compares official
left/right finger force and friction vectors against a direct candidate mapping
from ``mjw_data.contact.frame`` plus ``mjw_data.efc.force``.

The result validates or rejects the candidate mapping. It is not training and
does not replace the active Newton hydro base.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact

from newton_tactile_curiosity.phase00_mjw_direct_tactile_export import frame_matrix, geom_pair_to_shapes
from newton_tactile_curiosity.phase00_mjw_force_audit import force_table_for_world, to_numpy
from newton_tactile_curiosity.phase00_sync_hydro_diagnostic import SurfaceNullViewer, body_label


def vec_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=np.float32), axis=-1)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size < 2 or bb.size < 2:
        return None
    if float(np.std(aa)) <= 1.0e-12 or float(np.std(bb)) <= 1.0e-12:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def vector_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    cand = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    diff = cand - ref
    ref_norm = vec_norm(ref)
    cand_norm = vec_norm(cand)
    active = (ref_norm > 1.0e-6) | (cand_norm > 1.0e-6)
    if not np.any(active):
        return {
            "active_vector_count": 0,
            "rmse": 0.0,
            "relative_rmse": None,
            "mean_abs_error_norm": 0.0,
            "max_abs_error_norm": 0.0,
            "mean_cosine": None,
            "norm_correlation": None,
        }
    d_active = diff[active]
    ref_active = ref[active]
    cand_active = cand[active]
    ref_norm_active = vec_norm(ref_active)
    cand_norm_active = vec_norm(cand_active)
    err_norm = vec_norm(d_active)
    rmse = float(math.sqrt(float(np.mean(d_active * d_active))))
    scale = float(ref_norm_active.max(initial=0.0))
    cosine_mask = (ref_norm_active > 1.0e-6) & (cand_norm_active > 1.0e-6)
    if np.any(cosine_mask):
        cosine = np.einsum("ij,ij->i", ref_active[cosine_mask], cand_active[cosine_mask]) / (
            ref_norm_active[cosine_mask] * cand_norm_active[cosine_mask]
        )
        mean_cosine = float(np.mean(cosine))
    else:
        mean_cosine = None
    return {
        "active_vector_count": int(np.count_nonzero(active)),
        "rmse": rmse,
        "relative_rmse": float(rmse / max(scale, 1.0e-12)),
        "mean_abs_error_norm": float(np.mean(err_norm)),
        "max_abs_error_norm": float(err_norm.max(initial=0.0)),
        "mean_cosine": mean_cosine,
        "norm_correlation": safe_corr(cand_norm[active], ref_norm[active]),
    }


def choose_best(candidate_positive: np.ndarray, reference: np.ndarray) -> tuple[str, np.ndarray, dict, dict]:
    plus_metrics = vector_metrics(candidate_positive, reference)
    minus_metrics = vector_metrics(-candidate_positive, reference)
    plus_score = plus_metrics["relative_rmse"]
    minus_score = minus_metrics["relative_rmse"]
    if plus_score is None:
        plus_score = float("inf")
    if minus_score is None:
        minus_score = float("inf")
    if float(plus_score) <= float(minus_score):
        return "shape0_positive", candidate_positive, plus_metrics, minus_metrics
    return "shape0_negative", -candidate_positive, minus_metrics, plus_metrics


def candidate_from_mjw(
    solver,
    shape_body: np.ndarray,
    left_body_idx: int,
    right_body_idx: int,
    object_body_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate force/friction vectors and scalar counts.

    The vector convention initially assumes the EFC frame vector is positive on
    shape0 and negative on shape1. The caller also evaluates the opposite sign.
    """

    mjw_data = solver.mjw_data
    contact = mjw_data.contact
    nacon = int(to_numpy(mjw_data.nacon).reshape(-1)[0])
    nacon = max(0, min(nacon, int(mjw_data.naconmax)))
    force = np.zeros((2, 1, 3), dtype=np.float32)
    friction = np.zeros((2, 1, 3), dtype=np.float32)
    scalars = np.zeros(8, dtype=np.float32)
    if nacon <= 0:
        return force, friction, scalars

    geom = to_numpy(contact.geom)[:nacon]
    frames = to_numpy(contact.frame)[:nacon]
    efc_address = to_numpy(contact.efc_address)[:nacon]
    worldid = to_numpy(contact.worldid)[:nacon].reshape(-1)
    efc_force = to_numpy(mjw_data.efc.force)
    geom_to_shape = to_numpy(solver.mjc_geom_to_newton_shape)
    body_to_sensor_row = {int(left_body_idx): 0, int(right_body_idx): 1}

    for cidx in range(nacon):
        world = int(worldid[cidx]) if cidx < worldid.size else 0
        shape0, shape1 = geom_pair_to_shapes(geom[cidx], world, geom_to_shape)
        body0 = int(shape_body[shape0]) if 0 <= shape0 < len(shape_body) else -99
        body1 = int(shape_body[shape1]) if 0 <= shape1 < len(shape_body) else -99
        if body0 in body_to_sensor_row and body1 == object_body_idx:
            row = body_to_sensor_row[body0]
            sign = 1.0
        elif body1 in body_to_sensor_row and body0 == object_body_idx:
            row = body_to_sensor_row[body1]
            sign = -1.0
        else:
            continue

        force_row = force_table_for_world(efc_force, world)
        addresses = [int(a) for a in np.asarray(efc_address[cidx]).reshape(-1) if 0 <= int(a) < force_row.size]
        if not addresses:
            continue
        values = np.asarray([float(force_row[a]) for a in addresses], dtype=np.float32)
        mat = frame_matrix(frames[cidx])
        normal_vec = float(values[0]) * mat[0].astype(np.float32)
        tangent_vec = np.zeros(3, dtype=np.float32)
        for value, basis in zip(values[1:], mat[1 : 1 + max(0, len(values) - 1)], strict=False):
            tangent_vec += float(value) * basis.astype(np.float32)
        full_vec = normal_vec + tangent_vec
        force[row, 0] += sign * full_vec
        friction[row, 0] += sign * tangent_vec
        scalars[0] += 1.0
        scalars[1 + row] += 1.0
        scalars[3] += abs(float(values[0]))
        scalars[4] += float(np.linalg.norm(tangent_vec))
        scalars[5 + row] += abs(float(values[0]))
    return force, friction, scalars


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

    labels = list(example.model.body_label)
    left_idx = next((i for i, label in enumerate(labels) if "leftfinger" in label.lower()), None)
    right_idx = next((i for i, label in enumerate(labels) if "rightfinger" in label.lower()), None)
    object_idx = next((i for i, label in enumerate(labels) if label.endswith("object")), None)
    if left_idx is None or right_idx is None or object_idx is None:
        raise RuntimeError("missing left/right/object body labels for alignment probe")

    sensor = SensorContact(
        example.model,
        sensing_bodies=[labels[left_idx], labels[right_idx]],
        counterpart_bodies=[labels[object_idx]],
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

    shape_body = example.model.shape_body.numpy()
    object_pos = np.zeros((args.num_frames, 3), dtype=np.float32)
    official_force = np.zeros((args.num_frames, 2, 1, 3), dtype=np.float32)
    official_friction = np.zeros((args.num_frames, 2, 1, 3), dtype=np.float32)
    candidate_force_raw = np.zeros_like(official_force)
    candidate_friction_raw = np.zeros_like(official_friction)
    candidate_scalars = np.zeros((args.num_frames, 8), dtype=np.float32)
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
        candidate_force_raw[frame], candidate_friction_raw[frame], candidate_scalars[frame] = candidate_from_mjw(
            solver,
            shape_body,
            left_idx,
            right_idx,
            object_idx,
        )
        try:
            solver.update_contacts(contacts, example.state_0)
            sensor.update(example.state_0, contacts)
            wp.synchronize()
        except Exception as exc:  # noqa: BLE001
            update_errors.append(f"frame {frame}: {type(exc).__name__}: {exc}")
            continue
        contact_count[frame] = int(contacts.rigid_contact_count.numpy()[0])
        if sensor.force_matrix is not None:
            official_force[frame] = sensor.force_matrix.numpy().astype(np.float32)
        if sensor.force_matrix_friction is not None:
            official_friction[frame] = sensor.force_matrix_friction.numpy().astype(np.float32)

    viewer.close()

    force_sign, candidate_force_best, force_metrics_best, force_metrics_other = choose_best(
        candidate_force_raw.reshape(args.num_frames * 2, 3),
        official_force.reshape(args.num_frames * 2, 3),
    )
    friction_sign, candidate_friction_best, friction_metrics_best, friction_metrics_other = choose_best(
        candidate_friction_raw.reshape(args.num_frames * 2, 3),
        official_friction.reshape(args.num_frames * 2, 3),
    )
    candidate_force_best = candidate_force_best.reshape(official_force.shape)
    candidate_friction_best = candidate_friction_best.reshape(official_friction.shape)

    force_norm_official = vec_norm(official_force)
    friction_norm_official = vec_norm(official_friction)
    force_norm_candidate = vec_norm(candidate_force_best)
    friction_norm_candidate = vec_norm(candidate_friction_best)
    object_z = object_pos[:, 2]
    lift_threshold_m = 0.15
    lifted_mask = object_z >= initial_z + lift_threshold_m
    lift_success = bool(lifted_mask.any())
    first_lift_frame = int(np.argmax(lifted_mask)) if lift_success else None
    hold_frames = int((object_z[first_lift_frame:] >= initial_z + lift_threshold_m).sum()) if lift_success else 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "mjw_sensor_alignment_timeseries.npz"
    np.savez_compressed(
        npz_path,
        object_pos=object_pos,
        object_z=object_z,
        contact_count=contact_count,
        official_force=official_force,
        official_friction=official_friction,
        candidate_force_raw=candidate_force_raw,
        candidate_friction_raw=candidate_friction_raw,
        candidate_force_best=candidate_force_best,
        candidate_friction_best=candidate_friction_best,
        candidate_scalars=candidate_scalars,
    )

    force_rel = force_metrics_best["relative_rmse"]
    friction_rel = friction_metrics_best["relative_rmse"]
    force_cos = force_metrics_best["mean_cosine"]
    friction_cos = friction_metrics_best["mean_cosine"]
    aligned = (
        force_rel is not None
        and friction_rel is not None
        and force_rel <= args.max_relative_rmse
        and friction_rel <= args.max_relative_rmse
        and (force_cos is None or force_cos >= args.min_mean_cosine)
        and (friction_cos is None or friction_cos >= args.min_mean_cosine)
    )
    has_reference = float(friction_norm_official.max(initial=0.0)) > 0.0
    has_candidate = float(friction_norm_candidate.max(initial=0.0)) > 0.0
    if aligned:
        status = "pass_candidate_sensor_alignment"
    elif has_reference and has_candidate:
        status = "failed_candidate_sensor_alignment"
    else:
        status = "blocked_missing_reference_or_candidate_force"

    shape_mu = example.model.shape_material_mu.numpy()
    shape_kh = example.model.shape_material_kh.numpy()
    summary = {
        "classification": "phase00_mjwarp_candidate_vs_sensorcontact_alignment_v1",
        "run_tag": args.run_tag,
        "status": status,
        "not_training_result": True,
        "not_curiosity_success": True,
        "not_active_base_replacement": True,
        "variant": "official Panda scene/waypoints with SolverMuJoCo(use_mujoco_contacts=True), no Newton hydro collision pipeline",
        "validation_target": "candidate MJWarp EFC frame mapping vs official SensorContact.force_matrix and force_matrix_friction",
        "num_frames": int(args.num_frames),
        "scene": args.scene,
        "material_label": args.material_label,
        "requested_override_mu": args.override_mu,
        "requested_override_kh": args.override_kh,
        "observed_shape_material_mu_unique": sorted({float(v) for v in shape_mu.tolist()}),
        "observed_shape_material_kh_unique": sorted({float(v) for v in shape_kh.tolist()}),
        "sensing_bodies": [labels[left_idx], labels[right_idx]],
        "counterpart_bodies": [labels[object_idx]],
        "left_body_idx": int(left_idx),
        "right_body_idx": int(right_idx),
        "object_body_idx": int(object_idx),
        "max_contact_count_after_update": int(contact_count.max(initial=0)),
        "max_candidate_pad_object_contact_count": int(candidate_scalars[:, 0].max(initial=0)),
        "max_object_lift_m": float(object_z.max(initial=object_z[0]) - object_z[0]),
        "lift_success_threshold_m": lift_threshold_m,
        "lift_success": lift_success,
        "first_lift_frame": first_lift_frame,
        "hold_frames_above_lift_threshold": hold_frames,
        "max_official_force_norm": float(force_norm_official.max(initial=0.0)),
        "max_official_friction_norm": float(friction_norm_official.max(initial=0.0)),
        "max_candidate_force_norm": float(force_norm_candidate.max(initial=0.0)),
        "max_candidate_friction_norm": float(friction_norm_candidate.max(initial=0.0)),
        "force_best_sign": force_sign,
        "friction_best_sign": friction_sign,
        "force_metrics_best": force_metrics_best,
        "force_metrics_opposite_sign": force_metrics_other,
        "friction_metrics_best": friction_metrics_best,
        "friction_metrics_opposite_sign": friction_metrics_other,
        "max_relative_rmse_gate": args.max_relative_rmse,
        "min_mean_cosine_gate": args.min_mean_cosine,
        "update_error_count": len(update_errors),
        "update_errors_first": update_errors[:5],
        "npz_path": str(npz_path),
        "interpretation": (
            "candidate MJWarp EFC mapping is numerically aligned with official SensorContact on the compatible scene"
            if aligned
            else "candidate MJWarp EFC mapping is not yet validated against official SensorContact; use as candidate evidence only"
        ),
    }
    summary_path = args.output_dir / "mjw_sensor_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "mjw_sensor_alignment.md"
    report_path.write_text(
        "# Phase 00 MJWarp Candidate vs SensorContact Alignment\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{status}`\n"
        f"- force sign: `{force_sign}`\n"
        f"- friction sign: `{friction_sign}`\n"
        f"- force relative RMSE: `{force_metrics_best['relative_rmse']}`\n"
        f"- friction relative RMSE: `{friction_metrics_best['relative_rmse']}`\n"
        f"- force mean cosine: `{force_metrics_best['mean_cosine']}`\n"
        f"- friction mean cosine: `{friction_metrics_best['mean_cosine']}`\n"
        f"- max official/candidate friction norm: `{summary['max_official_friction_norm']}` / `{summary['max_candidate_friction_norm']}`\n"
        f"- source arrays: `{npz_path}`\n"
        f"- summary: `{summary_path}`\n\n"
        "This validates only the compatible MuJoCo-contact scene. It is not training, not curiosity success, and not final active-hydro tactile validation.\n",
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
    parser.add_argument("--max-relative-rmse", type=float, default=0.25)
    parser.add_argument("--min-mean-cosine", type=float, default=0.80)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    summary = run(args)
    return 0 if summary["status"] == "pass_candidate_sensor_alignment" else 1


if __name__ == "__main__":
    raise SystemExit(main())
