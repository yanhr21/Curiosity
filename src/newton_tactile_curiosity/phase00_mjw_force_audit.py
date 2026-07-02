#!/usr/bin/env python3
"""Audit MJWarp contact-force arrays for official Newton Panda hydro.

This is a compatibility diagnostic only. It intentionally does not call
``SolverMuJoCo.update_contacts`` because that official conversion path produced
CUDA illegal-memory-access evidence in the current Panda hydro setup. The goal
is to check whether the underlying MJWarp contact constraint arrays are present
and nonzero after official Newton simulation steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from newton_tactile_curiosity.phase00_sync_hydro_diagnostic import SurfaceNullViewer, classify_shape


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def first_scalar_int(array: Any, default: int = 0) -> int:
    arr = to_numpy(array).reshape(-1)
    if arr.size == 0:
        return default
    return int(arr[0])


def safe_shape(array: Any) -> list[int]:
    try:
        return [int(v) for v in to_numpy(array).shape]
    except Exception:  # noqa: BLE001
        return []


def force_table_for_world(efc_force: np.ndarray, world: int) -> np.ndarray:
    arr = np.asarray(efc_force)
    if arr.ndim == 1:
        return arr
    if arr.ndim >= 2:
        row = min(max(int(world), 0), arr.shape[0] - 1)
        return arr[row].reshape(-1)
    return arr.reshape(-1)


def frame_rows(frame: np.ndarray, contact_index: int) -> np.ndarray:
    row = np.asarray(frame[contact_index])
    if row.shape == (3, 3):
        return row.astype(np.float32)
    flat = row.reshape(-1)
    if flat.size >= 9:
        return flat[:9].reshape(3, 3).astype(np.float32)
    return np.zeros((3, 3), dtype=np.float32)


def classify_pair(shape_classes: list[str], shape0: int, shape1: int) -> tuple[str, bool]:
    c0 = shape_classes[shape0] if 0 <= shape0 < len(shape_classes) else "invalid"
    c1 = shape_classes[shape1] if 0 <= shape1 < len(shape_classes) else "invalid"
    pad = {"left_pad_or_finger", "right_pad_or_finger"}
    obj = {"object", "cup"}
    is_pad_obj = (c0 in pad and c1 in obj) or (c1 in pad and c0 in obj)
    return f"{c0}__{c1}", is_pad_obj


def summarize_contact(
    contact_index: int,
    world: int,
    geoms: np.ndarray,
    addresses: np.ndarray,
    efc_force: np.ndarray,
    geom_to_shape: np.ndarray,
    shape_classes: list[str],
    frame: np.ndarray,
) -> dict:
    world_idx = int(world)
    geom0 = int(geoms[0]) if geoms.size > 0 else -1
    geom1 = int(geoms[1]) if geoms.size > 1 else -1
    if geom_to_shape.ndim >= 2 and 0 <= world_idx < geom_to_shape.shape[0]:
        shape0 = int(geom_to_shape[world_idx, geom0]) if 0 <= geom0 < geom_to_shape.shape[1] else -1
        shape1 = int(geom_to_shape[world_idx, geom1]) if 0 <= geom1 < geom_to_shape.shape[1] else -1
    elif geom_to_shape.ndim == 1:
        shape0 = int(geom_to_shape[geom0]) if 0 <= geom0 < geom_to_shape.shape[0] else -1
        shape1 = int(geom_to_shape[geom1]) if 0 <= geom1 < geom_to_shape.shape[0] else -1
    else:
        shape0 = -1
        shape1 = -1

    force_row = force_table_for_world(efc_force, world_idx)
    valid_addresses = [int(a) for a in np.asarray(addresses).reshape(-1) if 0 <= int(a) < force_row.size]
    values = [float(force_row[a]) for a in valid_addresses]
    normal_abs = abs(values[0]) if values else 0.0
    tangent_abs = float(np.abs(values[1:]).sum()) if len(values) > 1 else 0.0
    pair_label, is_pad_object = classify_pair(shape_classes, shape0, shape1)
    frame_mat = frame_rows(frame, contact_index)
    return {
        "contact_index": int(contact_index),
        "world": world_idx,
        "geom0": geom0,
        "geom1": geom1,
        "shape0": shape0,
        "shape1": shape1,
        "pair_label": pair_label,
        "is_pad_object": bool(is_pad_object),
        "valid_efc_addresses": valid_addresses,
        "efc_force_values": values,
        "normal_efc_abs": float(normal_abs),
        "tangent_efc_abs_sum": float(tangent_abs),
        "contact_frame_rows": frame_mat.tolist(),
    }


def run(args: argparse.Namespace) -> dict:
    from newton.examples.robot.example_robot_panda_hydro import Example

    wp.set_device(args.device)
    viewer = SurfaceNullViewer(num_frames=args.num_frames)
    example = Example(viewer, SimpleNamespace(scene=args.scene, test=True, world_count=1))

    shape_body = example.model.shape_body.numpy()
    shape_classes = [classify_shape(i, shape_body, example.model) for i in range(example.model.shape_count)]

    nacon_series = np.zeros(args.num_frames, dtype=np.int32)
    valid_addr_series = np.zeros(args.num_frames, dtype=np.int32)
    efc_abs_sum_series = np.zeros(args.num_frames, dtype=np.float32)
    normal_abs_sum_series = np.zeros(args.num_frames, dtype=np.float32)
    tangent_abs_sum_series = np.zeros(args.num_frames, dtype=np.float32)
    pad_object_contact_series = np.zeros(args.num_frames, dtype=np.int32)
    pad_object_efc_abs_sum_series = np.zeros(args.num_frames, dtype=np.float32)
    pad_object_tangent_abs_sum_series = np.zeros(args.num_frames, dtype=np.float32)
    read_errors: list[str] = []
    max_sample: dict | None = None
    first_active_sample: dict | None = None
    array_shapes: dict[str, Any] = {}

    for frame_idx in range(args.num_frames):
        try:
            example.step()
            wp.synchronize()
            solver = example.solver
            mjw_data = solver.mjw_data
            contact = mjw_data.contact

            nacon = first_scalar_int(mjw_data.nacon)
            nacon = max(0, min(nacon, int(mjw_data.naconmax)))
            nacon_series[frame_idx] = nacon

            if frame_idx == 0:
                array_shapes = {
                    "mjw_data.nacon": safe_shape(mjw_data.nacon),
                    "mjw_data.naconmax": int(mjw_data.naconmax),
                    "mjw_data.njmax": int(mjw_data.njmax),
                    "mjw_data.contact.geom": safe_shape(contact.geom),
                    "mjw_data.contact.efc_address": safe_shape(contact.efc_address),
                    "mjw_data.contact.frame": safe_shape(contact.frame),
                    "mjw_data.contact.dim": safe_shape(contact.dim),
                    "mjw_data.contact.worldid": safe_shape(contact.worldid),
                    "mjw_data.efc.force": safe_shape(mjw_data.efc.force),
                    "solver.mjc_geom_to_newton_shape": safe_shape(solver.mjc_geom_to_newton_shape),
                    "mjw_model.opt.cone": int(solver.mjw_model.opt.cone),
                }

            if nacon <= 0:
                continue

            geom = to_numpy(contact.geom)[:nacon]
            efc_address = to_numpy(contact.efc_address)[:nacon]
            worldid = to_numpy(contact.worldid)[:nacon].reshape(-1)
            dim = to_numpy(contact.dim)[:nacon].reshape(-1)
            frame_rows_np = to_numpy(contact.frame)[:nacon]
            efc_force = to_numpy(mjw_data.efc.force)
            geom_to_shape = to_numpy(solver.mjc_geom_to_newton_shape)

            if first_active_sample is None:
                first_active_sample = {
                    "frame": int(frame_idx),
                    "nacon": int(nacon),
                    "dim_first_contacts": [int(v) for v in dim[: min(8, dim.size)]],
                }

            for cidx in range(nacon):
                sample = summarize_contact(
                    cidx,
                    int(worldid[cidx]) if cidx < worldid.size else 0,
                    np.asarray(geom[cidx]).reshape(-1),
                    np.asarray(efc_address[cidx]).reshape(-1),
                    efc_force,
                    geom_to_shape,
                    shape_classes,
                    frame_rows_np,
                )
                valid_addr_count = len(sample["valid_efc_addresses"])
                abs_sum = float(np.abs(sample["efc_force_values"]).sum()) if valid_addr_count else 0.0
                valid_addr_series[frame_idx] += valid_addr_count
                efc_abs_sum_series[frame_idx] += abs_sum
                normal_abs_sum_series[frame_idx] += sample["normal_efc_abs"]
                tangent_abs_sum_series[frame_idx] += sample["tangent_efc_abs_sum"]
                if sample["is_pad_object"]:
                    pad_object_contact_series[frame_idx] += 1
                    pad_object_efc_abs_sum_series[frame_idx] += abs_sum
                    pad_object_tangent_abs_sum_series[frame_idx] += sample["tangent_efc_abs_sum"]
                if abs_sum > 0.0 and (max_sample is None or abs_sum > float(max_sample["efc_abs_sum"])):
                    max_sample = {
                        "frame": int(frame_idx),
                        "efc_abs_sum": float(abs_sum),
                        **sample,
                    }
        except Exception as exc:  # noqa: BLE001
            read_errors.append(f"frame {frame_idx}: {type(exc).__name__}: {exc}")
            try:
                wp.synchronize()
            except Exception as sync_exc:  # noqa: BLE001
                read_errors.append(f"frame {frame_idx}: synchronize after error: {type(sync_exc).__name__}: {sync_exc}")
            continue

    viewer.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "mjw_force_audit_timeseries.npz"
    np.savez_compressed(
        npz_path,
        nacon=nacon_series,
        valid_efc_address_count=valid_addr_series,
        efc_abs_sum=efc_abs_sum_series,
        normal_efc_abs_sum=normal_abs_sum_series,
        tangent_efc_abs_sum=tangent_abs_sum_series,
        pad_object_contact_count=pad_object_contact_series,
        pad_object_efc_abs_sum=pad_object_efc_abs_sum_series,
        pad_object_tangent_efc_abs_sum=pad_object_tangent_abs_sum_series,
    )

    max_nacon = int(nacon_series.max(initial=0))
    max_efc = float(efc_abs_sum_series.max(initial=0.0))
    max_pad_efc = float(pad_object_efc_abs_sum_series.max(initial=0.0))
    max_tangent = float(tangent_abs_sum_series.max(initial=0.0))
    max_pad_tangent = float(pad_object_tangent_abs_sum_series.max(initial=0.0))
    has_force_arrays = max_nacon > 0 and int(valid_addr_series.max(initial=0)) > 0 and max_efc > 0.0
    has_pad_object_force = int(pad_object_contact_series.max(initial=0)) > 0 and max_pad_efc > 0.0
    has_tangent = max_tangent > 0.0
    status = "pass_mjw_force_arrays_nonzero" if has_force_arrays else "blocked_mjw_force_arrays_zero_or_unavailable"
    if has_force_arrays and not has_pad_object_force:
        status = "partial_mjw_force_nonzero_no_pad_object_force"
    if read_errors:
        status = "failed_read_errors" if not has_force_arrays else f"{status}_with_read_errors"

    summary = {
        "classification": "phase00_official_newton_hydro_mjwarp_force_array_audit_v1",
        "run_tag": args.run_tag,
        "status": status,
        "not_training_result": True,
        "not_curiosity_success": True,
        "official_example": "newton.examples.robot.example_robot_panda_hydro",
        "method": "step official Panda hydro; read SolverMuJoCo.mjw_data contact/efc arrays directly; do not call SolverMuJoCo.update_contacts",
        "why_no_update_contacts": "prior direct force probe produced CUDA illegal memory access in this setup",
        "num_frames": int(args.num_frames),
        "scene": args.scene,
        "device": args.device,
        "array_shapes": array_shapes,
        "max_nacon": max_nacon,
        "frames_with_contacts": int((nacon_series > 0).sum()),
        "max_valid_efc_address_count": int(valid_addr_series.max(initial=0)),
        "max_efc_abs_sum": max_efc,
        "max_normal_efc_abs_sum": float(normal_abs_sum_series.max(initial=0.0)),
        "max_tangent_efc_abs_sum": max_tangent,
        "frames_with_nonzero_efc": int((efc_abs_sum_series > 0.0).sum()),
        "max_pad_object_contact_count": int(pad_object_contact_series.max(initial=0)),
        "max_pad_object_efc_abs_sum": max_pad_efc,
        "max_pad_object_tangent_efc_abs_sum": max_pad_tangent,
        "frames_with_pad_object_force": int((pad_object_efc_abs_sum_series > 0.0).sum()),
        "has_force_arrays": bool(has_force_arrays),
        "has_pad_object_force": bool(has_pad_object_force),
        "has_tangent_efc_components": bool(has_tangent),
        "read_error_count": len(read_errors),
        "read_errors_first": read_errors[:5],
        "first_active_sample": first_active_sample,
        "max_sample": max_sample,
        "npz_path": str(npz_path),
        "direct_tactile_claim_allowed": False,
        "next_gate_if_pass": "candidate direct MJWarp force extractor must be validated against official SensorContact/update_contacts on a compatible MuJoCo-contact scene before direct tactile success claims",
    }

    summary_path = args.output_dir / "mjw_force_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "mjw_force_audit.md"
    report_path.write_text(
        "# Phase 00 MJWarp Force Array Audit\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{status}`\n"
        f"- max nacon: `{max_nacon}`\n"
        f"- max EFC abs sum: `{max_efc}`\n"
        f"- max tangent EFC abs sum: `{max_tangent}`\n"
        f"- max pad-object EFC abs sum: `{max_pad_efc}`\n"
        f"- max pad-object tangent EFC abs sum: `{max_pad_tangent}`\n"
        f"- read errors: `{len(read_errors)}`\n"
        f"- summary: `{summary_path}`\n"
        f"- timeseries: `{npz_path}`\n\n"
        "This is a compatibility diagnostic only. It does not prove tactile success and does not restart curiosity training.\n",
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
    parser.add_argument("--num-frames", type=int, default=90)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary["status"]).startswith(("pass_", "partial_")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
