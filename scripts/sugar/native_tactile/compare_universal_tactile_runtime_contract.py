#!/usr/bin/env python3
"""Compare actual IsaacLab TacSL and Newton native-contact runtime traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _clock(sequence: np.ndarray, timestamp: np.ndarray, dt: np.ndarray) -> dict:
    if not (len(sequence) == len(timestamp) == len(dt)):
        raise ValueError("Clock fields have different lengths.")
    if len(sequence) < 2 or not np.array_equal(np.diff(sequence), np.ones(len(sequence) - 1, dtype=sequence.dtype)):
        raise ValueError("Tactile sequence is not continuous.")
    if np.any(np.diff(timestamp) < 0.0) or not np.allclose(dt[1:], np.diff(timestamp), atol=1.0e-9):
        raise ValueError("Tactile timestamps and elapsed time disagree.")
    return {
        "sequence_interval": [int(sequence[0]), int(sequence[-1])],
        "timestamp_interval_s": [float(timestamp[0]), float(timestamp[-1])],
        "dt_s_range_after_first": [float(dt[1:].min()), float(dt[1:].max())],
    }


def _isaaclab(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as trace:
        normal = np.asarray(trace["normal_force"])
        shear = np.asarray(trace["signed_shear"])
        penetration = np.asarray(trace["penetration"])
        positions = np.asarray(trace["taxel_position_w"])
        orientation_key = (
            "taxel_orientation_w_xyzw"
            if "taxel_orientation_w_xyzw" in trace.files
            else "taxel_quaternion_w"
        )
        orientations = np.asarray(trace[orientation_key])
        if normal.ndim == 3:
            normal = normal[:, None]
            shear = shear[:, None]
            penetration = penetration[:, None]
            positions = positions[:, None]
            orientations = orientations[:, None]
        elif normal.ndim == 5:
            frames, sides, patches, rows, columns = normal.shape
            normal = normal.reshape(frames, sides * patches, rows, columns)
            shear = shear.reshape(frames, sides * patches, rows, columns, 2)
            penetration = penetration.reshape(frames, sides * patches, rows, columns)
            positions = positions.reshape(frames, sides * patches, rows, columns, 3)
            orientations = orientations.reshape(frames, sides * patches, rows, columns, 4)
        expected = normal.shape
        if shear.shape != (*expected, 2) or penetration.shape != expected:
            raise ValueError("IsaacLab force and penetration fields do not share the universal layout.")
        if positions.shape != (*expected, 3) or orientations.shape != (*expected, 4):
            raise ValueError("IsaacLab taxel pose fields do not share the universal layout.")
        quaternion_error = float(np.max(np.abs(np.linalg.norm(orientations, axis=-1) - 1.0)))
        if quaternion_error > 2.0e-4:
            raise ValueError("IsaacLab xyzw taxel quaternions are not normalized.")
        optical_rgb = trace["optical_rgb"]
        optical_depth = trace["optical_depth"]
        if optical_rgb.shape[0] != expected[0] or optical_depth.shape[0] != expected[0]:
            raise ValueError("IsaacLab optical and force timelines have different lengths.")
        clock = _clock(
            np.asarray(trace["tactile_sequence"]),
            np.asarray(trace["tactile_timestamp_s"]),
            np.asarray(trace["tactile_dt_s"]),
        )
        return {
            "backend": "isaaclab_tacsl",
            "frames": expected[0],
            "patches": expected[1],
            "grid_shape": list(expected[2:4]),
            "normal_shape": list(normal.shape),
            "shear_shape": list(shear.shape),
            "penetration_shape": list(penetration.shape),
            "taxel_position_shape": list(positions.shape),
            "taxel_orientation_xyzw_shape": list(orientations.shape),
            "maximum_quaternion_norm_error": quaternion_error,
            "signed_shear_has_both_signs": bool(np.any(shear < 0.0) and np.any(shear > 0.0)),
            "optical_available": True,
            "optical_rgb_shape": list(optical_rgb.shape),
            "optical_depth_shape": list(optical_depth.shape),
            "clock": clock,
        }


def _newton(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as trace:
        force = np.asarray(trace["force_patch_n"])
        penetration = np.asarray(trace["penetration_m"])
        active = np.asarray(trace["active"])
        positions = np.asarray(trace["taxel_position_w_m"])
        orientations = np.asarray(trace["taxel_orientation_w_xyzw"])
        scalar_shape = force.shape[:-1]
        if force.ndim != 5 or force.shape[-1] != 3:
            raise ValueError("Newton force must be [frame, patch, row, column, XYZ].")
        if penetration.shape != scalar_shape or active.shape != scalar_shape:
            raise ValueError("Newton force and penetration fields do not share the universal layout.")
        if positions.shape != (*scalar_shape, 3) or orientations.shape != (*scalar_shape, 4):
            raise ValueError("Newton taxel pose fields do not share the universal layout.")
        quaternion_error = float(np.max(np.abs(np.linalg.norm(orientations, axis=-1) - 1.0)))
        if quaternion_error > 2.0e-4:
            raise ValueError("Newton xyzw taxel quaternions are not normalized.")
        optical_available = np.asarray(trace["optical_available"], dtype=bool)
        if np.any(optical_available):
            raise ValueError("Newton must not fabricate a native optical stream.")
        raw_count = np.asarray(trace["raw_count"])
        raw_patch = np.asarray(trace["raw_patch"])
        raw_force = np.asarray(trace["raw_force_patch_n"])
        maximum_force_residual = 0.0
        maximum_force_scale = 0.0
        for frame, count in enumerate(raw_count):
            raw_sum = np.zeros((scalar_shape[1], 3), dtype=np.float64)
            for patch in range(scalar_shape[1]):
                mask = raw_patch[frame, :count] == patch
                raw_sum[patch] = raw_force[frame, :count][mask].sum(axis=0)
            dense_sum = force[frame].sum(axis=(1, 2), dtype=np.float64)
            maximum_force_residual = max(maximum_force_residual, float(np.max(np.abs(raw_sum - dense_sum))))
            maximum_force_scale = max(
                maximum_force_scale,
                float(np.max(np.abs(raw_sum))),
                float(np.max(np.abs(dense_sum))),
            )
        force_tolerance = max(1.0e-5, 1.0e-6 * maximum_force_scale)
        if maximum_force_residual > force_tolerance:
            raise ValueError(f"Newton raw-to-grid force residual is {maximum_force_residual:.6g} N.")
        clock = _clock(
            np.asarray(trace["tactile_sequence"]),
            np.asarray(trace["tactile_timestamp_s"]),
            np.asarray(trace["tactile_dt_s"]),
        )
        return {
            "backend": str(np.asarray(trace["backend"]).item()),
            "frames": scalar_shape[0],
            "patches": scalar_shape[1],
            "grid_shape": list(scalar_shape[2:4]),
            "normal_shape": list(force[..., 2].shape),
            "shear_shape": list(force[..., :2].shape),
            "penetration_shape": list(penetration.shape),
            "taxel_position_shape": list(positions.shape),
            "taxel_orientation_xyzw_shape": list(orientations.shape),
            "maximum_quaternion_norm_error": quaternion_error,
            "signed_shear_has_both_signs": bool(np.any(force[..., :2] < 0.0) and np.any(force[..., :2] > 0.0)),
            "optical_available": False,
            "raw_sample_fields_present": sorted(key for key in trace.files if key.startswith("raw_")),
            "maximum_raw_samples_per_frame": int(raw_count.max()),
            "maximum_raw_to_grid_force_residual_n": maximum_force_residual,
            "force_conservation_tolerance_n": force_tolerance,
            "clock": clock,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-trace", type=Path, required=True)
    parser.add_argument("--newton-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    isaaclab = _isaaclab(args.isaaclab_trace.resolve())
    newton = _newton(args.newton_trace.resolve())
    if isaaclab["grid_shape"] != newton["grid_shape"]:
        raise ValueError("The actual IsaacLab and Newton traces use different common grid shapes.")
    report = {
        "schema": "isaaclab_newton_actual_universal_tactile_contract_v1",
        "shared_runtime_layout": {
            "scalar": "[frame, patch, row, column]",
            "signed_shear": "[frame, patch, row, column, local-XY]",
            "world_orientation": "xyzw",
            "grid_shape": isaaclab["grid_shape"],
            "required_channels": [
                "signed local-Z force",
                "signed local-XY shear",
                "penetration",
                "active mask",
                "taxel world position",
                "taxel world orientation",
                "source clock",
            ],
        },
        "isaaclab": isaaclab,
        "newton": newton,
        "backend_difference_preserved": {
            "isaaclab_official_gelsight_rgb_depth": True,
            "newton_native_gelsight_rgb_depth": False,
            "newton_raw_solved_contacts_retained": True,
        },
        "training": False,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
