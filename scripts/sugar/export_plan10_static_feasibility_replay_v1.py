#!/usr/bin/env python3
"""Export exact official-body poses for one completed Plan-10 static scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path


HOST = socket.gethostname()
if HOST.startswith(("mgmtserver", "login")):
    raise SystemExit(f"Refusing Plan-10 replay export on login node: {HOST}")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Plan-10 replay export requires a retained allocation")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--unitree-repo", type=Path, required=True)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--geometry-root", type=Path, required=True)
parser.add_argument("--scan-root", type=Path, required=True)
parser.add_argument("--source-trace", type=Path, required=True)
parser.add_argument("--box-envelope", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--label", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

for path in (
    args.unitree_repo,
    args.asset_root,
    args.geometry_root,
    args.scan_root,
):
    if not path.exists():
        raise FileNotFoundError(path)
for path in (args.source_trace, args.box_envelope):
    if not path.is_file():
        raise FileNotFoundError(path)
if args.output_root.exists():
    raise FileExistsError(args.output_root)

os.environ["PROJECT_ROOT"] = str(args.asset_root.resolve())
sys.path.insert(0, str(args.unitree_repo.resolve()))
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from robots.unitree import G129_CFG_WITH_INSPIRE_WHOLEBODY  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def quat_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def quat_apply(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    q_vector = quaternion[1:]
    return (
        vector
        + 2.0 * quaternion[0] * np.cross(q_vector, vector)
        + 2.0 * np.cross(q_vector, np.cross(q_vector, vector))
    )


def reconstruct_box_pose(
    source_pose: np.ndarray,
    pca_center: np.ndarray,
    pca_basis: np.ndarray,
    pivot_pca: np.ndarray,
    tilt_deg: float,
    translation: np.ndarray,
) -> np.ndarray:
    axis_world = quat_apply(source_pose[3:7], pca_basis[:, 1])
    radians = np.deg2rad(tilt_deg)
    tilt_quaternion = np.concatenate(
        ((np.cos(0.5 * radians),), np.sin(0.5 * radians) * axis_world)
    )
    pivot_body = pca_center + pca_basis @ pivot_pca
    pivot_world = source_pose[:3] + quat_apply(source_pose[3:7], pivot_body)
    result = np.empty(7, dtype=np.float64)
    result[:3] = (
        pivot_world
        + quat_apply(tilt_quaternion, source_pose[:3] - pivot_world)
        + translation
    )
    result[3:7] = quat_mul(tilt_quaternion, source_pose[3:7])
    return result


def main() -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True)
    geometry_manifest_path = args.geometry_root / "manifest.json"
    geometry_path = args.geometry_root / "official_geometry.npz"
    scan_manifest_path = args.scan_root / "manifest.json"
    solutions_path = args.scan_root / "solutions.npz"
    geometry_manifest = json.loads(geometry_manifest_path.read_text())
    scan_manifest = json.loads(scan_manifest_path.read_text())
    if not geometry_manifest["export_passed"]:
        raise RuntimeError("The reused official geometry export did not pass")
    if sha256(geometry_path) != geometry_manifest["geometry_npz_sha256"]:
        raise RuntimeError("Official geometry hash mismatch")
    if not scan_manifest["audit_passed"]:
        raise RuntimeError("The completed static scan did not pass its structural audit")
    with np.load(solutions_path, allow_pickle=False) as solutions:
        joint_position = solutions["joint_position"].astype(np.float32)
        root_state = solutions["root_state"].astype(np.float32)
        tilt_deg = solutions["box_pca1_tilt_deg"].astype(np.float64)
        box_translation = np.column_stack(
            (
                solutions["box_x_delta_m"],
                solutions["box_y_delta_m"],
                solutions["box_z_delta_m"],
            )
        ).astype(np.float64)
        ranked_indices = solutions["ranked_indices"].astype(np.int32)
    records = scan_manifest["records"]
    if len(records) != joint_position.shape[0]:
        raise RuntimeError("Scan manifest/solution row mismatch")
    with np.load(args.source_trace, allow_pickle=False) as source:
        source_box_pose = source["initial_object_state"][:7].astype(np.float64)
    with np.load(args.box_envelope, allow_pickle=False) as envelope:
        pca_center = envelope["pca_center_b"].astype(np.float64)
        pca_basis = envelope["pca_basis_b"].astype(np.float64)
    pivot_pca = np.asarray(
        scan_manifest["parameters"]["box_pivot_pca_m"], dtype=np.float64
    )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args.device)
    )
    robot_cfg = G129_CFG_WITH_INSPIRE_WHOLEBODY.replace(prim_path="/World/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(cfg=robot_cfg)
    sim.reset()
    robot.reset()
    sim.forward()
    robot.update(0.0)
    if list(robot.body_names) != geometry_manifest["body_names"][:-1]:
        raise RuntimeError("Official runtime body order differs from geometry export")

    body_pose = np.empty(
        (len(records), len(robot.body_names) + 1, 7), dtype=np.float32
    )
    box_pose = np.empty((len(records), 7), dtype=np.float32)
    reconstructed_hand_position_error = []
    left_id = list(robot.body_names).index("left_hand_base_link")
    right_id = list(robot.body_names).index("right_hand_base_link")
    for index, record in enumerate(records):
        robot.write_root_state_to_sim(
            torch.as_tensor(root_state[index], device=robot.device).unsqueeze(0)
        )
        robot.write_joint_state_to_sim(
            torch.as_tensor(joint_position[index], device=robot.device).unsqueeze(0),
            torch.zeros((1, joint_position.shape[1]), device=robot.device),
        )
        sim.forward()
        robot.update(0.0)
        body_pose[index, :-1, :3] = (
            robot.data.body_link_pos_w[0].detach().cpu().numpy()
        )
        body_pose[index, :-1, 3:7] = (
            robot.data.body_link_quat_w[0].detach().cpu().numpy()
        )
        box_pose[index] = reconstruct_box_pose(
            source_box_pose,
            pca_center,
            pca_basis,
            pivot_pca,
            tilt_deg[index],
            box_translation[index],
        )
        body_pose[index, -1] = box_pose[index]
        expected_hand = np.asarray(record["actual_hand_position_w_m"])
        actual_hand = body_pose[index, (left_id, right_id), :3]
        reconstructed_hand_position_error.append(
            np.linalg.norm(actual_hand - expected_hand, axis=1)
        )
    reconstructed_hand_position_error = np.asarray(
        reconstructed_hand_position_error, dtype=np.float64
    )
    recorded_box_position = np.asarray(
        [record["box_position_w_m"] for record in records], dtype=np.float64
    )
    maximum_box_position_error = float(
        np.linalg.norm(box_pose[:, :3] - recorded_box_position, axis=1).max()
    )
    maximum_hand_position_error = float(reconstructed_hand_position_error.max())

    replay_path = output_root / "static_feasibility_replay.npz"
    np.savez_compressed(
        replay_path,
        body_pose_wxyz=body_pose,
        body_names=np.asarray(geometry_manifest["body_names"]),
        ranked_indices=ranked_indices,
        position_error_m=np.asarray(
            [record["position_error_m"] for record in records], dtype=np.float32
        ),
        rotation_error_rad=np.asarray(
            [record["rotation_error_rad"] for record in records], dtype=np.float32
        ),
        head_gap_m=np.asarray(
            [record["head_to_box_pca_obb_gap_m"] for record in records],
            dtype=np.float32,
        ),
        head_inside_vertices=np.asarray(
            [record["head_collision_vertices_inside_box_pca_obb"] for record in records],
            dtype=np.int32,
        ),
        nonhand_load_n=np.asarray(
            [record["nonhand_contact_normal_load_n"] for record in records],
            dtype=np.float32,
        ),
        tilt_deg=tilt_deg.astype(np.float32),
        box_translation_m=box_translation.astype(np.float32),
        root_delta_m=np.asarray(
            [
                (record["root_x_delta_m"], record["root_y_delta_m"], record["root_z_delta_m"])
                for record in records
            ],
            dtype=np.float32,
        ),
        active_contact_bodies=np.asarray(
            [",".join(record["active_contact_bodies"].keys()) for record in records]
        ),
        static_candidate=np.asarray(
            [record["static_candidate"] for record in records], dtype=bool
        ),
    )
    checks = {
        "official_geometry_export_passed": bool(geometry_manifest["export_passed"]),
        "official_geometry_hash_exact": sha256(geometry_path)
        == geometry_manifest["geometry_npz_sha256"],
        "static_scan_audit_passed": bool(scan_manifest["audit_passed"]),
        "all_scan_rows_exported": body_pose.shape[0] == len(records),
        "runtime_body_order_exact": list(robot.body_names)
        == geometry_manifest["body_names"][:-1],
        "reconstructed_hand_positions_within_2e_6m": maximum_hand_position_error
        <= 2.0e-6,
        "reconstructed_box_positions_within_2e_6m": maximum_box_position_error
        <= 2.0e-6,
        "no_static_candidate_preserved": not bool(np.any([record["static_candidate"] for record in records])),
    }
    payload = {
        "schema": "plan10_static_feasibility_official_pose_replay_v1",
        "claim_boundary": "Exact no-learning pose reconstruction for human review of a completed static feasibility scan. It is not dynamics, grasp, lift, tactile, policy, or success evidence.",
        "label": args.label,
        "host": HOST,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "geometry_manifest": str(geometry_manifest_path.resolve()),
        "geometry_manifest_sha256": sha256(geometry_manifest_path),
        "geometry_npz": str(geometry_path.resolve()),
        "geometry_npz_sha256": sha256(geometry_path),
        "scan_manifest": str(scan_manifest_path.resolve()),
        "scan_manifest_sha256": sha256(scan_manifest_path),
        "solutions": str(solutions_path.resolve()),
        "solutions_sha256": sha256(solutions_path),
        "source_trace": str(args.source_trace.resolve()),
        "source_trace_sha256": sha256(args.source_trace),
        "box_envelope": str(args.box_envelope.resolve()),
        "box_envelope_sha256": sha256(args.box_envelope),
        "replay": str(replay_path),
        "replay_sha256": sha256(replay_path),
        "frame_count": len(records),
        "maximum_reconstructed_hand_position_error_m": maximum_hand_position_error,
        "maximum_reconstructed_box_position_error_m": maximum_box_position_error,
        "checks": checks,
    }
    payload["export_passed"] = all(checks.values())
    atomic_json(output_root / "manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    simulation_app.close()
    if not payload["export_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
