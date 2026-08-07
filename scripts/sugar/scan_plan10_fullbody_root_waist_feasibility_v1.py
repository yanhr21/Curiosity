#!/usr/bin/env python3
"""No-learning root/waist feasibility scan for an exact recorded grasp target.

Each candidate restores the exact recorded official G1+Inspire state, changes
only the declared root pose coordinates and live waist-pitch joint, and uses the
official IsaacLab DifferentialIKController to return both seven-DoF arms to
the same v98 bilateral palm target.  The CarryBox is kept far away during IK,
then restored as a fixed collision query for exactly one PhysX step.  This is
a posture/geometry diagnostic, not a grasp, lift, tactile, or policy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import tempfile
from itertools import product
from pathlib import Path
from typing import Any


HOST = socket.gethostname()
if HOST.startswith(("mgmtserver", "login")):
    raise SystemExit(f"Refusing full-body feasibility scan on login node: {HOST}")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Full-body feasibility scan requires the retained allocation")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--unitree-repo", type=Path, required=True)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--box-usd", type=Path, required=True)
parser.add_argument("--source-trace", type=Path, required=True)
parser.add_argument(
    "--target-source",
    choices=("legacy_contact_anchor", "geometric_side_clamp"),
    default="legacy_contact_anchor",
    help=(
        "Select the exact bilateral pose stored by the source producer. The "
        "geometric side-clamp target is frozen before any IK reachability "
        "concession or dynamic contact and is required for mixed side/bottom scans."
    ),
)
parser.add_argument(
    "--source-index",
    type=int,
    default=None,
    help=(
        "Optional recorded source row. When provided, bind the scan to that "
        "row's robot, box and desired bilateral hand poses instead of the "
        "legacy initial-state/contact-anchor contract."
    ),
)
parser.add_argument("--collision-points", type=Path, required=True)
parser.add_argument("--box-envelope", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--maximum-ik-iterations", type=int, default=600)
parser.add_argument(
    "--box-z-deltas-m",
    type=float,
    nargs="+",
    default=(0.0,),
    help=(
        "Declared vertical translations applied to both the official CarryBox "
        "pose and its hash-bound bilateral hand targets."
    ),
)
parser.add_argument(
    "--box-x-deltas-m",
    type=float,
    nargs="+",
    default=(0.0,),
    help="Declared world-X translations of the box and bilateral targets.",
)
parser.add_argument(
    "--box-y-deltas-m",
    type=float,
    nargs="+",
    default=(0.0,),
    help="Declared world-Y translations of the box and bilateral targets.",
)
parser.add_argument(
    "--box-pca1-tilt-deg",
    type=float,
    nargs="+",
    default=(0.0,),
    help=(
        "Rigid diagnostic tilt of the box and both frozen contact targets "
        "about the declared cooked lower-edge pivot and box PCA1 axis."
    ),
)
parser.add_argument(
    "--box-pivot-pca-m",
    type=float,
    nargs=3,
    default=None,
    metavar=("PCA0", "PCA1", "PCA2"),
)
parser.add_argument(
    "--root-x-deltas-m",
    type=float,
    nargs="+",
    default=(0.0,),
)
parser.add_argument(
    "--root-y-deltas-m",
    type=float,
    nargs="+",
    default=(0.00, 0.01, 0.02, 0.03, 0.04, 0.05),
)
parser.add_argument(
    "--root-z-deltas-m",
    type=float,
    nargs="+",
    default=(0.0,),
)
parser.add_argument(
    "--root-yaw-deltas-deg",
    type=float,
    nargs="+",
    default=(0.0,),
)
parser.add_argument(
    "--left-normal-roll-deltas-deg",
    type=float,
    nargs="+",
    default=(0.0,),
    help=(
        "Local palm-X roll applied to the frozen left contact orientation. "
        "It changes wrist/camera clearance without moving the contact point."
    ),
)
parser.add_argument(
    "--right-normal-roll-deltas-deg",
    type=float,
    nargs="+",
    default=(0.0,),
    help=(
        "Local palm-X roll applied to the frozen right contact orientation. "
        "It changes wrist/camera clearance without moving the contact point."
    ),
)
parser.add_argument(
    "--waist-pitch-rad",
    type=float,
    nargs="+",
    default=(0.35, 0.40, 0.43, 0.45, 0.47, 0.49, 0.50, 0.51, 0.52),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

unitree_repo = args.unitree_repo.expanduser().resolve()
asset_root = args.asset_root.expanduser().resolve()
box_usd = args.box_usd.expanduser().resolve()
source_trace = args.source_trace.expanduser().resolve()
collision_points_path = args.collision_points.expanduser().resolve()
box_envelope_path = args.box_envelope.expanduser().resolve()
output_root = args.output_root.expanduser().resolve()
official_usd = (
    asset_root
    / "assets/robots/g1-29dof_wholebody_inspire/"
    "g1_29dof_with_inspire_rev_1_0.usd"
)
for path in (
    official_usd,
    box_usd,
    source_trace,
    collision_points_path,
    box_envelope_path,
):
    if not path.is_file():
        raise FileNotFoundError(path)
if output_root.exists():
    raise FileExistsError(f"Refusing overwrite: {output_root}")
if args.maximum_ik_iterations < 100:
    raise ValueError("The bounded serious DLS scan requires at least 100 iterations")
if args.target_source == "geometric_side_clamp" and args.source_index is not None:
    raise ValueError("The frozen geometric target cannot be combined with source-index")
if any(abs(value) > 1.0e-12 for value in args.box_pca1_tilt_deg):
    if args.box_pivot_pca_m is None:
        raise ValueError("Nonzero box PCA1 tilt requires a declared PCA pivot")
    if args.target_source != "geometric_side_clamp":
        raise ValueError("Box tilt requires the frozen geometric contact target")
if args.box_pivot_pca_m is not None and any(
    abs(value) > 0.30 for value in args.box_pivot_pca_m
):
    raise ValueError("Declared box PCA pivot is outside +/-0.30 m")

os.environ["PROJECT_ROOT"] = str(asset_root)
sys.path.insert(0, str(unitree_repo))
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import (  # noqa: E402
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from robots.unitree import G129_CFG_WITH_INSPIRE_WHOLEBODY  # noqa: E402


EXPECTED_USD_SHA256 = "86047174b87b4df485e996232fb4d2ece5901a9bcd6f9a54b78f961ce664730e"
SIDES = ("left", "right")
ROOT_Y_DELTAS_M = (0.00, 0.01, 0.02, 0.03, 0.04, 0.05)
WAIST_PITCH_ABSOLUTE_RAD = (0.35, 0.40, 0.43, 0.45, 0.47, 0.49, 0.50, 0.51, 0.52)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def is_anatomical_hand(name: str) -> bool:
    return name.endswith("hand_base_link") or name.startswith(("L_", "R_"))


def quat_apply_numpy(quaternion_wxyz: np.ndarray, points: np.ndarray) -> np.ndarray:
    vector = quaternion_wxyz[1:]
    return points + 2.0 * (
        quaternion_wxyz[0] * np.cross(vector, points)
        + np.cross(vector, np.cross(vector, points))
    )


def quat_apply_inverse_numpy(
    quaternion_wxyz: np.ndarray, points: np.ndarray
) -> np.ndarray:
    inverse = quaternion_wxyz.copy()
    inverse[1:] *= -1.0
    return quat_apply_numpy(inverse, points)


def head_to_box_obb_gap(
    head_vertices_body: np.ndarray,
    head_position_w: np.ndarray,
    head_quaternion_w: np.ndarray,
    box_pose_w: np.ndarray,
    pca_center_b: np.ndarray,
    pca_basis_b: np.ndarray,
    pca_bounds: np.ndarray,
) -> tuple[float, int]:
    vertices_w = (
        quat_apply_numpy(head_quaternion_w, head_vertices_body)
        + head_position_w
    )
    vertices_b = quat_apply_inverse_numpy(
        box_pose_w[3:7], vertices_w - box_pose_w[:3]
    )
    vertices_pca = (vertices_b - pca_center_b) @ pca_basis_b
    outside = np.maximum(
        np.maximum(pca_bounds[0] - vertices_pca, vertices_pca - pca_bounds[1]),
        0.0,
    )
    distance = np.linalg.norm(outside, axis=1)
    inside = np.all(
        (vertices_pca >= pca_bounds[0])
        & (vertices_pca <= pca_bounds[1]),
        axis=1,
    )
    return float(distance.min()), int(np.count_nonzero(inside))


def main() -> None:
    output_root.mkdir(parents=True)
    with np.load(source_trace, allow_pickle=False) as source:
        source_robot_body_names = source["robot_body_names"].astype(str).tolist()
        source_robot_joint_names = source["robot_joint_names"].astype(str).tolist()
        source_contact_body_names = source[
            "all_robot_contact_body_names"
        ].astype(str).tolist()
        if args.target_source == "geometric_side_clamp":
            source_root_state = source["initial_robot_root_state"].astype(
                np.float32
            )
            source_joint_position = source["robot_joint_position"][0].astype(
                np.float32
            )
            # The recorded first dynamic row follows the producer's offstage
            # IK attempt and can already be trapped at an unreachable
            # concession. Restore the exact official motion source for all 29
            # G1 body joints; retain only the separate Inspire finger values
            # from the first row because they do not enter the arm DLS solve.
            source_joint_position[:29] = source[
                "initial_robot_body_joint_position"
            ].astype(np.float32)
            source_box_state = source["initial_object_state"].astype(np.float32)
            desired_hand_position_w = source[
                "side_clamp_geometric_contact_pos_w"
            ].astype(np.float32)
            desired_hand_quaternion_w = source[
                "side_clamp_geometric_contact_quat_w"
            ].astype(np.float32)
            if desired_hand_position_w.shape != (2, 3):
                raise ValueError("Source trace lacks a bilateral geometric target")
            if desired_hand_quaternion_w.shape != (2, 4):
                raise ValueError("Source trace lacks geometric target orientations")
        elif args.source_index is None:
            source_root_state = source["robot_root_state"][0].astype(np.float32)
            source_joint_position = source["robot_joint_position"][0].astype(
                np.float32
            )
            source_box_state = source["initial_object_state"].astype(np.float32)
            desired_hand_position_w = source["ik_desired_contact_pos_w"].astype(
                np.float32
            )
            desired_hand_quaternion_w = source["ik_desired_contact_quat_w"].astype(
                np.float32
            )
        else:
            index = args.source_index
            if not (0 <= index < source["box_state"].shape[0]):
                raise ValueError("source-index is outside the recorded trace")
            source_root_state = source["robot_root_state"][index].astype(np.float32)
            source_joint_position = source["robot_joint_position"][index].astype(
                np.float32
            )
            source_box_state = source["box_state"][index].astype(np.float32)
            desired_hand_position_w = source["desired_hand_pos_w"][index].astype(
                np.float32
            )
            desired_hand_quaternion_w = source["desired_hand_quat_w"][index].astype(
                np.float32
            )
    with np.load(collision_points_path, allow_pickle=False) as geometry:
        head_vertices_body = geometry[
            "body_015_collision_00_geometry_00"
        ].astype(np.float64)
    with np.load(box_envelope_path, allow_pickle=False) as envelope:
        pca_center_b = envelope["pca_center_b"].astype(np.float64)
        pca_basis_b = envelope["pca_basis_b"].astype(np.float64)
        pca_bounds = envelope["pca_bounds"].astype(np.float64)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.005,
            device=args.device,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                min_position_iteration_count=4,
                max_position_iteration_count=16,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=4,
            ),
        )
    )
    robot_cfg = G129_CFG_WITH_INSPIRE_WHOLEBODY.replace(
        prim_path="/World/Robot"
    )
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
    robot = Articulation(cfg=robot_cfg)
    box = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/CarryBox",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(box_usd),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                    retain_accelerations=True,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(5.0, 5.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )
    )
    sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/Robot/.*",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            filter_prim_paths_expr=["/World/CarryBox"],
            track_friction_forces=True,
            max_contact_data_count_per_prim=256,
        )
    )
    sim.reset()
    _ = sensor.data
    robot.update(0.005)
    if list(robot.body_names) != source_robot_body_names:
        raise RuntimeError("Live/source robot body order differs")
    if list(robot.joint_names) != source_robot_joint_names:
        raise RuntimeError("Live/source robot joint order differs")
    if list(sensor.body_names) != source_contact_body_names:
        raise RuntimeError("Live/source contact body order differs")

    device = robot.device
    waist_pitch_id = robot.find_joints("waist_pitch_joint")[0]
    if len(waist_pitch_id) != 1:
        raise RuntimeError("Expected one waist_pitch_joint")
    waist_pitch_id = int(waist_pitch_id[0])
    lower = robot.data.joint_pos_limits[0, :, 0].clone()
    upper = robot.data.joint_pos_limits[0, :, 1].clone()
    waist_lower = float(lower[waist_pitch_id].item())
    waist_upper = float(upper[waist_pitch_id].item())
    # Preserve the declared scan while binding the endpoint to the exact live
    # float32 hard limit.  The official source records 0.52 rad, while the
    # runtime upper tensor can be one ULP below the same decimal value.
    feasible_waist_values = sorted(
        {
            float(np.clip(value, waist_lower, waist_upper))
            for value in args.waist_pitch_rad
        }
    )
    if not feasible_waist_values:
        raise RuntimeError("No declared waist-pitch value satisfies live limits")

    ee_body_ids: dict[str, int] = {}
    ee_jacobian_ids: dict[str, int] = {}
    arm_joint_ids: dict[str, list[int]] = {}
    controllers: dict[str, DifferentialIKController] = {}
    for side in SIDES:
        body_ids = robot.find_bodies(f"{side}_hand_base_link")[0]
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one {side} hand body")
        ee_body_ids[side] = int(body_ids[0])
        ee_jacobian_ids[side] = int(body_ids[0]) - 1
        arm_names = [
            f"{side}_shoulder_pitch_joint",
            f"{side}_shoulder_roll_joint",
            f"{side}_shoulder_yaw_joint",
            f"{side}_elbow_joint",
            f"{side}_wrist_roll_joint",
            f"{side}_wrist_pitch_joint",
            f"{side}_wrist_yaw_joint",
        ]
        ids = [int(value) for value in robot.find_joints(arm_names)[0]]
        if len(ids) != 7:
            raise RuntimeError(f"Expected seven {side} arm joints")
        arm_joint_ids[side] = ids
        controllers[side] = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=1,
            device=device,
        )

    desired_position_t = torch.as_tensor(
        desired_hand_position_w, device=device
    )
    desired_quaternion_t = torch.as_tensor(
        desired_hand_quaternion_w, device=device
    )
    source_joint_t = torch.as_tensor(source_joint_position, device=device)
    source_root_t = torch.as_tensor(source_root_state, device=device)
    box_pose_t = torch.as_tensor(source_box_state[:7], device=box.device).unsqueeze(0)
    pca_center_t = torch.as_tensor(
        pca_center_b, device=box.device, dtype=box_pose_t.dtype
    )
    pca_basis_t = torch.as_tensor(
        pca_basis_b, device=box.device, dtype=box_pose_t.dtype
    )
    box_pivot_pca_t = (
        torch.as_tensor(
            args.box_pivot_pca_m, device=box.device, dtype=box_pose_t.dtype
        )
        if args.box_pivot_pca_m is not None
        else torch.zeros(3, device=box.device, dtype=box_pose_t.dtype)
    )
    box_pivot_body_t = pca_center_t + pca_basis_t @ box_pivot_pca_t
    box_pivot_world_t = box_pose_t[0, :3] + math_utils.quat_apply(
        box_pose_t[0, 3:7], box_pivot_body_t
    )
    box_pca1_axis_world_t = math_utils.quat_apply(
        box_pose_t[0, 3:7], pca_basis_t[:, 1]
    )
    far_box_pose_t = torch.tensor(
        ((5.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0),),
        device=box.device,
        dtype=box_pose_t.dtype,
    )
    zero_box_velocity = torch.zeros((1, 6), device=box.device)
    contact_names = list(sensor.body_names)
    nonhand_indices = [
        index for index, name in enumerate(contact_names) if not is_anatomical_hand(name)
    ]
    head_contact_index = contact_names.index("head_link")
    head_robot_index = list(robot.body_names).index("head_link")

    records: list[dict[str, Any]] = []
    joint_solutions = []
    root_solutions = []
    for (
        box_x_delta_m,
        box_y_delta_m,
        box_z_delta_m,
        box_pca1_tilt_deg,
        root_x_delta_m,
        root_y_delta_m,
        root_z_delta_m,
        root_yaw_delta_deg,
        left_normal_roll_delta_deg,
        right_normal_roll_delta_deg,
        waist_pitch_rad,
    ) in product(
        args.box_x_deltas_m,
        args.box_y_deltas_m,
        args.box_z_deltas_m,
        args.box_pca1_tilt_deg,
        args.root_x_deltas_m,
        args.root_y_deltas_m,
        args.root_z_deltas_m,
        args.root_yaw_deltas_deg,
        args.left_normal_roll_deltas_deg,
        args.right_normal_roll_deltas_deg,
        feasible_waist_values,
    ):
            box_tilt_rad = np.deg2rad(box_pca1_tilt_deg)
            box_tilt_quaternion_t = torch.cat(
                (
                    torch.tensor(
                        [np.cos(0.5 * box_tilt_rad)],
                        device=box.device,
                        dtype=box_pose_t.dtype,
                    ),
                    np.sin(0.5 * box_tilt_rad) * box_pca1_axis_world_t,
                )
            )
            bilateral_tilt_quaternion_t = box_tilt_quaternion_t.unsqueeze(0).expand(
                2, -1
            )
            scan_desired_position_t = box_pivot_world_t + math_utils.quat_apply(
                bilateral_tilt_quaternion_t,
                desired_position_t - box_pivot_world_t,
            )
            scan_desired_quaternion_t = math_utils.quat_mul(
                bilateral_tilt_quaternion_t, desired_quaternion_t
            )
            scan_box_pose_t = box_pose_t.clone()
            scan_box_pose_t[0, :3] = box_pivot_world_t + math_utils.quat_apply(
                box_tilt_quaternion_t,
                box_pose_t[0, :3] - box_pivot_world_t,
            )
            scan_box_pose_t[0, 3:7] = math_utils.quat_mul(
                box_tilt_quaternion_t, box_pose_t[0, 3:7]
            )
            scan_desired_position_t[:, 0] += box_x_delta_m
            scan_desired_position_t[:, 1] += box_y_delta_m
            scan_desired_position_t[:, 2] += box_z_delta_m
            left_roll_delta_rad = np.deg2rad(left_normal_roll_delta_deg)
            left_roll_delta_quaternion = torch.tensor(
                (
                    np.cos(0.5 * left_roll_delta_rad),
                    np.sin(0.5 * left_roll_delta_rad),
                    0.0,
                    0.0,
                ),
                device=device,
                dtype=scan_desired_quaternion_t.dtype,
            )
            scan_desired_quaternion_t[0] = math_utils.quat_mul(
                scan_desired_quaternion_t[0], left_roll_delta_quaternion
            )
            right_roll_delta_rad = np.deg2rad(right_normal_roll_delta_deg)
            right_roll_delta_quaternion = torch.tensor(
                (
                    np.cos(0.5 * right_roll_delta_rad),
                    np.sin(0.5 * right_roll_delta_rad),
                    0.0,
                    0.0,
                ),
                device=device,
                dtype=scan_desired_quaternion_t.dtype,
            )
            scan_desired_quaternion_t[1] = math_utils.quat_mul(
                scan_desired_quaternion_t[1], right_roll_delta_quaternion
            )
            scan_box_pose_t[:, 0] += box_x_delta_m
            scan_box_pose_t[:, 1] += box_y_delta_m
            scan_box_pose_t[:, 2] += box_z_delta_m
            scan_box_state = source_box_state.copy()
            scan_box_state[:7] = cpu(scan_box_pose_t[0])
            candidate_root = source_root_t.clone()
            candidate_root[0] += root_x_delta_m
            candidate_root[1] += root_y_delta_m
            candidate_root[2] += root_z_delta_m
            yaw_delta_rad = np.deg2rad(root_yaw_delta_deg)
            yaw_delta_quat = torch.tensor(
                (
                    np.cos(0.5 * yaw_delta_rad),
                    0.0,
                    0.0,
                    np.sin(0.5 * yaw_delta_rad),
                ),
                device=device,
                dtype=candidate_root.dtype,
            )
            candidate_root[3:7] = math_utils.quat_mul(
                yaw_delta_quat, candidate_root[3:7]
            )
            candidate_joint = source_joint_t.clone()
            candidate_joint[waist_pitch_id] = waist_pitch_rad
            candidate_joint = torch.clamp(candidate_joint, lower, upper)
            box.write_root_pose_to_sim(far_box_pose_t)
            box.write_root_velocity_to_sim(zero_box_velocity)
            robot.write_root_state_to_sim(candidate_root.unsqueeze(0))
            robot.write_joint_state_to_sim(
                candidate_joint.unsqueeze(0),
                torch.zeros_like(candidate_joint).unsqueeze(0),
            )
            robot.set_joint_position_target(candidate_joint.unsqueeze(0))
            robot.set_joint_velocity_target(
                torch.zeros_like(candidate_joint).unsqueeze(0)
            )
            robot.write_data_to_sim()
            sim.forward()
            robot.update(0.005)
            for controller in controllers.values():
                controller.reset()

            iterations = 0
            for iterations in range(1, args.maximum_ik_iterations + 1):
                root_pose_w = robot.data.root_pose_w
                base_rotation = math_utils.matrix_from_quat(
                    math_utils.quat_inv(root_pose_w[:, 3:7])
                )
                updates = []
                for side_index, side in enumerate(SIDES):
                    body_id = ee_body_ids[side]
                    ids = arm_joint_ids[side]
                    desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
                        root_pose_w[:, :3],
                        root_pose_w[:, 3:7],
                        scan_desired_position_t[side_index].unsqueeze(0),
                        scan_desired_quaternion_t[side_index].unsqueeze(0),
                    )
                    jacobian = robot.root_physx_view.get_jacobians()[
                        :, ee_jacobian_ids[side], :, ids
                    ].clone()
                    jacobian[:, :3, :] = torch.bmm(
                        base_rotation, jacobian[:, :3, :]
                    )
                    jacobian[:, 3:, :] = torch.bmm(
                        base_rotation, jacobian[:, 3:, :]
                    )
                    ee_pose_w = robot.data.body_pose_w[:, body_id]
                    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                        root_pose_w[:, :3],
                        root_pose_w[:, 3:7],
                        ee_pose_w[:, :3],
                        ee_pose_w[:, 3:7],
                    )
                    controllers[side].set_command(
                        torch.cat((desired_pos_b, desired_quat_b), dim=1)
                    )
                    current = robot.data.joint_pos[:, ids]
                    proposed = controllers[side].compute(
                        ee_pos_b, ee_quat_b, jacobian, current
                    )
                    proposed = torch.maximum(
                        torch.minimum(proposed, upper[ids].unsqueeze(0)),
                        lower[ids].unsqueeze(0),
                    )
                    proposed = torch.maximum(
                        torch.minimum(proposed, current + 0.025), current - 0.025
                    )
                    updates.append((ids, proposed))
                for ids, proposed in updates:
                    robot.write_joint_state_to_sim(
                        proposed,
                        torch.zeros_like(proposed),
                        joint_ids=ids,
                    )
                    robot.set_joint_position_target(proposed, joint_ids=ids)
                robot.write_data_to_sim()
                sim.forward()
                robot.update(0.005)
                actual_position = torch.stack(
                    [robot.data.body_pos_w[0, ee_body_ids[side]] for side in SIDES]
                )
                actual_quaternion = torch.stack(
                    [robot.data.body_quat_w[0, ee_body_ids[side]] for side in SIDES]
                )
                position_error = torch.linalg.norm(
                    actual_position - scan_desired_position_t, dim=1
                )
                rotation_error = math_utils.quat_error_magnitude(
                    actual_quaternion, scan_desired_quaternion_t
                )
                if float(position_error.max().item()) <= 0.002 and float(
                    rotation_error.max().item()
                ) <= 0.05:
                    break

            actual_position = torch.stack(
                [robot.data.body_pos_w[0, ee_body_ids[side]].clone() for side in SIDES]
            )
            actual_quaternion = torch.stack(
                [robot.data.body_quat_w[0, ee_body_ids[side]].clone() for side in SIDES]
            )
            position_error = cpu(
                torch.linalg.norm(actual_position - scan_desired_position_t, dim=1)
            )
            rotation_error = cpu(
                math_utils.quat_error_magnitude(
                    actual_quaternion, scan_desired_quaternion_t
                )
            )
            solved_joint = cpu(robot.data.joint_pos[0])
            solved_root = cpu(robot.data.root_state_w[0])
            head_position = cpu(robot.data.body_pos_w[0, head_robot_index]).astype(
                np.float64
            )
            head_quaternion = cpu(
                robot.data.body_quat_w[0, head_robot_index]
            ).astype(np.float64)
            head_obb_gap_m, head_vertices_inside_obb = head_to_box_obb_gap(
                head_vertices_body,
                head_position,
                head_quaternion,
                scan_box_state[:7].astype(np.float64),
                pca_center_b,
                pca_basis_b,
                pca_bounds,
            )

            # One exact collision-query step with a fixed box; no claim uses
            # the resulting force magnitude, only named contact presence.
            box.write_root_pose_to_sim(scan_box_pose_t)
            box.write_root_velocity_to_sim(zero_box_velocity)
            sensor.reset()
            sim.forward()
            sim.step(render=False)
            robot.update(0.005)
            box.update(0.005)
            sensor.update(0.005, force_recompute=True)
            normal = cpu(sensor.data.force_matrix_w[0, :, 0]).astype(np.float64)
            friction = cpu(sensor.data.friction_forces_w[0, :, 0]).astype(np.float64)
            normal_load = np.linalg.norm(normal, axis=1)
            direct_load = np.linalg.norm(normal + friction, axis=1)
            active = np.flatnonzero(normal_load > 0.01)
            nonhand_load_n = float(normal_load[nonhand_indices].sum())
            candidate = {
                "box_x_delta_m": box_x_delta_m,
                "box_y_delta_m": box_y_delta_m,
                "box_z_delta_m": box_z_delta_m,
                "box_pca1_tilt_deg": box_pca1_tilt_deg,
                "root_x_delta_m": root_x_delta_m,
                "root_x_w_m": float(solved_root[0]),
                "root_y_delta_m": root_y_delta_m,
                "root_y_w_m": float(solved_root[1]),
                "root_z_delta_m": root_z_delta_m,
                "root_z_w_m": float(solved_root[2]),
                "root_yaw_delta_deg": root_yaw_delta_deg,
                "left_normal_roll_delta_deg": left_normal_roll_delta_deg,
                "right_normal_roll_delta_deg": right_normal_roll_delta_deg,
                "waist_pitch_rad": waist_pitch_rad,
                "ik_iterations": iterations,
                "position_error_m": position_error.tolist(),
                "position_error_vector_w_m": cpu(
                    actual_position - scan_desired_position_t
                ).tolist(),
                "actual_hand_position_w_m": cpu(actual_position).tolist(),
                "desired_hand_position_w_m": cpu(
                    scan_desired_position_t
                ).tolist(),
                "rotation_error_rad": rotation_error.tolist(),
                "maximum_position_error_m": float(position_error.max()),
                "maximum_rotation_error_rad": float(rotation_error.max()),
                "head_to_box_pca_obb_gap_m": head_obb_gap_m,
                "head_position_w_m": head_position.tolist(),
                "box_position_w_m": scan_box_state[:3].astype(float).tolist(),
                "head_collision_vertices_inside_box_pca_obb": head_vertices_inside_obb,
                "head_contact_normal_load_n": float(normal_load[head_contact_index]),
                "nonhand_contact_normal_load_n": nonhand_load_n,
                "active_contact_bodies": {
                    contact_names[index]: {
                        "normal_load_n": float(normal_load[index]),
                        "direct_load_n": float(direct_load[index]),
                    }
                    for index in active
                },
                "ik_reachable_5mm_0p15rad": bool(
                    position_error.max() <= 0.005
                    and rotation_error.max() <= 0.15
                ),
                "nonhand_contact_absent_0p01n": nonhand_load_n <= 0.01,
            }
            candidate["static_candidate"] = bool(
                candidate["ik_reachable_5mm_0p15rad"]
                and candidate["nonhand_contact_absent_0p01n"]
                and head_vertices_inside_obb == 0
            )
            records.append(candidate)
            joint_solutions.append(solved_joint.astype(np.float32))
            root_solutions.append(solved_root.astype(np.float32))
            print(json.dumps(candidate, sort_keys=True), flush=True)

    candidate_indices = [
        index for index, record in enumerate(records) if record["static_candidate"]
    ]
    live_upper_scan_value = max(feasible_waist_values)
    baseline = min(
        records,
        key=lambda record: (
            abs(record["box_z_delta_m"])
            + abs(record["box_x_delta_m"])
            + abs(record["box_y_delta_m"])
            + abs(np.deg2rad(record["box_pca1_tilt_deg"]))
            + abs(record["root_x_delta_m"])
            + abs(record["root_y_delta_m"])
            + abs(record["root_z_delta_m"])
            + abs(np.deg2rad(record["root_yaw_delta_deg"]))
            + abs(np.deg2rad(record["left_normal_roll_delta_deg"]))
            + abs(np.deg2rad(record["right_normal_roll_delta_deg"])),
            abs(record["waist_pitch_rad"] - live_upper_scan_value),
        ),
    )
    ranked_indices = sorted(
        range(len(records)),
        key=lambda index: (
            not records[index]["static_candidate"],
            records[index]["nonhand_contact_normal_load_n"],
            records[index]["maximum_position_error_m"],
            -records[index]["head_to_box_pca_obb_gap_m"],
        ),
    )
    arrays_path = output_root / "solutions.npz"
    atomic_npz(
        arrays_path,
        {
            "joint_position": np.stack(joint_solutions),
            "root_state": np.stack(root_solutions),
            "box_z_delta_m": np.asarray(
                [record["box_z_delta_m"] for record in records], dtype=np.float32
            ),
            "box_x_delta_m": np.asarray(
                [record["box_x_delta_m"] for record in records], dtype=np.float32
            ),
            "box_y_delta_m": np.asarray(
                [record["box_y_delta_m"] for record in records], dtype=np.float32
            ),
            "box_pca1_tilt_deg": np.asarray(
                [record["box_pca1_tilt_deg"] for record in records],
                dtype=np.float32,
            ),
            "root_x_delta_m": np.asarray(
                [record["root_x_delta_m"] for record in records], dtype=np.float32
            ),
            "root_y_delta_m": np.asarray(
                [record["root_y_delta_m"] for record in records], dtype=np.float32
            ),
            "root_z_delta_m": np.asarray(
                [record["root_z_delta_m"] for record in records], dtype=np.float32
            ),
            "root_yaw_delta_deg": np.asarray(
                [record["root_yaw_delta_deg"] for record in records],
                dtype=np.float32,
            ),
            "left_normal_roll_delta_deg": np.asarray(
                [record["left_normal_roll_delta_deg"] for record in records],
                dtype=np.float32,
            ),
            "right_normal_roll_delta_deg": np.asarray(
                [record["right_normal_roll_delta_deg"] for record in records],
                dtype=np.float32,
            ),
            "waist_pitch_rad": np.asarray(
                [record["waist_pitch_rad"] for record in records], dtype=np.float32
            ),
            "candidate_mask": np.asarray(
                [record["static_candidate"] for record in records], dtype=bool
            ),
            "ranked_indices": np.asarray(ranked_indices, dtype=np.int32),
        },
    )
    baseline_check_name = (
        "baseline_root020_waist052_reproduces_head_contact"
        if args.source_index is None
        and args.target_source == "legacy_contact_anchor"
        else "source_row_baseline_is_finite"
    )
    baseline_check = (
        bool(
            baseline["head_contact_normal_load_n"] > 0.01
            and baseline["head_collision_vertices_inside_box_pca_obb"] > 0
        )
        if args.source_index is None
        and args.target_source == "legacy_contact_anchor"
        else bool(
            np.isfinite(baseline["maximum_position_error_m"])
            and np.isfinite(baseline["maximum_rotation_error_rad"])
        )
    )
    checks = {
        "official_usd_hash_exact": sha256(official_usd)
        == EXPECTED_USD_SHA256,
        "exact_source_orders_reproduced": True,
        baseline_check_name: baseline_check,
        "all_declared_candidates_executed": len(records)
        == len(args.box_x_deltas_m)
        * len(args.box_y_deltas_m)
        * len(args.box_z_deltas_m)
        * len(args.box_pca1_tilt_deg)
        * len(args.root_x_deltas_m)
        * len(args.root_y_deltas_m)
        * len(args.root_z_deltas_m)
        * len(args.root_yaw_deltas_deg)
        * len(args.left_normal_roll_deltas_deg)
        * len(args.right_normal_roll_deltas_deg)
        * len(feasible_waist_values),
        "all_solutions_finite": bool(
            np.isfinite(np.stack(joint_solutions)).all()
            and np.isfinite(np.stack(root_solutions)).all()
        ),
    }
    report = {
        "schema": "plan10_fullbody_root_xyz_yaw_waist_static_feasibility_v3",
        "host": HOST,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": args.device,
        "sources": {
            "official_robot_usd": str(official_usd),
            "official_robot_usd_sha256": sha256(official_usd),
            "official_box_usd": str(box_usd),
            "official_box_usd_sha256": sha256(box_usd),
            "source_trace": str(source_trace),
            "source_trace_sha256": sha256(source_trace),
            "collision_points": str(collision_points_path),
            "collision_points_sha256": sha256(collision_points_path),
            "box_envelope": str(box_envelope_path),
            "box_envelope_sha256": sha256(box_envelope_path),
            "producer": str(Path(__file__).resolve()),
            "producer_sha256": sha256(Path(__file__).resolve()),
        },
        "parameters": {
            "source_index": args.source_index,
            "target_source": args.target_source,
            "box_x_deltas_m": list(args.box_x_deltas_m),
            "box_y_deltas_m": list(args.box_y_deltas_m),
            "box_z_deltas_m": list(args.box_z_deltas_m),
            "box_pca1_tilt_deg": list(args.box_pca1_tilt_deg),
            "box_pivot_pca_m": (
                list(args.box_pivot_pca_m)
                if args.box_pivot_pca_m is not None
                else None
            ),
            "root_x_deltas_m": list(args.root_x_deltas_m),
            "root_y_deltas_m": list(args.root_y_deltas_m),
            "root_z_deltas_m": list(args.root_z_deltas_m),
            "root_yaw_deltas_deg": list(args.root_yaw_deltas_deg),
            "left_normal_roll_deltas_deg": list(
                args.left_normal_roll_deltas_deg
            ),
            "right_normal_roll_deltas_deg": list(
                args.right_normal_roll_deltas_deg
            ),
            "waist_pitch_absolute_rad_declared": list(args.waist_pitch_rad),
            "waist_pitch_absolute_rad_live_feasible": feasible_waist_values,
            "maximum_ik_iterations": args.maximum_ik_iterations,
            "ik_controller": "official IsaacLab DifferentialIKController pose/dls",
            "position_gate_m": 0.005,
            "rotation_gate_rad": 0.15,
            "nonhand_contact_gate_n": 0.01,
        },
        "baseline": baseline,
        "candidate_count": len(candidate_indices),
        "candidate_indices": candidate_indices,
        "ranked_indices": ranked_indices,
        "records": records,
        "arrays": {
            "path": str(arrays_path),
            "sha256": sha256(arrays_path),
        },
        "checks": checks,
        "audit_passed": all(checks.values()),
        "claim_boundary": (
            "No-learning fixed-box geometry and IK feasibility only. A static "
            "candidate does not prove physical grasp, lift, force balance, standing, "
            "tactile sensing, recovery, exploration, or policy behavior."
        ),
    }
    atomic_json(output_root / "manifest.json", report)
    print(
        json.dumps(
            {
                "audit_passed": report["audit_passed"],
                "candidate_count": len(candidate_indices),
                "best_index": ranked_indices[0],
                "best": records[ranked_indices[0]],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not report["audit_passed"]:
        raise RuntimeError("Full-body root/waist feasibility audit failed")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
