"""Compute-only Taccel contact placement sweep for Phase 00."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh as tm
import warp as wp
from scipy.spatial.transform import Rotation as R

from taccel.taccel import TaccelModel
from warp_ipc.ipc_integrator import IPCIntegrator
from warp_ipc.robots import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--x-centers", default="-0.02,0.00,0.02,0.04,0.06,0.08,0.10,0.12")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def marker_local_by_pad(model: TaccelModel) -> list[np.ndarray]:
    current_all = model.tac_markers_local[0].detach().cpu().numpy()
    out = []
    offset = 0
    for rest in model.rest_markers[0]:
        count = rest.shape[0]
        out.append(current_all[offset : offset + count])
        offset += count
    return out


def force_norm(model: TaccelModel, handle, dt: float) -> float:
    try:
        model.update_contact_force(dt)
        return float(np.linalg.norm(model.get_body_resultant_contact_force(handle, dt).detach().cpu().numpy()))
    except Exception:
        return -1.0


def bbox_payload(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {"min": lo.tolist(), "max": hi.tolist(), "center": ((lo + hi) * 0.5).tolist()}


def min_vertex_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    best = float("inf")
    chunk = 512
    for start in range(0, len(a), chunk):
        diff = a[start : start + chunk, None, :] - b[None, :, :]
        best = min(best, float(np.sqrt(np.square(diff).sum(axis=-1)).min()))
    return best


def collision_count(model: TaccelModel) -> int:
    try:
        return int(model.cdw.num_collisions.numpy()[0])
    except Exception:
        return -1


def layer_debug(model: TaccelModel, object_handle) -> dict:
    try:
        layers_np = model.body_collision_layer.numpy().astype(int)
        filt_np = model.collision_layer_filter.numpy().astype(int)
        layers = layers_np.tolist()
        gel_body_indices = [int(handle.body_index) for handle in model.robots[0].gel_handles]
        object_body_index = int(object_handle.body_index)
        object_layer = int(layers[object_body_index])
        gel_layers = [int(layers[idx]) for idx in gel_body_indices]
        object_gel_filters = []
        filter_values = {}
        for gel_layer in gel_layers:
            for layer in (object_layer, gel_layer):
                if 0 <= layer < filt_np.size:
                    filter_values[str(layer)] = int(filt_np.reshape(-1)[layer])
            if filt_np.ndim == 2:
                object_gel_filters.append([int(filt_np[object_layer, gel_layer]), int(filt_np[gel_layer, object_layer])])
            else:
                idx_a = object_layer * model.NUM_COLLISION_LAYERS + gel_layer
                idx_b = gel_layer * model.NUM_COLLISION_LAYERS + object_layer
                if idx_a < filt_np.size and idx_b < filt_np.size:
                    object_gel_filters.append([int(filt_np.reshape(-1)[idx_a]), int(filt_np.reshape(-1)[idx_b])])
                else:
                    object_gel_filters.append(["out_of_range", "out_of_range"])
        return {
            "available": True,
            "num_collision_layers": int(model.NUM_COLLISION_LAYERS),
            "body_collision_layer_shape": list(layers_np.shape),
            "collision_layer_filter_shape": list(filt_np.shape),
            "collision_layer_filter_size": int(filt_np.size),
            "filter_values_for_used_layers": filter_values,
            "object_body_index": object_body_index,
            "gel_body_indices": gel_body_indices,
            "object_layer": object_layer,
            "gel_layers": gel_layers,
            "object_gel_filters": object_gel_filters,
        }
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def make_object(model: TaccelModel, handle_x: float):
    shift = handle_x - 0.02
    handle_extents = np.array([0.0075, 0.1, 0.01], dtype=np.float64) * 2.0
    obj_handle_mesh = tm.primitives.Box(extents=handle_extents).to_mesh()
    obj_handle_mesh.vertices = np.asarray(obj_handle_mesh.vertices) + np.array([[0.02 + shift, 0.0, 0.1]])
    board_extent = np.array([0.3, 0.3], dtype=np.float64)
    obj_board_mesh = tm.primitives.Box(extents=np.array([0.01, board_extent[0], board_extent[1]])).to_mesh()
    obj_board_mesh.vertices = np.asarray(obj_board_mesh.vertices) + np.array([[-0.05 + shift, 0.0, 0.1]])
    obj_mesh = tm.util.concatenate([obj_handle_mesh, obj_board_mesh])
    edge_verts = np.array(
        [
            [-0.055 + shift, board_extent[0] / 2.0, board_extent[1] / 2.0 + 0.1],
            [-0.055 + shift, -board_extent[0] / 2.0, board_extent[1] / 2.0 + 0.1],
            [-0.055 + shift, -board_extent[0] / 2.0, -board_extent[1] / 2.0 + 0.1],
            [-0.055 + shift, board_extent[0] / 2.0, -board_extent[1] / 2.0 + 0.1],
        ],
        dtype=np.float64,
    )
    handle = model.add_affine_body(
        obj_mesh.vertices,
        obj_mesh.faces.astype(np.int32),
        density=1000.0,
        E=1.0e9,
        mu=0.3,
        mass_xi=0.08 / len(obj_mesh.vertices),
        env_id=0,
        nu=0.3,
    )
    return handle, edge_verts


@torch.no_grad()
def run_variant(x_center: float, args: argparse.Namespace) -> dict:
    model = TaccelModel(num_envs=1, viz_envs=[], device=args.device)
    model.set_kinematic_stiffness(1.0e8)
    model.gravity = wp.vec3d([0.0, 0.0, 0.0])
    model.dhat = 1.0e-4
    object_handle, edge_verts = make_object(model, x_center)
    urdf_path, _, tac_path = Robot.get_fabr_path("tactile-pandahand", 1.0e-7)
    model.add_robot(urdf_path, tac_fab_path=tac_path, env_id=0, start_coll_layer=2, coll_layers=[], disable_coll_layers=[1])
    model.add_vbts_to_sim(model.robots[0], coll_layers=1)
    model.init()
    joint_axis = edge_verts[1] - edge_verts[0]
    joint_axis = joint_axis / np.linalg.norm(joint_axis)
    model.add_world_revolute_joint(object_handle, edge_verts[0], joint_axis)
    layers = layer_debug(model, object_handle)
    hand_tf = np.eye(4)
    hand_tf[:3, 3] = np.array([0.12, 0.0, 0.1])
    hand_tf[:3, :3] = R.from_euler("xyz", np.array([90, 0, -90]), degrees=True).as_matrix()
    model.finalize()
    integrator = IPCIntegrator(device=args.device)
    integrator.use_hard_kinematic_constraint = False
    integrator.use_cpu = False
    integrator.max_newton_iter = 50
    integrator.use_inversion_free_step_size_filter = True
    integrator.inversion_free_im_tol = 1.0e-6
    integrator.inversion_free_cubic_coef_tol = 1.0e-10
    release_q = 0.04
    grasp_q = 0.01 - 8.0e-4 + model.dhat
    init_q = {"panda_finger_joint1": release_q, "panda_finger_joint2": release_q}
    model.set_robot_states([init_q], hand_tf[None])
    model.set_robot_targets([init_q], hand_tf[None])
    model.apply_set_state()
    ref = None
    max_collision = 0
    max_object_force = 0.0
    max_gel_force = 0.0
    max_marker_z = 0.0
    final_geometry = {}
    for step in range(args.steps):
        alpha = min(1.0, step / max(args.steps - 1, 1))
        q = release_q + (grasp_q - release_q) * alpha
        hand_q = {"panda_finger_joint1": float(q), "panda_finger_joint2": float(q)}
        model.set_robot_targets([hand_q], hand_tf[None])
        integrator.simulate(model, dt=args.dt)
        max_collision = max(max_collision, collision_count(model))
        max_object_force = max(max_object_force, force_norm(model, object_handle, args.dt))
        for gel_handle in model.robots[0].gel_handles:
            max_gel_force = max(max_gel_force, force_norm(model, gel_handle, args.dt))
        markers = marker_local_by_pad(model)
        if ref is None:
            ref = [m.copy() for m in markers]
        for curr, base in zip(markers, ref):
            max_marker_z = max(max_marker_z, float(np.abs(curr[:, 2] - base[:, 2]).max()))
    object_vertices = np.asarray(model.get_affine_body_mesh_from_handle(object_handle).vertices)
    gel_vertices = [np.asarray(model.get_soft_body_pos(handle, True)) for handle in model.robots[0].gel_handles]
    final_geometry = {
        "object_bbox": bbox_payload(object_vertices),
        "gel_bboxes": [bbox_payload(vertices) for vertices in gel_vertices],
        "object_gel_min_vertex_distance_m": [min_vertex_distance(object_vertices, vertices) for vertices in gel_vertices],
    }
    return {
        "x_center": x_center,
        "layer_debug": layers,
        "final_geometry": final_geometry,
        "max_collision_count": int(max_collision),
        "max_object_force_norm": max_object_force,
        "max_gel_force_norm": max_gel_force,
        "max_marker_z_delta_m": max_marker_z,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    wp.config.kernel_cache_dir = os.environ.get("TACCEL_PTX_DIR", str(Path(args.output_json).parent / "ptx"))
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = int(os.environ.get("TACCEL_PTX_ARCH", "86"))
    wp.init()
    x_centers = [float(item) for item in args.x_centers.split(",") if item.strip()]
    results = [run_variant(x, args) for x in x_centers]
    payload = {
        "classification": "phase00_taccel_contact_placement_sweep_v1",
        "not_training_result": True,
        "not_curiosity_success": True,
        "results": results,
        "best_by_collision": max(results, key=lambda item: item["max_collision_count"]),
        "best_by_gel_force": max(results, key=lambda item: item["max_gel_force_norm"]),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
