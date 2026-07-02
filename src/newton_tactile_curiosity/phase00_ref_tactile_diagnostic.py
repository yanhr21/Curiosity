"""Phase 00 reference-video-aligned tactile environment diagnostic.

This script is intentionally compute-only. It must run inside a Curiosity
Slurm allocation via the phase00 launcher, not on a login node.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
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
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--object-density", type=float, default=7800.0)
    parser.add_argument("--object-youngs-modulus", type=float, default=1.0e9)
    parser.add_argument("--object-friction", type=float, default=0.3)
    parser.add_argument("--max-newton-iter", type=int, default=50)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image)
    if image.max() <= 1.5:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def heatmap(values: np.ndarray, vmax: float | None = None) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32))
    vmax = float(vmax if vmax is not None else max(values.max(), 1.0e-8))
    norm = np.clip(values / max(vmax, 1.0e-8), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def line_plot(series: list[float], size: tuple[int, int], label: str, color: tuple[int, int, int]) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (190, 190, 190), 1)
    cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    if len(series) < 2:
        return canvas
    arr = np.asarray(series, dtype=np.float32)
    ymin = min(0.0, float(np.nanmin(arr)))
    ymax = max(1.0e-8, float(np.nanmax(arr)))
    if abs(ymax - ymin) < 1.0e-8:
        ymax = ymin + 1.0
    xs = np.linspace(6, width - 8, len(arr)).astype(np.int32)
    ys = (height - 12 - (arr - ymin) / (ymax - ymin) * (height - 38)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{arr[-1]:.4g}", (width - 92, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return canvas


def draw_marker_flow(markers: np.ndarray, rest: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    pts = markers[:, :2]
    ref = rest[:, :2]
    all_pts = np.concatenate([pts, ref], axis=0)
    span = np.maximum(all_pts.max(axis=0) - all_pts.min(axis=0), 1.0e-6)
    lo = all_pts.min(axis=0) - span * 0.12
    scale = np.array([width - 24, height - 24]) / (span * 1.24)

    def project(xy: np.ndarray) -> np.ndarray:
        out = (xy - lo) * scale + np.array([12.0, 12.0])
        out[:, 1] = height - out[:, 1]
        return out.astype(np.int32)

    p = project(pts)
    r = project(ref)
    for start, end in zip(r, p):
        cv2.arrowedLine(canvas, tuple(start), tuple(end), (180, 30, 180), 1, cv2.LINE_AA, tipLength=0.25)
    for point in p:
        cv2.circle(canvas, tuple(point), 2, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.putText(canvas, "marker flow", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def split_tactile_markers(model: TaccelModel) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return per-pad marker positions in tactile-local coordinates."""

    current_all = model.tac_markers_local[0].detach().cpu().numpy()
    current_by_pad = []
    rest_by_pad = []
    offset = 0
    for rest in model.rest_markers[0]:
        rest_np = rest.detach().cpu().numpy()
        count = rest_np.shape[0]
        current_by_pad.append(current_all[offset : offset + count])
        rest_by_pad.append(rest_np)
        offset += count
    return current_by_pad, rest_by_pad


def draw_scene_projection(mesh: tm.Trimesh, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.size == 0:
        cv2.putText(canvas, "scene projection unavailable", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        return canvas
    pts = verts[:, [0, 2]]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-6)
    pad = span * 0.12
    lo = lo - pad
    span = span + 2.0 * pad
    proj = (pts - lo) / span * np.array([width - 24, height - 28]) + np.array([12, 14])
    proj[:, 1] = height - proj[:, 1]
    proj = proj.astype(np.int32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    step = max(1, len(faces) // 5000)
    for tri in faces[::step]:
        cv2.polylines(canvas, [proj[tri].reshape(-1, 1, 2)], True, (80, 110, 130), 1, cv2.LINE_AA)
    cv2.putText(canvas, "scene x-z projection", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def body_force_summary(model: TaccelModel, handle, dt: float) -> dict:
    summary: dict[str, object] = {"available": True}
    try:
        resultant = model.get_body_resultant_contact_force(handle, dt).detach().cpu().numpy()
        collision = model.get_body_nodal_collision_force(handle, dt).detach().cpu().numpy()
        friction = model.get_body_nodal_friction_force(handle, dt).detach().cpu().numpy()
        summary.update(
            {
                "resultant_contact_force_xyz": np.asarray(resultant).reshape(-1).astype(float).tolist(),
                "resultant_contact_force_norm": float(np.linalg.norm(resultant)),
                "collision_force_norm_sum": float(np.linalg.norm(collision.reshape(-1, 3), axis=1).sum()),
                "friction_force_norm_sum": float(np.linalg.norm(friction.reshape(-1, 3), axis=1).sum()),
                "nodal_force_count": int(collision.reshape(-1, 3).shape[0]),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report official API failure in artifact metadata.
        summary = {"available": False, "error": repr(exc)}
    return summary


def collision_summary(model: TaccelModel) -> dict[str, int | bool | str]:
    try:
        num_collisions = int(model.cdw.num_collisions.numpy()[0])
        num_hs_pair = int(model.cdw.num_hs_pair.numpy()[0])
        return {"available": True, "num_collisions": num_collisions, "num_hs_pair": num_hs_pair}
    except Exception as exc:  # noqa: BLE001 - record API mismatch as evidence.
        return {"available": False, "error": repr(exc)}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    visual_dir = Path(args.visual_dir)
    report_dir = Path(args.report_dir)
    for path in (output_dir, visual_dir, report_dir, output_dir / "scene_ply"):
        path.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    wp.config.kernel_cache_dir = os.environ.get("TACCEL_PTX_DIR", str(output_dir / "ptx"))
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = int(os.environ.get("TACCEL_PTX_ARCH", "86"))
    wp.init()

    model = TaccelModel(num_envs=1, viz_envs=[], device=args.device)
    model.set_kinematic_stiffness(1.0e8)
    model.gravity = wp.vec3d([0.0, 0.0, 0.0])
    model.dhat = 1.0e-4
    model.tol = 1.0e-3
    model.epsv = 1.0e-2

    handle_extents = np.array([0.0075, 0.1, 0.01], dtype=np.float64) * 2.0
    obj_handle_mesh = tm.primitives.Box(extents=handle_extents).to_mesh()
    obj_handle_mesh.vertices = np.asarray(obj_handle_mesh.vertices) + np.array([[0.02, 0.0, 0.1]])
    board_extent = np.array([0.3, 0.3], dtype=np.float64)
    obj_board_mesh = tm.primitives.Box(extents=np.array([0.01, board_extent[0], board_extent[1]])).to_mesh()
    obj_board_mesh.vertices = np.asarray(obj_board_mesh.vertices) + np.array([[-0.05, 0.0, 0.1]])
    steel_bar = tm.util.concatenate([obj_handle_mesh, obj_board_mesh])
    edge_verts = np.array(
        [
            [-0.055, board_extent[0] / 2.0, board_extent[1] / 2.0 + 0.1],
            [-0.055, -board_extent[0] / 2.0, board_extent[1] / 2.0 + 0.1],
            [-0.055, -board_extent[0] / 2.0, -board_extent[1] / 2.0 + 0.1],
            [-0.055, board_extent[0] / 2.0, -board_extent[1] / 2.0 + 0.1],
        ],
        dtype=np.float64,
    )
    object_handle = model.add_affine_body(
        steel_bar.vertices,
        steel_bar.faces.astype(np.int32),
        density=args.object_density,
        E=args.object_youngs_modulus,
        mu=args.object_friction,
        mass_xi=0.08 / len(steel_bar.vertices),
        env_id=0,
        nu=0.3,
    )

    urdf_path, _, tac_path = Robot.get_fabr_path("tactile-pandahand", 1.0e-7)
    model.add_robot(
        urdf_path,
        tac_fab_path=tac_path,
        env_id=0,
        start_coll_layer=2,
        coll_layers=[],
        disable_coll_layers=[1],
    )
    model.add_vbts_to_sim(model.robots[0], coll_layers=1)
    model.init()
    joint_axis = edge_verts[1] - edge_verts[0]
    joint_axis = joint_axis / np.linalg.norm(joint_axis)
    model.add_world_revolute_joint(object_handle, edge_verts[0], joint_axis)

    gripper_init_pos = np.array([0.12, 0.0, 0.1])
    gripper_init_rot = R.from_euler("xyz", np.array([90, 0, -90]), degrees=True).as_matrix()
    hand_tf = np.eye(4)
    hand_tf[:3, 3] = gripper_init_pos
    hand_tf[:3, :3] = gripper_init_rot
    model.finalize()

    integrator = IPCIntegrator(device=args.device)
    integrator.use_hard_kinematic_constraint = False
    integrator.use_cpu = False
    integrator.max_newton_iter = args.max_newton_iter
    integrator.max_cg_iter = 80
    integrator.cg_rel_tol = 1.0e-5
    integrator.use_inversion_free_step_size_filter = True
    integrator.inversion_free_im_tol = 1.0e-6
    integrator.inversion_free_cubic_coef_tol = 1.0e-10

    release_q = 0.04
    grasp_q = handle_extents[2] / 2.0 - 8.0e-4 + model.dhat
    init_q = {"panda_finger_joint1": release_q, "panda_finger_joint2": release_q}
    model.set_robot_states([init_q], hand_tf[None])
    model.set_robot_targets([init_q], hand_tf[None])

    video_path = visual_dir / "ref_tactile_diag.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    normal_frames: list[np.ndarray] = []
    marker_disp_mean: list[list[float]] = []
    marker_disp_max: list[list[float]] = []
    contact_area_px: list[list[int]] = []
    force_records: list[list[dict]] = []
    collision_records: list[dict] = []
    fps_series: list[float] = []
    frame_summaries: list[dict] = []
    marker_reference: list[np.ndarray] | None = None

    for step in range(args.steps):
        alpha = min(1.0, step / max(args.steps - 1, 1))
        q = release_q + (grasp_q - release_q) * alpha
        hand_q = {"panda_finger_joint1": float(q), "panda_finger_joint2": float(q)}
        model.set_robot_targets([hand_q], hand_tf[None])
        integrator.simulate(model, dt=args.dt)
        try:
            model.update_contact_force(args.dt)
        except Exception:
            pass
        collisions = collision_summary(model)

        try:
            model.vbts_depths.zero_()
            model.vbts_normals.zero_()
        except Exception:
            pass
        rgbs, depths, normals = model.render_tactile(False, False)
        rgbs = np.asarray(rgbs)
        depths = np.asarray(depths)
        normals = np.asarray(normals)
        markers, _rest_markers = split_tactile_markers(model)
        if marker_reference is None:
            marker_reference = [m.copy() for m in markers]

        per_pad_mean = []
        per_pad_max = []
        per_pad_area = []
        per_pad_force = []
        marker_panels = []
        for pad_id, handle in enumerate(model.robots[0].gel_handles):
            disp = np.asarray(markers[pad_id]) - marker_reference[pad_id]
            mag = np.linalg.norm(disp, axis=1)
            per_pad_mean.append(float(mag.mean()))
            per_pad_max.append(float(mag.max()))
            per_pad_area.append(int((depths[0, pad_id] > 1.0e-5).sum()))
            per_pad_force.append(body_force_summary(model, handle, args.dt))
            marker_panels.append(draw_marker_flow(np.asarray(markers[pad_id]), marker_reference[pad_id], (300, 210)))

        sim_step_time = float(integrator.profile_helper.current_timestep_data.get("total_timestep", 0.0))
        fps_series.append(float(1.0 / sim_step_time) if sim_step_time > 0 else 0.0)
        marker_disp_mean.append(per_pad_mean)
        marker_disp_max.append(per_pad_max)
        contact_area_px.append(per_pad_area)
        force_records.append(per_pad_force)
        collision_records.append(collisions)
        rgb_frames.append(rgbs[0].copy())
        depth_frames.append(depths[0].copy())
        normal_frames.append(normals[0].copy())

        if step % 8 == 0 or step == args.steps - 1:
            model.write_scene(str(output_dir / "scene_ply" / f"scene_{step:04d}.ply"))

        left_rgb = cv2.resize(as_uint8_rgb(rgbs[0, 0]), (240, 240))
        right_rgb = cv2.resize(as_uint8_rgb(rgbs[0, 1]), (240, 240))
        depth_vmax = max(float(depths[0].max()), 1.0e-6)
        left_depth = cv2.resize(heatmap(depths[0, 0], depth_vmax), (240, 240))
        right_depth = cv2.resize(heatmap(depths[0, 1], depth_vmax), (240, 240))
        scene_panel = draw_scene_projection(model.get_scene_mesh(), (420, 300))
        force_mag = [
            sum(float(pad.get("resultant_contact_force_norm", 0.0)) for pad in pads if pad.get("available"))
            for pads in force_records
        ]
        friction_mag = [
            sum(float(pad.get("friction_force_norm_sum", 0.0)) for pad in pads if pad.get("available"))
            for pads in force_records
        ]
        disp_mag = [float(np.mean(row)) for row in marker_disp_mean]
        area_mag = [float(np.sum(row)) for row in contact_area_px]
        plot_force = line_plot(force_mag, (420, 100), "Taccel resultant contact force norm", (20, 95, 200))
        plot_friction = line_plot(friction_mag, (420, 100), "Taccel nodal friction norm sum", (25, 135, 60))
        plot_disp = line_plot(disp_mag, (420, 100), "marker displacement mean (m)", (160, 45, 160))
        plot_area = line_plot(area_mag, (420, 100), "depth contact-area proxy (px)", (190, 110, 20))

        panel = np.full((720, 1280, 3), 242, dtype=np.uint8)
        panel[0:240, 0:240] = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
        panel[0:240, 250:490] = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
        panel[250:490, 0:240] = left_depth
        panel[250:490, 250:490] = right_depth
        panel[500:710, 0:300] = marker_panels[0]
        panel[500:710, 310:610] = marker_panels[1]
        panel[0:300, 650:1070] = scene_panel
        panel[320:420, 650:1070] = plot_force
        panel[430:530, 650:1070] = plot_friction
        panel[540:640, 650:1070] = plot_disp
        panel[650:720, 650:1070] = cv2.resize(plot_area, (420, 70))
        cv2.putText(panel, f"phase00/ref_tactile diag | frame {step:04d} | fps {fps_series[-1]:.2f}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "L/R tactile RGB", (20, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "L/R depth-compression maps", (20, 482), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(panel)

        frame_summaries.append(
            {
                "step": step,
                "q": float(q),
                "fps": fps_series[-1],
                "marker_disp_mean_m": per_pad_mean,
                "marker_disp_max_m": per_pad_max,
                "depth_contact_area_px": per_pad_area,
                "forces": per_pad_force,
                "collisions": collisions,
            }
        )

    writer.release()

    npz_path = output_dir / "dense_tactile_timeseries.npz"
    np.savez_compressed(
        npz_path,
        taccel_rgb=np.asarray(rgb_frames),
        taccel_depth=np.asarray(depth_frames),
        taccel_normal=np.asarray(normal_frames),
        marker_disp_mean=np.asarray(marker_disp_mean),
        marker_disp_max=np.asarray(marker_disp_max),
        contact_area_px=np.asarray(contact_area_px),
        fps=np.asarray(fps_series),
    )

    force_available = any(pad.get("available") for pads in force_records for pad in pads)
    force_nonzero = any(
        float(pad.get("resultant_contact_force_norm", 0.0)) > 1.0e-8
        or float(pad.get("collision_force_norm_sum", 0.0)) > 1.0e-8
        or float(pad.get("friction_force_norm_sum", 0.0)) > 1.0e-8
        for pads in force_records
        for pad in pads
        if pad.get("available")
    )
    collision_nonzero = any(
        bool(record.get("available")) and int(record.get("num_collisions", 0)) > 0 for record in collision_records
    )
    nonblank_rgb = bool(np.asarray(rgb_frames).std() > 1.0e-6)
    nonzero_depth = bool(np.asarray(depth_frames).max() > 1.0e-6)
    nonzero_marker = bool(np.asarray(marker_disp_max).max() > 1.0e-7)
    status = "fail"
    if video_path.exists() and nonblank_rgb and (nonzero_depth or nonzero_marker):
        status = "pass" if force_nonzero else "partial_pass_force_gap"
    summary = {
        "classification": "phase00_reference_tactile_rigid_metal_diagnostic_v1",
        "status": status,
        "run_tag": args.run_tag,
        "not_training_result": True,
        "not_curiosity_success": True,
        "engine": "Taccel official mainline API",
        "object_material": {
            "label": "steel_like_rigid_bar",
            "density": args.object_density,
            "youngs_modulus": args.object_youngs_modulus,
            "friction": args.object_friction,
        },
        "steps": args.steps,
        "dt": args.dt,
        "video": str(video_path),
        "npz": str(npz_path),
        "scene_ply_dir": str(output_dir / "scene_ply"),
        "nonblank_rgb": nonblank_rgb,
        "nonzero_depth": nonzero_depth,
        "nonzero_marker_displacement": nonzero_marker,
        "force_api_available": force_available,
        "mean_fps": float(np.mean(fps_series)),
        "min_fps": float(np.min(fps_series)),
        "max_marker_disp_m": float(np.asarray(marker_disp_max).max()),
        "max_depth_m": float(np.asarray(depth_frames).max()),
        "max_contact_area_px": int(np.asarray(contact_area_px).max()),
        "force_nonzero": force_nonzero,
        "collision_nonzero": collision_nonzero,
        "max_collision_count": max((int(record.get("num_collisions", 0)) for record in collision_records if record.get("available")), default=0),
        "field_provenance": {
            "taccel.rgb": "TaccelModel.render_tactile",
            "taccel.depth": "TaccelModel.render_tactile raycast depth",
            "taccel.normal": "TaccelModel.render_tactile depth_to_normal",
            "taccel.marker_flow": "TaccelModel marker barycentric points in tactile local frame",
            "taccel.contact_force": "ASRModel get_body_*contact/collision/friction_force API when available",
            "taccel.collision_count": "ASRModel cdw.num_collisions after IPC step",
            "candidate.contact_area_px": "depth threshold proxy, not final pressure-area field",
        },
        "known_gaps": [
            "This is an environment diagnostic, not a base grasp policy evaluation.",
            "Pressure maps are represented by Taccel depth/compression maps in this diagnostic; calibrated pressure requires an additional field mapping.",
            "If force_nonzero is false, mechanics force evidence is still incomplete even when tactile video/depth/marker fields are present.",
            "Rigid steel object stress tensor is not exported as a dense field in this first diagnostic.",
            "The scene view is an orthographic mesh projection, not a photorealistic camera render.",
        ],
        "frames": frame_summaries,
    }
    summary_json = output_dir / "diagnostic_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md = report_dir / "diagnostic_report.md"
    report_md.write_text(
        "# Phase 00 Reference Tactile Diagnostic\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{status}`\n"
        f"- video: `{video_path}`\n"
        f"- arrays: `{npz_path}`\n"
        f"- mean_fps: `{summary['mean_fps']:.3f}`\n"
        f"- force_api_available: `{force_available}`\n"
        f"- force_nonzero: `{force_nonzero}`\n"
        f"- collision_nonzero: `{collision_nonzero}`\n"
        f"- max_marker_disp_m: `{summary['max_marker_disp_m']:.6g}`\n"
        f"- max_depth_m: `{summary['max_depth_m']:.6g}`\n"
        "\nThis is environment evidence only, not training and not curiosity success.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
