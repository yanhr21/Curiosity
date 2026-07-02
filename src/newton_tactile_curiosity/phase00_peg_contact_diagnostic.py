"""Official Taccel peg-style two-sensor contact diagnostic for Phase 00."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp
from scipy.spatial.transform import Rotation as R, Slerp

from warp_ipc.ipc_integrator import IPCIntegrator
from warp_ipc.sim_model import ASRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=50.0)
    return parser.parse_args()


def interpolate_keyframes(keyframes, time_step):
    p_traj = []
    q_traj = []
    for i in range(len(keyframes) - 1):
        t0, p0, q0 = keyframes[i]
        t1, p1, q1 = keyframes[i + 1]
        step0 = int(round(t0 / time_step))
        step1 = int(round(t1 / time_step))
        if i > 0:
            p_traj = p_traj[:-1]
            q_traj = q_traj[:-1]
        p_traj += np.linspace(p0, p1, int(step1 - step0 + 1)).tolist()
        slerp = Slerp([t0, t1], R.from_quat([q0, q1]))
        q_traj += slerp(np.linspace(t0, t1, int(step1 - step0 + 1))).as_quat().tolist()
    return np.array(p_traj), np.array(q_traj)


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
    cv2.polylines(canvas, [np.stack([xs, ys], axis=1).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{arr[-1]:.4g}", (width - 92, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return canvas


def project_scene(mesh: tm.Trimesh, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pts = verts[:, [0, 2]]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-6)
    lo = lo - span * 0.12
    span = span * 1.24
    proj = (pts - lo) / span * np.array([width - 24, height - 28]) + np.array([12, 14])
    proj[:, 1] = height - proj[:, 1]
    proj = proj.astype(np.int32)
    step = max(1, len(faces) // 6000)
    for tri in faces[::step]:
        cv2.polylines(canvas, [proj[tri].reshape(-1, 1, 2)], True, (80, 110, 130), 1, cv2.LINE_AA)
    cv2.putText(canvas, "official peg scene x-z", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def deformation_panel(points: np.ndarray, reference: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    width, height = size
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    disp = points - reference
    mag = np.linalg.norm(disp, axis=1)
    xy = reference[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1.0e-6)
    proj = (xy - lo) / span * np.array([width - 24, height - 30]) + np.array([12, 20])
    proj[:, 1] = height - proj[:, 1]
    vmax = max(float(mag.max()), 1.0e-8)
    colors = cv2.applyColorMap((np.clip(mag / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    for p, c in zip(proj.astype(np.int32), colors.reshape(-1, 3)):
        cv2.circle(canvas, tuple(p), 2, tuple(int(x) for x in c), -1, cv2.LINE_AA)
    cv2.putText(canvas, "soft tactile deformation", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas, float(mag.mean()), float(mag.max())


def force_norm(model: ASRModel, handle, dt: float) -> tuple[float, float, float]:
    try:
        model.update_contact_force(dt)
        contact = model.get_body_resultant_contact_force(handle, dt).detach().cpu().numpy()
        coll = model.get_body_nodal_collision_force(handle, dt).detach().cpu().numpy()
        fric = model.get_body_nodal_friction_force(handle, dt).detach().cpu().numpy()
        return float(np.linalg.norm(contact)), float(np.linalg.norm(coll.reshape(-1, 3), axis=1).sum()), float(np.linalg.norm(fric.reshape(-1, 3), axis=1).sum())
    except Exception:
        return -1.0, -1.0, -1.0


def collision_count(model: ASRModel) -> int:
    try:
        return int(model.cdw.num_collisions.numpy()[0])
    except Exception:
        return -1


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    visual_dir = Path(args.visual_dir)
    report_dir = Path(args.report_dir)
    for path in (output_dir, visual_dir, report_dir, output_dir / "scene_ply"):
        path.mkdir(parents=True, exist_ok=True)

    wp.config.kernel_cache_dir = os.environ.get("TACCEL_PTX_DIR", str(output_dir / "ptx"))
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = int(os.environ.get("TACCEL_PTX_ARCH", "86"))
    wp.init()
    dt = 1.0 / 50.0

    model = ASRModel(num_envs=1)
    model.set_kinematic_stiffness(1.0e5)
    model.dhat = 1.0e-4
    model.tol = 1.0e-3
    model.epsv = 1.0e-2
    model.gravity = wp.vec3d([0.0, 0.0, -9.81])

    peg = tm.load("assets/objects/peg/peg.stl")
    hole = tm.load("assets/objects/peg/hole.stl")
    peg_handle = model.add_affine_body(peg.vertices, peg.faces.astype(np.int32), 1.0e3, 1.0e9, 0.2, mass_xi=0.0205 / len(peg.vertices), env_id=0)
    hole_handle = model.add_affine_body(hole.vertices, hole.faces.astype(np.int32), 1.0e3, 1.0e9, 0.2, 0.0239 / len(peg.vertices), env_id=0)

    sensor = pv.read("assets/robots/sensor.vtk")
    stick_idx = [1, 3, 5, 7, 12, 18, 24, 30, 36, 39, 42, 46, 49, 53, 56, 59, 63, 66, 69, 73, 76, 77, 82, 85, 86, 88, 90, 92, 94, 95, 96, 97, 99, 102, 103, 104, 105, 106, 107, 108, 115, 117, 118, 119, 120, 121, 122, 125, 126, 127, 128, 129, 131, 134, 135, 138]
    stick_mask = np.zeros(sensor.n_points, dtype=np.int32)
    stick_mask[stick_idx] = 1
    sensor_handles = [model.add_soft_vol_body(sensor, density=1.0e3, E=1.0e5, nu=0.4, mu=1.0, env_id=0) for _ in range(2)]
    model.init()
    model.enable_affine_kinematic_constraint(hole_handle)
    model.set_affine_state(hole_handle, np.eye(3), np.zeros(3))
    model.set_affine_kinematic_target(hole_handle, np.eye(3), np.zeros(3))
    model.set_affine_state(peg_handle, np.eye(3), np.array([0.0, 0.005, 0.021]))
    for handle in sensor_handles:
        model.set_soft_kinematic_constraint(handle, stick_mask)
    model.finalize()

    sensor_1_keyframes = [
        (0.0, [-0.0171, 0.005, 0.071], [0.70710677, 0.0, 0.70710677, 0.0]),
        (0.1, [-0.0165, 0.005, 0.071], [0.70710677, 0.0, 0.70710677, 0.0]),
        (2.0, [-0.0165, -0.006, 0.066], [0.70710677, 0.0, 0.70710677, 0.0]),
        (4.0, [-0.0165, -0.006, 0.05], [0.70710677, 0.0, 0.70710677, 0.0]),
    ]
    sensor_2_keyframes = [
        (0.0, [0.0171, 0.005, 0.071], [0.0, -0.70710677, 0.0, 0.70710677]),
        (0.1, [0.0165, 0.005, 0.071], [0.0, -0.70710677, 0.0, 0.70710677]),
        (2.0, [0.0165, -0.006, 0.066], [0.0, -0.70710677, 0.0, 0.70710677]),
        (4.0, [0.0165, -0.006, 0.05], [0.0, -0.70710677, 0.0, 0.70710677]),
    ]
    p1, q1 = interpolate_keyframes(sensor_1_keyframes, dt)
    p2, q2 = interpolate_keyframes(sensor_2_keyframes, dt)
    sensor_points = np.array(sensor.points)

    integrator = IPCIntegrator(device=args.device)
    integrator.use_hard_kinematic_constraint = False
    integrator.use_cpu = False
    integrator.max_cg_iter = 40
    integrator.cg_rel_tol = 1.0e-5
    integrator.max_newton_iter = 20

    video_path = visual_dir / "peg_contact_diag.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720))
    reference = None
    force_series = []
    collision_series = []
    deform_series = []
    frame_summaries = []
    steps = min(args.steps, len(p1), len(p2))
    for step in range(steps):
        t1 = np.eye(4)
        t1[:3, 3] = p1[step]
        t1[:3, :3] = R.from_quat(q1[step]).as_matrix()
        t2 = np.eye(4)
        t2[:3, 3] = p2[step]
        t2[:3, :3] = R.from_quat(q2[step]).as_matrix()
        targets = [sensor_points @ t1[:3, :3].T + t1[:3, 3], sensor_points @ t2[:3, :3].T + t2[:3, 3]]
        for i, handle in enumerate(sensor_handles):
            if step == 0:
                model.set_soft_state(handle, targets[i])
            model.set_soft_kinematic_target(handle, targets[i])
        if step == 0:
            model.apply_set_state()
        model.set_affine_kinematic_target(hole_handle, np.eye(3), np.zeros(3))
        integrator.simulate(model, dt=dt)
        current = [np.asarray(model.get_element_by_handle(handle, False)[0].cpu().numpy()) for handle in sensor_handles]
        if reference is None:
            reference = [arr.copy() for arr in current]
        panels = []
        deform_means = []
        deform_maxes = []
        for arr, ref in zip(current, reference):
            panel, mean_d, max_d = deformation_panel(arr, ref, (300, 250))
            panels.append(panel)
            deform_means.append(mean_d)
            deform_maxes.append(max_d)
        contact_norms = [force_norm(model, handle, dt) for handle in sensor_handles]
        force_sum = float(sum(max(0.0, item[0]) for item in contact_norms))
        collision = collision_count(model)
        force_series.append(force_sum)
        collision_series.append(max(0, collision))
        deform_series.append(float(np.mean(deform_means)))
        if step % 20 == 0 or step == steps - 1:
            model.write_scene(str(output_dir / "scene_ply" / f"scene_{step:04d}.ply"))
        scene = project_scene(model.get_scene_mesh(), (520, 300))
        panel = np.full((720, 1280, 3), 242, dtype=np.uint8)
        panel[0:250, 0:300] = panels[0]
        panel[0:250, 310:610] = panels[1]
        panel[0:300, 650:1170] = scene
        panel[320:420, 650:1070] = line_plot(force_series, (420, 100), "resultant contact force norm", (20, 95, 200))
        panel[430:530, 650:1070] = line_plot(collision_series, (420, 100), "IPC collision count", (190, 90, 20))
        panel[540:640, 650:1070] = line_plot(deform_series, (420, 100), "mean soft deformation (m)", (160, 45, 160))
        cv2.putText(panel, f"official peg contact diagnostic | frame {step:04d}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(panel)
        frame_summaries.append(
            {
                "step": step,
                "collision_count": collision,
                "sensor_forces": contact_norms,
                "deform_mean_m": deform_means,
                "deform_max_m": deform_maxes,
            }
        )
    writer.release()
    summary = {
        "classification": "phase00_official_peg_contact_diagnostic_v1",
        "run_tag": args.run_tag,
        "status": "pass" if max(collision_series or [0]) > 0 or max(force_series or [0]) > 0 else "partial_no_contact_force",
        "not_training_result": True,
        "not_curiosity_success": True,
        "video": str(video_path),
        "scene_ply_dir": str(output_dir / "scene_ply"),
        "max_collision_count": int(max(collision_series or [0])),
        "max_force_norm": float(max(force_series or [0.0])),
        "max_deform_mean_m": float(max(deform_series or [0.0])),
        "frames": frame_summaries,
    }
    (output_dir / "peg_contact_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "peg_contact_report.md").write_text(
        "# Phase 00 Official Peg Contact Diagnostic\n\n"
        f"- status: `{summary['status']}`\n"
        f"- max_collision_count: `{summary['max_collision_count']}`\n"
        f"- max_force_norm: `{summary['max_force_norm']:.6g}`\n"
        f"- max_deform_mean_m: `{summary['max_deform_mean_m']:.6g}`\n"
        f"- video: `{video_path}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] == "partial_no_contact_force":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
