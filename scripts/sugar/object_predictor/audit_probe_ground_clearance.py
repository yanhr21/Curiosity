"""Saved full-mesh height/pose audit; does not infer missing contact forces."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import load_sugar_outer_box
from sugar_newton.hand.patches import load_hand_mesh


def main(args):
    case = Path(args.case)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((case / 'PROTOCOL.json').read_text())
    path = case / 'episode_0000.npz'
    with np.load(path) as data:
        time = data['timestamp_s']
        pose = data['object_pose_w'].astype(np.float64)
        hands = data['hand_pose_w'].astype(np.float64)
        load = data['normal_load_n'].reshape(-1, 2, 27).sum(2)
        dimensions = data['object_dimensions_m']
    vertices, faces = load_sugar_outer_box()
    vertices = vertices * np.asarray(protocol['scale'], np.float32)
    assert np.array_equal(np.ptp(vertices, axis=0), dimensions[0])
    assert len(vertices) == 15626 and len(time) == 7500
    rotations = Rotation.from_quat(pose[:, 3:])
    matrix = rotations.as_matrix()
    bottom = np.empty(len(time))
    top = np.empty(len(time))
    for start in range(0, len(time), 128):
        end = min(start + 128, len(time))
        height = vertices.astype(np.float64) @ matrix[start:end, 2, :].T + pose[start:end, 2]
        bottom[start:end] = height.min(0)
        top[start:end] = height.max(0)
    # Independent full XYZ transform of predetermined frames verifies axis/order.
    selected = [0, 49, 999, 1699, 2199, 2398, 2499, 4399, 6999, 7499]
    errors = []
    for frame in selected:
        world = rotations[frame].apply(vertices) + pose[frame, :3]
        errors.append(abs(world[:, 2].min() - bottom[frame]))
        errors.append(abs(world[:, 2].max() - top[frame]))
    assert max(errors) < 1e-12
    angle = np.rad2deg((rotations * rotations[49].inv()).magnitude())
    np.savez_compressed(out / 'geometry.npz', timestamp_s=time, normal_load_n=load,
                        object_origin_z_m=pose[:, 2], full_mesh_min_z_m=bottom,
                        full_mesh_max_z_m=top, rotation_from_1s_deg=angle)
    rows = []
    for phase in range(3):
        mask = (time >= phase * 50 + 2) & (time < phase * 50 + 48)
        zero = mask & (load <= .01).all(1)
        rows.append(dict(phase=phase, approach_frames=int(mask.sum()),
                         both_pad_loads_at_most_001n_frames=int(zero.sum()),
                         mesh_min_z_range_m=[float(bottom[mask].min()), float(bottom[mask].max())],
                         origin_z_range_m=[float(pose[mask, 2].min()), float(pose[mask, 2].max())],
                         rotation_from_1s_range_deg=[float(angle[mask].min()), float(angle[mask].max())],
                         low_pad_load_and_mesh_min_z_above_1cm_frames=int((zero & (bottom > .01)).sum())))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time, pose[:, 2], label='Object body origin')
    axes[0].plot(time, bottom, label='Full mesh lowest vertex')
    axes[0].axhline(0., color='black', lw=.7, label='Ground plane')
    axes[0].set_ylabel('World height [m]'); axes[0].legend()
    axes[1].plot(time, angle); axes[1].set_ylabel('Rotation from 1 s [deg]')
    axes[2].plot(time, load[:, 0], label='Left saved pads')
    axes[2].plot(time, load[:, 1], label='Right saved pads')
    axes[2].set_ylabel('Recorded load [N]'); axes[2].legend(); axes[2].set_xlabel('Time [s]')
    fig.suptitle('Saved full-mesh geometry and partial tactile readings; no missing-force reconstruction')
    fig.tight_layout(); fig.savefig(out/'ground_clearance.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    frame = int(np.flatnonzero(np.isclose(time, 44.))[0])
    world = rotations[frame].apply(vertices) + pose[frame, :3]
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection='3d')
    clouds = [world]
    ax.scatter(*world[::8].T, s=2, c='gray', alpha=.3, label='Full object mesh (display subset)')
    for hand, side, color in [(0, 'left', 'tab:blue'), (1, 'right', 'tab:orange')]:
        mesh = load_hand_mesh(side)
        cloud = Rotation.from_quat(hands[frame, hand, 3:]).apply(mesh.vertices) + hands[frame, hand, :3]
        ax.scatter(*cloud[::20].T, s=2, c=color, alpha=.3, label=side+' hand')
        clouds.append(cloud)
    combined = np.concatenate(clouds)
    low, high = combined.min(0), combined.max(0)
    xx, yy = np.meshgrid([low[0], high[0]], [low[1], high[1]])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color='green', alpha=.2)
    center = (low+high)/2; radius = max(np.ptp(combined, axis=0))/2
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center, strict=True): setter(c-radius, c+radius)
    ax.set_box_aspect((1, 1, 1)); ax.set_xlabel('World X [m]'); ax.set_ylabel('World Y [m]'); ax.set_zlabel('World Z [m]')
    ax.set_title(f'Fixed 44 s: mesh min Z={bottom[frame]:.6f} m; pad loads={load[frame]} N')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'fixed_44s_geometry.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    result = dict(case=str(case), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  full_mesh_vertices=len(vertices), full_mesh_faces=len(faces),
                  full_xyz_transform_checks=len(errors), max_height_readback_error_m=max(errors),
                  phases=rows, fixed_frame_44s=dict(frame=frame, mesh_min_z_m=float(bottom[frame]),
                  origin_z_m=float(pose[frame, 2]), rotation_from_1s_deg=float(angle[frame]), pad_loads_n=load[frame].tolist()),
                  interpretation='Exact geometry of the saved state and known triangulated asset relative to z=0. Does not reconstruct solver forces, ground reactions, unassigned skin contacts or causality. Collision/SDF margins remain distinct from geometric clearance.',
                  new_optimizer_updates=0, physics_replays=0, model_calls=0)
    (out/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--case', required=True); p.add_argument('--output', required=True)
    main(p.parse_args())
