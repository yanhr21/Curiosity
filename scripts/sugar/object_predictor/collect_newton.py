"""Collect real SUGAR/Newton hand-contact observations for object estimation.

Uses the existing reference-driven physical scene. True object state only enters
the target record. The observation excludes reference commands and object-derived
slip/friction utilization. Contact centroids are ideal simulated tactile readings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial.transform import Rotation


def rotation(q):
    return Rotation.from_quat(q).as_matrix()


def hand_sites(palm_signs=(1.,-1.)):
    from sugar_newton.hand.patches import PATCH_SPECS, load_hand_mesh
    points = []
    normals = []
    for side, sign in zip(("left", "right"),palm_signs,strict=True):
        mesh = load_hand_mesh(side)
        v = np.asarray(mesh.vertices)
        for spec in PATCH_SPECS:
            distance = (v[:, 0] - spec.center_x_m)**2 + (v[:, 2] - spec.center_z_m)**2
            near = np.argsort(distance)[:32]
            idx = near[np.argmax(sign * v[near, 1])]
            points.append(v[idx])
            normals.append([0, sign, 0])
    return np.asarray(points, np.float32), np.asarray(normals, np.float32)


def observation(scene, sites, normals):
    bq = scene.state_0.body_q.numpy()
    poses = bq[scene.patch_frame]
    R = rotation(poses[:, 3:])
    geometry = np.concatenate([sites[s*27:(s+1)*27] @ R[s].T + poses[s, :3] for s in range(2)])
    normal_w = np.concatenate([normals[s*27:(s+1)*27] @ R[s].T for s in range(2)])
    field = scene.field.to_numpy()  # Full surface: no subsampling before force integration.
    local_centroid = sites.copy()
    normal_load = np.zeros(54, np.float32)
    shear_local = np.zeros((54, 3), np.float32)
    contact_area = np.zeros(54, np.float32)
    for pad in range(54):
        sel = field['pad'] == pad
        weights = field['area'][sel] * field['pressure'][sel]
        load = weights.sum()
        if load > 1e-6:
            local_centroid[pad] = (field['pos'][sel] * weights[:, None]).sum(0) / load
            normal_load[pad] = load
            contact_area[pad] = field['area'][sel].sum()
            shear_local[pad] = (field['traction_vec'][sel] * field['area'][sel, None]).sum(0)
    contact_w = np.concatenate([local_centroid[s*27:(s+1)*27] @ R[s].T + poses[s, :3] for s in range(2)])
    shear_w = np.concatenate([shear_local[s*27:(s+1)*27] @ R[s].T for s in range(2)])
    # No object-derived channels cross this return boundary.
    obs = dict(hand_pose_w=poses, hand_sites_w=geometry, hand_normals_w=normal_w,
               contact_position_w=contact_w, normal_load_n=normal_load,
               shear_force_w=shear_w, contact_area_m2=contact_area,
               robot_joint_pos=scene.state_0.joint_q.numpy()[:scene.n_joint_dofs].copy())
    audit = dict(surface_count=scene.field.total,
                 assigned_force_n=float(normal_load.sum()),
                 unassigned_faces=int(np.count_nonzero((field['pad'] < 0) | (field['pad'] >= 54))),
                 resolved_hand_normal_n=float(scene.tactile.normal_load.numpy().sum()),
                 overflow=scene.field.total > scene.field.capacity)
    return obs, audit


def collect(args):
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Collection requires a retained compute step')
    import warp as wp
    import newton
    from sugar_newton.validation.g1_carrybox import G1CarryBoxScene, load_clip
    wp.init()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sites, normals = hand_sites()
    scales = [(1., 1., 1.), (.88, 1.08, .94), (1.08, .92, 1.06)]
    masses = [.35, .65, 1.0]
    configurations = [(motion, scale, mass, repeat) for motion in args.motions
                      for scale in scales[:args.scale_count] for mass in masses[:args.mass_count]
                      for repeat in range(args.repeats)]
    protocol = dict(backend='Newton/SolverMuJoCo', seed=args.seed, frames=args.frames,
                    dt=.02, substeps=8, motions=args.motions, configurations=configurations,
                    sensing='ideal simulated tactile contact centroid and integrated field load',
                    excluded=['object velocity', 'object pose', 'mass', 'friction coefficient',
                              'utilization', 'true slip', 'reference command', 'future'],
                    driven_root=True, free_dynamic_object=True,
                    mesh_inertia=True,collision_each_substep=True)
    protocol_path = out / 'COLLECTION_PROTOCOL.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError('Existing collection protocol differs')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    summaries = []
    for index, (motion, scale, mass, repeat) in enumerate(configurations):
        prefix = out / f'episode_{index:04d}'
        summary_path = prefix.with_suffix('.json')
        if summary_path.exists() and prefix.with_suffix('.npz').exists():
            summaries.append(json.loads(summary_path.read_text()))
            continue
        rng = np.random.default_rng(args.seed + index)
        clip = load_clip(f'data_{motion:03d}')
        scene = G1CarryBoxScene(clip, object_scale=scale, object_mass=mass,
                               upper_body_only=False,collide_each_substep=True)
        scene.reset()
        # Independent physical pose perturbation, without changing the hand drive.
        q = scene.state_0.joint_q.numpy()
        offset = rng.uniform(-.015, .015, 3)
        offset[2] = abs(offset[2])
        q[-7:-4] += offset
        delta = Rotation.from_rotvec(rng.uniform(-.08, .08, 3))
        q[-4:] = (delta * Rotation.from_quat(q[-4:])).as_quat()
        scene.state_0.joint_q.assign(q)
        newton.eval_fk(scene.model, scene.state_0.joint_q, scene.state_0.joint_qd, scene.state_0)
        box_body = int(scene.model.shape_body.numpy()[scene.box_shape])
        true_mass = float(scene.model.body_mass.numpy()[box_body])
        if not np.isclose(true_mass, mass, rtol=1e-5):
            raise ValueError(f'Actual simulated mass {true_mass} differs from {mass}')
        rows, audits = [], []
        started = time.monotonic()
        count = min(args.frames, len(clip['joint_pos']))
        try:
            for frame in range(count):
                scene.step(.02, substeps=8)
                obs, audit = observation(scene, sites, normals)
                obj_pose = scene.state_0.body_q.numpy()[box_body].copy()
                row = {**obs, 'object_pose_w':obj_pose,
                       'object_mass_kg':np.float32(true_mass),
                       'object_dimensions_m':np.ptp(scene.box_verts, axis=0).astype(np.float32),
                       'object_local_center_m':((scene.box_verts.max(0)+scene.box_verts.min(0))*.5).astype(np.float32),
                       'timestamp_s':np.float64((frame+1)*.02)}
                if not all(np.isfinite(v).all() for v in row.values()):
                    raise FloatingPointError(f'Nonfinite physical state at {frame}')
                rows.append(row)
                audits.append(audit)
                if audit['overflow']:
                    raise RuntimeError('Contact surface observation overflow')
                if frame % 50 == 0:
                    print(json.dumps(dict(episode=index, frame=frame, contact_patches=int((obs['normal_load_n']>1e-3).sum()),
                                          force_n=audit['assigned_force_n'], elapsed=time.monotonic()-started)), flush=True)
        except BaseException as exc:
            if rows:
                np.savez_compressed(str(prefix)+'.partial.npz', **{k:np.stack([r[k] for r in rows]) for k in rows[0]})
            Path(str(prefix)+'.failure.json').write_text(json.dumps(dict(error=repr(exc),frames=len(rows)),indent=2))
            raise
        arrays = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(prefix.with_suffix('.npz'), **arrays)
        summary = dict(episode=index, motion=motion, scale=scale, mass_kg=true_mass, repeat=repeat,
                       frames=count, seed=args.seed+index, seconds=time.monotonic()-started,
                       contact_frames=int((arrays['normal_load_n'].sum(1)>1e-3).sum()),
                       max_contact_patches=int((arrays['normal_load_n']>1e-3).sum(1).max()),
                       max_assigned_force_n=max(a['assigned_force_n'] for a in audits),
                       max_unassigned_faces=max(a['unassigned_faces'] for a in audits),
                       max_total_resolved_force_n=max(a['resolved_hand_normal_n'] for a in audits),
                       source=dict(job=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'],host=socket.gethostname()))
        summary_path.write_text(json.dumps(summary,indent=2))
        summaries.append(summary)
        print('EPISODE_COMPLETE '+json.dumps(summary), flush=True)
        del scene
    (out/'COLLECTION_RESULT.json').write_text(json.dumps(dict(episodes=summaries),indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--motions',type=int,nargs='+',default=[45,90,96])
    ap.add_argument('--frames',type=int,default=400)
    ap.add_argument('--scale-count',type=int,default=3)
    ap.add_argument('--mass-count',type=int,default=3)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--seed',type=int,default=310015)
    collect(ap.parse_args())
