"""Physical qualification/collection for the exact-mesh tactile support fixture."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Requires compute step')
    import warp as wp
    from .support_scene import SupportScene
    from .collect_newton import hand_sites,observation
    wp.init()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sites,normals=hand_sites(palm_signs=(-1.,1.))
    summaries=[]
    rng=np.random.default_rng(args.seed)
    if args.dataset_episodes:
        if args.mass_only_dataset:
            configurations=[dict(mass=float(rng.uniform(.25,1.2)),scale=[1.,1.,1.],seed=args.geometry_seed)
                            for i in range(args.dataset_episodes)]
        else:
            configurations=[dict(mass=float(rng.uniform(.25,1.2)),scale=rng.uniform(.85,1.15,3).tolist(),seed=args.seed+i)
                            for i in range(args.dataset_episodes)]
    else:
        configurations=[dict(mass=mass,scale=[1.,1.,1.],seed=args.seed) for mass in args.masses]
    (out/'PROTOCOL.json').write_text(json.dumps(dict(configurations=configurations,
        frames=args.frames,scene='exact_mesh_support_fixture',full_body=False,
        palmar_surface_signs=[-1.,1.],initial_pose='widest face horizontal, COM near CAD support-plane centers',
        hand_layout='largest palmar CAD hull facet oriented upward; original full hand mesh collider retained; support centers 0.14m apart',
        object_collision='SUGAR solid_outer_shell_only, complete positive component, no remeshing',
        intended_use='independent perception pretraining, not policy success',
        mass_only_calibration_diagnostic=args.mass_only_dataset,
        geometry_seed_if_fixed=args.geometry_seed if args.mass_only_dataset else None,
        evaluation_only=args.evaluation_only,episode_offset=args.episode_offset,
        nominal_gravity_m_s2=9.81),indent=2))
    for i,configuration in enumerate(configurations):
        episode_id=i+args.episode_offset
        mass=configuration['mass']
        prefix=out/f'episode_{episode_id:04d}'
        if prefix.with_suffix('.npz').exists():
            raise FileExistsError(prefix)
        scene=SupportScene(**configuration)
        box_body=int(scene.model.shape_body.numpy()[scene.box_shape])
        actual_mass=float(scene.model.body_mass.numpy()[box_body])
        rows=[];audits=[];started=time.monotonic()
        for frame in range(args.frames):
            scene.step()
            obs,audit=observation(scene,sites,normals)
            if audit['overflow']:
                raise RuntimeError('Contact surface observation overflow')
            pose=scene.state_0.body_q.numpy()[box_body].copy()
            # Resolved net hand load is a validation field, excluded from inputs.
            hand_force=(scene.tactile.normal_vec.numpy()+scene.tactile.friction_vec.numpy()).sum(0)
            audit['object_support_force_z_n']=float(-hand_force[2])
            row={**obs,'object_pose_w':pose,'object_mass_kg':np.float32(actual_mass),
                 'object_dimensions_m':np.ptp(scene.box_verts,axis=0).astype(np.float32),
                 'object_local_center_m':((scene.box_verts.max(0)+scene.box_verts.min(0))*.5).astype(np.float32),
                 'timestamp_s':np.float64((frame+1)*.02)}
            if not all(np.isfinite(v).all() for v in row.values()):
                raise FloatingPointError(f'Nonfinite physical record {frame}')
            rows.append(row);audits.append(audit)
            if frame%25==0:
                print(json.dumps(dict(episode=i,frame=frame,mass=actual_mass,
                                      assigned_force_n=audit['assigned_force_n'],support_z_n=audit['object_support_force_z_n'],
                                      contact_patches=int((obs['normal_load_n']>1e-3).sum()),elapsed=time.monotonic()-started)),flush=True)
        arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
        # Validation-only fields never appear in data.OBSERVATION_KEYS.
        arrays['validation_support_force_z_n']=np.asarray([a['object_support_force_z_n'] for a in audits])
        arrays['validation_resolved_normal_n']=np.asarray([a['resolved_hand_normal_n'] for a in audits])
        arrays['validation_unassigned_faces']=np.asarray([a['unassigned_faces'] for a in audits])
        np.savez_compressed(prefix.with_suffix('.npz'),**arrays)
        # Last half-second before the hands start accelerating: actual support
        # should balance measured mass. Failure is evidence, not a new calibration.
        rest=np.asarray([a['object_support_force_z_n'] for a in audits])[25:49]
        balance=float(np.median(rest)/(actual_mass*9.81))
        normal_ratio=float(np.median(arrays['validation_resolved_normal_n'][25:49])/(actual_mass*9.81))
        assigned_fraction=float(np.median(arrays['normal_load_n'][25:49].sum(1)/np.maximum(arrays['validation_resolved_normal_n'][25:49],1e-8)))
        summary=dict(episode=episode_id,acquisition='exact_sugar_mesh_support_fixture',mass_kg=actual_mass,
                     frames=args.frames,seed=configuration['seed'],scale=configuration['scale'],seconds=time.monotonic()-started,
                     contact_frames=int((arrays['normal_load_n'].sum(1)>1e-3).sum()),
                     support_to_weight_ratio=balance,
                     support_balance_pass=bool(.8<=balance<=1.2),
                     normal_load_to_weight_ratio=normal_ratio,
                     normal_balance_pass=bool(.8<=normal_ratio<=1.5),
                     assigned_normal_fraction_at_rest=assigned_fraction,
                     object_min_z_m=float(arrays['object_pose_w'][:,2].min()),
                     max_assigned_load_n=float(arrays['normal_load_n'].sum(1).max()),
                     is_whole_body_control=False,optimizer_updates=0)
        if args.evaluation_only:
            summary['split']='test'
        elif args.dataset_episodes:
            boundary_train=int(args.dataset_episodes*2/3)
            boundary_val=int(args.dataset_episodes*5/6)
            summary['split']='train' if i<boundary_train else 'val' if i<boundary_val else 'test'
        prefix.with_suffix('.json').write_text(json.dumps(summary,indent=2))
        summaries.append(summary)
        print('SUPPORT_EPISODE_COMPLETE '+json.dumps(summary),flush=True)
        del scene
    (out/'RESULT.json').write_text(json.dumps({'episodes':summaries,'all_support_balance_pass':all(s['support_balance_pass'] for s in summaries)},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--masses',type=float,nargs='+',default=[.35,.7])
    ap.add_argument('--frames',type=int,default=200);ap.add_argument('--seed',type=int,default=310017)
    ap.add_argument('--dataset-episodes',type=int,default=0)
    ap.add_argument('--mass-only-dataset',action='store_true')
    ap.add_argument('--geometry-seed',type=int,default=310017)
    ap.add_argument('--evaluation-only',action='store_true')
    ap.add_argument('--episode-offset',type=int,default=0)
    main(ap.parse_args())
