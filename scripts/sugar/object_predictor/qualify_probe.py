"""Bounded real-physics check of multi-face acquisition before any training."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main(args):
    if not os.environ.get('SLURM_STEP_ID'): raise RuntimeError('Compute step required')
    import warp as wp
    from .probe_scene import ProbeScene
    from .collect_newton import observation,hand_sites
    wp.init();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    if args.stable_servo and not args.gentle: raise ValueError('Stable servo requires gentle mode')
    frames=7500 if args.stable_servo else (5400 if args.gentle else 1500)
    protocol=dict(frames=frames,dt=.02,mass=args.mass,scale=args.scale,seed=args.seed,
                  full_mesh_colliders=True,free_dynamic_object=True,ground_supported=True,
                  controller_input='clock and previous measured two-hand normal loads only',
                  phases=['opposing X sides','opposing Y sides','top'],
                  force_latch_threshold_n=.5,new_optimizer_updates=0,
                  gentle_force_servo=args.gentle,approach_speed_m_s=.006 if args.gentle else .06,
                  canonical_initial_orientation=args.canonical_init,
                  stable_servo=args.stable_servo,
                  maintained_load_target_n=1. if args.gentle else None,
                  required_both_contact_frames_each_phase=25 if args.gentle else 5,
                  maximum_allowed_load_n=30. if args.gentle else None,
                  scope='Acquisition qualification only; static probing does not make mass observable.')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    scene=ProbeScene(mass=args.mass,scale=args.scale,seed=args.seed,gentle=args.gentle,canonical_init=args.canonical_init,stable_servo=args.stable_servo)
    actual_mass=float(scene.model.body_mass.numpy()[scene.box_body])
    assert np.isclose(actual_mass,args.mass,rtol=1e-6)
    sites,normals=hand_sites(palm_signs=(-1.,1.));rows=[]
    try:
        for frame in range(frames):
            scene.step();obs,audit=observation(scene,sites,normals)
            if audit['overflow']: raise RuntimeError('Tactile surface overflow')
            obj=scene.state_0.body_q.numpy()[scene.box_body].copy()
            row={**obs,'object_pose_w':obj,'object_mass_kg':np.float32(actual_mass),
                 'object_dimensions_m':np.ptp(scene.box_verts,axis=0).astype(np.float32),
                 'object_local_center_m':((scene.box_verts.max(0)+scene.box_verts.min(0))*.5).astype(np.float32),
                 'timestamp_s':np.float64((frame+1)*.02),
                 'validation_probe_phase':np.int32(scene.probe_record['phase']),
                 'validation_probe_touched':scene.probe_record['touched'].copy()}
            if not all(np.isfinite(v).all() for v in row.values()): raise FloatingPointError(frame)
            rows.append(row)
            if frame%50==0: print(json.dumps(dict(frame=frame,phase=scene.probe_record['phase'],loads=obs['normal_load_n'].reshape(2,27).sum(1).tolist(),object_origin=obj[:3].tolist())),flush=True)
    except BaseException as exc:
        if rows: np.savez_compressed(out/'episode_0000.partial.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        (out/'FAILURE.json').write_text(json.dumps(dict(error=repr(exc),frames=len(rows)),indent=2));raise
    arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(out/'episode_0000.npz',**arrays)
    phases=[]
    for phase in range(3):
        mask=arrays['validation_probe_phase']==phase;load=arrays['normal_load_n'][mask].reshape(-1,2,27).sum(2)
        phases.append(dict(phase=phase,contact_frames_per_hand=(load>.01).sum(0).tolist(),both_contact_frames=int((load>.01).all(1).sum()),maximum_load_n=float(load.max())))
    result=dict(frames=len(rows),phases=phases,all_three_directions_both_hands_contact=all(x['both_contact_frames']>=protocol['required_both_contact_frames_each_phase'] for x in phases),new_optimizer_updates=0)
    if args.gentle:
        result['all_peak_loads_within_30n']=all(x['maximum_load_n']<=30. for x in phases)
        result['qualification_passed']=result['all_three_directions_both_hands_contact'] and result['all_peak_loads_within_30n']
    (out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--gentle',action='store_true')
    ap.add_argument('--mass',type=float,default=.7);ap.add_argument('--scale',type=float,nargs=3,default=[1.,1.,1.])
    ap.add_argument('--seed',type=int,default=310017);ap.add_argument('--canonical-init',action='store_true')
    ap.add_argument('--stable-servo',action='store_true')
    main(ap.parse_args())
