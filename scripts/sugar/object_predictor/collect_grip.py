"""Bounded exact-physics grip qualification with saved force/clearance evidence."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Retained compute step required')
    import warp as wp
    from .grip_scene import GripScene
    from .collect_newton import observation,hand_sites
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(frames=1200,dt=.02,mass_kg=args.mass,target_load_n=args.load,seed=args.seed,scale=[1.,1.,1.],
                  phases={'approach_and_grip':[0,16],'lift18cm_translate4cm':[16,20],'hold':[20,24]},
                  controller_inputs=['clock','previous measured palmar normal load of each hand'],
                  original_full_mesh_and_physics=True,model_updates=0,
                  criteria={'final_hold_bilateral_pad_contact_fraction_min':.8,'final_hold_clearance_above10cm_fraction_min':.8,
                            'final_hold_median_rise_min_m':.15,'all_recorded_peak_hand_load_max_n':100.,'final_hold_load_target_relative_error_max':.25},
                  source_sha256={n:hashlib.sha256(Path('scripts/sugar/object_predictor',n).read_bytes()).hexdigest()
                                 for n in ('grip_scene.py','probe_scene.py','support_scene.py','collect_newton.py','geometry.py')})
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    wp.init();scene=GripScene(mass=args.mass,scale=(1.,1.,1.),seed=args.seed,target_load_n=args.load)
    mass=float(scene.model.body_mass.numpy()[scene.box_body]);assert np.isclose(mass,args.mass,rtol=1e-6)
    sites,normals=hand_sites(palm_signs=(-1.,1.));rows=[];started=time.monotonic()
    try:
        for frame in range(1200):
            scene.step();obs,audit=observation(scene,sites,normals)
            if audit['overflow']:raise RuntimeError('Tactile surface overflow')
            pose=scene.state_0.body_q.numpy()[scene.box_body].copy()
            vertex_z=Rotation.from_quat(pose[3:]).apply(scene.box_verts)[:,2]+pose[2]
            normal=scene.tactile.normal_vec.numpy();friction=scene.tactile.friction_vec.numpy()
            row={**obs,'object_pose_w':pose,'object_mass_kg':np.float32(mass),
                 'object_dimensions_m':np.ptp(scene.box_verts,axis=0).astype(np.float32),
                 'object_local_center_m':((scene.box_verts.max(0)+scene.box_verts.min(0))*.5).astype(np.float32),
                 'timestamp_s':np.float64((frame+1)*.02),'validation_full_mesh_min_z_m':np.float64(vertex_z.min()),
                 'validation_hand_normal_vec_w':normal,'validation_hand_friction_vec_w':friction,
                 'validation_resolved_normal_n':scene.tactile.normal_load.numpy(),
                 'validation_unassigned_faces':np.int32(audit['unassigned_faces']),
                 'validation_controller_phase':np.int32(scene.probe_record['phase']),
                 'validation_controller_distance_m':scene.probe_record['distance'].copy()}
            if not all(np.isfinite(v).all() for v in row.values()):raise FloatingPointError(frame)
            rows.append(row)
            if frame%50==0:print(json.dumps(dict(frame=frame,time_s=(frame+1)*.02,loads=obs['normal_load_n'].reshape(2,27).sum(1).tolist(),
                clearance_m=float(vertex_z.min()),support_z_n=float(-(normal+friction).sum(0)[2]),elapsed=time.monotonic()-started)),flush=True)
    except BaseException as exc:
        if rows:np.savez_compressed(out/'episode_4000.partial.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        (out/'FAILURE.json').write_text(json.dumps(dict(error=repr(exc),frames=len(rows)),indent=2));raise
    a={k:np.stack([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(out/'episode_4000.npz',**a)
    hold=a['timestamp_s']>=20.;baseline=(a['timestamp_s']>=1.)&(a['timestamp_s']<=2.)
    loads=a['normal_load_n'].reshape(-1,2,27).sum(2)
    center=a['object_pose_w'][:,:3]+Rotation.from_quat(a['object_pose_w'][:,3:]).apply(a['object_local_center_m'])
    rise=float(np.median(center[hold,2])-np.median(center[baseline,2]))
    values=dict(bilateral_fraction=float((loads[hold]>.01).all(1).mean()),clearance_fraction=float((a['validation_full_mesh_min_z_m'][hold]>.1).mean()),
                median_rise_m=rise,peak_hand_load_n=float(loads.max()),hold_mean_hand_load_n=loads[hold].mean(0).tolist(),
                hold_mean_normal_n=a['validation_resolved_normal_n'][hold].mean(0).tolist(),
                hold_mean_support_z_n=float(-(a['validation_hand_normal_vec_w'][hold]+a['validation_hand_friction_vec_w'][hold]).sum(1)[:,2].mean()))
    checks=dict(bilateral=values['bilateral_fraction']>=.8,clearance=values['clearance_fraction']>=.8,rise=rise>=.15,peak=values['peak_hand_load_n']<=100.,
                load_target=bool(np.max(np.abs(loads[hold].mean(0)/args.load-1))<=.25))
    report=dict(episode=4000,split='test',acquisition_group='grip',sampling_stride=25,frames=1200,mass_kg=mass,scale=[1,1,1],seed=args.seed,
                passed=all(checks.values()),checks=checks,values=values,physics_steps=1200,optimizer_updates=0,
                scope='Prescribed dual-hand sensing fixture, not a whole-body policy. Validation-only forces/pose excluded from predictor.')
    (out/'episode_4000.json').write_text(json.dumps(report,indent=2));(out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--mass',type=float,default=.5);ap.add_argument('--load',type=float,default=12.);ap.add_argument('--seed',type=int,default=310017);main(ap.parse_args())
