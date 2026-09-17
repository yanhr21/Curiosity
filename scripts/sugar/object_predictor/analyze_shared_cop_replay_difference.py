"""Saved cross-recording chronology only; does not rerun either controller."""
import json
import hashlib
from pathlib import Path
import numpy as np

R=Path('experiments/object_predictor_v1/overfit_repair_v1')
ROOT=R/'shared_budget_cop_pilot_v1'


def run():
    cases=[];sources=[]
    fields=['hand_pose_w','object_pose_w','validation_object_velocity_w','normal_load_n',
            'validation_controller_floor_executed_target_pose_w','validation_controller_distance',
            'validation_controller_cop_travel_m','validation_controller_cop_world_m',
            'validation_controller_fit_normal_w','validation_controller_response_gain',
            'validation_controller_cop_servo_velocity_m_s']
    for e in (5012,5000):
        dirs=[R/s/'cases'/f'episode_{e}' for s in ('allocated_floor_cop_pilot_v1','shared_budget_cop_pilot_v1')]
        paths=[d/f'episode_{e}.npz' for d in dirs];sources+=paths
        data=[]
        for p in paths:
            with np.load(p) as z:data.append({k:z[k] for k in z.files})
        a,b=data;protos=[json.loads((d/'PROTOCOL.json').read_text()) for d in dirs]
        config={k:dict(baseline=protos[0][k],candidate=protos[1][k],exact=protos[0][k]==protos[1][k]) for k in
            ('frames','dt','mass_kg','target_load_n','seed','scale','geometry_group','approach_angle_deg','yaw_delta_deg','lift_height_m','lateral_xy','criteria','stability_criteria')}
        if not all(x['exact'] for x in config.values()):raise ValueError('Configuration mismatch')
        stats={}
        for k in fields:
            if k not in a or k not in b:continue
            x,y=a[k],b[k];diff=abs(x.astype(float)-y.astype(float));changed=np.any(diff.reshape(len(x),-1)!=0,axis=1)
            ids=np.flatnonzero(changed)
            stats[k]=dict(first_difference_frame=int(ids[0]) if len(ids) else None,
                first_difference_time_s=float(a['timestamp_s'][ids[0]]) if len(ids) else None,
                first_difference_max_abs=float(diff[ids[0]].max()) if len(ids) else 0,
                max_abs=float(diff.max()),changed_frames=len(ids),
                first_baseline=x[ids[0]].tolist() if len(ids) else None,
                first_candidate=y[ids[0]].tolist() if len(ids) else None)
        # Exact raw-field source chronology, without encoding/voxel averaging.
        surfaces=[]
        for d in dirs:
            p=d/'contact_surface.npz';sources.append(p)
            with np.load(p) as z:surfaces.append({k:z[k] for k in ('offset','position_hand_frame_m','area_m2','normal_pressure_pa','hand','pad')})
        first=None
        for i in range(2400):
            xs=[{k:v[s['offset'][i]:s['offset'][i+1]] for k,v in s.items() if k!='offset'} for s in surfaces]
            unequal=[k for k in xs[0] if not np.array_equal(xs[0][k],xs[1][k])]
            if unequal:
                first=dict(frame=i,time_s=float(a['timestamp_s'][i]),fields=unequal,
                    point_counts=[len(x['hand']) for x in xs],
                    max_abs={k:float(np.max(abs(xs[0][k].astype(float)-xs[1][k].astype(float))))
                             for k in unequal if xs[0][k].shape==xs[1][k].shape and xs[0][k].size})
                break
        cases.append(dict(episode=e,config=config,first_raw_surface_difference=first,fields=stats,
            actual_borrow_count=int(np.any(b['validation_controller_shared_borrow_applied'],axis=1).sum())))
    out=ROOT/'cross_recording_readback';out.mkdir(exist_ok=False)
    report=dict(cases=cases,complete=True,
        limit='Separate physical recordings. Earliest saved chronology can establish action-before-observation or observation-before-action, but cannot uniquely identify floating point/solver/runtime causality. No identical-state controller replay or new physics performed; inactive borrowing differences are not borrowing benefits.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[Path(__file__)]},new_physics_controls=0,new_model_forwards=0)
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{ 'episode':x['episode'],'surface':x['first_raw_surface_difference'],
        'first':{k:{a:b for a,b in v.items() if a.startswith('first_difference')} for k,v in x['fields'].items()}} for x in cases],indent=2))


if __name__=='__main__':run()
