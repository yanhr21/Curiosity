"""Saved-only fixed-three CoP admission/added-path accounting; no simulation."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('experiments/object_predictor_v1/overfit_repair_v1/prelift_cop_alignment_pilot_v1')


def runs(mask, times):
    indices = np.flatnonzero(mask)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    groups = [g for g in groups if len(g)]
    return dict(frames=len(indices), first_s=float(times[indices[0]]) if len(indices) else None,
                longest_s=max((len(g)*.02 for g in groups), default=0.),
                first_1s_end_s=next((float(times[g[49]]) for g in groups if len(g)>=50), None))


def case(config):
    episode = config['episode']; folder = ROOT/'cases'/f'episode_{episode}'
    with np.load(folder/f'episode_{episode}.npz') as z: a={k:z[k] for k in z.files}
    with np.load(folder/'contact_surface.npz') as z: s={k:z[k] for k in z.files}
    t=a['timestamp_s'];v=lambda key:a['validation_controller_'+key]
    measured=np.zeros((len(t),2))
    for i in range(1,len(t)):
        start,end=s['offset'][i-1:i+1]
        for side in (0,1):
            take=(s['pad'][start:end]//27)==side
            measured[i,side]=np.sum(s['area_m2'][start:end][take]*s['normal_pressure_pa'][start:end][take])
    fit=v('fit_valid').all(1);orientation=(v('alignment_error_deg')<=5).all(1)
    force=(abs(measured/config['load']-1)<=.25).all(1)
    cop=v('cop_valid').all(1)&(v('cop_line_angle_deg')<=5)
    masks=dict(plane=fit,palm_orientation=orientation,force_band=force,cop_gate=cop,
               original_ready=fit&orientation&force,all_ready=fit&orientation&force&cop)
    accounting=[]
    for side in (0,1):
        vel=v('cop_servo_velocity_m_s')[:,side];step=vel*.02;norm=np.linalg.norm(step,axis=1)
        arc=float(norm.sum());net=step.sum(0);active=norm>1e-12
        pairs=active[1:]&active[:-1]
        directions=step[active]/norm[active,None]
        cos=np.sum(directions[1:]*directions[:-1],axis=1)
        turn=np.degrees(np.arccos(np.clip(cos,-1,1)))
        # Adjacent active samples can straddle a pause; report explicitly.
        accounting.append(dict(arc_m=arc,net_added_translation_m=net.tolist(),net_norm_m=float(np.linalg.norm(net)),
            net_over_arc=float(np.linalg.norm(net)/arc) if arc else None,
            active_duration_s=float(active.sum()*.02),first_active_s=runs(active,t)['first_s'],
            budget_first_exhausted_s=runs(v('cop_travel_m')[:,side]>=.10-1e-12,t)['first_s'],
            consecutive_active_direction_reversals=int((cos<0).sum()),
            consecutive_active_turn_deg_quantiles=np.quantile(turn,[.5,.95,1]).tolist() if len(turn) else [],
            contiguous_clock_direction_reversals=int((np.sum(step[1:]*step[:-1],axis=1)[pairs]<0).sum())))
    rows=[]
    for clock in (12,16,20,24,28,30,32,33,34,35,36,38,40,42,44,48):
        i=int(np.argmin(abs(t-clock)))
        rows.append(dict(time_s=float(t[i]),previous_observation_s=float(t[i-1]),
            input_load_n=measured[i].tolist(),angle_deg=float(v('cop_line_angle_deg')[i]),
            tangent_m=v('cop_tangent_error_m')[i].tolist(),servo_m_s=v('cop_servo_velocity_m_s')[i].tolist(),
            gain=v('response_gain_m_per_ns')[i].tolist(),upper=v('response_upper_n_m')[i].tolist(),
            samples=v('response_samples')[i].tolist(),distance_m=v('distance_m')[i].tolist(),
            ready_s=float(v('ready_seconds')[i]),object_quat=a['object_pose_w'][i,3:].tolist()))
    windows={}
    for name,window in [('full',t>0),('admission_24_to_40',(t>=24)&(t<=40)),('before_deadline',t<=40)]:
        windows[name]={key:runs(mask&window,t) for key,mask in masks.items()}
        windows[name]['exclusive_blocker_frames']={
            key:int((window&~mask&np.logical_and.reduce([m for k,m in masks.items() if k in ('plane','palm_orientation','force_band','cop_gate') and k!=key])).sum())
            for key,mask in masks.items() if key in ('plane','palm_orientation','force_band','cop_gate')}
    return dict(episode=episode,target_n=config['load'],lift_start_s=float(v('lift_start_s')[-1]),
                path=accounting,readiness=windows,clocks=rows,
                sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/f'episode_{episode}.npz',folder/'contact_surface.npz',folder/'RESULT.json']})


def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());result=json.loads((ROOT/'RESULT.json').read_text())
    assert result['complete'] and [c['episode'] for c in result['cases']]==[5014,5012,5000]
    output=ROOT/'ADMISSION_BUDGET_READBACK.json'
    if output.exists():raise FileExistsError(output)
    report=dict(cases=[case(c) for c in protocol['configurations']],physics_controls=0,model_forwards=0,
                method='Loads integrated from previous observed pressure times area. Net translation sums ONLY the added CoP servo; excludes original closure/alignment and physical response. Reversal counts compare adjacent nonzero servo directions and are not a mechanical stability certificate.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
