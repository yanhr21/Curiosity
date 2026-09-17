"""Replay saved observed gain estimator and describe selected secant motion."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .response_gain import ObservedForceGain
from .analyze_cop_admission_budget import ROOT, runs


def case(config):
    folder=ROOT/'cases'/f"episode_{config['episode']}"
    with np.load(folder/f"episode_{config['episode']}.npz") as z:a={k:z[k] for k in z.files}
    with np.load(folder/'contact_surface.npz') as z:s={k:z[k] for k in z.files}
    t=a['timestamp_s'];v=lambda k:a['validation_controller_'+k]
    loads=np.zeros((len(t),2));distance=np.zeros((len(t),2));distance[1:]=v('distance_m')[:-1]
    for i in range(1,len(t)):
        start,end=s['offset'][i-1:i+1]
        for side in (0,1):
            mask=s['pad'][start:end]//27==side
            loads[i,side]=(s['area_m2'][start:end][mask]*s['normal_pressure_pa'][start:end][mask]).sum()
    model=ObservedForceGain();actual=np.zeros((len(t),2,3));slopes=[[],[]]
    for i in range(len(t)):
        for side in (0,1):
            actual[i,side]=model.update(side,float(t[i]),float(loads[i,side]),float(distance[i,side]),.02,bool(v('fit_valid')[i,side]))
            if i>=5:
                dx=distance[i,side]-distance[i-5,side];df=loads[i,side]-loads[i-5,side]
                if loads[i,side]>=.2 and loads[i-5,side]>=.2 and dx>1e-6 and df>.005:
                    slopes[side].append((i,df/dx))
    saved=np.stack([v('response_gain_m_per_ns'),v('response_upper_n_m'),v('response_samples')],axis=-1)
    diff=np.max(abs(actual-saved),axis=(0,1))
    if not np.allclose(actual,saved,rtol=1e-10,atol=1e-9):raise ValueError(('Estimator replay mismatch',config['episode'],diff))
    records=[]
    for side in (0,1):
        plateau=v('response_upper_n_m')[1199,side]
        candidates=[(i,slope) for i,slope in slopes[side] if np.isclose(slope,plateau,rtol=1e-10,atol=1e-9)]
        selected=[('maximum_new_secant',max(slopes[side],key=lambda x:x[1]))] if slopes[side] else []
        if candidates:selected.append(('secant_for_24s_retained_upper',candidates[0]))
        for label,(i,slope) in selected:
            j=i-5
            # Input observations j-1 and i-1 bracket commands j..i-1.
            rot=Rotation.from_quat(a['hand_pose_w'][[j-1,i-1],side,3:])
            servo=v('cop_servo_velocity_m_s')[j:i,side]*.02
            records.append(dict(side=side,label=label,old_clock_s=float(t[j]),new_clock_s=float(t[i]),
                old_load_n=float(loads[j,side]),new_load_n=float(loads[i,side]),
                delta_load_n=float(loads[i,side]-loads[j,side]),
                delta_closure_m=float(distance[i,side]-distance[j,side]),secant_n_m=float(slope),
                cop_added_translation_m=servo.sum(0).tolist(),cop_added_path_m=float(np.linalg.norm(servo,axis=1).sum()),
                hand_observed_rotation_deg=float(np.degrees((rot[1]*rot[0].inv()).magnitude())),
                parent_alignment_requested_frames=int(v('actual_alignment_active')[j:i,side].sum()),
                applied_gain=float(v('response_gain_m_per_ns')[i,side]),
                old_hand_position_m=a['hand_pose_w'][j-1,side,:3].tolist(),new_hand_position_m=a['hand_pose_w'][i-1,side,:3].tolist()))
    stale=[]
    for side in (0,1):
        upper=v('response_upper_n_m')[:,side];samples=v('response_samples')[:,side]
        valid=(samples<5)&(upper>0)
        stale.append(dict(side=side,nonzero_retained_with_fewer_than5_slopes=runs(valid,t),
            overload_while_stale_frames=int((valid&(loads[:,side]>15)&(t<=40)).sum()),
            maximum_upper=float(upper.max()),minimum_applied_gain=float(v('response_gain_m_per_ns')[:,side].min())))
    return dict(episode=config['episode'],max_replay_difference_gain_upper_samples=diff.tolist(),selected_secants=records,stale=stale)


def main():
    output=ROOT/'RESPONSE_SECANT_READBACK.json'
    if output.exists():raise FileExistsError(output)
    config=json.loads((ROOT/'PROTOCOL.json').read_text())['configurations']
    result=dict(cases=[case(c) for c in config],physics_controls=0,model_forwards=0,
        limits='Saved observation replay, not rerunning mechanics. Non-normal motion in the same secant interval invalidates a uniquely normal stiffness interpretation, but does not identify which displacement caused the load difference. These are inferred loop secants, not material constants. Samples<5 persistence is intentional existing code behavior; no live source altered.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(output)


if __name__=='__main__':main()
