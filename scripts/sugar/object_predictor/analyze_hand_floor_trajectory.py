"""Saved complete hand triangles versus public floor; no physical stepping."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sugar_newton.hand.patches import load_hand_mesh


def analyze(root):
    protocol=json.loads((root/'PROTOCOL.json').read_text());baseline=Path(protocol['baseline_root'])
    results=json.loads((root/'RESULT.json').read_text())
    if not results['complete']:raise ValueError('Require all fixed completed cases')
    meshes=[np.asarray(load_hand_mesh(side).vertices,np.float32).astype(float) for side in ('left','right')]
    records=[];arrays={}
    for case in results['cases']:
        e=case['episode'];record=dict(episode=e)
        for name,source in [('baseline',baseline),('candidate',root)]:
            file=source/'cases'/f'episode_{e}'/f'episode_{e}.npz'
            with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            hand=a['hand_pose_w'];t=a['timestamp_s'];minimum=np.zeros((len(t),2))
            for side,vertices in enumerate(meshes):
                z_axis=Rotation.from_quat(hand[:,side,3:]).as_matrix()[:,2,:]
                for lo in range(0,len(t),100):
                    hi=min(lo+100,len(t));minimum[lo:hi,side]=(z_axis[lo:hi]@vertices.T).min(1)+hand[lo:hi,side,2]
            sides=[]
            for side in (0,1):
                bad=np.flatnonzero(minimum[:,side]<-1e-6);worst=int(np.argmin(minimum[:,side]));frames=sorted(set([worst,*([int(bad[0])] if len(bad) else [])]))
                clocks=[]
                for i in frames:
                    clocks.append(dict(time_s=float(t[i]),min_z_m=float(minimum[i,side]),
                        cop_servo_velocity_m_s=a['validation_controller_cop_servo_velocity_m_s'][i,side].tolist(),
                        cop_travel_m=float(a['validation_controller_cop_travel_m'][i,side]),
                        actual_alignment_active=bool(a['validation_controller_actual_alignment_active'][i,side]),
                        lift_start_s=float(a['validation_controller_lift_start_s'][i]),
                        motion_elapsed_s=float(a['validation_controller_motion_elapsed_s'][i])))
                sides.append(dict(side=side,min_z_m=float(minimum[worst,side]),first_penetration_s=float(t[bad[0]]) if len(bad) else None,
                    penetrating_recorded_clocks=len(bad),total_recorded_clocks=len(t),events=clocks))
            record[name]=dict(recorded_control_clock_hand_floor_passed=bool((minimum>=-1e-6).all()),hands=sides,
                source_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            arrays[f'{e}_{name}_full_hand_min_z_m']=minimum
            arrays[f'{e}_{name}_timestamp_s']=t
        records.append(record)
    out=root/'HAND_FLOOR_TRAJECTORY.json'
    if out.exists():raise FileExistsError(out)
    np.savez_compressed(root/'HAND_FLOOR_TRAJECTORY.npz',**arrays)
    out.write_text(json.dumps(dict(cases=records,public_floor_z_m=0.,reporting_roundoff_m=1e-6,
        full_original_hand_meshes=True,physics_controls=0,model_forwards=0,
        original12_results_unchanged=True,scope='Independent additional hand-floor gate on all2400 saved20ms poses. Any negative vertex certifies triangle penetration; positive saved endpoints do not certify unsaved8 substeps. No thresholds of original gate changed.'),indent=2)+'\n')
    print(json.dumps(records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
