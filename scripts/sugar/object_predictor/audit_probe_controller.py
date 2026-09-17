"""Replay recorded sensor feedback through the actual acquisition controller.

CPU only: no physics repetition, no network, no optimizer. Verify every issued
hand pose and retained touch flag, and describe sampled force oscillation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .probe_scene import ProbeController


def main(root):
    root=Path(root);protocol=json.loads((root/'PROTOCOL.json').read_text())
    with np.load(root/'episode_0000.npz') as a: data={k:a[k] for k in a.files}
    controller=ProbeController(gentle=protocol.get('gentle_force_servo',False),stable_servo=protocol.get('stable_servo',False))
    controller.command(0.,np.zeros(2),.02);controller.last_positions=None
    load=data['normal_load_n'].reshape(-1,2,27).sum(2)
    position=[];rotation=[];flags=[]
    for frame in range(len(load)):
        p,_,record=controller.command((frame+1)*.02,load[frame-1] if frame else np.zeros(2),.02)
        actual=data['hand_pose_w'][frame]
        position.append(float(np.abs(p[:,:3]-actual[:,:3]).max()))
        rotation.append(float(np.abs(Rotation.from_quat(p[:,3:]).as_matrix()-Rotation.from_quat(actual[:,3:]).as_matrix()).max()))
        flags.append(np.array_equal(record['touched'],data['validation_probe_touched'][frame]))
    phases=[]
    for phase in range(3):
        # Fixed final four seconds of approach, before retract. No selection by
        # observed score or labels. These are diagnostics, not admission gates.
        t=data['timestamp_s']-phase*controller.segment_seconds
        mask=(t>=controller.segment_seconds-6)&(t<controller.segment_seconds-2)
        x=load[mask]
        phases.append(dict(phase=phase,windows=len(x),mean_hand_load_n=x.mean(0).tolist(),
                           sample_to_sample_contact_toggle_fraction=np.mean((x[1:]>.01)!=(x[:-1]>.01),axis=0).tolist()))
    result=dict(frames=len(load),max_position_difference_m=max(position),max_rotation_matrix_difference=max(rotation),
                all_touch_flags_exact=all(flags),tolerance=1e-5,
                passed=max(position)<=1e-5 and max(rotation)<=1e-5 and all(flags),
                phase_force_diagnostics=phases,physics_replays=0,new_optimizer_updates=0,
                input='Only previous saved measured hand loads and current clock; no object labels.')
    (root/'CONTROLLER_REPLAY_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not result['passed']: raise RuntimeError('Actual controller routing mismatch')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');main(ap.parse_args().root)
