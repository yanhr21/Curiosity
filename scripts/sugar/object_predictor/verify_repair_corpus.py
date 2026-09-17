"""Saved-only recovery from the first coordinator's exact-COM equality bug.

Preserves the original coordinator result, then verifies the same complete
trajectories with a 1e-12 m COM tolerance. No simulation, model, or data selection.
Must run only after the recorded coordinator has terminated.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from .overfit_data import FIXED_EPISODES, fixed_protocol, mass_supervision, uniform_history_indices


def verify(root, status):
    terminal = dict(line.split('=', 1) for line in status.read_text().splitlines())
    if 'exit_code' not in terminal:
        raise ValueError('Recorded coordinator must be terminal before saved-only verification')
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    if [c['episode'] for c in protocol['configurations']] != list(FIXED_EPISODES):
        raise ValueError('All fixed TRAIN IDs required')
    if protocol['overfit_data'] != fixed_protocol():
        raise ValueError('Fixed mass/time protocol changed')
    original = json.loads((root/'COLLECTION_RESULT.json').read_text())
    rows = []
    for config in protocol['configurations']:
        episode = config['episode']; directory = root/'cases'/f'episode_{episode}'
        row = dict(episode=episode, geometry_group=config['geometry_group'], split='train',
                   source=str(directory.resolve()), complete=False, controller_passed=False)
        try:
            result = json.loads((directory/'RESULT.json').read_text())
            recorded = json.loads((directory/'PROTOCOL.json').read_text())
            for key, value in config.items():
                mapped = {'mass':'mass_kg', 'load':'grip_target_per_hand_n'}.get(key, key)
                if result[mapped] != value and not (key=='mass' and np.isclose(result[mapped], value, rtol=1e-6)):
                    raise ValueError(f'Configuration mismatch: {key}')
            if recorded['controller_revision'] != 'sensor_feedback_v2' or recorded['frames'] != 2400:
                raise ValueError('Wrong revision or physical budget')
            with np.load(directory/f'episode_{episode}.npz', allow_pickle=False) as z:
                arrays={key:z[key] for key in ('timestamp_s', 'object_pose_w', 'object_com_w',
                    'validation_object_local_com_m', 'validation_full_mesh_min_z_m','normal_load_n')}
            if len(arrays['timestamp_s'])!=2400 or not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Incomplete or nonfinite physical labels')
            com=Rotation.from_quat(arrays['object_pose_w'][:,3:]).apply(arrays['validation_object_local_com_m'])+arrays['object_pose_w'][:,:3]
            error=float(np.max(abs(com-arrays['object_com_w'])))
            if not np.allclose(com, arrays['object_com_w'], rtol=0, atol=1e-12):
                raise ValueError(f'Actual COM transformation mismatch: {error} m')
            labels=[]
            for frame in protocol['overfit_data']['frames']:
                indices=uniform_history_indices(arrays['timestamp_s'], frame,32,.02)
                label=mass_supervision(arrays,indices,protocol['overfit_data']['mass_criteria'])
                labels.append(dict(frame=frame,mass_available=bool(label['mass_available']),
                    mass_status=int(label['mass_status']), diagnostics=label['evaluation_only']))
            if len(result['checks'])!=12 or result['passed']!=all(result['checks'].values()):
                raise ValueError('Original twelve physical checks missing/inconsistent')
            row.update(complete=True,controller_passed=result['passed'],controller_checks=result['checks'],
                values=result['values'],frames=2400,com_transform_max_abs_error_m=error,
                fit_clock_mass_labels=labels,fit_clock_qualified_mass_count=sum(v['mass_available'] for v in labels),
                late_fit_mass_count=sum(v['mass_available'] for v in labels if v['frame'] in (1581,2381)))
        except Exception as error:
            row['error']=repr(error)
        rows.append(row)
    final=dict(complete=all(r['complete'] for r in rows),records=rows,
        attempted_configurations=16,planned_configurations=16,controller_passes=sum(r['controller_passed'] for r in rows),
        qualification_passed=all(r['complete'] and r['controller_passed'] and r.get('late_fit_mass_count',0)>=1 for r in rows),
        new_physics_controls=0,new_model_forwards=0,new_optimizer_updates=0,
        verification=dict(coordinator_terminal=terminal,com_atol_m=1e-12,com_rtol=0,
            reason='Original exact equality wrongly rejected harmless single-frame versus batch Rotation.apply roundoff',
            physical_thresholds_changed=False,original_coordinator_result='COLLECTION_RESULT.original.json',
            original_qualification=original.get('qualification_passed')))
    for filename in ('COLLECTION_RESULT.json','QUALIFICATION.json'):
        old=root/filename
        backup=old.with_name(old.stem+'.original.json')
        if backup.exists():raise FileExistsError(backup)
        shutil.copy2(old,backup)
    for filename in ('COLLECTION_RESULT.json','QUALIFICATION.json','SAVED_VERIFICATION.json'):
        (root/filename).write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(json.dumps(final,indent=2))
    return 0 if final['qualification_passed'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--coordinator-status',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(verify(args.root,args.coordinator_status))
