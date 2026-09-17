"""Four fixed canonical-approach acquisition attempts, never blind-grasp repair.

Reuse the existing full-mesh Newton collector and qualified freeze controller.
Only the declared approach angle differs from each archived case configuration.
Object initialization still uses its known canonical mesh frame. No automatic
16-case expansion, model execution or training follows this pilot.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from .controller_interventions import INTERVENTIONS
from .overfit_data import fixed_protocol, mass_supervision, uniform_history_indices
from .retained_execution import require_active_resource


EPISODES = (5008, 5012, 5020, 5000)
INTERVENTION = 'freeze_postlift_alignment_v1'
ORIGINAL_PROTOCOL = Path('experiments/object_predictor_v1/overfit_repair_v1/controller_v2/PROTOCOL.json')
CHECKS = ('bilateral', 'clearance', 'rise', 'peak', 'load_target', 'lift_started',
          'hold_frames', 'target_stability', 'frame_load_stability', 'hold_drift',
          'motion_completed', 'actual_complete_hold_frames')


def protocol():
    source = json.loads(ORIGINAL_PROTOCOL.read_text())
    original = [deepcopy(next(c for c in source['configurations'] if c['episode']==i)) for i in EPISODES]
    executed = [dict(c, approach_angle_deg=0.) for c in original]
    return dict(study='canonical_approach_fixture_v1',
        source_original_protocol=str(ORIGINAL_PROTOCOL),
        original_configurations=original, configurations=executed,
        requested_approach_evaluation=[dict(episode=c['episode'],
            original_requested_approach_deg=c['approach_angle_deg'], executed_approach_deg=0.,
            requested_approach_status='NOT_EXECUTED_REPLACED' if c['approach_angle_deg']!=0. else 'UNCHANGED_EXECUTED')
            for c in original],
        controller_intervention=INTERVENTION, controls_per_case=2400, dt_s=.02,
        physics_substeps=8, planned_cases=4, planned_controls=9600,
        original_physical_checks=list(CHECKS),
        late_mass_check=dict(frame=2381, history=32, history_interval_s=.02,
                             criteria=fixed_protocol()['mass_criteria']),
        scope='Known canonical-mesh object initialization with fixed world-X opposing approach and kinematic hands; not per-case GT collision-certified grasp planning, arbitrary-shape control, or original 60/90/150-degree blind-grasp qualification.',
        configuration_change='Only approach_angle_deg is replaced by zero; scale/mass/load/seed/yaw/lift/lateral and original case identities remain exact. Freeze controller is explicit, not silently treated as original v2.',
        failure_policy='All four fixed attempts retained. All original 12 physical checks and the fixed late mass-availability window must pass in every case. No replacement, automatic expansion or training.',
        model_forwards=0, optimizer_updates=0)


def validate_protocol(value):
    for key, expected in protocol().items():
        if value.get(key)!=expected:
            raise ValueError('Fixed canonical fixture protocol changed: '+key)
    for field in ('prepared_source_sha256','baseline_artifact_sha256'):
        for path, expected in value.get(field,{}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise ValueError('Prepared fixture binding changed: '+path)


def collection_command(root, configuration):
    return [sys.executable,'-P','-m','scripts.sugar.object_predictor.collect_dense_grip',
        '--protocol',str(root/'PROTOCOL.json'),'--episode',str(configuration['episode']),
        '--output',str(root/'cases'/f"episode_{configuration['episode']}"),
        '--surface-feedback','--frames','2400','--force-gain','.000025',
        '--sensor-coverage','continuous_palmar_v1','--response-gain',
        '--controller-revision','sensor_feedback_v2','--controller-intervention',INTERVENTION,
        '--purpose','diagnostic']


def validate_saved_case(directory, config, declared):
    episode=config['episode']
    result=json.loads((directory/'RESULT.json').read_text())
    actual=json.loads((directory/'PROTOCOL.json').read_text())
    if actual.get('controller_intervention')!=INTERVENTION or result.get('controller_intervention')!=INTERVENTION:
        raise ValueError('Actual freeze controller evidence missing')
    mapping=dict(mass='mass_kg',load='target_load_n')
    for key in ('mass','load','seed','scale','geometry_group','approach_angle_deg',
                'yaw_delta_deg','lift_height_m','lateral_xy'):
        if actual[mapping.get(key,key)]!=config[key]:
            raise ValueError('Executed case differs from declared configuration: '+key)
    if set(result['checks'])!=set(CHECKS) or bool(result['passed'])!=all(result['checks'].values()):
        raise ValueError('Original twelve physical checks missing or inconsistent')
    with np.load(directory/f'episode_{episode}.npz',allow_pickle=False) as saved:
        arrays={k:saved[k] for k in ('timestamp_s','object_com_w','object_pose_w',
            'normal_load_n','validation_full_mesh_min_z_m','validation_object_local_com_m')}
        if len(arrays['timestamp_s'])!=2400 or not np.allclose(arrays['timestamp_s'],.02*np.arange(1,2401),rtol=0,atol=1e-8):
            raise ValueError('Actual 2400-control physical clock missing')
        for key in INTERVENTIONS[INTERVENTION][3]:
            value=saved['validation_controller_'+key]
            if len(value)!=2400 or not np.isfinite(value).all():
                raise ValueError('Missing finite actual controller evidence: '+key)
    actual_com=Rotation.from_quat(arrays['object_pose_w'][:,3:]).apply(
        arrays['validation_object_local_com_m'])+arrays['object_pose_w'][:,:3]
    if not np.allclose(actual_com,arrays['object_com_w'],rtol=0,atol=1e-12):
        raise ValueError('Saved COM differs from transformed actual local body COM')
    check=declared['late_mass_check']
    indices=uniform_history_indices(arrays['timestamp_s'],check['frame'],check['history'],check['history_interval_s'])
    mass=mass_supervision(arrays,indices,check['criteria'])
    return dict(complete=True,physical_passed=bool(result['passed']),physical_checks=result['checks'],
        physical_values=result['values'],actual_com_transform_max_error_m=float(abs(actual_com-arrays['object_com_w']).max()),
        late_mass_available=bool(mass['mass_available']),late_mass_status=int(mass['mass_status']),
        late_mass_window=dict(frame=check['frame'],first_frame=int(indices[0]),
            start_time_s=float(arrays['timestamp_s'][indices[0]]),end_time_s=float(arrays['timestamp_s'][indices[-1]]),
            diagnostics=mass['evaluation_only']),
        passed=bool(result['passed'] and mass['mass_available']))


def aggregate(rows, declared, resource):
    return dict(study=declared['study'],complete=len(rows)==4 and all(r['complete'] for r in rows),
        qualification_passed=len(rows)==4 and all(r['passed'] for r in rows),
        planned_cases=4,attempted_cases=len(rows),cases=rows,
        requested_approach_evaluation=declared['requested_approach_evaluation'],
        original_requested_approach_regression_repaired=False,
        scope=declared['scope'],retained_resource=resource,
        new_model_forwards=0,new_optimizer_updates=0,automatic_expansion_or_training=False)


def main(root):
    resource=require_active_resource()
    declared=json.loads((root/'PROTOCOL.json').read_text());validate_protocol(declared)
    if any(declared.get('planned_resource',{}).get(k)!=resource[k] for k in ('job_id','step_id','host')):
        raise RuntimeError('Canonical fixture protocol resource binding changed')
    for name in ('cases','logs'):(root/name).mkdir(exist_ok=False)
    rows=[]
    for config in declared['configurations']:
        episode=config['episode'];directory=root/'cases'/f'episode_{episode}'
        row=dict(episode=episode,complete=False,passed=False,physical_passed=False,late_mass_available=False)
        print('CANONICAL_FIXTURE_START',episode,flush=True)
        try:
            with (root/'logs'/f'episode_{episode}.log').open('x') as log:
                code=subprocess.run(collection_command(root,config),stdout=log,stderr=subprocess.STDOUT,pass_fds=(9,)).returncode
            row['exit_code']=code
            if code!=0:raise RuntimeError('Collector runtime exit '+str(code))
            row.update(validate_saved_case(directory,config,declared))
        except Exception as exc:
            row.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        rows.append(row)
        result=aggregate(rows,declared,resource)
        (root/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('CANONICAL_FIXTURE_COMPLETE',episode,row['complete'],row['passed'],flush=True)
    return 0 if result['qualification_passed'] else (2 if result['complete'] else 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().root))
