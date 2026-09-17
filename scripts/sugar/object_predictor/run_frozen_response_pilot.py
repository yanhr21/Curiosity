"""Three fixed diagnostic cases; explicit process-local controller registration.

Existing collectors, registry source and default controller classes stay intact.
No model training or automatic expansion follows this physical pilot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from .overfit_data import fixed_protocol, uniform_history_indices, mass_supervision
from .retained_execution import require_active_resource
from .freeze_postlift_response_gain import INTERVENTION, DESCRIPTION, TELEMETRY, register_intervention
from .run_canonical_approach_fixture_pilot import CHECKS

EPISODES = (5014,5012,5000)
BASELINE = Path('experiments/object_predictor_v1/overfit_repair_v1/controlled_fixture16_v1')


def protocol():
    source=json.loads((BASELINE/'PROTOCOL.json').read_text())
    return dict(study='frozen_response_gain_pilot_v1',controller_intervention=INTERVENTION,
        baseline_root=str(BASELINE),baseline_intervention='freeze_postlift_alignment_v1',
        configurations=[next(c for c in source['configurations'] if c['episode']==e) for e in EPISODES],
        description=DESCRIPTION,controls_per_case=2400,dt=.02,physics_substeps=8,planned_cases=3,
        original_physical_checks=list(CHECKS),late_mass_check=dict(frame=2381,history=32,
            history_interval_s=.02,criteria=fixed_protocol()['mass_criteria']),
        freeze_clock='Actual gain returned on first original lift-admission command, when motion_elapsed=0. Hold only on subsequent commands through final hold.',
        diagnostic_estimator='Original response upper/samples/candidate continue on actual new observations; candidate is telemetry, not a counterfactual replay. Stale-upper bug remains.',
        gain_limit_scope='Held actual admission gains remain <=1.5e-4 and original velocity caps remain. They no longer follow the live upper contraction cap or fit-invalid base-gain cap; this is exactly the declared applied-gain freeze.',
        unchanged='Full original physics, observation-only control, initial1s/band/fit/angle, live force error, PCA closure direction, postlift alignment freeze, prescribed4s path,2400 controls,original12checks and late2381 qualification.',
        scope='Controlled known-scene kinematic-hand physical diagnostic; not all16 repair, general-purpose control, real-hand policy or learned predictor.',
        automatic_expansion=False,automatic_training=False)


def validate_protocol(root):
    declared=json.loads((root/'PROTOCOL.json').read_text())
    for key,value in protocol().items():
        if declared.get(key)!=value:raise ValueError('Fixed pilot protocol differs: '+key)
    bindings=declared.get('prepared_source_sha256',{})
    if not bindings:raise ValueError('Prepared source bindings required')
    for path,expected in bindings.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Prepared pilot source changed: '+path)
    return declared


def collect_one(root,episode):
    require_active_resource();declared=validate_protocol(root)
    config=next(c for c in declared['configurations'] if c['episode']==episode)
    register_intervention()
    from .collect_dense_grip import main as collect
    destination=root/'cases'/f'episode_{episode}'
    collect(argparse.Namespace(**config,output=str(destination),surface_feedback=True,frames=2400,
        force_gain=.000025,force_integral_time=0.,purpose='diagnostic',sustained_feedback=False,
        alignment_deadband=False,object_sdf_resolution=128,normalized_acquisition=False,
        sensor_coverage='continuous_palmar_v1',response_gain=True,controller_revision='sensor_feedback_v2',
        controller_intervention=INTERVENTION))
    # Preserve the generic collector receipt. Correct its inherited v2 wording
    # explicitly; this changes no scene, command, observation or original check.
    path=destination/'PROTOCOL.json';raw=path.read_text()
    (destination/'BASE_COLLECTOR_PROTOCOL.json').write_text(raw)
    actual=json.loads(raw)
    original_response_rule=actual.get('response_rule')
    actual.update(intervention_baseline='freeze_postlift_alignment_v1',motion_progress='Original ungated4s path after original admission; no runtime ready gating.',
        alignment_progress='Original alignment through admission, then held except prescribed yaw.',
        response_rule='Diagnostic estimator candidate only: '+str(original_response_rule)+' Actual applied gain instead follows the explicit admission freeze after admission.',
        applied_response_gain_rule=declared['freeze_clock'],response_diagnostic_estimator=declared['diagnostic_estimator'],
        actual_gain_limit_scope=declared['gain_limit_scope'],
        metadata_adapter='run_frozen_response_pilot; generic source receipt retained as BASE_COLLECTOR_PROTOCOL.json')
    path.write_text(json.dumps(actual,indent=2)+'\n')


def validate_case(folder,config,declared):
    actual=json.loads((folder/'PROTOCOL.json').read_text());result=json.loads((folder/'RESULT.json').read_text())
    if actual.get('controller_intervention')!=INTERVENTION or result.get('controller_intervention')!=INTERVENTION:
        raise ValueError('Missing actual intervention')
    for key in ('mass','load','seed','scale','geometry_group','approach_angle_deg','yaw_delta_deg','lift_height_m','lateral_xy'):
        if actual[{'mass':'mass_kg','load':'target_load_n'}.get(key,key)]!=config[key]:
            raise ValueError('Actual case setting differs: '+key)
    if set(result['checks'])!=set(CHECKS) or bool(result['passed'])!=all(result['checks'].values()):
        raise ValueError('Original physical check set differs')
    with np.load(folder/f"episode_{config['episode']}.npz",allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    if len(a['timestamp_s'])!=2400 or not np.allclose(a['timestamp_s'],.02*np.arange(1,2401),rtol=0,atol=1e-8):
        raise ValueError('Incomplete original physical clock')
    for key in TELEMETRY:
        value=a['validation_controller_'+key]
        if len(value)!=2400 or not np.isfinite(value).all():raise ValueError('Missing telemetry: '+key)
    com=Rotation.from_quat(a['object_pose_w'][:,3:]).apply(a['validation_object_local_com_m'])+a['object_pose_w'][:,:3]
    if not np.allclose(com,a['object_com_w'],rtol=0,atol=1e-12):raise ValueError('Actual COM transform differs')
    admitted=np.flatnonzero(a['validation_controller_lift_start_s']>=0)
    frozen=a['validation_controller_response_applied_gain_frozen']
    if len(admitted):
        first=int(admitted[0]);g=a['validation_controller_response_gain_m_per_ns'][first]
        if (frozen[:first+1].any() or not frozen[first+1:].all()
                or not np.array_equal(a['validation_controller_response_gain_m_per_ns'][first+1:],np.broadcast_to(g,(2399-first,2)))
                or not np.array_equal(a['validation_controller_response_frozen_gain_m_per_ns'][first:],np.broadcast_to(g,(2400-first,2)))):
            raise ValueError('Applied response gains differ from actual admission capture')
    elif frozen.any():raise ValueError('Gain froze without admission')
    check=declared['late_mass_check'];indices=uniform_history_indices(a['timestamp_s'],check['frame'],check['history'],check['history_interval_s'])
    mass=mass_supervision(a,indices,check['criteria'])
    return dict(complete=True,physical_passed=result['passed'],physical_checks=result['checks'],
        physical_values=result['values'],late_mass_available=bool(mass['mass_available']),
        late_mass_diagnostics=mass['evaluation_only'],actual_gain_freeze_validated=True,
        passed=bool(result['passed'] and mass['mass_available']))


def run(root):
    resource=require_active_resource();declared=validate_protocol(root)
    if declared.get('planned_resource')!={k:resource[k] for k in ('job_id','step_id','host')}:
        raise ValueError('Pilot resource binding differs')
    for name in ('cases','logs'):(root/name).mkdir(exist_ok=False)
    rows=[]
    for config in declared['configurations']:
        episode=config['episode'];row=dict(episode=episode,complete=False,passed=False)
        try:
            with (root/'logs'/f'episode_{episode}.log').open('x') as log:
                code=subprocess.run([sys.executable,'-P','-m',__spec__.name,'--root',str(root),
                    '--collect-episode',str(episode)],stdout=log,stderr=subprocess.STDOUT,pass_fds=(9,)).returncode
            row['exit_code']=code
            if code:raise RuntimeError('Actual collector exit '+str(code))
            row.update(validate_case(root/'cases'/f'episode_{episode}',config,declared))
        except Exception as exc:row.update(error=repr(exc),traceback=traceback.format_exc())
        row['baseline']=json.loads((BASELINE/'cases'/f'episode_{episode}'/'RESULT.json').read_text())
        rows.append(row)
        report=dict(complete=len(rows)==3 and all(r['complete'] for r in rows),
            qualification_passed=len(rows)==3 and all(r['passed'] for r in rows),cases=rows,
            planned_cases=3,retained_resource=resource,scope=declared['scope'],
            model_forwards=0,optimizer_updates=0,automatic_training=False)
        (root/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
        print('FROZEN_RESPONSE_PILOT_CASE',episode,row['complete'],row['passed'],flush=True)
    return 0 if report['qualification_passed'] else (2 if report['complete'] else 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--collect-episode',type=int,choices=EPISODES)
    args=parser.parse_args()
    if args.collect_episode is not None:collect_one(args.root,args.collect_episode)
    else:raise SystemExit(run(args.root))
