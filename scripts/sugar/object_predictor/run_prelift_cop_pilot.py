"""Fixed three-case observed-CoP pilot; CPU preparation/readback, root-owned physics.

Only this new process-local entry registers the reviewed adapter. Original
collectors/controllers and the complete 15/16 baseline remain unchanged.
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

from .overfit_data import fixed_protocol, mass_supervision, uniform_history_indices
from .prelift_cop_alignment import (INTERVENTION, DESCRIPTION, TELEMETRY,
    PreliftCoPAlignmentController, observed_alignment, register_intervention)
from .run_canonical_approach_fixture_pilot import CHECKS
from .retained_execution import require_active_resource

EPISODES = (5014, 5012, 5000)
BASELINE = Path('experiments/object_predictor_v1/overfit_repair_v1/controlled_fixture16_v1')
RESOURCE = BASELINE.parent/'ACTIVE_RESOURCE.json'
MODULE = 'scripts.sugar.object_predictor.run_prelift_cop_pilot'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def protocol():
    source = json.loads((BASELINE/'PROTOCOL.json').read_text())
    return dict(study='prelift_cop_alignment_pilot_v1', controller_intervention=INTERVENTION,
        baseline_root=str(BASELINE), baseline_intervention='freeze_postlift_alignment_v1',
        configurations=[next(c for c in source['configurations'] if c['episode']==e) for e in EPISODES],
        description=DESCRIPTION, planned_cases=3, planned_controls=7200,
        controls_per_case=2400, dt=.02, physics_substeps=8,
        original_physical_checks=list(CHECKS),
        late_mass_check=dict(frame=2381, history=32, history_interval_s=.02,
                             criteria=fixed_protocol()['mass_criteria']),
        cop_rule=dict(weight='observed pressure times area', normal='oriented bilateral PCA normal difference, normalized',
            error='contact-center difference projected tangent to common normal',
            per_hand_velocity='opposed +/- .25 * tangent error; norm capped .004m/s',
            per_hand_accumulated_travel_max_m=.10, active_clock_s=[1.,40.], admission_angle_max_deg=5.,
            admission='Original fit/angle/load readiness AND observed CoP angle <=5deg continuously for1s; original earliest24/latest40s. Never forced.',
            post_admission='No added CoP translation on admission or any later command; original fixed4s path and postlift alignment freeze.'),
        response_gain='Original live ObservedForceGain, not the failed frozen-response intervention. Formula unchanged; new observations can change its values.',
        unchanged='All original scale/mass/load/seed/approach/yaw/lift/lateral settings, full mesh physics, ideal continuous palmar field, force targets, original12checks and fixed late2381 H32 criteria.',
        observation_timing='Controller row k uses previous saved field/hand pose row k-1. First initial observation is not saved; readback replays geometry rows1..2399 and validates all2400 telemetry clocks.',
        scope='Controlled known-scene kinematic-hand physical candidate; pressure center is not certified exact solver wrench. Not original blind-grasp/all16 repair or learned prediction.',
        failure_policy='Retain all3 attempts and all original failures; all3 physical and late mass gates required. No case replacement, extra force, longer clock, automatic16 expansion or training.',
        model_forwards=0, optimizer_updates=0, automatic_expansion=False, automatic_training=False)


def source_paths():
    names=('run_prelift_cop_pilot','prelift_cop_alignment','freeze_postlift_alignment',
        'fixed_path_surface_grip','alignment_rollback','surface_grip_scene','response_gain',
        'inspect_contact_surface','controller_interventions','collect_dense_grip','collect_newton',
        'grip_motion_scene','grip_scene','probe_scene','support_scene','geometry','palmar_coverage',
        'overfit_data','run_canonical_approach_fixture_pilot','retained_execution')
    return [Path(__file__).with_name(n+'.py').resolve() for n in names]


def baseline_paths():
    return [BASELINE/'PROTOCOL.json', *[BASELINE/'cases'/f'episode_{e}'/name
        for e in EPISODES for name in ('PROTOCOL.json','RESULT.json',f'episode_{e}.npz')]]


def prepare(root):
    declared=protocol();resource=json.loads(RESOURCE.read_text())
    if resource['state']!='RUNNING':raise ValueError('Require declared retained resource')
    root.mkdir(parents=True,exist_ok=False)
    declared.update(planned_resource={k:resource[k] for k in ('job_id','step_id','host')},
        prepared_source_sha256={str(p):sha(p) for p in source_paths()},
        baseline_artifact_sha256={str(p.resolve()):sha(p) for p in baseline_paths()})
    write(root/'PROTOCOL.json',declared)
    validate_protocol(root)
    return declared


def validate_protocol(root):
    declared=json.loads((root/'PROTOCOL.json').read_text())
    for key,value in protocol().items():
        if declared.get(key)!=value:raise ValueError('Fixed CoP pilot changed: '+key)
    for field,paths in (('prepared_source_sha256',source_paths()),('baseline_artifact_sha256',baseline_paths())):
        bindings=declared.get(field,{})
        if set(bindings)!={str(p.resolve()) for p in paths}:raise ValueError('Incomplete bindings: '+field)
        for path,expected in bindings.items():
            if sha(path)!=expected:raise ValueError('Bound source/baseline changed: '+path)
    resource=json.loads(RESOURCE.read_text())
    if declared['planned_resource']!={k:resource[k] for k in ('job_id','step_id','host')}:
        raise ValueError('Prepared resource differs')
    return declared


def collect_one(root,episode):
    require_active_resource();declared=validate_protocol(root)
    config=next(c for c in declared['configurations'] if c['episode']==episode)
    register_intervention()
    from .collect_dense_grip import main as collect
    folder=root/'cases'/f'episode_{episode}'
    collect(argparse.Namespace(**config,output=str(folder),surface_feedback=True,frames=2400,
        force_gain=.000025,force_integral_time=0.,purpose='diagnostic',sustained_feedback=False,
        alignment_deadband=False,object_sdf_resolution=128,normalized_acquisition=False,
        sensor_coverage='continuous_palmar_v1',response_gain=True,controller_revision='sensor_feedback_v2',
        controller_intervention=INTERVENTION))
    path=folder/'PROTOCOL.json';raw=path.read_text()
    (folder/'BASE_COLLECTOR_PROTOCOL.json').write_text(raw)
    actual=json.loads(raw)
    actual.update(intervention_baseline='freeze_postlift_alignment_v1',
        controller_inputs=['clock','hand proprioception','previous measured palmar loads',
            'observed contact positions, positive areas and pressures'],
        readiness=declared['cop_rule']['admission'],
        motion_progress='Original ungated4s path after joint original+CoP admission; no runtime ready gate.',
        alignment_progress='Original through admission, then freeze except prescribed yaw. Additional bounded prelift CoP translation only.',
        changed_rules=['prelift_pressure_center_tangential_registration','additional_observed_CoP_admission_gate'],
        unchanged_rules=['response_gain_formula','load_target','original_load_and_angle_bands',
            'original_physical_thresholds','postlift_alignment_freeze','fixed_runtime_path'],
        cop_rule=declared['cop_rule'], metadata_adapter=MODULE,
        generic_source_receipt='BASE_COLLECTOR_PROTOCOL.json')
    write(path,actual)


def telemetry_readback(a,config):
    """Validate applied record arithmetic and combined readiness, without GT."""
    n=len(a['timestamp_s']);dt=.02
    def v(key):return a['validation_controller_'+key]
    for key in TELEMETRY:
        if len(v(key))!=n or not np.isfinite(v(key)).all():raise ValueError('Missing finite telemetry: '+key)
    valid=v('cop_valid');angle=v('cop_line_angle_deg');aligned=valid.all(1)&(angle<=5.)
    if not np.array_equal(aligned,v('cop_admission_ready')):raise ValueError('CoP gate arithmetic differs')
    lift=v('lift_start_s');admitted=np.flatnonzero(lift>=0);first=int(admitted[0]) if len(admitted) else n
    if len(admitted):
        if not np.all(lift[first:]==lift[first]) or not np.isclose(lift[first],a['timestamp_s'][first],rtol=0,atol=1e-8):
            raise ValueError('Inconsistent actual admission clock')
        if not 24.<=lift[first]<=40. or first<49 or not aligned[first-49:first+1].all():
            raise ValueError('Lift without original clock and sustained CoP qualification')
    servo=v('cop_servo_velocity_m_s');travel=v('cop_travel_m')
    expected=np.zeros_like(servo,dtype=float);accumulated=np.zeros(2)
    for i,time in enumerate(a['timestamp_s']):
        if i<first and 1.<=time<=40. and valid[i].all() and not aligned[i]:
            candidate=.25*v('cop_tangent_error_m')[i]
            speed=np.linalg.norm(candidate)
            if speed>.004:candidate*=.004/speed
            for side,sign in ((0,1.),(1,-1.)):
                step=sign*candidate*dt;length=np.linalg.norm(step)
                remaining=max(0.,.10-accumulated[side])
                if length>remaining:step*=remaining/length
                accumulated[side]+=np.linalg.norm(step);expected[i,side]=step/dt
        if not np.allclose(travel[i],accumulated,rtol=0,atol=1e-12):
            raise ValueError('CoP accumulated travel differs from applied increments')
    if not np.allclose(servo,expected,rtol=0,atol=1e-12):raise ValueError('CoP bounded servo arithmetic differs')
    if (np.linalg.norm(servo,axis=2)>.004+1e-12).any() or (travel>.10+1e-12).any():
        raise ValueError('CoP speed or travel budget exceeded')
    if not np.array_equal(v('cop_budget_exhausted'),(travel>=.10-1e-12).any(1)&~aligned):
        raise ValueError('CoP exhausted-budget flag differs')
    if np.any(servo[first:]!=0):raise ValueError('Added translation persists after admission')
    # The load channel is the original full palmar pressure-area sum. Its saved
    # float32 reduction can round at a boundary, so exact gate replay below uses
    # the stored ready clock together with all original fit/angle requirements.
    if len(admitted) and (not v('fit_valid')[first-49:first+1].all()
        or (v('alignment_error_deg')[first-49:first+1]>5.).any()
        or v('ready_seconds')[first]<1.):raise ValueError('Original sustained geometry/readiness missing')
    if np.any(v('ready_seconds')[:first][~aligned[:first]]>0):raise ValueError('Unaligned prelift readiness accumulated')
    return dict(passed=True,admission_frame=first if len(admitted) else None,
        admission_time_s=float(lift[first]) if len(admitted) else None,
        max_servo_speed_m_s=float(np.linalg.norm(servo,axis=2).max()),
        final_travel_m=travel[-1].tolist(),post_admission_servo_zero=True,
        original_response_gain_rule_retained=True)


def geometry_readback(a,folder,config):
    """Recompute CoP and original readiness using previous saved measurements."""
    controller=PreliftCoPAlignmentController(target_load_n=config['load'],response_gain=True,
        force_gain=.000025,controller_revision='sensor_feedback_v2',
        approach_angle_deg=config['approach_angle_deg'],yaw_delta_deg=config['yaw_delta_deg'],
        lift_height_m=config['lift_height_m'],lateral_xy=config['lateral_xy'])
    with np.load(folder/'contact_surface.npz',allow_pickle=False) as z:s={k:z[k] for k in z.files}
    if len(s['offset'])!=2401 or s['offset'][0]!=0 or s['offset'][-1]!=len(s['pad']):
        raise ValueError('Incomplete saved contact surface clock')
    worst=0.;ready=0.
    if a['validation_controller_ready_seconds'][0]!=0:
        raise ValueError('First zero-load command must have zero readiness')
    for i in range(1,2400):
        sl=slice(s['offset'][i-1],s['offset'][i]);field=dict(
            pos=s['position_hand_frame_m'][sl],area=s['area_m2'][sl],
            pressure=s['normal_pressure_pa'][sl],pad=s['pad'][sl],patch=s['hand'][sl])
        valid,centers,tangent,angle=observed_alignment(a['hand_pose_w'][i-1],field,controller.support_normals)
        for key,expected in [('cop_valid',valid),('cop_world_m',centers),('cop_tangent_error_m',tangent),('cop_line_angle_deg',angle)]:
            saved=a['validation_controller_'+key][i]
            if not np.allclose(saved,expected,rtol=0,atol=1e-10):raise ValueError('Observed CoP replay differs: '+key)
            worst=max(worst,float(np.max(np.abs(np.asarray(saved,dtype=float)-np.asarray(expected,dtype=float)))))
        loads=np.array([(field['area'][(field['pad']//27)==side]*field['pressure'][(field['pad']//27)==side]).sum() for side in range(2)])
        original_ready=bool(a['validation_controller_fit_valid'][i].all()
            and (a['validation_controller_alignment_error_deg'][i]<=5.).all()
            and (abs(loads/config['load']-1)<=.25).all())
        prelift=i==0 or a['validation_controller_lift_start_s'][i-1]<0
        if prelift and not (valid.all() and angle<=5.):ready=-.02
        ready=ready+.02 if original_ready else 0.
        if not np.isclose(ready,a['validation_controller_ready_seconds'][i],rtol=0,atol=1e-10):
            raise ValueError('Original force+geometry and CoP sustained gate differs')
    return dict(passed=True,previous_observation_rows_replayed=2399,
        initial_unsaved_observation_not_replayed=True,max_geometry_abs_difference=worst,
        complete_original_load_band_and_ready_clock_replayed=True)


def validate_case(folder,config,declared):
    actual=json.loads((folder/'PROTOCOL.json').read_text());result=json.loads((folder/'RESULT.json').read_text())
    baseline=json.loads((BASELINE/'cases'/f"episode_{config['episode']}"/'PROTOCOL.json').read_text())
    if actual.get('controller_intervention')!=INTERVENTION or result.get('controller_intervention')!=INTERVENTION:
        raise ValueError('Actual CoP intervention missing')
    for key in ('criteria','stability_criteria','frames','dt','object_sdf_resolution',
                'sensor_coverage','response_gain','force_gain_m_per_ns','force_integral_time_s'):
        if actual.get(key)!=baseline.get(key):raise ValueError('Original physics/threshold differs: '+key)
    for key in ('mass','load','seed','scale','geometry_group','approach_angle_deg','yaw_delta_deg','lift_height_m','lateral_xy'):
        if actual[{'mass':'mass_kg','load':'target_load_n'}.get(key,key)]!=config[key]:raise ValueError('Actual setting differs: '+key)
    if set(result['checks'])!=set(CHECKS) or bool(result['passed'])!=all(result['checks'].values()):
        raise ValueError('Original twelve physical check set differs')
    for name,digest in actual['source_sha256'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Actual collector source changed: '+name)
    with np.load(folder/f"episode_{config['episode']}.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if len(a['timestamp_s'])!=2400 or not np.allclose(a['timestamp_s'],.02*np.arange(1,2401),rtol=0,atol=1e-8):
        raise ValueError('Incomplete original2400 physical clock')
    com=Rotation.from_quat(a['object_pose_w'][:,3:]).apply(a['validation_object_local_com_m'])+a['object_pose_w'][:,:3]
    if not np.allclose(com,a['object_com_w'],rtol=0,atol=1e-12):raise ValueError('Actual COM transform differs')
    telemetry=telemetry_readback(a,config);geometry=geometry_readback(a,folder,config)
    check=declared['late_mass_check'];indices=uniform_history_indices(a['timestamp_s'],check['frame'],check['history'],check['history_interval_s'])
    mass=mass_supervision(a,indices,check['criteria'])
    return dict(complete=True,physical_passed=bool(result['passed']),physical_checks=result['checks'],
        physical_values=result['values'],late_mass_available=bool(mass['mass_available']),
        late_mass_status=int(mass['mass_status']),late_mass_window=dict(frame=2381,first_frame=int(indices[0]),
            diagnostics=mass['evaluation_only']),controller_telemetry=telemetry,observed_geometry_replay=geometry,
        passed=bool(result['passed'] and mass['mass_available']))


def run(root):
    resource=require_active_resource();declared=validate_protocol(root)
    if declared['planned_resource']!={k:resource[k] for k in ('job_id','step_id','host')}:raise ValueError('Actual resource differs')
    for name in ('cases','logs'):(root/name).mkdir(exist_ok=False)
    rows=[]
    for config in declared['configurations']:
        episode=config['episode'];row=dict(episode=episode,complete=False,passed=False)
        try:
            with (root/'logs'/f'episode_{episode}.log').open('x') as log:
                code=subprocess.run([sys.executable,'-P','-m',MODULE,'--root',str(root),'--collect-episode',str(episode)],
                    stdout=log,stderr=subprocess.STDOUT,pass_fds=(9,)).returncode
            row['exit_code']=code
            if code:raise RuntimeError('Collector exit '+str(code))
            row.update(validate_case(root/'cases'/f'episode_{episode}',config,declared))
        except Exception as exc:row.update(error=repr(exc),traceback=traceback.format_exc())
        row['baseline']=json.loads((BASELINE/'cases'/f'episode_{episode}'/'RESULT.json').read_text())
        rows.append(row)
        report=dict(study=declared['study'],complete=len(rows)==3 and all(r['complete'] for r in rows),
            qualification_passed=len(rows)==3 and all(r['passed'] for r in rows),cases=rows,
            planned_cases=3,attempted_cases=len(rows),retained_resource=resource,scope=declared['scope'],
            model_forwards=0,optimizer_updates=0,automatic_expansion=False,automatic_training=False)
        write(root/'RESULT.json',report)
        print('PRELIFT_COP_CASE',episode,row['complete'],row['passed'],flush=True)
    return 0 if report['qualification_passed'] else (2 if report['complete'] else 1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--check',action='store_true');mode.add_argument('--collect-episode',type=int,choices=EPISODES)
    args=parser.parse_args()
    if args.prepare:prepare(args.root);print('CPU_PREPARED',args.root);return 0
    if args.check:
        declared=validate_protocol(args.root)
        print(json.dumps(dict(passed=True,episodes=list(EPISODES),planned_controls=7200,
            baseline_intervention=declared['baseline_intervention'],model_forwards=0,physics_controls=0)));return 0
    if args.collect_episode is not None:collect_one(args.root,args.collect_episode);return 0
    return run(args.root)


if __name__=='__main__':raise SystemExit(main())
