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
from .coordinated_rotation_cop import (INTERVENTION, DESCRIPTION, TELEMETRY,
    CoordinatedRotationCoPController as PreliftCoPAlignmentController, register_intervention, CLEARANCE_M, coordinate_rotation)
from .shared_budget_cop import MASKS, add_cop
from .qualified_cop_response import QualifiedObservedForceGain, command_evidence, ROUND_OFF_TOL
from .prelift_cop_alignment import observed_alignment
from . import run_prelift_cop_pilot as cop_baseline
from .run_canonical_approach_fixture_pilot import CHECKS
from .retained_execution import require_active_resource

EPISODES = (5014, 5012, 5000)
BASELINE = Path('experiments/object_predictor_v1/overfit_repair_v1/shared_budget_cop_pilot_v1')
RESOURCE = BASELINE.parent/'ACTIVE_RESOURCE.json'
MODULE = 'scripts.sugar.object_predictor.run_coordinated_rotation_cop_pilot'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def protocol():
    source=json.loads((BASELINE/'PROTOCOL.json').read_text())
    keep=('configurations','planned_cases','planned_controls','controls_per_case','dt','physics_substeps',
        'original_physical_checks','late_mass_check','cop_rule','response_gain','sample_rule','observation_timing',
        'known_hand_geometry','hand_floor','allocation','shared_budget')
    return dict(**{k:source[k] for k in keep},study='coordinated_rotation_cop_pilot_v1',
        controller_intervention=INTERVENTION,baseline_root=str(BASELINE),baseline_intervention='shared_budget_cop_v1',
        description=DESCRIPTION,
        rotation_coordination=dict(
            trigger='Prelift1..40s, bilateral measuredCoP valid but angle>5, committed parent with actual alignment request; transported fitted normals make predicted CoP line angle worse than fixed-current-normal prediction.',
            masks=[[True,False],[False,True],[True,True]],
            prediction='Current pressureCoPs and fitted normals transported by proposed hand rigid poses. No object GT; local contact refitting and load change unpredicted.',
            action='Undo selected alignment around same measured area-weighted local pivot. Preserve parent normal closure and selected CoP translations exactly; no extra speed/travel.',
            eligibility='Strictly better predictedCoP angle, both hands within original5deg of current measured fit normals, every full-hand sweep>=original1e-5m reserve.',
            selection='Minimum number suppressed; then smallest predicted angle; then fixed mask order. No eligible candidate retains exact parent.',
            state='Parent observed response consumed once; executed target/velocity/alignment/floor and NEXT response command evidence updated. Parent distance/travel/phase/readiness unchanged. Suppression only while no positive prelift ready/lift.',
            no_predicted_admission=True,postlift_unchanged=True,
            static_evidence='Saved5014 34..40s:301 independent safe one-hand candidates;69borrow per-step0.00405..0.00497deg. Not integrated, no one-stepcandidate reaches5deg, no success guarantee.'),
        unchanged='Shared200mm total, perhand4mm/s,40sdeadline, measured5deg/1sready, force loop, original12+lateH32 and all8substep floor gates; fixed3sameconfigs.',
        scope='Selective rotation coordination only; prior5014FAIL/shared2of3 unchanged; no automatic16/modeltraining.',
        failure_policy='All3 attempted and retained. New joint qualification requires all original12+late+actualfullhandfloor.',
        model_forwards=0,optimizer_updates=0,automatic_expansion=False,automatic_training=False)


def source_paths():
    from .run_shared_budget_cop_pilot import source_paths as parent_sources
    names=('run_coordinated_rotation_cop_pilot','coordinated_rotation_cop','coordinated_rotation_cop_scene',
           'collect_coordinated_rotation_cop','render_coordinated_rotation_cop_pilot')
    return list(dict.fromkeys([*parent_sources(),*[Path(__file__).with_name(n+'.py').resolve() for n in names]]))


def baseline_paths():
    return [BASELINE/'PROTOCOL.json',BASELINE/'RESULT.json', *[BASELINE/'cases'/f'episode_{e}'/name
        for e in EPISODES for name in ('PROTOCOL.json','RESULT.json',f'episode_{e}.npz','HAND_FLOOR_SUBSTEPS.npz')]]


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
    from .collect_coordinated_rotation_cop import main as collect
    folder=root/'cases'/f'episode_{episode}'
    collect(argparse.Namespace(**config,output=str(folder),surface_feedback=True,frames=2400,
        force_gain=.000025,force_integral_time=0.,purpose='diagnostic',sustained_feedback=False,
        alignment_deadband=False,object_sdf_resolution=128,normalized_acquisition=False,
        sensor_coverage='continuous_palmar_v1',response_gain=True,controller_revision='sensor_feedback_v2',
        controller_intervention=INTERVENTION))
    path=folder/'PROTOCOL.json';raw=path.read_text()
    (folder/'BASE_COLLECTOR_PROTOCOL.json').write_text(raw)
    actual=json.loads(raw)
    actual.update(intervention_baseline='shared_budget_cop_v1',
        controller_inputs=['clock','hand proprioception','previous measured palmar loads',
            'observed contact positions, positive areas and pressures'],
        readiness=declared['cop_rule']['admission'],
        motion_progress='Original4s path after original+CoP admission; prelift unsafe full commands try observed CoP allocation; no safe combination triggers whole hold. Postlift unsafe commands hold actual phase. No deadline extension.',
        alignment_progress='Measured prelift parent alignment with selective transport-opposing rotation suppression about the original area pivot; actual_alignment_active is final execution. Post-admission freeze and prescribed yaw remain original.',
        hand_floor=declared['hand_floor'], allocation=declared['allocation'], shared_budget=declared['shared_budget'], rotation_coordination=declared['rotation_coordination'], changed_rules=['selective_rotation_coordination_preserving_parent_force_and_shared_CoP'],
        unchanged_rules=['base_and_candidate_gain_formula_and_caps','load_target','original_load_and_angle_bands',
            'original_physical_thresholds','postlift_alignment_freeze','requested_runtime_path'],
        cop_rule=declared['cop_rule'], response_rule=declared['response_gain'], sample_rule=declared['sample_rule'], metadata_adapter=MODULE,
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
    accumulated=np.zeros(2)
    for i,time in enumerate(a['timestamp_s']):
        if np.any(servo[i]) and not (i<first and 1.<=time<=40. and valid[i].all() and not aligned[i] and not v('floor_blocked')[i]):
            raise ValueError('CoP motion outside original observation/clock/admission conditions')
        accumulated+=np.linalg.norm(servo[i]*dt,axis=1)
        if not np.allclose(travel[i],accumulated,rtol=0,atol=1e-12):
            raise ValueError('CoP accumulated travel differs from applied increments')
    if (np.linalg.norm(servo,axis=2)>.004+1e-12).any() or (travel.sum(1)>.2).any():
        raise ValueError('CoP speed or travel budget exceeded')
    if not np.array_equal(v('shared_total_travel_m'),travel.sum(1)) or not np.array_equal(v('shared_borrowed_travel_m'),np.maximum(0.,travel-.1)):
        raise ValueError('Actual shared/borrowed accounting differs')
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
        final_travel_m=travel[-1].tolist(),final_total_travel_m=float(travel[-1].sum()),
        final_borrowed_travel_m=v('shared_borrowed_travel_m')[-1].tolist(),
        applied_borrow_clocks=int(v('shared_borrow_applied').any(1).sum()),post_admission_servo_zero=True,
        original_gain_formula_and_caps_retained=True, qualified_upper_validity_repaired=True)


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
        if a['validation_controller_floor_blocked'][i]:ready=0.
        if not np.isclose(ready,a['validation_controller_ready_seconds'][i],rtol=0,atol=1e-10):
            raise ValueError('Original force+geometry and CoP sustained gate differs')
    return dict(passed=True,previous_observation_rows_replayed=2399,
        initial_unsaved_observation_not_replayed=True,max_geometry_abs_difference=worst,
        complete_original_load_band_and_ready_clock_replayed=True)


def response_readback(a,folder):
    """Replay original arithmetic with causal command-window qualification."""
    with np.load(folder/'contact_surface.npz',allow_pickle=False) as z:s={k:z[k] for k in z.files}
    v=lambda key:a['validation_controller_'+key]
    estimator=QualifiedObservedForceGain(.000025)
    for i,time in enumerate(a['timestamp_s']):
        loads=np.zeros(2)
        if i:
            lo,hi=s['offset'][i-1:i+1]
            for side in (0,1):
                select=s['pad'][lo:hi]//27==side
                loads[side]=(s['area_m2'][lo:hi][select]*s['normal_pressure_pa'][lo:hi][select]).sum()
        distance=v('distance_m')[i-1] if i else np.zeros(2)
        for side in (0,1):
            actual=estimator.update(side,float(time),float(loads[side]),float(distance[side]),.02,bool(v('fit_valid')[i,side]))
            expected=[v('response_gain_m_per_ns')[i,side],v('response_upper_n_m')[i,side],v('response_samples')[i,side]]
            if not np.array_equal(actual,expected):raise ValueError('Causal qualified response replay differs')
        for key,expected in [('response_window_pure',estimator.window_pure),
            ('response_secant_accepted',estimator.accepted),('response_qualified_secants_total',estimator.accepted_total),
            ('response_rejected_mixed_secants_total',estimator.rejected_total),('response_base_fallback',estimator.upper==0)]:
            if not np.array_equal(v(key)[i],expected):raise ValueError('Response qualification telemetry differs: '+key)
        travel=np.linalg.norm(v('cop_servo_velocity_m_s')[i],axis=1)*.02
        phase=float(v('motion_elapsed_s')[i]-(v('motion_elapsed_s')[i-1] if i else 0.))
        rotation=v('response_command_rotation_rad')[i]
        if i:
            old=Rotation.from_quat(a['hand_pose_w'][i-1,:,3:])
            new=Rotation.from_quat(v('response_command_target_quat_xyzw')[i])
            expected=(new*old.inv()).magnitude()
            if not np.allclose(rotation,expected,rtol=0,atol=1e-12):raise ValueError('Recorded command rotation differs')
        if not np.array_equal(travel,v('response_command_cop_travel_m')[i]) or phase!=v('response_command_phase_delta_s')[i]:
            raise ValueError('Recorded command nonnormal motion differs')
        pure=np.full(2,bool((rotation<=ROUND_OFF_TOL).all() and (travel==0).all() and abs(phase)<=ROUND_OFF_TOL))
        if not np.array_equal(v('response_command_pure')[i],pure):raise ValueError('Command purity differs')
        estimator.incoming_pure=pure
    accepted=estimator.accepted_total.tolist()
    return dict(passed=True,all2400_gain_upper_samples_exact=True,command_rotation_rows_replayed=2399,
        initial_unsaved_input_rotation_not_replayed=True,qualified_secants_per_hand=accepted,
        rejected_mixed_secants_per_hand=estimator.rejected_total.tolist(),
        base_fallback_frame_fraction=v('response_base_fallback').mean(0).tolist(),
        zero_qualified_means_base_fallback_not_identification=not any(accepted))


def allocation_row_readback(a,i,meshes,previous,initial,requested,field,support_normals):
    """Saved-only replay of narrow witness, conserved budget and all sweeps.

No object pose/label is accessed here. Parent normal/alignment proposal is saved
before CoP addition; exact actual response is independently replayed elsewhere.
"""
    from .hand_floor_feasibility import swept_min_height
    from .shared_budget_cop import budget_steps
    v=lambda key:a['validation_controller_'+key]
    parent=v('cop_allocation_parent_pose_w')[i]
    before=v('cop_travel_m')[i-1] if i else np.zeros(2)
    prelift=i==0 or v('lift_start_s')[i-1]<0
    time=a['timestamp_s'][i];raw=np.zeros((2,3));original=np.zeros((2,3))
    if (v('floor_requested_lift_start_s')[i]<0 and 1.<=time<=40.
            and v('cop_valid')[i].all() and not v('cop_admission_ready')[i]):
        velocity=.25*v('cop_tangent_error_m')[i];speed=np.linalg.norm(velocity)
        if speed>.004:velocity*=.004/speed
        for side,sign in ((0,1.),(1,-1.)):
            raw[side]=sign*velocity*.02;step=raw[side].copy();length=np.linalg.norm(step)
            remaining=max(0.,.1-before[side])
            if length>remaining:step*=remaining/length
            original[side]=step
    def group(desired):
        steps=np.array([budget_steps(desired,mask,before) for mask in MASKS])
        targets=[add_cop(parent,np.zeros((2,6)),step,np.ones(2,bool),.02)[0] for step in steps]
        heights=np.zeros((4,2));evaluated=np.zeros(4,bool);safe=np.zeros(4,bool)
        for j,target in enumerate(targets):
            if j and (safe[0] or not prelift):continue
            heights[j]=[swept_min_height(m,p,q) for m,p,q in zip(meshes,previous,target,strict=True)]
            evaluated[j]=True;safe[j]=bool((heights[j]>=CLEARANCE_M).all())
        choices=np.flatnonzero(safe);scores=np.linalg.norm(steps,axis=2).sum(1)
        selected=int(choices[np.argmax(scores[choices])]) if len(choices) else -1
        return dict(steps=steps,targets=targets,heights=heights,evaluated=evaluated,safe=safe,selected=selected)
    base=group(original);quota=np.linalg.norm(raw,axis=1)>np.linalg.norm(original,axis=1)
    donor=np.zeros(2,bool)
    if prelift and base['selected']>=0:
        donor=(np.linalg.norm(original,axis=1)>0)&~MASKS[base['selected']]&(base['heights'][0]<CLEARANCE_M)&(before<.1)
    eligible=quota&donor[::-1] if before.sum()<.2 else np.zeros(2,bool)
    desired=original.copy();desired[eligible]=raw[eligible]
    trial=group(desired) if eligible.any() else None
    chosen=base;used=original;applied=np.zeros(2,bool)
    if trial is not None and trial['selected']>=0:
        candidate=trial['steps'][trial['selected']];base_step=base['steps'][base['selected']]
        added=eligible&(np.linalg.norm(candidate,axis=1)>np.linalg.norm(base_step,axis=1))
        if added.any() and np.linalg.norm(candidate,axis=1).sum()>=np.linalg.norm(base_step,axis=1).sum():
            chosen=trial;used=desired;applied=added
    selected=chosen['selected'];committed=selected>=0
    mask=MASKS[selected] if committed else np.zeros(2,bool)
    executed=chosen['steps'][selected] if committed else np.zeros((2,3))
    changed=bool(committed and np.any(np.linalg.norm(used,axis=1)[~mask]>0.))
    expected=dict(shared_available_before_m=max(0.,.2-float(before.sum())),shared_desired_step_m=raw,
        shared_original_proposed_step_m=original,shared_quota_limited=quota,shared_floor_donor=donor,
        shared_borrow_eligible=eligible,shared_borrow_attempted=bool(eligible.any()),shared_borrow_applied=applied,
        shared_baseline_selected_mask=MASKS[base['selected']] if base['selected']>=0 else np.zeros(2,bool),
        shared_baseline_blocked=base['selected']<0,shared_baseline_evaluated=base['evaluated'],
        shared_baseline_safe=base['safe'],shared_baseline_swept_min_z_m=base['heights'],
        shared_baseline_candidate_steps_m=base['steps'],shared_candidate_steps_m=chosen['steps'],
        shared_borrow_candidate_evaluated=trial['evaluated'] if trial else np.zeros(4,bool),
        shared_borrow_candidate_safe=trial['safe'] if trial else np.zeros(4,bool),
        shared_borrow_candidate_swept_min_z_m=trial['heights'] if trial else np.zeros((4,2)),
        shared_borrow_candidate_steps_m=trial['steps'] if trial else np.zeros((4,2,3)),
        cop_allocation_attempted=bool(not chosen['safe'][0] and prelift),
        cop_allocation_evaluated=chosen['evaluated'],cop_allocation_safe=chosen['safe'],
        cop_allocation_swept_min_z_m=chosen['heights'],cop_allocation_selected_mask=mask,
        cop_allocation_changed=changed,cop_allocation_proposed_step_m=used,
        floor_parent_transaction_committed=committed,floor_blocked=not committed,
        floor_requested_target_pose_w=chosen['targets'][0],floor_requested_cop_velocity_m_s=chosen['steps'][0]/.02,
        cop_servo_velocity_m_s=executed/.02,
        shared_total_limited=bool(committed and np.any(np.linalg.norm(executed,axis=1)<np.linalg.norm(used*mask[:,None],axis=1))))
    for key,value in expected.items():
        if 'swept_min' in key:
            good=np.allclose(v(key)[i],value,rtol=0,atol=1e-12)
        else:good=np.array_equal(v(key)[i],value)
        if not good:raise ValueError('Shared allocation decision replay differs: '+key)
    for key,increment in [('cop_allocation_changed_count',int(changed)),('shared_borrow_applied_count',int(applied.any()))]:
        if v(key)[i]!=(int(v(key)[i-1]) if i else 0)+increment:raise ValueError('Actual allocation count differs: '+key)
    actual_travel=before+np.linalg.norm(executed,axis=1)
    if not np.array_equal(v('cop_travel_m')[i],actual_travel) or actual_travel.sum()>.2:
        raise ValueError('Shared actual travel differs or exceeds200mm')
    parent_target=chosen['targets'][selected] if committed else previous
    if not np.array_equal(v('rotation_coord_parent_target_pose_w')[i],parent_target):
        raise ValueError('Original shared parent target differs')
    parent_height=chosen['heights'][selected] if committed else initial
    parent_record={key.removeprefix('validation_controller_'):value[i] for key,value in a.items() if key.startswith('validation_controller_')}
    parent_record['actual_alignment_active']=v('rotation_coord_parent_actual_alignment_active')[i]
    parent_record['floor_executed_swept_min_z_m']=parent_height
    target,diag=coordinate_rotation(previous,parent_target,parent_record,field,support_normals,meshes,
        prelift=bool(i==0 or v('lift_start_s')[i-1]<0),time=float(a['timestamp_s'][i]))
    for key,value in diag.items():
        if np.asarray(value).dtype.kind in 'fc':good=np.allclose(v(key)[i],value,rtol=0,atol=1e-12)
        else:good=np.array_equal(v(key)[i],value)
        if not good:raise ValueError('Rotation coordination replay differs: '+key)
    suppressed=diag['rotation_coord_selected_suppressed']
    previous_count=int(v('rotation_coord_suppressed_count')[i-1]) if i else 0
    if int(v('rotation_coord_suppressed_count')[i])!=previous_count+int(suppressed.any()):
        raise ValueError('Actual rotation suppression clock count differs')
    if not np.array_equal(v('actual_alignment_active')[i],parent_record['actual_alignment_active']&~suppressed):
        raise ValueError('Actual alignment suppression differs')
    if not np.array_equal(v('floor_executed_target_pose_w')[i],target):raise ValueError('Final coordinated target differs')
    if not np.allclose(a['hand_pose_w'][i],target,rtol=0,atol=2e-7):raise ValueError('Actual hand differs from coordinated target')
    if committed:
        for actual,proposed in [('distance_m','floor_requested_distance_m'),('ready_seconds','floor_requested_ready_seconds'),
            ('motion_elapsed_s','floor_requested_motion_elapsed_s'),('lift_start_s','floor_requested_lift_start_s')]:
            if not np.array_equal(v(actual)[i],v(proposed)[i]):raise ValueError('Committed parent state differs: '+actual)
        if not np.array_equal(v('rotation_coord_parent_actual_alignment_active')[i],v('floor_requested_alignment_active')[i]):
            raise ValueError('Original parent alignment request differs')
    from .hand_floor_feasibility import swept_min_height
    return np.array([swept_min_height(m,p,q) for m,p,q in zip(meshes,previous,target,strict=True)])


def floor_readback(a,folder):
    """Independent full-mesh replay of all actual8substeps and transaction flags."""
    from sugar_newton.hand.patches import load_hand_mesh
    from .hand_floor_feasibility import swept_min_height
    meshes=[np.asarray(load_hand_mesh(s).vertices,np.float32).astype(float) for s in ('left','right')]
    with np.load(folder/'HAND_FLOOR_SUBSTEPS.npz',allow_pickle=False) as z:
        sub={k:z[k] for k in z.files}
    if len(sub['timestamp_s'])!=19200 or not np.allclose(sub['timestamp_s'],.0025*np.arange(1,19201),rtol=0,atol=1e-8):
        raise ValueError('All original8 substep clocks required')
    if not np.array_equal(sub['control_frame'],np.repeat(np.arange(2400),8)) or not np.array_equal(sub['substep'],np.tile(np.arange(8),2400)):
        raise ValueError('Substep identities differ')
    minima=[]
    for pose_key,min_key in [('fk_hand_pose_w','fk_min_z_m'),('actual_hand_pose_w','actual_min_z_m')]:
        pose=sub[pose_key]
        if pose.shape!=(19200,2,7) or not np.isfinite(pose).all():raise ValueError('Incomplete actual hand poses')
        actual=np.zeros((19200,2))
        for side,vertices in enumerate(meshes):
            rows=Rotation.from_quat(pose[:,side,3:]).as_matrix()[:,2,:]
            for lo in range(0,19200,100):
                hi=min(19200,lo+100)
                actual[lo:hi,side]=(rows[lo:hi]@vertices.T).min(1)+pose[lo:hi,side,2]
        if not np.allclose(actual,sub[min_key],rtol=0,atol=1e-12):raise ValueError('Actual full hand mesh minimum replay differs')
        minima.append(actual)
    if not np.array_equal(sub['actual_hand_pose_w'][7::8],a['hand_pose_w']):
        raise ValueError('Final substep differs from saved control pose')
    v=lambda k:a['validation_controller_'+k]
    blocked=v('floor_blocked').astype(bool)
    if not np.array_equal(v('floor_blocked_count'),np.cumsum(blocked)) or not np.array_equal(v('floor_parent_transaction_committed'),~blocked):
        raise ValueError('Floor transaction counts/commit differ')
    previous=sub['initial_hand_pose_w'].astype(float)
    initial_replay=np.array([swept_min_height(m,p,p) for m,p in zip(meshes,previous,strict=True)])
    if not np.allclose(initial_replay,sub['initial_min_z_m'],rtol=0,atol=1e-12):raise ValueError('Initial full hand minimum differs')
    maximum_height_error=0.
    from .geometry import palmar_support_frame
    support_normals=np.stack([palmar_support_frame(load_hand_mesh(s),sign)[0].inv().apply([0.,0.,1.]) for s,sign in [('left',-1.),('right',1.)]])
    with np.load(folder/'contact_surface.npz',allow_pickle=False) as z:surface={k:z[k] for k in z.files}
    for i in range(2400):
        proposed=v('floor_requested_target_pose_w')[i]
        initial=np.array([swept_min_height(m,p,p) for m,p in zip(meshes,previous,strict=True)])
        requested=np.array([swept_min_height(m,p,q) for m,p,q in zip(meshes,previous,proposed,strict=True)])
        if (initial<CLEARANCE_M).any():raise ValueError('Unsafe observed old hand accepted by controller')
        sl=slice(surface['offset'][i-1],surface['offset'][i]) if i else slice(0,0)
        field=dict(pos=surface['position_hand_frame_m'][sl],area=surface['area_m2'][sl],pressure=surface['normal_pressure_pa'][sl],pad=surface['pad'][sl],patch=surface['hand'][sl])
        expected=allocation_row_readback(a,i,meshes,previous,initial,requested,field,support_normals)
        for name,x in [('floor_observed_min_z_m',initial),('floor_requested_swept_min_z_m',requested),('floor_executed_swept_min_z_m',expected)]:
            error=float(np.max(abs(v(name)[i]-x)));maximum_height_error=max(maximum_height_error,error)
            if error>1e-12:raise ValueError('Swept certificate replay differs: '+name)
        if blocked[i]:
            for name in ('cop_servo_velocity_m_s','response_command_rotation_rad','response_command_cop_travel_m',
                         'response_command_phase_delta_s','actual_alignment_active','actualmotion_rate','ready_seconds'):
                if np.any(v(name)[i]!=0):raise ValueError('Rejected command claims executed motion/readiness: '+name)
            for name,zero in [('distance_m',np.zeros(2)),('cop_travel_m',np.zeros(2)),('motion_elapsed_s',0.),('lift_start_s',-1.),('lift_complete_s',-1.)]:
                before=v(name)[i-1] if i else zero
                if not np.array_equal(v(name)[i],before):raise ValueError('Rejected command changed motion state: '+name)
            if not np.allclose(a['hand_pose_w'][i],previous,rtol=0,atol=2e-7):
                raise ValueError('Rejected command did not hold actual pose')
        previous=a['hand_pose_w'][i].astype(float)
    stacked=np.stack(minima)
    return dict(passed=True,actual_substep_hand_floor_passed=bool((stacked>=0.).all()),
        actual_fk_min_z_m=minima[0].min(0).tolist(),actual_solver_min_z_m=minima[1].min(0).tolist(),
        all19200_substeps_both_hands_replayed=True,all2400_command_transactions_replayed=True,
        maximum_certificate_replay_error_m=maximum_height_error,blocked_commands=int(blocked.sum()),
        first_blocked_s=float(a['timestamp_s'][np.flatnonzero(blocked)[0]]) if blocked.any() else None,
        no_model_or_physics=True)


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
    response=response_readback(a,folder)
    floor=floor_readback(a,folder)
    check=declared['late_mass_check'];indices=uniform_history_indices(a['timestamp_s'],check['frame'],check['history'],check['history_interval_s'])
    mass=mass_supervision(a,indices,check['criteria'])
    return dict(complete=True,physical_passed=bool(result['passed']),physical_checks=result['checks'],
        response_readback=response,hand_floor_readback=floor,hand_floor_passed=floor['actual_substep_hand_floor_passed'],physical_values=result['values'],late_mass_available=bool(mass['mass_available']),
        late_mass_status=int(mass['mass_status']),late_mass_window=dict(frame=2381,first_frame=int(indices[0]),
            diagnostics=mass['evaluation_only']),controller_telemetry=telemetry,observed_geometry_replay=geometry,
        passed=bool(result['passed'] and mass['mass_available'] and floor['actual_substep_hand_floor_passed']))


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
        print('SHARED_BUDGET_COP_CASE',episode,row['complete'],row['passed'],flush=True)
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
