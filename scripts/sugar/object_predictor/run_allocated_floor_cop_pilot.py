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
from .allocated_floor_cop import (INTERVENTION, DESCRIPTION, TELEMETRY,
    AllocatedFloorCoPController as PreliftCoPAlignmentController, register_intervention, CLEARANCE_M, MASKS, add_cop)
from .qualified_cop_response import QualifiedObservedForceGain, command_evidence, ROUND_OFF_TOL
from .prelift_cop_alignment import observed_alignment
from . import run_prelift_cop_pilot as cop_baseline
from .run_canonical_approach_fixture_pilot import CHECKS
from .retained_execution import require_active_resource

EPISODES = (5014, 5012, 5000)
BASELINE = Path('experiments/object_predictor_v1/overfit_repair_v1/floor_safe_cop_pilot_v1')
RESOURCE = BASELINE.parent/'ACTIVE_RESOURCE.json'
MODULE = 'scripts.sugar.object_predictor.run_allocated_floor_cop_pilot'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def protocol():
    from sugar_newton.hand.patches import load_hand_mesh
    hand_geometry={}
    for side in ('left','right'):
        mesh=load_hand_mesh(side);vertices=np.asarray(mesh.vertices,np.float32);faces=np.asarray(mesh.faces,np.int64)
        hand_geometry[side]=dict(vertices=len(vertices),triangles=len(faces),
            float32_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
            int64_faces_sha256=hashlib.sha256(faces.tobytes()).hexdigest())
    source=json.loads((BASELINE/'PROTOCOL.json').read_text())
    keep=('configurations','planned_cases','planned_controls','controls_per_case','dt','physics_substeps',
        'original_physical_checks','late_mass_check','cop_rule','response_gain','sample_rule','observation_timing')
    return dict(**{k:source[k] for k in keep},study='allocated_floor_cop_pilot_v1',
        controller_intervention=INTERVENTION,baseline_root=str(BASELINE),
        baseline_intervention='floor_safe_qualified_cop_v1',description=DESCRIPTION,known_hand_geometry=hand_geometry,
        hand_floor=dict(public_floor_z_m=0.,reserved_clearance_m=CLEARANCE_M,
            geometry='Every vertex of both original complete CAD triangle meshes; affine triangle heights.',
            swept_rule='Analytic global minimum over complete linear-translation plus shortest-SLERP path; endpoints and all stationary points.',
            execution='Safe full original command unchanged. Otherwise prelift enumerate per-hand CoP keep/zero masks11,10,01,00; choose safe maximum retained original step length, fixed-order ties. Commit complete parent normal/alignment plus retained CoP; if no safe combination original full hold. Postlift never enumerate.',
            rejected_motion='Rollback closure distance, CoP travel, lift start/completion/phase, previous lift/yaw and previous poses; readiness resets0. Actual target=observed pose, velocity0.',
            response='Original update exactly once on real current observation and prior executed distance/command evidence; keep this observed state, replace next incoming_pure from actual selected command or hold.',
            substep_evidence='Record actual full hand pose and mesh minima after FK and after solver at every8substeps; new joint gate requires all minima>=0.',
            unsafe_initial='Raise explicit failure; never clamp wrist or teleport.',
            geometric_margin_not_acceptance_relaxation=True),
        allocation=dict(masks=MASKS.astype(int).tolist(), selection='maximum sum of retained original per-hand step norms; fixed mask order resolves ties',
            exact_parent_pose_saved=True, per_hand_actual_travel_only=True, response_evidence='actual executed pose and retained CoP; real force observation update once',
            full_hold_fallback=True, postlift_allocation=False, fixed_state_counterfactual_not_trajectory=True),
        unchanged='Original configurations, clock2400/48s, 12checks, late2381/H32, CoP5deg/40s/.1m/.004mps and original qualified gain formula.',
        scope='CoP allocation on completed floor-safe baseline2/3; no static-counterfactual trajectory or grasp success claim. Original15/16 and all negative interventions retained.',
        failure_policy='All3 attempted/retained; new joint qualification requires original12+late and actual full-hand floor. No case replacement, extra force, longer clock or automatic16/train.',
        model_forwards=0,optimizer_updates=0,automatic_expansion=False,automatic_training=False)


def source_paths():
    names=('run_allocated_floor_cop_pilot','allocated_floor_cop','allocated_floor_cop_scene','collect_allocated_floor_cop','render_allocated_floor_cop_pilot','floor_safe_cop','hand_floor_feasibility','run_floor_safe_cop_pilot','run_qualified_cop_response_pilot','qualified_cop_response','run_prelift_cop_pilot','prelift_cop_alignment','freeze_postlift_alignment',
        'fixed_path_surface_grip','alignment_rollback','surface_grip_scene','response_gain',
        'inspect_contact_surface','controller_interventions','collect_dense_grip','collect_newton',
        'grip_motion_scene','grip_scene','probe_scene','support_scene','geometry','palmar_coverage',
        'overfit_data','run_canonical_approach_fixture_pilot','retained_execution')
    return [Path(__file__).with_name(n+'.py').resolve() for n in names]


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
    from .collect_allocated_floor_cop import main as collect
    folder=root/'cases'/f'episode_{episode}'
    collect(argparse.Namespace(**config,output=str(folder),surface_feedback=True,frames=2400,
        force_gain=.000025,force_integral_time=0.,purpose='diagnostic',sustained_feedback=False,
        alignment_deadband=False,object_sdf_resolution=128,normalized_acquisition=False,
        sensor_coverage='continuous_palmar_v1',response_gain=True,controller_revision='sensor_feedback_v2',
        controller_intervention=INTERVENTION))
    path=folder/'PROTOCOL.json';raw=path.read_text()
    (folder/'BASE_COLLECTOR_PROTOCOL.json').write_text(raw)
    actual=json.loads(raw)
    actual.update(intervention_baseline='floor_safe_qualified_cop_v1',
        controller_inputs=['clock','hand proprioception','previous measured palmar loads',
            'observed contact positions, positive areas and pressures'],
        readiness=declared['cop_rule']['admission'],
        motion_progress='Original4s path after original+CoP admission; prelift unsafe full commands try observed CoP allocation; no safe combination triggers whole hold. Postlift unsafe commands hold actual phase. No deadline extension.',
        alignment_progress='Original through admission, then freeze except prescribed yaw. Additional bounded prelift CoP translation only.',
        hand_floor=declared['hand_floor'], allocation=declared['allocation'], changed_rules=['prelift_per_hand_cop_feasible_allocation'],
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
    expected=np.zeros_like(servo,dtype=float);accumulated=np.zeros(2)
    for i,time in enumerate(a['timestamp_s']):
        if i<first and 1.<=time<=40. and valid[i].all() and not aligned[i] and not v('floor_blocked')[i]:
            candidate=.25*v('cop_tangent_error_m')[i]
            speed=np.linalg.norm(candidate)
            if speed>.004:candidate*=.004/speed
            for side,sign in ((0,1.),(1,-1.)):
                step=sign*candidate*dt;length=np.linalg.norm(step)
                remaining=max(0.,.10-accumulated[side])
                if length>remaining:step*=remaining/length
                step*=v('cop_allocation_selected_mask')[i,side]
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


def allocation_row_readback(a,i,meshes,previous,initial,requested):
    """Replay all four command alternatives from saved exact pre-CoP pose.

Only hand proprioception, observed tangent error and known hand mesh enter.
No object state or label is read to choose a combination.
"""
    from .hand_floor_feasibility import swept_min_height
    v=lambda key:a['validation_controller_'+key]
    parent=v('cop_allocation_parent_pose_w')[i]
    step=v('cop_allocation_proposed_step_m')[i]
    before_travel=v('cop_travel_m')[i-1] if i else np.zeros(2)
    prelift=i==0 or v('lift_start_s')[i-1]<0
    expected_steps=np.zeros((2,3))
    time=a['timestamp_s'][i]
    if (v('floor_requested_lift_start_s')[i]<0 and 1.<=time<=40.
            and v('cop_valid')[i].all() and not v('cop_admission_ready')[i]):
        candidate=.25*v('cop_tangent_error_m')[i]
        speed=np.linalg.norm(candidate)
        if speed>.004:candidate*=.004/speed
        for side,sign in ((0,1.),(1,-1.)):
            delta=sign*candidate*.02;length=np.linalg.norm(delta)
            remaining=max(0.,.10-before_travel[side])
            if length>remaining:delta*=remaining/length
            expected_steps[side]=delta
    if not np.array_equal(step,expected_steps):raise ValueError('Requested CoP steps differ from original formula/budget')
    if not np.array_equal(v('floor_requested_cop_velocity_m_s')[i],step/.02):raise ValueError('Requested CoP velocity differs')
    targets=[add_cop(parent,np.zeros((2,6)),step,mask,.02)[0] for mask in MASKS]
    if not np.array_equal(targets[0],v('floor_requested_target_pose_w')[i]):raise ValueError('Exact parent plus full CoP differs')
    attempted=bool((requested<CLEARANCE_M).any() and prelift)
    evaluated=np.array([True,attempted,attempted,attempted])
    heights=np.zeros((4,2));heights[0]=requested
    if attempted:
        for index in (1,2,3):
            heights[index]=[swept_min_height(m,p,q) for m,p,q in zip(meshes,previous,targets[index],strict=True)]
    safe=evaluated & (heights>=CLEARANCE_M).all(1)
    choices=np.flatnonzero(safe)
    scores=np.linalg.norm(step,axis=1)@MASKS.T
    selected=int(choices[np.argmax(scores[choices])]) if len(choices) else -1
    committed=selected>=0;mask=MASKS[selected] if committed else np.zeros(2,bool)
    changed=bool(committed and np.any(np.linalg.norm(step,axis=1)[~mask]>0.))
    for key,value in [('cop_allocation_attempted',attempted),('cop_allocation_evaluated',evaluated),
        ('cop_allocation_safe',safe),('cop_allocation_selected_mask',mask),
        ('cop_allocation_changed',changed),('floor_parent_transaction_committed',committed),
        ('floor_blocked',not committed)]:
        if not np.array_equal(v(key)[i],value):raise ValueError('Allocation decision replay differs: '+key)
    count=(int(v('cop_allocation_changed_count')[i-1]) if i else 0)+int(changed)
    if v('cop_allocation_changed_count')[i]!=count:raise ValueError('Allocation count differs')
    if not np.allclose(v('cop_allocation_swept_min_z_m')[i],heights,rtol=0,atol=1e-12):
        raise ValueError('Allocation alternative swept minima differ')
    target=targets[selected] if committed else previous
    if not np.array_equal(v('floor_executed_target_pose_w')[i],target):raise ValueError('Executed allocation target differs')
    if not np.allclose(a['hand_pose_w'][i],target,rtol=0,atol=2e-7):raise ValueError('Actual final pose differs from allocated target')
    if not np.array_equal(v('cop_servo_velocity_m_s')[i],step*mask[:,None]/.02):
        raise ValueError('Suppressed CoP claimed executed')
    if committed:
        for actual,proposed in [('distance_m','floor_requested_distance_m'),
            ('ready_seconds','floor_requested_ready_seconds'),('motion_elapsed_s','floor_requested_motion_elapsed_s'),
            ('lift_start_s','floor_requested_lift_start_s'),('actual_alignment_active','floor_requested_alignment_active')]:
            if not np.array_equal(v(actual)[i],v(proposed)[i]):raise ValueError('Committed parent state differs: '+actual)
    return heights[selected] if committed else initial


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
    for i in range(2400):
        proposed=v('floor_requested_target_pose_w')[i]
        initial=np.array([swept_min_height(m,p,p) for m,p in zip(meshes,previous,strict=True)])
        requested=np.array([swept_min_height(m,p,q) for m,p,q in zip(meshes,previous,proposed,strict=True)])
        if (initial<CLEARANCE_M).any():raise ValueError('Unsafe observed old hand accepted by controller')
        expected=allocation_row_readback(a,i,meshes,previous,initial,requested)
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
        print('ALLOCATED_FLOOR_COP_CASE',episode,row['complete'],row['passed'],flush=True)
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
