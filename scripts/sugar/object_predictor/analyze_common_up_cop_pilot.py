"""Full-denominator saved-only response-qualification comparison; no forward."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import run_common_up_cop_pilot as pilot
from .analyze_prelift_cop_pilot import read, summarize


def response_summary(a):
    v=lambda key:a['validation_controller_'+key]
    t=a['timestamp_s'];gain=v('response_gain_m_per_ns');upper=v('response_upper_n_m')
    snapshots=[]
    for clock in (8,12,16,20,24,28,32,36,40,44,48):
        i=int(np.argmin(abs(t-clock)))
        snapshots.append(dict(time_s=float(t[i]),gain=gain[i].tolist(),upper_n_m=upper[i].tolist(),
            valid_slopes=v('response_samples')[i].tolist(),window_pure=v('response_window_pure')[i].tolist(),
            accepted=v('response_secant_accepted')[i].tolist(),candidate_base=v('response_base_fallback')[i].tolist(),
            cop_angle_deg=float(v('cop_line_angle_deg')[i]),cop_travel_m=v('cop_travel_m')[i].tolist(),
            ready_s=float(v('ready_seconds')[i]),motion_elapsed_s=float(v('motion_elapsed_s')[i])))
    candidate_base=v('response_base_fallback').astype(bool)
    return dict(qualified_secants=v('response_qualified_secants_total')[-1].tolist(),
        rejected_mixed_secants=v('response_rejected_mixed_secants_total')[-1].tolist(),
        pure_command_frames=v('response_command_pure').sum(0).tolist(),
        pure_complete_window_frames=v('response_window_pure').sum(0).tolist(),
        candidate_base_frames=candidate_base.sum(0).tolist(),
        applied_gain_exact_base_frames=(gain==.000025).sum(0).tolist(),
        candidate_base_but_applied_ramping_frames=(candidate_base&(gain<.000025)).sum(0).tolist(),
        upper_nonzero_with_insufficient_samples_frames=((upper>0)&(v('response_samples')<5)).sum(0).tolist(),
        applied_gain_min_max=np.stack([gain.min(0),gain.max(0)]).tolist(),
        upper_n_m_max=upper.max(0).tolist(),fixed_clock_snapshots=snapshots,
        interpretation='Candidate-base flag describes upper validity, not equality of actual gain to base during the preserved upward ramp. Pure command windows do not certify a stationary object or material stiffness.')


def saved_initial_certificate(folder,config):
    """Recompute full triangle floor/object separation from saved actual poses."""
    from scipy.spatial.transform import Rotation
    from sugar_newton.hand.patches import load_hand_mesh
    from .geometry import load_sugar_outer_box
    from .qualify_common_up_geometry import initial_poses
    raw=json.loads((folder/'INITIAL_GEOMETRY.json').read_text())
    if raw['frame']!=0 or raw['time_s']!=0 or not raw['passed']:
        raise ValueError('Actual prephysics certificate absent')
    controller,expected=initial_poses(config)
    poses=np.asarray(raw['actual_hand_poses_w']);obj=np.asarray(raw['actual_object_pose_w'])
    if not np.allclose(poses,expected,rtol=0,atol=2e-7):raise ValueError('Actual initial frame differs')
    vertices,_=load_sugar_outer_box();vertices*=np.asarray(config['scale'],np.float32)
    objects=Rotation.from_quat(obj[3:]).apply(vertices)+obj[:3]
    rows=[]
    for i,side in enumerate(('left','right')):
        rotation=Rotation.from_quat(poses[i,3:]);normal=rotation.apply(controller.support_normals[i])
        mesh=load_hand_mesh(side)
        hand=rotation.apply(np.asarray(mesh.vertices,np.float32))+poses[i,:3]
        floor=float(hand[:,2].min());gap=float((objects@normal).min()-(hand@normal).max())
        recorded=raw['hands'][i]
        if (recorded['side']!=side or abs(recorded['min_full_hand_z_m']-floor)>1e-12
                or abs(recorded['full_triangle_separating_gap_m']-gap)>1e-12
                or floor<=.002001 or gap<=.006001):
            raise ValueError('Actual full-mesh initial separation replay failed')
        rows.append(dict(side=side,full_hand_min_z_m=floor,full_triangle_object_gap_m=gap))
    return dict(passed=True,hands=rows,actual_object_min_z_m=float(objects[:,2].min()),
        GT_used_only_for_initial_safety_evaluation=True,physics_controls=0)


def run(root,output):
    declared=pilot.validate_protocol(root)
    result=json.loads((root/'RESULT.json').read_text())
    baseline=json.loads((pilot.BASELINE/'RESULT.json').read_text())
    for name,value in [('candidate',result),('baseline',baseline)]:
        if not value['complete'] or tuple(c['episode'] for c in value['cases'])!=pilot.EPISODES:
            raise ValueError('Require complete ordered three-case '+name)
    if baseline['qualification_passed'] or sum(c['physical_passed'] for c in baseline['cases'])!=2:
        raise ValueError('Declared completed2/3 qualified-CoP baseline changed')
    if output.exists():raise FileExistsError('Preserve previous readback: '+str(output))
    rows=[];arrays={}
    for config in declared['configurations']:
        episode=config['episode'];row=dict(episode=episode)
        folder=root/'cases'/f'episode_{episode}'
        validated=pilot.validate_case(folder,config,declared)
        original_record=next(c for c in result['cases'] if c['episode']==episode)
        for key in ('physical_passed','physical_checks','late_mass_available','passed'):
            if validated[key]!=original_record[key]:raise ValueError('Original qualification decision changed: '+key)
        row['complete_new_case_revalidation']=validated
        row['actual_initial_geometry_sha256']=pilot.sha(folder/'INITIAL_GEOMETRY.json')
        row['independent_actual_initial_fullmesh_replay']=saved_initial_certificate(folder,config)
        for name,source in [('baseline',pilot.BASELINE),('candidate',root)]:
            summary,geometry=summarize(source/'cases'/f'episode_{episode}',config)
            row[name]=summary
            arrays.update({f'{episode}_{name}_{key}':value for key,value in geometry.items()})
        a=read(folder/f'episode_{episode}.npz')
        row['response_qualification']=response_summary(a)
        if any(row['response_qualification']['upper_nonzero_with_insufficient_samples_frames']):
            raise ValueError('Stale upper survived new validity rule')
        # Preserve all2400 actual numeric evidence, even for physical FAIL.
        for key in ('response_command_rotation_rad','response_command_cop_travel_m',
            'response_command_phase_delta_s','response_command_pure','response_window_pure',
            'response_secant_accepted','response_qualified_secants_total',
            'response_rejected_mixed_secants_total','response_base_fallback',
            'response_gain_m_per_ns','response_upper_n_m','response_samples'):
            arrays[f'{episode}_{key}']=a['validation_controller_'+key]
        arrays[f'{episode}_timestamp_s']=a['timestamp_s']
        rows.append(row)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'ACTUAL_RESPONSE_READBACK.npz',**arrays)
    report=dict(complete=True,planned_cases=3,actual_cases=3,controls_per_case=2400,
        baseline_physical_passes=2,new_physical_passes=sum(r['candidate']['physical_passed'] for r in rows),
        new_physical_and_late_passes=sum(r['candidate']['physical_passed'] and r['candidate']['late_mass_available'] for r in rows),
        cases=rows,all_case_gain_upper_samples_and_window_telemetry_replayed=True,
        all_original12_and_late_decisions_preserved=True,all_actual_initial_mesh_certificates_revalidated=True,model_forwards=0,physics_controls=0,optimizer_updates=0,
        source_sha256={str(p):pilot.sha(p) for p in (Path(__file__),
            Path(__file__).with_name('analyze_prelift_cop_pilot.py'),root/'PROTOCOL.json',root/'INITIAL_GEOMETRY_CPU.json',root/'RESULT.json',
            pilot.BASELINE/'PROTOCOL.json',pilot.BASELINE/'RESULT.json')},
        limits='Only initial hand tangent roll changes relative to qualified CoP2/3. Static placement qualification is not carrying success. No automatic16 qualification, model training, force-target changes, timeout relaxation or deleted failed cases. The original15/16 corpus and blind-control failures remain separate.')
    pilot.write(output/'RESULT.json',report)
    lines=['# Common-up initial hand frames: actual three-case comparison','',
        '|Case|Physical old→new|Late mass old→new|Peak N old→new|Lift time old→new|Qualified samples L/R|Candidate-base frames L/R|',
        '|---|---|---|---|---|---|---|']
    for row in rows:
        old=row['baseline'];new=row['candidate'];q=row['response_qualification']
        lines.append(f"|{row['episode']}|{old['physical_passed']}→{new['physical_passed']}|{old['late_mass_available']}→{new['late_mass_available']}|{old['physical_values']['peak_hand_load_n']:.2f}→{new['physical_values']['peak_hand_load_n']:.2f}|{old['events']['lift_start_s']}→{new['events']['lift_start_s']}|{q['qualified_secants']}|{q['candidate_base_frames']}|")
    lines+=['',f"Physical {report['new_physical_passes']}/3; physical plus fixed late mass {report['new_physical_and_late_passes']}/3. All three complete cases retained.",'',
        'Each actual prephysics INITIAL_GEOMETRY certificate was checked against declared common-up poses; first command must have no roll jump. All2400 gain/upper/sample values and command-window flags were replayed from actual previous measurements. All12 original physical decisions and fixed frame2381/H32 mass availability were recomputed. Actual command rotation is replayed for2399 saved input rows; the initial unsaved input remains explicitly outside that geometry replay.',
        '',q['interpretation'],'',report['limits']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True,physical=report['new_physical_passes'],physical_and_late=report['new_physical_and_late_passes'],denominator=3,output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.output)
