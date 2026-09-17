"""Saved-only CoP rigid/local-migration/normal-basis decomposition; no GT object.

Command[k] uses pose[k-1] and CoP[k]. Its outcome is observed as CoP[k+1].
The vector decomposition is an algebraic identity, not a causal mechanics fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def transported(points,old,new):
    result=np.zeros((2,3))
    for h in (0,1):
        local=Rotation.from_quat(old[h,3:]).inv().apply(points[h]-old[h,:3])
        result[h]=Rotation.from_quat(new[h,3:]).apply(local)+new[h,:3]
    return result


def line_angle(delta,normal):
    separation=float(delta@normal)
    return float(np.degrees(np.arctan2(np.linalg.norm(delta-normal*separation),separation)))


def run(root):
    # No partial-run trajectory claims: wait for the complete three-case result.
    result=json.loads((root/'RESULT.json').read_text())
    if not result['complete'] or len(result['cases'])!=3:raise ValueError('Require all3 completed cases')
    source=root/'cases/episode_5014/episode_5014.npz'
    with np.load(source) as z:a={k:z[k] for k in z.files}
    v=lambda key:a['validation_controller_'+key]
    output=root/'cop_kinematic_decomposition';output.mkdir(exist_ok=False)
    records=[];unavailable=[]
    for i,t in enumerate(a['timestamp_s'][:-1]):
        if not 34.<=t<=40. or i==0:continue
        if not v('cop_valid')[i:i+2].all():
            unavailable.append(dict(frame=i,command_s=float(t),reason='current_or_next_bilateral_CoP_invalid'))
            continue
        centers=v('cop_world_m')[i];nextcenters=v('cop_world_m')[i+1]
        old=a['hand_pose_w'][i-1];actual=a['hand_pose_w'][i]
        parent=v('cop_allocation_parent_pose_w')[i]
        rigid=transported(centers,old,actual)
        parent_rigid=transported(centers,old,parent)
        delta=centers[1]-centers[0];nextdelta=nextcenters[1]-nextcenters[0]
        e0=v('cop_tangent_error_m')[i];e1=v('cop_tangent_error_m')[i+1]
        n0=delta-e0;n0/=np.linalg.norm(n0)
        n1=nextdelta-e1;n1/=np.linalg.norm(n1)
        p0=np.eye(3)-np.outer(n0,n0);p1=np.eye(3)-np.outer(n1,n1)
        dr=rigid[1]-rigid[0];dp=parent_rigid[1]-parent_rigid[0]
        migration=nextcenters-rigid
        parent_component=p0@(dp-delta)
        executed_cop_component=p0@(dr-dp)
        migration_component=p0@(nextdelta-dr)
        basis_component=(p1-p0)@nextdelta
        reconstructed=parent_component+executed_cop_component+migration_component+basis_component
        residual=float(np.max(abs(reconstructed-(e1-e0))))
        if residual>1e-12:raise ValueError('Tangent decomposition algebra differs')
        if abs(line_angle(delta,n0)-v('cop_line_angle_deg')[i])>1e-8:
            raise ValueError('Recovered common fitted normal disagrees with recorded angle')
        committed=bool(v('floor_parent_transaction_committed')[i])
        if not committed:raise ValueError('This fixed interval includes a hold: parent decomposition needs an explicit hold convention')
        records.append(dict(frame=i,command_s=float(t),outcome_seen_command_s=float(a['timestamp_s'][i+1]),
            borrowed=bool(v('shared_borrow_applied')[i].any()),
            executed_cop_m=(v('cop_servo_velocity_m_s')[i]*.02).tolist(),
            centroid_world_change_m=(nextcenters-centers).tolist(),
            same_hand_material_point_rigid_world_change_m=(rigid-centers).tolist(),
            pressure_centroid_hand_relative_migration_world_m=migration.tolist(),
            common_normal_before=n0.tolist(),common_normal_after=n1.tolist(),
            common_normal_rotation_deg=float(np.degrees(np.arccos(np.clip(n0@n1,-1,1)))),
            tangent_error_before_m=e0.tolist(),tangent_error_after_m=e1.tolist(),
            tangent_parent_normal_and_alignment_component_m=parent_component.tolist(),
            tangent_executed_cop_translation_component_m=executed_cop_component.tolist(),
            tangent_pressure_centroid_migration_component_m=migration_component.tolist(),
            tangent_common_normal_basis_component_m=basis_component.tolist(),
            angle_sequence_deg=[line_angle(d,n) for d,n in ((delta,n0),(dp,n0),(dr,n0),(nextdelta,n0),(nextdelta,n1))],
            identity_max_abs_error_m=residual))
    groups={}
    for name,select in [('before_borrow',lambda r:r['command_s']<36.58),
                        ('actual69_borrow',lambda r:r['borrowed']),
                        ('within_borrow_span_not_borrow',lambda r:36.58<=r['command_s']<=37.96 and not r['borrowed']),
                        ('after_borrow',lambda r:r['command_s']>37.96)]:
        rows=[r for r in records if select(r)]
        if not rows:continue
        components={key:np.array([r[key] for r in rows]).sum(0).tolist() for key in
            ('tangent_parent_normal_and_alignment_component_m','tangent_executed_cop_translation_component_m',
             'tangent_pressure_centroid_migration_component_m','tangent_common_normal_basis_component_m')}
        groups[name]=dict(clocks=len(rows),first_command_s=rows[0]['command_s'],last_command_s=rows[-1]['command_s'],
            components_sum_world_m=components,
            angle_change_deg=sum(r['angle_sequence_deg'][-1]-r['angle_sequence_deg'][0] for r in rows),
            sequential_angle_increment_deg=np.diff(np.array([r['angle_sequence_deg'] for r in rows]),axis=1).sum(0).tolist(),
            absolute_local_migration_path_m=np.linalg.norm(np.array([r['pressure_centroid_hand_relative_migration_world_m'] for r in rows]),axis=2).sum(0).tolist())
    if groups['actual69_borrow']['clocks']!=69:raise ValueError('Unexpected actual borrow count')
    report=dict(scope='Exact saved kinematic identity for fixed5014; not a force-response regression or controller intervention.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,Path(__file__),root/'RESULT.json')},
        all_rows=records,groups=groups,unavailable_rows=unavailable,
        interval_control_denominator=len(records)+len(unavailable),
        angle_sequence_definition=['original','parent_normal_alignment_same_normal','actual_CoP_same_normal','measured_next_centers_same_normal','measured_next_centers_next_normal'],
        interpretation='Pressure centers are not persistent material points. Components depend on stated projection/order; normal-basis contribution is measured algebra, not an independently controlled causal mechanism.',
        complete_clock_mapping='command[k] consumes hand_pose[k-1]/centers[k]; endpoint measured_centers[k+1] follows actualpose[k].',
        uses_object_ground_truth=False,new_physics_controls=0,new_model_forwards=0,new_optimizer_updates=0)
    (output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Actual5014 CoP motion decomposition','',report['complete_clock_mapping'],'',report['interpretation'],'',
        '|Stage|Controls|Parent angle contribution deg|CoP translation deg|Local centroid migration deg|Common-normal change deg|Net angle deg|',
        '|---|---|---|---|---|---|---|']
    for name,g in groups.items():
        values='|'.join(f'{x:.8f}' for x in g['sequential_angle_increment_deg'])
        lines.append(f"|{name}|{g['clocks']}|{values}|{g['angle_change_deg']:.8f}|")
    lines+=['','No thresholds, travel budgets, old controller code, model or physics were changed.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
