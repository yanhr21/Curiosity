"""Saved-only exact split of common-normal drift; no object GT or simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .analyze_shared_cop_kinematics import line_angle


def unit(x):
    return x / np.linalg.norm(x)


def degrees(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def run(root):
    source = root / 'cases/episode_5014/episode_5014.npz'
    previous = root / 'cop_kinematic_decomposition/RESULT.json'
    report = json.loads(previous.read_text())
    with np.load(source) as z:
        normals = z['validation_controller_fit_normal_w']
        poses = z['hand_pose_w']
        loads = z['normal_load_n'].reshape(-1, 2, 27).sum(2)
        all_centers = z['validation_controller_cop_world_m']
    rows = []
    for r in report['all_rows']:
        i = r['frame']
        n0 = np.asarray(r['common_normal_before'])
        n1 = np.asarray(r['common_normal_after'])
        old = normals[i]
        new = normals[i + 1]
        if max(np.max(abs(unit(old[0]-old[1])-n0)),
               np.max(abs(unit(new[0]-new[1])-n1))) > 1e-10:
            raise ValueError('Saved parent fit normals differ from CoP common normal')
        rotation = Rotation.from_quat(poses[i, :, 3:]) * Rotation.from_quat(poses[i-1, :, 3:]).inv()
        moved = rotation.apply(old)
        nr = unit(moved[0]-moved[1])
        # Recover the actual next center difference from saved observation.
        c = all_centers[i+1]
        delta = c[1]-c[0]
        p0 = np.eye(3)-np.outer(n0,n0)
        pr = np.eye(3)-np.outer(nr,nr)
        p1 = np.eye(3)-np.outer(n1,n1)
        rigid = (pr-p0)@delta
        local = (p1-pr)@delta
        expected = np.asarray(r['tangent_common_normal_basis_component_m'])
        err = float(np.max(abs(rigid+local-expected)))
        if err > 1e-12:
            raise ValueError('Normal transport identity failed')
        rows.append(dict(frame=i,command_s=r['command_s'],borrowed=r['borrowed'],
            common_normal_rigid_transported=nr.tolist(),
            common_normal_rigid_rotation_deg=degrees(n0,nr),
            common_normal_local_refit_rotation_deg=degrees(nr,n1),
            each_hand_actual_rotation_deg=np.degrees(rotation.magnitude()).tolist(),
            each_hand_local_refit_angle_deg=[degrees(unit(moved[h]),unit(new[h])) for h in (0,1)],
            rigid_transport_tangent_component_m=rigid.tolist(),
            local_refit_tangent_component_m=local.tolist(),
            angle_rigid_transport_contribution_deg=line_angle(delta,nr)-line_angle(delta,n0),
            angle_local_refit_contribution_deg=line_angle(delta,n1)-line_angle(delta,nr),
            outcome_load_n=loads[i].tolist(),
            identity_max_abs_error_m=err))
    groups = {}
    for name, spec in report['groups'].items():
        selected = [r for r in rows if spec['first_command_s'] <= r['command_s'] <= spec['last_command_s']
                    and (r['borrowed'] if name=='actual69_borrow' else not r['borrowed'] if name=='within_borrow_span_not_borrow' else True)]
        groups[name] = dict(clocks=len(selected),
            angle_rigid_transport_sum_deg=sum(r['angle_rigid_transport_contribution_deg'] for r in selected),
            angle_local_refit_sum_deg=sum(r['angle_local_refit_contribution_deg'] for r in selected),
            each_hand_actual_rotation_path_deg=np.sum([r['each_hand_actual_rotation_deg'] for r in selected],axis=0).tolist(),
            each_hand_local_refit_path_deg=np.sum([r['each_hand_local_refit_angle_deg'] for r in selected],axis=0).tolist(),
            outcome_both_loads_in_original_9_15_n=sum(bool(np.all((np.array(r['outcome_load_n'])>=9)&(np.array(r['outcome_load_n'])<=15))) for r in selected))
    out = root/'cop_kinematic_normal_decomposition'
    out.mkdir(exist_ok=False)
    result = dict(complete=True,groups=groups,rows=rows,
        scope='Ordered exact algebra on saved trajectory; rigid rotations are actual controlled motion, while local refit changes can depend on contact mechanics, translation, rotation and sample support. This is not an independent causal intervention.',
        normal_definition='normalize(left fitted world normal - right fitted world normal)',
        transport='R_actual[k] R_actual[k-1]^-1 applied separately to the old fitted normals; then recompute common normal',
        timing='command[k] fit consumes pose[k-1]; command[k+1] fit consumes actual pose[k]',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,previous,Path(__file__))},
        new_physics_controls=0,new_model_forwards=0,uses_object_gt=False)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Common fitted-normal drift split','',result['scope'],'',result['timing'],'',
        '|Stage|Controls|Rigid hand rotation contribution °|Local refit contribution °|Both outcome loads 9–15 N|',
        '|---|---:|---:|---:|---:|']
    for name,g in groups.items():
        lines.append(f"|{name}|{g['clocks']}|{g['angle_rigid_transport_sum_deg']:.8f}|{g['angle_local_refit_sum_deg']:.8f}|{g['outcome_both_loads_in_original_9_15_n']}|")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
