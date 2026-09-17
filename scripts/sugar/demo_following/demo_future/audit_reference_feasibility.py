"""Fixed-phase TRAIN-reference feasibility before another actual future pair.

Uses a single causal rigid reference transform, never per-frame/window matching,
and never transforms recorded actual state or replaces executed actions.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT/'experiments/demo_following/demo_future_smp_v1/refiner_pair_feasibility'
sys.path.insert(0,str(ROOT/'MimicKit/mimickit'))
from util import torch_util as u


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--primary-source',type=int,default=45)
    parser.add_argument('--switch',type=int,default=197)
    args=parser.parse_args()
    run=args.run
    primary=args.primary_source
    a = np.load(run/'original/TRACE.npz')
    names = np.load(run/'original/STARTUP.npz')['body_names'].tolist()
    route = np.load(RUN.parent/'dataset_heading_local_v3/train/routing.npz')
    candidates = sorted(set(route['base_source_motion_id'][route['base_task']==0].tolist()))
    switch = args.switch
    primary_length=len(np.load(ROOT/f'SUGAR/data/CarryBox/data_{primary:03d}/robot_50hz.npz')['joint_pos'])
    live = a['robot_body_state_before_w'][switch,0]
    obj = a['object_state_before_w'][switch,0]
    torso = names.index('torso_link')
    ee = [names.index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
    qlive = torch.tensor(live[torso,[4,5,6,3]])
    box_rotation = u.quat_to_matrix(torch.tensor(obj[[4,5,6,3]])).numpy()
    rows = []
    for source in sorted(set(candidates+[45,96])):
        folder = ROOT/f'SUGAR/data/CarryBox/data_{source:03d}'
        robot = np.load(folder/'robot_50hz.npz')
        with (folder/'obj_motion_global_50hz.pkl').open('rb') as stream:
            ref = pickle.load(stream)
        length = len(robot['joint_pos'])
        frame = round(switch/(primary_length-1)*(length-1))
        qref = torch.tensor(robot['body_quat_w'][frame,torso,[1,2,3,0]])
        delta = u.quat_mul(u.calc_heading_quat(qlive),u.calc_heading_quat_inv(qref))
        rotation = u.quat_to_matrix(delta).numpy()
        shift = obj[:3]-np.asarray(ref['obj_trans'][frame])@rotation.T
        shift[2] = 0.
        body = robot['body_pos_w'][frame]@rotation.T+shift
        box = np.asarray(ref['obj_trans'][frame])@rotation.T+shift
        # Same formula as MotionCommand._update_command and bad_motion_body_pos:
        # reference torso heading is now matched; target torso XY is live XY,
        # while reference torso height is retained.
        anchor = live[torso,:3].copy();anchor[2]=body[torso,2]
        target_ee = anchor+(body[ee]-body[torso])
        ee_errors = np.linalg.norm(target_ee-live[ee,:3],axis=-1)
        target_box_rotation = rotation@np.asarray(ref['obj_rot'][frame])
        angle = np.arccos(np.clip((np.trace(target_box_rotation@box_rotation.T)-1)/2,-1,1))
        target_torso_rotation = rotation@u.quat_to_matrix(qref).numpy()
        actual_torso_rotation = u.quat_to_matrix(qlive).numpy()
        gravity_z_error = abs(float(target_torso_rotation[2,2]-actual_torso_rotation[2,2]))
        future_frame = round(min(1.,(switch+140)/(primary_length-1))*(length-1))
        displacement = (np.asarray(ref['obj_trans'][future_frame])-np.asarray(ref['obj_trans'][frame]))@rotation.T
        row = dict(source=source,train_candidate=source in candidates,reference_frame=frame,
                   rotation_xyzw=delta.tolist(),translation_xyz=shift.tolist(),
                   box_distance_m=float(np.linalg.norm(box-obj[:3])),
                   torso_distance_m=float(np.linalg.norm(body[torso]-live[torso,:3])),
                   official_relative_effector_distances_m=ee_errors.tolist(),
                   box_orientation_error_rad=float(angle),torso_gravity_z_error=gravity_z_error,
                   reference_future_box_displacement_2p8s=displacement.tolist())
        row['within_original_position_orientation_limits'] = bool(row['box_distance_m']<.3 and row['torso_distance_m']<.3 and max(ee_errors)<.3 and angle<.8 and gravity_z_error<.8)
        row['conservative_compatibility'] = bool(row['box_distance_m']<.15 and row['torso_distance_m']<.15 and max(ee_errors)<.2 and angle<.4 and gravity_z_error<.4)
        rows.append(row)
    baseline = next(r for r in rows if r['source']==primary)
    for row in rows:
        row['future_box_displacement_difference_from_primary_m'] = float(np.linalg.norm(np.asarray(row['reference_future_box_displacement_2p8s'])-baseline['reference_future_box_displacement_2p8s']))
    feasible = [r for r in rows if r['train_candidate'] and r['source']!=primary and r['conservative_compatibility'] and r['future_box_displacement_difference_from_primary_m']>=.15]
    # Predeclared deterministic selection: lowest maximum normalized CURRENT
    # compatibility error among TRAIN references with different known futures.
    def cost(r):
        return max(r['box_distance_m']/.15,r['torso_distance_m']/.15,max(r['official_relative_effector_distances_m'])/.2,r['box_orientation_error_rad']/.4,r['torso_gravity_z_error']/.4)
    feasible.sort(key=lambda r:(cost(r),r['source']))
    result = dict(execution_completed=True,physics_rerun=False,phase=switch/(primary_length-1),primary_source=primary,switch_control_frame=switch,
                  candidate_count=len(candidates),selected_train_reference=feasible[0]['source'] if feasible else None,
                  feasible_count=len(feasible),rows=rows,
                  alignment_rule='Single fixed yaw and horizontal translation from causal live torso heading and box XY at the fixed declared switch frame; preserve source height and all future relative motion. Same rule for original and alternate. No spatial change to actual state.',
                  automatic_next_action='Run matched original/repeat/selected-reference frozen-teacher pair with the same one-time alignment rule, original termination thresholds and450-frame budget.' if feasible else 'No conservative paired reference at this fixed phase; inspect native-reference teacher data before spending another pair.',
                  scope='Exploratory TRAIN-only reference feasibility selection, not an independent held-out following evaluation or proof of physical feasibility. Original45/96 failure remains recorded.')
    (run/'TRAIN_REFERENCE_FEASIBILITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    print(json.dumps(feasible[:5]),flush=True)


if __name__=='__main__':
    main()
