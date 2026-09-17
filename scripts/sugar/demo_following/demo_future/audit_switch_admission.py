"""Read-only reference-switch termination reconstruction from saved real states.

Reproduce the existing causal reference transform and official term formulas.
This is a static compatibility screen, not simulated recovery or policy success.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scripts.sugar.smp.sugar_g1_box_schema import TRACKED_BODY_NAMES

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'generator_actual_branch_coverage_dense'


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[..., [1,2,3,0]])


def yaw(r):
    matrix = r.as_matrix()
    return Rotation.from_rotvec([0.,0.,np.arctan2(matrix[1,0],matrix[0,0])])


def evaluate_switch_state(state, phase, arm, bank, names):
    tracked = [names.index(n) for n in TRACKED_BODY_NAMES]
    torso = names.index('torso_link')
    ee = [names.index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
    body = state['robot_body_state_before_w'][phase,0]
    obj_actual = state['object_state_before_w'][phase,0]
    robot,obj = bank[arm]
    actual_anchor = rotation(body[torso,3:7])
    ref_anchor = rotation(robot['body_quat_w'][phase,torso])
    delta = yaw(actual_anchor) * yaw(ref_anchor).inv()
    shift = obj_actual[:3] - delta.apply(obj['obj_trans'][phase]); shift[2]=0.
    ref_body = delta.apply(robot['body_pos_w'][phase])+shift
    ref_anchor = delta * ref_anchor
    ref_object = delta.apply(obj['obj_trans'][phase])+shift
    relative_delta = yaw(actual_anchor * ref_anchor.inv())
    robot_anchor = body[torso,:3].copy(); robot_anchor[2]=ref_body[torso,2]
    relative_body = robot_anchor + relative_delta.apply(ref_body-ref_body[torso])
    ee_error = np.linalg.norm(relative_body[ee]-body[ee,:3],axis=-1)
    box_error = float(np.linalg.norm(ref_object-obj_actual[:3]))
    anchor_error = float(np.linalg.norm(ref_body[torso]-body[torso,:3]))
    gravity=np.array([0.,0.,-1.])
    anchor_ori_error = float(abs(ref_anchor.inv().apply(gravity)[2]-actual_anchor.inv().apply(gravity)[2]))
    ref_obj_rot = delta * Rotation.from_matrix(obj['obj_rot'][phase])
    object_ori_error = float((ref_obj_rot * rotation(obj_actual[3:7]).inv()).magnitude())
    terms=dict(trajectory_complete=phase+1>=len(robot['joint_pos']),anchor_ori=anchor_ori_error>.8,
        ee_body_pos=bool(np.any(ee_error>.3)),obj_pos=box_error>.3,obj_ori=object_ori_error>.8,anchor_pos=anchor_error>.3)
    return dict(terms=terms,all_terms_false=not any(terms.values()),box_position_error_m=box_error,
        ee_position_errors_m=ee_error.tolist(),anchor_position_error_m=anchor_error,
        anchor_projected_gravity_z_error=anchor_ori_error,object_orientation_error_rad=object_ori_error), ref_body[tracked],ref_object

def main():
    endpoint = json.loads((RUN / 'RESULT.json').read_text())
    if endpoint['all_phase_data_passed'] or endpoint['per_switch']['237']['admitted']:
        raise RuntimeError('This audit addresses the observed237pre-action failure')
    out = RUN / 'switch_admission_audit'; out.mkdir(exist_ok=False)
    names = np.load(RUN / 'switch_237/original/STARTUP.npz')['body_names'].tolist()
    tracked = [names.index(n) for n in TRACKED_BODY_NAMES]
    torso = names.index('torso_link')
    ee = [names.index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
    baseline_path = RUN / 'switch_318/original/TRACE.npz'
    baseline = dict(np.load(baseline_path))
    bank = []
    for folder in sorted((RUN / 'switch_237/motions').glob('data_*')):
        robot = dict(np.load(folder / 'robot_50hz.npz'))
        with (folder / 'obj_motion_global_50hz.pkl').open('rb') as f:
            obj = pickle.load(f)
        bank.append((robot,obj))
    def evaluate(state, phase, arm):
        return evaluate_switch_state(state, phase, arm, bank, names)
    checks, observed = {}, {}
    for phase in (197,237,277):
        for ai,arm in enumerate(('original','alternate')):
            folder=RUN/f'switch_{phase}'/arm;trace=dict(np.load(folder/'TRACE.npz'))
            row,body,obj=evaluate(trace,phase,ai)
            live=json.loads((folder/'SWITCH_TERMS_BEFORE_ACTION.json').read_text())
            key=f'{phase}_{arm}'
            checks[key+'_term_decisions_exact']=row['terms']==live
            checks[key+'_reference_body_reproduced']=bool(np.allclose(body,trace['reference_body_pos_w'][phase,0],rtol=0,atol=2e-6))
            checks[key+'_reference_object_reproduced']=bool(np.allclose(obj,trace['reference_object_pos_w'][phase,0],rtol=0,atol=2e-6))
            checks[key+'_unswitched_baseline_state_exact']=all(np.array_equal(trace[k][phase],baseline[k][phase]) for k in ('robot_body_state_before_w','object_state_before_w','joint_pos_before','joint_vel_before'))
            observed[key]=row
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    eligible=set(plan['phase_selection']['238']['eligible_phases'])
    scan={}
    for phase in range(223,254):
        rows={arm:evaluate(baseline,phase,ai)[0] for ai,arm in enumerate(('original','alternate'))}
        scan[str(phase)]=dict(source_future_clocks_disjoint=phase in eligible,arms=rows,
            static_compatible=all(v['all_terms_false'] for v in rows.values()))
    candidates=[int(p) for p,v in scan.items() if v['source_future_clocks_disjoint'] and v['static_compatible']]
    choice=min(candidates,key=lambda p:(abs(p-238),p)) if candidates and all(checks.values()) else None
    report=dict(execution_completed=True,checks=checks,checks_passed=all(checks.values()),observed=observed,
        candidate_phases=scan,selected_next_phase=choice,
        selection_rule='Nearest to238 among the original clock-eligible223..253 window with all original term formulas false for both references on the saved unswitched full607state; ties lower. No outcome-dependent threshold changes.',
        authoritative_formula_sources=['SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/terminations.py',
            'SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_tracker/base_tracker_env_cfg.py',
            'scripts/sugar/demo_following/demo_future/collect_refiner_pair.py'],
        unswitched_actual_state_source=str(baseline_path),new_optimizer_updates=0,new_physics_steps=0,
        scope='Static source-formula reconstruction checked against all six actual pre-action term decisions and aligned reference arrays. Candidate state comes before its recorded318reference transform. Passing is only switch compatibility; actual finite-horizon rollout and independent following remain unproven.237failure and all cumulative denominators remain unchanged.')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['checks_passed']:
        raise RuntimeError('Static reconstruction differs from the actual official reference/term records')
    print(json.dumps(dict(checks=checks,observed=observed,selected_next_phase=choice,candidates=candidates)),flush=True)


if __name__=='__main__':main()
