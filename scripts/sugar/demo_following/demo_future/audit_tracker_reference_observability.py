"""Offline rigid-translation observability diagnostic using full official actors.

Counterfactual tensor queries only: no new PhysX trajectory, model or training.
The current robot, box and entire causal history translate together while the
selected reference stays fixed. Local proprioception and robot-to-box geometry
are invariant; the teacher's reference-position terms change analytically.
"""
import ast
import json
from pathlib import Path
import re

import numpy as np
import torch
from rsl_rl.networks.mlp import MLP

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/demo_following/demo_future_smp_v1'
RUN=BASE/'online_tracker_future128'


def term_slices(log,group):
    section=log.split(f"Active Observation Terms in Group: '{group}'",1)[1]
    section=section.split('Active Observation Terms in Group:',1)[0]
    rows=re.findall(r'\|\s*\d+\s*\|\s*(\w+)\s*\|\s*\((\d+),\)\s*\|',section)
    result={};offset=0
    for name,size in rows:
        result[name]=slice(offset,offset+int(size));offset+=int(size)
    return result,offset


def main():
    torch.set_num_threads(1)
    out=RUN/'reference_translation_diagnostic';out.mkdir(exist_ok=False)
    source=ROOT/'external/IsaacLab/source/isaaclab/isaaclab/utils/math.py'
    function=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='quat_apply_inverse')
    function.decorator_list=[];namespace={'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    inverse=namespace['quat_apply_inverse']
    log=(BASE/'online128_same_state_teacher_original.log').read_text()
    policy_terms,policy_size=term_slices(log,'policy');teacher_terms,teacher_size=term_slices(log,'teacher')
    if (policy_size,teacher_size)!=(510,890):raise RuntimeError('Actual observation layout differs')
    teacher_path=ROOT/'experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt'
    actors={};initial={}
    for name,path,width in (('student',RUN/'training/model_383.pt',798),('teacher',teacher_path,890)):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        state={k.removeprefix('actor.'):v for k,v in payload['model_state_dict'].items() if k.startswith('actor.')}
        model=MLP(input_dim=width,output_dim=29,hidden_dims=[512,256,128],activation='elu')
        model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
        actors[name]=model;initial[name]={k:v.clone() for k,v in model.state_dict().items()}
    summaries={}
    for arm in ('original','alternate'):
        path=RUN/('actual_state_teacher_'+arm)
        if not json.loads((path/'TEACHER_INTERFACE_AUDIT.json').read_text())['passed']:raise RuntimeError('Actual teacher interface failed')
        raw=dict(np.load(path/'TRACE.npz',allow_pickle=False))
        original=dict(np.load(RUN/'evaluation'/arm/'TRACE.npz',allow_pickle=False))
        if not all(np.array_equal(v,raw[k]) for k,v in original.items()):raise RuntimeError('Teacher queries changed actual trajectory')
        valid=~raw['done'].reshape(-1)
        names=np.load(path/'STARTUP.npz',allow_pickle=False)['body_names'].tolist()
        anchor=torch.from_numpy(raw['robot_body_state_before_w'][valid,0,names.index('torso_link')])
        obj=torch.from_numpy(raw['object_state_before_w'][valid,0])
        policy=torch.from_numpy(raw['teacher_observation'][valid,0])
        future=torch.from_numpy(raw['planned_reference_command'][valid,0].reshape(-1,288))
        teacher=torch.from_numpy(raw['bcppo_teacher_observation'][valid,0])
        teacher_labels=torch.from_numpy(raw['same_world_teacher_action'][valid,0])
        local_box=inverse(anchor[:,3:7],obj[:,:3]-anchor[:,:3])
        local_box_error=float((local_box-policy[:,policy_terms['obj_pos_b']]).abs().max())
        with torch.inference_mode():
            student_action=actors['student'](torch.cat([policy,future],dim=1))
            teacher_action=actors['teacher'](teacher)
            teacher_error=float((teacher_action-teacher_labels).abs().max())
            student_error=float((student_action-torch.from_numpy(raw['requested_action'][valid,0])).abs().max())
            if max(local_box_error,teacher_error,student_error)>1e-4:raise RuntimeError('Recorded/full-actor baseline reconstruction differs')
            shifts=[];arrays={}
            for shift_y in (-.15,-.05,.05,.15):
                shift=torch.zeros_like(obj[:,:3]);shift[:,1]=shift_y
                changed_policy=policy.clone()
                changed_policy[:,policy_terms['obj_pos_b']]=inverse(anchor[:,3:7],(obj[:,:3]+shift)-(anchor[:,:3]+shift))
                changed_teacher=teacher.clone()
                for term,quat in (('motion_anchor_pos_b_future',anchor[:,3:7]),('ref_obj_pos_b_future',obj[:,3:7])):
                    target=changed_teacher[:,teacher_terms[term]].reshape(-1,8,3)
                    target-=inverse(quat,shift)[:,None,:]
                predicted=actors['student'](torch.cat([changed_policy,future],dim=1))
                target=actors['teacher'](changed_teacher)
                target_delta=(target-teacher_action).square().mean(dim=1)
                arrays[str(shift_y)+'_teacher_delta_mse']=target_delta.numpy()
                shifts.append(dict(world_y_shift_m=shift_y,student_input_max_difference=float((changed_policy-policy).abs().max()),student_action_max_difference=float((predicted-student_action).abs().max()),teacher_action_difference_mse_mean=float(target_delta.mean()),teacher_action_difference_mse_p90=float(torch.quantile(target_delta,.9)),equal_input_two_label_mse_floor_mean=float(target_delta.mean()/4)))
        np.savez(out/(arm+'_counterfactual_teacher_differences.npz'),**arrays)
        summaries[arm]=dict(valid_actual_source_frames=int(valid.sum()),baseline_local_box_error=local_box_error,baseline_teacher_action_error=teacher_error,baseline_student_action_error=student_error,shifts=shifts)
    unchanged=all(torch.equal(v,initial[name][k]) for name,actor in actors.items() for k,v in actor.state_dict().items())
    result=dict(execution_completed=True,full_actors_unchanged=unchanged,optimizer_updates=0,physical_transitions_created=0,official_math_source=str(source),actual_observation_layout_source='online128_same_state_teacher_original.log',arms=summaries,
                invariant_terms='Selected current/future reference commands fixed; constant translation of complete actual history leaves angular velocity, joints, past actions, gravity and box orientation in robot frame unchanged. Box position in robot frame is independently recomputed.',
                teacher_transform='Only current-frame-relative future anchor/object position blocks change by -R_inverse*translation, exactly as the official observation formulas. Other teacher terms are translation-invariant.',
                lower_bound_scope='Analytical deterministic equal-input two-label MSE floor is teacher label difference MSE/4. Floating-point local-position cancellation is reported separately; copied invariant terms are justified by formulas, not claimed as a second physical observation.',
                scope='Synthetic rigid-translation diagnostic grounded in actual recorded states. Does not prove teacher recovery, unique cause of late drift, feasibility of a new controller, or actual counterfactual physics. No input or parameter has been added to the deployed student.')
    result['numerical_diagnostic_passed']=unchanged and all(x['student_input_max_difference']<1e-5 and x['student_action_max_difference']<1e-4 for arm in summaries.values() for x in arm['shifts'])
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['numerical_diagnostic_passed']:raise RuntimeError('Translation invariance numerical check failed')


if __name__=='__main__':main()
