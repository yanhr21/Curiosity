"""Exact live-state official SMP feature/scoring adapter; no policy updates."""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import socket
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'experiments/demo_following/demo_future_smp_v1/refiner_pair_feasibility'
sys.path.insert(0, str(ROOT/'scripts/sugar/smp'))
sys.path.insert(0, str(ROOT/'MimicKit/mimickit'))
from run_selected_demo_tinymdm import build_feature_windows, load_clip, load_official_feature_functions
from run_conditional_taskwide_tinymdm import load_shared_prior, conditional_raw_energy
from sugar_g1_box_schema import G1_JOINT_NAMES, TRACKED_BODY_NAMES
from learning.tinymdm.tinymdm_model import TinyMDMModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--compare-run',type=Path,help='Compare already exported original-arm features on an identical common phase/batch/noise schedule')
    parser.add_argument('--common-output',type=Path)
    parser.add_argument('--window-start',type=int,default=0,help='Common saved-feature window start for matched comparison; no rescanning or best-window selection')
    parser.add_argument('--compare-arm',choices=('original','alternate'),default='original')
    args=parser.parse_args()
    run=args.run
    protocol=json.loads((run/'PROTOCOL.json').read_text())
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Official prior evaluation requires retained compute step')
    if args.compare_run:
        if args.common_output is None:raise ValueError('Specify a separate common output')
        out=args.common_output;out.mkdir(exist_ok=False)
        runs=[run,args.compare_run]
        features=[]
        for source in runs:
            result=json.loads((source/'official_smp/RESULT.json').read_text())
            if not result['frozen_unchanged'] or not result['exact_live_fields_used']:raise RuntimeError('Source SMP field audit failed')
            features.append(np.load(source/f'official_smp/{args.compare_arm}_features.npy',allow_pickle=False))
        start=args.window_start
        if start<0:raise ValueError('Window start must be nonnegative')
        count=min(len(x) for x in features)-start
        if count<=0:raise ValueError('No common actual feature windows at the declared start')
        device=torch.device('cuda:0')
        prior=load_shared_prior(ROOT/'experiments/demo_following/conditional_taskwide_smp_v1',device)
        if not isinstance(prior,TinyMDMModel):raise TypeError('Expected complete official TinyMDM')
        initial={k:v.clone() for k,v in prior.state_dict().items()}
        summaries=[]
        for index,array in enumerate(features):
            if array.shape[1:]!=(10,216):raise RuntimeError('Official feature shape drift')
            energies={str(label):conditional_raw_energy(prior,array[start:start+count],label,device,chunk_size=count) for label in (0,1)}
            np.savez(out/f'run{index}_energies.npz',**energies)
            summaries.append(dict(run=str(runs[index]),windows=count,mean_carry_energy=float(energies['0'].mean()),mean_kick_energy=float(energies['1'].mean()),carry_preferred_fraction=float((energies['0']<energies['1']).mean())))
        unchanged=all(torch.equal(v,initial[k]) for k,v in prior.state_dict().items())
        if not unchanged:raise RuntimeError('Frozen official prior changed')
        result=dict(execution_completed=True,frozen_unchanged=True,optimizer_updates=0,common_window_start_frames=[start,start+count-1],common_window_end_frames=[start+9,start+count+8],actual_frames_covered=count+9,identical_batch_shapes_and_score_seed_schedule=True,runs=summaries,scope='Task-class prior on identical phase ranges. Recomputed equal-size batches pair the Monte Carlo schedule; slicing independently scored unequal-size batches is not claimed as noise-matched. This is not selected-demo distance or physical-following success.')
        result['compared_arm']=args.compare_arm
        result['full_prior_parameter_count']=sum(p.numel() for p in prior.parameters())
        (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        return
    out = run/'official_smp'
    out.mkdir(exist_ok=False)
    device = torch.device('cuda:0')
    _, util = load_official_feature_functions()
    prior = load_shared_prior(ROOT/'experiments/demo_following/conditional_taskwide_smp_v1',device)
    if not isinstance(prior,TinyMDMModel):
        raise TypeError('Expected official TinyMDM')
    state = {k:v.detach().clone() for k,v in prior.state_dict().items()}
    details = {}
    for arm in protocol['arms']:
        trace = dict(np.load(run/arm/'TRACE.npz',allow_pickle=False))
        startup = dict(np.load(run/arm/'STARTUP.npz',allow_pickle=False))
        body_names = startup['body_names'].tolist()
        joint_names = startup['joint_names'].tolist()
        if len(set(body_names)) != len(body_names) or len(set(joint_names)) != len(joint_names):
            raise RuntimeError('Ambiguous live state names')
        body_ids = [body_names.index(name) for name in TRACKED_BODY_NAMES]
        joint_ids = [joint_names.index(name) for name in G1_JOINT_NAMES]
        # Stop before the recorded terminal transition. Never bridge resets.
        end = int(np.flatnonzero(trace['done'].reshape(-1))[0]) if trace['done'].any() else len(trace['done'])
        body = trace['robot_body_state_before_w'][:end,0][:,body_ids]
        obj_state = trace['object_state_before_w'][:end,0]
        robot = {'body_pos_w':body[...,:3], 'body_quat_w':body[...,3:7],
                 'body_lin_vel_w':body[...,7:10], 'body_ang_vel_w':body[...,10:13],
                 'joint_pos':trace['joint_pos_before'][:end,0][:,joint_ids],
                 'joint_vel':trace['joint_vel_before'][:end,0][:,joint_ids]}
        obj = {'obj_trans':obj_state[:,:3],
               'obj_rot':util.quat_to_matrix(torch.as_tensor(obj_state[:,[4,5,6,3]],device=device)).cpu().numpy(),
               'obj_lin_vel':obj_state[:,7:10], 'obj_ang_vel':obj_state[:,10:13]}
        features = build_feature_windows(robot,obj,device)
        np.save(out/f'{arm}_features.npy',features)
        energies = {str(label):conditional_raw_energy(prior,features,label,device) for label in (0,1)}
        np.savez(out/f'{arm}_energies.npz',**energies)
        after = np.arange(len(features)) >= protocol['switch_control_frame']
        details[arm] = {'feature_shape':list(features.shape),'actual_valid_frames':end,
                        'body_indices_from_live_names':body_ids,'joint_indices_from_live_names':joint_ids,
                        'mean_carry_energy':float(energies['0'].mean()),
                        'mean_kick_energy':float(energies['1'].mean()),
                        'carry_preferred_fraction':float((energies['0']<energies['1']).mean()),
                        'post_switch_carry_energy':float(energies['0'][after].mean()) if after.any() else None,
                        'post_switch_carry_preferred_fraction':float((energies['0'][after]<energies['1'][after]).mean()) if after.any() else None}
        print(json.dumps({'arm':arm,**details[arm]}),flush=True)
    # Source demos are task-prior context only; do not subtract these energies
    # and call the result a selected-demo distance.
    references = {}
    for source in protocol['reference_pair']:
        robot,obj = load_clip(ROOT/f'SUGAR/data/CarryBox/data_{source:03d}')
        features = build_feature_windows(robot,obj,device)[::10]
        energies = conditional_raw_energy(prior,features,0,device)
        references[str(source)] = {'sampled_windows':len(features),'mean_carry_energy':float(energies.mean())}
    changed = [k for k,v in prior.state_dict().items() if not torch.equal(v,state[k])]
    result = {'execution_completed':True,'exact_live_fields_used':True,'official_feature_shape':[10,216],
              'full_prior_parameter_count':sum(p.numel() for p in prior.parameters()),
              'post_switch_window_definition':'Window start is at or after selection; all ten window frames are post-selection.',
              'strict_official_model_loaded':True,'frozen_unchanged':not changed,'changed_state_keys':changed,
              'arms':details,'reference_task_prior_context':references,'optimizer_updates':0,
              'scope':'Official task-class SMP prior scoring on newly measured complete live PhysX states; not a selected-demo latent/distance, not proof of following, and not yet a policy reward.'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
