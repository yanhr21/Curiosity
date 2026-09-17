"""Full144-row preservation checks for the official measured-robot input adapter."""
import argparse
import json
import os
import socket
from pathlib import Path

import dill
import hydra
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE,write
from scripts.sugar.demo_following.demo_future.train_generator_geometry import training_config
from scripts.sugar.demo_following.demo_future.generator_measured_robot import WEIGHT_KEY,EXTRA_KEY,extend_robot_state_dict,original_robot_slice
from scripts.sugar.demo_following.demo_future.preflight_generator_latent_replay import same_rng,gradients
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state

RUN=BASE/'matched_generator_branch_measured_robot32512'
SOURCE=BASE/'matched_generator_branch_self_replay01_gaps512'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    device=torch.device(parser.parse_args().device)
    if device.type=='cuda':assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8 if device.type=='cuda' else 1)
    plan=json.loads((RUN/'PROTOCOL.json').read_text());old=json.loads((SOURCE/'PROTOCOL.json').read_text())
    output=Path(plan['robot_preflight_directory'])/('GENERATED_BF16_PREFLIGHT.json' if device.type=='cuda' else 'GENERATED_CPU_PREFLIGHT.json')
    output.parent.mkdir(exist_ok=True);assert not output.exists()
    decision=json.loads(Path(plan['input_conditioning_decision']).read_text())
    assert decision['checks_passed'] and decision['prepare_existing_robot_input_extension']
    checks={k+'_settings_unchanged':plan[k]==old[k] for k in ('branch_dataset','batch_size','data','normalizer_state','seed','epochs','actual_optimizer_updates','optimizer','scheduler','frozen_evaluation','paired_objective','generated_state_objective','noise_coupling','goal_condition_mode')}
    dataset=hydra.utils.instantiate(training_config(plan,'demo_geometry').task.dataset)
    reference_dataset=hydra.utils.instantiate(training_config(old,'demo_geometry').task.dataset)
    checks['all_replay_arrays_exact']=set(dataset.replay)==set(reference_dataset.replay) and all(np.array_equal(v,reference_dataset.replay[k]) for k,v in dataset.replay.items())
    checks['full144rows18realTRAIN']=len(dataset)==144 and dataset.real_case_count==18
    samples=[dataset[i] for i in range(144)]
    checks['all144_original_samples_exact']=all(all((all(torch.equal(v[n],reference_dataset[i][k][n]) for n in v) if isinstance(v,dict) else torch.equal(v,reference_dataset[i][k])) for k,v in s.items()) for i,s in enumerate(samples))
    torch.manual_seed(plan['seed']);order=next(iter(torch.utils.data.DataLoader(dataset,batch_size=144,shuffle=True,num_workers=0)))['sample_index'].tolist()
    torch.manual_seed(plan['seed']);other=next(iter(torch.utils.data.DataLoader(reference_dataset,batch_size=144,shuffle=True,num_workers=0)))['sample_index'].tolist()
    checks['full144_batch_order_exact']=order==other
    def collate(indices):
        rows=[samples[i] for i in indices]
        return {k:({n:torch.stack([s[k][n] for s in rows]).to(device) for n in rows[0][k]} if isinstance(rows[0][k],dict) else torch.stack([s[k] for s in rows]).to(device)) for k in rows[0]}
    batch=collate(order);plain=collate(range(144));statistics={};optimizer_bindings={}
    def projected_equal(left,right):return set(right)==set(left)|{EXTRA_KEY} and all(torch.equal(v,right[k]) for k,v in left.items())
    def loss_grad(policy,seed=272240):
        captured=[]
        def capture(module,args,kwargs,result):captured.append((args[0].detach().clone(),args[1].detach().clone(),result[0].detach().clone()))
        handle=policy.model.register_forward_hook(capture,with_kwargs=True)
        policy.train();policy.generated_state_step=0;torch.manual_seed(seed)
        try:
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):loss=policy.compute_loss(batch,training=True)
        finally:handle.remove()
        values=gradients(loss,policy)
        return float(loss.detach()),values,rng_state(device),captured,dict(policy.last_paired_loss)
    for arm in ('zero_context','demo_geometry'):
        for snapshot in ('initial','learned'):
            policies=[];factory_rng=[]
            for settings in (old,plan):
                torch.manual_seed(272241)
                policy=hydra.utils.instantiate(training_config(settings,arm).policy)
                factory_rng.append(rng_state(torch.device('cpu')))
                policy.set_normalizer(dataset.get_normalizer());policy.normalizer.requires_grad_(False)
                policies.append(policy)
            key=arm+'_'+snapshot;checks[key+'_factory_rng_exact']=same_rng(*factory_rng)
            if snapshot=='learned':
                with (SOURCE/arm/'checkpoints/endpoint.ckpt').open('rb') as f:state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
                policies[0].load_state_dict(state,strict=True);policies[1].load_state_dict(extend_robot_state_dict(state),strict=True)
            original,current=policies
            saved={k:v.cpu().clone() for k,v in original.state_dict().items()}
            expected=extend_robot_state_dict(saved)
            checks[key+'_all_original_weights_normalizer_exact']=projected_equal(saved,current.state_dict())
            checks[key+'_exact_zero8192_columns']=current.state_dict()[WEIGHT_KEY].shape==(256,36) and current.state_dict()[EXTRA_KEY].shape==(256,32) and not bool(current.state_dict()[EXTRA_KEY].count_nonzero())
            checks[key+'_full_model_counts']=sum(p.numel() for p in original.parameters())==8319216 and sum(p.numel() for p in current.parameters())==8327408
            # The official AdamW grouping must include the widened input exactly once.
            optimizers=[p.get_optimizer(**plan['optimizer']) for p in policies]
            bindings=[];named_ids=[]
            for p,opt in zip(policies,optimizers):
                names={id(v):n for n,v in p.named_parameters() if v.requires_grad}
                groups=[[names[id(v)] for v in group['params']] for group in opt.param_groups]
                bindings.append(groups)
                named_ids.append([dict(name=names[id(value)],parameter_id=index) for group,state_group in zip(opt.param_groups,opt.state_dict()['param_groups']) for value,index in zip(group['params'],state_group['params'])])
                checks[key+('_new' if p is current else '_old')+'_optimizer_all_parameters_once']=len([n for g in groups for n in g])==len(names) and set(n for g in groups for n in g)==set(names.values())
            checks[key+'_official_adam_groups_and_settings_exact']=bindings[0]==[[n for n in group if n!=EXTRA_KEY] for group in bindings[1]] and all({k:v for k,v in a.items() if k!='params'}=={k:v for k,v in b.items() if k!='params'} for a,b in zip(optimizers[0].param_groups,optimizers[1].param_groups))
            optimizer_bindings[key]=dict(old=named_ids[0],new=named_ids[1])
            del optimizers
            for p in policies:p.to(device)
            left,right=[loss_grad(p) for p in policies]
            checks[key+'_full_combined_loss_records_exact']=left[0]==right[0] and left[4]==right[4]
            checks[key+'_every_original_parameter_gradient_exact']=projected_equal(left[1],right[1])
            checks[key+'_nonzero_finite_new_column_gradient']=bool(right[1][EXTRA_KEY].count_nonzero()) and all(bool(torch.isfinite(v).all()) for v in right[1].values())
            checks[key+'_all_noisy_inputs_times_forwards_exact']=len(left[3])==len(right[3]) and all(torch.equal(a,b) for l,r in zip(left[3],right[3]) for a,b in zip(l,r))
            checks[key+'_next_rng_exact']=same_rng(left[2],right[2])
            checks[key+'_zero_context_demo_columns_zero']=arm!='zero_context' or not bool(right[1]['obs_encoder.target_state_net.0.weight'][:,9:].count_nonzero())
            statistics[key]=dict(loss=right[0],new_column_gradient_norm=float(right[1][EXTRA_KEY].double().square().sum().sqrt()),full_forward_count=len(right[3]))
            del left,right
            # Match actual official inference, restoring its original CPU scheduler constants.
            for p in policies:p.eval();p.noise_scheduler.alphas_cumprod=p.generated_state_alphas_cpu.clone()
            eval_phases=plan['data']['train']['phases'] if device.type=='cuda' else plan['data']['train']['phases'][:1]
            for phase in eval_phases:
                pi=plan['data']['train']['phases'].index(phase)
                variant='initial_shared_goal_zero' if snapshot=='initial' else ('trained_zero_context' if arm=='zero_context' else 'trained_demo_geometry_correct')
                with np.load(SOURCE/f'frozen_evaluation/phase_{phase}/{variant}.npz') as a:reference=a['predictions'][:2].copy()
                for si,seed in enumerate(plan['frozen_evaluation']['sample_seeds'][:2]):
                    for branch in (0,1):
                        obs={k:v[2*pi+branch:2*pi+branch+1] for k,v in plain['obs'].items()}
                        with torch.no_grad():
                            torch.manual_seed(seed);a=original.predict_action(obs)
                            torch.manual_seed(seed);b=current.predict_action(obs)
                        checks[f'{key}_{phase}_{seed}_{branch}_full_sampler_exact']=torch.equal(a,b)
                        if device.type=='cuda':checks[f'{key}_{phase}_{seed}_{branch}_saved_primary_exact']=np.array_equal(b.cpu().numpy()[0],reference[si,branch])
            checks[key+'_complete_states_unchanged']=all(torch.equal(v.cpu(),saved[k]) for k,v in original.state_dict().items()) and all(torch.equal(v.cpu(),expected[k]) for k,v in current.state_dict().items())
            write(output.with_name(output.name+'.partial'),dict(checks=checks,statistics=statistics))
            assert all(checks.values()),[k for k,v in checks.items() if not v]
            print(json.dumps(dict(state=key,checks=len(checks),statistics=statistics[key])),flush=True)
            if arm=='demo_geometry' and snapshot=='learned':break
            del policies,original,current
    for ti in range(16):
        values=[];grads=[];streams=[]
        for p in (original,current):
            p.eval();torch.manual_seed(272240)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):loss=p.compute_generated_state_loss(plain,False,ti)
            values.append(float(loss.detach()));grads.append(gradients(loss,p));streams.append(rng_state(device))
        checks[f'all144_aux_time{ti}_loss_old_gradient_rng_exact']=values[0]==values[1] and projected_equal(grads[0],grads[1]) and same_rng(*streams)
        checks[f'all144_aux_time{ti}_new_grad_finite']=all(bool(torch.isfinite(v).all()) for v in grads[1].values())
    randint=torch.randint
    def all_times(low,high,size,**kwargs):
        assert (low,high,size)==(0,50,(144,))
        return torch.arange(144,device=kwargs.get('device'))%50
    try:
        torch.randint=all_times
        for clock in (0,1,15,16,511):
            values=[];grads=[];streams=[];records=[]
            for p in (original,current):
                p.train();p.generated_state_step=clock;torch.manual_seed(272240)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):loss=p.compute_loss(plain,True)
                values.append(float(loss.detach()));grads.append(gradients(loss,p));streams.append(rng_state(device));records.append(dict(p.last_paired_loss))
            checks[f'all50_q_clock{clock}_full_loss_gradient_rng_exact']=values[0]==values[1] and records[0]==records[1] and projected_equal(grads[0],grads[1]) and same_rng(*streams)
            checks[f'all50_q_clock{clock}_aux_clock_exact']=all(p.generated_state_step==clock+1 for p in (original,current)) and records[1]['generated_time_index']==clock%16
    finally:torch.randint=randint
    checks['final_full_states_unchanged']=all(torch.equal(v.cpu(),saved[k]) for k,v in original.state_dict().items()) and all(torch.equal(v.cpu(),expected[k]) for k,v in current.state_dict().items())
    checks['no_parameter_grad_accumulation']=all(p.grad is None for policy in (original,current) for p in policy.parameters())
    report=dict(execution_completed=True,passed=all(checks.values()),checks=checks,statistics=statistics,optimizer_bindings=optimizer_bindings,parameter_layout='partitioned_leaf',device=str(device),bf16_autocast=device.type=='cuda',robot_state_extension=True,full_parameter_count=8327408,actual_train_cases=18,batch_rows=144,new_optimizer_updates=0,new_physics_steps=0,
        scope='Complete official model and mathematical widened robot first affine, zero8192newcoefficients, exact original8319216 state/gradients/forward/RNG and official Adam groups. All16aux/all50q; botharms initial+learned source model. Old learned endpoint is only a preflight reference. New training will start at release initialization. CPU firstphase and CUDA all9TRAIN first2 saved primary seeds exactly replayed; no performance claim.')
    write(output,report);assert report['passed'],[k for k,v in checks.items() if not v]
    print(json.dumps(dict(passed=True,device=str(device),checks=len(checks))),flush=True)


if __name__=='__main__':main()
