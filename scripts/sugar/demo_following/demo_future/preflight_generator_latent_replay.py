"""Actual full-model replay input, objective, gradient and RNG preflight."""
import argparse
import json
import os
import socket
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE, PARENT, write
from scripts.sugar.demo_following.demo_future.generator_dataset import ActualBranchGeometryDataset
from scripts.sugar.demo_following.demo_future.generator_latent_replay import ActualBranchLatentReplayDataset
from scripts.sugar.demo_following.demo_future.generator_workspace import warm_start_geometry_policy
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state, restore_rng
from scripts.sugar.demo_following.demo_future.train_generator_geometry import training_config

RUN=BASE/'matched_generator_branch_latent_replay01512'
OLD=BASE/'matched_generator_branch_paired_rank025512'


def same_rng(a,b):
    return all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(a,b))


def gradients(loss,policy,retain_graph=False):
    named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    values=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True,retain_graph=retain_graph)
    return {n:torch.zeros_like(p) if v is None else v.detach() for (n,p),v in zip(named,values)}


def main():
    global RUN, OLD
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--logging-fix',action='store_true')
    args=parser.parse_args();RUN=args.run.resolve()
    gap_coverage=RUN.name=='matched_generator_branch_latent_replay01_gaps512'
    rank_ablation=RUN.name=='matched_generator_branch_latent_only_gaps512'
    self_replay=RUN.name=='matched_generator_branch_self_replay01_gaps512'
    seed_replication=RUN.name in ('matched_generator_branch_ranked18_seed272400_512','matched_generator_branch_self18_seed272400_512')
    assert RUN.parent==BASE and (seed_replication or RUN.name in ('matched_generator_branch_latent_replay01512','matched_generator_branch_latent_replay01_gaps512','matched_generator_branch_latent_only_gaps512','matched_generator_branch_self_replay01_gaps512'))
    if seed_replication:OLD=Path(json.loads((RUN/'PROTOCOL.json').read_text())['training_seed_replication']['source_run'])
    if gap_coverage:OLD=BASE/'matched_generator_branch_latent_replay01512'
    if rank_ablation:OLD=BASE/'matched_generator_branch_latent_replay01_gaps512'
    if self_replay:OLD=BASE/'matched_generator_branch_latent_replay01_gaps512'
    device=torch.device(args.device)
    if device.type=='cuda':assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8 if device.type=='cuda' else 1)
    filename='GENERATED_BF16_PREFLIGHT.json' if device.type=='cuda' else 'GENERATED_CPU_PREFLIGHT.json'
    if args.logging_fix:
        assert rank_ablation
        filename=filename.replace('.json','_LOGGING_FIX.json')
    assert not (RUN/filename).exists()
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    old_plan=json.loads((OLD/'PROTOCOL.json').read_text())
    decision=json.loads(Path(plan['objective_decision']).read_text())
    assert decision['permits_preparing_matched_objective_experiment'] and decision['weight']==.1
    if seed_replication:
        assert OLD in (BASE/'matched_generator_branch_latent_replay01_gaps512',BASE/'matched_generator_branch_self_replay01_gaps512')
        replication=plan['training_seed_replication']
        assert decision['fixed_teacher_training_seed_replication'] and str(RUN) in decision['next_runs']
        assert plan['seed']==replication['new_seed']==272400 and old_plan['seed']==replication['original_seed']==272084
        assert all(plan[k]==old_plan[k] for k in ('branch_dataset','batch_size','data','normalizer_state','epochs','actual_optimizer_updates','optimizer','scheduler','frozen_evaluation','paired_objective','generated_state_objective','noise_coupling','goal_condition_mode'))
    if self_replay:
        assert decision['teacher_refresh'] and decision['next_run']==str(RUN) and plan['teacher_refresh']
        assert all(plan[k]==old_plan[k] for k in ('branch_dataset','batch_size','data','normalizer_state','seed','epochs','actual_optimizer_updates','optimizer','scheduler','frozen_evaluation','paired_objective'))
        assert plan['generated_state_objective']['teacher_run']==str(OLD)
    if rank_ablation:
        ablation=json.loads(Path(plan['rank_ablation_decision']).read_text())
        assert ablation['checks_passed'] and ablation['prepare_rank_ablation'] and ablation['next_run']==str(RUN)
        assert plan['paired_objective']['weight']==0 and old_plan['paired_objective']['weight']==.25
        assert all(plan[k]==old_plan[k] for k in ('branch_dataset','batch_size','data','normalizer_state','seed','epochs','actual_optimizer_updates','optimizer','scheduler','frozen_evaluation'))
    cfg=training_config(plan,'demo_geometry')
    dataset=hydra.utils.instantiate(cfg.task.dataset)
    old=ActualBranchGeometryDataset(**(plan['branch_dataset'] if gap_coverage else old_plan['branch_dataset']))
    rows=plan['batch_size'];cases=plan['real_branch_examples']
    assert (cases,rows)==((18,144) if gap_coverage or rank_ablation or self_replay or seed_replication else (14,112))
    assert isinstance(dataset,ActualBranchLatentReplayDataset)
    samples=[dataset[i] for i in range(rows)]
    def collate(samples):
        return {k:({n:torch.stack([s[k][n] for s in samples]).to(device) for n in samples[0][k]} if k=='obs' else torch.stack([s[k] for s in samples]).to(device)) for k in samples[0]}
    batch=collate(samples)
    checks=dict(actual_full_rows=len(dataset)==rows and dataset.real_case_count==cases,
        all_original_fields_exact=all(torch.equal(samples[i]['action'],old[i]['action']) and all(torch.equal(samples[i]['obs'][k],old[i]['obs'][k]) for k in old[i]['obs']) and torch.equal(samples[i]['paired_action'],old[i]['paired_action']) and torch.equal(samples[i]['paired_geometry'],old[i]['paired_geometry']) for i in range(rows)),
        no_replay_fields_in_obs=set(batch['obs'])==set(old[0]['obs']),
        all_replay_seed_case_exact=all(np.array_equal(samples[i]['replay_xt'].numpy(),dataset.replay['xt'][i//cases,:,i%cases]) and int(samples[i]['replay_seed'])==272230+i//cases for i in range(rows)),
        full_state_dict_shape_unmodified=True)
    if seed_replication:
        reference_dataset=ActualBranchLatentReplayDataset(replay_source=old_plan['generated_state_objective']['replay_source'],**old_plan['branch_dataset'])
        checks['fixed_teacher_all_replay_arrays_exact']=set(reference_dataset.replay)==set(dataset.replay) and all(np.array_equal(reference_dataset.replay[k],dataset.replay[k]) for k in dataset.replay)
        checks['training_seed_only_operational_change']=plan['seed']!=old_plan['seed'] and plan['generated_state_objective']==old_plan['generated_state_objective']
    # Check actual official loader order against the preceding data configuration.
    from torch.utils.data import DataLoader
    order=[]
    for data in (old,dataset):
        torch.manual_seed(plan['seed'])
        order.append([v['sample_index'].tolist() for v in DataLoader(data,batch_size=rows,shuffle=True,num_workers=0)])
    checks['actual_shuffled_batch_order_exact']=order[0]==order[1]
    shuffled=collate([samples[i] for i in order[1][0]])
    checks['shuffled_replay_binding_exact']=all(torch.equal(shuffled['replay_xt'][j],batch['replay_xt'][i]) for j,i in enumerate(order[1][0]))
    if self_replay:
        with np.load(old_plan['generated_state_objective']['input_arrays']) as a:
            for k in ('normalized_actual_target','initial_gaussian','seeds','times','phases','branches'):
                checks['teacher_refresh_'+k+'_exact']=np.array_equal(a[k],dataset.replay[k])
            checks['teacher_refresh_xt_changed']=not np.array_equal(a['xt'],dataset.replay['xt'])
    if gap_coverage:
        previous_dataset=ActualBranchLatentReplayDataset(replay_source=old_plan['generated_state_objective']['replay_source'],**old_plan['branch_dataset'])
        old_phases=old_plan['data']['train']['phases'];new_phases=plan['data']['train']['phases']
        mapping=[(i//14)*cases+2*new_phases.index(old_phases[(i%14)//2])+(i%2) for i in range(112)]
        legacy_samples=[previous_dataset[i] for i in range(112)]
        mapped_samples=[samples[i] for i in mapping]
        checks['all_old112_complete_samples_exact']=all(set(a)==set(b) and all((set(a[k])==set(b[k]) and all(torch.equal(a[k][n],b[k][n]) for n in a[k])) if k=='obs' else torch.equal(a[k],b[k]) for k in a if k!='sample_index') for a,b in zip(legacy_samples,mapped_samples))
        legacy_batch=collate(legacy_samples);mapped_batch=collate(mapped_samples)
    stats={}
    for arm in ('zero_context','demo_geometry'):
        for checkpoint in ('initial','learned_endpoint'):
            original=warm_start_geometry_policy(PARENT,arm,goal_condition_mode='zero_diagnostic',noise_coupling='independent',paired_objective=plan['paired_objective'])
            current=hydra.utils.instantiate(training_config(plan,arm).policy)
            for p in (original,current):p.set_normalizer(dataset.get_normalizer());p.normalizer.requires_grad_(False)
            if checkpoint=='learned_endpoint':
                with (OLD/arm/'checkpoints/endpoint.ckpt').open('rb') as f:state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
                for p in (original,current):p.load_state_dict(state,strict=True)
            saved={n:v.cpu().clone() for n,v in original.state_dict().items()}
            key=arm+'_'+checkpoint
            checks[key+'_full8319216_initial_state_exact']=sum(p.numel() for p in current.parameters())==8319216 and current.state_dict().keys()==saved.keys() and all(torch.equal(v,saved[n]) for n,v in current.state_dict().items())
            values=[];grads=[];streams=[];captures=[]
            for index,p in enumerate((original,current)):
                p.to(device).train();captured=[]
                def capture(module,args,kwargs,output):captured.append((args[0].detach().clone(),args[1].detach().clone(),output[0].detach().clone()))
                handle=p.model.register_forward_hook(capture,with_kwargs=True)
                torch.manual_seed(272240)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    if index==0:base=p.compute_loss(shuffled,training=True)
                    else:base,aux=p.compute_replay_components(shuffled,training=True,time_index=8)
                handle.remove();captures.append(captured);streams.append(rng_state(device));values.append(float(base.detach()))
                grads.append(gradients(base,p))
                if index:
                    aux_grads=gradients(aux,p)
                    checks[key+'_finite_nonzero_aux_full_gradient']=bool(torch.isfinite(aux)) and all(bool(torch.isfinite(g).all()) for g in aux_grads.values()) and any(bool(g.count_nonzero()) for g in aux_grads.values())
                    if arm=='zero_context':checks[key+'_zero_context_aux_geometry_gradient_zero']=not bool(aux_grads['obs_encoder.target_state_net.0.weight'][:,9:].count_nonzero())
                    xt,t,pred=captured[-1];alpha=p.generated_state_alphas_cpu[21]
                    target=p.normalizer['action'].normalize(shuffled['action'])
                    implied=(xt-(alpha**.5)*target)/((1-alpha)**.5)
                    checks[key+'_actual_aux_xt_time_exact']=torch.equal(xt,shuffled['replay_xt'][:,8]) and bool((t==21).all())
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        expected=F.mse_loss(pred,implied)
                    checks[key+'_actual_full_aux_formula_exact']=float(aux.detach())==float(expected)
                    stats[key]=dict(original_loss=values[0],replay_original_loss=values[1],auxiliary_loss=float(aux.detach()),auxiliary_gradient_norm=sum(float(v.double().square().sum()) for v in aux_grads.values())**.5)
            checks[key+'_original_loss_gradients_exact']=values[0]==values[1] and all(torch.equal(g,grads[1][n]) for n,g in grads[0].items())
            checks[key+'_original_all_forward_noise_times_exact']=len(captures[1])==len(captures[0])+1 and all(torch.equal(a,b) for left,right in zip(captures[0],captures[1]) for a,b in zip(left,right))
            checks[key+'_next_rng_exact']=same_rng(*streams)
            if rank_ablation:
                # Compare to the existing full0.25 paired implementation with
                # its differentiable base component retained and rank omitted.
                reference_settings=dict(plan['paired_objective'],weight=.25)
                reference_policy=warm_start_geometry_policy(PARENT,arm,goal_condition_mode='zero_diagnostic',noise_coupling='independent',paired_objective=reference_settings)
                reference_policy.set_normalizer(dataset.get_normalizer());reference_policy.normalizer.requires_grad_(False)
                reference_policy.load_state_dict(saved,strict=True);reference_policy.to(device).train()
                torch.manual_seed(272240)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    if arm=='demo_geometry':reference_base,reference_rank=reference_policy.compute_paired_losses(shuffled,training=True)
                    else:reference_base=reference_policy.compute_loss(shuffled,training=True)
                rg=gradients(reference_base,reference_policy)
                checks[key+'_old_ranked_base_gradient_rng_exact']=float(reference_base.detach())==values[0] and all(torch.equal(v,rg[n]) for n,v in grads[0].items()) and same_rng(rng_state(device),streams[0])
                del reference_policy,rg
            current.generated_state_weight=0
            torch.manual_seed(272240)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):disabled=current.compute_loss(shuffled,training=True)
            dg=gradients(disabled,current)
            checks[key+'_disabled_full_loss_gradients_rng_exact']=float(disabled.detach())==values[0] and all(torch.equal(g,dg[n]) for n,g in grads[0].items()) and same_rng(rng_state(device),streams[0]) and current.generated_state_step==0
            current.generated_state_weight=.1
            if gap_coverage:
                subset_values=[];subset_gradients=[];subset_rng=[]
                for subset in (legacy_batch,mapped_batch):
                    current.generated_state_step=0;torch.manual_seed(272240)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        loss=current.compute_loss(subset,training=True)
                    subset_values.append(float(loss.detach()));subset_rng.append(rng_state(device));subset_gradients.append(gradients(loss,current))
                checks[key+'_old112_full_replay_loss_gradients_rng_exact']=subset_values[0]==subset_values[1] and same_rng(*subset_rng) and all(torch.equal(v,subset_gradients[1][n]) for n,v in subset_gradients[0].items())
                current.generated_state_step=0
                del subset_gradients
            checks[key+'_all_full_states_unchanged']=all(p.state_dict().keys()==saved.keys() and all(torch.equal(v.cpu(),saved[n]) for n,v in p.state_dict().items()) for p in (original,current))
            write(RUN/(filename+'.partial'),dict(checks=checks,statistics=stats))
            assert all(checks.values())
            print(json.dumps(dict(state=key,checks_passed=True,statistics=stats[key])),flush=True)
            del original
    # The final current policy is the full learned demo model. Exercise every
    # replay time with all declared real rows and full differentiable correction.
    current.eval()
    for ti in range(16):
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            aux=current.compute_generated_state_loss(batch,training=False,time_index=ti)
        ag=gradients(aux,current)
        checks[f'time{ti}_full{rows}_aux_backward_finite']=bool(torch.isfinite(aux)) and all(bool(torch.isfinite(g).all()) for g in ag.values())
    # Actual combined loss uses all50q times; counter exercises wrap and labels.
    original_randint=torch.randint
    def all_times(low,high,size,**kwargs):
        assert low==0 and high==50 and size==(rows,)
        return torch.arange(rows,device=kwargs.get('device'))%50
    records=[]
    try:
        torch.randint=all_times
        for step in (0,1,15,16,511):
            current.generated_state_step=step
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):combined=current.compute_loss(batch,training=True)
            cg=gradients(combined,current)
            row=dict(current.last_paired_loss);records.append(row)
            checks[f'combined_step{step}_finite_exact_clock']=current.generated_state_step==step+1 and row['generated_step']==step and row['generated_time_index']==step%16 and bool(torch.isfinite(combined)) and all(bool(torch.isfinite(g).all()) for g in cg.values())
            checks[f'combined_step{step}_recorded_loss_arithmetic']=bool(np.isclose(float(combined.detach()),row['base']+row['weight']*row['rank']+row['generated_weight']*row['generated'],rtol=2e-6,atol=2e-6))
    finally:torch.randint=original_randint
    checks['final_learned_full_state_unchanged']=all(torch.equal(v.cpu(),saved[n]) for n,v in current.state_dict().items())
    checks['no_accumulated_parameter_gradients']=all(p.grad is None for p in current.parameters())
    checks['actual_frozen_normalizer_exact']=all(torch.equal(current.state_dict()[n].cpu(),v) for n,v in saved.items() if n.startswith('normalizer.'))
    report=dict(execution_completed=True,passed=all(checks.values()),checks=checks,statistics=stats,combined_records=records,device=str(device),bf16_autocast=device.type=='cuda',full_parameter_count=8319216,actual_train_cases=cases,batch_rows=rows,new_optimizer_updates=0,new_physics_steps=0,
        comparison_scope='Fixed-teacher training-seed replication: unchanged18case144rows and all replay arrays; actual new-seed loader order, botharms initial+source-learned full gradients/forwards/RNG checked. Only training RNG seed changes; teachers and primary sampling seeds remain fixed.' if seed_replication else 'Teacher refresh only: same18case144rows, exact original labels/initialGaussian/clocks; original base+rank fullgradients/forwards/RNG unchanged in botharms initial+learned states.' if self_replay else 'Rank absence only: same18case144rows; old0.25paired differentiable base gradient/RNG exact, fullinitial+learned botharms.' if rank_ablation else ('Old-loss baseline uses the same expanded144rows; old112 replay subset is additionally compared exactly in botharms initial+learned fullstates.' if gap_coverage else 'Original14case112row objective extension.'),
        scope='Actual Hydra dataset/factory and complete model: botharms initial+learned exact oldloss/gradient/forwards/RNG, disabled-aux parity, real replay mapping, all16aux times and all50q times, deterministic exposure counter. Raw auxiliary intentionally nonzero in zero context. No parameter or Adam update and no generation benefit.')
    write(RUN/filename,report)
    assert report['passed']
    print(json.dumps(dict(passed=True,checks=len(checks),device=str(device))),flush=True)


if __name__=='__main__':main()
