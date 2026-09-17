"""Prepare and verify the bounded full-model paired-condition experiment."""
import argparse
import json
import os
import socket
from pathlib import Path
import torch
import dill
from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE, PARENT, ActualBranchGeometryDataset, write
from scripts.sugar.demo_following.demo_future.generator_workspace import warm_start_geometry_policy
from scripts.sugar.demo_following.demo_future.generator_demo_geometry import CONTEXT_KEY
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state

PREVIOUS=BASE/'matched_generator_branch_dense_fit512'
RUN=BASE/'matched_generator_branch_paired_rank512'


def prepare():
    audit=json.loads((PREVIOUS/'paired_objective_audit/RESULT.json').read_text())
    decision=audit['decision']
    if not decision['permits_preparing_matched_objective_experiment'] or decision['smallest_qualifying_weight']!=.1:
        raise RuntimeError('Require declared full-model TRAIN gradient decision')
    plan=json.loads((PREVIOUS/'PROTOCOL.json').read_text())
    plan['run_name']=RUN.name
    plan['paired_objective']=dict(weight=.1,margin_fraction=.1)
    plan['branch_dataset']['paired_supervision']=True
    plan['data_coverage_predecessor']=str(PREVIOUS)
    plan['objective_audit']=str(PREVIOUS/'paired_objective_audit/RESULT.json')
    plan['comparison_scope']='Same fourteen actual TRAIN cases, full initialization/normalizer, seed272084,512updates, batch112, IID base noise, AdamW1e-4 and schedule. Only training objective adds local paired ranking weight0.1. Additional wrong-context forward/backward increases treatment compute; equal updates/examples, not equal FLOPs. Compare full zero-context control reproducibility and all nine frozen phases against the previous complete base-loss endpoint.'
    plan['training']=plan['optimization']=plan['comparison_scope']
    plan['paired_objective_definition']='L=epsilon_MSE+0.1*mean(s*softplus(0.1+(e_correct-e_wrong)/s));s=alpha/(1-alpha)*mean((actual_future-actual_paired_future)^2), computed over all original50uniformly sampled training times. Wrong geometry only, same noisy target and matched stochastic masks. Actual paired labels outside obs. No new model and no official Zero-WAM/SMP implementation claim.'
    plan['automatic_next_action']='Complete both512full-model arms and all9phase5condition32draw frozen evaluations, full-state/Adam/batch/previous-control checks and all72panels. Compare seven TRAIN criteria and reused218/258 separately with base-loss predecessor. Both reused checks must pass, all seven TRAIN phases must pass, and correct MSE must not degrade on previously passing158/245 before native prediction and original-position/goal prerequisites; otherwise inspect objective effects without update513, generated physics or claiming SMP benefit.'
    plan['generator_training_started']=False
    RUN.mkdir(exist_ok=False);write(RUN/'PROTOCOL.json',plan)
    print(str(RUN),flush=True)



def prepare_next():
    if RUN.name == 'matched_generator_branch_paired_hinge025512':
        return prepare_hinge()
    plan=json.loads((PREVIOUS/'NEXT_MATCHED_PROTOCOL.json').read_text())
    comparison=json.loads((PREVIOUS/'frozen_evaluation/PAIRED_COMPARISON.json').read_text())
    routing = RUN.name == 'matched_generator_branch_encoder_rank025512'
    expected = dict(weight=.25,margin_fraction=.1)
    if routing: expected['gradient_scope']='target_encoder_only'
    if RUN.name != plan['run_name'] or plan['paired_objective'] != expected:
        raise RuntimeError('Only the declared next paired-objective run may be prepared')
    train=[v for v in comparison['phases'].values() if v['split']=='train']
    if not comparison['checks_passed'] or not all(v['rank_mse']<v['base_mse'] for v in train) or sum(v['rank_pass'] for v in train)<=sum(v['base_pass'] for v in train) or not all(v['rank_pass'] for v in train if v['base_pass']):
        raise RuntimeError('Declared TRAIN-only weight follow-up rule failed')
    previous_plan=json.loads((PREVIOUS/'PROTOCOL.json').read_text())
    keys=('branch_dataset','seed','epochs','batch_size','normalizer_state','phase_corpora','noise_coupling','goal_condition_mode','optimizer','scheduler','actual_optimizer_updates')
    same={k:plan[k]==previous_plan[k] for k in keys}
    if routing:
        audit=json.loads((PREVIOUS/'paired_objective_audit/RESULT.json').read_text())
        selection_ok=(audit['checks_passed'] and audit['all32gradient_rows_and_seven_phase_errors_inspected']
            and audit['target_encoder_only_rank_direction_diagnostic']['all32_base_first_order_descent']
            and audit['aggregate']['endpoint']['all_noise_mean']['denoiser']['dot']<0
            and audit['aggregate']['endpoint']['all_noise_mean']['target_encoder']['dot']>0)
    else: selection_ok=all(plan['weight_selection']['checks'].values())
    if not all(same.values()) or not selection_ok:
        raise RuntimeError('The declared follow-up changed its full data/configuration or lacks TRAIN evidence')
    plan['implementation_status']='prepared_requires_full_gradient_routing_preflight' if routing else 'prepared'
    RUN.mkdir(exist_ok=False);write(RUN/'PROTOCOL.json',plan)
    # This full CPU input/initial-sampling test does not exercise the coefficient.
    # Reuse it explicitly only after all of its input/model settings match.
    prior=json.loads((PREVIOUS/'PREFLIGHT.json').read_text())
    if not prior['passed']:raise RuntimeError('Previous full input preflight failed')
    prior.update(reused_from=str(PREVIOUS/'PREFLIGHT.json'),unchanged_input_model_settings=same,
        reuse_scope='No new CPU forward/backward/sample draw. Complete prior full-model input test reused with exactly matching settings; coefficient-specific full GPU BF16 preflight remains required.')
    write(RUN/'PREFLIGHT.json',prior)
    print(json.dumps(dict(run=str(RUN),weight=.25,unchanged_settings=same,gpu_paired_preflight_completed=False)),flush=True)


def prepare_hinge():
    evidence=BASE/'matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/rank_margin_readback/hinge_gradient_audit'
    audit=json.loads((evidence/'RESULT.json').read_text())
    assert audit['checks_passed'] and audit['decision']['permits_preparing_matched_objective_experiment']
    assert audit['all64gradient_rows_and_phase_metrics_inspected']
    plan=json.loads((PREVIOUS/'PROTOCOL.json').read_text())
    plan.update(run_name=RUN.name,paired_objective=dict(weight=.25,margin_fraction=.1,rank_function='hinge'),
        data_coverage_predecessor=str(PREVIOUS),objective_audit=str(evidence/'RESULT.json'),
        generator_training_started=False,implementation_status='prepared_requires_cpu_and_bf16_hinge_preflight',
        paired_objective_definition='L=epsilon_MSE+0.25*mean(s*relu(0.1+(e_correct-e_wrong)/s)); same original50uniform training times and s=alpha/(1-alpha)*actual paired future separation. Rank derivative zero once fixed margin is satisfied. Full model and actual targets unchanged; no new model/officialZero-WAM/SMP claim.',
        comparison_scope='Same full8319216parameter model/released initialization/oldnormalizer/14actualcases/112batch/seed272084/512updates/AdamW1e-4/schedule/IID-noise/goalzero/intact history as fullsoftplus0.25. Only softplus-to-hinge scalar ranking penalty changes; no weight or margin sweep. Both arms retrain from release, zero fullmodel and Adam must exactly reproduce predecessor.',
        automatic_next_action='Complete CPU and actualGPU BF16 full-model hinge preflights, then both512arms and all9phase5condition32draws/fullstates/Adam/savedreadback/72panels/strict predecessor comparison. Original sevenTRAIN and reused218/258 criteria unchanged; all must pass and all previous passing TRAIN MSEs must not regress before broader/native and physical prerequisites. Otherwise retain negatives and diagnose saved objective effects without update513, generated physics or SMP claim.')
    plan['training']=plan['optimization']=plan['comparison_scope']
    # Keep historical selection metadata explicitly separate from this hypothesis.
    for key in ('weight_selection','gradient_routing_selection'):
        if key in plan:plan['predecessor_'+key]=plan.pop(key)
    prior=json.loads((PREVIOUS/'PREFLIGHT.json').read_text());assert prior['passed']
    keys=('branch_dataset','seed','epochs','batch_size','normalizer_state','phase_corpora','noise_coupling','goal_condition_mode','optimizer','scheduler','actual_optimizer_updates')
    previous=json.loads((PREVIOUS/'PROTOCOL.json').read_text())
    same={k:plan[k]==previous[k] for k in keys};assert all(same.values())
    prior.update(reused_from=str(PREVIOUS/'PREFLIGHT.json'),unchanged_input_model_settings=same,
        reuse_scope='Full prior input/initial-sampling preflight reused only for identical model/data settings; actual new hinge CPU and BF16 full-loss/gradient checks separately required.')
    RUN.mkdir(exist_ok=False);write(RUN/'PROTOCOL.json',plan);write(RUN/'PREFLIGHT.json',prior)
    write(evidence/'NEXT_MATCHED_PROTOCOL.json',plan)
    print(json.dumps(dict(run=str(RUN),unchanged_settings=same,training_started=False)),flush=True)


def preflight(device="cpu"):
    device=torch.device(device)
    if device.type=="cuda" and (not os.environ.get("SLURM_STEP_ID") or socket.gethostname().startswith(("login","mgmtserver"))):
        raise RuntimeError("GPU preflight requires the retained compute step")
    torch.set_num_threads(8 if device.type=="cuda" else 1)
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    special = plan['paired_objective'].get('gradient_scope')=='target_encoder_only' or plan['paired_objective'].get('rank_function')=='hinge'
    preflight_name='PAIRED_CPU_PREFLIGHT.json' if device.type=='cpu' and special else 'PAIRED_PREFLIGHT.json'
    assert not (RUN/preflight_name).exists()
    dataset=ActualBranchGeometryDataset(**plan['branch_dataset'])
    old=ActualBranchGeometryDataset(**json.loads((PREVIOUS/'PROTOCOL.json').read_text())['branch_dataset'])
    rows=[dataset[i] for i in range(len(dataset))]
    batch={k:({n:torch.stack([r[k][n] for r in rows]) for n in rows[0][k]} if k=='obs' else torch.stack([r[k] for r in rows])) for k in rows[0]}
    batch={k:({n:v.to(device) for n,v in value.items()} if k=='obs' else value.to(device)) for k,value in batch.items()}
    checks=dict(actual_base_samples_exact=all(torch.equal(dataset[i]['action'],old[i]['action']) and all(torch.equal(dataset[i]['obs'][k],old[i]['obs'][k]) for k in old[i]['obs']) for i in range(len(dataset))),
        paired_targets_exact=all(torch.equal(dataset[i]['paired_action'],old[(i%14)^1]['action']) and torch.equal(dataset[i]['paired_geometry'],old[(i%14)^1]['obs'][CONTEXT_KEY]) for i in range(len(dataset))),
        no_paired_labels_in_obs=set(batch['obs'])==set(old[0]['obs']),balanced_rows=len(dataset)==112)
    observed=[];stats={}
    for arm in ('zero_context','demo_geometry'):
        policies=[]
        for settings in (None,plan['paired_objective']):
            p=warm_start_geometry_policy(PARENT,arm,goal_condition_mode='zero_diagnostic',noise_coupling='independent',paired_objective=settings)
            p.set_normalizer(dataset.get_normalizer());p.normalizer.requires_grad_(False);p.to(device).train();policies.append(p)
        original=policies[0].state_dict()
        checks[arm+'_full_initial_state_exact']=all(torch.equal(v,original[k]) for k,v in policies[1].state_dict().items())
        losses,grads,states,calls=[],[],[],[]
        for p in policies:
            captured=[]
            def hook(module,args,kwargs,output):
                captured.append((args[0].detach().clone(),args[1].detach().clone(),output[0].detach().clone()))
            handle=p.model.register_forward_hook(hook,with_kwargs=True)
            torch.manual_seed(272190)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                loss=p.compute_loss(batch,training=True)
            loss.backward();handle.remove()
            losses.append(float(loss.detach()));grads.append({n:(None if v.grad is None else v.grad.detach().clone()) for n,v in p.named_parameters()});states.append(rng_state(device));calls.append(captured)
            checks[arm+('_paired' if len(losses)==2 else '_base')+'_finite_backward']=bool(torch.isfinite(loss)) and all(g is None or bool(torch.isfinite(g).all()) for g in grads[-1].values())
        checks[arm+'_base_forward_and_times_exact']=all(torch.equal(a,b) for a,b in zip(calls[0][0],calls[1][0]))
        checks[arm+'_next_rng_exact']=all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(*states))
        if arm=='zero_context':
            checks['zero_control_loss_and_gradients_exact']=losses[0]==losses[1] and all((g is None and grads[1][k] is None) or (g is not None and grads[1][k] is not None and torch.equal(g,grads[1][k])) for k,g in grads[0].items())
        else:
            checks['wrong_forward_same_noise_time']=torch.equal(calls[1][0][0],calls[1][-1][0]) and torch.equal(calls[1][0][1],calls[1][-1][1])
            checks['initial_zero_columns_same_dropout_predictions']=torch.equal(calls[1][0][2],calls[1][-1][2])
            if plan['paired_objective'].get('gradient_scope')=='target_encoder_only':
                checks['rank_correct_forward_exact']=len(calls[1])==3 and all(torch.equal(a,b) for a,b in zip(calls[1][0],calls[1][1]))
            key='obs_encoder.target_state_net.0.weight'
            difference=grads[1][key][:,9:]-grads[0][key][:,9:]
            checks['ranking_adds_finite_geometry_gradient']=bool(torch.isfinite(difference).all()) and bool(difference.count_nonzero())
            stats['rank_geometry_gradient_increment_norm']=float(difference.norm())
            # Cover every scheduler time, including both endpoints, with the
            # full112-row actual loss. No parameter or optimizer update.
            p=policies[1];p.zero_grad(set_to_none=True);p.eval()
            original_randint=torch.randint
            def all_times(low,high,size,**kwargs):
                assert low==0 and high==50 and size==(112,)
                return torch.arange(112,device=kwargs.get('device'))%50
            try:
                torch.randint=all_times
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    value=p.compute_loss(batch,training=True)
                value.backward()
            finally:torch.randint=original_randint
            checks['all50times_finite_full_backward']=bool(torch.isfinite(value)) and all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in p.parameters())
        checks[arm+'_all_full_states_unchanged']=all(all(torch.equal(v,original[k]) for k,v in p.state_dict().items()) for p in policies)
        stats[arm]=dict(base_loss=losses[0],paired_loss=losses[1])
        for p in policies:p.zero_grad(set_to_none=True)
    if plan['paired_objective'].get('gradient_scope')=='target_encoder_only':
        route=gradient_routing_preflight(plan,dataset,batch,device)
        checks.update(route['checks']);stats['gradient_routing']=route['statistics']
    if plan['paired_objective'].get('rank_function')=='hinge':
        hinge=hinge_preflight(plan,dataset,batch,device)
        checks.update(hinge['checks']);stats['hinge']=hinge['statistics']
    result=dict(execution_completed=True,passed=all(checks.values()),checks=checks,statistics=stats,full_parameter_count=sum(p.numel() for p in policies[1].parameters()),actual_train_cases=14,batch_rows=112,optimizer_updates=0,physics_steps=0,
        device=str(device),bf16_autocast=device.type=='cuda',scope='Full loss/backward stochastic-control preflight with all50training times; device and actual autocast reported. No optimizer updates or training-benefit claim.')
    write(RUN/preflight_name,result);print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Full paired objective preflight failed')


def hinge_preflight(plan,dataset,batch,device):
    """Full initial/learned paired forwards; exact base gradients and hinge rule."""
    checks={};stats={}
    for checkpoint in ('initial','learned_endpoint'):
        policies=[]
        for function in ('softplus','hinge'):
            settings=dict(weight=.25,margin_fraction=.1,rank_function=function)
            p=warm_start_geometry_policy(PARENT,'demo_geometry',goal_condition_mode='zero_diagnostic',noise_coupling='independent',paired_objective=settings)
            p.set_normalizer(dataset.get_normalizer());p.normalizer.requires_grad_(False)
            if checkpoint=='learned_endpoint':
                with (PREVIOUS/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
                    state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
                p.load_state_dict(state,strict=True)
            p.to(device).train();policies.append(p)
        original={k:v.clone() for k,v in policies[0].state_dict().items()}
        values=[];base_grad=[];streams=[];captures=[]
        for p in policies:
            captured=[]
            def hook(module,args,kwargs,output):
                captured.append(tuple(v.detach().clone() for v in (args[0],args[1],output[0])))
            h=p.model.register_forward_hook(hook,with_kwargs=True)
            torch.manual_seed(272194)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                base,rank=p.compute_paired_losses(batch,training=True)
            named=[(n,v) for n,v in p.named_parameters() if v.requires_grad]
            raw=torch.autograd.grad(base,[v for _,v in named],retain_graph=True,allow_unused=True)
            base_grad.append({n:torch.zeros_like(v) if g is None else g.detach().clone() for (n,v),g in zip(named,raw)})
            rg=torch.autograd.grad(rank,[v for _,v in named],allow_unused=True)
            checks[f'{checkpoint}_{p.paired_rank_function}_finite_rank_gradients']=all(g is None or bool(torch.isfinite(g).all()) for g in rg)
            values.append((float(base.detach()),float(rank.detach())));streams.append(rng_state(device));captures.append(captured);h.remove()
            checks[f'{checkpoint}_{p.paired_rank_function}_full_state_unchanged']=all(torch.equal(v,original[k]) for k,v in p.state_dict().items())
        checks[checkpoint+'_base_loss_and_full_gradients_exact']=values[0][0]==values[1][0] and all(torch.equal(v,base_grad[1][k]) for k,v in base_grad[0].items())
        checks[checkpoint+'_all_forward_noise_time_values_exact']=len(captures[0])==len(captures[1])==2 and all(torch.equal(a,b) for c,d in zip(*captures) for a,b in zip(c,d))
        checks[checkpoint+'_next_rng_exact']=all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(*streams))
        # The exact full-model audit records per-case error/scale; verify the
        # scalar rule and its derivative on those real cases, including satisfied rows.
        audit=json.loads(Path(plan['objective_audit']).read_text())
        state_name='initial' if checkpoint=='initial' else 'endpoint'
        cases=[v for row in audit['rows'] if row['checkpoint_state']==state_name for v in row['phase_metrics'].values()]
        ec=torch.tensor([x for v in cases for x in v['epsilon_correct']],device=device,dtype=torch.float64,requires_grad=True)
        ew=torch.tensor([x for v in cases for x in v['epsilon_wrong']],device=device,dtype=torch.float64,requires_grad=True)
        scale=torch.tensor([x for v in cases for x in v['scale']],device=device,dtype=torch.float64)
        z=.1+(ec-ew)/scale
        loss=(scale*torch.relu(z)).sum();gc,gw=torch.autograd.grad(loss,(ec,ew))
        active=z>0
        checks[checkpoint+'_satisfied_hinge_derivative_exact_zero']=bool((gc[~active]==0).all()) and bool((gw[~active]==0).all())
        checks[checkpoint+'_active_hinge_derivative_roundoff']=torch.allclose(gc,active.to(gc.dtype),rtol=0,atol=1e-12) and torch.allclose(gw,-active.to(gw.dtype),rtol=0,atol=1e-12)
        stats[checkpoint]=dict(softplus_base_rank=values[0],hinge_base_rank=values[1],audit_cases=len(ec),satisfied_cases=int((~active).sum()))
        del policies,base_grad,rg
    return dict(checks=checks,statistics=stats)


def gradient_routing_preflight(plan,dataset,batch,device):
    """Full initial and learned states: forward parity and exact rank isolation."""
    checks,stats={},{}
    for checkpoint in ('initial','learned_endpoint'):
        policies=[]
        for scope in ('full_model','target_encoder_only'):
            settings=dict(weight=.25,margin_fraction=.1,gradient_scope=scope)
            p=warm_start_geometry_policy(PARENT,'demo_geometry',goal_condition_mode='zero_diagnostic',noise_coupling='independent',paired_objective=settings)
            p.set_normalizer(dataset.get_normalizer())
            if checkpoint=='learned_endpoint':
                with (PREVIOUS/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
                    state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
                p.load_state_dict(state,strict=True)
            p.normalizer.requires_grad_(False);p.to(device).train();policies.append(p)
        original={n:v.detach().clone() for n,v in policies[0].state_dict().items()}
        checks[checkpoint+'_routing_full_initial_state_exact']=all(torch.equal(v,original[n]) for n,v in policies[1].state_dict().items())
        all_grads,all_values,all_calls,all_rng=[],[],[],[]
        for policy in policies:
            calls=[]
            def hook(module,args,kwargs,output):
                calls.append((args[0].detach().clone(),args[1].detach().clone(),output[0].detach().clone()))
            handle=policy.model.register_forward_hook(hook,with_kwargs=True)
            torch.manual_seed(272190)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                base,rank=policy.compute_paired_losses(batch,training=True)
            handle.remove();all_rng.append(rng_state(device));all_calls.append(calls)
            named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
            params=[p for _,p in named]
            grads=[]
            for i,value in enumerate((base,rank)):
                values=torch.autograd.grad(value,params,retain_graph=i==0,allow_unused=True)
                grads.append({n:torch.zeros_like(p) if g is None else g.detach() for (n,p),g in zip(named,values)})
            all_grads.append(grads);all_values.append((float(base.detach()),float(rank.detach())))
        checks[checkpoint+'_same_base_and_rank_values']=all_values[0]==all_values[1]
        checks[checkpoint+'_base_and_wrong_forward_exact']=all(torch.equal(a,b) for i,j in ((0,0),(1,2)) for a,b in zip(all_calls[0][i],all_calls[1][j]))
        checks[checkpoint+'_rank_correct_forward_exact']=all(torch.equal(a,b) for a,b in zip(all_calls[1][0],all_calls[1][1]))
        checks[checkpoint+'_routing_next_rng_exact']=all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(*all_rng))
        checks[checkpoint+'_base_all_gradients_exact']=all(torch.equal(g,all_grads[1][0][n]) for n,g in all_grads[0][0].items())
        target=lambda n:n.startswith('obs_encoder.target_state_net.')
        checks[checkpoint+'_rank_outside_target_encoder_exact_zero']=all(not bool(g.count_nonzero()) for n,g in all_grads[1][1].items() if not target(n))
        checks[checkpoint+'_rank_target_encoder_nonzero']=any(bool(g.count_nonzero()) for n,g in all_grads[1][1].items() if target(n))
        checks[checkpoint+'_rank_target_encoder_gradient_exact']=all(torch.equal(g,all_grads[1][1][n]) for n,g in all_grads[0][1].items() if target(n))
        checks[checkpoint+'_routing_all_gradients_finite']=all(bool(torch.isfinite(g).all()) for groups in all_grads for group in groups for g in group.values())
        checks[checkpoint+'_routing_full_states_unchanged']=all(all(torch.equal(v,original[n]) for n,v in p.state_dict().items()) for p in policies)
        stats[checkpoint]=dict(base_loss=all_values[1][0],rank_loss=all_values[1][1],target_rank_norm=sum(float(g.double().square().sum()) for n,g in all_grads[1][1].items() if target(n))**.5,
            target_gradient_max_abs_difference=max(float((g-all_grads[1][1][n]).abs().max()) for n,g in all_grads[0][1].items() if target(n)))
        print(json.dumps(dict(routing_preflight_state=checkpoint,checks={k:v for k,v in checks.items() if k.startswith(checkpoint+'_')},statistics=stats[checkpoint])),flush=True)
    return dict(checks=checks,statistics=stats)



def readback():
    from numpy import load as load_arrays, array_equal
    out=RUN/'frozen_evaluation/PAIRED_COMPARISON.json'
    if out.exists():raise RuntimeError('Preserve completed paired comparison')
    roots=(PREVIOUS,RUN)
    endpoints=[json.loads((p/'RESULT.json').read_text()) for p in roots]
    plans=[json.loads((p/'PROTOCOL.json').read_text()) for p in roots]
    if not all(d['execution_completed'] and d['horizon_plots_inspected'] for d in endpoints):
        raise RuntimeError('Finish every sample and inspect all phase curves')
    checks={key:plans[0][key]==plans[1][key] for key in ('seed','epochs','batch_size','actual_optimizer_updates','optimizer','scheduler','normalizer_state','phase_corpora')}
    for arm in ('zero_context','demo_geometry'):
        checks[arm+'_actual_batch_order_exact']=(roots[0]/arm/'BATCH_ORDER.json').read_text()==(roots[1]/arm/'BATCH_ORDER.json').read_text()
        checks[arm+'_actual_budget_exact']=(roots[0]/arm/'DATA_AND_BUDGET.json').read_text()==(roots[1]/arm/'DATA_AND_BUDGET.json').read_text()
    checks['full_zero_control_model_and_adam_exact']=all(json.loads((RUN/'zero_context/RESULT.json').read_text())['checks'][k] for k in ('preceding_zero_control_full_model_exact','preceding_zero_control_full_adam_exact'))
    rows={}
    for phase in endpoints[0]['phases']:
        old,new=[d['phases'][phase] for d in endpoints]
        for variant in ('initial_shared_goal_zero','trained_zero_context','trained_demo_geometry_correct'):
            paths=[p/f'frozen_evaluation/phase_{phase}/{variant}.npz' for p in roots]
            with load_arrays(paths[0]) as a,load_arrays(paths[1]) as b:
                checks[phase+'_'+variant+'_targets_and_seeds_exact']=array_equal(a['actual_future_command_target'],b['actual_future_command_target']) and array_equal(a['sample_seeds'],b['sample_seeds'])
                if variant!='trained_demo_geometry_correct':checks[phase+'_'+variant+'_samples_exact']=array_equal(a['predictions'],b['predictions'])
        a,b=[v['variants']['trained_demo_geometry_correct'] for v in (old,new)]
        rows[phase]=dict(split=new['evaluation_split'],base_mse=a['normalized_mse'],rank_mse=b['normalized_mse'],rank_over_base_mse=b['normalized_mse']/a['normalized_mse'],base_correct_draws=a['per_branch_correct_preference_draws'],rank_correct_draws=b['per_branch_correct_preference_draws'],base_pass=old['branch_identifiability_passed'],rank_pass=new['branch_identifiability_passed'],base_correct_wrong_ratio=old['comparisons']['trained_demo_geometry_wrong']['correct_mse_ratio'],rank_correct_wrong_ratio=new['comparisons']['trained_demo_geometry_wrong']['correct_mse_ratio'])
    no_regression={p:row['rank_mse']<=row['base_mse'] for p,row in rows.items() if row['base_pass']}
    decision=dict(all_seven_train_pass=all(endpoints[1]['training_phase_checks'].values()),both_reused_phase_checks_pass=all(endpoints[1]['heldout_phase_checks'].values()),previously_passing_phase_mse_not_worse=no_regression)
    decision['all_declared_prediction_requirements_passed']=all(checks.values()) and decision['all_seven_train_pass'] and decision['both_reused_phase_checks_pass'] and all(no_regression.values())
    reference=json.loads((BASE/'matched_generator_branch_dense_fit512/RESULT.json').read_text())
    for phase,row in rows.items():
        base=reference['phases'][phase]
        row['original_unranked_mse']=base['variants']['trained_demo_geometry_correct']['normalized_mse']
        row['current_over_original_unranked_mse']=row['rank_mse']/row['original_unranked_mse']
        row['original_unranked_pass']=base['branch_identifiability_passed']
    report=dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,decision=decision,new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,
        previous_run=str(PREVIOUS),current_run=str(RUN),previous_weight=plans[0].get('paired_objective',{}).get('weight',0.),current_weight=plans[1]['paired_objective']['weight'],
        previous_gradient_scope=plans[0].get('paired_objective',{}).get('gradient_scope','full_model'),current_gradient_scope=plans[1]['paired_objective'].get('gradient_scope','full_model'),
        previous_rank_function=plans[0].get('paired_objective',{}).get('rank_function','softplus'),current_rank_function=plans[1]['paired_objective'].get('rank_function','softplus'),
        scope=plans[1]['comparison_scope']+' Original unranked results included descriptively. All are known TRAIN sources and reused phase checks, not independent training seeds or generated physical execution.')
    write(out,report);print(json.dumps(dict(checks_passed=report['checks_passed'],phases=rows,decision=decision)),flush=True)
    if not all(checks.values()):raise RuntimeError('Matched base/rank comparison integrity failed')


def main():
    global RUN, PREVIOUS
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--run',type=Path,default=RUN);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--preflight',action='store_true');g.add_argument('--readback',action='store_true');g.add_argument('--prepare-next',action='store_true');a=p.parse_args()
    RUN=a.run.resolve()
    if RUN==BASE/'matched_generator_branch_paired_rank025512':
        PREVIOUS=BASE/'matched_generator_branch_paired_rank512'
    elif RUN in (BASE/'matched_generator_branch_encoder_rank025512',BASE/'matched_generator_branch_paired_hinge025512'):
        PREVIOUS=BASE/'matched_generator_branch_paired_rank025512'
    elif RUN!=BASE/'matched_generator_branch_paired_rank512':
        raise RuntimeError('Use a predeclared paired-objective run')
    if a.prepare_next:prepare_next()
    elif a.prepare:prepare()
    elif a.readback:readback()
    else:preflight(a.device)

if __name__=='__main__':main()
