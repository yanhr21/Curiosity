"""Zero-update objective audit on the complete official Generator.

No model replacement or optimizer. A proposed local paired denoising ranking
objective is evaluated on real TRAIN branches only; no claim of official IFP.
"""
import argparse
import json
import os
from pathlib import Path
import socket

import dill
import numpy as np
import torch
import torch.nn.functional as F

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import (
    BASE, PARENT, ActualBranchGeometryDataset, GeneratorWrapper, CONTEXT_KEY,
    restore_geometry_state, write,
)

RUN = BASE / 'matched_generator_branch_dense_fit512'
SEEDS = list(range(272180, 272184))
TIMES = [15, 24, 33, 42]
WEIGHTS = [0.1, 0.25, 0.5, 1., 2., 4.]


def prepare(run):
    endpoint = json.loads((run / 'RESULT.json').read_text())
    prior = json.loads((run / 'frozen_evaluation/DENSE_COMPARISON.json').read_text())
    if not endpoint['all_horizon_plots_inspected'] or endpoint['heldout_phase_transfer_passed'] or not prior['checks_passed']:
        raise RuntimeError('Require complete negative dense endpoint and comparison first')
    out = run / 'paired_objective_audit'; out.mkdir(exist_ok=False)
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    write(out / 'PROTOCOL.json', dict(
        model_run=str(run), checkpoint_states=['initial', 'endpoint'],
        train_phases=plan['data']['train']['phases'], actual_cases=14,
        noise_seeds=SEEDS, diffusion_times=TIMES, candidate_weights=WEIGHTS,
        margin_fraction=0.1,
        base_loss='Mean per-case epsilon MSE with correct geometry; complete model in eval mode to isolate objective gradients without dropout.',
        rank_loss='mean(s * softplus(0.1 + (e_correct - e_wrong)/s)), s = alpha/(1-alpha) * mean((x0 - paired_x0)^2). Both errors predict the same sampled epsilon from the same target-corrupted input; only original-demo geometry swaps.',
        rationale='Scale the ranking margin by the epsilon displacement implied by the two real observed futures. This is a diagnostic discriminative surrogate, not a new denoising target or a generated counterfactual label. All targets remain actual8x36 commands.',
        pairing='One IID noise tensor per actual shared-world pair; same saved tensor in both condition calls and both complete checkpoint states. Same actual observations, old frozen normalizer and normalized goal zero. No validation gradients.',
        scope='Full8319216parameter official architecture with existing input adapter. No optimizer, weight mutation, model replacement, generative sampling, physics or SMP evaluation. Raw-gradient directional predictions are local SGD geometry, not AdamW steps or demonstrated training benefit.',
        checks='Exact data/phases/normalizer and full state; only geometry swapped; finite losses and gradients; zero-context ranking gradient zero up to numerical roundoff. Correct/wrong equal outputs alone must not be called demonstration use.',
        automatic_next_action='Inspect all32checkpoint/seed/time rows and all7phase errors. If ranking has finite nonzero geometry-column gradients in both states and an inspected fixed weight offers joint descent for both objectives in aggregate and at least3/4seed means at each state, predeclare a separate matched full-model ranking-versus-base experiment with unchanged data and budget. Otherwise locate TRAIN-phase gradient conflicts before any further training. Gradient checks never prove generative or physical benefit. Preserve all negative512endpoints.',
        optimizer_updates=0, new_physics_steps=0, new_sample_draws=0))
    print(str(out / 'PROTOCOL.json'), flush=True)


def gradient_stats(names, gd, gr):
    groups = {}
    for group in ('full_model', 'geometry_columns', 'target_encoder', 'denoiser'):
        dd = torch.zeros((), device=gd[0].device, dtype=torch.float64)
        rr, dot = dd.clone(), dd.clone()
        for name, a, b in zip(names, gd, gr):
            if group == 'geometry_columns':
                if name != 'obs_encoder.target_state_net.0.weight': continue
                a, b = a[:, 9:], b[:, 9:]
            elif group == 'target_encoder' and not name.startswith('obs_encoder.target_state_net.'): continue
            elif group == 'denoiser' and not name.startswith('model.'): continue
            a, b = a.double(), b.double()
            dd += a.square().sum(); rr += b.square().sum(); dot += (a * b).sum()
        dd, rr, dot = map(float, (dd, rr, dot))
        groups[group] = dict(base_norm=dd**.5, rank_norm=rr**.5, dot=dot,
                             cosine=dot / (dd * rr)**.5 if dd*rr > 0 else None,
                             weights={str(w): dict(base_dot_direction=dd+w*dot, rank_dot_direction=dot+w*rr,
                                both_local_descent=dd+w*dot > 0 and dot+w*rr > 0) for w in WEIGHTS})
    return groups


def run_audit(run, out=None):
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Require retained compute step')
    torch.set_num_threads(8)
    out = run / 'paired_objective_audit' if out is None else out
    protocol = json.loads((out / 'PROTOCOL.json').read_text())
    if (out / 'PARTIAL_RESULT.json').exists() or (out / 'RESULT.json').exists():
        raise RuntimeError('Preserve existing audit; inspect exact child before recovery')
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    dataset = ActualBranchGeometryDataset(**plan['branch_dataset'])
    if dataset.real_case_count != 14 or protocol['train_phases'] != plan['data']['train']['phases']:
        raise RuntimeError('TRAIN case scope changed')
    obs = {k: torch.stack([dataset[i]['obs'][k] for i in range(14)]).cuda() for k in dataset[0]['obs']}
    raw_target = torch.stack([dataset[i]['action'] for i in range(14)]).cuda()
    permutation = torch.arange(14, device='cuda') ^ 1
    checks = dict(causal_inputs_exact=all(torch.equal(obs[k], obs[k][permutation]) for k in ('obj_pos_b','obj_ori_b','last_action')))
    noises = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        noises[seed] = torch.randn(7, 8, 36, device='cuda').repeat_interleave(2, dim=0)
    np.savez_compressed(out / 'COMMON_NOISE.npz', **{str(k):v.cpu().numpy() for k,v in noises.items()})
    rows, aggregate = [], {}
    reference_normalized = None
    for checkpoint_state in protocol['checkpoint_states']:
        if checkpoint_state == 'initial':
            state = torch.load(run / 'demo_geometry/INITIAL_MODEL.pt', map_location='cpu')['model']
        else:
            with (run / 'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as stream:
                state = torch.load(stream, pickle_module=dill, map_location='cpu')['state_dicts']['model']
        policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
        restore_geometry_state(policy, state, 'demo_geometry', goal_condition_mode='zero_diagnostic')
        policy.cuda().eval(); policy.normalizer.requires_grad_(False)
        full_parameter_count=sum(p.numel() for p in policy.parameters())
        checks[checkpoint_state+'_full_parameter_count']=full_parameter_count == 8319216
        normalized = policy.normalizer.normalize(obs)
        target = policy.normalizer['action'].normalize(raw_target)
        if reference_normalized is None:
            reference_normalized = {k:v.clone() for k,v in normalized.items()}
            reference_target = target.clone()
        checks[checkpoint_state+'_normalizer_exact'] = torch.equal(target, reference_target) and all(torch.equal(v,reference_normalized[k]) for k,v in normalized.items())
        wrong = dict(normalized); wrong[CONTEXT_KEY] = normalized[CONTEXT_KEY][permutation]
        checks[checkpoint_state+'_only_geometry_swapped'] = all(torch.equal(v,wrong[k]) for k,v in normalized.items() if k != CONTEXT_KEY)
        named = [(n,p) for n,p in policy.named_parameters() if p.requires_grad]
        names, parameters = zip(*named)
        summed = [[torch.zeros_like(p) for p in parameters] for _ in range(2)]
        seeds_summary = {}
        for seed in SEEDS:
            seed_grad = [[torch.zeros_like(p) for p in parameters] for _ in range(2)]
            for time in TIMES:
                t = torch.full((14,), time, dtype=torch.long, device='cuda')
                noisy = policy.noise_scheduler.add_noise(target, noises[seed], t)
                alpha = policy.noise_scheduler.alphas_cumprod[time].cuda()
                separation = (target - target[permutation]).square().mean((1,2))
                scale = (alpha / (1-alpha) * separation).detach()
                assert bool((scale > 0).all())
                errors = []
                for inputs in (normalized,wrong):
                    cond = policy.obs_encoder(inputs,training=False)
                    pred = policy.model(noisy,t,cond=cond,training=False,gen_attn_map=False)[0]
                    errors.append((pred-noises[seed]).square().mean((1,2)))
                base = errors[0].mean()
                margin = 0.1 + (errors[0]-errors[1])/scale
                rank_each = scale * (F.relu(margin) if protocol.get('rank_function') == 'hinge' else F.softplus(margin))
                rank = rank_each.mean()
                grads=[]
                for index,loss in enumerate((base,rank)):
                    raw = torch.autograd.grad(loss,parameters,retain_graph=index==0,allow_unused=True)
                    grads.append([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,raw)])
                finite = all(bool(torch.isfinite(g).all()) for group in grads for g in group) and bool(torch.isfinite(base+rank))
                checks[f'{checkpoint_state}_{seed}_{time}_finite'] = finite
                if not finite: raise RuntimeError('Nonfinite full-model objective gradient')
                for k in (0,1):
                    for i,g in enumerate(grads[k]):
                        summed[k][i] += g / (len(SEEDS) * len(TIMES))
                        seed_grad[k][i] += g / len(TIMES)
                row = dict(checkpoint_state=checkpoint_state,seed=seed,time=time,base_loss=float(base.detach()),rank_loss=float(rank.detach()),
                    satisfied_margin_cases=int((margin.detach()<=0).sum()),
                    phase_metrics={str(p):dict(epsilon_correct=errors[0][2*i:2*i+2].detach().cpu().tolist(),epsilon_wrong=errors[1][2*i:2*i+2].detach().cpu().tolist(),rank_loss=rank_each[2*i:2*i+2].detach().cpu().tolist(),scale=scale[2*i:2*i+2].cpu().tolist()) for i,p in enumerate(protocol['train_phases'])},
                    gradient_groups=gradient_stats(names,*grads))
                rows.append(row)
                write(out / 'PARTIAL_RESULT.json', rows)
            seeds_summary[str(seed)] = gradient_stats(names,*seed_grad)
            print(json.dumps(dict(state=checkpoint_state,seed_completed=seed)),flush=True)
        aggregate[checkpoint_state]=dict(all_noise_mean=gradient_stats(names,*summed),seed_means=seeds_summary)
        # Complete zero-context control at the predeclared first seed and t24.
        policy.obs_encoder.condition_mode='zero_context'
        t=torch.full((14,),24,dtype=torch.long,device='cuda');noisy=policy.noise_scheduler.add_noise(target,noises[SEEDS[0]],t)
        errors=[]
        for inputs in (normalized,wrong):
            pred=policy.model(noisy,t,cond=policy.obs_encoder(inputs,training=False),training=False,gen_attn_map=False)[0]
            errors.append((pred-noises[SEEDS[0]]).square().mean((1,2)))
        zero_loss=(errors[0]-errors[1]).mean()
        zg=torch.autograd.grad(zero_loss,parameters,allow_unused=True)
        zero_max=max(float(g.abs().max()) for g in zg if g is not None)
        checks[checkpoint_state+'_zero_context_equal_errors']=torch.equal(errors[0],errors[1])
        checks[checkpoint_state+'_zero_context_rank_gradient_roundoff']=zero_max < 1e-7
        aggregate[checkpoint_state]['zero_context_max_gradient']=zero_max
        policy.obs_encoder.condition_mode='demo_geometry'
        checks[checkpoint_state+'_full_state_unchanged']=all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        checks[checkpoint_state+'_no_accumulated_parameter_gradients']=all(p.grad is None for p in policy.parameters())
        del policy,summed,seed_grad,grads,zg
        torch.cuda.empty_cache()
    candidates={}
    for weight in WEIGHTS:
        candidates[str(weight)]={state:dict(mean_pass=aggregate[state]['all_noise_mean']['full_model']['weights'][str(weight)]['both_local_descent'],
            seed_passes=sum(row['full_model']['weights'][str(weight)]['both_local_descent'] for row in aggregate[state]['seed_means'].values())) for state in protocol['checkpoint_states']}
    usable=[float(w) for w,states in candidates.items() if all(v['mean_pass'] and v['seed_passes']>=3 for v in states.values())]
    geometry_signal=all(row['gradient_groups']['geometry_columns']['rank_norm']>0 for row in rows)
    decision=dict(gradient_integrity_passed=all(checks.values()),geometry_signal_in_all32_rows=geometry_signal,candidate_weights=candidates,
        qualifying_weights=usable,smallest_qualifying_weight=min(usable) if usable else None,
        permits_preparing_matched_objective_experiment=all(checks.values()) and geometry_signal and bool(usable),
        interpretation='Local raw-gradient support for a bounded method experiment only; no actual optimizer step or performance conclusion.')
    if run.name == 'matched_generator_branch_paired_rank025512':
        group = aggregate['endpoint']['all_noise_mean']['full_model']
        decision = dict(gradient_integrity_passed=all(checks.values()),
            geometry_signal_in_all32_rows=geometry_signal, assessed_weight=.25,
            negative_base_rank_dot=group['dot'] < 0,
            current_direction_increases_base_to_first_order=group['weights']['0.25']['base_dot_direction'] < 0,
            seed_negative_dot_count=sum(v['full_model']['dot'] < 0 for v in aggregate['endpoint']['seed_means'].values()),
            row_negative_dot_count=sum(v['gradient_groups']['full_model']['dot'] < 0 for v in rows),
            coefficient_search_performed=False, next_action=protocol['automatic_next_action'],
            interpretation='Existing0.25weight only. Eight-time eval-mode raw-gradient diagnosis; not all50time unbiased training gradient, BF16/dropout training or an AdamW step. No new method admitted.')
    if protocol.get('rank_function') == 'hinge':
        support = {}
        for checkpoint_state, summary in aggregate.items():
            mean = summary['all_noise_mean']['full_model']['weights']['0.25']
            seed_rows = [v['full_model']['weights']['0.25'] for v in summary['seed_means'].values()]
            support[checkpoint_state] = dict(
                mean_base_descent=mean['base_dot_direction'] > 0,
                mean_rank_nonincrease=mean['rank_dot_direction'] >= -1e-10,
                compatible_seed_means=sum(v['base_dot_direction'] > 0 and v['rank_dot_direction'] >= -1e-10 for v in seed_rows))
        signal = any(v['gradient_groups']['geometry_columns']['rank_norm'] > 0 for v in rows if v['checkpoint_state']=='initial')
        decision = dict(gradient_integrity_passed=all(checks.values()), assessed_weight=.25,
            initial_geometry_signal=signal, checkpoint_support=support,
            permits_preparing_matched_objective_experiment=all(checks.values()) and signal and all(
                v['mean_base_descent'] and v['mean_rank_nonincrease'] and v['compatible_seed_means'] >= 3 for v in support.values()),
            coefficient_search_performed=False,
            interpretation='Fixed hinge0.1margin/0.25weight full-model raw-gradient evidence. Zero rank gradient is legitimate when all margins are satisfied. Not an actual AdamW update, BF16 training or demonstrated generation gain.')
    write(out/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,rows=rows,aggregate=aggregate,decision=decision,
        full_parameter_count=full_parameter_count,actual_train_cases=14,new_optimizer_updates=0,new_physics_steps=0,new_sample_draws=0,scope=protocol['scope'],next_action=protocol['automatic_next_action']))
    print(json.dumps(decision),flush=True)
    if not all(checks.values()): raise RuntimeError('Full objective audit integrity failed')


def endpoint_conflict():
    """Diagnose the observed accuracy/ranking tradeoff; no weight search."""
    global TIMES, WEIGHTS
    run = BASE / 'matched_generator_branch_paired_rank025512'
    evidence = run / 'frozen_evaluation/rank_component_audit'
    result = json.loads((evidence / 'RESULT.json').read_text())
    if not result['checks_passed'] or not result['plot_inspected']:
        raise RuntimeError('Finish matched full-model component evidence first')
    TIMES = [0, 3, 9, 15, 24, 33, 42, 49]
    WEIGHTS = [.25]
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    out = run / 'paired_objective_audit'; out.mkdir(exist_ok=False)
    write(out / 'PROTOCOL.json', dict(model_run=str(run), checkpoint_states=['endpoint'],
        train_phases=plan['data']['train']['phases'], actual_cases=14,
        noise_seeds=SEEDS, diffusion_times=TIMES, candidate_weights=WEIGHTS,
        margin_fraction=.1,
        evidence=str(evidence / 'RESULT.json'),
        scope='Frozen full8319216parameter025endpoint TRAIN-only gradient conflict diagnosis at eight predeclared low/mid/high times and four saved paired noises. Only the existing0.25weight is assessed; no coefficient selection or search. Exact original local objective, eval-mode gradients and no optimizer step. Selected times are diagnostic coverage, not an unbiased all50time training-gradient estimate. Raw gradients are not AdamW directions.',
        automatic_next_action='Separate TRAIN gradient conflict from weak condition signal and correct-denoising error. A conflict supports designing one explicit base-accuracy-preserving objective experiment; it does not authorize a coefficient sweep or prove any new method. If no conflict appears, inspect solver trajectories using the unchanged frozen endpoint. No update513, generated physics or SMP claim.',
        optimizer_updates=0,new_physics_steps=0,new_sample_draws=0))
    run_audit(run)


def hinge_margin_audit():
    """One fixed alternative on full initial/learned states; no tuning sweep."""
    global TIMES, WEIGHTS
    run = BASE / 'matched_generator_branch_paired_rank025512'
    evidence = BASE / 'matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe/rank_margin_readback'
    result = json.loads((evidence / 'RESULT.json').read_text())
    assert result['checks_passed'] and result['satisfied_margin_still_exerts_scalar_rank_derivative']
    TIMES = [0, 3, 9, 15, 24, 33, 42, 49]
    WEIGHTS = [.25]
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    out = evidence / 'hinge_gradient_audit'; out.mkdir(exist_ok=False)
    write(out / 'PROTOCOL.json', dict(model_run=str(run), checkpoint_states=['initial', 'endpoint'],
        train_phases=plan['data']['train']['phases'], actual_cases=14,
        noise_seeds=SEEDS, diffusion_times=TIMES, candidate_weights=WEIGHTS,
        margin_fraction=.1, rank_function='hinge', evidence=str(evidence / 'RESULT.json'),
        rank_loss='mean(s*relu(0.1+(e_correct-e_wrong)/s)); original full model and same true paired targets, fixed0.25weight.',
        decision_rule='Require full integrity and nonzero initial geometry signal; both checkpoint mean directions descend base and do not increase hinge, with at least3/4compatible seed means per checkpoint. Zero hinge gradients on already-satisfied rows are allowed. No weight/margin search.',
        scope='Full8319216parameter official initial and learned025endpoint, fourteen actual TRAIN cases, four original paired noise seeds and eight predeclared times.64raw-gradient rows; no optimization/sampling/physics. Same0.1margin0.25weight, replaces only softplus by hinge for gradient diagnosis. Eight-time eval gradients are not all50time/BF16/dropout/Adam dynamics.',
        automatic_next_action='On machine-checked gradient support, prepare one separate full-model paired-hinge512 matched experiment from original released initialization, same14cases/112batch/oldnormalizer/seed/budget/0.25weight, then full CPU/BF16 preflight before training and unchanged all9phase32draw criteria. On failure, retain result and diagnose recorded TRAIN gradient conflict before declaring any new training budget. Do not append old endpoints or claim generation/SMP benefit.',
        optimizer_updates=0, new_physics_steps=0, new_sample_draws=0))
    run_audit(run, out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--audit',action='store_true')
    mode.add_argument('--endpoint-conflict',action='store_true')
    mode.add_argument('--hinge-margin-audit',action='store_true')
    args=parser.parse_args()
    if args.endpoint_conflict:
        endpoint_conflict(); return
    if args.hinge_margin_audit:
        hinge_margin_audit(); return
    if args.run.resolve()!=RUN:raise RuntimeError('Use the declared fourteen-case full-model endpoint')
    (prepare if args.prepare else run_audit)(args.run.resolve())

if __name__=='__main__':main()
