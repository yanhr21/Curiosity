"""Two matched fixed20 full-gradient semantic repairs of complete official Utonia.

Preparation/CPU qualification are separate from retained-resource execution.
Never selects a best checkpoint or launches a retry/extension automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from . import train_overfit as original
from . import overfit_fullbatch_refinement as source_helper
from . import semantic_state_data as data
from . import semantic_state_readout as readout
from . import semantic_state_loss as objective
from . import semantic_state_training as study


def save_evaluation(root,stem,model,dataset,vertices,device,scales,condition):
    report,arrays=study.evaluate(model,dataset,vertices,device,scales,condition)
    original.save_json(Path(root)/(stem+'.json'),report)
    np.savez_compressed(Path(root)/(stem+'.npz'),**arrays)
    return report,arrays


def gradient_norms(optimizer):
    result={}
    for group in optimizer.param_groups:
        terms=[p.grad.detach().float().square().sum() for p in group['params'] if p.grad is not None]
        result[group['name']]=float(torch.stack(terms).sum().sqrt()) if terms else 0.
    return result


def source_output_difference(source,fit,dense):
    # This is historical cross-process output readback, not a bit-exact gate.
    result={}
    for label,name,arrays in (('fit','fit_02100.npz',fit),('development','same_trajectory_interpolation.npz',dense)):
        with np.load(Path(source)/name) as z:
            old={k:z[k] for k in z.files}
        lookup={(int(e),int(f)):i for i,(e,f) in enumerate(zip(arrays['episode'],arrays['frame']))}
        indices=[lookup[int(e),int(f)] for e,f in zip(old['episode'],old['frame'])]
        identity={k:bool(np.array_equal(arrays[k][indices],old[k])) for k in study.IDENTITY_KEYS if k in old}
        if not all(identity.values()):raise ValueError('Source target/mask/clock changed')
        result[label]=dict(identity_exact=identity,rows=len(indices),
            maximum_abs_differences={k:float(np.max(np.abs(arrays[k][indices]-old[k]))) for k in
                ('prediction','force_prediction_n','availability_probability','contact_probability')})
    return dict(result,scope='Fresh source2100 versus archived cross-process predictions; differences are measured, not asserted exact. Direct same-forward zero-addition check is separate.')


def run_arm(root,p,dataset,dense,full_vertices,sampled,resource):
    condition=p['condition'];root=Path(root)
    if any((root/n).exists() for n in ('RESULT.json','FAILURE.json','model.pt','train.jsonl','ARTIFACTS.json')):
        raise FileExistsError('No overwrite/restart of a semantic arm')
    manifest=study.begin_artifacts(root)
    random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed'])
    torch.cuda.manual_seed_all(p['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda')
    from .model import disable_stochastic_regularizers
    model=original.FullUtoniaOverfit(p['checkpoint'],history=32).to(device)
    optimizer=torch.optim.AdamW(original.parameter_groups(model),weight_decay=.01)
    # Source2100 has four groups. Split empty construction groups by actual
    # original parameter identity, then load complete source model AND Adam.
    source_helper.split_readout_group(optimizer,model.predictor.head.parameters(),model.auxiliary.parameters())
    endpoint=torch.load(Path(p['source_endpoint'])/'model.pt',map_location='cpu',weights_only=False,mmap=True)
    if endpoint['step']!=2100:raise ValueError('Only nofloor2100 can initialize this study')
    model.load_state_dict(endpoint['model'],strict=True)
    optimizer.load_state_dict(endpoint['optimizer'])
    restored=source_helper.verify_restored_optimizer(model,optimizer,2100)
    # Exact tensor readback includes all full-model values and all original
    # first/second moments; no optimizer action occurs during installation.
    parameters_exact=all(torch.equal(v.detach().cpu(),endpoint['model'][n]) for n,v in model.state_dict().items())
    saved_optimizer=endpoint['optimizer'];live_optimizer=optimizer.state_dict()
    groups_exact=live_optimizer['param_groups']==saved_optimizer['param_groups']
    moments_exact=(set(live_optimizer['state'])==set(saved_optimizer['state']) and all(
        torch.equal(v.detach().cpu(),saved_optimizer['state'][k][name]) if isinstance(v,torch.Tensor) else v==saved_optimizer['state'][k][name]
        for k,state in live_optimizer['state'].items() for name,v in state.items()))
    if not (parameters_exact and groups_exact and moments_exact):raise RuntimeError('Full source model/Adam restoration not exact')
    del endpoint,saved_optimizer,live_optimizer
    install=readout.install_semantic_readouts(model)
    if install['old_readout_width']!=88704 or install['added_parameters']!=5280:
        raise ValueError('Actual official readout structure differs from fixed protocol')
    added=readout.add_observation_optimizer_groups(model,optimizer,
        state_lr=study.LR_BASE['semantic_state_observation'],classification_lr=study.LR_BASE['semantic_classification_observation'])
    initial={n:v.detach().cpu().clone() for n,v in model.named_parameters()}
    original.save_json(root/'RESTORED_OPTIMIZER.json',dict(passed=True,restored=restored,install=install,
        new_groups=added,source_parameters_exact=parameters_exact,source_optimizer_groups_exact=groups_exact,
        source_all_optimizer_tensors_exact=moments_exact,source_checkpoint=p['source_bindings']['model.pt'],
        model_parameters=sum(v.numel() for v in model.parameters()),official_backbone_parameters=model.predictor.original_parameter_count))
    counter={'calls':0,'rows':0}
    def count_forward(module,args,output):
        counter['calls']+=1;counter['rows']+=len(args[0]['offset'])//32
    hook=model.register_forward_hook(count_forward)
    scales=torch.tensor(p['summary_scales'],dtype=torch.float32,device=device)
    vertices=torch.as_tensor(full_vertices[sampled],device=device)
    started=time.monotonic()
    initial_report,initial_fit=save_evaluation(root,'initial_fit',model,dataset,full_vertices,device,scales,condition)
    _,initial_dense=save_evaluation(root,'initial_same_trajectory_interpolation',model,dense,full_vertices,device,scales,condition)
    zero=readout.end_zero_effect_checks(model)
    if any(row!={'calls':1904,'items':1904} for row in zero.values()):
        raise RuntimeError('Every initial464+1440 readout must be directly zero-effect checked')
    original.save_json(root/'INITIAL_COMPATIBILITY.json',dict(passed=True,same_forward_readout_max_abs=0.,
        readouts=zero,source_model_adam_exact=True,condition=condition,
        historical_difference=source_output_difference(p['source_endpoint'],initial_fit,initial_dense),
        scope='Full official forward, exact original-width state/aux readout versus same computed readout plus zero. No historical cross-process bitidentity assertion.'))
    iterator=study.fixed_batches(dataset,p['seed']+1)
    with (root/'train.jsonl').open('x',buffering=1) as log:
        for update in range(1,21):
            if original.require_active_resource()!=resource:raise RuntimeError('Retained resource changed')
            rates=study.apply_learning_rates(optimizer,update)
            batches=next(iterator)
            model.train();disable_stochastic_regularizers(model);optimizer.zero_grad(set_to_none=True)
            totals={k:0. for k in objective.WEIGHTS};micro=[]
            for micro_index,indices in enumerate(batches,1):
                rows=[dataset[i] for i in indices]
                batch,history=study.prepare_batch(rows,device,scales,condition)
                output=readout.forward_semantic_observations(model,batch['inputs'],history)
                parts=objective.semantic_loss_parts(output,batch,vertices,p['denominators'])
                loss=objective.semantic_total_loss(parts)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite semantic objective')
                loss.backward() # contributions already use global denominators
                values={k:float(v.detach()) for k,v in parts.items()}
                for k,v in values.items():totals[k]+=v
                ns=int(batch['state_precision_eligible'].sum());nm=int(batch['mass_available'].sum())
                micro.append(dict(microbatch=micro_index,indices=indices,
                    samples=[dict(episode=r['metadata']['episode'],frame=r['metadata']['frame']) for r in rows],
                    state_candidates=ns,available_mass=nm,availability_positive=nm,
                    availability_negative=4-nm,current_contact_total=4,parts=values))
                del output,parts,loss,batch,history
            counts={k:sum(m[k] for m in micro) for k in ('state_candidates','available_mass','availability_positive','availability_negative','current_contact_total')}
            if any(counts[k]!=p['denominators'][k] for k in counts):raise RuntimeError('Full464 global denominators changed')
            norms=gradient_norms(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),100.,error_if_nonfinite=True)
            tracked={n:v.detach().clone() for n,v in model.named_parameters() if n.startswith('predictor.head.') or n.startswith('auxiliary.')}
            optimizer.step()
            delta={n:float((dict(model.named_parameters())[n].detach()-v).norm()) for n,v in tracked.items()}
            del tracked
            record=dict(condition=condition,new_optimizer_update=update,total_optimizer_updates=2100+update,
                new_training_microbatches=update*116,training_rows_exposed=update*464,
                learning_rates=rates,parts=totals,loss=sum(objective.WEIGHTS[k]*v for k,v in totals.items()),
                grad_norm_before_clip=float(norm),group_grad_norms_before_clip=norms,
                clip_multiplier=min(1.,100./(float(norm)+1e-6)),actual_readout_update_l2=delta,
                global_denominators=counts,microbatches=micro,elapsed_s=time.monotonic()-started)
            log.write(json.dumps(record)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k!='microbatches'}),flush=True)
    updates=study.parameter_update_report(model,optimizer,initial,condition)
    original.save_json(root/'PARAMETER_UPDATES.json',updates);del initial
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=2120,
        source_optimizer_updates=2100,new_optimizer_updates=20,new_observation_parameter_steps=20,
        protocol=p,condition=condition),root/'model.pt')
    fit_report,fit=save_evaluation(root,'fit_02120',model,dataset,full_vertices,device,scales,condition)
    dense_report,final_dense=save_evaluation(root,'same_trajectory_interpolation',model,dense,full_vertices,device,scales,condition)
    model.eval();batch,history=study.prepare_batch([dataset[0]],device,scales,condition)
    with torch.no_grad():
        before={k:v.detach().clone() for k,v in readout.forward_semantic_observations(model,batch['inputs'],history).items()}
        saved=torch.load(root/'model.pt',map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(saved['model'],strict=True)
        after=readout.forward_semantic_observations(model,batch['inputs'],history)
        reload_difference=max(float((before[k]-after[k]).abs().max()) for k in before)
    del saved,batch,history;hook.remove()
    exact_fit=all(np.array_equal(initial_fit[k],fit[k]) for k in study.IDENTITY_KEYS)
    exact_dense=all(np.array_equal(initial_dense[k],final_dense[k]) for k in study.IDENTITY_KEYS)
    if counter!={'calls':6130,'rows':13090}:raise RuntimeError('Actual forward/batch count differs: '+str(counter))
    integrity=bool(updates['passed'] and reload_difference==0 and exact_fit and exact_dense)
    semantic=bool(fit_report['passed'] and dense_report['passed'] and integrity)
    legacy=bool(fit_report['legacy_acceptance_passed'] and dense_report['legacy_acceptance_passed'] and integrity)
    study.finish_artifacts(root,manifest)
    result=dict(execution_complete=True,study=study.STUDY,condition=condition,acceptance_passed=semantic,
        semantic_acceptance_passed=semantic,legacy_acceptance_passed=legacy,
        legacy_force_gate_passed=bool(fit_report['legacy_force_gate_passed'] and dense_report['legacy_force_gate_passed']),
        fit_passed=fit_report['passed'],same_trajectory_interpolation_passed=dense_report['passed'],
        source_optimizer_updates=2100,optimizer_updates=20,total_optimizer_updates=2120,
        new_observation_parameter_steps=20,training_microbatches=2320,training_rows_exposed=9280,
        fit_items=464,interpolation_items=1440,endpoint_prediction_file='fit_02120.npz',
        initial_prediction_total_optimizer_updates=2100,supervision_profile=data.PROFILE,
        physical_success_gate_passed=False,original_historical_overall_result_modified=False,
        full_parameter_update_passed=updates['passed'],full_endpoint_reload_max_abs=reload_difference,
        initial_same_forward_readout_max_abs=0.,initial_final_fit_clocks_targets_exact=exact_fit,
        initial_final_interpolation_clocks_targets_exact=exact_dense,actual_model_forwards=counter,
        data_qualification_passed=True,retained_resource=resource,elapsed_s=time.monotonic()-started,
        scope=p['scope'],next_stage_scope='User authorizes continued root-managed repairs; this adapter performs only the fixed two-arm budget. No automatic extension.',
        legacy_force_note=p['original_force_rows'])
    original.save_json(root/'RESULT.json',result)
    print(json.dumps(result),flush=True)
    del model,optimizer;torch.cuda.empty_cache()
    return result


def run(args):
    root=Path(args.output).resolve();p=study.validate_protocol(root)
    if any((root/n).exists() for n in ('RESULT.json','FAILURE.json')):raise FileExistsError('Pair already attempted; preserve terminal result')
    if not args.check_data:resource=original.require_active_resource()
    print('SEMANTIC_FIT_ENCODING_BEGIN',flush=True)
    dataset=data.SemanticStateDataset(p['source_data'],p['data'])
    reports={condition:study.qualify_condition(dataset,condition,p['fit_limits']) for condition in study.CONDITIONS}
    actual_denominators=data.task_denominators(dataset)
    if actual_denominators!=p['denominators']:raise ValueError('Prepared global denominators changed')
    cache=study.cache_observations(dataset,p,verify_qualified_fit=True)
    report=dict(passed=all(r['passed'] for r in reports.values()),conditions=reports,
        observation_readback=cache,denominators=actual_denominators,rows=464,
        model_forwards=0,cuda_initialized=torch.cuda.is_initialized(),fit_input=True)
    original.save_json(root/('FIT_INPUT_CHECK.json' if args.check_data else 'DATA_QUALIFICATION.json'),report)
    if not report['passed']:raise RuntimeError('Both actual input contracts must be learnable before model construction')
    if args.check_data:
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print(json.dumps({k:v for k,v in report.items() if k not in ('conditions','observation_readback')}),flush=True)
        return 0
    print('SEMANTIC_DEVELOPMENT_ENCODING_BEGIN',flush=True)
    dense=data.SemanticStateDataset(p['source_data'],p['data'],role='development_interpolation')
    dense_reports={c:study.qualify_condition(dense,c,p['fit_limits']) for c in study.CONDITIONS}
    dense_cache=study.cache_observations(dense,p)
    original.save_json(root/'INTERPOLATION_DATA_QUALIFICATION.json',dict(conditions=dense_reports,observation_readback=dense_cache))
    if not all(r['passed'] for r in dense_reports.values()):raise RuntimeError('Dense input conflicts failed before model construction')
    if original.require_active_resource()!=resource:raise RuntimeError('Retained resource changed')
    full_vertices,sampled,mesh_report=original.read_known_mesh(p['mesh'])
    original.save_json(root/'KNOWN_MESH.json',mesh_report)
    results={}
    for condition in study.CONDITIONS:
        arm=root/condition
        expected=study.arm_protocol(p,condition)
        if json.loads((arm/'PROTOCOL.json').read_text())!=expected:raise ValueError('Prepared arm protocol changed')
        try:results[condition]=run_arm(arm,expected,dataset,dense,full_vertices,sampled,resource)
        except Exception as e:
            original.save_json(arm/'FAILURE.json',dict(execution_complete=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
            raise
    pair_checks={}
    for filename in ('initial_fit.npz','fit_02120.npz','initial_same_trajectory_interpolation.npz','same_trajectory_interpolation.npz'):
        with np.load(root/'no_summary'/filename) as a,np.load(root/'observed_summary'/filename) as b:
            pair_checks[filename]=all(np.array_equal(a[k],b[k]) for k in study.IDENTITY_KEYS)
    logs=[(root/c/'train.jsonl').read_text().splitlines() for c in study.CONDITIONS]
    matched_batches=len(logs[0])==len(logs[1])==20 and all(
        [m['samples'] for m in json.loads(a)['microbatches']]==[m['samples'] for m in json.loads(b)['microbatches']]
        for a,b in zip(*logs,strict=True))
    study.validate_protocol(root)
    result=dict(study=study.STUDY,execution_complete=True,conditions=results,
        matched_clocks_targets_masks=pair_checks,matched_actual_training_batches=matched_batches,
        acceptance_passed=bool(all(pair_checks.values()) and matched_batches and all(r['semantic_acceptance_passed'] for r in results.values())),
        physical_success_gate_passed=False,legacy_scientific_failures_preserved=True,
        optimizer_updates_per_arm=20,total_training_microbatches=4640,
        scope=p['evaluation_scope'],renders='Root-owned saved-only true/no_summary/observed_summary full16 rendering pending')
    original.save_json(root/'RESULT.json',result)
    return 0 if result['acceptance_passed'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--prepare',action='store_true');modes.add_argument('--check-data',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        protocol=study.source_protocol(args.source,args.qualification)
        args.output.mkdir(parents=True,exist_ok=True)
        if (args.output/'PROTOCOL.json').exists():raise FileExistsError('Prepared pair already exists')
        original.save_json(args.output/'PROTOCOL.json',protocol)
        for condition in study.CONDITIONS:
            arm=args.output/condition;arm.mkdir(exist_ok=False)
            original.save_json(arm/'PROTOCOL.json',study.arm_protocol(protocol,condition))
        return
    try:
        p=json.loads((args.output/'PROTOCOL.json').read_text())
        if str(args.source.resolve())!=p['source_endpoint'] or str(args.qualification.resolve())!=p['qualification']:
            raise ValueError('CLI sources differ from prepared pair')
        code=run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error,FileExistsError):
            original.save_json(args.output/'FAILURE.json',dict(execution_complete=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise
    raise SystemExit(code)


if __name__=='__main__':main()
