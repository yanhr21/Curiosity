"""Floor-only controlled observation experiment: same100 full-gradient updates.

No SVD coefficients, replacement architecture, data filtering, loss changes or
relaxed accuracy gates. Source2000 model+Adam are restored; endpoint is2100.
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
from . import overfit_fullbatch_refinement as base
from . import overfit_floor_observation as repair


def save_evaluation(root,stem,model,dataset,vertices,device):
    report,arrays=repair.evaluate_floor(model,dataset,vertices,device)
    original.save_json(Path(root)/(stem+'.json'),report)
    np.savez_compressed(Path(root)/(stem+'.npz'),**arrays)
    return report,arrays


def gradient_norms(optimizer):
    report={}
    for group in optimizer.param_groups:
        terms=[parameter.grad.detach().float().square().sum() for parameter in group['params'] if parameter.grad is not None]
        report[group['name']]=float(torch.stack(terms).sum().sqrt()) if terms else 0.
    return report


def fixed_epoch_batches(dataset,iterator):
    batches=[next(iterator) for _ in range(repair.MICROBATCHES)]
    if sorted(index for indices in batches for index in indices)!=list(range(80)):
        raise RuntimeError('Every effective update must contain every fixed row once')
    if any(len(indices)!=4 for indices in batches):raise RuntimeError('Original microbatch size changed')
    return batches


def validate_prepared(source,root,matched_baseline):
    declared=json.loads((Path(root)/'PROTOCOL.json').read_text())
    if declared!=repair.source_protocol(source,matched_baseline):raise ValueError('Prepared refinement protocol/source changed')
    return declared


def run(args):
    root=Path(args.output);source=Path(args.source).resolve()
    protocol=validate_prepared(source,root,args.matched_baseline)
    if any((root/name).exists() for name in ('RESULT.json','FAILURE.json','model.pt','train.jsonl')):
        raise FileExistsError('No overwrite, automatic restart or budget extension')
    if not args.check_data:
        resource=original.require_active_resource()
        initial_artifacts=repair.begin_artifacts(root)
    dataset=repair.FloorDataset(protocol['source_data'],protocol['data'])
    data_report=original.qualification(dataset)
    original.save_json(root/('FIT_INPUT_CHECK.json' if args.check_data else 'DATA_QUALIFICATION.json'),data_report)
    if not data_report['passed']:raise RuntimeError('Original complete failure-aware fit data gate failed')
    floor_check=repair.validate_floor_inputs(dataset)
    original.save_json(root/('FLOOR_INPUT_CHECK.json' if args.check_data else 'FLOOR_DATA_QUALIFICATION.json'),floor_check)
    if args.check_data:
        if torch.cuda.is_initialized():raise RuntimeError('CPU fit preflight initialized CUDA')
        print(json.dumps(dict(fit_input=True,rows=len(dataset),passed=True,
            cuda_initialized=False,model_forwards=0,dense_check='Required inside actual training before model construction')),flush=True)
        return 0
    interpolation=repair.FloorDataset(protocol['source_data'],protocol['data'],
                                                       interpolation_frames=original.INTERPOLATION_FRAMES)
    dense_report=original.failure_aware_dense_qualification(interpolation,
        frames=original.INTERPOLATION_FRAMES,fit_limits=original.FIT_LIMITS)
    original.save_json(root/'INTERPOLATION_DATA_QUALIFICATION.json',dense_report)
    if not dense_report['passed']:raise RuntimeError('Original dense data gate failed before model construction')
    original.save_json(root/'FLOOR_INTERPOLATION_INPUT_CHECK.json',repair.validate_floor_inputs(interpolation))
    if original.require_active_resource()!=resource:raise RuntimeError('Retained resource changed during CPU admission')
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    torch.cuda.manual_seed_all(protocol['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda')
    full_vertices,sampled,mesh_report=original.read_known_mesh(protocol['mesh'])
    original.save_json(root/'KNOWN_MESH.json',mesh_report)
    vertices=torch.as_tensor(full_vertices[sampled],device=device)
    from .model import disable_stochastic_regularizers
    model=repair.FloorObservationOverfit(protocol['checkpoint'],history=32).to(device)
    optimizer=torch.optim.AdamW(original.parameter_groups(model),weight_decay=.01)
    endpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False,mmap=True)
    if endpoint['step']!=repair.SOURCE_UPDATES:raise ValueError('Wrong source checkpoint clock')
    model.load_state_dict(endpoint['model'],strict=True)
    optimizer.load_state_dict(endpoint['optimizer'])
    restored=base.verify_restored_optimizer(model,optimizer,repair.SOURCE_UPDATES)
    base.split_readout_group(optimizer,model.predictor.head.parameters(),model.auxiliary.parameters())
    after_split=base.verify_restored_optimizer(model,optimizer,repair.SOURCE_UPDATES)
    if restored!=after_split:raise RuntimeError('Splitting readout groups changed complete restored Adam')
    del endpoint
    model.add_floor_branches()
    repair.add_optimizer_groups(model,optimizer)
    added=repair.verify_optimizer(model,optimizer,repair.SOURCE_UPDATES,0)
    initial={name:parameter.detach().cpu().clone() for name,parameter in model.named_parameters()}
    original.save_json(root/'RESTORED_OPTIMIZER.json',dict(before_split=restored,after_split=after_split,after_floor_branches=added,
        no_moment_reset=True,total_model_parameters=sum(p.numel() for p in model.parameters()),
        original_backbone_parameters=model.predictor.original_parameter_count,
        groups=[dict(name=g['name'],parameters=sum(p.numel() for p in g['params']),lr=g['lr']) for g in optimizer.param_groups]))
    initial_report,initial_arrays=save_evaluation(root,'initial_fit',model,dataset,full_vertices,device)
    _,initial_dense=save_evaluation(root,'initial_same_trajectory_interpolation',model,interpolation,full_vertices,device)
    if model.zero_replay_items!=1520 or model.zero_replay_calls!=1520:
        raise RuntimeError('Require all80+1440 same-forward exact zero-branch checks')
    original.save_json(root/'ZERO_BRANCH_REPLAY.json',dict(passed=True,fit_items=80,dense_items=1440,
        same_original_forward_comparisons=model.zero_replay_items,max_abs_difference=0.,
        source_checkpoint=protocol['source_bindings']['model.pt'],
        scope='Same original source2000 forward vs zero-branch-added outputs for every initial item; no cross-process bitidentity claim'))
    model.zero_replay_enabled=False
    iterator=original.fixed_batches(dataset,protocol['seed']+1)
    for _ in range(repair.SOURCE_UPDATES):next(iterator)
    started=time.monotonic()
    with (root/'train.jsonl').open('x',buffering=1) as log:
        for update in range(1,repair.NEW_UPDATES+1):
            rates=repair.apply_learning_rates(optimizer,update)
            batches=fixed_epoch_batches(dataset,iterator)
            model.train();disable_stochastic_regularizers(model)
            optimizer.zero_grad(set_to_none=True)
            micro_records=[];totals={key:0. for key in original.LOSS_WEIGHTS}
            for micro_index,indices in enumerate(batches,1):
                batch=original.device_batch(repair.collate_floor([dataset[index] for index in indices]),device)
                output=model(batch['inputs'])
                parts=original.conditional_task_losses(output,batch,vertices)
                loss=base.scaled_microbatch_loss(parts)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite original objective')
                loss.backward()
                values={key:float(value.detach()) for key,value in parts.items()}
                for key,value in values.items():totals[key]+=value/repair.MICROBATCHES
                micro_records.append(dict(microbatch=micro_index,
                    cumulative_training_microbatch=repair.SOURCE_UPDATES+(update-1)*repair.MICROBATCHES+micro_index,
                    samples=[dict(episode=row['episode'],frame=row['frame']) for row in batch['metadata']],
                    state_precision_denominator=int(batch['state_precision_eligible'].sum()),
                    mass_precision_denominator=int(batch['mass_available'].sum()),parts=values))
                del loss,parts,output,batch
            state_count=sum(row['state_precision_denominator'] for row in micro_records)
            mass_count=sum(row['mass_precision_denominator'] for row in micro_records)
            if state_count!=63 or mass_count!=43:raise RuntimeError('Original fixed mask denominators changed')
            group_norms=gradient_norms(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),100.,error_if_nonfinite=True)
            readouts=[*model.predictor.head.parameters(),*model.auxiliary.parameters()]
            readout_before=[p.detach().clone() for p in readouts]
            optimizer.step()
            readout_delta={name:float(torch.stack([(parameter.detach()-before).square().sum()
                for parameter,before in zip(readouts[start:start+2],readout_before[start:start+2],strict=True)]).sum().sqrt())
                for name,start in (('state_readout',0),('auxiliary_readout',2))}
            del readout_before
            record=dict(new_optimizer_update=update,total_optimizer_updates=repair.SOURCE_UPDATES+update,
                new_training_microbatches=update*repair.MICROBATCHES,learning_rates=rates,
                loss=sum(original.LOSS_WEIGHTS[key]*value for key,value in totals.items()),parts=totals,
                grad_norm_before_clip=float(norm),group_grad_norms_before_clip=group_norms,
                actual_readout_parameter_update_l2=readout_delta,
                state_precision_denominator=state_count,mass_precision_denominator=mass_count,
                all_rows=80,microbatches=micro_records,elapsed_s=time.monotonic()-started)
            log.write(json.dumps(record)+'\n')
            print(json.dumps({key:value for key,value in record.items() if key!='microbatches'}),flush=True)
            if update%25==0:
                total=repair.SOURCE_UPDATES+update
                report,arrays=save_evaluation(root,f'fit_{total:05d}',model,dataset,full_vertices,device)
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=total,
                    new_optimizer_updates=update,protocol=protocol),root/'latest.pt')
    total=repair.SOURCE_UPDATES+repair.NEW_UPDATES
    final_optimizer=repair.verify_optimizer(model,optimizer,total,repair.NEW_UPDATES)
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=total,
                    new_optimizer_updates=repair.NEW_UPDATES,protocol=protocol),root/'model.pt')
    update_report=repair.update_report(model,optimizer,initial)
    original.save_json(root/'PARAMETER_UPDATES.json',dict(update_report,source_optimizer_updates=repair.SOURCE_UPDATES,
        new_optimizer_updates=repair.NEW_UPDATES,total_optimizer_updates=total,
        parameters_compared_to='Actual source2000 checkpoint before any refinement',final_optimizer=final_optimizer))
    del initial
    batch=original.device_batch(repair.collate_floor([dataset[0]]),device)
    model.eval()
    with torch.no_grad():
        before={key:value.detach().clone() for key,value in model(batch['inputs']).items()}
        saved=torch.load(root/'model.pt',map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(saved['model'],strict=True)
        after=model(batch['inputs'])
        reload_difference=max(float((before[key]-after[key]).abs().max()) for key in before)
    del saved
    dense_result,dense_arrays=save_evaluation(root,'same_trajectory_interpolation',model,interpolation,full_vertices,device)
    clock_keys=('episode','frame','timestamp_s','target','force_target_n','mass_available','mass_status',
                'contact_present','state_precision_eligible','state_contact_history_frames','state_evidence_status','floor_height_m')
    exact=all(np.array_equal(initial_dense[key],dense_arrays[key]) for key in clock_keys)
    fit_exact=all(np.array_equal(initial_arrays[key],arrays[key]) for key in clock_keys)
    validate_prepared(source,root,args.matched_baseline)
    repair.finish_artifacts(root,initial_artifacts)
    accepted=bool(report['passed'] and dense_result['passed'] and update_report['passed'] and reload_difference==0 and exact and fit_exact)
    result=dict(execution_complete=True,study=repair.STUDY,steps=repair.NEW_UPDATES,
        optimizer_updates=repair.NEW_UPDATES,source_optimizer_updates=repair.SOURCE_UPDATES,
        total_optimizer_updates=total,training_microbatches=repair.NEW_UPDATES*repair.MICROBATCHES,
        acceptance_passed=accepted,failure_aware_learning_acceptance_passed=accepted,
        fit_passed=report['passed'],same_trajectory_interpolation_passed=dense_result['passed'],
        data_qualification_passed=True,full_parameter_update_passed=update_report['passed'],
        full_endpoint_reload_max_abs=reload_difference,fit_items=80,interpolation_items=len(interpolation),
        initial_final_interpolation_clocks_targets_exact=exact,initial_final_fit_clocks_targets_exact=fit_exact,
        initial_prediction_total_optimizer_updates=repair.SOURCE_UPDATES,endpoint_prediction_file='fit_02100.npz',
        physical_success_gate_passed=data_report['physical_success_gate_passed'],
        original_conditional_qualification_passed=data_report['original_conditional_qualification_passed'],
        learnable_data_gate_passed=data_report['learnable_data_gate_passed'],
        mass_cases_with_available_fit_targets=report['mass_cases_with_available_targets'],
        mass_cases_without_available_fit_targets=report['mass_cases_without_available_targets'],
        supervision_profile=original.FAILURE_AWARE_PROFILE,physical_success_or_tactile_benefit_claimed=False,
        svd_coefficients_imported=False,retained_resource=resource,elapsed_s=time.monotonic()-started,
        floor_observation_only=True,zero_branch_all_initial_predictions_exact=True,
        matched_baseline=protocol['matched_baseline'],
        scope=protocol['scope'],renders='Pending root actual full-mesh renderer; all sixteen retained',
        continuation_scope='User authorized continued bounded repairs; next experiments remain separately declared, no automatic extension of this100-update run')
    original.save_json(root/'RESULT.json',result)
    print(json.dumps(result),flush=True)
    return original.completion_exit_code(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--matched-baseline',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--check-data',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        args.output.mkdir(parents=True,exist_ok=True)
        if (args.output/'PROTOCOL.json').exists():raise FileExistsError('Prepared protocol already exists')
        original.save_json(args.output/'PROTOCOL.json',repair.source_protocol(args.source,args.matched_baseline))
        return
    try:code=run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error,FileExistsError):
            original.save_json(args.output/'FAILURE.json',dict(execution_complete=False,
                error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise
    raise SystemExit(code)


if __name__=='__main__':main()
