"""One fresh2100 force-head block update, then unchanged80/1440 evaluation.

Preparation and CPU checks never construct or forward the complete model.
Execution requires the original retained-resource/lock guard and separate launch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from . import train_overfit as original
from . import overfit_force_head_solve as repair
from . import audit_overfit_readout_features as capture
from .train_overfit_fullbatch_refinement import save_evaluation


class FullForwardCounter:
    """Count successful original full-model calls, excluding direct head calls."""
    def __init__(self,model):self.model=model;self.calls=0;self.handle=None
    def __enter__(self):
        def count(_module,_args,_output):self.calls+=1
        self.handle=self.model.register_forward_hook(count)
        return self
    def __exit__(self,*_error):
        self.handle.remove();self.handle=None


def load_arrays(path):
    with np.load(path,allow_pickle=False) as value:return {k:value[k] for k in value.files}


def validate_prepared(source,root):
    saved=json.loads((Path(root)/'PROTOCOL.json').read_text())
    if saved!=repair.source_protocol(source):raise ValueError('Prepared source/protocol changed')
    return saved


def evaluate_capture(model,dataset,vertices,device):
    """Original evaluate performs the only full forwards; hooks change nothing."""
    features=[]
    handle=model.predictor.head.register_forward_pre_hook(
        lambda _module,args:features.append(args[0].detach().cpu().numpy().copy()))
    try:report,arrays=original.evaluate(model,dataset,vertices,device)
    finally:handle.remove()
    width=model.predictor.head.in_features
    if len(features)!=len(dataset) or any(x.shape!=(1,width) for x in features):
        raise RuntimeError('Exactly one original chronological head input per row required')
    matrix=np.concatenate(features)
    if matrix.dtype!=np.float32 or not np.isfinite(matrix).all():
        raise FloatingPointError('Original float32 features must be finite')
    return report,arrays,matrix


def reload_pair(model,state,batch):
    """Two same-process complete forwards around a full strict checkpoint load."""
    model.eval()
    with torch.no_grad():
        before={k:v.detach().clone() for k,v in capture.forward_observations(model,batch).items()}
        model.load_state_dict(state,strict=True)
        after=capture.forward_observations(model,batch)
    exact={k:bool(torch.equal(before[k],after[k])) for k in before}
    difference={k:float((before[k]-after[k]).abs().max()) for k in before}
    return dict(passed=all(exact.values()),exact=exact,max_abs_difference=difference,
        maximum=max(difference.values()),scope='Same-process same-input full checkpoint reload; separate from historical cross-process comparison')


@torch.no_grad()
def cached_readouts(model,features,device):
    x=torch.as_tensor(features,device=device)
    # Original existing modules, including all10 auxiliary rows. No new model.
    return dict(state=model.predictor.head(x).detach().cpu().numpy(),
                auxiliary=model.auxiliary(x).detach().cpu().numpy())


def same_feature_check(before,after,targets):
    if any(not np.isfinite(v).all() for values in (before,after) for v in values.values()):
        raise FloatingPointError('Nonfinite actual cached-head output')
    exact_state=bool(np.array_equal(before['state'],after['state']))
    exact_other=bool(np.array_equal(before['auxiliary'][:,8:],after['auxiliary'][:,8:]))
    if not exact_state or not exact_other:
        raise RuntimeError('Non-force output changed for the identical captured features')
    residual=after['auxiliary'][:,:8].astype(float)-targets.astype(float)
    per_row=np.sqrt(np.mean(residual**2,axis=1))
    return dict(passed=True,state13_exact=exact_state,other_auxiliary2_logits_exact=exact_other,
        force_item_rmse_n=per_row.tolist(),force_rmse_n=float(np.sqrt(np.mean(residual**2))),
        force_every_item_gate=bool((per_row<=original.FIT_LIMITS['force_rmse_n']).all()),
        arithmetic='Original float32 existing readout modules, same80 captured features, CUDA/TF32 as declared; no full-backbone recapture here')


def force_gate(report):
    return bool(all(row['checks']['force'] for row in report['per_case'].values()))


def run(args):
    root,source=Path(args.output),Path(args.source).resolve()
    protocol=validate_prepared(source,root)
    if any((root/name).exists() for name in ('RESULT.json','FAILURE.json','model.pt','FRESH_FEATURES.npz')):
        raise FileExistsError('No overwrite, restart or automatic retry')
    if not args.check_data:
        resource=original.require_active_resource()
        artifacts=repair.begin_artifacts(root)
        original.save_json(root/'UPDATE_LEDGER.json',dict(learning_updates=0,closed_form_parameter_updates=0,
            optimizer_updates=0,phase='preflight',source_optimizer_updates=2100))
    fit=original.FailureAwareOverfitDataset(protocol['source_data'],protocol['data'])
    data_report=original.qualification(fit)
    source_fit=load_arrays(source/'fit_02100.npz')
    fresh=capture.check_saved_rows(fit,source_fit)
    if not data_report['passed']:raise ValueError('Complete original failure-aware fit gate failed')
    original.save_json(root/('CPU_INPUT_CHECK.json' if args.check_data else 'DATA_QUALIFICATION.json'),
        dict(data_report,source_fit_targets_masks_exact=True,fresh_feature_forwards=0,
             checkpoint_feature_layout=protocol['feature_layout'],cuda_initialized=torch.cuda.is_initialized()))
    if args.check_data:
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print(json.dumps(dict(passed=True,fit_input=True,rows=80,state_rows=int(fresh['state_precision_eligible'].sum()),
            mass_rows=int(fresh['mass_available'].sum()),full_model_constructed=False,model_forwards=0,cuda_initialized=False)),flush=True)
        return 0
    dense=original.FailureAwareOverfitDataset(protocol['source_data'],protocol['data'],
        interpolation_frames=original.INTERPOLATION_FRAMES)
    dense_data=original.failure_aware_dense_qualification(dense,frames=original.INTERPOLATION_FRAMES,fit_limits=original.FIT_LIMITS)
    original.save_json(root/'INTERPOLATION_DATA_QUALIFICATION.json',dense_data)
    if not dense_data['passed']:raise ValueError('Dense original data gate failed before model construction')
    if original.require_active_resource()!=resource:raise RuntimeError('Retained resource changed')
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    torch.cuda.manual_seed_all(protocol['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda')
    full_vertices,_sampled,mesh_report=original.read_known_mesh(protocol['mesh'])
    original.save_json(root/'KNOWN_MESH.json',mesh_report)
    from .model import disable_stochastic_regularizers
    model=original.FullUtoniaOverfit(protocol['checkpoint'],history=32).to(device)
    endpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False,mmap=True)
    if endpoint['step']!=2100:raise ValueError('Only actual source2100 is allowed')
    model.load_state_dict(endpoint['model'],strict=True)
    model.eval();disable_stochastic_regularizers(model)
    with FullForwardCounter(model) as forward_counter:
        restored=repair.preserved_state(endpoint['model'],model.state_dict())
        if not all(row['source_equals_endpoint'] for row in restored['named_state_tensors']):
            raise ValueError('Initial whole-model restoration is not exact')
        # No optimizer is constructed or stepped. Preserve complete original Adam
        # only as archival evidence; this checkpoint will forbid ordinary resumption.
        if len(endpoint['optimizer']['state'])!=457 or any(int(s['step'])!=2100 for s in endpoint['optimizer']['state'].values()):
            raise ValueError('Source complete Adam2100 archive missing')
        batch=original.device_batch(original.collate_conditional([fit[0]]),device)
        initial_reload=reload_pair(model,endpoint['model'],batch)
        original.save_json(root/'INITIAL_SAME_PROCESS_RELOAD.json',initial_reload)
        if not initial_reload['passed']:raise RuntimeError('Same-process source checkpoint reload differs; no learning update applied')
        started=time.monotonic()
        initial_report,initial_arrays,x=evaluate_capture(model,fit,full_vertices,device)
        original.save_json(root/'initial_fit.json',initial_report)
        np.savez_compressed(root/'initial_fit.npz',**initial_arrays)
        np.savez_compressed(root/'FRESH_FEATURES.npz',features=x,**initial_arrays)
        source_comparison=capture.replay_comparison(initial_arrays,source_fit)
        source_comparison.update(source_optimizer_updates=2100,scope='Fresh actual source2100 restoration versus historical source2100 saved fit; differences recorded, no cross-process exactness requirement or GT correction',same_process_reload=initial_reload)
        original.save_json(root/'SOURCE_RELOAD_COMPARISON.json',source_comparison)
        # Explicit historical source baseline, never advertised as fresh inference.
        shutil.copyfile(source/'same_trajectory_interpolation.npz',root/'initial_same_trajectory_interpolation.npz')
        shutil.copyfile(source/'same_trajectory_interpolation.json',root/'initial_same_trajectory_interpolation.json')
        old_dense=load_arrays(source/'same_trajectory_interpolation.npz')
        original.save_json(root/'HISTORICAL_BASELINE.json',dict(dense_source=str(source/'same_trajectory_interpolation.npz'),
            sha256=original.sha256_file(source/'same_trajectory_interpolation.npz'),fresh_dense_forwards=0,total_optimizer_updates=2100))
        before_cached=cached_readouts(model,x,device)
        w,b,solve=repair.minimum_increment(x,initial_arrays['force_target_n'],
            model.auxiliary.weight[:8].detach().cpu().numpy(),model.auxiliary.bias[:8].detach().cpu().numpy())
        original.save_json(root/'SOLVE.json',solve)
        repair.apply_force_rows(model,w,b)  # Exactly one explicitly counted learning update.
        original.save_json(root/'UPDATE_LEDGER.json',dict(learning_updates=1,closed_form_parameter_updates=1,
            optimizer_updates=0,phase='applied_exactly_once',source_optimizer_updates=2100))
        after_cached=cached_readouts(model,x,device)
        same=same_feature_check(before_cached,after_cached,initial_arrays['force_target_n'])
        original.save_json(root/'NON_FORCE_SAME_FEATURES.json',same)
        preserved=repair.preserved_state(endpoint['model'],model.state_dict())
        original.save_json(root/'PARAMETER_PRESERVATION.json',preserved)
        state={key:value.detach().cpu() for key,value in model.state_dict().items()}
        torch.save(dict(model=state,optimizer_archive=endpoint['optimizer'],optimizer_resume_supported=False,
            step=2100,source_optimizer_updates=2100,new_optimizer_updates=0,closed_form_parameter_updates=1,
            total_learning_updates=2101,protocol=protocol),root/'model.pt')
        saved=torch.load(root/'model.pt',map_location='cpu',weights_only=False,mmap=True)
        final_reload=reload_pair(model,saved['model'],batch)
        original.save_json(root/'FINAL_SAME_PROCESS_RELOAD.json',final_reload)
        fit_report,fit_arrays=save_evaluation(root,'fit_force_update_0001',model,fit,full_vertices,device)
        dense_report,dense_arrays=save_evaluation(root,'same_trajectory_interpolation',model,dense,full_vertices,device)
        keys=('episode','frame','timestamp_s','target','force_target_n','mass_available','mass_status',
            'contact_present','state_precision_eligible','state_contact_history_frames','state_evidence_status')
        exact_fit=all(np.array_equal(initial_arrays[k],fit_arrays[k]) for k in keys)
        exact_dense=all(np.array_equal(old_dense[k],dense_arrays[k]) for k in keys)
        fresh_nonforce={key:dict(exact=bool(np.array_equal(initial_arrays[key],fit_arrays[key])),
            maximum=float(np.max(np.abs(initial_arrays[key].astype(float)-fit_arrays[key].astype(float)))))
            for key in ('prediction','availability_probability','contact_probability')}
        if forward_counter.calls!=protocol['full_forward_budget']:
            raise RuntimeError('Actual full-forward budget differs from declaration')
        validate_prepared(source,root)
        repair.finish_artifacts(root,artifacts)
        complete_integrity=bool(exact_fit and exact_dense and final_reload['passed'] and preserved['passed'] and same['passed'])
        accepted=bool(complete_integrity and fit_report['passed'] and dense_report['passed'])
        force_passed=bool(complete_integrity and same['force_every_item_gate'] and force_gate(fit_report) and force_gate(dense_report))
        result=dict(execution_complete=True,study=repair.STUDY,optimizer_updates=0,learning_updates=1,
            closed_form_parameter_updates=1,source_optimizer_updates=2100,total_adam_optimizer_updates=2100,
            total_learning_updates=2101,full_backbone_updated=False,all_other_model_parameters_exact=True,
            complete_original_model_retained=True,source_optimizer_archived=True,optimizer_resume_supported=False,
            acceptance_passed=accepted,force_readout_qualification_passed=force_passed,
            fit_passed=fit_report['passed'],same_trajectory_interpolation_passed=dense_report['passed'],
            fit_force_passed=force_gate(fit_report),interpolation_force_passed=force_gate(dense_report),
            cached_force_gate_passed=same['force_every_item_gate'],integrity_passed=complete_integrity,
            full_endpoint_reload_max_abs=final_reload['maximum'],same_feature_non_force_exact=True,
            fresh_full_forward_non_force_differences=fresh_nonforce,
            historical_source_replay_exact=source_comparison['passed'],fit_items=80,interpolation_items=1440,
            initial_final_fit_clocks_targets_exact=exact_fit,initial_final_interpolation_clocks_targets_exact=exact_dense,
            initial_fit_source='Fresh source2100 forwards',initial_dense_source='Historical actual source2100 saved evaluation',
            endpoint_prediction_file='fit_force_update_0001.npz',model_forwards=forward_counter.calls,
            physical_success_gate_passed=data_report['physical_success_gate_passed'],
            original_conditional_qualification_passed=data_report['original_conditional_qualification_passed'],
            learnable_data_gate_passed=data_report['learnable_data_gate_passed'],
            retained_resource=resource,elapsed_s=time.monotonic()-started,
            scope=protocol['scope'],continuation='Prepared/executed one-update qualification only; no automatic deployment, Adam resumption, extra solve or full-model training.')
        original.save_json(root/'RESULT.json',result)
        print(json.dumps(result),flush=True)
        return 0 if accepted else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--check-data',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        args.output.mkdir(parents=True,exist_ok=True)
        if (args.output/'PROTOCOL.json').exists():raise FileExistsError('Prepared protocol exists')
        original.save_json(args.output/'PROTOCOL.json',repair.source_protocol(args.source))
        return
    try:code=run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error,FileExistsError):
            original.save_json(args.output/'FAILURE.json',dict(execution_complete=False,error_type=type(error).__name__,
                error=str(error),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise
    raise SystemExit(code)


if __name__=='__main__':main()
