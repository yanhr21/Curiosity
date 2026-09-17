"""One fixed full-official mode-AE continuation: mode1000→2000 atlr1e-5.

Restore every original parameter and Adam state; change only optimizer group lr.
Original model/public posterior mode, KLNearFar, query rules and geometric gates
remain. This is an explicitly new continuation, not a matched fresh1000 run or
proof that a local single-step improvement repairs the final shape. No best
checkpoint selection, variable budget, mesh filtering or automatic follow-up.
prepare/check are CPU only. Root alone launches run in the retained shared lock.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
from types import FunctionType
import numpy as np
from . import michelangelo_lowstep_continuation_data as sampling
from . import train_michelangelo_mode_grid as prior
from . import train_michelangelo_overfit as original
from . import diagnose_michelangelo_adam as state
from .qualify_michelangelo_ae import CHECKPOINT_SHA,OBJECT_IDS,VENDOR,digest,dependency_check

SCOPE=sampling.SCOPE
REVISION=sampling.REVISION
batch_loss=prior.batch_loss
refresh=prior.refresh
# Original complete model/Adam equality audit with private endpoint ceiling2000.
# No change to the frozen function or its globals/algorithm.
audit_endpoint=FunctionType(original.audit_saved_checkpoint.__code__,
    dict(original.audit_saved_checkpoint.__globals__,STEPS=sampling.END),
    original.audit_saved_checkpoint.__name__,original.audit_saved_checkpoint.__defaults__)


def bindings(directory,manifest):
    return dict(data_manifest_sha256=digest(directory/'MANIFEST.json'),
        data_adapter_sha256=digest(Path(sampling.__file__)),trainer_sha256=digest(Path(__file__)),
        state_helper_sha256=digest(Path(state.__file__)),prior_trainer_sha256=digest(Path(prior.__file__)),
        original_trainer_sha256=digest(Path(original.__file__)),
        original_sources={str(p.relative_to(VENDOR)):digest(p) for p in VENDOR.rglob('*.py')},
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'])


def load_source_checkpoint(manifest):
    import torch
    source=sampling.SOURCE
    saved=torch.load(source/manifest['sources']['checkpoint_file'],map_location='cpu',mmap=True,weights_only=False)
    protocol=json.loads((source/'PROTOCOL.json').read_text())
    if (saved['step']!=1000 or saved['protocol']!=protocol or saved['initial_checkpoint_sha256']!=CHECKPOINT_SHA
            or len(saved['parameter_names'])!=307 or len(saved['optimizer']['state'])!=307):
        raise ValueError('Require actual full mode1000 checkpoint and complete Adam')
    groups=saved['optimizer']['param_groups']
    if len(groups)!=1 or groups[0]['lr']!=1e-4 or tuple(groups[0]['betas'])!=(.9,.99):
        raise ValueError('Source optimizer recipe changed')
    for field,value in [('eps',1e-6),('weight_decay',.01)]:
        if groups[0][field]!=value:raise ValueError('Source Adam setting changed: '+field)
    names=saved['parameter_names'];ids=[i for group in groups for i in group['params']]
    if len(ids)!=307 or sum(saved['model'][name].numel() for name in names)!=184659585:
        raise ValueError('Incomplete source full model')
    for index,name in zip(ids,names):
        row=saved['optimizer']['state'][index]
        if set(row)!={'step','exp_avg','exp_avg_sq'} or float(row['step'])!=1000:
            raise ValueError('Source Adam clock/fields changed')
        if any(not torch.isfinite(x).all() for x in row.values()) or not torch.isfinite(saved['model'][name]).all():
            raise ValueError('Nonfinite source model or Adam')
        if any(row[k].shape!=saved['model'][name].shape for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Source Adam shape mismatch')
    return saved


def restore_for_continuation(model,optimizer,saved,device):
    """Reuse actual diagnosed no-alias restore, then modify only group lr."""
    source_hash=state.snapshot_digest(saved)
    receipt=state.restore(model,optimizer,saved,device)
    for group in optimizer.param_groups:group['lr']=sampling.LR
    # Compare every live moment/clock and full parameter to the source after
    # replacing just its expected group lr. Source cached dictionaries stay raw.
    expected=dict(saved,optimizer=dict(saved['optimizer'],param_groups=[dict(g,lr=sampling.LR)
                    for g in saved['optimizer']['param_groups']]))
    check=state.match_state(model,optimizer,expected)
    if state.snapshot_digest(saved)!=source_hash:raise RuntimeError('Restore mutated cached source')
    return dict(original_restore=receipt,after_only_lr_change=check,source_cached_tensors_unchanged=True,
                lr_before=1e-4,lr_after=sampling.LR,all_adam_steps_before=1000),source_hash


def run(args):
    from .retained_execution import require_active_resource
    resource=require_active_resource()
    import torch
    manifest,queries,cases,pools,source_result=sampling.read_data(args.data)
    args.output.mkdir(parents=True,exist_ok=False)
    bound=bindings(args.data,manifest)
    protocol=dict(scope=SCOPE,sampling_revision=REVISION,bindings=bound,resource=resource,
        recipe=sampling.recipe(),model_updates=1000,source_model_updates=1000,total_model_updates=2000,
        updates_per_object=250,source_updates_per_object=250,total_updates_per_object=500,initial_checkpoint_sha256=CHECKPOINT_SHA,
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],
        checkpoint_reset=False,loss='original michelangelo.models.tsal.loss.KLNearFar',near_weight=.1,kl_weight=.001,
        optimizer=dict(name='AdamW',lr=sampling.LR,betas=[.9,.99],eps=1e-6,weight_decay=.01),
        schedule='fixed lr1e-5 atabsolute1001..2000; no sweep, selection or early stop',
        train_seed=original.TRAIN_SEED,batch_size=1,batch_queries=dict(volume=1024,near=1024),input_points=4096,
        train_posterior='mode',refresh_posterior='mode',eval_posterior='mode',full_model_trainable=True,
        data_manifest=manifest,data_manifest_sha256=digest(args.data/'MANIFEST.json'),
        previous_result=str((sampling.SOURCE/'RESULT.json').resolve()),
        previous_result_sha256=manifest['sources']['previous_result_sha256'],previous_cases=manifest['previous_cases'],
        gates_unchanged=dict(cd_x9000_max=.45,fscore_min=.95,area_ratio=[.7,1.5],
            max_vertex_and_face_probe=.01,occupancy_iou_min=.90),
        sources=bound['original_sources'],adapter_sha256=bound['trainer_sha256'],
        initial_mesh_output='bound mode1000 source four raw/corrected mesh NPZs; not re-extracted',
        initial_pool_source=manifest['sources']['pool_sources'],
        geometry_evaluation='unchanged original endpoint_meshes helper including official raw and corrected coordinates',
        visual_inspection_required=True,physics_controls=0,tactile_model_forwards=0,
        task_adaptation='deterministic native AE continuation; no unknown SUGAR shape/mass success claim')
    original.write_json(args.output/'PROTOCOL.json',protocol)
    saved_source=load_source_checkpoint(manifest);device=torch.device('cuda')
    loading=source_result['loading'];cls=state.official_class(loading['official_config'])
    model=cls(device=None,dtype=None,**loading['official_config']['params']).to(device)
    if [n for n,_ in model.named_parameters()]!=saved_source['parameter_names']:
        raise RuntimeError('Full original parameter names/order changed')
    optimizer=original.optimizer_for(model);criterion=original.original_loss()
    restored,source_hash=restore_for_continuation(model,optimizer,saved_source,device)
    original.write_json(args.output/'INITIAL_RESTORE.json',restored)
    data=[c['original'] for c in cases]
    initial_eval=original.evaluate_queries(model,criterion,data,device)
    expected=json.loads((sampling.SOURCE/'RELOAD.json').read_text())['cases']
    if initial_eval!=expected:raise RuntimeError('Initial mode1000 heldout/latent outputs are not EXACT; 0newupdates')
    reports=[dict(step=1000,cases=initial_eval)]
    original.write_json(args.output/'CURVE.json',reports)
    original.write_json(args.output/'INITIAL_STATE.json',dict(model_updates=1000,new_model_updates=0,
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],
        original_heldout_and_latent_exact=True,checkpoint_selection=False,cases=manifest['previous_cases']))
    print('LOWSTEP_INITIAL_FULL_MODEL_ADAM_AND_HELDOUT_EXACT',flush=True)
    traces=[];refreshes=[];current_refresh=1000
    try:
        with (args.output/'updates.jsonl').open('x') as log:
            for step in range(1001,2001):
                if step-1 in sampling.REFRESH_STEPS:
                    current_refresh=step-1
                    pools,record=refresh(model,queries,cases,device,args.output/'grid_refresh',current_refresh)
                    refreshes.append(record);original.write_json(args.output/'REFRESHES.json',refreshes)
                index,item,indices,volume,near,info=sampling.batch_for_step(step,queries,cases,pools)
                torch.manual_seed(original.TRAIN_SEED+step);model.train();optimizer.zero_grad(set_to_none=True)
                loss,components=batch_loss(model,criterion,item,indices,device)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite original loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                if step in (1001,2000):original.write_json(args.output/f'GRADIENT_STEP_{step}.json',original.gradients(model))
                optimizer.step()
                if any(float(v['step'])!=step for v in optimizer.state.values()):
                    raise RuntimeError('Full Adam absolute step drift')
                traces.append((step,index,current_refresh,volume.copy(),near.copy()))
                log.write(json.dumps(dict(step=step,object_id=OBJECT_IDS[index],seed=original.TRAIN_SEED+step,
                    loss=float(loss),gradient_norm=float(norm),loss_components={k:float(v) for k,v in components.items()},
                    learning_rate=sampling.LR,refresh_step=current_refresh,sampling=info,
                    volume_indices_sha256=prior.array_sha(volume),near_indices_sha256=prior.array_sha(near)))+'\n');log.flush()
                if step%100==0:
                    row=dict(step=step,cases=original.evaluate_queries(model,criterion,data,device))
                    reports.append(row);original.write_json(args.output/'CURVE.json',reports)
                    print('LOWSTEP_UPDATE',step,row,flush=True)
    finally:
        np.savez_compressed(args.output/'SAMPLING_TRACE.npz',steps=np.array([x[0] for x in traces],np.int32),
            case_indices=np.array([x[1] for x in traces],np.int32),refresh_steps=np.array([x[2] for x in traces],np.int32),
            volume_grid_indices=np.stack([x[3] for x in traces]) if traces else np.empty((0,1024),np.int32),
            near_indices=np.stack([x[4] for x in traces]) if traces else np.empty((0,1024),np.int32))
    if len(traces)!=1000:raise RuntimeError('Incomplete fixed continuation budget')
    if state.snapshot_digest(saved_source)!=source_hash:raise RuntimeError('Cached source was mutated by continuation')
    original.write_json(args.output/'SOURCE_CACHE_READBACK.json',dict(passed=True,
        initial_hash=source_hash,final_hash=state.snapshot_digest(saved_source),all_original_adam_steps=1000))
    del saved_source;gc.collect()
    # Save full endpoint before downstream grid/mesh evaluation can fail.
    checkpoint=args.output/'endpoint.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=2000,
        parameter_names=[n for n,_ in model.named_parameters()],initial_checkpoint_sha256=CHECKPOINT_SHA,
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],protocol=protocol,
        rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),checkpoint)
    audit,saved=audit_endpoint(checkpoint,model,optimizer)
    original.write_json(args.output/'FULL_ADAM_READBACK.json',audit)
    _,endpoint_grid=refresh(model,queries,cases,device,args.output/'endpoint_grid',2000)
    original.write_json(args.output/'ENDPOINT_GRID.json',endpoint_grid)
    before=original.evaluate_queries(model,criterion,data,device)
    config=loading['official_config']['params']
    del model,optimizer,loss,components;gc.collect();torch.cuda.empty_cache()
    model=cls(device=None,dtype=None,**config);model.load_state_dict(saved['model'],strict=True)
    model=model.to(device).requires_grad_(False).eval();del saved;gc.collect()
    after=original.evaluate_queries(model,criterion,data,device)
    if before!=after:raise RuntimeError('Actual complete2000 endpoint reload differs')
    original.write_json(args.output/'RELOAD.json',dict(passed=True,all_four_query_metrics_exact=True,
        all_four_logits_and_latents_exact=True,cases=after))
    records=original.endpoint_meshes(model,data,manifest['original_manifest'],args.output,device)
    result=dict(complete=True,scope=SCOPE,sampling_revision=REVISION,cases=records,
        model_updates=1000,source_model_updates=1000,total_model_updates=2000,
        checkpoint_reset=False,
        updates_per_object=250,source_updates_per_object=250,total_updates_per_object=500,initial_checkpoint_sha256=CHECKPOINT_SHA,
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],checkpoint_file=checkpoint.name,
        checkpoint_sha256=digest(checkpoint),numerical_passed=all(r['checks']['passed'] for r in records),
        representation_qualified=False,visual_inspection='pending',full_adam_audit=audit,
        source_full_gradient_qualification_passed=source_result['full_gradient_qualification_passed'],
        continuation_gradient_check_steps=[1001,2000],actual_checkpoint_reload_passed=True,loading=loading,
        endpoint_grid=endpoint_grid,initial_pool_step=1000,formal_refresh_steps=[r['step'] for r in refreshes],
        physics_controls=0,tactile_model_forwards=0,clip_alignment_continued=False)
    original.write_json(args.output/'RESULT.json',result)
    return 0 if result['numerical_passed'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','run'))
    parser.add_argument('--data',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':
        if args.output is None:parser.error('prepare requires new --output')
        print(json.dumps(sampling.prepare(args.output)));return 0
    if args.data is None:parser.error('check/run require --data')
    if args.stage=='check':
        manifest,queries,cases,pools,result=sampling.read_data(args.data)
        deps=dependency_check();criterion=original.original_loss()
        import torch
        saved=load_source_checkpoint(manifest);cls=state.official_class(result['loading']['official_config'])
        with torch.device('meta'):model=cls(device=None,dtype=None,**result['loading']['official_config']['params'])
        if [n for n,_ in model.named_parameters()]!=saved['parameter_names']:
            raise RuntimeError('Official architecture/source parameter order mismatch')
        if torch.cuda.is_initialized() or not all(deps.values()):raise RuntimeError('CPU dependency qualification failed')
        print(json.dumps(dict(passed=True,scope=SCOPE,recipe=sampling.recipe(),dependencies=deps,
            parameters=184659585,parameter_tensors=307,source_adam_states=307,every_source_adam_step=1000,
            grid_nodes=len(queries),initial_error_pool_counts=[{k:len(v) for k,v in p.items()} for p in pools],
            original_class=type(model).__module__+'.'+type(model).__name__,
            original_loss=type(criterion).__module__+'.'+type(criterion).__name__,
            cuda_initialized=False,model_forwards=0,model_updates=0)));return 0
    if args.output is None:parser.error('run requires new --output')
    return run(args)

if __name__=='__main__':raise SystemExit(main())
