"""Official full AE lowstep2000->2832 with fixed persistent query replay.

prepare/check are CPU only. qualify has4 encode/4 decode/48 query calls and no
updates; all four saved source latents/heldout logits must replay EXACT before
querying TRAIN-near. run restores the same full source+Adam and performs exactly
832 new updates. No heldout-guided sampler, mesh filtering, best checkpoint,
automatic budget extension, replacement model or SUGAR learning claim.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import traceback
from types import FunctionType
import numpy as np
from . import michelangelo_persistent_replay_data as sampling
from . import train_michelangelo_mode_grid as prior
from . import train_michelangelo_overfit as original
from . import diagnose_michelangelo_adam as state
from .audit_michelangelo_extraction_signs import sign_arrays_and_metrics
from .qualify_michelangelo_ae import CHECKPOINT_SHA,OBJECT_IDS,VENDOR,digest,dependency_check

SCOPE,REVISION=sampling.SCOPE,sampling.REVISION
batch_loss=prior.batch_loss
FORMAL_BUDGET=dict(forward=832,encode=920,decode=916,query_geometry=9960)
audit_endpoint=FunctionType(original.audit_saved_checkpoint.__code__,
    dict(original.audit_saved_checkpoint.__globals__,STEPS=sampling.END),
    original.audit_saved_checkpoint.__name__,original.audit_saved_checkpoint.__defaults__)


class MethodCounter:
    """Count actual official bound methods, including failed attempts."""
    def __init__(self,budget):
        self.budget={k:budget[k] for k in ('forward','encode','decode','query_geometry')}
        self.attempted=Counter();self.completed=Counter();self.originals=[]
    def install(self,model):
        if self.originals:raise RuntimeError('Detach previous counted model first')
        for name in self.budget:
            method=getattr(model,name);self.originals.append((model,name,method))
            def wrapper(*args,_name=name,_method=method,**kwargs):
                self.attempted[_name]+=1
                if self.attempted[_name]>self.budget[_name]:raise RuntimeError('Declared call budget exceeded: '+_name)
                value=_method(*args,**kwargs);self.completed[_name]+=1;return value
            setattr(model,name,wrapper)
    def detach(self):
        for model,name,method in self.originals:setattr(model,name,method)
        self.originals.clear()
    def report(self):
        return dict(budget=self.budget,attempted={k:self.attempted[k] for k in self.budget},
                    completed={k:self.completed[k] for k in self.budget})
    def require_complete(self):
        if any(self.attempted[k]!=v or self.completed[k]!=v for k,v in self.budget.items()):
            raise RuntimeError('Actual official calls differ from fixed complete budget')


def bindings(directory,manifest):
    return dict(data_manifest_sha256=digest(directory/'MANIFEST.json'),
        data_adapter_sha256=digest(Path(sampling.__file__)),trainer_sha256=digest(Path(__file__)),
        state_helper_sha256=digest(Path(state.__file__)),prior_trainer_sha256=digest(Path(prior.__file__)),
        original_trainer_sha256=digest(Path(original.__file__)),
        original_sources={str(p.relative_to(VENDOR)):digest(p) for p in VENDOR.rglob('*.py')},
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'])


def load_source_checkpoint(manifest):
    import torch
    path=sampling.SOURCE/manifest['sources']['checkpoint_file']
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    protocol=json.loads((sampling.SOURCE/'PROTOCOL.json').read_text())
    if (saved['step']!=sampling.START or saved['protocol']!=protocol
            or saved['initial_checkpoint_sha256']!=CHECKPOINT_SHA
            or len(saved['parameter_names'])!=307 or len(saved['optimizer']['state'])!=307):
        raise ValueError('Require complete original lowstep2000 model and Adam')
    groups=saved['optimizer']['param_groups']
    if len(groups)!=1 or groups[0]['lr']!=sampling.LR or tuple(groups[0]['betas'])!=(.9,.99):
        raise ValueError('Source Adam recipe differs')
    if groups[0]['eps']!=1e-6 or groups[0]['weight_decay']!=.01:raise ValueError('Source Adam settings differ')
    names=saved['parameter_names'];ids=groups[0]['params']
    if len(ids)!=307 or sum(saved['model'][n].numel() for n in names)!=184659585:
        raise ValueError('Incomplete source architecture')
    for key,name in zip(ids,names,strict=True):
        row=saved['optimizer']['state'][key]
        if set(row)!={'step','exp_avg','exp_avg_sq'} or float(row['step'])!=sampling.START:
            raise ValueError('Source Adam clock/fields differ')
        if any(not torch.isfinite(v).all() for v in row.values()) or not torch.isfinite(saved['model'][name]).all():
            raise ValueError('Nonfinite source parameter/moment')
        if any(row[k].shape!=saved['model'][name].shape for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Source moment shape differs')
    return saved


def restore_source(model,optimizer,saved,device):
    """No lr change, no moment reset, no alias of source CPU step scalars."""
    import torch
    source_hash=state.snapshot_digest(saved)
    model.load_state_dict(saved['model'],strict=True);model.requires_grad_(True).eval()
    optimizer.load_state_dict(state.clone_optimizer_state(saved['optimizer'],device))
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(saved['rng_state'].clone())
    if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng_state'].clone(),device)
    receipt=state.match_state(model,optimizer,saved,expected_step=sampling.START)
    receipt.update(cpu_rng_exact=torch.equal(torch.get_rng_state(),saved['rng_state']),
        cuda_rng_exact=device.type!='cuda' or torch.equal(torch.cuda.get_rng_state(device),saved['cuda_rng_state']),
        source_cache_unchanged=state.snapshot_digest(saved)==source_hash,lr_unchanged=sampling.LR)
    if not all(receipt[k] for k in ('cpu_rng_exact','cuda_rng_exact','source_cache_unchanged')):
        raise RuntimeError('Source model/Adam/RNG restore failed')
    return receipt,source_hash


def runtime():
    import torch
    return dict(torch_version=torch.__version__,cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,num_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),settings_changed_by_adapter=False)


def new_model(source_result,saved,device):
    cls=state.official_class(source_result['loading']['official_config'])
    model=cls(device=None,dtype=None,**source_result['loading']['official_config']['params']).to(device)
    if [n for n,_ in model.named_parameters()]!=saved['parameter_names']:
        raise RuntimeError('Original model parameter ordering changed')
    return model,cls


def qualify(args):
    from .retained_execution import require_active_resource
    resource=require_active_resource()
    import torch
    manifest,queries,cases,pools,histories,source=sampling.read_data(args.data)
    args.output.mkdir(parents=True,exist_ok=False);bound=bindings(args.data,manifest)
    protocol=dict(scope=SCOPE+'_qualification',sampling_revision=REVISION,bindings=bound,resource=resource,
        recipe=sampling.recipe(),budget=sampling.QUALIFICATION_BUDGET,runtime=runtime(),
        optimizer_updates=0,backward_calls=0,training_near_labels_only=True)
    original.write_json(args.output/'PROTOCOL.json',protocol)
    counter=MethodCounter(sampling.QUALIFICATION_BUDGET);saved=None;model=None
    records=[dict(object_id=oid,passed=False,source_exact=False) for oid in OBJECT_IDS]
    error=None;restored=None;preserved=None;cache=None
    try:
        saved=load_source_checkpoint(manifest);device=torch.device('cuda')
        model,_=new_model(source,saved,device);optimizer=original.optimizer_for(model)
        restored,cache=restore_source(model,optimizer,saved,device);counter.install(model)
        decoded=[];latents=[]
        # All four exact source gates precede every new TRAIN-near query.
        with prior.refresh_context(model,device):
            for i,case in enumerate(cases):
                item=case['original']
                _,latent,_=model.encode(original.tensor(item['input_surface_xyz_vae'],device),
                    original.tensor(item['input_normals'],device),sample_posterior=False)
                features=model.decode(latent);decoded.append(features);latents.append(latent)
                logits=torch.cat([model.query_geometry(original.tensor(item['eval_queries_vae'][s:s+4096],device),features)
                                  for s in range(0,32768,4096)],dim=1)[0].cpu().numpy()
                with np.load(manifest['previous_cases'][i]['file'],allow_pickle=False) as prior_case:
                    if not (np.array_equal(latent[0].cpu().numpy(),prior_case['shape_latent'])
                            and np.array_equal(logits,prior_case['occupancy_logits'])
                            and np.array_equal(item['eval_queries_vae'],prior_case['occupancy_queries_vae'])):
                        raise RuntimeError('Source latent/heldout replay is not EXACT: '+case['object_id'])
                records[i]['source_exact']=True
            for i,case in enumerate(cases):
                item=case['original'];query=item['train_queries_vae'][16384:]
                logits=torch.cat([model.query_geometry(original.tensor(query[s:s+4096],device),decoded[i])
                                  for s in range(0,16384,4096)],dim=1)[0].cpu().numpy()
                near=sampling.near_error_ids(logits,item);path=args.output/(case['object_id']+'_train_near.npz')
                np.savez_compressed(path,train_near_logits=logits,train_near_error_indices=near,
                    shape_latent=latents[i][0].cpu().numpy())
                records[i].update(passed=True,file=path.name,sha256=digest(path),train_near_queries=16384,
                    train_near_errors=len(near),near_query_sha256=prior.array_sha(query),
                    near_label_sha256=prior.array_sha(item['train_labels'][16384:]))
        counter.require_complete()
        preserved=state.match_state(model,optimizer,saved,expected_step=sampling.START)
        if state.snapshot_digest(saved)!=cache:raise RuntimeError('Qualification mutated source cache')
    except Exception:error=traceback.format_exc()
    finally:counter.detach()
    result=dict(scope=SCOPE+'_qualification',sampling_revision=REVISION,bindings=bound,recipe=sampling.recipe(),
        resource=resource,passed=error is None and all(r['passed'] for r in records),cases=records,
        actual_calls=counter.report(),budget=sampling.QUALIFICATION_BUDGET,initial_restore=restored,
        final_model_and_adam_exact=preserved,source_cached_digest=cache,error=error,
        optimizer_updates=0,backward_calls=0,physics_controls=0)
    original.write_json(args.output/'RESULT.json',result)
    return 0 if result['passed'] else 1


def check_qualification(path,bound,resource,cases):
    q=json.loads((path/'RESULT.json').read_text())
    budget={k:sampling.QUALIFICATION_BUDGET[k] for k in ('forward','encode','decode','query_geometry')}
    if (q['scope']!=SCOPE+'_qualification' or not q['passed'] or q['sampling_revision']!=REVISION
            or q['bindings']!=bound or q['resource']!=resource or q['recipe']!=sampling.recipe()
            or q['optimizer_updates']!=0 or q['backward_calls']!=0
            or q['actual_calls']['completed']!=budget or q['actual_calls']['attempted']!=budget
            or tuple(r['object_id'] for r in q['cases'])!=OBJECT_IDS
            or not q['final_model_and_adam_exact']['passed']):
        raise ValueError('Require exact successful same-source actual TRAIN-near qualification')
    near=[]
    for row,case in zip(q['cases'],cases,strict=True):
        item=case['original'];p=path/row['file']
        if (not row['passed'] or not row['source_exact'] or digest(p)!=row['sha256']
                or row['near_query_sha256']!=prior.array_sha(item['train_queries_vae'][16384:])
                or row['near_label_sha256']!=prior.array_sha(item['train_labels'][16384:])):
            raise ValueError('Qualified actual TRAIN-near source differs')
        with np.load(p,allow_pickle=False) as z:
            ids=sampling.near_error_ids(z['train_near_logits'],item)
            if not np.array_equal(ids,z['train_near_error_indices']):raise ValueError('TRAIN-near error replay differs')
        near.append(ids)
    return q,near


def refresh(model,queries,cases,device,output,step):
    """Original official mean-posterior calls, now also on the fixed TRAIN-near pool."""
    import torch
    directory=output/f'step_{step:04d}';directory.mkdir(parents=True,exist_ok=False)
    before_cpu=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state(device).clone();mode=model.training
    pools=[];near=[];records=[]
    with prior.refresh_context(model,device):
        for case in cases:
            item=case['original']
            _,latent,_=model.encode(original.tensor(item['input_surface_xyz_vae'],device),
                original.tensor(item['input_normals'],device),sample_posterior=False)
            decoded=model.decode(latent)
            values=[model.query_geometry(original.tensor(queries[s:s+10000],device),decoded)[0].cpu().numpy().copy()
                    for s in range(0,len(queries),10000)]
            logits=np.concatenate(values);pool=sampling.dynamic.error_pools(logits,case)
            query=item['train_queries_vae'][16384:]
            near_logits=np.concatenate([model.query_geometry(original.tensor(query[s:s+4096],device),decoded)[0].cpu().numpy().copy()
                                        for s in range(0,16384,4096)])
            errors=sampling.near_error_ids(near_logits,item)
            _,metrics=sign_arrays_and_metrics(logits,case['labels'],case['boundary'])
            path=directory/(case['object_id']+'.npz')
            np.savez_compressed(path,grid_logits=logits,fp_indices=pool['fp'],fn_indices=pool['fn'],
                shape_latent=latent[0].cpu().numpy(),train_near_logits=near_logits,train_near_error_indices=errors)
            records.append(dict(object_id=case['object_id'],file=str(path.relative_to(output)),file_sha256=digest(path),
                fp_pool_count=len(pool['fp']),fn_pool_count=len(pool['fn']),train_near_error_count=len(errors),
                grid_metrics=metrics,encode_calls=1,decode_calls=1,grid_query_geometry_calls=len(values),train_near_query_geometry_calls=4))
            pools.append(pool);near.append(errors)
    exact=torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state(device))
    if not exact or model.training!=mode:raise RuntimeError('Refresh changed training RNG or mode')
    report=dict(step=step,cases=records,cpu_rng_exact=True,cuda_rng_exact=True,model_mode_restored=True,
        mode_before=mode,mode_after=model.training,grid_nodes=len(queries),model_updates=0,backward_calls=0)
    original.write_json(directory/'RESULT.json',report)
    print('PERSISTENT_REPLAY_REFRESH',step,[(r['object_id'],r['fp_pool_count'],r['fn_pool_count'],r['train_near_error_count']) for r in records],flush=True)
    return pools,near,report


def run(args):
    from .retained_execution import require_active_resource
    resource=require_active_resource()
    import torch
    manifest,queries,cases,pools,histories,source=sampling.read_data(args.data)
    bound=bindings(args.data,manifest)
    qualified,near=check_qualification(args.qualification,bound,resource,cases)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope=SCOPE,sampling_revision=REVISION,bindings=bound,resource=resource,recipe=sampling.recipe(),
        model_updates=832,source_model_updates=2000,total_model_updates=2832,checkpoint_reset=False,
        updates_per_object=208,source_updates_per_object=500,total_updates_per_object=708,
        initial_checkpoint_sha256=CHECKPOINT_SHA,source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],
        loss='original michelangelo.models.tsal.loss.KLNearFar',near_weight=.1,kl_weight=.001,
        optimizer=dict(name='AdamW',lr=sampling.LR,betas=[.9,.99],eps=1e-6,weight_decay=.01),
        train_seed=original.TRAIN_SEED,batch_size=1,batch_queries=dict(volume=1024,near=1024),input_points=4096,
        train_posterior='mode',refresh_posterior='mode',eval_posterior='mode',full_model_trainable=True,
        data_manifest=manifest,data_manifest_sha256=digest(args.data/'MANIFEST.json'),
        qualification_result=str((args.qualification/'RESULT.json').resolve()),qualification_sha256=digest(args.qualification/'RESULT.json'),
        previous_result=str((sampling.SOURCE/'RESULT.json').resolve()),previous_result_sha256=manifest['sources']['previous_result_sha256'],
        previous_cases=manifest['previous_cases'],initial_pool_source=manifest['sources']['pool_sources'],
        gates_unchanged=dict(cd_x9000_max=.45,fscore_min=.95,area_ratio=[.7,1.5],max_vertex_and_face_probe=.01,occupancy_iou_min=.90),
        sources=bound['original_sources'],adapter_sha256=bound['trainer_sha256'],formal_call_budget=FORMAL_BUDGET,
        runtime=runtime(),initial_mesh_output='bound actual lowstep2000; no new initial extraction',
        visual_inspection_required=True,physics_controls=0,tactile_model_forwards=0,
        task_adaptation='native AE query sampler repair; no SUGAR teacher/student learning claim')
    original.write_json(args.output/'PROTOCOL.json',protocol)
    saved_source=load_source_checkpoint(manifest);device=torch.device('cuda')
    model,cls=new_model(source,saved_source,device);optimizer=original.optimizer_for(model);criterion=original.original_loss()
    restored,source_hash=restore_source(model,optimizer,saved_source,device)
    original.write_json(args.output/'INITIAL_RESTORE.json',restored)
    counter=MethodCounter(FORMAL_BUDGET);counter.install(model)
    data=[case['original'] for case in cases]
    initial=original.evaluate_queries(model,criterion,data,device)
    if initial!=json.loads((sampling.SOURCE/'RELOAD.json').read_text())['cases']:
        raise RuntimeError('Actual formal source2000 heldout/latent replay not EXACT; 0 updates')
    original.write_json(args.output/'INITIAL_STATE.json',dict(model_updates=2000,new_model_updates=0,
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],original_heldout_and_latent_exact=True,
        checkpoint_selection=False,cases=manifest['previous_cases']))
    reports=[dict(step=2000,cases=initial)];original.write_json(args.output/'CURVE.json',reports)
    sampler=sampling.ReplaySampler(cases,pools,histories,near);traces=[];refreshes=[]
    try:
        with (args.output/'updates.jsonl').open('x') as log:
            for step in range(2001,2833):
                if step-1 in sampling.REFRESH_STEPS:
                    pools,near,record=refresh(model,queries,cases,device,args.output/'grid_refresh',step-1)
                    sampler.refresh(step-1,pools,near);refreshes.append(record)
                    original.write_json(args.output/'REFRESHES.json',refreshes)
                index,item,indices,volume,chosen_near,info=sampler.batch(step,queries)
                torch.manual_seed(original.TRAIN_SEED+step);model.train();optimizer.zero_grad(set_to_none=True)
                loss,parts=batch_loss(model,criterion,item,indices,device)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite original loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                if step in (2001,2832):original.write_json(args.output/f'GRADIENT_STEP_{step}.json',original.gradients(model))
                optimizer.step()
                if any(float(v['step'])!=step for v in optimizer.state.values()):raise RuntimeError('Full Adam clock differs')
                traces.append((step,index,sampler.refresh_step,volume.copy(),chosen_near.copy()))
                log.write(json.dumps(dict(step=step,object_id=OBJECT_IDS[index],seed=original.TRAIN_SEED+step,
                    loss=float(loss),gradient_norm=float(norm),loss_components={k:float(v) for k,v in parts.items()},
                    learning_rate=sampling.LR,sampling=info,volume_indices_sha256=prior.array_sha(volume),
                    near_indices_sha256=prior.array_sha(chosen_near)))+'\n');log.flush()
                if step%100==0:
                    reports.append(dict(step=step,cases=original.evaluate_queries(model,criterion,data,device)))
                    original.write_json(args.output/'CURVE.json',reports)
                    print('PERSISTENT_REPLAY_UPDATE',step,reports[-1],flush=True)
    finally:
        np.savez_compressed(args.output/'SAMPLING_TRACE.npz',steps=np.array([r[0] for r in traces],np.int32),
            case_indices=np.array([r[1] for r in traces],np.int32),refresh_steps=np.array([r[2] for r in traces],np.int32),
            volume_grid_indices=np.stack([r[3] for r in traces]) if traces else np.empty((0,1024),np.int32),
            near_indices=np.stack([r[4] for r in traces]) if traces else np.empty((0,1024),np.int32))
        original.write_json(args.output/'SAMPLING_COVERAGE.json',sampler.coverage())
        original.write_json(args.output/'CALL_COUNTS.json',counter.report())
    if len(traces)!=832 or not sampler.coverage()['source_union_coverage_complete']:
        raise RuntimeError('Fixed832 updates/source-union coverage incomplete; no automatic extension')
    if state.snapshot_digest(saved_source)!=source_hash:raise RuntimeError('Source cached model/Adam mutated')
    original.write_json(args.output/'SOURCE_CACHE_READBACK.json',dict(passed=True,initial_hash=source_hash,
        final_hash=state.snapshot_digest(saved_source),all_original_adam_steps=2000))
    del saved_source;gc.collect()
    checkpoint=args.output/'endpoint.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=2832,
        parameter_names=[n for n,_ in model.named_parameters()],initial_checkpoint_sha256=CHECKPOINT_SHA,
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],protocol=protocol,
        rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),checkpoint)
    audit,saved=audit_endpoint(checkpoint,model,optimizer)
    original.write_json(args.output/'FULL_ADAM_READBACK.json',audit)
    final_pools,final_near,endpoint=refresh(model,queries,cases,device,args.output/'endpoint_grid',2832)
    original.write_json(args.output/'ENDPOINT_GRID.json',endpoint)
    # Endpoint observations are diagnostic only; they cannot cause another update.
    for i,h in enumerate(sampler.history):
        h['fp'].append(final_pools[i]['fp']);h['fn'].append(final_pools[i]['fn']);h['near'].append(final_near[i])
    coverage=sampler.coverage();original.write_json(args.output/'SAMPLING_COVERAGE.json',coverage)
    before=original.evaluate_queries(model,criterion,data,device)
    reports.append(dict(step=2832,cases=before));original.write_json(args.output/'CURVE.json',reports)
    counter.detach();config=source['loading']['official_config']['params']
    del model,optimizer,loss,parts;gc.collect();torch.cuda.empty_cache()
    model=cls(device=None,dtype=None,**config);model.load_state_dict(saved['model'],strict=True)
    model=model.to(device).requires_grad_(False).eval();del saved;gc.collect();counter.install(model)
    after=original.evaluate_queries(model,criterion,data,device)
    if before!=after:raise RuntimeError('Full2832 checkpoint reload not EXACT')
    original.write_json(args.output/'RELOAD.json',dict(passed=True,all_four_query_metrics_exact=True,
        all_four_logits_and_latents_exact=True,cases=after))
    records=original.endpoint_meshes(model,data,manifest['original_manifest'],args.output,device)
    if all(r['complete'] for r in records):counter.require_complete()
    actual_calls=counter.report();counter.detach();original.write_json(args.output/'CALL_COUNTS.json',actual_calls)
    result=dict(complete=True,scope=SCOPE,sampling_revision=REVISION,cases=records,model_updates=832,
        source_model_updates=2000,total_model_updates=2832,checkpoint_reset=False,updates_per_object=208,
        source_updates_per_object=500,total_updates_per_object=708,initial_checkpoint_sha256=CHECKPOINT_SHA,
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],checkpoint_file=checkpoint.name,
        checkpoint_sha256=digest(checkpoint),numerical_passed=all(r['checks']['passed'] for r in records),
        representation_qualified=False,visual_inspection='pending',full_adam_audit=audit,
        source_full_gradient_qualification_passed=source['source_full_gradient_qualification_passed'],
        continuation_gradient_check_steps=[2001,2832],actual_checkpoint_reload_passed=True,loading=source['loading'],
        endpoint_grid=endpoint,initial_pool_step=2000,formal_refresh_steps=[r['step'] for r in refreshes],
        actual_calls=actual_calls,sampling_coverage=coverage,actual_optimizer_updates=832,
        physics_controls=0,tactile_model_forwards=0,clip_alignment_continued=False)
    original.write_json(args.output/'RESULT.json',result)
    return 0 if result['numerical_passed'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','qualify','run'))
    parser.add_argument('--data',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--qualification',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':
        if args.output is None:parser.error('prepare requires new --output')
        print(json.dumps(sampling.prepare(args.output)));return 0
    if args.data is None:parser.error('check/qualify/run require --data')
    if args.stage=='check':
        manifest,queries,cases,pools,histories,result=sampling.read_data(args.data)
        deps=dependency_check();criterion=original.original_loss()
        import torch
        saved=load_source_checkpoint(manifest);cls=state.official_class(result['loading']['official_config'])
        with torch.device('meta'):model=cls(device=None,dtype=None,**result['loading']['official_config']['params'])
        if [n for n,_ in model.named_parameters()]!=saved['parameter_names']:raise RuntimeError('Meta original architecture differs')
        if torch.cuda.is_initialized() or not all(deps.values()):raise RuntimeError('CPU dependency check failed')
        print(json.dumps(dict(passed=True,scope=SCOPE,recipe=sampling.recipe(),dependencies=deps,
            parameters=184659585,parameter_tensors=307,source_adam_states=307,every_source_adam_step=2000,
            grid_nodes=len(queries),historical_counts=manifest['source_history_counts'],train_near_logits_qualification_pending=True,
            original_class=type(model).__module__+'.'+type(model).__name__,original_loss=type(criterion).__module__+'.'+type(criterion).__name__,
            cuda_initialized=False,model_forwards=0,model_updates=0)));return 0
    if args.output is None:parser.error('qualify/run require new --output')
    if args.stage=='qualify':return qualify(args)
    if args.qualification is None:parser.error('run requires successful --qualification')
    return run(args)

if __name__=='__main__':raise SystemExit(main())
