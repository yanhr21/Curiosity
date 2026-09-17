"""Full official mode-AE2832->2848 on one frozen four-object TRAIN objective.

No model replacement or vendor mutation. The original complete model, decoder
checkpoint implementation, KLNearFar and source AdamW are retained. Every
optimizer step follows all four complete object gradients; no refresh, new
query selection, best endpoint, automatic extension or mesh postprocessing.
prepare/check are CPU only; root owns every actual retained GPU launch.
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

from . import michelangelo_static_global_data as sampling
from . import train_michelangelo_persistent_replay as previous
from . import train_michelangelo_overfit as original
from . import diagnose_michelangelo_adam as state
from .qualify_michelangelo_ae import CHECKPOINT_SHA, OBJECT_IDS, VENDOR, digest, dependency_check

SCOPE, REVISION = sampling.SCOPE, sampling.REVISION
QUALIFICATION_BUDGET = dict(forward=0, encode=9, decode=9, query_geometry=187)
FORMAL_BUDGET = dict(forward=0, encode=100, decode=96, query_geometry=3634)
QUALIFICATION_COMPUTE = dict(forward_compute=187, backward_recompute=150)
FORMAL_COMPUTE = dict(forward_compute=3634, backward_recompute=2400)
EVAL_UPDATES = (0, 4, 8, 12, 16)
CHUNK_EQUIVALENCE = dict(object_id='11898', train_near_points=4096,
                         monolithic_chunk=4096, split_chunk=1024, atol=1e-5, rtol=1e-5)
MethodCounter = previous.MethodCounter
runtime = previous.runtime
new_model = previous.new_model
load_source_checkpoint = FunctionType(previous.load_source_checkpoint.__code__,
    dict(previous.load_source_checkpoint.__globals__, sampling=sampling),
    previous.load_source_checkpoint.__name__, previous.load_source_checkpoint.__defaults__)
restore_source = FunctionType(previous.restore_source.__code__,
    dict(previous.restore_source.__globals__, sampling=sampling),
    previous.restore_source.__name__, previous.restore_source.__defaults__)
audit_endpoint = FunctionType(original.audit_saved_checkpoint.__code__,
    dict(original.audit_saved_checkpoint.__globals__, STEPS=sampling.END),
    original.audit_saved_checkpoint.__name__, original.audit_saved_checkpoint.__defaults__)


def bindings(directory, manifest):
    return dict(data_manifest_sha256=digest(Path(directory)/'MANIFEST.json'),
        data_adapter_sha256=digest(Path(sampling.__file__)), trainer_sha256=digest(Path(__file__)),
        state_helper_sha256=digest(Path(state.__file__)), previous_trainer_sha256=digest(Path(previous.__file__)),
        original_trainer_sha256=digest(Path(original.__file__)),
        renderer_sha256=digest(Path(__file__).with_name('render_michelangelo_static_global.py')),
        original_sources={str(p.relative_to(VENDOR)):digest(p) for p in VENDOR.rglob('*.py')},
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'])


class DecoderComputeCounter:
    """Count actual original decoder computation, including checkpoint backward.

The official geo_decoder checkpoint wraps its entire _forward. Its forward
body executes with autograd disabled; its backward recomputation enables it.
Only this binding is wrapped; the original function/architecture is unchanged.
"""
    def __init__(self, budget):
        self.budget=dict(budget); self.attempted=Counter(); self.completed=Counter()
        self.owner=None; self.method=None

    def install(self, model):
        import torch
        if self.owner is not None or model.geo_decoder.use_checkpoint is not True:
            raise RuntimeError('Require original full-decoder checkpoint path')
        self.owner=model.geo_decoder; self.method=self.owner._forward
        method=self.method
        def counted(*args, **kwargs):
            name='backward_recompute' if torch.is_grad_enabled() else 'forward_compute'
            self.attempted[name]+=1
            if self.attempted[name]>self.budget[name]:
                raise RuntimeError('Decoder computation budget exceeded: '+name)
            value=method(*args, **kwargs); self.completed[name]+=1
            return value
        self.owner._forward=counted

    def detach(self):
        if self.owner is not None: self.owner._forward=self.method
        self.owner=None; self.method=None

    def report(self):
        return dict(budget=self.budget, attempted={k:self.attempted[k] for k in self.budget},
                    completed={k:self.completed[k] for k in self.budget})

    def require_complete(self):
        if any(self.attempted[k]!=v or self.completed[k]!=v for k,v in self.budget.items()):
            raise RuntimeError('Actual decoder forward/backward budget incomplete')


def original_mixture_loss(criterion, posterior, far_logits, near_logits, far_labels, near_labels):
    """Three ORIGINAL criteria, shared near logits/posterior, one object loss.

No chunk means are averaged. Each original criterion sees its entire far
bucket and the entire shared near pool; .5+.25+.25=1 keeps KL and near weights
exactly once. Caller divides by4 and performs one backward per object.
"""
    import torch
    if len(far_logits)!=3 or len(far_labels)!=3:
        raise ValueError('Three fixed far buckets required')
    if criterion.near_weight!=.1 or criterion.kl_weight!=.001:
        raise ValueError('Original near/KL weights changed')
    count=near_logits.shape[1]
    if count!=near_labels.shape[1] or count<=0:
        raise ValueError('Malformed complete near pool')
    saved=criterion.num_near_samples; criterion.num_near_samples=count
    losses=[]; records=[]
    try:
        for weight, logits, labels in zip(sampling.WEIGHTS,far_logits,far_labels,strict=True):
            if logits.shape!=labels.shape or logits.shape[1]<=0:
                raise ValueError('Malformed complete far bucket')
            loss, log=criterion(posterior,torch.cat((logits,near_logits),dim=1),
                                torch.cat((labels,near_labels),dim=1),split='train')
            losses.append(weight*loss)
            records.append(dict(weight=weight, far_count=logits.shape[1], near_count=count,
                                components={k:float(v) for k,v in log.items()}))
    finally: criterion.num_near_samples=saved
    return sum(losses), records


def query_chunks(model, queries, decoded, device, chunk=sampling.CHUNK):
    import torch
    if chunk<=0 or len(queries)<=0: raise ValueError('Nonempty positive-size query chunks required')
    return torch.cat([model.query_geometry(original.tensor(queries[s:s+chunk],device),decoded)
                      for s in range(0,len(queries),chunk)],dim=1)


def object_loss(model, criterion, grid, case, buckets, device):
    item=case['original']
    # Explicit native observation allowlist. No GT label/query/identity input to encode.
    _, latent, posterior=model.encode(original.tensor(item['input_surface_xyz_vae'],device),
        original.tensor(item['input_normals'],device),sample_posterior=False)
    decoded=model.decode(latent)
    logits=[query_chunks(model,grid[buckets[k]],decoded,device) for k in sampling.BUCKETS]
    labels=[original.tensor(case['labels'][buckets[k]],device) for k in sampling.BUCKETS]
    near=item['train_queries_vae'][buckets['near_indices']]
    near_logits=query_chunks(model,near,decoded,device)
    near_labels=original.tensor(item['train_labels'][buckets['near_indices']],device)
    loss, records=original_mixture_loss(criterion,posterior,logits,near_logits,labels,near_labels)
    return loss, dict(object_id=case['object_id'],object_weight=.25,loss=float(loss),buckets=records)


def global_objective(model, criterion, grid, cases, buckets, device, *, backward):
    import torch
    if tuple(c['object_id'] for c in cases)!=OBJECT_IDS or len(buckets)!=4:
        raise ValueError('All fixed four objects required')
    records=[]; total=0.
    for case, fixed in zip(cases,buckets,strict=True):
        loss, record=object_loss(model,criterion,grid,case,fixed,device)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite full fixed objective')
        total+=float(loss)/4
        if backward: (loss/4).backward()
        records.append(record)
        del loss
    return dict(loss=total,cases=records,object_backward_calls=4 if backward else 0,
                query_points=590052,query_geometry_calls=150,optimizer_updates=0)


def chunk_equivalence(model, case, device):
    import torch
    item=case['original']; query=item['train_queries_vae'][16384:20480]
    if case['object_id']!='11898' or len(query)!=4096:raise ValueError('Fixed actual TRAIN subset changed')
    model.eval()
    with torch.no_grad():
        _, latent, _=model.encode(original.tensor(item['input_surface_xyz_vae'],device),
            original.tensor(item['input_normals'],device),sample_posterior=False)
        decoded=model.decode(latent)
        full=query_chunks(model,query,decoded,device,4096)
        chunks=query_chunks(model,query,decoded,device,1024)
        error=(full-chunks).abs()
        passed=bool(torch.allclose(full,chunks,atol=1e-5,rtol=1e-5)
                    and torch.equal(full>=0,chunks>=0))
        result=dict(CHUNK_EQUIVALENCE,passed=passed,max_abs_difference=float(error.max()),
                    signs_exact=torch.equal(full>=0,chunks>=0),query_sha256=sampling.array_sha(query))
    if not passed:raise RuntimeError('Actual original query chunk equivalence failed: '+str(result))
    return result


def qualify(args):
    from .retained_execution import require_active_resource
    import torch
    resource=require_active_resource()
    manifest,grid,cases,buckets,source=sampling.read_data(args.data)
    args.output.mkdir(parents=True,exist_ok=False); bound=bindings(args.data,manifest)
    protocol=dict(scope=SCOPE+'_qualification',sampling_revision=REVISION,bindings=bound,resource=resource,
        recipe=sampling.recipe(),budget=QUALIFICATION_BUDGET,decoder_compute=QUALIFICATION_COMPUTE,
        chunk_equivalence=CHUNK_EQUIVALENCE,runtime=runtime(),optimizer_updates=0,
        object_backward_calls=4,global_gradient_qualifications=1)
    original.write_json(args.output/'PROTOCOL.json',protocol)
    calls=MethodCounter(QUALIFICATION_BUDGET); compute=DecoderComputeCounter(QUALIFICATION_COMPUTE)
    result=dict(scope=SCOPE+'_qualification',sampling_revision=REVISION,bindings=bound,resource=resource,
                recipe=sampling.recipe(),passed=False,optimizer_updates=0,physics_controls=0)
    try:
        saved=load_source_checkpoint(manifest); device=torch.device('cuda')
        model,_=new_model(source,saved,device); optimizer=original.optimizer_for(model)
        criterion=original.original_loss(); restored,cache=restore_source(model,optimizer,saved,device)
        calls.install(model); compute.install(model)
        baseline=original.evaluate_queries(model,criterion,[c['original'] for c in cases],device)
        if baseline!=json.loads((sampling.SOURCE/'RELOAD.json').read_text())['cases']:
            raise RuntimeError('All four source2832 heldout/latent outputs must replay EXACT')
        equivalence=chunk_equivalence(model,cases[1],device)
        torch.cuda.reset_peak_memory_stats(device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        objective=global_objective(model,criterion,grid,cases,buckets,device,backward=True)
        gradient=original.gradients(model)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        optimizer.zero_grad(set_to_none=True)
        preserved=state.match_state(model,optimizer,saved,expected_step=sampling.START)
        if state.snapshot_digest(saved)!=cache:raise RuntimeError('Qualification mutated source cache')
        calls.require_complete(); compute.require_complete()
        result.update(passed=True,initial_restore=restored,source_replay_exact=True,source_cases=baseline,
            chunk_equivalence=equivalence,global_objective=objective,global_gradient=gradient,
            gradient_norm=float(norm),memory=memory,object_backward_calls=4,
            final_model_and_adam_exact=preserved,source_cached_digest=cache)
    except Exception:result['error']=traceback.format_exc()
    finally:
        calls.detach();compute.detach()
        result.update(actual_calls=calls.report(),decoder_compute=compute.report())
        original.write_json(args.output/'RESULT.json',result)
    return 0 if result['passed'] else 1


def check_qualification(path, bound, resource):
    result=json.loads((Path(path)/'RESULT.json').read_text())
    if (result['scope']!=SCOPE+'_qualification' or result['sampling_revision']!=REVISION
            or not result['passed'] or result['bindings']!=bound or result['resource']!=resource
            or result['recipe']!=sampling.recipe() or result['optimizer_updates']!=0
            or result['object_backward_calls']!=4 or not result['source_replay_exact']
            or not result['global_gradient']['passed'] or not result['chunk_equivalence']['passed']
            or not result['final_model_and_adam_exact']['passed']):
        raise ValueError('Require actual complete same-source full-global-gradient qualification')
    for field,budget in [('actual_calls',QUALIFICATION_BUDGET),('decoder_compute',QUALIFICATION_COMPUTE)]:
        if result[field]['budget']!=budget or result[field]['attempted']!=budget or result[field]['completed']!=budget:
            raise ValueError('Actual qualification computation budget changed')
    return result


def run(args):
    from .retained_execution import require_active_resource
    import torch
    resource=require_active_resource()
    manifest,grid,cases,buckets,source=sampling.read_data(args.data)
    bound=bindings(args.data,manifest);qualified=check_qualification(args.qualification,bound,resource)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope=SCOPE,sampling_revision=REVISION,bindings=bound,resource=resource,
        recipe=sampling.recipe(),model_updates=16,source_model_updates=2832,total_model_updates=2848,
        checkpoint_reset=False,optimizer_update_kind='one global complete4-object objective',
        new_object_gradient_exposures_per_object=16,source_object_gradient_exposures_per_object=708,
        initial_checkpoint_sha256=CHECKPOINT_SHA,source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],
        loss='unmodified original michelangelo.models.tsal.loss.KLNearFar',near_weight=.1,kl_weight=.001,
        optimizer=dict(name='AdamW',lr=sampling.LR,betas=[.9,.99],eps=1e-6,weight_decay=.01),
        train_seed=original.TRAIN_SEED,train_posterior='mode',eval_posterior='mode',
        full_model_trainable=True,input_points=4096,objects_per_global_update=4,
        fixed_data_manifest=manifest,data_manifest_sha256=digest(args.data/'MANIFEST.json'),
        qualification_result=str((args.qualification/'RESULT.json').resolve()),
        qualification_sha256=digest(args.qualification/'RESULT.json'),
        previous_result=str((sampling.SOURCE/'RESULT.json').resolve()),
        previous_result_sha256=manifest['sources']['previous_result_sha256'],previous_cases=manifest['previous_cases'],
        gates_unchanged=dict(cd_x9000_max=.45,fscore_min=.95,area_ratio=[.7,1.5],
            max_vertex_and_face_probe=.01,occupancy_iou_min=.90),
        sources=bound['original_sources'],adapter_sha256=bound['trainer_sha256'],
        formal_call_budget=FORMAL_BUDGET,formal_decoder_compute=FORMAL_COMPUTE,
        budget_breakdown=dict(training=dict(encode=64,decode=64,query_geometry=2400,
            decoder_backward_recompute=2400,object_backward_calls=64,optimizer_updates=16),
            heldout_fixed0_4_8_12_16=dict(encode=20,decode=20,query_geometry=160),
            final_fixed_training_objective=dict(encode=4,decode=4,query_geometry=150),
            endpoint_reload=dict(encode=4,decode=4,query_geometry=32),
            original_endpoint_meshes=dict(encode=8,decode=4,query_geometry=892),
            qualification=dict(official_calls=QUALIFICATION_BUDGET,decoder_compute=QUALIFICATION_COMPUTE,optimizer_updates=0),
            renderer=dict(frames=384,stills=12,model_forwards=0,physics_controls=0)),
        runtime=runtime(),initial_mesh_output='bound actual source2832 raw/corrected meshes, no re-extraction',
        visual_inspection_required=True,physics_controls=0,tactile_model_forwards=0,
        task_adaptation='fixed native TRAIN objective and full4-object gradient accumulation; no SUGAR/student/generalization success claim')
    original.write_json(args.output/'PROTOCOL.json',protocol)
    saved=load_source_checkpoint(manifest);device=torch.device('cuda')
    model,cls=new_model(source,saved,device);optimizer=original.optimizer_for(model);criterion=original.original_loss()
    restored,source_hash=restore_source(model,optimizer,saved,device)
    original.write_json(args.output/'INITIAL_RESTORE.json',restored)
    calls=MethodCounter(FORMAL_BUDGET);compute=DecoderComputeCounter(FORMAL_COMPUTE)
    calls.install(model);compute.install(model)
    data=[c['original'] for c in cases]; completed=0;object_backwards=0
    try:
        initial=original.evaluate_queries(model,criterion,data,device)
        if initial!=json.loads((sampling.SOURCE/'RELOAD.json').read_text())['cases']:
            raise RuntimeError('Formal source2832 heldout/latent replay must be EXACT before updates')
        original.write_json(args.output/'INITIAL_STATE.json',dict(model_updates=2832,new_model_updates=0,
            source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],
            original_heldout_and_latent_exact=True,checkpoint_selection=False,cases=manifest['previous_cases']))
        curve=[dict(step=2832,cases=initial)];original.write_json(args.output/'CURVE.json',curve)
        with (args.output/'updates.jsonl').open('x') as log:
            for step in range(2833,2849):
                if require_active_resource()!=resource:raise RuntimeError('Retained resource changed')
                torch.manual_seed(original.TRAIN_SEED+step);model.train();optimizer.zero_grad(set_to_none=True)
                objective=global_objective(model,criterion,grid,cases,buckets,device,backward=True)
                object_backwards+=objective['object_backward_calls']
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                if step in (2833,2848):
                    original.write_json(args.output/f'GRADIENT_STEP_{step}.json',original.gradients(model))
                optimizer.step();completed+=1
                if any(float(v['step'])!=step for v in optimizer.state.values()):
                    raise RuntimeError('Full307 Adam global clock differs')
                row=dict(step=step,new_global_updates=completed,learning_rate=sampling.LR,
                    seed=original.TRAIN_SEED+step,objective=objective,gradient_norm=float(norm),
                    complete_object_order=list(OBJECT_IDS),optimizer_updates=1,
                    source_data_manifest_sha256=bound['data_manifest_sha256'])
                log.write(json.dumps(row)+'\n');log.flush()
                print('STATIC_GLOBAL_UPDATE',step,objective['loss'],flush=True)
                if completed in EVAL_UPDATES:
                    curve.append(dict(step=step,cases=original.evaluate_queries(model,criterion,data,device)))
                    original.write_json(args.output/'CURVE.json',curve)
        if completed!=16 or object_backwards!=64:raise RuntimeError('Incomplete fixed16 global updates')
        if state.snapshot_digest(saved)!=source_hash:raise RuntimeError('Source cached model/Adam mutated')
        original.write_json(args.output/'SOURCE_CACHE_READBACK.json',dict(passed=True,initial_hash=source_hash,
            final_hash=state.snapshot_digest(saved),all_original_adam_steps=2832))
        del saved;gc.collect()
        checkpoint=args.output/'endpoint.pt'
        # Preserve the true endpoint before any optional metrics or mesh work can fail.
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=2848,
            parameter_names=[n for n,_ in model.named_parameters()],initial_checkpoint_sha256=CHECKPOINT_SHA,
            source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],protocol=protocol,
            rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),checkpoint)
        audit,snapshot=audit_endpoint(checkpoint,model,optimizer)
        original.write_json(args.output/'FULL_ADAM_READBACK.json',audit)
        model.eval()
        with torch.no_grad():final_objective=global_objective(model,criterion,grid,cases,buckets,device,backward=False)
        original.write_json(args.output/'FINAL_FIXED_OBJECTIVE.json',final_objective)
        before=curve[-1]['cases'];calls.detach();compute.detach()
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
        model=cls(device=None,dtype=None,**source['loading']['official_config']['params'])
        model.load_state_dict(snapshot['model'],strict=True);model=model.to(device).requires_grad_(False).eval()
        del snapshot;gc.collect();calls.install(model);compute.install(model)
        after=original.evaluate_queries(model,criterion,data,device)
        if before!=after:raise RuntimeError('Actual full2848 checkpoint reload changed endpoint output')
        original.write_json(args.output/'RELOAD.json',dict(passed=True,all_four_query_metrics_exact=True,
            all_four_logits_and_latents_exact=True,cases=after))
        records=original.endpoint_meshes(model,data,manifest['original_manifest'],args.output,device)
        if all(r['complete'] for r in records):calls.require_complete();compute.require_complete()
        result=dict(complete=True,scope=SCOPE,sampling_revision=REVISION,cases=records,
            model_updates=16,source_model_updates=2832,total_model_updates=2848,checkpoint_reset=False,
            actual_optimizer_updates=16,actual_object_backward_calls=64,
            optimizer_update_kind='global4-object',new_object_gradient_exposures_per_object=16,
            initial_checkpoint_sha256=CHECKPOINT_SHA,source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],
            checkpoint_file=checkpoint.name,checkpoint_sha256=digest(checkpoint),
            numerical_passed=all(r['checks']['passed'] for r in records),representation_qualified=False,
            visual_inspection='pending',full_adam_audit=audit,source_full_gradient_qualification_passed=qualified['passed'],
            actual_checkpoint_reload_passed=True,loading=source['loading'],
            actual_calls=calls.report(),decoder_compute=compute.report(),final_fixed_objective=final_objective,
            physics_controls=0,tactile_model_forwards=0,clip_alignment_continued=False)
        original.write_json(args.output/'RESULT.json',result)
        return 0 if result['numerical_passed'] else 2
    finally:
        calls.detach();compute.detach()
        original.write_json(args.output/'CALL_COUNTS.json',dict(official_calls=calls.report(),
            decoder_compute=compute.report(),actual_optimizer_updates=completed,actual_object_backward_calls=object_backwards))


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
        manifest,grid,cases,buckets,source=sampling.read_data(args.data)
        dependencies=dependency_check();criterion=original.original_loss()
        import torch
        saved=load_source_checkpoint(manifest);cls=state.official_class(source['loading']['official_config'])
        with torch.device('meta'):model=cls(device=None,dtype=None,**source['loading']['official_config']['params'])
        if [n for n,_ in model.named_parameters()]!=saved['parameter_names']:
            raise RuntimeError('Meta official architecture parameter ordering changed')
        if torch.cuda.is_initialized() or not all(dependencies.values()):raise RuntimeError('CPU dependencies/CUDA boundary failed')
        print(json.dumps(dict(passed=True,scope=SCOPE,recipe=sampling.recipe(),dependencies=dependencies,
            parameter_tensors=307,parameters=184659585,source_adam_states=307,every_source_adam_step=2832,
            case_query_counts=[sampling.counts(b) for b in buckets],
            original_class=type(model).__module__+'.'+type(model).__name__,
            original_loss=type(criterion).__module__+'.'+type(criterion).__name__,
            qualification_budget=QUALIFICATION_BUDGET,formal_budget=FORMAL_BUDGET,
            cuda_initialized=False,model_forwards=0,optimizer_updates=0)));return 0
    if args.output is None:parser.error('qualify/run require new --output')
    if args.stage=='run' and args.qualification is None:parser.error('run requires successful --qualification')
    try:return qualify(args) if args.stage=='qualify' else run(args)
    except Exception:
        if args.output.is_dir():
            original.write_json(args.output/'FAILURE.json',dict(scope=SCOPE,stage=args.stage,
                execution_complete=False,error=traceback.format_exc()))
        raise


if __name__=='__main__':raise SystemExit(main())
