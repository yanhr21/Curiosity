"""Fixed1000 full-official-AE comparison: train with posterior mode.

The sole neural change from the frozen dynamic-grid run is the official public
forward flag sample_posterior=False. Full architecture, original KLNearFar,
AdamW, 1000 updates, seeds, dynamic sampling rules, old GT-near and evaluation
remain. This is a project deterministic native-AE task adaptation, not the
original stochastic VAE training recipe. Dynamic error pools can differ.

prepare/check are CPU only; root separately queues qualify (four full backwards,
zero optimizer updates) and run (fresh release and empty Adam). No extra budget,
checkpoint selection, mesh filtering, model replacement or automatic next run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from . import michelangelo_mode_grid_data as sampling
from . import train_michelangelo_overfit as original
from .audit_michelangelo_extraction_signs import sign_arrays_and_metrics
from .qualify_michelangelo_ae import (
    ABC_TO_VAE,CHECKPOINT,CHECKPOINT_BYTES,CHECKPOINT_SHA,OBJECT_IDS,VENDOR,digest,
    load_official_model,dependency_check,
)

SCOPE=sampling.SCOPE
REVISION=sampling.REVISION
GRID_CHUNK=10000


def batch_loss(model,criterion,item,indices,device,capture=None):
    """Original full-forward/loss glue; only the public posterior flag is False.

    Qualification may retain posterior gradients. This does not alter the
    forward values or loss; formal training has no retained diagnostic graph.
    """
    _,logits,posterior=model(original.tensor(item['input_surface_xyz_vae'],device),
                            original.tensor(item['input_normals'],device),
                            original.tensor(item['train_queries_vae'][indices],device),
                            sample_posterior=False)
    if capture is not None:
        posterior.mean.retain_grad();posterior.logvar.retain_grad()
        capture.append(posterior)
    return criterion(posterior,logits,original.tensor(item['train_labels'][indices],device))


def posterior_gradient_check(posterior,kl_weight):
    """Mode reconstruction has no logvar path; original KL must still train it."""
    import torch
    mean_grad=posterior.mean.grad;logvar_grad=posterior.logvar.grad
    if (mean_grad is None or logvar_grad is None
            or not torch.isfinite(mean_grad).all() or not torch.isfinite(logvar_grad).all()
            or not mean_grad.count_nonzero() or not logvar_grad.count_nonzero()):
        raise RuntimeError('Missing, nonfinite or zero mean/logvar qualification gradient')
    expected=.5*kl_weight*(posterior.var.detach()-1)/posterior.logvar.numel()
    if not torch.allclose(logvar_grad,expected,rtol=1e-5,atol=1e-12):
        raise RuntimeError('Mode posterior logvar gradient differs from unchanged mean-reduced KL')
    return dict(passed=True,mean_gradient_norm=float(mean_grad.double().norm()),
                logvar_gradient_norm=float(logvar_grad.double().norm()),
                logvar_expected_kl_gradient_max_error=float((logvar_grad-expected).abs().max()),
                latent_coordinates=posterior.mean.numel(),sample_posterior=False,
                interpretation='mean receives reconstruction and KL; logvar receives original KL only')


def bindings(directory,manifest):
    return dict(data_manifest_sha256=digest(directory/'MANIFEST.json'),
                data_sources=manifest['sources'],data_adapter_sha256=manifest['data_adapter_sha256'],
                trainer_sha256=digest(Path(__file__)),initial_checkpoint_sha256=CHECKPOINT_SHA,
                original_sources={str(p.relative_to(VENDOR)):digest(p) for p in VENDOR.rglob('*.py')},
                sign_metric_helper_sha256=digest(Path(__file__).with_name('audit_michelangelo_extraction_signs.py')))


def released_model(checkpoint,device):
    if checkpoint.stat().st_size!=CHECKPOINT_BYTES or digest(checkpoint)!=CHECKPOINT_SHA:
        raise ValueError('Require the exact official released checkpoint, never a previous trained endpoint')
    return load_official_model(checkpoint,device)


@contextmanager
def refresh_context(model,device):
    """Restore both model mode and CPU/CUDA RNG; no refresh gradient graph."""
    import torch
    training=model.training
    devices=[] if device.type=='cpu' else [torch.cuda.current_device() if device.index is None else device.index]
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            with torch.no_grad():yield
        finally:
            model.train(training)


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def refresh(model,queries,cases,device,output,step):
    """Complete, original-method mean-posterior grid forwards; GT only afterward."""
    import torch
    directory=output/f'step_{step:04d}';directory.mkdir(parents=True,exist_ok=False)
    before_cpu=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state(device).clone()
    before_mode=model.training;pools=[];records=[]
    with refresh_context(model,device):
        for item in cases:
            observed=item['original']
            _,latent,_=model.encode(original.tensor(observed['input_surface_xyz_vae'],device),
                                    original.tensor(observed['input_normals'],device),sample_posterior=False)
            decoded=model.decode(latent)
            logits=[];calls=0
            for start in range(0,len(queries),GRID_CHUNK):
                # The same float32 endpoint-inclusive prepared grid and chunk size
                # were qualified against original extract_geometry. No GT label
                # participates in any of these official neural method arguments.
                values=model.query_geometry(original.tensor(queries[start:start+GRID_CHUNK],device),decoded)
                logits.append(values[0].cpu().numpy().copy());calls+=1
            logits=np.concatenate(logits)
            pool=sampling.error_pools(logits,item)
            _,metrics=sign_arrays_and_metrics(logits,item['labels'],item['boundary'])
            path=directory/(item['object_id']+'.npz')
            np.savez_compressed(path,grid_logits=logits,fp_indices=pool['fp'],fn_indices=pool['fn'],
                                shape_latent=latent[0].cpu().numpy())
            records.append(dict(object_id=item['object_id'],file=str(path.relative_to(output)),
                                file_sha256=digest(path),fp_pool_count=len(pool['fp']),fn_pool_count=len(pool['fn']),
                                eligible_nodes=len(item['eligible_indices']),grid_metrics=metrics,
                                encode_calls=1,decode_calls=1,grid_query_geometry_calls=calls))
            pools.append(pool)
    cpu_exact=torch.equal(before_cpu,torch.get_rng_state())
    cuda_exact=torch.equal(before_cuda,torch.cuda.get_rng_state(device))
    mode_exact=model.training==before_mode
    if not (cpu_exact and cuda_exact and mode_exact):
        raise RuntimeError('Full-grid refresh changed formal training RNG or model mode')
    report=dict(step=step,cases=records,cpu_rng_exact=cpu_exact,cuda_rng_exact=cuda_exact,
                model_mode_restored=mode_exact,mode_before=before_mode,mode_after=model.training,
                grid_nodes=len(queries),model_updates=0,backward_calls=0)
    original.write_json(directory/'RESULT.json',report)
    print('MODE_GRID_REFRESH',step,[(r['object_id'],r['fp_pool_count'],r['fn_pool_count']) for r in records],flush=True)
    return pools,report


def qualify(args):
    from .retained_execution import require_active_resource
    resource=require_active_resource()
    import torch
    manifest,queries,cases=sampling.read_data(args.data)
    args.output.mkdir(parents=True,exist_ok=False)
    bound=bindings(args.data,manifest)
    original.write_json(args.output/'PROTOCOL.json',dict(scope=SCOPE+'_qualification',bindings=bound,
                        recipe=sampling.fixed_recipe(),resource=resource,optimizer_updates=0))
    torch.manual_seed(original.TRAIN_SEED);device=torch.device('cuda')
    model,loading=released_model(args.checkpoint,device);model.requires_grad_(True).train()
    criterion=original.original_loss()
    pools,refresh_report=refresh(model,queries,cases,device,args.output/'grid_refresh',0)
    records=[];backward_attempts=0;backward_completed=0
    for index,oid in enumerate(OBJECT_IDS):
        row=dict(object_id=oid,passed=False)
        try:
            step=index+1;torch.manual_seed(original.TRAIN_SEED+step)
            model.train();model.zero_grad(set_to_none=True)
            chosen,item,indices,volume,near,info=sampling.batch_for_step(step,queries,cases,pools)
            if chosen!=index:raise ValueError('Qualification case sequence changed')
            capture=[]
            loss,components=batch_loss(model,criterion,item,indices,device,capture=capture)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite qualification loss')
            backward_attempts+=1;loss.backward();backward_completed+=1
            grad=original.gradients(model)
            posterior_grad=posterior_gradient_check(capture[0],criterion.kl_weight)
            path=args.output/(oid+'_sampling.npz');np.savez_compressed(path,volume_grid_indices=volume,near_indices=near)
            row.update(passed=True,loss=float(loss),loss_components={k:float(v) for k,v in components.items()},
                       gradients=grad,posterior_gradients=posterior_grad,sampling=info,
                       sampling_file=path.name,sampling_sha256=digest(path))
        except Exception:row['error']=traceback.format_exc()
        records.append(row);original.write_json(args.output/'PROGRESS.json',records)
        print('MODE_GRID_FULL_GRADIENT',oid,row.get('loss'),row['passed'],flush=True)
    report=dict(scope=SCOPE+'_qualification',bindings=bound,recipe=sampling.fixed_recipe(),resource=resource,
                cases=records,loading=loading,passed=all(r['passed'] for r in records),
                optimizer_updates=0,attempted_full_model_backwards=backward_attempts,
                actual_full_model_backwards=backward_completed,
                initial_refresh=refresh_report,physics_controls=0)
    original.write_json(args.output/'RESULT.json',report)
    return 0 if report['passed'] else 2


def check_qualification(path,bound,resource):
    report=json.loads((path/'RESULT.json').read_text())
    if (report['scope']!=SCOPE+'_qualification' or not report['passed']
            or report['bindings']!=bound or report['recipe']!=sampling.fixed_recipe()
            or report['resource']!=resource or report['optimizer_updates']!=0
            or report['actual_full_model_backwards']!=4
            or tuple(x['object_id'] for x in report['cases'])!=OBJECT_IDS
            or not all(x['passed'] and x['gradients']['passed']
                       and x['posterior_gradients']['passed'] for x in report['cases'])):
        raise ValueError('Require successful same-source/resource four-case full-gradient qualification, zero updates')
    return report


def run(args):
    from .retained_execution import require_active_resource
    resource=require_active_resource()
    import torch
    manifest,queries,cases=sampling.read_data(args.data);bound=bindings(args.data,manifest)
    qualification=check_qualification(args.qualification,bound,resource)
    args.output.mkdir(parents=True,exist_ok=False)
    source=manifest['sources'];previous=Path(source['previous_run'])
    protocol=dict(scope=SCOPE,sampling_revision=REVISION,bindings=bound,resource=resource,
                  recipe=sampling.fixed_recipe(),model_updates=1000,updates_per_object=250,
                  initial_checkpoint_sha256=CHECKPOINT_SHA,checkpoint_reset='fresh official release after separate qualification',
                  qualification_result=str((args.qualification/'RESULT.json').resolve()),
                  qualification_sha256=digest(args.qualification/'RESULT.json'),gradient_qualification_updates=0,
                  loss='original michelangelo.models.tsal.loss.KLNearFar',near_weight=.1,kl_weight=.001,
                  optimizer=dict(name='AdamW',lr=1e-4,betas=[.9,.99],eps=1e-6,weight_decay=.01),
                  warmup=None,schedule='fixed lr; no sweep/early stopping',train_seed=original.TRAIN_SEED,
                  batch_size=1,batch_queries=dict(volume=1024,near=1024),input_points=4096,
                  train_posterior='mode',refresh_posterior='mode',eval_posterior='mode',abc_to_vae_scale=ABC_TO_VAE,
                  full_model_trainable=True,clip_alignment_continued=False,
                  gates_unchanged=dict(cd_x9000_max=.45,fscore_min=.95,area_ratio=[.7,1.5],
                                       max_vertex_and_face_probe=.01,occupancy_iou_min=.90),
                  data_manifest=manifest,data_manifest_sha256=digest(args.data/'MANIFEST.json'),
                  previous_result=str((previous/'RESULT.json').resolve()),previous_result_sha256=source['previous_result_sha256'],
                  previous_cases=manifest['previous_cases'],visual_inspection_required=True,
                  initial_mesh_output='initial_state',
                  initial_mesh_evaluation='original full raw/corrected mesh helper; unchanged gates; no selection',
                  sources=bound['original_sources'],adapter_sha256=bound['trainer_sha256'],
                  physics_controls=0,tactile_model_forwards=0,
                  task_adaptation='deterministic native AE, not original stochastic VAE training',
                  only_training_change='official forward sample_posterior=False',
                  dynamic_pool_note='same sampler rule, model-dependent pools and volume IDs may differ')
    original.write_json(args.output/'PROTOCOL.json',protocol)
    device=torch.device('cuda');torch.manual_seed(original.TRAIN_SEED)
    model,loading=released_model(args.checkpoint,device)
    if loading!=qualification['loading']:raise ValueError('Original released qualification/formal model differs')
    model.requires_grad_(True).train();optimizer=original.optimizer_for(model);criterion=original.original_loss()
    if optimizer.state:raise ValueError('Formal Adam was not reset')
    original_data=[case['original'] for case in cases]
    reports=[dict(step=0,cases=original.evaluate_queries(model,criterion,original_data,device))]
    original.write_json(args.output/'CURVE.json',reports)
    initial_output=args.output/'initial_state';initial_output.mkdir(exist_ok=False)
    initial_records=original.endpoint_meshes(model,original_data,manifest['original_manifest'],initial_output,device)
    original.write_json(args.output/'INITIAL_STATE.json',dict(model_updates=0,cases=initial_records,
                        initial_checkpoint_sha256=CHECKPOINT_SHA,checkpoint_selection=False))
    if not all(row['complete'] for row in initial_records):
        raise RuntimeError('Could not preserve all four actual initial raw meshes; no training started')
    traces=[];refreshes=[];current_refresh=None;pools=None
    try:
        with (args.output/'updates.jsonl').open('x') as log:
            for step in range(1,1001):
                if step-1 in sampling.REFRESH_STEPS:
                    current_refresh=step-1
                    pools,record=refresh(model,queries,cases,device,args.output/'grid_refresh',current_refresh)
                    refreshes.append(record);original.write_json(args.output/'REFRESHES.json',refreshes)
                index,item,indices,volume,near,sampling_info=sampling.batch_for_step(step,queries,cases,pools)
                torch.manual_seed(original.TRAIN_SEED+step);model.train();optimizer.zero_grad(set_to_none=True)
                loss,components=batch_loss(model,criterion,item,indices,device)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite original loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                if step in (1,1000):original.write_json(args.output/f'GRADIENT_STEP_{step}.json',original.gradients(model))
                optimizer.step()
                traces.append((step,index,current_refresh,volume.copy(),near.copy()))
                log.write(json.dumps(dict(step=step,object_id=OBJECT_IDS[index],seed=original.TRAIN_SEED+step,
                    loss=float(loss),gradient_norm=float(norm),loss_components={k:float(v) for k,v in components.items()},
                    refresh_step=current_refresh,sampling=sampling_info,volume_indices_sha256=array_sha(volume),
                    near_indices_sha256=array_sha(near)))+'\n');log.flush()
                if step%100==0:
                    row=dict(step=step,cases=original.evaluate_queries(model,criterion,original_data,device))
                    reports.append(row);original.write_json(args.output/'CURVE.json',reports)
                    print('MODE_GRID_UPDATE',step,row,flush=True)
    finally:
        np.savez_compressed(args.output/'SAMPLING_TRACE.npz',steps=np.array([x[0] for x in traces],np.int32),
            case_indices=np.array([x[1] for x in traces],np.int32),refresh_steps=np.array([x[2] for x in traces],np.int32),
            volume_grid_indices=np.stack([x[3] for x in traces]) if traces else np.empty((0,1024),np.int32),
            near_indices=np.stack([x[4] for x in traces]) if traces else np.empty((0,1024),np.int32))
    # Preserve the actual complete endpoint and Adam before any optional-to-
    # training endpoint diagnostic can fail. No optimizer action follows this.
    checkpoint=args.output/'endpoint.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=1000,
                    parameter_names=[n for n,_ in model.named_parameters()],initial_checkpoint_sha256=CHECKPOINT_SHA,
                    protocol=protocol,rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),checkpoint)
    audit,saved=original.audit_saved_checkpoint(checkpoint,model,optimizer)
    original.write_json(args.output/'FULL_ADAM_READBACK.json',audit)
    # Fixed endpoint diagnostic, never used to choose another update/checkpoint.
    _,endpoint_grid=refresh(model,queries,cases,device,args.output/'endpoint_grid',1000)
    original.write_json(args.output/'ENDPOINT_GRID.json',endpoint_grid)
    before=original.evaluate_queries(model,criterion,original_data,device)
    original_class=type(model);config=loading['official_config']['params']
    del model,optimizer,loss,components;gc.collect();torch.cuda.empty_cache()
    model=original_class(device=None,dtype=None,**config);model.load_state_dict(saved['model'],strict=True)
    model=model.to(device).requires_grad_(False).eval();del saved;gc.collect()
    after=original.evaluate_queries(model,criterion,original_data,device)
    if before!=after:raise RuntimeError('Full checkpoint reload changed exact evaluation')
    original.write_json(args.output/'RELOAD.json',dict(passed=True,all_four_query_metrics_exact=True,
                        all_four_logits_and_latents_exact=True,cases=after))
    records=original.endpoint_meshes(model,original_data,manifest['original_manifest'],args.output,device)
    result=dict(complete=True,scope=SCOPE,sampling_revision=REVISION,cases=records,model_updates=1000,updates_per_object=250,
                initial_checkpoint_sha256=CHECKPOINT_SHA,checkpoint_file=checkpoint.name,checkpoint_sha256=digest(checkpoint),
                numerical_passed=all(r['checks']['passed'] for r in records),representation_qualified=False,
                visual_inspection='pending',full_gradient_qualification_passed=True,full_adam_audit=audit,
                actual_checkpoint_reload_passed=True,loading=loading,endpoint_grid=endpoint_grid,
                formal_refresh_steps=[x['step'] for x in refreshes],physics_controls=0,tactile_model_forwards=0,
                clip_alignment_continued=False)
    original.write_json(args.output/'RESULT.json',result)
    return 0 if result['numerical_passed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','check','qualify','run'))
    p.add_argument('--data',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--qualification',type=Path);p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    args=p.parse_args()
    if args.stage=='prepare':
        if args.output is None:p.error('prepare requires --output')
        manifest=sampling.prepare(args.output);print(json.dumps(manifest));return 0
    if args.data is None:p.error('check/qualify/run require --data')
    if args.stage=='check':
        manifest,queries,cases=sampling.read_data(args.data)
        deps=dependency_check();criterion=original.original_loss()
        import torch
        if not all(deps.values()) or torch.cuda.is_initialized():raise RuntimeError('Dependency check failed or CUDA initialized')
        print(json.dumps(dict(passed=True,scope=SCOPE,recipe=sampling.fixed_recipe(),dependencies=deps,
                              cases=[x['object_id'] for x in cases],grid_nodes=len(queries),
                              original_loss_class=type(criterion).__module__+'.'+type(criterion).__name__,
                              cuda_initialized=False,model_forwards=0,model_updates=0)))
        return 0
    if args.output is None:p.error('qualify/run require a new --output directory')
    if args.stage=='qualify':return qualify(args)
    if args.qualification is None:p.error('run requires successful --qualification')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
