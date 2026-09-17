"""Bounded local optimization diagnostic of the complete official shape AE.

prepare/check are CPU only. run requires the retained allocation and shared
lock, and is launched by the root coordinator only. This is not continuation:
four full gradients and eight independent temporary Adam steps start from the
same mode1000 endpoint and complete optimizer. No deployable checkpoint, mesh,
new grid scan, physics, historical-cause claim or automatic next run is made.
All fixed-batch forwards use eval mode and the public posterior-mean flag.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from . import michelangelo_mode_grid_data as mode_data
from . import train_michelangelo_overfit as original
from .qualify_michelangelo_ae import OBJECT_IDS, VENDOR, CONFIG, digest, dependency_check

ROOT=original.EXPERIMENT/'overfit_repair_v1'
SOURCE=ROOT/'michelangelo_mode_grid_overfit_v1'
MODE_DATA=ROOT/'michelangelo_mode_grid_data_v1'
SCOPE='full_official_mode_ae_local_adam_diagnostic_only'
SEED_STEPS=(997,998,999,1000)
SCALES=(1.,.1)
BUDGET=dict(forward=44,encode=80,decode=80,query_geometry=332,backwards=4,
            temporary_optimizer_steps=8,grid_scans=0,mesh_extractions=0,physics_controls=0)


def recipe():
    return dict(scope=SCOPE,source_step=1000,trial_step=1001,objects=list(OBJECT_IDS),
        diagnostic_seed_steps=list(SEED_STEPS),scales=list(SCALES),model_mode='eval',
        posterior='mode',loss='original KLNearFar: far + .1 near + .001 KL',
        batches='endpoint1000 saved error pools; unchanged sampler and original near indices',
        historical_replay=False,baseline_full_forward_repeats=2,budget=BUDGET,
        reset='all parameters, all Adam states/groups and saved CPU/CUDA RNG before every trial',
        outputs='scalar/matrix diagnostics only; no deployable checkpoint',
        limitation='Local endpoint evidence cannot reconstruct historical100-to200 regression')


def load_sources():
    manifest,queries,cases=mode_data.read_data(MODE_DATA)
    result=json.loads((SOURCE/'RESULT.json').read_text())
    protocol=json.loads((SOURCE/'PROTOCOL.json').read_text())
    endpoint=json.loads((SOURCE/'ENDPOINT_GRID.json').read_text())
    if (result['scope']!=mode_data.SCOPE or not result['complete']
            or result['model_updates']!=1000 or protocol['train_posterior']!='mode'
            or endpoint['step']!=1000 or tuple(c['object_id'] for c in endpoint['cases'])!=OBJECT_IDS
            or tuple(c['object_id'] for c in result['cases'])!=OBJECT_IDS
            or digest(MODE_DATA/'MANIFEST.json')!=protocol['data_manifest_sha256']):
        raise ValueError('Require completed unchanged four-object mode1000 endpoint')
    for rel,sha in protocol['sources'].items():
        if digest(VENDOR/rel)!=sha:raise ValueError('Official source changed: '+rel)
    if digest(Path(__file__).with_name('train_michelangelo_mode_grid.py'))!=protocol['adapter_sha256']:
        raise ValueError('Executed mode trainer source changed')
    pools=[];pool_sources=[]
    for row in endpoint['cases']:
        path=SOURCE/'endpoint_grid'/row['file']
        if digest(path)!=row['file_sha256']:raise ValueError('Endpoint error-pool source changed')
        with np.load(path,allow_pickle=False) as z:
            pools.append(dict(fp=z['fp_indices'].copy(),fn=z['fn_indices'].copy()))
        pool_sources.append(dict(object_id=row['object_id'],file=str(path.resolve()),sha256=row['file_sha256']))
    fixed=[]
    for index,step in enumerate(SEED_STEPS):
        chosen,item,indices,volume,near,info=mode_data.batch_for_step(step,queries,cases,pools)
        if chosen!=index:raise ValueError('Wrong diagnostic object order')
        fixed.append(dict(object_id=OBJECT_IDS[index],seed_step=step,item=item,indices=indices,
                          volume=volume,near=near,sampling=info))
    sources=dict(source=str(SOURCE.resolve()),result_sha256=digest(SOURCE/'RESULT.json'),
        protocol_sha256=digest(SOURCE/'PROTOCOL.json'),reload_sha256=digest(SOURCE/'RELOAD.json'),
        endpoint_grid_sha256=digest(SOURCE/'ENDPOINT_GRID.json'),config_sha256=digest(VENDOR/CONFIG),
        checkpoint_file=result['checkpoint_file'],checkpoint_sha256=result['checkpoint_sha256'],
        mode_data=str(MODE_DATA.resolve()),mode_manifest_sha256=digest(MODE_DATA/'MANIFEST.json'),
        original_trainer_sha256=digest(Path(original.__file__)),
        executed_mode_trainer_sha256=protocol['adapter_sha256'],official_sources=protocol['sources'],
        pool_sources=pool_sources)
    return fixed,[c['original'] for c in cases],result,sources


def fixed_arrays(case):
    item=case['item']
    return dict(input_surface_xyz_vae=item['input_surface_xyz_vae'],input_normals=item['input_normals'],
                queries_vae=item['train_queries_vae'],labels=item['train_labels'],
                volume_grid_indices=case['volume'],near_indices=case['near'])


def prepare(output):
    fixed,_,result,sources=load_sources()
    if digest(SOURCE/sources['checkpoint_file'])!=sources['checkpoint_sha256']:
        raise ValueError('Actual full-model checkpoint differs')
    output.mkdir(parents=True,exist_ok=False);records=[]
    for c in fixed:
        path=output/(c['object_id']+'.npz');np.savez_compressed(path,**fixed_arrays(c))
        records.append(dict(object_id=c['object_id'],seed_step=c['seed_step'],file=path.name,
                            sha256=digest(path),sampling=c['sampling']))
    manifest=dict(scope=SCOPE,recipe=recipe(),sources=sources,cases=records,
                  adapter_sha256=digest(Path(__file__)),model_forwards=0,optimizer_updates=0,
                  source_numerical_passed=result['numerical_passed'])
    original.write_json(output/'MANIFEST.json',manifest)
    return manifest


def read_data(directory):
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    fixed,original_items,result,sources=load_sources()
    if (manifest['scope']!=SCOPE or manifest['recipe']!=recipe() or sources!=manifest['sources']
            or manifest['adapter_sha256']!=digest(Path(__file__)) or len(manifest['cases'])!=4):
        raise ValueError('Bound diagnostic data/source protocol changed')
    if digest(SOURCE/sources['checkpoint_file'])!=sources['checkpoint_sha256']:
        raise ValueError('Actual full-model checkpoint no longer matches the bound source')
    for c,row in zip(fixed,manifest['cases']):
        path=directory/row['file']
        if row['object_id']!=c['object_id'] or row['seed_step']!=c['seed_step'] or digest(path)!=row['sha256']:
            raise ValueError('Diagnostic batch identity/hash changed')
        with np.load(path,allow_pickle=False) as z:
            expected=fixed_arrays(c)
            if set(z.files)!=set(expected) or any(not np.array_equal(z[k],v) for k,v in expected.items()):
                raise ValueError('Fixed batch does not replay original sampler/labels/near')
    return manifest,fixed,original_items,result


def official_class(config):
    if config['target']!='michelangelo.models.tsal.sal_perceiver.AlignedShapeLatentPerceiver':
        raise ValueError('Require complete original aligned shape AE')
    sys.path.insert(0,str(VENDOR));name,cls=config['target'].rsplit('.',1)
    module=importlib.import_module(name)
    if Path(module.__file__).resolve()!=(VENDOR/'michelangelo/models/tsal/sal_perceiver.py').resolve():
        raise ValueError('Unofficial implementation imported')
    return getattr(module,cls)


class CallCounter:
    """Delegate every call to the actual original bound method, counting attempts."""
    def __init__(self):self.attempted=Counter();self.completed=Counter()
    def install(self,model):
        for name in ('forward','encode','decode','query_geometry'):
            original_method=getattr(model,name)
            def counted(*args,_name=name,_method=original_method,**kwargs):
                self.attempted[_name]+=1
                if self.attempted[_name]>BUDGET[_name]:raise RuntimeError('Forward budget exceeded: '+_name)
                value=_method(*args,**kwargs);self.completed[_name]+=1;return value
            setattr(model,name,counted)


def clone_optimizer_state(state,device):
    """Never alias CPU Adam step scalars or moments from the immutable snapshot."""
    import torch
    result=dict(param_groups=copy.deepcopy(state['param_groups']),state={})
    for key,values in state['state'].items():
        result['state'][key]={field:value.to(device=value.device if field=='step' else device,copy=True)
                            if isinstance(value,torch.Tensor) else copy.deepcopy(value)
                            for field,value in values.items()}
    return result


def match_state(model,optimizer,saved,expected_step=1000):
    import torch
    current=model.state_dict();live=optimizer.state_dict()
    if current.keys()!=saved['model'].keys() or live['param_groups']!=saved['optimizer']['param_groups']:
        raise RuntimeError('Full model/Adam groups differ from the endpoint')
    if live['state'].keys()!=saved['optimizer']['state'].keys():raise RuntimeError('Incomplete Adam state')
    for key,value in current.items():
        if not torch.equal(value.detach().cpu(),saved['model'][key]):raise RuntimeError('Model restore differs: '+key)
    for key,values in live['state'].items():
        if set(values)!=set(saved['optimizer']['state'][key]) or float(values['step'])!=expected_step:
            raise RuntimeError('Wrong restored Adam clock/fields')
        for field,value in values.items():
            if not torch.equal(value.detach().cpu(),saved['optimizer']['state'][key][field]):
                raise RuntimeError('Adam restore differs')
    return dict(passed=True,model_tensors=len(current),adam_states=len(live['state']),every_step=expected_step)


def restore(model,optimizer,saved,device):
    import torch
    model.load_state_dict(saved['model'],strict=True);model.requires_grad_(True).eval()
    optimizer.load_state_dict(clone_optimizer_state(saved['optimizer'],device))
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(saved['rng_state'].clone())
    if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng_state'].clone(),device)
    receipt=match_state(model,optimizer,saved)
    receipt.update(cpu_rng_exact=torch.equal(torch.get_rng_state(),saved['rng_state']),
        cuda_rng_exact=device.type!='cuda' or torch.equal(torch.cuda.get_rng_state(device),saved['cuda_rng_state']),
        model_mode='eval')
    if not receipt['cpu_rng_exact'] or not receipt['cuda_rng_exact']:raise RuntimeError('RNG restore differs')
    return receipt


def array_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def forward_loss(model,criterion,case,device):
    """Full original model forward and unchanged original criterion; no decoder substitute."""
    import torch
    if model.training:raise RuntimeError('All diagnostic forwards must use the same eval mode')
    torch.manual_seed(original.TRAIN_SEED+case['seed_step'])
    item=case['item']
    _,logits,posterior=model(original.tensor(item['input_surface_xyz_vae'],device),
        original.tensor(item['input_normals'],device),original.tensor(item['train_queries_vae'],device),
        sample_posterior=False)
    loss,parts=criterion(posterior,logits,original.tensor(item['train_labels'],device))
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite diagnostic loss')
    record=dict(object_id=case['object_id'],loss=float(loss.detach()),
        components={k:float(v) for k,v in parts.items()},logits_sha256=array_digest(logits),
        posterior_mean_sha256=array_digest(posterior.mean),posterior_logvar_sha256=array_digest(posterior.logvar),
        seed=original.TRAIN_SEED+case['seed_step'],model_mode='eval',sample_posterior=False)
    return loss,record


def gradient_matrix(gradients):
    import torch
    if len(gradients)!=4 or len({tuple(x) for x in gradients})!=1:
        raise ValueError('Require four complete gradients with matching parameter names')
    matrix=np.zeros((4,4),np.float64)
    for name in gradients[0]:
        vectors=[g[name].reshape(-1).double() for g in gradients]
        for i in range(4):
            for j in range(i,4):matrix[i,j]+=float(torch.dot(vectors[i],vectors[j]))
    matrix=matrix+np.triu(matrix,1).T
    norms=np.sqrt(np.diag(matrix))
    if not np.isfinite(matrix).all() or not (norms>0).all():raise ValueError('Invalid full gradient matrix')
    return matrix,matrix/np.outer(norms,norms)


def displacement(model,saved,gradients):
    import torch
    projections=np.zeros(4,np.float64);squared=0.;maximum=0.;changed=0
    for name,parameter in model.named_parameters():
        actual=parameter.detach().cpu()
        delta=actual.double()-saved['model'][name].double()
        if not torch.isfinite(delta).all():raise RuntimeError('Nonfinite parameter update')
        squared+=float(delta.square().sum());maximum=max(maximum,float(delta.abs().max()))
        changed+=int(delta.count_nonzero())
        for j,g in enumerate(gradients):projections[j]+=float(torch.dot(g[name].reshape(-1).double(),delta.reshape(-1)))
    if changed==0:raise RuntimeError('Actual optimizer step changed no parameter')
    return dict(l2=float(np.sqrt(squared)),maximum_absolute=maximum,changed_elements=changed,
                gradient_dot_actual_displacement=projections.tolist())


def validate_budget(counter,backwards,updates):
    for name in ('forward','encode','decode','query_geometry'):
        if counter.attempted[name]!=BUDGET[name] or counter.completed[name]!=BUDGET[name]:
            raise RuntimeError('Incomplete/mismatched actual call budget: '+name)
    if backwards!=4 or updates!=8:raise RuntimeError('Wrong backward/temporary-update budget')


def snapshot_digest(saved):
    """Read all cached model/Adam tensors, including CPU scalars, without mutation."""
    import torch
    hasher=hashlib.sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):
            hasher.update(str((str(value.dtype),tuple(value.shape))).encode())
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):hasher.update(str(key).encode());visit(value[key])
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        else:hasher.update(repr(value).encode())
    for key in ('model','optimizer','rng_state','cuda_rng_state'):visit(saved[key])
    return hasher.hexdigest()


def heldout(model,criterion,items,device):
    import torch
    if model.training:raise RuntimeError('Heldout and fixed-batch model modes differ')
    torch.manual_seed(original.TRAIN_SEED)
    return original.evaluate_queries(model,criterion,items,device)


def run(args):
    from .retained_execution import require_active_resource
    resource=require_active_resource()
    import torch
    manifest,fixed,items,source_result=read_data(args.data)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope=SCOPE,diagnostic_only=True,recipe=recipe(),resource=resource,
        data_manifest_sha256=digest(args.data/'MANIFEST.json'),data_manifest=manifest,
        driver_sha256=digest(Path(__file__)),source_model_updates=1000,
        source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],
        no_deployable_checkpoint=True,model_acceptance_evaluated=False,
        runtime=dict(torch_version=str(torch.__version__),numpy_version=np.__version__,
            python_version=sys.version,cuda_build=torch.version.cuda,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            num_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            settings_changed_by_diagnostic=False))
    original.write_json(args.output/'PROTOCOL.json',protocol)
    device=torch.device('cuda');counter=CallCounter()
    attempted_backwards=completed_backwards=attempted_updates=completed_updates=0
    model=optimizer=saved=None;trials=[];grad_records=[];error=None;final_restore=None
    baseline_digest=None;gram=cosine=None;fixed_baseline=None;eval_baseline=None
    try:
        checkpoint=SOURCE/manifest['sources']['checkpoint_file']
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        source_protocol=json.loads((SOURCE/'PROTOCOL.json').read_text())
        if saved['step']!=1000 or saved['protocol']!=source_protocol:
            raise RuntimeError('Saved checkpoint step/protocol differs from the completed source')
        cls=official_class(source_result['loading']['official_config'])
        model=cls(device=None,dtype=None,**source_result['loading']['official_config']['params']).to(device)
        protocol['runtime'].update(parameter_dtype=str(next(model.parameters()).dtype),
                                   parameter_device=str(next(model.parameters()).device))
        original.write_json(args.output/'PROTOCOL.json',protocol)
        names=[name for name,_ in model.named_parameters()]
        if names!=saved['parameter_names'] or len(names)!=307 or sum(p.numel() for p in model.parameters())!=184659585:
            raise RuntimeError('Incomplete original model/parameter ordering')
        optimizer=original.optimizer_for(model);criterion=original.original_loss()
        baseline_digest=snapshot_digest(saved)
        initial_restore=restore(model,optimizer,saved,device);counter.install(model)
        eval_baseline=heldout(model,criterion,items,device)
        expected=json.loads((SOURCE/'RELOAD.json').read_text())['cases']
        if eval_baseline!=expected:
            raise RuntimeError('Original heldout/latent baseline is not exact: abort before trial updates')
        original.write_json(args.output/'BASELINE_HELDOUT.json',dict(passed=True,cases=eval_baseline,
            original_outputs_exact=True,initial_restore=initial_restore))
        print('ADAM_DIAGNOSTIC_BASELINE_HELDOUT_EXACT',flush=True)
        fixed_baseline=[]
        with torch.no_grad():
            for case in fixed:
                _,a=forward_loss(model,criterion,case,device)
                _,b=forward_loss(model,criterion,case,device)
                if a!=b:raise RuntimeError('Same-input full baseline repeats differ')
                fixed_baseline.append(a)
        original.write_json(args.output/'BASELINE_FIXED_REPEATS.json',dict(passed=True,
            full_forward_calls=8,repeats_per_object=2,cases=fixed_baseline))
        gradients=[]
        for index,case in enumerate(fixed):
            restored=restore(model,optimizer,saved,device)
            loss,record=forward_loss(model,criterion,case,device)
            if record!=fixed_baseline[index]:raise RuntimeError('Gradient and no-grad baseline forward differ')
            attempted_backwards+=1
            loss.backward();completed_backwards+=1
            summary=original.gradients(model)
            # Same source operation: infinite threshold checks finite norm;
            # there is no finite clipping or altered original gradient.
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            gradients.append({name:param.grad.detach().cpu().clone() for name,param in model.named_parameters()})
            grad_records.append(dict(object_id=case['object_id'],loss=record,restore=restored,
                complete_gradient=summary,gradient_norm=float(norm),parameter_tensors=307))
            original.write_json(args.output/'GRADIENTS.json',grad_records)
            print('ADAM_DIAGNOSTIC_FULL_GRADIENT',case['object_id'],float(norm),flush=True)
        gram,cosine=gradient_matrix(gradients)
        original.write_json(args.output/'GRADIENT_MATRIX.json',dict(object_ids=list(OBJECT_IDS),
            full_gradient_dot=gram.tolist(),cosine=cosine.tolist(),reduction='per-tensor float64 over all parameters'))
        for index,case in enumerate(fixed):
            for scale in SCALES:
                restored=restore(model,optimizer,saved,device)
                # A full cached-state hash checks alias contamination beyond
                # merely comparing current live state with a polluted cache.
                if snapshot_digest(saved)!=baseline_digest:raise RuntimeError('Cached source state was mutated')
                for name,param in model.named_parameters():
                    param.grad=gradients[index][name].to(device=device,copy=True)
                for group in optimizer.param_groups:group['lr']=1e-4*scale
                if attempted_updates>=8:raise RuntimeError('Temporary optimizer budget exceeded')
                attempted_updates+=1;optimizer.step();completed_updates+=1
                actual_steps=[float(state['step']) for state in optimizer.state.values()]
                if len(actual_steps)!=307 or any(step!=1001 for step in actual_steps):
                    raise RuntimeError('Independent trial did not step all Adam states1000-to1001')
                change=displacement(model,saved,gradients)
                after_fixed=[]
                with torch.no_grad():
                    for other in fixed:
                        _,value=forward_loss(model,criterion,other,device);after_fixed.append(value)
                after_eval=heldout(model,criterion,items,device)
                delta=[value['loss']-before['loss'] for value,before in zip(after_fixed,fixed_baseline)]
                projections=change['gradient_dot_actual_displacement']
                row=dict(trial_index=len(trials),gradient_object_id=case['object_id'],learning_rate_scale=scale,
                    learning_rate=1e-4*scale,restore=restored,all_adam_steps_before=1000,all_adam_steps_after=1001,
                    actual_temporary_optimizer_steps=1,actual_parameter_displacement=change,
                    fixed_batch_before=fixed_baseline,fixed_batch_after=after_fixed,fixed_loss_change=delta,
                    loss_change_minus_first_order=[x-y for x,y in zip(delta,projections)],
                    heldout_before=eval_baseline,heldout_after=after_eval,
                    heldout_loss_change=[a['loss']-b['loss'] for a,b in zip(after_eval,eval_baseline)],
                    heldout_iou_change=[a['occupancy_iou']-b['occupancy_iou'] for a,b in zip(after_eval,eval_baseline)])
                trials.append(row);original.write_json(args.output/'TRIALS.json',trials)
                print('ADAM_DIAGNOSTIC_TEMPORARY_TRIAL',len(trials),case['object_id'],scale,delta,flush=True)
        validate_budget(counter,completed_backwards,completed_updates)
    except Exception:error=traceback.format_exc()
    finally:
        if model is not None and optimizer is not None and saved is not None:
            try:
                if snapshot_digest(saved)!=baseline_digest:raise RuntimeError('Cached source changed before final restore')
                final_restore=restore(model,optimizer,saved,device)
                if snapshot_digest(saved)!=baseline_digest:raise RuntimeError('Final restore mutated cached source')
                final_restore['cached_source_all_tensors_exact']=True
            except Exception:
                error=(error or '')+'\nFINAL_RESTORE_FAILURE\n'+traceback.format_exc()
    complete=error is None and completed_updates==8 and completed_backwards==4 and final_restore is not None
    result=dict(scope=SCOPE,diagnostic_only=True,complete=complete,diagnostic_execution_passed=complete,
        recipe=recipe(),resource=resource,source_checkpoint_sha256=manifest['sources']['checkpoint_sha256'],
        protocol_sha256=digest(args.output/'PROTOCOL.json'),driver_sha256=digest(Path(__file__)),
        model_acceptance_evaluated=False,no_deployable_checkpoint=True,source_model_updates=1000,
        original1000_finally_restored=bool(final_restore and final_restore.get('passed')),
        final_restoration=final_restore,attempted_full_backwards=attempted_backwards,
        actual_full_backwards=completed_backwards,temporary_optimizer_step_attempts=attempted_updates,
        actual_temporary_optimizer_steps=completed_updates,trial_adam_clock='1000-to1001 independently; never cumulative1008',
        api_attempted=dict(counter.attempted),api_completed=dict(counter.completed),
        gradient_matrix=None if gram is None else gram.tolist(),gradient_cosine=None if cosine is None else cosine.tolist(),
        gradients=grad_records,trials=trials,physics_controls=0,grid_scans=0,mesh_extractions=0,
        historical_cause_proven=False,error=error)
    original.write_json(args.output/'RESULT.json',result)
    print('ADAM_DIAGNOSTIC_COMPLETE',complete,'TEMPORARY_STEPS',completed_updates,flush=True)
    return 0 if complete else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','check','run'))
    parser.add_argument('--data',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':
        if args.output is None:parser.error('prepare requires a new --output')
        print(json.dumps(prepare(args.output)));return 0
    if args.data is None:parser.error('check/run require --data')
    if args.stage=='check':
        manifest,fixed,items,result=read_data(args.data)
        criterion=original.original_loss();deps=dependency_check()
        import torch
        cls=official_class(result['loading']['official_config'])
        with torch.device('meta'):model=cls(device=None,dtype=None,**result['loading']['official_config']['params'])
        count=len(list(model.parameters()));elements=sum(p.numel() for p in model.parameters())
        if count!=307 or elements!=184659585 or not all(deps.values()) or torch.cuda.is_initialized():
            raise RuntimeError('Official model/dependency CPU qualification failed')
        print(json.dumps(dict(passed=True,scope=SCOPE,recipe=recipe(),parameter_tensors=count,
            parameter_elements=elements,original_class=type(model).__module__+'.'+type(model).__name__,
            original_loss=type(criterion).__module__+'.'+type(criterion).__name__,cuda_initialized=False,
            model_forwards=0,optimizer_updates=0,dependencies=deps)));return 0
    if args.output is None:parser.error('run requires a new --output')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
