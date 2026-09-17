"""Fixed optimization repair helpers; original full model/loss/gates are reused."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from . import train_overfit as original

STUDY = 'failure_aware_fullbatch_refinement_v1'
SOURCE_UPDATES = 2000
NEW_UPDATES = 100
MICROBATCHES = 20
LR_BASE = dict(full_official_backbone=1e-6,sensor_affine=5e-5,
               state_readout=2e-6,auxiliary_readout=1e-3)
LR_FLOOR = .05
EXTRA_SOURCES = ('scripts.sugar.object_predictor.overfit_fullbatch_refinement',
                 'scripts.sugar.object_predictor.train_overfit_fullbatch_refinement')
ARTIFACTS = ('model.pt','initial_fit.npz','fit_02100.npz',
             'initial_same_trajectory_interpolation.npz','same_trajectory_interpolation.npz')


def scaled_microbatch_loss(parts):
    """Preserve the old macroaverage of twenty separately masked objectives."""
    return original.total_loss(parts)/MICROBATCHES


def cosine_learning_rates(update):
    if not isinstance(update,int) or not 1<=update<=NEW_UPDATES:
        raise ValueError('Only the fixed one hundred effective updates are allowed')
    factor=LR_FLOOR+(1-LR_FLOOR)*.5*(1+math.cos(math.pi*(update-1)/(NEW_UPDATES-1)))
    return {key:value*factor for key,value in LR_BASE.items()}


def split_readout_group(optimizer,state_parameters,auxiliary_parameters):
    """Keep actual Adam state objects; change only group membership and rates."""
    groups=optimizer.param_groups
    if [group['name'] for group in groups]!=['full_official_backbone','sensor_affine','task_readouts']:
        raise ValueError('Require the original three-group Adam checkpoint')
    state_parameters=list(state_parameters);auxiliary_parameters=list(auxiliary_parameters)
    if [id(p) for p in groups[2]['params']] != [id(p) for p in state_parameters+auxiliary_parameters]:
        raise ValueError('Original readout order differs; never remap moments by shape')
    if len(state_parameters)!=2 or len(auxiliary_parameters)!=2:
        raise ValueError('Both complete original linear readouts are required')
    before={id(p):optimizer.state.get(p) for group in groups for p in group['params']}
    state_group=dict(groups[2],params=state_parameters,name='state_readout')
    auxiliary_group=dict(groups[2],params=auxiliary_parameters,name='auxiliary_readout')
    optimizer.param_groups=[groups[0],groups[1],state_group,auxiliary_group]
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    if len({id(p) for p in parameters})!=len(parameters) or {id(p) for p in parameters}!=set(before):
        raise ValueError('Refinement must bind every original parameter exactly once')
    if any(optimizer.state.get(p) is not before[id(p)] for p in parameters):
        raise RuntimeError('Splitting groups changed restored Adam state')
    apply_learning_rates(optimizer,1)


def apply_learning_rates(optimizer,update):
    rates=cosine_learning_rates(update)
    if {g['name'] for g in optimizer.param_groups}!=set(rates):
        raise ValueError('Unexpected refinement parameter groups')
    for group in optimizer.param_groups:group['lr']=rates[group['name']]
    return rates


def verify_restored_optimizer(model,optimizer,expected_step):
    names={id(p):name for name,p in model.named_parameters()}
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    if len({id(p) for p in parameters})!=len(parameters) or {id(p) for p in parameters}!=set(names):
        raise ValueError('Incomplete or duplicated full-model optimizer binding')
    total=0;active=0
    for p in parameters:
        name=names[id(p)];state=optimizer.state.get(p)
        if name=='predictor.backbone.embedding.mask_token':
            if state:raise ValueError('Previously unused official token unexpectedly has optimizer state')
            continue
        if not state or int(state['step'])!=expected_step:
            raise ValueError('Missing/wrong restored Adam clock: '+name)
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].shape!=p.shape or not bool(torch.isfinite(state[key]).all()):
                raise ValueError('Malformed full Adam moment: '+name)
        active+=1;total+=p.numel()
    return dict(passed=True,active_parameter_tensors=active,active_parameter_elements=total,
                all_active_adam_steps=expected_step,all_model_parameters_bound=True)


def source_protocol(source):
    source=Path(source).resolve()
    original.verify_artifacts(source)
    p=json.loads((source/'PROTOCOL.json').read_text())
    r=json.loads((source/'RESULT.json').read_text())
    if (p['supervision_profile']!=original.FAILURE_AWARE_PROFILE
            or r.get('execution_complete') is not True or r.get('optimizer_updates')!=SOURCE_UPDATES
            or r.get('full_parameter_update_passed') is not True or r.get('full_endpoint_reload_max_abs')!=0):
        raise ValueError('Require the complete original failure-aware two-thousand-update endpoint')
    if p['loss_scales']!=original.LOSS_SCALES or p['loss_weights']!=original.LOSS_WEIGHTS:
        raise ValueError('Original losses changed')
    if p['fit_limits']!=original.FIT_LIMITS or p['interpolation_limits']!=original.INTERPOLATION_LIMITS:
        raise ValueError('Original numerical gates changed')
    bindings={name:dict(path=str(source/name),sha256=original.sha256_file(source/name))
              for name in ('model.pt','ARTIFACTS.json','PROTOCOL.json','RESULT.json')}
    return dict(p,study=STUDY,source_endpoint=str(source),source_bindings=bindings,
        source_optimizer_updates=SOURCE_UPDATES,steps=NEW_UPDATES,optimizer_updates=NEW_UPDATES,
        total_optimizer_updates=SOURCE_UPDATES+NEW_UPDATES,microbatch=4,effective_batch=80,
        microbatches_per_update=MICROBATCHES,total_training_microbatches=MICROBATCHES*NEW_UPDATES,
        training_row_exposures=80*NEW_UPDATES,evaluate_every=25,
        learning_rates=LR_BASE,learning_rate_schedule=dict(kind='fixed cosine',updates=NEW_UPDATES,
            first_factor=1.,last_factor=LR_FLOOR,formula='floor+(1-floor)/2*(1+cos(pi*(new_update-1)/99))'),
        sampler='Continue original deterministic epoch iterator after its2000 microbatches; each effective update visits all20 original four-item batches once. Every original microbatch loss is divided by20; global63/43 denominators are NOT substituted.',
        optimization_change='Full-gradient accumulation, state/aux separate learning rates and fixed cosine decay: a multi-factor stability repair, not a single-factor batch causal test.',
        restore='Complete original model and all Adam moments/steps. Split existing readout group by parameter identity only, without resetting state. All backbone parameters remain trainable.',
        gradient_clip='Exactly one global norm100 clip after all20 backward calls, then exactly one AdamW step.',
        initial_prediction_total_optimizer_updates=SOURCE_UPDATES,endpoint_prediction_file='fit_02100.npz',
        interpolation_evaluation='Fresh batch1 before source2000->2100 refinement and at fixed endpoint, all1440 clocks/targets/masks; original gates unchanged.',
        scope=p['scope']+' Multi-factor optimization repair of the complete model; no SVD coefficients imported, no feature replacement.',
        stop='Exactly100 new optimizer updates/2000 training microbatch forwards-backwards; never choose best checkpoint or extend budget automatically. Preserve every result and all raw errors.')


def actual_source_bindings():
    import importlib.util
    result=original.actual_source_bindings()
    for name in EXTRA_SOURCES:
        path=Path(importlib.util.find_spec(name).origin).resolve()
        result[name]=dict(path=str(path),sha256=original.sha256_file(path))
    return result


def begin_artifacts(root):
    root=Path(root)
    manifest=dict(schema=2,kind=STUDY,complete=False,sources=actual_source_bindings(),
        protocol=dict(path='PROTOCOL.json',sha256=original.sha256_file(root/'PROTOCOL.json')),artifacts={})
    original.save_json(root/'ARTIFACTS.json',manifest)
    return manifest


def verify_artifacts(root,*,require_complete=True,manifest=None):
    root=Path(root)
    if manifest is None:manifest=json.loads((root/'ARTIFACTS.json').read_text())
    if (manifest.get('schema')!=2 or manifest.get('kind')!=STUDY
            or (require_complete and manifest.get('complete') is not True)):
        raise ValueError('Require explicit completed refinement artifact manifest')
    if set(manifest['sources'])!=set(original.SOURCE_MODULES)|set(EXTRA_SOURCES):
        raise ValueError('Unexpected full-model refinement source set')
    for name,binding in manifest['sources'].items():
        if original.sha256_file(binding['path'])!=binding['sha256']:
            raise ValueError('Refinement source changed: '+name)
    if manifest['protocol']!=dict(path='PROTOCOL.json',sha256=original.sha256_file(root/'PROTOCOL.json')):
        raise ValueError('Refinement protocol changed')
    if require_complete:
        if set(manifest['artifacts'])!=set(ARTIFACTS):raise ValueError('Incomplete refinement artifacts')
        for name,binding in manifest['artifacts'].items():
            if binding!=dict(path=name,sha256=original.sha256_file(root/name)):
                raise ValueError('Refinement checkpoint/prediction binding changed: '+name)
    return manifest


def finish_artifacts(root,initial):
    root=Path(root)
    verify_artifacts(root,manifest=initial,require_complete=False)
    if actual_source_bindings()!=initial['sources']:raise ValueError('Actual refinement imports changed')
    final=dict(initial,complete=True,artifacts={name:dict(path=name,sha256=original.sha256_file(root/name)) for name in ARTIFACTS})
    original.save_json(root/'ARTIFACTS.json',final)
    return final


def prepare_counterfactual(source,capture_root):
    """CPU readout moment reconstruction; never historical whole-model motion."""
    source=Path(source);capture_root=Path(capture_root)
    checkpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False,mmap=True)
    group=next(g for g in checkpoint['optimizer']['param_groups'] if g['name']=='task_readouts')
    names=('predictor.head.weight','predictor.head.bias','auxiliary.weight','auxiliary.bias')
    deltas=[];moments={}
    for name,index in zip(names,group['params'],strict=True):
        weight=checkpoint['model'][name].numpy().astype(np.float64)
        state=checkpoint['optimizer']['state'][index]
        m=state['exp_avg'].numpy().astype(np.float64);v=state['exp_avg_sq'].numpy().astype(np.float64)
        if m.shape!=weight.shape or v.shape!=weight.shape or int(state['step'])!=SOURCE_UPDATES:
            raise ValueError('Malformed actual source readout moments')
        t=int(state['step']);b1,b2=group['betas'];lr=group['lr'];wd=group['weight_decay']
        adaptive=lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+group['eps'])
        previous=(weight+adaptive)/(1-lr*wd)
        delta=weight-previous;deltas.append(delta)
        moments[name]=dict(step=t,parameter_l2=float(np.linalg.norm(weight)),
            exp_avg_l2=float(np.linalg.norm(m)),adaptive_step_l2=float(np.linalg.norm(adaptive)),
            max_abs_step=float(abs(delta).max()))
    with np.load(capture_root/'FEATURES.npz') as data:features=data['features'].astype(np.float64)
    state_delta=features@deltas[0].T+deltas[1]
    auxiliary_delta=features@deltas[2].T+deltas[3]
    paths=dict(checkpoint=source/'model.pt',train_log=source/'train.jsonl',
               captured_features=capture_root/'FEATURES.npz',method=Path(__file__).resolve())
    return dict(bindings={key:dict(path=str(path.resolve()),sha256=original.sha256_file(path)) for key,path in paths.items()},
        original_optimizer_group={key:group[key] for key in ('lr','betas','eps','weight_decay')},
        moments=moments,last_step_record=json.loads((source/'train.jsonl').read_text().splitlines()[-1]),
        fixed_r1_feature_projection=dict(center_delta_mean_cm=float(np.linalg.norm(state_delta[:,:3],axis=1).mean()*100),
            center_delta_max_cm=float(np.linalg.norm(state_delta[:,:3],axis=1).max()*100),
            logsize_delta_max_abs=float(abs(state_delta[:,9:12]).max()),
            logmass_delta_max_abs=float(abs(state_delta[:,12]).max()),
            force_delta_rmse_n=float(np.sqrt(np.mean(auxiliary_delta[:,:8]**2))),
            force_delta_max_abs_n=float(abs(auxiliary_delta[:,:8]).max())),
        formula='adaptive=lr*m/(1-beta1^t)/(sqrt(v/(1-beta2^t))+eps); previous=(saved_parameter+adaptive)/(1-lr*weight_decay); delta=saved-previous; projected_delta=X_r1*deltaW+deltaBias.',
        scope='Counterfactual CPU reconstruction ignoring optimizer operation roundoff, evaluated at fixed r1 features. Not actual historical features or whole-model motion; historical features were not saved. No gradient, model forward or optimizer update.',
        cuda_initialized=torch.cuda.is_initialized(),model_forwards=0,optimizer_updates=0)
