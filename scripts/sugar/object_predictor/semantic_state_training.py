"""Frozen two-arm semantic-state training contract and faithful evaluation glue.

The official full Utonia and its original readouts remain the predictor. The
new352-column observation blocks are additive linear input adapters, not a
replacement model. Legacy force/physical failures remain separate outcomes.
"""
from __future__ import annotations

from copy import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import train_overfit as original
from . import semantic_state_data as data
from . import semantic_state_readout as readout
from . import semantic_state_loss as objective

STUDY='semantic_state_training_v1'
SCHEMA=6
CONDITIONS=('no_summary','observed_summary')
SOURCE_UPDATES=2100
UPDATES=20
MICROBATCHES=116
FIT_ROWS=464
FINAL_STEP=2120
LR_BASE=dict(full_official_backbone=1e-6,sensor_affine=5e-5,state_readout=2e-6,
    auxiliary_readout=1e-3,semantic_state_observation=1e-6,semantic_classification_observation=5e-4)
NEW_NAMES=('predictor.head.summary_weight','auxiliary.summary_weight')
EXTRA_SOURCES=tuple('scripts.sugar.object_predictor.'+name for name in (
    'semantic_state_data','semantic_state_readout','semantic_state_loss',
    'semantic_state_training','train_semantic_state','observed_force_summary',
    'overfit_fullbatch_refinement'))
ARTIFACTS=('model.pt','initial_fit.npz','fit_02120.npz',
    'initial_same_trajectory_interpolation.npz','same_trajectory_interpolation.npz')
IDENTITY_KEYS=('episode','frame','timestamp_s','target','force_target_n','mass_available',
    'mass_status','contact_present','state_precision_eligible','state_contact_history_frames',
    'state_evidence_status','original_fit_item','semantic_role')


def actual_source_bindings():
    result=original.actual_source_bindings()
    for name in EXTRA_SOURCES:
        path=Path(importlib.util.find_spec(name).origin).resolve()
        result[name]=dict(path=str(path),sha256=data.sha(path))
    return result


def source_protocol(source,qualification):
    p,binding=data.source_contract(source)
    qualification=Path(qualification).resolve()
    q=json.loads((qualification/'RESULT.json').read_text())
    qp=json.loads((qualification/'PROTOCOL.json').read_text())
    if not q.get('passed') or q.get('rows')!=464 or not q.get('original80_exact'):
        raise ValueError('Actual complete464 CPU qualification is required')
    if q['source_bindings']!=binding or qp['source']!=binding:
        raise ValueError('CPU qualification belongs to a different source2100')
    for path,digest in qp['prepared_sources'].items():
        if data.sha(path)!=digest:raise ValueError('Qualified data source changed: '+path)
    archive=qualification/'OBSERVATION_QUALIFICATION.npz'
    if data.sha(archive)!=q['qualification_npz_sha256']:raise ValueError('Qualified observations changed')
    with np.load(archive) as z:scales,scale_report=readout.fit_observation_scales(z['summary'])
    return dict(study=STUDY,schema=SCHEMA,supervision_profile=data.PROFILE,
        collection_study='controlled_fixture16_v1',
        state_evidence_status_codes={'0':'PRIOR_UNKNOWN','1':'CONTACT_CANDIDATE_OBSERVABILITY_UNPROVEN'},
        source_endpoint=str(Path(source).resolve()),
        source_bindings=binding['bindings'],source_data=p['source_data'],data=p['data'],
        checkpoint=p['checkpoint'],mesh=p['mesh'],seed=p['seed'],conditions=list(CONDITIONS),
        qualification=str(qualification),qualification_bindings={name:dict(path=str(qualification/name),sha256=data.sha(qualification/name))
            for name in ('PROTOCOL.json','RESULT.json','OBSERVATION_QUALIFICATION.npz')},
        prepared_sources=actual_source_bindings(),source_optimizer_updates=SOURCE_UPDATES,
        optimizer_updates=UPDATES,total_optimizer_updates=FINAL_STEP,steps=UPDATES,
        batch=4,effective_batch=FIT_ROWS,microbatches_per_update=MICROBATCHES,
        training_microbatches_per_arm=UPDATES*MICROBATCHES,training_row_exposures_per_arm=UPDATES*FIT_ROWS,
        fit_frames=list(data.FIT_FRAMES),original_fit_frames=list(data.FIXED_FRAMES),
        additional_fit_frames=list(data.ADDED_FRAMES),interpolation_frames=list(data.EVALUATION_FRAMES),
        fit_items=464,interpolation_items=1440,summary_fields=list(data.SUMMARY_FIELDS),
        summary_shape_per_item=[32,11],summary_scales=scales.tolist(),summary_scaling=scale_report,
        denominators=q['denominators'],learning_rates=LR_BASE,learning_rate_schedule=dict(
            kind='cosine',updates=20,first_factor=1.,last_factor=.05),weight_decay=.01,gradient_clip_norm=100.,
        loss_scales=original.LOSS_SCALES,semantic_loss_weights=objective.WEIGHTS,
        loss_normalization='Sum116 microbatch contributions with fixed global251/43/43positive421negative/464 denominators. No division by116.',
        fit_limits=original.FIT_LIMITS,interpolation_limits=original.INTERPOLATION_LIMITS,
        semantic_acceptance='Exactly original state/mass/availability thresholds; exclude only legacy force checks explicitly. Raw force diagnostics and independent legacy acceptance retained.',
        no_summary='Same architecture and added parameters, but zero explicit352 history inputs; original20features still contain tactile observations. Not a proprio-only ablation.',
        history_height='Actual hand_pose_w[history_indices,0,2]/0.2m for each of32 causal frames; publicfloorz0, no object min_z/GT/mask input.',
        adapter='Original-width F.linear unchanged plus separate zero-initialized bias-free blocks (state13x352/classification2x352). Original parameters and Adam clocks2100 preserved; new blocks clock0.',
        original_force_rows='No new summary columns/no force loss. Original8 auxiliary rows retained; shared features, old Adam moments and decay can change their predictions. Errors are legacy diagnostics, not learned mechanics.',
        initial_evaluation='All464+1440 actual batch1 forwards; original readout versus zero-addition exact in same forward. Historical source differences measured, never asserted cross-process exact.',
        endpoint_prediction_file='fit_02120.npz',initial_prediction_total_optimizer_updates=2100,
        evaluation_schedule=[20],precision='float32; TF32 enabled, original stochastic regularizers disabled',
        lr_rationale=dict(normalized_fit_l1=scale_report['normalized_l1'],
            new_state_lr=1e-6,new_classification_lr=5e-4,
            state_coherent_first_step_bound_m=scale_report['normalized_l1']['maximum']*1e-6,
            classification_coherent_first_step_bound_logit=scale_report['normalized_l1']['maximum']*5e-4,
            note='L1*LR is an idealized coherent-direction bound, not actual Adam movement. Conservative state residual and faster classification; negative short-run outcome does not prove input unhelpful.'),
        fairness='Both new arms use same source/464/semantic objective/batches/20full updates. 9280row exposures/arm versus previous8000;20Adam steps versus100. Data/objective changes shared, observation-only between-arm difference.',
        evaluation_scope='Original80/additional384/all464 fit and previously-viewed1440 development interpolation. No untouched test/generalization/tactile benefit claim.',
        scope='Known-mesh controlled failed/successful TRAIN state qualification. Physical15/16 remains FAIL; original blind/conditional failures are unchanged.',
        stop='One fixed20-update endpoint per arm; no best checkpoint, retry, extra budget or automatic next neural experiment.')


def validate_protocol(root):
    root=Path(root);p=json.loads((root/'PROTOCOL.json').read_text())
    expected=source_protocol(p['source_endpoint'],p['qualification'])
    if p!=expected:raise ValueError('Prepared pair protocol/source changed')
    return p


def arm_protocol(protocol,condition):
    if condition not in CONDITIONS:raise ValueError('Unknown observation condition')
    return dict(protocol,condition=condition)


def begin_artifacts(root):
    root=Path(root)
    manifest=dict(schema=SCHEMA,kind=STUDY,complete=False,sources=actual_source_bindings(),
        protocol=dict(path='PROTOCOL.json',sha256=data.sha(root/'PROTOCOL.json')),artifacts={})
    original.save_json(root/'ARTIFACTS.json',manifest)
    return manifest


def verify_artifacts(root,*,require_complete=True,manifest=None):
    root=Path(root)
    m=manifest or json.loads((root/'ARTIFACTS.json').read_text())
    if m.get('schema')!=SCHEMA or m.get('kind')!=STUDY or (require_complete and m.get('complete') is not True):
        raise ValueError('Require completed semantic arm schema6')
    if set(m['sources'])!=set(original.SOURCE_MODULES)|set(EXTRA_SOURCES):raise ValueError('Unexpected semantic source set')
    for name,b in m['sources'].items():
        if data.sha(b['path'])!=b['sha256']:raise ValueError('Semantic source changed: '+name)
    if m['protocol']!=dict(path='PROTOCOL.json',sha256=data.sha(root/'PROTOCOL.json')):raise ValueError('Arm protocol changed')
    if require_complete:
        if set(m['artifacts'])!=set(ARTIFACTS):raise ValueError('Incomplete semantic predictions/checkpoint')
        for name,b in m['artifacts'].items():
            if b!=dict(path=name,sha256=data.sha(root/name)):raise ValueError('Semantic artifact changed: '+name)
    return m


def finish_artifacts(root,manifest):
    verify_artifacts(root,require_complete=False,manifest=manifest)
    final=dict(manifest,complete=True,artifacts={n:dict(path=n,sha256=data.sha(Path(root)/n)) for n in ARTIFACTS})
    original.save_json(Path(root)/'ARTIFACTS.json',final)
    return final


def qualify_condition(dataset,condition,limits):
    # The no-summary arm does not receive historical absolute height. Checking
    # only the augmented-input grouping would miss a contradiction in this arm.
    proxy=copy(dataset)
    if condition=='no_summary':
        proxy.rows=[dict(row,observed_left_hand_world_height_m=np.zeros_like(row['observed_left_hand_world_height_m'])) for row in dataset.rows]
    elif condition!='observed_summary':raise ValueError('Unknown condition')
    return data.qualification(proxy,limits)


def cache_observations(dataset,protocol,*,verify_qualified_fit=False):
    values=[];fingerprints=[]
    for start in range(0,len(dataset),4):
        rows=dataset.rows[start:start+4]
        summary=data.collate_semantic_observations(rows)['observation_history'].numpy()
        values.append(summary)
        for row in rows:
            h=hashlib.sha256()
            for key in ('coord','grid_coord','feat'):
                for v in row['inputs'][key]:h.update(np.ascontiguousarray(v).tobytes())
            h.update(row['observed_left_hand_world_height_m'].tobytes());fingerprints.append(h.hexdigest())
    values=np.concatenate(values)
    if verify_qualified_fit:
        with np.load(Path(protocol['qualification'])/'OBSERVATION_QUALIFICATION.npz') as z:
            checks=dict(summary=np.array_equal(values,z['summary']),inputs=np.array_equal(fingerprints,z['input_sha256']),
                target=np.array_equal(np.stack([r['target'] for r in dataset.rows]),z['target']),
                masks=np.array_equal([[r['supervision']['state_precision_eligible'],r['supervision']['mass_available']] for r in dataset.rows],z['masks']),
                episode=np.array_equal([r['metadata']['episode'] for r in dataset.rows],z['episode']),
                frame=np.array_equal([r['metadata']['frame'] for r in dataset.rows],z['frame']))
        if not all(checks.values()):raise ValueError('Fresh464 differs from actual prepared qualification: '+str(checks))
    else:checks=dict(actual_causal_history=True)
    for row,value in zip(dataset.rows,values,strict=True):row['_observed_summary']=value
    return dict(passed=True,rows=len(dataset),checks=checks,summary_shape=list(values.shape),
        labels_never_used_to_construct_summary=True)


def prepare_batch(rows,device,scales,condition):
    batch=original.device_batch(original.collate_conditional(rows),device)
    history=torch.from_numpy(np.stack([r['_observed_summary'] for r in rows])).to(device)
    history=readout.scale_observation_history(history,scales)
    if condition=='no_summary':history=torch.zeros_like(history)
    elif condition!='observed_summary':raise ValueError('Unknown condition')
    return batch,history


def fixed_batches(dataset,seed):
    lookup={(r['metadata']['episode'],r['metadata']['frame']):i for i,r in enumerate(dataset.rows)}
    groups=(range(5000,5004),range(5008,5012),range(5012,5016),range(5020,5024))
    batches=[[lookup[e,f] for e in group] for group in groups for f in data.FIT_FRAMES]
    if len(batches)!=116 or sorted(i for b in batches for i in b)!=list(range(464)):
        raise ValueError('Every464 item must occur once in116 four-item batches')
    rng=np.random.default_rng(seed)
    while True:
        yield [batches[i] for i in rng.permutation(len(batches))]


def learning_rates(update):
    if type(update) is not int or not 1<=update<=20:raise ValueError('Fixed20-update budget')
    factor=.05+.95*.5*(1+math.cos(math.pi*(update-1)/19))
    return {k:v*factor for k,v in LR_BASE.items()}


def apply_learning_rates(optimizer,update):
    rates=learning_rates(update)
    if {g['name'] for g in optimizer.param_groups}!=set(rates):raise ValueError('Unexpected semantic groups')
    for group in optimizer.param_groups:group['lr']=rates[group['name']]
    return rates


def acceptance(arrays,role,*,expected_frames=None):
    """Reuse old numerical and N/A rules; change clocks then separate force."""
    frames=tuple(expected_frames or (data.FIT_FRAMES if role=='fit' else data.EVALUATION_FRAMES))
    def base(values,clock_role):
        raw=original.acceptance(values,clock_role)
        key='all_five_clocks_present' if clock_role=='fit' else 'all_dense_clocks_present'
        for episode,row in raw['per_case'].items():
            if 'checks' not in row:continue
            row['checks'][key]=sorted(values['frame'][values['episode']==int(episode)].tolist())==list(frames)
            row['passed']=all(row['checks'].values())
        raw['checks']={f'{e}/{k}':bool(v) for e,row in raw['per_case'].items() for k,v in row.get('checks',{'present':False}).items()}
        raw['passed']=all(raw['checks'].values())
        return raw
    legacy=original.failure_aware_acceptance(arrays,role,original_acceptance=base,
        fit_limits=original.FIT_LIMITS,interpolation_limits=original.INTERPOLATION_LIMITS,episodes=data.FIXED_EPISODES)
    semantic_checks={k:v for k,v in legacy['checks'].items() if k.rsplit('/',1)[-1]!='force'}
    semantic_cases={}
    for episode,row in legacy['per_case'].items():
        checks={k:v for k,v in row.get('checks',{'present':False}).items() if k!='force'}
        semantic_cases[episode]=dict(row,checks=checks,passed=all(v for v in checks.values() if v is not None))
    return dict(legacy,passed=all(semantic_checks.values()),checks=semantic_checks,per_case=semantic_cases,
        semantic_acceptance_passed=all(semantic_checks.values()),legacy_acceptance_passed=legacy['passed'],
        legacy_force_gate_passed=all(v for k,v in legacy['checks'].items() if k.rsplit('/',1)[-1]=='force'),
        legacy_checks=legacy['checks'],legacy_per_case=legacy['per_case'],
        expected_frames=list(frames),semantic_gate_excludes=['force'],
        original_historical_overall_result_modified=False,
        scope='Semantic state/mass/availability only; input-derived force regression is legacy diagnostic. Contact candidates are not observability certificates; unknown raw errors remain visible.')


@torch.no_grad()
def evaluate(model,dataset,full_vertices,device,scales,condition):
    # Reuse every original batch1 metric/full15626vertex geometry operation.
    # The proxy only supplies the explicit observation history to the same full
    # model; original evaluator's hardcoded80 clock report is discarded/rebuilt.
    class EvaluationCall:
        def __init__(self):self.index=0
        def eval(self):model.eval()
        def __call__(self,points):
            row=dataset.rows[self.index];self.index+=1
            history=torch.from_numpy(row['_observed_summary'][None]).to(device)
            history=readout.scale_observation_history(history,scales)
            if condition=='no_summary':history=torch.zeros_like(history)
            return readout.forward_semantic_observations(model,points,history)
    proxy=copy(dataset);proxy.supervision_profile=original.FAILURE_AWARE_PROFILE
    role='fit' if dataset.semantic_role=='fit' else 'same_trajectory_interpolation'
    proxy.clock_role=role;call=EvaluationCall()
    _,arrays=original.evaluate(call,proxy,full_vertices,device)
    if call.index!=len(dataset):raise RuntimeError('Evaluation skipped or added rows')
    arrays['original_fit_item']=np.isin(arrays['frame'],data.FIXED_FRAMES).astype(np.int8)
    arrays['semantic_role']=np.full(len(dataset),0 if role=='fit' else 1,dtype=np.int8)
    result=acceptance(arrays,role)
    if role=='fit':
        result['fit_subsets']={}
        for label,frames in (('original80',data.FIXED_FRAMES),('additional384',data.ADDED_FRAMES)):
            selected=np.isin(arrays['frame'],frames)
            result['fit_subsets'][label]=acceptance({k:v[selected] for k,v in arrays.items()},'fit',expected_frames=frames)
    return result,arrays


def parameter_update_report(model,optimizer,initial,condition):
    rows=[];modules={};bound=[p for g in optimizer.param_groups for p in g['params']]
    for name,p in model.named_parameters():
        state=optimizer.state.get(p,{})
        change=p.detach().cpu()-initial[name];n=int(torch.count_nonzero(change))
        unused=name=='predictor.backbone.embedding.mask_token';new=name in NEW_NAMES
        expected=None if unused else (20 if new else 2120)
        clock=int(state['step']) if state else None
        row=dict(name=name,numel=p.numel(),changed_elements=n,max_abs_change=float(change.abs().max()),
            adam_step=clock,expected_step=expected,finite=bool(torch.isfinite(p).all()),
            moments_finite=all(bool(torch.isfinite(state[k]).all()) for k in ('exp_avg','exp_avg_sq')) if state else unused)
        rows.append(row)
        if name.startswith('predictor.backbone.') and not unused:
            key=name.rsplit('.',1)[0];modules[key]=modules.get(key,0)+n
    checks=dict(all_parameters_bound_exactly_once=len({id(p) for p in bound})==len(bound) and {id(p) for p in bound}=={id(p) for p in model.parameters()},
        original457_steps2120_and_new2_steps20=all(r['adam_step']==r['expected_step'] for r in rows),
        full_parameters_and_moments_finite=all(r['finite'] and r['moments_finite'] for r in rows),
        every_active_official_module_changed=bool(modules) and all(n>0 for n in modules.values()),
        unused_official_token_unchanged=all(r['changed_elements']==0 for r in rows if r['expected_step'] is None),
        observation_arm_blocks_changed_or_zero_arm_blocks_preserved=all((r['changed_elements']>0 if condition=='observed_summary' else r['changed_elements']==0) for r in rows if r['name'] in NEW_NAMES))
    return dict(passed=all(checks.values()),checks=checks,named_parameters=rows,official_module_changed_elements=modules)
