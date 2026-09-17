"""Project-only dynamic extraction-grid TRAIN query sampler; no model code.

Exactly 1000 updates, same four TRAIN shapes, original full-surface input and
original GT-near pool/indices. Each volume half: 512 uniform eligible nodes,
256 current false positives, 256 current false negatives. Error pools refresh
at fixed absolute steps 0,100,...900 from full deterministic extraction-grid
logits. Small pools use replacement; empty FP/FN pools fall back respectively
to uniformly sampled eligible GT-exterior/GT-interior nodes. No mesh/component
selection, evaluation feedback, altered labels, or GT as encoder features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import train_michelangelo_overfit as original
from . import prepare_michelangelo_extraction_grid as grid_source
from .qualify_michelangelo_ae import OBJECT_IDS, digest

SCOPE = 'full_surface_native_ae_dynamic_grid_overfit_only'
OLD_DATA = original.EXPERIMENT/'overfit_repair_v1/michelangelo_overfit_data_v1'
PREVIOUS_RUN = original.EXPERIMENT/'overfit_repair_v1/michelangelo_coverage_overfit_v2'
GRID_DATA = grid_source.OUTPUT
REFRESH_STEPS = tuple(range(0, 1000, 100))
VOLUME_SEED = 681729
UNIFORM_COUNT, FP_COUNT, FN_COUNT = 512, 256, 256


def fixed_recipe():
    return dict(steps=1000,updates_per_object=250,refresh_steps=list(REFRESH_STEPS),
                volume_queries=dict(uniform=UNIFORM_COUNT,false_positive=FP_COUNT,false_negative=FN_COUNT),
                near_queries=1024,near_sequence='exact original TRAIN_SEED+step indices',
                volume_rng_seed=VOLUME_SEED,volume_rng='independent NumPy generator with VOLUME_SEED+absolute_step',
                short_error_pool='replacement iff pool size < 256',
                empty_fp_pool='eligible GT exterior uniform',empty_fn_pool='eligible GT interior uniform',
                grid_sign_convention='occupied iff logit >= 0',boundary_and_eval_overlap_ineligible=True,
                uniform_sampling='without replacement per batch; no claim every node is optimized',
                refresh='complete grid, eval mode, posterior mean, no grad, restore model mode and torch RNG',
                checkpoint_reset='official released weights; no continuation',
                old_eval_and_geometric_gates_unchanged=True)


def load_sources(old_data=OLD_DATA,grid_data=GRID_DATA,previous_run=PREVIOUS_RUN):
    old_manifest,items=original.read_data(old_data)
    grid=json.loads((grid_data/'RESULT.json').read_text())
    previous=json.loads((previous_run/'RESULT.json').read_text())
    if (grid['scope']!=grid_source.SCOPE or not grid['complete'] or not grid['passed']
            or tuple(x['object_id'] for x in grid['cases'])!=OBJECT_IDS
            or previous['scope']!='full_surface_native_ae_coverage_overfit_only'
            or previous['model_updates']!=1000 or not previous['complete']
            or tuple(x['object_id'] for x in previous['cases'])!=OBJECT_IDS):
        raise ValueError('Require complete fixed four original data/grid/previous endpoint cases')
    path=grid_data/grid['grid_file']
    if digest(path)!=grid['grid_sha256']:
        raise ValueError('Prepared official extraction grid changed')
    with np.load(path,allow_pickle=False) as saved:queries=saved['queries_vae'].copy()
    expected,_,_=grid_source.official_grid()
    if queries.dtype!=np.float32 or not np.array_equal(queries,expected):
        raise ValueError('Query coordinates/order differ from the released extraction grid')
    cases=[]
    for row,old_row,item,prev_row in zip(grid['cases'],old_manifest['cases'],items,previous['cases']):
        path=grid_data/row['file']
        if (digest(path)!=row['file_sha256'] or row['old_evaluation_file_sha256']!=old_row['file_sha256']
                or row['source_vertices_sha256']!=old_row['source_vertices_sha256']
                or row['source_faces_sha256']!=old_row['source_faces_sha256']
                or digest(previous_run/prev_row['output_file'])!=prev_row['case_output_sha256']):
            raise ValueError('Original inputs, GT labels or previous comparison source changed')
        with np.load(path,allow_pickle=False) as z:
            labels=z['occupancy_labels'].copy();eligible=z['training_eligible'].copy()
            boundary=z['boundary_ambiguous'].copy();overlap=z['evaluation_overlap'].copy()
        if (any(x.dtype!=bool or x.shape!=(len(queries),) for x in (labels,eligible,boundary,overlap))
                or not np.array_equal(eligible,~(boundary|overlap))
                or not (eligible&labels).any() or not (eligible&~labels).any()):
            raise ValueError('Incomplete or inconsistent GT-grid training eligibility')
        cases.append(dict(object_id=row['object_id'],original=item,labels=labels,eligible=eligible,
                          boundary=boundary,eligible_indices=np.flatnonzero(eligible),
                          exterior_indices=np.flatnonzero(eligible&~labels),interior_indices=np.flatnonzero(eligible&labels)))
    sources=dict(old_data=str(old_data.resolve()),old_manifest_sha256=digest(old_data/'MANIFEST.json'),
                 grid_data=str(grid_data.resolve()),grid_result_sha256=digest(grid_data/'RESULT.json'),
                 grid_file_sha256=grid['grid_sha256'],previous_run=str(previous_run.resolve()),
                 previous_result_sha256=digest(previous_run/'RESULT.json'),
                 original_trainer_sha256=digest(Path(original.__file__)))
    return old_manifest,queries,cases,previous,sources


def prepare(output):
    old,queries,cases,previous,sources=load_sources()
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(scope=SCOPE,recipe=fixed_recipe(),sources=sources,object_ids=list(OBJECT_IDS),
                  grid_nodes=len(queries),cases=old['cases'],original_manifest=old,
                  previous_cases=[dict(object_id=r['object_id'],file=str((PREVIOUS_RUN/r['output_file']).resolve()),
                                       sha256=r['case_output_sha256']) for r in previous['cases']],
                  data_adapter_sha256=digest(Path(__file__)),model_forwards=0,model_updates=0,
                  counts=[dict(object_id=c['object_id'],eligible_nodes=len(c['eligible_indices']),
                               gt_interior=len(c['interior_indices']),gt_exterior=len(c['exterior_indices']),
                               ambiguous_nodes=int(c['boundary'].sum())) for c in cases])
    original.write_json(output/'MANIFEST.json',manifest)
    return manifest


def read_data(directory):
    m=json.loads((directory/'MANIFEST.json').read_text())
    if (m['scope']!=SCOPE or m['recipe']!=fixed_recipe() or tuple(m['object_ids'])!=OBJECT_IDS
            or m['data_adapter_sha256']!=digest(Path(__file__))):
        raise ValueError('Fixed dynamic-grid data protocol changed')
    s=m['sources']
    old,q,cases,previous,sources=load_sources(Path(s['old_data']),Path(s['grid_data']),Path(s['previous_run']))
    if sources!=s or old!=m['original_manifest']:
        raise ValueError('Bound original data/GT grid/source changed')
    return m,q,cases


def error_pools(logits,case):
    logits=np.asarray(logits)
    if logits.shape!=case['labels'].shape or not np.isfinite(logits).all():
        raise ValueError('Require complete finite current full-grid logits')
    predicted=logits>=0
    return dict(fp=np.flatnonzero(case['eligible']&predicted&~case['labels']),
                fn=np.flatnonzero(case['eligible']&~predicted&case['labels']))


def sample_volume(step,case,pools):
    if not 1<=step<=1000:
        raise ValueError('Step outside the fixed budget')
    rng=np.random.default_rng(VOLUME_SEED+step)
    uniform=rng.choice(case['eligible_indices'],UNIFORM_COUNT,replace=False)
    sampled=[];summary=dict(fp_pool_count=len(pools['fp']),fn_pool_count=len(pools['fn']))
    for name,count,fallback in [('fp',FP_COUNT,'exterior_indices'),('fn',FN_COUNT,'interior_indices')]:
        candidates=np.asarray(pools[name])
        if candidates.ndim!=1 or not np.issubdtype(candidates.dtype,np.integer):
            raise ValueError('Error pools must be integer grid node IDs')
        if len(candidates) and (candidates.min()<0 or candidates.max()>=len(case['labels'])
                              or not case['eligible'][candidates].all()
                              or not np.all(case['labels'][candidates]==(name=='fn'))):
            raise ValueError('Invalid error-pool labels or eligibility')
        used_fallback=not len(candidates)
        if used_fallback:candidates=case[fallback]
        replace=len(candidates)<count
        sampled.append(rng.choice(candidates,count,replace=replace))
        summary[name+'_source']='gt_class_fallback' if used_fallback else 'current_error_pool'
        summary[name+'_replacement']=replace
    indices=np.concatenate((uniform,*sampled)).astype(np.int32)
    summary.update(unique_volume_nodes=int(np.unique(indices).size),
                   uniform_gt_interior=int(case['labels'][uniform].sum()))
    return indices,summary


def batch_for_step(step,queries,cases,pools):
    index,original_indices=original.train_indices(step)
    case=cases[index];near_indices=original_indices[1024:]
    volume_indices,summary=sample_volume(step,case,pools[index])
    item=dict(case['original'])
    item['train_queries_vae']=np.concatenate((queries[volume_indices],item['train_queries_vae'][near_indices]))
    item['train_labels']=np.concatenate((case['labels'][volume_indices],item['train_labels'][near_indices]))
    return index,item,np.arange(2048),volume_indices,near_indices.astype(np.int32),summary
