"""Bound mode1000 data/pools for a single fixed low-step continuation.

The only query-sampler adaptation extends absolute step guards to2000. Original
NumPy algorithms/seeds/order and GT-near/input/evaluation arrays remain exact.
No global or frozen module is mutated. Initial error pools are the actual
mode1000 full-grid endpoint, not resampled/recomputed from a selected checkpoint.
"""
from __future__ import annotations
import ast
import inspect
import json
from pathlib import Path
from types import FunctionType
import numpy as np
from . import michelangelo_mode_grid_data as previous_data
from . import michelangelo_dynamic_grid_data as dynamic
from . import train_michelangelo_overfit as original
from .qualify_michelangelo_ae import OBJECT_IDS,VENDOR,digest

ROOT=original.EXPERIMENT/'overfit_repair_v1'
SOURCE=ROOT/'michelangelo_mode_grid_overfit_v1'
SOURCE_DATA=ROOT/'michelangelo_mode_grid_data_v1'
SCOPE='full_surface_native_ae_lowstep_continuation_only'
REVISION='dynamic_extraction_grid_mode_lowstep_continuation_v1'
START,END,LR=1000,2000,1e-5
REFRESH_STEPS=tuple(range(1100,2000,100))

# Identical original code object; a private copied globals mapping only extends
# the protocol's absolute-step bound. Original STEPS and all modules stay intact.
train_indices=FunctionType(original.train_indices.__code__,
    dict(original.train_indices.__globals__,STEPS=END),
    original.train_indices.__name__,original.train_indices.__defaults__)


def extended_sample_volume():
    """Original sampler AST, with just its literal step ceiling1000→2000."""
    tree=ast.parse(inspect.getsource(dynamic.sample_volume))
    guard=tree.body[0].body[0].test
    if not (isinstance(guard,ast.UnaryOp) and isinstance(guard.op,ast.Not)
            and ast.dump(guard.operand)==ast.dump(ast.parse('1 <= step <= 1000',mode='eval').body)):
        raise RuntimeError('Original sampler budget guard changed')
    guard.operand.comparators[-1]=ast.Constant(value=END)
    ast.fix_missing_locations(tree)
    namespace=dict(dynamic.sample_volume.__globals__)
    exec(compile(tree,str(Path(dynamic.__file__))+'[absolute_step_bound_2000]','exec'),namespace)
    return namespace['sample_volume']


sample_volume=extended_sample_volume()
error_pools=dynamic.error_pools


def recipe():
    return dict(start_step=START,end_step=END,new_updates=1000,total_updates=2000,
        new_updates_per_object=250,total_updates_per_object=500,lr=LR,
        optimizer='restore full source AdamW; change only group lr; do not reset moments/steps',
        train_posterior='mode',loss='original KLNearFar: far + .1 near + .001 KL',
        initial_pool='bound actual mode1000 ENDPOINT_GRID; no new initial grid scan',
        actual_refresh_steps=list(REFRESH_STEPS),endpoint_grid_step=END,
        batch_queries=dict(uniform=512,false_positive=256,false_negative=256,near=1024),
        near_rng='unchanged original TRAIN_SEED + absolute_step',
        volume_rng='unchanged independent NumPy VOLUME_SEED + absolute_step',
        model_rng='unchanged torch TRAIN_SEED + absolute_step',
        evaluation='original32768 queries and raw/corrected mesh gates unchanged',
        checkpoint_selection=False,learning_rate_sweep=False,
        changed='one fixed continuation from mode1000, lr1e-5; no new architecture/loss/data rule')


def load_sources():
    manifest,queries,cases=previous_data.read_data(SOURCE_DATA)
    result=json.loads((SOURCE/'RESULT.json').read_text())
    protocol=json.loads((SOURCE/'PROTOCOL.json').read_text())
    grid=json.loads((SOURCE/'ENDPOINT_GRID.json').read_text())
    reload=json.loads((SOURCE/'RELOAD.json').read_text())
    if (result['scope']!=previous_data.SCOPE or not result['complete'] or result['model_updates']!=START
            or tuple(c['object_id'] for c in result['cases'])!=OBJECT_IDS
            or tuple(c['object_id'] for c in grid['cases'])!=OBJECT_IDS or grid['step']!=START
            or protocol['train_posterior']!='mode' or not reload['passed']
            or digest(SOURCE_DATA/'MANIFEST.json')!=protocol['data_manifest_sha256']):
        raise ValueError('Require completed mode1000 full-model endpoint and pools in fixed object order')
    for rel,sha in protocol['sources'].items():
        if digest(VENDOR/rel)!=sha:raise ValueError('Original model source changed')
    mode_trainer=Path(__file__).with_name('train_michelangelo_mode_grid.py')
    if digest(mode_trainer)!=protocol['adapter_sha256']:raise ValueError('Executed source trainer changed')
    pools=[];pool_rows=[];previous=[]
    for row,case,mesh in zip(grid['cases'],cases,result['cases']):
        p=SOURCE/'endpoint_grid'/row['file']
        if digest(p)!=row['file_sha256']:raise ValueError('Initial pool bytes changed')
        with np.load(p,allow_pickle=False) as z:
            actual=error_pools(z['grid_logits'],case)
            if any(not np.array_equal(actual[k],z[k+'_indices']) for k in ('fp','fn')):
                raise ValueError('Saved initial pool does not match actual endpoint logits/GT labels')
        pools.append(actual)
        pool_rows.append(dict(object_id=row['object_id'],file=str(p.resolve()),sha256=row['file_sha256']))
        p=SOURCE/mesh['output_file']
        if digest(p)!=mesh['case_output_sha256']:raise ValueError('Actual initial mesh changed')
        previous.append(dict(object_id=mesh['object_id'],file=str(p.resolve()),sha256=mesh['case_output_sha256']))
    sources=dict(previous_run=str(SOURCE.resolve()),previous_result_sha256=digest(SOURCE/'RESULT.json'),
        previous_protocol_sha256=digest(SOURCE/'PROTOCOL.json'),previous_reload_sha256=digest(SOURCE/'RELOAD.json'),
        checkpoint_file=result['checkpoint_file'],checkpoint_sha256=result['checkpoint_sha256'],
        endpoint_grid_sha256=digest(SOURCE/'ENDPOINT_GRID.json'),pool_sources=pool_rows,
        source_data_manifest_sha256=digest(SOURCE_DATA/'MANIFEST.json'),
        mode_trainer_sha256=digest(mode_trainer),original_trainer_sha256=digest(Path(original.__file__)),
        dynamic_sampler_sha256=digest(Path(dynamic.__file__)),mode_data_sha256=digest(Path(previous_data.__file__)))
    return manifest,queries,cases,pools,result,sources,previous


def batch_for_step(step,queries,cases,pools):
    index,indices=train_indices(step)
    case=cases[index];near=indices[1024:]
    volume,info=sample_volume(step,case,pools[index])
    item=dict(case['original'])
    item['train_queries_vae']=np.concatenate((queries[volume],item['train_queries_vae'][near]))
    item['train_labels']=np.concatenate((case['labels'][volume],item['train_labels'][near]))
    return index,item,np.arange(2048),volume,near.astype(np.int32),info


def prepare(output):
    old,q,cases,pools,result,sources,previous=load_sources()
    if digest(SOURCE/result['checkpoint_file'])!=result['checkpoint_sha256']:
        raise ValueError('Actual full source checkpoint hash mismatch')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(scope=SCOPE,sampling_revision=REVISION,recipe=recipe(),sources=sources,
        original_manifest=old['original_manifest'],cases=old['cases'],object_ids=list(OBJECT_IDS),
        previous_cases=previous,grid_nodes=len(q),counts=old['counts'],
        data_adapter_sha256=digest(Path(__file__)),model_forwards=0,model_updates=0,
        source_numerical_passed=result['numerical_passed'])
    original.write_json(output/'MANIFEST.json',manifest);return manifest


def read_data(directory):
    m=json.loads((directory/'MANIFEST.json').read_text())
    old,q,cases,pools,result,sources,previous=load_sources()
    if (m['scope']!=SCOPE or m['sampling_revision']!=REVISION or m['recipe']!=recipe()
            or m['sources']!=sources or m['previous_cases']!=previous
            or m['data_adapter_sha256']!=digest(Path(__file__))
            or m['original_manifest']!=old['original_manifest'] or m['cases']!=old['cases']):
        raise ValueError('Fixed continuation source/data binding changed')
    if digest(SOURCE/result['checkpoint_file'])!=result['checkpoint_sha256']:
        raise ValueError('Actual full source checkpoint hash mismatch')
    return m,q,cases,pools,result
