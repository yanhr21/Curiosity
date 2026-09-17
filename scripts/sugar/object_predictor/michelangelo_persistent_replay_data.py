"""Fixed, project-only persistent grid/TRAIN-near query replay.

No network, physics, GT encoder feature, heldout-guided sampling or altered label.
Historical grid pools are the complete saved mode0..lowstep2000 unions. The
TRAIN-near error pool requires separately qualified real source2000 logits.
Corrected entries remain in persistent queues; refresh only appends unseen IDs.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import FunctionType
import numpy as np
from . import michelangelo_lowstep_continuation_data as previous
from . import michelangelo_dynamic_grid_data as dynamic
from . import train_michelangelo_overfit as original
from .qualify_michelangelo_ae import OBJECT_IDS,VENDOR,digest

ROOT=original.EXPERIMENT/'overfit_repair_v1'
SOURCE=ROOT/'michelangelo_lowstep_continuation_v1'
SOURCE_DATA=ROOT/'michelangelo_lowstep_continuation_data_v1'
SCOPE='full_surface_native_ae_persistent_replay_continuation_only'
REVISION='persistent_grid_and_train_near_replay_v1'
START,END,UPDATES,LR=2000,2832,832,1e-5
REFRESH_STEPS=tuple(range(2100,2801,100))
QUALIFICATION_BUDGET=dict(forward=0,encode=4,decode=4,query_geometry=48,
                          backward=0,optimizer_updates=0,grid_scans=0,mesh_extractions=0)
train_indices=FunctionType(original.train_indices.__code__,
    dict(original.train_indices.__globals__,STEPS=END),
    original.train_indices.__name__,original.train_indices.__defaults__)


def recipe():
    return dict(start_step=START,end_step=END,new_updates=UPDATES,updates_per_object=208,lr=LR,
        source='complete lowstep2000 full model and all307 Adam states; no reset',
        train_posterior='mode',loss='original KLNearFar: far + .1 near + .001 KL',
        far_slots=dict(uniform=512,current_fp=128,historical_fp=128,current_fn=128,historical_fn=128),
        near_slots=dict(original_uniform_anchor=512,current_error=256,historical_error=256),
        near_anchor='first512 of original1024 TRAIN_SEED+absolute_step near indices; same pool/distribution, not all old1024 IDs',
        far_anchor='exact original512 eligible uniform VOLUME_SEED+absolute_step',
        historical='shuffle once, persistent cyclic cursor; corrected retained, new IDs append; refresh never resets history',
        current='shuffle at refresh, cycle without replacement until exhausted',
        empty_grid_pool='original corresponding GT class; FP exterior/FN interior',
        empty_near_pool='original complete TRAIN-near pool, never heldout',
        seed='independent NumPy SeedSequence[VOLUME_SEED,object_index,branch_id,refresh_or_cycle]',
        refresh_steps=list(REFRESH_STEPS),endpoint_grid_step=END,qualification_budget=QUALIFICATION_BUDGET,
        coverage='all source union IDs must be drawn at least once through their historical slots; new union coverage reported separately',
        budget_basis='208*128=26624 > maximum source FN union26371; four objects, no automatic extension',
        original_heldout_and_geometric_gates_unchanged=True,checkpoint_selection=False)


def load_sources():
    # Use original fixed data loader, without changing frozen modules/globals.
    old,queries,cases,_,_,_,_=previous.load_sources()
    result=json.loads((SOURCE/'RESULT.json').read_text())
    protocol=json.loads((SOURCE/'PROTOCOL.json').read_text())
    endpoint=json.loads((SOURCE/'ENDPOINT_GRID.json').read_text())
    reload=json.loads((SOURCE/'RELOAD.json').read_text())
    history=json.loads((SOURCE/'HISTORICAL_ERROR_UNION.json').read_text())
    if (result['scope']!=previous.SCOPE or not result['complete']
            or result['total_model_updates']!=START or result['model_updates']!=1000
            or result['checkpoint_reset'] is not False or protocol['train_posterior']!='mode'
            or endpoint['step']!=START or not reload['passed']
            or tuple(r['object_id'] for r in result['cases'])!=OBJECT_IDS
            or tuple(r['object_id'] for r in endpoint['cases'])!=OBJECT_IDS
            or tuple(r['object_id'] for r in history['cases'])!=OBJECT_IDS
            or protocol['data_manifest_sha256']!=digest(SOURCE_DATA/'MANIFEST.json')):
        raise ValueError('Require complete fixed source lowstep2000 and actual source history')
    if digest(Path(__file__).with_name('train_michelangelo_lowstep_continuation.py'))!=protocol['adapter_sha256']:
        raise ValueError('Frozen executed source trainer changed')
    for rel,sha in protocol['sources'].items():
        if digest(VENDOR/rel)!=sha:raise ValueError('Original official source changed')
    if (history['source_lowstep_result_sha256']!=digest(SOURCE/'RESULT.json')
            or history['union_sha256']!=digest(SOURCE/history['union_file'])):
        raise ValueError('Actual saved union/source binding differs')
    for path,sha in history['source_bindings'].items():
        if digest(Path(path))!=sha:raise ValueError('Historical actual error-pool source changed')
    pools=[];histories=[];pool_rows=[];previous_cases=[]
    with np.load(SOURCE/history['union_file'],allow_pickle=False) as union:
        for index,(case,row,mesh) in enumerate(zip(cases,endpoint['cases'],result['cases'],strict=True)):
            path=SOURCE/'endpoint_grid'/row['file']
            if digest(path)!=row['file_sha256']:raise ValueError('Current source pool bytes changed')
            with np.load(path,allow_pickle=False) as z:
                pool=dynamic.error_pools(z['grid_logits'],case)
                if any(not np.array_equal(pool[k],z[k+'_indices']) for k in ('fp','fn')):
                    raise ValueError('Current pools differ from actual source logits/GT labels')
            hist={k:union[case['object_id']+'_'+k+'_indices'].copy() for k in ('fp','fn')}
            for k in ('fp','fn'):
                validate_grid_ids(hist[k],case,k)
                if not np.array_equal(hist[k],np.unique(hist[k])) or not np.isin(pool[k],hist[k]).all():
                    raise ValueError('Historical IDs must be sorted unique and include current errors')
                if len(hist[k])!=history['cases'][index][k+'_union'] or len(hist[k])>208*128:
                    raise ValueError('Actual source history no longer fits the fixed coverage budget')
            pools.append(pool);histories.append(hist)
            pool_rows.append(dict(object_id=case['object_id'],file=str(path.resolve()),sha256=row['file_sha256']))
            path=SOURCE/mesh['output_file']
            if digest(path)!=mesh['case_output_sha256']:raise ValueError('Source complete mesh changed')
            previous_cases.append(dict(object_id=case['object_id'],file=str(path.resolve()),sha256=mesh['case_output_sha256']))
    sources=dict(previous_run=str(SOURCE.resolve()),previous_result_sha256=digest(SOURCE/'RESULT.json'),
        previous_protocol_sha256=digest(SOURCE/'PROTOCOL.json'),previous_reload_sha256=digest(SOURCE/'RELOAD.json'),
        checkpoint_file=result['checkpoint_file'],checkpoint_sha256=result['checkpoint_sha256'],
        endpoint_grid_sha256=digest(SOURCE/'ENDPOINT_GRID.json'),pool_sources=pool_rows,
        history_report_sha256=digest(SOURCE/'HISTORICAL_ERROR_UNION.json'),history_file_sha256=history['union_sha256'],
        source_data_manifest_sha256=digest(SOURCE_DATA/'MANIFEST.json'),
        previous_data_adapter_sha256=digest(Path(previous.__file__)),
        original_trainer_sha256=digest(Path(original.__file__)),dynamic_adapter_sha256=digest(Path(dynamic.__file__)))
    return old,queries,cases,pools,histories,result,sources,previous_cases


def validate_grid_ids(ids,case,kind):
    ids=np.asarray(ids)
    if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer)
            or (len(ids) and (ids.min()<0 or ids.max()>=len(case['labels'])))
            or not case['eligible'][ids].all() or not np.all(case['labels'][ids]==(kind=='fn'))):
        raise ValueError('Invalid eligible grid error-class IDs: '+kind)


def near_error_ids(logits,item):
    logits=np.asarray(logits)
    if logits.shape!=(16384,) or not np.isfinite(logits).all():
        raise ValueError('Require actual complete finite TRAIN-near logits')
    return (16384+np.flatnonzero((logits>=0)!=item['train_labels'][16384:])).astype(np.int32)


def validate_near_ids(ids):
    ids=np.asarray(ids)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or (len(ids) and (ids.min()<16384 or ids.max()>=32768)):
        raise ValueError('Only original TRAIN-near global indices are legal')


class PersistentCycle:
    """Numeric index queue, not a model. Source entries are never reordered on append."""
    def __init__(self,ids,object_index,branch):
        ids=np.asarray(ids,np.int64)
        if ids.ndim!=1 or len(np.unique(ids))!=len(ids):raise ValueError('Unique 1D IDs required')
        self.source=ids.copy();self.members=set(ids.tolist());self.seen=set()
        self.object_index=object_index;self.branch=branch;self.cycle=0;self.cursor=0
        self.order=self._shuffle(ids,0)

    def _shuffle(self,ids,cycle):
        rng=np.random.default_rng(np.random.SeedSequence([dynamic.VOLUME_SEED,self.object_index,self.branch,cycle]))
        return rng.permutation(ids)

    def append(self,ids):
        fresh=np.array(sorted(set(np.asarray(ids,np.int64).tolist())-self.members),np.int64)
        if len(fresh):
            # Appending cannot postpone any unvisited source element.
            self.order=np.concatenate((self.order,fresh));self.members.update(fresh.tolist())

    def take(self,count):
        if not self.members:raise ValueError('Empty queue requires an explicit semantic fallback')
        pieces=[];left=count
        while left:
            if self.cursor==len(self.order):
                self.cycle+=1;self.cursor=0
                self.order=self._shuffle(np.array(sorted(self.members),np.int64),self.cycle)
            size=min(left,len(self.order)-self.cursor)
            selected=self.order[self.cursor:self.cursor+size];pieces.append(selected)
            self.seen.update(selected.tolist());self.cursor+=size;left-=size
        return np.concatenate(pieces).astype(np.int32)

    def coverage(self):
        source=set(self.source.tolist());new=self.members-source
        return dict(source_nodes=len(source),source_seen=len(source&self.seen),
            source_complete=source<=self.seen,new_nodes=len(new),new_seen=len(new&self.seen),
            all_nodes=len(self.members),all_seen=len(self.members&self.seen),
            all_complete=self.members<=self.seen,cycle=self.cycle,cursor=self.cursor)


class ReplaySampler:
    def __init__(self,cases,pools,histories,near_errors):
        if not all(len(x)==4 for x in (cases,pools,histories,near_errors)):raise ValueError('All four cases required')
        self.cases=cases;self.last_step=START;self.refresh_step=START
        self.history=[];self.current=[]
        for i in range(4):
            for k in ('fp','fn'):validate_grid_ids(histories[i][k],cases[i],k)
            validate_near_ids(near_errors[i])
            h={k:PersistentCycle(histories[i][k],i,10+j) for j,k in enumerate(('fp','fn'))}
            h['near']=PersistentCycle(near_errors[i],i,12);self.history.append(h)
        self.refresh(START,pools,near_errors)

    def refresh(self,step,pools,near_errors):
        if step==START and (self.last_step!=START or self.current):
            raise ValueError('Initial refresh cannot reset an existing sampler')
        if step!=START and (step not in REFRESH_STEPS or self.last_step!=step):
            raise ValueError('Refresh must follow the actual fixed absolute update')
        if len(pools)!=4 or len(near_errors)!=4:raise ValueError('No case may be dropped')
        current=[]
        for i in range(4):
            entry={}
            for j,k in enumerate(('fp','fn','near')):
                ids=near_errors[i] if k=='near' else pools[i][k]
                if k=='near':validate_near_ids(ids)
                else:validate_grid_ids(ids,self.cases[i],k)
                self.history[i][k].append(ids)
                entry[k]=PersistentCycle(np.unique(ids),i,100+step*3+j)
            current.append(entry)
        self.current=current;self.refresh_step=step

    def _take(self,i,kind,count,historical,step):
        queue=(self.history if historical else self.current)[i][kind]
        if queue.members:return queue.take(count),False
        if kind=='near':candidates=np.arange(16384,32768)
        else:candidates=self.cases[i]['exterior_indices' if kind=='fp' else 'interior_indices']
        branch=200+10*int(historical)+('fp','fn','near').index(kind)
        rng=np.random.default_rng(np.random.SeedSequence([dynamic.VOLUME_SEED,i,branch,step]))
        return rng.choice(candidates,count,replace=len(candidates)<count).astype(np.int32),True

    def indices(self,step):
        if step!=self.last_step+1 or not START<step<=END:raise ValueError('Exactly832 ordered absolute steps required')
        if step-1 in REFRESH_STEPS and self.refresh_step!=step-1:raise ValueError('Scheduled real refresh is missing')
        i,original_indices=train_indices(step);case=self.cases[i]
        rng=np.random.default_rng(dynamic.VOLUME_SEED+step)
        uniform=rng.choice(case['eligible_indices'],512,replace=False).astype(np.int32)
        far=[uniform];near=[original_indices[1024:1536].astype(np.int32)];fallback={}
        for kind in ('fp','fn','near'):
            for historical in (False,True):
                count=256 if kind=='near' else 128
                ids,used=self._take(i,kind,count,historical,step)
                (near if kind=='near' else far).append(ids)
                fallback[('history_' if historical else 'current_')+kind]=used
        self.last_step=step
        return i,np.concatenate(far),np.concatenate(near),dict(refresh_step=self.refresh_step,fallback=fallback,
            current_pool_counts={k:len(v.members) for k,v in self.current[i].items()},
            historical_coverage={k:v.coverage() for k,v in self.history[i].items()})

    def batch(self,step,queries):
        i,volume,near,info=self.indices(step);case=self.cases[i];source=case['original']
        # Explicit observation allowlist. Evaluation arrays never enter sampler or model input.
        item=dict(input_surface_xyz_vae=source['input_surface_xyz_vae'],input_normals=source['input_normals'],
            train_queries_vae=np.concatenate((queries[volume],source['train_queries_vae'][near])),
            train_labels=np.concatenate((case['labels'][volume],source['train_labels'][near])))
        return i,item,np.arange(2048),volume,near,info

    def coverage(self):
        rows=[dict(object_id=c['object_id'],pools={k:v.coverage() for k,v in h.items()})
              for c,h in zip(self.cases,self.history,strict=True)]
        return dict(step=self.last_step,cases=rows,
            source_union_coverage_complete=all(v.coverage()['source_complete'] for h in self.history for v in h.values()),
            all_observed_union_coverage_complete=all(v.coverage()['all_complete'] for h in self.history for v in h.values()))


def prepare(output):
    old,q,cases,pools,histories,result,sources,previous_cases=load_sources()
    if digest(SOURCE/result['checkpoint_file'])!=result['checkpoint_sha256']:raise ValueError('Full checkpoint bytes differ')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(scope=SCOPE,sampling_revision=REVISION,recipe=recipe(),sources=sources,
        object_ids=list(OBJECT_IDS),original_manifest=old['original_manifest'],cases=old['cases'],
        previous_cases=previous_cases,grid_nodes=len(q),source_history_counts=[{k:len(v) for k,v in h.items()} for h in histories],
        data_adapter_sha256=digest(Path(__file__)),train_near_logits='PENDING separately qualified actual source2000 forward',
        model_forwards=0,model_updates=0)
    original.write_json(output/'MANIFEST.json',manifest);return manifest


def read_data(directory):
    m=json.loads((directory/'MANIFEST.json').read_text())
    old,q,cases,pools,histories,result,sources,previous_cases=load_sources()
    if (m['scope']!=SCOPE or m['sampling_revision']!=REVISION or m['recipe']!=recipe()
            or m['sources']!=sources or m['previous_cases']!=previous_cases
            or m['data_adapter_sha256']!=digest(Path(__file__))
            or m['original_manifest']!=old['original_manifest'] or m['cases']!=old['cases']
            or tuple(m['object_ids'])!=OBJECT_IDS
            or m['source_history_counts']!=[{k:len(v) for k,v in h.items()} for h in histories]):
        raise ValueError('Fixed source/history/data/protocol changed')
    if digest(SOURCE/result['checkpoint_file'])!=result['checkpoint_sha256']:raise ValueError('Full checkpoint bytes differ')
    return m,q,cases,pools,histories,result
