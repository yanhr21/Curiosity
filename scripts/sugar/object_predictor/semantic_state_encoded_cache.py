"""Exact complete semantic input packs; loading never re-encodes or rounds."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from . import semantic_state_data as data

CACHE_STUDY='semantic_state_encoded_cache_v1'
ROLES=('fit','development_interpolation')
SCALARS=('mass_available','mass_uncertain','mass_status','state_precision_eligible','state_contact_history_frames')


def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def write_json(path,value):Path(path).write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')


def row_hash(row):
    h=hashlib.sha256()
    for key in ('coord','grid_coord','feat'):
        for value in row['inputs'][key]:h.update(np.ascontiguousarray(value).tobytes())
    h.update(row['observed_left_hand_world_height_m'].tobytes())
    return h.hexdigest()


def recording_bindings(root):
    root=Path(root)
    records=json.loads((root/'COLLECTION_RESULT.json').read_text())['records']
    paths=[root/'PROTOCOL.json',root/'COLLECTION_RESULT.json']
    for row in records:
        if row['episode'] not in data.FIXED_EPISODES:continue
        source=Path(row['source'])
        paths.extend(source/name for name in ('PROTOCOL.json','RESULT.json',f"episode_{row['episode']}.npz",'contact_surface.npz'))
    return {str(p.resolve()):data.sha(p) for p in paths}


def save_role(root,dataset):
    role=dataset.semantic_role
    if role not in ROLES:raise ValueError('Unknown cache role')
    rows=dataset.rows;offset=[0]
    arrays={k:np.concatenate([v for row in rows for v in row['inputs'][k]]) for k in ('coord','grid_coord','feat')}
    for row in rows:
        for v in row['inputs']['coord']:offset.append(offset[-1]+len(v))
    arrays.update(offset=np.asarray(offset,dtype=np.int64),
        target=np.stack([r['target'] for r in rows]),
        height=np.stack([r['observed_left_hand_world_height_m'] for r in rows]),
        height_timestamp_s=np.stack([r['observed_height_timestamp_s'] for r in rows]),
        history_indices=np.asarray([r['metadata']['history_indices'] for r in rows],np.int64),
        episode=np.asarray([r['metadata']['episode'] for r in rows],np.int64),
        frame=np.asarray([r['metadata']['frame'] for r in rows],np.int64),
        timestamp_s=np.asarray([r['metadata']['timestamp_s'] for r in rows],np.float64),
        summary=np.stack([r['_observed_summary'] for r in rows]),
        input_sha256=np.asarray([row_hash(r) for r in rows]))
    for key in SCALARS:arrays['supervision__'+key]=np.asarray([r['supervision'][key] for r in rows])
    physics=tuple(rows[0]['supervision']['physics'])
    for key in physics:arrays['physics__'+key]=np.stack([r['supervision']['physics'][key] for r in rows])
    attributes={key:getattr(dataset,key) for key in ('collection_result','collection_records','qualified',
        'failure_source_validation','controlled_source_validation','summary_input_provenance','conflicts')}
    meta=dict(role=role,rows=len(rows),expected_frames=list(dataset.expected_frames),physics_fields=list(physics),
        attributes=attributes,metadata=[r['metadata'] for r in rows],evaluation_only=[r['evaluation_only'] for r in rows],
        array_schema={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k,v in arrays.items()})
    np.savez(Path(root)/(role+'.npz'),**arrays)
    write_json(Path(root)/(role+'.json'),meta)
    return meta


def validate_arrays(arrays,meta):
    schema=meta['array_schema']
    if set(arrays)!=set(schema):raise ValueError('Cache array set changed')
    for key,value in arrays.items():
        expected=schema[key]
        if value.dtype.str!=expected['dtype'] or list(value.shape)!=expected['shape']:
            raise ValueError('Cache dtype/shape changed: '+key)
        if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()!=expected['sha256']:
            raise ValueError('Cache exact array bytes changed: '+key)
        if value.dtype.kind!='U' and not np.isfinite(value).all():raise ValueError('Nonfinite cached array')
    n=meta['rows'];offset=arrays['offset']
    if offset.dtype!=np.int64 or offset.shape!=(n*32+1,) or offset[0]!=0 or (np.diff(offset)<=0).any():
        raise ValueError('Cache H32 frame offsets malformed')
    points=int(offset[-1])
    for key,width,dtype in (('coord',3,np.float32),('grid_coord',3,np.int32),('feat',20,np.float32)):
        if arrays[key].shape!=(points,width) or arrays[key].dtype!=dtype:raise ValueError('Original point interface changed')
    if arrays['height'].shape!=(n,32,1) or arrays['summary'].shape!=(n,32,11) or arrays['target'].shape!=(n,13):
        raise ValueError('Malformed complete cached row')
    if not np.array_equal(arrays['summary'][:,:,10],arrays['height'][:,:,0]/.2):raise ValueError('Historical public height changed')


class CachedSemanticDataset:
    """Reconstruct only array views/labels; no raw encoder or model invocation."""
    def __init__(self,root,role):
        if role not in ROLES:raise ValueError('Unknown cache role')
        root=Path(root);meta=json.loads((root/(role+'.json')).read_text())
        with np.load(root/(role+'.npz'),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        validate_arrays(arrays,meta)
        frames=data.FIT_FRAMES if role=='fit' else data.EVALUATION_FRAMES
        expected=[(e,f) for e in data.FIXED_EPISODES for f in frames]
        if meta['role']!=role or meta['expected_frames']!=list(frames) or list(zip(arrays['episode'],arrays['frame']))!=expected:
            raise ValueError('Cached row identity/order changed')
        for name,value in meta['attributes'].items():setattr(self,name,value)
        self.collection_records={int(k):v for k,v in self.collection_records.items()}
        self.semantic_role=role;self.expected_frames=frames;self.supervision_profile=data.PROFILE
        self.clock_role='fit' if role=='fit' else 'same_trajectory_interpolation'
        self.rows=[];offset=arrays['offset']
        for i in range(meta['rows']):
            inputs={key:[arrays[key][offset[j]:offset[j+1]] for j in range(i*32,(i+1)*32)] for key in ('coord','grid_coord','feat')}
            metadata=meta['metadata'][i]
            if (metadata['episode'],metadata['frame'])!=expected[i] or not np.array_equal(metadata['history_indices'],arrays['history_indices'][i]):
                raise ValueError('Cached metadata/history differs from actual arrays')
            if metadata['timestamp_s']!=arrays['timestamp_s'][i]:raise ValueError('Cached timestamp metadata differs')
            supervision={key:arrays['supervision__'+key][i] for key in SCALARS}
            supervision['physics']={key:arrays['physics__'+key][i] for key in meta['physics_fields']}
            row=dict(inputs=inputs,target=arrays['target'][i],supervision=supervision,metadata=metadata,
                evaluation_only=meta['evaluation_only'][i],observed_left_hand_world_height_m=arrays['height'][i],
                observed_height_timestamp_s=arrays['height_timestamp_s'][i],_observed_summary=arrays['summary'][i])
            if row_hash(row)!=arrays['input_sha256'][i]:raise ValueError('Reconstructed model input bytes changed')
            self.rows.append(row)
        self.cached_input_hashes=arrays['input_sha256'].copy()
        self.cache_readback=dict(passed=True,role=role,rows=len(self),points=int(offset[-1]),
            all_array_bytes_exact=True,all_row_input_bytes_exact=True,encoder_calls=0,rounding_applied=False,
            source_qualification='Physical/source/GT labels qualified during cache generation and retained by raw-source hashes; observed contact/conflicts recomputed on these exact inputs.')

    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]


def verify_cache(root):
    root=Path(root);p=json.loads((root/'PROTOCOL.json').read_text());r=json.loads((root/'RESULT.json').read_text())
    if p.get('study')!=CACHE_STUDY or r.get('passed') is not True or r.get('rows')!=1904:
        raise ValueError('Require complete fixed1904 encoded cache')
    for name,binding in r['artifacts'].items():
        if data.sha(root/name)!=binding['sha256']:raise ValueError('Encoded cache artifact changed: '+name)
    if set(r['artifacts'])!={'fit.npz','fit.json','development_interpolation.npz','development_interpolation.json'}:
        raise ValueError('Incomplete encoded cache artifacts')
    if r['protocol_sha256']!=data.sha(root/'PROTOCOL.json'):raise ValueError('Cache protocol changed')
    for path,digest in p['source_bindings'].items():
        if data.sha(path)!=digest:raise ValueError('Cache source changed: '+path)
    for path,digest in p['recording_bindings'].items():
        if data.sha(path)!=digest:raise ValueError('Bound physical recording changed: '+path)
    for path,digest in p['upstream_artifacts'].items():
        if data.sha(path)!=digest:raise ValueError('Bound upstream checkpoint/protocol changed: '+path)
    return p,r
