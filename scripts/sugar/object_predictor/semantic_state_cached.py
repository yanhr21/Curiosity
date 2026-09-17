"""Independent r1 contract: identical semantic experiment, frozen input bytes."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from . import semantic_state_training as base
from . import semantic_state_data as data
from . import semantic_state_encoded_cache as encoded

STUDY='semantic_state_training_v1_r1'
SCHEMA=7
CONDITIONS=base.CONDITIONS
IDENTITY_KEYS=base.IDENTITY_KEYS
LR_BASE=base.LR_BASE
ARTIFACTS=base.ARTIFACTS
NEW_SOURCES=tuple('scripts.sugar.object_predictor.'+n for n in (
    'semantic_state_encoded_cache','build_semantic_state_encoded_cache','semantic_state_cached','train_semantic_state_cached'))
qualify_condition=base.qualify_condition
prepare_batch=base.prepare_batch
fixed_batches=base.fixed_batches
apply_learning_rates=base.apply_learning_rates
parameter_update_report=base.parameter_update_report
evaluate=base.evaluate


def actual_source_bindings():
    result=base.actual_source_bindings()
    for name in NEW_SOURCES:
        path=Path(importlib.util.find_spec(name).origin).resolve()
        result[name]=dict(path=str(path),sha256=data.sha(path))
    return result


def source_protocol(source,qualification,cache):
    p=base.source_protocol(source,qualification)
    cache=Path(cache).resolve();cp,cr=encoded.verify_cache(cache)
    if cp['original_model_source']!=str(Path(source).resolve()) or cp['original_qualification']!=str(Path(qualification).resolve()):
        raise ValueError('Frozen cache source/qualification differs')
    if cp['source_data']!=p['source_data'] or cp['data']!=p['data'] or cp['model_source_bindings']!=p['source_bindings']:
        raise ValueError('Frozen cache belongs to a different original dataset/checkpoint')
    return dict(p,study=STUDY,schema=SCHEMA,cache=str(cache),prepared_sources=actual_source_bindings(),
        cache_bindings={name:dict(path=str(cache/name),sha256=data.sha(cache/name)) for name in
            ('PROTOCOL.json','RESULT.json','fit.npz','fit.json','development_interpolation.npz','development_interpolation.json')},
        input_revision='r1 loads one complete1904 byte-frozen encoding; original failed schema6 and all28sources remain unchanged. No re-encoding, rounding or hash tolerance.',
        cache_label_provenance='Actual raw-source/controller/COM labels validated at generation and bound to original recordings. Loaded observation contact masks, actual-arm conflicts and global denominators checked again.',
        previous_attempt='semantic_state_training_v1 stopped at CPU preflight before all model forwards or optimizer steps; only137 coordinate float32 components differ across CPU hosts, maximum11.92nm. This is reproducibility plumbing, not evidence for centimeter prediction error.')


def validate_protocol(root):
    p=json.loads((Path(root)/'PROTOCOL.json').read_text())
    if p!=source_protocol(p['source_endpoint'],p['qualification'],p['cache']):raise ValueError('Prepared cached protocol changed')
    return p


def arm_protocol(protocol,condition):return base.arm_protocol(protocol,condition)


def load_dataset(protocol,role):return encoded.CachedSemanticDataset(protocol['cache'],role)


def cache_observations(dataset,protocol,*,verify_qualified_fit=False):
    # No encoder, coordinate operation, PCA or summary reconstruction here.
    hashes=[encoded.row_hash(r) for r in dataset.rows]
    if hashes!=dataset.cached_input_hashes.tolist():raise ValueError('Loaded model input bytes changed')
    if verify_qualified_fit:
        with np.load(Path(protocol['qualification'])/'OBSERVATION_QUALIFICATION.npz') as z:
            checks=dict(inputs=np.array_equal(hashes,z['input_sha256']),
                summary=np.array_equal(np.stack([r['_observed_summary'] for r in dataset.rows]),z['summary']),
                targets=np.array_equal(np.stack([r['target'] for r in dataset.rows]),z['target']),
                masks=np.array_equal([[r['supervision']['state_precision_eligible'],r['supervision']['mass_available']] for r in dataset.rows],z['masks']),
                episode=np.array_equal([r['metadata']['episode'] for r in dataset.rows],z['episode']),
                frame=np.array_equal([r['metadata']['frame'] for r in dataset.rows],z['frame']))
        if not all(checks.values()):raise ValueError('Cached464 differs from original qualification')
    else:checks=dict(all_loaded_row_hashes_exact=True)
    return dict(dataset.cache_readback,checks=checks,loaded_summary_bytes=True,encoder_calls=0,
        rounding_applied=False,hash_tolerance_relaxed=False)


def begin_artifacts(root):
    root=Path(root)
    manifest=dict(schema=SCHEMA,kind=STUDY,complete=False,sources=actual_source_bindings(),
        protocol=dict(path='PROTOCOL.json',sha256=data.sha(root/'PROTOCOL.json')),artifacts={})
    base.original.save_json(root/'ARTIFACTS.json',manifest);return manifest


def verify_artifacts(root,*,require_complete=True,manifest=None):
    root=Path(root);m=manifest or json.loads((root/'ARTIFACTS.json').read_text())
    if m.get('schema')!=SCHEMA or m.get('kind')!=STUDY or (require_complete and m.get('complete') is not True):
        raise ValueError('Require completed cached semantic arm schema7')
    expected=set(base.original.SOURCE_MODULES)|set(base.EXTRA_SOURCES)|set(NEW_SOURCES)
    if set(m['sources'])!=expected:raise ValueError('Unexpected cached training source set')
    for name,b in m['sources'].items():
        if data.sha(b['path'])!=b['sha256']:raise ValueError('Cached training source changed: '+name)
    if m['protocol']!=dict(path='PROTOCOL.json',sha256=data.sha(root/'PROTOCOL.json')):raise ValueError('Arm protocol changed')
    p=json.loads((root/'PROTOCOL.json').read_text())
    for name,b in p['cache_bindings'].items():
        if data.sha(b['path'])!=b['sha256']:raise ValueError('Cached input binding changed: '+name)
    if require_complete:
        if set(m['artifacts'])!=set(ARTIFACTS):raise ValueError('Incomplete cached semantic artifacts')
        for name,b in m['artifacts'].items():
            if b!=dict(path=name,sha256=data.sha(root/name)):raise ValueError('Cached semantic artifact changed: '+name)
    return m


def finish_artifacts(root,manifest):
    verify_artifacts(root,require_complete=False,manifest=manifest)
    final=dict(manifest,complete=True,artifacts={n:dict(path=n,sha256=data.sha(Path(root)/n)) for n in ARTIFACTS})
    base.original.save_json(Path(root)/'ARTIFACTS.json',final);return final
