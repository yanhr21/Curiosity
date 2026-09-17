"""Frozen, already-observed TRAIN query sets for the full official shape AE.

This module only prepares/reads arrays. It never evaluates a network, changes
GT labels, selects queries using held-out errors, or modifies a source run.
Uniform anchors are every unique actual uniform-slot grid ID from the previous
832 updates. History includes the complete saved source union and every saved
refresh through the fixed endpoint2832, including corrected errors.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from . import michelangelo_persistent_replay_data as previous
from . import train_michelangelo_overfit as original
from .qualify_michelangelo_ae import OBJECT_IDS, VENDOR, digest

ROOT = previous.ROOT
SOURCE = ROOT / 'michelangelo_persistent_replay_v1'
SOURCE_DATA = ROOT / 'michelangelo_persistent_replay_data_v1'
SCOPE = 'full_surface_native_ae_static_global_continuation_only'
REVISION = 'frozen_train_global_objective_v1'
START, END, UPDATES, LR, CHUNK = 2832, 2848, 16, 1e-5, 4096
BUCKETS = ('uniform_grid_indices', 'history_fp_indices', 'history_fn_indices')
WEIGHTS = (.5, .25, .25)
EXPECTED_COUNTS = ((103972,18726,11286,16384), (103895,16232,14181,16384),
                   (103910,4186,5839,16384), (103927,11990,26372,16384))


def recipe():
    return dict(start_step=START,end_step=END,new_optimizer_updates=UPDATES,
        objects_per_global_update=4,object_backward_calls_per_update=4,
        lr=LR,optimizer='source complete AdamW, no moment reset',posterior='mode',
        model='complete original released architecture and all307 source2832 parameters',
        objective='mean_over4_objects(.5*K(U,N)+.25*K(historyFP,N)+.25*K(historyFN,N))',
        K='original KLNearFar, near_weight=.1, kl_weight=.001, num_near_samples=16384',
        denominators='Each original criterion computes the full exact bucket mean; same near/ KL shared across weights summing1; each object weight1/4.',
        uniform='All unique actual uniform512-slot grid IDs from source832 updates; no quality filter.',
        history='Complete original source union plus all saved2100..2832 FP/FN, including corrected entries.',
        near='Complete original16384 TRAIN-near pool; no current-error oversampling.',
        distribution_change='Frozen bucket means replace time-varying current/history mixtures; uniform duplicate draws become unique-node equal weights; near returns to complete original pool.',
        same_coordinate_in_multiple_buckets='Retained in each bucket to preserve explicit mixture weights.',
        refreshes=[],query_chunk=CHUNK,bucket_counts=[list(c) for c in EXPECTED_COUNTS],
        queries_per_global_step=590052,query_geometry_calls_per_global_step=150,
        heldout_used_for_query_selection=False,automatic_budget_extension=False,
        checkpoint_selection=False,mesh_filtering=False)


def array_sha(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def frozen_buckets(case_index, trace, source_history, refreshes):
    """No model score, heldout value, mesh component or quality filter accepted."""
    steps = np.asarray(trace['steps'])
    cases = np.asarray(trace['case_indices'])
    volume = np.asarray(trace['volume_grid_indices'])
    if (not np.array_equal(steps, np.arange(2001, 2833))
            or not np.array_equal(cases, (steps - 1) % 4)
            or volume.shape != (832, 1024)):
        raise ValueError('Require the complete actual fixed832 TRAIN sampling trace')
    result = {'uniform_grid_indices': np.unique(volume[cases == case_index, :512]).astype(np.int32)}
    for kind in ('fp', 'fn'):
        values = [np.asarray(source_history[kind])]
        values.extend(np.asarray(row[kind + '_indices']) for row in refreshes)
        result['history_' + kind + '_indices'] = np.unique(np.concatenate(values)).astype(np.int32)
    result['near_indices'] = np.arange(16384, 32768, dtype=np.int32)
    return result


def validate_buckets(buckets, case):
    if set(buckets) != set(BUCKETS) | {'near_indices'}:
        raise ValueError('Unexpected frozen query fields')
    for key in BUCKETS:
        ids = buckets[key]
        if (ids.dtype != np.int32 or ids.ndim != 1 or not len(ids)
                or not np.array_equal(ids, np.unique(ids))
                or ids.min() < 0 or ids.max() >= len(case['labels'])
                or not case['eligible'][ids].all()):
            raise ValueError('Invalid frozen TRAIN grid IDs: ' + key)
    if (case['labels'][buckets['history_fp_indices']].any()
            or not case['labels'][buckets['history_fn_indices']].all()):
        raise ValueError('Historical FP/FN classes no longer match actual GT labels')
    if not np.array_equal(buckets['near_indices'], np.arange(16384, 32768, dtype=np.int32)):
        raise ValueError('Complete original TRAIN-near pool is required')


def counts(buckets):
    sizes = {key: len(value) for key, value in buckets.items()}
    return dict(bucket_sizes=sizes, query_points=sum(sizes.values()),
                query_geometry_calls=sum((n + CHUNK - 1) // CHUNK for n in sizes.values()))


def load_sources():
    old, queries, cases, _, histories, _ = previous.read_data(SOURCE_DATA)
    result = json.loads((SOURCE / 'RESULT.json').read_text())
    protocol = json.loads((SOURCE / 'PROTOCOL.json').read_text())
    reload = json.loads((SOURCE / 'RELOAD.json').read_text())
    coverage = json.loads((SOURCE / 'SAMPLING_COVERAGE.json').read_text())
    if (result.get('scope') != previous.SCOPE or not result.get('complete')
            or result.get('total_model_updates') != START or result.get('model_updates') != 832
            or result.get('checkpoint_reset') is not False or not reload.get('passed')
            or protocol['train_posterior'] != 'mode'
            or tuple(r['object_id'] for r in result['cases']) != OBJECT_IDS
            or tuple(r['object_id'] for r in coverage['cases']) != OBJECT_IDS
            or coverage['step'] != START):
        raise ValueError('Require the complete actual four-case source2832')
    source_files = [SOURCE / n for n in ('RESULT.json', 'PROTOCOL.json', 'RELOAD.json',
                    'SAMPLING_TRACE.npz', 'SAMPLING_COVERAGE.json', 'FULL_ADAM_READBACK.json')]
    if digest(SOURCE / result['checkpoint_file']) != result['checkpoint_sha256']:
        raise ValueError('Source2832 complete checkpoint changed')
    if digest(Path(__file__).with_name('train_michelangelo_persistent_replay.py')) != protocol['adapter_sha256']:
        raise ValueError('Executed source trainer changed')
    for path, sha in protocol['sources'].items():
        if digest(VENDOR / path) != sha:
            raise ValueError('Original official model source changed')
    with np.load(SOURCE / 'SAMPLING_TRACE.npz', allow_pickle=False) as z:
        trace = {k: z[k] for k in z.files}
    by_object = [[] for _ in OBJECT_IDS]
    for step in (*range(2100, 2801, 100), START):
        directory = SOURCE / ('endpoint_grid' if step == START else 'grid_refresh') / f'step_{step}'
        receipt = directory / 'RESULT.json'; record = json.loads(receipt.read_text())
        source_files.append(receipt)
        if record['step'] != step or tuple(r['object_id'] for r in record['cases']) != OBJECT_IDS:
            raise ValueError('Saved history identity/order changed')
        for i, row in enumerate(record['cases']):
            path = directory / (row['object_id'] + '.npz')
            if digest(path) != row['file_sha256']:
                raise ValueError('Actual saved history bytes changed')
            source_files.append(path)
            with np.load(path, allow_pickle=False) as z:
                fields = {kind + '_indices': z[kind + '_indices'] for kind in ('fp', 'fn')}
            for kind in ('fp', 'fn'):
                previous.validate_grid_ids(fields[kind + '_indices'], cases[i], kind)
            by_object[i].append(fields)
    buckets = []; previous_cases = []
    for i, (case, row) in enumerate(zip(cases, result['cases'], strict=True)):
        values = frozen_buckets(i, trace, histories[i], by_object[i])
        validate_buckets(values, case)
        for kind in ('fp', 'fn'):
            if len(values['history_' + kind + '_indices']) != coverage['cases'][i]['pools'][kind]['all_nodes']:
                raise ValueError('Complete actual history union count changed')
        buckets.append(values)
        path = SOURCE / row['output_file']
        if digest(path) != row['case_output_sha256']:
            raise ValueError('Actual source raw/corrected mesh artifact changed')
        source_files.append(path)
        previous_cases.append(dict(object_id=case['object_id'], file=str(path.resolve()),
                                   sha256=row['case_output_sha256']))
    sources = dict(previous_run=str(SOURCE.resolve()), checkpoint_file=result['checkpoint_file'],
        checkpoint_sha256=result['checkpoint_sha256'], previous_result_sha256=digest(SOURCE / 'RESULT.json'),
        previous_data_manifest_sha256=digest(SOURCE_DATA / 'MANIFEST.json'),
        previous_adapter_sha256=digest(Path(previous.__file__)),
        files={str(path.resolve()): digest(path) for path in source_files})
    return old, queries, cases, buckets, result, sources, previous_cases


def prepare(output):
    output=Path(output)
    old,queries,cases,buckets,result,sources,previous_cases=load_sources()
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for i,(case,values) in enumerate(zip(cases,buckets,strict=True)):
        if tuple(len(values[k]) for k in (*BUCKETS,'near_indices'))!=EXPECTED_COUNTS[i]:
            raise ValueError('Actual frozen bucket inventory changed')
        path=output/(case['object_id']+'.npz');np.savez_compressed(path,**values)
        item=case['original']
        records.append(dict(object_id=case['object_id'],file=path.name,file_sha256=digest(path),
            **counts(values),bucket_array_sha256={k:array_sha(v) for k,v in values.items()},
            encoder_xyz_sha256=array_sha(item['input_surface_xyz_vae']),
            encoder_normals_sha256=array_sha(item['input_normals']),
            train_near_queries_sha256=array_sha(item['train_queries_vae'][16384:]),
            train_near_labels_sha256=array_sha(item['train_labels'][16384:]),
            fixed_bucket_labels_sha256={k:array_sha(case['labels'][values[k]]) for k in BUCKETS}))
    manifest=dict(scope=SCOPE,sampling_revision=REVISION,recipe=recipe(),sources=sources,
        object_ids=list(OBJECT_IDS),cases=records,previous_cases=previous_cases,
        original_manifest=old['original_manifest'],data_adapter_sha256=digest(Path(__file__)),
        model_forwards=0,optimizer_updates=0,physics_controls=0,
        encoder_input='Only unchanged original native full4096 surfaceXYZ and normals. Query coordinates are decoder supervision queries; GT occupancy never enters encoder.',
        history_labels='Actual full-grid GT labels only; original ambiguous/evaluation-overlap exclusions retained.',
        heldout='Original independent32768 preserved; never used to define new query buckets.')
    original.write_json(output/'MANIFEST.json',manifest)
    return manifest


def read_data(directory):
    directory=Path(directory);m=json.loads((directory/'MANIFEST.json').read_text())
    old,queries,cases,buckets,result,sources,previous_cases=load_sources()
    if (m['scope']!=SCOPE or m['sampling_revision']!=REVISION or m['recipe']!=recipe()
            or m['sources']!=sources or m['previous_cases']!=previous_cases
            or m['original_manifest']!=old['original_manifest']
            or m['data_adapter_sha256']!=digest(Path(__file__))
            or tuple(m['object_ids'])!=OBJECT_IDS
            or tuple(r['object_id'] for r in m['cases'])!=OBJECT_IDS):
        raise ValueError('Frozen TRAIN/global-objective source binding changed')
    for i,(row,case,expected) in enumerate(zip(m['cases'],cases,buckets,strict=True)):
        path=directory/row['file']
        if digest(path)!=row['file_sha256']:raise ValueError('Prepared fixed bucket file changed')
        with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        validate_buckets(values,case)
        if (any(not np.array_equal(values[k],expected[k]) for k in expected)
                or any(array_sha(values[k])!=row['bucket_array_sha256'][k] for k in values)
                or counts(values)['query_points']!=row['query_points']
                or counts(values)['query_geometry_calls']!=row['query_geometry_calls']
                or counts(values)['bucket_sizes']!=row['bucket_sizes']
                or tuple(len(values[k]) for k in (*BUCKETS,'near_indices'))!=EXPECTED_COUNTS[i]):
            raise ValueError('Actual all-history/uniform TRAIN replay changed')
        item=case['original']
        actual=dict(encoder_xyz_sha256=array_sha(item['input_surface_xyz_vae']),
            encoder_normals_sha256=array_sha(item['input_normals']),
            train_near_queries_sha256=array_sha(item['train_queries_vae'][16384:]),
            train_near_labels_sha256=array_sha(item['train_labels'][16384:]),
            fixed_bucket_labels_sha256={k:array_sha(case['labels'][values[k]]) for k in BUCKETS})
        if any(row[k]!=v for k,v in actual.items()):raise ValueError('Original observation/label bytes changed')
    return m,queries,cases,buckets,result
