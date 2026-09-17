"""Revision 2 project occupancy sampler, not an official Michelangelo algorithm.

Only the original volume half changes: 8192 uniform full extraction-cube queries
and 8192 queries around previous predicted triangles. All previous face IDs have
equal sampling probability; neither truth errors nor connected components select
faces. Each selected face contributes its center sample and ONE signed normal
offset of half an extraction cell (2.5/128/2 VAE units). Signs alternate between
fixed pair slots, with equal +/- counts in pools and batches. No vertex/face is
deleted, clipped, remeshed or repaired. Degenerate source faces abort preparation.
Offsets outside the cube are retained and counted, not clipped. The center is
the sampled barycentric point on a corrected prediction triangle, not its face
centroid and never a point moved toward ground truth. Revision 1 snapshots and
data remain intact; its +/- pairs could step across thin spurious sheets.

The previous GT-near half, true surface encoder input, and frozen evaluation
queries/labels are copied bit for bit. Per-update GT-near indices also remain
exactly the previous indices. This is a TRAIN-only, post hoc fixed data adapter;
GT occupancy labels are not encoder features. Boundary resampling retains the
existing 1e-6 ABC numerical-label exclusion and records all rejected pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import train_michelangelo_overfit as previous
from .qualify_michelangelo_ae import ABC_TO_VAE, OBJECT_IDS, closest_distance, digest

SCOPE = 'full_surface_native_ae_coverage_overfit_only'
STRATUM = previous.POOL_COUNT//2
HALF_BATCH = previous.BATCH_COUNT//2
HALF_CELL_VAE = 2.5/128/2
OLD_DATA = previous.EXPERIMENT/'overfit_repair_v1/michelangelo_overfit_data_v1'
OLD_RUN = previous.EXPERIMENT/'overfit_repair_v1/michelangelo_native_overfit_v1'
V1_DATA = previous.EXPERIMENT/'overfit_repair_v1/michelangelo_coverage_data_v1'
REVISION = 2
EXTRA_KEYS = {'volume_origin', 'surface_face_indices', 'surface_barycentric',
              'surface_normals', 'surface_offsets_vae'}
OLD_STATISTICS = ('boundary_rejections', 'volume_positive', 'near_positive')


def train_indices(step):
    case, old_indices = previous.train_indices(step)
    rng = np.random.default_rng(previous.TRAIN_SEED+step)
    uniform = rng.choice(STRATUM, HALF_BATCH, replace=False)
    # 256 complete pairs: 128 with negative offset, 128 with positive offset.
    pairs = np.concatenate((2*rng.choice(STRATUM//4, HALF_BATCH//4, replace=False),
                            2*rng.choice(STRATUM//4, HALF_BATCH//4, replace=False)+1))
    surface = STRATUM+(2*pairs[:, None]+np.arange(2)).ravel()
    volume = np.concatenate((uniform, surface))
    return case, np.concatenate((volume, old_indices[previous.BATCH_COUNT:]))


def source_triangles(vertices_vae, faces):
    v, f = np.asarray(vertices_vae), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('Require all finite valid original prediction triangles')
    triangles = v[f].astype(np.float64)
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Degenerate prediction faces: do not silently filter their IDs')
    return triangles, normal/lengths[:, None]


def paired_surface_queries(triangles, normals, rng, pairs, pair_slots=None):
    """Every face ID eligible with equal probability, independently of GT/area."""
    ids = rng.integers(0, len(triangles), size=pairs)
    uv = rng.random((pairs, 2)); root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1-root, root*(1-uv[:, 1]), root*uv[:, 1]))
    points = np.einsum('ni,nij->nj', bary, triangles[ids])
    ids = np.repeat(ids, 2); bary = np.repeat(bary, 2, axis=0)
    slots = np.arange(pairs) if pair_slots is None else np.asarray(pair_slots)
    if slots.shape != (pairs,) or not np.issubdtype(slots.dtype, np.integer):
        raise ValueError('Pair slots determine the fixed balanced sign, not GT')
    offsets = np.column_stack((np.zeros(pairs), np.where(slots % 2 == 0, -HALF_CELL_VAE, HALF_CELL_VAE))).ravel()
    queries = (np.repeat(points, 2, axis=0)+normals[ids]*offsets[:, None]).astype(np.float32)
    return dict(queries=queries, surface_face_indices=ids, surface_barycentric=bary,
                surface_normals=normals[ids], surface_offsets_vae=offsets)


def volume_pool(truth, vertices_vae, faces, index, fixed_uniform, stratum=STRATUM):
    if stratum % 4:
        raise ValueError('Surface stratum must contain balanced center/offset pairs')
    triangles, normals = source_triangles(vertices_vae, faces)
    rng = np.random.default_rng(previous.TRAIN_SEED+index)
    uniform, uniform_labels, uniform_distance = [np.asarray(x).copy() for x in fixed_uniform]
    if uniform.shape != (stratum, 3) or uniform_labels.shape != (stratum,) or uniform_distance.shape != (stratum,):
        raise ValueError('Require the unchanged revision1 full-cube uniform stratum')
    pending = np.arange(stratum//2); rejected_pairs = 0; surface = None
    surface_distance = np.empty(stratum, dtype=np.float64)
    for _ in range(20):
        if not len(pending):
            break
        part = paired_surface_queries(triangles, normals, rng, len(pending), pair_slots=pending)
        if surface is None:
            surface = {k: np.empty((stratum,)+v.shape[1:], dtype=v.dtype) for k, v in part.items()}
        d = closest_distance(truth, part['queries'].astype(np.float64)/ABC_TO_VAE)
        keep_pair = (d.reshape(-1, 2) > previous.BOUNDARY_ABC).all(1)
        keep = np.repeat(keep_pair, 2)
        slots = (2*pending[keep_pair, None]+np.arange(2)).ravel()
        for key in surface:
            surface[key][slots] = part[key][keep]
        surface_distance[slots] = d[keep]
        pending = pending[~keep_pair]; rejected_pairs += int((~keep_pair).sum())
    if len(pending):
        raise RuntimeError('Could not prepare complete boundary-qualified surface pairs')
    queries = np.concatenate((uniform, surface.pop('queries')))
    labels = np.concatenate((uniform_labels, truth.contains(queries[stratum:].astype(np.float64)/ABC_TO_VAE)))
    distance = np.concatenate((uniform_distance, surface_distance))
    surface['volume_origin'] = np.concatenate((np.zeros(stratum, np.uint8), np.ones(stratum, np.uint8)))
    summary = dict(uniform_reused_bit_exact=True, surface_boundary_pair_rejections=rejected_pairs,
                   uniform_positive=int(labels[:stratum].sum()), surface_positive=int(labels[stratum:].sum()),
                   surface_negative=int((~labels[stratum:]).sum()),
                   eligible_prediction_face_count=len(faces), sampled_unique_face_ids=int(np.unique(surface['surface_face_indices']).size),
                   surface_queries_outside_cube=int((np.abs(queries[stratum:]).max(1) > 1.25).sum()))
    return queries, labels, distance, surface, summary


def prepare(args):
    import trimesh
    old_manifest, old_data = previous.read_data(args.old_data)
    v1_manifest = json.loads((V1_DATA/'MANIFEST.json').read_text())
    if digest(V1_DATA/'SOURCE_SNAPSHOT/michelangelo_coverage_data.py') != v1_manifest['data_adapter_sha256']:
        raise ValueError('Revision1 adapter snapshot must match its original data receipt')
    result = json.loads((args.previous_run/'RESULT.json').read_text())
    if (result['scope'] != previous.SCOPE or result['model_updates'] != previous.STEPS
            or result['initial_checkpoint_sha256'] != previous.CHECKPOINT_SHA
            or tuple(r['object_id'] for r in result['cases']) != OBJECT_IDS):
        raise ValueError('Require the complete actual previous fixed1000 native run')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (row, old, pred_row) in enumerate(zip(old_manifest['cases'], old_data, result['cases'])):
        oid = row['object_id']; source = args.previous_run/f'{oid}.npz'
        if digest(source) != pred_row['case_output_sha256']:
            raise ValueError('Previous actual prediction source changed')
        v1_row = v1_manifest['cases'][index]
        v1_path = V1_DATA/v1_row['file']
        if v1_row['object_id'] != oid or digest(v1_path) != v1_row['file_sha256']:
            raise ValueError('Revision1 uniform source must remain the fixed actual pool')
        with np.load(v1_path, allow_pickle=False) as z:
            uniform = tuple(z[key][:STRATUM] for key in ('train_queries_vae', 'train_labels', 'train_surface_distance_abc'))
        truth_v = np.load(row['source_vertices'], allow_pickle=False)
        truth_f = np.load(row['source_faces'], allow_pickle=False)
        truth = trimesh.Trimesh(truth_v, truth_f, process=False)
        if not truth.is_watertight or not truth.is_winding_consistent or truth.volume <= 0:
            raise ValueError('Ambiguous full truth occupancy; do not repair')
        with np.load(source, allow_pickle=False) as z:
            if (not np.array_equal(z['truth_vertices_canonical'], truth_v)
                    or not np.array_equal(z['truth_faces'], truth_f)):
                raise ValueError('Previous prediction uses a different truth mesh')
            q, label, d, provenance, summary = volume_pool(truth, z['reconstruction_vertices_vae'],
                                                          z['reconstruction_faces'], index, uniform)
        arrays = {key: value.copy() for key, value in old.items()}
        arrays['train_queries_vae'][:previous.POOL_COUNT] = q
        arrays['train_labels'][:previous.POOL_COUNT] = label
        arrays['train_surface_distance_abc'][:previous.POOL_COUNT] = d
        arrays.update(provenance)
        void = np.dtype((np.void, 3*np.dtype(np.float32).itemsize))
        train = np.ascontiguousarray(arrays['train_queries_vae']).view(void).ravel()
        evaluation = np.ascontiguousarray(arrays['eval_queries_vae'], dtype=np.float32).view(void).ravel()
        if np.intersect1d(train, evaluation).size:
            raise ValueError('Coverage training query overlaps frozen evaluation query')
        path = args.output/f'{oid}.npz'; np.savez_compressed(path, **arrays)
        updated = dict(row, file=path.name, file_sha256=digest(path), old_pool_sha256=row['file_sha256'],
                       previous_prediction_file=str(source.resolve()), previous_prediction_sha256=digest(source),
                       coverage_sampling=summary)
        updated['previous_pool_statistics'] = {key: updated.pop(key) for key in OLD_STATISTICS}
        rows.append(updated)
        print('MICHELANGELO_COVERAGE_POOL', oid, summary, flush=True)
    manifest = dict(old_manifest, scope=SCOPE, sampling_revision=REVISION, cases=rows, volume_range=[-1.25, 1.25],
                    old_data=str(args.old_data.resolve()), old_manifest_sha256=digest(args.old_data/'MANIFEST.json'),
                    previous_result=str((args.previous_run/'RESULT.json').resolve()),
                    previous_result_sha256=digest(args.previous_run/'RESULT.json'),
                    adapter='project fixed coverage sampler, not an official method claim',
                    single_changed_factor='volume-query distribution only',
                    volume_strata=dict(uniform=STRATUM, all_prediction_faces=STRATUM),
                    batch_volume_strata=dict(uniform=HALF_BATCH, all_prediction_faces=HALF_BATCH),
                    batch_surface_composition=dict(centers=256, negative_offsets=128, positive_offsets=128),
                    paired_normal_offset_vae=HALF_CELL_VAE, near_pool_and_per_step_indices_bit_exact=True,
                    surface_pair_scheme='center_then_signed_offset; equal +/- slots, complete balanced pairs per batch',
                    previous_coverage_data=str(V1_DATA.resolve()), previous_coverage_manifest_sha256=digest(V1_DATA/'MANIFEST.json'),
                    uniform_reused_bit_exact=True,
                    encoder_inputs_and_evaluation_bit_exact=True, triangle_selection='equal face-ID probability, no GT/component filter',
                    pinned_previous_trainer_sha256=digest(Path(previous.__file__)),
                    data_adapter_sha256=digest(Path(__file__)))
    previous.write_json(args.output/'MANIFEST.json', manifest)
    return 0


def read_data(directory):
    manifest = json.loads((directory/'MANIFEST.json').read_text())
    if (manifest['scope'] != SCOPE or tuple(manifest['object_ids']) != OBJECT_IDS
            or manifest.get('sampling_revision') != REVISION
            or manifest['seed'] != previous.TRAIN_SEED
            or manifest['pool_per_kind'] != previous.POOL_COUNT
            or manifest['volume_strata'] != dict(uniform=STRATUM, all_prediction_faces=STRATUM)
            or manifest['batch_volume_strata'] != dict(uniform=HALF_BATCH, all_prediction_faces=HALF_BATCH)
            or manifest['paired_normal_offset_vae'] != HALF_CELL_VAE
            or digest(Path(previous.__file__)) != manifest['pinned_previous_trainer_sha256']):
        raise ValueError('Fixed sampling-only protocol or previous source changed')
    old_path = Path(manifest['old_data'])
    if (digest(old_path/'MANIFEST.json') != manifest['old_manifest_sha256']
            or digest(Path(manifest['previous_result'])) != manifest['previous_result_sha256']):
        raise ValueError('Previous fixed run/data changed')
    old_manifest, old_data = previous.read_data(old_path)
    v1_path = Path(manifest['previous_coverage_data'])
    if digest(v1_path/'MANIFEST.json') != manifest['previous_coverage_manifest_sha256']:
        raise ValueError('Revision1 uniform pool receipt changed')
    v1_manifest = json.loads((v1_path/'MANIFEST.json').read_text())
    if tuple(r['object_id'] for r in manifest['cases']) != OBJECT_IDS:
        raise ValueError('All four fixed cases required')
    data = []
    for row, old_row, old, v1_row in zip(manifest['cases'], old_manifest['cases'], old_data, v1_manifest['cases']):
        path = directory/row['file']; source = Path(row['previous_prediction_file'])
        if digest(path) != row['file_sha256'] or digest(source) != row['previous_prediction_sha256']:
            raise ValueError('Prepared pool/previous actual predicted mesh changed')
        for key in old_row:
            current = row['previous_pool_statistics'][key] if key in OLD_STATISTICS else row[key]
            if key not in ('file', 'file_sha256') and current != old_row[key]:
                raise ValueError('Original source/metadata unexpectedly changed: '+key)
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != previous.POOL_KEYS | EXTRA_KEYS:
                raise ValueError('Unexpected coverage input/provenance keys')
            item = {k: z[k] for k in z.files}
        if digest(v1_path/v1_row['file']) != v1_row['file_sha256']:
            raise ValueError('Revision1 uniform file changed')
        with np.load(v1_path/v1_row['file'], allow_pickle=False) as z:
            for key in ('train_queries_vae', 'train_labels', 'train_surface_distance_abc'):
                if not np.array_equal(item[key][:STRATUM], z[key][:STRATUM]):
                    raise ValueError('Revision1 uniform query/label/distance must be bit exact')
        for key in previous.POOL_KEYS:
            if item[key].shape != old[key].shape or item[key].dtype != old[key].dtype or not np.isfinite(item[key]).all():
                raise ValueError('Original schema/dtype or finite values changed')
            start = previous.POOL_COUNT if key.startswith('train_') else 0
            if not np.array_equal(item[key][start:], old[key][start:]):
                raise ValueError('Only the volume half may change: '+key)
        if (not np.isin(item['train_labels'], [0, 1]).all()
                or item['train_surface_distance_abc'].min() <= previous.BOUNDARY_ABC
                or not np.array_equal(item['volume_origin'], np.repeat(np.array([0, 1], np.uint8), STRATUM))
                or np.abs(item['train_queries_vae'][:STRATUM]).max() > 1.25):
            raise ValueError('Invalid volume labels/domain/provenance')
        with np.load(source, allow_pickle=False) as z:
            tri, normal = source_triangles(z['reconstruction_vertices_vae'], z['reconstruction_faces'])
        ids, bary = item['surface_face_indices'], item['surface_barycentric']
        offsets = item['surface_offsets_vae']
        if (ids.shape != (STRATUM,) or bary.shape != (STRATUM, 3) or item['surface_normals'].shape != (STRATUM, 3)
                or not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0 or ids.max() >= len(tri)
                or not np.isfinite(bary).all() or (bary < 0).any()
                or not np.allclose(bary.sum(1), 1, atol=1e-14, rtol=0)
                or not np.array_equal(offsets, np.tile([0., -HALF_CELL_VAE, 0., HALF_CELL_VAE], STRATUM//4))
                or not np.array_equal(ids[::2], ids[1::2])
                or not np.array_equal(bary[::2], bary[1::2])
                or not np.array_equal(item['surface_normals'], normal[ids])):
            raise ValueError('Invalid full-face +/- normal source provenance')
        replay = (np.einsum('ni,nij->nj', bary, tri[ids])+normal[ids]*offsets[:, None]).astype(np.float32)
        if not np.array_equal(replay, item['train_queries_vae'][STRATUM:previous.POOL_COUNT]):
            raise ValueError('Actual triangle/barycentric/normal replay must be bit exact')
        data.append(item)
    return manifest, data
