"""CPU-only GT occupancy at the exact released Michelangelo extraction nodes.

This is a new supervision/diagnostic asset, never encoder input or a training
run. The unchanged official generate_dense_grid_points creates 129^3 float32
queries, endpoint-inclusive [-1.25,1.25]^3, ij indexing and C flatten order.
The public x_vae=6.138*x_ABC transform is inverted before exact-mesh queries.
Strictly outside the GT AABB implies outside occupancy. Remaining nodes use
Trimesh.contains on the complete, unmodified watertight GT mesh. Actual nearest
surface distance identifies <=1e-6 ABC boundary ambiguity; raw ray labels are
retained there, but training_eligible is false. No grid node is moved/dropped.
Only the expanded AABB needs boundary-distance tests: points outside it have
a coordinate separation greater than the boundary tolerance. An exact float32
intersection with the old held-out evaluation set is recorded and also made
ineligible for any later training. No model, CUDA, physics or rendering call.

This grid does not prove arbitrary surface correctness. An MC point outside
GT and farther from its surface than the cell diagonal (.00551 ABC) has all
cell corners outside GT; its zero crossing requires a wrong inside sign. A
similarly deep interior point instead requires a wrong outside sign. Subcell
surface/interpolation errors need not imply wrong corner signs.
Actual model grid logits are NOT produced by this preparation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .qualify_michelangelo_ae import ABC_TO_VAE, EXPERIMENT, OBJECT_IDS, VENDOR, digest

SCOPE = 'exact_extraction_grid_gt_occupancy_cpu_diagnostic_only'
OLD_DATA = EXPERIMENT/'overfit_repair_v1/michelangelo_overfit_data_v1'
OUTPUT = EXPERIMENT/'overfit_repair_v1/michelangelo_extraction_grid_diagnostic_v1'
OFFICIAL_GRID = VENDOR/'michelangelo/graphics/primitives/volume.py'
BOUNDARY_ABC = 1e-6
DEPTH = 7
BOUNDS = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25)
CHUNK = 2048
AUDIT_DIRECTIONS = np.array([[.313, .547, .777], [-.811, .331, .479]], dtype=np.float64)
SEED = 190917


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def official_grid():
    # Import only this unmodified numpy-only official source, not its package
    # __init__, model code, graphics dependencies or any CUDA API.
    spec = importlib.util.spec_from_file_location('michelangelo_official_grid_cpu', OFFICIAL_GRID)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points, shape, length = module.generate_dense_grid_points(
        np.array(BOUNDS[:3]), np.array(BOUNDS[3:]), DEPTH, indexing='ij')
    if points.dtype != np.float32 or shape != [129, 129, 129] or points.shape != (129**3, 3):
        raise ValueError('Official extraction grid shape/dtype changed')
    return points, np.array(shape, dtype=np.int32), length


def row_keys(points):
    points = np.array(points, dtype=np.float32, order='C', copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Require finite float32 3D queries')
    # Signed zero has identical model coordinates; do not miss such overlap.
    points[points == 0] = 0
    return points.view(np.dtype((np.void, 12))).ravel()


def overlap_mask(points, evaluation):
    return np.isin(row_keys(points), row_keys(evaluation))


def label_grid(mesh, queries_vae, evaluation, *, chunk=CHUNK):
    import trimesh
    if (not mesh.is_watertight or not mesh.is_winding_consistent
            or mesh.volume <= 0 or not np.isfinite(mesh.vertices).all()):
        raise ValueError('Require finite, closed, consistent positive-volume GT; do not repair')
    points = np.asarray(queries_vae, dtype=np.float64)/ABC_TO_VAE
    inside_box = np.all((points >= mesh.bounds[0]) & (points <= mesh.bounds[1]), axis=1)
    near_box = np.all((points >= mesh.bounds[0]-BOUNDARY_ABC)
                      & (points <= mesh.bounds[1]+BOUNDARY_ABC), axis=1)
    labels = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(inside_box)
    for start in range(0, len(indices), chunk):
        chosen = indices[start:start+chunk]
        labels[chosen] = mesh.contains(points[chosen])
    candidates = np.flatnonzero(near_box).astype(np.int32)
    distances = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), chunk):
        _, distance, _ = trimesh.proximity.closest_point(mesh, points[candidates[start:start+chunk]])
        distances[start:start+len(distance)] = distance
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError('Invalid exact nearest-surface distance')
    ambiguous = np.zeros(len(points), dtype=bool)
    ambiguous[candidates[distances <= BOUNDARY_ABC]] = True
    overlap = overlap_mask(queries_vae, evaluation)
    return dict(occupancy_labels=labels, boundary_ambiguous=ambiguous,
                strict_aabb_outside=~inside_box, evaluation_overlap=overlap,
                training_eligible=~(ambiguous | overlap),
                boundary_candidate_indices=candidates,
                boundary_candidate_distance_abc=distances)


def alternate_ray_check(mesh, queries_vae, arrays, index):
    """A bounded stratified check with two new ray directions, not new labels."""
    from trimesh.ray.ray_util import contains_points
    rng = np.random.default_rng(SEED+index)
    labels, boundary = arrays['occupancy_labels'], arrays['boundary_ambiguous']
    outside = arrays['strict_aabb_outside']
    strata = [labels & ~boundary, ~labels & ~outside & ~boundary, outside & ~boundary, boundary]
    selected = []
    counts = []
    for mask in strata:
        eligible = np.flatnonzero(mask)
        chosen = rng.choice(eligible, min(256, len(eligible)), replace=False)
        selected.append(chosen); counts.append(len(chosen))
    chosen = np.concatenate(selected).astype(np.int32)
    points = queries_vae[chosen].astype(np.float64)/ABC_TO_VAE
    alternate = np.stack([contains_points(mesh.ray, points, check_direction=d)
                          for d in AUDIT_DIRECTIONS])
    nonboundary = ~boundary[chosen]
    errors = (alternate[:, nonboundary] != labels[chosen][None, nonboundary]).sum(axis=1)
    boundary_disagreement = (alternate[:, ~nonboundary] != labels[chosen][None, ~nonboundary]).sum(axis=1)
    arrays.update(audit_indices=chosen, audit_alternate_labels=alternate,
                  audit_directions=AUDIT_DIRECTIONS.copy())
    return dict(strata=['inside', 'aabb_inside_unoccupied', 'strict_aabb_outside', 'boundary'],
                stratum_counts=counts, checked_nonboundary=int(nonboundary.sum()),
                alternate_direction_mismatches=errors.tolist(),
                boundary_disagreements_retained=boundary_disagreement.tolist(),
                passed=bool(np.all(errors == 0)),
                limitation='Independent directions using Trimesh ray parity, not an independent algorithm')


def prepare(args):
    import trimesh
    manifest_path = args.old_data/'MANIFEST.json'
    old = json.loads(manifest_path.read_text())
    if tuple(row['object_id'] for row in old['cases']) != OBJECT_IDS:
        raise ValueError('Require all four fixed TRAIN objects in original order')
    args.output.mkdir(parents=True, exist_ok=False)
    queries, shape, length = official_grid()
    grid_path = args.output/'grid.npz'
    np.savez_compressed(grid_path, queries_vae=queries, grid_shape=shape,
                        bounds_vae=np.array(BOUNDS), cell_width_vae=length/(shape-1))
    protocol = dict(scope=SCOPE, object_ids=list(OBJECT_IDS), total_grid_nodes=len(queries),
                    grid_shape=shape.tolist(), bounds_vae=BOUNDS, grid_order='ij then C flatten',
                    dtype='float32', endpoint_inclusive=True, abc_to_vae_scale=ABC_TO_VAE,
                    boundary_tolerance_abc=BOUNDARY_ABC,
                    boundary_policy='keep raw labels and every node; mark boundary training-ineligible',
                    eval_policy='old eval remains unchanged; exact float32 overlap training-ineligible',
                    cell_diagonal_abc=float(np.linalg.norm(length/(shape-1))/ABC_TO_VAE),
                    grid_file=grid_path.name, grid_sha256=digest(grid_path),
                    official_grid_source=str(OFFICIAL_GRID), official_grid_sha256=digest(OFFICIAL_GRID),
                    adapter_sha256=digest(Path(__file__)), original_manifest=str(manifest_path.resolve()),
                    original_manifest_sha256=digest(manifest_path),
                    encoder_input=False, model_forwards=0, model_updates=0, physics_controls=0)
    write_json(args.output/'PROTOCOL.json', protocol)
    rows = []
    for index, source in enumerate(old['cases']):
        start = time.perf_counter(); oid = source['object_id']
        row = dict(object_id=oid, complete=False, passed=False)
        try:
            vpath, fpath = Path(source['source_vertices']), Path(source['source_faces'])
            eval_path = args.old_data/source['file']
            if (digest(vpath) != source['source_vertices_sha256']
                    or digest(fpath) != source['source_faces_sha256']
                    or digest(eval_path) != source['file_sha256']):
                raise ValueError('Original truth/evaluation asset changed')
            vertices, faces = np.load(vpath, allow_pickle=False), np.load(fpath, allow_pickle=False)
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                    or faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
                    or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError('Invalid source mesh; no repair or object replacement')
            mesh = trimesh.Trimesh(vertices, faces, process=False)
            with np.load(eval_path, allow_pickle=False) as saved:
                evaluation = saved['eval_queries_vae'].copy()
                evaluation_labels = saved['eval_labels'].copy()
            arrays = label_grid(mesh, queries, evaluation)
            audit = alternate_ray_check(mesh, queries, arrays, index)
            path = args.output/f'{oid}.npz'; np.savez_compressed(path, **arrays)
            # Re-read the original old evaluation arrays after preparation.
            with np.load(eval_path, allow_pickle=False) as saved:
                exact_eval = (np.array_equal(evaluation, saved['eval_queries_vae'])
                              and np.array_equal(evaluation_labels, saved['eval_labels']))
            row.update(complete=True, file=path.name, file_sha256=digest(path),
                       source_vertices=str(vpath), source_faces=str(fpath),
                       source_vertices_sha256=digest(vpath), source_faces_sha256=digest(fpath),
                       old_evaluation_file=str(eval_path.resolve()), old_evaluation_file_sha256=digest(eval_path),
                       vertices=len(vertices), faces=len(faces), watertight=bool(mesh.is_watertight),
                       winding_consistent=bool(mesh.is_winding_consistent), volume_abc=float(mesh.volume),
                       bounds_abc=mesh.bounds.tolist(), occupied_nodes=int(arrays['occupancy_labels'].sum()),
                       contains_evaluated_nodes=int((~arrays['strict_aabb_outside']).sum()),
                       strict_aabb_outside_nodes=int(arrays['strict_aabb_outside'].sum()),
                       boundary_distance_evaluated_nodes=len(arrays['boundary_candidate_indices']),
                       boundary_ambiguous_nodes=int(arrays['boundary_ambiguous'].sum()),
                       boundary_raw_inside_nodes=int((arrays['boundary_ambiguous'] & arrays['occupancy_labels']).sum()),
                       old_evaluation_count=len(evaluation), old_evaluation_unchanged=exact_eval,
                       evaluation_overlap_nodes=int(arrays['evaluation_overlap'].sum()),
                       training_eligible_nodes=int(arrays['training_eligible'].sum()),
                       alternate_ray_check=audit, elapsed_seconds=time.perf_counter()-start,
                       passed=bool(audit['passed'] and exact_eval))
        except Exception:
            row.update(error=traceback.format_exc(), elapsed_seconds=time.perf_counter()-start)
        rows.append(row)
        write_json(args.output/'PROGRESS.json', rows)
        print('EXTRACTION_GRID_CASE', json.dumps(row), flush=True)
    result = dict(protocol, cases=rows, complete=all(r['complete'] for r in rows),
                  passed=all(r['passed'] for r in rows),
                  representation_qualified=False, model_grid_logits_available=False)
    write_json(args.output/'RESULT.json', result)
    return 0 if result['passed'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-data', type=Path, default=OLD_DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    return prepare(args)


if __name__ == '__main__':
    raise SystemExit(main())
