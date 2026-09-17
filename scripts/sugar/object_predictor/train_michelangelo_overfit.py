"""Fixed four-mesh geometry-only finetuning of the FULL released Michelangelo AE.

No new model or loss implementation: original AlignedShapeLatentPerceiver and
KLNearFar are called directly. This does not continue CLIP alignment and is not
tactile prediction. CPU prepare freezes independent occupancy pools; GPU run is
root-queued only, behind require_active_resource/fd9. Four full gradient checks
perform ZERO optimizer updates, then model/Adam/seed reset before exactly 1000
updates (250 per fixed TRAIN object). No continuation, sweep or early stopping.

Original 4096 surface points/normals, 1024 volume + 1024 near queries per step,
KLNearFar near=.1/KL=.001 and AdamW .9/.99, eps1e-6, wd=.01 are retained.
The fixed lr1e-4 follows the paper; omitting its long pretraining warmup for this
1000-step overfit is an explicit engineering choice. Original posterior sampling
is used in training; deterministic posterior mean in evaluation/teacher checks.

Training labels use actual float32 query coordinates and complete watertight
meshes. A fixed 1e-6 ABC boundary exclusion avoids ray-label ambiguity only in
new training pools. Frozen evaluation queries/labels (including one previously
observed boundary disagreement) remain unchanged and never sampled for training.
Every 100 steps evaluates all fixed held-out query sets; endpoint evaluation uses
full actual meshes, original gates, exact checkpoint reload, and later rendering.
Meshes are saved before any geometric metric so failed evaluation retains output.
Both original extractor coordinates and analytically corrected coordinates are
saved. Corrected coordinates are the explicitly revised primary mesh evaluation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from .qualify_michelangelo_ae import (
    ABC_TO_VAE, CHECKPOINT, CHECKPOINT_BYTES, CHECKPOINT_SHA, CONFIG, DATASET,
    EXPERIMENT, INPUT_POINTS, OBJECT_IDS, SEED, VENDOR, closest_distance,
    dependency_check, digest, evaluate_mesh, load_official_model, numerical_gate,
    read_cases, sample_surface,
)
from .michelangelo_geometry_coordinates import correct_extracted_coordinates

BASELINE = EXPERIMENT/'overfit_repair_v1/michelangelo_native_ae_v1_r1'
POOL_COUNT = 16384
BATCH_COUNT = 1024
TRAIN_SEED = 481729
BOUNDARY_ABC = 1e-6
STEPS = 1000
SCOPE = 'full_surface_native_ae_overfit_only'
PARAMETER_GROUPS = ('encoder', 'pre_kl', 'post_kl', 'transformer', 'geo_decoder')
POOL_KEYS = {'input_surface_xyz_vae', 'input_normals', 'train_queries_vae',
             'train_labels', 'train_surface_distance_abc', 'eval_queries_vae',
             'eval_labels'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def occupancy_metrics(labels, logits):
    labels, logits = np.asarray(labels), np.asarray(logits)
    if labels.shape != logits.shape or not np.isfinite(logits).all():
        raise ValueError('Invalid saved occupancy output')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Occupancy labels must be binary')
    labels = labels.astype(bool); pred = logits >= 0
    union = np.logical_or(labels, pred).sum()
    return dict(occupancy_iou=float(np.logical_and(labels, pred).sum()/union) if union else 0.,
                occupancy_accuracy=float((pred == labels).mean()),
                false_negative=int((labels & ~pred).sum()),
                false_positive=int((~labels & pred).sum()))


def train_indices(step, count=POOL_COUNT):
    if not 1 <= step <= STEPS or count < BATCH_COUNT:
        raise ValueError('Step/count outside the fixed protocol')
    rng = np.random.default_rng(TRAIN_SEED+step)
    return ((step-1) % 4,
            np.concatenate((rng.choice(count, BATCH_COUNT, replace=False),
                            count+rng.choice(count, BATCH_COUNT, replace=False))))


def training_pool(case, index, count=POOL_COUNT):
    """CPU exact-mesh labels, volume first/near second; no evaluation inputs."""
    import trimesh
    v, f = case['vertices'], case['faces']
    mesh = trimesh.Trimesh(v, f, process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Require positive-volume closed winding-consistent truth; no repair')
    rng = np.random.default_rng(TRAIN_SEED+index)
    pieces, distances, rejected = [], [], []
    for kind in ('volume', 'near'):
        accepted, accepted_dist = [], []; have = 0; discarded = 0
        for attempt in range(20):
            needed = count-have
            if needed == 0:
                break
            if kind == 'volume':
                query = rng.uniform(-1.1, 1.1, (needed, 3))
            else:
                points, _, _, _ = sample_surface(v, f, needed, TRAIN_SEED+1000+100*index+attempt)
                query = points*ABC_TO_VAE+rng.normal(0, .02, points.shape)
            query = query.astype(np.float32)
            distance = closest_distance(mesh, query.astype(np.float64)/ABC_TO_VAE)
            keep = distance > BOUNDARY_ABC
            accepted.append(query[keep]); accepted_dist.append(distance[keep])
            have += int(keep.sum()); discarded += int((~keep).sum())
        if have != count:
            raise RuntimeError('Could not produce the declared fixed pool; do not silently shorten')
        pieces.append(np.concatenate(accepted)); distances.append(np.concatenate(accepted_dist))
        rejected.append(discarded)
    queries = np.concatenate(pieces)
    labels = mesh.contains(queries.astype(np.float64)/ABC_TO_VAE)
    return queries, labels, np.concatenate(distances), dict(
        boundary_rejections=dict(zip(('volume', 'near'), rejected)),
        volume_positive=int(labels[:count].sum()), near_positive=int(labels[count:].sum()))


def prepare(args):
    cases = read_cases(args.dataset)
    baseline_result = json.loads((args.baseline/'RESULT.json').read_text())
    if (tuple(r['object_id'] for r in baseline_result['cases']) != OBJECT_IDS
            or baseline_result['checkpoint_sha256'] != CHECKPOINT_SHA
            or baseline_result['model_updates'] != 0):
        raise ValueError('Require the fixed original released-weight four-case baseline')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, case in enumerate(cases):
        oid = case['object_id']; source = args.baseline/f'{oid}.npz'
        with np.load(source, allow_pickle=False) as z:
            if (not np.array_equal(z['truth_vertices_canonical'], case['vertices'])
                    or not np.array_equal(z['truth_faces'], case['faces'])):
                raise ValueError('Baseline truth must be the same complete original asset')
            pc, normal, _, _ = sample_surface(case['vertices'], case['faces'], INPUT_POINTS, SEED+index)
            if (not np.array_equal((pc*ABC_TO_VAE).astype(np.float32), z['input_surface_xyz_vae'])
                    or not np.array_equal(normal, z['input_normals'])):
                raise ValueError('AE input surface changed from the frozen baseline')
            arrays = dict(input_surface_xyz_vae=z['input_surface_xyz_vae'],
                          input_normals=z['input_normals'].astype(np.float32),
                          eval_queries_vae=z['occupancy_queries_vae'], eval_labels=z['occupancy_labels'])
        q, labels, distance, summary = training_pool(case, index)
        arrays.update(train_queries_vae=q, train_labels=labels, train_surface_distance_abc=distance)
        # Compare coordinates as actually consumed (float32), not just RNG seeds.
        key_dtype = np.dtype((np.void, 3*np.dtype(np.float32).itemsize))
        train_keys = np.ascontiguousarray(q).view(key_dtype).ravel()
        eval_keys = np.ascontiguousarray(arrays['eval_queries_vae'], dtype=np.float32).view(key_dtype).ravel()
        if np.intersect1d(train_keys, eval_keys).size:
            raise ValueError('Training pool contains a held-out evaluation query')
        path = args.output/f'{oid}.npz'; np.savez_compressed(path, **arrays)
        row = {k: v for k, v in case.items() if k not in ('vertices', 'faces')}
        row.update(file=path.name, file_sha256=digest(path), baseline_file=str(source.resolve()),
                   baseline_sha256=digest(source), **summary)
        rows.append(row)
        print('MICHELANGELO_POOL', oid, summary, flush=True)
    manifest = dict(scope=SCOPE, seed=TRAIN_SEED, object_ids=list(OBJECT_IDS),
                    pool_per_kind=POOL_COUNT, volume_range=[-1.1, 1.1], near_sigma_vae=.02,
                    boundary_exclusion_abc=BOUNDARY_ABC, abc_to_vae_scale=ABC_TO_VAE,
                    evaluation_unchanged=True, evaluation_used_for_training=False,
                    baseline_result=str((args.baseline/'RESULT.json').resolve()),
                    baseline_result_sha256=digest(args.baseline/'RESULT.json'),
                    preserved_eval_boundary_note='15737: saved/replayed contains differ at one query, '
                    'distance 4.080725923e-7 ABC; saved frozen labels remain unchanged.',
                    cases=rows, model_forwards=0, physics_controls=0)
    write_json(args.output/'MANIFEST.json', manifest)
    return 0


def read_data(directory):
    manifest = json.loads((directory/'MANIFEST.json').read_text())
    if (manifest['scope'] != SCOPE or tuple(manifest['object_ids']) != OBJECT_IDS
            or manifest['pool_per_kind'] != POOL_COUNT or manifest['seed'] != TRAIN_SEED
            or manifest['abc_to_vae_scale'] != ABC_TO_VAE
            or manifest['evaluation_used_for_training']):
        raise ValueError('Fixed preparation protocol changed')
    if digest(Path(manifest['baseline_result'])) != manifest['baseline_result_sha256']:
        raise ValueError('Frozen baseline RESULT changed')
    data = []
    if tuple(row['object_id'] for row in manifest['cases']) != OBJECT_IDS:
        raise ValueError('All four cases are required in original order')
    for row in manifest['cases']:
        path = directory/row['file']
        if (digest(path) != row['file_sha256']
                or digest(Path(row['baseline_file'])) != row['baseline_sha256']):
            raise ValueError('Fixed pool/baseline file changed')
        for key in ('vertices', 'faces'):
            if digest(Path(row['source_'+key])) != row['source_'+key+'_sha256']:
                raise ValueError('Original complete mesh changed')
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != POOL_KEYS:
                raise ValueError('Unexpected pool schema')
            a = {key: z[key] for key in z.files}
        expected = dict(input_surface_xyz_vae=(4096, 3), input_normals=(4096, 3),
                        train_queries_vae=(2*POOL_COUNT, 3), train_labels=(2*POOL_COUNT,),
                        train_surface_distance_abc=(2*POOL_COUNT,),
                        eval_queries_vae=(32768, 3), eval_labels=(32768,))
        if any(a[k].shape != shape or not np.isfinite(a[k]).all() for k, shape in expected.items()):
            raise ValueError('Invalid fixed pool dimensions/finite values')
        if (not np.isin(a['train_labels'], [0, 1]).all()
                or not np.isin(a['eval_labels'], [0, 1]).all()
                or a['train_surface_distance_abc'].min() <= BOUNDARY_ABC):
            raise ValueError('Invalid occupancy labels or boundary-qualified pool')
        data.append(a)
    return manifest, data


def original_loss():
    sys.path.insert(0, str(VENDOR))
    module = importlib.import_module('michelangelo.models.tsal.loss')
    if Path(module.__file__).resolve() != (VENDOR/'michelangelo/models/tsal/loss.py').resolve():
        raise ValueError('Loss must come from the unchanged official repository')
    return module.KLNearFar(near_weight=.1, kl_weight=.001, num_near_samples=BATCH_COUNT)


def optimizer_for(model):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .99), eps=1e-6, weight_decay=.01)


def tensor(array, device):
    import torch
    return torch.as_tensor(array, dtype=torch.float32, device=device)[None]


def gradients(model):
    """Require all real shape parameter tensors connected and finite; group signal."""
    import torch
    groups = {k: dict(tensors=0, parameters=0, squared_norm=0.) for k in PARAMETER_GROUPS}
    zeros = []
    for name, parameter in model.named_parameters():
        group = name.split('.')[0]
        if group not in groups or not parameter.requires_grad or parameter.grad is None:
            raise RuntimeError('Disconnected/frozen/unexpected shape parameter: '+name)
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite shape gradient: '+name)
        norm = float(parameter.grad.double().square().sum())
        groups[group]['tensors'] += 1; groups[group]['parameters'] += parameter.numel()
        groups[group]['squared_norm'] += norm
        if norm == 0:
            zeros.append(name)
    if any(not g['tensors'] or g['squared_norm'] <= 0 for g in groups.values()):
        raise RuntimeError('An entire original AE branch has zero gradient')
    return dict(groups=groups, zero_gradient_tensors=zeros, passed=True)


def batch_loss(model, criterion, item, indices, device):
    # Correct original Aligned.forward tuple; plain ShapeAsLatentPLModule expects
    # a different tuple and is deliberately not used for this geometry-only run.
    _, logits, posterior = model(tensor(item['input_surface_xyz_vae'], device),
                                 tensor(item['input_normals'], device),
                                 tensor(item['train_queries_vae'][indices], device),
                                 sample_posterior=True)
    return criterion(posterior, logits, tensor(item['train_labels'][indices], device))


def evaluate_queries(model, criterion, data, device):
    import torch
    rows = []
    model.eval()
    with torch.no_grad():
        for oid, item in zip(OBJECT_IDS, data):
            _, z, posterior = model.encode(tensor(item['input_surface_xyz_vae'], device),
                                           tensor(item['input_normals'], device), sample_posterior=False)
            decoded = model.decode(z)
            logits = torch.cat([model.query_geometry(tensor(item['eval_queries_vae'][i:i+4096], device), decoded)
                                for i in range(0, len(item['eval_labels']), 4096)], dim=1)
            # Evaluation contains 16384 volume + 16384 near: original class with
            # its equal-halves option, rather than training's explicit 1024.
            saved_count = criterion.num_near_samples; criterion.num_near_samples = None
            try:
                loss, log = criterion(posterior, logits, tensor(item['eval_labels'], device), split='eval')
            finally:
                criterion.num_near_samples = saved_count
            values = logits[0].cpu().numpy()
            rows.append(dict(object_id=oid, **occupancy_metrics(item['eval_labels'], values),
                             loss=float(loss), loss_components={k: float(v) for k, v in log.items()},
                             logits_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                             latent_sha256=hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest()))
    return rows


def endpoint_meshes(model, data, manifest, output, device):
    import torch
    from michelangelo.models.tsal.inference_utils import extract_geometry
    records = []
    model.eval()
    with torch.no_grad():
        for index, (row, item) in enumerate(zip(manifest['cases'], data)):
            oid = row['object_id']; record = dict(row)
            try:
                v = np.load(row['source_vertices'], allow_pickle=False)
                f = np.load(row['source_faces'], allow_pickle=False)
                embed, z, posterior = model.encode(tensor(item['input_surface_xyz_vae'], device),
                                                   tensor(item['input_normals'], device), sample_posterior=False)
                _, repeat, _ = model.encode(tensor(item['input_surface_xyz_vae'], device),
                                            tensor(item['input_normals'], device), sample_posterior=False)
                if z.shape != (1, 256, 64) or not torch.equal(z, repeat):
                    raise RuntimeError('Invalid/non-deterministic original latent')
                decoded = model.decode(z)
                geometry = lambda queries: model.query_geometry(queries, decoded)
                meshes, valid = extract_geometry(geometry, device, batch_size=1, octree_depth=7, num_chunks=10000)
                if not bool(valid[0]):
                    raise RuntimeError('No actual extractable endpoint surface')
                raw, faces = meshes[0]; corrected = correct_extracted_coordinates(raw)
                arrays = dict(truth_vertices_canonical=v, truth_faces=f,
                              input_surface_xyz_vae=item['input_surface_xyz_vae'], input_normals=item['input_normals'],
                              shape_latent=z[0].cpu().numpy(), aligned_shape_embed=embed[0].cpu().numpy(),
                              posterior_mean=posterior.mean[0].cpu().numpy(),
                              posterior_logvar=posterior.logvar[0].cpu().numpy(),
                              reconstruction_vertices_canonical=corrected/ABC_TO_VAE,
                              reconstruction_vertices_vae=corrected,
                              reconstruction_vertices_vae_raw_official=raw,
                              reconstruction_vertices_canonical_raw_official=raw/ABC_TO_VAE,
                              reconstruction_faces=faces, abc_to_vae_scale=np.float64(ABC_TO_VAE))
                path = output/f'{oid}.npz'; np.savez_compressed(path, **arrays)
                record.update(output_file=path.name, raw_reconstruction_saved=True)
                logits = torch.cat([geometry(tensor(item['eval_queries_vae'][i:i+4096], device))
                                    for i in range(0, len(item['eval_labels']), 4096)], dim=1)[0].cpu().numpy()
                arrays.update(occupancy_queries_vae=item['eval_queries_vae'], occupancy_labels=item['eval_labels'],
                              occupancy_logits=logits)
                np.savez_compressed(path, **arrays)
                occupancy = occupancy_metrics(item['eval_labels'], logits)
                metrics = dict(evaluate_mesh(v, f, corrected/ABC_TO_VAE, faces, SEED+300+index), **occupancy)
                raw_metrics = dict(evaluate_mesh(v, f, raw/ABC_TO_VAE, faces, SEED+300+index), **occupancy)
                record.update(complete=True, metrics=metrics, raw_official_metrics=raw_metrics,
                              checks=numerical_gate(metrics), repeat_latent_max_error=0.)
            except Exception as error:
                record.update(complete=False, checks=dict(passed=False),
                              error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            if (output/f'{oid}.npz').is_file():
                record['case_output_sha256'] = digest(output/f'{oid}.npz')
            records.append(record)
            write_json(output/'PROGRESS.json', records)
            print('MICHELANGELO_OVERFIT_MESH', oid, record.get('metrics'), record['checks'], flush=True)
    return records


def audit_saved_checkpoint(checkpoint, model, optimizer):
    """Read the actual saved endpoint and compare every parameter/Adam moment."""
    import torch
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['step'] != STEPS or saved['initial_checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Wrong endpoint step/source')
    live = model.state_dict()
    if saved['model'].keys() != live.keys():
        raise ValueError('Incomplete saved full model')
    for name in live:
        if not torch.equal(live[name].detach().cpu(), saved['model'][name]):
            raise ValueError('Saved model differs: '+name)
    live_opt = optimizer.state_dict()
    if (saved['optimizer']['param_groups'] != live_opt['param_groups']
            or saved['optimizer']['state'].keys() != live_opt['state'].keys()):
        raise ValueError('Incomplete Adam groups/state')
    names = saved['parameter_names']
    if names != [n for n, _ in model.named_parameters()]:
        raise ValueError('Parameter-name binding changed')
    ids = [i for g in live_opt['param_groups'] for i in g['params']]
    if len(ids) != len(names) or len(saved['optimizer']['state']) != len(names):
        raise ValueError('Adam does not cover all original parameters')
    elements = 0
    for key, (name, parameter) in zip(ids, model.named_parameters()):
        state = saved['optimizer']['state'][key]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != STEPS:
            raise ValueError('Wrong Adam update count: '+name)
        for field in state:
            value = state[field]
            if (not torch.isfinite(value).all()
                    or not torch.equal(value, live_opt['state'][key][field].detach().cpu())):
                raise ValueError('Invalid/different Adam state: '+name+'/'+field)
        if state['exp_avg'].shape != parameter.shape or state['exp_avg_sq'].shape != parameter.shape:
            raise ValueError('Wrong Adam moment dimensions')
        elements += parameter.numel()
    return dict(passed=True, full_model_tensors=len(live), adam_parameter_tensors=len(names),
                adam_parameter_elements=elements, every_adam_step=STEPS), saved


def run(args):
    from .retained_execution import require_active_resource
    resource = require_active_resource()
    import torch
    if args.checkpoint.stat().st_size != CHECKPOINT_BYTES or digest(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Official released checkpoint bytes changed')
    manifest, data = read_data(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(scope=SCOPE, initial_checkpoint_sha256=CHECKPOINT_SHA, object_ids=list(OBJECT_IDS),
                    model_updates=STEPS, updates_per_object=250, gradient_qualification_updates=0,
                    loss='original michelangelo.models.tsal.loss.KLNearFar', near_weight=.1, kl_weight=.001,
                    optimizer=dict(name='AdamW', lr=1e-4, betas=[.9, .99], eps=1e-6, weight_decay=.01),
                    warmup=None, schedule='fixed lr, no sweep/early stopping',
                    input_points=4096, batch_queries=dict(volume=1024, near=1024), batch_size=1,
                    train_seed=TRAIN_SEED, train_posterior='sample', eval_posterior='mode',
                    abc_to_vae_scale=ABC_TO_VAE, full_model_trainable=True, clip_alignment_continued=False,
                    geometry_coordinates='v1 corrected n/(n-1), original output retained separately',
                    gates_unchanged=dict(cd_x9000_max=.45, fscore_min=.95, area_ratio=[.7, 1.5],
                                         max_vertex_and_face_probe=.01, occupancy_iou_min=.90),
                    visual_inspection_required=True, data_manifest=manifest, data_manifest_sha256=digest(args.data/'MANIFEST.json'),
                    baseline_result=manifest['baseline_result'], baseline_result_sha256=manifest['baseline_result_sha256'],
                    baseline_cases=[dict(object_id=r['object_id'], file=r['baseline_file'], sha256=r['baseline_sha256'])
                                    for r in manifest['cases']],
                    sources={str(p.relative_to(VENDOR)): digest(p) for p in VENDOR.rglob('*.py')},
                    adapter_sha256=digest(Path(__file__)), resource=resource, physics_controls=0,
                    tactile_model_forwards=0)
    write_json(args.output/'PROTOCOL.json', protocol)
    device = torch.device('cuda'); torch.manual_seed(TRAIN_SEED)
    model, loading = load_official_model(args.checkpoint, device)
    model.requires_grad_(True).train(); criterion = original_loss()
    qualification = []
    for index, oid in enumerate(OBJECT_IDS):
        try:
            torch.manual_seed(TRAIN_SEED+index+1); model.zero_grad(set_to_none=True)
            _, indices = train_indices(index+1)
            loss, components = batch_loss(model, criterion, data[index], indices, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite original qualification loss')
            loss.backward(); grad = gradients(model)
            qualification.append(dict(object_id=oid, loss=float(loss), gradients=grad,
                                      loss_components={k: float(v) for k, v in components.items()}))
        except Exception as error:
            qualification.append(dict(object_id=oid, passed=False,
                                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()))
            write_json(args.output/'GRADIENT_QUALIFICATION.json', dict(passed=False, cases=qualification,
                       optimizer_updates=0, expected_cases=list(OBJECT_IDS)))
            raise
        write_json(args.output/'GRADIENT_QUALIFICATION.json', dict(passed=len(qualification) == 4,
                   cases=qualification, optimizer_updates=0, actual_full_model_backwards=len(qualification)))
        print('MICHELANGELO_FULL_GRADIENT', oid, float(loss), grad['passed'], flush=True)
    write_json(args.output/'GRADIENT_QUALIFICATION.json', dict(passed=True, cases=qualification,
                                                           optimizer_updates=0, actual_full_model_backwards=4))
    # Qualification never creates Adam or steps model parameters. Nonetheless use
    # a fresh complete released model and reset all training randomness explicitly.
    del model, loss, components
    gc.collect(); torch.cuda.empty_cache(); torch.manual_seed(TRAIN_SEED)
    model, reset_loading = load_official_model(args.checkpoint, device)
    if loading != reset_loading:
        raise ValueError('Qualification/reset original model loading differs')
    model.requires_grad_(True); optimizer = optimizer_for(model)
    if optimizer.state:
        raise ValueError('Fresh formal Adam must have no state')
    reports = [dict(step=0, cases=evaluate_queries(model, criterion, data, device))]
    write_json(args.output/'CURVE.json', reports)
    with (args.output/'updates.jsonl').open('x') as log:
        for step in range(1, STEPS+1):
            index, indices = train_indices(step)
            torch.manual_seed(TRAIN_SEED+step); model.train(); optimizer.zero_grad(set_to_none=True)
            loss, components = batch_loss(model, criterion, data[index], indices, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite original training loss')
            loss.backward()
            # Infinity is an explicit finite-gradient check, not gradient clipping.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            if step in (1, STEPS):
                write_json(args.output/f'GRADIENT_STEP_{step}.json', gradients(model))
            optimizer.step()
            log.write(json.dumps(dict(step=step, object_id=OBJECT_IDS[index], seed=TRAIN_SEED+step,
                                      loss=float(loss), gradient_norm=float(norm),
                                      loss_components={k: float(v) for k, v in components.items()}))+'\n')
            log.flush()
            if step % 100 == 0:
                report = dict(step=step, cases=evaluate_queries(model, criterion, data, device))
                reports.append(report); write_json(args.output/'CURVE.json', reports)
                print('MICHELANGELO_OVERFIT_UPDATE', step, report, flush=True)
    checkpoint = args.output/'endpoint.pt'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=STEPS,
                    parameter_names=[n for n, _ in model.named_parameters()],
                    initial_checkpoint_sha256=CHECKPOINT_SHA, protocol=protocol,
                    rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state()), checkpoint)
    audit, saved = audit_saved_checkpoint(checkpoint, model, optimizer)
    write_json(args.output/'FULL_ADAM_READBACK.json', audit)
    before_reload = evaluate_queries(model, criterion, data, device)
    # Actual new original architecture instance; strict complete saved-model load.
    original_class = type(model)
    config = loading['official_config']['params']
    del model, optimizer, loss, components
    gc.collect(); torch.cuda.empty_cache()
    model = original_class(device=None, dtype=None, **config)
    model.load_state_dict(saved['model'], strict=True)
    model = model.to(device).requires_grad_(False).eval()
    del saved; gc.collect()
    after_reload = evaluate_queries(model, criterion, data, device)
    if before_reload != after_reload:
        raise RuntimeError('Actual complete checkpoint reload changed deterministic evaluation')
    write_json(args.output/'RELOAD.json', dict(passed=True, all_four_query_metrics_exact=True,
                                             all_four_logits_and_latents_exact=True, cases=after_reload))
    records = endpoint_meshes(model, data, manifest, args.output, device)
    result = dict(complete=True, scope=SCOPE, cases=records, model_updates=STEPS,
                  updates_per_object=250, initial_checkpoint_sha256=CHECKPOINT_SHA,
                  checkpoint_file=checkpoint.name, checkpoint_sha256=digest(checkpoint),
                  numerical_passed=all(r['checks']['passed'] for r in records),
                  representation_qualified=False, visual_inspection='pending',
                  full_gradient_qualification_passed=True, full_adam_audit=audit,
                  actual_checkpoint_reload_passed=True, loading=loading,
                  physics_controls=0, tactile_model_forwards=0, clip_alignment_continued=False)
    write_json(args.output/'RESULT.json', result)
    return 0 if result['numerical_passed'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'check', 'run'))
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.output is None:
            parser.error('prepare requires a new --output directory')
        return prepare(args)
    if args.data is None:
        parser.error('check/run require --data')
    if args.stage == 'check':
        manifest, data = read_data(args.data)
        deps = dependency_check(); criterion = original_loss()
        import torch
        print(json.dumps(dict(dependencies=deps, objects=manifest['object_ids'],
                              original_loss_class=type(criterion).__module__+'.'+type(criterion).__name__,
                              pool_sizes=[len(d['train_labels']) for d in data],
                              cuda_initialized=torch.cuda.is_initialized(), model_forwards=0), indent=2))
        return 0 if all(deps.values()) and not torch.cuda.is_initialized() else 2
    if args.output is None:
        parser.error('run requires a new --output directory')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
