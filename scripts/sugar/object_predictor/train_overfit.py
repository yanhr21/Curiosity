"""Bounded full-Utonia fit and same-trajectory interpolation qualification.

Exactly one 2000-update run, no automatic retries, extensions, or best-checkpoint
selection.  A completed run may fail acceptance.  Nothing here certifies unseen
objects, arbitrary shape reconstruction, material, or policy/demo transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
import torch

from .overfit_data import (FIXED_EPISODES, FIXED_FRAMES, OverfitDataset,
                           collate_overfit, fixed_protocol)
from .overfit_model import (FORCE_TARGET_FIELDS, LOSS_SCALES, LOSS_WEIGHTS,
    FullUtoniaOverfit, force_target, parameter_groups, relative_mass_error,
    rotation_matrices, state_vertices, task_losses, total_loss)
from .retained_execution import require_active_resource
from .overfit_supervision_conflicts import state_supervision_feasibility
from .conditional_overfit_data import (PROFILE as CONDITIONAL_PROFILE, STUDY as CONTROLLED_STUDY,
    ConditionalOverfitDataset, check_collection_scope, collate_conditional,
    precision_eligible_conflict_groups, state_history_evidence)
from .conditional_overfit_training import conditional_task_losses, conditional_acceptance
from .failure_aware_overfit import (PROFILE as FAILURE_AWARE_PROFILE, SCOPE as FAILURE_AWARE_SCOPE,
    FailureAwareOverfitDataset, failure_aware_qualification, failure_aware_acceptance,
    failure_aware_dense_qualification)


FIT_LIMITS = dict(center_cm=1., mesh_nn_cm=1., size_max_relative=.05,
                  available_mass_relative=.05, force_rmse_n=.5,
                  availability_accuracy=1.)
INTERPOLATION_LIMITS = dict(center_mean_cm=1., center_p95_cm=2.,
    mesh_mean_cm=1., mesh_p95_cm=2., available_mass_mean_relative=.05,
    available_mass_p95_relative=.10, force_rmse_n=.5,
    availability_balanced_accuracy=.95, availability_false_confident_fraction=.05)
INTERPOLATION_FRAMES = tuple(frame for frame in range(31, 2400, 25) if frame not in FIXED_FRAMES)
REQUIRED_CONTROLLER_CHECKS = (
    'bilateral', 'clearance', 'rise', 'peak', 'load_target', 'lift_started',
    'hold_frames', 'target_stability', 'frame_load_stability', 'hold_drift',
    'motion_completed', 'actual_complete_hold_frames',
)
SOURCE_MODULES = (
    'scripts.sugar.object_predictor.train_overfit',
    'scripts.sugar.object_predictor.retained_execution',
    'scripts.sugar.object_predictor.overfit_model',
    'scripts.sugar.object_predictor.overfit_data',
    'scripts.sugar.object_predictor.overfit_supervision_conflicts',
    'scripts.sugar.object_predictor.conditional_overfit_data',
    'scripts.sugar.object_predictor.conditional_overfit_training',
    'scripts.sugar.object_predictor.failure_aware_overfit',
    'scripts.sugar.object_predictor.run_canonical_approach_fixture_pilot',
    'scripts.sugar.object_predictor.collect_controlled_fixture_corpus',
    'scripts.sugar.object_predictor.model',
    'scripts.sugar.object_predictor.data',
    'scripts.sugar.object_predictor.surface_data',
    'scripts.sugar.object_predictor.hand_surface_normals',
    'scripts.sugar.object_predictor.inspect_contact_surface',
    'sugar_newton.hand.patches', 'utonia', 'utonia.model',
    'utonia.structure', 'utonia.module', 'utonia.utils',
)
REQUIRED_ARTIFACTS = ('model.pt', 'initial_fit.npz', 'fit_02000.npz',
                    'initial_same_trajectory_interpolation.npz', 'same_trajectory_interpolation.npz')


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def actual_source_bindings():
    """Resolve the actual imported modules, including namespace-package origins."""
    result = {}
    for name in SOURCE_MODULES:
        loaded = sys.modules.get(name)
        if name == 'scripts.sugar.object_predictor.train_overfit':
            origin = __file__
        elif loaded is not None and getattr(loaded, '__file__', None):
            origin = loaded.__file__
        else:
            # Resolve the official submodule without executing package __init__
            # merely to collect provenance; model construction imports it later.
            if name.startswith('utonia.'):
                package = importlib.util.find_spec('utonia')
                spec = None if package is None else PathFinder.find_spec(name, package.submodule_search_locations)
            else:
                spec = importlib.util.find_spec(name)
            if spec is None or spec.origin is None:
                raise ImportError('Cannot resolve required training source: ' + name)
            origin = spec.origin
        path = Path(origin).resolve()
        result[name] = dict(path=str(path), sha256=sha256_file(path))
    return result


def begin_artifacts(root):
    """One manifest records initialization provenance and is finalized once."""
    root = Path(root)
    manifest = dict(schema=1, complete=False, sources=actual_source_bindings(),
                    protocol=dict(path='PROTOCOL.json', sha256=sha256_file(root / 'PROTOCOL.json')),
                    artifacts={})
    save_json(root / 'ARTIFACTS.json', manifest)
    return manifest


def check_manifest_bindings(root, manifest, *, require_complete=True):
    root = Path(root)
    if manifest.get('schema') != 1 or (require_complete and manifest.get('complete') is not True):
        raise ValueError('Artifact manifest is incomplete or unsupported')
    if set(manifest['sources']) != set(SOURCE_MODULES):
        raise ValueError('Artifact manifest has an unexpected training source set')
    for name, binding in manifest['sources'].items():
        if sha256_file(binding['path']) != binding['sha256']:
            raise ValueError('Training source changed: ' + name)
    binding = manifest['protocol']
    if binding['path'] != 'PROTOCOL.json' or sha256_file(root / binding['path']) != binding['sha256']:
        raise ValueError('Training protocol binding differs')
    if require_complete:
        if set(manifest['artifacts']) != set(REQUIRED_ARTIFACTS):
            raise ValueError('Artifact manifest does not bind every checkpoint/prediction file')
        for name, binding in manifest['artifacts'].items():
            if binding['path'] != name or sha256_file(root / name) != binding['sha256']:
                raise ValueError('Training artifact binding differs: ' + name)
    return manifest


def finish_artifacts(root, initial_manifest):
    root = Path(root)
    check_manifest_bindings(root, initial_manifest, require_complete=False)
    # Final actual resolution must agree with the source origins selected before
    # construction, not merely with a hard-coded repository pathname.
    if actual_source_bindings() != initial_manifest['sources']:
        raise ValueError('Actual training source resolution changed during the run')
    manifest = dict(initial_manifest, complete=True,
        artifacts={name: dict(path=name, sha256=sha256_file(root / name)) for name in REQUIRED_ARTIFACTS})
    save_json(root / 'ARTIFACTS.json', manifest)
    return manifest


def verify_artifacts(root):
    """Renderer preflight: reject one changed source/protocol/checkpoint/NPZ."""
    root = Path(root)
    return check_manifest_bindings(root, json.loads((root / 'ARTIFACTS.json').read_text()))


def read_known_mesh(path):
    with np.load(path) as data:
        vertices = np.asarray(data['vertices'], np.float64)
        faces = np.asarray(data['faces'])
    if vertices.shape != (15626, 3) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Use the original complete 15626-vertex CarryBox mesh')
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid complete mesh')
    dimensions = np.ptp(vertices, axis=0)
    if np.any(dimensions <= 0):
        raise ValueError('Degenerate mesh dimensions')
    center = (vertices.min(0) + vertices.max(0)) / 2
    normalized = (vertices - center) / dimensions
    # Public known geometry only; no observed pose, label, or predicted error is
    # involved in selecting the fixed 1024 original vertices used by the loss.
    sampled = np.sort(np.random.default_rng(310117).choice(len(vertices), 1024, replace=False))
    return normalized.astype(np.float32), sampled, dict(vertices=len(vertices), faces=len(faces),
        source_dimensions_m=dimensions.tolist(), sampled_original_vertex_indices=sampled.tolist())


def qualification(dataset):
    if getattr(dataset, 'supervision_profile', 'all_state') == FAILURE_AWARE_PROFILE:
        return failure_aware_qualification(dataset, original_qualification=qualification)
    collection = dataset.collection_result
    conditional = getattr(dataset, 'supervision_profile', 'all_state') == CONDITIONAL_PROFILE
    physical_records = dataset.collection_records
    physical_cases = {}
    for episode in FIXED_EPISODES:
        record = physical_records[episode]
        checks = record.get('controller_checks', {})
        physical_cases[str(episode)] = dict(complete=record.get('complete') is True,
            controller_passed=record.get('controller_passed') is True,
            exact_declared_check_set=set(checks) == set(REQUIRED_CONTROLLER_CHECKS),
            all_twelve_checks_pass=all(checks.get(name) is True for name in REQUIRED_CONTROLLER_CHECKS),
            original_checks=checks)
    coverage = {}
    for episode in FIXED_EPISODES:
        late = [row for row in dataset.rows if row['metadata']['episode'] == episode
                and row['metadata']['frame'] in (1581, 2381)]
        coverage[str(episode)] = dict(late_frames=[r['metadata']['frame'] for r in late],
            available_late_frames=[r['metadata']['frame'] for r in late if r['supervision']['mass_available'] > .5])
    coverage_pass = all(len(value['available_late_frames']) >= 1 for value in coverage.values())
    contradictory_available_mass = []
    for group in dataset.conflicts:
        available_targets = [float(dataset.rows[i]['target'][12]) for i in group['row_indices']
                             if dataset.rows[i]['supervision']['mass_available'] > .5]
        if len(set(available_targets)) > 1:
            contradictory_available_mass.append(group)
    state_groups = precision_eligible_conflict_groups(dataset) if conditional else dataset.conflicts
    state_feasibility = state_supervision_feasibility(dataset.rows, state_groups,
        center_limit_m=FIT_LIMITS['center_cm'] / 100.,
        size_max_relative_limit=FIT_LIMITS['size_max_relative'])
    checks = dict(original_collection_qualification_passed=collection.get('qualification_passed') is True,
                  original_collection_complete=collection.get('complete') is True,
                  exactly_fixed_16_collected_records=tuple(r['episode'] for r in collection['records']) == FIXED_EPISODES,
                  all_16_physical_qualifications_pass=all(all(row[key] for key in ('complete', 'controller_passed',
                      'exact_declared_check_set', 'all_twelve_checks_pass')) for row in physical_cases.values()),
                  all_80_fixed_items_present=len(dataset) == 80,
                  actual_com_labels_present=dataset.qualified,
                  all_16_cases_have_predeclared_late_mass=coverage_pass,
                  no_identical_input_different_available_mass=not contradictory_available_mass,
                  no_provably_incompatible_identical_input_state=state_feasibility['passed'])
    if conditional:
        checks.update(
            explicit_controlled_collection_scope=collection.get('study') == CONTROLLED_STUDY,
            actual_controlled_sources_verified=getattr(dataset, 'controlled_source_validation', {}).get('passed') is True,
            all_state_masks_match_observed_history=all(all(row['supervision'][key] == value
                for key, value in state_history_evidence(row['inputs']).items()) for row in dataset.rows),
            all_16_fixed_late_mass_labels=all(dataset.collection_records[episode].get('late_mass_available') is True
                and any(row['metadata']['episode'] == episode and row['metadata']['frame'] == 2381
                    and row['supervision']['mass_available'] > .5 and row['supervision']['state_precision_eligible'] > .5
                    for row in dataset.rows) for episode in FIXED_EPISODES))
        state_feasibility['applied_scope'] = 'Only observation-derived contact-candidate state precision targets; every raw input conflict remains separately reported'
    else:
        checks['original_collection_scope_only'] = collection.get('study') is None
    return dict(passed=all(checks.values()), checks=checks, coverage=coverage,
        original_collection_qualification_passed=collection.get('qualification_passed'),
        physical_cases=physical_cases,
        contradictory_available_mass=contradictory_available_mass,
        state_supervision_feasibility=state_feasibility,
        mass_available_count=sum(float(r['supervision']['mass_available']) for r in dataset.rows),
        total_rows=len(dataset), input_conflicts=dataset.conflicts,
        state_precision_eligible_count=sum(float(row['supervision']['state_precision_eligible']) for row in dataset.rows) if conditional else len(dataset),
        supervision_profile=CONDITIONAL_PROFILE if conditional else 'all_state',
        scope='Controlled conditional TRAIN-only data qualification; original blind/all-state qualification not claimed' if conditional else 'TRAIN-only data qualification; no model accuracy or generalization result')


def fixed_batches(dataset, seed):
    """Every 20 steps visits all 80 items once, four paired settings per batch."""
    lookup = {(r['metadata']['episode'], r['metadata']['frame']): i for i, r in enumerate(dataset.rows)}
    groups = (tuple(range(5000, 5004)), tuple(range(5008, 5012)),
              tuple(range(5012, 5016)), tuple(range(5020, 5024)))
    batches = [[lookup[episode, frame] for episode in group] for group in groups for frame in FIXED_FRAMES]
    if sorted(i for batch in batches for i in batch) != list(range(80)):
        raise ValueError('Every fixed item must appear once per epoch')
    rng = np.random.default_rng(seed)
    while True:
        for index in rng.permutation(len(batches)):
            yield batches[index]


def device_batch(batch, device):
    result = dict(batch)
    for key in ('inputs', 'physics'):
        result[key] = {name: value.to(device) for name, value in batch[key].items()}
    for key in ('target', 'mass_available', 'mass_uncertain', 'mass_status'):
        result[key] = batch[key].to(device)
    for key in ('state_precision_eligible', 'state_contact_history_frames'):
        if key in batch:
            result[key] = batch[key].to(device)
    return result


def availability_statistics(probability, labels):
    predicted = np.asarray(probability) >= .5
    truth = np.asarray(labels) > .5
    positive, negative = int(truth.sum()), int((~truth).sum())
    tpr = float(predicted[truth].mean()) if positive else None
    tnr = float((~predicted[~truth]).mean()) if negative else None
    return dict(accuracy=float((predicted == truth).mean()), positive_count=positive, negative_count=negative,
                balanced_accuracy=(tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
                false_confident_fraction=float(predicted[~truth].mean()) if negative else None)


def summarise_metrics(arrays):
    available = arrays['mass_available'] > .5
    result = {}
    for key in ('center_cm', 'mesh_nn_cm', 'rotation_deg', 'size_mean_relative', 'size_max_relative', 'mass_relative'):
        values = arrays[key]
        result[key] = dict(mean=float(values.mean()), p95=float(np.quantile(values, .95)), maximum=float(values.max()))
    values = arrays['mass_relative'][available]
    result['available_mass_relative'] = (dict(mean=float(values.mean()), p95=float(np.quantile(values, .95)),
        maximum=float(values.max()), count=len(values)) if len(values) else None)
    result['force_rmse_n'] = float(np.sqrt(np.mean(arrays['force_rmse_n'] ** 2)))
    result['force_max_item_rmse_n'] = float(arrays['force_rmse_n'].max())
    result['availability'] = availability_statistics(arrays['availability_probability'], arrays['mass_available'])
    return result


def acceptance(arrays, role):
    per_case, all_checks = {}, {}
    for episode in FIXED_EPISODES:
        mask = arrays['episode'] == episode
        if not mask.any():
            per_case[str(episode)] = dict(passed=False, reason='missing configuration')
            all_checks[f'{episode}/present'] = False
            continue
        row = {key: value[mask] for key, value in arrays.items()}
        summary = summarise_metrics(row)
        available = row['mass_available'] > .5
        if role == 'fit':
            limits = FIT_LIMITS
            checks = dict(
                all_five_clocks_present=sorted(row['frame'].tolist()) == list(FIXED_FRAMES),
                center=bool(np.all(row['center_cm'] <= limits['center_cm'])),
                mesh=bool(np.all(row['mesh_nn_cm'] <= limits['mesh_nn_cm'])),
                size=bool(np.all(row['size_max_relative'] <= limits['size_max_relative'])),
                mass=bool(available.any() and np.all(row['mass_relative'][available] <= limits['available_mass_relative'])),
                force=bool(np.all(row['force_rmse_n'] <= limits['force_rmse_n'])),
                availability=summary['availability']['accuracy'] >= limits['availability_accuracy'],
            )
        elif role == 'same_trajectory_interpolation':
            limits = INTERPOLATION_LIMITS
            mass = summary['available_mass_relative']
            availability = summary['availability']
            checks = dict(
                all_dense_clocks_present=sorted(row['frame'].tolist()) == list(INTERPOLATION_FRAMES),
                center_mean=summary['center_cm']['mean'] <= limits['center_mean_cm'],
                center_p95=summary['center_cm']['p95'] <= limits['center_p95_cm'],
                mesh_mean=summary['mesh_nn_cm']['mean'] <= limits['mesh_mean_cm'],
                mesh_p95=summary['mesh_nn_cm']['p95'] <= limits['mesh_p95_cm'],
                mass_mean=mass is not None and mass['mean'] <= limits['available_mass_mean_relative'],
                mass_p95=mass is not None and mass['p95'] <= limits['available_mass_p95_relative'],
                force=summary['force_rmse_n'] <= limits['force_rmse_n'],
                availability_balanced=availability['balanced_accuracy'] is not None and availability['balanced_accuracy'] >= limits['availability_balanced_accuracy'],
                availability_false_confident=availability['false_confident_fraction'] is not None and availability['false_confident_fraction'] <= limits['availability_false_confident_fraction'],
            )
        else:
            raise ValueError('Unknown evaluation role')
        per_case[str(episode)] = dict(passed=all(checks.values()), checks=checks, metrics=summary)
        all_checks.update({f'{episode}/{key}': bool(value) for key, value in checks.items()})
    return dict(passed=all(all_checks.values()), checks=all_checks, per_case=per_case,
                all_items=summarise_metrics(arrays), role=role,
                no_contact_note='State errors include every no-contact clock; these are learned priors, not certified tactile observability.')


@torch.no_grad()
def evaluate(model, dataset, full_vertices, device):
    """Independent batch1 evaluation on every declared clock, including failures."""
    model.eval()
    records = []
    vertices = torch.as_tensor(full_vertices, dtype=torch.float32)
    profile = getattr(dataset, 'supervision_profile', 'all_state')
    conditional = profile in (CONDITIONAL_PROFILE, FAILURE_AWARE_PROFILE)
    collator = collate_conditional if conditional else collate_overfit
    for index, row in enumerate(dataset.rows):
        batch = device_batch(collator([row]), device)
        output = model(batch['inputs'])
        prediction, target = output['state'].detach().cpu(), batch['target'].detach().cpu()
        predicted_mesh = state_vertices(prediction, vertices)[0].numpy()
        target_mesh = state_vertices(target, vertices)[0].numpy()
        if not all(np.isfinite(value).all() for value in (predicted_mesh, target_mesh, prediction.numpy())):
            raise FloatingPointError('Nonfinite evaluation geometry/state')
        distances = .5 * (cKDTree(target_mesh).query(predicted_mesh, workers=1)[0].mean()
                          + cKDTree(predicted_mesh).query(target_mesh, workers=1)[0].mean())
        predicted_rotation, target_rotation = rotation_matrices(prediction[:, 3:9]), rotation_matrices(target[:, 3:9])
        relative = predicted_rotation.transpose(1, 2) @ target_rotation
        angle = (((relative.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1).acos() * 180 / torch.pi)
        size_error = torch.expm1(prediction[:, 9:12] - target[:, 9:12]).abs()
        predicted_force = output['force'].detach().cpu().numpy()[0]
        true_force = force_target(batch['physics']).detach().cpu().numpy()[0]
        records.append(dict(episode=row['metadata']['episode'], frame=row['metadata']['frame'],
            timestamp_s=row['metadata']['timestamp_s'], prediction=prediction.numpy()[0], target=target.numpy()[0],
            force_prediction_n=predicted_force, force_target_n=true_force,
            center_cm=float((prediction[:, :3] - target[:, :3]).norm(dim=1)[0] * 100),
            mesh_nn_cm=float(distances * 100), rotation_deg=float(angle[0]),
            size_mean_relative=float(size_error.mean()), size_max_relative=float(size_error.max()),
            mass_relative=float(relative_mass_error(prediction, target)[0]),
            mass_available=float(row['supervision']['mass_available']), mass_status=int(row['supervision']['mass_status']),
            availability_probability=float(output['availability_logit'].sigmoid()[0]),
            contact_probability=float(output['contact_logit'].sigmoid()[0]),
            contact_present=float(row['supervision']['physics']['contact_present'][0]),
            force_rmse_n=float(np.sqrt(np.mean((predicted_force - true_force) ** 2))),
        ))
        if conditional:
            records[-1].update(state_precision_eligible=float(row['supervision']['state_precision_eligible']),
                state_contact_history_frames=int(row['supervision']['state_contact_history_frames']),
                state_evidence_status=int(row['supervision']['state_precision_eligible'] > .5))
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(event='evaluation_progress', role=dataset.clock_role, complete=index + 1, total=len(dataset))), flush=True)
    arrays = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Nonfinite evaluation result')
    conditional_gate = failure_aware_acceptance if profile == FAILURE_AWARE_PROFILE else conditional_acceptance
    report = (conditional_gate(arrays, dataset.clock_role, original_acceptance=acceptance,
        fit_limits=FIT_LIMITS, interpolation_limits=INTERPOLATION_LIMITS, episodes=FIXED_EPISODES)
        if conditional else acceptance(arrays, dataset.clock_role))
    return report, arrays


def parameter_update_report(model, optimizer, initial, steps):
    rows = []; active_backbone_changed = 0; active_backbone_elements = 0
    unused_name = 'predictor.backbone.embedding.mask_token'
    for name, parameter in model.named_parameters():
        current = parameter.detach().cpu()
        difference = current - initial[name]
        changed = int(torch.count_nonzero(difference))
        state = optimizer.state.get(parameter, {})
        clock = int(state['step'].item()) if 'step' in state else None
        expected_active = name != unused_name
        if name.startswith('predictor.backbone.') and expected_active:
            active_backbone_changed += changed
            active_backbone_elements += parameter.numel()
        rows.append(dict(name=name, numel=parameter.numel(), changed_elements=changed,
            max_abs_change=float(difference.abs().max()), adam_step=clock,
            expected_active=expected_active,
            finite_first_moment=bool(torch.isfinite(state['exp_avg']).all()) if state else None,
            finite_second_moment=bool(torch.isfinite(state['exp_avg_sq']).all()) if state else None))
    backbone_modules = {}
    for row in rows:
        if row['name'].startswith('predictor.backbone.') and row['expected_active']:
            module = row['name'].rsplit('.', 1)[0]
            value = backbone_modules.setdefault(module, dict(numel=0, changed_elements=0))
            value['numel'] += row['numel']; value['changed_elements'] += row['changed_elements']
    checks = dict(all_parameters_bound=sum(p.numel() for g in optimizer.param_groups for p in g['params']) == sum(p.numel() for p in model.parameters()),
        all_active_adam_clocks_complete=all(row['adam_step'] == steps for row in rows if row['expected_active']),
        all_active_adam_moments_finite=all(row['finite_first_moment'] and row['finite_second_moment'] for row in rows if row['expected_active']),
        complete_backbone_really_changed=active_backbone_changed > 0,
        every_active_official_module_really_changed=bool(backbone_modules) and all(row['changed_elements'] > 0 for row in backbone_modules.values()),
        original_unmasked_unused_token_preserved=all(row['changed_elements'] == 0 for row in rows if not row['expected_active']))
    return dict(passed=all(checks.values()), checks=checks,
        active_backbone_elements=active_backbone_elements, changed_backbone_elements=active_backbone_changed,
        backbone_modules=backbone_modules, named_parameters=rows)


def completion_exit_code(result):
    return 0 if result['execution_complete'] and result['acceptance_passed'] else 2


def run(args):
    profile = getattr(args, 'supervision_profile', 'all_state')
    if profile not in ('all_state', CONDITIONAL_PROFILE, FAILURE_AWARE_PROFILE):
        raise ValueError('Unknown supervision profile')
    failure_aware = profile == FAILURE_AWARE_PROFILE
    conditional = profile in (CONDITIONAL_PROFILE, FAILURE_AWARE_PROFILE)
    dataset_type = (FailureAwareOverfitDataset if failure_aware else
                    ConditionalOverfitDataset if conditional else OverfitDataset)
    collator = collate_conditional if conditional else collate_overfit
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('PROTOCOL.json', 'train.jsonl', 'model.pt', 'RESULT.json')):
        raise FileExistsError('Use a new output directory; no overwrite, restart, or budget extension')
    protocol = dict(data=fixed_protocol(), source_data=str(Path(args.data).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()), mesh=str(Path(args.mesh).resolve()),
        seed=args.seed, steps=2000, batch=4, evaluate_every=500,
        sampler='Each shuffled20-batch epoch covers all80 items exactly once; four paired settings/group/clock',
        learning_rates=dict(backbone=1e-5, sensor_affine=5e-4, readouts=1e-4),
        optimizer='AdamW', weight_decay=.01, gradient_clip_norm=100.,
        loss_scales=LOSS_SCALES, loss_weights=LOSS_WEIGHTS,
        fit_limits=FIT_LIMITS, interpolation_limits=INTERPOLATION_LIMITS,
        interpolation_frames=list(INTERPOLATION_FRAMES), force_target_fields=FORCE_TARGET_FIELDS,
        interpolation_evaluation='Same fixed1440 dataset evaluated batch1 before any update and at final2000; all clocks/targets retained for actual before/after rendering',
        precision='float32; TF32 enabled', disable_stochastic_regularizers=True,
        mesh_training='Fixed1024 original vertices; detached exact SciPy KDTree nearest indices; torch symmetric Euclidean distances. Not official Chamfer.',
        mesh_evaluation='All15626 original vertices, symmetric nearest-vertex distance',
        model='Original complete official Utonia ObjectPredictor plus10 linear auxiliary outputs; original forward unchanged',
        input='Existing20 observation columns only; H32 uniform0.02s; never GT/model-error selection',
        physical_data_gate=dict(collection_qualification_passed=True, all_16_cases_required=True,
            unchanged_controller_checks=list(REQUIRED_CONTROLLER_CHECKS)),
        availability='Learned sigmoid threshold0.5; GT support/stability only target/mask/evaluation',
        state_mask='None: state metrics/loss include no-contact and every failed configuration',
        state_supervision_conflict_check='Before model construction, reject identical-input targets whose center pairwise minimax bound or per-axis relative-size minimax bound exceeds the unchanged fit gate; raw rotation spread alone does not reject. Necessary checks, not a complete observability certificate.',
        scope='Bounded TRAIN overfit and same-trajectory interpolation; no generalization/arbitraryshape/material/demo claim',
        stop='After fixed endpoint, full evaluation, report and actual renders by root, stop for user approval. No automatic continuation.',
        source=dict(host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'), step=os.environ.get('SLURM_STEP_ID')))
    if conditional:
        protocol.update(supervision_profile=profile, collection_study=CONTROLLED_STUDY,
            state_mask='Precision loss and precision gate use any real contact occupancy in the actual H32 input. Every row still forwards and retains raw errors. No GT/error selection; contact is only a regression candidate.',
            state_display='0: PRIOR/UNKNOWN (no contact in H32); 1: CONTACT_CANDIDATE (observability unproven). This flag is observed evidence, not learned calibrated confidence.',
            state_evidence_status_codes={'0': 'PRIOR_UNKNOWN', '1': 'CONTACT_CANDIDATE_OBSERVABILITY_UNPROVEN'},
            state_supervision_conflict_check='Original center/size feasibility checks still reject incompatible identical-input contact candidates; all raw conflicts retained.',
            physical_data_gate=dict(collection_qualification_passed=True, study=CONTROLLED_STUDY,
                all_16_cases_required=True, unchanged_controller_checks=list(REQUIRED_CONTROLLER_CHECKS),
                fixed_late_mass_frame=2381, original_requested_approach_regression_repaired=False),
            scope='Controlled known-scene conditional TRAIN fit and same-trajectory interpolation. Not original blind/all-state qualification, generalization, arbitraryshape, calibrated state confidence or demonstrated tactile benefit.',
            proprio_comparison_plan='Pending separate matched full-model training on the same clocks with hand-only geometry encoder; remove contact-dependent coordinates/normals/occupancy/area/force and point counts. Same initialization/budget, all raw metrics. Frozen zero-force is a separate intervention. This run executes only one full-observation arm and cannot claim tactile benefit.')
    if failure_aware:
        protocol.update(scope=FAILURE_AWARE_SCOPE,
            physical_data_gate=dict(study=CONTROLLED_STUDY, all_16_cases_required=True,
                original_physical_success_is_separate=True,
                unchanged_controller_checks=list(REQUIRED_CONTROLLER_CHECKS),
                original_requested_approach_regression_repaired=False),
            learnable_data_gate='All16 complete, same actual source/config/controller, full2400 clocks/finite observed data/COM, fresh original mass masks and candidate-state input conflict checks. Original collection qualification and late mass success remain separately reported, never relabeled.',
            mass_precision_not_applicable='No available targets in a case means N/A and denominator0, never mass PASS. Available targets retain original numeric limits.',
            availability_single_class='Per-case dense all-negative uses specificity>=.95; all-positive uses sensitivity>=.95; global requires balanced>=.95 and false-confident<=.05. Fit accuracy remains1. All rows assessed.',
            state_history_limit='Any observed H32 contact remains only a regression candidate, not current support or an observability certificate. No-contact H32 stays PRIOR/UNKNOWN. Failed case identity/physical gate/GT never changes inputs or evidence masks.')
    save_json(output / 'PROTOCOL.json', protocol)
    legacy = args.legacy_input_check
    if legacy and not args.check_data:
        raise ValueError('Legacy missing-COM inputs can only be checked, never trained')
    if conditional and legacy:
        raise ValueError('Conditional controlled collection cannot use legacy input fallback')
    check_collection_scope(args.data, CONDITIONAL_PROFILE if failure_aware else profile)
    if not args.check_data:
        require_active_resource()
        initial_artifacts = begin_artifacts(output)
    dataset = dataset_type(args.data, protocol['data'], purpose='input_check' if legacy else 'overfit')
    data_report = qualification(dataset)
    save_json(output / 'DATA_QUALIFICATION.json', data_report)
    if args.check_data:
        print(json.dumps(data_report), flush=True)
        return 0 if data_report['passed'] else 2
    if not data_report['passed']:
        raise RuntimeError('Data qualification failed; preserve all cases and resolve the declared data gate before training')
    from .model import disable_stochastic_regularizers
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    normalized_vertices, sampled, mesh_report = read_known_mesh(args.mesh)
    save_json(output / 'KNOWN_MESH.json', mesh_report)
    interpolation = dataset_type(args.data, protocol['data'], interpolation_frames=INTERPOLATION_FRAMES)
    if failure_aware:
        dense_data_report = failure_aware_dense_qualification(interpolation,
            frames=INTERPOLATION_FRAMES, fit_limits=FIT_LIMITS)
        save_json(output / 'INTERPOLATION_DATA_QUALIFICATION.json', dense_data_report)
        if not dense_data_report['passed']:
            raise RuntimeError('Dense same-trajectory data qualification failed before model construction')
    model = FullUtoniaOverfit(args.checkpoint, history=32).to(device)
    groups = parameter_groups(model)
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    save_json(output / 'MODEL_SCOPE.json', dict(full_original_backbone_parameters=model.predictor.original_parameter_count,
        original_predictor_parameters=sum(p.numel() for p in model.predictor.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()),
        optimizer_parameters=sum(p.numel() for group in groups for p in group['params']),
        groups=[dict(name=g['name'], lr=g['lr'], parameters=sum(p.numel() for p in g['params'])) for g in groups]))
    vertices = torch.as_tensor(normalized_vertices[sampled], device=device)
    initial_report, initial_arrays = evaluate(model, dataset, normalized_vertices, device)
    save_json(output / 'initial_fit.json', initial_report)
    np.savez_compressed(output / 'initial_fit.npz', **initial_arrays)
    initial_interpolation_report, initial_interpolation_arrays = evaluate(model, interpolation, normalized_vertices, device)
    save_json(output / 'initial_same_trajectory_interpolation.json', initial_interpolation_report)
    np.savez_compressed(output / 'initial_same_trajectory_interpolation.npz', **initial_interpolation_arrays)
    iterator = fixed_batches(dataset, args.seed + 1)
    started = time.monotonic()
    with (output / 'train.jsonl').open('x', buffering=1) as log:
        for step in range(1, 2001):
            indices = next(iterator)
            batch = device_batch(collator([dataset[index] for index in indices]), device)
            model.train(); disable_stochastic_regularizers(model)
            optimizer.zero_grad(set_to_none=True)
            output_values = model(batch['inputs'])
            parts = (conditional_task_losses if conditional else task_losses)(output_values, batch, vertices)
            loss = total_loss(parts)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite supervised loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100., error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step, loss=float(loss.detach()),
                parts={key: float(value.detach()) for key, value in parts.items()}, grad_norm_before_clip=float(norm),
                available_mass_samples=int(batch['mass_available'].sum()),
                samples=[dict(episode=r['episode'], frame=r['frame']) for r in batch['metadata']],
                elapsed_s=time.monotonic() - started)
            if conditional:
                record['state_precision_eligible_samples'] = int(batch['state_precision_eligible'].sum())
            log.write(json.dumps(record) + '\n')
            if step == 1 or step % 20 == 0:
                print(json.dumps(record), flush=True)
            if step % 500 == 0:
                report, arrays = evaluate(model, dataset, normalized_vertices, device)
                save_json(output / f'fit_{step:05d}.json', report)
                np.savez_compressed(output / f'fit_{step:05d}.npz', **arrays)
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                                protocol=protocol), output / 'latest.pt')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=2000,
                    protocol=protocol), output / 'model.pt')
    update_report = parameter_update_report(model, optimizer, initial, 2000)
    save_json(output / 'PARAMETER_UPDATES.json', update_report)
    del initial
    # Reproduce the saved endpoint on one real batch, including all learned heads.
    batch = device_batch(collator([dataset[0]]), device)
    model.eval()
    with torch.no_grad():
        before = {key: value.detach().clone() for key, value in model(batch['inputs']).items()}
        saved = torch.load(output / 'model.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        after = model(batch['inputs'])
        reload_difference = max(float((before[key] - after[key]).abs().max()) for key in before)
    del saved
    interpolation_report, interpolation_arrays = evaluate(model, interpolation, normalized_vertices, device)
    save_json(output / 'same_trajectory_interpolation.json', interpolation_report)
    np.savez_compressed(output / 'same_trajectory_interpolation.npz', **interpolation_arrays)
    clock_target_keys = ('episode', 'frame', 'timestamp_s', 'target', 'force_target_n', 'mass_available', 'mass_status', 'contact_present')
    if conditional:
        clock_target_keys += ('state_precision_eligible', 'state_contact_history_frames', 'state_evidence_status')
    identical_clock_targets = all(np.array_equal(initial_interpolation_arrays[key], interpolation_arrays[key])
        for key in clock_target_keys)
    finish_artifacts(output, initial_artifacts)
    result = dict(execution_complete=True, steps=2000, optimizer_updates=2000,
        acceptance_passed=bool(report['passed'] and interpolation_report['passed'] and update_report['passed'] and reload_difference == 0 and identical_clock_targets),
        fit_passed=report['passed'], same_trajectory_interpolation_passed=interpolation_report['passed'],
        data_qualification_passed=data_report['passed'], full_parameter_update_passed=update_report['passed'],
        full_endpoint_reload_max_abs=reload_difference, fit_items=len(dataset), interpolation_items=len(interpolation),
        initial_final_interpolation_clocks_targets_exact=identical_clock_targets,
        renders='Pending separate actual full-mesh root renderer; this script does not render',
        elapsed_s=time.monotonic() - started, scope=protocol['scope'],
        user_approval_required_before_next_stage=True)
    if failure_aware:
        result.update(supervision_profile=profile,
            physical_success_gate_passed=data_report['physical_success_gate_passed'],
            original_conditional_qualification_passed=data_report['original_conditional_qualification_passed'],
            learnable_data_gate_passed=data_report['learnable_data_gate_passed'],
            failure_aware_learning_acceptance_passed=result['acceptance_passed'],
            mass_cases_with_available_fit_targets=report['mass_cases_with_available_targets'],
            mass_cases_without_available_fit_targets=report['mass_cases_without_available_targets'],
            physical_success_or_tactile_benefit_claimed=False)
    save_json(output / 'RESULT.json', result)
    print(json.dumps(result), flush=True)
    return completion_exit_code(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', default='experiments/object_predictor_v1/checkpoints/utonia.pth')
    parser.add_argument('--mesh', default='experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=310116)
    parser.add_argument('--check-data', action='store_true', help='CPU observations/labels only; no model construction')
    parser.add_argument('--legacy-input-check', action='store_true', help='Only with --check-data; missing COM remains unqualified')
    parser.add_argument('--supervision-profile', choices=('all_state', CONDITIONAL_PROFILE, FAILURE_AWARE_PROFILE), default='all_state',
        help='Explicit controlled profiles require controlled_fixture16_v1; failure-aware separates physical failure from learning eligibility. Original default unchanged.')
    args = parser.parse_args()
    try:
        code = run(args)
    except Exception as exc:
        output = Path(args.output)
        # Do not overwrite an existing run's result when refusing a duplicate.
        if output.exists() and not isinstance(exc, FileExistsError):
            save_json(output / 'FAILURE.json', dict(execution_complete=False,
                error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise
    raise SystemExit(code)


if __name__ == '__main__':
    main()
