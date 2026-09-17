"""Sampling-only four-shape AE experiment, reset from official released weights.

The explicit run body is identical to train_michelangelo_overfit.run except for
additional protocol metadata and the imported coverage read_data/train_indices/
SCOPE. All original full-model/loss/optimizer/1000-step/seed/gates/reload functions
are reused. The previous pinned source is not changed or monkeypatched. Geometry
is never filtered; both raw official and corrected full meshes are saved.
CPU prepare/check only; root alone queues the guarded GPU run.
"""
from __future__ import annotations

import argparse
import gc
import traceback
import json
from pathlib import Path

from . import michelangelo_coverage_data as coverage
from .michelangelo_coverage_data import SCOPE, read_data, train_indices
from .train_michelangelo_overfit import (
    ABC_TO_VAE, CHECKPOINT, CHECKPOINT_BYTES, CHECKPOINT_SHA, OBJECT_IDS,
    STEPS, TRAIN_SEED, VENDOR, audit_saved_checkpoint, batch_loss,
    dependency_check, digest, endpoint_meshes, evaluate_queries, gradients,
    load_official_model, optimizer_for, original_loss, write_json,
)


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
    protocol.update(single_changed_factor='project volume-query sampling adapter only',
                    sampling_revision=manifest['sampling_revision'],
                    surface_pair_scheme=manifest['surface_pair_scheme'],
                    batch_surface_composition=manifest['batch_surface_composition'],
                    previous_coverage_manifest_sha256=manifest['previous_coverage_manifest_sha256'],
                    uniform_reused_bit_exact=True,
                    volume_strata=manifest['volume_strata'], batch_volume_strata=manifest['batch_volume_strata'],
                    paired_normal_offset_vae=manifest['paired_normal_offset_vae'],
                    near_pool_and_per_step_indices_bit_exact=True,
                    encoder_inputs_and_evaluation_bit_exact=True,
                    previous_result=manifest['previous_result'],
                    previous_result_sha256=manifest['previous_result_sha256'],
                    previous_cases=[dict(object_id=r['object_id'], file=r['previous_prediction_file'],
                                         sha256=r['previous_prediction_sha256']) for r in manifest['cases']],
                    unchanged_previous_trainer_sha256=manifest['pinned_previous_trainer_sha256'],
                    data_adapter_sha256=digest(Path(coverage.__file__)))
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
    parser.add_argument('--old-data', type=Path, default=coverage.OLD_DATA)
    parser.add_argument('--previous-run', type=Path, default=coverage.OLD_RUN)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.output is None:
            parser.error('prepare requires new --output')
        return coverage.prepare(args)
    if args.data is None:
        parser.error('check/run require --data')
    if args.stage == 'check':
        manifest, data = read_data(args.data)
        deps = dependency_check(); loss = original_loss()
        import torch
        print(json.dumps(dict(scope=SCOPE, dependencies=deps, object_ids=manifest['object_ids'],
                              original_loss=type(loss).__module__+'.'+type(loss).__name__,
                              original_near_input_eval_bit_exact=True,
                              volume_strata=manifest['volume_strata'],
                              batch_volume_strata=manifest['batch_volume_strata'],
                              cuda_initialized=torch.cuda.is_initialized(), model_forwards=0), indent=2))
        return 0 if all(deps.values()) and not torch.cuda.is_initialized() else 2
    if args.output is None:
        parser.error('run requires new --output')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
