"""Bindings for one deterministic-posterior native AE comparison.

Reuse the frozen dynamic sampler and original data, without rewriting any
arrays. The comparison endpoint is the completed stochastic dynamic-grid run.
Only the training posterior flag changes in the independent trainer. Dynamic
error pools may consequently differ; identical volume IDs are not promised.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import michelangelo_dynamic_grid_data as dynamic
from . import train_michelangelo_overfit as original
from .qualify_michelangelo_ae import OBJECT_IDS, digest

SCOPE = 'full_surface_native_ae_mode_grid_overfit_only'
REVISION = 'dynamic_extraction_grid_mode_training_v1'
DYNAMIC_DATA = original.EXPERIMENT/'overfit_repair_v1/michelangelo_dynamic_grid_data_v1'
PREVIOUS_RUN = original.EXPERIMENT/'overfit_repair_v1/michelangelo_dynamic_grid_overfit_v1'
DYNAMIC_TRAINER = Path(__file__).with_name('train_michelangelo_dynamic_grid.py')
REFRESH_STEPS = dynamic.REFRESH_STEPS
fixed_recipe = dynamic.fixed_recipe
error_pools = dynamic.error_pools
sample_volume = dynamic.sample_volume
batch_for_step = dynamic.batch_for_step


def load_sources(dynamic_data=DYNAMIC_DATA, previous_run=PREVIOUS_RUN):
    manifest, queries, cases = dynamic.read_data(dynamic_data)
    previous = json.loads((previous_run/'RESULT.json').read_text())
    protocol = json.loads((previous_run/'PROTOCOL.json').read_text())
    if (previous['scope'] != dynamic.SCOPE or not previous['complete']
            or previous['sampling_revision'] != 'dynamic_extraction_grid_v1'
            or previous['model_updates'] != 1000
            or tuple(x['object_id'] for x in previous['cases']) != OBJECT_IDS
            or protocol['train_posterior'] != 'sample'
            or protocol['data_manifest_sha256'] != digest(dynamic_data/'MANIFEST.json')
            or protocol['bindings']['trainer_sha256'] != digest(DYNAMIC_TRAINER)):
        raise ValueError('Require the complete unchanged stochastic dynamic-grid comparison')
    for row in previous['cases']:
        if digest(previous_run/row['output_file']) != row['case_output_sha256']:
            raise ValueError('Actual stochastic comparison output changed')
    sources = dict(dynamic_data=str(dynamic_data.resolve()),
                   dynamic_manifest_sha256=digest(dynamic_data/'MANIFEST.json'),
                   dynamic_adapter_sha256=digest(Path(dynamic.__file__)),
                   dynamic_trainer_sha256=digest(DYNAMIC_TRAINER),
                   original_trainer_sha256=digest(Path(original.__file__)),
                   previous_run=str(previous_run.resolve()),
                   previous_result_sha256=digest(previous_run/'RESULT.json'),
                   previous_protocol_sha256=digest(previous_run/'PROTOCOL.json'))
    return manifest, queries, cases, previous, sources


def prepare(output):
    dynamic_manifest, queries, cases, previous, sources = load_sources()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(scope=SCOPE, sampling_revision=REVISION, recipe=fixed_recipe(),
                    only_training_change='official full-model sample_posterior=False',
                    recipe_interpretation='project deterministic native-AE task adaptation; not original stochastic VAE recipe',
                    sources=sources, dynamic_manifest=dynamic_manifest,
                    original_manifest=dynamic_manifest['original_manifest'],
                    cases=dynamic_manifest['cases'], object_ids=list(OBJECT_IDS),
                    grid_nodes=len(queries), counts=dynamic_manifest['counts'],
                    previous_cases=[dict(object_id=r['object_id'],
                        file=str((PREVIOUS_RUN/r['output_file']).resolve()),
                        sha256=r['case_output_sha256']) for r in previous['cases']],
                    data_adapter_sha256=digest(Path(__file__)),
                    model_forwards=0, model_updates=0)
    original.write_json(output/'MANIFEST.json', manifest)
    return manifest


def read_data(directory):
    manifest = json.loads((directory/'MANIFEST.json').read_text())
    if (manifest['scope'] != SCOPE or manifest['sampling_revision'] != REVISION
            or manifest['recipe'] != fixed_recipe()
            or tuple(manifest['object_ids']) != OBJECT_IDS
            or manifest['data_adapter_sha256'] != digest(Path(__file__))):
        raise ValueError('Deterministic-posterior comparison binding changed')
    sources = manifest['sources']
    dynamic_manifest, queries, cases, previous, actual = load_sources(
        Path(sources['dynamic_data']), Path(sources['previous_run']))
    expected_previous = [dict(object_id=r['object_id'],
                        file=str((Path(sources['previous_run'])/r['output_file']).resolve()),
                        sha256=r['case_output_sha256']) for r in previous['cases']]
    if (actual != sources or dynamic_manifest != manifest['dynamic_manifest']
            or manifest['original_manifest'] != dynamic_manifest['original_manifest']
            or manifest['previous_cases'] != expected_previous):
        raise ValueError('Bound original inputs, sampler or comparison outputs changed')
    return manifest, queries, cases
