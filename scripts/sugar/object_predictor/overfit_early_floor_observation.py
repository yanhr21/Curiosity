"""Early observed-floor glue around the unchanged complete official Utonia."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

from . import train_overfit as original
from . import overfit_fullbatch_refinement as base
from . import overfit_floor_observation as late

STUDY = 'failure_aware_early_floor_observation_v1'
SOURCE_UPDATES = base.SOURCE_UPDATES
NEW_UPDATES = base.NEW_UPDATES
MICROBATCHES = base.MICROBATCHES
HEIGHT_SCALE_M = late.HEIGHT_SCALE_M
POINT_KEYS = late.POINT_KEYS
GROUP = 'early_floor_observation'
EXTRA_SOURCES = (*late.EXTRA_SOURCES,
    'scripts.sugar.object_predictor.overfit_early_floor_observation',
    'scripts.sugar.object_predictor.train_overfit_early_floor_observation')
ARTIFACTS = (*base.ARTIFACTS, 'ZERO_STEM_REPLAY.json')
FloorDataset = late.FloorDataset
collate_floor = late.collate_floor
evaluate_floor = late.evaluate_floor


def derive_stem_width(state):
    prefix = 'predictor.backbone.embedding.stem.linear.'
    weight, bias, extra = (state[prefix + name] for name in
                           ('original.weight', 'original.bias', 'extra.weight'))
    if (weight.ndim != 2 or weight.shape[1] != 9 or weight.shape[0] <= 0
            or bias.shape != (weight.shape[0],) or extra.shape != (weight.shape[0], 11)):
        raise ValueError('Source original9+sensor11 stem shape mismatch')
    return int(weight.shape[0])


def point_floor_height(height, offset, history, total_points):
    """Repeat CURRENT sample height across its H frames and their unequal points."""
    if (history <= 0 or offset.ndim != 1 or not len(offset)
            or offset.dtype not in (torch.int32, torch.int64)
            or len(offset) % history or height.shape != (len(offset) // history, 1)
            or not bool(torch.isfinite(height).all())):
        raise ValueError('Incomplete causal sample/history/floor layout')
    counts = torch.diff(torch.cat((offset.new_zeros(1), offset))).long()
    if bool((counts <= 0).any()) or int(offset[-1]) != total_points:
        raise ValueError('Offsets must cover every nonempty original frame exactly')
    frame_height = torch.repeat_interleave(height, history, dim=0)
    return torch.repeat_interleave(frame_height, counts.to(height.device), dim=0)


def validate_floor_inputs(dataset):
    report = late.validate_floor_inputs(dataset)
    for start in range(0, len(dataset), 4):
        rows = dataset.rows[start:start+4]
        inputs = collate_floor(rows)['inputs']
        mapped = point_floor_height(inputs['floor_height_m'], inputs['offset'],
                                    32, len(inputs['feat']))
        point_start = 0
        for index, row in enumerate(rows):
            point_end = int(inputs['offset'][(index+1)*32-1])
            if not bool((mapped[point_start:point_end] == float(row['floor_height_m'][0])).all()):
                raise ValueError('Cross-sample current floor mapping')
            point_start = point_end
    return dict(report, current_height_repeated_across_each_complete_history=True,
                all_unequal_point_offsets_checked=True)


class EarlyFloorObservationOverfit(original.FullUtoniaOverfit):
    def add_floor_branch(self):
        if hasattr(self, 'floor_embedding'):
            raise ValueError('Early floor branch already installed')
        old = {name: id(p) for name, p in self.named_parameters()}
        stem = self.predictor.backbone.embedding.stem.linear
        width = stem.original.out_features
        if stem.extra.in_features != 11 or stem.extra.out_features != width:
            raise ValueError('Require unmodified original SensorAffine')
        parameter = stem.original.weight
        devices = [parameter.device.index] if parameter.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self.floor_embedding = nn.Linear(1, width, bias=False,
                device=parameter.device, dtype=parameter.dtype)
            nn.init.zeros_(self.floor_embedding.weight)
        if any(id(dict(self.named_parameters())[name]) != identity for name, identity in old.items()):
            raise RuntimeError('Installing floor glue changed original parameter identity')
        self.zero_stem_enabled = True
        self.zero_stem_items = 0
        self.zero_stem_calls = 0
        self.zero_stem_points = 0
        self._floor_hook_active = False

    def forward(self, inputs):
        if set(inputs) != set(POINT_KEYS) | {'floor_height_m'}:
            raise ValueError('Only original four point inputs and observed floor height allowed')
        if self._floor_hook_active:
            raise RuntimeError('Reentrant floor hook is forbidden')
        heights = point_floor_height(inputs['floor_height_m'], inputs['offset'],
                                     self.history, len(inputs['feat']))
        calls = 0

        def inject(module, args, value):
            nonlocal calls
            calls += 1
            if calls != 1 or value.shape != (len(heights), self.floor_embedding.out_features):
                raise RuntimeError('Unexpected original SensorAffine execution/layout')
            delta = self.floor_embedding(heights.to(value) / HEIGHT_SCALE_M)
            updated = value + delta
            if self.zero_stem_enabled:
                if not torch.equal(value, updated):
                    raise RuntimeError('Zero floor branch changed original stem output')
                self.zero_stem_calls += 1
                self.zero_stem_items += len(inputs['floor_height_m'])
                self.zero_stem_points += len(heights)
            return updated

        stem = self.predictor.backbone.embedding.stem.linear
        self._floor_hook_active = True
        handle = stem.register_forward_hook(inject)
        try:
            output = super().forward({key: inputs[key] for key in POINT_KEYS})
            if calls != 1:
                raise RuntimeError('Original forward did not execute exactly one SensorAffine')
            return output
        finally:
            handle.remove()
            self._floor_hook_active = False


def add_optimizer_groups(model, optimizer):
    existing = {id(p) for group in optimizer.param_groups for p in group['params']}
    parameter = model.floor_embedding.weight
    if id(parameter) in existing or parameter in optimizer.state:
        raise ValueError('New floor embedding must begin outside restored Adam')
    sensors = [group for group in optimizer.param_groups if group['name'] == 'sensor_affine']
    if len(sensors) != 1:raise ValueError('Require the restored sensor optimizer group')
    optimizer.add_param_group(dict(sensors[0], params=[parameter], name=GROUP,
                                   lr=base.LR_BASE['sensor_affine']))


def apply_learning_rates(optimizer, update):
    rates = base.cosine_learning_rates(update)
    rates[GROUP] = rates['sensor_affine']
    if {group['name'] for group in optimizer.param_groups} != set(rates):
        raise ValueError('Expected four original groups plus one early floor group')
    for group in optimizer.param_groups:
        group['lr'] = rates[group['name']]
    return rates


def verify_optimizer(model, optimizer, old_step, new_step):
    names = {id(p): name for name, p in model.named_parameters()}
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if len(parameters) != len(names) or {id(p) for p in parameters} != set(names):
        raise ValueError('Incomplete or duplicate optimizer binding')
    old_count = new_count = 0
    for parameter in parameters:
        name = names[id(parameter)]
        state = optimizer.state.get(parameter)
        if name == 'predictor.backbone.embedding.mask_token':
            if state:raise ValueError('Unused original token unexpectedly updated')
            continue
        is_new = name == 'floor_embedding.weight'
        expected = new_step if is_new else old_step
        if is_new:new_count += 1
        else:old_count += 1
        if expected == 0:
            if state:raise ValueError('New floor Adam must start without moments')
            continue
        if not state or int(state['step']) != expected:
            raise ValueError('Incorrect Adam clock: ' + name)
        for key in ('exp_avg', 'exp_avg_sq'):
            if state[key].shape != parameter.shape or not bool(torch.isfinite(state[key]).all()):
                raise ValueError('Malformed Adam moments: ' + name)
    if new_count != 1:
        raise ValueError('Exactly one new embedding tensor required')
    return dict(passed=True, old_active_tensors=old_count, new_active_tensors=new_count,
                old_adam_step=old_step, new_adam_step=new_step, all_parameters_bound=True)


def update_report(model, optimizer, initial):
    result = original.parameter_update_report(model, optimizer, initial, SOURCE_UPDATES+NEW_UPDATES)
    result['checks']['all_active_adam_clocks_complete'] = all(
        row['adam_step'] == (NEW_UPDATES if row['name'] == 'floor_embedding.weight' else SOURCE_UPDATES+NEW_UPDATES)
        for row in result['named_parameters'] if row['expected_active'])
    new = [row for row in result['named_parameters'] if row['name'] == 'floor_embedding.weight']
    result['checks']['early_floor_embedding_really_updated'] = len(new) == 1 and new[0]['changed_elements'] > 0
    result['passed'] = all(result['checks'].values())
    return result


def source_protocol(source, matched_baseline, late_baseline):
    protocol = base.source_protocol(source)
    control = Path(matched_baseline).resolve()
    late_control = Path(late_baseline).resolve()
    base.verify_artifacts(control)
    late.verify_artifacts(late_control)
    if json.loads((control/'PROTOCOL.json').read_text()) != protocol:
        raise ValueError('Require exact completed no-floor100 baseline')
    if json.loads((late_control/'PROTOCOL.json').read_text()) != late.source_protocol(source, control):
        raise ValueError('Require exact completed late-floor100 comparison')
    comparisons = {}
    for role, folder in (('no_floor', control), ('late_floor', late_control)):
        result = json.loads((folder/'RESULT.json').read_text())
        if (result.get('execution_complete') is not True or result.get('optimizer_updates') != 100
                or result.get('total_optimizer_updates') != 2100 or result.get('full_endpoint_reload_max_abs') != 0):
            raise ValueError('Comparison endpoint incomplete: ' + role)
        comparisons[role] = {name: dict(path=str(folder/name), sha256=original.sha256_file(folder/name))
            for name in ('PROTOCOL.json', 'RESULT.json', 'ARTIFACTS.json', 'fit_02100.npz', 'same_trajectory_interpolation.npz')}
    checkpoint = torch.load(Path(source)/'model.pt', map_location='meta', weights_only=False, mmap=True)
    width = derive_stem_width(checkpoint['model'])
    del checkpoint
    return dict(protocol, study=STUDY, matched_baseline=str(control), late_baseline=str(late_control),
        comparison_bindings=comparisons,
        prepared_adapter_sources={name:dict(path=str(Path(__file__).with_name(name)),
            sha256=original.sha256_file(Path(__file__).with_name(name))) for name in
            ('overfit_early_floor_observation.py', 'train_overfit_early_floor_observation.py')},
        model='Complete original official Utonia and SensorAffine/readouts, plus one zero bias-free observed-floor embedding branch',
        floor_observation=dict(source='Current LEFT hand hand_pose_w[frame,0,2] minus public floor z0',
            public_floor_z_m=0., normalization_m=HEIGHT_SCALE_M, embedding_width=width, new_weights=width,
            scale_source='Same fixed public0.01m center loss normalization as late-floor arm',
            injection='Temporary SensorAffine output hook before original LayerNorm/GELU; finally removed',
            point_assignment='Original offset defines H32 frame blocks; repeat each sample CURRENT height across all its history points',
            zero_validation='1520 actual forward calls directly verify stem-before equals stem-plus-zero; no direct dual final-output comparison claimed',
            exclusions=['GT object geometry/min_z/mass', 'mass availability', 'history absolute heights', 'eight-force summaries', 'late output branches'],
            limitations='LayerNorm removes channel-common shift. Early interaction is possible, not guaranteed; shared current height can remain a trajectory shortcut.'),
        learning_rates={**base.LR_BASE, GROUP:base.LR_BASE['sensor_affine']},
        restore=protocol['restore']+' One new embedding tensor starts zero/no-state and ends at100; original active Adam ends at2100.',
        optimization_change='Same original four schedules/100x20 objective/clip100/gates; new floor group uses existing sensor-affine5e-5 cosine schedule. Early versus late differs in location, added parameter count and meaningful LR units.',
        scope=original.FAILURE_AWARE_SCOPE+' Early current-floor observation glue compared with fixed no-floor and late-floor2100 endpoints. No physics or tactile-benefit claim.')


def actual_source_bindings():
    result = original.actual_source_bindings()
    for name in EXTRA_SOURCES:
        path = Path(importlib.util.find_spec(name).origin).resolve()
        result[name] = dict(path=str(path), sha256=original.sha256_file(path))
    return result


def begin_artifacts(root):
    root = Path(root)
    result = dict(schema=4, kind=STUDY, complete=False, sources=actual_source_bindings(),
        protocol=dict(path='PROTOCOL.json', sha256=original.sha256_file(root/'PROTOCOL.json')), artifacts={})
    original.save_json(root/'ARTIFACTS.json', result)
    return result


def verify_artifacts(root, *, require_complete=True, manifest=None):
    root = Path(root)
    if manifest is None:manifest = json.loads((root/'ARTIFACTS.json').read_text())
    if manifest.get('schema') != 4 or manifest.get('kind') != STUDY or (require_complete and manifest.get('complete') is not True):
        raise ValueError('Require explicit early-floor artifact manifest')
    if set(manifest['sources']) != set(original.SOURCE_MODULES) | set(EXTRA_SOURCES):
        raise ValueError('Wrong early-floor source set')
    for name, binding in manifest['sources'].items():
        if original.sha256_file(binding['path']) != binding['sha256']:
            raise ValueError('Early-floor source changed: ' + name)
    if manifest['protocol'] != dict(path='PROTOCOL.json', sha256=original.sha256_file(root/'PROTOCOL.json')):
        raise ValueError('Early-floor protocol changed')
    if require_complete:
        if set(manifest['artifacts']) != set(ARTIFACTS):raise ValueError('Incomplete early-floor artifacts')
        for name, binding in manifest['artifacts'].items():
            if binding != dict(path=name, sha256=original.sha256_file(root/name)):
                raise ValueError('Early-floor artifact changed: ' + name)
    return manifest


def finish_artifacts(root, initial):
    root = Path(root)
    verify_artifacts(root, manifest=initial, require_complete=False)
    if actual_source_bindings() != initial['sources']:raise ValueError('Actual early-floor imports changed')
    final = dict(initial, complete=True, artifacts={name:dict(path=name,
        sha256=original.sha256_file(root/name)) for name in ARTIFACTS})
    original.save_json(root/'ARTIFACTS.json', final)
    return final
