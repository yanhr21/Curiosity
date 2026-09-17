"""Explicit floor-only observation adapter; original full model/objective retained."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import train_overfit as original
from . import overfit_fullbatch_refinement as base

STUDY = 'failure_aware_floor_observation_v1'
SOURCE_UPDATES = base.SOURCE_UPDATES
NEW_UPDATES = base.NEW_UPDATES
MICROBATCHES = base.MICROBATCHES
FLOOR_Z_M = 0.
HEIGHT_SCALE_M = original.LOSS_SCALES['center_m']
POINT_KEYS = ('coord', 'grid_coord', 'feat', 'offset')
FLOOR_GROUPS = dict(floor_state_observation='state_readout',
                   floor_auxiliary_observation='auxiliary_readout')
EXTRA_SOURCES = (*base.EXTRA_SOURCES,
    'scripts.sugar.object_predictor.overfit_floor_observation',
    'scripts.sugar.object_predictor.train_overfit_floor_observation')
ARTIFACTS = (*base.ARTIFACTS, 'ZERO_BRANCH_REPLAY.json')


def current_floor_height(hand_pose_w, frame):
    """Only observed current LEFT hand pose and the public fixed floor enter."""
    pose = np.asarray(hand_pose_w)
    if pose.ndim != 3 or pose.shape[1:] != (2, 7) or not 0 <= frame < len(pose):
        raise ValueError('Expected recorded bilateral hand poses and current clock')
    value = pose[frame, 0, 2] - FLOOR_Z_M
    if not np.isfinite(value):raise ValueError('Nonfinite observed floor height')
    return np.array([value], dtype=np.float32)


class FloorDataset(original.FailureAwareOverfitDataset):
    """Preserve original rows, all gates and point inputs; append proprioception."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        poses = {}
        for episode, record in self.collection_records.items():
            with np.load(Path(record['source']) / f'episode_{episode:04d}.npz', allow_pickle=False) as saved:
                poses[episode] = saved['hand_pose_w']
        for row in self.rows:
            row['floor_height_m'] = current_floor_height(
                poses[row['metadata']['episode']], row['metadata']['frame'])


def collate_floor(rows):
    result = original.collate_conditional(rows)
    if set(result['inputs']) != set(POINT_KEYS):raise ValueError('Original point interface changed')
    height = np.stack([row['floor_height_m'] for row in rows])
    if height.shape != (len(rows), 1) or not np.isfinite(height).all():
        raise ValueError('Each causal item must have one finite observed floor height')
    result['inputs']['floor_height_m'] = torch.from_numpy(height)
    return result


def validate_floor_inputs(dataset):
    count = 0
    for begin in range(0, len(dataset), 4):
        rows = dataset.rows[begin:begin+4]
        reference = original.collate_conditional(rows)['inputs']
        augmented = collate_floor(rows)['inputs']
        if any(not torch.equal(reference[key], augmented[key]) for key in POINT_KEYS):
            raise ValueError('Floor adapter changed an original point tensor')
        count += len(rows)
    values = np.array([row['floor_height_m'][0] for row in dataset.rows], dtype=np.float64)
    return dict(passed=True,items=count,all_original_four_point_tensors_exact=True,
        current_left_hand_height_only=True,public_floor_z_m=FLOOR_Z_M,
        height_scale_m=HEIGHT_SCALE_M,height_min_m=float(values.min()),height_max_m=float(values.max()),
        model_forwards=0,cuda_initialized=torch.cuda.is_initialized())


class FloorObservationOverfit(original.FullUtoniaOverfit):
    """The full inherited original forward plus two explicit zero linear branches."""
    def add_floor_branches(self):
        if hasattr(self, 'floor_state'):raise ValueError('Floor branches already installed')
        parameter = next(self.parameters())
        devices = [parameter.device.index] if parameter.is_cuda else []
        # nn.Linear initialization must not consume the matched model RNG stream.
        with torch.random.fork_rng(devices=devices):
            self.floor_state = nn.Linear(1, 13, bias=False, device=parameter.device, dtype=parameter.dtype)
            self.floor_auxiliary = nn.Linear(1, 10, bias=False, device=parameter.device, dtype=parameter.dtype)
            nn.init.zeros_(self.floor_state.weight)
            nn.init.zeros_(self.floor_auxiliary.weight)
        self.zero_replay_enabled = True
        self.zero_replay_items = 0
        self.zero_replay_calls = 0

    def forward(self, inputs):
        if set(inputs) != set(POINT_KEYS) | {'floor_height_m'}:
            raise ValueError('Floor model accepts only original four point inputs and observed height')
        height = inputs['floor_height_m']
        frames = len(inputs['offset'])
        if (frames % self.history or height.shape != (frames // self.history, 1)
                or not bool(torch.isfinite(height).all())):
            raise ValueError('Floor observation does not match complete causal histories')
        original_output = super().forward({key: inputs[key] for key in POINT_KEYS})
        scalar = height.to(original_output['state']) / HEIGHT_SCALE_M
        state_delta = self.floor_state(scalar)
        auxiliary_delta = self.floor_auxiliary(scalar)
        output = dict(state=original_output['state'] + state_delta,
            force=original_output['force'] + auxiliary_delta[:, :8],
            availability_logit=original_output['availability_logit'] + auxiliary_delta[:, 8],
            contact_logit=original_output['contact_logit'] + auxiliary_delta[:, 9])
        if self.zero_replay_enabled:
            if any(not torch.equal(original_output[key], output[key]) for key in original_output):
                raise RuntimeError('Zero-initialized floor branch changed source2000 output')
            self.zero_replay_items += len(height)
            self.zero_replay_calls += 1
        return output


def add_optimizer_groups(model, optimizer):
    original_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    for name, parameters, parent in (
        ('floor_state_observation', model.floor_state.parameters(), 'state_readout'),
        ('floor_auxiliary_observation', model.floor_auxiliary.parameters(), 'auxiliary_readout')):
        source = next(group for group in optimizer.param_groups if group['name'] == parent)
        optimizer.add_param_group(dict(source, params=list(parameters), name=name))
    bound = [p for group in optimizer.param_groups for p in group['params']]
    if len(bound) != len({id(p) for p in bound}) or {id(p) for p in bound} != {id(p) for p in model.parameters()}:
        raise ValueError('Floor optimizer must bind the complete model exactly once')
    if any(optimizer.state.get(p) for p in bound if id(p) not in original_ids):
        raise ValueError('New floor branches must have fresh optimizer state')


def apply_learning_rates(optimizer, update):
    rates = base.cosine_learning_rates(update)
    rates.update({name: rates[parent] for name, parent in FLOOR_GROUPS.items()})
    if {group['name'] for group in optimizer.param_groups} != set(rates):
        raise ValueError('Require four unchanged original groups and two floor groups')
    for group in optimizer.param_groups:group['lr'] = rates[group['name']]
    return rates


def verify_optimizer(model, optimizer, old_step, new_step):
    names = {id(p): name for name, p in model.named_parameters()}
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if len(parameters) != len(names) or {id(p) for p in parameters} != set(names):
        raise ValueError('Incomplete or duplicate optimizer binding')
    old_count = new_count = 0
    for parameter in parameters:
        name = names[id(parameter)];state = optimizer.state.get(parameter)
        if name == 'predictor.backbone.embedding.mask_token':
            if state:raise ValueError('Unused original token unexpectedly updated')
            continue
        is_new = name.startswith(('floor_state.', 'floor_auxiliary.'))
        expected = new_step if is_new else old_step
        if is_new:new_count += 1
        else:old_count += 1
        if expected == 0:
            if state:raise ValueError('New floor Adam must start without moments')
            continue
        if not state or int(state['step']) != expected:raise ValueError('Incorrect Adam clock: ' + name)
        for key in ('exp_avg', 'exp_avg_sq'):
            if state[key].shape != parameter.shape or not bool(torch.isfinite(state[key]).all()):
                raise ValueError('Malformed Adam moments: ' + name)
    return dict(passed=True,old_active_tensors=old_count,new_active_tensors=new_count,
                old_adam_step=old_step,new_adam_step=new_step,all_parameters_bound=True)


def update_report(model, optimizer, initial):
    result = original.parameter_update_report(model, optimizer, initial, SOURCE_UPDATES + NEW_UPDATES)
    # Only the two newly introduced tensors have100 rather than2100 Adam updates.
    result['checks']['all_active_adam_clocks_complete'] = all(
        row['adam_step'] == (NEW_UPDATES if row['name'].startswith(('floor_state.', 'floor_auxiliary.')) else SOURCE_UPDATES+NEW_UPDATES)
        for row in result['named_parameters'] if row['expected_active'])
    result['checks']['both_floor_branches_really_updated'] = all(row['changed_elements'] > 0
        for row in result['named_parameters'] if row['name'].startswith(('floor_state.', 'floor_auxiliary.')))
    result['passed'] = all(result['checks'].values())
    return result


def source_protocol(source, matched_baseline):
    protocol = base.source_protocol(source)
    baseline = Path(matched_baseline).resolve()
    base.verify_artifacts(baseline)
    bp = json.loads((baseline/'PROTOCOL.json').read_text())
    br = json.loads((baseline/'RESULT.json').read_text())
    if bp != protocol or not br['execution_complete'] or br['optimizer_updates'] != NEW_UPDATES:
        raise ValueError('Require the completed exact matched100 fullbatch baseline')
    bindings = {name:dict(path=str(baseline/name),sha256=original.sha256_file(baseline/name))
        for name in ('PROTOCOL.json','RESULT.json','ARTIFACTS.json','initial_fit.npz','initial_same_trajectory_interpolation.npz')}
    adapter_sources={name:dict(path=str(Path(__file__).with_name(name)),sha256=original.sha256_file(Path(__file__).with_name(name)))
        for name in ('overfit_floor_observation.py','train_overfit_floor_observation.py')}
    return dict(protocol,study=STUDY,matched_baseline=str(baseline),matched_baseline_bindings=bindings,
        prepared_adapter_sources=adapter_sources,
        model='Complete original official Utonia+sensor affine+original13/10 readouts, plus23 zero-initialized floor-observation weights',
        floor_observation=dict(source='Current LEFT hand hand_pose_w[frame,0,2] minus public floor z0',
            coordinate='Signed world vertical height in meters; gravity already gives plane normal in current hand frame',
            public_floor_z_m=FLOOR_Z_M,normalization_m=HEIGHT_SCALE_M,
            scale_source='Existing fixed public center loss scale0.01m; not selected from fit/interpolation statistics',
            adapter='Bias-free zero linear1->13 state and1->10 auxiliary residuals;23 new weights. No force summary.',
            zero_equivalence='All80+1440 initial rows compare original source2000 outputs with branch-added outputs within the same original forward; require exact0. Cross-process historical tiny differences are only reported.',
            limitations='Linear additive height does not explicitly supply floor-by-relative-object-state interactions; negative result cannot disprove floor information utility.'),
        learning_rates={**base.LR_BASE,**{name:base.LR_BASE[parent] for name,parent in FLOOR_GROUPS.items()}},
        restore=protocol['restore']+' New floor parameters/moments start at zero/no-state, ending at100; all original active Adam ends at2100.',
        optimization_change='Only add explicit observed floor-height zero branches to the matched fullbatch protocol; all four original schedules,20-microbatch objective,100-update budget and gates unchanged.',
        scope=original.FAILURE_AWARE_SCOPE+' Floor-only observation-adapter comparison against completed matched100 fullbatch control; no8force summary, no GT object min-z input, no new physics or tactile-benefit claim.')


def actual_source_bindings():
    result = original.actual_source_bindings()
    for name in EXTRA_SOURCES:
        path = Path(importlib.util.find_spec(name).origin).resolve()
        result[name] = dict(path=str(path),sha256=original.sha256_file(path))
    return result


def begin_artifacts(root):
    root = Path(root)
    manifest = dict(schema=3,kind=STUDY,complete=False,sources=actual_source_bindings(),
        protocol=dict(path='PROTOCOL.json',sha256=original.sha256_file(root/'PROTOCOL.json')),artifacts={})
    original.save_json(root/'ARTIFACTS.json',manifest)
    return manifest


def verify_artifacts(root, *, require_complete=True, manifest=None):
    root = Path(root)
    if manifest is None:manifest = json.loads((root/'ARTIFACTS.json').read_text())
    if manifest.get('schema') != 3 or manifest.get('kind') != STUDY or (require_complete and manifest.get('complete') is not True):
        raise ValueError('Require explicit floor-only artifact manifest')
    if set(manifest['sources']) != set(original.SOURCE_MODULES) | set(EXTRA_SOURCES):raise ValueError('Wrong floor-only source set')
    for name,binding in manifest['sources'].items():
        if original.sha256_file(binding['path']) != binding['sha256']:raise ValueError('Floor source changed: '+name)
    if manifest['protocol'] != dict(path='PROTOCOL.json',sha256=original.sha256_file(root/'PROTOCOL.json')):
        raise ValueError('Floor protocol changed')
    if require_complete:
        if set(manifest['artifacts']) != set(ARTIFACTS):raise ValueError('Incomplete floor artifacts')
        for name,binding in manifest['artifacts'].items():
            if binding != dict(path=name,sha256=original.sha256_file(root/name)):raise ValueError('Floor artifact changed: '+name)
    return manifest


def finish_artifacts(root, initial):
    root = Path(root)
    verify_artifacts(root,manifest=initial,require_complete=False)
    if actual_source_bindings() != initial['sources']:raise ValueError('Actual floor imports changed')
    final = dict(initial,complete=True,artifacts={name:dict(path=name,sha256=original.sha256_file(root/name)) for name in ARTIFACTS})
    original.save_json(root/'ARTIFACTS.json',final)
    return final


@torch.no_grad()
def evaluate_floor(model, dataset, full_vertices, device):
    """Independent batch1 evaluation on every declared clock, including failures."""
    model.eval()
    records = []
    vertices = torch.as_tensor(full_vertices, dtype=torch.float32)
    profile = getattr(dataset, 'supervision_profile', 'all_state')
    conditional = profile in (original.CONDITIONAL_PROFILE, original.FAILURE_AWARE_PROFILE)
    collator = collate_floor
    for index, row in enumerate(dataset.rows):
        batch = original.device_batch(collator([row]), device)
        output = model(batch['inputs'])
        prediction, target = output['state'].detach().cpu(), batch['target'].detach().cpu()
        predicted_mesh = original.state_vertices(prediction, vertices)[0].numpy()
        target_mesh = original.state_vertices(target, vertices)[0].numpy()
        if not all(np.isfinite(value).all() for value in (predicted_mesh, target_mesh, prediction.numpy())):
            raise FloatingPointError('Nonfinite evaluation geometry/state')
        distances = .5 * (original.cKDTree(target_mesh).query(predicted_mesh, workers=1)[0].mean()
                          + original.cKDTree(predicted_mesh).query(target_mesh, workers=1)[0].mean())
        predicted_rotation, target_rotation = original.rotation_matrices(prediction[:, 3:9]), original.rotation_matrices(target[:, 3:9])
        relative = predicted_rotation.transpose(1, 2) @ target_rotation
        angle = (((relative.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1).acos() * 180 / torch.pi)
        size_error = torch.expm1(prediction[:, 9:12] - target[:, 9:12]).abs()
        predicted_force = output['force'].detach().cpu().numpy()[0]
        true_force = original.force_target(batch['physics']).detach().cpu().numpy()[0]
        records.append(dict(episode=row['metadata']['episode'], frame=row['metadata']['frame'],
            timestamp_s=row['metadata']['timestamp_s'], floor_height_m=float(row['floor_height_m'][0]), prediction=prediction.numpy()[0], target=target.numpy()[0],
            force_prediction_n=predicted_force, force_target_n=true_force,
            center_cm=float((prediction[:, :3] - target[:, :3]).norm(dim=1)[0] * 100),
            mesh_nn_cm=float(distances * 100), rotation_deg=float(angle[0]),
            size_mean_relative=float(size_error.mean()), size_max_relative=float(size_error.max()),
            mass_relative=float(original.relative_mass_error(prediction, target)[0]),
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
    conditional_gate = original.failure_aware_acceptance if profile == original.FAILURE_AWARE_PROFILE else original.conditional_acceptance
    report = (conditional_gate(arrays, dataset.clock_role, original_acceptance=original.acceptance,
        fit_limits=original.FIT_LIMITS, interpolation_limits=original.INTERPOLATION_LIMITS, episodes=original.FIXED_EPISODES)
        if conditional else original.acceptance(arrays, dataset.clock_role))
    return report, arrays
