"""One fixed continuation of the unchanged native geometry objective.

Only the completed 1000-update geometry run can resume. Exactly 3000 new
updates reach absolute step 4000, with the original model, Adam moments,
seed+absolute-step sampling, eight cases, losses and numerical thresholds.
These helpers perform no model inference and contain no replacement network.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


START = 1000
END = 4000
ADDITIONAL = END-START


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_resume(args, public_rows, selection, require_engine=True):
    from . import active3d_geometry_loss as geometry

    source = Path(args.resume_from).resolve()
    protocol = json.loads((source/'PROTOCOL.json').read_text())
    result = json.loads((source/'RESULT.json').read_text())
    require(args.geometry_repair, 'Continuation requires the unchanged geometry objective')
    require(protocol.get('fixed_updates') == START and not protocol.get('continuation'),
            'Only the original complete 1000-update run may resume; no chained extension')
    require(result.get('complete') is True and result.get('optimizer_updates') == START
            and result.get('backwards') == START and result.get('fixed_input_cases') == 8
            and result.get('checkpoint_reload', {}).get('passed') is True,
            'Source must preserve all eight cases and complete 1000 updates with reload')
    require(protocol.get('cases') == public_rows and all(protocol.get(k) == v for k, v in selection.items()),
            'Fixed native inputs, source bindings or selection changed')
    require(protocol.get('seed') == args.seed
            and Path(protocol['dataset']).resolve() == args.dataset.resolve()
            and Path(protocol['checkpoint']).resolve() == args.checkpoint.resolve(),
            'Continuation seed/dataset/released initialization must match')
    require(protocol.get('geometry_repair') == geometry.configuration()
            and protocol.get('geometry_repair_source_sha256') == digest(geometry.__file__),
            'Geometry objective source/configuration must remain unchanged')
    require(protocol.get('optimizer') == dict(name='Adam', learning_rate=3e-4, weight_decay=0)
            and (protocol.get('samples'), protocol.get('repeats'), protocol.get('loss_scale')) == (30000, 3, 9000.)
            and protocol.get('batch_size') == 8 and protocol.get('report_every') == 100,
            'Original optimizer/loss/batch/report contract changed')
    checkpoint = source/f'step_{START:04d}'/'model'
    optimizer = checkpoint.with_name('optim')
    require(checkpoint.resolve() == Path(result['checkpoint']).resolve()
            and digest(checkpoint) == result['checkpoint_sha256'], 'Source endpoint binding mismatch')
    receipt = dict(source=str(source), start_absolute_step=START, end_absolute_step=END,
                   additional_updates=ADDITIONAL, source_model_sha256=digest(checkpoint),
                   source_optimizer_sha256=digest(optimizer), source_result_sha256=digest(source/'RESULT.json'),
                   source_protocol_sha256=digest(source/'PROTOCOL.json'),
                   absolute_seed_rule='original_seed + absolute_step',
                   original_initial_metrics='Copied unchanged released step_0000, not new inference',
                   source_scientific_gate=result.get('native_repair_numeric_gate_passed'))
    verification = getattr(args, 'engine_verification', None)
    if require_engine:
        require(verification is not None, 'Actual continuation requires --engine-verification RESULT.json')
        verification = Path(verification).resolve()
        engine = json.loads(verification.read_text())
        require(engine.get('complete') is True and engine.get('passed') is True
                and engine.get('input_cases') == 8 and engine.get('model_forwards') == 1
                and engine.get('optimizer_updates') == 0 and engine.get('backwards') == 0
                and Path(engine['compared_source']).resolve() == source
                and engine.get('released_checkpoint_sha256') == protocol['checkpoint_sha256'],
                'Require actual original Engine eight-input equivalence PASS for this source')
        expected = {row['name'] for row in public_rows}
        records = engine.get('cases', [])
        require(len(records) == 8 and {row['name'] for row in records} == expected
                and all(row.get('passed') is True for row in records),
                'Engine receipt must retain all eight passing cases')
        receipt.update(engine_verification=str(verification), engine_verification_sha256=digest(verification),
                       engine_equivalence_passed=True)
    return dict(source=source, protocol=protocol, result=result, model=checkpoint,
                optimizer=optimizer, receipt=receipt)


def nested_equal(a, b):
    import torch

    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(nested_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(nested_equal(x, y) for x, y in zip(a, b)))
    return a == b


def restore(model, optimizer, resume):
    """Restore and compare all saved weights, moments, step counters and groups."""
    import torch

    require(digest(resume['model']) == resume['receipt']['source_model_sha256']
            and digest(resume['optimizer']) == resume['receipt']['source_optimizer_sha256'],
            'Source checkpoint changed after preflight')
    weights = torch.load(resume['model'], map_location='cpu', weights_only=True)
    moments = torch.load(resume['optimizer'], map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    optimizer.load_state_dict(moments)
    require(nested_equal(model.state_dict(), weights), 'Restored complete model differs')
    require(nested_equal(optimizer.state_dict(), moments), 'Restored complete Adam differs')
    parameters = list(model.parameters())
    require(len(optimizer.state) == len(parameters), 'Every official parameter requires its saved Adam state')
    for parameter in parameters:
        state = optimizer.state[parameter]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and float(state['step']) == START,
                'Every Adam counter must be exactly 1000')
        for key in ('exp_avg', 'exp_avg_sq'):
            require(state[key].shape == parameter.shape and bool(torch.isfinite(state[key]).all()),
                    'Invalid full-parameter Adam moment')
    require(all(group['lr'] == 3e-4 and group['weight_decay'] == 0
                and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
                for group in optimizer.param_groups), 'Saved Adam hyperparameters changed')
    return weights


def copy_initial_and_check_inputs(resume, output, rows, graph, initial_mesh):
    import numpy as np

    source = resume['source']
    with np.load(source/'GRAPH.npz', allow_pickle=False) as old:
        require(np.array_equal(old['initial_mesh'], initial_mesh.detach().cpu().numpy()), 'Initial template changed')
        require(all(np.array_equal(old[key], value.detach().cpu().numpy()) for key, value in graph.items()),
                'Full original graph changed')
    destination = output/'step_0000'
    destination.mkdir()
    for row in rows:
        filename = row['name']+'.npz'
        with np.load(source/'step_0000'/filename, allow_pickle=False) as old:
            for key, current in (('input_touch_charts', row['touch']), ('target_points_canonical', row['target']),
                                 ('truth_vertices_canonical', row['truth_vertices']), ('truth_faces', row['truth_faces'])):
                require(np.array_equal(old[key], current), 'Original case input/label changed: '+row['name']+'/'+key)
        shutil.copyfile(source/'step_0000'/filename, destination/filename)
    shutil.copyfile(source/'step_0000/METRICS.json', destination/'METRICS.json')
    return json.loads((destination/'METRICS.json').read_text())


def check_replay(resume, output, rows):
    import numpy as np

    maximum = 0.
    for row in rows:
        relative = Path(f'step_{START:04d}')/(row['name']+'.npz')
        with np.load(resume['source']/relative, allow_pickle=False) as before, np.load(output/relative, allow_pickle=False) as after:
            for key in ('observed_vertices_canonical', 'empty_vertices_canonical'):
                difference = float(np.max(np.abs(before[key]-after[key])))
                require(np.isfinite(difference) and difference <= 1e-7, 'Actual resumed replay differs: '+row['name']+'/'+key)
                maximum = max(maximum, difference)
            for key in ('observed_masks', 'empty_masks'):
                require(np.array_equal(before[key], after[key]), 'Actual resumed masks differ')
    return dict(passed=True, full_model_and_adam_exact=True, all_eight_observed_and_empty=True,
                output_max_abs=maximum, absolute_step=START, optimizer_updates=0)


def formal_continuation_gate(training, rendering=None, inspection=None):
    """Same scientific/render checks; explicitly account for 1000 + 3000 clocks.

    The original geometry module is unchanged. Only its original-run clock
    check is replaced for this separately declared continuation protocol.
    """
    from .active3d_geometry_loss import formal_repair_gate

    result = formal_repair_gate(training, rendering, inspection)
    c = training.get('continuation', {})
    rendered = rendering or {}
    result['checks']['actual_complete_render'] = bool(result['checks']['actual_complete_render']
        and rendered.get('rendered_endpoint') == 'step_4000'
        and rendered.get('training_total_optimizer_updates') == END
        and rendered.get('training_new_optimizer_updates') == ADDITIONAL)
    result['checks']['complete_fixed_training'] = bool(
        training.get('complete') is True and training.get('fixed_input_cases') == 8
        and training.get('optimizer_updates') == ADDITIONAL and training.get('backwards') == ADDITIONAL
        and training.get('total_optimizer_updates') == END
        and c.get('start_absolute_step') == START and c.get('end_absolute_step') == END
        and c.get('additional_updates') == ADDITIONAL and c.get('engine_equivalence_passed') is True
        and training.get('resume_replay', {}).get('passed') is True
        and training.get('checkpoint_reload', {}).get('passed') is True)
    result['passed'] = all(result['checks'].values())
    return result
