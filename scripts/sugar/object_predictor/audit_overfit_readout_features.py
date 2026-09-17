"""DIAGNOSTIC ONLY: exact endpoint replay and in-sample linear span analysis.

The full original Utonia executes 80 batch-one forwards, with zero updates.
CPU SVD fits are mathematical diagnostics, not replacement models, trained
predictors, acceptance results, or weights to deploy. No fitted weights are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STUDY = 'failure_aware_readout_feature_diagnostic_v1'
PROFILE = 'failure_aware_controlled_contact_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def feature_layout(state, history, enc_channels):
    """Read actual tensor shapes, cross-checking the unchanged official config."""
    head = tuple(state['predictor.head.weight'].shape)
    auxiliary = tuple(state['auxiliary.weight'].shape)
    if len(head) != 2 or head[0] != 13 or head[1] <= 0:
        raise ValueError('Malformed actual state readout tensor')
    dimension = int(head[1])
    if (auxiliary != (10, dimension)
            or tuple(state['predictor.head.bias'].shape) != (13,)
            or tuple(state['auxiliary.bias'].shape) != (10,)):
        raise ValueError('Actual state and auxiliary readout shapes disagree')
    channels = [int(value) for value in enc_channels]
    if (history != 32 or not channels or min(channels) <= 0
            or dimension != 2 * sum(channels) * history):
        raise ValueError('Actual readout width disagrees with official channels/history')
    return dict(feature_dimension=dimension, head_weight=list(head),
        auxiliary_weight=list(auxiliary), history=history, enc_channels=channels,
        frame_feature_dimension=dimension//history,
        source='Actual endpoint readout tensors, cross-checked against released official enc_channels and original H32')


def checkpoint_feature_layout(source, training_protocol):
    """CPU tensor metadata only; do not construct or forward any model."""
    import torch
    endpoint = torch.load(Path(source)/'model.pt', map_location='cpu', weights_only=False, mmap=True)
    released = torch.load(training_protocol['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    return feature_layout(endpoint['model'], training_protocol['data']['history'],
                          released['config']['enc_channels'])


def protocol(source):
    source = Path(source).resolve()
    p = json.loads((source/'PROTOCOL.json').read_text())
    r = json.loads((source/'RESULT.json').read_text())
    if (p.get('supervision_profile') != PROFILE or r.get('execution_complete') is not True
            or r.get('optimizer_updates') != 2000 or r.get('full_endpoint_reload_max_abs') != 0
            or r.get('full_parameter_update_passed') is not True):
        raise ValueError('Require the completed actual fixed failure-aware endpoint')
    layout = checkpoint_feature_layout(source, p)
    return dict(study=STUDY, source=str(source), profile=PROFILE,
        model_artifact_manifest_sha256=digest(source/'ARTIFACTS.json'),
        source_result_sha256=digest(source/'RESULT.json'),
        adapter_sha256=digest(__file__),
        fixed_episodes=p['data']['episodes'], fixed_frames=p['data']['frames'],
        fixed_rows=80, feature_dimension=layout['feature_dimension'], feature_layout=layout, history=32, batch=1,
        full_original_model=True, strict_replay='Exact array equality for all state13, force8 and both sigmoid probabilities against actual fit_02000.npz; no tolerance or truth correction.',
        runtime='Original eval(), deterministic pooling, disabled stochastic regularizers, float32 and TF32 enabled; same retained resource and source manifest.',
        masks='Observation-only H32 state candidate mask for center/logsize; original stable-support mass availability for logmass; every row for observed force8. All80 captured and retained.',
        linear_targets=dict(center='hand-relative meters,3', logsize='log meters,3',
            logmass='log kilograms,1', force='original observation-derived Newton targets,8'),
        solver='CPU float64 SVD of ORIGINAL features plus a bias column, no feature normalization. Moore-Penrose cutoff=max(design.shape)*eps64*smax. Also report conventional eps32 rank indicator and coefficient-rounded CPU float32 residuals; these are not CUDA/TF32 equivalence guarantees.',
        interpretation='In-sample span/conditioning/optimization diagnostic only. Actual full-readout features greatly exceed63/43/80 rows; exact interpolation can be memorization or numerically unstable. No holdout, geometry/rotation gate, generalization or tactile-benefit claim.',
        fitted_weights_saved=False, fitted_weights_deployed=False,
        optimizer_updates=0, physics_controls=0, planned_model_forwards=80)


def validate_protocol(source, output):
    actual = json.loads((Path(output)/'PROTOCOL.json').read_text())
    expected = protocol(source)
    if actual != expected:
        raise ValueError('Prepared readout diagnostic protocol/source changed')
    return actual


def check_saved_rows(dataset, saved):
    """GT remains target/mask/evaluation only; inputs are the unchanged allowlist."""
    import torch
    from .overfit_model import force_target, FORCE_TARGET_FIELDS
    from .overfit_data import FIXED_EPISODES, FIXED_FRAMES
    clocks = [(r['metadata']['episode'], r['metadata']['frame']) for r in dataset.rows]
    if clocks != [(e,f) for e in FIXED_EPISODES for f in FIXED_FRAMES] or len(clocks) != 80:
        raise ValueError('Require all original eighty fixed rows in order')
    physics = {key:torch.from_numpy(np.stack([r['supervision']['physics'][key]
        for r in dataset.rows])) for key,_width in FORCE_TARGET_FIELDS}
    fresh = dict(episode=np.array([e for e,f in clocks]), frame=np.array([f for e,f in clocks]),
        target=np.stack([r['target'] for r in dataset.rows]),
        mass_available=np.array([r['supervision']['mass_available'] for r in dataset.rows]),
        state_precision_eligible=np.array([r['supervision']['state_precision_eligible'] for r in dataset.rows]),
        state_contact_history_frames=np.array([r['supervision']['state_contact_history_frames'] for r in dataset.rows]),
        force_target_n=force_target(physics).numpy())
    if any(not np.array_equal(value, saved[key]) for key,value in fresh.items()):
        raise ValueError('Actual clocks/targets/masks differ from saved endpoint fit')
    return fresh


def replay_comparison(actual, saved):
    fields = ('prediction', 'force_prediction_n', 'availability_probability', 'contact_probability')
    checks, differences = {}, {}
    for key in fields:
        a, b = np.asarray(actual[key]), np.asarray(saved[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Malformed/nonfinite replay output: '+key)
        checks[key] = bool(np.array_equal(a, b))
        differences[key] = float(np.max(abs(a.astype(np.float64)-b.astype(np.float64))))
    return dict(passed=all(checks.values()), exact_checks=checks, max_abs_difference=differences)


def forward_observations(model, batch):
    """Use the original model boundary; labels and masks cannot enter forward."""
    inputs = batch['inputs']
    if set(inputs) != {'coord', 'grid_coord', 'feat', 'offset'}:
        raise ValueError('Model input must contain exactly the original observation allowlist')
    return model(inputs)


def residual_statistics(residual, task):
    residual = np.asarray(residual)
    if not np.isfinite(residual).all():
        return dict(finite=False, nonfinite_elements=int((~np.isfinite(residual)).sum()))
    result = dict(finite=True, component_rmse=float(np.sqrt(np.mean(residual**2))),
                  component_max_abs=float(abs(residual).max()))
    if task == 'center':
        values = np.linalg.norm(residual, axis=1)*100
        result['center_error_cm'] = dict(mean=float(values.mean()), maximum=float(values.max()))
    elif task in ('logsize', 'logmass'):
        with np.errstate(over='ignore', invalid='ignore'):
            relative = abs(np.expm1(residual))
        result['relative_error_finite'] = bool(np.isfinite(relative).all())
        result['relative_error'] = (dict(mean=float(relative.mean()), maximum=float(relative.max()))
                                    if np.isfinite(relative).all() else None)
    elif task == 'force':
        result['force_rmse_n'] = result['component_rmse']
    return result


def linear_span_diagnostic(features, targets, mask, task):
    """Return residuals, never coefficients; no predictor/model is produced."""
    from scipy.linalg import svd
    features, targets, mask = np.asarray(features), np.asarray(targets), np.asarray(mask)
    if (features.ndim != 2 or targets.ndim != 2 or len(targets) != len(features)
            or mask.shape != (len(features),) or not np.isin(mask, (0,1)).all()
            or not np.isfinite(features).all()):
        raise ValueError('Malformed original feature/target/mask matrices')
    selected = mask.astype(bool)
    if not selected.any():
        return dict(status='NOT_APPLICABLE', rows=0), np.empty_like(targets[:0]), np.empty_like(targets[:0])
    x, y = features[selected].astype(np.float64), targets[selected].astype(np.float64)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite selected diagnostic target')
    design = np.column_stack((x, np.ones(len(x))))
    u, singular, vt = svd(design, full_matrices=False, check_finite=True, lapack_driver='gesdd')
    cutoff = max(design.shape)*np.finfo(np.float64).eps*singular[0]
    keep = singular > cutoff
    rank = int(keep.sum())
    coefficients = vt[keep].T @ ((u[:,keep].T@y)/singular[keep,None])
    residual64 = design@coefficients-y
    # This is CPU float32 arithmetic, not an assertion of CUDA/TF32 equivalence.
    residual32 = (design.astype(np.float32)@coefficients.astype(np.float32)).astype(np.float64)-y
    cutoff32 = max(design.shape)*np.finfo(np.float32).eps*singular[0]
    result = dict(status='DIAGNOSTIC_ONLY', rows=len(x), original_features=x.shape[1],
        affine_design_columns=design.shape[1], rank_float64=rank,
        cutoff_float64=float(cutoff), rank_float32_roundoff_indicator=int((singular>cutoff32).sum()),
        cutoff_float32_roundoff_indicator=float(cutoff32), singular_values=singular.tolist(),
        condition_retained=float(singular[0]/singular[keep][-1]),
        full_row_condition=float(singular[0]/singular[-1]) if rank==len(x) else None,
        coefficient_frobenius_norm=float(np.linalg.norm(coefficients)),
        coefficient_max_abs=float(abs(coefficients).max()),
        float64_residual=residual_statistics(residual64,task),
        coefficient_rounded_cpu_float32_residual=residual_statistics(residual32,task),
        note='All fitted rows are the diagnostic training rows, not holdout. Float32 rank indicator is a conventional dimensional roundoff threshold, not proof of information loss. No fitted coefficients are saved/deployed.')
    return result, residual64, residual32


def analyze_capture(capture, output, feature_dimension):
    x = capture['features']
    if x.shape != (80, feature_dimension) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Require eighty complete original checkpoint-width float32 features')
    tasks = dict(center=(capture['target'][:,:3],capture['state_precision_eligible']),
        logsize=(capture['target'][:,9:12],capture['state_precision_eligible']),
        logmass=(capture['target'][:,12:13],capture['mass_available']),
        force=(capture['force_target_n'],np.ones(80,bool)))
    reports, residuals = {}, {}
    for task,(target,mask) in tasks.items():
        report,r64,r32 = linear_span_diagnostic(x,target,mask,task)
        reports[task] = report
        residuals[task+'_row_indices'] = np.flatnonzero(mask)
        residuals[task+'_residual_float64'] = r64
        residuals[task+'_residual_cpu_float32'] = r32
    norms = dict(row_l2=np.linalg.norm(x.astype(np.float64),axis=1).tolist(),
        row_l1=abs(x.astype(np.float64)).sum(1).tolist(), row_max_abs=abs(x).max(1).tolist())
    np.savez_compressed(Path(output)/'LINEAR_PROJECTION_RESIDUALS.npz', **residuals)
    save_json(Path(output)/'LINEAR_SPAN_DIAGNOSTIC.json', dict(tasks=reports, feature_norms=norms,
        no_holdout=True, replacement_model=False, model_acceptance_claimed=False,
        fitted_weights_saved=False, fitted_weights_deployed=False,
        rotation_or_mesh_acceptance_tested=False))
    return reports


def run(source, output, *, check=False):
    import torch
    from .train_overfit import verify_artifacts, device_batch
    from .failure_aware_overfit import FailureAwareOverfitDataset
    from .conditional_overfit_data import collate_conditional
    source,output = Path(source).resolve(),Path(output)
    declared = validate_protocol(source,output)
    feature_dimension = declared['feature_dimension']
    verify_artifacts(source)
    if any((output/name).exists() for name in ('FEATURES.npz','RESULT.json','FAILURE.json')):
        raise FileExistsError('Use a fresh diagnostic output; no overwrite or automatic retry')
    if not check:
        from .retained_execution import require_active_resource
        resource = require_active_resource()
    p = json.loads((source/'PROTOCOL.json').read_text())
    dataset = FailureAwareOverfitDataset(p['source_data'],p['data'])
    with np.load(source/'fit_02000.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
    fresh = check_saved_rows(dataset,saved)
    if check:
        result=dict(cpu_inputs_checked=True, rows=len(dataset), feature_forwards=0,
            checkpoint_feature_layout=declared['feature_layout'],
            state_candidate_rows=int(fresh['state_precision_eligible'].sum()),
            available_mass_rows=int(fresh['mass_available'].sum()),cuda_initialized=torch.cuda.is_initialized())
        if result['cuda_initialized']:raise RuntimeError('CPU check initialized CUDA')
        save_json(output/'CPU_INPUT_CHECK.json',result);print(json.dumps(result),flush=True)
        return 0
    from .overfit_model import FullUtoniaOverfit
    from .model import disable_stochastic_regularizers
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    model = FullUtoniaOverfit(p['checkpoint'],history=32).to('cuda')
    endpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(endpoint['model'],strict=True);del endpoint
    model.eval();disable_stochastic_regularizers(model)
    if model.predictor.head.in_features != feature_dimension or model.auxiliary.in_features != feature_dimension:
        raise ValueError('Original complete chronological readout width changed')
    captures,outputs=[],[]
    def hook(_module,args):captures.append(args[0].detach().cpu().numpy().copy())
    handle=model.predictor.head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for index,row in enumerate(dataset.rows):
                before=len(captures)
                batch=device_batch(collate_conditional([row]),torch.device('cuda'))
                values=forward_observations(model,batch)
                if len(captures)!=before+1 or captures[-1].shape!=(1,feature_dimension):
                    raise RuntimeError('Expected exactly one unchanged original head feature per row')
                outputs.append(dict(prediction=values['state'].cpu().numpy()[0],
                    force_prediction_n=values['force'].cpu().numpy()[0],
                    availability_probability=float(values['availability_logit'].sigmoid()[0]),
                    contact_probability=float(values['contact_logit'].sigmoid()[0])))
                if (index+1)%10==0:print(json.dumps(dict(event='diagnostic_forward',rows=index+1,total=80)),flush=True)
    finally:handle.remove()
    actual={key:np.asarray([row[key] for row in outputs]) for key in outputs[0]}
    replay=replay_comparison(actual,saved)
    capture=dict(fresh,**actual,features=np.concatenate(captures))
    np.savez_compressed(output/'FEATURES.npz',**capture)
    save_json(output/'REPLAY.json',replay)
    if not replay['passed']:
        save_json(output/'RESULT.json',dict(execution_complete=False,strict_replay_passed=False,
            optimizer_updates=0,physics_controls=0,model_forwards=80,linear_diagnostic_executed=False))
        return 2
    del model;torch.cuda.empty_cache()
    reports=analyze_capture(capture,output,feature_dimension)
    verify_artifacts(source);validate_protocol(source,output)
    result=dict(execution_complete=True,study=STUDY,strict_replay_passed=True,
        model_forwards=80,optimizer_updates=0,physics_controls=0,retained_resource=resource,
        original_model_acceptance=json.loads((source/'RESULT.json').read_text())['acceptance_passed'],
        fixed_rows=80,feature_dimension=feature_dimension,task_rows={k:v['rows'] for k,v in reports.items()},
        replacement_model=False,fitted_weights_saved=False,fitted_weights_deployed=False,
        generalization_or_tactile_benefit_claimed=False,
        scope=declared['interpretation'])
    save_json(output/'RESULT.json',result);print(json.dumps(result),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare',action='store_true')
    actions.add_argument('--check',action='store_true',help='CPU source/input comparison only, no model construction')
    args=parser.parse_args()
    if args.prepare:
        args.output.mkdir(parents=True,exist_ok=False)
        save_json(args.output/'PROTOCOL.json',protocol(args.source));return
    try:code=run(args.source,args.output,check=args.check)
    except Exception as error:
        if args.output.exists() and not isinstance(error,FileExistsError):
            save_json(args.output/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),
                replacement_model=False,automatic_retry=False))
        raise
    raise SystemExit(code)


if __name__=='__main__':main()
