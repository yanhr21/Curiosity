"""One declared least-squares update to the existing eight force-readout rows.

The complete official model is retained. This helper adds no learned module.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import train_overfit as original
from . import overfit_fullbatch_refinement as baseline
from . import audit_overfit_readout_features as capture

STUDY = 'failure_aware_force_head_solve_v1'
SOURCE_UPDATES = 2100
FORCE_ROWS = 8
EXTRA_SOURCES = ('scripts.sugar.object_predictor.audit_overfit_readout_features',
    'scripts.sugar.object_predictor.overfit_force_head_solve',
    'scripts.sugar.object_predictor.train_overfit_force_head_solve')
ARTIFACTS = ('model.pt', 'initial_fit.npz', 'fit_force_update_0001.npz',
    'initial_same_trajectory_interpolation.npz', 'same_trajectory_interpolation.npz',
    'FRESH_FEATURES.npz', 'SOLVE.json', 'SOURCE_RELOAD_COMPARISON.json',
    'NON_FORCE_SAME_FEATURES.json', 'PARAMETER_PRESERVATION.json', 'UPDATE_LEDGER.json')


def minimum_increment(features, targets, weight, bias):
    """Unscaled affine Moore-Penrose minimum *increment*, float64 -> float32.

    No ridge, clipping, spectrum tuning or old fitted coefficients. The extra
    column is the existing head bias, so its Euclidean weight is explicitly1.
    """
    from scipy.linalg import svd
    x, y, w, b = [np.asarray(v) for v in (features, targets, weight, bias)]
    if (x.ndim != 2 or y.shape != (len(x), FORCE_ROWS)
            or w.shape != (FORCE_ROWS, x.shape[1]) or b.shape != (FORCE_ROWS,)
            or not len(x) or x.shape[1] < 1
            or any(not np.isfinite(v).all() for v in (x, y, w, b))):
        raise ValueError('Malformed/nonfinite original force-head arrays')
    if x.dtype != np.float32 or w.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError('Require actual float32 captured features and original weights')
    a = np.column_stack((x.astype(np.float64), np.ones(len(x))))
    old = np.column_stack((w.astype(np.float64), b.astype(np.float64)))
    target = y.astype(np.float64)
    u, singular, vt = svd(a, full_matrices=False, check_finite=True, lapack_driver='gesdd')
    cutoff = max(a.shape) * np.finfo(np.float64).eps * singular[0]
    keep = singular > cutoff
    if not keep.any():
        raise ValueError('No retained affine design direction')
    residual = target - a @ old.T
    delta = ((vt[keep].T / singular[keep]) @ (u[:, keep].T @ residual)).T
    candidate64 = old + delta
    with np.errstate(over='ignore', invalid='ignore'):
        candidate = candidate64.astype(np.float32)
    if not np.isfinite(delta).all() or not np.isfinite(candidate).all():
        raise FloatingPointError('Nonfinite float64 solution or float32 application; no fallback')
    actual_delta = candidate.astype(np.float64) - old
    row_basis = vt[keep]
    null64 = delta - (delta @ row_basis.T) @ row_basis
    null32 = actual_delta - (actual_delta @ row_basis.T) @ row_basis
    rounded_residual = a @ candidate.astype(np.float64).T - target
    with np.errstate(over='ignore',invalid='ignore'):
        cached32 = a.astype(np.float32) @ candidate.T - y.astype(np.float32)
    if not np.isfinite(rounded_residual).all() or not np.isfinite(cached32).all():
        raise FloatingPointError('Nonfinite rounded cached-head arithmetic; no application')
    report = dict(rows=len(x), features=x.shape[1], affine_columns=a.shape[1],
        rank=int(keep.sum()), cutoff_float64=float(cutoff), singular_values=singular.tolist(),
        retained_condition=float(singular[0] / singular[keep][-1]),
        retained_condition_times_eps32=float(singular[0] / singular[keep][-1] * np.finfo(np.float32).eps),
        condition_is_report_only=True, delta_frobenius_float64=float(np.linalg.norm(delta)),
        delta_max_abs_float64=float(abs(delta).max()),
        applied_delta_frobenius_float32=float(np.linalg.norm(actual_delta)),
        applied_parameter_max_abs=float(abs(candidate).max()),
        exact_increment_nullspace_leak_l2=float(np.linalg.norm(null64)),
        float32_rounding_nullspace_leak_l2=float(np.linalg.norm(null32)),
        solve_residual_rmse_n=float(np.sqrt(np.mean((a @ candidate64.T - target)**2))),
        rounded_parameters_cpu64_residual_rmse_n=float(np.sqrt(np.mean(rounded_residual**2))),
        rounded_parameters_cpu32_residual_rmse_n=float(np.sqrt(np.mean(cached32.astype(np.float64)**2))),
        nullspace_scope='The float64 minimum increment lies in the retained row span; float32 casting can perturb the old nullspace and is measured, not claimed bitexact.',
        optimizer_updates=0, learning_updates=1, closed_form_parameter_updates=1,
        algorithm='Unscaled affine float64 scipy.linalg.svd(gesdd), rcond=max(A.shape)*eps64; minimum Euclidean weight-and-bias increment; one float32 cast.',
        numerical_failure_policy='Finite is mandatory. Rank/condition/delta are recorded without tuned rejection or automatic truncation/ridge/retry; actual cached-head and full-forward original gates decide success.')
    return candidate[:, :-1].copy(), candidate[:, -1].copy(), report


def apply_force_rows(model, weight, bias):
    """Update exactly the existing rows0:8; no replacement head or architecture."""
    if (model.auxiliary.weight.ndim != 2 or model.auxiliary.weight.shape[0] != 10
            or model.auxiliary.bias.shape != (10,)):
        raise ValueError('Require the original ten-output auxiliary head')
    w = torch.as_tensor(weight, dtype=model.auxiliary.weight.dtype, device=model.auxiliary.weight.device)
    b = torch.as_tensor(bias, dtype=model.auxiliary.bias.dtype, device=model.auxiliary.bias.device)
    if w.shape != model.auxiliary.weight[:8].shape or b.shape != (8,):
        raise ValueError('Force update shape differs')
    if not bool(torch.isfinite(w).all() and torch.isfinite(b).all()):
        raise FloatingPointError('Nonfinite force update')
    with torch.no_grad():
        model.auxiliary.weight[:8].copy_(w)
        model.auxiliary.bias[:8].copy_(b)


def preserved_state(source, current):
    """Every original state tensor bitexact except the declared eight rows."""
    if set(source) != set(current):
        raise ValueError('Model state keys changed')
    rows = []
    for name, value in source.items():
        after = current[name].detach().cpu()
        before = value.detach().cpu()
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError('Original model tensor shape/dtype changed: ' + name)
        if not bool(torch.isfinite(after).all()):
            raise FloatingPointError('Nonfinite endpoint tensor: ' + name)
        allowed = name in ('auxiliary.weight', 'auxiliary.bias')
        def tensor_hash(v):
            return hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        before_hash=tensor_hash(before[8:] if allowed else before)
        after_hash=tensor_hash(after[8:] if allowed else after)
        exact = before_hash==after_hash
        if not exact:
            raise ValueError('Forbidden tensor/auxiliary row changed: ' + name)
        rows.append(dict(name=name, source_equals_endpoint=bool(torch.equal(before,after)),
            checked_source_sha256=before_hash,checked_endpoint_sha256=after_hash,
            checked_non_force_elements=after[8:].numel() if allowed else after.numel(),
            permitted_force_elements=after[:8].numel() if allowed else 0))
    return dict(passed=True, named_state_tensors=rows, all_non_force_tensors_exact=True,
        original_two_other_auxiliary_rows_exact=True, full_official_backbone_updated=False,
        learning_updates=1, optimizer_updates=0)


def source_protocol(source):
    source = Path(source).resolve()
    baseline.verify_artifacts(source)
    p = json.loads((source/'PROTOCOL.json').read_text())
    r = json.loads((source/'RESULT.json').read_text())
    if (p.get('study') != baseline.STUDY or r.get('execution_complete') is not True
            or r.get('total_optimizer_updates') != 2100 or r.get('optimizer_updates') != 100
            or r.get('full_parameter_update_passed') is not True or r.get('full_endpoint_reload_max_abs') != 0):
        raise ValueError('Require actual completed no-floor2100 full-model refinement')
    layout = capture.checkpoint_feature_layout(source, p)
    bindings = {name:dict(path=str(source/name), sha256=original.sha256_file(source/name))
        for name in ('model.pt','PROTOCOL.json','RESULT.json','ARTIFACTS.json',
                     'fit_02100.npz','fit_02100.json','same_trajectory_interpolation.npz','same_trajectory_interpolation.json')}
    return dict(study=STUDY, source_endpoint=str(source), source_bindings=bindings,
        prepared_adapter_sources=actual_source_bindings(),
        checkpoint=p['checkpoint'], source_data=p['source_data'], data=p['data'], mesh=p['mesh'], seed=p['seed'],
        supervision_profile=original.FAILURE_AWARE_PROFILE, feature_layout=layout,
        source_optimizer_updates=2100, optimizer_updates=0, learning_updates=1,
        closed_form_parameter_updates=1, total_adam_optimizer_updates=2100,
        total_learning_updates=2101, full_official_model=True, full_backbone_updated=False,
        fixed_fit_rows=80, interpolation_frames=list(original.INTERPOLATION_FRAMES), fixed_interpolation_rows=1440,
        fit_limits=original.FIT_LIMITS, interpolation_limits=original.INTERPOLATION_LIMITS,
        force_target_fields=[list(value) for value in original.FORCE_TARGET_FIELDS],
        force_targets='Fresh original observation-derived eight Newton targets, all80 rows; never GT object labels in model inputs.',
        input_allowlist=['coord','grid_coord','feat','offset'], history=32, history_interval_s=.02,
        model_arithmetic='Original eval/batch1 full official Utonia, float32, TF32 enabled, original deterministic pooling and disabled stochastic regularizers.',
        solver=dict(method='scipy.linalg.svd', driver='gesdd', arithmetic='float64',
            design='Fresh80 actual2100 original88704 features, unscaled, append one for original bias',
            cutoff='max(A.shape)*eps64*smax', update='delta=A_pinv@(Y-A@W0); apply one cast of W0+delta to float32',
            rank_or_condition_sweep=False, ridge=False, clipping=False, automatic_retry=False,
            weight_and_bias_metric='Unscaled Euclidean/Frobenius minimum increment; bias column has weight1.',
            finite_required=True, condition_report_only=True,
            provenance='Same fixed float64 MP tolerance as saved conditioning study; CPUfloat32 representability was demonstrated on source2000, but fresh2100 is not presumed identical.'),
        unchanged='Every tensor except auxiliary.weight[:8] and auxiliary.bias[:8] is bitexact; preserve old row-nullspace mathematically, measure finite-precision leakage; never modify13-state or2classification rows.',
        update_scope='One closed-form learning update to original force-readout rows, not zero training, full-backbone optimization, replacement model, or import of old SVD coefficients.',
        adam_archive='Save original complete optimizer state unchanged under optimizer_archive; explicitly not resume-compatible after the non-Adam update.',
        initial_control='Fresh80 full forwards compared with source historical fit; all exact flags and differences saved, no cross-process bitidentity claim. Before dense is the explicitly historical saved source2100 evaluation.',
        evaluation='After one update actual80+1440 ordinary full forwards, original all-task gates/raw metrics; same-feature original non-force outputs must be exact. No dense labels/features used to choose solver/rank.',
        full_forward_budget=1604, forward_budget_detail=dict(fresh_fit_capture=80, after_fit=80, after_dense=1440,
            initial_same_process_source_reload_pair=2, final_checkpoint_reload_pair=2),
        readout_only_calls='Two80-row existing head/auxiliary readouts on identical fresh cached features to certify non-force outputs; not full-model forwards.',
        failure_policy='Preserve all16/80/1440 and every failure. Nonfinite prevents application; no automatic regularization, rank change, retry, or extension. Original every-row/per-case force gates remain.',
        scope='Failure-aware controlled-scene readout qualification on TRAIN fit and same-trajectory interpolation. PhysicalFAIL, prior/unknown states and missing mass denominators remain. No overall repair, arbitrary shape, generalization or tactile-benefit claim.')


def actual_source_bindings():
    result = baseline.actual_source_bindings()
    for name in EXTRA_SOURCES:
        path = Path(importlib.util.find_spec(name).origin).resolve()
        result[name] = dict(path=str(path), sha256=original.sha256_file(path))
    return result


def begin_artifacts(root):
    root = Path(root)
    manifest = dict(schema=5, kind=STUDY, complete=False, sources=actual_source_bindings(),
        protocol=dict(path='PROTOCOL.json', sha256=original.sha256_file(root/'PROTOCOL.json')), artifacts={})
    original.save_json(root/'ARTIFACTS.json', manifest)
    return manifest


def verify_artifacts(root, *, require_complete=True, manifest=None):
    root = Path(root)
    manifest = manifest or json.loads((root/'ARTIFACTS.json').read_text())
    if manifest.get('schema') != 5 or manifest.get('kind') != STUDY or (require_complete and not manifest.get('complete')):
        raise ValueError('Wrong/incomplete force-head study manifest')
    if manifest['sources'] != actual_source_bindings():
        raise ValueError('Actual force-head study source changed')
    if manifest['protocol'] != dict(path='PROTOCOL.json',sha256=original.sha256_file(root/'PROTOCOL.json')):
        raise ValueError('Force-head protocol changed')
    if require_complete:
        if set(manifest['artifacts']) != set(ARTIFACTS):raise ValueError('Incomplete force-head artifacts')
        for name,binding in manifest['artifacts'].items():
            if binding != dict(path=name,sha256=original.sha256_file(root/name)):
                raise ValueError('Changed force-head artifact: '+name)
    return manifest


def finish_artifacts(root, initial):
    root = Path(root)
    verify_artifacts(root,require_complete=False,manifest=initial)
    final = dict(initial,complete=True,artifacts={name:dict(path=name,sha256=original.sha256_file(root/name)) for name in ARTIFACTS})
    original.save_json(root/'ARTIFACTS.json',final)
    return final
