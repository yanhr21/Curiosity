"""Read saved full-model denoising errors to audit fixed ranking-margin activity."""
import json
from pathlib import Path

import numpy as np


BASE = Path('/public/home/yanhongru/Curiosity/experiments/demo_following/demo_future_smp_v1')
SOURCE = BASE / 'matched_generator_branch_paired_rank025512'
RECORDED = BASE / 'matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe'
OUT = RECORDED / 'rank_margin_readback'


def main():
    fixed = json.loads((RECORDED / 'fixed_xt_condition/RESULT.json').read_text())
    assert fixed['checks_passed'] and fixed['plot_inspected']
    plan = json.loads((SOURCE / 'PROTOCOL.json').read_text())
    prior = SOURCE / 'frozen_evaluation/rank_component_audit'
    audit = json.loads((prior / 'RESULT.json').read_text())
    assert audit['checks_passed'] and audit['plot_inspected']
    OUT.mkdir(exist_ok=False)
    protocol = dict(source=str(prior), source_model_index=2, margin_fraction=0.1,
        scope='Saved full0.25 endpoint supervised denoising errors, same4noise seeds/all50times/18cases. Relative epsilon error gap divided by alpha/(1-alpha)*paired separation equals unclipped clean-error gap divided by paired separation. No new forward/gradient/optimizer/sampling/physics. Report all cases; only14TRAIN cases motivate followup. Sigmoid is a scalar derivative of softplus with respect to normalized margin, not a parameter gradient or measured Adam damage.',
        hypothesis='Existing softplus continues exerting a nonzero ranking derivative after the fixed0.1 margin is satisfied. Audit its activity before any claim that stopping satisfied-margin rank gradients helps accurate generation.',
        automatic_next_action='If satisfied TRAIN rows still carry material scalar rank derivative, run one full-model TRAIN gradient audit of a fixed0.1 hinge-margin alternative at released initialization and retainedfull0.25 endpoint before declaring a new matched512 experiment. Preserve complete network, weight0.25 and all prior settings; no coefficient/margin sweep, old-budget extension or benefit claim.')
    (OUT / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n')
    with np.load(prior / 'MATCHED_DENOISING.npz') as data:
        clean = data['clean_unclipped_mse'][2].copy()
        phases = data['phases'].copy(); times = data['times'].copy(); seeds = data['seeds'].copy()
    checks = dict(shape_exact=clean.shape == (4, 50, 2, 18), all50times=times.tolist() == list(range(50)))
    rows = {}; train_values = []
    for phase in plan['frozen_evaluation']['evaluation_phases']:
        with np.load(RECORDED / f'phase_{phase}_steps_16.npz') as data:
            target = data['normalized_actual_target'].copy()
        separation = float(np.square(target[0]-target[1]).mean())
        indices = np.flatnonzero(phases == phase)
        checks[f'{phase}_two_distinct_targets'] = len(indices) == 2 and separation > 0
        margin = .1 + (clean[:, :, 0, indices]-clean[:, :, 1, indices])/separation
        derivative = np.exp(-np.logaddexp(0, -margin))
        satisfied = margin <= 0
        row = dict(split='train' if phase in plan['data']['train']['phases'] else 'reused_check',
            case_noise_time_rows=int(margin.size), margin_satisfied_fraction=float(satisfied.mean()),
            mean_softplus_scalar_derivative=float(derivative.mean()),
            satisfied_derivative_sum_fraction=float(derivative[satisfied].sum()/derivative.sum()),
            mean_satisfied_derivative=float(derivative[satisfied].mean()) if satisfied.any() else None,
            minimum_margin=float(margin.min()), maximum_margin=float(margin.max()))
        rows[str(phase)] = row
        if row['split'] == 'train':
            train_values.append((margin.reshape(-1), derivative.reshape(-1)))
    margin = np.concatenate([v[0] for v in train_values]); derivative = np.concatenate([v[1] for v in train_values])
    checks['exact_train_rows'] = margin.size == 14*4*50
    checks['finite'] = bool(np.isfinite(margin).all() and np.isfinite(derivative).all())
    train = dict(rows=int(margin.size), satisfied_fraction=float((margin <= 0).mean()),
        satisfied_derivative_sum_fraction=float(derivative[margin <= 0].sum()/derivative.sum()),
        mean_satisfied_derivative=float(derivative[margin <= 0].mean()))
    # This is a derivative-presence decision, not a tuned threshold or scientific success.
    decision = bool(np.any((margin <= 0) & (derivative > 0)))
    result = dict(checks=checks, checks_passed=all(checks.values()), phases=rows, train=train,
        satisfied_margin_still_exerts_scalar_rank_derivative=decision,
        next_action=protocol['automatic_next_action'] if decision else 'Retain completed evidence; no hinge-gradient hypothesis supported by this audit.',
        scope=protocol['scope'], new_optimizer_updates=0, new_model_forwards=0, new_physics_steps=0)
    (OUT / 'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    assert all(checks.values())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
