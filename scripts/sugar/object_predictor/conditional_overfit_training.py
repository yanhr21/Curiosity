"""Conditional-contact supervision and gates; the original profile is separate."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .overfit_model import (LOSS_SCALES, force_target, masked_mass_loss,
                           state_vertices, symmetric_mesh_distance)


def conditional_task_losses(output, batch, normalized_sample_vertices):
    """Only no-evidence state precision is masked; other original losses remain.

    Selection precedes geometry/error computation. All rows still passed through
    the full model. Contact eligibility does not certify full observability.
    """
    state, target = output['state'], batch['target']
    selected = batch['state_precision_eligible'] > .5
    if bool(selected.any()):
        prediction, truth = state[selected], target[selected]
        center = (prediction[:, :3] - truth[:, :3]).norm(dim=1).mean() / LOSS_SCALES['center_m']
        mesh = symmetric_mesh_distance(state_vertices(prediction, normalized_sample_vertices),
                                       state_vertices(truth, normalized_sample_vertices)).mean() / LOSS_SCALES['mesh_m']
        size = torch.expm1(prediction[:, 9:12] - truth[:, 9:12]).abs().mean() / LOSS_SCALES['size_relative']
    else:
        center = mesh = size = state[:, :12].sum() * 0.
    return dict(center=center, mesh=mesh, size=size,
        mass=masked_mass_loss(state, target, batch['mass_available']),
        force=F.smooth_l1_loss(output['force'] / LOSS_SCALES['force_n'],
            force_target(batch['physics']) / LOSS_SCALES['force_n'], beta=1.),
        availability=F.binary_cross_entropy_with_logits(output['availability_logit'], batch['mass_available']),
        contact=F.binary_cross_entropy_with_logits(output['contact_logit'], batch['physics']['contact_present'][:, 0]))


def conditional_acceptance(arrays, role, *, original_acceptance, fit_limits,
                           interpolation_limits, episodes):
    """Keep raw all-state report and original non-state gates on every row."""
    original = original_acceptance(arrays, role)
    per_case, all_checks = {}, {}
    for episode in episodes:
        mask = arrays['episode'] == episode
        raw = original['per_case'][str(episode)]
        if not mask.any():
            per_case[str(episode)] = raw
            all_checks[f'{episode}/present'] = False
            continue
        row = {key: value[mask] for key, value in arrays.items()}
        eligible = row['state_precision_eligible'] > .5
        checks = dict(raw['checks'])
        checks['contact_candidate_state_rows_present'] = bool(eligible.any())
        state_metrics = {}
        for key in ('center_cm', 'mesh_nn_cm', 'rotation_deg', 'size_mean_relative', 'size_max_relative'):
            values = row[key][eligible]
            state_metrics[key] = (dict(mean=float(values.mean()), p95=float(np.quantile(values, .95)),
                maximum=float(values.max())) if len(values) else None)
        if role == 'fit':
            checks['late_2381_state_evidence'] = bool(np.any(eligible & (row['frame'] == 2381)))
            checks.update(center=bool(eligible.any() and np.all(row['center_cm'][eligible] <= fit_limits['center_cm'])),
                mesh=bool(eligible.any() and np.all(row['mesh_nn_cm'][eligible] <= fit_limits['mesh_nn_cm'])),
                size=bool(eligible.any() and np.all(row['size_max_relative'][eligible] <= fit_limits['size_max_relative'])))
        elif role == 'same_trajectory_interpolation':
            for name, metric, stat, limit in (
                ('center_mean', 'center_cm', 'mean', 'center_mean_cm'),
                ('center_p95', 'center_cm', 'p95', 'center_p95_cm'),
                ('mesh_mean', 'mesh_nn_cm', 'mean', 'mesh_mean_cm'),
                ('mesh_p95', 'mesh_nn_cm', 'p95', 'mesh_p95_cm')):
                checks[name] = bool(eligible.any() and state_metrics[metric][stat] <= interpolation_limits[limit])
        else:
            raise ValueError('Unknown conditional evaluation role')
        per_case[str(episode)] = dict(passed=all(checks.values()), checks=checks,
            metrics=raw['metrics'], original_all_state_gate_passed=raw['passed'],
            state_candidate_metrics=state_metrics, all_rows=int(mask.sum()),
            state_candidate_rows=int(eligible.sum()), prior_unknown_rows=int((~eligible).sum()))
        all_checks.update({f'{episode}/{key}': bool(value) for key, value in checks.items()})
    return dict(passed=all(all_checks.values()), checks=all_checks, per_case=per_case,
        all_items=original['all_items'], role=role,
        original_all_state_gate_passed=original['passed'],
        state_candidate_rows=int(np.count_nonzero(arrays['state_precision_eligible'] > .5)),
        prior_unknown_rows=int(np.count_nonzero(arrays['state_precision_eligible'] <= .5)),
        no_contact_note='All predictions and raw state errors are retained. H32 without observed contact is PRIOR/UNKNOWN; contact rows are regression candidates, not certified identifiable states.',
        scope='Controlled-fixture conditional TRAIN fit/interpolation only; not original blind/all-state qualification, calibrated uncertainty, generalization or tactile benefit.')
