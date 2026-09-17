"""Necessary feasibility checks for identical-input state supervision.

Only targets and verified exact-input groups are inspected. No model, observed
error, filtering, input change, or physics is involved. A failed bound proves
the original fit gates cannot all hold for a deterministic shared prediction.
A passing bound does not establish observability or joint mesh feasibility.
"""
from __future__ import annotations

import numpy as np


def state_supervision_feasibility(rows, exact_input_groups, *, center_limit_m,
                                  size_max_relative_limit):
    """Check original unmasked center and max-per-axis relative-size gates.

    ``exact_input_groups`` must come from overfit_data.input_conflicts, which
    verifies each model-input array. That function also retains raw rotations;
    they are deliberately not used as a geometric inequivalence certificate.

    For two centers, the minimum worst error is half their separation. For
    positive dimensions a <= b, the minimum worst relative error of one shared
    estimate is (b-a)/(b+a), attained at 2ab/(a+b). The size acceptance uses a
    maximum over axes (not the mean used by the training loss). Checking group
    extrema is therefore equivalent to intersecting all per-axis gate intervals.
    Pairwise center bounds are necessary, not sufficient for three or more rows.
    """
    if (not np.isfinite(center_limit_m) or center_limit_m < 0 or
            not np.isfinite(size_max_relative_limit) or
            not 0 <= size_max_relative_limit < 1):
        raise ValueError('Expected finite original center/max-relative-size gates')
    reports = []
    for group in exact_input_groups:
        members = list(group['row_indices'])
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError('Expected a verified group of distinct identical-input rows')
        if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(rows) for i in members):
            raise ValueError('Identical-input row index outside the dataset')
        target = np.asarray([rows[i]['target'] for i in members], dtype=np.float64)
        if target.shape != (len(members), 13) or not np.isfinite(target).all():
            raise ValueError('Expected finite 13-dimensional saved targets')
        separations = np.linalg.norm(target[:, None, :3] - target[None, :, :3], axis=-1)
        pair = np.unravel_index(int(np.argmax(separations)), separations.shape)
        center_bound = float(separations[pair] / 2.)
        # tanh(log(b/a)/2) is algebraically (b-a)/(b+a), without exp overflow.
        size_bounds = np.tanh(np.ptp(target[:, 9:12], axis=0) / 2.)
        size_witnesses = [[members[int(np.argmin(target[:, 9+axis]))],
                           members[int(np.argmax(target[:, 9+axis]))]] for axis in range(3)]
        center_impossible = center_bound > center_limit_m
        size_impossible = bool(np.any(size_bounds > size_max_relative_limit))
        reports.append(dict(row_indices=members,
            samples=[rows[i]['metadata'] for i in members],
            center_pair_row_indices=[members[pair[0]], members[pair[1]]],
            center_pairwise_minimum_worst_error_m=center_bound,
            size_axis_extrema_row_indices=size_witnesses,
            size_axis_minimum_worst_relative_error=size_bounds.tolist(),
            raw_rotation6_spread=np.ptp(target[:, 3:9], axis=0).tolist(),
            center_gate_provably_impossible=bool(center_impossible),
            size_gate_provably_impossible=size_impossible,
            provably_impossible=bool(center_impossible or size_impossible)))
    impossible = [row for row in reports if row['provably_impossible']]
    return dict(passed=not impossible, groups=reports, impossible_groups=impossible,
        center_limit_m=float(center_limit_m),
        size_max_relative_limit=float(size_max_relative_limit),
        scope='Original all-row state supervision; exact-input pairwise center and max-per-axis size necessary bounds only. Raw rotation is diagnostic; no symmetry/mesh equivalence or observability certificate.')
