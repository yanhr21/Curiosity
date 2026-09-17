"""Parameter-free CPU summaries of the existing collated SUGAR observations.

Call after collate_surface and before moving the batch to CUDA. Only the four
ordinary observation tensors are accepted; no labels, eligibility masks, object
state, model, checkpoint or simulator is read. Outputs retain all H32 frames in
the CURRENT LEFT HAND reference frame, in Newtons. They are encoded-observation
summaries, not independent predictions or complete physical object wrenches.

NumPy float64 inverse transforms/reductions followed by float32 storage preserve
the existing observed_physics_targets numerical contract. The oracle is NOT
called here. The CPU implementation is deliberate: a future GPU implementation
must separately qualify its numerical semantics. No gradients/parameters exist.
"""
from __future__ import annotations

import numpy as np
import torch

HISTORY = 32
INPUT_KEYS = frozenset(('coord', 'grid_coord', 'feat', 'offset'))
FORCE_FIELDS = ('normal_load_encoded_left_n', 'normal_load_encoded_right_n',
                'shear_on_hand_x_n', 'shear_on_hand_y_n', 'shear_on_hand_z_n',
                'normal_on_object_approx_x_n', 'normal_on_object_approx_y_n',
                'normal_on_object_approx_z_n')
SUPPORT_FIELDS = ('shear_support_up_n', 'combined_support_up_approx_n')


def _frames(inputs, history):
    if history != HISTORY or isinstance(history, bool):
        raise ValueError('Require the declared H32 observation history')
    if set(inputs) != INPUT_KEYS:
        raise ValueError('Accept only coord/grid_coord/feat/offset observations')
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.requires_grad:
            raise ValueError('Require non-gradient CPU observation tensors: ' + key)
    feat = inputs['feat'].numpy()
    coord = inputs['coord'].numpy()
    grid = inputs['grid_coord'].numpy()
    offset = inputs['offset'].numpy()
    n = len(feat)
    if (feat.ndim != 2 or feat.shape[1] != 20 or feat.dtype != np.float32 or n == 0
            or coord.shape != (n, 3) or grid.shape != (n, 3)
            or not np.issubdtype(grid.dtype, np.integer)
            or offset.ndim != 1 or len(offset) == 0 or len(offset) % history
            or not np.issubdtype(offset.dtype, np.integer)
            or offset[-1] != n or np.any(np.diff(np.r_[0, offset]) <= 0)
            or not np.isfinite(feat).all() or not np.isfinite(coord).all()):
        raise ValueError('Malformed finite H32 collated observation arrays')
    if np.any(feat[:, 10] < 0) or np.any(feat[:, 14] < 0) or np.any((feat[:, 16] < 0) | (feat[:, 16] > 1)):
        raise ValueError('Invalid encoded load/area/hand fraction')
    result = []
    start = 0
    for stop in offset:
        frame = feat[start:stop].astype(np.float64)
        down = frame[0, 17:20]
        if (not np.allclose(frame[:, 17:20], down, rtol=0, atol=1e-6)
                or not np.isclose(np.linalg.norm(down), 1, atol=1e-6)):
            raise ValueError('Each frame requires its known unit gravity direction')
        result.append(frame)
        start = int(stop)
    return result


def _summaries(inputs, history):
    force, support = [], []
    for frame in _frames(inputs, history):
        with np.errstate(over='raise', invalid='raise'):
            load = np.expm1(frame[:, 10])
            shear = 2 * np.sinh(frame[:, 11:14])
            normal = frame[:, 6:9] * load[:, None]
            side_weights = np.stack((1 - frame[:, 16], frame[:, 16]), axis=1)
            shear_sum = shear.sum(0)
            force.append(np.concatenate((side_weights.T @ load, shear_sum, normal.sum(0))).astype(np.float32))
            down = frame[0, 17:20]
            # Keep the oracle's per-point subtraction BEFORE summation: changing
            # this order is mathematically equivalent but need not be bitexact.
            support.append(np.array([shear_sum @ down, -(normal - shear).sum(0) @ down], np.float32))
    force = np.stack(force).reshape(-1, history, 8)
    support = np.stack(support).reshape(-1, history, 2)
    if not np.isfinite(force).all() or not np.isfinite(support).all():
        raise FloatingPointError('Nonfinite decoded observation summary')
    return torch.from_numpy(force), torch.from_numpy(support)


def summarize_observed_forces(inputs, *, history=HISTORY):
    """Return [B,32,8] float32 CPU Newton summaries, never sum across time.

Frame k=b*32+t occupies [offset[k-1],offset[k]); k=0 starts at0.
Thus [:,31,:] is each window's current frame. The two load components are
encoded-side weighted sums; mixed-hand voxels need not preserve true hand loads.
The last three components use normalized CAD/fitted normals, not true normals.
"""
    return _summaries(inputs, history)[0]


def summarize_support_projections(inputs, *, history=HISTORY):
    """Return [B,32,2] shear-up and approximate combined-up Newton summaries.

Gravity feat17:20 points DOWN in the common current-left-hand frame. Shear
acts ON the hand, so its down projection is object-support up. Approximate
object force is normal minus shear. Neither projection is a mass estimate.
"""
    return _summaries(inputs, history)[1]
