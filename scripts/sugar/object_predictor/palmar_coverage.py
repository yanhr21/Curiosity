"""Explicit alternative ideal continuous-palmar sensor, not original hardware.

Only the measured contact field and known hand layout are used. The original
54 region IDs stay fixed; previously uncovered palmar samples join the nearest
XZ footprint for aggregation, retaining original labels for every sample.
"""
import numpy as np
from sugar_newton.hand.patches import patch_footprints

COVERAGES = ('anatomical27', 'continuous_palmar_v1')


def continuous_palmar_field(field, palm_signs=(-1., 1.)):
    pos = np.asarray(field['pos']); hand = np.asarray(field['patch']); raw = np.asarray(field['pad'])
    if pos.shape != (len(raw), 3) or hand.shape != raw.shape:
        raise ValueError('Invalid contact field dimensions')
    if not np.isfinite(pos).all() or not np.isin(hand, (0, 1)).all():
        raise ValueError('Invalid hand contact position/identity')
    if not ((raw == -1) | ((raw >= 0) & (raw < 54))).all():
        raise ValueError('Unexpected anatomical region ID')
    assigned = raw >= 0
    if not np.array_equal(raw[assigned]//27, hand[assigned]):
        raise ValueError('Anatomical pad/hand mismatch')
    signs = np.asarray(palm_signs)
    if signs.shape != (2,) or not np.isin(signs, (-1., 1.)).all():
        raise ValueError('Explicit hand halfspace signs required')
    eligible = (~assigned) & (pos[:, 1]*signs[hand] > 0)
    pad = raw.copy()
    if eligible.any():
        fp = patch_footprints().astype(float)
        delta = pos[eligible].astype(float)[:, None, [0, 2]]-fp[None, :, :2]
        ca = np.cos(fp[:, 4]); sa = np.sin(fp[:, 4])
        lx = delta[..., 0]*ca+delta[..., 1]*sa
        lz = -delta[..., 0]*sa+delta[..., 1]*ca
        distance_sq = np.maximum(abs(lx)-fp[:, 2], 0.)**2+np.maximum(abs(lz)-fp[:, 3], 0.)**2
        pad[eligible] = distance_sq.argmin(1)+hand[eligible]*27
    return {**field, 'pad': pad, 'anatomical_pad': raw.copy()}


class ContinuousPalmarField:
    """Observation-only wrapper; delegates actual field generation unchanged."""
    def __init__(self, raw, palm_signs):
        self.raw = raw
        self.palm_signs = tuple(palm_signs)

    def to_numpy(self):
        return continuous_palmar_field(self.raw.to_numpy(), self.palm_signs)

    def __getattr__(self, name):
        return getattr(self.raw, name)
