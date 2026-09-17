"""Single-factor diagnostic: use one resolved observed region per hand.

All force feedback still consumes the caller's complete hand loads. Only the
geometry view used by the unchanged bilateral controller and its pivot rollback
is restricted; the full observed surface is restored even when the call fails.
Pad IDs describe known hand footprints, not hidden object faces.
"""
import numpy as np

from .bilateral_common_alignment import BilateralCommonAlignmentController
from .inspect_contact_surface import fit_surface


REGIONAL_RECORD_FIELDS = (
    'regional_selected_pad', 'regional_selected_area_m2',
    'regional_total_area_m2', 'regional_selected_count', 'regional_selected_rms_m',
    'regional_selected_center_hand_m', 'regional_selected_pca_normal_hand',
    'regional_valid_candidate_count', 'regional_selection_changed',
    'regional_input_hand_load_n',
    'regional_candidate_area_m2', 'regional_candidate_count',
    'regional_candidate_rms_m', 'regional_candidate_valid',
)


def region_geometry_view(surface):
    """Observed-only largest-area eligible pad; exact original fit thresholds."""
    selected = np.full(2, -1, dtype=np.int32)
    area_selected = np.zeros(2)
    total_area = np.zeros(2)
    counts = np.zeros(2, dtype=np.int32)
    rms_selected = np.zeros((2, 3))
    centers = np.zeros((2, 3))
    normals = np.zeros((2, 3))
    candidate_counts = np.zeros(2, dtype=np.int32)
    pad_areas = np.zeros(54)
    pad_counts = np.zeros(54, dtype=np.int32)
    pad_rms = np.zeros((54, 3))
    pad_valid = np.zeros(54, dtype=bool)
    keep = np.zeros(len(surface['area']), dtype=bool)
    for side in (0, 1):
        observed = ((surface['patch'] == side) & (surface['pad'] >= 0)
                    & (surface['pressure'] > 0))
        total_area[side] = surface['area'][observed].sum()
        active = observed & (surface['area'] > 0)
        points = surface['pos'][active]
        areas = surface['area'][active]
        if not np.isfinite(points).all() or not np.isfinite(areas).all():
            raise ValueError('Nonfinite observed contact geometry')
        # np.unique is sorted: keeping the first area tie chooses smaller pad.
        for pad in np.unique(surface['pad'][active]):
            if not side * 27 <= pad < (side + 1) * 27:
                raise ValueError('Observed pad ID disagrees with the fixed 54-pad hand layout')
            region = active & (surface['pad'] == pad)
            count = int(region.sum())
            area = float(surface['area'][region].sum())
            pad_areas[pad] = area
            pad_counts[pad] = count
            if count < 6:
                continue
            center, rms, normal = fit_surface(surface['pos'][region].astype(float),
                                              surface['area'][region].astype(float))
            pad_rms[pad] = rms
            if rms[1] < .0002 or rms[0] / max(rms[1], 1e-12) > .35:
                continue
            candidate_counts[side] += 1
            pad_valid[pad] = True
            if selected[side] < 0 or area > area_selected[side]:
                selected[side] = pad
                area_selected[side] = area
                counts[side] = count
                rms_selected[side] = rms
                centers[side] = center
                normals[side] = normal
        if selected[side] >= 0:
            keep |= active & (surface['pad'] == selected[side])
    view = {key: value[keep] for key, value in surface.items()}
    record = dict(regional_selected_pad=selected, regional_selected_area_m2=area_selected,
                  regional_total_area_m2=total_area, regional_selected_count=counts,
                  regional_selected_rms_m=rms_selected,
                  regional_selected_center_hand_m=centers,
                  # Raw PCA sign is arbitrary. Actual oriented control normal
                  # remains the parent's fit_normal_w, never this audit field.
                  regional_selected_pca_normal_hand=normals,
                  regional_valid_candidate_count=candidate_counts,
                  regional_candidate_area_m2=pad_areas, regional_candidate_count=pad_counts,
                  regional_candidate_rms_m=pad_rms, regional_candidate_valid=pad_valid)
    return view, record


class RegionalBilateralAlignmentController(BilateralCommonAlignmentController):
    intervention_name = 'regional_bilateral_alignment_v1'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_selected_pad = np.full(2, -1, dtype=np.int32)

    def command(self, time, loads, dt):
        complete_surface = self.surface
        if complete_surface is None:
            empty = dict(pos=np.empty((0, 3)), area=np.empty(0), pressure=np.empty(0),
                         pad=np.empty(0, dtype=np.int32), patch=np.empty(0, dtype=np.int32))
            _, regional = region_geometry_view(empty)
        else:
            self.surface, regional = region_geometry_view(complete_surface)
        try:
            # No reduction/replacement of the original all-hand load argument.
            poses, velocity, record = super().command(time, loads, dt)
        finally:
            self.surface = complete_surface
        selected = regional['regional_selected_pad']
        regional['regional_selection_changed'] = selected != self.previous_selected_pad
        regional['regional_input_hand_load_n'] = np.asarray(loads).copy()
        self.previous_selected_pad = selected.copy()
        record.update(regional)
        # Preserve the original public record's whole-hand area meaning.
        # Geometry-support area is separately named, never a replacement load.
        if 'contact_area_m2' in record:
            record['contact_area_m2'] = regional['regional_total_area_m2'].copy()
        return poses, velocity, record
