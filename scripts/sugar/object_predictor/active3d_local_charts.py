"""Explicit observation-only local fallback for one failed tactile pad chart.

Call supported_local_chart instead of supported_chart only in a separately
declared input adapter. Existing collectors, reports, defaults, saved arrays,
qualification gates and models are untouched. Inputs must already belong to
one pad and one observed time, in the same measured hand-local metre frame.

The original full-pad builder runs first. If it succeeds, its exact output and
metadata are returned. Otherwise every actual observed point is a candidate
centre for a fixed radius 2*max_edge_m neighbourhood (3.4 mm at the original
1.7 mm support scale). The same original builder/guards run on each subset;
largest supported chart area wins, with smallest original centre index as tie
break. This is a declared local support assumption, not fitted to GT geometry.

All returned source_indices/support_triangles address the ORIGINAL supplied pad
array, not the local subset. Interpolation weights/topology are untouched. A
successful local chart is still one partial observation, not full-pad coverage,
an optical touch-CNN output or a certificate of physical surface connectivity.
No masks, force acceptance, coordinates outside the measured support, GT object
mesh, object pose, mass, or model prediction are inferred here.
"""
from __future__ import annotations

import numpy as np

from .active3d_charts import supported_chart


def supported_local_chart(points, areas, template, max_edge_m=.0017):
    """Return (chart, info), preserving the original successful full-pad path."""
    points = np.asarray(points, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or areas.shape != (len(points),):
        raise ValueError('Expected one pad of measured XYZ centroids and areas')
    if not np.isfinite(max_edge_m) or max_edge_m <= 0:
        raise ValueError('Positive finite declared support edge scale required')
    # Original validation rejects nonfinite positions, nonpositive areas and
    # invalid inputs. Such errors must propagate, never become valid subpatches.
    original, full_info = supported_chart(points, areas, template, max_edge_m)
    if original is not None:
        return original, full_info

    radius = 2.*max_edge_m
    selected = None
    selected_info = None
    candidates = []
    for centre, position in enumerate(points):
        indices = np.flatnonzero(np.linalg.norm(points-position, axis=1) <= radius)
        chart, info = supported_chart(points[indices], areas[indices], template, max_edge_m)
        candidates.append(dict(centre_source_index=centre, neighbourhood_points=len(indices), **info))
        if chart is None:
            continue
        # Every group contains an actual centre point. Strict > preserves the
        # earliest original index for exact supported-area ties.
        if selected_info is not None and info['chart_area_m2'] <= selected_info['chart_area_m2']:
            continue
        selected = dict(chart)
        selected['source_indices'] = indices[chart['source_indices']]
        selected['support_triangles'] = indices[chart['support_triangles']]
        selected['local_source_indices'] = indices.copy()
        selected['local_centre_source_index'] = np.int64(centre)
        selected_info = dict(info, local_centre_source_index=centre,
                             local_source_indices=indices.tolist())

    common = dict(adapter='supported_local_chart_v1', full_pad_failure=full_info,
                  local_radius_m=radius, local_candidate_count=len(candidates),
                  valid_local_candidates=sum(row['reason'] == 'accepted' for row in candidates),
                  local_candidates=candidates,
                  source_index_convention='Original input pad array; caller maps pad indices to snapshot indices',
                  selection='Maximum supported chart area; ties use lowest original centre index')
    if selected is None:
        return None, dict(reason='no_supported_local_chart', **common)
    replay = np.einsum('ni,nij->nj', selected['barycentric'], points[selected['source_indices']])
    if not np.allclose(replay, selected['points'], atol=1e-12, rtol=0):
        raise AssertionError('Local-to-full-pad barycentric source replay mismatch')
    return selected, dict(selected_info, reason='accepted_local_supported_chart', **common)
