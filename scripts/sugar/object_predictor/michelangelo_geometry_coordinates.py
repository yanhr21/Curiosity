"""Correct only the released mesh extractor's grid-index coordinate conversion.

Official queries use n=2**depth+1 endpoint-inclusive samples. Marching-cubes
indices therefore have n-1 intervals; the released extractor divides by n.
This invertible affine correction requires no object geometry or fitted pose.
It preserves every vertex/face and does not repair missing or extra surfaces.
"""
import numpy as np


def correct_extracted_coordinates(vertices, *, depth=7,
                                  bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25)):
    vertices = np.asarray(vertices)
    bounds = np.asarray(bounds, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or bounds.shape != (6,) or not np.isfinite(bounds).all()
            or np.any(bounds[3:] <= bounds[:3]) or not isinstance(depth, int) or depth < 1):
        raise ValueError('Invalid vertices or declared official extraction grid')
    n = 2**depth + 1
    return (vertices.astype(np.float64) - bounds[:3]) * (n / (n - 1)) + bounds[:3]
