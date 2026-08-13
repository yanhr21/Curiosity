#!/usr/bin/env python3
"""Freeze common Plan-15 patch scales from the completed live mass sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CHANNELS = (
    "contact",
    "normal_load_n",
    "mean_pressure_pa",
    "signed_shear_x_n",
    "signed_shear_y_n",
    "friction_utilization",
    "slip_score",
    "incipient_slip",
    "gross_slip",
)


def _nonzero_quantile(values: np.ndarray, quantile: float) -> float:
    magnitude = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    magnitude = magnitude[np.isfinite(magnitude) & (magnitude > 0.0)]
    if not len(magnitude):
        raise ValueError("live sweep contains no nonzero samples for a required channel")
    return float(np.quantile(magnitude, quantile))


def fit_channel_scales(
    patch_features: np.ndarray,
    slip_features: np.ndarray,
    *,
    quantile: float = 0.995,
) -> list[float]:
    """Return nine common actor scales in the fixed patch-channel order."""

    patch = np.asarray(patch_features)
    slip = np.asarray(slip_features)
    if patch.ndim != 4 or patch.shape[1:] != (2, 27, 6):
        raise ValueError(f"patch features must be [frames,2,27,6], got {patch.shape}")
    if slip.ndim != 4 or slip.shape[1:] != (2, 27, 3):
        raise ValueError(f"slip features must be [frames,2,27,3], got {slip.shape}")
    if len(patch) != len(slip):
        raise ValueError("patch and slip trace lengths differ")
    if not 0.9 <= quantile < 1.0:
        raise ValueError("scale quantile must lie in [0.9, 1.0)")
    if not np.isfinite(patch).all() or not np.isfinite(slip).all():
        raise ValueError("live patch scale input contains non-finite values")

    shear_scale = _nonzero_quantile(patch[..., 3:5], quantile)
    scales = [
        1.0,
        _nonzero_quantile(patch[..., 1], quantile),
        _nonzero_quantile(patch[..., 2], quantile),
        shear_scale,
        shear_scale,
        _nonzero_quantile(patch[..., 5], quantile),
        _nonzero_quantile(slip[..., 0], quantile),
        1.0,
        1.0,
    ]
    if any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise RuntimeError("fitted patch scale is not positive and finite")
    return scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.995)
    args = parser.parse_args()

    patch_rows = []
    slip_rows = []
    resolved = []
    for value in args.trace:
        path = value.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            if "patch_features" not in payload or "slip_features" not in payload:
                raise KeyError(f"{path} lacks patch_features or slip_features")
            patch_rows.append(np.asarray(payload["patch_features"]))
            slip_rows.append(np.asarray(payload["slip_features"]))
        resolved.append(str(path))

    patches = np.concatenate(patch_rows, axis=0)
    slips = np.concatenate(slip_rows, axis=0)
    scales = fit_channel_scales(patches, slips, quantile=float(args.quantile))
    result = {
        "schema": "plan15_live_patch_channel_scales_v1",
        "source": "online IsaacLab mass-sweep traces",
        "quantile_over_nonzero_magnitude": float(args.quantile),
        "channel_order": list(CHANNELS),
        "patch_channel_scales": scales,
        "source_traces": resolved,
        "frames": int(len(patches)),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
