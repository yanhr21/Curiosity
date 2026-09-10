#!/usr/bin/env python3
"""Split the retained Refiner reward panel into two digitized scalar plots.

The original TensorBoard event file is no longer retained.  This utility recovers
the two Matplotlib traces from the preserved, lossless PNG by their canonical
``tab:blue`` and ``tab:orange`` colors.  The resulting values are therefore
pixel-digitized approximations, not replacements for the missing event scalars.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
VIS = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/visualizations"
)
SOURCE = VIS / "refiner_model10000_reward_curve.png"

# Pixel calibration from the retained 2160 x 710 lossless source panel.
# Major x ticks: (184, 0), (555, 2000), ..., (2040, 10000).
# Major y ticks: (606, 0), (490, 100), ..., (146, 400).
X0_PX = 184
X10000_PX = 2040
Y0_PX = 606.0
Y400_PX = 146.0
PLOT_TOP_PX = 63
PLOT_BOTTOM_PX = 631

SERIES = {
    "mean_reward": {
        "rgb": np.array([31, 119, 180], dtype=np.int16),
        "label": "Train/mean_reward",
        "color": "tab:blue",
    },
    "mean_episode_length": {
        "rgb": np.array([255, 127, 14], dtype=np.int16),
        "label": "Train/mean_episode_length",
        "color": "tab:orange",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digitize(rgb_image: np.ndarray, target_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    crop = rgb_image[PLOT_TOP_PX : PLOT_BOTTOM_PX + 1, X0_PX : X10000_PX + 1]
    distance = np.max(np.abs(crop.astype(np.int16) - target_rgb), axis=2)
    mask = distance <= 18

    # Remove the two colored line samples in the source legend.  No actual blue
    # or orange trace occupies this rectangle at these late iterations.
    legend_x0 = 1750 - X0_PX
    legend_x1 = min(2125, X10000_PX) - X0_PX
    legend_y0 = 72 - PLOT_TOP_PX
    legend_y1 = 130 - PLOT_TOP_PX
    mask[legend_y0 : legend_y1 + 1, legend_x0 : legend_x1 + 1] = False

    xs: list[float] = []
    ys: list[float] = []
    for local_x in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, local_x])
        if rows.size == 0:
            continue
        image_y = float(np.median(rows + PLOT_TOP_PX))
        iteration = (local_x + X0_PX - X0_PX) * 10000.0 / (X10000_PX - X0_PX)
        value = (Y0_PX - image_y) * 400.0 / (Y0_PX - Y400_PX)
        xs.append(iteration)
        ys.append(value)
    return np.asarray(xs), np.asarray(ys)


def write_csv(path: Path, xs: np.ndarray, ys: np.ndarray, label: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["training_iteration_pixel_digitized", f"{label}_pixel_digitized"])
        writer.writerows(zip(xs, ys, strict=True))


def render(path: Path, xs: np.ndarray, ys: np.ndarray, label: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    ax.plot(xs, ys, color=color, linewidth=1.6, label=label)
    ax.set_xlim(0, 10000)
    margin = max(1.0, 0.06 * float(np.ptp(ys)))
    ax.set_ylim(float(np.min(ys) - margin), float(np.max(ys) + margin))
    ax.set_xlabel("training iteration")
    ax.set_ylabel(label)
    ax.set_title(f"Official SUGAR CarryBox Refiner model_10000: {label}")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.text(
        0.5,
        0.005,
        "Pixel-digitized from the retained lossless TensorBoard-rendered PNG; values are approximate.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    image = np.asarray(Image.open(SOURCE).convert("RGB"))
    if image.shape != (710, 2160, 3):
        raise RuntimeError(f"unexpected source shape: {image.shape}")

    outputs: dict[str, object] = {}
    for name, spec in SERIES.items():
        xs, ys = digitize(image, spec["rgb"])
        if xs.size < 1700:
            raise RuntimeError(f"too few extracted columns for {name}: {xs.size}")
        png = VIS / f"refiner_model10000_{name}.png"
        csv_path = VIS / f"refiner_model10000_{name}_pixel_digitized.csv"
        render(png, xs, ys, str(spec["label"]), str(spec["color"]))
        write_csv(csv_path, xs, ys, str(spec["label"]))
        outputs[name] = {
            "png": str(png),
            "png_sha256": sha256(png),
            "png_resolution": list(Image.open(png).size),
            "csv": str(csv_path),
            "csv_sha256": sha256(csv_path),
            "digitized_points": int(xs.size),
            "approx_min": float(np.min(ys)),
            "approx_max": float(np.max(ys)),
            "approx_final": float(ys[-1]),
        }

    provenance = {
        "method": "color-trace pixel digitization from retained lossless PNG",
        "limitation": "approximate image-derived values; original TensorBoard scalar event is unavailable",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "source_resolution": [2160, 710],
        "calibration": {
            "x_pixels": {"iteration_0": X0_PX, "iteration_10000": X10000_PX},
            "y_pixels": {"value_0": Y0_PX, "value_400": Y400_PX},
            "rgb_max_distance": 18,
        },
        "outputs": outputs,
    }
    provenance_path = VIS / "refiner_model10000_split_curves.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**provenance, "provenance": str(provenance_path)}, indent=2))


if __name__ == "__main__":
    main()
