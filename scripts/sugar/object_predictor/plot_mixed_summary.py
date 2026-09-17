"""Plot the declared acquisition-specific metrics from completed saved readbacks."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_mass_transfer(root, report, modes, labels, colors):
    """Display every held-out mass, including temporal spread and force-zero."""
    support_ids = set(map(int, report["reports"]["geometry"]["support_all"]["episodes"]))
    scenarios = ((root, support_ids, "Held-out support configurations"),
                 (root.parent / "fresh_mass", None, "Additional mass configurations"))
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
    records = []
    sources = {}
    extent = [0.]
    for row, (folder, selected, title) in enumerate(scenarios):
        anchor = None
        for col, mode in enumerate(modes):
            filename = "force_zero_predictions.npz" if col == 3 else "predictions.npz"
            source = folder / ("geometry_contact_force" if col == 3 else mode) / filename
            sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            with np.load(source) as data:
                a = {key: data[key] for key in data.files}
            if anchor is None:
                anchor = a
            else:
                for key in ("episode", "frame", "target", "contact"):
                    if not np.array_equal(a[key], anchor[key]):
                        raise ValueError(f"Unmatched mass-transfer data: {source}/{key}")
            episodes = set(map(int, a["episode"])) if selected is None else selected
            if not episodes <= set(map(int, a["episode"])):
                raise ValueError("Missing support configuration")
            for episode in sorted(episodes):
                mask = a["episode"] == episode
                truth = np.exp(a["target"][mask, 12].astype(np.float64))
                prediction = np.exp(a["prediction"][mask, 12].astype(np.float64))
                if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
                    raise ValueError("Nonfinite mass prediction")
                x, y = float(truth.mean()), float(prediction.mean())
                low, high = map(float, np.quantile(prediction, [.1, .9]))
                axes[row, col].vlines(x, low, high, color=colors[col], alpha=.6)
                axes[row, col].scatter([x], [y], color=colors[col], s=26, zorder=3)
                extent.extend((x, y, high))
                records.append(dict(scenario=title, mode=mode, episode=episode,
                                    samples=int(mask.sum()), true_mass_kg=x,
                                    mean_predicted_mass_kg=y, p10_kg=low, p90_kg=high,
                                    mean_relative_error=float(np.abs(prediction / truth - 1.).mean())))
            axes[row, col].set_title(labels[col])
            if row == 1:
                axes[row, col].set_xlabel("True mass [kg]")
            if col == 0:
                axes[row, col].set_ylabel(title + "\nPredicted mass [kg]")
    limit = max(extent) * 1.07
    for ax in axes.flat:
        ax.plot([0., limit], [0., limit], "k--", linewidth=1.)
        ax.set_xlim(0., limit)
        ax.set_ylim(0., limit)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15)
    fig.suptitle("Mass prediction across every held-out support configuration", fontsize=14)
    fig.text(.5, .02, "Point: mean over every saved test window. Vertical line: temporal 10–90% range, not a confidence interval.\n"
             "Dashed line: perfect prediction. Identical geometry/support setup; additional masses were not used for training.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0., .08, 1., .95))
    fig.subplots_adjust(hspace=.35)
    image_path = root / "mass_transfer_comparison.png"
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    output = dict(sources=sources, records=records,
                  image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                  temporal_spread_is_not_confidence_interval=True,
                  new_model_forwards=0, new_optimizer_updates=0,
                  requires_agent_visual_inspection=True)
    (root / "MASS_TRANSFER_PLOT.json").write_text(json.dumps(output, indent=2) + "\n")


def main(root):
    root = Path(root)
    source = root / "MIXED_METRICS.json"
    report = json.loads(source.read_text())
    if not report["readback_passed"]:
        raise ValueError("The completed matched readback is required")
    modes = ("geometry", "geometry_contact", "geometry_contact_force",
             "force_zero_keep_contact_geometry_and_area")
    labels = ("Geometry", "+ Contact", "+ Force", "Force model\nforces zeroed")
    colors = ("#64748b", "#0284c7", "#059669", "#d97706")
    specifications = (
        ("probe_primary_140_148s", "position_cm", 1., 5., "Center error [cm]"),
        ("probe_primary_140_148s", "rotation_deg", 1., 15., "Rotation error [deg]"),
        ("probe_primary_140_148s", "size_relative", 100., 10., "Size error [%]"),
        ("support_all", "mass_relative", 100., 10., "Mass error [%]"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    plotted = {}
    for ax, (group, metric, scale, threshold, title) in zip(axes, specifications):
        values = [report["reports"][mode][group]["equal_episode_mean"][metric] * scale
                  for mode in modes]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite saved metric")
        bars = ax.bar(range(4), values, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.4,
                   label=f"Declared limit: {threshold:g}")
        ax.set_xticks(range(4), labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0., max(max(values), threshold) * 1.22)
        ax.set_title(title)
        ax.set_ylabel("Equal episode mean; lower is better")
        ax.legend(loc="upper right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.5, -.36, "Multi-face probes, fixed 140–148 s" if group.startswith("probe")
                else "Support episodes, all saved test windows", transform=ax.transAxes,
                ha="center", fontsize=9)
        plotted[metric] = {"group": group, "display_scale": scale,
                           "declared_limit": threshold, "values": dict(zip(modes, values))}
    fig.suptitle("Held-out object-state prediction: three matched trained models + force-zero intervention",
                 fontsize=13)
    fig.text(.5, .025,
             "One known scanned family. All failed test acquisitions retained. "
             "All arms share force-feedback hand trajectories.\n"
             "Force-zero keeps geometry, contact and area; it is an inference intervention, not a fourth trained model.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.055, right=.985, top=.86, bottom=.32, wspace=.33)
    output = root / "primary_state_comparison.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    manifest = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "plotted_metrics": plotted, "new_model_forwards": 0, "new_optimizer_updates": 0,
                "requires_agent_visual_inspection": True}
    (root / "PRIMARY_STATE_PLOT.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_mass_transfer(root, report, modes, labels, colors)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_root")
    main(parser.parse_args().evaluation_root)
