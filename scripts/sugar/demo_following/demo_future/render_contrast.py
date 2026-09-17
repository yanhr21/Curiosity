"""Plot actual and predicted same-task demonstration mismatch differences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "experiments/demo_following/demo_future_smp_v1"
PARENT = ROOT / "experiments/demo_following/contact_event_reward_redesign_v1/phase_aware_goal_core_dataset_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=BASE / "matched_future_supervision")
    parser.add_argument("--control", default="aux_detached")
    parser.add_argument("--treatment", default="aux_attached")
    args = parser.parse_args()
    protocol = json.loads((args.run / "PROTOCOL.json").read_text())
    dataset = Path(protocol.get("dataset", BASE / "dataset"))
    split = "validation"
    with np.load(dataset / split / "routing.npz", allow_pickle=False) as z:
        route = {key: z[key] for key in z.files}
    with np.load(protocol.get("normalization_path", PARENT / "NORMALIZATION.npz"), allow_pickle=False) as z:
        scale = z["target_scale"]
    truth = np.log1p(np.load(dataset / split / "target_mismatch.npy") / scale)
    predictions = {}
    for arm in (args.control, args.treatment):
        with np.load(args.run / arm / f"{split}_predictions.npz", allow_pickle=False) as z:
            predictions[arm] = z["full"]
    output = args.run / "contrast_figures"
    output.mkdir(exist_ok=False)
    records = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for task, name in ((0, "CarryBox"), (1, "KickBox")):
        # Deterministic first validation motion, never selected by a favorable plot.
        source = int(np.min(route["base_source_motion_id"][route["base_task"] == task]))
        selected = (route["base_task"] == task) & (route["base_source_motion_id"] == source)
        time = route["base_anchor_frame"][selected] * 0.02
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        for horizon, ax in enumerate(axes):
            def difference(array):
                return (array[selected, 1, horizon] - array[selected, 0, horizon]).mean(axis=-1)
            ax.plot(time, difference(truth), color="#222222", label="Actual future mismatch difference", linewidth=1.8)
            ax.plot(time, difference(predictions[args.control]), color="#bd7024", label=args.control, linewidth=1.3)
            ax.plot(time, difference(predictions[args.treatment]), color="#2169a6", label=args.treatment, linewidth=1.3)
            ax.axhline(0, color="#999999", linestyle=":", linewidth=0.8)
            ax.set_ylabel(f"{(0.2,0.4,0.8,1.4)[horizon]:.1f}s future\nMismatch gap")
            ax.grid(alpha=0.15)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Causal history endpoint (seconds)")
        fig.suptitle(f"{name} source {source:03d}: alternate minus original demonstration\n"
                     "Positive means actual trajectory matches original better; predictor diagnostic, not a policy rollout", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = output / f"{name}_source{source:03d}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        records.append({"task": name, "source": source, "rows": int(selected.sum()), "path": str(path)})
    with (output / "RENDER_RESULT.json").open("x") as stream:
        json.dump({"execution_completed": True, "records": records,
                   "scope": "Native log1p-scaled 13-target mean mismatch gap on fixed first validation motion per task; no physics or generated-video claim"}, stream, indent=2)


if __name__ == "__main__":
    main()
