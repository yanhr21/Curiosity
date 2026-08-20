#!/usr/bin/env python3
"""Render official SUGAR training logs without loading or changing any model."""

from __future__ import annotations

import argparse
import json
import socket
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, default=None)
    return parser.parse_args()


def load_tensorboard_scalars(event_files: list[Path]) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Resume attempts create multiple event files with overlapping iteration
    # numbers. Prefer the run with the broadest coverage for a duplicate step,
    # so a one-iteration failed retry cannot replace the end of a long run.
    points: dict[str, dict[int, tuple[int, float, float]]] = defaultdict(dict)
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            scalars = accumulator.Scalars(tag)
            run_coverage = len({scalar.step for scalar in scalars})
            for scalar in scalars:
                previous = points[tag].get(scalar.step)
                candidate = (run_coverage, scalar.wall_time, scalar.value)
                if previous is None or candidate[:2] >= previous[:2]:
                    points[tag][scalar.step] = candidate
    return {
        tag: [(step, run_wall_value[2]) for step, run_wall_value in sorted(step_values.items())]
        for tag, step_values in points.items()
    }


def plot_rl_stage(stage: str, event_files: list[Path], plot_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    scalars = load_tensorboard_scalars(event_files)
    panels = [
        ("Reward", ["Train/mean_reward", "Train/mean_episode_length"]),
        ("PPO losses", ["Loss/value_function", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate"]),
        (
            "Motion errors",
            [
                "Metrics/motion/error_anchor_pos",
                "Metrics/motion/error_body_pos",
                "Metrics/motion/error_joint_pos",
                "Metrics/motion/error_obj_pos",
            ],
        ),
    ]
    available_panels = [(title, [tag for tag in tags if tag in scalars]) for title, tags in panels]
    available_panels = [(title, tags) for title, tags in available_panels if tags]
    if not available_panels:
        return None

    fig, axes = plt.subplots(len(available_panels), 1, figsize=(12, 4 * len(available_panels)), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    for axis, (title, tags) in zip(axes[:, 0], available_panels, strict=True):
        axis.set_facecolor("white")
        axis.patch.set_alpha(1.0)
        for tag in tags:
            values = scalars[tag]
            axis.plot([item[0] for item in values], [item[1] for item in values], label=tag)
        axis.set_title(f"Official SUGAR CarryBox {stage}: {title}")
        axis.set_xlabel("training iteration")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    output_path = plot_dir / f"{stage}_training_curves.png"
    fig.savefig(output_path, dpi=180, facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return output_path


def load_json_lines(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def plot_generator(log_path: Path, plot_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    records = load_json_lines(log_path)
    groups = [
        ("Loss", ["train_loss", "val_loss"]),
        ("Action MSE", ["train_action_mse_error", "val_action_mse_error"]),
    ]
    available_groups = []
    for title, keys in groups:
        present_keys = [key for key in keys if any(key in record for record in records)]
        if present_keys:
            available_groups.append((title, present_keys))
    if not available_groups:
        return None

    fig, axes = plt.subplots(len(available_groups), 1, figsize=(12, 4 * len(available_groups)), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    for axis, (title, keys) in zip(axes[:, 0], available_groups, strict=True):
        axis.set_facecolor("white")
        axis.patch.set_alpha(1.0)
        for key in keys:
            values = [
                (record.get("epoch", record.get("global_step")), record[key])
                for record in records
                if key in record and record.get("epoch", record.get("global_step")) is not None
            ]
            axis.plot([item[0] for item in values], [item[1] for item in values], label=key)
        axis.set_title(f"Official SUGAR CarryBox generator: {title}")
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    output_path = plot_dir / "generator_training_curves.png"
    fig.savefig(output_path, dpi=180, facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return output_path


def main() -> None:
    if socket.gethostname().startswith("mgmtserver"):
        raise SystemExit("Refusing visualization generation on a login/management node")

    args = parse_args()
    output_dir = args.output_dir.resolve()
    plot_dir = (args.plot_dir or (output_dir / "visualizations")).resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for stage in ("refiner", "tracker"):
        event_files = sorted((output_dir / "logs" / stage).glob("events.out.tfevents.*"))
        if event_files:
            output_path = plot_rl_stage(stage, event_files, plot_dir)
            if output_path is not None:
                outputs.append(output_path)

    generator_log = output_dir / "logs" / "generator" / "logs.json.txt"
    if generator_log.is_file():
        output_path = plot_generator(generator_log, plot_dir)
        if output_path is not None:
            outputs.append(output_path)

    if not outputs:
        raise SystemExit(f"No renderable official SUGAR training logs found under {output_dir}")
    for output_path in outputs:
        print(f"[SUGAR-TRAINING-VIS] wrote={output_path}")


if __name__ == "__main__":
    main()
