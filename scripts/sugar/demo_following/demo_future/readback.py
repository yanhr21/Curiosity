"""Read-only per-horizon and per-target analysis of the matched endpoints."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
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
    run = args.run
    arms = (args.control, args.treatment)
    result = json.loads((run / "RESULT.json").read_text())
    if not result["execution_completed"]:
        raise RuntimeError("Matched endpoints are incomplete")
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    dataset = Path(protocol.get("dataset", BASE / "dataset"))
    manifest = json.loads((BASE / "dataset/MANIFEST.json").read_text())
    with np.load(protocol.get("normalization_path", PARENT / "NORMALIZATION.npz"), allow_pickle=False) as z:
        scale = z["target_scale"]
    with np.load(dataset / "train/routing.npz", allow_pickle=False) as z:
        train_base_rows = len(z["base_task"])
    expected_steps = ((train_base_rows * 4 + protocol["batch_size"] - 1) // protocol["batch_size"]) * protocol["epochs"]
    execution = {}
    for arm in arms:
        trace = [json.loads(line) for line in (run / arm / "TRAIN_TRACE.jsonl").read_text().splitlines()]
        execution[arm] = {
            "steps": len(trace), "expected_steps": expected_steps,
            "contiguous_steps": [row["step"] for row in trace] == list(range(expected_steps)),
            "all_updates_applied": all(row["optimizer_applied"] for row in trace),
            "sample_count": sum(row["samples"] for row in trace),
            "expected_sample_count": train_base_rows * 4 * protocol["epochs"],
            "all_recorded_losses_and_gradients_finite": all(np.isfinite([row["loss"], row["base_gradient_norm"], row["aux_gradient_norm"], *row["horizon_nll"]]).all() for row in trace),
        }
        e = execution[arm]
        if not (e["contiguous_steps"] and e["all_updates_applied"] and e["sample_count"] == e["expected_sample_count"] and e["all_recorded_losses_and_gradients_finite"]):
            raise RuntimeError("Training execution evidence failed")
    detail = {}
    for split in ("validation", "test"):
        with np.load(dataset / split / "routing.npz", allow_pickle=False) as z:
            task, source = z["base_task"], z["base_source_motion_id"]
        target = np.log1p(np.load(dataset / split / "target_mismatch.npy") / scale)
        predictions = {}
        for arm in arms:
            with np.load(run / arm / f"{split}_predictions.npz", allow_pickle=False) as z:
                predictions[arm] = z["full"]
        detail[split] = {}
        for task_id in (0, 1):
            selection = task == task_id
            errors = {}
            for arm, pred in predictions.items():
                error = np.abs(pred - target)[:, :3].mean(axis=1)
                # Equal motion weights, preserving horizons and target channels.
                means = [error[selection & (source == sid)].mean(axis=0) for sid in np.unique(source[selection])]
                errors[arm] = np.mean(means, axis=0)
            horizon_rows = {}
            for horizon, offset in enumerate(manifest["future_start_offsets"]):
                horizon_rows[str(offset)] = {
                    name: {"control_mae": float(errors[args.control][horizon, k]),
                           "treatment_mae": float(errors[args.treatment][horizon, k]),
                           "ratio": float(errors[args.treatment][horizon, k] / max(errors[args.control][horizon, k], 1e-12))}
                    for k, name in enumerate(manifest["target_names"])
                }
            detail[split][str(task_id)] = horizon_rows
    failed = [key for key, passed in result["checks"].items() if not passed]
    next_action = protocol["if_pass"] if result["passed"] else protocol["if_fail"]
    report = {"arms": list(arms), "execution_audit": execution, "failed_scientific_checks": failed,
              "per_task_horizon_target": detail, "next_action": next_action,
              "scope": "Read-only analysis; original scientific thresholds unchanged; single training seed is exploratory"}
    with (run / "READBACK.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"execution_audit": execution, "failed_scientific_checks": failed, "next_action": next_action}), flush=True)


if __name__ == "__main__":
    main()
