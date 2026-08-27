# SPDX-License-Identifier: BSD-3-Clause
"""Fixed-20 physical ceiling for the exact released CarryBox Tracker.

This is a parameter-free teacher audit.  It uses the same Newton worlds,
raw-motion resets, physical metrics and termination rules as the acting-Refiner
gate, but the deployed actor is the exact released 510-D Tracker itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl.train_bcppo import activate_rsl_rl
from sugar_newton.validation.refiner_open_loop import (
    DEFAULT_CLIPS,
    evaluate_batch,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKER = ROOT / "SUGAR/demo_ckpts/CarryBox/tracker.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--motion-root", type=Path, default=ROOT / "SUGAR/data/CarryBox")
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--minimum-lift", type=float, default=0.05)
    parser.add_argument("--minimum-pass-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"released Tracker checkpoint not found: {args.checkpoint}")
    if not 1 <= args.num_envs <= 8:
        raise SystemExit("--num-envs must be in the validated range 1..8")
    if not args.clips:
        raise SystemExit("the Tracker ceiling requires at least one clip")
    missing = [clip for clip in args.clips if not (args.motion_root / clip).is_dir()]
    if missing:
        raise SystemExit(f"missing source clips: {missing}")

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("official Tracker ceiling must run on a Slurm CUDA node")
    activate_rsl_rl(args.rsl_rl_root)
    sugar_rl_source = ROOT / "SUGAR/source/sugar_rl"
    if str(sugar_rl_source) not in sys.path:
        sys.path.insert(0, str(sugar_rl_source))
    from sugar_rl.utils.frozen_expert_transition_actor_critic import (
        ACTION_DIM,
        OFFICIAL_HIDDEN_DIMS,
        TRACKER_OBSERVATION_DIM,
        _released_tracker,
    )

    device = torch.device(args.device)
    tracker, tracker_std = _released_tracker(args.checkpoint, device=device)
    first = tracker[0]
    last = tracker[-1]
    if not isinstance(first, torch.nn.Linear) or not isinstance(last, torch.nn.Linear):
        raise RuntimeError("released Tracker MLP geometry is not inspectable")
    geometry = (
        int(first.in_features),
        tuple(
            int(layer.out_features)
            for layer in tracker
            if isinstance(layer, torch.nn.Linear)
        ),
    )
    expected_outputs = (*OFFICIAL_HIDDEN_DIMS, ACTION_DIM)
    if geometry != (TRACKER_OBSERVATION_DIM, expected_outputs):
        raise RuntimeError(f"released Tracker geometry drift: {geometry}")

    from sugar_newton.rl.carrybox_env import CarryBoxEnv

    env = CarryBoxEnv(
        num_envs=args.num_envs,
        clip_names=list(args.clips),
        motion_root=args.motion_root,
        teacher_motion_root=args.motion_root,
        episode_length=10**9,
        substeps=args.substeps,
        mu=args.mu,
        device=args.device,
        seed=0,
        auto_reset=False,
    )
    records = []
    for begin in range(0, len(args.clips), args.num_envs):
        motion_ids = list(range(begin, min(begin + args.num_envs, len(args.clips))))
        records.extend(
            evaluate_batch(
                env,
                tracker,
                motion_ids,
                actor_observation="tracker",
            )
        )

    lifted = [record["box_peak_lift_m"] >= args.minimum_lift for record in records]
    strict_complete = [
        record["trajectory_timeout"] and not record["strict_failure"]
        for record in records
    ]
    finite = [record["all_finite"] for record in records]
    needed = math.ceil(args.minimum_pass_fraction * len(records))
    checks = {
        "checkpoint_actor_is_exact_official_510_to_512_256_128_to_29_mlp": True,
        "tracker_std_is_finite_29d": (
            tuple(tracker_std.shape) == (ACTION_DIM,)
            and bool(torch.isfinite(tracker_std).all())
        ),
        "all_profiles_finished": all(record["finished"] for record in records),
        "no_active_profile_diverged": all(finite) and not any(
            "diverged" in record["strict_failure_reasons"] for record in records
        ),
        "physical_lift_fraction_passes": sum(lifted) >= needed,
        "strict_completion_fraction_passes": sum(strict_complete) >= needed,
    }
    result = {
        "protocol": "sugar_newton_official_tracker_open_loop_ceiling_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "official_actor_geometry": [510, 512, 256, 128, 29],
        "motion_root": str(args.motion_root.resolve()),
        "num_profiles": len(records),
        "minimum_lift_m": args.minimum_lift,
        "minimum_pass_fraction": args.minimum_pass_fraction,
        "required_profile_count": needed,
        "lifted_profile_count": sum(lifted),
        "strict_complete_profile_count": sum(strict_complete),
        "finite_profile_count": sum(finite),
        "mean_peak_lift_m": sum(r["box_peak_lift_m"] for r in records) / len(records),
        "mean_bilateral_contact_fraction": sum(
            r["bilateral_contact_fraction"] for r in records
        ) / len(records),
        "checks": checks,
        "passed": all(checks.values()),
        "profiles": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
