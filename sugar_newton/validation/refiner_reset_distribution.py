# SPDX-License-Identifier: BSD-3-Clause
"""Audit the exact reset distribution used by Newton Refiner PPO transfer.

The frozen physical gate always starts at frame zero.  Training instead samples uniformly
from every reference frame that leaves one full episode.  Since ``CarryBoxEnv.reset`` writes
the reference object pose into Newton state, a nonzero sampled frame can initialize the box
already lifted.  This script measures that distinction from the same validated clip loader and
the exact span formula used by the environment; it does not run or approximate a policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sugar_newton.rl.carrybox_env import load_clips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=Path("SUGAR/data/CarryBox"))
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--minimum-lift", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.episode_length < 1:
        raise SystemExit("--episode-length must be positive")
    if args.minimum_lift <= 0.0:
        raise SystemExit("--minimum-lift must be positive")
    clip_names = sorted(path.name for path in args.motion_root.glob("data_*") if path.is_dir())
    if not clip_names:
        raise SystemExit(f"no data_* clips under {args.motion_root}")

    clips = load_clips(args.motion_root, clip_names)
    profile_records = []
    for motion_id, clip_name in enumerate(clip_names):
        length = int(clips["length"][motion_id])
        # Exact CarryBoxEnv.reset formula: span=max(length-episode_length-1, 1),
        # followed by floor(U[0,1) * span), hence valid starts are 0..span-1.
        span = max(length - args.episode_length - 1, 1)
        starts = np.arange(span, dtype=np.int64)
        object_z = clips["obj_pos"][motion_id, :length, 2]
        lift_from_frame_zero = object_z[starts] - object_z[0]
        prelifted = lift_from_frame_zero >= args.minimum_lift
        contact = clips["contact"][motion_id, starts] > 0.5
        profile_records.append(
            {
                "clip": clip_name,
                "source_frames": length,
                "valid_training_start_count": span,
                "frame_zero_sampling_probability": 1.0 / span,
                "prelifted_start_count": int(prelifted.sum()),
                "prelifted_start_fraction": float(prelifted.mean()),
                "contact_positive_start_fraction": float(contact.mean()),
                "mean_start_lift_from_frame_zero_m": float(lift_from_frame_zero.mean()),
                "maximum_start_lift_from_frame_zero_m": float(lift_from_frame_zero.max()),
                "reference_peak_lift_from_frame_zero_m": float(object_z.max() - object_z[0]),
            }
        )

    frame_zero_probability = float(
        np.mean([record["frame_zero_sampling_probability"] for record in profile_records])
    )
    prelifted_fraction = float(
        np.mean([record["prelifted_start_fraction"] for record in profile_records])
    )
    contact_fraction = float(
        np.mean([record["contact_positive_start_fraction"] for record in profile_records])
    )
    mean_start_lift = float(
        np.mean([record["mean_start_lift_from_frame_zero_m"] for record in profile_records])
    )
    checks = {
        "all_100_carrybox_profiles_present": len(profile_records) == 100,
        "every_profile_has_a_valid_training_start": all(
            record["valid_training_start_count"] >= 1 for record in profile_records
        ),
        "training_can_initialize_above_the_frozen_gate_lift_threshold": any(
            record["prelifted_start_count"] > 0 for record in profile_records
        ),
        "frozen_frame_zero_has_less_than_one_percent_training_probability": (
            frame_zero_probability < 0.01
        ),
    }
    result = {
        "protocol": "sugar_newton_refiner_reset_distribution_audit_v1",
        "motion_root": str(args.motion_root.resolve()),
        "episode_length": args.episode_length,
        "minimum_lift_m": args.minimum_lift,
        "num_profiles": len(profile_records),
        "expected_frame_zero_sampling_probability": frame_zero_probability,
        "expected_prelifted_reset_fraction": prelifted_fraction,
        "expected_contact_positive_reset_fraction": contact_fraction,
        "expected_start_lift_from_frame_zero_m": mean_start_lift,
        "profiles_with_any_prelifted_start": sum(
            record["prelifted_start_count"] > 0 for record in profile_records
        ),
        "checks": checks,
        "reset_distribution_mismatch": all(checks.values()),
        "profiles": profile_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "profiles"}, indent=2))
    print(f"wrote {args.output}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
