"""Audit Newton shooting corrections in simultaneous matched world pairs.

Serial frontier restoration made high-survival shooting candidates sensitive to
contact execution order.  This bounded diagnostic constructs two tensor-exact
copies of each of eight physical frontiers inside one Newton environment and
executes each matched pair in the same ``env.step`` call.  It changes replay
topology only: the released Refiner, saved v2 correction, horizon, strict
termination rules and fixed replay tolerance are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl import obs_890
from sugar_newton.rl.carrybox_env import CarryBoxEnv
from sugar_newton.rl.train_bcppo import activate_rsl_rl
from sugar_newton.validation.newton_native_action_shooting import (
    DEFAULT_REFINER,
    REPLAY_TOLERANCE,
    aggregate_world_records,
    capture_frontier,
    replay_population,
    world_records_match,
)
from sugar_newton.validation.refiner_open_loop import (
    load_official_teacher,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "experiments/sugar_reproduction/outputs/newton_native_action_shooting_20260828"
    / "seed171740_data000_prefix200_h35_matched8_v2"
)


def _paired_snapshot(snapshot: dict, pair_count: int) -> dict:
    """Copy the first matched block into the second block tensor-exactly."""

    world_count = 2 * pair_count

    def pair_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0 or value.shape[0] != world_count:
            raise RuntimeError(
                f"frontier field {name!r} does not expose {world_count} worlds: "
                f"shape={tuple(value.shape)}"
            )
        paired = value.clone()
        paired[pair_count:].copy_(paired[:pair_count])
        return paired

    paired: dict = {}
    for key, value in snapshot.items():
        if key == "history":
            paired[key] = {
                history_key: pair_tensor(
                    f"history.{history_key}", history_value
                )
                for history_key, history_value in value.items()
            }
        elif torch.is_tensor(value):
            paired[key] = pair_tensor(key, value)
        else:
            paired[key] = value
    return paired


def _maximum_pair_delta(snapshot: dict, pair_count: int) -> float:
    deltas: list[torch.Tensor] = []
    for key, value in snapshot.items():
        values = value.values() if key == "history" else [value]
        for tensor in values:
            if torch.is_tensor(tensor):
                first = tensor[:pair_count]
                second = tensor[pair_count:]
                if tensor.dtype == torch.bool:
                    deltas.append((first != second).float().max())
                else:
                    deltas.append((first - second).abs().max())
    return float(torch.stack(deltas).max().item())


def _load_correction(path: Path, *, action_dim: int, device: torch.device) -> torch.Tensor:
    correction = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(correction, dict):
        for key in ("correction", "correction_knots", "best_knots"):
            if key in correction:
                correction = correction[key]
                break
    correction = torch.as_tensor(correction, dtype=torch.float32, device=device)
    if correction.shape != (7, action_dim):
        raise RuntimeError(
            "saved shooting correction geometry drifted: "
            f"expected {(7, action_dim)}, got {tuple(correction.shape)}"
        )
    return correction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=171743)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--motion-root", type=Path, default=ROOT / "SUGAR/data/CarryBox"
    )
    parser.add_argument("--refiner-checkpoint", type=Path, default=DEFAULT_REFINER)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", type=Path, required=True)
    args = parser.parse_args()

    pair_count = 8
    num_envs = 2 * pair_count
    prefix_steps = 200
    horizon = 35
    knot_count = 7
    steps_per_knot = 5

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("simultaneous Newton replay must run on a Slurm CUDA node")
    activate_rsl_rl(args.rsl_rl_root)
    device = torch.device(args.device)
    teacher, hidden_dims = load_official_teacher(args.refiner_checkpoint, device)
    if hidden_dims != [512, 256, 128]:
        raise RuntimeError(f"official Refiner geometry drifted: {hidden_dims}")
    teacher_before = {
        key: value.detach().clone() for key, value in teacher.state_dict().items()
    }

    env = CarryBoxEnv(
        num_envs=num_envs,
        clip_names=["data_000"],
        motion_root=args.motion_root,
        teacher_motion_root=args.motion_root,
        episode_length=10**9,
        substeps=4,
        mu=1.0,
        device=args.device,
        seed=args.seed,
        auto_reset=False,
    )
    action_dim = int(env.a_scale.numel())
    ids = torch.arange(num_envs, device=device)
    env.reset(
        ids,
        motion_ids=torch.zeros(num_envs, device=device),
        start_frames=0,
    )
    for _ in range(prefix_steps):
        privileged = obs_890.build(env, teacher=True)
        with torch.inference_mode():
            exact_action = teacher(privileged)
        _, _, done, _ = env.step(exact_action)
        if bool(done.any()):
            raise RuntimeError("exact Refiner failed before the paired frontier")

    frontier = _paired_snapshot(capture_frontier(env), pair_count)
    maximum_frontier_pair_delta = _maximum_pair_delta(frontier, pair_count)
    zero_knots = torch.zeros(num_envs, knot_count, action_dim, device=device)
    zero_records, _, zero_restore_delta = replay_population(
        env,
        teacher,
        zero_knots,
        frontier=frontier,
        steps_per_knot=steps_per_knot,
    )
    zero_matches, zero_pair_delta = world_records_match(
        zero_records[:pair_count], zero_records[pair_count:]
    )
    zero = aggregate_world_records(zero_records)

    correction_path = args.source / "BEST_CORRECTION.pt"
    source_correction = _load_correction(
        correction_path, action_dim=action_dim, device=device
    )
    source_knots = source_correction.unsqueeze(0).expand(
        num_envs, -1, -1
    ).clone()
    source_records, _, source_restore_delta = replay_population(
        env,
        teacher,
        source_knots,
        frontier=frontier,
        steps_per_knot=steps_per_knot,
    )
    source_matches, source_pair_delta = world_records_match(
        source_records[:pair_count], source_records[pair_count:]
    )
    source = aggregate_world_records(source_records)

    teacher_delta = max(
        float((value - teacher_before[key]).abs().max().item())
        for key, value in teacher.state_dict().items()
    )
    zero_obj_pos = (
        zero["alive_world_count"] == 0
        and all(
            record["survived_recovery_steps"] == horizon
            and "obj_pos" in record["strict_failure_reasons"]
            for record in zero_records
        )
    )
    source_strict = (
        source["alive_world_count"] == num_envs
        and all(
            record["survived_recovery_steps"] == horizon
            and not record["strict_failure_reasons"]
            for record in source_records
        )
    )
    checks = {
        "official_refiner_parameter_exact": teacher_delta == 0.0,
        "paired_frontier_is_tensor_exact": maximum_frontier_pair_delta == 0.0,
        "frontier_restore_is_tensor_exact": max(
            zero_restore_delta, source_restore_delta
        )
        == 0.0,
        "zero_pairs_match_fixed_tolerance": zero_matches,
        "zero_baseline_reproduces_step235_obj_pos_in_all_copies": zero_obj_pos,
        "source_pairs_match_fixed_tolerance": source_matches,
        "source_correction_is_strict_in_all_copies": source_strict,
        "zero_replay_has_no_triangle_pair_overflow": not zero[
            "triangle_pair_overflow"
        ],
        "source_replay_has_no_triangle_pair_overflow": not source[
            "triangle_pair_overflow"
        ],
    }
    result = {
        "protocol": "sugar_newton_native_simultaneous_matched_replay_v4_smoke",
        "seed": args.seed,
        "pair_count": pair_count,
        "world_count": num_envs,
        "prefix_steps": prefix_steps,
        "horizon": horizon,
        "replay_tolerance": REPLAY_TOLERANCE,
        "refiner_checkpoint": str(args.refiner_checkpoint.resolve()),
        "refiner_sha256": sha256(args.refiner_checkpoint),
        "source_correction": str(correction_path.resolve()),
        "source_correction_sha256": sha256(correction_path),
        "teacher_parameter_max_delta": teacher_delta,
        "maximum_frontier_pair_delta": maximum_frontier_pair_delta,
        "maximum_frontier_restore_delta": max(
            zero_restore_delta, source_restore_delta
        ),
        "zero_pair_metric_delta": zero_pair_delta,
        "source_pair_metric_delta": source_pair_delta,
        "zero": zero,
        "source": source,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
