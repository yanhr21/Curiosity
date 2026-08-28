"""Generate a Newton-native recovery action target by matched CEM shooting.

This is a training-label feasibility audit, not a deployed controller.  It
replays the parameter-exact released Refiner for 200 physical Newton steps,
captures the complete causal/solver frontier, and then searches a full 35-step,
29-D additive action sequence in the unchanged strict environment.  Every CEM
candidate is evaluated over the same eight restored Newton worlds.  Future
simulated states are used only by the offline shooting teacher and never enter
a policy observation.
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
from sugar_newton.validation.refiner_open_loop import (
    TERMINATION_KEYS,
    load_official_teacher,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFINER = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt"
)
REPLAY_TOLERANCE = 5.0e-4  # 0.5 mm / normalized-margin unit, fixed by smoke.


def triangle_pair_usage(env: CarryBoxEnv) -> tuple[int, int]:
    """Read Newton's public narrow-phase triangle-pair counter and capacity."""

    narrow_phase = env.pipeline.narrow_phase
    counter = narrow_phase.triangle_pairs_count
    if counter is None:
        return 0, int(narrow_phase.max_triangle_pairs)
    return int(wp.to_torch(counter)[0].item()), int(narrow_phase.max_triangle_pairs)


def capture_frontier(env: CarryBoxEnv) -> dict:
    """Capture every persistent quantity needed for an exact causal restart."""

    contact = env.contact_history
    return {
        "joint_q": env.q.detach().clone(),
        "joint_qd": env.qd.detach().clone(),
        "body_q": env._body_q().detach().clone(),
        "body_qd": env._body_qd().detach().clone(),
        "t": env.t.detach().clone(),
        "start": env.start.detach().clone(),
        "motion_id": env.motion_id.detach().clone(),
        "episode_initial_box_z": env.episode_initial_box_z.detach().clone(),
        "last_action": env.last_action.detach().clone(),
        "prev_action": env.prev_action.detach().clone(),
        "prev_qd": env.prev_qd.detach().clone(),
        "history": {key: value.detach().clone() for key, value in env.hist.items()},
        "contact_net": contact.net.detach().clone(),
        "contact_box": contact.box.detach().clone(),
        "foot_contact": contact.foot_contact.detach().clone(),
        "first_foot_contact": contact.first_foot_contact.detach().clone(),
        "current_air_time": contact.current_air_time.detach().clone(),
        "last_air_time": contact.last_air_time.detach().clone(),
        "qacc_warmstart": wp.to_torch(
            env.solver.mjw_data.qacc_warmstart
        ).detach().clone(),
        "solver_step": int(env.solver._step),
    }


def restore_frontier(env: CarryBoxEnv, snapshot: dict) -> float:
    """Restore a frontier and return the maximum tensor-exact audit delta."""

    world_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    matcher = getattr(env.pipeline, "_contact_matcher", None)
    if matcher is not None:
        matcher.reset()
    env.solver.reset(
        env.state_0,
        world_mask=wp.from_torch(world_mask),
        flags=0,
    )
    env.q.copy_(snapshot["joint_q"])
    env.qd.copy_(snapshot["joint_qd"])
    env._body_q().copy_(snapshot["body_q"])
    env._body_qd().copy_(snapshot["body_qd"])
    env.t.copy_(snapshot["t"])
    env.start.copy_(snapshot["start"])
    env.motion_id.copy_(snapshot["motion_id"])
    env.episode_initial_box_z.copy_(snapshot["episode_initial_box_z"])
    env.last_action = snapshot["last_action"].clone()
    env.prev_action = snapshot["prev_action"].clone()
    env.prev_qd = snapshot["prev_qd"].clone()
    env.hist = {key: value.clone() for key, value in snapshot["history"].items()}
    contact = env.contact_history
    contact.net = snapshot["contact_net"].clone()
    contact.box = snapshot["contact_box"].clone()
    contact.foot_contact = snapshot["foot_contact"].clone()
    contact.first_foot_contact = snapshot["first_foot_contact"].clone()
    contact.current_air_time = snapshot["current_air_time"].clone()
    contact.last_air_time = snapshot["last_air_time"].clone()
    wp.to_torch(env.solver.mjw_data.qacc_warmstart).copy_(
        snapshot["qacc_warmstart"]
    )
    env.solver._step = snapshot["solver_step"]

    deltas = [
        (env.q - snapshot["joint_q"]).abs().max(),
        (env.qd - snapshot["joint_qd"]).abs().max(),
        (env._body_q() - snapshot["body_q"]).abs().max(),
        (env._body_qd() - snapshot["body_qd"]).abs().max(),
        (
            wp.to_torch(env.solver.mjw_data.qacc_warmstart)
            - snapshot["qacc_warmstart"]
        ).abs().max(),
    ]
    deltas.extend(
        (env.hist[key] - value).abs().max()
        for key, value in snapshot["history"].items()
    )
    return float(torch.stack(deltas).max().item())


def replay_population(
    env: CarryBoxEnv,
    teacher,
    correction_knots: torch.Tensor,
    *,
    frontier: dict,
    steps_per_knot: int,
    collect_first: bool = False,
) -> tuple[list[dict], dict[str, torch.Tensor] | None, float]:
    """Evaluate one correction over the same restored eight-world frontier."""

    batch = correction_knots.shape[0]
    if batch != env.num_envs:
        raise ValueError("shooting batches must fill every constructed Newton world")
    action_dim = int(env.a_scale.numel())
    if correction_knots.shape[2] != action_dim:
        raise ValueError("correction action dimension drifted")
    horizon = correction_knots.shape[1] * steps_per_knot
    correction = correction_knots.repeat_interleave(steps_per_knot, dim=1)
    restore_delta = restore_frontier(env, frontier)

    initial_box_z = env.episode_initial_box_z.clone()
    active = torch.ones(batch, dtype=torch.bool, device=env.device)
    survived = torch.zeros(batch, dtype=torch.long, device=env.device)
    peak_lift = torch.zeros(batch, device=env.device)
    bilateral_steps = torch.zeros(batch, device=env.device)
    minimum_object_margin = torch.ones(batch, device=env.device)
    minimum_ee_margin = torch.ones(batch, device=env.device)
    maximum_triangle_pairs = 0
    triangle_pair_capacity = int(env.pipeline.narrow_phase.max_triangle_pairs)
    failure_seen = {
        key: torch.zeros(batch, dtype=torch.bool, device=env.device)
        for key in TERMINATION_KEYS
    }
    trajectory = (
        {"observation": [], "base_action": [], "action": [], "correction": []}
        if collect_first
        else None
    )

    for step in range(horizon):
        privileged = obs_890.build(env, teacher=True)
        with torch.inference_mode():
            base_action = teacher(privileged)
        action = base_action + correction[:, step]
        if trajectory is not None:
            trajectory["observation"].append(privileged[0].detach().cpu())
            trajectory["base_action"].append(base_action[0].detach().cpu())
            trajectory["action"].append(action[0].detach().cpu())
            trajectory["correction"].append(correction[0, step].detach().cpu())

        _, _, done, extras = env.step(action)
        triangle_pairs, triangle_pair_capacity = triangle_pair_usage(env)
        maximum_triangle_pairs = max(maximum_triangle_pairs, triangle_pairs)
        survived += active.long()
        body_q = env._body_q()
        lift = body_q[:, env.box_body, 2] - initial_box_z
        peak_lift = torch.where(active, torch.maximum(peak_lift, lift), peak_lift)
        terms = extras["physical_recovery_terms"]
        bilateral_steps += active.float() * terms["bilateral_contact"]
        minimum_object_margin = torch.where(
            active,
            torch.minimum(minimum_object_margin, terms["object_margin"]),
            minimum_object_margin,
        )
        minimum_ee_margin = torch.where(
            active,
            torch.minimum(minimum_ee_margin, terms["end_effector_margin"]),
            minimum_ee_margin,
        )
        newly_done = active & done
        for key in TERMINATION_KEYS:
            failure_seen[key] |= newly_done & extras["termination_terms"][key]
        active &= ~newly_done

    records: list[dict] = []
    correction_energy = correction.square().mean(dim=(1, 2))
    for index in range(batch):
        reasons = [key for key in TERMINATION_KEYS if bool(failure_seen[key][index])]
        records.append(
            {
                "alive_after_horizon": bool(active[index]),
                "survived_recovery_steps": int(survived[index]),
                "peak_lift_m": float(peak_lift[index]),
                "bilateral_contact_fraction": float(
                    bilateral_steps[index] / max(int(survived[index]), 1)
                ),
                "minimum_object_margin": float(minimum_object_margin[index]),
                "minimum_end_effector_margin": float(minimum_ee_margin[index]),
                "correction_rms": float(correction_energy[index].sqrt()),
                "maximum_triangle_pair_count": maximum_triangle_pairs,
                "triangle_pair_capacity": triangle_pair_capacity,
                "triangle_pair_overflow": (
                    maximum_triangle_pairs > triangle_pair_capacity
                ),
                "strict_failure_reasons": reasons,
            }
        )

    if trajectory is not None:
        trajectory = {key: torch.stack(value) for key, value in trajectory.items()}
    return records, trajectory, restore_delta


def aggregate_world_records(records: list[dict]) -> dict:
    """Fixed worst-case/mean record for one candidate on all Newton worlds."""

    if not records:
        raise ValueError("cannot aggregate an empty world record list")
    return {
        "alive_world_count": sum(record["alive_after_horizon"] for record in records),
        "minimum_survived_recovery_steps": min(
            record["survived_recovery_steps"] for record in records
        ),
        "minimum_object_margin": min(
            record["minimum_object_margin"] for record in records
        ),
        "minimum_peak_lift_m": min(record["peak_lift_m"] for record in records),
        "mean_peak_lift_m": sum(record["peak_lift_m"] for record in records)
        / len(records),
        "minimum_bilateral_contact_fraction": min(
            record["bilateral_contact_fraction"] for record in records
        ),
        "mean_bilateral_contact_fraction": sum(
            record["bilateral_contact_fraction"] for record in records
        )
        / len(records),
        "minimum_end_effector_margin": min(
            record["minimum_end_effector_margin"] for record in records
        ),
        "correction_rms": records[0]["correction_rms"],
        "maximum_triangle_pair_count": max(
            record["maximum_triangle_pair_count"] for record in records
        ),
        "triangle_pair_capacity": records[0]["triangle_pair_capacity"],
        "triangle_pair_overflow": any(
            record["triangle_pair_overflow"] for record in records
        ),
        "world_records": records,
    }


def world_records_match(first: list[dict], second: list[dict]) -> tuple[bool, float]:
    """Compare matched replays with categorical exactness and a fixed tolerance."""

    if len(first) != len(second):
        return False, float("inf")
    numeric_fields = (
        "peak_lift_m",
        "bilateral_contact_fraction",
        "minimum_object_margin",
        "minimum_end_effector_margin",
        "correction_rms",
    )
    maximum_delta = 0.0
    categorical_match = True
    for left, right in zip(first, second, strict=True):
        categorical_match &= (
            left["alive_after_horizon"] == right["alive_after_horizon"]
            and left["survived_recovery_steps"]
            == right["survived_recovery_steps"]
            and left["strict_failure_reasons"] == right["strict_failure_reasons"]
            and left.get("triangle_pair_overflow", False)
            == right.get("triangle_pair_overflow", False)
        )
        maximum_delta = max(
            maximum_delta,
            *(abs(left[field] - right[field]) for field in numeric_fields),
        )
    return categorical_match and maximum_delta <= REPLAY_TOLERANCE, maximum_delta


def rank_key(record: dict) -> tuple:
    """Fixed lexicographic strict-physics ordering; no scalar reward weights."""

    return (
        record["alive_world_count"],
        record["minimum_survived_recovery_steps"],
        record["minimum_object_margin"],
        record["minimum_peak_lift_m"],
        record["mean_peak_lift_m"],
        record["minimum_bilateral_contact_fraction"],
        record["mean_bilateral_contact_fraction"],
        record["minimum_end_effector_margin"],
        -record["correction_rms"],
    )


def conservative_replay_pair(
    first: dict,
    second: dict,
    *,
    replay_matches: bool,
    maximum_replay_delta: float,
) -> dict:
    """Build the fixed worst-of-two record used by robust matched-v3 CEM."""

    return {
        "replay_matches_fixed_tolerance": replay_matches,
        "maximum_replay_delta": maximum_replay_delta,
        "triangle_pair_overflow": (
            first["triangle_pair_overflow"] or second["triangle_pair_overflow"]
        ),
        "maximum_triangle_pair_count": max(
            first["maximum_triangle_pair_count"],
            second["maximum_triangle_pair_count"],
        ),
        "triangle_pair_capacity": first["triangle_pair_capacity"],
        "alive_world_count": min(
            first["alive_world_count"], second["alive_world_count"]
        ),
        "minimum_survived_recovery_steps": min(
            first["minimum_survived_recovery_steps"],
            second["minimum_survived_recovery_steps"],
        ),
        "minimum_object_margin": min(
            first["minimum_object_margin"], second["minimum_object_margin"]
        ),
        "minimum_peak_lift_m": min(
            first["minimum_peak_lift_m"], second["minimum_peak_lift_m"]
        ),
        "mean_peak_lift_m": min(
            first["mean_peak_lift_m"], second["mean_peak_lift_m"]
        ),
        "minimum_bilateral_contact_fraction": min(
            first["minimum_bilateral_contact_fraction"],
            second["minimum_bilateral_contact_fraction"],
        ),
        "mean_bilateral_contact_fraction": min(
            first["mean_bilateral_contact_fraction"],
            second["mean_bilateral_contact_fraction"],
        ),
        "minimum_end_effector_margin": min(
            first["minimum_end_effector_margin"],
            second["minimum_end_effector_margin"],
        ),
        "correction_rms": max(first["correction_rms"], second["correction_rms"]),
        "first": first,
        "second": second,
    }


def robust_rank_key(record: dict) -> tuple:
    """Put replay/collision validity ahead of the unchanged physics ordering."""

    return (
        int(
            record["replay_matches_fixed_tolerance"]
            and not record["triangle_pair_overflow"]
        ),
        int(not record["triangle_pair_overflow"]),
        int(record["replay_matches_fixed_tolerance"]),
        *rank_key(record),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refiner-checkpoint", type=Path, default=DEFAULT_REFINER)
    parser.add_argument("--motion-root", type=Path, default=ROOT / "SUGAR/data/CarryBox")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=171740)
    parser.add_argument(
        "--frontier-restore-smoke",
        action="store_true",
        help="replay the zero correction twice from one captured frontier",
    )
    parser.add_argument(
        "--robust-double-replay",
        action="store_true",
        help=(
            "matched-v3: evaluate every CEM candidate twice and rank only "
            "fixed-tolerance, overflow-free replay pairs"
        ),
    )
    args = parser.parse_args()

    # Fixed protocol constants are deliberately not CLI sweep axes.
    num_envs = 8
    population_size = 32
    elite_count = 8
    generations = 4
    prefix_steps = 200
    horizon = 35
    knot_count = 7
    steps_per_knot = 5
    initial_std = 0.35
    minimum_std = 0.05
    if horizon != knot_count * steps_per_knot:
        raise RuntimeError("fixed shooting horizon geometry drifted")

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("Newton-native shooting must run on a Slurm CUDA node")
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
    generator = torch.Generator(device=device).manual_seed(args.seed)
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
            raise RuntimeError("exact Refiner failed before the fixed shooting frontier")
    frontier = capture_frontier(env)
    frontier_world_state_spread = max(
        float((env.q - env.q[:1]).abs().max().item()),
        float((env.qd - env.qd[:1]).abs().max().item()),
    )

    if args.frontier_restore_smoke:
        zero = torch.zeros(num_envs, knot_count, action_dim, device=device)
        first_worlds, _, first_restore_delta = replay_population(
            env,
            teacher,
            zero,
            frontier=frontier,
            steps_per_knot=steps_per_knot,
        )
        second_worlds, _, second_restore_delta = replay_population(
            env,
            teacher,
            zero,
            frontier=frontier,
            steps_per_knot=steps_per_knot,
        )
        first = aggregate_world_records(first_worlds)
        second = aggregate_world_records(second_worlds)
        replay_matches, maximum_replay_delta = world_records_match(
            first_worlds, second_worlds
        )
        smoke_checks = {
            "frontier_restore_is_tensor_exact": max(
                first_restore_delta, second_restore_delta
            )
            == 0.0,
            "same_world_zero_replay_matches_fixed_tolerance": replay_matches,
            "zero_baseline_fails_obj_pos_in_all_worlds": (
                first["alive_world_count"] == 0
                and all(
                    record["survived_recovery_steps"] == horizon
                    and "obj_pos" in record["strict_failure_reasons"]
                    for record in first["world_records"]
                )
            ),
        }
        smoke = {
            "protocol": "sugar_newton_frontier_restore_smoke_v1",
            "seed": args.seed,
            "frontier_world_state_spread": frontier_world_state_spread,
            "replay_tolerance": REPLAY_TOLERANCE,
            "maximum_world_metric_replay_delta": maximum_replay_delta,
            "first": first,
            "second": second,
            "checks": smoke_checks,
            "passed": all(smoke_checks.values()),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "RESULT.json").write_text(
            json.dumps(smoke, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0 if smoke["passed"] else 1

    mean = torch.zeros(knot_count, action_dim, device=device)
    std = torch.full_like(mean, initial_std)
    generation_reports = []
    best_record = None
    best_knots = None
    maximum_frontier_restore_delta = 0.0

    for generation in range(generations):
        population = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            population_size,
            knot_count,
            action_dim,
            device=device,
            generator=generator,
        )
        population.clamp_(-1.0, 1.0)
        population[0].zero_()  # exact Refiner baseline in every generation
        population[1].copy_(mean.clamp(-1.0, 1.0))

        records = []
        for candidate_index in range(population_size):
            repeated_candidate = population[candidate_index : candidate_index + 1].expand(
                num_envs, -1, -1
            )
            world_records, _, restore_delta = replay_population(
                env,
                teacher,
                repeated_candidate,
                frontier=frontier,
                steps_per_knot=steps_per_knot,
            )
            maximum_frontier_restore_delta = max(
                maximum_frontier_restore_delta, restore_delta
            )
            first = aggregate_world_records(world_records)
            if args.robust_double_replay:
                second_world_records, _, second_restore_delta = replay_population(
                    env,
                    teacher,
                    repeated_candidate,
                    frontier=frontier,
                    steps_per_knot=steps_per_knot,
                )
                maximum_frontier_restore_delta = max(
                    maximum_frontier_restore_delta, second_restore_delta
                )
                second = aggregate_world_records(second_world_records)
                replay_matches, maximum_replay_delta = world_records_match(
                    first["world_records"], second["world_records"]
                )
                records.append(
                    conservative_replay_pair(
                        first,
                        second,
                        replay_matches=replay_matches,
                        maximum_replay_delta=maximum_replay_delta,
                    )
                )
            else:
                records.append(first)

        ranking = robust_rank_key if args.robust_double_replay else rank_key
        order = sorted(
            range(population_size), key=lambda i: ranking(records[i]), reverse=True
        )
        elites = population[order[:elite_count]]
        elite_mean = elites.mean(dim=0)
        elite_std = elites.std(dim=0, unbiased=False).clamp_min(minimum_std)
        mean = elite_mean
        std = 0.3 * std + 0.7 * elite_std
        candidate = records[order[0]]
        if best_record is None or ranking(candidate) > ranking(best_record):
            best_record = candidate
            best_knots = population[order[0]].detach().clone()
        generation_reports.append(
            {
                "generation": generation,
                "best": candidate,
                "all_world_alive_candidate_count": sum(
                    r["alive_world_count"] == num_envs for r in records
                ),
                **(
                    {
                        "fixed_tolerance_replay_candidate_count": sum(
                            r["replay_matches_fixed_tolerance"] for r in records
                        ),
                        "overflow_free_candidate_count": sum(
                            not r["triangle_pair_overflow"] for r in records
                        ),
                        "fully_robust_candidate_count": sum(
                            r["replay_matches_fixed_tolerance"]
                            and not r["triangle_pair_overflow"]
                            for r in records
                        ),
                    }
                    if args.robust_double_replay
                    else {}
                ),
                "baseline": records[0],
                "elite_mean_correction_rms": float(elites.square().mean().sqrt()),
                "mean_std": float(std.mean()),
            }
        )
        print(json.dumps(generation_reports[-1], sort_keys=True), flush=True)

    if best_knots is None or best_record is None:
        raise RuntimeError("shooting search produced no candidate")
    replay_knots = best_knots.unsqueeze(0).expand(num_envs, -1, -1).clone()
    replay_world_records, trajectory, restore_delta = replay_population(
        env,
        teacher,
        replay_knots,
        frontier=frontier,
        steps_per_knot=steps_per_knot,
        collect_first=True,
    )
    maximum_frontier_restore_delta = max(
        maximum_frontier_restore_delta, restore_delta
    )
    replay = aggregate_world_records(replay_world_records)
    if trajectory is None:
        raise RuntimeError("best trajectory collection failed")

    teacher_after = teacher.state_dict()
    teacher_parameter_delta = max(
        float((teacher_after[key] - value).abs().max())
        for key, value in teacher_before.items()
    )
    if args.robust_double_replay:
        first_final_matches, first_final_delta = world_records_match(
            best_record["first"]["world_records"], replay_world_records
        )
        second_final_matches, second_final_delta = world_records_match(
            best_record["second"]["world_records"], replay_world_records
        )
        replay_matches = first_final_matches and second_final_matches
        maximum_replay_delta = max(first_final_delta, second_final_delta)
    else:
        replay_matches, maximum_replay_delta = world_records_match(
            best_record["world_records"], replay_world_records
        )
    baseline_record = generation_reports[0]["baseline"]
    if args.robust_double_replay:
        baseline_record = baseline_record["first"]
    checks = {
        "official_refiner_parameter_exact": teacher_parameter_delta == 0.0,
        "frontier_restore_is_tensor_exact": maximum_frontier_restore_delta == 0.0,
        "baseline_reproduces_step235_obj_pos_failure_in_all_worlds": (
            baseline_record["alive_world_count"] == 0
            and all(
                record["survived_recovery_steps"] == horizon
                and not record["alive_after_horizon"]
                and "obj_pos" in record["strict_failure_reasons"]
                for record in baseline_record["world_records"]
            )
        ),
        "best_replay_matches_fixed_tolerance": replay_matches,
        "best_replay_has_no_triangle_pair_overflow": not replay[
            "triangle_pair_overflow"
        ],
        "best_candidate_crosses_step235_in_all_worlds": (
            replay["alive_world_count"] == num_envs
        ),
        **(
            {
                "selected_pair_matches_fixed_tolerance": best_record[
                    "replay_matches_fixed_tolerance"
                ],
                "selected_pair_has_no_triangle_pair_overflow": not best_record[
                    "triangle_pair_overflow"
                ],
            }
            if args.robust_double_replay
            else {}
        ),
    }
    result = {
        "protocol": (
            "sugar_newton_native_robust_matched_action_shooting_v3"
            if args.robust_double_replay
            else "sugar_newton_native_matched_action_shooting_v2"
        ),
        "seed": args.seed,
        "refiner_checkpoint": str(args.refiner_checkpoint.resolve()),
        "refiner_sha256": sha256(args.refiner_checkpoint),
        "motion": "data_000",
        "prefix_steps": prefix_steps,
        "horizon": horizon,
        "action_dim": action_dim,
        "knot_count": knot_count,
        "steps_per_knot": steps_per_knot,
        "population_size": population_size,
        "elite_count": elite_count,
        "generations": generations,
        "initial_std": initial_std,
        "minimum_std": minimum_std,
        "correction_bound": 1.0,
        "matched_world_count_per_candidate": num_envs,
        "independent_replays_per_candidate": (
            2 if args.robust_double_replay else 1
        ),
        "final_unselected_replay_count": 1,
        "triangle_pair_capacity": replay["triangle_pair_capacity"],
        "future_state_is_training_label_only": True,
        "deployed_actor_observation_augmented": False,
        "teacher_parameter_max_delta": teacher_parameter_delta,
        "frontier_world_state_spread": frontier_world_state_spread,
        "maximum_frontier_restore_delta": maximum_frontier_restore_delta,
        "replay_tolerance": REPLAY_TOLERANCE,
        "maximum_world_metric_replay_delta": maximum_replay_delta,
        "generation_reports": generation_reports,
        "best_search_record": best_record,
        "best_replay_record": replay,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    torch.save(
        {
            "protocol": result["protocol"],
            "correction_knots": best_knots.cpu(),
            "steps_per_knot": steps_per_knot,
            "record": replay,
        },
        args.output / "BEST_CORRECTION.pt",
    )
    torch.save(trajectory, args.output / "BEST_TRAJECTORY.pt")
    print(json.dumps({key: value for key, value in result.items() if key != "generation_reports"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
