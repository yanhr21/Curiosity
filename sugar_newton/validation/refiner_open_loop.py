# SPDX-License-Identifier: BSD-3-Clause
"""Frozen official-Refiner transfer gate in the real Newton CarryBox loop.

The released Refiner is used exactly as SUGAR uses it: its checkpoint actor is rebuilt
with the compatible official ``rsl_rl.networks.MLP``, receives the exact 890-D teacher
observation, and emits the deployed 29-D joint-position action.  This validator never
trains, fits, or substitutes a policy.

The fixed default cohort contains twenty source motions spaced across CarryBox.  Every
profile starts at frame zero and runs deterministically until the official strict
termination or trajectory end.  The gate is deliberately physical: at least 80 percent
must lift the box by five centimetres and reach the trajectory end without a strict
failure, and no active profile may diverge.  A negative result directs Plan 16 to retrain
the teacher in Newton; it must not be hidden by starting Tracker training anyway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl import obs_890
from sugar_newton.rl.carrybox_env import CarryBoxEnv
from sugar_newton.rl.train_bcppo import activate_rsl_rl


DEFAULT_CLIPS = tuple(f"data_{motion_id:03d}" for motion_id in range(0, 100, 5))
TERMINATION_KEYS = (
    "anchor_ori", "anchor_pos", "ee_pos", "obj_pos", "obj_ori", "diverged"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_teacher(checkpoint: Path, device: torch.device):
    """Rebuild the exact checkpoint actor using SUGAR's own rsl_rl MLP class."""
    from rsl_rl.networks.mlp import MLP

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    weight_keys = sorted(
        (key for key in state if key.startswith("actor.") and key.endswith("weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if not weight_keys:
        raise KeyError("official Refiner checkpoint contains no actor weights")
    input_dim = int(state[weight_keys[0]].shape[1])
    output_dim = int(state[weight_keys[-1]].shape[0])
    hidden_dims = [int(state[key].shape[0]) for key in weight_keys[:-1]]
    if (input_dim, output_dim) != (obs_890.OBS_DIM_890, 29):
        raise ValueError(
            f"official Refiner geometry is {input_dim}->{output_dim}, expected 890->29"
        )
    model = MLP(input_dim, output_dim, hidden_dims, "elu").to(device)
    actor_state = {
        key.removeprefix("actor."): value for key, value in state.items()
        if key.startswith("actor.")
    }
    model.load_state_dict(actor_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, hidden_dims


@torch.inference_mode()
def evaluate_batch(env: CarryBoxEnv, teacher, motion_ids: list[int]) -> list[dict]:
    count = len(motion_ids)
    if count > env.num_envs:
        raise ValueError("batch is larger than the constructed Newton world count")
    padded = motion_ids + [motion_ids[0]] * (env.num_envs - count)
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset(ids, motion_ids=torch.tensor(padded, device=env.device), start_frames=0)

    initial_box = env._body_q()[:, env.box_body, :3].clone()
    initial_root = env._body_q()[:, env.body_idx[0], 2].clone()
    peak_box_z = initial_box[:, 2].clone()
    min_root_z = initial_root.clone()
    reward_sum = torch.zeros(env.num_envs, device=env.device)
    bilateral_steps = torch.zeros(env.num_envs, device=env.device)
    evaluated_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active[:count] = True
    finished = torch.zeros_like(active)
    timeout_at_finish = torch.zeros_like(active)
    all_finite = torch.ones_like(active)
    max_abs_action = torch.zeros(env.num_envs, device=env.device)
    reason_seen = {
        key: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for key in TERMINATION_KEYS
    }

    max_steps = int(env.ref["length"][torch.tensor(padded, device=env.device)].max()) + 2
    for _ in range(max_steps):
        if not bool(active[:count].any()):
            break
        teacher_obs = obs_890.build(env, teacher=True)
        action = teacher(teacher_obs)
        action = torch.where(active[:, None], action, torch.zeros_like(action))
        finite_action = torch.isfinite(action).all(dim=1)
        all_finite &= ~active | finite_action
        max_abs_action = torch.maximum(max_abs_action, action.abs().amax(dim=1))

        _, reward, done, extras = env.step(action)
        body_q = env._body_q()
        finite_state = (
            torch.isfinite(body_q).all(dim=(1, 2))
            & torch.isfinite(env.q).all(dim=1)
            & torch.isfinite(env.qd).all(dim=1)
        )
        all_finite &= ~active | finite_state
        peak_box_z = torch.where(
            active, torch.maximum(peak_box_z, body_q[:, env.box_body, 2]), peak_box_z
        )
        min_root_z = torch.where(
            active, torch.minimum(min_root_z, body_q[:, env.body_idx[0], 2]), min_root_z
        )
        reward_sum += torch.where(active, reward, torch.zeros_like(reward))
        evaluated_steps += active.long()

        hand_force = env.contact_history.box[:, :, env.contact_history.hand_idx]
        bilateral = (hand_force.norm(dim=-1).amax(dim=1) > 0.1).all(dim=1)
        bilateral_steps += (active & bilateral).float()

        newly_finished = active & done
        timeout_at_finish |= newly_finished & extras["timeout"]
        for key in TERMINATION_KEYS:
            reason_seen[key] |= newly_finished & extras["termination_terms"][key]
        finished |= newly_finished
        active &= ~newly_finished

    records = []
    for local_index, motion_index in enumerate(motion_ids):
        clip = env.clip_names[motion_index]
        length = int(env.ref["length"][motion_index])
        ref_z = env.ref["obj_pos"][motion_index, :length, 2]
        steps = int(evaluated_steps[local_index])
        strict_reasons = [key for key in TERMINATION_KEYS if bool(reason_seen[key][local_index])]
        records.append({
            "clip": clip,
            "start_frame": 0,
            "source_frames": length,
            "evaluated_steps": steps,
            "finished": bool(finished[local_index]),
            "trajectory_timeout": bool(timeout_at_finish[local_index]),
            "strict_failure": bool(strict_reasons),
            "strict_failure_reasons": strict_reasons,
            "all_finite": bool(all_finite[local_index]),
            "box_peak_lift_m": float(peak_box_z[local_index] - initial_box[local_index, 2]),
            "reference_peak_lift_m": float(ref_z.max() - ref_z[0]),
            "root_height_loss_m": float(initial_root[local_index] - min_root_z[local_index]),
            "root_below_0_65": bool(min_root_z[local_index] < 0.65),
            "bilateral_contact_fraction": float(
                bilateral_steps[local_index] / max(steps, 1)
            ),
            "mean_reward_per_step": float(reward_sum[local_index] / max(steps, 1)),
            "max_abs_action": float(max_abs_action[local_index]),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--official-refiner-base-checkpoint",
        type=Path,
        default=None,
        help="interpret checkpoint as a frozen-official-Refiner residual policy",
    )
    parser.add_argument("--motion-root", type=Path, default=Path("SUGAR/data/CarryBox"))
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--minimum-lift", type=float, default=0.05)
    parser.add_argument("--minimum-pass-fraction", type=float, default=0.80)
    parser.add_argument(
        "--torso-hull",
        action="store_true",
        help="diagnostic: replace only the torso mesh collider with its convex hull",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not 1 <= args.num_envs <= 8:
        raise SystemExit("--num-envs must be in the validated range 1..8")
    if not args.clips:
        raise SystemExit("the frozen gate requires at least one clip")
    missing = [clip for clip in args.clips if not (args.motion_root / clip).is_dir()]
    if missing:
        raise SystemExit(f"missing source clips: {missing}")

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("official Refiner gate must run on a Slurm CUDA compute node")
    activate_rsl_rl(args.rsl_rl_root)
    device = torch.device(args.device)
    residual_audit = None
    if args.official_refiner_base_checkpoint is None:
        teacher, hidden_dims = load_official_teacher(args.checkpoint, device)
    else:
        if not args.official_refiner_base_checkpoint.is_file():
            raise SystemExit(
                "official Refiner base checkpoint not found: "
                f"{args.official_refiner_base_checkpoint}"
            )
        sugar_rl_source = Path(__file__).resolve().parents[2] / "SUGAR/source/sugar_rl"
        if str(sugar_rl_source) not in sys.path:
            sys.path.insert(0, str(sugar_rl_source))
        from sugar_rl.utils.frozen_expert_transition_actor_critic import (
            FrozenOfficialRefinerResidual,
        )

        teacher = FrozenOfficialRefinerResidual(
            args.official_refiner_base_checkpoint
        ).to(device)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in payload["model_state_dict"].items()
            if key.startswith("actor.")
        }
        teacher.load_state_dict(actor_state, strict=True)
        teacher.eval().requires_grad_(False)
        base = torch.load(
            args.official_refiner_base_checkpoint,
            map_location=device,
            weights_only=False,
        )["model_state_dict"]
        expert_weight_delta = max(
            float(
                (
                    actor_state["expert." + key.removeprefix("actor.")] - value
                ).abs().max().item()
            )
            for key, value in base.items()
            if key.startswith("actor.")
        )
        if "std" in base:
            base_std = base["std"]
        elif "log_std" in base:
            base_std = base["log_std"].exp()
        else:
            raise KeyError("official Refiner checkpoint is missing std/log_std")
        expert_std_delta = float(
            (actor_state["expert_std"] - base_std).abs().max().item()
        )
        expert_delta = max(expert_weight_delta, expert_std_delta)
        if expert_delta != 0.0:
            raise RuntimeError(f"frozen official Refiner expert drift: {expert_delta}")
        residual_audit = {
            "official_refiner_base_checkpoint": str(
                args.official_refiner_base_checkpoint.resolve()
            ),
            "official_refiner_base_sha256": sha256(
                args.official_refiner_base_checkpoint
            ),
            "frozen_expert_parameter_max_delta": expert_delta,
            "frozen_expert_weight_max_delta": expert_weight_delta,
            "frozen_expert_std_max_delta": expert_std_delta,
            "residual_hidden_dims": [512, 256, 128],
            "residual_limit": float(teacher.residual_limit),
        }
        hidden_dims = [512, 256, 128]

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
        torso_hull=args.torso_hull,
        auto_reset=False,
    )
    records = []
    for begin in range(0, len(args.clips), args.num_envs):
        motion_ids = list(range(begin, min(begin + args.num_envs, len(args.clips))))
        records.extend(evaluate_batch(env, teacher, motion_ids))

    lifted = [record["box_peak_lift_m"] >= args.minimum_lift for record in records]
    strict_complete = [
        record["trajectory_timeout"] and not record["strict_failure"] for record in records
    ]
    finite = [record["all_finite"] for record in records]
    needed = math.ceil(args.minimum_pass_fraction * len(records))
    checks = {
        "checkpoint_actor_is_exact_890_to_29_official_mlp": True,
        "frozen_expert_residual_is_parameter_exact": (
            residual_audit is None
            or residual_audit["frozen_expert_parameter_max_delta"] == 0.0
        ),
        "all_profiles_finished": all(record["finished"] for record in records),
        "no_active_profile_diverged": all(finite) and not any(
            "diverged" in record["strict_failure_reasons"] for record in records
        ),
        "physical_lift_fraction_passes": sum(lifted) >= needed,
        "strict_completion_fraction_passes": sum(strict_complete) >= needed,
    }
    result = {
        "protocol": "sugar_newton_official_refiner_open_loop_gate_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "official_mlp_hidden_dims": hidden_dims,
        "frozen_expert_residual_audit": residual_audit,
        "motion_root": str(args.motion_root.resolve()),
        "num_profiles": len(records),
        "minimum_lift_m": args.minimum_lift,
        "minimum_pass_fraction": args.minimum_pass_fraction,
        "torso_hull_diagnostic": args.torso_hull,
        "torso_hull_original_shape_count": env.torso_hull_original_shape_count,
        "torso_hull_triangle_count": env.torso_hull_triangle_count,
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
    print(json.dumps({key: value for key, value in result.items() if key != "profiles"}, indent=2))
    print(f"wrote {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
