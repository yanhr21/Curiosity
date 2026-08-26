#!/usr/bin/env python3
"""Score frozen CHORD-OFF/ON rollouts against reconstructed KickBox21 geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher
from isaaclab.utils.math import quat_from_matrix


ROOT = Path(__file__).resolve().parents[3]
NUM_BASIS = 512
SUPPORT_THRESHOLD = 1.0e-3


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT
        / "experiments/demo_following/sugar_demo_chord_geometry_v2/kick21/contact_geometry.npz",
    )
    parser.add_argument(
        "--official-chord-root",
        type=Path,
        default=ROOT / "experiments/runtime_assets/official_chord_5654c50e",
    )
    parser.add_argument(
        "--object-usd",
        type=Path,
        default=ROOT / "SUGAR/descriptions/objects/big_box/obj_aligned.usd",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _support_engine(args, device: torch.device):
    from sugar_rl.utils.official_chord_runtime_reward import (
        _basis,
        _load_official_utils,
        _object_mesh_radius,
    )

    chord, source = _load_official_utils(args.official_chord_root.resolve())
    radius = _object_mesh_radius(args.object_usd.resolve())
    basis = _basis(device)
    theta = torch.linspace(0.0, 2.0 * torch.pi, steps=9, device=device)[:-1]
    return chord, source, radius, basis, torch.cos(theta).view(1, -1, 1), torch.sin(
        theta
    ).view(1, -1, 1)


def _supports(
    chord,
    radius: float,
    basis: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    positions: torch.Tensor,
    forces: torch.Tensor,
    object_position: torch.Tensor,
    object_quat: torch.Tensor,
) -> torch.Tensor:
    count = positions.shape[0]
    points_com, normals_com = chord.wrench_preprocess_jit(
        contact_positions_w=positions.reshape(count, 1, 1, 3),
        contact_forces_first_hist_w=forces.reshape(count, 1, 1, 3),
        object_com_position_w=object_position.reshape(count, 1, 1, 3),
        object_com_orientation_w=object_quat.reshape(count, 1, 1, 4),
        num_envs=count,
        num_bodies=1,
        num_robot_contacts=1,
    )
    return chord.wrench_support_one_body_jit(
        contact_points=points_com[:, 0],
        contact_normals=normals_com[:, 0],
        cos_t=cos_t,
        sin_t=sin_t,
        basis=basis,
        rc=radius,
        friction_coefficients=0.1,
    )


def _reference(args, engine, device: torch.device) -> torch.Tensor:
    chord, _, radius, basis, cos_t, sin_t = engine
    with np.load(args.reference, allow_pickle=False) as archive:
        active = torch.as_tensor(archive["contact_active"], device=device)
        position = torch.as_tensor(
            archive["hand_contact_position_w"], device=device, dtype=torch.float32
        )
        normal = torch.as_tensor(
            archive["hand_contact_normal_w"], device=device, dtype=torch.float32
        )
        object_position = torch.as_tensor(
            archive["object_position_w"], device=device, dtype=torch.float32
        )
        object_rotation = torch.as_tensor(
            archive["object_rotation_w"], device=device, dtype=torch.float32
        )
    position = torch.where(active.unsqueeze(-1), position, torch.zeros_like(position))
    normal = torch.where(active.unsqueeze(-1), normal, torch.zeros_like(normal))
    object_quat = quat_from_matrix(object_rotation)
    return torch.stack(
        [
            _supports(
                chord,
                radius,
                basis,
                cos_t,
                sin_t,
                position[:, side],
                normal[:, side],
                object_position,
                object_quat,
            )
            for side in range(2)
        ],
        dim=1,
    )


def _score(path: Path, reference: torch.Tensor, engine, device: torch.device) -> dict:
    chord, _, radius, basis, cos_t, sin_t = engine
    with np.load(path / "trace.npz", allow_pickle=False) as archive:
        position = torch.as_tensor(
            archive["foot_contact_position_w"], device=device, dtype=torch.float32
        )
        valid = torch.as_tensor(archive["foot_contact_position_valid"], device=device)
        force = torch.as_tensor(
            archive["foot_contact_force_w"], device=device, dtype=torch.float32
        )
        obj = torch.as_tensor(
            archive["object_root_state_w"], device=device, dtype=torch.float32
        )
        frames = torch.as_tensor(archive["motion_frame"], device=device, dtype=torch.long)
    steps, envs = frames.shape
    force_valid = torch.linalg.vector_norm(force, dim=-1) > 0.1
    valid = valid & force_valid
    position = torch.where(valid.unsqueeze(-1), position, torch.zeros_like(position))
    force = torch.where(valid.unsqueeze(-1), force, torch.zeros_like(force))
    current = torch.stack(
        [
            _supports(
                chord,
                radius,
                basis,
                cos_t,
                sin_t,
                position[:, :, side].reshape(-1, 3),
                force[:, :, side].reshape(-1, 3),
                obj[:, :, :3].reshape(-1, 3),
                obj[:, :, 3:7].reshape(-1, 4),
            ).reshape(steps, envs, NUM_BASIS)
            for side in range(2)
        ],
        dim=2,
    )
    command = reference[frames]
    cur = current.reshape(-1, 2, NUM_BASIS)
    cmd = command.reshape(-1, 2, NUM_BASIS)
    cmd_active = cmd > SUPPORT_THRESHOLD
    cur_active = cur > SUPPORT_THRESHOLD
    cmd_body = cmd_active.any(dim=-1, keepdim=True)
    cur_body = cur_active.any(dim=-1, keepdim=True)
    cws = chord.contact_wrench_support_reward_jit(
        right_cmd_active=cmd_active[:, 1].unsqueeze(1),
        right_cur_active=cur_active[:, 1].unsqueeze(1),
        left_cmd_active=cmd_active[:, 0].unsqueeze(1),
        left_cur_active=cur_active[:, 0].unsqueeze(1),
        right_cmd_active_per_body=cmd_body[:, 1],
        left_cmd_active_per_body=cmd_body[:, 0],
        right_cmd_supports=cmd[:, 1].unsqueeze(1),
        right_cur_supports=cur[:, 1].unsqueeze(1),
        left_cmd_supports=cmd[:, 0].unsqueeze(1),
        left_cur_supports=cur[:, 0].unsqueeze(1),
        tolerance=0.1,
        var=0.1,
    )
    missed = chord.missed_contact_penalty_jit(
        right_cmd_active=cmd_active[:, 1].unsqueeze(1),
        right_cur_active=cur_active[:, 1].unsqueeze(1),
        left_cmd_active=cmd_active[:, 0].unsqueeze(1),
        left_cur_active=cur_active[:, 0].unsqueeze(1),
        right_cmd_active_per_body=cmd_body[:, 1],
        left_cmd_active_per_body=cmd_body[:, 0],
    )
    unintended = chord.unintended_contact_penalty_jit(
        right_cmd_active_per_body=cmd_body[:, 1],
        right_cur_active_per_body=cur_body[:, 1],
        left_cmd_active_per_body=cmd_body[:, 0],
        left_cur_active_per_body=cur_body[:, 0],
        right_cur_supports=cur[:, 1].unsqueeze(1),
        left_cur_supports=cur[:, 0].unsqueeze(1),
        num_bodies=1,
    )
    active = cmd_body.any(dim=1).squeeze(-1)
    result = json.loads((path / "RESULT.json").read_text(encoding="utf-8"))
    return {
        "mean_cws_on_reference_contact": float(cws[active].mean().item()),
        "mean_missed_on_reference_contact": float(missed[active].mean().item()),
        "mean_unintended_all_frames": float(unintended.mean().item()),
        "reference_contact_samples": int(active.sum().item()),
        "safe_kick_success_count": int(result["aggregate"]["safe_kick_success_count"]),
        "physical_fall_count": int(result["aggregate"]["physical_fall_count"]),
    }


def main() -> None:
    args = _args()
    args.headless = True
    app = AppLauncher(args).app
    try:
        device = torch.device(args.device)
        engine = _support_engine(args, device)
        reference = _reference(args, engine, device)
        rows = {}
        for arm in ("off", "on"):
            for prefix in (37, 45, 53, 61):
                rows[f"{arm}_prefix{prefix}"] = _score(
                    args.pair_root
                    / arm
                    / f"evaluation/prefix{prefix}/learned_kick",
                    reference,
                    engine,
                    device,
                )
        means = {}
        for arm in ("off", "on"):
            selected = [rows[f"{arm}_prefix{prefix}"] for prefix in (37, 45, 53, 61)]
            means[arm] = {
                key: sum(row[key] for row in selected) / len(selected)
                for key in (
                    "mean_cws_on_reference_contact",
                    "mean_missed_on_reference_contact",
                    "mean_unintended_all_frames",
                )
            }
        payload = {
            "protocol": "sugar_official_chord_causal_pair_geometry_eval_v1",
            "official_source": str(engine[1].resolve()),
            "reference": str(args.reference.resolve()),
            "rows": rows,
            "means": means,
            "on_minus_off": {
                key: means["on"][key] - means["off"][key]
                for key in means["off"]
            },
            "all_finite": bool(
                all(
                    np.isfinite(value)
                    for row in rows.values()
                    for value in row.values()
                )
            ),
            "claim_boundary": (
                "Frozen rollout representation score; physical benefit is decided separately "
                "from safe-kick and fall outcomes."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not payload["all_finite"]:
            raise SystemExit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
