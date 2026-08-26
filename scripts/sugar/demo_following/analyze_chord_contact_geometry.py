#!/usr/bin/env python3
"""Apply the released CHORD wrench representation to frozen SUGAR rollouts.

This is a representation diagnostic, not a CHORD policy result.  The SUGAR
Carry45/Kick21 archives do not contain reference contact points or normals, so
the non-fabricated reference here is the released Kick21 expert's own online
PhysX contact geometry from one fixed, successful expert rollout, indexed by
the exact SUGAR motion frame.  A per-frame median across randomized expert
environments is invalid here: their contact events are asynchronous, so the
median silently erases a real but temporally sparse kick.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_CHORD_COMMIT = "5654c50edc1f3dea8e3145bf2dbfc277dbf27b4c"
NUM_BASIS = 512
NUM_FRICTION_EDGES = 8
FRICTION_COEFFICIENT = 0.1
SUPPORT_THRESHOLD = 1.0e-3
BASIS_SEED = 26070033


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _load_official_chord_utils(checkout: Path):
    source = (
        checkout
        / "robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/utils_jit.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"official CHORD utils_jit.py is missing: {source}")
    spec = importlib.util.spec_from_file_location("official_chord_utils_jit", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official CHORD utils_jit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def _object_mesh_radius(path: Path) -> float:
    """Match CHORD's maximum vertex radius about the mesh centroid."""

    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"cannot open object USD: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points: list[np.ndarray] = []
    predicate = Usd.TraverseInstanceProxies()
    for prim in Usd.PrimRange.Stage(stage, predicate):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        value = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not value:
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        transformed = [
            transform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            for p in value
        ]
        points.append(np.asarray(transformed, dtype=np.float64))
    if not points:
        raise RuntimeError("object USD contains no readable mesh vertices")
    vertices = np.concatenate(points, axis=0)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(centered, axis=-1).max())
    if not np.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("object mesh radius is invalid")
    return radius


def _official_basis(device: torch.device) -> torch.Tensor:
    """Exact body of CHORD sample_wrench_space_basis_scaled(rc=1)."""

    generator_state = torch.random.get_rng_state()
    torch.manual_seed(BASIS_SEED)
    basis = torch.randn(NUM_BASIS, 6, device=device, dtype=torch.float32)
    basis[:, 3:] = basis[:, 3:] / 1.0
    basis = basis / basis.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    torch.random.set_rng_state(generator_state)
    return basis


def _trace_supports(
    trace: dict[str, np.ndarray],
    *,
    position_key: str,
    valid_key: str,
    force_key: str,
    role_indices: tuple[int, int],
    radius: float,
    chord,
) -> np.ndarray:
    position = torch.as_tensor(trace[position_key][:, :, role_indices, :])
    valid = torch.as_tensor(trace[valid_key][:, :, role_indices])
    force = torch.as_tensor(trace[force_key][:, :, role_indices, :])
    position = torch.where(valid.unsqueeze(-1), position, torch.zeros_like(position))
    force = torch.where(valid.unsqueeze(-1), force, torch.zeros_like(force))
    obj = torch.as_tensor(trace["object_root_state_w"])
    steps, envs = position.shape[:2]
    basis = _official_basis(position.device)
    theta = torch.linspace(
        0.0, 2.0 * torch.pi, steps=NUM_FRICTION_EDGES + 1, device=position.device
    )[:-1]
    cos_t = torch.cos(theta).view(1, -1, 1)
    sin_t = torch.sin(theta).view(1, -1, 1)
    output: list[torch.Tensor] = []
    for side in range(2):
        points_com, normals_com = chord.wrench_preprocess_jit(
            contact_positions_w=position[:, :, side : side + 1]
            .reshape(steps * envs, 1, 1, 3),
            contact_forces_first_hist_w=force[:, :, side : side + 1]
            .reshape(steps * envs, 1, 1, 3),
            object_com_position_w=obj[:, :, :3].reshape(steps * envs, 1, 1, 3),
            object_com_orientation_w=obj[:, :, 3:7].reshape(
                steps * envs, 1, 1, 4
            ),
            num_envs=steps * envs,
            num_bodies=1,
            num_robot_contacts=1,
        )
        support = chord.wrench_support_one_body_jit(
            contact_points=points_com[:, 0],
            contact_normals=normals_com[:, 0],
            cos_t=cos_t,
            sin_t=sin_t,
            basis=basis,
            rc=radius,
            friction_coefficients=FRICTION_COEFFICIENT,
        )
        output.append(support.reshape(steps, envs, NUM_BASIS))
    return torch.stack(output, dim=2).cpu().numpy()


def _reference_by_frame(
    frames: np.ndarray, supports: np.ndarray, *, profile: int
) -> dict[int, np.ndarray]:
    """Return one coherent expert trajectory rather than an asynchronous median."""

    if frames.ndim != 2 or supports.ndim != 4:
        raise RuntimeError("native reference trace geometry drift")
    if not (0 <= profile < frames.shape[1]):
        raise ValueError(f"reference profile {profile} is out of range")
    selected_frames = frames[:, profile]
    if np.unique(selected_frames).size != selected_frames.size:
        raise RuntimeError("native reference motion frame is not one-to-one")
    return {
        int(frame): supports[step, profile].astype(np.float32)
        for step, frame in enumerate(selected_frames)
    }


def _score_arm(
    arm_dir: Path,
    reference: dict[int, np.ndarray],
    radius: float,
    chord,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    result = json.loads((arm_dir / "RESULT.json").read_text(encoding="utf-8"))
    with np.load(arm_dir / "trace.npz", allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    supports = _trace_supports(
        trace,
        position_key="foot_contact_position_w",
        valid_key="foot_contact_position_valid",
        force_key="foot_contact_force_w",
        role_indices=(0, 1),
        radius=radius,
        chord=chord,
    )
    frames = trace["motion_frame"]
    missing = sorted(set(int(v) for v in np.unique(frames)) - set(reference))
    if missing:
        raise RuntimeError(f"native Kick reference misses motion frames: {missing[:8]}")
    commanded = np.stack(
        [reference[int(frame)] for frame in frames.reshape(-1)], axis=0
    ).reshape(supports.shape)

    current = torch.as_tensor(supports.reshape(-1, 2, NUM_BASIS))
    command = torch.as_tensor(commanded.reshape(-1, 2, NUM_BASIS))
    cmd_active = command > SUPPORT_THRESHOLD
    cur_active = current > SUPPORT_THRESHOLD
    cmd_body = cmd_active.any(dim=-1, keepdim=True)
    cur_body = cur_active.any(dim=-1, keepdim=True)
    cws = chord.contact_wrench_support_reward_jit(
        right_cmd_active=cmd_active[:, 1].unsqueeze(1),
        right_cur_active=cur_active[:, 1].unsqueeze(1),
        left_cmd_active=cmd_active[:, 0].unsqueeze(1),
        left_cur_active=cur_active[:, 0].unsqueeze(1),
        right_cmd_active_per_body=cmd_body[:, 1],
        left_cmd_active_per_body=cmd_body[:, 0],
        right_cmd_supports=command[:, 1].unsqueeze(1),
        right_cur_supports=current[:, 1].unsqueeze(1),
        left_cmd_supports=command[:, 0].unsqueeze(1),
        left_cur_supports=current[:, 0].unsqueeze(1),
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
        right_cur_supports=current[:, 1].unsqueeze(1),
        left_cur_supports=current[:, 0].unsqueeze(1),
        num_bodies=1,
    )
    steps, envs = frames.shape
    cws_np = cws.reshape(steps, envs).cpu().numpy()
    missed_np = missed.reshape(steps, envs).cpu().numpy()
    unintended_np = unintended.reshape(steps, envs).cpu().numpy()
    reference_active = cmd_body.any(dim=1).reshape(steps, envs).cpu().numpy()

    profiles: list[dict[str, object]] = []
    for profile, physical in enumerate(result["profiles"]):
        active = reference_active[:, profile]
        profiles.append(
            {
                "profile": profile,
                "mean_cws_on_reference_contact": float(cws_np[active, profile].mean()),
                "mean_missed_contact_on_reference_contact": float(
                    missed_np[active, profile].mean()
                ),
                "mean_unintended_contact_all_frames": float(
                    unintended_np[:, profile].mean()
                ),
                "safe_kick_success": bool(physical["safe_kick_success"]),
                "physical_robot_fall": bool(physical["physical_robot_fall"]),
                "planar_object_net_displacement_m": float(
                    physical["planar_object_net_displacement_m"]
                ),
            }
        )
    summary = {
        "mean_cws_on_reference_contact": float(
            cws_np[reference_active].mean()
        ),
        "mean_missed_contact_on_reference_contact": float(
            missed_np[reference_active].mean()
        ),
        "mean_unintended_contact_all_frames": float(unintended_np.mean()),
        "reference_contact_frame_fraction": float(reference_active.mean()),
        "safe_kick_success_count": int(result["aggregate"]["safe_kick_success_count"]),
        "physical_fall_count": int(result["aggregate"]["physical_fall_count"]),
        "profiles": profiles,
    }
    arrays = {
        "motion_frame": frames,
        "cws": cws_np,
        "missed_contact": missed_np,
        "unintended_contact": unintended_np,
        "reference_active": reference_active,
        "current_support": supports,
        "command_support": commanded,
    }
    return summary, arrays


def main() -> None:
    args = _arguments()
    args.headless = True
    app = AppLauncher(args).app
    try:
        _main(args)
    finally:
        app.close()


def _main(args: argparse.Namespace) -> None:
    collection = args.collection_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    chord, source = _load_official_chord_utils(
        args.official_chord_root.expanduser().resolve()
    )
    radius = _object_mesh_radius(args.object_usd.expanduser().resolve())

    native_dir = collection / "native_kick21"
    native_result = json.loads(
        (native_dir / "RESULT.json").read_text(encoding="utf-8")
    )
    with np.load(native_dir / "TRACE.npz", allow_pickle=False) as archive:
        native = {name: archive[name] for name in archive.files}
    role_names = native["contact_role_names"].tolist()
    foot_roles = (role_names.index("left_foot"), role_names.index("right_foot"))
    native_supports = _trace_supports(
        native,
        position_key="contact_position_w",
        valid_key="contact_position_valid",
        force_key="contact_force_w",
        role_indices=foot_roles,
        radius=radius,
        chord=chord,
    )
    # Profile 0 is fixed independently of its contact score or object outcome.
    # It is a successful released-expert rollout and supplies one physically
    # coherent contact sequence, as a single demonstration would.
    reference_profile = 0
    reference = _reference_by_frame(
        native["motion_frame"], native_supports, profile=reference_profile
    )
    pre, pre_arrays = _score_arm(
        collection / "pre_update_kick", reference, radius, chord
    )
    learned, learned_arrays = _score_arm(
        collection / "learned_kick", reference, radius, chord
    )
    paired_cws = np.asarray(
        [
            learned["profiles"][i]["mean_cws_on_reference_contact"]
            - pre["profiles"][i]["mean_cws_on_reference_contact"]
            for i in range(len(pre["profiles"]))
        ]
    )
    native_profile = native_result["profiles"][reference_profile]
    scalar_metrics = np.asarray(
        [
            pre["mean_cws_on_reference_contact"],
            learned["mean_cws_on_reference_contact"],
            pre["mean_missed_contact_on_reference_contact"],
            learned["mean_missed_contact_on_reference_contact"],
            pre["mean_unintended_contact_all_frames"],
            learned["mean_unintended_contact_all_frames"],
            *paired_cws.tolist(),
        ],
        dtype=np.float64,
    )
    checks = {
        "native_reference_profile_is_safe_kick": bool(
            native_profile["kick_success"]
            and not native_profile["physical_robot_fall"]
        ),
        "native_reference_has_contact_in_transition_clock": bool(
            pre["reference_contact_frame_fraction"] > 0.0
            and learned["reference_contact_frame_fraction"] > 0.0
        ),
        "official_metrics_are_finite": bool(np.isfinite(scalar_metrics).all()),
        "matched_physical_profile_counts": bool(
            len(pre["profiles"]) == len(learned["profiles"]) == 20
        ),
    }
    payload = {
        "protocol": "sugar_official_chord_representation_diagnostic_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "official_chord": {
            "repository": "https://github.com/nvidia-isaac/video_to_data",
            "commit": OFFICIAL_CHORD_COMMIT,
            "loaded_source": str(source),
            "official_functions": [
                "wrench_preprocess_jit",
                "wrench_support_one_body_jit",
                "contact_wrench_support_reward_jit",
                "missed_contact_penalty_jit",
                "unintended_contact_penalty_jit",
            ],
            "num_basis": NUM_BASIS,
            "num_friction_cone_edges": NUM_FRICTION_EDGES,
            "friction_coefficient": FRICTION_COEFFICIENT,
        },
        "object_mesh_radius_m": radius,
        "reference": {
            "kind": "fixed profile-0 live PhysX wrench support of released Kick21 expert by exact motion frame",
            "profile": reference_profile,
            "selection_rule": "fixed before scoring; no outcome/contact-based profile selection",
            "native_kick_success_count": int(
                native_result["aggregate"]["kick_success_count"]
            ),
            "native_physical_fall_count": int(
                native_result["aggregate"]["physical_fall_count"]
            ),
            "raw_sugar_demo_has_contact_points_or_normals": False,
            "therefore_not_human_demo_chord_reward": True,
        },
        "pre_update": pre,
        "learned": learned,
        "paired": {
            "mean_learned_minus_pre_cws": float(paired_cws.mean()),
            "profiles_with_higher_learned_cws": int(np.count_nonzero(paired_cws > 0.0)),
            "profiles_with_lower_learned_cws": int(np.count_nonzero(paired_cws < 0.0)),
            "safe_kick_success_delta": int(
                learned["safe_kick_success_count"] - pre["safe_kick_success_count"]
            ),
            "physical_fall_delta": int(
                learned["physical_fall_count"] - pre["physical_fall_count"]
            ),
        },
        "claim_boundary": (
            "The official CHORD contact-wrench representation is executable on live SUGAR PhysX "
            "contacts. The target is a released Kick21 robot-expert reference because the SUGAR "
            "demo archive lacks per-contact positions/normals. This diagnoses contact geometry; "
            "it is not a faithful human-demo CHORD training result and proves no policy benefit."
        ),
    }
    (output / "RESULT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "METRICS.npz",
        pre_motion_frame=pre_arrays["motion_frame"],
        pre_cws=pre_arrays["cws"],
        pre_missed_contact=pre_arrays["missed_contact"],
        pre_unintended_contact=pre_arrays["unintended_contact"],
        pre_reference_active=pre_arrays["reference_active"],
        learned_motion_frame=learned_arrays["motion_frame"],
        learned_cws=learned_arrays["cws"],
        learned_missed_contact=learned_arrays["missed_contact"],
        learned_unintended_contact=learned_arrays["unintended_contact"],
        learned_reference_active=learned_arrays["reference_active"],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise RuntimeError("official CHORD representation diagnostic failed")


if __name__ == "__main__":
    main()
