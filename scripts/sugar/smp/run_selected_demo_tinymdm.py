#!/usr/bin/env python3
"""Prepare, train, and cross-score official single-clip TinyMDM priors.

This file is adapter and experiment orchestration only. The representation uses
MimicKit's official ``compute_disc_obs`` implementation and the prior, EMA,
diffusion scheduler, training objective, and ESM/SDS energy all come directly
from the pinned official ``TinyMDMModel``. No local encoder or surrogate world
model is introduced.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sugar_g1_box_schema import (
    CHARACTER_FEATURE_DIM,
    FEATURE_DIM,
    G1_JOINT_AXES,
    G1_JOINT_NAMES,
    KEY_BODY_INDICES,
    OBJECT_FEATURE_DIM,
    ROOT_BODY_INDEX,
    SOURCE_BODY_INDICES,
    SOURCE_BODY_NAMES,
    TRACKED_BODY_NAMES,
    WINDOW_SIZE,
)


ROOT = Path(__file__).resolve().parents[3]
MIMICKIT = ROOT / "MimicKit"
MIMICKIT_PYTHON = MIMICKIT / "mimickit"
OUTPUT_ROOT = ROOT / "experiments/demo_following/selected_demo_smp_v1"
PINNED_MIMICKIT_COMMIT = "2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69"
DIFFUSION_STEPS = (22, 15, 8)
SCORE_SEEDS = tuple(range(1701, 1709))

CLIPS = {
    "carry45": ROOT / "SUGAR/data/CarryBox/data_045",
    "kick21": ROOT / "SUGAR/data/KickBox/data_021",
    "carry96": ROOT / "SUGAR/data/CarryBox/data_096",
    "kick22": ROOT / "SUGAR/data/KickBox/data_022",
}

SINGLE_CLIP_CONFIG = {
    "model_name": "tiny_mdm",
    "arch_name": "DiT",
    "T": 50,
    "loss_type": "l1",
    "estimate_mode": "epsilon",
    "noise_schedule_mode": "squaredcos_cap_v2",
    "num_layers": 2,
    "num_attention_heads": 4,
    "model_ema": True,
    "model_ema_decay": 0.995,
    "model_ema_steps": 10,
    "model_ema_update_after": 5_000,
    "normalizer_std_clip": 0.2,
    "batch_size": 512,
    "lr": 0.0001,
    "num_iterations": 50_000,
    "num_samples_stat": 10_000,
    "output_iter": 2_000,
    "grad_clip_norm": 1.0,
    "control_freq": 50,
    "input_channel": FEATURE_DIM,
    "num_disc_obs_steps": WINDOW_SIZE,
    "input_dim": FEATURE_DIM * WINDOW_SIZE,
}

ENV_CONFIG = {
    "global_obs": False,
    "root_height_obs": True,
    "pose_termination": False,
    "enable_phase_obs": False,
    "enable_tar_obs": False,
    "disc_dof_vel_obs": False,
    "num_disc_obs_steps": WINDOW_SIZE,
    "key_bodies": [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ],
    "sugar_object_features": {
        "enabled": True,
        "coordinate_frame": "final-frame torso heading-local",
        "fields": [
            "box_position",
            "box_rotation_tangent_normal_6d",
            "box_linear_velocity",
            "box_angular_velocity",
        ],
    },
}


def require_compute_gpu(device: torch.device) -> None:
    if socket.gethostname().startswith(("mgmtserver", "login")):
        raise SystemExit("Refusing TinyMDM work on a login node")
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("selected-demo TinyMDM requires a retained compute GPU")
    head = subprocess.run(
        ["git", "-C", str(MIMICKIT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != PINNED_MIMICKIT_COMMIT:
        raise RuntimeError(f"MimicKit checkout {head} != pinned {PINNED_MIMICKIT_COMMIT}")


def fix_seed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_official_feature_functions() -> tuple[Any, Any]:
    sys.path.insert(0, str(MIMICKIT_PYTHON))
    from envs.amp_env import compute_disc_obs  # noqa: PLC0415
    from util import torch_util  # noqa: PLC0415

    return compute_disc_obs, torch_util


def load_clip(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    robot_path = path / "robot_50hz.npz"
    object_path = path / "obj_motion_global_50hz.pkl"
    if not robot_path.is_file() or not object_path.is_file():
        raise FileNotFoundError(path)
    with np.load(robot_path, allow_pickle=False) as archive:
        robot = {name: np.asarray(archive[name]) for name in archive.files}
    with object_path.open("rb") as stream:
        raw_object = pickle.load(stream)  # noqa: S301 - trusted repository asset
    obj = {name: np.asarray(value) for name, value in raw_object.items()}
    length = int(robot["joint_pos"].shape[0])
    expected_robot = {
        "joint_pos": (length, len(G1_JOINT_NAMES)),
        "joint_vel": (length, len(G1_JOINT_NAMES)),
        "body_pos_w": (length, len(SOURCE_BODY_NAMES), 3),
        "body_quat_w": (length, len(SOURCE_BODY_NAMES), 4),
        "body_lin_vel_w": (length, len(SOURCE_BODY_NAMES), 3),
        "body_ang_vel_w": (length, len(SOURCE_BODY_NAMES), 3),
    }
    expected_object = {
        "obj_trans": (length, 3),
        "obj_rot": (length, 3, 3),
        "obj_lin_vel": (length, 3),
        "obj_ang_vel": (length, 3),
    }
    for name, shape in expected_robot.items():
        if robot[name].shape != shape or not np.isfinite(robot[name]).all():
            raise ValueError(f"{path}: invalid {name} {robot[name].shape}")
    for name, shape in expected_object.items():
        if obj[name].shape[0] < length or obj[name].shape[1:] != shape[1:]:
            raise ValueError(f"{path}: invalid {name} {obj[name].shape}")
        obj[name] = obj[name][:length]
        if not np.isfinite(obj[name]).all():
            raise ValueError(f"{path}: non-finite {name}")
    for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        robot[name] = robot[name][:, SOURCE_BODY_INDICES]
    if length < WINDOW_SIZE:
        raise ValueError(f"{path}: clip shorter than {WINDOW_SIZE} frames")
    return robot, obj


@torch.no_grad()
def build_feature_windows(
    robot: dict[str, np.ndarray],
    obj: dict[str, np.ndarray],
    device: torch.device,
    chunk_size: int = 256,
) -> np.ndarray:
    compute_disc_obs, torch_util = load_official_feature_functions()
    length = int(robot["joint_pos"].shape[0])
    starts = np.arange(length - WINDOW_SIZE + 1, dtype=np.int64)
    offsets = np.arange(WINDOW_SIZE, dtype=np.int64)
    output = np.empty((len(starts), WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
    joint_axes = torch.as_tensor(G1_JOINT_AXES, dtype=torch.float32, device=device)

    for begin in range(0, len(starts), chunk_size):
        end = min(begin + chunk_size, len(starts))
        frame_ids = starts[begin:end, None] + offsets[None, :]
        batch = len(frame_ids)
        body_pos = torch.as_tensor(robot["body_pos_w"][frame_ids], device=device)
        body_quat_wxyz = robot["body_quat_w"][frame_ids]
        body_quat = torch.as_tensor(
            body_quat_wxyz[..., (1, 2, 3, 0)], device=device
        )
        body_lin_vel = torch.as_tensor(
            robot["body_lin_vel_w"][frame_ids], device=device
        )
        body_ang_vel = torch.as_tensor(
            robot["body_ang_vel_w"][frame_ids], device=device
        )
        dof_pos = torch.as_tensor(robot["joint_pos"][frame_ids], device=device)
        dof_vel = torch.as_tensor(robot["joint_vel"][frame_ids], device=device)

        root_pos = body_pos[:, :, ROOT_BODY_INDEX]
        root_rot = body_quat[:, :, ROOT_BODY_INDEX]
        axes = joint_axes.view(1, 1, len(G1_JOINT_NAMES), 3).expand(
            batch, WINDOW_SIZE, -1, -1
        )
        joint_rot = torch_util.axis_angle_to_quat(axes, dof_pos)
        character = compute_disc_obs(
            ref_root_pos=root_pos[:, -1],
            ref_root_rot=root_rot[:, -1],
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=body_lin_vel[:, :, ROOT_BODY_INDEX],
            root_ang_vel=body_ang_vel[:, :, ROOT_BODY_INDEX],
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=body_pos[:, :, list(KEY_BODY_INDICES)],
            global_obs=False,
            root_height_obs=True,
            dof_vel_obs=False,
        ).reshape(batch, WINDOW_SIZE, -1)
        if character.shape[-1] != CHARACTER_FEATURE_DIM:
            raise RuntimeError(f"official character feature dim {character.shape[-1]}")

        heading_inv = torch_util.calc_heading_quat_inv(root_rot[:, -1])
        heading_steps = heading_inv[:, None].expand(-1, WINDOW_SIZE, -1)
        obj_pos = torch.as_tensor(obj["obj_trans"][frame_ids], device=device)
        obj_rot = torch.as_tensor(obj["obj_rot"][frame_ids], device=device)
        obj_lin_vel = torch.as_tensor(obj["obj_lin_vel"][frame_ids], device=device)
        obj_ang_vel = torch.as_tensor(obj["obj_ang_vel"][frame_ids], device=device)
        object_features = torch.cat(
            (
                torch_util.quat_rotate(heading_steps, obj_pos - root_pos),
                torch_util.quat_rotate(heading_steps, obj_rot[..., :, 0]),
                torch_util.quat_rotate(heading_steps, obj_rot[..., :, 2]),
                torch_util.quat_rotate(heading_steps, obj_lin_vel),
                torch_util.quat_rotate(heading_steps, obj_ang_vel),
            ),
            dim=-1,
        )
        if object_features.shape[-1] != OBJECT_FEATURE_DIM:
            raise RuntimeError(f"object feature dim {object_features.shape[-1]}")
        features = torch.cat((character, object_features), dim=-1)
        if features.shape[1:] != (WINDOW_SIZE, FEATURE_DIM):
            raise RuntimeError(f"feature geometry {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise RuntimeError("non-finite selected-demo features")
        output[begin:end] = features.cpu().numpy().astype(np.float32, copy=False)
    return output


def prepare_dataset(output_root: Path, device: torch.device) -> dict[str, Any]:
    dataset_root = output_root / "dataset"
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["arrays"].values():
            if not Path(record["path"]).is_file():
                raise FileNotFoundError(record["path"])
        return manifest
    if dataset_root.exists():
        raise FileExistsError(f"incomplete dataset directory: {dataset_root}")
    dataset_root.mkdir(parents=True)
    arrays: dict[str, dict[str, Any]] = {}
    for name, source in CLIPS.items():
        robot, obj = load_clip(source)
        features = build_feature_windows(robot, obj, device)
        starts = np.arange(features.shape[0], dtype=np.int64)
        if name in {"carry45", "kick21"}:
            selections = {
                "train": starts % 5 != 0,
                "holdout": starts % 5 == 0,
            }
        else:
            selections = {"semantic": starts % 5 == 0}
        for split, mask in selections.items():
            selected = features[mask]
            path = dataset_root / f"{name}_{split}.npy"
            atomic_npy(path, selected)
            key = f"{name}_{split}"
            arrays[key] = {
                "path": str(path),
                "shape": list(selected.shape),
                "source": str(source),
                "window_start_rule": (
                    "start % 5 != 0" if split == "train" else "start % 5 == 0"
                ),
            }
    expected_shapes = {
        "carry45_train": [520, 10, 216],
        "carry45_holdout": [131, 10, 216],
        "kick21_train": [520, 10, 216],
        "kick21_holdout": [131, 10, 216],
        "carry96_semantic": [105, 10, 216],
        "kick22_semantic": [107, 10, 216],
    }
    actual_shapes = {name: record["shape"] for name, record in arrays.items()}
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"selected-demo dataset shapes {actual_shapes}")
    manifest = {
        "protocol": "sugar_selected_demo_tinymdm_dataset_v1",
        "mimickit_commit": PINNED_MIMICKIT_COMMIT,
        "representation": "official MimicKit compute_disc_obs plus 15-D box feature",
        "feature_geometry": [WINDOW_SIZE, FEATURE_DIM],
        "split_scope": (
            "Interleaved exact-window holdout across the selected clip; adjacent windows overlap, "
            "so this is an identity gate, not independent clip generalization."
        ),
        "arrays": arrays,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def load_array(manifest: dict[str, Any], key: str) -> np.ndarray:
    record = manifest["arrays"][key]
    array = np.load(record["path"], allow_pickle=False)
    if list(array.shape) != record["shape"] or array.dtype != np.float32:
        raise ValueError(f"dataset drift for {key}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite dataset {key}")
    return array


def runtime_config(output_dir: Path) -> dict[str, Any]:
    env_path = output_dir / "env_config.yaml"
    config_path = output_dir / "diffusion_config.yaml"
    env_path.write_text(yaml.safe_dump(ENV_CONFIG, sort_keys=False), encoding="utf-8")
    config = dict(SINGLE_CLIP_CONFIG)
    config["env_config"] = str(env_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


@torch.no_grad()
def validation_loss(model: Any, windows: torch.Tensor, seed: int) -> float:
    training = model.training
    model.eval()
    devices = [windows.device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        normalized = model.normalize(windows).reshape(windows.shape[0], -1)
        value = float(model(normalized).item())
    model.train(training)
    return value


@torch.no_grad()
def save_samples(model: Any, path: Path, device: torch.device) -> None:
    model.eval()
    normalized = model.sample_ema(
        shape=(WINDOW_SIZE * FEATURE_DIM,), batch_size=16, device=device
    )
    samples = model.unnormalize(
        normalized.reshape(16, WINDOW_SIZE, FEATURE_DIM)
    ).cpu().numpy().astype(np.float32)
    atomic_npy(path, samples)


def train_prior(
    output_root: Path,
    manifest: dict[str, Any],
    clip: str,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if clip not in {"carry45", "kick21"}:
        raise ValueError("train clip must be carry45 or kick21")
    output_dir = output_root / "priors" / clip
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("passed") is not True or result.get("completed_iterations") != 50_000:
            raise RuntimeError(f"invalid existing prior: {output_dir}")
        return result
    if output_dir.exists():
        raise FileExistsError(f"incomplete prior directory: {output_dir}")
    output_dir.mkdir(parents=True)
    config = runtime_config(output_dir)
    train_np = load_array(manifest, f"{clip}_train")
    holdout_np = load_array(manifest, f"{clip}_holdout")
    train = torch.as_tensor(train_np, device=device)
    holdout = torch.as_tensor(holdout_np, device=device)
    fix_seed(seed)
    sys.path.insert(0, str(MIMICKIT_PYTHON))
    from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: PLC0415

    model = TinyMDMModel(config, device).to(device)
    stat_generator = torch.Generator(device=device).manual_seed(seed + 1)
    stat_indices = torch.randint(
        train.shape[0],
        (int(config["num_samples_stat"]),),
        generator=stat_generator,
        device=device,
    )
    model.update_normalizer(train.index_select(0, stat_indices))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["lr"]),
    )
    sample_generator = torch.Generator(device=device).manual_seed(seed + 2)
    batch_size = int(config["batch_size"])
    output_iter = int(config["output_iter"])
    gradient_clip = float(config["grad_clip_norm"])
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != 2_836_096:
        raise RuntimeError(f"official TinyMDM parameter count {parameter_count}")
    model.train()
    loss_sum = 0.0
    log_path = output_dir / "train_metrics.jsonl"
    for iteration in range(1, int(config["num_iterations"]) + 1):
        indices = torch.randint(
            train.shape[0],
            (batch_size,),
            generator=sample_generator,
            device=device,
        )
        samples = train.index_select(0, indices)
        normalized = model.normalize(samples).reshape(batch_size, -1)
        loss = model(normalized)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if model.model_ema is not False:
            model.ema_dmodel.update()
        loss_sum += float(loss.item())
        if iteration % output_iter == 0:
            metrics = {
                "iteration": iteration,
                "train_loss_mean": loss_sum / output_iter,
                "holdout_loss": validation_loss(model, holdout, seed + 3),
                "gradient_norm_before_clip": float(gradient_norm),
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps({"clip": clip, **metrics}, sort_keys=True), flush=True)
            loss_sum = 0.0
            model.train()
    model.eval()
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("non-finite TinyMDM endpoint")
    atomic_torch(output_dir / "model.pt", model.state_dict())
    save_samples(model, output_dir / "feature_samples.npy", device)
    result = {
        "protocol": "sugar_selected_demo_official_single_clip_tinymdm_v1",
        "passed": True,
        "clip": clip,
        "seed": seed,
        "mimickit_commit": PINNED_MIMICKIT_COMMIT,
        "official_recipe": "MimicKit tinymdm_single_clip.yaml adapted only to 50 Hz 10x216 input",
        "parameter_count": parameter_count,
        "train_windows": int(train.shape[0]),
        "holdout_windows": int(holdout.shape[0]),
        "completed_iterations": int(config["num_iterations"]),
        "model": str(output_dir / "model.pt"),
        "config": str(output_dir / "diffusion_config.yaml"),
    }
    atomic_json(result_path, result)
    return result


def load_prior(path: Path, device: torch.device) -> Any:
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    if (
        result.get("protocol")
        != "sugar_selected_demo_official_single_clip_tinymdm_v1"
        or result.get("passed") is not True
        or result.get("completed_iterations") != 50_000
    ):
        raise RuntimeError(f"unadmitted selected-demo prior: {path}")
    config = yaml.safe_load((path / "diffusion_config.yaml").read_text(encoding="utf-8"))
    config["env_config"] = str(path / "env_config.yaml")
    sys.path.insert(0, str(MIMICKIT_PYTHON))
    from learning.tinymdm.tinymdm_model import TinyMDMModel  # noqa: PLC0415

    model = TinyMDMModel(config, device).to(device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def raw_energy(model: Any, array: np.ndarray, device: torch.device) -> np.ndarray:
    windows = torch.as_tensor(array, device=device)
    normalized = model.normalize(windows).reshape(windows.shape[0], -1)
    repeats = []
    for seed in SCORE_SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        energy = model.ESM_SDS_loss(
            normalized, t_lst=list(DIFFUSION_STEPS)
        ).mean(dim=-1)
        repeats.append(energy.cpu().numpy())
    output = np.mean(np.stack(repeats, axis=0), axis=0).astype(np.float32)
    if output.shape != (array.shape[0],) or not np.isfinite(output).all():
        raise RuntimeError("invalid ESM/SDS energy")
    return output


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std_across_windows": float(values.std()),
        "median": float(np.median(values)),
    }


def preference(own: np.ndarray, other: np.ndarray) -> dict[str, float]:
    return {
        "pairwise_probability_own_energy_lower": float(
            np.mean(own[:, None] < other[None, :])
        ),
        "mean_other_minus_own": float(other.mean() - own.mean()),
        "median_other_minus_own": float(np.median(other) - np.median(own)),
    }


def score_priors(
    output_root: Path, manifest: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    score_root = output_root / "cross_score"
    result_path = score_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if score_root.exists():
        raise FileExistsError(f"incomplete score directory: {score_root}")
    score_root.mkdir(parents=True)
    arrays = {
        "carry45_holdout": load_array(manifest, "carry45_holdout"),
        "kick21_holdout": load_array(manifest, "kick21_holdout"),
        "carry96_semantic": load_array(manifest, "carry96_semantic"),
        "kick22_semantic": load_array(manifest, "kick22_semantic"),
    }
    energies: dict[str, dict[str, np.ndarray]] = {}
    for prior_name in ("carry45", "kick21"):
        model = load_prior(output_root / "priors" / prior_name, device)
        energies[prior_name] = {
            name: raw_energy(model, array, device) for name, array in arrays.items()
        }
        del model
        torch.cuda.empty_cache()
    exact_preferences = {
        "carry45_prior": preference(
            energies["carry45"]["carry45_holdout"],
            energies["carry45"]["kick21_holdout"],
        ),
        "kick21_prior": preference(
            energies["kick21"]["kick21_holdout"],
            energies["kick21"]["carry45_holdout"],
        ),
    }
    semantic_preferences = {
        "carry45_prior": preference(
            energies["carry45"]["carry96_semantic"],
            energies["carry45"]["kick22_semantic"],
        ),
        "kick21_prior": preference(
            energies["kick21"]["kick22_semantic"],
            energies["kick21"]["carry96_semantic"],
        ),
    }
    exact_gate = all(
        record["mean_other_minus_own"] > 0.0
        and record["pairwise_probability_own_energy_lower"] >= 0.75
        for record in exact_preferences.values()
    )
    semantic_gate = all(
        record["mean_other_minus_own"] > 0.0
        and record["pairwise_probability_own_energy_lower"] >= 0.60
        for record in semantic_preferences.values()
    )
    overall_gate = exact_gate and semantic_gate
    result = {
        "protocol": "sugar_selected_demo_official_tinymdm_cross_score_v1",
        "passed": overall_gate,
        "selected_demo_identity_gate_passed": exact_gate,
        "semantic_extension_gate_passed": semantic_gate,
        "policy_integration_supported": overall_gate,
        "diffusion_steps": list(DIFFUSION_STEPS),
        "noise_repeats": len(SCORE_SEEDS),
        "matrix": {
            prior_name: {
                array_name: summarize(values)
                for array_name, values in by_array.items()
            }
            for prior_name, by_array in energies.items()
        },
        "exact_selected_demo_preferences": exact_preferences,
        "heldout_same_task_preferences": semantic_preferences,
        "claim_scope": (
            "Official TinyMDM ESM/SDS identity and same-task extension gate only. "
            "This is not policy improvement and not an official latent embedding."
        ),
    }
    for prior_name, by_array in energies.items():
        for array_name, values in by_array.items():
            atomic_npy(score_root / f"{prior_name}__{array_name}.npy", values)
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "score"))
    parser.add_argument("--clip", choices=("carry45", "kick21"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=160821)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    if not output_root.is_relative_to(experiments):
        raise ValueError("outputs must stay inside ignored experiments/")
    device = torch.device(args.device)
    require_compute_gpu(device)
    manifest = prepare_dataset(output_root, device)
    if args.command == "prepare":
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "train":
        if args.clip is None:
            raise ValueError("train requires --clip")
        result = train_prior(output_root, manifest, args.clip, device, args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        score_priors(output_root, manifest, device)


if __name__ == "__main__":
    main()
