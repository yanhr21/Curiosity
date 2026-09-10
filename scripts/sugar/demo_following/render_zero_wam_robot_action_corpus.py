#!/usr/bin/env python3
"""Render clean robot RGB from exact action-corpus physical states.

The renderer never reruns a policy or physics.  It places the recorded pre-step
robot/object state into the official SUGAR scene, emits one RGB frame every five
50 Hz control steps, and appends the exact post-state after the final action.
Thus 141 frames delimit 140 five-action intervals and cover all 700 executed
actions without temporal nearest-neighbor matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback


ROOT = Path(__file__).resolve().parents[3]
SUGAR = ROOT / "SUGAR"
sys.path.insert(0, str(SUGAR / "scripts/sugar_rl"))
os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(SUGAR / "descriptions/terrain/sugar_ground_plane.usda"),
)
os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("DISPLAY", "")
job_id = os.environ.get("SLURM_JOB_ID", "local")
os.environ.setdefault("ISAACLAB_TMP_ROOT", f"/tmp/Curiosity_zero_wam_rgb_{job_id}")
os.environ.setdefault(
    "SUGAR_UNITREE_TMP_ROOT", f"/tmp/Curiosity_zero_wam_rgb_unitree_{job_id}"
)
os.chdir(SUGAR)

from isaaclab.app import AppLauncher


if socket.gethostname().startswith(("mgmtserver", "login")):
    raise SystemExit("Zero-WAM robot RGB rendering is forbidden on a login node")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Zero-WAM robot RGB rendering requires retained Slurm")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--frame-stride", type=int, default=5)
parser.add_argument("--camera-width", type=int, default=320)
parser.add_argument("--camera-height", type=int, default=320)
parser.add_argument(
    "--source-motion-ids",
    type=int,
    nargs="+",
    default=None,
    help="Optional exact subset of environments to render from a candidate trace.",
)
parser.add_argument(
    "--bpp-candidate-variant-id",
    type=int,
    choices=range(10),
    default=None,
)
parser.add_argument(
    "--bpp-camera-contract",
    type=Path,
    default=None,
    help=(
        "Passed, immutable dual-camera visibility contract. Required for BPP "
        "variants and forbidden for the original Zero-WAM renderer."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.frame_stride != 5:
    parser.error("the frozen 50 Hz -> 10 Hz contract requires --frame-stride 5")
if (args.bpp_candidate_variant_id is None) != (args.bpp_camera_contract is None):
    parser.error("BPP variant ID and --bpp-camera-contract must be provided together")
args.enable_cameras = True
simulation_app = AppLauncher(args).app
print("ZERO_WAM_RGB_STAGE app_ready", flush=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from sugar_rl.tasks.locomanip.robots.g129dof.inference.carry_box_inference_env_cfg import (  # noqa: E402
    RobotPlayEnvCfg as CarryPlayEnvCfg,
)
from sugar_rl.tasks.locomanip.robots.g129dof.inference.kick_box_inference_env_cfg import (  # noqa: E402
    RobotPlayEnvCfg as KickPlayEnvCfg,
)


CONTROL_HZ = 50
TARGET_VIDEO_HZ = 10
EXPECTED_ACTIONS = 700
FRAME_STRIDE = CONTROL_HZ // TARGET_VIDEO_HZ
EXPECTED_FRAMES = EXPECTED_ACTIONS // FRAME_STRIDE + 1
RTX_RENDER_SIZE = 640
ISOLATED_ENV_SPACING_M = 30.0
CAMERA_FAR_CLIP_M = 20.0
TASK_CFG = {"CarryBox": CarryPlayEnvCfg, "KickBox": KickPlayEnvCfg}


def split_for_motion(motion_id: int) -> str:
    remainder = int(motion_id) % 10
    if remainder == 8:
        return "validation"
    if remainder == 9:
        return "test"
    return "train"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_cfg(
    *,
    prim_path: str,
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
    convention: str,
    focal_length_mm: float = 24.0,
    horizontal_aperture_mm: float = 20.955,
) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=position,
            rot=quaternion_wxyz,
            convention=convention,
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length_mm,
            focus_distance=4.0,
            horizontal_aperture=horizontal_aperture_mm,
            clipping_range=(0.05, CAMERA_FAR_CLIP_M),
        ),
        width=RTX_RENDER_SIZE,
        height=RTX_RENDER_SIZE,
    )


def default_agentview_camera_cfg() -> TiledCameraCfg:
    return camera_cfg(
        prim_path="{ENV_REGEX_NS}/ZeroWamWorldCamera",
        position=(3.6, 3.6, 2.4),
        quaternion_wxyz=(
            0.3043649418,
            0.2319667899,
            0.5600173703,
            0.7348019703,
        ),
        convention="opengl",
    )


def load_bpp_camera_contract(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("protocol") != "sugar_bpp_dual_camera_render_contract_v1"
        or value.get("passed_visibility_audit") is not True
        or value.get("camera_keys") != ["agentview_rgb", "eye_in_hand_rgb"]
        or value.get("saved_resolution") != [320, 320]
        or value.get("rtx_resolution") != [640, 640]
        or value.get("saved_fps") != 10
        or float(value.get("camera_far_clip_m", -1.0)) != CAMERA_FAR_CLIP_M
        or float(value.get("neighbor_isolation_lower_bound_m", -1.0))
        <= CAMERA_FAR_CLIP_M
    ):
        raise RuntimeError("BPP dual-camera visibility contract is absent or failed")
    for key in ("agentview_rgb", "eye_in_hand_rgb"):
        spec = value.get("cameras", {}).get(key, {})
        if (
            not isinstance(spec.get("prim_path"), str)
            or len(spec.get("position", ())) != 3
            or len(spec.get("quaternion_wxyz", ())) != 4
            or spec.get("convention") not in {"opengl", "ros", "world"}
        ):
            raise RuntimeError(f"invalid BPP camera specification: {key}")
    eye_path = str(value["cameras"]["eye_in_hand_rgb"]["prim_path"])
    if "/Robot/d435_link/" not in eye_path:
        raise RuntimeError("eye_in_hand_rgb must be rigidly parented to G1 d435_link")
    return value


def camera_cfg_from_contract(spec: dict) -> TiledCameraCfg:
    return camera_cfg(
        prim_path=str(spec["prim_path"]),
        position=tuple(map(float, spec["position"])),
        quaternion_wxyz=tuple(map(float, spec["quaternion_wxyz"])),
        convention=str(spec["convention"]),
        focal_length_mm=float(spec.get("focal_length_mm", 24.0)),
        horizontal_aperture_mm=float(
            spec.get("horizontal_aperture_mm", 20.955)
        ),
    )


def disable_randomization(cfg) -> None:
    for name in (
        "robot_physics_material", "obj_physics_material", "obj_mass",
        "add_joint_default_pos", "base_com", "push_robot", "push_object",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def load_trace(trace_path: Path) -> dict[str, np.ndarray]:
    with np.load(trace_path, allow_pickle=False) as archive:
        required = {
            "robot_root_state_before_w",
            "robot_joint_pos_before",
            "robot_joint_vel_before",
            "object_root_state_before_w",
            "robot_root_state_w",
            "robot_joint_pos",
            "robot_joint_vel",
            "object_root_state_w",
            "environment_origin_w",
            "source_motion_id_by_local_motion",
            "transition_time_s",
            "executed_action",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"action trace is missing exact-state arrays: {missing}")
        data = {name: np.asarray(archive[name]) for name in required}
    steps, envs, action_dim = data["executed_action"].shape
    if (steps, action_dim) != (EXPECTED_ACTIONS, 29):
        raise RuntimeError(f"unexpected action tensor {data['executed_action'].shape}")
    expected_shapes = {
        "robot_root_state_before_w": (steps, envs, 13),
        "robot_joint_pos_before": (steps, envs, 29),
        "robot_joint_vel_before": (steps, envs, 29),
        "object_root_state_before_w": (steps, envs, 13),
        "robot_root_state_w": (steps, envs, 13),
        "robot_joint_pos": (steps, envs, 29),
        "robot_joint_vel": (steps, envs, 29),
        "object_root_state_w": (steps, envs, 13),
        "environment_origin_w": (envs, 3),
        "source_motion_id_by_local_motion": (envs,),
        "transition_time_s": (steps,),
    }
    for name, shape in expected_shapes.items():
        if data[name].shape != shape:
            raise RuntimeError(f"{name} shape drift: {data[name].shape} != {shape}")
        if data[name].dtype.kind in "fc" and not np.isfinite(data[name]).all():
            raise RuntimeError(f"non-finite state array: {name}")
    data["trace_environment_index"] = np.arange(envs, dtype=np.int64)
    return data


def state_at(
    data: dict[str, np.ndarray], frame_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if frame_index < EXPECTED_FRAMES - 1:
        step = frame_index * FRAME_STRIDE
        return (
            data["robot_root_state_before_w"][step],
            data["robot_joint_pos_before"][step],
            data["robot_joint_vel_before"][step],
            data["object_root_state_before_w"][step],
        )
    return (
        data["robot_root_state_w"][-1],
        data["robot_joint_pos"][-1],
        data["robot_joint_vel"][-1],
        data["object_root_state_w"][-1],
    )


def select_source_environments(
    data: dict[str, np.ndarray], requested_source_ids: list[int] | None
) -> dict[str, np.ndarray]:
    if requested_source_ids is None:
        return data
    source_ids = data["source_motion_id_by_local_motion"].astype(np.int64)
    if len(set(requested_source_ids)) != len(requested_source_ids):
        raise ValueError("--source-motion-ids contains duplicates")
    requested = set(map(int, requested_source_ids))
    available = set(map(int, source_ids.tolist()))
    if not requested.issubset(available):
        raise ValueError(
            f"requested source IDs are absent: {sorted(requested - available)}"
        )
    indices = np.asarray(
        [index for index, source_id in enumerate(source_ids.tolist()) if source_id in requested],
        dtype=np.int64,
    )
    if len(indices) != len(requested):
        raise RuntimeError("source environment selection geometry drift")
    original_envs = len(source_ids)
    selected: dict[str, np.ndarray] = {}
    for name, value in data.items():
        if value.ndim >= 2 and value.shape[1] == original_envs:
            selected[name] = value[:, indices]
        elif value.ndim >= 1 and value.shape[0] == original_envs:
            selected[name] = value[indices]
        else:
            selected[name] = value
    return selected


def main() -> None:
    trace_path = args.trace.expanduser().resolve()
    result_path = trace_path.parent / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("passed") is not True:
        raise RuntimeError("source official collection gate did not pass")
    if args.bpp_candidate_variant_id is not None and result.get(
        "bpp_variant_id"
    ) != args.bpp_candidate_variant_id:
        raise RuntimeError("BPP candidate variant identity mismatch")
    if (args.source_motion_ids is None) != (
        args.bpp_candidate_variant_id is None
    ):
        raise RuntimeError(
            "BPP source subset and candidate variant ID must be specified together"
        )
    camera_contract_path = (
        args.bpp_camera_contract.expanduser().resolve()
        if args.bpp_camera_contract is not None
        else None
    )
    camera_contract = (
        load_bpp_camera_contract(camera_contract_path)
        if camera_contract_path is not None
        else None
    )
    camera_keys = (
        ["agentview_rgb", "eye_in_hand_rgb"]
        if camera_contract is not None
        else ["agentview_rgb"]
    )
    task = str(result["task_family"])
    if task not in TASK_CFG:
        raise RuntimeError(f"unsupported task: {task}")
    output = args.output_root.expanduser().resolve()
    if not output.is_relative_to((ROOT / "experiments").resolve()):
        raise ValueError("robot RGB corpus must remain under ignored experiments/")
    data = select_source_environments(
        load_trace(trace_path), args.source_motion_ids
    )
    source_ids = data["source_motion_id_by_local_motion"].astype(np.int64)
    envs = len(source_ids)
    frame_dirs = [
        output / split_for_motion(int(source_id)) / task / str(int(source_id))
        for source_id in source_ids
    ]
    for directory in frame_dirs:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if camera_contract is not None:
            for camera_key in camera_keys:
                (directory / camera_key).mkdir(exist_ok=False)

    cfg = TASK_CFG[task]()
    cfg.scene.num_envs = envs
    cfg.scene.env_spacing = ISOLATED_ENV_SPACING_M
    cfg.seed = int(result["seed"])
    cfg.sim.device = args.device
    if camera_contract is None:
        cfg.scene.world_camera = default_agentview_camera_cfg()
    else:
        cfg.scene.world_camera = camera_cfg_from_contract(
            camera_contract["cameras"]["agentview_rgb"]
        )
        cfg.scene.eye_in_hand_camera = camera_cfg_from_contract(
            camera_contract["cameras"]["eye_in_hand_rgb"]
        )
    cfg.sim.render_interval = cfg.decimation
    cfg.observations.policy.enable_corruption = False
    disable_randomization(cfg)
    print(
        f"ZERO_WAM_RGB_STAGE sources_ready task={task} envs={envs} "
        f"frames={EXPECTED_FRAMES}",
        flush=True,
    )

    sim = SimulationContext(cfg.sim)
    scene = InteractiveScene(cfg.scene)
    sim.reset()
    scene.reset()
    scene_origins = scene.env_origins
    trace_origins = torch.as_tensor(
        data["environment_origin_w"], device=scene_origins.device
    )
    initial_focus_xy = 0.5 * (
        torch.as_tensor(
            data["robot_root_state_before_w"][0, :, :2],
            device=scene_origins.device,
        )
        + torch.as_tensor(
            data["object_root_state_before_w"][0, :, :2],
            device=scene_origins.device,
        )
    ) - trace_origins[:, :2]
    env_ids = torch.arange(envs, device=scene_origins.device, dtype=torch.long)
    frame_timestamps = np.arange(EXPECTED_FRAMES, dtype=np.float64) / TARGET_VIDEO_HZ
    frame_action_ranges = []
    for frame_index in range(EXPECTED_FRAMES):
        robot_root_np, joint_pos_np, joint_vel_np, object_root_np = state_at(
            data, frame_index
        )
        robot_root = torch.as_tensor(
            robot_root_np, device=scene_origins.device
        ).clone()
        object_root = torch.as_tensor(
            object_root_np, device=scene_origins.device
        ).clone()
        translation = scene_origins - trace_origins
        robot_root[:, :3] += translation
        object_root[:, :3] += translation
        robot_root[:, :2] -= initial_focus_xy
        object_root[:, :2] -= initial_focus_xy
        joint_pos = torch.as_tensor(joint_pos_np, device=scene_origins.device)
        joint_vel = torch.as_tensor(joint_vel_np, device=scene_origins.device)
        scene["robot"].write_root_state_to_sim(robot_root, env_ids=env_ids)
        scene["robot"].write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids
        )
        scene["obj"].write_root_state_to_sim(object_root, env_ids=env_ids)
        scene.write_data_to_sim()
        sim.forward()
        sim.render()
        scene.update(dt=0.0)
        scene_camera_names = {
            "agentview_rgb": "world_camera",
            "eye_in_hand_rgb": "eye_in_hand_camera",
        }
        for camera_key in camera_keys:
            camera = scene[scene_camera_names[camera_key]]
            camera.update(0.0, force_recompute=True)
            rendered = camera.data.output["rgb"][:, ..., :3].detach().cpu().numpy()
            if rendered.shape != (envs, RTX_RENDER_SIZE, RTX_RENDER_SIZE, 3):
                raise RuntimeError(
                    f"unexpected {camera_key} tensor {rendered.shape}"
                )
            rgb = np.stack(
                [
                    cv2.resize(
                        image,
                        (args.camera_width, args.camera_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    for image in rendered
                ]
            )
            for env_index, image in enumerate(rgb):
                if int(image.max()) == int(image.min()):
                    raise RuntimeError(
                        f"constant {camera_key} RGB for "
                        f"{task}:{int(source_ids[env_index])}"
                    )
                camera_directory = (
                    frame_dirs[env_index] / camera_key
                    if camera_contract is not None
                    else frame_dirs[env_index]
                )
                path = camera_directory / f"{frame_index:03d}.png"
                if not cv2.imwrite(
                    str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                ):
                    raise RuntimeError(f"failed to write {path}")
        action_start = min(frame_index * FRAME_STRIDE, EXPECTED_ACTIONS)
        action_stop = min((frame_index + 1) * FRAME_STRIDE, EXPECTED_ACTIONS)
        frame_action_ranges.append([action_start, action_stop])
        if frame_index % 20 == 0 or frame_index == EXPECTED_FRAMES - 1:
            print(
                f"ZERO_WAM_RGB_PROGRESS task={task} "
                f"frame={frame_index + 1}/{EXPECTED_FRAMES}",
                flush=True,
            )

    per_motion = []
    for env_index, source_id_value in enumerate(source_ids.tolist()):
        directory = frame_dirs[env_index]
        camera_frame_directories = {
            camera_key: (
                directory / camera_key if camera_contract is not None else directory
            )
            for camera_key in camera_keys
        }
        for camera_key, camera_directory in camera_frame_directories.items():
            paths = sorted(camera_directory.glob("*.png"))
            if len(paths) != EXPECTED_FRAMES:
                raise RuntimeError(
                    f"incomplete {camera_key} sequence: {camera_directory}"
                )
        record = {
            "task": task,
            "source_motion_id": int(source_id_value),
            "split": split_for_motion(int(source_id_value)),
            "frame_directory": str(directory.relative_to(ROOT)),
            "camera_keys": camera_keys,
            "camera_frame_directories": {
                key: str(value.relative_to(ROOT))
                for key, value in camera_frame_directories.items()
            },
            "frame_count": EXPECTED_FRAMES,
            "frame_timestamps_s": frame_timestamps.tolist(),
            "action_ranges_50hz": frame_action_ranges,
            "trace_environment_index": int(
                data["trace_environment_index"][env_index]
            ),
        }
        (directory / "FRAME_MAP.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        per_motion.append(record)

    render_result = {
        "protocol": "sugar_zero_wam_exact_state_robot_rgb_v1",
        "passed": True,
        "task": task,
        "source_trace": str(trace_path.relative_to(ROOT)),
        "source_collector_protocol": result["protocol"],
        "source_motion_ids": source_ids.tolist(),
        "bpp_candidate_variant_id": args.bpp_candidate_variant_id,
        "source_subset_requested": (
            sorted(map(int, args.source_motion_ids))
            if args.source_motion_ids is not None else None
        ),
        "trajectory_count": envs,
        "frame_count_per_trajectory": EXPECTED_FRAMES,
        "camera_keys": camera_keys,
        "total_rgb_frames": EXPECTED_FRAMES * envs * len(camera_keys),
        "resolution": [args.camera_height, args.camera_width],
        "rtx_render_resolution": [RTX_RENDER_SIZE, RTX_RENDER_SIZE],
        "environment_spacing_m": ISOLATED_ENV_SPACING_M,
        "camera_far_clip_m": CAMERA_FAR_CLIP_M,
        "neighbor_geometry_gate": {
            "conservative_camera_to_neighbor_center_lower_bound_m": (
                float(camera_contract["neighbor_isolation_lower_bound_m"])
                if camera_contract is not None
                else ISOLATED_ENV_SPACING_M - float(np.hypot(3.6, 3.6))
            ),
            "neighbor_center_is_beyond_far_clip": (
                float(camera_contract["neighbor_isolation_lower_bound_m"])
                > CAMERA_FAR_CLIP_M
                if camera_contract is not None
                else ISOLATED_ENV_SPACING_M - float(np.hypot(3.6, 3.6))
                > CAMERA_FAR_CLIP_M
            ),
        },
        "camera_contract": (
            {
                "path": str(camera_contract_path.relative_to(ROOT)),
                "sha256": file_sha256(camera_contract_path),
            }
            if camera_contract_path is not None
            else None
        ),
        "control_hz": CONTROL_HZ,
        "target_video_hz": TARGET_VIDEO_HZ,
        "actions_per_video_interval": FRAME_STRIDE,
        "fixed_trajectory_centering": (
            "subtract the per-trajectory first-frame robot/object XY midpoint "
            "once; no future or per-frame camera tracking"
        ),
        "motion_records": per_motion,
        "rendering_semantics": (
            "Exact recorded official Generator+Tracker pre/post physical states; "
            "no policy inference, physics replay, labels, text, plots or borders."
        ),
        "claim_boundary": (
            "This is synchronized robot-domain observation data, not generated "
            "future quality or demo-following evidence."
        ),
    }
    render_result_path = output / f"RENDER_RESULT_{trace_path.parent.name}.json"
    render_result_path.write_text(
        json.dumps(render_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(render_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        # On the H200 cluster, Kit can block indefinitely during teardown after
        # every frame and the machine-readable result have been durably written.
        # There is no live simulator state to preserve: this process only replays
        # immutable tensors into RTX.  Flush the evidence and terminate this
        # one-shot renderer without entering the non-scientific teardown hang.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
