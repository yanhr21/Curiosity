#!/usr/bin/env python3
"""Model-independent dual-camera visibility audit on one exact SUGAR trace.

The audit renders one fixed external view and every predeclared rigid G1 d435
candidate on fifteen full-horizon physical states.  Semantic masks prove robot
and task-object visibility; no BPP model, feature, action prediction or loss is
loaded.  Run it once on CarryBox source 0 and once on KickBox source 0, then use
``finalize_bpp_dual_camera_contract.py`` to freeze the common passing camera.
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
os.environ.setdefault("ISAACLAB_TMP_ROOT", f"/tmp/Curiosity_bpp_camera_{job_id}")
os.environ.setdefault(
    "SUGAR_UNITREE_TMP_ROOT", f"/tmp/Curiosity_bpp_camera_unitree_{job_id}"
)
os.chdir(SUGAR)

from isaaclab.app import AppLauncher


if socket.gethostname().startswith(("mgmtserver", "login")):
    raise SystemExit("BPP camera audit is forbidden on a login node")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("BPP camera audit requires a retained Slurm allocation")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--source-motion-id", type=int, default=0)
parser.add_argument("--candidate-contract", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app
print("BPP_CAMERA_AUDIT_STAGE app_ready", flush=True)

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


EXPECTED_ACTIONS = 700
EXPECTED_FRAMES = 141
FRAME_STRIDE = 5
RTX_SIZE = 640
FAR_CLIP_M = 20.0
TASK_CFG = {"CarryBox": CarryPlayEnvCfg, "KickBox": KickPlayEnvCfg}
ROBOT_COLOR = np.asarray([0, 255, 0, 255], dtype=np.uint8)
OBJECT_COLOR = np.asarray([255, 0, 0, 255], dtype=np.uint8)
SEMANTIC_MAPPING = {
    "class:robot": tuple(map(int, ROBOT_COLOR)),
    "class:object": tuple(map(int, OBJECT_COLOR)),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def camera_cfg(spec: dict) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=str(spec["prim_path"]),
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(map(float, spec["position"])),
            rot=tuple(map(float, spec["quaternion_wxyz"])),
            convention=str(spec["convention"]),
        ),
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping=SEMANTIC_MAPPING,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(spec["focal_length_mm"]),
            focus_distance=4.0,
            horizontal_aperture=float(spec["horizontal_aperture_mm"]),
            clipping_range=(0.05, FAR_CLIP_M),
        ),
        width=RTX_SIZE,
        height=RTX_SIZE,
    )


def disable_randomization(cfg) -> None:
    for name in (
        "robot_physics_material",
        "obj_physics_material",
        "obj_mass",
        "add_joint_default_pos",
        "base_com",
        "push_robot",
        "push_object",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def load_source(trace_path: Path, source_motion_id: int) -> dict[str, np.ndarray]:
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
            "executed_action",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"trace missing exact-state arrays: {missing}")
        raw = {name: np.asarray(archive[name]) for name in required}
    source_ids = raw["source_motion_id_by_local_motion"].astype(np.int64)
    matches = np.flatnonzero(source_ids == int(source_motion_id))
    if matches.shape != (1,):
        raise RuntimeError(
            f"source motion {source_motion_id} occurs {len(matches)} times"
        )
    env_index = int(matches[0])
    if raw["executed_action"].shape[0] != EXPECTED_ACTIONS:
        raise RuntimeError("trace is not a complete 700-action trajectory")
    data: dict[str, np.ndarray] = {}
    env_count = len(source_ids)
    for name, value in raw.items():
        if value.ndim >= 2 and value.shape[1] == env_count:
            data[name] = value[:, env_index : env_index + 1]
        elif value.ndim >= 1 and value.shape[0] == env_count:
            data[name] = value[env_index : env_index + 1]
        else:
            data[name] = value
    data["trace_environment_index"] = np.asarray([env_index], dtype=np.int64)
    return data


def state_at(
    data: dict[str, np.ndarray], saved_frame_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if saved_frame_index < EXPECTED_FRAMES - 1:
        step = saved_frame_index * FRAME_STRIDE
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


def pixel_count(segmentation: np.ndarray, color: np.ndarray) -> int:
    if segmentation.shape != (RTX_SIZE, RTX_SIZE, 4):
        raise RuntimeError(f"semantic tensor shape drift: {segmentation.shape}")
    return int(np.count_nonzero(np.all(segmentation == color, axis=-1)))


def main() -> None:
    trace_path = args.trace.expanduser().resolve()
    result_path = trace_path.parent / "RESULT.json"
    collector_result = read_json(result_path)
    if collector_result.get("passed") is not True:
        raise RuntimeError("source collector result did not pass")
    task = str(collector_result.get("task_family"))
    if task not in TASK_CFG:
        raise RuntimeError(f"unsupported source task: {task}")
    candidate_path = args.candidate_contract.expanduser().resolve()
    candidates = read_json(candidate_path)
    if (
        candidates.get("protocol")
        != "sugar_bpp_dual_camera_candidate_contract_v1"
        or candidates.get("camera_keys")
        != ["agentview_rgb", "eye_in_hand_rgb"]
        or candidates.get("sampled_saved_frame_indices")
        != list(range(0, EXPECTED_FRAMES, 10))
        or len(candidates.get("eye_in_hand_candidates", [])) != 3
    ):
        raise RuntimeError("dual-camera candidate contract drift")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit: {output}")
    if not output.is_relative_to((ROOT / "experiments").resolve()):
        raise ValueError("camera evidence must remain under ignored experiments/")
    output.mkdir(parents=True, exist_ok=False)

    data = load_source(trace_path, args.source_motion_id)
    cfg = TASK_CFG[task]()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = float(candidates["environment_spacing_m"])
    cfg.seed = int(collector_result["seed"])
    cfg.sim.device = args.device
    cfg.observations.policy.enable_corruption = False
    cfg.scene.robot.spawn.semantic_tags = [("class", "robot")]
    cfg.scene.obj.spawn.semantic_tags = [("class", "object")]
    cfg.scene.bpp_agentview = camera_cfg(candidates["agentview_rgb"])
    sensor_names = ["bpp_agentview"]
    for spec in candidates["eye_in_hand_candidates"]:
        candidate_id = int(spec["candidate_id"])
        name = f"bpp_eye_candidate_{candidate_id}"
        setattr(cfg.scene, name, camera_cfg(spec))
        sensor_names.append(name)
    cfg.sim.render_interval = cfg.decimation
    disable_randomization(cfg)

    sim = SimulationContext(cfg.sim)
    scene = InteractiveScene(cfg.scene)
    sim.reset()
    scene.reset()
    scene_origin = scene.env_origins
    trace_origin = torch.as_tensor(
        data["environment_origin_w"], device=scene_origin.device
    )
    initial_focus_xy = 0.5 * (
        torch.as_tensor(
            data["robot_root_state_before_w"][0, :, :2],
            device=scene_origin.device,
        )
        + torch.as_tensor(
            data["object_root_state_before_w"][0, :, :2],
            device=scene_origin.device,
        )
    ) - trace_origin[:, :2]
    env_ids = torch.zeros(1, device=scene_origin.device, dtype=torch.long)
    measurements = {
        name: {
            "robot_pixels": [],
            "object_pixels": [],
            "rgb_sha256": [],
            "camera_position_w": [],
        }
        for name in sensor_names
    }
    frame_state_hashes = []
    for frame_index in candidates["sampled_saved_frame_indices"]:
        robot_root_np, joint_pos_np, joint_vel_np, object_root_np = state_at(
            data, int(frame_index)
        )
        frame_state_hashes.append(
            array_sha256(
                np.concatenate(
                    [
                        robot_root_np.reshape(-1),
                        joint_pos_np.reshape(-1),
                        joint_vel_np.reshape(-1),
                        object_root_np.reshape(-1),
                    ]
                )
            )
        )
        robot_root = torch.as_tensor(
            robot_root_np, device=scene_origin.device
        ).clone()
        object_root = torch.as_tensor(
            object_root_np, device=scene_origin.device
        ).clone()
        translation = scene_origin - trace_origin
        robot_root[:, :3] += translation
        object_root[:, :3] += translation
        robot_root[:, :2] -= initial_focus_xy
        object_root[:, :2] -= initial_focus_xy
        scene["robot"].write_root_state_to_sim(robot_root, env_ids=env_ids)
        scene["robot"].write_joint_state_to_sim(
            torch.as_tensor(joint_pos_np, device=scene_origin.device),
            torch.as_tensor(joint_vel_np, device=scene_origin.device),
            env_ids=env_ids,
        )
        scene["obj"].write_root_state_to_sim(object_root, env_ids=env_ids)
        scene.write_data_to_sim()
        sim.forward()
        sim.render()
        scene.update(dt=0.0)
        for sensor_name in sensor_names:
            camera = scene[sensor_name]
            camera.update(0.0, force_recompute=True)
            rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            segmentation = (
                camera.data.output["semantic_segmentation"][0]
                .detach()
                .cpu()
                .numpy()
            )
            if rgb.shape != (RTX_SIZE, RTX_SIZE, 3):
                raise RuntimeError(f"RGB tensor shape drift: {rgb.shape}")
            measurements[sensor_name]["robot_pixels"].append(
                pixel_count(segmentation, ROBOT_COLOR)
            )
            measurements[sensor_name]["object_pixels"].append(
                pixel_count(segmentation, OBJECT_COLOR)
            )
            measurements[sensor_name]["rgb_sha256"].append(array_sha256(rgb))
            measurements[sensor_name]["camera_position_w"].append(
                camera.data.pos_w[0].detach().cpu().tolist()
            )
            save_dir = output / sensor_name
            save_dir.mkdir(exist_ok=True)
            image_path = save_dir / f"{int(frame_index):03d}.png"
            if not cv2.imwrite(
                str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"failed to write {image_path}")
        print(
            f"BPP_CAMERA_AUDIT_PROGRESS task={task} frame={frame_index}",
            flush=True,
        )

    thresholds = candidates["visibility_thresholds"]
    agent = measurements["bpp_agentview"]
    agent_pass_count = sum(
        robot >= int(thresholds["minimum_robot_pixels_640x640"])
        and obj >= int(thresholds["minimum_object_pixels_640x640"])
        for robot, obj in zip(agent["robot_pixels"], agent["object_pixels"])
    )
    agent_passed = agent_pass_count >= int(
        thresholds["minimum_passing_frames_per_task"]
    )
    candidate_results = []
    for spec in candidates["eye_in_hand_candidates"]:
        candidate_id = int(spec["candidate_id"])
        name = f"bpp_eye_candidate_{candidate_id}"
        values = measurements[name]
        visibility_pass_count = sum(
            robot >= int(thresholds["minimum_robot_pixels_640x640"])
            and obj >= int(thresholds["minimum_object_pixels_640x640"])
            for robot, obj in zip(values["robot_pixels"], values["object_pixels"])
        )
        positions = np.asarray(values["camera_position_w"], dtype=np.float64)
        translation_range = float(
            np.max(np.linalg.norm(positions - positions[0], axis=-1))
        )
        identical_count = sum(
            left == right
            for left, right in zip(values["rgb_sha256"], agent["rgb_sha256"])
        )
        passed = bool(
            agent_passed
            and visibility_pass_count
            >= int(thresholds["minimum_passing_frames_per_task"])
            and translation_range
            >= float(thresholds["minimum_eye_camera_translation_range_m"])
            and identical_count == 0
        )
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "passed": passed,
                "visibility_passing_frames": visibility_pass_count,
                "robot_pixels": values["robot_pixels"],
                "object_pixels": values["object_pixels"],
                "robot_pixel_median": float(np.median(values["robot_pixels"])),
                "object_pixel_median": float(np.median(values["object_pixels"])),
                "eye_camera_translation_range_m": translation_range,
                "rgb_identical_to_agentview_frames": identical_count,
                "camera_position_w": values["camera_position_w"],
                "camera_spec": spec,
            }
        )
    result = {
        "protocol": "sugar_bpp_dual_camera_single_task_visibility_v1",
        "passed": bool(agent_passed and any(row["passed"] for row in candidate_results)),
        "task": task,
        "source_motion_id": int(args.source_motion_id),
        "source_trace": {
            "path": str(trace_path.relative_to(ROOT)),
            "sha256": file_sha256(trace_path),
            "environment_index": int(data["trace_environment_index"][0]),
        },
        "candidate_contract": {
            "path": str(candidate_path.relative_to(ROOT)),
            "sha256": file_sha256(candidate_path),
        },
        "sampled_saved_frame_indices": candidates["sampled_saved_frame_indices"],
        "frame_state_sha256": frame_state_hashes,
        "agentview": {
            "passed": agent_passed,
            "visibility_passing_frames": agent_pass_count,
            "robot_pixels": agent["robot_pixels"],
            "object_pixels": agent["object_pixels"],
            "rgb_sha256": agent["rgb_sha256"],
            "camera_position_w": agent["camera_position_w"],
            "camera_spec": candidates["agentview_rgb"],
        },
        "eye_in_hand_candidates": candidate_results,
        "thresholds": thresholds,
        "claim_boundary": (
            "Model-independent physical-state camera visibility only; no BPP "
            "checkpoint, feature, loss or predicted action was loaded."
        ),
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError(f"no camera candidate passed {task} visibility")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
