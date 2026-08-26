#!/usr/bin/env python3
"""Render clean exact-pose SUGAR reference videos for official XIRL/TCC or XSkill.

The output is an image-folder corpus in the exact layout consumed by the
released Google Research XIRL ``VideoDataset``::

    <output>/<split>/<CarryBox|KickBox>/<motion_id>/<frame_id>.png

Frames are produced by the IsaacLab RTX world camera from exact SUGAR robot
joint/root and object trajectories.  They contain no text, plots, borders or
policy results.  Rendering is kinematic reference playback, not a physics
rollout.  Exactly 64 normalized-time frames are emitted per source motion so
that frame index is a shared causal progress clock for the TCC gate.

The default ``g1`` embodiment preserves the immutable XIRL corpus contract.
The optional ``sphere`` embodiment follows the released XSkill simulation
intervention: hide the original agent and retain only fixed-radius spheres at
its task-independent end effectors.  Franka XSkill uses two 0.05 m gripper
spheres; the G1 compatibility rendering uses the same radius at both hands and
both feet so the visible embodiment is fixed across CarryBox and KickBox.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
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
os.environ.setdefault("ISAACLAB_TMP_ROOT", f"/tmp/Curiosity_xirl_render_{job_id}")
os.environ.setdefault(
    "SUGAR_UNITREE_TMP_ROOT", f"/tmp/Curiosity_xirl_render_unitree_{job_id}"
)
os.chdir(SUGAR)

from isaaclab.app import AppLauncher


if socket.gethostname().startswith(("mgmtserver", "login")):
    raise SystemExit("XIRL corpus rendering is forbidden on a login node")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("XIRL corpus rendering requires retained Slurm compute")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("CarryBox", "KickBox"), required=True)
parser.add_argument("--motion-ids", type=int, nargs="+", required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--frames-per-motion", type=int, default=64)
parser.add_argument("--camera-width", type=int, default=320)
parser.add_argument("--camera-height", type=int, default=320)
parser.add_argument("--embodiment", choices=("g1", "sphere"), default="g1")
parser.add_argument("--write-preview-mp4", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app
print("XIRL_RENDER_STAGE app_ready", flush=True)

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import isaacsim.core.utils.prims as prim_utils  # noqa: E402
import sugar_rl.tasks  # noqa: E402,F401
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from sugar_rl.tasks.locomanip.robots.g129dof.train_tracker.carry_box_tracker_env_cfg import (  # noqa: E402
    RobotPlayEnvCfg as CarryPlayEnvCfg,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_tracker.kick_box_tracker_env_cfg import (  # noqa: E402
    RobotPlayEnvCfg as KickPlayEnvCfg,
)
print("XIRL_RENDER_STAGE task_imports_ready", flush=True)


SOURCE_HZ = 50
OUTPUT_FRAME_COUNT = 64
RTX_RENDER_SIZE = 640
XSKILL_SPHERE_RADIUS_M = 0.05
XSKILL_SPHERE_BODY_NAMES = (
    "left_rubber_hand",
    "right_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
TASK_SPEC = {
    "CarryBox": (CarryPlayEnvCfg, 100),
    "KickBox": (KickPlayEnvCfg, 99),
}


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int = 10) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input closed")
        self.process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg preview encoding failed")


def split_for_motion(motion_id: int) -> str:
    remainder = int(motion_id) % 10
    if remainder == 8:
        return "valid"
    if remainder == 9:
        return "test"
    return "train"


def load_motion(task: str, motion_id: int) -> dict[str, np.ndarray]:
    directory = ROOT / f"SUGAR/data/{task}/data_{motion_id:03d}"
    with np.load(directory / "robot_50hz.npz", allow_pickle=False) as archive:
        robot = {name: np.asarray(archive[name]) for name in archive.files}
    with (directory / "obj_motion_global_50hz.pkl").open("rb") as stream:
        obj = {name: np.asarray(value) for name, value in pickle.load(stream).items()}
    fps = int(np.asarray(robot["fps"]).reshape(-1)[0])
    length = min(
        len(robot["joint_pos"]),
        len(robot["joint_vel"]),
        len(robot["body_pos_w"]),
        len(obj["obj_trans"]),
        len(obj["obj_rot"]),
        len(obj["obj_lin_vel"]),
        len(obj["obj_ang_vel"]),
    )
    if fps != SOURCE_HZ or length < args.frames_per_motion:
        raise RuntimeError(f"invalid source {task}:{motion_id}, fps={fps}, length={length}")
    object_rotation = torch.from_numpy(obj["obj_rot"][:length]).float()
    object_quaternion = math_utils.quat_from_matrix(object_rotation).cpu().numpy()
    robot_root = np.concatenate(
        (
            robot["body_pos_w"][:length, 0],
            robot["body_quat_w"][:length, 0],
            robot["body_lin_vel_w"][:length, 0],
            robot["body_ang_vel_w"][:length, 0],
        ),
        axis=-1,
    ).astype(np.float32)
    object_root = np.concatenate(
        (
            obj["obj_trans"][:length], object_quaternion,
            obj["obj_lin_vel"][:length], obj["obj_ang_vel"][:length],
        ),
        axis=-1,
    ).astype(np.float32)
    indices = np.rint(np.linspace(0, length - 1, args.frames_per_motion)).astype(np.int64)
    if len(np.unique(indices)) != args.frames_per_motion:
        raise RuntimeError("normalized frame sampling produced duplicates")
    return {
        "source_indices": indices,
        "source_length": np.asarray([length], dtype=np.int32),
        "robot_root": robot_root[indices],
        "object_root": object_root[indices],
        "joint_pos": np.asarray(robot["joint_pos"][:length][indices], dtype=np.float32),
        "joint_vel": np.asarray(robot["joint_vel"][:length][indices], dtype=np.float32),
    }


def disable_randomization(cfg) -> None:
    for name in (
        "robot_physics_material", "obj_physics_material", "obj_mass",
        "add_joint_default_pos", "base_com", "push_robot", "push_object",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/XirlWorldCamera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.6, 3.6, 2.4),
            rot=(0.3043649418, 0.2319667899, 0.5600173703, 0.7348019703),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        # Render above the DLSS minimum and downsample only the clean RGB
        # output.  Passing a 320 px render target makes Isaac Sim 5.1 choose
        # an internal 186 px DLSS input and can stall RTX sensor startup.
        width=RTX_RENDER_SIZE,
        height=RTX_RENDER_SIZE,
    )


def sphere_agent_cfg() -> VisualizationMarkersCfg:
    """Return the task-independent released-XSkill-style sphere agent."""

    return VisualizationMarkersCfg(
        prim_path="/Visuals/XSkillSphereAgent",
        markers={
            "end_effector": sim_utils.SphereCfg(
                radius=XSKILL_SPHERE_RADIUS_M,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.99, 0.92),
                    roughness=0.5,
                ),
            )
        },
    )


def decode_preview(path: Path) -> bool:
    process = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
            "-map", "0:v:0", "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    return process.returncode == 0 and path.stat().st_size > 4096


def main() -> None:
    cfg_class, motion_count = TASK_SPEC[args.task]
    motion_ids = tuple(sorted(set(int(value) for value in args.motion_ids)))
    if len(motion_ids) != len(args.motion_ids):
        raise ValueError("motion IDs must be unique")
    if not motion_ids or min(motion_ids) < 0 or max(motion_ids) >= motion_count:
        raise ValueError("motion ID outside official task range")
    if args.frames_per_motion != OUTPUT_FRAME_COUNT:
        raise ValueError(f"frozen corpus requires exactly {OUTPUT_FRAME_COUNT} frames")
    output = args.output_root.expanduser().resolve()
    if not output.is_relative_to((ROOT / "experiments").resolve()):
        raise ValueError("XIRL corpus must remain under ignored experiments/")
    motions = [load_motion(args.task, motion_id) for motion_id in motion_ids]
    print(
        f"XIRL_RENDER_STAGE sources_ready task={args.task} motions={motion_ids}",
        flush=True,
    )
    frame_dirs = [
        output / split_for_motion(motion_id) / args.task / str(motion_id)
        for motion_id in motion_ids
    ]
    for directory in frame_dirs:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(directory)
        directory.mkdir(parents=True, exist_ok=True)

    cfg = cfg_class()
    cfg.scene.num_envs = len(motions)
    cfg.seed = 271401
    cfg.sim.device = args.device
    cfg.scene.world_camera = camera_cfg()
    cfg.sim.render_interval = cfg.decimation
    cfg.observations.policy.enable_corruption = False
    disable_randomization(cfg)
    writers: list[FfmpegWriter] = []
    try:
        print(f"XIRL_RENDER_STAGE creating_scene task={args.task}", flush=True)
        sim = SimulationContext(cfg.sim)
        scene = InteractiveScene(cfg.scene)
        sim.reset()
        scene.reset()
        sphere_agent = None
        sphere_body_ids: list[int] = []
        sphere_body_names: list[str] = []
        if args.embodiment == "sphere":
            sphere_body_ids, sphere_body_names = scene["robot"].find_bodies(
                XSKILL_SPHERE_BODY_NAMES,
                preserve_order=True,
            )
            if tuple(sphere_body_names) != XSKILL_SPHERE_BODY_NAMES:
                raise RuntimeError(
                    "sphere embodiment end-effector mismatch: "
                    f"expected={XSKILL_SPHERE_BODY_NAMES}, actual={sphere_body_names}"
                )
            robot_prims = sim_utils.find_matching_prims(scene["robot"].cfg.prim_path)
            if len(robot_prims) != len(motions):
                raise RuntimeError(
                    f"expected {len(motions)} robot prims, found {len(robot_prims)}"
                )
            for robot_prim in robot_prims:
                prim_utils.set_prim_visibility(robot_prim, False)
            sphere_agent = VisualizationMarkers(sphere_agent_cfg())
        print(f"XIRL_RENDER_STAGE scene_ready task={args.task}", flush=True)
        origin = scene.env_origins
        device = origin.device
        env_ids = torch.arange(len(motions), device=device, dtype=torch.long)
        if args.write_preview_mp4:
            writers = [
                FfmpegWriter(
                    output / "preview_videos" / f"{args.task}_{motion_id:03d}.mp4",
                    args.camera_width,
                    args.camera_height,
                )
                for motion_id in motion_ids
            ]
        for frame_id in range(args.frames_per_motion):
            robot_root = torch.as_tensor(
                np.stack([motion["robot_root"][frame_id] for motion in motions]),
                device=device,
            ).clone()
            object_root = torch.as_tensor(
                np.stack([motion["object_root"][frame_id] for motion in motions]),
                device=device,
            ).clone()
            robot_root[:, :3] += origin
            object_root[:, :3] += origin
            joint_pos = torch.as_tensor(
                np.stack([motion["joint_pos"][frame_id] for motion in motions]),
                device=device,
            )
            joint_vel = torch.as_tensor(
                np.stack([motion["joint_vel"][frame_id] for motion in motions]),
                device=device,
            )
            scene["robot"].write_root_state_to_sim(robot_root, env_ids=env_ids)
            scene["robot"].write_joint_state_to_sim(
                joint_pos, joint_vel, env_ids=env_ids
            )
            scene["obj"].write_root_state_to_sim(object_root, env_ids=env_ids)
            scene.write_data_to_sim()
            sim.forward()
            if sphere_agent is not None:
                # Pull the exact kinematic body transforms before RTX renders.
                # The marker array is global, so flatten env-major positions.
                scene["robot"].update(0.0)
                sphere_positions = scene["robot"].data.body_pos_w[
                    :, sphere_body_ids, :
                ].reshape(-1, 3)
                if not torch.isfinite(sphere_positions).all():
                    raise RuntimeError("non-finite XSkill sphere-agent position")
                sphere_agent.visualize(translations=sphere_positions)
            sim.render()
            scene.update(dt=0.0)
            camera = scene["world_camera"]
            camera.update(0.0, force_recompute=True)
            rendered = camera.data.output["rgb"][:, ..., :3].detach().cpu().numpy()
            if rendered.shape != (
                len(motions), RTX_RENDER_SIZE, RTX_RENDER_SIZE, 3
            ):
                raise RuntimeError(f"unexpected camera tensor {rendered.shape}")
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
            if rgb.shape != (
                len(motions), args.camera_height, args.camera_width, 3
            ):
                raise RuntimeError(f"unexpected camera tensor {rgb.shape}")
            for row, image in enumerate(rgb):
                if int(image.max()) == int(image.min()):
                    raise RuntimeError(f"constant camera frame for motion {motion_ids[row]}")
                path = frame_dirs[row] / f"{frame_id}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"failed to write {path}")
                if writers:
                    writers[row].append(image)
            if frame_id in (0, 15, 31, 47, 63):
                print(
                    f"XIRL_RENDER_PROGRESS task={args.task} frame={frame_id + 1}/64 "
                    f"motions={motion_ids}",
                    flush=True,
                )
    finally:
        for writer in writers:
            writer.close()

    # Do not call SimulationContext.stop() here.  In the headless RTX sensor
    # application it can block after all camera frames have been delivered.
    # The outer simulation_app.close() is the single owner of Kit teardown.

    videos = [
        output / "preview_videos" / f"{args.task}_{motion_id:03d}.mp4"
        for motion_id in motion_ids
    ]
    frame_counts = [len(list(directory.glob("*.png"))) for directory in frame_dirs]
    result = {
        "protocol": (
            "sugar_clean_xirl_reference_render_v1"
            if args.embodiment == "g1"
            else "sugar_clean_xskill_sphere_reference_render_v1"
        ),
        "passed": bool(
            all(count == OUTPUT_FRAME_COUNT for count in frame_counts)
            and (not args.write_preview_mp4 or all(decode_preview(path) for path in videos))
        ),
        "task": args.task,
        "embodiment": args.embodiment,
        "motion_ids": list(motion_ids),
        "split_by_motion_id": {
            str(motion_id): split_for_motion(motion_id) for motion_id in motion_ids
        },
        "frames_per_motion": OUTPUT_FRAME_COUNT,
        "resolution": [args.camera_width, args.camera_height],
        "rtx_render_resolution": [RTX_RENDER_SIZE, RTX_RENDER_SIZE],
        "source": "exact SUGAR 50Hz root/joint/object trajectory",
        "render": "IsaacLab RTX TiledCamera exact-pose playback; no physics replay",
        "clean_frame_contract": "RGB only; no text, plot, border, metric or policy output",
        "sphere_agent_contract": (
            {
                "original_g1_visibility": False,
                "body_names": list(XSKILL_SPHERE_BODY_NAMES),
                "radius_m": XSKILL_SPHERE_RADIUS_M,
                "task_independent_visible_body_set": True,
                "released_xskill_analogy": (
                    "Franka meshes transparent; two 0.05 m gripper spheres"
                ),
            }
            if args.embodiment == "sphere"
            else None
        ),
        "frame_counts": frame_counts,
        "preview_videos": [str(path) for path in videos] if writers else [],
    }
    embodiment_prefix = "" if args.embodiment == "g1" else "SPHERE_"
    result_path = output / (
        f"RENDER_RESULT_{embodiment_prefix}{args.task}_"
        f"{motion_ids[0]:03d}_{motion_ids[-1]:03d}.json"
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"XIRL clean render failed: {result_path}")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    # Isaac Sim 5.1 can block indefinitely while tearing down a headless RTX
    # TiledCamera application, even after every frame and result file has been
    # flushed.  Each task is intentionally rendered in an isolated process, so
    # terminate at the process boundary and let the OS reclaim Kit/CUDA state.
    # This is a lifecycle workaround only; it does not alter frames or labels.
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
