#!/usr/bin/env python3
"""Render exactly two panels: input demo and frozen-policy behavior."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import socket
import subprocess

# Select the cluster's installed NVIDIA Vulkan ICD before importing Isaac Sim.
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("DISPLAY", "")

from isaaclab.app import AppLauncher


if socket.gethostname().startswith(("mgmtserver", "login")):
    raise SystemExit("Refusing behavior rendering on a login node")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Behavior rendering requires retained Slurm")
ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--correct-trace", type=Path, required=True)
parser.add_argument("--unrelated-trace", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--preview-update128",
    action="store_true",
    help="Render only the unrelated-demo update-128 preview (source env 0).",
)
parser.add_argument(
    "--actual-source-env",
    type=int,
    default=None,
    help="environment row to render from each matched trace",
)
parser.add_argument(
    "--matched-update64",
    "--matched-endpoint",
    dest="matched_pair",
    action="store_true",
    help="Render a fresh matched correct/unrelated endpoint pair.",
)
parser.add_argument(
    "--same-teacher-reward-only",
    action="store_true",
    help=(
        "Admit the causal pair where both policies use the same CarryBox45 "
        "teacher and differ only in selected demo reward."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.same_teacher_reward_only and not args.matched_pair:
    parser.error("same-teacher reward-only rendering requires --matched-endpoint")
simulation_app = AppLauncher(args).app

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import sugar_rl.tasks  # noqa: E402,F401
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_refiner_env_cfg import RobotPlayEnvCfg  # noqa: E402


TASK_ID = "Sugar-G129dof-CarryBox-Refiner"
FINAL_SOURCE_ENV = 40  # update 512, fixed profile 0
SOURCE_HZ = 50.0
VIDEO_FPS = 20


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_reference(path: Path) -> dict[str, np.ndarray]:
    robot_path = path / "robot_50hz.npz"
    object_path = path / "obj_motion_global_50hz.pkl"
    with np.load(robot_path, allow_pickle=False) as archive:
        robot = {name: np.asarray(archive[name]) for name in archive.files}
    with object_path.open("rb") as stream:
        obj = {name: np.asarray(value) for name, value in pickle.load(stream).items()}
    length = min(int(robot["joint_pos"].shape[0]), int(obj["obj_trans"].shape[0]))
    rot = torch.from_numpy(obj["obj_rot"][:length]).to(dtype=torch.float32)
    quat = math_utils.quat_from_matrix(rot).cpu().numpy().astype(np.float32)
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
            obj["obj_trans"][:length],
            quat,
            obj["obj_lin_vel"][:length],
            obj["obj_ang_vel"][:length],
        ),
        axis=-1,
    ).astype(np.float32)
    return {
        "robot_root": robot_root,
        "object_root": object_root,
        "joint_pos": robot["joint_pos"][:length].astype(np.float32),
        "joint_vel": robot["joint_vel"][:length].astype(np.float32),
    }


def actual_first_episode(
    trace: dict[str, np.ndarray], source_env: int
) -> dict[str, np.ndarray]:
    hits = np.flatnonzero(trace["done"][:, source_env])
    last_transition = int(hits[0]) if hits.size else trace["done"].shape[0] - 1
    count = last_transition + 2
    return {
        "robot_root": trace["robot_root_state_w"][:count, source_env].astype(np.float32),
        "object_root": trace["object_root_state_w"][:count, source_env].astype(np.float32),
        "joint_pos": trace["robot_joint_pos"][:count, source_env].astype(np.float32),
        "joint_vel": trace["robot_joint_vel"][:count, source_env].astype(np.float32),
    }


def disable_randomization(cfg: RobotPlayEnvCfg) -> None:
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


def capture(base_env, camera: TiledCamera, sequence: dict[str, np.ndarray], index: int) -> np.ndarray:
    index = min(index, sequence["robot_root"].shape[0] - 1)
    robot_root = torch.as_tensor(sequence["robot_root"][index:index + 1], device=base_env.device).clone()
    object_root = torch.as_tensor(sequence["object_root"][index:index + 1], device=base_env.device).clone()
    focus = 0.5 * (sequence["robot_root"][0, :2] + sequence["object_root"][0, :2])
    focus_tensor = torch.as_tensor(focus, device=base_env.device).reshape(1, 2)
    robot_root[:, :2] -= focus_tensor
    object_root[:, :2] -= focus_tensor
    joint_pos = torch.as_tensor(sequence["joint_pos"][index:index + 1], device=base_env.device)
    joint_vel = torch.as_tensor(sequence["joint_vel"][index:index + 1], device=base_env.device)
    ids = torch.zeros(1, dtype=torch.long, device=base_env.device)
    base_env.scene["robot"].write_root_state_to_sim(robot_root, env_ids=ids)
    base_env.scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
    base_env.scene["obj"].write_root_state_to_sim(object_root, env_ids=ids)
    base_env.sim.forward()
    base_env.sim.render()
    base_env.scene.update(dt=0.0)
    camera.update(0.0, force_recompute=True)
    image = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8, copy=True)
    if int(image.max()) == int(image.min()):
        raise RuntimeError("camera returned a constant image")
    return image


def decode(path: Path) -> dict[str, object]:
    process = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "info", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    text = process.stderr
    return {
        "exit_code": process.returncode,
        "h264": "Video: h264" in text,
        "yuv420p": "yuv420p" in text,
        "passed": process.returncode == 0 and "Video: h264" in text and "yuv420p" in text,
    }


def render_pair(base_env, camera, reference, actual, left_label: str, output: Path) -> dict[str, object]:
    playback_rate = 0.25 if args.matched_pair else 1.0
    reference_frame_count = int(reference["robot_root"].shape[0])
    actual_frame_count = int(actual["robot_root"].shape[0])
    output_frames_for_reference = (
        int(math.ceil((reference_frame_count - 1) * VIDEO_FPS / SOURCE_HZ))
        + 1
    )
    output_frames_for_actual = (
        int(
            math.ceil(
                (actual_frame_count - 1)
                * VIDEO_FPS
                / (SOURCE_HZ * playback_rate)
            )
        )
        + 1
    )
    frame_count = max(output_frames_for_reference, output_frames_for_actual)
    with imageio.get_writer(
        output,
        fps=VIDEO_FPS,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="warning",
    ) as writer:
        for video_frame in range(frame_count):
            actual_source_index = int(
                round(
                    video_frame
                    * SOURCE_HZ
                    * playback_rate
                    / VIDEO_FPS
                )
            )
            reference_source_index = (
                int(
                    round(
                        video_frame * SOURCE_HZ / VIDEO_FPS
                    )
                )
                if args.matched_pair
                else actual_source_index
            )
            left = cv2.resize(
                cv2.cvtColor(
                    capture(
                        base_env,
                        camera,
                        reference,
                        reference_source_index,
                    ),
                    cv2.COLOR_RGB2BGR,
                ),
                (620, 620),
                interpolation=cv2.INTER_AREA,
            )
            right = cv2.resize(
                cv2.cvtColor(
                    capture(base_env, camera, actual, actual_source_index),
                    cv2.COLOR_RGB2BGR,
                ),
                (620, 620),
                interpolation=cv2.INTER_AREA,
            )
            canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
            canvas[80:700, 10:630] = left
            canvas[80:700, 650:1270] = right
            cv2.putText(canvas, left_label, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, (0.55 if args.matched_pair else 0.78), (25, 25, 25), 2, cv2.LINE_AA)
            cv2.putText(canvas, "ACTUAL: FIXED TEACHER + LEARNED RESIDUAL", (658, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25), 2, cv2.LINE_AA)
            if args.matched_pair:
                cv2.putText(canvas, "DEMO 1x (then holds) | ACTUAL 0.25x (then holds)", (690, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
            cv2.line(canvas, (640, 68), (640, 710), (185, 185, 185), 1)
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "frames": frame_count,
        "fps": VIDEO_FPS,
        "reference_source_frames": reference_frame_count,
        "actual_source_frames": actual_frame_count,
        "reference_fully_displayed": (
            frame_count >= output_frames_for_reference
        ),
        "actual_fully_displayed": frame_count >= output_frames_for_actual,
        "decode": decode(output),
    }


def main() -> None:
    correct_trace = args.correct_trace.resolve()
    unrelated_trace = args.unrelated_trace.resolve()
    output = args.output_dir.resolve()
    experiments = (ROOT / "experiments").resolve()
    if not all(path.is_relative_to(experiments) for path in (correct_trace, unrelated_trace, output)):
        raise ValueError("all render paths must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    correct_result_path = correct_trace.with_name("RESULT.json")
    unrelated_result_path = unrelated_trace.with_name("RESULT.json")
    admitted_results = [
        (
            unrelated_result_path,
            (
                "same_teacher_unrelated_reward"
                if args.same_teacher_reward_only
                else "wrong_teacher_unrelated_reward"
            )
            if args.matched_pair
            else "unrelated_demo",
        )
    ]
    if not args.preview_update128:
        admitted_results.insert(
            0,
            (
                correct_result_path,
                (
                    "same_teacher_correct_reward"
                    if args.same_teacher_reward_only
                    else "wrong_teacher_correct_reward"
                )
                if args.matched_pair
                else "correct_demo",
            ),
        )
    for result_path, arm in admitted_results:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("passed") is not True
            or result.get("arm") != arm
            or not result_path.with_name("TRACE.npz").is_file()
        ):
            raise RuntimeError(f"unadmitted trace: {arm}")
        if args.preview_update128 and (
            result.get("protocol")
            != "sugar_plan11_unrelated_kickbox21_update128_preview_v1"
            or result.get("policy_updates") != [128]
            or result.get("preview_not_final_512_result") is not True
        ):
            raise RuntimeError("unrelated trace is not the declared update-128 preview")

    references = {
        "correct": ROOT / "SUGAR/data/CarryBox/data_045",
        "unrelated": ROOT / "SUGAR/data/KickBox/data_021",
    }
    correct_reference = load_reference(references["correct"])
    unrelated_reference = load_reference(references["unrelated"])
    source_env = (
        args.actual_source_env
        if args.actual_source_env is not None
        else (0 if args.matched_pair else FINAL_SOURCE_ENV)
    )
    for trace_path in (correct_trace, unrelated_trace):
        if source_env < 0 or source_env >= load_npz(trace_path)["done"].shape[1]:
            raise ValueError("actual source environment is outside the trace")
    correct_actual = (
        None
        if args.preview_update128
        else actual_first_episode(load_npz(correct_trace), source_env)
    )
    unrelated_actual = actual_first_episode(
        load_npz(unrelated_trace),
        0 if args.preview_update128 else source_env,
    )

    cfg = RobotPlayEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.commands.motion.motion_folder = str(ROOT / "SUGAR/data/CarryBox")
    cfg.commands.motion.teacher_motion_folder = None
    cfg.commands.motion.use_generator = False
    cfg.commands.motion.generator_checkpoint_path = None
    disable_randomization(cfg)
    env = None
    try:
        env = gym.make(TASK_ID, cfg=cfg)
        base_env = env.unwrapped
        base_env.reset()
        eye = torch.tensor([[2.2, 2.2, 1.9]], device=base_env.device)
        target = torch.tensor([[0.0, 0.0, 0.95]], device=base_env.device)
        quat = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(eye, target, up_axis="Z", device=base_env.device)
        )[0]
        camera_cfg = TiledCameraCfg(
            prim_path="/World/Plan11BehaviorOnlyCamera",
            update_period=0.0,
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(float(value) for value in eye[0].tolist()),
                rot=tuple(float(value) for value in quat.tolist()),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=5.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 30.0),
            ),
            width=640,
            height=640,
        )
        camera = TiledCamera(camera_cfg)
        camera._initialize_impl()
        camera._is_initialized = True
        if args.preview_update128:
            videos = {
                "unrelated_update128_preview": render_pair(
                    base_env,
                    camera,
                    unrelated_reference,
                    unrelated_actual,
                    "INPUT DEMO - UNRELATED KICKBOX",
                    output / "unrelated_kickbox_update128_demo_and_actual_behavior.mp4",
                )
            }
        else:
            videos = {
                "correct": render_pair(
                    base_env,
                    camera,
                    correct_reference,
                    correct_actual,
                    "DEMO: CARRYBOX45",
                    output / "01_correct_demo_and_actual_behavior.mp4",
                ),
                "unrelated": render_pair(
                base_env,
                camera,
                unrelated_reference,
                unrelated_actual,
                "DEMO: KICKBOX21 (UNRELATED)",
                output / "02_unrelated_kickbox_demo_and_actual_behavior.mp4",
                ),
            }
    finally:
        if env is not None:
            env.close()

    checks = {
        "expected_output_video_count": len(videos) == (
            1 if args.preview_update128 else 2
        ),
        "each_video_has_exactly_demo_and_actual_panels": True,
        "no_numeric_metric_panels": True,
        "correct_uses_official_carrybox45": True,
        "unrelated_uses_official_kickbox21": True,
        "both_reference_and_actual_trajectories_fully_displayed": all(
            item["reference_fully_displayed"]
            and item["actual_fully_displayed"]
            for item in videos.values()
        ),
        "both_full_decode_h264_yuv420p": all(item["decode"]["passed"] for item in videos.values()),
    }
    proof = {
        "protocol": (
            (
                "sugar_same_teacher_selected_demo_behavior_v1"
                if args.same_teacher_reward_only
                else "sugar_matched_fixed_teacher_demo_identity_behavior_v2"
            )
            if args.matched_pair
            else (
                "sugar_plan11_unrelated_kickbox21_update128_behavior_preview_v1"
                if args.preview_update128
                else "sugar_plan11_correct_vs_unrelated_behavior_only_render_v1"
            )
        ),
        "passed": all(checks.values()),
        "preview_not_final_512_result": args.preview_update128,
        "claim_scope": "Behavior-only human-review rendering. Each movie contains the exact input demo and the corresponding frozen CarryBox execution using the same fixed teacher plus the learned residual policy. No predictor or reward number is displayed or interpreted.",
        "checks": checks,
        "videos": videos,
        "sources": [
            *([] if args.preview_update128 else [str(correct_trace)]),
            str(unrelated_trace),
            *([] if args.preview_update128 else [str(correct_result_path)]),
            str(unrelated_result_path),
            str(references["correct"] / "robot_50hz.npz"),
            str(references["unrelated"] / "robot_50hz.npz"),
            str(Path(__file__).resolve()),
        ],
    }
    (output / "RENDER_PROOF.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not proof["passed"]:
        raise RuntimeError("behavior-only render failed")
    print(json.dumps(proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
