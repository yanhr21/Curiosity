#!/usr/bin/env python3
"""Record native anatomical whole-hand tactile fields on a SUGAR G1 object task.

This is a no-learning visualization collector. The official frozen SUGAR
Refiner controls the sensorized G1. A release failure relaxes every joint
target after ``release_step``. A closure failure returns only the right arm
to its neutral action before box contact, so the left and right native tactile
maps expose the resulting partial/missed closure. The box remains a dynamic
PhysX body in every condition.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback


HOST = socket.gethostname()
if HOST.startswith(("mgmtserver", "login")):
    raise SystemExit(f"Refusing IsaacLab collection on login node: {HOST}")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("A retained Slurm allocation is required")

os.environ.setdefault("DISPLAY", "")

from isaaclab.app import AppLauncher


ROOT = Path(os.environ.get("CURIOSITY_ROOT", Path(__file__).resolve().parents[3])).resolve()
REFINER_TASK_ID = (
    "Sugar-G129dof-CarryBox-Official-Refiner-Anatomical27-"
    "WholeHand-TacSL-Audit"
)
REFINER_CHECKPOINT = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/"
    "baseline/ckpts/refiner_model10000.pt"
)
PICKBOTTLE_TASK_ID = "Sugar-G129dof-PickBottle-Tracker"
PICKBOTTLE_TRACKER_CHECKPOINT = ROOT / "SUGAR/demo_ckpts/PickBottle/tracker.pt"
PATCHES = (
    *(f"palm_r{row}_c{column}" for row in range(4) for column in range(3)),
    *(
        f"{digit}_{segment}"
        for digit in ("thumb", "index", "middle", "ring", "little")
        for segment in ("proximal", "middle", "distal")
    ),
)
SIDES = ("left", "right")

# Exact principal frame of the official SUGAR CarryBox rigid mesh. Columns are
# the PCA axes in the object root frame; bounds are measured over all 50,004
# mesh vertices. The open-loop demo uses PCA2 as the physical bottom/top axis
# and PCA0 for the braced side.
CARRYBOX_PCA_CENTER_B = (-0.0011075759, -0.0005471044, 0.0052253723)
CARRYBOX_PCA_BASIS_B = (
    (-0.0745516238, 0.9870564599, -0.1419915504),
    (0.9083135305, 0.0084440003, -0.4182047694),
    (-0.4115927425, -0.1601506911, -0.8971862518),
)
CARRYBOX_PCA0_MAX_M = 0.2197713927
CARRYBOX_PCA1_EDGE_INSET_M = 0.075
# The scanned CarryBox bottom is not planar. At the selected palm-support
# strip (PCA1=-0.075 m), the outer shell is at PCA2=-0.179 m; using the global
# -0.189 m vertex leaves the palm about a centimetre below the local surface.
CARRYBOX_LOCAL_SUPPORT_PCA2_M = -0.179

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument(
    "--object-kind",
    choices=("carrybox", "bottle", "palm_grip"),
    default="carrybox",
    help="Dynamic IsaacLab object grasped by the complete sensorized G1.",
)
parser.add_argument(
    "--object-scale",
    type=float,
    nargs=3,
    default=None,
    metavar=("SX", "SY", "SZ"),
    help="Optional root scale override for the selected rigid object.",
)
parser.add_argument(
    "--action-trace",
    type=Path,
    default=None,
    help="Replay applied_action from a completed G1 trace instead of querying the teacher online.",
)
parser.add_argument(
    "--scenario",
    choices=(
        "unmodified_official_policy",
        "successful_grasp",
        "failed_grasp",
        "failed_closure",
        "bottom_support_lift",
    ),
    required=True,
)
parser.add_argument("--seed", type=int, default=4263)
parser.add_argument("--motion-id", type=int, default=45)
parser.add_argument(
    "--start-step",
    type=int,
    default=0,
    help="Initialize robot and object together from this official motion frame.",
)
parser.add_argument("--max-steps", type=int, default=660)
parser.add_argument(
    "--continue-after-termination",
    action="store_true",
    help=(
        "Keep recording the same physical rollout after a task termination "
        "signal. The environment reset path remains disabled; actions, "
        "physics, and native tactile sensing are unchanged."
    ),
)
parser.add_argument(
    "--bottom-support-open-loop",
    action="store_true",
    help=(
        "Use bilateral IsaacLab DLS IK to place the left palm under the plain "
        "CarryBox, brace its side with the right palm, and lift both hands."
    ),
)
parser.add_argument("--release-step", type=int, default=360)
parser.add_argument("--closure-fault-step", type=int, default=210)
parser.add_argument(
    "--mass-kg",
    type=float,
    default=None,
    help="Object mass. Defaults to the official task mass for each object.",
)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument(
    "--force-only",
    action="store_true",
    help="Disable RTX/world and optical cameras while retaining all native TacSL force fields.",
)
parser.add_argument(
    "--disable-optical",
    action="store_true",
    help="Keep the world camera but disable GelSight optical cameras.",
)
parser.add_argument(
    "--hold-reset-pose",
    action="store_true",
    help="Hold the exact reset joint pose for a controlled rigid palm-press sample.",
)
parser.add_argument("--physical-stiffness", type=float, default=1500.0)
parser.add_argument("--physical-damping", type=float, default=300.0)
parser.add_argument("--normal-stiffness", type=float, default=199.35014495534745)
parser.add_argument("--tangential-stiffness", type=float, default=19.935014495534745)
parser.add_argument(
    "--contact-friction",
    type=float,
    default=None,
    help=(
        "Exact physical patch/object friction and TacSL Coulomb coefficient. "
        "Omit to retain the task's seeded object-material draw and nominal 0.5 patch/TacSL value."
    ),
)
parser.add_argument(
    "--wrist-yaw-target-offset-rad",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("LEFT", "RIGHT"),
    help=(
        "Constant left/right wrist-yaw joint-target offsets in radians, applied "
        "after the official policy action."
    ),
)
parser.add_argument(
    "--shoulder-pitch-target-offset-rad",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("LEFT", "RIGHT"),
)
parser.add_argument(
    "--shoulder-roll-target-offset-rad",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("LEFT", "RIGHT"),
)
parser.add_argument(
    "--joint-offset-ramp-steps",
    type=int,
    default=0,
    help="Linearly introduce the declared joint-target offsets over this many control steps.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

output_root = args.output_root.expanduser().resolve()
action_trace_path = (
    args.action_trace.expanduser().resolve() if args.action_trace is not None else None
)
experiment_root = (ROOT / "experiments").resolve()
if not output_root.is_relative_to(experiment_root):
    raise SystemExit("Output must remain below experiments/")
if output_root.exists():
    raise SystemExit(f"Refusing overwrite: {output_root}")
active_checkpoint = (
    PICKBOTTLE_TRACKER_CHECKPOINT
    if args.object_kind == "bottle"
    else REFINER_CHECKPOINT
)
if args.action_trace is None and not active_checkpoint.is_file():
    raise SystemExit(f"Missing official SUGAR checkpoint: {active_checkpoint}")
if action_trace_path is not None and not action_trace_path.is_file():
    raise SystemExit(f"Missing G1 action trace: {action_trace_path}")
if args.object_kind == "bottle" and args.action_trace is not None:
    raise SystemExit(
        "PickBottle must use its released official Tracker, not a CarryBox action trace"
    )
if args.max_steps < 30:
    raise SystemExit("At least 30 recorded control steps are required")
if args.start_step < 0:
    raise SystemExit("start-step must be nonnegative")
if args.contact_friction is not None and not 0.0 <= args.contact_friction <= 2.0:
    raise SystemExit("contact-friction must lie in [0, 2]")
if args.mass_kg is not None and args.mass_kg <= 0.0:
    raise SystemExit("mass-kg must be positive")
if args.joint_offset_ramp_steps < 0:
    raise SystemExit("joint-offset-ramp-steps must be nonnegative")
if args.bottom_support_open_loop and args.object_kind != "carrybox":
    raise SystemExit("bottom-support-open-loop requires object-kind=carrybox")
if args.bottom_support_open_loop and args.action_trace is not None:
    raise SystemExit("bottom-support-open-loop does not accept an action trace")
if args.bottom_support_open_loop != (args.scenario == "bottom_support_lift"):
    raise SystemExit(
        "scenario=bottom_support_lift and --bottom-support-open-loop must be used together"
    )
if args.scenario == "failed_grasp" and not 1 <= args.release_step < args.max_steps:
    raise SystemExit("release_step must lie inside the failed-grasp rollout")
if args.scenario == "failed_closure" and not 1 <= args.closure_fault_step < args.max_steps:
    raise SystemExit("closure_fault_step must lie inside the failed-closure rollout")

os.environ["CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_STIFFNESS"] = str(
    args.physical_stiffness
)
os.environ["CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_DAMPING"] = str(
    args.physical_damping
)
os.environ["CURIOSITY_ANATOMICAL_TACSL_NORMAL_STIFFNESS"] = str(
    args.normal_stiffness
)
os.environ["CURIOSITY_ANATOMICAL_TACSL_TANGENTIAL_STIFFNESS"] = str(
    args.tangential_stiffness
)
contact_friction = 0.5 if args.contact_friction is None else args.contact_friction
os.environ["CURIOSITY_ANATOMICAL_TACSL_FRICTION_COEFFICIENT"] = str(
    contact_friction
)
if args.contact_friction is not None:
    os.environ["CURIOSITY_ANATOMICAL_PHYSX_STATIC_FRICTION"] = str(
        args.contact_friction
    )
    os.environ["CURIOSITY_ANATOMICAL_PHYSX_DYNAMIC_FRICTION"] = str(
        args.contact_friction
    )
os.environ["CURIOSITY_ENABLE_ANATOMICAL27_WHOLE_HAND_TACSL_AUDIT"] = "1"
os.environ["SUGAR_DISABLE_TRAIN_DEBUG_VIS"] = "1"
os.environ["CURIOSITY_TACSL_CALIBRATION_DIR"] = str(
    ROOT / "experiments/sugar_reproduction/assets/official_tacsl/calibration"
)
os.environ["ISAACLAB_GROUND_PLANE_USD"] = str(
    ROOT / "SUGAR/descriptions/terrain/sugar_ground_plane.usda"
)
os.environ["ISAACLAB_USE_LOCAL_FRAME_MARKER"] = "1"
os.chdir(ROOT / "SUGAR")
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.controllers import (  # noqa: E402
    DifferentialIKController,
    DifferentialIKControllerCfg,
)

sys.path.insert(0, str(ROOT))
from scripts.sugar.native_tactile.slip import TactileSlipDetector  # noqa: E402
from scripts.sugar.native_tactile.universal import IsaacLabTacSLAdapter  # noqa: E402

from sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1 import (  # noqa: E402
    ANATOMICAL_WHOLE_HAND_PATCH_SPECS,
    anatomical_whole_hand_sensor_names,
)
from sugar_rl.assets.objects.tactile_objects import SdfUsdFileCfg  # noqa: E402
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg import (  # noqa: E402
    OfficialRefinerAnatomicalWholeHandTacSLAuditEnvCfg,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg import (  # noqa: E402
    OfficialRefinerAnatomicalWholeHandTacSLEnvCfg,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_tracker.pick_bottle_anatomical_whole_hand_tacsl_env_cfg import (  # noqa: E402
    PickBottleAnatomicalWholeHandTacSLEnvCfg,
)
from sugar_rl.utils.official_refiner_nominal_teacher import (  # noqa: E402
    FrozenOfficialRefinerTeacher,
)

sys.path.insert(0, str(ROOT / "SUGAR/scripts/sugar_rl"))
from official_refiner_anatomical_whole_hand_tacsl_audit_task_registration import (  # noqa: E402
    register_official_refiner_anatomical_whole_hand_tacsl_audit_task,
)


def cpu(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=dtype).numpy()


def load_official_pickbottle_tracker(device: torch.device) -> torch.nn.Module:
    """Load the released deterministic 510-D PickBottle Tracker actor."""

    payload = torch.load(
        PICKBOTTLE_TRACKER_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    actor = torch.nn.Sequential(
        torch.nn.Linear(510, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 29),
    ).to(device)
    actor.load_state_dict(
        {
            name.removeprefix("actor."): value
            for name, value in payload["model_state_dict"].items()
            if name.startswith("actor.")
        },
        strict=True,
    )
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def termination_after_grace(
    env,
    original_func,
    original_params: dict[str, object],
    grace_steps: int,
) -> torch.Tensor:
    return (env.episode_length_buf > grace_steps) & original_func(
        env, **original_params
    )


class FfmpegRgbWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self.path = path
        self.process = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")


def main() -> None:
    output_root.mkdir(parents=True)
    trace_path = output_root / "whole_hand_trace.npz"
    world_path = output_root / f"world_{args.object_kind}.mp4"
    summary_path = output_root / "summary.json"
    default_mass_kg = {
        "bottle": 0.75,
        "carrybox": 0.3023375868797302,
        "palm_grip": 0.5,
    }[args.object_kind]
    object_mass_kg = float(
        default_mass_kg if args.mass_kg is None else args.mass_kg
    )

    register_official_refiner_anatomical_whole_hand_tacsl_audit_task()
    # The audit scene adds three large raw ContactSensors for force-balance
    # figures. They are not part of the 54-patch native tactile signal, so the
    # force-only runtime uses the sensorized G1 scene without them.
    if args.object_kind == "bottle":
        cfg = PickBottleAnatomicalWholeHandTacSLEnvCfg()
        # The released Tracker actor consumes only the 510-D policy group.
        # The base training config also declares Refiner-only critic/teacher
        # groups that require a separate teacher-motion dataset; they are not
        # used during this frozen no-learning Tracker rollout.
        cfg.observations.critic = None
        cfg.observations.teacher = None
        task_id = PICKBOTTLE_TASK_ID
    else:
        cfg = (
            OfficialRefinerAnatomicalWholeHandTacSLEnvCfg()
            if args.force_only or args.object_kind == "palm_grip"
            else OfficialRefinerAnatomicalWholeHandTacSLAuditEnvCfg()
        )
        task_id = REFINER_TASK_ID
    actual_object_scale = (1.0, 1.0, 1.0)
    if args.object_kind == "carrybox" and args.object_scale is not None:
        actual_object_scale = tuple(args.object_scale)
        cfg.scene.obj.spawn.scale = actual_object_scale
    if args.object_kind == "palm_grip":
        object_path = (
            ROOT
            / "SUGAR/descriptions/objects/palm_fit_fixture/palm_grip_object.usda"
        )
        if not object_path.is_file():
            raise FileNotFoundError(object_path)
        object_scale = tuple(args.object_scale or (1.0, 1.0, 1.0))
        actual_object_scale = object_scale
        cfg.scene.obj = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obj",
            spawn=SdfUsdFileCfg(
                usd_path=str(object_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=2.0,
                    angular_damping=2.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=object_mass_kg),
                scale=object_scale,
                solid_outer_shell_only=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.object_kind == "bottle":
        motion_folder = ROOT / f"SUGAR/data/PickBottle/data_{args.motion_id:03d}"
        if not motion_folder.is_dir():
            raise FileNotFoundError(motion_folder)
        cfg.commands.motion.motion_folder = str(motion_folder)
        fixed_motion_index = 0
    else:
        cfg.commands.motion.motion_folder = "data/CarryBox"
        fixed_motion_index = args.motion_id
    cfg.commands.motion.init_with_ref = True
    cfg.commands.motion.start_init_env_ratio = 1.0
    cfg.commands.motion.pose_range = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    cfg.events.push_robot = None
    cfg.events.push_object = None
    if args.contact_friction is not None:
        cfg.events.obj_physics_material.params.update(
            static_friction_range=(args.contact_friction, args.contact_friction),
            dynamic_friction_range=(args.contact_friction, args.contact_friction),
            restitution_range=(0.0, 0.0),
            num_buckets=1,
        )
    if args.hold_reset_pose:
        cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        arm_drive = cfg.scene.robot.actuators["arms"]
        arm_drive.effort_limit_sim = 200.0
        arm_drive.stiffness = 1000.0
        arm_drive.damping = 50.0
    cfg.scene.left_hand_camera = None
    cfg.scene.right_hand_camera = None
    source_mass_kg = {
        "bottle": 0.75,
        "carrybox": 0.5,
        "palm_grip": 0.3023375868797302,
    }[args.object_kind]
    mass_scale = (
        1.0
        if args.object_kind == "palm_grip"
        else object_mass_kg / source_mass_kg
    )
    cfg.events.obj_mass.params["mass_distribution_params"] = (
        mass_scale,
        mass_scale,
    )
    for group_name in ("policy", "critic"):
        group = getattr(cfg.observations, group_name, None)
        if group is not None:
            group.enable_corruption = False
    for termination_name in (
        "anchor_ori",
        "ee_body_pos",
        "obj_pos",
        "obj_ori",
        "anchor_pos",
    ):
        term = getattr(cfg.terminations, termination_name, None)
        if term is None:
            continue
        original_func = term.func
        original_params = dict(term.params)
        term.func = termination_after_grace
        term.params = {
            "original_func": original_func,
            "original_params": original_params,
            "grace_steps": 2,
        }
    for sensor_name in anatomical_whole_hand_sensor_names():
        sensor_cfg = getattr(cfg.scene, sensor_name)
        sensor_cfg.update_period = float(cfg.sim.dt)
        if args.force_only or args.disable_optical:
            sensor_cfg.enable_camera_tactile = False
            sensor_cfg.camera_cfg = None
    if args.force_only:
        cfg.scene.world_camera = None

    env = gym.make(
        task_id,
        cfg=cfg,
        render_mode=None if args.force_only else "rgb_array",
    )
    writer: FfmpegRgbWriter | None = None
    original_reset_idx = None
    original_scene_update = None
    try:
        base_env = env.unwrapped
        command = base_env.command_manager.get_term("motion")
        official_motion_frames = int(
            command.motion.time_step_total_permotion[fixed_motion_index].item()
        )
        if args.start_step + args.max_steps >= official_motion_frames:
            raise RuntimeError(
                f"Requested frames {args.start_step} through "
                f"{args.start_step + args.max_steps - 1}, but env.step computes the "
                f"next observation and official motion {args.motion_id} has "
                f"{official_motion_frames} indexable robot frames"
            )

        def fixed_start(env_ids) -> None:
            ids = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.as_tensor(env_ids, dtype=torch.long, device=base_env.device)
            )
            command.motion_id[ids] = fixed_motion_index
            command.time_steps[ids] = args.start_step
            command._use_motion_data[ids] = True

        command._sample_init_state = fixed_start
        env.reset()
        # ManagerBasedRLEnv.reset() may complete without resampling this command
        # term when a previously constructed environment is reused.  Invoke the
        # official command reset path once explicitly so --start-step controls
        # the actual robot/object state, not only the command counters.
        reset_env_ids = torch.arange(
            base_env.num_envs, dtype=torch.long, device=base_env.device
        )
        command._resample_command(reset_env_ids)
        base_env.scene.write_data_to_sim()
        base_env.sim.forward()
        original_reset_idx = base_env._reset_idx
        base_env._reset_idx = lambda env_ids: None
        # This visualization task loads the official checkpoint directly and
        # deliberately skips the historical hash gate.  Architecture, weights,
        # observations, and deterministic inference remain unchanged.
        teacher = None
        tracker_actor = None
        replay_actions = None
        if args.object_kind == "bottle":
            tracker_actor = load_official_pickbottle_tracker(base_env.device)
        elif args.hold_reset_pose:
            pass
        elif args.action_trace is None:
            teacher = FrozenOfficialRefinerTeacher(
                base_env,
                REFINER_CHECKPOINT,
                expected_sha256=None,
            )
        else:
            with np.load(action_trace_path, allow_pickle=False) as replay:
                replay_actions = np.asarray(replay["applied_action"], dtype=np.float32)
            if replay_actions.ndim != 2 or replay_actions.shape[1] != 29:
                raise RuntimeError(f"Unexpected replay action shape: {replay_actions.shape}")
            if len(replay_actions) < args.max_steps:
                raise RuntimeError(
                    f"Action trace has {len(replay_actions)} rows, need {args.max_steps}"
                )
        robot = base_env.scene["robot"]
        obj = base_env.scene["obj"]
        action_term = base_env.action_manager.get_term("JointPositionAction")
        action_joint_names = tuple(action_term._joint_names)
        offset_action_indices = tuple(
            action_joint_names.index(f"{side}_{joint}_joint")
            for joint in ("wrist_yaw", "shoulder_pitch", "shoulder_roll")
            for side in SIDES
        )
        target_offset_rad = torch.as_tensor(
            (
                *args.wrist_yaw_target_offset_rad,
                *args.shoulder_pitch_target_offset_rad,
                *args.shoulder_roll_target_offset_rad,
            ),
            dtype=torch.float32,
            device=base_env.device,
        ).reshape(1, 6)
        action_offset = (
            target_offset_rad
            / action_term._scale[:, offset_action_indices]
        )
        right_arm_action_indices = tuple(
            index
            for index, name in enumerate(action_joint_names)
            if name.startswith("right_")
            and any(part in name for part in ("shoulder", "elbow", "wrist"))
        )
        if len(right_arm_action_indices) != 7:
            raise RuntimeError(
                "Expected seven right-arm action indices, got "
                f"{right_arm_action_indices} from {action_joint_names}"
            )
        bottom_support = None
        if args.bottom_support_open_loop:
            device = base_env.device
            dtype = robot.data.joint_pos.dtype
            palm_center_h = {
                "left": torch.tensor(
                    (0.040, -0.0149731754, -0.010), device=device, dtype=dtype
                ),
                "right": torch.tensor(
                    (0.040, 0.0149688583, -0.010), device=device, dtype=dtype
                ),
            }
            hand_body_ids = {}
            hand_jacobian_ids = {}
            arm_joint_ids = {}
            arm_action_indices = {}
            lift_controllers = {}
            for side in SIDES:
                body_ids = robot.find_bodies(f"{side}_rubber_hand")[0]
                if len(body_ids) != 1:
                    raise RuntimeError(
                        f"Expected one {side} rubber hand, got {body_ids}"
                    )
                body_id = int(body_ids[0])
                joint_names = [
                    f"{side}_shoulder_pitch_joint",
                    f"{side}_shoulder_roll_joint",
                    f"{side}_shoulder_yaw_joint",
                    f"{side}_elbow_joint",
                    f"{side}_wrist_roll_joint",
                    f"{side}_wrist_pitch_joint",
                    f"{side}_wrist_yaw_joint",
                ]
                ids = [int(value) for value in robot.find_joints(joint_names)[0]]
                if len(ids) != 7:
                    raise RuntimeError(f"Expected seven {side} arm joints, got {ids}")
                hand_body_ids[side] = body_id
                hand_jacobian_ids[side] = body_id - 1
                arm_joint_ids[side] = ids
                arm_action_indices[side] = [
                    action_joint_names.index(name) for name in joint_names
                ]
                lift_controllers[side] = DifferentialIKController(
                    DifferentialIKControllerCfg(
                        command_type="pose", use_relative_mode=False, ik_method="dls"
                    ),
                    num_envs=1,
                    device=device,
                )

            # Initial configuration only: move the box away and rotate the
            # two wrists to a reachable mixed-support pose derived from
            # official CarryBox motion 45, frame 380. The left palm faces
            # upward; the right palm faces horizontally toward the left.
            far_object_state = obj.data.root_state_w.clone()
            far_object_state[:, :3] = torch.tensor(
                (5.0, 5.0, 1.0), device=device, dtype=dtype
            )
            far_object_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(far_object_state)
            base_env.scene.write_data_to_sim()
            base_env.sim.forward()
            robot.update(0.0)

            # Wrist targets solved at official motion-45 frame 249 so the
            # Refiner keeps its original stable whole-body/arm trajectory:
            # left palm faces upward and right palm braces the side.
            wrist_setup_rad = {
                "left": (-0.527079, -0.001817, -1.570921),
                "right": (0.334558, -1.604436, -0.025278),
            }
            wrist_joint_ids = {}
            for side in SIDES:
                ids = [
                    int(value)
                    for value in robot.find_joints(
                        [
                            f"{side}_wrist_roll_joint",
                            f"{side}_wrist_pitch_joint",
                            f"{side}_wrist_yaw_joint",
                        ]
                    )[0]
                ]
                wrist_joint_ids[side] = ids
                target = torch.tensor(
                    wrist_setup_rad[side], device=device, dtype=dtype
                ).unsqueeze(0)
                robot.write_joint_state_to_sim(
                    target,
                    torch.zeros_like(target),
                    joint_ids=ids,
                )
            base_env.scene.write_data_to_sim()
            base_env.sim.forward()
            robot.update(0.0)

            hand_rotation_w = {
                side: math_utils.matrix_from_quat(
                    robot.data.body_quat_w[:, hand_body_ids[side]]
                )[0]
                for side in SIDES
            }
            palm_center_w = {
                side: robot.data.body_pos_w[0, hand_body_ids[side]].clone()
                + hand_rotation_w[side] @ palm_center_h[side]
                for side in SIDES
            }
            palm_outward_w = {
                "left": hand_rotation_w["left"]
                @ torch.tensor((0.0, -1.0, 0.0), device=device, dtype=dtype),
                "right": hand_rotation_w["right"]
                @ torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype),
            }
            up_w = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype)
            support_axis_w = palm_center_w["right"] - palm_center_w["left"]
            support_axis_w[2] = 0.0
            support_axis_w = support_axis_w / torch.linalg.vector_norm(support_axis_w)
            transverse_w = torch.linalg.cross(up_w, support_axis_w)
            box_pca_basis_b = torch.tensor(
                CARRYBOX_PCA_BASIS_B, device=device, dtype=dtype
            )
            desired_pca_frame_w = torch.stack(
                (support_axis_w, transverse_w, up_w), dim=1
            )
            box_rotation_w = desired_pca_frame_w @ box_pca_basis_b.T
            box_quaternion_w = math_utils.quat_from_matrix(
                box_rotation_w.unsqueeze(0)
            )[0]
            box_pca_center_w = palm_center_w["right"] - support_axis_w * (
                CARRYBOX_PCA0_MAX_M - 0.001
            )
            # Put both contact points near the PCA1 edge. The palm remains on
            # the bottom face while the four protruding distal finger regions
            # lie beyond that edge instead of intercepting the flat bottom.
            box_pca_center_w += (
                CARRYBOX_PCA1_EDGE_INSET_M * transverse_w
            )
            box_pca_center_w[2] = (
                palm_center_w["left"][2]
                - CARRYBOX_LOCAL_SUPPORT_PCA2_M
                - 0.003
            )
            box_root_position_w = box_pca_center_w - box_rotation_w @ torch.tensor(
                CARRYBOX_PCA_CENTER_B, device=device, dtype=dtype
            )
            object_state = obj.data.root_state_w.clone()
            object_state[0, :3] = box_root_position_w
            object_state[0, 3:7] = box_quaternion_w
            object_state[0, 7:] = 0.0
            obj.write_root_state_to_sim(object_state)
            base_env.scene.write_data_to_sim()
            base_env.sim.forward()
            robot.update(0.0)
            obj.update(0.0)
            for controller in lift_controllers.values():
                controller.reset()
            target_hand_position_w = {
                side: robot.data.body_pos_w[0, hand_body_ids[side]].clone()
                for side in SIDES
            }
            target_hand_quaternion_w = {
                side: robot.data.body_quat_w[0, hand_body_ids[side]].clone()
                for side in SIDES
            }
            bottom_support = {
                "hand_body_ids": hand_body_ids,
                "hand_jacobian_ids": hand_jacobian_ids,
                "arm_joint_ids": arm_joint_ids,
                "arm_action_indices": arm_action_indices,
                "controllers": lift_controllers,
                "target_hand_position_w": target_hand_position_w,
                "target_hand_quaternion_w": target_hand_quaternion_w,
                "setup_position_error_m": {"left": 0.0, "right": 0.0},
                "setup_rotation_error_rad": {"left": 0.0, "right": 0.0},
                "wrist_setup_rad": wrist_setup_rad,
                "wrist_joint_ids": wrist_joint_ids,
                "wrist_action_indices": {
                    side: [
                        action_joint_names.index(
                            f"{side}_wrist_{axis}_joint"
                        )
                        for axis in ("roll", "pitch", "yaw")
                    ]
                    for side in SIDES
                },
                "left_palm_target_w": palm_center_w["left"],
                "right_palm_target_w": palm_center_w["right"],
                "palm_outward_w": palm_outward_w,
                "box_pca_center_w": box_pca_center_w,
                "box_root_position_w": box_root_position_w,
                "setup_joint_position": robot.data.joint_pos.clone(),
                "initial_reference_object_position_w": command.obj_ref_pos_w[
                    0
                ].clone(),
                "settle_steps": 0,
                "lift_steps": 0,
                "lift_height_m": 0.0,
            }
            print(
                json.dumps(
                    {
                        "bottom_support_wrist_setup_rad": wrist_setup_rad,
                        "left_palm_target_w": cpu(palm_center_w["left"]).tolist(),
                        "right_palm_target_w": cpu(palm_center_w["right"]).tolist(),
                        "left_palm_outward_w": cpu(palm_outward_w["left"]).tolist(),
                        "right_palm_outward_w": cpu(palm_outward_w["right"]).tolist(),
                        "box_pca_center_w": cpu(box_pca_center_w).tolist(),
                    }
                ),
                flush=True,
            )
        hold_action = None
        if args.hold_reset_pose:
            reset_joint_position = robot.data.joint_pos.clone()
            reset_root_state = robot.data.root_state_w.clone()
            reset_root_state[:, 7:] = 0.0
            robot.write_joint_state_to_sim(
                reset_joint_position,
                torch.zeros_like(reset_joint_position),
            )
            robot.write_root_state_to_sim(reset_root_state)
            base_env.scene.write_data_to_sim()
            base_env.sim.forward()
            hold_action = (
                reset_joint_position[:, action_term._joint_ids] - action_term._offset
            ) / action_term._scale
        sensors = [base_env.scene[name] for name in anatomical_whole_hand_sensor_names()]
        if len(sensors) != 54:
            raise RuntimeError(f"Expected 54 sensors, found {len(sensors)}")
        common_patch_names = tuple(
            f"{side}_{patch}" for side in SIDES for patch in PATCHES
        )
        common_patch_sizes_m = [
            (spec.width_m, spec.length_m)
            for _side in SIDES
            for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS
        ]
        tactile_adapter = IsaacLabTacSLAdapter(
            common_patch_names,
            grid_shape=(20, 25),
            patch_size_m=common_patch_sizes_m,
        )
        # Save the exact post-reset geometry before the first policy/physics
        # step. This is the correct frame for constructing controlled contact
        # fixtures; a later rollout pose can already differ by millimeters.
        reset_adapter = IsaacLabTacSLAdapter(
            common_patch_names,
            grid_shape=(20, 25),
            patch_size_m=common_patch_sizes_m,
        )
        for sensor in sensors:
            sensor.update(0.0, force_recompute=True)
        reset_tactile = reset_adapter.update(
            {args.object_kind: [sensor.data for sensor in sensors]},
            timestamp_s=0.0,
            optical_timestamp_s=None,
        )
        np.savez_compressed(
            output_root / "reset_geometry.npz",
            taxel_position_w=cpu(reset_tactile.taxel_position_w_m[0]).reshape(
                2, 27, 20, 25, 3
            ),
            taxel_quaternion_w=cpu(reset_tactile.taxel_orientation_w_xyzw[0]).reshape(
                2, 27, 20, 25, 4
            ),
            penetration=cpu(reset_tactile.penetration_m[0]).reshape(
                2, 27, 20, 25
            ),
            normal_force=cpu(reset_tactile.normal_force_n[0]).reshape(
                2, 27, 20, 25
            ),
            signed_shear=cpu(reset_tactile.shear_force_xy_n[0]).reshape(
                2, 27, 20, 25, 2
            ),
            tactile_patch_size_m=np.asarray(common_patch_sizes_m, np.float32).reshape(
                2, 27, 2
            ),
            object_state_w=cpu(obj.data.root_state_w[0]),
            motion_frame=np.asarray(int(command.time_steps[0]), np.int64),
            patch_order=np.asarray(PATCHES),
            side_order=np.asarray(SIDES),
        )
        slip_detector = TactileSlipDetector(
            common_patch_names,
            friction_coefficient=0.5,
        )
        optical_baseline_rgb: list[np.ndarray] = []
        optical_baseline_depth: list[np.ndarray] = []
        if not args.force_only and not args.disable_optical:
            center_optical = [sensors[4], sensors[31]]
            for side, sensor in zip(SIDES, center_optical, strict=True):
                if int(torch.count_nonzero(sensor.data.tactile_normal_force).item()) != 0:
                    raise RuntimeError(
                        f"{side} center R15 baseline was requested under contact"
                    )
                camera = sensor._camera_sensor
                depth = None
                for _ in range(16):
                    base_env.sim.render()
                    camera.update(0.0, force_recompute=True)
                    depth = camera.data.output["distance_to_image_plane"]
                    if bool(torch.isfinite(depth).all().item()):
                        break
                if depth is None or not bool(torch.isfinite(depth).all().item()):
                    raise RuntimeError(f"{side} R15 no-contact baseline is non-finite")
                sensor.get_initial_render()
                zero_deformation = torch.zeros_like(depth[..., 0])
                optical_baseline_rgb.append(
                    cpu(
                        sensor._tactile_rgb_render.render(zero_deformation)[0],
                        torch.uint8,
                    )
                )
                optical_baseline_depth.append(cpu(depth[0]))
        world_camera = None if args.force_only else base_env.scene["world_camera"]
        if not args.force_only:
            writer = FfmpegRgbWriter(world_path, 1280, 720, args.fps)

        normal_rows: list[np.ndarray] = []
        shear_rows: list[np.ndarray] = []
        penetration_rows: list[np.ndarray] = []
        object_rows: list[np.ndarray] = []
        object_velocity_rows: list[np.ndarray] = []
        joint_rows: list[np.ndarray] = []
        joint_velocity_rows: list[np.ndarray] = []
        robot_root_state_rows: list[np.ndarray] = []
        robot_root_velocity_rows: list[np.ndarray] = []
        robot_body_state_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        position_rows: list[np.ndarray] = []
        quaternion_rows: list[np.ndarray] = []
        contact_normal_rows: list[np.ndarray] = []
        relative_velocity_rows: list[np.ndarray] = []
        optical_rgb_rows: list[np.ndarray] = []
        optical_depth_rows: list[np.ndarray] = []
        tactile_sequence_rows: list[int] = []
        tactile_timestamp_rows: list[float] = []
        tactile_dt_rows: list[float] = []
        optical_sequence_rows: list[int] = []
        optical_timestamp_rows: list[float] = []
        optical_dt_rows: list[float] = []
        slip_state_rows: list[np.ndarray] = []
        slip_normal_load_rows: list[np.ndarray] = []
        slip_tangential_load_rows: list[np.ndarray] = []
        slip_friction_utilization_rows: list[np.ndarray] = []
        slip_cop_speed_rows: list[np.ndarray] = []
        slip_footprint_rate_rows: list[np.ndarray] = []
        slip_normal_loss_rate_rows: list[np.ndarray] = []
        patch_box_force_rows: list[np.ndarray] = []
        patch_box_friction_rows: list[np.ndarray] = []
        robot_box_force_rows: list[np.ndarray] = []
        robot_box_friction_rows: list[np.ndarray] = []
        physics_object_state_rows: list[np.ndarray] = []
        physics_object_velocity_rows: list[np.ndarray] = []
        physics_robot_box_force_rows: list[np.ndarray] = []
        physics_robot_box_friction_rows: list[np.ndarray] = []
        physics_control_steps: list[int] = []
        physics_substeps: list[int] = []
        source_frames: list[int] = []
        terminated_rows: list[bool] = []
        truncated_rows: list[bool] = []
        termination_names = tuple(base_env.termination_manager.active_terms)
        termination_rows = {name: [] for name in termination_names}
        audit_contacts_available = (
            "all_robot_box_contact" in base_env.scene.sensors
            and "left_patch_box_contact" in base_env.scene.sensors
            and "right_patch_box_contact" in base_env.scene.sensors
        )
        all_robot_box_contact = (
            base_env.scene["all_robot_box_contact"]
            if audit_contacts_available
            else None
        )
        object_material_properties = cpu(obj.root_physx_view.get_material_properties())
        capture_state = {"enabled": False, "control_step": -1, "substep": 0}
        original_scene_update = base_env.scene.update

        def scene_update_with_physics_balance(dt: float) -> None:
            original_scene_update(dt)
            if not capture_state["enabled"]:
                return
            physics_object_state_rows.append(cpu(obj.data.root_state_w[0]))
            physics_object_velocity_rows.append(cpu(obj.data.root_vel_w[0]))
            if all_robot_box_contact is None:
                physics_robot_box_force_rows.append(np.empty((0, 3), np.float32))
                physics_robot_box_friction_rows.append(np.empty((0, 3), np.float32))
            else:
                contact = all_robot_box_contact.data
                if contact.force_matrix_w is None or contact.friction_forces_w is None:
                    raise RuntimeError("Substep robot/object force data is absent")
                physics_robot_box_force_rows.append(
                    cpu(contact.force_matrix_w[0, :, 0])
                )
                physics_robot_box_friction_rows.append(
                    cpu(contact.friction_forces_w[0, :, 0])
                )
            physics_control_steps.append(int(capture_state["control_step"]))
            physics_substeps.append(int(capture_state["substep"]))
            capture_state["substep"] += 1

        base_env.scene.update = scene_update_with_physics_balance

        def bottom_support_action(source_step: int) -> torch.Tensor:
            assert bottom_support is not None
            assert teacher is not None
            reference_offset_w = (
                command.obj_ref_pos_w[0]
                - bottom_support["initial_reference_object_position_w"]
            )
            root_pose_w = robot.data.root_pose_w
            base_rotation = math_utils.matrix_from_quat(
                math_utils.quat_inv(root_pose_w[:, 3:7])
            )
            desired_joint_position = bottom_support[
                "setup_joint_position"
            ].clone()
            for side in SIDES:
                target_position_w = (
                    bottom_support["target_hand_position_w"][side]
                    + reference_offset_w
                )
                target_quaternion_w = bottom_support[
                    "target_hand_quaternion_w"
                ][side]
                desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
                    root_pose_w[:, :3],
                    root_pose_w[:, 3:7],
                    target_position_w.unsqueeze(0),
                    target_quaternion_w.unsqueeze(0),
                )
                controller = bottom_support["controllers"][side]
                controller.set_command(
                    torch.cat((desired_pos_b, desired_quat_b), dim=1)
                )
                arm_ids = bottom_support["arm_joint_ids"][side]
                jacobian = robot.root_physx_view.get_jacobians()[
                    :,
                    bottom_support["hand_jacobian_ids"][side],
                    :,
                    arm_ids,
                ].clone()
                jacobian[:, :3, :] = torch.bmm(
                    base_rotation, jacobian[:, :3, :]
                )
                jacobian[:, 3:, :] = torch.bmm(
                    base_rotation, jacobian[:, 3:, :]
                )
                ee_pose_w = robot.data.body_pose_w[
                    :, bottom_support["hand_body_ids"][side]
                ]
                ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                    root_pose_w[:, :3],
                    root_pose_w[:, 3:7],
                    ee_pose_w[:, :3],
                    ee_pose_w[:, 3:7],
                )
                desired = controller.compute(
                    ee_pos_b,
                    ee_quat_b,
                    jacobian,
                    robot.data.joint_pos[:, arm_ids],
                )
                limits = robot.data.joint_pos_limits[0, arm_ids]
                desired_joint_position[:, arm_ids] = torch.clamp(
                    desired, limits[:, 0], limits[:, 1]
                )
            desired_action = (
                desired_joint_position[:, action_term._joint_ids]
                - action_term._offset
            ) / action_term._scale
            _, action = teacher.action()
            action = action.clone()
            for side in SIDES:
                indices = bottom_support["arm_action_indices"][side]
                action[:, indices] = desired_action[:, indices]
            return action

        for source_step in range(args.max_steps):
            source_frames.append(int(command.time_steps[0]))
            if bottom_support is not None:
                action = bottom_support_action(source_step)
            elif hold_action is not None:
                action = hold_action
            elif tracker_actor is not None:
                policy_observation = base_env.observation_manager.compute()["policy"]
                if tuple(policy_observation.shape) != (1, 510):
                    raise RuntimeError(
                        "Unexpected official PickBottle Tracker observation shape: "
                        f"{tuple(policy_observation.shape)}"
                    )
                action = tracker_actor(policy_observation)
            elif replay_actions is None:
                assert teacher is not None
                _, action = teacher.action()
            else:
                action = torch.as_tensor(
                    replay_actions[source_step : source_step + 1],
                    device=base_env.device,
                )
            if bool(torch.any(target_offset_rad != 0.0).item()):
                ramp = (
                    1.0
                    if args.joint_offset_ramp_steps == 0
                    else min(
                        (source_step + 1) / float(args.joint_offset_ramp_steps),
                        1.0,
                    )
                )
                action = action.clone()
                action[:, offset_action_indices] += ramp * action_offset
            if args.scenario == "failed_grasp" and source_step >= args.release_step:
                action = torch.zeros_like(action)
            elif (
                args.scenario == "failed_closure"
                and source_step >= args.closure_fault_step
            ):
                action = action.clone()
                action[:, right_arm_action_indices] = 0.0
            capture_state.update(
                enabled=True,
                control_step=source_step,
                substep=0,
            )
            try:
                _, _, terminated, truncated, _ = env.step(action)
            finally:
                capture_state["enabled"] = False
            if capture_state["substep"] != int(cfg.decimation):
                raise RuntimeError(
                    "Expected "
                    f"{cfg.decimation} physics samples, got "
                    f"{capture_state['substep']} at control step {source_step}"
                )
            if hold_action is not None:
                # This is a controlled contact-calibration sample: keep the
                # complete G1 at the declared reset pose after each physics
                # interval, then recompute the official TacSL SDF field at
                # that real geometry. No taxel value is generated or copied.
                robot.write_joint_state_to_sim(
                    reset_joint_position,
                    torch.zeros_like(reset_joint_position),
                )
                robot.write_root_state_to_sim(reset_root_state)
                base_env.scene.write_data_to_sim()
                base_env.sim.forward()
                for sensor in sensors:
                    sensor.update(0.0, force_recompute=True)

            tactile_frame = tactile_adapter.update(
                {args.object_kind: [sensor.data for sensor in sensors]},
                timestamp_s=(source_step + 1) * float(cfg.decimation * cfg.sim.dt),
                optical_timestamp_s=(
                    None
                    if args.force_only or args.disable_optical
                    else (source_step + 1) * float(cfg.decimation * cfg.sim.dt)
                ),
            )
            if not args.force_only and not args.disable_optical and tactile_frame.optical.clock is None:
                raise RuntimeError(
                    "Available official RGB/depth has no optical clock"
                )
            tactile_sequence_rows.append(tactile_frame.clock.sequence)
            tactile_timestamp_rows.append(tactile_frame.clock.timestamp_s)
            tactile_dt_rows.append(tactile_frame.clock.dt_s)
            if tactile_frame.optical.clock is not None:
                optical_sequence_rows.append(tactile_frame.optical.clock.sequence)
                optical_timestamp_rows.append(
                    tactile_frame.optical.clock.timestamp_s
                )
                optical_dt_rows.append(tactile_frame.optical.clock.dt_s)
            slip_evidence = slip_detector.update(tactile_frame)
            normal = cpu(tactile_frame.normal_force_n[0]).reshape(
                2, 27, 20, 25
            )
            shear = cpu(tactile_frame.shear_force_xy_n[0]).reshape(
                2, 27, 20, 25, 2
            )
            penetration = cpu(tactile_frame.penetration_m[0]).reshape(
                2, 27, 20, 25
            )
            position = cpu(tactile_frame.taxel_position_w_m[0]).reshape(
                2, 27, 20, 25, 3
            )
            quaternion = cpu(
                tactile_frame.taxel_orientation_w_xyzw[0]
            ).reshape(2, 27, 20, 25, 4)
            contact_normal = np.stack(
                [cpu(sensor.data.tactile_contact_normal_w[0]) for sensor in sensors]
            ).reshape(2, 27, 20, 25, 3)
            relative_velocity = np.stack(
                [
                    cpu(sensor.data.tactile_relative_tangential_velocity_w[0])
                    for sensor in sensors
                ]
            ).reshape(2, 27, 20, 25, 3)
            normal_rows.append(normal)
            shear_rows.append(shear)
            penetration_rows.append(penetration)
            position_rows.append(position)
            quaternion_rows.append(quaternion)
            contact_normal_rows.append(contact_normal)
            relative_velocity_rows.append(relative_velocity)
            slip_state_rows.append(slip_evidence.state.reshape(2, 27))
            slip_normal_load_rows.append(
                slip_evidence.normal_load_n.reshape(2, 27)
            )
            slip_tangential_load_rows.append(
                slip_evidence.tangential_load_n.reshape(2, 27)
            )
            slip_friction_utilization_rows.append(
                slip_evidence.friction_utilization.reshape(2, 27)
            )
            slip_cop_speed_rows.append(
                slip_evidence.center_of_pressure_speed_m_s.reshape(2, 27)
            )
            slip_footprint_rate_rows.append(
                slip_evidence.footprint_change_rate_s.reshape(2, 27)
            )
            slip_normal_loss_rate_rows.append(
                slip_evidence.normal_loss_rate_s.reshape(2, 27)
            )
            object_rows.append(cpu(obj.data.root_state_w[0]))
            object_velocity_rows.append(cpu(obj.data.root_vel_w[0]))
            joint_rows.append(cpu(robot.data.joint_pos[0]))
            joint_velocity_rows.append(cpu(robot.data.joint_vel[0]))
            robot_root_state_rows.append(cpu(robot.data.root_state_w[0]))
            robot_root_velocity_rows.append(cpu(robot.data.root_vel_w[0]))
            robot_body_state_rows.append(cpu(robot.data.body_state_w[0]))
            action_rows.append(cpu(action[0]))
            if not args.force_only and not args.disable_optical:
                optical_rgb_rows.append(
                    np.stack(
                        [
                            cpu(
                                sensors[hand_index * 27 + 4].data.tactile_rgb_image[0],
                                torch.uint8,
                            )
                            for hand_index in range(2)
                        ]
                    )
                )
                optical_depth_rows.append(
                    np.stack(
                        [
                            cpu(
                                sensors[hand_index * 27 + 4].data.tactile_depth_image[0]
                            )
                            for hand_index in range(2)
                        ]
                    )
                )
            if audit_contacts_available:
                robot_contact = base_env.scene["all_robot_box_contact"].data
                if robot_contact.force_matrix_w is None:
                    raise RuntimeError("All-robot object-contact force matrix is absent")
                robot_box_force_rows.append(cpu(robot_contact.force_matrix_w[0, :, 0]))
                if robot_contact.friction_forces_w is None:
                    raise RuntimeError("All-robot object friction-force matrix is absent")
                robot_box_friction_rows.append(
                    cpu(robot_contact.friction_forces_w[0, :, 0])
                )
                patch_force_by_hand = []
                patch_friction_by_hand = []
                for side in SIDES:
                    patch_contact = base_env.scene[f"{side}_patch_box_contact"].data
                    if patch_contact.force_matrix_w is None:
                        raise RuntimeError(f"{side} patch object-contact force matrix is absent")
                    if patch_contact.friction_forces_w is None:
                        raise RuntimeError(
                            f"{side} patch object friction-force matrix is absent"
                        )
                    patch_force_by_hand.append(
                        cpu(patch_contact.force_matrix_w[0, :, 0])
                    )
                    patch_friction_by_hand.append(
                        cpu(patch_contact.friction_forces_w[0, :, 0])
                    )
                patch_box_force_rows.append(np.stack(patch_force_by_hand))
                patch_box_friction_rows.append(np.stack(patch_friction_by_hand))
            else:
                robot_box_force_rows.append(np.empty((0, 3), np.float32))
                robot_box_friction_rows.append(np.empty((0, 3), np.float32))
                patch_box_force_rows.append(np.empty((2, 0, 3), np.float32))
                patch_box_friction_rows.append(np.empty((2, 0, 3), np.float32))
            terminated_rows.append(bool(terminated[0].item()))
            truncated_rows.append(bool(truncated[0].item()))
            for name in termination_names:
                termination_rows[name].append(
                    bool(base_env.termination_manager.get_term(name)[0].item())
                )

            if world_camera is not None and writer is not None:
                rgb = cpu(world_camera.data.output["rgb"][0, ..., :3], torch.uint8)
                writer.append(rgb)
            if source_step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "source_step": source_step,
                            "object_z_m": float(object_rows[-1][2]),
                            "active_taxels_left": int(np.count_nonzero(normal[0])),
                            "active_taxels_right": int(np.count_nonzero(normal[1])),
                        }
                    ),
                    flush=True,
                )
            if (
                args.object_kind == "carrybox"
                and args.scenario != "failed_closure"
                and not args.continue_after_termination
                and (terminated_rows[-1] or truncated_rows[-1])
            ):
                break

        if writer is not None:
            writer.close()
            writer = None
        normal_array = np.stack(normal_rows).astype(np.float32)
        shear_array = np.stack(shear_rows).astype(np.float32)
        penetration_array = np.stack(penetration_rows).astype(np.float32)
        object_array = np.stack(object_rows).astype(np.float32)
        object_velocity_array = np.stack(object_velocity_rows).astype(np.float32)
        active = np.count_nonzero(penetration_array > 0.0, axis=(-1, -2))
        bilateral = np.all(np.any(active > 0, axis=-1), axis=-1)
        active_palm_patches = np.count_nonzero(active[:, :, :12] > 0, axis=-1)
        bilateral_palm_contact = np.all(active_palm_patches > 0, axis=-1)
        bilateral_six_palm_patches = np.all(active_palm_patches >= 6, axis=-1)
        relative_lift = object_array[:, 2] - object_array[0, 2]
        termination_arrays = {
            f"termination_{name}": np.asarray(values, dtype=np.bool_)
            for name, values in termination_rows.items()
        }
        np.savez_compressed(
            trace_path,
            normal_force=normal_array,
            signed_shear=shear_array,
            penetration=penetration_array,
            taxel_position_w=np.stack(position_rows).astype(np.float32),
            taxel_quaternion_w=np.stack(quaternion_rows).astype(np.float32),
            tactile_contact_normal_w=np.stack(contact_normal_rows).astype(np.float32),
            tactile_relative_tangential_velocity_w=np.stack(
                relative_velocity_rows
            ).astype(np.float32),
            tactile_only_slip_state=np.stack(slip_state_rows).astype(np.int8),
            tactile_only_slip_normal_load_n=np.stack(slip_normal_load_rows).astype(
                np.float32
            ),
            tactile_only_slip_tangential_load_n=np.stack(
                slip_tangential_load_rows
            ).astype(np.float32),
            tactile_only_slip_friction_utilization=np.stack(
                slip_friction_utilization_rows
            ).astype(np.float32),
            tactile_only_slip_cop_speed_m_s=np.stack(slip_cop_speed_rows).astype(
                np.float32
            ),
            tactile_only_slip_footprint_rate_s=np.stack(
                slip_footprint_rate_rows
            ).astype(np.float32),
            tactile_only_slip_normal_loss_rate_s=np.stack(
                slip_normal_loss_rate_rows
            ).astype(np.float32),
            optical_rgb=(
                np.stack(optical_rgb_rows).astype(np.uint8)
                if optical_rgb_rows
                else np.empty((len(normal_array), 2, 0), dtype=np.uint8)
            ),
            optical_depth=(
                np.stack(optical_depth_rows).astype(np.float32)
                if optical_depth_rows
                else np.empty((len(normal_array), 2, 0), dtype=np.float32)
            ),
            optical_baseline_rgb=(
                np.stack(optical_baseline_rgb).astype(np.uint8)
                if optical_baseline_rgb
                else np.empty((2, 0), dtype=np.uint8)
            ),
            optical_baseline_depth=(
                np.stack(optical_baseline_depth).astype(np.float32)
                if optical_baseline_depth
                else np.empty((2, 0), dtype=np.float32)
            ),
            tactile_sequence=np.asarray(tactile_sequence_rows, dtype=np.int64),
            tactile_timestamp_s=np.asarray(
                tactile_timestamp_rows, dtype=np.float64
            ),
            tactile_dt_s=np.asarray(tactile_dt_rows, dtype=np.float64),
            optical_sequence=np.asarray(optical_sequence_rows, dtype=np.int64),
            optical_timestamp_s=np.asarray(
                optical_timestamp_rows, dtype=np.float64
            ),
            optical_dt_s=np.asarray(optical_dt_rows, dtype=np.float64),
            active_taxels=active.astype(np.int32),
            bilateral_contact=bilateral.astype(np.bool_),
            object_state_w=object_array,
            object_velocity_w=object_velocity_array,
            patch_box_force_w=np.stack(patch_box_force_rows).astype(np.float32),
            patch_box_friction_force_w=np.stack(patch_box_friction_rows).astype(
                np.float32
            ),
            robot_box_force_w=np.stack(robot_box_force_rows).astype(np.float32),
            robot_box_friction_force_w=np.stack(robot_box_friction_rows).astype(
                np.float32
            ),
            robot_box_force_body_names=np.asarray(
                base_env.scene["all_robot_box_contact"].body_names
                if audit_contacts_available
                else []
            ),
            robot_joint_position=np.stack(joint_rows).astype(np.float32),
            robot_joint_velocity=np.stack(joint_velocity_rows).astype(np.float32),
            robot_joint_names=np.asarray(robot.joint_names),
            robot_root_state_w=np.stack(robot_root_state_rows).astype(np.float32),
            robot_root_velocity_w=np.stack(robot_root_velocity_rows).astype(np.float32),
            robot_body_state_w=np.stack(robot_body_state_rows).astype(np.float32),
            robot_body_names=np.asarray(robot.body_names),
            applied_action=np.stack(action_rows).astype(np.float32),
            source_step=np.arange(len(normal_array), dtype=np.int32),
            motion_frame_before_action=np.asarray(source_frames, dtype=np.int32),
            terminated=np.asarray(terminated_rows, dtype=np.bool_),
            truncated=np.asarray(truncated_rows, dtype=np.bool_),
            patch_order=np.asarray(PATCHES),
            side_order=np.asarray(SIDES),
            sensor_names=np.asarray(anatomical_whole_hand_sensor_names()),
            tactile_patch_size_m=np.asarray(common_patch_sizes_m, dtype=np.float32).reshape(
                2, 27, 2
            ),
            gravity_w=np.asarray(cfg.sim.gravity, dtype=np.float32),
            physics_dt_s=np.asarray(cfg.sim.dt, dtype=np.float64),
            control_dt_s=np.asarray(cfg.decimation * cfg.sim.dt, dtype=np.float64),
            physics_object_state_w=np.stack(physics_object_state_rows).astype(
                np.float32
            ),
            physics_object_velocity_w=np.stack(
                physics_object_velocity_rows
            ).astype(np.float32),
            physics_robot_box_force_w=np.stack(
                physics_robot_box_force_rows
            ).astype(np.float32),
            physics_robot_box_friction_force_w=np.stack(
                physics_robot_box_friction_rows
            ).astype(np.float32),
            physics_control_step=np.asarray(physics_control_steps, dtype=np.int32),
            physics_substep=np.asarray(physics_substeps, dtype=np.int8),
            object_material_properties=object_material_properties.astype(np.float32),
            **termination_arrays,
        )
        reasons = [
            name for name, values in termination_rows.items() if any(values)
        ]
        lifted_and_bilateral = (relative_lift >= 0.20) & bilateral
        summary = {
            "schema": "sugar_g1_anatomical27_object_native_tactile_v1",
            "scenario": args.scenario,
            "object_kind": args.object_kind,
            "object_scale": list(actual_object_scale),
            "wrist_yaw_target_offset_rad": list(
                map(float, args.wrist_yaw_target_offset_rad)
            ),
            "shoulder_pitch_target_offset_rad": list(
                map(float, args.shoulder_pitch_target_offset_rad)
            ),
            "shoulder_roll_target_offset_rad": list(
                map(float, args.shoulder_roll_target_offset_rad)
            ),
            "joint_offset_ramp_steps": int(args.joint_offset_ramp_steps),
            "action_source": (
                "official_refiner_body_plus_bilateral_isaaclab_dls_arms"
                if bottom_support is not None
                else (
                    "controlled_reset_pose_hold"
                    if hold_action is not None
                    else (
                        "released_official_pickbottle_tracker"
                        if tracker_actor is not None
                        else (
                            "frozen_official_refiner"
                            if args.action_trace is None
                            else str(action_trace_path)
                        )
                    )
                )
            ),
            "bottom_support_open_loop": args.bottom_support_open_loop,
            "bottom_support_setup": (
                None
                if bottom_support is None
                else {
                    "left_palm_target_w_m": cpu(
                        bottom_support["left_palm_target_w"]
                    ).tolist(),
                    "right_palm_target_w_m": cpu(
                        bottom_support["right_palm_target_w"]
                    ).tolist(),
                    "box_pca_center_w_m": cpu(
                        bottom_support["box_pca_center_w"]
                    ).tolist(),
                    "box_root_position_w_m": cpu(
                        bottom_support["box_root_position_w"]
                    ).tolist(),
                    "ik_position_error_m": bottom_support[
                        "setup_position_error_m"
                    ],
                    "ik_rotation_error_rad": bottom_support[
                        "setup_rotation_error_rad"
                    ],
                    "settle_steps": bottom_support["settle_steps"],
                    "lift_steps": bottom_support["lift_steps"],
                    "commanded_lift_m": bottom_support["lift_height_m"],
                }
            ),
            "host": HOST,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "task_id": task_id,
            "motion_id": args.motion_id,
            "source_start_step": args.start_step,
            "seed": args.seed,
            "source_frames": int(len(normal_array)),
            "release_step": (
                args.release_step if args.scenario == "failed_grasp" else None
            ),
            "closure_fault_step": (
                args.closure_fault_step
                if args.scenario == "failed_closure"
                else None
            ),
            "closure_fault_action_joint_names": (
                [action_joint_names[index] for index in right_arm_action_indices]
                if args.scenario == "failed_closure"
                else None
            ),
            "continued_after_task_termination_for_visualization": (
                args.continue_after_termination
                or args.scenario == "failed_closure"
                or args.object_kind != "carrybox"
            ),
            "box_mass_requested_kg": object_mass_kg,
            "box_mass_readback_kg": float(
                cpu(obj.root_physx_view.get_masses())[0].sum()
            ),
            "object_mass_requested_kg": object_mass_kg,
            "object_mass_readback_kg": float(
                cpu(obj.root_physx_view.get_masses())[0].sum()
            ),
            "physx_contact_audit_available": audit_contacts_available,
            "normal_shape": list(normal_array.shape),
            "shear_shape": list(shear_array.shape),
            "common_tactile_backend": "isaaclab_tacsl",
            "common_tactile_patch_count": len(common_patch_names),
            "common_tactile_patch_size_shape": [2, 27, 2],
            "common_tactile_patch_size_order": "row/local-X then column/local-Y",
            "tactile_only_slip_state_shape": list(
                np.stack(slip_state_rows).shape
            ),
            "tactile_only_slip_inputs": [
                "signed_local_z_force",
                "signed_local_xy_shear",
                "penetration",
                "timestamps",
            ],
            "taxel_position_shape": [len(normal_array), 2, 27, 20, 25, 3],
            "taxel_quaternion_shape": [len(normal_array), 2, 27, 20, 25, 4],
            "taxel_quaternion_order": "xyzw (official IsaacLab wxyz reordered by common adapter)",
            "tactile_clock_fields": [
                "tactile_sequence",
                "tactile_timestamp_s",
                "tactile_dt_s",
            ],
            "optical_clock_fields": [
                "optical_sequence",
                "optical_timestamp_s",
                "optical_dt_s",
            ],
            "optical_available": not args.force_only and not args.disable_optical,
            "optical_unavailable_reason": (
                "Force-only collection explicitly disabled the optical cameras."
                if args.force_only
                else (
                    "Controlled force visualization disabled optical cameras."
                    if args.disable_optical
                    else None
                )
            ),
            "optical_rgb_shape": (
                list(np.stack(optical_rgb_rows).shape)
                if optical_rgb_rows
                else [len(normal_array), 2, 0]
            ),
            "optical_depth_shape": (
                list(np.stack(optical_depth_rows).shape)
                if optical_depth_rows
                else [len(normal_array), 2, 0]
            ),
            "optical_baseline_rgb_shape": (
                list(np.stack(optical_baseline_rgb).shape)
                if optical_baseline_rgb
                else [2, 0]
            ),
            "optical_baseline_depth_shape": (
                list(np.stack(optical_baseline_depth).shape)
                if optical_baseline_depth
                else [2, 0]
            ),
            "patch_box_force_shape": list(np.stack(patch_box_force_rows).shape),
            "patch_box_friction_force_shape": list(
                np.stack(patch_box_friction_rows).shape
            ),
            "robot_box_force_shape": list(np.stack(robot_box_force_rows).shape),
            "robot_box_friction_force_shape": list(
                np.stack(robot_box_friction_rows).shape
            ),
            "robot_root_state_shape": list(np.stack(robot_root_state_rows).shape),
            "robot_body_state_shape": list(np.stack(robot_body_state_rows).shape),
            "robot_state_quaternion_order": "wxyz (native IsaacLab state)",
            "physics_substeps_per_control_step": int(cfg.decimation),
            "physics_robot_box_force_shape": list(
                np.stack(physics_robot_box_force_rows).shape
            ),
            "maximum_relative_lift_m": float(relative_lift.max()),
            "final_relative_lift_m": float(relative_lift[-1]),
            "bilateral_contact_frames": int(bilateral.sum()),
            "lifted_bilateral_frames": int(lifted_and_bilateral.sum()),
            "maximum_active_taxels_left": int(active[:, 0].sum(axis=-1).max()),
            "maximum_active_taxels_right": int(active[:, 1].sum(axis=-1).max()),
            "maximum_active_palm_patches_left": int(
                active_palm_patches[:, 0].max()
            ),
            "maximum_active_palm_patches_right": int(
                active_palm_patches[:, 1].max()
            ),
            "bilateral_palm_contact_frames": int(bilateral_palm_contact.sum()),
            "bilateral_six_or_more_palm_patch_frames": int(
                bilateral_six_palm_patches.sum()
            ),
            "termination_reasons": reasons,
            "world_video": None if args.force_only else str(world_path),
            "trace": str(trace_path),
            "claim_boundary": (
                "No-learning native-sensor behavior trace. Final success/failure "
                "and visible correspondence require the paired video review."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if original_scene_update is not None:
            base_env.scene.update = original_scene_update
        if original_reset_idx is not None:
            base_env._reset_idx = original_reset_idx
        if writer is not None:
            writer.process.kill()
            writer.process.wait()
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
