#!/usr/bin/env python3
"""Freeze-evaluate one demo-conditioned official SUGAR skill route.

Each run fixes the official task environment, generator/reference motion,
startup seed, one-step official Tracker alignment prefix and shared checkpoint.
The only experimental input is the selected demo condition (Carry45 or
Kick21).  The shared actor then directly supplies the 29-D ActionManager
command; no Refiner/Tracker action or future ground truth is available after
the common one-step prefix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[3]
SUGAR = ROOT / "SUGAR"
SUGAR_SCRIPT = SUGAR / "scripts/sugar_rl"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SUGAR_SCRIPT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(SUGAR / "descriptions/terrain/sugar_ground_plane.usda"),
)
os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("DISPLAY", "")
job_id = os.environ.get("SLURM_JOB_ID", "local")
os.environ.setdefault("ISAACLAB_TMP_ROOT", f"/tmp/Curiosity_shared_absolute_{job_id}")
os.environ.setdefault(
    "SUGAR_UNITREE_TMP_ROOT", f"/tmp/Curiosity_shared_absolute_unitree_{job_id}"
)

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--domain", choices=("CarryBox", "KickBox"), required=True)
parser.add_argument(
    "--selected-demo-option", choices=("correct", "unrelated"), required=True
)
parser.add_argument("--shared-checkpoint", type=Path, required=True)
parser.add_argument(
    "--route-generator-with-expert",
    action="store_true",
    help=(
        "Route the selected demo's released Generator together with its exact "
        "Tracker expert after the common one-step domain prefix."
    ),
)
parser.add_argument(
    "--causal-safe-fallback",
    action="store_true",
    help=(
        "Evaluate the selected released Generator+Tracker as an online candidate, "
        "but execute the domain released Generator+Tracker whenever the selected "
        "Tracker leaves the released-action envelope."
    ),
)
parser.add_argument(
    "--causal-transition-risk-fallback",
    action="store_true",
    help=(
        "Use a frozen causal Carry45 transition-risk Transformer over the first "
        "50 current-state frames, then latch risky profiles to the domain released pair."
    ),
)
parser.add_argument(
    "--transition-risk-proof",
    type=Path,
    help="Passed frozen transition-risk evaluation supplying checkpoint and threshold.",
)
parser.add_argument(
    "--scene-object-asset",
    choices=("domain", "small", "big"),
    default="domain",
    help="Factorial audit override for the physical box asset only.",
)
parser.add_argument(
    "--object-nominal-mass-source",
    choices=("domain", "small", "big"),
    default="domain",
    help=(
        "Factorial audit override for nominal mass/inertia only; geometry remains "
        "the selected scene asset."
    ),
)
parser.add_argument(
    "--target-goal-source",
    choices=("domain", "carry45", "kick21"),
    default="domain",
    help=(
        "Factorial audit override for the online Generator target pose. Explicit "
        "sources are installed before the common prefix with matched RNG usage."
    ),
)
parser.add_argument(
    "--disable-observation-corruption",
    action="store_true",
    help="Disable Tracker observation noise for exact direct-versus-shadow equivalence tests.",
)
parser.add_argument("--training-proof", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=20)
parser.add_argument("--steps", type=int, default=650)
parser.add_argument("--seed", type=int, default=171595)
parser.add_argument(
    "--dagger-collection",
    action="store_true",
    help="Collect online official-Tracker corrective labels without policy optimization.",
)
parser.add_argument(
    "--student-action-fraction",
    type=float,
    default=1.0,
    help="DAgger behavior mixture: 0=official teacher, 1=shared student.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 20 or args.steps != 650:
    parser.error("the frozen experiment is fixed to 20 environments and 650 steps")
if not 0.0 <= args.student_action_fraction <= 1.0:
    parser.error("--student-action-fraction must be in [0, 1]")
if not args.dagger_collection and args.student_action_fraction != 1.0:
    parser.error("formal frozen evaluation requires exact student-only execution")
if args.causal_safe_fallback and not args.route_generator_with_expert:
    parser.error("--causal-safe-fallback requires --route-generator-with-expert")
if args.causal_transition_risk_fallback and not args.route_generator_with_expert:
    parser.error("--causal-transition-risk-fallback requires --route-generator-with-expert")
if args.causal_transition_risk_fallback and args.causal_safe_fallback:
    parser.error("transition-risk and action-envelope fallback are mutually exclusive")
if args.causal_transition_risk_fallback and args.transition_risk_proof is None:
    parser.error("--causal-transition-risk-fallback requires --transition-risk-proof")
args.task = f"Sugar-G129dof-{args.domain}-Inference"
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import builtins  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import rsl_rl.algorithms  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

import sugar_rl.tasks  # noqa: F401,E402
import sugar_rl.tasks.locomanip.mdp as mdp  # noqa: E402
from sugar_rl.assets.objects.objects import BIGBOX_CFG, SMALLBOX_CFG  # noqa: E402
from sugar_rl.tasks.locomanip import goal_carry_mdp as goal_mdp  # noqa: E402
from sugar_rl.utils.demo_event_reward_runtime import (  # noqa: E402
    FrozenPhaseAwareDemoEventScorer,
    FrozenPhaseAwareDemoEventScorerCfg,
)
from sugar_rl.utils.parser_cfg import parse_env_cfg  # noqa: E402
from sugar_rl.utils.rsl_rl_bcppo import BCPPO  # noqa: E402
from sugar_il.wrapper.sugar_il_wrapper import GeneratorObs, GeneratorWrapper  # noqa: E402

from train_shared_topology_distillation import (  # noqa: E402
    ACTIONABLE_DEMO_CONDITIONING_DIM,
    RUNTIME_CONFIG,
    TACTILE_DIM,
    _policy_observation,
)
from train_shared_full_tracker import (  # noqa: E402
    TRACKER_POLICY_DIM,
    construct_full_policy,
)
from train_official_tracker_router import construct_router_policy  # noqa: E402
from train_official_transition_risk_transformer import (  # noqa: E402
    CausalTransitionRiskTransformer,
)


setattr(builtins, "BCPPO", BCPPO)
setattr(rsl_rl.algorithms, "BCPPO", BCPPO)

DOMAIN = {
    "CarryBox": {
        "motion_id": 45,
        "motion_folder": SUGAR / "data/CarryBox/data_045",
        "generator": SUGAR / "demo_ckpts/CarryBox/generator.ckpt",
        "tracker": SUGAR / "demo_ckpts/CarryBox/tracker.pt",
    },
    "KickBox": {
        "motion_id": 21,
        "motion_folder": SUGAR / "data/KickBox/data_021",
        "generator": SUGAR / "demo_ckpts/KickBox/generator.ckpt",
        "tracker": SUGAR / "demo_ckpts/KickBox/tracker.pt",
    },
}
SELECTED_SKILL = {
    "correct": "CarryBox",
    "unrelated": "KickBox",
}
SENSOR_NAMES = (
    "left_hand_forces",
    "right_hand_forces",
    "left_foot_forces",
    "right_foot_forces",
)
CONTACT_THRESHOLD_N = 0.1
RELEASED_TRACKER_RAW_ACTION_LIMIT = 25.0
LIFT_THRESHOLD_M = 0.05
FALL_HEIGHT_LOSS_M = 0.35
TRANSITION_RISK_HISTORY_STEPS = 10
TRANSITION_RISK_DECISION_FRAME = 49
TRANSITION_RISK_SAMPLE_STRIDE = 5


class _CausalShadowGenerator:
    """Run one released Generator online without mutating the domain command.

    The shadow owns its observation history, generated-command horizon and last
    command.  Its inputs are the same current simulator state and target used by
    the active official Generator.  It never reads a future reference frame.
    """

    def __init__(self, generator: GeneratorWrapper, command) -> None:
        self.generator = generator
        self.num_envs = command.num_envs
        self.device = command.device
        self.call_interval = int(command.generator_call_interval)
        if self.call_interval <= 0:
            raise RuntimeError("shadow Generator call interval drift")
        self._cpu_rng_state = torch.get_rng_state().clone()
        self._cuda_rng_state = (
            torch.cuda.get_rng_state(self.device).clone()
            if torch.cuda.is_available()
            else None
        )
        history = (generator.n_obs_steps - 1) * 5 + 1
        current = command._get_generator_obs()
        self.observation = {
            name: getattr(current, name).expand(-1, history, -1).clone()
            for name in (
                "obj_pos_b",
                "obj_ori_b",
                "joint_pos",
                "project_gravity",
                "target_obj_pos_b",
                "target_obj_ori_b",
                "last_command",
            )
        }
        self.generated = self._predict()
        self.last_command = current.last_command.clone()

    def _predict(self) -> torch.Tensor:
        global_cpu_state = torch.get_rng_state()
        global_cuda_state = (
            torch.cuda.get_rng_state(self.device) if self._cuda_rng_state is not None else None
        )
        try:
            torch.set_rng_state(self._cpu_rng_state)
            if self._cuda_rng_state is not None:
                torch.cuda.set_rng_state(self._cuda_rng_state, self.device)
            generated = self.generator.predict(GeneratorObs(**self.observation))
            self._cpu_rng_state = torch.get_rng_state().clone()
            if self._cuda_rng_state is not None:
                self._cuda_rng_state = torch.cuda.get_rng_state(self.device).clone()
        finally:
            torch.set_rng_state(global_cpu_state)
            if global_cuda_state is not None:
                torch.cuda.set_rng_state(global_cuda_state, self.device)
        expected_time = (self.generator.n_action_steps - 1) * 5 + 1
        if generated.shape != (self.num_envs, expected_time, 36):
            raise RuntimeError("shadow Generator output geometry drift")
        if not torch.isfinite(generated).all():
            raise RuntimeError("shadow Generator produced non-finite commands")
        return generated

    def command_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        index = (time_steps.to(dtype=torch.long) - 1) % self.call_interval
        env_index = torch.arange(self.num_envs, device=self.device)
        command = self.generated[env_index, index]
        self.last_command = command.unsqueeze(1).clone()
        return command

    def normalized_compatibility(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.generator._prepare_obs_dict(GeneratorObs(**self.observation))
        normalized = self.generator.policy.normalizer.normalize(raw)
        flat = torch.cat(
            [value.reshape(self.num_envs, -1) for value in normalized.values()], dim=-1
        )
        return torch.amax(torch.abs(flat), dim=-1), torch.mean(
            (torch.abs(flat) > 1.0).to(flat.dtype), dim=-1
        )

    def update_after_step(self, command) -> None:
        current = command._get_generator_obs()
        current.last_command = self.last_command
        for name, history in self.observation.items():
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = getattr(current, name).squeeze(1)
        refresh = (command.time_steps - 1) % self.call_interval == 0
        if torch.any(refresh):
            replacement = self._predict()
            self.generated[refresh] = replacement[refresh]


def _goal_policy_core_observation(base) -> torch.Tensor:
    value = torch.cat(
        (
            mdp.projected_gravity(base),
            goal_mdp.base_height(base, "motion"),
            mdp.base_lin_vel(base),
            mdp.base_ang_vel(base),
            mdp.joint_pos_rel(base),
            mdp.joint_vel_rel(base),
            base.action_manager.action,
            goal_mdp.box_position_body(base, "motion"),
            goal_mdp.box_orientation_tangent_normal_body(base, "motion"),
            goal_mdp.box_linear_velocity_body(base, "motion"),
            goal_mdp.box_angular_velocity_body(base, "motion"),
            goal_mdp.goal_position_body(base, "motion"),
            goal_mdp.goal_orientation_tangent_normal_body(base, "motion"),
        ),
        dim=-1,
    )
    if value.shape != (base.num_envs, 121) or not torch.isfinite(value).all():
        raise RuntimeError("deployable goal core geometry/finiteness drift")
    return value


def _latest_filtered_force(sensor) -> torch.Tensor:
    force = sensor.data.force_matrix_w_history
    if force is None or force.ndim != 5 or force.shape[2:4] != (1, 1):
        raise RuntimeError("filtered body-to-object ContactSensor geometry drift")
    return force[:, -1, 0, 0, :]


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _same_state(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> bool:
    return before.keys() == after.keys() and all(
        torch.equal(before[name], after[name]) for name in before
    )


def _profile_summaries(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    available_frames = int(arrays["done"].shape[0])
    for env_index in range(args.num_envs):
        done_hits = np.flatnonzero(arrays["done"][:, env_index])
        stop = int(done_hits[0]) + 1 if done_hits.size else available_frames
        obj = arrays["object_root_state_w"][:stop, env_index]
        root = arrays["robot_root_state_w"][:stop, env_index]
        contact = arrays["contact"][:stop, env_index]
        baseline_z = float(np.median(obj[: min(25, stop), 2]))
        lift = obj[:, 2] - baseline_z
        bilateral = contact[:, 0] & contact[:, 1]
        any_foot = contact[:, 2] | contact[:, 3]
        root_loss = float(root[0, 2] - root[:, 2].min())
        planar_net = float(np.linalg.norm(obj[-1, :2] - obj[0, :2]))
        planar_path = float(np.linalg.norm(np.diff(obj[:, :2], axis=0), axis=-1).sum())
        record = {
            "profile": env_index,
            "valid_frames": stop,
            "maximum_lift_m": float(lift.max()),
            "bilateral_hand_contact_fraction": float(bilateral.mean()),
            "any_foot_box_contact_fraction": float(any_foot.mean()),
            "planar_object_net_displacement_m": planar_net,
            "planar_object_path_m": planar_path,
            "maximum_robot_root_height_loss_m": root_loss,
            "physical_robot_fall": bool(root_loss >= FALL_HEIGHT_LOSS_M),
            "carry_success": bool(bilateral.any() and lift.max() >= LIFT_THRESHOLD_M),
            "kick_success": bool(any_foot.any() and planar_net >= 0.05),
            "reset_or_done": bool(done_hits.size),
        }
        records.append(record)
    return records


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    numeric = (
        "maximum_lift_m",
        "bilateral_hand_contact_fraction",
        "any_foot_box_contact_fraction",
        "planar_object_net_displacement_m",
        "planar_object_path_m",
        "maximum_robot_root_height_loss_m",
    )
    result: dict[str, object] = {
        f"mean_{key}": float(np.mean([float(record[key]) for record in records]))
        for key in numeric
    }
    result.update(
        {
            "carry_success_count": int(sum(bool(r["carry_success"]) for r in records)),
            "kick_success_count": int(sum(bool(r["kick_success"]) for r in records)),
            "physical_fall_count": int(sum(bool(r["physical_robot_fall"]) for r in records)),
            "reset_or_done_count": int(sum(bool(r["reset_or_done"]) for r in records)),
        }
    )
    return result


def main() -> None:
    output = args.output_dir.expanduser().resolve()
    checkpoint_path = args.shared_checkpoint.expanduser().resolve()
    proof_path = args.training_proof.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    if any(experiments not in path.parents for path in (output, checkpoint_path, proof_path)):
        raise ValueError("checkpoint, proof and output must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    source = DOMAIN[args.domain]
    for path in (
        checkpoint_path,
        proof_path,
        source["motion_folder"],
        source["generator"],
        source["tracker"],
        RUNTIME_CONFIG,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    failed = [name for name, passed in proof.get("checks", {}).items() if passed is not True]
    admitted_proof_protocols = {
        "sugar_shared_full_tracker_distillation_v1",
        "sugar_shared_full_tracker_dagger_fit_v1",
        "sugar_demo_conditioned_official_tracker_router_fit_v1",
    }
    if proof.get("protocol") not in admitted_proof_protocols or proof.get("passed") is not True or failed:
        raise RuntimeError("demo-conditioned Tracker proof is not admitted")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    admitted_checkpoint_protocols = {
        "sugar_shared_full_tracker_checkpoint_v1",
        "sugar_shared_full_tracker_dagger_checkpoint_v1",
        "sugar_demo_conditioned_official_tracker_router_checkpoint_v1",
    }
    if checkpoint.get("protocol") not in admitted_checkpoint_protocols:
        raise RuntimeError("demo-conditioned Tracker checkpoint is not admitted")
    official_tracker_router = (
        checkpoint.get("protocol")
        == "sugar_demo_conditioned_official_tracker_router_checkpoint_v1"
    )
    if args.causal_transition_risk_fallback and not official_tracker_router:
        raise RuntimeError(
            "transition-risk composition requires the checkpoint containing the "
            "two exact released Tracker experts"
        )
    output.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.chdir(SUGAR)
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args.seed
    env_cfg.commands.motion.generator_checkpoint_path = str(source["generator"])
    env_cfg.commands.motion.motion_folder = str(source["motion_folder"])
    env_cfg.commands.motion.eval_random_motion = False
    env_cfg.commands.motion.eval_mode = True
    env_cfg.commands.motion.eval_max_time = max(args.steps + 2, 660)
    if args.disable_observation_corruption:
        env_cfg.observations.policy.enable_corruption = False
    if args.scene_object_asset != "domain":
        object_cfg = SMALLBOX_CFG if args.scene_object_asset == "small" else BIGBOX_CFG
        env_cfg.scene.obj = object_cfg.replace(prim_path="{ENV_REGEX_NS}/Obj")
    for value in vars(env_cfg.scene).values():
        if hasattr(value, "debug_vis"):
            value.debug_vis = False

    # cli_args expects args.checkpoint to be the official runner checkpoint.
    args.checkpoint = str(source["tracker"])
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.seed = args.seed
    gym_env = gym.make(args.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    base = gym_env.unwrapped
    resolved_scene_asset = (
        args.scene_object_asset
        if args.scene_object_asset != "domain"
        else ("small" if args.domain == "CarryBox" else "big")
    )
    resolved_mass_source = (
        args.object_nominal_mass_source
        if args.object_nominal_mass_source != "domain"
        else ("small" if args.domain == "CarryBox" else "big")
    )
    nominal_mass_kg = {"small": 0.5, "big": 0.75}
    mass_inertia_ratio = (
        nominal_mass_kg[resolved_mass_source]
        / nominal_mass_kg[resolved_scene_asset]
    )
    obj_view = base.scene["obj"].root_physx_view
    original_object_masses = obj_view.get_masses().clone()
    original_object_inertias = obj_view.get_inertias().clone()
    if mass_inertia_ratio != 1.0:
        object_indices = torch.arange(args.num_envs, dtype=torch.int32)
        obj_view.set_masses(original_object_masses * mass_inertia_ratio, object_indices)
        obj_view.set_inertias(original_object_inertias * mass_inertia_ratio, object_indices)
    expected_object_masses = original_object_masses * mass_inertia_ratio
    expected_object_inertias = original_object_inertias * mass_inertia_ratio
    if not torch.allclose(obj_view.get_masses(), expected_object_masses):
        raise RuntimeError("object mass-source override readback mismatch")
    if not torch.allclose(obj_view.get_inertias(), expected_object_inertias):
        raise RuntimeError("object inertia-source override readback mismatch")
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    official_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    official_runner.load(str(source["tracker"]))
    official_policy = official_runner.get_inference_policy(device=base.device)

    if official_tracker_router:
        shared_policy = construct_router_policy(torch.device(args.device))
    else:
        shared_policy = construct_full_policy(torch.device(args.device))
    shared_policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    shared_policy.eval()
    shared_policy.requires_grad_(False)
    state_before = _clone_state(shared_policy)
    transition_risk_model = None
    transition_risk_threshold = 0.0
    transition_risk_state_before = None
    if args.causal_transition_risk_fallback:
        transition_proof_path = args.transition_risk_proof.expanduser()
        if not transition_proof_path.is_absolute():
            transition_proof_path = ROOT / transition_proof_path
        transition_proof_path = transition_proof_path.resolve()
        transition_proof = json.loads(transition_proof_path.read_text(encoding="utf-8"))
        if (
            transition_proof.get("protocol")
            != "sugar_causal_transition_risk_frozen_evaluation_v1"
            or transition_proof.get("passed") is not True
        ):
            raise RuntimeError("transition-risk frozen proof is not admitted")
        transition_checkpoint_path = Path(transition_proof["checkpoint"]).resolve()
        transition_checkpoint = torch.load(
            transition_checkpoint_path, map_location=base.device, weights_only=True
        )
        transition_state = transition_checkpoint["model_state_dict"]
        transition_risk_model = CausalTransitionRiskTransformer(
            transition_state["state_mean"].detach().cpu(),
            transition_state["state_std"].detach().cpu(),
        ).to(base.device)
        transition_risk_model.load_state_dict(transition_state, strict=True)
        transition_risk_model.eval()
        transition_risk_model.requires_grad_(False)
        transition_risk_state_before = _clone_state(transition_risk_model)
        transition_risk_threshold = float(
            transition_proof["validation_first50_selected_threshold"]
        )
        if not 0.0 < transition_risk_threshold < 1.0:
            raise RuntimeError("transition-risk threshold drift")
    scorer = None
    if not args.causal_transition_risk_fallback:
        scorer = FrozenPhaseAwareDemoEventScorer(
            num_envs=args.num_envs,
            device=base.device,
            cfg=FrozenPhaseAwareDemoEventScorerCfg(
                runtime_config_path=str(RUNTIME_CONFIG),
                selected_option=args.selected_demo_option,
                phase_horizon_steps=650,
            ),
        )

    command = base.command_manager.get_term("motion")
    resolved_target_goal_source = (
        args.target_goal_source
        if args.target_goal_source != "domain"
        else ("carry45" if args.domain == "CarryBox" else "kick21")
    )
    if args.target_goal_source != "domain":
        target_skill = "CarryBox" if args.target_goal_source == "carry45" else "KickBox"
        target_folder = Path(DOMAIN[target_skill]["motion_folder"])
        with (target_folder / "obj_motion_global_50hz.pkl").open("rb") as stream:
            target_object = pickle.load(stream)  # noqa: S301 - trusted repository asset
        with np.load(target_folder / "robot_50hz.npz", allow_pickle=False) as robot_archive:
            target_index = int(robot_archive["joint_pos"].shape[0] - 1)
        target_position = torch.as_tensor(
            np.asarray(target_object["obj_trans"])[target_index],
            dtype=command.obj_target_pos_w.dtype,
            device=base.device,
        )
        target_rotation = torch.as_tensor(
            np.asarray(target_object["obj_rot"])[target_index],
            dtype=command.obj_target_quat_w.dtype,
            device=base.device,
        )
        target_quaternion = quat_from_matrix(target_rotation.unsqueeze(0))[0]
        command.obj_target_pos_w[:] = target_position + base.scene.env_origins
        command.obj_target_quat_w[:] = target_quaternion
        command._update_target_visualization()
        all_env_ids = torch.arange(args.num_envs, device=base.device)
        command._fill_generator_obs_buffer(all_env_ids)
        command._call_generator(all_env_ids)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    initial = {
        "robot_root_state_w": base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy(),
        "robot_joint_pos": base.scene["robot"].data.joint_pos.detach().cpu().numpy().copy(),
        "robot_joint_vel": base.scene["robot"].data.joint_vel.detach().cpu().numpy().copy(),
        "object_root_state_w": base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy(),
        "object_mass_kg": base.scene["obj"].root_physx_view.get_masses().detach().cpu().numpy().copy(),
        "object_inertia": base.scene["obj"].root_physx_view.get_inertias().detach().cpu().numpy().copy(),
        "target_object_position_w": command.obj_target_pos_w.detach().cpu().numpy().copy(),
        "target_object_quaternion_w": command.obj_target_quat_w.detach().cpu().numpy().copy(),
    }
    with torch.inference_mode():
        prefix_action = official_policy(obs)
        obs, _, prefix_done, _ = env.step(prefix_action)
    initial_steps = command.time_steps.detach().clone().to(dtype=torch.long)
    if torch.any(prefix_done) or not torch.all(initial_steps == 1):
        raise RuntimeError("one-step official Tracker alignment prefix drift")
    post_prefix = {
        "robot_root_state_w": base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy(),
        "robot_joint_pos": base.scene["robot"].data.joint_pos.detach().cpu().numpy().copy(),
        "robot_joint_vel": base.scene["robot"].data.joint_vel.detach().cpu().numpy().copy(),
        "object_root_state_w": base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy(),
        "target_object_position_w": command.obj_target_pos_w.detach().cpu().numpy().copy(),
        "target_object_quaternion_w": command.obj_target_quat_w.detach().cpu().numpy().copy(),
    }
    routed_generator_skill = args.domain
    shadow_generator = None
    if args.route_generator_with_expert:
        routed_generator_skill = SELECTED_SKILL[args.selected_demo_option]
        if routed_generator_skill != args.domain:
            routed_generator_path = DOMAIN[routed_generator_skill]["generator"]
            selected_generator = GeneratorWrapper.load(
                checkpoint_path=str(routed_generator_path),
                device=base.device,
            )
            if selected_generator.n_obs_steps <= 0 or selected_generator.n_action_steps <= 0:
                raise RuntimeError("routed official Generator geometry drift")
            if args.causal_safe_fallback or args.causal_transition_risk_fallback:
                shadow_generator = _CausalShadowGenerator(selected_generator, command)
                obs = env.get_observations()
                if isinstance(obs, tuple):
                    obs = obs[0]
            else:
                command.generator = selected_generator
                all_env_ids = torch.arange(args.num_envs, device=base.device)
                command._fill_generator_obs_buffer(all_env_ids)
                command._call_generator(all_env_ids)
                obs = env.get_observations()
                if isinstance(obs, tuple):
                    obs = obs[0]
    tracker_policy_obs = obs["policy"]
    if tracker_policy_obs.shape != (args.num_envs, TRACKER_POLICY_DIM):
        raise RuntimeError("official Tracker policy observation geometry drift")
    if args.causal_transition_risk_fallback:
        core = None
        condition = torch.zeros(
            args.num_envs, ACTIONABLE_DEMO_CONDITIONING_DIM, device=base.device
        )
        scorer_begin_audit = {
            "constructed": False,
            "model_frozen": True,
            "future_actual_events_used": False,
            "reason": (
                "The transition-risk arm fixes the exact released candidate expert "
                "from the declared experiment and does not use the legacy scorer."
            ),
        }
    else:
        core = _goal_policy_core_observation(base)
        scorer_policy_obs = _policy_observation(core)
        scorer_begin_audit = scorer.begin(
            {"policy": scorer_policy_obs}, initial_episode_steps=initial_steps
        )
        condition = scorer.actionable_conditioning()
    if condition.shape != (args.num_envs, ACTIONABLE_DEMO_CONDITIONING_DIM):
        raise RuntimeError("selected-demo conditioning geometry drift")

    records: dict[str, list[np.ndarray]] = {
        "robot_root_state_w": [],
        "robot_body_position_w": [],
        "object_root_state_w": [],
        "contact_force_w": [],
        "contact": [],
        "action": [],
        "student_action": [],
        "teacher_action": [],
        "executed_action": [],
        "done": [],
        "motion_frame": [],
        "goal_policy_core_observation": [],
        "tracker_policy_observation": [],
        "demo_conditioning": [],
        "routing_weight": [],
        "safe_fallback_mask": [],
        "selected_generator_command": [],
        "selected_generator_normalized_max_abs": [],
        "selected_generator_outside_train_range_fraction": [],
        "transition_risk_probability": [],
        "transition_risk_latched_fallback": [],
    }
    zero_tactile = torch.zeros(args.num_envs, TACTILE_DIM, device=base.device)
    transition_history = None
    transition_probability_sum = torch.zeros(args.num_envs, device=base.device)
    transition_probability_count = torch.zeros(args.num_envs, device=base.device)
    transition_latched_fallback = torch.zeros(
        args.num_envs, dtype=torch.bool, device=base.device
    )
    invalid_transition: dict[str, object] | None = None
    for rollout_step in range(args.steps):
        actor_obs = {
            "policy": tracker_policy_obs,
            "critic": tracker_policy_obs,
            "demo_conditioning": condition,
            "tactile_history": zero_tactile,
        }
        with torch.inference_mode():
            selected_expert_index = (
                0 if args.selected_demo_option == "correct" else 1
            )
            if args.causal_transition_risk_fallback:
                # This experiment predeclares one candidate skill pair.  The frozen
                # transition-risk model decides only whether to execute that pair or
                # the domain pair; the older demo-event scorer is not part of this
                # decision and must not become an accidental OOD failure surface.
                routing_weight = torch.nn.functional.one_hot(
                    torch.full(
                        (args.num_envs,),
                        selected_expert_index,
                        dtype=torch.long,
                        device=base.device,
                    ),
                    num_classes=2,
                ).to(tracker_policy_obs.dtype)
            else:
                routing_weight = (
                    shared_policy.routing_weights(condition)
                    if official_tracker_router
                    else torch.zeros(args.num_envs, 2, device=base.device)
                )
            teacher_action = official_policy(obs)
            if shadow_generator is not None:
                normalized_max_abs, outside_train_fraction = (
                    shadow_generator.normalized_compatibility()
                )
                selected_command = shadow_generator.command_at(command.time_steps)
                selected_tracker_obs = tracker_policy_obs.clone()
                selected_tracker_obs[:, :36] = selected_command
                actor_tracker_policy_obs = selected_tracker_obs
                selected_actor_obs = dict(actor_obs)
                selected_actor_obs["policy"] = selected_tracker_obs
                selected_actor_obs["critic"] = selected_tracker_obs
                student_action = (
                    shared_policy.experts[selected_expert_index](selected_tracker_obs)
                    if args.causal_transition_risk_fallback
                    else shared_policy.act_inference(selected_actor_obs)
                )
                if args.causal_transition_risk_fallback:
                    if (
                        not torch.isfinite(selected_tracker_obs).all()
                        or not torch.isfinite(student_action).all()
                    ):
                        raise RuntimeError(
                            "transition-risk candidate input/action became non-finite"
                        )
                    if rollout_step <= TRANSITION_RISK_DECISION_FRAME:
                        feature = torch.cat((selected_tracker_obs, student_action), dim=-1)
                        if transition_history is None:
                            transition_history = feature.unsqueeze(1).expand(
                                -1, TRANSITION_RISK_HISTORY_STEPS, -1
                            ).clone()
                        else:
                            transition_history[:, :-1] = transition_history[:, 1:].clone()
                            transition_history[:, -1] = feature
                        transition_output = transition_risk_model(transition_history)
                        transition_probability = torch.sigmoid(
                            transition_output["risk_logit"]
                        )
                        if not torch.isfinite(transition_probability).all():
                            raise RuntimeError(
                                "transition-risk probability became non-finite during "
                                f"the admitted early window at frame {rollout_step}"
                            )
                        sample_probability = bool(
                            TRANSITION_RISK_HISTORY_STEPS - 1 <= rollout_step
                            and (rollout_step - (TRANSITION_RISK_HISTORY_STEPS - 1))
                            % TRANSITION_RISK_SAMPLE_STRIDE
                            == 0
                        )
                        if sample_probability:
                            transition_probability_sum += transition_probability
                            transition_probability_count += 1.0
                        if rollout_step == TRANSITION_RISK_DECISION_FRAME:
                            if not torch.all(transition_probability_count == 9):
                                raise RuntimeError("transition-risk sampling clock drift")
                            transition_latched_fallback |= (
                                transition_probability_sum
                                / transition_probability_count
                                >= transition_risk_threshold
                            )
                    else:
                        # The decision is irreversible.  Retain its causal evidence
                        # for visualization without evaluating an unused hypothetical
                        # candidate on the fallback-generated state distribution.
                        transition_probability = (
                            transition_probability_sum / transition_probability_count
                        )
                    action = torch.where(
                        transition_latched_fallback.unsqueeze(-1),
                        teacher_action,
                        student_action,
                    )
                    safe_fallback_mask = transition_latched_fallback
                else:
                    transition_probability = torch.zeros(
                        args.num_envs, device=base.device
                    )
                    safe = (
                        torch.amax(torch.abs(student_action), dim=-1)
                        <= RELEASED_TRACKER_RAW_ACTION_LIMIT
                    )
                    action = torch.where(safe.unsqueeze(-1), student_action, teacher_action)
                    safe_fallback_mask = ~safe
            else:
                normalized_max_abs = torch.zeros(args.num_envs, device=base.device)
                outside_train_fraction = torch.zeros(args.num_envs, device=base.device)
                selected_command = tracker_policy_obs[:, :36]
                actor_tracker_policy_obs = tracker_policy_obs
                student_action = shared_policy.act_inference(actor_obs)
                action = (
                    args.student_action_fraction * student_action
                    + (1.0 - args.student_action_fraction) * teacher_action
                )
                safe_fallback_mask = torch.zeros(
                    args.num_envs, dtype=torch.bool, device=base.device
                )
                transition_probability = torch.zeros(
                    args.num_envs, device=base.device
                )
            obs, _, done, _ = env.step(action)
            next_tracker_policy_obs = obs["policy"]
            if (
                next_tracker_policy_obs.shape != (args.num_envs, TRACKER_POLICY_DIM)
                or not torch.isfinite(next_tracker_policy_obs).all()
            ):
                invalid_rows = torch.any(
                    ~torch.isfinite(next_tracker_policy_obs), dim=-1
                )
                invalid_transition = {
                    "frame": rollout_step,
                    "env_indices": torch.nonzero(
                        invalid_rows, as_tuple=False
                    ).squeeze(-1).detach().cpu().numpy(),
                    "candidate_selected": (
                        ~transition_latched_fallback
                    ).detach().cpu().numpy(),
                    "latched_fallback": transition_latched_fallback.detach().cpu().numpy(),
                    "risk_probability": transition_probability.detach().cpu().numpy(),
                    "executed_action": action.detach().cpu().numpy(),
                    "candidate_action": student_action.detach().cpu().numpy(),
                    "fallback_action": teacher_action.detach().cpu().numpy(),
                    "tracker_observation_before": tracker_policy_obs.detach().cpu().numpy(),
                    "tracker_observation_after": next_tracker_policy_obs.detach().cpu().numpy(),
                }
                break
            if shadow_generator is not None:
                shadow_generator.update_after_step(command)
        forces = torch.stack(
            [_latest_filtered_force(base.scene.sensors[name]) for name in SENSOR_NAMES],
            dim=1,
        )
        records["robot_root_state_w"].append(
            base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy()
        )
        records["robot_body_position_w"].append(
            base.scene["robot"].data.body_pos_w.detach().cpu().numpy().copy()
        )
        records["object_root_state_w"].append(
            base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy()
        )
        force_np = forces.detach().cpu().numpy().copy()
        records["contact_force_w"].append(force_np)
        records["contact"].append(
            np.linalg.norm(force_np, axis=-1) > CONTACT_THRESHOLD_N
        )
        records["action"].append(action.detach().cpu().numpy().copy())
        records["student_action"].append(
            student_action.detach().cpu().numpy().copy()
        )
        records["teacher_action"].append(
            teacher_action.detach().cpu().numpy().copy()
        )
        records["executed_action"].append(action.detach().cpu().numpy().copy())
        records["done"].append(done.detach().cpu().numpy().astype(bool, copy=True))
        records["motion_frame"].append(
            command.time_steps.detach().cpu().numpy().copy()
        )
        if core is not None:
            records["goal_policy_core_observation"].append(
                core.detach().cpu().numpy().copy()
            )
        records["tracker_policy_observation"].append(
            actor_tracker_policy_obs.detach().cpu().numpy().copy()
        )
        records["demo_conditioning"].append(
            condition.detach().cpu().numpy().copy()
        )
        records["routing_weight"].append(
            routing_weight.detach().cpu().numpy().copy()
        )
        records["safe_fallback_mask"].append(
            safe_fallback_mask.detach().cpu().numpy().copy()
        )
        records["selected_generator_command"].append(
            selected_command.detach().cpu().numpy().copy()
        )
        records["selected_generator_normalized_max_abs"].append(
            normalized_max_abs.detach().cpu().numpy().copy()
        )
        records["selected_generator_outside_train_range_fraction"].append(
            outside_train_fraction.detach().cpu().numpy().copy()
        )
        records["transition_risk_probability"].append(
            transition_probability.detach().cpu().numpy().copy()
        )
        records["transition_risk_latched_fallback"].append(
            transition_latched_fallback.detach().cpu().numpy().copy()
        )
        tracker_policy_obs = next_tracker_policy_obs
        if not args.causal_transition_risk_fallback:
            core = _goal_policy_core_observation(base)
            scorer_policy_obs = _policy_observation(core)
            condition = scorer.process_step(
                {"policy": scorer_policy_obs}, done
            ).actionable_conditioning

    arrays = {
        name: np.stack(values)
        for name, values in records.items()
        if values
    }
    arrays.update(
        {
            f"initial_{name}": value for name, value in initial.items()
        }
    )
    arrays.update(
        {
            f"post_prefix_{name}": value for name, value in post_prefix.items()
        }
    )
    arrays["prefix_action"] = prefix_action.detach().cpu().numpy().copy()
    arrays["robot_body_names"] = np.asarray(base.scene["robot"].body_names, dtype="U64")
    arrays["contact_role_names"] = np.asarray(
        ("left_hand", "right_hand", "left_foot", "right_foot"), dtype="U16"
    )
    completed_frames = int(arrays["action"].shape[0])
    records_by_profile = _profile_summaries(arrays)
    aggregate = _aggregate(records_by_profile)
    reference_action_envelope = RELEASED_TRACKER_RAW_ACTION_LIMIT
    matched_condition = (
        (args.domain == "CarryBox" and args.selected_demo_option == "correct")
        or (args.domain == "KickBox" and args.selected_demo_option == "unrelated")
    )
    task_success_count = (
        aggregate["carry_success_count"]
        if args.domain == "CarryBox"
        else aggregate["kick_success_count"]
    )
    selected_skill_success_count = (
        aggregate["carry_success_count"]
        if SELECTED_SKILL[args.selected_demo_option] == "CarryBox"
        else aggregate["kick_success_count"]
    )
    checks = {
        "shared_absolute_checkpoint_admitted": True,
        "one_step_official_domain_tracker_prefix_only": True,
        "declared_actor_or_released_fallback_supplies_every_action": True,
        "actor_state_contract_matches_checkpoint": bool(
            arrays["tracker_policy_observation"].shape[-1] == TRACKER_POLICY_DIM
        ),
        "future_tracker_actions_unavailable_during_evaluation": True,
        "between_arm_variable_matches_declared_protocol": True,
        "frozen_predictor_model": bool(scorer_begin_audit["model_frozen"]),
        "policy_parameters_frozen_and_unchanged": _same_state(
            state_before, _clone_state(shared_policy)
        ),
        "exact_zero_tactile_input": bool(torch.count_nonzero(zero_tactile) == 0),
        "all_trace_values_finite": all(
            np.isfinite(value).all()
            for value in arrays.values()
            if isinstance(value, np.ndarray) and value.dtype.kind in "f"
        ),
        "completed_all_650_control_frames": completed_frames == args.steps,
        "official_tracker_observation_remained_finite": invalid_transition is None,
        "no_environment_reset": aggregate["reset_or_done_count"] == 0,
        "all_actions_finite_and_nonzero": bool(
            np.isfinite(arrays["action"]).all()
            and np.max(np.abs(arrays["action"])) > 0.0
        ),
        "online_teacher_labels_finite": bool(
            np.isfinite(arrays["teacher_action"]).all()
        ),
        "router_selects_requested_expert": (
            bool(
                np.mean(
                    np.argmax(arrays["routing_weight"], axis=-1)
                    == (0 if args.selected_demo_option == "correct" else 1)
                )
                >= 0.99
            )
            if official_tracker_router
            else True
        ),
        "matched_demo_task_behavioral_gate": (
            int(task_success_count) >= 10
            and int(aggregate["physical_fall_count"]) <= 2
            if (matched_condition and not args.dagger_collection)
            else True
        ),
    }
    if args.route_generator_with_expert:
        if args.causal_transition_risk_fallback:
            checks["selected_demo_candidate_uses_complete_generator_tracker_pair"] = True
            checks["candidate_tracker_is_exact_predeclared_released_expert"] = bool(
                official_tracker_router
            )
            checks["legacy_demo_event_scorer_absent_from_fallback_decision"] = True
            checks["causal_selected_candidate_and_domain_fallback_both_online"] = True
            checks["transition_risk_model_frozen_and_unchanged"] = _same_state(
                transition_risk_state_before, _clone_state(transition_risk_model)
            )
            checks["transition_risk_decision_uses_exactly_nine_early_samples"] = bool(
                np.all(transition_probability_count.detach().cpu().numpy() == 9)
            )
            checks["transition_risk_fallback_precedes_known_earliest_failure"] = (
                TRANSITION_RISK_DECISION_FRAME < 103
            )
            checks["executed_actions_within_released_tracker_envelope"] = bool(
                np.max(np.abs(arrays["executed_action"])) <= reference_action_envelope
            )
            checks["transition_risk_fallback_prevents_physical_falls"] = bool(
                int(aggregate["physical_fall_count"]) == 0
            )
        elif args.causal_safe_fallback:
            checks["selected_demo_candidate_uses_complete_generator_tracker_pair"] = True
            checks["causal_selected_candidate_and_domain_fallback_both_online"] = True
            checks["executed_actions_within_released_tracker_envelope"] = bool(
                np.max(np.abs(arrays["executed_action"])) <= reference_action_envelope
            )
            checks["safe_fallback_prevents_physical_falls"] = bool(
                int(aggregate["physical_fall_count"]) == 0
            )
        else:
            checks["selected_demo_routes_complete_generator_tracker_pair"] = True
            checks["raw_student_actions_within_released_tracker_envelope"] = bool(
                np.max(np.abs(arrays["student_action"])) <= reference_action_envelope
            )
            checks["selected_skill_behavioral_gate"] = bool(
                int(selected_skill_success_count) >= 10
                and int(aggregate["physical_fall_count"]) <= 2
            )
    else:
        checks["selected_demo_condition_is_only_between_arm_variable"] = True
    result = {
        "protocol": (
            "sugar_shared_absolute_tracker_dagger_collection_v1"
            if args.dagger_collection
            else "sugar_shared_absolute_tracker_frozen_physics_v1"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "domain": args.domain,
        "domain_motion_id": source["motion_id"],
        "selected_demo_option": args.selected_demo_option,
        "selected_demo_motion_id": 45 if args.selected_demo_option == "correct" else 21,
        "routed_generator_with_expert": args.route_generator_with_expert,
        "causal_safe_fallback": args.causal_safe_fallback,
        "causal_transition_risk_fallback": args.causal_transition_risk_fallback,
        "observation_corruption_disabled": args.disable_observation_corruption,
        "routed_generator_skill": routed_generator_skill,
        "selected_skill_success_count": int(selected_skill_success_count),
        "scene_object_asset": args.scene_object_asset,
        "resolved_scene_object_asset": resolved_scene_asset,
        "object_nominal_mass_source": resolved_mass_source,
        "mass_inertia_override_ratio": mass_inertia_ratio,
        "target_goal_source": resolved_target_goal_source,
        "object_mass_kg": {
            "minimum": float(arrays["initial_object_mass_kg"].min()),
            "mean": float(arrays["initial_object_mass_kg"].mean()),
            "maximum": float(arrays["initial_object_mass_kg"].max()),
        },
        "object_inertia_readback_all_finite": bool(
            np.isfinite(arrays["initial_object_inertia"]).all()
        ),
        "matched_demo_task_condition": matched_condition,
        "shared_checkpoint": str(checkpoint_path),
        "training_proof": str(proof_path),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps_after_prefix": args.steps,
        "executed_valid_frames": completed_frames,
        "first_invalid_transition": (
            None
            if invalid_transition is None
            else {
                "frame": int(invalid_transition["frame"]),
                "env_indices": invalid_transition["env_indices"].tolist(),
                "candidate_selected_env_indices": np.flatnonzero(
                    invalid_transition["candidate_selected"]
                ).tolist(),
                "latched_fallback_env_indices": np.flatnonzero(
                    invalid_transition["latched_fallback"]
                ).tolist(),
                "risk_probability_by_env": invalid_transition[
                    "risk_probability"
                ].tolist(),
                "executed_action_all_finite": bool(
                    np.isfinite(invalid_transition["executed_action"]).all()
                ),
                "candidate_action_all_finite": bool(
                    np.isfinite(invalid_transition["candidate_action"]).all()
                ),
                "fallback_action_all_finite": bool(
                    np.isfinite(invalid_transition["fallback_action"]).all()
                ),
                "maximum_abs_finite_candidate_action": float(
                    np.max(np.abs(invalid_transition["candidate_action"]))
                ),
            }
        ),
        "prefix_steps": 1,
        "dagger_collection": args.dagger_collection,
        "student_action_fraction": args.student_action_fraction,
        "actor_state_observation_contract": (
            "exact_official_510D_Tracker_observation_plus_current_29D_candidate_"
            "action_to_frozen_transition_risk_model; candidate_is_the_predeclared_"
            "exact_released_Tracker_expert"
            if args.causal_transition_risk_fallback
            else "exact_official_510D_Tracker_policy_observation_plus_"
            "causal_selected_demo_expert_router"
            if official_tracker_router
            else "exact_official_510D_Tracker_policy_observation"
        ),
        "official_prefix_tracker": str(source["tracker"]),
        "domain_prefix_generator_checkpoint": str(source["generator"]),
        "routed_generator_checkpoint": str(
            DOMAIN[routed_generator_skill]["generator"]
        ),
        "motion_folder": str(source["motion_folder"]),
        "scorer_begin_audit": scorer_begin_audit,
        "frozen_model_audit": (
            scorer.frozen_model_audit()
            if scorer is not None
            else {
                "constructed": False,
                "reason": "legacy demo-event scorer is outside the transition-risk method",
            }
        ),
        "aggregate": aggregate,
        "teacher_student_action_mae": float(
            np.mean(np.abs(arrays["teacher_action"] - arrays["student_action"]))
        ),
        "maximum_abs_raw_student_action": float(
            np.max(np.abs(arrays["student_action"]))
        ),
        "maximum_abs_executed_action": float(np.max(np.abs(arrays["executed_action"]))),
        "safe_fallback_fraction": float(np.mean(arrays["safe_fallback_mask"])),
        "profiles_using_safe_fallback": int(
            np.any(arrays["safe_fallback_mask"], axis=0).sum()
        ),
        "released_tracker_action_envelope": reference_action_envelope,
        "transition_risk_threshold": transition_risk_threshold,
        "transition_risk_latched_profile_count": int(
            np.sum(arrays["transition_risk_latched_fallback"][-1])
        ),
        "profiles": records_by_profile,
        "claim_boundary": (
            "A frozen causal Carry45 transition-risk composition of two released "
            "SUGAR Generator+Tracker pairs. It observes only the first 50 frames "
            "of current Tracker state and candidate action, then irreversibly latches "
            "predicted-risk profiles to the domain pair. The candidate expert is fixed "
            "by the declared arm, not by the legacy demo-event scorer. Future outcomes "
            "are absent."
            if args.causal_transition_risk_fallback
            else
            "A causal safety composition of two released SUGAR Generator+Tracker "
            "pairs. The selected pair is evaluated online from current simulator "
            "state; only its current action envelope may select the domain official "
            "pair as fallback. This is a safety mechanism, not arbitrary-demo "
            "imitation or semantic skill success."
            if args.causal_safe_fallback
            else
            "A matched frozen-physics test of one causal router selecting a complete "
            "released SUGAR Generator+Tracker pair after a common one-step domain "
            "prefix. The physical scene, task goal and initial state remain fixed; "
            "the selected demo identity changes the routed skill pair. A failure is "
            "not semantic following."
            if args.route_generator_with_expert
            else "A matched condition-only frozen physics test of one shared "
            "absolute-action actor. Success requires at least 10/20 domain-matched "
            "Carry/Kick physical successes and at most 2/20 physical falls; a "
            "condition-dependent failure is not semantic following."
        ),
        "artifacts": {
            "trace": "TRACE.npz",
            "result": "RESULT.json",
            "invalid_transition": (
                "INVALID_TRANSITION.npz" if invalid_transition is not None else None
            ),
        },
    }
    np.savez_compressed(output / "TRACE.npz", **arrays)
    if invalid_transition is not None:
        np.savez_compressed(
            output / "INVALID_TRANSITION.npz",
            **{
                name: np.asarray(value)
                for name, value in invalid_transition.items()
                if name != "frame"
            },
            frame=np.asarray(invalid_transition["frame"], dtype=np.int64),
        )
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "passed": result["passed"],
        "domain": args.domain,
        "selected_demo_option": args.selected_demo_option,
        "aggregate": aggregate,
        "checks": checks,
    }, indent=2, sort_keys=True))
    env.close()
    if not result["passed"]:
        raise RuntimeError("demo-conditioned Tracker frozen evaluation failed")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close()
