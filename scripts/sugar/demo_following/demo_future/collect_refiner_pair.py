"""Frozen official SUGAR controller reference-switch diagnostic in real PhysX.

The project-trained step10000 checkpoint is used unchanged through the existing
official observation/ActorCritic adapter. This is data feasibility, not policy
training, a new controller, or an official released Refiner checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "experiments/demo_following/demo_future_smp_v1/refiner_pair_feasibility"
CHECKPOINT = ROOT / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt"


def main():
    if not os.environ.get("SLURM_STEP_ID") or socket.gethostname().startswith(("login", "mgmtserver")):
        raise RuntimeError("PhysX requires the retained compute step")
    sugar = ROOT / "SUGAR"
    sys.path.insert(0, str(sugar / "scripts/sugar_rl"))
    os.environ.setdefault("ISAACLAB_GROUND_PLANE_USD", str(sugar / "descriptions/terrain/sugar_ground_plane.usda"))
    os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
    os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("DISPLAY", "")
    os.environ.setdefault("ISAACLAB_TMP_ROOT", f"/tmp/Curiosity_refiner_pair_{os.environ['SLURM_JOB_ID']}")
    os.environ.setdefault("SUGAR_UNITREE_TMP_ROOT", f"/tmp/Curiosity_refiner_pair_unitree_{os.environ['SLURM_JOB_ID']}")
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("original", "repeat", "alternate"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--video", action="store_true", help="Record a separate actual camera-enabled rollout, not a replay of camera-free statistics")
    parser.add_argument("--controller", choices=("refiner", "tracker"), default="refiner")
    parser.add_argument("--tracker-checkpoint",type=Path,help="Exact full Tracker adaptation endpoint; default is released checkpoint")
    parser.add_argument('--original-position-feedback',action='store_true',help='Frozen native diagnostic: replace only48 position inputs with original numeric-demo positions')
    parser.add_argument('--robot-usd',type=Path,help='Reuse unchanged official URDF-converter output, preserving full robot and common spawn settings')
    parser.add_argument("--audit-teacher-bank",type=Path,help="Zero-update same-world audit of official BCPPO teacher observation against direct Refiner")
    parser.add_argument("--teacher-handoff-step",type=int,help="Diagnostic live handoff to the exact frozen Refiner without resetting state")
    parser.add_argument("--record-bc-replay",action='store_true',help='Record actual critic/reward/next observations for a fixed-data official BC-stage diagnostic')
    parser.add_argument('--disable-fabric-gpu-interop',action='store_true',help='Diagnose Vulkan device loss by disabling only PhysX-to-Fabric GPU interop; physics remains GPU PhysX')
    parser.add_argument('--disable-fabric',action='store_true',help='Use official USD readback instead of Fabric for runtime diagnosis; GPU PhysX solver and physical parameters unchanged')
    parser.add_argument('--clear-cuda-mask',action='store_true',help='Diagnose CUDA/Vulkan enumeration conflict only when Slurm/NVML expose exactly one GPU')
    parser.add_argument('--default-renderer-settings',action='store_true',help='Camera runtime diagnostic: use official AppLauncher renderer settings without extra multiGpu overrides')
    parser.add_argument('--synchronous-renderer-init',action='store_true',help='Camera-only runtime diagnostic: disable the official asynchronous renderer initialization setting')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    prior_cuda_mask=os.environ.get('CUDA_VISIBLE_DEVICES')
    if args.clear_cuda_mask:
        visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
        if len(visible)!=1 or prior_cuda_mask!='0':raise RuntimeError('Clearing CUDA mask requires exactly one cgroup-visible GPU and original local mask0')
        os.environ.pop('CUDA_VISIBLE_DEVICES')
    if args.tracker_checkpoint:args.tracker_checkpoint=args.tracker_checkpoint.resolve()
    if args.robot_usd:args.robot_usd=args.robot_usd.resolve()
    if args.audit_teacher_bank:args.audit_teacher_bank=args.audit_teacher_bank.resolve()
    if args.teacher_handoff_step is not None and (not args.audit_teacher_bank or args.teacher_handoff_step<0):
        raise ValueError('Teacher handoff requires the audited teacher bank and a nonnegative step')
    if args.record_bc_replay and not args.audit_teacher_bank:raise ValueError('BC replay recording requires the exact teacher interface')
    args.enable_cameras = args.video
    args.kit_args = "--/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false"
    if args.default_renderer_settings:
        if not args.video:raise ValueError('Default-renderer diagnostic is camera-only')
        args.kit_args = ""
    if args.synchronous_renderer_init:
        if not args.video:raise ValueError('Synchronous renderer initialization diagnostic is camera-only')
        args.kit_args += ' --/renderer/asyncInit=false'
    if args.disable_fabric_gpu_interop:args.kit_args+=' --/physics/fabricUseGPUInterop=false'
    if not args.video:
        args.kit_args += " --/renderer/enabled="
    run = args.run.resolve()
    plan = json.loads((run / "PROTOCOL.json").read_text())
    if plan.get('generated_rollout') and (args.controller != 'tracker' or args.audit_teacher_bank or args.original_position_feedback or args.video):
        raise RuntimeError('Generated-command diagnostic requires the declared frozen Tracker and known position interface')
    sources = plan["reference_pair"]
    aligned = plan.get("reference_alignment") == "single_causal_yaw_box_xy"
    if args.original_position_feedback and (args.controller != 'tracker' or plan.get('reference_alignment') != 'none' or args.arm != 'original'):
        raise ValueError('Original-position diagnostic requires the native original Tracker arm without alignment')
    out = (args.output or run / args.arm).resolve()
    out.mkdir(parents=True, exist_ok=False)
    portable = Path(f"/tmp/Curiosity_refiner_pair_kit_{os.environ['SLURM_JOB_ID']}_{os.getpid()}")
    portable.mkdir(exist_ok=False)
    sys.argv.extend(["--portable-root", str(portable)])
    app = AppLauncher(args, multi_gpu=False, max_gpu_count=1).app
    import numpy as np
    import torch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import sugar_rl.tasks  # noqa: F401
    from sugar_rl.utils.parser_cfg import parse_env_cfg
    from sugar_rl.utils.official_refiner_nominal_teacher import FrozenOfficialRefinerTeacher
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

    os.chdir(sugar)
    seed, steps, switch = plan["seed"], plan["steps_per_arm"], plan["switch_control_frame"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    folder = run / "motions"
    if not folder.is_dir():
        raise RuntimeError("Predeclared motion folder is missing")
    task = "Sugar-G129dof-CarryBox-" + args.controller.title()
    cfg = parse_env_cfg(task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric,
                        entry_point_key="play_env_cfg_entry_point")
    asset_adapter=None
    if args.robot_usd:
        from scripts.sugar.demo_following.demo_future.robot_asset import use_converted_robot_usd
        asset_adapter=use_converted_robot_usd(cfg.scene.robot,args.robot_usd)
    cfg.seed = seed
    cfg.commands.motion.motion_folder = str(folder)
    if args.audit_teacher_bank:
        cfg.commands.motion.teacher_motion_folder = str(args.audit_teacher_bank.resolve())
    cfg.commands.motion.use_generator = False
    cfg.commands.motion.eval_mode = True
    cfg.commands.motion.eval_random_motion = False
    cfg.commands.motion.eval_max_time = max(steps + 2, *[
        len(np.load(p / "robot_50hz.npz")["joint_pos"])
        for p in sorted(folder.glob("data_*"))])
    cfg.commands.motion.pose_range = {k: (0., 0.) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.commands.motion.joint_position_range = (0., 0.)
    cfg.events.push_robot = None
    cfg.events.push_object = None
    # One explicitly nominal feasibility condition, identical across all arms.
    for name in ("robot_physics_material", "obj_physics_material"):
        term = getattr(cfg.events, name)
        term.params.update(static_friction_range=(1., 1.), dynamic_friction_range=(1., 1.),
                           restitution_range=(0., 0.))
    cfg.events.obj_mass.params["mass_distribution_params"] = (1., 1.)
    cfg.observations.policy.enable_corruption = False
    cfg.observations.critic.enable_corruption = False
    if args.controller == 'tracker' and not args.audit_teacher_bank:
        # This training-only BCPPO observation group requires a separate
        # teacher motion bank. Frozen released-actor inference never reads it.
        cfg.observations.teacher = None
    if args.video:
        from isaaclab.sensors import TiledCameraCfg
        import isaaclab.sim as sim_utils
        cfg.scene.world_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/WorldCamera", update_period=.02,
            offset=TiledCameraCfg.OffsetCfg(pos=(3.6,3.6,2.4),
                rot=(.3043649418,.2319667899,.5600173703,.7348019703), convention="opengl"),
            data_types=["rgb"],width=960,height=540,
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.,focus_distance=4.,
                horizontal_aperture=24.,clipping_range=(.05,20.)))
        cfg.sim.render_interval = cfg.decimation
    # Keep original termination definitions. Stop at the first failure; never
    # splice an automatic reset into a purported shared-history future.
    print("REFINER_PAIR_PHASE=before_gym_make", flush=True)
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    env.reset(seed=seed)
    command = base.command_manager.get_term("motion")
    robot, obj = base.scene["robot"], base.scene["obj"]
    termination_compute = base.termination_manager.compute
    def capture_terminal_before_reset():
        value = termination_compute()
        if value.any() and not (out/'TERMINAL_STATE.npz').exists():
            # Read-only hook at the original termination decision, before
            # ManagerBasedRLEnv overwrites the failed world with a reset.
            np.savez(out/'TERMINAL_STATE.npz',
                     robot_body_state_w=robot.data.body_state_w.detach().cpu().numpy(),
                     robot_root_state_w=robot.data.root_state_w.detach().cpu().numpy(),
                     joint_pos=robot.data.joint_pos.detach().cpu().numpy(),
                     joint_vel=robot.data.joint_vel.detach().cpu().numpy(),
                     object_state_w=obj.data.root_state_w.detach().cpu().numpy(),
                     requested_action=base.action_manager.action.detach().cpu().numpy())
            terms={name:bool(term.func(base,**term.params).any()) for name,term in
                   zip(base.termination_manager._term_names,base.termination_manager._term_cfgs)}
            (out/'TERMINAL_TERMS.json').write_text(json.dumps(terms,indent=2)+'\n')
        return value
    base.termination_manager.compute = capture_terminal_before_reset
    if command.motion.num_motion != 2 or command.motion_id.item() != 0:
        raise RuntimeError("Expected Carry45 initialization and ordered Carry45/Carry96 bank")
    checkpoint = CHECKPOINT
    origin = "project-trained official SUGAR architecture"
    actor_contract = "unchanged frozen890-D privileged teacher;29-D executed actions"
    if args.controller == 'tracker':
        from scripts.sugar.demo_following.demo_future.frozen_tracker import FrozenReleasedTracker
        checkpoint = (args.tracker_checkpoint or ROOT/'SUGAR/demo_ckpts/CarryBox/tracker.pt').resolve()
        origin = 'project-adapted full official Tracker endpoint' if args.tracker_checkpoint else 'author-released full SUGAR Tracker actor'
        actor_contract = 'unchanged frozen510-D Tracker; current36-D reference command, causal proprioception history and current object pose;29-D executed actions'
        position_provider = None
        if args.original_position_feedback:
            from scripts.sugar.demo_following.demo_future.generator_tracker_routing import OriginalNumericPositionFeedback
            position_provider = OriginalNumericPositionFeedback(base, sources)
        teacher = FrozenReleasedTracker(base, checkpoint, origin=origin, position_feedback_provider=position_provider)
        if teacher.paired_input:
            actor_contract=f'Full official Tracker with798-D adapted input: original510-D plus288-D known future reference packet ({teacher.paired_input}); unchanged512/256/128 hidden widths and29-D executed actions'
        if teacher.reference_feedback:
            actor_contract=f'Full official846-D Tracker:510 current+288 known future command+48 reference position feedback ({teacher.reference_feedback});512/256/128 hidden widths,29 actual actions. New causal localization-dependent research interface.'
    else:
        teacher = FrozenOfficialRefinerTeacher(base, checkpoint, expected_sha256=None)
    generated_controller = None
    if plan.get('generated_rollout'):
        from scripts.sugar.demo_following.demo_future.generated_rollout_controller import GeneratedRolloutController
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            generated_controller = GeneratedRolloutController(base, teacher, run, args.arm, out)
    teacher_interface = None
    if args.audit_teacher_bank:
        if args.controller != 'tracker':raise RuntimeError('Teacher-interface audit requires Tracker task')
        teacher_interface = FrozenOfficialRefinerTeacher(base, CHECKPOINT, expected_sha256=None)
        released_control = FrozenReleasedTracker(base, ROOT/'SUGAR/demo_ckpts/CarryBox/tracker.pt')
    lengths = command.motion.time_step_total_permotion.detach().cpu().tolist()
    start_frame = int(command.time_steps.item())
    protocol = dict(arm=args.arm, seed=seed, requested_steps=steps, switch_control_frame=switch,
                    source_ids=sources, source_lengths=lengths, initial_reference_frame=start_frame,
                    checkpoint=str(checkpoint), teacher_origin=origin, controller=args.controller,
                    physics=f"original {args.controller} PhysX scene; friction1/restitution0/mass1; zero reset pose/joint noise; no external pushes",
                    actor=actor_contract,
                    alignment=plan["alignment"],
                    scope="bounded data feasibility, not learned-policy following or a deployment interface",
                    slurm_job_id=os.environ["SLURM_JOB_ID"], slurm_step_id=os.environ["SLURM_STEP_ID"],
                    host=socket.gethostname(), pid=os.getpid())
    protocol['camera_enabled'] = args.video
    protocol['position_feedback_source'] = 'original_numeric_demo' if args.original_position_feedback else 'unchanged_measured_reference'
    protocol['robot_asset_adapter']=asset_adapter
    protocol['tracker_readonly_warmup']=getattr(teacher,'warmup_audit',None)
    protocol['fabric_gpu_interop_disabled']=args.disable_fabric_gpu_interop
    protocol['fabric_disabled']=args.disable_fabric
    protocol['cuda_mask_cleared']=args.clear_cuda_mask
    protocol['original_cuda_mask']=prior_cuda_mask
    protocol['default_renderer_settings']=args.default_renderer_settings
    protocol['synchronous_renderer_init']=args.synchronous_renderer_init
    protocol['teacher_handoff_step'] = args.teacher_handoff_step
    if generated_controller is not None:
        protocol['generated_rollout'] = plan['generated_rollout']
        protocol['scope'] = 'Bounded actual generated-current/future-command PhysX diagnostic with known48position assistance. Original terms, no optimizer, no independent deployment or SMP benefit claim.'
    if args.teacher_handoff_step is not None:
        protocol['scope']='Live student-to-frozen-Refiner recovery diagnostic. No state reset, no training, not a deployable learned controller or paired-demo success.'
    protocol['video_scope'] = 'This separate actual PhysX rollout proves only its own trajectory; camera-free traces remain authoritative paired statistics.' if args.video else None
    (out / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    startup = {"joint_names": np.asarray(robot.joint_names), "body_names": np.asarray(robot.body_names),
               "default_joint_pos": robot.data.default_joint_pos.cpu().numpy(),
               "default_joint_vel": robot.data.default_joint_vel.cpu().numpy(),
               "robot_material": robot.root_physx_view.get_material_properties().cpu().numpy(),
               "object_material": obj.root_physx_view.get_material_properties().cpu().numpy(),
               "object_mass": obj.root_physx_view.get_masses().cpu().numpy(),
               "object_inertia": obj.root_physx_view.get_inertias().cpu().numpy()}
    np.savez(out / "STARTUP.npz", **startup)
    writer = None
    if args.video:
        import imageio.v2 as imageio
        import cv2
        writer = imageio.get_writer(str(out/'ACTUAL_WORLD.mp4'),fps=25,codec='libx264',
                                   macro_block_size=1,pixelformat='yuv420p')
    records = {}
    def save(name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().copy()
        records.setdefault(name, []).append(value)

    def select_reference(index, frame):
        # Change only the reference selector/clock and its derived bookkeeping;
        # never write robot/object/actuator/sensor state at the switch.
        command.motion_id.fill_(index)
        command.time_steps.fill_(frame)
        last = lengths[index] - 1
        command.obj_target_pos_w[:] = command.motion.obj_pos[index, last] + base.scene.env_origins
        command.obj_target_quat_w[:] = command.motion.obj_quat[index, last]
        anchor = command.anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        robot_anchor = command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        robot_anchor[..., 2] = anchor[..., 2]
        delta = yaw_quat(quat_mul(command.robot_anchor_quat_w, quat_inv(command.anchor_quat_w)))[:, None, :]
        delta = delta.expand(-1, len(command.cfg.body_names), -1)
        command.body_quat_relative_w[:] = quat_mul(delta, command.body_quat_w)
        command.body_pos_relative_w[:] = robot_anchor + quat_apply(delta, command.body_pos_w - anchor)

    print("REFINER_PAIR_PHASE=rollout_started", flush=True)
    initial_box_z = float(obj.data.root_pos_w[0, 2])
    for step in range(steps):
        frame0 = min(start_frame + step, lengths[0] - 1)
        frame1 = round(frame0 / (lengths[0] - 1) * (lengths[1] - 1))
        active = int(args.arm == "alternate" and step >= switch)
        frame = (frame0, frame1)[active]
        with torch.inference_mode():
            select_reference(active, frame)
            if aligned and step == switch:
                # One rigid reference coordinate transform on the selected
                # numeric demonstration. Original and alternate arms use the
                # identical causal rule; actual world/history remains intact.
                delta = quat_mul(yaw_quat(command.robot_anchor_quat_w),
                                 quat_inv(yaw_quat(command.anchor_quat_w)))[0]
                ref_box = command.motion.obj_pos[active,frame].clone()
                shift = obj.data.root_pos_w[0]-quat_apply(delta,ref_box)
                shift[2] = 0.
                def rotate(value):
                    return quat_apply(delta.expand(*value.shape[:-1],4),value)
                motion = command.motion
                motion._body_pos_w[active] = rotate(motion._body_pos_w[active])+shift
                motion._body_quat_w[active] = quat_mul(delta.expand_as(motion._body_quat_w[active]),motion._body_quat_w[active])
                motion._body_lin_vel_w[active] = rotate(motion._body_lin_vel_w[active])
                motion._body_ang_vel_w[active] = rotate(motion._body_ang_vel_w[active])
                motion.obj_pos[active] = rotate(motion.obj_pos[active])+shift
                motion.obj_quat[active] = quat_mul(delta.expand_as(motion.obj_quat[active]),motion.obj_quat[active])
                motion.obj_lin_vel[active] = rotate(motion.obj_lin_vel[active])
                motion.obj_ang_vel[active] = rotate(motion.obj_ang_vel[active])
                (out/'REFERENCE_TRANSFORM.json').write_text(json.dumps(dict(
                    source=sources[active],frame=frame,rotation_wxyz=delta.cpu().tolist(),
                    translation_xyz=shift.cpu().tolist(),actual_state_written=False),indent=2)+'\n')
                select_reference(active,frame)
            observation, action = teacher.action()
            original_actor_input=getattr(teacher,'last_actor_input',observation).clone()
            # Counterfactual action query on EXACTLY the same live state. Its
            # action is never executed; this alone is not future-data evidence.
            select_reference(1-active, (frame0, frame1)[1-active])
            other_observation, other_action = teacher.action()
            select_reference(active, frame)
            restored_observation, restored_action = teacher.action()
            if not torch.equal(observation, restored_observation) or not torch.equal(action, restored_action):
                restored_actor_input=getattr(teacher,'last_actor_input',restored_observation)
                np.savez(out/'QUERY_RESTORE_FAILURE.npz',original_input=original_actor_input.cpu().numpy(),restored_input=restored_actor_input.cpu().numpy(),original_action=action.cpu().numpy(),restored_action=restored_action.cpu().numpy(),original_observation=observation.cpu().numpy(),restored_observation=restored_observation.cpu().numpy())
                failure=dict(step=step,observation_exact=bool(torch.equal(observation,restored_observation)),actor_input_exact=bool(torch.equal(original_actor_input,restored_actor_input)),input_max_error=float((original_actor_input-restored_actor_input).abs().max()),action_max_error=float((action-restored_action).abs().max()),matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
                (out/'QUERY_RESTORE_FAILURE.json').write_text(json.dumps(failure,indent=2)+'\n')
                print(json.dumps(failure),flush=True)
                raise RuntimeError("Reference query failed exact restoration")
            if teacher_interface is not None:
                official_teacher_obs = base.observation_manager.compute_group('teacher',update_history=False)
                student_motion = command.motion
                try:
                    command.motion = command.teacher_motion
                    direct_teacher_obs, direct_teacher_action = teacher_interface.action()
                finally:
                    command.motion = student_motion
                observation_error = (official_teacher_obs-direct_teacher_obs).abs()
                from tensordict import TensorDict
                official_teacher_action = teacher_interface.actor.act_inference(TensorDict({'policy':official_teacher_obs},batch_size=[1]))
                control_obs, control_action = released_control.action()
                if not torch.equal(control_obs,observation):raise RuntimeError('Teacher query changed Tracker input')
                save('bcppo_teacher_observation',official_teacher_obs)
                save('direct_teacher_observation',direct_teacher_obs)
                save('teacher_observation_max_error',observation_error.max())
                save('teacher_action_max_error',(official_teacher_action-direct_teacher_action).abs().max())
                save('same_world_teacher_action',direct_teacher_action)
                save('same_world_released_tracker_action',control_action)
                save('nominal_tracker_action',action)
                handoff=args.teacher_handoff_step is not None and step>=args.teacher_handoff_step
                save('frozen_refiner_executed',handoff)
                if handoff:action=direct_teacher_action
            if generated_controller is not None:
                try:
                    action = generated_controller.action(step, observation, action, save)
                except Exception as error:
                    np.savez_compressed(out/'PARTIAL_TRACE.npz', **{k:np.asarray(v) for k,v in records.items()})
                    (out/'EXECUTION_FAILURE.json').write_text(json.dumps(dict(step_before_action=step,
                        completed_physics_steps=len(records.get('done', [])),generated_controls=generated_controller.calls,
                        error=repr(error)),indent=2)+'\n')
                    raise
            save("robot_body_state_before_w", robot.data.body_state_w)
            save("robot_root_state_before_w", robot.data.root_state_w)
            save("joint_pos_before", robot.data.joint_pos)
            save("joint_vel_before", robot.data.joint_vel)
            save("object_state_before_w", obj.data.root_state_w)
            save("teacher_observation", observation)
            save("requested_action", action)
            save("alternate_query_action", other_action)
            save("reference_id", sources[active])
            save("reference_frame", frame)
            save("reference_object_pos_w", command.obj_ref_pos_w)
            save("reference_body_pos_w", command.body_pos_w)
            if args.record_bc_replay:
                save('critic_observation',base.observation_manager.compute_group('critic',update_history=False))
            if args.record_bc_replay or (args.controller=='tracker' and teacher.paired_input):
                from scripts.sugar.demo_following.demo_future.frozen_tracker import planned_reference_commands
                save('planned_reference_command',planned_reference_commands(base))
                if getattr(teacher,'reference_feedback',None):
                    save('raw_reference_position_feedback',teacher.last_raw_reference_feedback)
                    save('actor_reference_position_feedback',teacher.last_reference_feedback)
            # Preserve full live substep history for the exact official
            # Refiner-to-motion contact-label conversion, separately from
            # latest-sample physical contact statistics and executed actions.
            save("contact_force_history_before_w", torch.stack([
                base.scene[name].data.force_matrix_w_history[:, :, 0, 0]
                for name in ("left_hand_forces", "right_hand_forces", "left_foot_forces", "right_foot_forces")], dim=1))
            save("reference_command", torch.cat([command.joint_pos, command.root_lin_vel_b,
                                                  command.root_ang_vel_b, command.contact_label[:, None]], dim=-1))
            save("target_object_pos_w", command.obj_target_pos_w)
            save("target_object_quat_w", command.obj_target_quat_w)
            save("project_gravity", quat_apply(quat_inv(command.robot_base_quat_w),
                                               torch.tensor([[0., 0., -1.]], device=base.device)))
            if step == switch:
                # Evaluate original terms independently: manager's episode
                # summary may retain only the last triggered term.
                pre_terms = {name: bool(term.func(base,**term.params).any())
                             for name,term in zip(base.termination_manager._term_names,
                                                  base.termination_manager._term_cfgs)}
                (out/'SWITCH_TERMS_BEFORE_ACTION.json').write_text(json.dumps(pre_terms,indent=2)+'\n')
            next_observations, reward, terminated, truncated, info = env.step(action)
            if args.record_bc_replay:
                save('reward',reward)
                for group in ('policy','critic','teacher'):
                    save('next_'+group+'_observation',next_observations[group])
            save("executed_action", base.action_manager.action)
            save("done", terminated | truncated)
            save("robot_body_state_after_w", robot.data.body_state_w)
            save("object_state_after_w", obj.data.root_state_w)
            forces = []
            for name in ("left_hand_forces", "right_hand_forces", "left_foot_forces", "right_foot_forces"):
                force = base.scene[name].data.force_matrix_w_history
                if force is None or force.shape[2:4] != (1, 1):
                    raise RuntimeError("Named body-to-box force shape drift")
                forces.append(force[:, -1, 0, 0])
            save("contact_force_after_w", torch.stack(forces, dim=1))
            if writer is not None and step % 2 == 0 and not (terminated | truncated).any():
                rgb = base.scene['world_camera'].data.output['rgb'][0,...,:3].cpu().numpy().copy()
                acting='Refiner (handoff)' if args.teacher_handoff_step is not None and step>=args.teacher_handoff_step else args.controller
                label=f"Frozen {acting} | actual PhysX | {args.arm} | demo {sources[active]} | {(step+1)*.02:.2f}s"
                cv2.putText(rgb,label,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(rgb,'Camera-enabled diagnostic; not camera-free trace replay',(12,47),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
                writer.append_data(rgb)
                if step % 100 == 0:
                    imageio.imwrite(out/f'WORLD_{step:04d}.png',rgb)
            if not torch.equal(action, base.action_manager.action) and not (terminated | truncated).any():
                raise RuntimeError("Requested action differs from action-manager execution")
        if step % 50 == 0:
            print(f"REFINER_PAIR_STEP={step} reference={sources[active]} frame={frame}", flush=True)
        if (terminated | truncated).any():
            # The final post-state may be auto-reset; explicit done field makes
            # it ineligible for any future label or behavior success statistic.
            break
    arrays = {k: np.asarray(v) for k, v in records.items()}
    if teacher_interface is not None:
        observation_error=float(arrays['teacher_observation_max_error'].max())
        action_error=float(arrays['teacher_action_max_error'].max())
        labels=arrays['same_world_teacher_action']
        audit=dict(execution_completed=True,frames=len(labels),observation_max_error=observation_error,
                   action_max_error=action_error,passed=observation_error<=1e-5 and action_error<=1e-5,
                   direct_refiner_frozen=teacher_interface.frozen_audit(),
                   released_action_teacher_mse=float(np.mean((arrays['same_world_released_tracker_action']-labels)**2)),
                   active_action_teacher_mse=float(np.mean((arrays['nominal_tracker_action']-labels)**2)),
                   scope='Same actual state, complete official890-D teacher inputs and frozen actor; no optimization. Agreement does not prove teacher recovery from arbitrary student states.')
        (out/'TEACHER_INTERFACE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    if writer is not None:
        writer.close()
    np.savez_compressed(out / "TRACE.npz", **arrays)
    valid = ~arrays["done"].reshape(-1)
    z = arrays["object_state_after_w"][:, 0, 2][valid]
    lifted = z > initial_box_z + .05
    frozen = teacher.frozen_audit()
    result = dict(execution_completed=True, actual_steps=len(valid), reset_or_termination=bool((~valid).any()),
                  all_numeric_finite=all(np.isfinite(v).all() for v in arrays.values()),
                  full_budget_without_reset=bool(valid.all() and len(valid) == steps),
                  maximum_lift_m=float((z-initial_box_z).max()) if len(z) else None,
                  at_least_ten_consecutive_lifted_frames=bool(len(lifted) >= 10 and np.convolve(lifted.astype(int), np.ones(10,dtype=int), 'valid').max() == 10),
                  teacher_frozen=frozen, actual_alternate_reference_executed=bool((arrays['reference_id']==sources[1]).any()),
                  scope=protocol['scope'])
    if generated_controller is not None:
        result['generator_frozen'] = generated_controller.frozen_audit()
        result['generated_control_steps'] = generated_controller.calls
    (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    # Evidence is flushed before bypassing the known slow Kit teardown. The
    # process exits, leaving the retained allocation shell alive.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
