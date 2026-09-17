"""Full official OnPolicyRunner/BCPPO with live known-reference future input.

Project glue only: no replacement policy, simulator, optimizer or PPO objective.
The optional preflight executes24 actual steps without updating any weight.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/demo_following/demo_future_smp_v1'


def live_future_packet(env):
    from scripts.sugar.demo_following.demo_future.frozen_tracker import planned_reference_commands
    return planned_reference_commands(env).flatten(1)


def live_position_packet(env,arm):
    import torch
    from scripts.sugar.demo_following.demo_future.frozen_tracker import reference_position_feedback
    value=reference_position_feedback(env)
    return torch.zeros_like(value) if arm=='zero_feedback' else value


def main():
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Use recorded retained compute step')
    sugar=ROOT/'SUGAR'
    os.environ.update(ISAACLAB_GROUND_PLANE_USD=str(sugar/'descriptions/terrain/sugar_ground_plane.usda'),
                     ISAACLAB_USE_LOCAL_FRAME_MARKER='1',SUGAR_DISABLE_TRAIN_DEBUG_VIS='1',
                     VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',DISPLAY='')
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--robot-usd',type=Path,required=True)
    parser.add_argument('--feedback-arm',choices=('zero_feedback','reference_feedback'))
    parser.add_argument('--coverage-arm',choices=('pair_control','broad_train'))
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.robot_usd=args.robot_usd.resolve()
    coverage=args.coverage_arm is not None
    if coverage and args.feedback_arm is not None:parser.error('Coverage uses the same true reference feedback in both arms')
    if coverage:args.feedback_arm='reference_feedback'
    feedback=args.feedback_arm is not None
    updates=128 if coverage else (96 if feedback else 128)
    start_update=480 if coverage else (384 if feedback else 256)
    end_update=start_update+updates
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    args.kit_args='--/renderer/enabled= --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false'
    portable=Path(f'/tmp/Curiosity_online_future_{os.environ["SLURM_JOB_ID"]}_{os.getpid()}')
    portable.mkdir();sys.argv+=['--portable-root',str(portable)]
    app=AppLauncher(args,multi_gpu=False,max_gpu_count=1).app
    import builtins
    import numpy as np
    import torch
    import gymnasium as gym
    import isaaclab_tasks
    import sugar_rl.tasks
    import rsl_rl.algorithms
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.utils import configclass
    from sugar_rl.utils.parser_cfg import parse_env_cfg
    from sugar_rl.utils.rsl_rl_bcppo import BCPPO
    from sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg import BCPPORunnerCfg
    from sugar_rl.utils.official_refiner_nominal_teacher import FrozenOfficialRefinerTeacher
    from scripts.sugar.demo_following.demo_future.robot_asset import use_converted_robot_usd
    from scripts.sugar.demo_following.demo_future.frozen_tracker import FrozenReleasedTracker, initialize_relative_reference
    from scripts.sugar.demo_following.demo_future.tracker_coverage import restore_optimizer, equal_state
    setattr(builtins,'BCPPO',BCPPO);setattr(rsl_rl.algorithms,'BCPPO',BCPPO)
    os.chdir(sugar)
    torch.set_num_threads(8);seed=272049 if coverage else (272043 if feedback else 272042);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    parent=BASE/'matched_tracker_future_plan/future_plan/model_255.pt'
    if feedback:parent=BASE/'online_tracker_future128/training/model_383.pt'
    if coverage:parent=BASE/'matched_reference_feedback96/reference_feedback/training/model_479.pt'
    teacher_path=ROOT/'experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt'
    student=BASE/'refiner_aligned96_90_feasibility/refined_motion_export/rl_dataset'
    teacher_bank=BASE/'refiner_aligned96_90_feasibility/aligned_original_teacher_bank'
    num_envs=64
    coverage_plan=None
    if coverage:
        coverage_root=BASE/'matched_train_coverage128'
        coverage_plan=json.loads((coverage_root/'PROTOCOL.json').read_text())
        if not coverage_plan['banks_prepared'] or coverage_plan['new_updates_per_arm']!=128:
            raise RuntimeError('Completed predeclared TRAIN banks required')
        num_envs=coverage_plan['resolved_num_envs']
        student=coverage_root/args.coverage_arm/'student_motion_bank'
        teacher_bank=coverage_root/args.coverage_arm/'original_teacher_bank'
    @configclass
    class FutureGroup(ObservationGroupCfg):
        packet=ObservationTermCfg(func=live_future_packet)
    @configclass
    class PositionGroup(ObservationGroupCfg):
        packet=ObservationTermCfg(func=live_position_packet,params={'arm':args.feedback_arm})
    cfg=parse_env_cfg('Sugar-G129dof-CarryBox-Tracker',device=args.device,num_envs=num_envs)
    asset=use_converted_robot_usd(cfg.scene.robot,args.robot_usd.resolve())
    cfg.seed=seed;cfg.episode_length_s=8.
    cfg.commands.motion.motion_folder=str(student);cfg.commands.motion.teacher_motion_folder=str(teacher_bank)
    cfg.commands.motion.use_generator=False;cfg.commands.motion.start_init_env_ratio=1.
    cfg.commands.motion.pose_range={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.commands.motion.joint_position_range=(0.,0.)
    cfg.events.push_robot=None;cfg.events.push_object=None
    for name in ('robot_physics_material','obj_physics_material'):
        getattr(cfg.events,name).params.update(static_friction_range=(1.,1.),dynamic_friction_range=(1.,1.),restitution_range=(0.,0.))
    cfg.events.obj_mass.params['mass_distribution_params']=(1.,1.)
    for group in ('policy','critic','teacher'):getattr(cfg.observations,group).enable_corruption=False
    cfg.observations.future_plan=FutureGroup(concatenate_terms=True,enable_corruption=False)
    if feedback:cfg.observations.reference_feedback=PositionGroup(concatenate_terms=True,enable_corruption=False)
    observation_groups=('policy','future_plan','teacher','critic')+(('reference_feedback',) if feedback else ())
    env=gym.make('Sugar-G129dof-CarryBox-Tracker',cfg=cfg)
    base=env.unwrapped;wrapped=RslRlVecEnvWrapper(env)
    command=base.command_manager.get_term('motion')
    with torch.inference_mode():
        for _ in range(8):
            for name in observation_groups:
                base.observation_manager.compute_group(name,update_history=False)
        world_before={name:base.scene[name].data.root_state_w.clone() for name in ('robot','obj')}
        body_before=base.scene['robot'].data.body_state_w.clone()
        clock_before=command.time_steps.clone()
        groups_before={name:base.observation_manager.compute_group(name,update_history=False).clone() for name in observation_groups}
        initial_terms={name:int(term.func(base,**term.params).sum()) for name,term in zip(base.termination_manager._term_names,base.termination_manager._term_cfgs)}
        initial_cache_max=float(command.body_pos_relative_w.abs().max())
        initialize_relative_reference(command)
        refreshed_terms={name:int(term.func(base,**term.params).sum()) for name,term in zip(base.termination_manager._term_names,base.termination_manager._term_cfgs)}
        cache_audit=dict(initial_term_counts=initial_terms,refreshed_term_counts=refreshed_terms,initial_relative_position_cache_max_abs=initial_cache_max,
                         clock_exact=torch.equal(clock_before,command.time_steps),body_state_exact=torch.equal(body_before,base.scene['robot'].data.body_state_w),
                         root_states_exact=all(torch.equal(v,base.scene[name].data.root_state_w) for name,v in world_before.items()),
                         observation_groups_exact={name:torch.equal(v,base.observation_manager.compute_group(name,update_history=False)) for name,v in groups_before.items()},
                         physics_steps=0,scope='Initialize only the official derived relative-reference cache after native reset; no actual world/history/time write.')
        cache_audit['passed']=cache_audit['clock_exact'] and cache_audit['body_state_exact'] and cache_audit['root_states_exact'] and all(cache_audit['observation_groups_exact'].values()) and not any(refreshed_terms.values())
        (out/'REFERENCE_CACHE_INITIALIZATION.json').write_text(json.dumps(cache_audit,indent=2)+'\n')
        if not cache_audit['passed']:raise RuntimeError('Reference cache initialization changed inputs/world or leaves invalid initial terms')
    agent=BCPPORunnerCfg();agent.seed=seed;agent.device=args.device;agent.save_interval=64
    agent.obs_groups={'policy':['policy','future_plan'],'critic':['critic'],'teacher':['teacher']}
    if feedback:agent.obs_groups['policy'].append('reference_feedback')
    agent.algorithm.teacher_ckpt=str(teacher_path);agent.algorithm.learning_rate=1e-4
    agent.policy.actor_obs_normalization=False;agent.policy.critic_obs_normalization=False
    runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=str(out),device=args.device)
    payload=torch.load(parent,map_location=args.device,weights_only=True)
    if coverage:
        from scripts.sugar.demo_following.demo_future.tracker_reference_feedback import audit_position_packet
        if payload['infos']['bcppo_update_step']!=480 or payload['model_state_dict']['actor.0.weight'].shape!=(512,846):
            raise RuntimeError('Expected complete reference-feedback480 parent')
        runner.alg.policy.load_state_dict(payload['model_state_dict'],strict=True)
        runner.alg.optimizer.load_state_dict(payload['optimizer_state_dict']);runner.alg.update_step=480
        clocks=sorted({int(v['step']) for v in runner.alg.optimizer.state.values() if 'step' in v})
        if not equal_state(runner.alg.optimizer.state_dict(),payload['optimizer_state_dict']) or clocks!=[9600]:
            raise RuntimeError('Complete480 Adam restoration failed')
        if (runner.alg.bc_only_steps,runner.alg.critic_warmup_steps,runner.alg.full_ppo_warmup_steps)!=(500,1000,2000):
            raise RuntimeError('Official BCPPO stage boundaries changed')
        if any(group['lr']!=1e-4 for group in runner.alg.optimizer.param_groups):raise RuntimeError('Parent learning rate changed')
        resume=dict(passed=True,old_model_and_optimizer_exact=True,bcppo_update_step=480,optimizer_clocks=clocks,newly_applied_updates=0,
                    rng='Explicit matched seed272049; no parent RNG continuation claim')
    elif feedback:
        from scripts.sugar.demo_following.demo_future.tracker_reference_feedback import restore_feedback_parent,audit_position_packet
        payload,resume=restore_feedback_parent(runner.alg,payload)
    else:
        runner.alg.policy.load_state_dict(payload['model_state_dict'],strict=True)
        resume=restore_optimizer(runner.alg,payload)
    runner.current_learning_iteration=start_update
    runner.alg.learning_rate=1e-4
    if not equal_state(runner.alg.policy.state_dict(),payload['model_state_dict']):raise RuntimeError('Full online initial state differs')
    teacher_initial={k:v.clone() for k,v in runner.alg.teacher_model.state_dict().items()}
    critic_initial={k:v.clone() for k,v in runner.alg.policy.state_dict().items() if k.startswith('critic.')}
    frozen=FrozenReleasedTracker(base,parent)
    with torch.inference_mode():
        obs=wrapped.get_observations()
        expected={'policy':(num_envs,510),'future_plan':(num_envs,288),'teacher':(num_envs,890),'critic':(num_envs,890)}
        if feedback:expected['reference_feedback']=(num_envs,48)
        if any(tuple(obs[k].shape)!=shape for k,shape in expected.items()):raise RuntimeError('Online observation contract drift')
        _,reference_action=frozen.action()
        actor_error=float((runner.alg.policy.act_inference(obs)-reference_action).abs().max())
        direct=FrozenOfficialRefinerTeacher(base,teacher_path,expected_sha256=None)
        command=base.command_manager.get_term('motion');motion=command.motion
        try:
            command.motion=command.teacher_motion
            teacher_obs,teacher_action=direct.action()
        finally:command.motion=motion
        teacher_obs_error=float((obs['teacher']-teacher_obs).abs().max())
        teacher_action_error=float((runner.alg.teacher_model(obs['teacher'])-teacher_action).abs().max())
        if feedback:
            from scripts.sugar.demo_following.demo_future.frozen_tracker import reference_position_feedback
            packet_audit=audit_position_packet(base)
            if not packet_audit['passed']:raise RuntimeError('Official feedback packet readback failed')
            (out/'POSITION_PACKET_AUDIT.json').write_text(json.dumps(packet_audit,indent=2)+'\n')
            np.savez(out/'INITIAL_OBSERVATIONS.npz',**{k:v.cpu().numpy() for k,v in obs.items()},parent_action=reference_action.cpu().numpy(),adapted_action=runner.alg.policy.act_inference(obs).cpu().numpy(),raw_reference_feedback=reference_position_feedback(base).cpu().numpy())
    if max(actor_error,teacher_obs_error,teacher_action_error)>1e-5:raise RuntimeError('Online full-model interface preflight failed')
    protocol=dict(kind='Full official online BCPPO future-input state-coverage diagnostic',parent=str(parent),seed=seed,num_envs=num_envs,steps_per_update=24,updates=0 if args.preflight_only else updates,
                  actor='Complete official798/890 ActorCritic,512/256/128 hidden layers,29 executed motor actions',resume=resume,asset_adapter=asset,
                  only_runtime_adapter='Additional known-reference288-D observation group; unmodified official OnPolicyRunner.learn and BCPPO.update',
                  data=f'Every update uses current-policy stochastic actions in{num_envs} actual PhysX environments. No replay transitions or teacher action execution.',
                  reference_schedule='Native official training motion selection and fixed prepared banks, reference chosen from episode start. No extra mid-episode alignment in training; frozen evaluation keeps its original unannounced158-frame reference switch and causal alignment.',
                  precision='FP32, same explicit new RNG seed as offline comparison; no claim of parent RNG continuation',
                  scope='Exploratory online-state coverage. Training references/reset schedule differ from fixed replay; no single-factor causal claim from offline-versus-online outcome.',
                  reference_cache_initialization=cache_audit,actor_interface_max_error=actor_error,teacher_observation_max_error=teacher_obs_error,teacher_action_max_error=teacher_action_error)
    if feedback:
        protocol.update(kind='Matched full official Tracker reference-position feedback',feedback_arm=args.feedback_arm,updates=0 if args.preflight_only else updates,
                        actor='Complete official846/890 ActorCritic,512/256/128 hidden layers,29 actual motor actions',
                        only_runtime_adapter='Original future288 plus48 exact official reference-position terms, zero versus true. Full model and Adam expansion with old values exact. Existing native cache initialization retained.',
                        scope='Matched input comparison on a TRAIN pair; causal current-localization-dependent new interface. Native reference schedule differs from unannounced frozen switch; no broad deployment or generalization claim.',position_packet_audit=packet_audit)
    if coverage:
        ids=command.motion_id.detach().cpu().tolist()
        if sorted(set(ids))!=list(range(num_envs)) or command.motion.num_motion!=num_envs:
            raise RuntimeError('Not every predeclared reference slot has an actual environment')
        protocol.update(kind='Matched broader TRAIN state coverage with full official846-D Tracker',
                        coverage_arm=args.coverage_arm,num_envs=num_envs,updates=0 if args.preflight_only else 128,
                        only_runtime_adapter='Existing complete846-D actor and Adam restored unchanged; only the predeclared TRAIN reference distribution changes.',
                        source_coverage_protocol=str(coverage_root/'PROTOCOL.json'),initial_reference_slots=ids,
                        reference_schedule='Official fixed env_id modulo motion_count native initialization.4N environments cover all4N predeclared reference slots; half are original pair, half broader TRAIN references in treatment.',
                        stage_schedule='480..499 pure BC,500..607 unchanged official critic warmup plus full BC; no PPO surrogate or entropy credit.',
                        scope='Matched TRAIN reference-distribution experiment; both arms use true reference feedback and exact same full480 parent/Adam. New source slots intentionally have different physical starts. Validation remains optimizer-free.')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps(protocol),flush=True)
    np.savez(out/'INITIAL_STATE.npz',robot_root=base.scene['robot'].data.root_state_w.cpu().numpy(),object_root=base.scene['obj'].data.root_state_w.cpu().numpy(),reference_id=command.motion_id.cpu().numpy(),reference_frame=command.time_steps.cpu().numpy())
    original_save=runner.save
    def save(path,infos=None):
        metadata=dict(infos or {},paired_input='future_plan',bcppo_update_step=runner.alg.update_step,scope=protocol['kind'])
        if feedback:metadata['reference_feedback']=args.feedback_arm
        if coverage:metadata['coverage_arm']=args.coverage_arm
        return original_save(path,metadata)
    runner.save=save
    actual_steps=0;terminations=0;terminal_records=[];preflight_records=[];preflight_trace={};position_checks=[]
    termination_compute=base.termination_manager.compute
    def inspect_termination():
        value=termination_compute()
        if value.any():
            terms={name:int(term.func(base,**term.params).sum()) for name,term in zip(base.termination_manager._term_names,base.termination_manager._term_cfgs)}
            terminal_records.append(dict(control_step=actual_steps,terminated_environments=int(value.sum()),terms=terms,
                                         reference_frames=command.time_steps.detach().cpu().tolist()))
            if args.preflight_only and not (out/'FIRST_TERMINAL_STATE.npz').exists():
                np.savez(out/'FIRST_TERMINAL_STATE.npz',robot_root=base.scene['robot'].data.root_state_w.cpu().numpy(),object_root=base.scene['obj'].data.root_state_w.cpu().numpy(),reference_anchor=command.anchor_pos_w.cpu().numpy(),actual_anchor=command.robot_anchor_pos_w.cpu().numpy(),reference_object=command.obj_ref_pos_w.cpu().numpy(),done=value.cpu().numpy())
        return value
    base.termination_manager.compute=inspect_termination
    original_step=wrapped.step
    def step(actions):
        nonlocal actual_steps,terminations
        if actions.shape!=(num_envs,29) or not torch.isfinite(actions).all():raise RuntimeError('Invalid actual online action')
        if feedback and args.preflight_only:
            values=dict(actions=actions,robot_root=base.scene['robot'].data.root_state_w,object_root=base.scene['obj'].data.root_state_w,body_state=base.scene['robot'].data.body_state_w,reference_id=command.motion_id,reference_frame=command.time_steps)
            for name,value in values.items():preflight_trace.setdefault(name,[]).append(value.detach().cpu().numpy().copy())
        output=original_step(actions);actual_steps+=1;terminations+=int(output[2].sum())
        if not all(torch.isfinite(output[0][k]).all() for k in output[0].keys()):raise RuntimeError('Nonfinite online observation')
        if args.preflight_only:
            preflight_records.append(dict(control_step=actual_steps,terminations=int(output[2].sum()),reference_frame_min=int(command.time_steps.min()),reference_frame_max=int(command.time_steps.max())))
            if feedback:
                check=audit_position_packet(base);position_checks.append(check)
                if not check['passed']:raise RuntimeError('Live feedback geometry drift')
        return output
    wrapped.step=step
    torch.manual_seed(seed)
    if args.preflight_only:
        with torch.inference_mode():
            for _ in range(24):
                actions=runner.alg.policy.act(obs)
                obs,_,_,_=wrapped.step(actions)
        model_exact=equal_state(runner.alg.policy.state_dict(),payload['model_state_dict'])
        optimizer_exact=equal_state(runner.alg.optimizer.state_dict(),payload['optimizer_state_dict'])
        result=dict(execution_completed=True,passed=model_exact and optimizer_exact,actual_env_steps=actual_steps,actual_transitions=actual_steps*num_envs,reset_terminations=terminations,optimizer_updates=0,model_unchanged=model_exact,optimizer_unchanged=optimizer_exact,scope=f'Actual{num_envs}-environment full-interface preflight, not policy success.')
        if feedback:
            runner._prepare_logging_writer();runner.current_learning_iteration=start_update-1
            runner.save(str(out/f'model_{start_update-1}.pt'))
            np.savez(out/'PREFLIGHT_TRACE.npz',**{k:np.asarray(v) for k,v in preflight_trace.items()})
            result['position_readback_max_error']=max(x['max_absolute_error'] for x in position_checks)
    else:
        preflight_path=out.parent/'preflight/RESULT.json' if feedback else out.parent/'preflight_initialized/RESULT.json'
        if not json.loads(preflight_path.read_text())['passed']:raise RuntimeError('Actual initialized online preflight incomplete')
        if feedback and not json.loads((out.parent.parent/'MATCHED_PREFLIGHT.json').read_text())['passed']:raise RuntimeError('Both actual matched preflights required')
        # Official save expects the logger fields normally initialized by learn.
        # Prepare its existing idempotent writer before our initial-state save.
        runner._prepare_logging_writer()
        runner.current_learning_iteration=start_update-1
        runner.save(str(out/f'model_{start_update-1}.pt'))
        runner.current_learning_iteration=start_update
        stage_records=[]
        if coverage:
            optimizer_parameter_names={id(p):name for name,p in runner.alg.policy.named_parameters()}
            def optimizer_clocks_by_role():
                roles={'actor':set(),'critic':set()}
                for parameter,state in runner.alg.optimizer.state.items():
                    if 'step' in state:
                        role='critic' if optimizer_parameter_names[id(parameter)].startswith('critic.') else 'actor'
                        roles[role].add(int(state['step']))
                return {k:sorted(v) for k,v in roles.items()}
            original_update=runner.alg.update
            def audited_update(*args,**kwargs):
                before=runner.alg.update_step
                losses=original_update(*args,**kwargs)
                if before in (480,499,500,501,607):
                    row=dict(applied_update_index=before,bcppo_update_step=runner.alg.update_step,
                             optimizer_clocks=optimizer_clocks_by_role(),
                             critic_unchanged=all(torch.equal(runner.alg.policy.state_dict()[k],v) for k,v in critic_initial.items()),
                             all_model_values_finite=all(torch.isfinite(v).all().item() for v in runner.alg.policy.state_dict().values()),
                             stage='pure_bc' if before<500 else 'critic_warmup_plus_bc')
                    if not row['all_model_values_finite']:raise RuntimeError('Nonfinite coverage update')
                    stage_records.append(row)
                    (out/'OFFICIAL_STAGE_READBACK.json').write_text(json.dumps(stage_records,indent=2)+'\n')
                return losses
            runner.alg.update=audited_update
        runner.learn(num_learning_iterations=updates,init_at_random_ep_len=False)
        clocks=sorted({int(v['step']) for v in runner.alg.optimizer.state.values() if 'step' in v})
        frozen_teacher=all(torch.equal(v,teacher_initial[k]) for k,v in runner.alg.teacher_model.state_dict().items())
        frozen_critic=all(torch.equal(runner.alg.policy.state_dict()[k],v) for k,v in critic_initial.items())
        if coverage:
            role_clocks=optimizer_clocks_by_role()
            if role_clocks!={'actor':[12160],'critic':[2160]} or frozen_critic:
                raise RuntimeError('Original critic-warmup optimizer roles or expected critic update differ')
        elif clocks!=[end_update*20] or not frozen_critic:
            raise RuntimeError('Pure-BC optimizer clocks or frozen critic differ')
        if actual_steps!=updates*24 or runner.alg.update_step!=end_update or not frozen_teacher:raise RuntimeError('Online endpoint budget or frozen-teacher audit failed')
        runner.save(str(out/f'model_{end_update-1}.pt'))
        result=dict(execution_completed=True,newly_applied_updates=updates,bcppo_updates=end_update,optimizer_clocks=clocks,actual_env_steps=actual_steps,actual_transitions=actual_steps*num_envs,reset_terminations=terminations,teacher_unchanged=frozen_teacher,critic_unchanged=frozen_critic,scope=protocol['scope'],next_action=f'Freeze model{end_update-1} and run all original/repeat/alternate400-step independent mean-action evaluations, including failures.')
        if coverage:result.update(optimizer_clocks_by_role=role_clocks,coverage_arm=args.coverage_arm,next_action='Freeze full608 endpoint; all three TRAIN-pair arms and all eight fixed native validation sources, with no validation updates.')
    (out/'TERMINATION_READBACK.json').write_text(json.dumps(dict(terminal_events=terminal_records,step_records=preflight_records),indent=2)+'\n')
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    sys.stdout.flush();os._exit(0)


if __name__=='__main__':main()
