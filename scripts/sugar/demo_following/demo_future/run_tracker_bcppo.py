"""Bounded official full Tracker BCPPO adaptation and frozen physical endpoint.

Runs repository SUGAR train.py, BCPPO, ActorCritic, official observations and
checkpoints. Changes only declared data, nominal conditions and bounded budget.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/demo_following/demo_future_smp_v1'
OUT=BASE/'tracker_bcppo_refined_pair_v1'


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--initial-exploration-std',type=float)
    cli=parser.parse_args();OUT=cli.output.resolve()
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Retained compute step required')
    for arm in ('original','alternate'):
        r=json.loads((BASE/f'aligned_original_teacher_feasibility/{arm}/RESULT.json').read_text())
        if not (r['full_budget_without_reset'] and r['teacher_frozen']['passed'] and r['at_least_ten_consecutive_lifted_frames']):
            raise RuntimeError('Original teacher bank failed physical floor; diagnose before training')
    if cli.initial_exploration_std is not None:
        audit=json.loads((BASE/'tracker_bcppo_refined_pair_v1/teacher_interface_audit/TEACHER_INTERFACE_AUDIT.json').read_text())
        if not audit['passed']:raise RuntimeError('Resolve teacher interface mismatch before exploration experiment')
        if cli.initial_exploration_std<=0:raise ValueError('Exploration std must be positive')
    OUT.mkdir(exist_ok=False)
    student=BASE/'refiner_aligned96_90_feasibility/refined_motion_export/rl_dataset'
    teacher=BASE/'refiner_aligned96_90_feasibility/aligned_original_teacher_bank'
    teacher_ckpt=ROOT/'experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt'
    released=ROOT/'SUGAR/demo_ckpts/CarryBox/tracker.pt'
    plan=dict(purpose='Full official Tracker execution overfit prerequisite for selected-demo Generator research',
              updates=64,num_envs=16,steps_per_env=24,seed=272041,learning_rate=1e-4,
              architecture='Complete official512/256/128 ActorCritic,510-D Tracker observation,29-D actions',
              algorithm='Repository official-default BCPPO KL teacher distillation, all64 updates in its initial500-update BC stage; no PPO surrogate/value credit',
              initialization='Strict complete released ActorCritic weights; fresh optimizer and update clock, not optimizer continuation',
              student_motion=str(student),teacher_motion=str(teacher),teacher_checkpoint=str(teacher_ckpt),
              method_variant='Two TRAIN reference timelines; nominal physics;16 environments;64 updates;lr1e-4. No architecture reduction, SMP reward, new Generator or world model.',
              endpoint='Frozen same400-step nominal seed272012 physics. If original passes run repeat/alternate and independent reference readback. A loss decrease alone is insufficient.',
              evaluation_reference_alignment='Same one-time causal yaw/boxXY rule as released Tracker baseline; training uses the fixed prepared reference banks without an extra mid-episode alignment.',
              original_artifacts_preserved=True)
    plan['initial_exploration_std_override']=cli.initial_exploration_std
    if cli.initial_exploration_std is not None:
        plan['matched_control']=str(BASE/'tracker_bcppo_refined_pair_v1')
        plan['single_changed_factor']='Initial exploration std only. Complete actor/critic mean weights, seed, optimizer,64-update budget, data, teacher, physics and frozen endpoint are unchanged; on-policy sampled histories may differ as the intended consequence.'
    (OUT/'PROTOCOL.json').write_text(json.dumps(plan,indent=2)+'\n')
    sugar=ROOT/'SUGAR'
    os.environ.update(ISAACLAB_GROUND_PLANE_USD=str(sugar/'descriptions/terrain/sugar_ground_plane.usda'),
                     ISAACLAB_USE_LOCAL_FRAME_MARKER='1',SUGAR_DISABLE_TRAIN_DEBUG_VIS='1',
                     SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT='1',SUGAR_INIT_AT_RANDOM_EP_LEN='0',
                     VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',DISPLAY='',HYDRA_FULL_ERROR='1')
    if cli.initial_exploration_std is not None:
        os.environ['SUGAR_ACTOR_CRITIC_WARM_START_EXPLORATION_STD']=str(cli.initial_exploration_std)
    else:
        os.environ.pop('SUGAR_ACTOR_CRITIC_WARM_START_EXPLORATION_STD',None)
    children=[]
    def execute(tag,args,cwd=ROOT):
        log=OUT/f'{tag}.log'
        with open(log,'xb') as stream:
            child=subprocess.Popen(args,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
            children.append(dict(tag=tag,pid=child.pid,pgid=os.getpgid(child.pid),command=args))
            (OUT/'CHILDREN.json').write_text(json.dumps(children,indent=2)+'\n')
            print(f'STAGE={tag} PID={child.pid} PGID={os.getpgid(child.pid)} LOG={log}',flush=True)
            code=child.wait()
        children[-1]['exit_code']=code
        (OUT/'CHILDREN.json').write_text(json.dumps(children,indent=2)+'\n')
        if code:raise RuntimeError(f'{tag} exited{code}; inspect its retained log before retry')
    args=[sys.executable,str(sugar/'scripts/sugar_rl/train.py'),'--task','Sugar-G129dof-CarryBox-Tracker',
          '--num_envs','16','--seed','272041','--max_iterations','64','--headless',
          '--motion_folder',str(student),'--teacher_motion_folder',str(teacher),
          '--teacher_ckpt',str(teacher_ckpt),'--actor_critic_warm_start_checkpoint_path',str(released),
          '--log_dir',str(OUT/'training'),'--kit_args','--/renderer/enabled= --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false',
          'agent.algorithm.learning_rate=0.0001','agent.save_interval=32',
          'env.commands.motion.use_generator=false','env.commands.motion.start_init_env_ratio=1.0',
          'env.commands.motion.joint_position_range=[0.0,0.0]',
          'env.commands.motion.pose_range={x:[0.0,0.0],y:[0.0,0.0],z:[0.0,0.0],roll:[0.0,0.0],pitch:[0.0,0.0],yaw:[0.0,0.0]}',
          'env.events.push_robot=null','env.events.push_object=null','env.episode_length_s=8.0',
          'env.observations.policy.enable_corruption=false','env.observations.critic.enable_corruption=false',
          'env.events.obj_mass.params.mass_distribution_params=[1.0,1.0]']
    for material in ('robot_physics_material','obj_physics_material'):
        for key,value in (('static_friction_range','[1.0,1.0]'),('dynamic_friction_range','[1.0,1.0]'),('restitution_range','[0.0,0.0]')):
            args.append(f'env.events.{material}.params.{key}={value}')
    execute('training',args,sugar)
    import torch
    endpoint=OUT/'training/model_63.pt'
    checkpoint=torch.load(endpoint,map_location='cpu',weights_only=False)
    if checkpoint['iter']!=63 or not checkpoint['optimizer_state_dict']['state']:
        raise RuntimeError('Missing complete64-update endpoint/optimizer')
    if not all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values()):
        raise RuntimeError('Nonfinite endpoint')
    evaluation=OUT/'evaluation';evaluation.mkdir()
    eval_plan=json.loads((BASE/'tracker_refined96_90_feasibility/PROTOCOL.json').read_text())
    eval_plan.update(checkpoint_origin='Project64-update full official BCPPO Tracker adaptation',checkpoint=str(endpoint))
    (evaluation/'PROTOCOL.json').write_text(json.dumps(eval_plan,indent=2)+'\n')
    (evaluation/'motions').symlink_to(student,target_is_directory=True)
    bootstrap=ROOT/'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
    def module(tag,name,*extra):execute(tag,[sys.executable,str(bootstrap),f'scripts.sugar.demo_following.demo_future.{name}',*map(str,extra)])
    module('eval_original','collect_refiner_pair','--headless','--controller','tracker','--tracker-checkpoint',endpoint,'--run',evaluation,'--arm','original')
    original=json.loads((evaluation/'original/RESULT.json').read_text())
    passes=original['full_budget_without_reset'] and original['at_least_ten_consecutive_lifted_frames']
    if passes:
        for arm in ('repeat','alternate'):
            module(f'eval_{arm}','collect_refiner_pair','--headless','--controller','tracker','--tracker-checkpoint',endpoint,'--run',evaluation,'--arm',arm)
        module('pair_readback','readback_refiner_pair','--run',evaluation)
        pairing=json.loads((evaluation/'READBACK.json').read_text())
        passes=pairing['data_pair_feasibility_passed']
        if passes:
            module('reference_following','readback_reference_following','--run',evaluation)
            module('official_smp','score_refiner_smp','--run',evaluation)
    result=dict(execution_completed=True,actual_updates=64,endpoint=str(endpoint),
                original_physical_result=original,data_pair_feasibility_passed=bool(passes),
                scope='Single TRAIN-pair execution diagnostic, not new Generator/SMP/Zero-WAM success.',
                next_action='Inspect physical/reference endpoints and actual visuals before Generator data admission.' if passes else 'Diagnose failed full Tracker endpoint before another bounded stage; no automatic budget extension.')
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
