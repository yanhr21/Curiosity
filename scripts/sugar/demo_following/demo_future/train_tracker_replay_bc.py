"""Fixed actual-rollout adapter to the repository's full official BCPPO BC stage.

This is offline action-fit diagnosis, not new on-policy sampling or PPO. Every
observation, reward and executed action comes from recorded real PhysX. Returns
and student log-probabilities are computed only for the unchanged storage API;
the original BC-only loss gives them no surrogate/value/entropy credit.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from sugar_rl.utils.rsl_rl_bcppo import BCPPO

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/demo_following/demo_future_smp_v1'
OUT=BASE/'tracker_teacher_supported_fixed_replay_bc'


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--paired-input',choices=['current_only','future_plan'],help='Matched two-trajectory experiment with full official actor plus288 zero-initialized input columns')
    parser.add_argument('--paired-original-source',type=Path,help='Completed original real trajectory; missing plans require the independently passed known-reference extraction audit')
    parser.add_argument('--coverage-arm',choices=['replay_control','aggregate_recovery'])
    args=parser.parse_args()
    coverage=args.coverage_arm is not None
    if coverage and args.paired_input!='future_plan':raise RuntimeError('Coverage retains the future-plan interface')
    updates=128 if coverage else 256
    seed=272042 if coverage else 272041
    OUT=args.output.resolve()
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Use retained compute step')
    source=BASE/'tracker_bcppo_refined_pair_low_exploration/teacher_supported_replay'
    result=json.loads((source/'RESULT.json').read_text())
    interface=json.loads((source/'TEACHER_INTERFACE_AUDIT.json').read_text())
    if not result['full_budget_without_reset'] or not interface['passed']:
        raise RuntimeError('Actual successful teacher-supported source and exact interface required')
    raw=dict(np.load(source/'TRACE.npz',allow_pickle=False))
    prior=dict(np.load(source.parent/'teacher_handoff100/TRACE.npz',allow_pickle=False))
    same={key:bool(np.array_equal(raw[key],prior[key])) for key in
          ('robot_body_state_before_w','object_state_before_w','teacher_observation','executed_action')}
    if not all(same.values()) or raw['done'].any():raise RuntimeError('Replay recording changed the accepted recovery trajectory')
    num_envs,n=16,384
    paired=args.paired_input is not None
    if paired:
        audit=json.loads((source.parent/'CURRENT_INPUT_FUTURE_TARGET_AUDIT.json').read_text())
        if not audit['current_input_ambiguity_observed']:raise RuntimeError('No observed input ambiguity')
        if args.paired_original_source is None:raise RuntimeError('Specify the completed original trajectory source')
        sources=[args.paired_original_source.resolve(),source.parent/'teacher_supported_alternate']
        records=[]
        for path in sources:
            if not json.loads((path/'RESULT.json').read_text())['full_budget_without_reset'] or not json.loads((path/'TEACHER_INTERFACE_AUDIT.json').read_text())['passed']:
                raise RuntimeError('Both actual teacher-supported sources must pass')
            records.append(dict(np.load(path/'TRACE.npz',allow_pickle=False)))
        live_same={k:bool(np.array_equal(records[0][k],raw[k])) for k in raw}
        if not all(live_same.values()):raise RuntimeError('Original source changed existing trajectory fields')
        packet_audit=json.loads((source.parent/'KNOWN_REFERENCE_PLAN_AUDIT.json').read_text())
        if not packet_audit['passed']:raise RuntimeError('Known-reference plan extraction failed')
        packets=np.load(source.parent/'KNOWN_REFERENCE_PLANS.npz',allow_pickle=False)
        # Use the same reference extraction in both arms and both trajectories.
        # The original stream was not recorded live; alternate full packets and
        # both current-command streams independently validate this extraction.
        for name,record in zip(['original','alternate'],records):record['planned_reference_command']=packets[name]
        for record in records:
            if record['planned_reference_command'].shape!=(400,1,8,36) or record['done'].any():raise RuntimeError('Invalid live future plan')
            if np.max(abs(record['planned_reference_command'][:,:,0]-record['reference_command']))>1e-5:raise RuntimeError('Plan starts at a different current command')
        # Both traces have one unannounced branch at158. Before that actor call
        # the selected reference remains original, including the future packet.
        for key in ['teacher_observation','planned_reference_command','executed_action','robot_body_state_before_w']:
            if not np.array_equal(records[0][key][:158],records[1][key][:158]):raise RuntimeError('Different pre-selection history or plan')
        if coverage:
            from scripts.sugar.demo_following.demo_future.tracker_coverage import recovery_pool
            records,sources,common_records,common_sources=recovery_pool(BASE,records,sources,args.coverage_arm)
        raw={key:np.concatenate([r[key][:384] for r in records]) for key in records[0]}
        next_plan=np.concatenate([r['planned_reference_command'][1:385] for r in records])
        num_envs,n=16*len(records),384*len(records)
    OUT.mkdir(exist_ok=False)
    plan=dict(kind='Fixed-data full official Tracker action-fit diagnostic',updates=256,
              unique_actual_transitions=384,replay_layout='16 logical24-frame segments from the first384 actual frames of one400-frame trajectory; these are not16 physical rollouts.',
              source=str(source),source_recollection_exact=same,seed=272041,learning_rate=1e-4,
              architecture='Released complete ActorCritic510/890 ->512/256/128 ->29/1; strict whole-model warm start',
              algorithm='Unmodified repository BCPPO.update and RolloutStorage, initial BC-only KL(T||S) objective,5epochs/4minibatches; no PPO/value/entropy credit',
              label='Exact frozen Refiner action distribution queried on each actual state via recorded890-D teacher observation; saved29-D executed behavior actions remain distinct.',
              budget_boundary='256 fixed-data updates, all before official BC boundary500. This is not a budget-matched online64-update treatment.',
              fit_checks={'all384_teacher_mean_action_mse_at_most':.002,'p90_frame_action_mse_at_most':.005},
              automatic_next_action='Freeze model255 and evaluate the same400-step real PhysX original arm, regardless of offline fit. Preserve all results; fit alone is not physical success.',
              simulated_training_transitions=0,actual_source_transitions=400)
    plan['numerical_precision']='FP32 matmul throughout this fixed-data diagnostic. Measure batched TF32 teacher difference before updates; no threshold relaxation.'
    if paired:
        plan.update(kind='Matched full official Tracker future-plan input adaptation',paired_input=args.paired_input,
                    source=[str(p) for p in sources],unique_actual_transitions=n,actual_source_transitions=800,
                    replay_layout='32 logical24-frame segments, first384 frames from each of two actual400-frame trajectories; not32 physical rollouts.',
                    architecture='Full official ActorCritic798/890 ->512/256/128 ->29/1, all released parameters retained; only actor first layer gains288 zero columns.',
                    only_arm_difference='Future288-D known-reference input is zero in current_only, actual selected8x36 plan in future_plan. Identical complete initialization/data/budget/optimizer/minibatch RNG.',
                    original_source_fields_unchanged=live_same,
                    future_plan_extraction_audit=packet_audit,
                    future_plan_causality='Derived from the already selected known reference and recorded current frame. Full alternate live packets and both current streams match within1e-5; original full packet was not recorded live. No future actual simulator states; no alternate plan before selection158.',
                    automatic_next_action='Freeze model255 and evaluate original/repeat/alternate400-step PhysX for both arms, including failed arms for matched diagnosis; inspect terminal state, independent following and actual video. No additional update follows automatically.')
        plan['fit_checks']['each_branchpoint_teacher_action_mse_at_most']=.002
    if coverage:
        parent_path=BASE/'matched_tracker_future_plan/future_plan/model_255.pt'
        plan.update(kind='Matched full Tracker actual-state coverage continuation',coverage_arm=args.coverage_arm,updates=128,seed=seed,parent=str(parent_path),
                    unique_actual_transitions=768 if args.coverage_arm=='replay_control' else 1536,
                    actual_source_transitions=800 if args.coverage_arm=='replay_control' else 1600,
                    entries_per_update=n,replay_layout='64 logical24-frame segments, four384-frame blocks. Control repeats old pair; treatment uses old plus recovery pair. Not64 physical environments.',
                    architecture='Unchanged complete official ActorCritic798/890 ->512/256/128 ->29/1 from future model255',
                    only_arm_difference='Actual state coverage: repeated old pair versus old plus recovery pair. Same future interface, complete parent weights/Adam, matched new RNG seed and128-update budget.',
                    budget_boundary='Restore BCPPO256, apply128 BC-only updates to384, below500. Parent RNG absent: both explicitly reseeded272042.',
                    future_plan_causality='Old pair uses validated reference-derived packets, new pair uses recorded live known-reference packets. No future simulator-state actor inputs.',
                    common_evaluation_sources=[str(x) for x in common_sources],
                    automatic_next_action='Freeze model383 and independently evaluate all matched original/repeat/alternate400-step student endpoints. Inspect failures before further updates.')
    (OUT/'PROTOCOL.json').write_text(json.dumps(plan,indent=2)+'\n')
    device=torch.device('cuda:0');torch.set_num_threads(8);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    fields={'policy':'teacher_observation','critic':'critic_observation','teacher':'bcppo_teacher_observation'}
    tensors={group:torch.as_tensor(raw[key][:n,0],device=device) for group,key in fields.items()}
    next_tensors={group:torch.as_tensor(raw['next_'+group+'_observation'][:n,0],device=device) for group in fields}
    if paired:
        tensors['future_plan']=torch.as_tensor(raw['planned_reference_command'][:,0].reshape(n,288),device=device)
        next_tensors['future_plan']=torch.as_tensor(next_plan[:,0].reshape(n,288),device=device)
        if args.paired_input=='current_only':
            tensors['future_plan']=torch.zeros_like(tensors['future_plan']);next_tensors['future_plan']=torch.zeros_like(next_tensors['future_plan'])
    actions=torch.as_tensor(raw['executed_action'][:n,0],device=device)
    rewards=torch.as_tensor(raw['reward'][:n,0],device=device)
    dones=torch.as_tensor(raw['done'][:n,0],device=device)
    recorded_labels=torch.as_tensor(raw['same_world_teacher_action'][:n,0],device=device)
    indices=torch.arange(num_envs,device=device)*24
    def batch(data,ids):return TensorDict({k:v[ids] for k,v in data.items()},batch_size=[len(ids)])
    observation=batch(tensors,indices)
    policy=ActorCritic(observation,{'policy':['policy','future_plan'] if paired else ['policy'],'critic':['critic'],'teacher':['teacher']},29,
                       actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',
                       actor_obs_normalization=False,critic_obs_normalization=False,init_noise_std=.5).to(device)
    released=torch.load(parent_path if coverage else ROOT/'SUGAR/demo_ckpts/CarryBox/tracker.pt',map_location=device,weights_only=False)
    if coverage and (released.get('infos') or {}).get('paired_input')!='future_plan':raise RuntimeError('Wrong coverage parent interface')
    warm={k:v.clone() for k,v in released['model_state_dict'].items()}
    first='actor.0.weight'
    if paired and not coverage:
        if warm[first].shape!=(512,510):raise RuntimeError('Released first layer differs')
        warm[first]=torch.cat([warm[first],torch.zeros(512,288,device=device)],dim=1)
    policy.load_state_dict(warm,strict=True)
    initial={k:v.clone() for k,v in policy.state_dict().items()}
    if not all(torch.equal(v,warm[k]) for k,v in initial.items()):raise RuntimeError('Full released warm start differs')
    if paired and not coverage:
        with torch.no_grad():
            from rsl_rl.networks import MLP
            baseline=MLP(510,29,[512,256,128],'elu').to(device)
            baseline.load_state_dict({k.removeprefix('actor.'):v for k,v in released['model_state_dict'].items() if k.startswith('actor.')},strict=True)
            before=policy.act_inference(TensorDict(tensors,batch_size=[n]))
            baseline_error=float((before-baseline(tensors['policy'])).abs().max())
        preflight=dict(released_initial_action_max_error=baseline_error,all_released_parameters_exact=True,new_input_columns_exact_zero=bool((initial[first][:,510:]==0).all()),optimizer_updates=0)
        (OUT/'INITIALIZATION_PREFLIGHT.json').write_text(json.dumps(preflight,indent=2)+'\n')
        if baseline_error>1e-5:raise RuntimeError('Extended full actor fails initial released-action equivalence')
    teacher_path=ROOT/'experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt'
    alg=BCPPO(policy,teacher_ckpt=str(teacher_path),device=device,num_learning_epochs=5,num_mini_batches=4,
              clip_param=.2,gamma=.99,lam=.95,value_loss_coef=1.,entropy_coef=.005,
              learning_rate=1e-4,max_grad_norm=1.,use_clipped_value_loss=True,schedule='adaptive',desired_kl=.01)
    if coverage:
        from scripts.sugar.demo_following.demo_future.tracker_coverage import restore_optimizer
        resume=restore_optimizer(alg,released)
        resume['whole_model_state_exact']=all(torch.equal(v,released['model_state_dict'][k]) for k,v in policy.state_dict().items())
        if not resume['whole_model_state_exact']:raise RuntimeError('Full parent weights changed on restoration')
        (OUT/'RESUME_PREFLIGHT.json').write_text(json.dumps(resume,indent=2)+'\n')
        print(json.dumps(resume),flush=True)
    initial_update=alg.update_step
    teacher_before={k:v.clone() for k,v in alg.teacher_model.state_dict().items()}
    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32=True
        tf32_labels=alg.teacher_model(tensors['teacher'])
        torch.backends.cuda.matmul.allow_tf32=False
        labels=alg.teacher_model(tensors['teacher'])
    label_error=float((labels-recorded_labels).abs().max())
    precision=dict(tf32_batched_teacher_max_error=float((tf32_labels-recorded_labels).abs().max()),
                   fp32_batched_teacher_max_error=label_error,
                   tf32_batched_teacher_mean_squared_error=float((tf32_labels-recorded_labels).square().mean()),
                   fp32_batched_teacher_mean_squared_error=float((labels-recorded_labels).square().mean()),
                   required_max_error=1e-4,optimizer_updates=0)
    (OUT/'TEACHER_PRECISION_PREFLIGHT.json').write_text(json.dumps(precision,indent=2)+'\n')
    print(json.dumps(precision),flush=True)
    if label_error>1e-4:raise RuntimeError('Offline official teacher differs from actual same-world query')
    alg.init_storage('rl',num_envs,24,observation,[29])
    def snapshot(index):
        torch.save(dict(model_state_dict=policy.state_dict(),optimizer_state_dict=alg.optimizer.state_dict(),
                        iter=index,infos=dict(bcppo_update_step=alg.update_step,scope=plan['kind'],paired_input=args.paired_input)),
                   OUT/f'model_{index}.pt')
    snapshot(-1)
    @torch.no_grad()
    def fit():
        prediction=policy.act_inference(TensorDict(tensors,batch_size=[n]))
        per_frame=(prediction-labels).square().mean(-1)
        result=dict(teacher_mean_action_mse=float(per_frame.mean()),
                    p90_frame_action_mse=float(torch.quantile(per_frame,.9)),
                    mean_exploration_std=float(policy.std.mean()))
        if paired:result['branchpoint_action_mse']=[float(per_frame[i]) for i in range(158,n,384)]
        return result
    if coverage:
        common={group:torch.as_tensor(np.concatenate([r[key][:384,0] for r in common_records]),device=device) for group,key in fields.items()}
        common['future_plan']=torch.as_tensor(np.concatenate([r['planned_reference_command'][:384,0].reshape(384,288) for r in common_records]),device=device)
        common_recorded=torch.as_tensor(np.concatenate([r['same_world_teacher_action'][:384,0] for r in common_records]),device=device)
        with torch.no_grad():common_labels=alg.teacher_model(common['teacher'])
        if float((common_labels-common_recorded).abs().max())>1e-4:raise RuntimeError('Common teacher evaluation labels differ')
        @torch.no_grad()
        def common_fit():
            predictions=policy.act_inference(TensorDict(common,batch_size=[1536]));error=(predictions-common_labels).square().mean(-1)
            return dict(teacher_mean_action_mse=float(error.mean()),p90_frame_action_mse=float(torch.quantile(error,.9)),source_mse=[float(v.mean()) for v in error.reshape(4,384)],branchpoint_action_mse=error[torch.arange(4,device=device)*384+158].tolist())
        common_initial=common_fit()
    start=fit();trace=[]
    with open(OUT/'UPDATES.jsonl','x') as log:
        for local_update in range(updates):
            update=initial_update+local_update
            if alg.update_step>=alg.bc_only_steps:raise RuntimeError('Offline data must never receive PPO credit')
            with torch.no_grad():
                for step in range(24):
                    ids=indices+step;obs=batch(tensors,ids);following=batch(next_tensors,ids)
                    alg.act(obs)  # Distribution/value bookkeeping only; sampled action is never executed.
                    alg.transition.actions=actions[ids]
                    alg.transition.actions_log_prob=policy.get_actions_log_prob(actions[ids]).detach()
                    alg.process_env_step(following,rewards[ids],dones[ids],{})
                alg.compute_returns(following)
            loss=alg.update()
            row=dict(update=update,bcppo_update_step=alg.update_step,loss=loss)
            if update%16==0 or local_update==updates-1:row['fixed_data_fit']=fit();print(json.dumps(row),flush=True)
            log.write(json.dumps(row)+'\n');log.flush();trace.append(row)
            if local_update in (63,127,255) or local_update==updates-1:snapshot(update)
    final=fit()
    if paired:
        saved=tensors['future_plan']
        ablations={}
        for name,value in [('zero',torch.zeros_like(saved)),('other_reference_same_phase',saved.reshape(-1,2,384,288).flip(1).reshape(n,288))]:
            tensors['future_plan']=value
            ablations[name]=fit()
        tensors['future_plan']=saved
        (OUT/'FUTURE_INPUT_ABLATION.json').write_text(json.dumps(ablations,indent=2)+'\n')
    frozen=all(torch.equal(v,teacher_before[k]) for k,v in alg.teacher_model.state_dict().items())
    critic_equal=all(torch.equal(v,initial[k]) for k,v in policy.state_dict().items() if k.startswith('critic.'))
    clocks=sorted(set(int(v['step']) for v in alg.optimizer.state.values() if 'step'in v))
    with torch.no_grad():predictions=policy.act_inference(TensorDict(tensors,batch_size=[n]))
    np.savez(OUT/'FIT_PREDICTIONS.npz',prediction=predictions.cpu().numpy(),teacher_mean=labels.cpu().numpy(),
             actual_executed_action=actions.cpu().numpy(),source_frame=np.arange(n)%384,source_arm=np.arange(n)//384)
    report=dict(execution_completed=True,bcppo_updates=alg.update_step,unique_actual_transitions=plan['unique_actual_transitions'],
                newly_applied_updates=updates,initial_bcppo_update=initial_update,entries_per_update=n,
                replayed_transition_uses=n*updates,optimizer_steps=clocks,teacher_action_query_max_error=label_error,
                frozen_teacher_unchanged=frozen,critic_unchanged=critic_equal,initial_fit=start,final_fit=final,
                all_parameters_finite=all(bool(torch.isfinite(v).all()) for v in policy.state_dict().values()),
                fit_passed=final['teacher_mean_action_mse']<=.002 and final['p90_frame_action_mse']<=.005 and (not paired or max(final['branchpoint_action_mse'])<=.002),
                scope=plan['kind'],next_action=plan['automatic_next_action'])
    if coverage:
        report['common_initial_fit']=common_initial;report['common_final_fit']=common_fit()
        report['fit_passed']=report['fit_passed'] and report['common_final_fit']['teacher_mean_action_mse']<=.002 and report['common_final_fit']['p90_frame_action_mse']<=.005 and max(report['common_final_fit']['branchpoint_action_mse'])<=.002
        with torch.no_grad():common_predictions=policy.act_inference(TensorDict(common,batch_size=[1536]))
        np.savez(OUT/'COMMON_FIT_PREDICTIONS.npz',prediction=common_predictions.cpu().numpy(),teacher_mean=common_labels.cpu().numpy(),source_frame=np.arange(1536)%384,source_arm=np.arange(1536)//384)
        if clocks!=[7680] or alg.update_step!=384:raise RuntimeError('Coverage optimizer clocks or applied budget failed')
    (OUT/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
