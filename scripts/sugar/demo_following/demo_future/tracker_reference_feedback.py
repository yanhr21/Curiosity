"""State-preserving input/Adam expansion and official position-packet readback."""
import copy
import torch

from scripts.sugar.demo_following.demo_future.tracker_coverage import equal_state


def restore_feedback_parent(alg,parent):
    if parent['infos']['bcppo_update_step']!=384:raise RuntimeError('Expected online384 parent')
    payload=copy.deepcopy(parent)
    key='actor.0.weight';old=parent['model_state_dict'][key]
    if old.shape!=(512,798):raise RuntimeError('Full parent input shape changed')
    payload['model_state_dict'][key]=torch.cat([old,old.new_zeros(512,48)],dim=1)
    names={id(p):name for name,p in alg.policy.named_parameters()}
    changed=[]
    for current,stored in zip(alg.optimizer.param_groups,payload['optimizer_state_dict']['param_groups'],strict=True):
        for parameter,identifier in zip(current['params'],stored['params'],strict=True):
            if names[id(parameter)]==key:
                state=payload['optimizer_state_dict']['state'][identifier]
                for moment in ('exp_avg','exp_avg_sq'):
                    value=state[moment]
                    if value.shape!=old.shape:raise RuntimeError('Parent moment geometry changed')
                    state[moment]=torch.cat([value,value.new_zeros(512,48)],dim=1)
                changed.append(identifier)
    if len(changed)!=1:raise RuntimeError('Ambiguous first-layer Adam mapping')
    alg.policy.load_state_dict(payload['model_state_dict'],strict=True)
    alg.optimizer.load_state_dict(payload['optimizer_state_dict']);alg.update_step=384
    if not equal_state(alg.policy.state_dict(),payload['model_state_dict']) or not equal_state(alg.optimizer.state_dict(),payload['optimizer_state_dict']):
        raise RuntimeError('Expanded whole model/Adam restoration failed')
    trimmed=copy.deepcopy(payload)
    trimmed['model_state_dict'][key]=trimmed['model_state_dict'][key][:,:798]
    for moment in ('exp_avg','exp_avg_sq'):
        trimmed['optimizer_state_dict']['state'][changed[0]][moment]=trimmed['optimizer_state_dict']['state'][changed[0]][moment][:,:798]
    old_exact=equal_state(trimmed['model_state_dict'],parent['model_state_dict']) and equal_state(trimmed['optimizer_state_dict'],parent['optimizer_state_dict'])
    clocks=sorted({int(v['step']) for v in alg.optimizer.state.values() if 'step' in v})
    zeros=not bool(payload['model_state_dict'][key][:,798:].any()) and all(not bool(payload['optimizer_state_dict']['state'][changed[0]][m][:,798:].any()) for m in ('exp_avg','exp_avg_sq'))
    if not old_exact or not zeros or clocks!=[7680] or any(g['lr']!=1e-4 for g in alg.optimizer.param_groups):raise RuntimeError('Parent state preservation audit failed')
    return payload,dict(passed=True,old_model_and_optimizer_exact=old_exact,new_columns_and_moments_zero=zeros,bcppo_update_step=384,optimizer_clocks=clocks,newly_applied_updates=0,rng='Explicit new matched seed272043; no parent RNG continuation claim')


@torch.inference_mode()
def audit_position_packet(env):
    from isaaclab.utils.math import quat_apply_inverse
    from scripts.sugar.demo_following.demo_future.frozen_tracker import reference_position_feedback
    command=env.command_manager.get_term('motion');motion=command.motion
    ids,times=command.get_future_index()
    anchor=motion.body_pos_w[ids,times,command.motion_anchor_body_index]+env.scene.env_origins[:,None,:]
    obj=motion.obj_pos[ids,times]+env.scene.env_origins[:,None,:]
    robot_quat=command.robot_anchor_quat_w[:,None,:].expand(-1,8,-1)
    obj_quat=command.obj_quat_w[:,None,:].expand(-1,8,-1)
    derived=torch.cat([quat_apply_inverse(robot_quat,anchor-command.robot_anchor_pos_w[:,None,:]).flatten(1),quat_apply_inverse(obj_quat,obj-command.obj_pos_w[:,None,:]).flatten(1)],dim=1)
    packet=reference_position_feedback(env)
    error=float((derived-packet).abs().max())
    return dict(passed=packet.shape==(env.num_envs,48) and error<=1e-5,max_absolute_error=error,shape=list(packet.shape),reference_offsets=list(range(8)),physical_steps=0,scope='Independent official inverse-rotation readback against unchanged official observation functions; known reference positions relative to current measured pose.')
