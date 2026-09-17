"""Inference glue for the released full SUGAR Tracker actor and official observations."""
from pathlib import Path
import json
import torch
from rsl_rl.networks.mlp import MLP


def initialize_relative_reference(command):
    """Populate the official MotionCommand derived cache without a clock/physics step.

    Same formulas as the final block of MotionCommand._update_command, whose
    full invocation also advances its clock and initialization pool.
    """
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat
    count=len(command.cfg.body_names)
    anchor_pos=command.anchor_pos_w[:,None,:].repeat(1,count,1)
    anchor_quat=command.anchor_quat_w[:,None,:].repeat(1,count,1)
    robot_pos=command.robot_anchor_pos_w[:,None,:].repeat(1,count,1)
    robot_quat=command.robot_anchor_quat_w[:,None,:].repeat(1,count,1)
    robot_pos[...,2]=anchor_pos[...,2]
    delta=yaw_quat(quat_mul(robot_quat,quat_inv(anchor_quat)))
    command.body_quat_relative_w=quat_mul(delta,command.body_quat_w)
    command.body_pos_relative_w=robot_pos+quat_apply(delta,command.body_pos_w-anchor_pos)


def planned_reference_commands(env, steps=8, stride=5):
    """Read the already-selected reference plan, never future simulator state."""
    from isaaclab.utils.math import quat_apply_inverse
    command=env.command_manager.get_term('motion')
    motion=command.motion
    ids=command.motion_id[:,None].expand(-1,steps)
    times=command.time_steps[:,None]+stride*torch.arange(steps,device=env.device)[None,:]
    times=torch.minimum(times,motion.time_step_total_permotion[command.motion_id,None]-1)
    quat=motion.body_quat_w[ids,times,0]
    lin=quat_apply_inverse(quat,motion.body_lin_vel_w[ids,times,0])
    ang=quat_apply_inverse(quat,motion.body_ang_vel_w[ids,times,0])
    return torch.cat([motion.joint_pos[ids,times],lin,ang,motion.contact_label[ids,times,None]],dim=-1)


def reference_position_feedback(env):
    """Official relative reference positions, requiring current causal localization.

    Eight consecutive50-Hz known reference frames, in current robot/object
    coordinates. No future actual state and no invented observation formula.
    This additional interface is distinct from the released Tracker contract.
    """
    from sugar_rl.tasks.locomanip.mdp.observations import motion_anchor_pos_b_future, obj_motion_pos_future
    packet=torch.cat([motion_anchor_pos_b_future(env,'motion'),obj_motion_pos_future(env,'motion')],dim=1)
    if packet.shape!=(env.num_envs,48) or not torch.isfinite(packet).all():
        raise RuntimeError('Official reference-position feedback contract changed')
    return packet


class FrozenReleasedTracker:
    def __init__(self, env, checkpoint: Path, origin='Author-released SUGAR Tracker actor', position_feedback_provider=None):
        self.env = env
        self.origin = origin
        payload = torch.load(checkpoint, map_location=env.device, weights_only=True)
        self.paired_input=(payload.get('infos') or {}).get('paired_input')
        self.reference_feedback=(payload.get('infos') or {}).get('reference_feedback')
        if self.reference_feedback not in (None,'zero_feedback','reference_feedback') or (self.reference_feedback and self.paired_input!='future_plan'):
            raise RuntimeError('Unknown reference-feedback interface')
        if self.paired_input not in (None,'current_only','future_plan'):
            raise RuntimeError('Unknown full Tracker future-plan interface')
        if position_feedback_provider is not None and self.reference_feedback != 'reference_feedback':
            raise RuntimeError('A position source comparison requires the full trained reference-feedback interface')
        self.position_feedback_provider = position_feedback_provider
        state = {k.removeprefix('actor.'): v for k, v in payload['model_state_dict'].items()
                 if k.startswith('actor.')}
        self.actor = MLP(input_dim=846 if self.reference_feedback else (798 if self.paired_input else 510), output_dim=29, hidden_dims=[512, 256, 128],
                         activation='elu').to(env.device)
        self.actor.load_state_dict(state, strict=True)
        self.actor.eval().requires_grad_(False)
        self.initial = {k: v.clone() for k, v in self.actor.state_dict().items()}
        self.warmup_audit=None
        if self.paired_input:
            # Official scripted quaternion math can change its floating-point
            # fusion after initial calls on the new8-step shape. Warm read-only
            # inference until inputs/actions stabilize; keep exact query restore
            # checks and never advance observation history or physics here.
            world={name:env.scene[name].data.root_state_w.clone() for name in ('robot','obj')}
            previous=None;streak=0;rows=[]
            for index in range(8):
                obs,action=self.action()
                current=(self.last_actor_input.clone(),action.clone())
                if previous is not None:
                    input_error=float((current[0]-previous[0]).abs().max())
                    action_error=float((current[1]-previous[1]).abs().max())
                    exact=torch.equal(current[0],previous[0]) and torch.equal(current[1],previous[1])
                    streak=streak+1 if exact else 0
                    rows.append(dict(call=index+1,input_max_error=input_error,action_max_error=action_error,exact=exact))
                    if max(input_error,action_error)>1e-5:raise RuntimeError('Tracker read-only warmup changed beyond numerical tolerance')
                previous=current
                if streak>=3:break
            unchanged=all(torch.equal(value,env.scene[name].data.root_state_w) for name,value in world.items())
            self.warmup_audit=dict(calls=index+1,consecutive_exact=streak,world_unchanged=unchanged,comparisons=rows,passed=streak>=3 and unchanged,physics_steps=0,observation_history_updates=0)
            print(json.dumps({'tracker_readonly_warmup':self.warmup_audit}),flush=True)
            if not self.warmup_audit['passed']:raise RuntimeError('Tracker read-only inference did not stabilize')

    @torch.inference_mode()
    def action(self):
        # The environment advances history once per physical step. Reference
        # queries recompute current command terms without advancing history.
        obs = self.env.observation_manager.compute_group('policy', update_history=False)
        if obs.shape != (self.env.num_envs, 510) or not torch.isfinite(obs).all():
            raise RuntimeError('Released Tracker observation contract failed')
        actor_obs=obs
        if self.paired_input:
            future=planned_reference_commands(self.env).flatten(1)
            if future.shape!=(self.env.num_envs,288) or not torch.isfinite(future).all():
                raise RuntimeError('Invalid causal future reference packet')
            if self.paired_input=='current_only':future=torch.zeros_like(future)
            actor_obs=torch.cat([obs,future],dim=-1)
        if self.reference_feedback:
            feedback=reference_position_feedback(self.env) if self.position_feedback_provider is None else self.position_feedback_provider(self.env)
            if feedback.shape!=(self.env.num_envs,48) or not torch.isfinite(feedback).all():
                raise RuntimeError('Invalid full reference-position feedback packet')
            self.last_raw_reference_feedback=feedback
            if self.reference_feedback=='zero_feedback':feedback=torch.zeros_like(feedback)
            self.last_reference_feedback=feedback
            actor_obs=torch.cat([actor_obs,feedback],dim=-1)
        self.last_actor_input=actor_obs
        action = self.actor(actor_obs)
        if action.shape != (self.env.num_envs, 29) or not torch.isfinite(action).all():
            raise RuntimeError('Released Tracker action contract failed')
        return obs, action

    def frozen_audit(self):
        changed = [k for k, v in self.actor.state_dict().items() if not torch.equal(v, self.initial[k])]
        trainable = [k for k, p in self.actor.named_parameters() if p.requires_grad]
        gradients = [k for k, p in self.actor.named_parameters() if p.grad is not None]
        return dict(passed=not (changed or trainable or gradients), changed_state_keys=changed,
                    requires_grad_parameter_names=trainable, gradient_parameter_names=gradients,
                    optimizer_constructed=False, origin=self.origin)
