"""Full official Generator-to-Tracker glue for a bounded reference-assisted diagnostic.

Only the current36 and future288 command entries are generated. The48 position
entries retain the declared known reference; this is not independent deployment.
"""
import json
from pathlib import Path

import dill
import numpy as np
import torch

from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
from scripts.sugar.demo_following.demo_future.generator_demo_geometry import CONTEXT_KEY, restore_geometry_state
from scripts.sugar.demo_following.demo_future.generator_tracker_routing import IssuedCommandHistory, compose_generated_tracker_input

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'


def prepare_live_input(wrapper, raw, previous, context):
    obs = wrapper._prepare_obs_dict(raw)
    obs['last_action'] = previous
    obs['joint_pos'] = raw.joint_pos
    obs['project_gravity'] = raw.project_gravity
    obs[CONTEXT_KEY] = context
    assert all(v.shape[:2] == (1, 1) and torch.isfinite(v).all() for v in obs.values())
    return obs


class GeneratedRolloutController:
    def __init__(self, env, tracker, run, arm, out):
        self.env, self.tracker, self.out = env, tracker, Path(out)
        self.plan = json.loads((Path(run) / 'PROTOCOL.json').read_text())['generated_rollout']
        self.switch = self.plan['handoff_control_frame']
        assert tracker.paired_input == 'future_plan' and tracker.reference_feedback == 'reference_feedback'
        assert tracker.position_feedback_provider is None
        model_run = Path(self.plan['generator_run'])
        self.mode = self.plan.get('generator_condition_mode', 'demo_geometry')
        assert self.mode in ('demo_geometry', 'zero_context')
        model_plan = json.loads((model_run / 'PROTOCOL.json').read_text())
        assert model_plan['goal_condition_mode'] == 'zero_diagnostic'
        self.wrapper = GeneratorWrapper.load(str(ROOT / 'SUGAR/demo_ckpts/CarryBox/generator.ckpt'), device=env.device)
        with (model_run / self.mode / 'checkpoints/endpoint.ckpt').open('rb') as f:
            state = torch.load(f, pickle_module=dill, map_location=env.device)['state_dicts']['model']
        audit = restore_geometry_state(self.wrapper.policy, state, self.mode, 'zero_diagnostic')
        assert audit['full_parameter_count'] == 8327408
        self.wrapper.policy.eval().requires_grad_(False)
        self.initial = {k: v.detach().clone() for k, v in self.wrapper.policy.state_dict().items()}
        assert self.wrapper.policy.num_inference_steps == 16
        self.branch = int(arm == 'alternate')
        with np.load(Path(run) / 'ORIGINAL_DEMO_CONTEXT.npz') as data:
            self.context = torch.as_tensor(data['geometry'][self.branch], device=env.device)
        with np.load(Path(self.plan['baseline_run']) / 'branch_samples/BRANCH_SAMPLES.npz') as data:
            self.reference_input = {k: torch.as_tensor(data[k][self.branch:self.branch+1], device=env.device) for k in
                ('obj_pos_b', 'obj_ori_b', 'joint_pos', 'project_gravity', 'last_action', CONTEXT_KEY)}
        self.prefix, self.history, self.calls = [], None, 0
        self.audit = dict(full_restore=audit, sample_seed_rule=self.plan['sample_seed_rule'], new_optimizer_updates=0,
            generated_fields='current36 + future288 only', position_fields='unchanged known measured reference48',
            observation_history='actual issued36 command at control t-5', handoff_control_frame=self.switch)
        (self.out / 'GENERATOR_RESTORE.json').write_text(json.dumps(self.audit, indent=2)+'\n')

    @torch.inference_mode()
    def action(self, step, observation, known_action, save):
        command = self.env.command_manager.get_term('motion')
        issued = observation[:, :36].clone()
        if step < self.switch:
            self.prefix.append(issued)
            self.prefix = self.prefix[-5:]
            save('generated_control', False)
            save('issued_command', issued)
            return known_action
        if self.history is None:
            assert step == self.switch and len(self.prefix) == 5
            self.history = IssuedCommandHistory(step, torch.stack(self.prefix, dim=1))
        previous = self.history.previous_10hz_command(step)
        # Built-in Generator scheduling is disabled; provide the real issued
        # history required by its unchanged observation getter, then restore it.
        had_history = hasattr(command, 'last_command')
        old_history = getattr(command, 'last_command', None)
        try:
            command.last_command = previous.clone()
            raw = command._get_generator_obs()
        finally:
            if had_history: command.last_command = old_history
            else: del command.last_command
        context = self.context[step].reshape(1, 1, 168)
        obs = prepare_live_input(self.wrapper, raw, previous, context)
        if step == self.switch:
            errors = {k: float((obs[k]-v).abs().max()) for k, v in self.reference_input.items()}
            report = dict(max_absolute_error=errors, passed=all(v <= 1e-5 for v in errors.values()),
                          scope='Live first handoff against saved actual branch inputs; no future state used.')
            (self.out/'HANDOFF_INPUT_CHECK.json').write_text(json.dumps(report, indent=2)+'\n')
            assert report['passed'], report
        # Isolate diffusion sampling from the simulator/global RNG and pair all arms.
        seed = self.plan['sample_seed_base'] + step-self.switch
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            predicted = self.wrapper.policy.predict_action(obs)
        if step == self.switch:
            variant = 'trained_demo_geometry_correct' if self.mode == 'demo_geometry' else 'trained_zero_context'
            with np.load(Path(self.plan['generator_run'])/f'frozen_evaluation/phase_158/{variant}.npz') as data:
                expected = torch.as_tensor(data['predictions'][0,self.branch], device=self.env.device)
                assert int(data['sample_seeds'][0]) == seed
            error = float((predicted[0]-expected).abs().max())
            (self.out/'FIRST_GENERATED_SAMPLE_CHECK.json').write_text(json.dumps(dict(
                passed=error<=1e-4,max_absolute_error=error,seed=seed),indent=2)+'\n')
            assert error<=1e-4, error
        dense = self.wrapper._parse_action(predicted)
        positions = self.tracker.last_reference_feedback.clone()
        actor_input = compose_generated_tracker_input(observation, dense, positions, plan_frame=step, control_frame=step)
        assert torch.equal(actor_input[:, 36:510], observation[:, 36:])
        assert torch.equal(actor_input[:, 798:], positions)
        action = self.tracker.actor(actor_input)
        assert action.shape == (1, 29) and torch.isfinite(action).all()
        self.history.record_issued(step, dense[:, 0])
        self.calls += 1
        save('generated_control', True)
        save('issued_command', dense[:, 0])
        save('generator_control_frame', step)
        save('generator_seed', seed)
        save('generator_prediction', predicted)
        save('generator_dense_plan', dense)
        save('generated_actor_input', actor_input)
        save('known_reference_action_same_world', known_action)
        for key, value in obs.items(): save('generator_input_'+key, value)
        return action

    def frozen_audit(self):
        changed = [k for k, v in self.wrapper.policy.state_dict().items() if not torch.equal(v, self.initial[k])]
        trainable = [k for k, p in self.wrapper.policy.named_parameters() if p.requires_grad or p.grad is not None]
        return dict(passed=not changed and not trainable, changed_state_keys=changed,
                    trainable_or_gradient_parameters=trainable, generated_control_steps=self.calls,
                    full_parameter_count=8327408, optimizer_updates=0,
                    scope='Actual generated-command execution with known48position assistance; no SMP benefit or independent deployment claim.')
