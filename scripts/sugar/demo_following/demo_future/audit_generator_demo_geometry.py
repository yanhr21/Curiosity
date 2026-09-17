"""Zero-update full released Generator audit with original-demo geometry input."""
import copy
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch

from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
from scripts.sugar.demo_following.demo_future.generator_demo_geometry import (
    CONTEXT_KEY, CONTEXT_WIDTH, install_geometry_context)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
PARENT = BASE / 'matched_reference_feedback96'
DATA = PARENT / 'reference_feedback/evaluation'
OUT = PARENT / 'generator_demo_geometry_preflight'


def main():
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    OUT.mkdir(exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    prepared = json.loads((DATA / 'original_demo_geometry_context/RESULT.json').read_text())
    if not prepared['execution_completed'] or prepared['optimizer_updates']:
        raise RuntimeError('Original-demo preparation incomplete')
    keys = ('obj_pos_b', 'obj_ori_b', 'last_action', 'target_obj_pos_b', 'target_obj_ori_b')
    obs = {k: [] for k in keys}
    contexts, targets, other = [], [], []
    for arm in ('original', 'alternate'):
        il = dict(np.load(DATA / f'official_il_data/{arm}_IL.npz'))
        data = dict(np.load(DATA / f'original_demo_geometry_context/{arm}.npz'))
        starts = data['official_il_chunk_start']
        for key in keys:
            obs[key].append(il[key][starts, None])
        contexts.append(data['original_demo_geometry'].reshape(-1, 1, CONTEXT_WIDTH))
        other.append(data['alternate_demo_geometry'].reshape(-1, 1, CONTEXT_WIDTH))
        targets.append(data['official_command_target'])
    obs = {k: torch.as_tensor(np.concatenate(v), device=device) for k, v in obs.items()}
    geometry = torch.as_tensor(np.concatenate(contexts), device=device)
    alternate_geometry = torch.as_tensor(np.concatenate(other), device=device)
    target = torch.as_tensor(np.concatenate(targets), device=device)
    wrapper = GeneratorWrapper.load(str(ROOT / 'SUGAR/demo_ckpts/CarryBox/generator.ckpt'), device=str(device))
    parent = wrapper.policy.eval().requires_grad_(False)
    parent_state = {k: v.clone() for k, v in parent.state_dict().items()}
    scheduler_class = type(parent.noise_scheduler).__name__
    results = {}
    policies = {}
    adapted_states = {}
    for arm in ('zero_context', 'demo_geometry'):
        policies[arm] = copy.deepcopy(parent)
        results[arm] = install_geometry_context(policies[arm], geometry, arm)
        policies[arm].eval().requires_grad_(False)
        adapted_states[arm] = {k: v.clone() for k, v in policies[arm].state_dict().items()}
    state0, state1 = (p.state_dict() for p in policies.values())
    matched = state0.keys() == state1.keys() and all(torch.equal(v, state1[k]) for k, v in state0.items())
    batch_obs = dict(obs, **{CONTEXT_KEY: geometry})
    torch.manual_seed(272045)
    sample = torch.randn_like(target)
    times = torch.full((len(target),), 17, device=device, dtype=torch.long)
    with torch.no_grad():
        base_tokens = parent.obs_encoder(parent.normalizer.normalize(obs), training=False)
        base_pred, _ = parent.model(sample, times, base_tokens, training=False)
        torch.manual_seed(272046)
        base_sample = parent.predict_action(obs)
        for arm, policy in policies.items():
            tokens = policy.obs_encoder(policy.normalizer.normalize(batch_obs), training=False)
            pred, _ = policy.model(sample, times, tokens, training=False)
            changed_obs = dict(batch_obs, **{CONTEXT_KEY: alternate_geometry})
            other_tokens = policy.obs_encoder(policy.normalizer.normalize(changed_obs), training=False)
            other_pred, _ = policy.model(sample, times, other_tokens, training=False)
            torch.manual_seed(272046)
            generated = policy.predict_action(batch_obs)
            results[arm].update(full_condition_shape=list(tokens.shape),
                initial_condition_max_error=float((tokens - base_tokens).abs().max()),
                initial_full_denoiser_max_error=float((pred - base_pred).abs().max()),
                initial_full_sampler_max_error=float((generated - base_sample).abs().max()),
                initial_alternate_context_denoiser_max_error=float((other_pred - pred).abs().max()),
                all_outputs_finite=bool(torch.isfinite(generated).all() and torch.isfinite(pred).all()))
    for arm, policy in policies.items():
        policy.requires_grad_(True)
        policy.normalizer.requires_grad_(False)
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(272047)
        loss = policy.compute_loss(dict(obs=batch_obs, action=target), training=False)
        loss.backward()
        grad = policy.obs_encoder.target_state_net[0].weight.grad[:, 9:]
        results[arm].update(actual_official_loss=float(loss.detach()),
                            new_input_column_gradient_norm=float(grad.norm()),
                            full_gradients_finite=all(torch.isfinite(p.grad).all().item() for p in policy.parameters() if p.grad is not None))
        policy.zero_grad(set_to_none=True)
        policy.requires_grad_(False)
    checks = dict(whole_adapted_initial_states_exact=matched,
                  original_full_generator_unchanged=all(torch.equal(v, parent.state_dict()[k]) for k, v in parent_state.items()),
                  zero_context_new_columns_gradient_zero=results['zero_context']['new_input_column_gradient_norm'] == 0,
                  geometry_context_new_columns_gradient_nonzero=results['demo_geometry']['new_input_column_gradient_norm'] > 0)
    for arm, row in results.items():
        checks[arm + '_full_forward_equivalence'] = row['initial_full_denoiser_max_error'] <= 1e-5
        checks[arm + '_full16step_sample_equivalence'] = row['initial_full_sampler_max_error'] <= 1e-4
        checks[arm + '_finite_forward_backward'] = row['all_outputs_finite'] and row['full_gradients_finite']
        checks[arm + '_correct_full_condition_shape'] = row['full_condition_shape'] == [82, 3, 256]
        current = policies[arm].state_dict()
        checks[arm + '_no_parameter_update'] = all(torch.equal(v, adapted_states[arm][k]) for k, v in current.items())
    result = dict(execution_completed=True, passed=all(checks.values()), checks=checks, arms=results,
                  actual_training_chunks=len(target), optimizer_updates=0,
                  optimizer_constructed=False, physics_steps=0,
                  official_scheduler=scheduler_class, inference_steps=parent.num_inference_steps,
                  scope='Actual full released12x256 Generator forward,16-step sampling and backward only. Original-demo geometry adapts an existing encoder input; this is not Zero-WAM reproduction, an SMP latent, trained future generation or physical success.')
    (OUT / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
