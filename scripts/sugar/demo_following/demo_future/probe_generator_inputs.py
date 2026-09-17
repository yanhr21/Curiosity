"""Frozen full Generator input-branch diagnostic, without optimization or physics.

Use official noise assignment, scheduler and the complete saved denoiser. Input
swaps are sensitivity interventions, not counterfactual physical supervision.
"""
import json
from pathlib import Path

import dill
import numpy as np
import torch

from sugar_il.policy.generator import noise_assignment
from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
from scripts.sugar.demo_following.demo_future.generator_dataset import ActualDemoGeometryDataset, FrozenGeometryEvaluationDataset
from scripts.sugar.demo_following.demo_future.generator_demo_geometry import CONTEXT_KEY, restore_geometry_state
from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import write

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'matched_generator_demo_geometry16'
OUT = RUN / 'frozen_evaluation/input_branch_probe'
FIELDS = {'demo': (CONTEXT_KEY,), 'goal': ('target_obj_pos_b', 'target_obj_ori_b'),
          'history': ('last_action',), 'object': ('obj_pos_b', 'obj_ori_b')}


def selected_batch(dataset, selected_sources):
    indices, metadata = [], []
    for source in selected_sources:
        available = [i for i, (episode, _, _) in enumerate(dataset.indices)
                     if dataset.timeline_metadata[episode]['source'] == source]
        for fraction in (0., .5, 1.):
            index = available[round(fraction * (len(available) - 1))]
            indices.append(index)
            metadata.append(dict(source=source, relative_chunk_fraction=fraction, dataset_index=index))
    samples = [dataset[index] for index in indices]
    obs = {key: torch.stack([sample['obs'][key] for sample in samples]) for key in samples[0]['obs']}
    target = torch.stack([sample['action'] for sample in samples])
    # Move to the next source with the same relative chunk phase.
    permutation = (torch.arange(len(indices)) + 3) % len(indices)
    return obs, target, permutation, metadata


def main():
    torch.set_num_threads(1)
    if not json.loads((RUN / 'RESULT.json').read_text())['execution_completed']:
        raise RuntimeError('Complete the original matched prediction endpoint first')
    OUT.mkdir(exist_ok=False)
    parent = ROOT / 'SUGAR/demo_ckpts/CarryBox/generator.ckpt'
    train = ActualDemoGeometryDataset(BASE / 'tracker_train_generator_corpus76', parent)
    validation = FrozenGeometryEvaluationDataset(BASE / 'generator_native_validation8')
    available = sorted(row['source'] for row in train.timeline_metadata)
    sources = {'train': [available[i] for i in np.linspace(0, len(available) - 1, 8).round().astype(int)],
               'validation': [8, 28, 38, 58, 68, 78, 88, 98]}
    batches = {name: selected_batch(dataset, sources[name])
               for name, dataset in (('train', train), ('validation', validation))}
    seeds = list(range(272071, 272075))
    write(OUT / 'PROTOCOL.json', dict(sources=sources, selected_samples={k: v[3] for k, v in batches.items()},
        noise_seeds=seeds, device='cpu', optimizer_updates=0, physics_steps=0,
        model='Complete frozen demo_geometry endpoint; original official denoiser, noise assignment and scheduler.',
        intervention='Cycle one entire normalized input branch to the next source at the same relative chunk phase. All other inputs, noisy targets, noise assignment and diffusion times remain fixed. Swapped branches are diagnostic and are not counterfactual physical labels.',
        times='First, middle and last training diffusion timestep; equal source/phase/noise averaging within each time, not a replacement training or evaluation metric.',
        automatic_next_action='Inspect whether TRAIN and validation use the new demo input, whether existing goal/history dominate, and whether high-noise gradients reach the demo input. Keep failed primary conditioning result unchanged; choose a bounded full-model adaptation only from this evidence.'))
    policy = GeneratorWrapper.load(str(parent), device='cpu').policy
    with (RUN / 'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as stream:
        state = torch.load(stream, pickle_module=dill, map_location='cpu')['state_dicts']['model']
    restore_geometry_state(policy, state, 'demo_geometry')
    policy.eval().requires_grad_(False)
    times = [0, policy.noise_scheduler.config.num_train_timesteps // 2,
             policy.noise_scheduler.config.num_train_timesteps - 1]
    prediction_type = policy.noise_scheduler.config.prediction_type
    result = {}
    for split, (obs, target, permutation, metadata) in batches.items():
        normalized = policy.normalizer.normalize(obs)
        trajectory = policy.normalizer['action'].normalize(target)
        variants = {'correct': normalized}
        for branch, fields in FIELDS.items():
            swapped = dict(normalized)
            for field in fields:
                swapped[field] = normalized[field][permutation].clone()
            assert all(torch.equal(swapped[key], value) for key, value in normalized.items() if key not in fields)
            variants[branch] = swapped
        with torch.no_grad():
            tokens = {key: policy.obs_encoder(value, training=False) for key, value in variants.items()}
        token_changes = {key: ((value - tokens['correct']) ** 2).mean((0, 2)).tolist()
                         for key, value in tokens.items()}
        layer = policy.obs_encoder.target_state_net[0]
        goal = torch.cat([normalized['target_obj_pos_b'], normalized['target_obj_ori_b']], -1)
        projection = dict(goal_rms=float((goal @ layer.weight[:, :9].T).square().mean().sqrt()),
                          demo_rms=float((normalized[CONTEXT_KEY] @ layer.weight[:, 9:].T).square().mean().sqrt()))
        rows, gradients = [], None
        for seed in seeds:
            torch.manual_seed(seed)
            noise = torch.randn_like(trajectory)
            noise = noise[noise_assignment(trajectory, noise)]
            expected = noise if prediction_type == 'epsilon' else trajectory
            if prediction_type not in ('epsilon', 'sample'):
                raise RuntimeError('Unsupported official prediction type')
            for time in times:
                t = torch.full((len(target),), time, dtype=torch.long)
                noisy = policy.noise_scheduler.add_noise(trajectory, noise, t)
                with torch.no_grad():
                    outputs = {key: policy.model(noisy, t, cond=value, training=False, gen_attn_map=False)[0]
                               for key, value in tokens.items()}
                row = dict(seed=seed, time=time, variants={key: dict(
                    denoising_mse=float((value - expected).square().mean()),
                    prediction_change_mse=float((value - outputs['correct']).square().mean()))
                    for key, value in outputs.items()})
                if not all(torch.isfinite(value).all() for value in outputs.values()):
                    raise RuntimeError('Nonfinite full-model probe')
                rows.append(row)
                if seed == seeds[0] and time == times[-1]:
                    inputs = {key: value.detach().clone().requires_grad_(True) for key, value in normalized.items()}
                    features = policy.obs_encoder(inputs, training=False)
                    output = policy.model(noisy, t, cond=features, training=False, gen_attn_map=False)[0]
                    loss = (output - expected).square().mean()
                    values = torch.autograd.grad(loss, tuple(inputs.values()), allow_unused=True)
                    gradients = {key: None if value is None else dict(norm=float(value.norm()), rms=float(value.square().mean().sqrt()))
                                 for key, value in zip(inputs, values)}
        summary = {str(time): {variant: {metric: float(np.mean([
            row['variants'][variant][metric] for row in rows if row['time'] == time]))
            for metric in ('denoising_mse', 'prediction_change_mse')} for variant in variants} for time in times}
        result[split] = dict(samples=metadata, token_change_mse_by_object_history_goal_token=token_changes,
                             first_layer_contribution_rms=projection, high_noise_normalized_input_gradients=gradients,
                             per_seed_time=rows, summary_by_time=summary)
        write(OUT / 'PARTIAL_RESULT.json', result)
        print(json.dumps(dict(split=split, projection=projection, summary=summary)), flush=True)
    unchanged = all(torch.equal(value.cpu(), state[key]) for key, value in policy.state_dict().items())
    if not unchanged or any(p.grad is not None for p in policy.parameters()):
        raise RuntimeError('Frozen model or parameter gradients changed')
    write(OUT / 'RESULT.json', dict(execution_completed=True, model_state_exact=unchanged,
          optimizer_updates=0, physics_steps=0, full_parameter_count=sum(p.numel() for p in policy.parameters()),
          prediction_type=prediction_type, diffusion_times=times, per_split=result,
          scope='Complete frozen-model input sensitivity at predeclared TRAIN/validation probes. No new optimization, full diffusion samples, actual rollout, or change to the failed primary conditioning result.'))


if __name__ == '__main__':
    main()
