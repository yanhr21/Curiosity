"""Frozen prompt intervention at identical recorded official sampler states."""
import argparse
import json
import os
import socket
from pathlib import Path

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import (
    SOURCE, OUT as RECORDED, PARENT, ActualBranchGeometryDataset,
    GeneratorWrapper, CONTEXT_KEY, restore_geometry_state, write,
)

OUT = RECORDED / 'fixed_xt_condition'
REFERENCE_SOURCE = SOURCE


def prepare():
    recorded = json.loads((RECORDED / 'RESULT.json').read_text())
    assert recorded['checks_passed'] and recorded['plot_inspected']
    transfer = SOURCE != REFERENCE_SOURCE
    if transfer:
        reference=json.loads((RECORDED/'fixed_xt_condition/RESULT.json').read_text())
        component=json.loads((SOURCE/'frozen_evaluation/rank_component_audit/RESULT.json').read_text())
        comparison=json.loads((SOURCE/'frozen_evaluation/PAIRED_COMPARISON.json').read_text())
        assert reference['checks_passed'] and reference['plot_inspected'] and component['checks_passed'] and component['plot_inspected'] and comparison['checks_passed']
    plan = json.loads((RECORDED / 'PROTOCOL.json').read_text())
    OUT.mkdir(exist_ok=False)
    write(OUT / 'PROTOCOL.json', dict(
        source_run=str(SOURCE), recorded_run=str(RECORDED),
        phases=plan['phases'], seeds=plan['seeds'], steps=16,
        model_forward_calls=2304, optimizer_updates=0, new_physics_steps=0,
        transferred_model=transfer, replayed_own_sample_calls=144 if transfer else 0,
        scope=('Full hinge model on identical saved xt from full025 correct-prompt16step paths, compared with previously exact-audited full025 predictions. Full official encoder and denoiser,4savedseeds/9phases/2branches/16times/correct+wrong.144own hinge final samples exactly replayed plus2304point forwards;4608total denoiser forwards. No new independent samples, optimizer or physical labels. ' if transfer else 'Same saved xt from correct-prompt original16step paths; full official encoder and denoiser. Swap only demo context while holding xt/time/causal observations fixed. Four existing seeds, all nine phases, two branches, all16 times, correct/wrong. Exact recorded correct epsilon and scheduler clean prediction reconstruction required. ')+ 'Wrong prompt on this xt is a sensitivity intervention, not a physical counterfactual label or a newly sampled trajectory.',
        selection='Diagnose TRAIN277/298 plateau observed in recorded reverse paths; reused218/258 reported separately and never select method settings. No intermediate output or solver selection.',
        automatic_next_action='Inspect all nine fixed-xt condition curves and full-state/input/RNG checks. Compare whether correct conditioning improves actual clean-target error at early versus late recorded times; distinguish same-xt effects from divergent complete sampling paths. Use this evidence before declaring any new matched training objective. No old-budget extension or SMP benefit claim.',
        transfer_scope='When transferred_model is true, evaluate full hinge endpoint on exact originalfull025 recorded xt; baseline full025 outputs are reused from the previously exact-reproduced fixed-xt audit. Replay own hinge first4saved16step samples per phase/condition/branch exactly before transfer. This is a same-input diagnostic on another models trajectories, not newly generated execution or evidence that hinge follows those paths. No ranking coefficient/margin sweep.'))


def run():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login', 'mgmtserver'))
    torch.set_num_threads(8)
    protocol = json.loads((OUT / 'PROTOCOL.json').read_text())
    assert not (OUT / 'PARTIAL_RESULT.json').exists() and not (OUT / 'RESULT.json').exists()
    plan = json.loads((SOURCE / 'PROTOCOL.json').read_text())
    with (SOURCE / 'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as stream:
        state = torch.load(stream, pickle_module=dill, map_location='cpu')['state_dicts']['model']
    policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
    restore_geometry_state(policy, state, 'demo_geometry', goal_condition_mode='zero_diagnostic')
    policy.cuda().eval().requires_grad_(False)
    assert sum(p.numel() for p in policy.parameters()) == 8319216
    config = policy.noise_scheduler.config
    assert config.prediction_type == 'epsilon' and config.clip_sample and config.clip_sample_range == 1.0 and not config.thresholding
    checks = {}; rows = {}; calls = 0; own_replays = 0
    for phase in protocol['phases']:
        dataset = ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)], plan['normalizer_state'])
        samples = [dataset[i] for i in (0, 1)]
        obs = {k: torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
        target = policy.normalizer['action'].normalize(torch.stack([s['action'] for s in samples]).cuda()).cpu().numpy()
        with np.load(RECORDED / f'phase_{phase}_steps_16.npz') as archive:
            xt = archive['xt'][:, 0].copy()
            expected_epsilon = archive['epsilon'][:, 0].copy()
            expected_clean = archive['pred_original'][:, 0].copy()
            times = archive['times'].copy()
            checks[f'{phase}_targets_exact'] = np.array_equal(target.astype(np.float64), archive['normalized_actual_target'])
            checks[f'{phase}_seeds_exact'] = archive['seeds'].tolist() == protocol['seeds']
        baseline = None
        if protocol.get('transferred_model'):
            with np.load(RECORDED/'fixed_xt_condition'/f'phase_{phase}.npz') as archive:
                baseline=archive['clean'].copy()
                checks[f'{phase}_reference_epsilon_exact']=np.array_equal(archive['epsilon'][:,0],expected_epsilon)
                checks[f'{phase}_reference_clean_exact']=np.array_equal(baseline[:,0],expected_clean)
                checks[f'{phase}_reference_targets_exact']=np.array_equal(archive['target'],target)
            policy.num_inference_steps=16
            for ci,condition_name in enumerate(('correct','wrong')):
                inputs=dict(obs)
                if ci:inputs[CONTEXT_KEY]=obs[CONTEXT_KEY].flip(0)
                draws=[]
                with torch.inference_mode():
                    for seed in protocol['seeds']:
                        pair=[]
                        for branch in (0,1):
                            torch.manual_seed(seed)
                            pair.append(policy.predict_action({k:v[branch:branch+1] for k,v in inputs.items()}))
                            own_replays += 1
                        draws.append(torch.cat(pair))
                with np.load(SOURCE/f'frozen_evaluation/phase_{phase}/trained_demo_geometry_{condition_name}.npz') as archive:
                    checks[f'{phase}_{condition_name}_own_four_samples_exact']=np.array_equal(torch.stack(draws).cpu().numpy(),archive['predictions'][:4])
            if not all(checks.values()):raise RuntimeError('Transferred model failed own exact replay or reference binding')
        cpu_rng = torch.get_rng_state().clone(); gpu_rng = torch.cuda.get_rng_state().clone()
        epsilon = np.empty((4, 2, 2, 16, 8, 36), dtype=np.float32)
        clean = np.empty_like(epsilon); unclipped = np.empty_like(epsilon)
        with torch.inference_mode():
            for ci in (0, 1):
                inputs = dict(obs)
                if ci:
                    inputs[CONTEXT_KEY] = obs[CONTEXT_KEY].flip(0)
                checks[f'{phase}_{ci}_causal_fields_exact'] = all(torch.equal(inputs[k], obs[k]) for k in obs if k != CONTEXT_KEY)
                for branch in (0, 1):
                    normalized = policy.normalizer.normalize({k: v[branch:branch+1] for k, v in inputs.items()})
                    condition = policy.obs_encoder(normalized, training=False)
                    for si in range(4):
                        for ti, time in enumerate(times):
                            sample = torch.from_numpy(xt[si, branch, ti:ti+1]).cuda()
                            t = torch.tensor(int(time), device='cuda')
                            predicted, _ = policy.model(sample, t, condition, training=False, gen_attn_map=False)
                            calls += 1
                            alpha = policy.noise_scheduler.alphas_cumprod[int(time)]
                            # Exact installed official DDPMScheduler epsilon-to-clean readback;
                            # no scheduler transition or random draw is performed here.
                            raw = (sample - (1-alpha)**0.5 * predicted) / alpha**0.5
                            epsilon[si, ci, branch, ti] = predicted[0].cpu().numpy()
                            unclipped[si, ci, branch, ti] = raw[0].cpu().numpy()
                            clean[si, ci, branch, ti] = raw.clamp(-1, 1)[0].cpu().numpy()
        if not protocol.get('transferred_model'):
            checks[f'{phase}_all_correct_epsilon_exact'] = np.array_equal(epsilon[:, 0], expected_epsilon)
            checks[f'{phase}_all_correct_clean_exact'] = np.array_equal(clean[:, 0], expected_clean)
        checks[f'{phase}_rng_unchanged'] = torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
        checks[f'{phase}_finite'] = all(np.isfinite(a).all() for a in (epsilon, clean, unclipped))
        np.savez_compressed(OUT / f'phase_{phase}.npz', epsilon=epsilon, clean=clean,
                            unclipped=unclipped, times=times, target=target, seeds=protocol['seeds'])
        error = ((clean.astype(np.float64)-target[None, None, :, None])**2).mean((-1, -2))
        raw_error = ((unclipped.astype(np.float64)-target[None, None, :, None])**2).mean((-1, -2))
        response = clean[:, 0].astype(np.float64)-clean[:, 1]
        delta = target.astype(np.float64)-target[::-1]
        projection = (response*delta[None, :, None]).mean((-1, -2))/np.square(delta).mean((-1, -2))[None, :, None]
        rows[str(phase)] = dict(times=times.tolist(), axes=['condition', 'branch', 'time'],
            clean_mse=error.mean(0).tolist(), unclipped_mse=raw_error.mean(0).tolist(),
            correct_better_than_wrong_draws=(error[:, 0] < error[:, 1]).sum(0).tolist(),
            prompt_response_projection=projection.mean(0).tolist())
        if baseline is not None:
            reference_error=((baseline.astype(np.float64)-target[None,None,:,None])**2).mean((-1,-2))
            rows[str(phase)]['reference_full025_clean_mse']=reference_error.mean(0).tolist()
        write(OUT / 'PARTIAL_RESULT.json', dict(checks=checks, phases=rows, model_forward_calls=calls))
        if not all(checks.values()):
            raise RuntimeError('Fixed-xt diagnostic failed exact replay/input/RNG checks')
        print(json.dumps(dict(phase=phase, checks_passed=True)), flush=True)
    checks['full_state_unchanged'] = state.keys() == policy.state_dict().keys() and all(torch.equal(v.cpu(), state[k]) for k, v in policy.state_dict().items())
    checks['no_gradients'] = all(p.grad is None for p in policy.parameters())
    checks['declared_calls_exact'] = calls == protocol['model_forward_calls']
    checks['declared_own_replays_exact'] = own_replays == protocol.get('replayed_own_sample_calls',0)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for phase, ax in zip(protocol['phases'], axes.flat):
        row = rows[str(phase)]; mse = np.asarray(row['clean_mse'])
        for branch in (0, 1):
            if protocol.get('transferred_model'):
                ref=np.asarray(row['reference_full025_clean_mse'])
                ax.plot(row['times'],ref[0,branch],color=f'C{branch}',linestyle='--',label=f'full025 correct branch{branch}')
                ax.plot(row['times'],mse[0,branch],color=f'C{branch}',linestyle='-',label=f'hinge correct branch{branch}')
            else:
                for ci, style in enumerate(('-', '--')):
                    ax.plot(row['times'], mse[ci, branch], color=f'C{branch}', linestyle=style,
                            label=f'branch{branch} '+('correct' if ci == 0 else 'wrong'))
        ax.set_title(f'Phase{phase}: identical recorded xt'); ax.set_yscale('log')
        ax.invert_xaxis(); ax.set_xlabel('Reverse diffusion time'); ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Full official denoiser prompt intervention; saved4seeds; no new sampling or physics')
    fig.tight_layout(); fig.savefig(OUT / 'FIXED_XT_CONDITION.png', dpi=140); plt.close(fig)
    write(OUT / 'RESULT.json', dict(execution_completed=True, checks_passed=all(checks.values()),
        checks=checks, phases=rows, model_forward_calls=calls, full_parameter_count=8319216,
        replayed_own_sample_calls=own_replays,total_denoiser_forwards=calls+16*own_replays,
        plot_inspected=False, new_optimizer_updates=0, new_physics_steps=0, scope=protocol['scope']))
    assert all(checks.values())


def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run'], required=True)
    parser.add_argument('--run',type=Path,default=SOURCE)
    args = parser.parse_args()
    selected=args.run.resolve()
    if selected == REFERENCE_SOURCE.parent/'matched_generator_branch_paired_hinge025512':
        SOURCE=selected;OUT=SOURCE/'frozen_evaluation/recorded_path_transfer'
    elif selected != REFERENCE_SOURCE:
        raise RuntimeError('Use only the declared full025 or hinge fixed-state model')
    prepare() if args.mode == 'prepare' else run()


if __name__ == '__main__':
    main()
