"""Full official Generator identifiability and actual branch-phase diagnostics.

There is no replacement model or generated trajectory. Goal removal is an
explicit diagnostic input intervention, not a deployable observation contract.
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket

import dill
import numpy as np
import torch

from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
from scripts.sugar.demo_following.demo_future.generator_dataset import ActualBranchGeometryDataset, ActualBranchReplayDataset
from scripts.sugar.demo_following.demo_future.generator_demo_geometry import CONTEXT_KEY, restore_geometry_state
from scripts.sugar.demo_following.demo_future.generator_workspace import warm_start_geometry_policy
from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import write

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'matched_generator_actual_branch256'
PARENT = ROOT / 'SUGAR/demo_ckpts/CarryBox/generator.ckpt'
ARMS = ('zero_context', 'demo_geometry')


def prepare():
    source = BASE / 'generator_actual_branch_point'
    result = json.loads((source / 'RESULT.json').read_text())
    if not result['execution_completed'] or not all(result['checks'].values()):
        raise RuntimeError('Complete actual shared-world branch preparation first')
    previous = BASE / 'matched_generator_demo_geometry16'
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    for key in ('native_train_corpus', 'additional_paired_train', 'frozen_validation'):
        plan.pop(key, None)
    plan.update(seed=272084, epochs=256, batch_size=64, checkpoint_every=64,
        optimizer=dict(lr=1e-4, weight_decay=1e-4, betas=[.95, .999]),
        scheduler=dict(name='cosine', warmup_steps=16),
        branch_dataset=dict(corpus=str(source), normalizer_state=str(previous / 'zero_context/INITIAL_MODEL.pt'), repetitions=32),
        goal_condition_mode='zero_diagnostic', post_prediction_position_floor=None,
        actual_optimizer_updates=256, real_branch_examples=2, repeated_examples_per_update=64,
        data='Exactly two real TRAIN futures from shared world158, sources96/90. Repeat each32 times per balanced batch only to sample diffusion noise; no extra trajectories or independent examples. Original command8x36 labels remain separate from actual29-D actions.',
        conditioning='Both normalized9-D goal fields zeroed identically inside the existing full encoder. Object pose and last-command inputs are exactly shared. Only original-demo8x21 geometry differs; matched control zeros it after normalization. No source ID, labels or actual future state enters context.',
        normalization='Restore all previous TRAIN-only normalization from its zero-update initial full model. No refit on these two examples or validation.',
        optimization='Unchanged full official Generator loss, noise assignment, scheduler and TrainGeneratorWorkspace.run. Full12x256/8-head model, original16-step DDPM. Fresh AdamW, one balanced64-row batch per epoch,256updates; local diagnostic LR1e-4/warmup16, no history dropout.',
        checkpoint='Retain full initial state, full model/Adam every64epochs and complete endpoint, actual batch IDs and clocks. No best-checkpoint selection or interrupted-loop resume claim.',
        retained_goal_provenance='Actual original goals remain in the data archive but both normalized goal fields are explicitly zeroed in every diagnostic training and evaluation call. No Refiner-goal cue enters the model in this diagnostic.',
        frozen_evaluation=dict(sample_seeds=list(range(272090, 272122)), samples_per_branch=32,
            variants=['initial_shared_goal_zero', 'trained_zero_context', 'trained_demo_geometry_correct', 'trained_demo_geometry_wrong', 'trained_demo_geometry_zero'],
            inference_steps=16, same_noise='Each branch receives identical RNG seed for each draw; no-context outputs must therefore be exactly identical and swapped geometry must swap outputs exactly.',
            criterion='Correct-context normalized all-draw MSE <=0.5 times each unconditional two-label mean baseline, trained zero-context, wrong-context and own-zero baseline. At least28/32 draws per branch nearer its correct target; wrong context must prefer the other target at least28/32. All matching and intervention checks required.',
            scope='Identifiability on two known TRAIN futures from one actual shared world. No validation/generalization, human-video, generated physical execution or SMP-benefit claim.'),
        automatic_next_action='Read all32 draws and horizon/channel evidence. If branch identifiability passes, investigate actual branching data coverage and prospective TRAIN sources before broader matched adaptation; do not call this generalization. If it fails, inspect full-model fitting and gradient/context evidence before any further budget. Resolve independent original-position and original-goal compatibility before generated execution.',
        generator_training_started=False)
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), real_examples=2, updates_per_arm=256)), flush=True)


def preflight(output_name='PREFLIGHT.json'):
    torch.set_num_threads(1)
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    dataset = ActualBranchGeometryDataset(**plan['branch_dataset'])
    batch_rows = plan.get('repeated_examples_per_update', 64)
    samples = [dataset[i] for i in range(dataset.real_case_count)]
    obs = {key: torch.stack([sample['obs'][key] for sample in samples]) for key in samples[0]['obs']}
    target = torch.stack([sample['action'] for sample in samples])
    checks, gradients = {}, {}
    if plan.get('phase_evaluation'):
        heldout = ActualBranchGeometryDataset(**dict(plan['branch_dataset'], split='heldout_phase', paired_supervision=False))
        checks['all_heldout_actual_arrays_exact'] = heldout.real_case_count == plan['data']['heldout_phase']['real_examples']
        checks['declared_train_phases_exact'] = [r['shared_world_control_frame'] for r in dataset.timeline_metadata] == [p for p in plan['data']['train']['phases'] for _ in range(2)]
        checks['declared_heldout_phases_exact'] = [r['shared_world_control_frame'] for r in heldout.timeline_metadata] == [p for p in plan['data']['heldout_phase']['phases'] for _ in range(2)]
    initial = torch.load(plan['branch_dataset']['normalizer_state'], map_location='cpu')['model']
    for arm in ARMS:
        coupling = plan.get('noise_coupling', 'official_assignment')
        policy = warm_start_geometry_policy(PARENT, arm, goal_condition_mode='zero_diagnostic', noise_coupling=coupling)
        policy.set_normalizer(dataset.get_normalizer()); policy.normalizer.requires_grad_(False)
        policy.eval()
        captured = []
        handle = policy.obs_encoder.target_state_net[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        normalized = policy.normalizer.normalize(obs)
        features = policy.obs_encoder(normalized, training=False)
        checks[arm + '_goal_exact_zero'] = not bool(captured[-1][:, :9].count_nonzero())
        checks[arm + '_initial_tokens_shared'] = all(torch.equal(features[i], features[i + 1]) for i in range(0, len(samples), 2))
        checks[arm + '_full_original_state_exact'] = policy.state_dict().keys() == initial.keys() and all(torch.equal(value, initial[key]) for key, value in policy.state_dict().items())
        checks[arm + '_all_declared_rows_balanced'] = len(dataset) == batch_rows and all(torch.equal(dataset[i]['action'], target[i % len(samples)]) for i in range(batch_rows))
        from sugar_il.policy.generator import noise_assignment
        loss_target = target.repeat(dataset.repetitions, 1, 1)
        loss_obs = {key: value.repeat(dataset.repetitions, 1, 1) for key, value in obs.items()}
        trajectory = policy.normalizer['action'].normalize(loss_target)
        torch.manual_seed(272085)
        expected_noise = torch.randn(trajectory.shape)
        assigned_noise = expected_noise[noise_assignment(trajectory, expected_noise)]
        checks[arm + '_noise_probe_exercises_nonidentity_pairing'] = not torch.equal(expected_noise, assigned_noise)
        if coupling == 'official_assignment':
            expected_noise = assigned_noise
        expected_time = torch.randint(0, policy.noise_scheduler.config.num_train_timesteps, (batch_rows,)).long()
        expected_noisy = policy.noise_scheduler.add_noise(trajectory, expected_noise, expected_time)
        model_inputs = []
        noise_hook = policy.model.register_forward_pre_hook(lambda module, args, kwargs: model_inputs.append((args[0].detach().clone(), args[1].detach().clone())), with_kwargs=True)
        torch.manual_seed(272085)
        loss = policy.compute_loss(dict(obs=loss_obs, action=loss_target), training=True)
        noise_hook.remove()
        checks[arm + '_declared_noisy_input_exact'] = torch.equal(model_inputs[0][0], expected_noisy)
        checks[arm + '_original_timestep_stream_exact'] = torch.equal(model_inputs[0][1], expected_time)
        loss.backward()
        grad = policy.obs_encoder.target_state_net[0].weight.grad[:, 9:]
        gradients[arm] = float(grad.norm())
        checks[arm + '_geometry_gradient_contract'] = gradients[arm] == 0. if arm == 'zero_context' else gradients[arm] > 0.
        policy.zero_grad(set_to_none=True)
        predictions = []
        with torch.inference_mode():
            for branch in range(len(samples)):
                torch.manual_seed(272086)
                predictions.append(policy.predict_action({key: value[branch:branch + 1] for key, value in obs.items()}))
        checks[arm + '_initial_same_noise_outputs_shared'] = all(torch.equal(predictions[i], predictions[i + 1]) for i in range(0, len(samples), 2))
        checks[arm + '_state_unchanged_after_backward'] = all(torch.equal(value, initial[key]) for key, value in policy.state_dict().items())
        handle.remove()
    checks['causal_inputs_exact'] = all(torch.equal(obs[key][i], obs[key][i + 1]) for key in ('obj_pos_b', 'obj_ori_b', 'last_action') for i in range(0, len(samples), 2))
    norm_target = policy.normalizer['action'].normalize(target)
    paired_target = norm_target.reshape(-1, 2, 8, 36)
    baseline = float((paired_target - paired_target.mean(1, keepdim=True)).square().mean())
    report = dict(execution_completed=True, passed=all(checks.values()), checks=checks,
        new_column_gradient_norms=gradients, two_target_mean_normalized_mse=baseline,
        real_training_branch_examples=len(samples),
        balanced_batch_rows=batch_rows,
        full_parameter_count=sum(p.numel() for p in policy.parameters()), optimizer_updates=0, physics_steps=0,
        scope='Full-model CPU input/gradient/sampling preflight only; missing goal input is explicit diagnostic intervention.')
    write(RUN / output_name, report)
    if not report['passed']:
        raise RuntimeError('Actual-branch full-model preflight failed')
    print(json.dumps(report), flush=True)


def evaluate_branch(run=RUN, *, transfer=False, phase=None):
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the retained compute step for full frozen sampling')
    run = Path(run)
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    if phase is not None:
        if plan.get('phase_frozen_evaluation'):
            plan['frozen_evaluation'] = plan['phase_frozen_evaluation']
        if transfer or phase not in plan['frozen_evaluation']['evaluation_phases']:
            raise RuntimeError('Use only the declared phase evaluation')
        plan['branch_dataset'] = dict(corpus=plan['phase_corpora'][str(phase)],
                                     normalizer_state=plan['branch_dataset']['normalizer_state'], repetitions=32)
    if transfer:
        endpoint = json.loads((run / 'RESULT.json').read_text())
        if not endpoint['execution_completed'] or endpoint['preliminary_conditioning_passed'] or plan.get('goal_condition_mode') != 'zero_diagnostic':
            raise RuntimeError('Transfer probe requires the completed negative goal-removed broader endpoint')
        reference = json.loads((BASE / 'matched_generator_actual_branch_iid256/PROTOCOL.json').read_text())
        plan['branch_dataset'] = reference['branch_dataset']
        plan['frozen_evaluation'] = reference['frozen_evaluation']
        plan['frozen_evaluation']['scope'] = 'Zero-update transfer probe of the full broader-trained goal-removed Generator on two exact shared-world TRAIN-source branch examples omitted by the original stride phase. Adjacent same-source chunks were in TRAIN: this is not motion-held-out or untouched-test evidence. All32draws, common causal inputs, actual observed8x36 targets, no generated physics or SMP benefit.'
        plan['automatic_next_action'] = 'Compare all full-model branch predictions and context interventions with the earlier two-example fit. If this broader endpoint cannot select the actual branches, use the actual common-world data-coverage result to specify a bounded new real paired-data collection before any new Generator budget. Never count repeated examples as independent sources.'
    initial = [torch.load(run / arm / 'INITIAL_MODEL.pt', map_location='cpu')['model'] for arm in ARMS]
    checks = dict(full_initial_states_exact=initial[0].keys() == initial[1].keys() and all(torch.equal(value, initial[1][key]) for key, value in initial[0].items()),
        actual_batch_order_exact=(run / ARMS[0] / 'BATCH_ORDER.json').read_text() == (run / ARMS[1] / 'BATCH_ORDER.json').read_text(),
        actual_data_budget_exact=(run / ARMS[0] / 'DATA_AND_BUDGET.json').read_text() == (run / ARMS[1] / 'DATA_AND_BUDGET.json').read_text(),
        both_full_endpoints=all(json.loads((run / arm / 'RESULT.json').read_text())['endpoint_checks_passed'] for arm in ARMS))
    if plan.get('preceding_noise_assignment_experiment'):
        previous = Path(plan['preceding_noise_assignment_experiment'])
        for arm, state in zip(ARMS, initial):
            old = torch.load(previous / arm / 'INITIAL_MODEL.pt', map_location='cpu')['model']
            checks[arm + '_preceding_initial_exact'] = old.keys() == state.keys() and all(torch.equal(value, old[key]) for key, value in state.items())
            checks[arm + '_preceding_batch_order_exact'] = (previous / arm / 'BATCH_ORDER.json').read_text() == (run / arm / 'BATCH_ORDER.json').read_text()
    if not all(checks.values()):
        raise RuntimeError('Actual branch training was not matched')
    dataset = ActualBranchGeometryDataset(**plan['branch_dataset'])
    samples = [dataset[i] for i in (0, 1)]
    obs = {key: torch.stack([sample['obs'][key] for sample in samples]).cuda() for key in samples[0]['obs']}
    target = torch.stack([sample['action'] for sample in samples]).cuda()
    out = run / ('frozen_branch_transfer' if transfer else 'frozen_evaluation')
    if phase is not None:
        out = out / f'phase_{phase}'
    out.mkdir(parents=phase is not None, exist_ok=False)
    if transfer:
        write(out / 'PROTOCOL.json', dict(
            checkpoint_run=str(run), branch_dataset=plan['branch_dataset'],
            frozen_evaluation=plan['frozen_evaluation'], goal_condition_mode='zero_diagnostic',
            optimizer_updates=0, actual_new_physics_steps=0,
            automatic_next_action=plan['automatic_next_action']))
    results, saved = {}, {}
    for variant in plan['frozen_evaluation']['variants']:
        policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
        if variant == 'initial_shared_goal_zero':
            state = initial[0]
        else:
            arm = 'zero_context' if variant == 'trained_zero_context' else 'demo_geometry'
            with (run / arm / 'checkpoints/endpoint.ckpt').open('rb') as stream:
                state = torch.load(stream, pickle_module=dill, map_location='cpu')['state_dicts']['model']
        mode = 'demo_geometry' if variant in ('trained_demo_geometry_correct', 'trained_demo_geometry_wrong') else 'zero_context'
        restore_geometry_state(policy, state, mode, goal_condition_mode='zero_diagnostic')
        policy.cuda().eval().requires_grad_(False)
        inputs = dict(obs)
        if variant == 'trained_demo_geometry_wrong':
            inputs[CONTEXT_KEY] = obs[CONTEXT_KEY].flip(0)
        if plan.get('solver_comparison'):
            solver = plan['solver_comparison']
            if phase is None or solver['new_steps'] != 50 or policy.num_inference_steps != 16:
                raise RuntimeError('Require the separately declared16-to50 full DDPM probe')
            with np.load(Path(solver['source_run']) / f'frozen_evaluation/phase_{phase}/{variant}.npz') as archive:
                old_predictions = archive['predictions'][:2].copy()
                checks[variant+'_solver_targets_exact'] = np.array_equal(archive['actual_future_command_target'],target.cpu().numpy())
            replay=[]
            with torch.inference_mode():
                for seed in plan['frozen_evaluation']['sample_seeds'][:2]:
                    pair=[]
                    for branch in (0,1):
                        torch.manual_seed(seed)
                        pair.append(policy.predict_action({k:v[branch:branch+1] for k,v in inputs.items()}))
                    replay.append(torch.cat(pair))
            checks[variant+'_first_two_original16_draws_exact']=np.array_equal(torch.stack(replay).cpu().numpy(),old_predictions)
            if not checks[variant+'_first_two_original16_draws_exact']:
                raise RuntimeError('Unchanged16step replay differs before solver probe')
            policy.num_inference_steps=50
        draws = []
        with torch.inference_mode():
            for seed in plan['frozen_evaluation']['sample_seeds']:
                per_branch = []
                for branch in (0, 1):
                    torch.manual_seed(seed)
                    per_branch.append(policy.predict_action({key: value[branch:branch + 1] for key, value in inputs.items()}))
                draws.append(torch.cat(per_branch))
            prediction = torch.stack(draws)
            if not torch.isfinite(prediction).all() or prediction.shape != (32, 2, 8, 36):
                raise RuntimeError('Invalid complete sampled branch futures')
            normalized = policy.normalizer['action'].normalize(prediction)
            normalized_target = policy.normalizer['action'].normalize(target)
            error = (normalized - normalized_target[None]).square()
            other_error = (normalized - normalized_target.flip(0)[None]).square()
            preferred = error.mean((2, 3)) < other_error.mean((2, 3))
            baseline = (normalized_target - normalized_target.mean(0)).square()
            last = obs['last_action'].expand(-1, 8, -1)
            last_error = (policy.normalizer['action'].normalize(last) - normalized_target).square()
        results[variant] = dict(normalized_mse=float(error.mean()), per_branch_mse=error.mean((0, 2, 3)).cpu().tolist(),
            per_branch_correct_preference_draws=preferred.sum(0).cpu().tolist(),
            per_branch_other_preference_draws=(other_error.mean((2, 3)) < error.mean((2, 3))).sum(0).cpu().tolist(),
            normalized_mse_by_branch_horizon_dimension=error.mean(0).cpu().tolist())
        saved[variant] = prediction.cpu().numpy()
        np.savez_compressed(out / f'{variant}.npz', predictions=saved[variant], actual_future_command_target=target.cpu().numpy(), previous_command=last.cpu().numpy(), sample_seeds=np.asarray(plan['frozen_evaluation']['sample_seeds']))
        write(out / 'PARTIAL_RESULT.json', results)
        print(json.dumps(dict(variant=variant, **{k: v for k, v in results[variant].items() if 'horizon' not in k})), flush=True)
        if plan.get('solver_comparison'):
            checks[variant+'_full50_time_sequence']=policy.noise_scheduler.timesteps.tolist()==list(range(49,-1,-1))
            checks[variant+'_solver_full_state_unchanged']=all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        del policy
        torch.cuda.empty_cache()
    for variant in ('initial_shared_goal_zero', 'trained_zero_context', 'trained_demo_geometry_zero'):
        checks[variant + '_same_noise_outputs_exact'] = np.array_equal(saved[variant][:, 0], saved[variant][:, 1])
    checks['swapped_context_swaps_outputs_exactly'] = np.array_equal(saved['trained_demo_geometry_correct'][:, ::-1], saved['trained_demo_geometry_wrong'])
    correct = results['trained_demo_geometry_correct']
    comparisons = {}
    for name, value in [('two_target_mean', float(baseline.mean()))] + [(name, results[name]['normalized_mse']) for name in ('trained_zero_context', 'trained_demo_geometry_wrong', 'trained_demo_geometry_zero')]:
        ratio = correct['normalized_mse'] / value
        comparisons[name] = dict(correct_mse_ratio=ratio, passed=ratio <= .5)
    preference = min(correct['per_branch_correct_preference_draws']) >= 28 and min(results['trained_demo_geometry_wrong']['per_branch_other_preference_draws']) >= 28
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    groups = {'joint29': slice(0, 29), 'linear3': slice(29, 32), 'angular3': slice(32, 35), 'contact1': slice(35, 36)}
    for branch in (0, 1):
        for ax, (name, section) in zip(axes[branch], groups.items()):
            for variant, row in results.items():
                errors = np.asarray(row['normalized_mse_by_branch_horizon_dimension'])[branch, :, section].mean(-1)
                ax.plot(np.arange(8) * .1, errors, label=variant)
            ax.plot(np.arange(8) * .1, baseline[branch, :, section].mean(-1).cpu(), 'k--', label='two-target mean')
            if phase is not None:
                ax.plot(np.arange(8) * .1, last_error[branch, :, section].mean(-1).cpu(), color='gray', linestyle=':', label='repeat last command')
            ax.set_title(f'Branch{branch} {name}'); ax.grid(alpha=.2)
    axes[1, 0].legend(fontsize=6)
    title = 'Broader-trained transfer to two TRAIN-source branches' if transfer else 'Two real TRAIN branches'
    if phase is not None:
        split = 'fitted phase' if phase in plan['data']['train']['phases'] else 'reused phase check'
        title = f'Actual phase{phase}: {split}, two known source motions'
    fig.suptitle(title + (': full50stepDDPM' if plan.get('solver_comparison') else '') + ': shared goal, all32 draws; no generated physics')
    fig.tight_layout(); fig.savefig(out / 'HORIZON_ERRORS.png', dpi=150); plt.close(fig)
    final = dict(execution_completed=True, matched_and_intervention_checks=checks, variants=results,
        two_target_mean_normalized_mse=float(baseline.mean()), comparisons=comparisons, preference_criterion_passed=preference,
        branch_identifiability_passed=all(checks.values()) and all(row['passed'] for row in comparisons.values()) and preference,
        real_training_branch_examples=0 if transfer else 2, evaluated_actual_branch_examples=2,
        actual_updates_per_arm=json.loads((run / ARMS[0] / 'RESULT.json').read_text())['actual_optimizer_updates'],
        new_optimizer_updates=0, new_physics_steps=0, horizon_plot_inspected=False,
        scope=plan['frozen_evaluation']['scope'], next_action=plan['automatic_next_action'])
    final['noise_coupling'] = plan.get('noise_coupling', 'official_assignment')
    if plan.get('solver_comparison'):
        final['solver_comparison']=plan['solver_comparison']
    if phase is not None:
        final['evaluation_phase'] = phase
        final['evaluation_split'] = 'train' if phase in plan['data']['train']['phases'] else 'heldout_phase'
        final['real_training_branch_examples'] = plan['data']['train']['real_examples']
        final['last_command_normalized_mse'] = float(last_error.mean())
        final['last_command_error_by_branch_horizon_dimension'] = last_error.cpu().tolist()
    if plan.get('preceding_noise_assignment_experiment'):
        old = json.loads((Path(plan['preceding_noise_assignment_experiment']) / 'RESULT.json').read_text())
        final['preceding_noise_assignment_comparison'] = dict(correct_mse_ratio=correct['normalized_mse'] / old['variants']['trained_demo_geometry_correct']['normalized_mse'],
            old_correct_preference_draws=old['variants']['trained_demo_geometry_correct']['per_branch_correct_preference_draws'])
    write(out / 'RESULT.json', final)
    return final


def prepare_phase_transfer():
    corpus = BASE / 'generator_actual_branch_coverage4'
    declared = json.loads((corpus / 'NEXT_MATCHED_PROTOCOL.json').read_text())
    result = json.loads((corpus / 'RESULT.json').read_text())
    if not result['all_phase_data_passed'] or not result['physical_plots_inspected']:
        raise RuntimeError('Complete actual phase data and physical inspection first')
    if RUN.name != declared['run_name']:
        raise RuntimeError('Use the declared separate phase-transfer run directory')
    plan = json.loads((BASE / 'matched_generator_actual_branch_iid256/PROTOCOL.json').read_text())
    plan.pop('preceding_noise_assignment_experiment', None)
    plan.update(declared)
    plan.update(branch_dataset=dict(corpus=str(corpus), normalizer_state=declared['normalizer_state'], repetitions=16, split='train'),
                phase_corpora={str(p): str(corpus / f'switch_{p}/branch_samples') for p in declared['frozen_evaluation']['evaluation_phases']},
                phase_evaluation=True, real_branch_examples=4, repeated_examples_per_update=64,
                noise_coupling='independent', goal_condition_mode='zero_diagnostic',
                optimization=declared['training'] + ' ' + declared['local_method'],
                implementation_status='Full official-model dataset/phase evaluator glue prepared; require full-model preflight before training.')
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), real_training_examples=4, heldout_phase_examples=4, updates_per_arm=256)), flush=True)


def evaluate_phases(run):
    run = Path(run)
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    if plan.get('phase_frozen_evaluation'):
        plan['frozen_evaluation'] = plan['phase_frozen_evaluation']
    results = {}
    for phase in plan['frozen_evaluation']['evaluation_phases']:
        results[str(phase)] = evaluate_branch(run, phase=phase)
    checks = {str(phase): results[str(phase)]['branch_identifiability_passed'] for phase in plan['data']['heldout_phase']['phases']}
    final = dict(execution_completed=True, phases=results, heldout_phase_checks=checks,
                 heldout_phase_transfer_passed=all(checks.values()),
                 training_phase_checks={str(p): results[str(p)]['branch_identifiability_passed'] for p in plan['data']['train']['phases']},
                 real_training_examples=plan['data']['train']['real_examples'] + plan.get('replay_composition', {}).get('native_real_chunks', 0),
                 real_training_branch_examples=plan['data']['train']['real_examples'], real_heldout_phase_examples=plan['data']['heldout_phase']['real_examples'],
                 actual_updates_per_arm=json.loads((run / ARMS[0] / 'RESULT.json').read_text())['actual_optimizer_updates'],
                 native_replay_composition=plan.get('replay_composition'),
                 new_physics_steps=0, horizon_plots_inspected=False,
                 scope=plan['frozen_evaluation']['scope'], next_action=plan['automatic_next_action'])
    write(run / 'frozen_evaluation/RESULT.json', final)
    readback_phase_samples(run)
    return final


def readback_phase_samples(run=RUN):
    """Read every saved draw and actual phase input, with no new sampling."""
    run = Path(run)
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    if plan.get('phase_frozen_evaluation'):
        plan['frozen_evaluation'] = plan['phase_frozen_evaluation']
    endpoint = json.loads((run / 'frozen_evaluation/RESULT.json').read_text())
    if not endpoint['execution_completed']:
        raise RuntimeError('Complete all phase sampling before saved readback')
    torch.set_num_threads(1)
    dataset = ActualBranchGeometryDataset(**plan['branch_dataset'])
    normalizer = dataset.get_normalizer()
    initial = torch.load(plan['branch_dataset']['normalizer_state'], map_location='cpu')['model']
    scale = initial['normalizer.params_dict.action.scale'].double().numpy()
    fields = dict(object=('obj_pos_b', 'obj_ori_b'), history=('last_action',), geometry=(CONTEXT_KEY,))
    if plan.get('robot_state_conditioning'):
        fields['measured_robot']=('joint_pos','project_gravity')
    train_obs = normalizer.normalize({key: torch.stack([dataset[i]['obs'][key] for i in range(dataset.real_case_count)]) for key in dataset[0]['obs']})
    train_features = {name: torch.cat([train_obs[k] for k in keys], dim=-1).reshape(dataset.real_case_count, -1).double().numpy()
                      for name, keys in fields.items()}
    checks, rows = {}, {}
    for phase, result in endpoint['phases'].items():
        folder = run / f'frozen_evaluation/phase_{phase}'
        decomposition = {}
        for variant, reference in result['variants'].items():
            with np.load(folder / f'{variant}.npz') as archive:
                prediction = archive['predictions'].astype(np.float64) * scale
                target = archive['actual_future_command_target'].astype(np.float64) * scale
                total = float(np.mean((prediction - target[None]) ** 2))
                mean_error = float(np.mean((prediction.mean(0) - target) ** 2))
                variance = float(np.mean((prediction - prediction.mean(0)[None]) ** 2))
                own = ((prediction - target[None]) ** 2).mean((2, 3))
                other = ((prediction - target[::-1][None]) ** 2).mean((2, 3))
                checks[phase + '_' + variant + '_saved_metrics_exact'] = bool(np.isclose(total, reference['normalized_mse'], rtol=1e-5)
                    and np.isclose(total, mean_error + variance, rtol=1e-10)
                    and np.array_equal((own < other).sum(0), reference['per_branch_correct_preference_draws'])
                    and np.array_equal(archive['sample_seeds'], plan['frozen_evaluation']['sample_seeds']))
                toward_first = ((prediction - target[0][None, None]) ** 2).mean((2, 3)) < ((prediction - target[1][None, None]) ** 2).mean((2, 3))
                decomposition[variant] = dict(total_mse=total, draw_mean_mse=mean_error, draw_variance=variance,
                    same_future_under_both_prompts_draws=int((toward_first[:, 0] == toward_first[:, 1]).sum()),
                    both_prompts_correct_draws=int((toward_first[:, 0] & ~toward_first[:, 1]).sum()))
        phase_dataset = ActualBranchGeometryDataset(plan['phase_corpora'][phase], plan['branch_dataset']['normalizer_state'])
        obs = normalizer.normalize({key: torch.stack([phase_dataset[i]['obs'][key] for i in (0, 1)]) for key in phase_dataset[0]['obs']})
        inputs = {}
        for name, keys in fields.items():
            current = torch.cat([obs[k] for k in keys], dim=-1).reshape(2, -1).double().numpy()
            train = train_features[name]
            inputs[name] = dict(nearest_train_case_normalized_rms=np.sqrt(((current[:, None] - train[None]) ** 2).mean(-1)).min(1).tolist(),
                fraction_coordinates_outside_fitted_branch_range=((current < train.min(0)) | (current > train.max(0))).mean(1).tolist(),
                within_pair_normalized_rms=float(np.sqrt(np.mean((current[0] - current[1]) ** 2))))
        rows[phase] = dict(split=result['evaluation_split'], decomposition=decomposition, actual_input_comparison=inputs)
    report = dict(execution_completed=True, saved_metrics_reproduced=all(checks.values()), checks=checks, phases=rows,
                  new_optimizer_updates=0, new_sample_draws=0, new_physics_steps=0,
                  native_replay_composition=plan.get('replay_composition'),
                  fitted_branch_examples=dataset.real_case_count,
                  scope='All saved32draws and actual input comparison against the explicit fitted branch examples only; broader native replay is reported separately. Averaged predictions are diagnostic decomposition only, not a deployed controller or replacement primary metric. Coordinate ranges from these few branch cases cannot establish global distribution support or a sole failure cause.')
    write(run / 'frozen_evaluation/SAVED_PHASE_READBACK.json', report)
    if not report['saved_metrics_reproduced']:
        raise RuntimeError('Saved full phase draws differ from reported metrics')
    print(json.dumps(dict(saved_phase_metrics_reproduced=True, phases={p: v['decomposition']['trained_demo_geometry_correct'] for p, v in rows.items()})), flush=True)
    return report


def probe_phase_inputs(run=RUN, geometry_only=False):
    """Frozen full-model sensitivity probe; changed inputs have no physics label."""
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Full-model probe sampling requires the retained compute step')
    run = Path(run)
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    endpoint = json.loads((run / 'RESULT.json').read_text())
    readback = json.loads((run / 'frozen_evaluation/SAVED_PHASE_READBACK.json').read_text())
    if endpoint['heldout_phase_transfer_passed'] or not readback['saved_metrics_reproduced']:
        raise RuntimeError('Require the completed negative phase endpoint and saved metric readback')
    out = run / 'frozen_evaluation' / ('geometry_phase_probe' if geometry_only else 'input_branch_probe')
    out.mkdir(exist_ok=False)
    seeds = plan['frozen_evaluation']['sample_seeds'][:8]
    modes = dict(actual=(), train_history=('last_action',),
                 train_object=('obj_pos_b', 'obj_ori_b'),
                 train_object_and_history=('obj_pos_b', 'obj_ori_b', 'last_action'))
    phases = [218, 258]
    donors = {}
    if geometry_only:
        if not endpoint['horizon_plots_inspected']:
            raise RuntimeError('Inspect complete phase endpoint before geometry sensitivity probe')
        phases = [int(p) for p, result in endpoint['phases'].items() if not result['branch_identifiability_passed']]
        modes = dict(actual=(), **{f'train_geometry_{p}': (CONTEXT_KEY,) for p in plan['data']['train']['phases']})
        for p in plan['data']['train']['phases']:
            dataset = ActualBranchGeometryDataset(plan['phase_corpora'][str(p)], plan['normalizer_state'])
            donors[f'train_geometry_{p}'] = torch.stack([dataset[i]['obs'][CONTEXT_KEY] for i in (0, 1)]).cuda()
    protocol = dict(checkpoint=str(run / 'demo_geometry/checkpoints/endpoint.ckpt'),
        heldout_phases=[218, 258], replacement_phase=178, sample_seeds=seeds,
        modes=modes, contexts=['correct', 'wrong'], inference_steps=16, new_optimizer_updates=0,
        scope='Full-model diagnostic only. Replace declared raw object/history fields with an earlier actual fitted-phase observation, retain the real later-phase geometry and zero normalized goal. These mixed inputs are not a consistent physical state; target distances are descriptive, not counterfactual ground truth or a replacement validation result.',
        automatic_next_action='Inspect exact actual-input replay and per-mode context sensitivity, plus all saved prediction distances. Use the changed branch evidence to select a bounded actual-data coverage experiment. Do not extend the4case budget or deploy mixed-input substitutions.')
    if geometry_only:
        protocol.update(heldout_phases=plan['data']['heldout_phase']['phases'], evaluated_failed_phases=phases,
            replacement_phase=plan['data']['train']['phases'],
            scope='Frozen full-model sensitivity only. Keep actual current object/history/goal fields and replace only original-demo geometry with each of the four fitted phases, preserving source order96/90 before correct/wrong swap. All donors reported; no best donor selection. Cross-phase mixtures are physically inconsistent and target distances are descriptive, not counterfactual labels, deployable retrieval or primary success.',
            automatic_next_action='Compare all donor phases and both contexts, requiring exact actual/self-donor replay and full frozen state. If changing geometry restores strong context response while actual geometry fails, inspect representation-to-state/phase correspondence; otherwise inspect joint conditioning rather than declaring geometry the sole cause. No new training, physics or extension of512 budget follows from a favorable donor alone.')
    write(out / 'PROTOCOL.json', protocol)
    with (run / 'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
        state = torch.load(f, pickle_module=dill, map_location='cpu')['state_dicts']['model']
    policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
    restore_geometry_state(policy, state, 'demo_geometry', goal_condition_mode='zero_diagnostic')
    policy.cuda().eval().requires_grad_(False)
    earlier = {}
    if not geometry_only:
        prior = ActualBranchGeometryDataset(plan['phase_corpora']['178'], plan['normalizer_state'])
        earlier = {k: prior[0]['obs'][k].cuda() for k in prior[0]['obs']}
    results, checks = {}, {}
    for phase in phases:
        dataset = ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)], plan['normalizer_state'])
        actual = {k: torch.stack([dataset[i]['obs'][k] for i in (0, 1)]).cuda() for k in dataset[0]['obs']}
        target = torch.stack([dataset[i]['action'] for i in (0, 1)]).cuda()
        normalized_target = policy.normalizer['action'].normalize(target)
        rows = {}
        for mode, fields in modes.items():
            obs = {k: v.clone() for k, v in actual.items()}
            for key in fields:
                obs[key] = donors[mode].clone() if geometry_only else earlier[key][None].expand_as(obs[key]).clone()
            checks[f'{phase}_{mode}_only_declared_fields_changed'] = all(torch.equal(obs[k], actual[k]) for k in actual if k not in fields)
            predictions, stats = {}, {}
            for context in ('correct', 'wrong'):
                inputs = dict(obs)
                if context == 'wrong':
                    inputs[CONTEXT_KEY] = obs[CONTEXT_KEY].flip(0)
                draws = []
                with torch.inference_mode():
                    for seed in seeds:
                        pair = []
                        for branch in (0, 1):
                            torch.manual_seed(seed)
                            pair.append(policy.predict_action({k: v[branch:branch + 1] for k, v in inputs.items()}))
                        draws.append(torch.cat(pair))
                    prediction = torch.stack(draws)
                    normalized = policy.normalizer['action'].normalize(prediction)
                    own = (normalized - normalized_target[None]).square().mean((2, 3))
                    other = (normalized - normalized_target.flip(0)[None]).square().mean((2, 3))
                predictions[context] = prediction.cpu().numpy()
                stats[context] = dict(descriptive_target_mse=float(own.mean()),
                    per_branch_correct_preference_draws=(own < other).sum(0).cpu().tolist())
                np.savez_compressed(out / f'phase{phase}_{mode}_{context}.npz', predictions=predictions[context],
                    actual_later_phase_targets=target.cpu().numpy(), sample_seeds=np.asarray(seeds))
                if mode == 'actual' or (geometry_only and mode == f'train_geometry_{phase}'):
                    variant = 'trained_demo_geometry_' + context
                    with np.load(run / f'frozen_evaluation/phase_{phase}/{variant}.npz') as saved:
                        checks[f'{phase}_{mode}_{context}_actual_samples_replayed_exactly'] = np.array_equal(predictions[context], saved['predictions'][:len(seeds)])
            scale = policy.normalizer['action'].params_dict['scale'].detach().cpu().numpy()
            stats['same_noise_context_change_mse'] = float(np.mean(((predictions['correct'] - predictions['wrong']) * scale) ** 2))
            rows[mode] = stats
            print(json.dumps(dict(phase=phase, mode=mode, **stats)), flush=True)
        results[str(phase)] = rows
        write(out / 'PARTIAL_RESULT.json', results)
    checks['full_model_state_unchanged'] = all(torch.equal(v.detach().cpu(), state[k]) for k, v in policy.state_dict().items())
    report = dict(execution_completed=True, checks=checks, checks_passed=all(checks.values()), phases=results,
                  full_parameter_count=sum(p.numel() for p in policy.parameters()),
                  new_optimizer_updates=0, new_physics_steps=0, sampling_draws_per_case=8,
                  scope=protocol['scope'], next_action=protocol['automatic_next_action'])
    write(out / 'RESULT.json', report)
    if not report['checks_passed']:
        raise RuntimeError('Actual full-model input probe failed its replay or state checks')
    return report


def audit_noise_coupling():
    """Probe official cross-example noise pairing, without a model update."""
    from sugar_il.policy.generator import noise_assignment
    torch.set_num_threads(1)
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    result = json.loads((RUN / 'RESULT.json').read_text())
    if not result['execution_completed']:
        raise RuntimeError('Read the complete actual-branch endpoint first')
    dataset = ActualBranchGeometryDataset(**plan['branch_dataset'])
    normalizer = dataset.get_normalizer()['action']
    target = normalizer.normalize(torch.stack([dataset[i]['action'] for i in (0, 1)]))
    direction = (target[0] - target[1]).flatten()
    direction = direction / direction.norm()
    batches = json.loads((RUN / 'demo_geometry/BATCH_ORDER.json').read_text())
    rows = []
    for probe, batch in enumerate(batches[:32]):
        labels = torch.tensor(batch) % 2
        trajectory = target[labels]
        seed = 272130 + probe
        noise = torch.randn(trajectory.shape, generator=torch.Generator().manual_seed(seed))
        assignment = noise_assignment(trajectory, noise)
        before = noise.flatten(1) @ direction
        after = noise[assignment].flatten(1) @ direction
        accuracy_before = float(((before > 0) == (labels == 0)).float().mean())
        accuracy_after = float(((after > 0) == (labels == 0)).float().mean())
        median = after.sort().values[31:33].mean()
        rows.append(dict(probe_seed=seed, actual_training_batch_index=probe,
            branch_labels=labels.tolist(), assignment=assignment.tolist(),
            original_projection=before.tolist(), assigned_projection=after.tolist(),
            zero_threshold_accuracy_before=accuracy_before, zero_threshold_accuracy_after=accuracy_after,
            median_threshold_accuracy_after=float(((after > median) == (labels == 0)).float().mean()),
            assigned_projection_mean_by_branch=[float(after[labels == branch].mean()) for branch in (0, 1)]))
    decomposition = {}
    for variant in result['variants']:
        with np.load(RUN / 'frozen_evaluation' / f'{variant}.npz') as arrays:
            pred = normalizer.normalize(torch.from_numpy(arrays['predictions']))
            mean = pred.mean(0)
            mse_mean = float((mean - target).square().mean())
            variance = float((pred - mean[None]).square().mean())
            total = float((pred - target[None]).square().mean())
            if not np.isclose(total, mse_mean + variance, rtol=1e-5):
                raise RuntimeError('Saved branch error decomposition differs')
            e0 = (pred - target[0]).square().mean((2, 3))
            e1 = (pred - target[1]).square().mean((2, 3))
            selected = (e1 < e0).long()
            decomposition[variant] = dict(total_mse=total, draw_mean_mse=mse_mean, draw_variance=variance,
                same_target_selected_for_both_prompts=int((selected[:, 0] == selected[:, 1]).sum()),
                both_prompts_correct=int(((selected[:, 0] == 0) & (selected[:, 1] == 1)).sum()))
    summary = {key: float(np.mean([row[key] for row in rows])) for key in
               ('zero_threshold_accuracy_before', 'zero_threshold_accuracy_after', 'median_threshold_accuracy_after')}
    report = dict(execution_completed=True, optimizer_updates=0, new_model_samples=0, new_physics_steps=0,
        official_noise_assignment_source='SUGAR/source/sugar_il/sugar_il/policy/generator.py:noise_assignment',
        fresh_noise_probe_summary=summary, per_probe=rows, saved_sample_decomposition=decomposition,
        scope='Fresh CPU Gaussian probes on32 recorded balanced branch batches, not the exact historical training noise. The analytic direction uses the two diagnostic target labels only to measure target-dependent pairing; it is not an actor input or learned classifier.',
        interpretation='Global official noise assignment couples noise to different branch targets. This may let noise carry branch information while inference draws noise independently of the selected prompt. Distributional evidence is not proof that this alone caused the endpoint failure; a matched full-model independent-noise comparison is the next isolating experiment if the coupling is strong.')
    write(RUN / 'frozen_evaluation/NOISE_COUPLING_READBACK.json', report)
    print(json.dumps(dict(summary=summary, decomposition=decomposition)), flush=True)


def prepare_independent_noise():
    previous = BASE / 'matched_generator_actual_branch256'
    if RUN == previous:
        raise RuntimeError('Preserve the completed official-assignment diagnostic')
    evidence = json.loads((previous / 'frozen_evaluation/NOISE_COUPLING_READBACK.json').read_text())
    if evidence['fresh_noise_probe_summary']['zero_threshold_accuracy_after'] < .8:
        raise RuntimeError('Require evidence for strong target-dependent noise coupling')
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    plan['noise_coupling'] = 'independent'
    plan['preceding_noise_assignment_experiment'] = str(previous)
    plan['optimization'] = 'Local independent-noise method variant of the full official Generator loss: omit only target-dependent cross-example noise_assignment. Keep complete model, normalization, random timestep stream, original scheduler, epsilon MSE, unchanged official workspace and AdamW, same256update budget and seed. No replacement model or additional synthetic data.'
    plan['automatic_next_action'] = 'Inspect complete same-budget results against the preceding official-assignment experiment and all unchanged branch identifiability criteria. If positive, assess actual branching-data coverage before broader matched training; two TRAIN branches cannot prove generalization or generated physics. If negative, inspect full fitting/conditioning evidence before another budget. Keep all previous negative endpoints.'
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), noise_coupling='independent', updates_per_arm=256)), flush=True)


def prepare_replay():
    previous = BASE / 'matched_generator_branch_phase_iid256'
    endpoint = json.loads((previous / 'RESULT.json').read_text())
    probe = json.loads((previous / 'frozen_evaluation/input_branch_probe/RESULT.json').read_text())
    if not endpoint['execution_completed'] or endpoint['heldout_phase_transfer_passed'] or not all(probe['checks'].values()):
        raise RuntimeError('Require completed negative phase transfer and exact frozen input probe')
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    native = json.loads((BASE / 'matched_generator_goal_removed_iid16/PROTOCOL.json').read_text())
    plan['phase_frozen_evaluation'] = plan['frozen_evaluation']
    for key in ('frozen_evaluation', 'frozen_validation', 'native_train_corpus', 'additional_paired_train'):
        plan[key] = native[key]
    for key in ('repeated_examples_per_update', 'repetitions_per_real_training_example', 'implementation_status'):
        plan.pop(key, None)
    plan.update(run_name=RUN.name, epochs=16, batch_size=128, seed=272051, checkpoint_every=4,
        optimizer=dict(lr=5e-5, weight_decay=1e-4, betas=[.95, .999]), scheduler=dict(name='cosine', warmup_steps=32),
        replay_dataset=dict(corpus=plan['branch_dataset']['corpus'], normalizer_state=plan['normalizer_state'],
            native_corpus=plan['native_train_corpus'], parent_checkpoint=str(PARENT), paired_train=plan['additional_paired_train']),
        training='Full official training loop and shuffled DataLoader. Each retained native chunk once per epoch; each of4early branch cases repeated floor(native_count/12) times, approximately25% branch rows. Repetitions are noise exposures, not independent observations. Both arms share complete release initialization, frozen old TRAIN normalizer, actual batch order, fresh AdamW and fixed16epoch budget.',
        normalization='All prior released/old TRAIN geometry statistics frozen exactly. Prior statistics include source96/90 geometry; excluded future labels are disjoint in this adaptation only, not untouched/pretraining-held-out evidence.',
        conditioning='Full actual object and causal t-5 command history retained, normalized goal9 zero in TRAIN and all frozen calls. Only the new8x21 original-demo geometry differs between matched arms. No sample/source identifier or future actual state enters model observation.',
        checkpoint='Keep complete initial model, actual batch order, full model/Adam every4epochs and full endpoint. No checkpoint selection or extension.',
        comparison_scope='Native dataset, weighting and update count differ from previous experiments. Prior absolute errors are descriptive; this run isolates added geometry through its two exactly matched arms, not a single-factor historical replay comparison.',
        automatic_next_action='Complete all raw dataset readback and full-model mixed-batch preflight, then both fixed-budget arms serially and all native/phase frozen sampling and saved readbacks. Inspect every horizon panel. Require both native conditioning and both heldout branch phases to pass before considering generated-command prerequisites; retain original-position7/8 and missing camera evidence. On failure diagnose saved source/horizon/branch evidence and actual coverage before a new bounded experiment; no same-budget extension, generated physics or SMP claim.',
        post_prediction_position_floor=None, generator_training_started=False)
    dataset = ActualBranchReplayDataset(**plan['replay_dataset'])
    plan['replay_composition'] = dataset.composition
    plan['actual_optimizer_updates'] = plan['exact_optimizer_updates_per_arm'] = plan['epochs'] * math.ceil(len(dataset) / plan['batch_size'])
    plan['optimization'] = plan['training'] + ' ' + plan['local_method']
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), composition=dataset.composition, updates_per_arm=plan['actual_optimizer_updates'])), flush=True)


def preflight_replay():
    preflight('BRANCH_PREFLIGHT.json')
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    dataset = ActualBranchReplayDataset(**plan['replay_dataset'])
    branch_count = dataset.branch.real_case_count
    checks = dict(branch_preflight_passed=json.loads((RUN / 'BRANCH_PREFLIGHT.json').read_text())['passed'],
        composition_exact=dataset.composition == plan['replay_composition'],
        budget_exact=plan['actual_optimizer_updates'] == plan['epochs'] * math.ceil(len(dataset) / plan['batch_size']),
        native_sources_exclude_branch_and_validation=not bool(set(dataset.composition['native_sources']) & {96, 90, *plan['frozen_evaluation']['validation_sources']}))
    all_exact = True
    for i in range(len(dataset)):
        sample = dataset[i]
        original = dataset.native[dataset.native_indices[i]] if i < dataset.native_count else dataset.branch[(i - dataset.native_count) % branch_count]
        all_exact &= torch.equal(sample['action'], original['action']) and tuple(sample['obs']) == dataset.observation_keys
        all_exact &= (set(original['obs']) - set(sample['obs'])) == (set() if i < dataset.native_count else {'joint_pos', 'project_gravity'})
        all_exact &= all(torch.equal(value, original['obs'][key]) for key, value in sample['obs'].items()) and int(sample['sample_index']) == i
    checks['all_actual_native_and_repeated_branch_fields_exact'] = bool(all_exact)
    generator = torch.Generator().manual_seed(plan['seed'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)
    epoch_ids, batch = [], None
    for actual_batch in loader:
        if batch is None: batch = actual_batch
        epoch_ids.extend(actual_batch['sample_index'].tolist())
    checks['entire_shuffled_epoch_collates_exactly_once_per_row'] = sorted(epoch_ids) == list(range(len(dataset)))
    from torch.utils.data import default_collate
    native_first = default_collate([dataset[0], dataset[dataset.native_count]])
    branch_first = default_collate([dataset[dataset.native_count], dataset[0]])
    checks['both_collation_orders_same_schema_and_values'] = native_first['obs'].keys() == branch_first['obs'].keys() and all(torch.equal(value.flip(0), branch_first['obs'][key]) for key, value in native_first['obs'].items())
    ids = batch['sample_index'].tolist()
    branches = [i for i in ids if i >= dataset.native_count]
    checks['mixed_real_batch_has_native_and_all_declared_branches'] = len(branches) < 128 and {(i - dataset.native_count) % branch_count for i in branches} == set(range(branch_count))
    initial = torch.load(plan['normalizer_state'], map_location='cpu')['model']
    gradients = {}
    sampled = []
    for arm in ARMS:
        policy = warm_start_geometry_policy(PARENT, arm, goal_condition_mode='zero_diagnostic', noise_coupling='independent')
        policy.set_normalizer(dataset.get_normalizer()); policy.normalizer.requires_grad_(False); policy.eval()
        checks[arm + '_full_initial_state_exact'] = policy.state_dict().keys() == initial.keys() and all(torch.equal(value, initial[key]) for key, value in policy.state_dict().items())
        original_branch = default_collate([dataset.branch[i] for i in range(branch_count)])['obs']
        schema_branch = default_collate([dataset[dataset.native_count + i] for i in range(branch_count)])['obs']
        with torch.no_grad():
            checks[arm + '_official_schema_preserves_complete_branch_tokens'] = torch.equal(
                policy.obs_encoder(policy.normalizer.normalize(original_branch), training=False),
                policy.obs_encoder(policy.normalizer.normalize(schema_branch), training=False))
        trajectory = policy.normalizer['action'].normalize(batch['action'])
        torch.manual_seed(272125)
        noise = torch.randn(trajectory.shape)
        times = torch.randint(0, policy.noise_scheduler.config.num_train_timesteps, (128,)).long()
        expected = policy.noise_scheduler.add_noise(trajectory, noise, times)
        denoiser_inputs, goal_inputs = [], []
        hook = policy.model.register_forward_pre_hook(lambda module, args, kwargs: denoiser_inputs.append((args[0].detach().clone(), args[1].detach().clone())), with_kwargs=True)
        goal_hook = policy.obs_encoder.target_state_net[0].register_forward_pre_hook(lambda module, args: goal_inputs.append(args[0].detach().clone()))
        torch.manual_seed(272125)
        loss = policy.compute_loss(batch, training=True)
        hook.remove(); goal_hook.remove()
        checks[arm + '_exact_iid_noisy_input_and_times'] = torch.equal(denoiser_inputs[0][0], expected) and torch.equal(denoiser_inputs[0][1], times)
        checks[arm + '_all_goal_inputs_zero'] = not bool(goal_inputs[0][:, :9].count_nonzero())
        loss.backward()
        gradients[arm] = float(policy.obs_encoder.target_state_net[0].weight.grad[:, 9:].norm())
        checks[arm + '_finite_full_backward_and_context_gradient'] = bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in policy.parameters()) and (gradients[arm] == 0 if arm == 'zero_context' else gradients[arm] > 0)
        policy.zero_grad(set_to_none=True)
        with torch.inference_mode():
            torch.manual_seed(272126)
            sampled.append(policy.predict_action({key: value[:8] for key, value in batch['obs'].items()}))
        checks[arm + '_full_state_unchanged'] = all(torch.equal(value, initial[key]) for key, value in policy.state_dict().items())
    checks['full_16step_initial_samples_exact_and_finite'] = torch.equal(*sampled) and bool(torch.isfinite(sampled[0]).all())
    report = dict(execution_completed=True, passed=all(checks.values()), checks=checks, composition=dataset.composition,
        mixed_batch_indices=ids, mixed_batch_branch_rows=len(branches), new_column_gradient_norms=gradients,
        full_parameter_count=sum(p.numel() for p in policy.parameters()), optimizer_updates=0, physics_steps=0,
        scope='All actual replay rows compared to their original formatter outputs; full128row mixed-batch IID loss/backward and complete initial sampling. Dataset repetitions are not independent trajectories.')
    write(RUN / 'PREFLIGHT.json', report)
    if not report['passed']:
        raise RuntimeError('Full actual replay preflight failed')
    print(json.dumps(report), flush=True)


def probe_replay_objective(run=RUN):
    """Frozen full-model gradients of actual native and early-branch objectives."""
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Full-corpus gradient probe requires the retained compute step')
    run = Path(run)
    endpoint = json.loads((run / 'RESULT.json').read_text())
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    if not endpoint['execution_completed'] or not endpoint['all_horizon_plots_inspected'] or endpoint['combined_conditioning_passed']:
        raise RuntimeError('Complete negative matched replay and all horizon inspection first')
    out = run / 'objective_probe'
    declared = json.loads((out / 'PROTOCOL.json').read_text())
    if (out / 'RESULT.json').exists():
        raise RuntimeError('Do not overwrite the completed objective probe')
    dataset = ActualBranchReplayDataset(**plan['replay_dataset'])
    with (run / 'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as stream:
        initial = torch.load(stream, pickle_module=dill, map_location='cpu')['state_dicts']['model']
    policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
    restore_geometry_state(policy, initial, 'demo_geometry', goal_condition_mode='zero_diagnostic')
    from scripts.sugar.demo_following.demo_future.generator_noise_coupling import configure_noise_coupling
    configure_noise_coupling(policy, 'independent')
    policy.cuda().eval(); policy.normalizer.requires_grad_(False)
    parameters = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    checks = dict(composition_exact=dataset.composition == plan['replay_composition'])
    weight = dataset.composition['branch_row_fraction']
    from torch.utils.data import DataLoader, Subset
    native = Subset(dataset, range(dataset.native_count))
    branch = Subset(dataset, [dataset.native_count + i % dataset.branch.real_case_count for i in range(128)])
    rows = []
    for seed in declared['noise_seeds']:
        gradients, losses = {}, {}
        for name, data in (('native', native), ('branch', branch)):
            policy.zero_grad(set_to_none=True)
            # Separate fixed streams prevent native batch count from changing
            # branch noise. Every actual native row participates once per seed.
            torch.manual_seed(seed + (100 if name == 'branch' else 0))
            total_loss, seen = 0., 0
            for batch in DataLoader(data, batch_size=128, shuffle=False, num_workers=0):
                count = len(batch['action'])
                gpu = dict(action=batch['action'].cuda(), obs={k: v.cuda() for k, v in batch['obs'].items()})
                loss = policy.compute_loss(gpu, training=True)
                (loss * (count / len(data))).backward()
                total_loss += float(loss.detach()) * count / len(data)
                seen += count
            checks[f'{seed}_{name}_all_rows_exact'] = seen == len(data)
            checks[f'{seed}_{name}_finite_loss_and_gradients'] = bool(np.isfinite(total_loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for _, p in parameters)
            gradients[name] = {key: (torch.zeros_like(p) if p.grad is None else p.grad.detach()).cpu().clone() for key, p in parameters}
            losses[name] = total_loss
        groups = {}
        for group in ('full_model', 'new_geometry_columns', 'denoiser'):
            nnorm = bnorm = dot = 0.
            for key, ng in gradients['native'].items():
                bg = gradients['branch'][key]
                if group == 'new_geometry_columns':
                    if key != 'obs_encoder.target_state_net.0.weight': continue
                    ng, bg = ng[:, 9:], bg[:, 9:]
                elif group == 'denoiser' and not key.startswith('model.'):
                    continue
                ng, bg = ng.double(), bg.double()
                nnorm += float(ng.square().sum()); bnorm += float(bg.square().sum()); dot += float((ng * bg).sum())
            groups[group] = dict(native_gradient_norm=nnorm ** .5, branch_gradient_norm=bnorm ** .5,
                cosine=dot / (nnorm * bnorm) ** .5 if nnorm * bnorm > 0 else None,
                negative_gradient_dot=dot < 0,
                branch_dot_weighted_training_gradient=(1 - weight) * dot + weight * bnorm,
                native_dot_weighted_training_gradient=(1 - weight) * nnorm + weight * dot)
        rows.append(dict(seed=seed, losses=losses, gradient_groups=groups))
        write(out / 'PARTIAL_RESULT.json', rows)
        print(json.dumps(rows[-1]), flush=True)
    policy.zero_grad(set_to_none=True)
    checks['complete_frozen_state_unchanged'] = policy.state_dict().keys() == initial.keys() and all(torch.equal(value.cpu(), initial[key]) for key, value in policy.state_dict().items())
    report = dict(execution_completed=True, checks_passed=all(checks.values()), checks=checks, rows=rows,
        native_rows_per_seed=dataset.native_count, real_branch_examples=dataset.branch.real_case_count, branch_noise_rows_per_seed=128,
        branch_training_row_weight=weight, full_parameter_count=sum(p.numel() for p in policy.parameters()),
        new_optimizer_updates=0, new_physics_steps=0, scope=declared['scope'], next_action=declared['next_action'])
    write(out / 'RESULT.json', report)
    if not report['checks_passed']:
        raise RuntimeError('Frozen complete objective probe failed its state or data checks')


def readback_replay_coverage(run=RUN):
    """Descriptive actual-data neighbors; no fitted model or new prediction draw."""
    run = Path(run)
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    endpoint = json.loads((run / 'RESULT.json').read_text())
    if not endpoint['execution_completed'] or not endpoint['all_horizon_plots_inspected']:
        raise RuntimeError('Complete matched replay and inspect all actual horizon evidence first')
    out = run / 'frozen_evaluation/REPLAY_COVERAGE_READBACK.json'
    if out.exists():
        raise RuntimeError('Do not overwrite completed coverage readback')
    torch.set_num_threads(1)
    dataset = ActualBranchReplayDataset(**plan['replay_dataset'])
    from torch.utils.data import default_collate
    # Count each real early branch once, excluding the repeated noise exposures.
    branch_count = dataset.branch.real_case_count
    batch = default_collate([dataset[i] for i in range(dataset.native_count + branch_count)])
    normalizer = dataset.get_normalizer()
    normalized = normalizer.normalize(batch['obs'])
    fields = dict(object=('obj_pos_b', 'obj_ori_b'), history=('last_action',), geometry=(CONTEXT_KEY,))
    train = {name: torch.cat([normalized[k] for k in keys], dim=-1).reshape(len(batch['action']), -1).numpy()
             for name, keys in fields.items()}
    targets = normalizer['action'].normalize(batch['action']).numpy()
    metadata = []
    for i in dataset.native_indices:
        episode, start, context_index = dataset.native.indices[i]
        row = dataset.native.timeline_metadata[episode]
        metadata.append(dict(source=row['source'], role=row['role'], official_row=start, context_row=context_index,
                             label_path=row['label_path'], context_path=row['context_path']))
    metadata.extend(dict(row, role='fitted_actual_branch') for row in dataset.branch.timeline_metadata)
    checks = dict(actual_composition_exact=dataset.composition == plan['replay_composition'],
        each_real_example_counted_once=len(metadata) == dataset.native_count + branch_count,
        no_heldout_native_sources=not bool(set(dataset.composition['native_sources']) & {96, 90}))
    rows = {}
    for phase in plan['phase_frozen_evaluation']['evaluation_phases']:
        branch = ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)], plan['normalizer_state'])
        query = default_collate([branch[i] for i in (0, 1)])
        obs = normalizer.normalize(query['obs'])
        query_targets = normalizer['action'].normalize(query['action']).numpy()
        distances = {}
        for name, keys in fields.items():
            feature = torch.cat([obs[k] for k in keys], dim=-1).reshape(2, -1).numpy()
            distances[name] = np.mean((feature[:, None] - train[name][None]) ** 2, axis=-1)
        modes = dict(geometry=distances['geometry'], state=(distances['object'] + distances['history']) / 2,
                     state_and_geometry=sum(distances.values()) / 3)
        phase_rows = []
        for i in (0, 1):
            target_distance = np.mean((targets - query_targets[i]) ** 2, axis=(1, 2))
            other_distance = np.mean((targets - query_targets[1 - i]) ** 2, axis=(1, 2))
            neighbors = {}
            for name, matrix in modes.items():
                indices = np.argsort(matrix[i], kind='stable')[:5]
                neighbors[name] = [dict(training_index=int(j), metadata=metadata[j], normalized_input_rms=float(matrix[i, j] ** .5),
                    target_command_mse=float(target_distance[j]), other_target_command_mse=float(other_distance[j]),
                    stored_command_nearer_own_branch=bool(target_distance[j] < other_distance[j])) for j in indices]
            oracle = int(target_distance.argmin())
            phase_rows.append(dict(branch=i, nearest_input_examples=neighbors,
                label_aware_best_stored_command=dict(training_index=oracle, metadata=metadata[oracle],
                    target_command_mse=float(target_distance[oracle]), other_target_command_mse=float(other_distance[oracle])),
                minimum_mse_among_all_stored_commands=float(target_distance[oracle]),
                paired_label_mean_mse=float(np.mean((query_targets[i] - query_targets.mean(0)) ** 2))))
        if phase in plan['data']['train']['phases']:
            checks[f'{phase}_fitted_geometry_self_neighbor_exact'] = all(float(distances['geometry'][i].min()) == 0. for i in (0, 1))
        rows[str(phase)] = phase_rows
    report = dict(execution_completed=True, checks_passed=all(checks.values()), checks=checks, phases=rows,
        native_real_chunks=dataset.native_count, real_fitted_branch_examples=branch_count, distinct_native_sources=len(dataset.composition['native_sources']),
        field_distance_rule='Squared normalized-coordinate mean within object/history/geometry separately; state averages object/history equally, combined averages all3equally. No target or source ID selects input neighbors. First5stable neighbors are descriptive, not a deployed retrieval model.',
        scope='Actual TRAIN coverage readback, no fitted substitute model, optimizer or new sampling/physics. The label-aware minimum is an oracle over stored commands only and is not an achievable predictor metric or a model-class lower bound. Neighbor similarity alone cannot prove sufficient coverage, ambiguity or causation.',
        new_optimizer_updates=0, new_sample_draws=0, new_physics_steps=0)
    write(out, report)
    if not report['checks_passed']:
        raise RuntimeError('Actual data coverage readback failed')
    print(json.dumps({phase: [dict(branch=x['branch'], geometry_source=x['nearest_input_examples']['geometry'][0]['metadata'],
        combined_source=x['nearest_input_examples']['state_and_geometry'][0]['metadata'], best_stored=x['label_aware_best_stored_command'], mean_baseline=x['paired_label_mean_mse']) for x in row] for phase, row in rows.items()}), flush=True)


def prepare_interpolation_replay():
    corpus = BASE / 'generator_actual_branch_coverage_bracket'
    actual = json.loads((corpus / 'RESULT.json').read_text())
    plan = json.loads((corpus / 'NEXT_MATCHED_PROTOCOL.json').read_text())
    if not actual['all_phase_data_passed'] or not actual['physical_plots_inspected'] or RUN.name != plan['run_name']:
        raise RuntimeError('Use the declared interpolation run after complete actual-data inspection')
    dataset = ActualBranchReplayDataset(**plan['replay_dataset'])
    if dataset.composition != plan['replay_composition'] or plan['epochs'] * math.ceil(len(dataset) / plan['batch_size']) != plan['actual_optimizer_updates']:
        raise RuntimeError('Interpolation actual data or budget differs')
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), composition=dataset.composition, updates_per_arm=plan['actual_optimizer_updates'])), flush=True)


def prepare_interpolation_fit():
    previous = BASE / 'matched_generator_branch_interpolation_replay16'
    result = json.loads((previous / 'RESULT.json').read_text())
    if not result['execution_completed'] or not result['horizon_plots_inspected']:
        raise RuntimeError('Complete preceding interpolation endpoint and curve inspection first')
    if result['combined_conditioning_passed'] or any(result['phase_evaluation']['training_phase_checks'].values()):
        raise RuntimeError('This diagnostic addresses the observed all-TRAIN-phase fitting failure')
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    plan['frozen_evaluation'] = plan.pop('phase_frozen_evaluation')
    for key in ('replay_dataset', 'replay_composition', 'data_coverage_predecessor',
                'native_train_corpus', 'additional_paired_train', 'frozen_validation'):
        plan.pop(key, None)
    description = ('Full official eight-real-case fitting diagnostic, not a matched native-replay '
        'improvement claim. Fresh complete released model and AdamW; same frozen normalization, '
        'IID noise, zero normalized goal and intact causal history. Eight cases from phases '
        '158/178/298/318, each repeated eight times per balanced64 batch. Exactly512 updates '
        'per arm gives4096 noise exposures per real case, matching the earlier four-case256 '
        'diagnostic exposure. LR1e-4, seed272084, cosine warmup16. No new data or physics.')
    plan.update(run_name=RUN.name, seed=272084, epochs=512, batch_size=64,
        actual_optimizer_updates=512, exact_optimizer_updates_per_arm=512,
        checkpoint_every=64, repeated_examples_per_update=64,
        optimizer=dict(lr=1e-4, weight_decay=1e-4, betas=[.95, .999]),
        scheduler=dict(name='cosine', warmup_steps=16),
        training=description, optimization=description, comparison_scope=description,
        checkpoint='Full initial state, full model/Adam every64epochs and endpoint, all actual batch IDs. No best-checkpoint selection or extension.',
        diagnostic_predecessor=str(previous), generator_training_started=False,
        automatic_next_action='Complete both512update arms and all six phases/five conditions/32draws, saved readback and curves. Report all four TRAIN criteria separately from both already-used218/258 interpolation checks. If TRAIN fails, inspect saved full-model denoising/context fitting before any new budget; if TRAIN passes but interpolation fails, investigate representation/phase correspondence without claiming generalization. If both pass, next address native prediction and original goal/position execution prerequisites; no direct generated execution or SMP benefit claim.')
    dataset = ActualBranchGeometryDataset(**plan['branch_dataset'])
    if dataset.real_case_count != 8 or len(dataset) != 64 or RUN.name != 'matched_generator_branch_interpolation_fit512':
        raise RuntimeError('Use the declared eight-case balanced full-model diagnostic')
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), real_examples=8, updates_per_arm=512)), flush=True)


def prepare_dense_fit():
    corpus = BASE / 'generator_actual_branch_coverage_compatible'
    actual = json.loads((corpus / 'RESULT.json').read_text())
    plan = json.loads((corpus / 'NEXT_MATCHED_PROTOCOL.json').read_text())
    if not actual['all_phase_data_passed'] or not actual['physical_plots_inspected'] or RUN.name != plan['run_name']:
        raise RuntimeError('Complete the declared dense actual corpus and use its separate model run')
    dataset = ActualBranchGeometryDataset(**plan['branch_dataset'])
    if dataset.real_case_count != 14 or len(dataset) != 112 or plan['batch_size'] != 112 or plan['actual_optimizer_updates'] != 512:
        raise RuntimeError('Preserve the declared fourteen-case512 diagnostic')
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN),real_examples=14,batch_rows=112,updates_per_arm=512)),flush=True)


def audit_output_range():
    """Read-only bound from actual scheduler and frozen target normalization."""
    torch.set_num_threads(1)
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    endpoint = json.loads((RUN / 'RESULT.json').read_text())
    out = RUN / 'frozen_evaluation/OUTPUT_RANGE_READBACK.json'
    if not endpoint['execution_completed'] or out.exists():
        raise RuntimeError('Require a complete endpoint and preserve any earlier range audit')
    policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
    state = torch.load(plan['normalizer_state'], map_location='cpu')['model']
    restore_geometry_state(policy, state, 'demo_geometry', goal_condition_mode='zero_diagnostic')
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(policy.num_inference_steps)
    bound = float(scheduler.config.clip_sample_range)
    checks = dict(clipping_enabled=bool(scheduler.config.clip_sample),
                  no_dynamic_thresholding=not bool(scheduler.config.thresholding),
                  final_inference_time_zero=int(scheduler.timesteps[-1]) == 0)
    if not all(checks.values()):
        raise RuntimeError('This analytic bound requires the observed fixed clipping at final t0')
    rows = {}
    for phase, reference in endpoint['phases'].items():
        dataset = ActualBranchGeometryDataset(plan['phase_corpora'][phase], plan['normalizer_state'])
        target = policy.normalizer['action'].normalize(torch.stack([dataset[i]['action'] for i in (0, 1)])).double()
        error = (target - target.clamp(-bound, bound)).square()
        with np.load(RUN / f'frozen_evaluation/phase_{phase}/trained_demo_geometry_correct.npz') as saved:
            prediction = policy.normalizer['action'].normalize(torch.from_numpy(saved['predictions'])).double()
        mse = float((prediction - target[None]).square().mean())
        floor = float(error.mean())
        checks[phase + '_saved_predictions_within_scheduler_bound'] = bool(prediction.abs().max() <= bound + 1e-5)
        checks[phase + '_bound_below_actual_error'] = floor <= mse + 1e-7
        checks[phase + '_saved_primary_mse_reproduced'] = bool(np.isclose(mse, reference['variants']['trained_demo_geometry_correct']['normalized_mse'], rtol=1e-5))
        rows[phase] = dict(split=reference['evaluation_split'], target_min=float(target.min()), target_max=float(target.max()),
            fraction_target_coordinates_outside_bound=float((target.abs() > bound + 1e-6).double().mean()),
            unavoidable_clipping_mse=floor, per_branch_clipping_mse=error.mean((1, 2)).tolist(),
            per_channel_clipping_mse={name: float(error[..., sl].mean()) for name, sl in
                dict(joint=slice(0, 29), linear=slice(29, 32), angular=slice(32, 35), contact=slice(35, 36)).items()},
            actual_sample_mse=mse, floor_fraction_of_actual_mse=floor / mse,
            mean_baseline_criterion_impossible_from_clipping=floor > .5 * reference['two_target_mean_normalized_mse'])
    report = dict(execution_completed=True, checks=checks, checks_passed=all(checks.values()), phases=rows,
        scheduler=dict(clip_sample=True, clip_sample_range=bound, prediction_type=scheduler.config.prediction_type,
                       inference_timesteps=scheduler.timesteps.tolist()), new_optimizer_updates=0, new_sample_draws=0,
        new_physics_steps=0, scope='Analytic nearest-point bound for the actual final clipped DDPM output and fixed target normalizer. Does not modify data, normalization, model, solver or primary criteria; small bounds do not establish any alternative root cause.')
    write(out, report)
    if not report['checks_passed']:
        raise RuntimeError('Saved output range or exact metric checks failed')
    print(json.dumps(report), flush=True)


def probe_phase_denoising():
    """Frozen complete denoiser at every training time; never a rollout metric."""
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the retained explicit compute step')
    torch.set_num_threads(8)
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    endpoint = json.loads((RUN / 'RESULT.json').read_text())
    bounds = json.loads((RUN / 'frozen_evaluation/OUTPUT_RANGE_READBACK.json').read_text())
    if not endpoint['horizon_plots_inspected'] or endpoint['heldout_phase_transfer_passed'] or not bounds['checks_passed']:
        raise RuntimeError('Complete negative phase and output-bound evidence first')
    out = RUN / 'frozen_evaluation/denoising_probe'; out.mkdir(exist_ok=False)
    phases = plan['frozen_evaluation']['evaluation_phases']
    seeds = list(range(272160, 272164))
    write(out / 'PROTOCOL.json', dict(checkpoint=str(RUN / 'demo_geometry/checkpoints/endpoint.ckpt'),
        phases=phases, noise_seeds=seeds, diffusion_times='All training times from actual released scheduler',
        conditions=['correct', 'paired_wrong'], goal='normalized zero', full_model=True,
        new_optimizer_updates=0, new_physics_steps=0,
        scope='Actual target-corrupted denoising with paired common IID noise at every training time. Actual target enters noisy trajectory as in training, so this is supervised fitting/sensitivity evidence, never generative sampling, counterfactual ground truth or physical success. Report all seeds/times/cases and retain failed primary criteria.',
        automatic_next_action='Compare fitted and withheld phases across noise levels, correct/wrong epsilon error and reconstructed clean-target error, including the actual16-step solver times. Separate clipping floor from residual denoising error. Use evidence to predeclare a next bounded full-model or representation diagnostic; do not append updates to the512 endpoint.'))
    with (RUN / 'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
        state = torch.load(f, pickle_module=dill, map_location='cpu')['state_dicts']['model']
    policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
    restore_geometry_state(policy, state, 'demo_geometry', goal_condition_mode='zero_diagnostic')
    policy.cuda().eval().requires_grad_(False)
    samples = []
    for phase in phases:
        dataset = ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)], plan['normalizer_state'])
        samples.extend([dataset[i] for i in (0, 1)])
    obs = {k: torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
    target = policy.normalizer['action'].normalize(torch.stack([s['action'] for s in samples]).cuda())
    normalized = policy.normalizer.normalize(obs)
    permutation = torch.arange(len(samples), device='cuda') ^ 1
    wrong = dict(normalized); wrong[CONTEXT_KEY] = normalized[CONTEXT_KEY][permutation]
    scheduler = policy.noise_scheduler
    if scheduler.config.prediction_type != 'epsilon':
        raise RuntimeError('This declared readback requires the actual epsilon prediction model')
    times = list(range(scheduler.config.num_train_timesteps))
    scheduler.set_timesteps(policy.num_inference_steps)
    inference_times = scheduler.timesteps.tolist()
    metrics = {key: np.empty((len(seeds), len(times), 2, len(samples)), dtype=np.float64)
               for key in ('epsilon_mse', 'clean_unclipped_mse', 'clean_clipped_mse')}
    with torch.inference_mode():
        tokens = [policy.obs_encoder(v, training=False) for v in (normalized, wrong)]
        token_delta = (tokens[0] - tokens[1]).square().mean((1, 2)).cpu().tolist()
        for si, seed in enumerate(seeds):
            torch.manual_seed(seed)
            noise = torch.randn((len(phases), 8, 36), device='cuda').repeat_interleave(2, dim=0)
            for ti, time in enumerate(times):
                t = torch.full((len(samples),), time, dtype=torch.long, device='cuda')
                noisy = scheduler.add_noise(target, noise, t)
                alpha = scheduler.alphas_cumprod[time].to(device='cuda')
                for ci, cond in enumerate(tokens):
                    prediction = policy.model(noisy, t, cond=cond, training=False, gen_attn_map=False)[0]
                    clean = (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                    clipped = clean.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
                    for key, error in [('epsilon_mse', (prediction-noise).square()),
                                       ('clean_unclipped_mse', (clean-target).square()),
                                       ('clean_clipped_mse', (clipped-target).square())]:
                        metrics[key][si, ti, ci] = error.mean((1, 2)).cpu().numpy()
            print(json.dumps(dict(denoising_probe_seed_complete=seed, times=len(times), actual_cases=len(samples))), flush=True)
    np.savez_compressed(out / 'ALL_TIME_METRICS.npz', **metrics, times=np.asarray(times),
                        seeds=np.asarray(seeds), phases=np.repeat(phases, 2), branches=np.tile([0, 1], len(phases)))
    checks = dict(full_state_unchanged=all(torch.equal(v.cpu(), state[k]) for k,v in policy.state_dict().items()),
                  no_parameter_gradients=all(p.grad is None for p in policy.parameters()),
                  all_metrics_finite=all(np.isfinite(v).all() for v in metrics.values()),
                  only_geometry_swapped=all(torch.equal(normalized[k],wrong[k]) for k in normalized if k != CONTEXT_KEY))
    summary = {}
    for pi, phase in enumerate(phases):
        summary[str(phase)] = dict(split=endpoint['phases'][str(phase)]['evaluation_split'],
            token_swap_mse=token_delta[2*pi:2*pi+2],
            by_time={str(t): {key: metrics[key][:, ti, :, 2*pi:2*pi+2].mean(0).tolist() for key in metrics}
                     for ti,t in enumerate(times)})
    report = dict(execution_completed=True, checks=checks, checks_passed=all(checks.values()),
        full_parameter_count=sum(p.numel() for p in policy.parameters()),
        inference_timesteps=inference_times, alphas_cumprod=scheduler.alphas_cumprod.cpu().tolist(),
        conditions=['correct','paired_wrong'], metric_array_axes=['noise_seed','diffusion_time','condition','actual_case'],
        phases=summary, new_optimizer_updates=0, new_sample_draws=0, new_physics_steps=0,
        scope='Frozen full-model supervised denoising probe, not new generation or physical execution. Clean reconstruction at high noise amplifies epsilon error; all times and both branches reported without selection.')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plot_rows = (len(phases) + 2) // 3
    fig, axes = plt.subplots(plot_rows, 3, figsize=(16, 4 * plot_rows), squeeze=False)
    for ax in list(axes.flat)[len(phases):]:
        ax.set_visible(False)
    for pi, (phase, ax) in enumerate(zip(phases, axes.flat)):
        for branch, color in enumerate(('tab:blue', 'tab:orange')):
            for ci, style in enumerate(('-', '--')):
                values = metrics['clean_clipped_mse'][:, :, ci, 2*pi+branch].mean(0)
                ax.plot(times, values, color=color, linestyle=style,
                        label=f'branch{branch} ' + ('correct' if ci == 0 else 'wrong'))
        ax.axhline(bounds['phases'][str(phase)]['unavoidable_clipping_mse'], color='gray', linestyle=':', label='pair-mean clipping floor')
        ax.set_title(f'Phase{phase}: {summary[str(phase)]["split"]}')
        ax.set_yscale('log'); ax.set_xlabel('Diffusion training time'); ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Frozen full model: clipped clean reconstruction from target-corrupted inputs; all4 noise seeds; not generation')
    fig.tight_layout(); fig.savefig(out / 'DENOISING_ERRORS.png', dpi=140); plt.close(fig)
    report['denoising_plot_inspected'] = False
    write(out / 'RESULT.json', report)
    if not report['checks_passed']:
        raise RuntimeError('Frozen full denoising probe checks failed')


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--noise-coupling-readback', action='store_true')
    mode.add_argument('--prepare-independent-noise', action='store_true')
    mode.add_argument('--transfer-only', action='store_true')
    mode.add_argument('--prepare-phase-transfer', action='store_true')
    mode.add_argument('--phase-readback-only', action='store_true')
    mode.add_argument('--phase-input-probe', action='store_true')
    mode.add_argument('--prepare-replay', action='store_true')
    mode.add_argument('--preflight-replay', action='store_true')
    mode.add_argument('--replay-objective-probe', action='store_true')
    mode.add_argument('--replay-coverage-readback', action='store_true')
    mode.add_argument('--prepare-interpolation-replay', action='store_true')
    mode.add_argument('--prepare-interpolation-fit', action='store_true')
    mode.add_argument('--output-range-readback', action='store_true')
    mode.add_argument('--phase-denoising-probe', action='store_true')
    mode.add_argument('--phase-geometry-probe', action='store_true')
    mode.add_argument('--prepare-dense-fit', action='store_true')
    args = parser.parse_args()
    RUN = args.run.resolve()
    if RUN.parent != BASE:
        raise RuntimeError('Use the existing local experiment root')
    if args.prepare_dense_fit: prepare_dense_fit()
    elif args.phase_geometry_probe: probe_phase_inputs(RUN, geometry_only=True)
    elif args.phase_denoising_probe: probe_phase_denoising()
    elif args.output_range_readback: audit_output_range()
    elif args.prepare_interpolation_fit: prepare_interpolation_fit()
    elif args.prepare_interpolation_replay: prepare_interpolation_replay()
    elif args.replay_coverage_readback: readback_replay_coverage(RUN)
    elif args.replay_objective_probe: probe_replay_objective(RUN)
    elif args.prepare_replay: prepare_replay()
    elif args.preflight_replay: preflight_replay()
    elif args.phase_input_probe: probe_phase_inputs(RUN)
    elif args.phase_readback_only: readback_phase_samples(RUN)
    elif args.prepare_phase_transfer: prepare_phase_transfer()
    elif args.transfer_only: evaluate_branch(RUN, transfer=True)
    elif args.prepare_independent_noise: prepare_independent_noise()
    elif args.prepare: prepare()
    elif args.preflight: preflight()
    else: audit_noise_coupling()


if __name__ == '__main__': main()
