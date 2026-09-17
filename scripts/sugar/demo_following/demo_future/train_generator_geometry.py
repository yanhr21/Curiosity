"""Bounded matched full official Generator adaptation on actual TRAIN data.

The official training loop, full denoiser and AdamW remain unchanged. Default
experiments use the official loss; explicit noise-coupling variants are recorded.
This driver supplies explicit dataset/configuration and endpoint bookkeeping.
Generated closed-loop routing and SMP benefit are separate scientific stages.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import dill
import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.sugar.demo_following.demo_future.generator_dataset import ActualDemoGeometryDataset, FrozenGeometryEvaluationDataset, ActualBranchGeometryDataset, ActualBranchReplayDataset
from scripts.sugar.demo_following.demo_future.generator_demo_geometry import CONTEXT_KEY, restore_geometry_state
from scripts.sugar.demo_following.demo_future.generator_workspace import GeometryGeneratorWorkspace, warm_start_geometry_policy
from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import write, passed
from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'matched_generator_demo_geometry16'
CORPUS = BASE / 'tracker_train_generator_corpus76'
PAIRED = BASE / 'matched_reference_feedback96/reference_feedback/evaluation'
PARENT = ROOT / 'SUGAR/demo_ckpts/CarryBox/generator.ckpt'
ARMS = ('zero_context', 'demo_geometry')


def position_floor_protocol():
    return dict(sources=[8, 28, 38, 58, 68, 78, 88, 98],
                arms=['measured_reference', 'original_numeric_demo'], optimizer_updates=0,
                checkpoint=str(BASE / 'matched_train_coverage128/broad_train/training/model_607.pt'),
                rollouts=16, total_requested_control_steps=5964,
                seed_and_world='Copy each existing native validation protocol unchanged; both arms use its same seed, source bank, initial world, horizon and physics on one retained GPU. Compare actual startup and first state exactly.',
                intervention='Only48 reference-position entries change. Control reads measured frozen-Refiner reference positions; treatment reads the original numeric demo at identical native offsets0..7 with official current-frame transforms. Full846-D Tracker, all510 old observations and288 known command entries remain unchanged.',
                physical_criterion='Same complete predeclared horizon, no reset, finite data, frozen full weights and ten consecutive lifted frames. Preserve source58/88 short budgets326/256 and failed teachers18/48 in denominator10. Pass this prerequisite only if both arms pass8/8.',
                scope='Frozen original-position compatibility test after the Generator prediction endpoint. Known reference commands remain inputs; this is not generated closed-loop execution, learned positions, an independent training replicate or SMP benefit.',
                automatic_next_action='Read all physical outcomes and trajectories. If original positions preserve8/8 execution and Generator prediction checks pass, separately evaluate replacing the Generator9-D measured-Refiner endpoint goal with the original-demo goal at the same declared source clock before original-demo-only generated-command handoff. This position-only comparison contains no Generator and cannot test its goal input. Otherwise diagnose the failed component using the retained complete models; no silent position substitution or placeholder world model.')


def prepare():
    RUN.mkdir(exist_ok=False)
    protocol = dict(
        arms=list(ARMS), parent_checkpoint=str(PARENT), native_train_corpus=str(CORPUS),
        additional_paired_train=str(PAIRED), frozen_validation=str(BASE / 'generator_native_validation8'),
        seed=272051, epochs=16, batch_size=128,
        actual_optimizer_updates='16 * ceil(actual admitted TRAIN chunks /128), fixed before either arm starts',
        optimizer=dict(lr=5e-5, weight_decay=1e-4, betas=[.95, .999]),
        scheduler=dict(name='cosine', warmup_steps=32),
        optimization='Unmodified official TrainGeneratorWorkspace.run, Generator.compute_loss, fused AdamW groups, bf16 Accelerator, gradient clip0.5, one update per batch. Fresh AdamW: release has no optimizer. LR/batch/warmup/16-epoch budget are explicitly local adaptation settings.',
        model='Full released12x256/8-head8x36 Generator, full16-step DDPM. Only existing target encoder input gains168 zero columns; every original model and normalization state preserved. Both arms same full initial state.',
        data='Complete all76 source attempts and actual dataset readback first. Only full physical passes; retain failures in76/80 denominators. Include both old TRAIN timelines as82 extra chunks, not new independent sources. Uniform shuffle of actual complete chunks, no padded or reference-switch-crossing windows.',
        conditioning='TRAIN-only8x21 original numeric-demo box/limb geometry. Control zeros this field after normalization. No action targets, real future states, force/reward or sourceID enters condition. Retain original task target and causal state/last-command input in both arms.',
        retained_goal_provenance='Existing9-D Generator goal is the final object pose of the measured frozen-Refiner reference bank. All matched arms and frozen context interventions retain it identically. This experiment tests added geometry conditional on that refined goal; it does not prove independence from Refiner reference data.',
        normalization='All released statistics frozen; only new geometry field fit to the same admitted TRAIN chunks in both arms. Validation never fits statistics.',
        checkpoint='Keep complete endpoint model and full AdamW, exact epoch/update clocks, all epoch checkpoints. No best-validation selection. Initial model captured before first update. Exact interrupted-loop scheduler/sampler resumption is not implemented.',
        frozen_evaluation=dict(
            validation_sources=[8, 28, 38, 58, 68, 78, 88, 98], original_validation_denominator=10,
            excluded_failed_teachers=[18, 48], samples_per_chunk=8,
            sample_seeds=list(range(272060, 272068)), inference_steps=16,
            variants=['released', 'trained_zero_context', 'trained_demo_geometry_correct',
                      'trained_demo_geometry_wrong', 'trained_demo_geometry_zero'],
            wrong_context='Cyclic next source at normalized raw phase; all other input fields and observed-future labels fixed. This is not a counterfactual physical target.',
            baselines=['Repeat last observed36-D command over all8 horizons'],
            metrics='Per-source and equal-source means of released-normalized command MSE; also separate joint29/linear3/angular3/contact1 raw and normalized errors at each horizon. Report all draws, no best-of8 selection.',
            preliminary_conditioning_pass='Equal-source correct-context normalized MSE <=0.95 times each matched zero-context, wrong-context and last-command baseline; correct beats each in at least5/8 sources. Numerical validity and matching/data contracts must also pass. This is only observed-future prediction evidence.',
            scope='Already-used motion-disjoint native validation, one adaptation seed. No untouched-test, generated closed-loop, human-video or SMP-benefit claim.'),
        automatic_next_action='Finish actual TRAIN corpus and direct readback, run both matched full-model arms serially, freeze complete endpoints and evaluate all predeclared variants. If prediction checks fail, inspect full horizon/channel/context evidence and diagnose before another bounded experiment. If they pass, resolve complete generated command routing and real PhysX endpoints, then independently compare frozen SMP. Camera evidence remains outstanding.',
        generator_training_started=False, post_prediction_position_floor=position_floor_protocol())
    write(RUN / 'PROTOCOL.json', protocol)
    print(json.dumps(protocol), flush=True)


def training_config(plan, arm):
    with PARENT.open('rb') as stream:
        released = torch.load(stream, pickle_module=dill, map_location='cpu')
    cfg = copy.deepcopy(released['cfg'])
    OmegaConf.resolve(cfg)
    cfg.policy = OmegaConf.create(dict(
        _target_='scripts.sugar.demo_following.demo_future.generator_workspace.warm_start_geometry_policy',
        parent_checkpoint=str(PARENT), condition_mode=arm, start_ckpt_path=None,
        history_dropout_probability=plan.get('history_condition_dropout', {}).get('probability', 0.),
        history_dropout_seed=plan.get('history_condition_dropout', {}).get('seed', 272077),
        goal_condition_mode=plan.get('goal_condition_mode', 'provided'),
        noise_coupling=plan.get('noise_coupling', 'official_assignment'),
        paired_objective=plan.get('paired_objective'),
        generated_state_objective=plan.get('generated_state_objective'),
        robot_state_conditioning=plan.get('robot_state_conditioning'),
        actual_state_supervision=plan.get('actual_state_supervision')))
    cfg.task.dataset = OmegaConf.create(dict(
        _target_='scripts.sugar.demo_following.demo_future.generator_dataset.ActualDemoGeometryDataset',
        corpus=str(CORPUS), parent_checkpoint=str(PARENT), paired_train=str(PAIRED)))
    if plan.get('branch_dataset'):
        cfg.task.dataset = OmegaConf.create(dict(
            _target_='scripts.sugar.demo_following.demo_future.generator_dataset.ActualBranchGeometryDataset',
            **plan['branch_dataset']))
    if plan.get('replay_dataset'):
        cfg.task.dataset = OmegaConf.create(dict(
            _target_='scripts.sugar.demo_following.demo_future.generator_dataset.ActualBranchReplayDataset',
            **plan['replay_dataset']))
    if plan.get('generated_state_objective'):
        cfg.task.dataset = OmegaConf.create(dict(
            _target_='scripts.sugar.demo_following.demo_future.generator_latent_replay.ActualBranchLatentReplayDataset',
            replay_source=plan['generated_state_objective']['replay_source'], **plan['branch_dataset']))
    # Upstream GeneratorRunner intentionally performs no physics. Real frozen
    # PhysX evaluation is external and may never be inferred from its empty log.
    cfg.task.env_runner = OmegaConf.create(dict(_target_='sugar_il.env_runner.generator_runner.GeneratorRunner'))
    cfg.optimizer = OmegaConf.create(plan['optimizer'])
    for key, value in dict(seed=plan['seed'], num_epochs=plan['epochs'], resume=False, debug=False,
                           gradient_accumulate_every=1, use_ema=False, lr_scheduler='cosine',
                           lr_warmup_steps=plan['scheduler']['warmup_steps'], checkpoint_every=1,
                           sample_every=plan['epochs'], val_every=plan['epochs'], rollout_every=plan['epochs'],
                           gen_attn_map=False, max_train_steps=None, max_val_steps=None).items():
        cfg.training[key] = value
    loader = dict(batch_size=plan['batch_size'], num_workers=0, shuffle=True,
                  pin_memory=True, persistent_workers=False, drop_last=False)
    cfg.dataloader = OmegaConf.create(loader)
    cfg.val_dataloader = OmegaConf.create(dict(loader, shuffle=False))
    cfg.checkpoint.topk.k = 0
    cfg.training.checkpoint_every = plan.get('checkpoint_every', 1)
    cfg.checkpoint.save_last_ckpt = True
    cfg.checkpoint.save_last_snapshot = False
    cfg.logging.project = 'matched_generator_demo_geometry16'
    return cfg


def train_arm(arm):
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Full Generator training requires the recorded retained compute step')
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    readback = json.loads((CORPUS / 'READBACK.json').read_text())
    if plan.get('generated_state_objective'):
        for name,device in (('GENERATED_CPU_PREFLIGHT.json','cpu'),('GENERATED_BF16_PREFLIGHT.json','cuda')):
            preflight_root=Path(plan['robot_preflight_directory']) if plan.get('robot_state_conditioning') else RUN
            check=json.loads((preflight_root/name).read_text())
            if not (check['passed'] and check['device']==device and (device=='cpu' or check['bf16_autocast'])):
                raise RuntimeError('Complete both actual full replay preflights before training')
            if plan.get('robot_state_conditioning') and not (check.get('robot_state_extension') and check['full_parameter_count']==8327408 and check['batch_rows']==144):
                raise RuntimeError('Require both full measured32 robot input preflights')
    if plan.get('paired_objective') and not plan.get('generated_state_objective') and not json.loads((RUN / 'PAIRED_PREFLIGHT.json').read_text())['passed']:
        raise RuntimeError('Complete full paired-objective stochastic matching checks first')
    if plan.get('actual_state_supervision'):
        for name in ('ACTUAL_STATE_CPU_PREFLIGHT.json','ACTUAL_STATE_BF16_PREFLIGHT.json'):
            report=json.loads((RUN/name).read_text())
            if not (report['checks_passed'] and report.get('loss_module_training_mode')):raise RuntimeError('Complete actual-state full-model objective preflight in actual training mode first')
    if plan.get('paired_objective', {}).get('gradient_scope') == 'target_encoder_only' or plan.get('paired_objective', {}).get('rank_function') == 'hinge':
        cpu = json.loads((RUN / 'PAIRED_CPU_PREFLIGHT.json').read_text())
        gpu = json.loads((RUN / 'PAIRED_PREFLIGHT.json').read_text())
        if not (cpu['passed'] and cpu['device'] == 'cpu' and gpu['passed'] and gpu['bf16_autocast']):
            raise RuntimeError('Complete both CPU and actual BF16 gradient-routing checks first')
    if plan.get('history_condition_dropout') and not json.loads((RUN / 'PREFLIGHT.json').read_text())['passed']:
        raise RuntimeError('Full history dropout preflight must pass before training')
    if plan.get('goal_condition_mode') == 'zero_diagnostic' and not plan.get('branch_dataset'):
        if not json.loads((RUN / 'PREFLIGHT.json').read_text())['passed']:
            raise RuntimeError('Complete full-model goal-removal preflight first')
    if plan.get('noise_coupling') == 'independent' and plan.get('method_adapter_preflight'):
        if not json.loads(Path(plan['method_adapter_preflight']).read_text())['passed']:
            raise RuntimeError('Complete full-model independent-noise adapter checks first')
    if not readback['data_integrity_passed']:
        raise RuntimeError('Complete actual TRAIN dataset checks first')
    if plan.get('generated_state_objective'):
        from scripts.sugar.demo_following.demo_future.generator_latent_replay import ActualBranchLatentReplayDataset
        dataset=ActualBranchLatentReplayDataset(replay_source=plan['generated_state_objective']['replay_source'], **plan['branch_dataset'])
    elif plan.get('replay_dataset'):
        if not json.loads((RUN / 'PREFLIGHT.json').read_text())['passed']:
            raise RuntimeError('Complete full-model replay preflight before training')
        dataset = ActualBranchReplayDataset(**plan['replay_dataset'])
    elif plan.get('branch_dataset'):
        if not json.loads((RUN / 'PREFLIGHT.json').read_text())['passed']:
            raise RuntimeError('Complete full actual-branch input preflight first')
        dataset = ActualBranchGeometryDataset(**plan['branch_dataset'])
    else:
        dataset = ActualDemoGeometryDataset(CORPUS, PARENT, PAIRED)
    budget = plan['epochs'] * math.ceil(len(dataset) / plan['batch_size'])
    if plan.get('replay_dataset') and (dataset.composition != plan['replay_composition'] or budget != plan['actual_optimizer_updates']):
        raise RuntimeError('Actual replay data or budget differs from the predeclared protocol')
    out = RUN / arm
    out.mkdir(exist_ok=False)
    write(out / 'DATA_AND_BUDGET.json', dict(actual_chunks=len(dataset), timeline_metadata=dataset.timeline_metadata,
                                            epochs=plan['epochs'], exact_optimizer_updates=budget,
                                            batch_size=plan['batch_size'], seed=plan['seed'],
                                            replay_composition=getattr(dataset, 'composition', None)))
    cfg = training_config(plan, arm)
    workspace = GeometryGeneratorWorkspace(cfg, output_dir=out)
    workspace.model.set_normalizer(dataset.get_normalizer())
    workspace.model.normalizer.requires_grad_(False)
    initial = {key: value.detach().cpu().clone() for key, value in workspace.model.state_dict().items()}
    predecessor = plan.get('comparison_source') if plan.get('robot_state_conditioning') else plan.get('preceding_experiment') or plan.get('data_coverage_predecessor')
    if predecessor:
        previous_initial = torch.load(Path(predecessor) / arm / 'INITIAL_MODEL.pt', map_location='cpu')['model']
        if plan.get('robot_state_conditioning'):
            from scripts.sugar.demo_following.demo_future.generator_measured_robot import extend_robot_state_dict,EXTRA_KEY
            if EXTRA_KEY not in previous_initial:previous_initial=extend_robot_state_dict(previous_initial)
        if initial.keys() != previous_initial.keys() or not all(torch.equal(value, previous_initial[key]) for key, value in initial.items()):
            raise RuntimeError('Preceding full model or TRAIN normalization differs before the first update')
    torch.save(dict(model=initial, arm=arm, optimizer_updates=0), out / 'INITIAL_MODEL.pt')
    batches, history_masks, paired_losses, binding = [], [], [], []

    def record_batch(module, args, kwargs):
        if kwargs.get('training', True):
            if plan.get('robot_state_conditioning') and not binding:
                # Official run() constructs and wraps Adam before this first forward.
                names={id(p):n for n,p in module.named_parameters() if p.requires_grad}
                state_groups=workspace.optimizer.state_dict()['param_groups']
                binding.extend(dict(name=names[id(p)],parameter_id=index,shape=list(p.shape))
                               for group,saved_group in zip(workspace.optimizer.param_groups,state_groups)
                               for p,index in zip(group['params'],saved_group['params']))
                if len(binding)!=len(names) or len({v['name'] for v in binding})!=len(names):
                    raise RuntimeError('Measured robot optimizer omitted or duplicated a full-model parameter')
                write(out/'MEASURED_ROBOT_ADAM_BINDING.json',dict(parameters=binding,full_parameter_count=sum(p.numel() for p in module.parameters()),recorded_before_first_forward=True))
            batches.append(args[0]['sample_index'].detach().cpu().tolist())

    hook = workspace.model.register_forward_pre_hook(record_batch, with_kwargs=True)
    def record_history_mask(module, args, kwargs, output):
        if kwargs.get('training', True) and plan.get('paired_objective'):
            row = dict(module.last_paired_loss) if module.last_paired_loss is not None else dict(base=float(output.detach()), rank=0., weight=plan['paired_objective']['weight'], rank_not_evaluated_zero_context=True)
            row['total'] = float(output.detach())
            paired_losses.append(row)
        if kwargs.get('training', True) and plan.get('history_condition_dropout'):
            mask = module.obs_encoder.last_history_dropout_mask
            if mask is None or len(mask) != len(args[0]['sample_index']):
                raise RuntimeError('Actual history dropout mask missing or misaligned')
            history_masks.append(mask.tolist())
    mask_hook = workspace.model.register_forward_hook(record_history_mask, with_kwargs=True)
    try:
        workspace.run()
    except Exception as error:
        failure_checkpoint = workspace.save_checkpoint(tag='execution_failure')
        write(out / 'BATCH_ORDER_AT_FAILURE.json', batches)
        if plan.get('paired_objective'): write(out / 'PAIRED_LOSSES_AT_FAILURE.json', paired_losses)
        write(out / 'EXECUTION_FAILURE.json', dict(error=repr(error), checkpoint=failure_checkpoint,
            observed_global_step=workspace.global_step, observed_epoch=workspace.epoch,
            scope='Complete current model and AdamW retained after execution failure. Not a valid endpoint or an exact scheduler/sampler resume checkpoint.'))
        raise
    finally:
        hook.remove()
        mask_hook.remove()
    write(out / 'BATCH_ORDER.json', batches)
    if plan.get('paired_objective'):
        if len(paired_losses) != budget or not all(math.isfinite(v) for row in paired_losses for k, v in row.items() if k != 'rank_not_evaluated_zero_context'):
            raise RuntimeError('Actual paired-loss records do not match the finite applied budget')
        write(out / 'PAIRED_LOSSES.json', paired_losses)
    if plan.get('generated_state_objective'):
        replay_checks=dict(full512_training_calls=len(paired_losses)==512,
            deterministic_times=all(row['generated_step']==i and row['generated_time_index']==i%16 and row['generated_time']==plan['generated_state_objective']['times'][i%16] for i,row in enumerate(paired_losses)),
            fixed_both_arm_auxiliary=all(row['generated_weight']==.1 and row['generated_rows']==plan['batch_size'] for row in paired_losses),
            exact_total_rows=sum(row['generated_rows'] for row in paired_losses)==plan['generated_state_objective']['expected_total_auxiliary_rows'])
        write(out/'GENERATED_STATE_EXPOSURES.json',dict(checks=replay_checks,passed=all(replay_checks.values()),rows=plan['generated_state_objective']['expected_total_auxiliary_rows'],per_real_case=plan['generated_state_objective']['expected_exposures_per_real_case']))
        if not all(replay_checks.values()):raise RuntimeError('Actual replay exposure differs from the fixed protocol')
    if plan.get('actual_state_supervision'):
        policy=workspace.model
        expected=torch.zeros_like(policy.expert_counts)
        for step in range(budget):
            indices=policy.expert_indices(step)
            expected.index_add_(0,indices,torch.ones_like(indices))
        expert_checks=dict(exact512_calls=policy.expert_step==budget==512,
            actual_exposure_counts_exact=torch.equal(policy.expert_counts,expected),
            all355_examples_seen=bool((expected>0).all()),
            all73728_extra_rows=int(expected.sum())==73728,
            loss_clocks_exact=all(v['expert_step']==i and v['expert_seed']==272500+i and v['expert_rows']==144 and v['expert_weight']==.1 for i,v in enumerate(paired_losses)))
        write(out/'EXPERT_STATE_EXPOSURES.json',dict(checks_passed=all(expert_checks.values()),checks=expert_checks,
            actual_counts=policy.expert_counts.tolist(),expected_counts=expected.tolist(),additional_expert_rows=73728,
            label_kind='desired known expert command plans on355actually visited states; no counterfactual physical futures'))
        if not all(expert_checks.values()):raise RuntimeError('Actual-state expert supervision clocks or exposures differ')
    if plan.get('history_condition_dropout'):
        if len(history_masks) != budget or sum(map(len, history_masks)) != len(dataset) * plan['epochs']:
            raise RuntimeError('History mask exposure does not match actual training examples')
        write(out / 'HISTORY_MASKS.json', history_masks)
        torch.save(workspace.model.obs_encoder.history_dropout_generator.get_state(), out / 'HISTORY_MASK_RNG.pt')
    if workspace.global_step != budget or workspace.epoch != plan['epochs']:
        raise RuntimeError('Official workspace did not complete its exact matched budget')
    if len(batches) != budget or sum(map(len, batches)) != len(dataset) * plan['epochs']:
        raise RuntimeError('Actual full-dataset batch exposure differs from the matched budget')
    endpoint = workspace.save_checkpoint(tag='endpoint')
    with open(endpoint, 'rb') as stream:
        saved = torch.load(stream, pickle_module=dill, map_location='cpu')
    model, optimizer = saved['state_dicts']['model'], saved['state_dicts']['optimizer']
    clocks = {int(value['step'].item()) for value in optimizer['state'].values()}
    original_norm = {k: v for k, v in initial.items() if k.startswith('normalizer.')}
    checks = dict(full_model_finite=all(torch.isfinite(value).all().item() for value in model.values()),
                  full_adapted_state_shape=model.keys() == initial.keys()
                  and all(value.shape == initial[key].shape for key, value in model.items()),
                  actual_weights_changed=any(not torch.equal(value, initial[key]) for key, value in model.items()),
                  normalization_frozen=all(torch.equal(value, model[key]) for key, value in original_norm.items()),
                  optimizer_present=bool(optimizer['state']), optimizer_clocks_exact=clocks == {budget},
                  optimizer_moments_finite=all(torch.isfinite(value[k]).all().item()
                                               for value in optimizer['state'].values() for k in ('exp_avg', 'exp_avg_sq')))
    if plan.get('robot_state_conditioning'):
        from scripts.sugar.demo_following.demo_future.generator_measured_robot import WEIGHT_KEY,EXTRA_KEY
        checks['full8327408_parameters']=sum(p.numel() for p in workspace.model.parameters())==8327408
        checks['new8192_input_coefficients_trained']=model[WEIGHT_KEY].shape==(256,36) and model[EXTRA_KEY].shape==(256,32) and bool(model[EXTRA_KEY].count_nonzero())
        checks['all_adam_moments_bound_to_full_parameters']=all(
            v['parameter_id'] in optimizer['state'] and all(tuple(optimizer['state'][v['parameter_id']][key].shape)==tuple(v['shape']) for key in ('exp_avg','exp_avg_sq')) for v in binding)
    if plan.get('rank_ablation_decision') and arm=='zero_context':
        from scripts.sugar.demo_following.demo_future.compare_generator_rank_ablation import equal
        source=Path(plan['data_coverage_predecessor'])/arm
        with (source/'checkpoints/endpoint.ckpt').open('rb') as stream:
            reference=torch.load(stream,pickle_module=dill,map_location='cpu')['state_dicts']
        previous_losses=json.loads((source/'PAIRED_LOSSES.json').read_text())
        zero_checks=dict(full_model_exact=equal(model,reference['model']),full_adam_exact=equal(optimizer,reference['optimizer']),
            all512_batches_exact=batches==json.loads((source/'BATCH_ORDER.json').read_text()),
            all512_actual_losses_exact=len(previous_losses)==len(paired_losses)==512 and all(all(a[k]==b[k] for k in ('base','rank','generated','total','generated_time','generated_step')) for a,b in zip(previous_losses,paired_losses)))
        write(out/'RANK_ABLATION_ZERO_CONTROL.json',dict(checks=zero_checks,checks_passed=all(zero_checks.values()),source=str(source)))
        if not all(zero_checks.values()):raise RuntimeError('Rank absence changed the supposedly identical full zero-context control')
    if plan.get('exact_training_replay_source'):
        from scripts.sugar.demo_following.demo_future.compare_generator_rank_ablation import equal
        source=Path(plan['exact_training_replay_source'])/arm
        with (source/'checkpoints/endpoint.ckpt').open('rb') as stream:
            reference=torch.load(stream,pickle_module=dill,map_location='cpu')['state_dicts']
        previous_losses=json.loads((source/'PAIRED_LOSSES.json').read_text())
        replay_checks=dict(full_model_exact=equal(model,reference['model']),full_adam_exact=equal(optimizer,reference['optimizer']),
            full_initial_state_exact=equal(initial,torch.load(source/'INITIAL_MODEL.pt',map_location='cpu')['model']),
            all512_batches_exact=batches==json.loads((source/'BATCH_ORDER.json').read_text()),
            all512_nonbase_loss_fields_exact=len(previous_losses)==len(paired_losses)==512 and all(a.keys()==b.keys() and all(a[k]==b[k] for k in a if k!='base') for a,b in zip(previous_losses,paired_losses)),
            all512_corrected_arithmetic=all(math.isclose(v['total'],v['base']+v['weight']*v['rank']+v['generated_weight']*v['generated'],rel_tol=2e-6,abs_tol=2e-6) for v in paired_losses))
        write(out/'EXACT_TRAINING_REPLAY_READBACK.json',dict(checks=replay_checks,checks_passed=all(replay_checks.values()),source=str(source),extra_executed_training_updates=512,new_unique_trajectory_updates=0,scope='Independent512step replay from identical original initialization; original endpoint never updated. Exact fullmodel+Adam,all512actual batches/totals/auxiliary/metadata except repaired base field.'))
        if not all(replay_checks.values()):raise RuntimeError('Full training logging repair failed exact trajectory reproduction')
    if plan.get('paired_objective') and not plan.get('generated_state_objective') and arm == 'zero_context':
        with (Path(plan['data_coverage_predecessor']) / arm / 'checkpoints/endpoint.ckpt').open('rb') as stream:
            previous = torch.load(stream, pickle_module=dill, map_location='cpu')['state_dicts']
        def exact_tree(a, b):
            if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
            if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(exact_tree(v, b[k]) for k, v in a.items())
            if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(exact_tree(v, w) for v, w in zip(a, b))
            return a == b
        checks['preceding_zero_control_full_model_exact'] = exact_tree(model, previous['model'])
        checks['preceding_zero_control_full_adam_exact'] = exact_tree(optimizer, previous['optimizer'])
    write(out / 'RESULT.json', dict(execution_completed=True, endpoint_checks_passed=all(checks.values()),
                                    checks=checks, actual_optimizer_updates=workspace.global_step,
                                    epochs=workspace.epoch, endpoint=endpoint,
                                    scope='Complete official Generator adaptation endpoint only; frozen sampling, prompt dependence and physical execution remain separate.'))
    if not all(checks.values()):
        raise RuntimeError('Inspect complete Generator endpoint failure')


def evaluate(output_name='frozen_evaluation'):
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Full frozen Generator sampling requires the retained compute step')
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    frozen = plan['frozen_evaluation']
    states = [torch.load(RUN / arm / 'INITIAL_MODEL.pt', map_location='cpu')['model'] for arm in ARMS]
    checks = dict(initial_full_states_exact=states[0].keys() == states[1].keys()
                  and all(torch.equal(value, states[1][key]) for key, value in states[0].items()),
                  actual_batch_order_exact=json.loads((RUN / ARMS[0] / 'BATCH_ORDER.json').read_text())
                  == json.loads((RUN / ARMS[1] / 'BATCH_ORDER.json').read_text()),
                  actual_data_and_budget_exact=json.loads((RUN / ARMS[0] / 'DATA_AND_BUDGET.json').read_text())
                  == json.loads((RUN / ARMS[1] / 'DATA_AND_BUDGET.json').read_text()),
                  both_complete_endpoints=all(json.loads((RUN / arm / 'RESULT.json').read_text())['endpoint_checks_passed'] for arm in ARMS))
    if plan.get('history_condition_dropout'):
        checks['actual_history_masks_exact'] = json.loads((RUN / ARMS[0] / 'HISTORY_MASKS.json').read_text()) == json.loads((RUN / ARMS[1] / 'HISTORY_MASKS.json').read_text())
    predecessor = plan.get('preceding_experiment') or plan.get('data_coverage_predecessor')
    if predecessor:
        previous = Path(predecessor)
        for arm, state in zip(ARMS, states):
            old = torch.load(previous / arm / 'INITIAL_MODEL.pt', map_location='cpu')['model']
            checks[arm + '_preceding_initial_exact'] = old.keys() == state.keys() and all(torch.equal(value, old[key]) for key, value in state.items())
            checks[arm + '_preceding_batch_order_exact'] = json.loads((previous / arm / 'BATCH_ORDER.json').read_text()) == json.loads((RUN / arm / 'BATCH_ORDER.json').read_text())
    if not all(checks.values()):
        write(RUN / 'MATCHED_CHECK_FAILURE.json', checks)
        raise RuntimeError('Full matched initialization, actual data order or endpoints differ')
    out = RUN / output_name; out.mkdir(exist_ok=False)
    dataset = FrozenGeometryEvaluationDataset(plan['frozen_validation'])
    groups = {'joint29': slice(0, 29), 'linear3': slice(29, 32), 'angular3': slice(32, 35), 'contact1': slice(35, 36)}
    source_indices = {}
    for index, (episode, _, _) in enumerate(dataset.indices):
        source = dataset.timeline_metadata[episode]['source']
        source_indices.setdefault(source, []).append(index)
    results = {}
    device = torch.device('cuda:0')
    for variant in frozen['variants']:
        policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
        if variant != 'released':
            arm = 'zero_context' if variant == 'trained_zero_context' else 'demo_geometry'
            with (RUN / arm / 'checkpoints/endpoint.ckpt').open('rb') as stream:
                payload = torch.load(stream, pickle_module=dill, map_location='cpu')
            mode = 'zero_context' if variant in ('trained_zero_context', 'trained_demo_geometry_zero') else 'demo_geometry'
            restore_geometry_state(policy, payload['state_dicts']['model'], mode,
                                   goal_condition_mode=plan.get('goal_condition_mode', 'provided'))
        elif plan.get('goal_condition_mode') == 'zero_diagnostic':
            # The released baseline receives the same removed goal. Its added
            # geometry columns are exactly zero in the saved common initial state.
            restore_geometry_state(policy, states[0], 'zero_context', goal_condition_mode='zero_diagnostic')
        policy.to(device).eval().requires_grad_(False)
        variant_out = out / variant; variant_out.mkdir()
        per_source = {}
        for source, indices in source_indices.items():
            samples = [dataset.with_wrong_context(i) if variant == 'trained_demo_geometry_wrong' else dataset[i] for i in indices]
            obs = {key: torch.stack([sample['obs'][key] for sample in samples]).to(device)
                   for key in samples[0]['obs'] if not (variant == 'released' and key == CONTEXT_KEY
                       and plan.get('goal_condition_mode', 'provided') == 'provided')}
            target = torch.stack([sample['action'] for sample in samples]).to(device)
            draws = []
            with torch.inference_mode():
                for seed in frozen['sample_seeds']:
                    torch.manual_seed(seed)
                    generated = policy.predict_action(obs)
                    if generated.shape != target.shape or not torch.isfinite(generated).all():
                        raise RuntimeError('Full frozen sampled command shape or numeric failure')
                    draws.append(generated.detach().clone())
                prediction = torch.stack(draws)
                raw_error = (prediction - target[None]).square()
                normalized_error = (policy.normalizer['action'].normalize(prediction)
                                    - policy.normalizer['action'].normalize(target)[None]).square()
                last = obs['last_action'].expand(-1, 8, -1)
                last_error = (policy.normalizer['action'].normalize(last)
                              - policy.normalizer['action'].normalize(target)).square()
            per_source[str(source)] = dict(
                chunks=len(indices), normalized_mse=float(normalized_error.mean()),
                per_draw_normalized_mse=normalized_error.mean((1, 2, 3)).cpu().tolist(),
                last_command_normalized_mse=float(last_error.mean()),
                last_command_normalized_mse_by_group_horizon={name: last_error[..., section].mean((0, 2)).cpu().tolist() for name, section in groups.items()},
                raw_mse_by_group_horizon={name: raw_error[..., section].mean((0, 1, 3)).cpu().tolist() for name, section in groups.items()},
                normalized_mse_by_group_horizon={name: normalized_error[..., section].mean((0, 1, 3)).cpu().tolist() for name, section in groups.items()})
            np.savez_compressed(variant_out / f'source_{source:03d}.npz', predictions=prediction.cpu().numpy(),
                                observed_future_target=target.cpu().numpy(), previous_command=last.cpu().numpy(),
                                dataset_indices=np.asarray(indices), sample_seeds=np.asarray(frozen['sample_seeds']))
            print(json.dumps(dict(variant=variant, source=source, normalized_mse=per_source[str(source)]['normalized_mse'])), flush=True)
        results[variant] = dict(per_source=per_source,
                                equal_source_normalized_mse=float(np.mean([row['normalized_mse'] for row in per_source.values()])))
        write(out / 'PARTIAL_RESULT.json', results)
        del policy
        torch.cuda.empty_cache()
    correct = results['trained_demo_geometry_correct']
    comparisons = {}
    for name in ('trained_zero_context', 'trained_demo_geometry_wrong', 'last_command'):
        baseline = {source: row['last_command_normalized_mse'] for source, row in correct['per_source'].items()} if name == 'last_command' else {
            source: row['normalized_mse'] for source, row in results[name]['per_source'].items()}
        baseline_mean = float(np.mean(list(baseline.values())))
        ratio = correct['equal_source_normalized_mse'] / baseline_mean if baseline_mean > 0 else None
        wins = sum(row['normalized_mse'] < baseline[source] for source, row in correct['per_source'].items())
        comparisons[name] = dict(equal_source_mse_ratio=ratio, source_wins=wins,
                                 passed=ratio is not None and ratio <= .95 and wins >= 5)
    if plan.get('preceding_experiment'):
        previous = json.loads((Path(plan['preceding_experiment']) / 'RESULT.json').read_text())['variants']['trained_demo_geometry_correct']
        ratio = correct['equal_source_normalized_mse'] / previous['equal_source_normalized_mse']
        wins = sum(row['normalized_mse'] < previous['per_source'][source]['normalized_mse'] for source, row in correct['per_source'].items())
        comparisons['preceding_correct_context'] = dict(equal_source_mse_ratio=ratio, source_wins=wins, passed=ratio <= .95 and wins >= 5)
    final = dict(execution_completed=True, matched_checks=checks, variants=results, comparisons=comparisons,
                 preliminary_conditioning_passed=all(row['passed'] for row in comparisons.values()),
                 actual_validation_sources=len(source_indices), original_validation_denominator=10,
                 sampling_draws_per_chunk=len(frozen['sample_seeds']), new_physics_rollouts=0,
                 scope=frozen['scope'], retained_goal_provenance=plan['retained_goal_provenance'],
                 history_condition_dropout=plan.get('history_condition_dropout'),
                 noise_coupling=plan.get('noise_coupling', 'official_assignment'),
                 goal_condition_mode=plan.get('goal_condition_mode', 'provided'),
                 next_action=plan['automatic_next_action'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    horizon = np.arange(8) * .1
    labels = {'released': 'Released', 'trained_zero_context': 'Trained zero context',
              'trained_demo_geometry_correct': 'Correct demo', 'trained_demo_geometry_wrong': 'Wrong demo',
              'trained_demo_geometry_zero': 'Geometry model with zero context'}
    if plan.get('goal_condition_mode') == 'zero_diagnostic':
        labels['released'] = 'Released weights, goal removed'
    for ax, group in zip(axes.flat, groups):
        for variant, data in results.items():
            values = np.mean([row['normalized_mse_by_group_horizon'][group] for row in data['per_source'].values()], axis=0)
            ax.plot(horizon, values, label=labels[variant], marker='.', linewidth=1.4)
        previous = np.mean([row['last_command_normalized_mse_by_group_horizon'][group]
                            for row in correct['per_source'].values()], axis=0)
        ax.plot(horizon, previous, label='Repeat previous command', color='black', linestyle='--')
        ax.set_title(group); ax.set_xlabel('Future command horizon (s)')
        ax.set_ylabel('Released-normalized MSE'); ax.grid(alpha=.2)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Frozen observed-future prediction: equal-source mean, all8 draws; no new physics')
    fig.tight_layout(rect=(0, .09, 1, .96)); fig.savefig(out / 'HORIZON_ERRORS.png', dpi=150); plt.close(fig)
    final['horizon_error_plot'] = str(out / 'HORIZON_ERRORS.png')
    final['horizon_plot_inspected'] = False
    write(out / 'RESULT.json', final)
    return final


def diagnose_saved_conditioning(output_name='frozen_evaluation'):
    """Decompose the already saved full draws without training or resampling."""
    out = RUN / output_name
    result = json.loads((out / 'RESULT.json').read_text())
    if not result['execution_completed'] or not all(result['matched_checks'].values()):
        raise RuntimeError('Complete the matched frozen endpoint before interpreting saved draws')
    models = {}
    for arm in ARMS:
        with (RUN / arm / 'checkpoints/endpoint.ckpt').open('rb') as stream:
            models[arm] = torch.load(stream, pickle_module=dill, map_location='cpu')['state_dicts']['model']
    scale = models['demo_geometry']['normalizer.params_dict.action.scale'].numpy()
    if not torch.equal(models['demo_geometry']['normalizer.params_dict.action.scale'],
                       models['zero_context']['normalizer.params_dict.action.scale']):
        raise RuntimeError('Action normalization differs across matched models')
    decomposition, intervention = {}, {}
    sources = list(result['variants']['trained_demo_geometry_correct']['per_source'])
    for variant in result['variants']:
        rows = {}
        for source in sources:
            with np.load(out / variant / f'source_{int(source):03d}.npz') as arrays:
                pred = arrays['predictions'].astype(np.float64) * scale
                target = arrays['observed_future_target'].astype(np.float64) * scale
                mean = pred.mean(0)
                total = float(np.mean((pred - target[None]) ** 2))
                bias = float(np.mean((mean - target) ** 2))
                variance = float(np.mean((pred - mean[None]) ** 2))
                if not np.isclose(total, bias + variance, rtol=1e-10, atol=1e-12):
                    raise RuntimeError('Saved-draw squared-error decomposition differs')
                if not np.isclose(total, result['variants'][variant]['per_source'][source]['normalized_mse'], rtol=1e-5):
                    raise RuntimeError('Saved full predictions do not reproduce the reported error')
                rows[source] = dict(total_mse=total, eight_draw_mean_mse=bias, draw_variance=variance)
        decomposition[variant] = dict(per_source=rows,
            equal_source_mean={key: float(np.mean([r[key] for r in rows.values()])) for key in next(iter(rows.values()))})
    for source in sources:
        path = f'source_{int(source):03d}.npz'
        with np.load(out / 'trained_demo_geometry_correct' / path) as correct:
            rows = {}
            for variant in ('trained_demo_geometry_wrong', 'trained_demo_geometry_zero'):
                with np.load(out / variant / path) as changed:
                    if not all(np.array_equal(correct[key], changed[key]) for key in
                               ('observed_future_target', 'previous_command', 'dataset_indices', 'sample_seeds')):
                        raise RuntimeError('Context intervention changed target, history, indices or sampling seeds')
                    delta = (correct['predictions'].astype(np.float64) - changed['predictions'].astype(np.float64)) * scale
                    mse = float(np.mean(delta ** 2))
                    rows[variant] = dict(paired_prediction_change_mse=mse,
                        ratio_to_correct_draw_variance=mse / decomposition['trained_demo_geometry_correct']['per_source'][source]['draw_variance'])
            intervention[source] = rows
    weights = {}
    for arm, model in models.items():
        value = model['obs_encoder.target_state_net.0.weight']
        weights[arm] = dict(original9_column_norm=float(value[:, :9].norm()),
                            new168_column_norm=float(value[:, 9:].norm()),
                            new168_nonzero_entries=int(value[:, 9:].count_nonzero()))
    report = dict(execution_completed=True, optimizer_updates=0, new_sample_draws=0, new_physics_steps=0,
                  saved_prediction_metrics_reproduced=True, decomposition=decomposition,
                  same_noise_context_intervention=intervention, adapter_weights=weights,
                  scope='Post-endpoint diagnostic of all saved draws and full model weights. The mean of8 draws is a decomposition term, not a deployed controller or replacement primary metric. This cannot reverse the failed predeclared conditioning result.')
    write(out / 'CONDITIONING_READBACK.json', report)
    print(json.dumps(dict(adapter_weights=weights,
        decomposition={k: v['equal_source_mean'] for k, v in decomposition.items()})), flush=True)
    return report


def pipeline():
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    if (RUN / 'CHILDREN.json').exists():
        raise RuntimeError('Inspect existing exact Generator children before changing execution')
    records = []
    bootstrap = ROOT / 'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
    for arm in ARMS:
        command = [sys.executable, str(bootstrap),
                   'scripts.sugar.demo_following.demo_future.train_generator_geometry', '--run', str(RUN), '--arm', arm]
        with (RUN / f'{arm}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(arm=arm, pid=child.pid, pgid=os.getpgid(child.pid), command=command)
            records.append(row); write(RUN / 'CHILDREN.json', records)
            row['returncode'] = child.wait(); write(RUN / 'CHILDREN.json', records)
        if row['returncode']:
            raise RuntimeError('Inspect exact full Generator child failure: ' + arm)
    if json.loads((RUN / 'PROTOCOL.json').read_text()).get('replay_dataset'):
        native = evaluate('native_evaluation')
        diagnose_saved_conditioning('native_evaluation')
        from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import evaluate_phases
        phases = evaluate_phases(RUN)
        result = dict(execution_completed=True, native_evaluation=native, phase_evaluation=phases,
                      native_conditioning_passed=native['preliminary_conditioning_passed'],
                      heldout_phase_transfer_passed=phases['heldout_phase_transfer_passed'],
                      combined_conditioning_passed=native['preliminary_conditioning_passed'] and phases['heldout_phase_transfer_passed'],
                      new_physics_steps=0, scope='Matched full-model native replay and declared actual branch supervision. Known-source withheld phases and existing motion-disjoint native checks remain separate. No generated physics or SMP benefit.')
    elif json.loads((RUN / 'PROTOCOL.json').read_text()).get('phase_evaluation'):
        from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import evaluate_phases
        result = evaluate_phases(RUN)
    elif json.loads((RUN / 'PROTOCOL.json').read_text()).get('branch_dataset'):
        from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import evaluate_branch
        result = evaluate_branch(RUN)
    else:
        result = evaluate()
    write(RUN / 'RESULT.json', result)
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    if not plan.get('branch_dataset') and (plan.get('history_condition_dropout') or plan.get('noise_coupling') == 'independent'):
        diagnose_saved_conditioning()
    if json.loads((RUN / 'PROTOCOL.json').read_text()).get('post_prediction_position_floor'):
        original_position_floor()


def prepare_history_dropout():
    previous = BASE / 'matched_generator_demo_geometry16'
    if RUN == previous:
        raise RuntimeError('Use a distinct run directory and preserve the original matched experiment')
    result = json.loads((previous / 'RESULT.json').read_text())
    probe = json.loads((previous / 'frozen_evaluation/input_branch_probe/RESULT.json').read_text())
    if not result['execution_completed'] or result['preliminary_conditioning_passed'] or not probe['model_state_exact']:
        raise RuntimeError('Require the completed failed conditioning endpoint and frozen branch diagnosis')
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    plan['preceding_experiment'] = str(previous)
    plan['history_condition_dropout'] = dict(probability=.5, seed=272077,
        scope='Local training-only condition dropout in the existing full official encoder. Mask the whole normalized last-command token to zero for independently sampled50% of examples. All actual validation inputs remain intact. Independent CPU RNG preserves original noise/shuffle streams; save every actual mask and compare arms.')
    plan['automatic_next_action'] = 'Complete both full-model matched arms and unchanged intact-history frozen evaluation. Require the original correct-versus-zero/wrong/last-command criteria and report the preceding experiment. If still negative, inspect TRAIN information and branch evidence before another experiment; do not claim selected-demo following or open generated physics. If positive, resolve the independent position-source and original-goal compatibility prerequisites before generated handoff.'
    plan['post_prediction_position_floor'] = None
    plan['additional_improvement_criterion'] = 'In addition to every original intact-history conditioning criterion, correct-context MSE must be <=0.95 times the preceding unmasked correct-context endpoint, with wins on at least5/8 sources. Merely weakening the matched zero-context arm is insufficient.'
    plan['generator_training_started'] = False
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), history_condition_dropout=plan['history_condition_dropout'], epochs=plan['epochs'])), flush=True)


def prepare_independent_noise():
    previous = BASE / 'matched_generator_demo_geometry16'
    branch = BASE / 'matched_generator_actual_branch_iid256'
    if RUN == previous:
        raise RuntimeError('Preserve the original matched experiment')
    evidence = json.loads((branch / 'RESULT.json').read_text())
    coverage = json.loads((previous / 'frozen_evaluation/CONDITION_COVERAGE.json').read_text())
    if not evidence['branch_identifiability_passed'] or not coverage['execution_completed']:
        raise RuntimeError('Require complete branch identifiability and actual TRAIN coverage evidence')
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    plan['preceding_experiment'] = str(previous)
    plan['noise_coupling'] = 'independent'
    plan['method_adapter_preflight'] = str(branch / 'PREFLIGHT.json')
    plan['goal_condition_mode'] = 'provided'
    plan['optimization'] = 'Same complete official model/workspace/AdamW, data, normalization, seed and528update budget as the original broader matched experiment. Local method variant omits only target-dependent cross-example noise_assignment; keep independent Gaussian noise, original timestep stream/scheduler and epsilon MSE. Original goal and intact history remain supplied; no history dropout or branch-only goal removal.'
    plan['additional_improvement_criterion'] = 'In addition to every original correct-versus-zero/wrong/last-command validation criterion, correct-context MSE must be <=0.95 times the preceding original-assignment correct-context endpoint, with wins on at least5/8 sources. Exact prior full initialization and actual sample order required.'
    plan['data_scope'] = 'Reuse the exact original4125 TRAIN chunks, including82 existing paired chunks; do not append the two branch-diagnostic examples or alter phases. Two-branch success does not establish generalization; this is the next existing fixed-validation comparison.'
    plan['post_prediction_position_floor'] = None
    plan['automatic_next_action'] = 'Complete both matched full-model arms and intact original fixed-validation evaluation, then inspect all draws/horizons and preceding-method comparison. If positive, separately resolve original-goal and position-source compatibility before generated physics. If negative, use actual branch/data and learning evidence to select the next bounded experiment. Do not treat two TRAIN branch fits or loss changes as generalization or SMP benefit.'
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), noise_coupling='independent', epochs=16, updates_per_arm=528, goal='provided')), flush=True)


def prepare_goal_removal():
    previous = BASE / 'matched_generator_demo_geometry_iid16'
    result = json.loads((previous / 'RESULT.json').read_text())
    if not result['execution_completed'] or result['preliminary_conditioning_passed'] or not result['horizon_plot_inspected']:
        raise RuntimeError('Require the completed and inspected broader IID negative endpoint')
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    plan['preceding_experiment'] = str(previous)
    plan['goal_condition_mode'] = 'zero_diagnostic'
    plan['optimization'] = 'Full official model/workspace and local IID-noise variant, same4125actual TRAIN chunks, complete initial states, batch order, seed and528updates per arm as preceding IID experiment. Only remove the normalized9-D goal in both TRAIN and evaluation, preserving all history and object inputs. This is an explicit goal-dependency diagnostic, not unchanged official conditioning.'
    plan['conditioning'] = 'Original-demo8x21 geometry versus normalized-zero geometry. Both arms remove the entire normalized9-D measured-Refiner goal at the encoder; no actual future, executed actions or sourceID added. Full causal history/object fields remain intact.'
    plan['retained_goal_provenance'] = 'The archived dataset still preserves the original measured-Refiner endpoint goal, but all9normalized goal entries are zeroed before every target-encoder call, during training and all frozen variants including released weights. This diagnostic removes use of that goal; it does not solve the separate Tracker48-D position dependency.'
    plan['additional_improvement_criterion'] = 'Keep all original conditional prediction criteria and require correct-context MSE <=0.95 times the preceding provided-goal IID correct endpoint, with at least5/8source wins. Report condition benefit and prior absolute-performance comparison separately; weakening controls alone is not a deployment improvement.'
    plan['automatic_next_action'] = 'Complete the exact paired budget, frozen five-condition eight-draw evaluation and saved-sample readback. Inspect all horizon groups. If conditional separation remains absent, do not increase this budget or scan goal/dropout strengths: next assess actual shared-world branching coverage using full official components. If it passes, resolve the separate original48-D position execution failure before generated physics. No SMP or generalization claim from TRAIN diagnostics.'
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(dict(run=str(RUN), goal_condition_mode='zero_diagnostic', updates_per_arm=528)), flush=True)


def preflight_goal_removal():
    """Full actual-data check of goal removal and restored inference behavior."""
    torch.set_num_threads(1)
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    if plan['goal_condition_mode'] != 'zero_diagnostic':
        raise RuntimeError('This preflight is only for the declared goal-removal diagnostic')
    dataset = ActualDemoGeometryDataset(CORPUS, PARENT, PAIRED)
    indices = np.linspace(0, len(dataset) - 1, 32).round().astype(int).tolist()
    samples = [dataset[i] for i in indices]
    obs = {k: torch.stack([s['obs'][k] for s in samples]) for k in samples[0]['obs']}
    checks = {}
    for arm in ARMS:
        policy = warm_start_geometry_policy(PARENT, arm, goal_condition_mode='zero_diagnostic', noise_coupling='independent')
        policy.set_normalizer(dataset.get_normalizer())
        policy.requires_grad_(False).eval()
        state = torch.load(Path(plan['preceding_experiment']) / arm / 'INITIAL_MODEL.pt', map_location='cpu')['model']
        checks[arm + '_complete_initial_state_exact'] = state.keys() == policy.state_dict().keys() and all(torch.equal(v, state[k]) for k, v in policy.state_dict().items())
        captured = {}
        handles = []
        for name in ('target', 'robot', 'obj'):
            def capture(module, args, name=name):
                captured[name] = args[0].detach().clone()
            handles.append(getattr(policy.obs_encoder, name + '_state_net').register_forward_pre_hook(capture))
        normalized = policy.normalizer.normalize(obs)
        for training in (True, False):
            policy.train(training)
            policy.obs_encoder(normalized, training=training)
            prefix = arm + ('_train_' if training else '_eval_')
            checks[prefix + 'goal_exact_zero'] = not bool(captured['target'][..., :9].count_nonzero())
            expected = normalized[CONTEXT_KEY] if arm == 'demo_geometry' else torch.zeros_like(normalized[CONTEXT_KEY])
            checks[prefix + 'geometry_exact'] = torch.equal(captured['target'][..., 9:].reshape_as(expected), expected)
            checks[prefix + 'history_intact'] = torch.equal(captured['robot'].reshape_as(normalized['last_action']), normalized['last_action'])
            obj = torch.cat([normalized['obj_pos_b'], normalized['obj_ori_b']], dim=-1)
            checks[prefix + 'object_intact'] = torch.equal(captured['obj'].reshape_as(obj), obj)
        for handle in handles:
            handle.remove()
        policy.eval()
        small = {k: v[:2].clone() for k, v in obs.items()}
        swapped = {k: v.clone() for k, v in small.items()}
        for key in ('target_obj_pos_b', 'target_obj_ori_b'):
            swapped[key] = obs[key][-2:].clone()
        checks[arm + '_goal_intervention_nontrivial'] = any(not torch.equal(small[k], swapped[k]) for k in ('target_obj_pos_b', 'target_obj_ori_b'))
        restored = GeneratorWrapper.load(str(PARENT), device='cpu').policy
        restore_geometry_state(restored, state, arm, goal_condition_mode='zero_diagnostic')
        restored.requires_grad_(False).eval()
        with torch.inference_mode():
            torch.manual_seed(272140)
            prediction = policy.predict_action(small)
            torch.manual_seed(272140)
            changed = policy.predict_action(swapped)
            torch.manual_seed(272140)
            roundtrip = restored.predict_action(small)
        checks[arm + '_all16step_samples_goal_invariant'] = torch.equal(prediction, changed)
        checks[arm + '_full_inference_restore_exact'] = torch.equal(prediction, roundtrip)
        checks[arm + '_all_weights_unchanged'] = all(torch.equal(v, state[k]) for k, v in policy.state_dict().items())
    report = dict(execution_completed=True, passed=all(checks.values()), checks=checks,
                  sample_indices=indices, actual_train_chunks=len(dataset), optimizer_updates=0, physics_steps=0,
                  full_parameter_count=sum(p.numel() for p in policy.parameters()),
                  scope='Full official-model input and complete sampling check on actual TRAIN observations. No optimization, validation fitting, substitute model or physical result.')
    write(RUN / 'PREFLIGHT.json', report)
    if not report['passed']:
        raise RuntimeError('Full goal-removal preflight failed')
    print(json.dumps(report), flush=True)


def preflight_history_dropout():
    """Check the full input adapter and sampling before its bounded training."""
    torch.set_num_threads(1)
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    dropout = plan['history_condition_dropout']
    dataset = ActualDemoGeometryDataset(CORPUS, PARENT, PAIRED)
    indices = np.linspace(0, len(dataset) - 1, 32).round().astype(int).tolist()
    samples = [dataset[index] for index in indices]
    obs = {key: torch.stack([sample['obs'][key] for sample in samples]) for key in samples[0]['obs']}
    masks, features, rng_after, checks = {}, {}, {}, {}
    for arm in ARMS:
        policy = warm_start_geometry_policy(PARENT, arm,
            history_dropout_probability=dropout['probability'], history_dropout_seed=dropout['seed'])
        policy.set_normalizer(dataset.get_normalizer())
        policy.normalizer.requires_grad_(False)
        policy.requires_grad_(False)
        original = torch.load(Path(plan['preceding_experiment']) / arm / 'INITIAL_MODEL.pt', map_location='cpu')['model']
        checks[arm + '_original_full_state_exact'] = original.keys() == policy.state_dict().keys() and all(torch.equal(value, original[key]) for key, value in policy.state_dict().items())
        captured = []
        handle = policy.obs_encoder.robot_state_net.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
        normalized = policy.normalizer.normalize(obs)
        policy.train()
        torch.manual_seed(272078)
        features[arm] = policy.obs_encoder(normalized, training=True).detach()
        rng_after[arm] = torch.get_rng_state().clone()
        masks[arm] = policy.obs_encoder.last_history_dropout_mask.clone()
        actual = captured[-1].reshape_as(normalized['last_action'])
        checks[arm + '_masked_history_exact_zero'] = not bool(actual[masks[arm]].count_nonzero())
        checks[arm + '_unmasked_history_exact'] = torch.equal(actual[~masks[arm]], normalized['last_action'][~masks[arm]])
        checks[arm + '_both_mask_outcomes_present'] = bool(masks[arm].any() and (~masks[arm]).any())
        policy.obs_encoder.history_dropout_probability = 0.
        torch.manual_seed(272078)
        policy.obs_encoder(normalized, training=True)
        checks[arm + '_original_global_rng_stream_preserved'] = torch.equal(rng_after[arm], torch.get_rng_state())
        policy.obs_encoder.history_dropout_probability = dropout['probability']
        rng_mask = policy.obs_encoder.history_dropout_generator.get_state().clone()
        policy.eval()
        policy.obs_encoder(normalized, training=False)
        checks[arm + '_evaluation_history_intact'] = torch.equal(captured[-1].reshape_as(normalized['last_action']), normalized['last_action'])
        checks[arm + '_evaluation_does_not_draw_masks'] = torch.equal(rng_mask, policy.obs_encoder.history_dropout_generator.get_state()) and policy.obs_encoder.last_history_dropout_mask is None
        handle.remove()
        small = {key: value[:2] for key, value in obs.items()}
        torch.manual_seed(272079)
        with torch.inference_mode():
            prediction = policy.predict_action(small)
            policy.obs_encoder.history_dropout_probability = 0.
            torch.manual_seed(272079)
            control = policy.predict_action(small)
        checks[arm + '_full_eval_sampling_unchanged'] = torch.equal(prediction, control)
    checks['both_actual_masks_exact'] = torch.equal(masks[ARMS[0]], masks[ARMS[1]])
    checks['initial_train_features_exact'] = torch.equal(features[ARMS[0]], features[ARMS[1]])
    checks['both_global_rng_states_exact'] = torch.equal(rng_after[ARMS[0]], rng_after[ARMS[1]])
    report = dict(execution_completed=True, passed=all(checks.values()), checks=checks,
        sample_indices=indices, actual_mask=masks[ARMS[0]].tolist(),
        optimizer_updates=0, physics_steps=0, full_parameter_count=sum(p.numel() for p in policy.parameters()),
        scope='Full official-model CPU adapter/sampling check. TRAIN-only history mask, intact evaluation, identical initial states and masks. Not a learned-conditioning or physical result.')
    write(RUN / 'PREFLIGHT.json', report)
    if not report['passed']:
        raise RuntimeError('Full history-condition dropout preflight failed')
    print(json.dumps(report), flush=True)


def original_position_floor():
    """Matched frozen physics check of the remaining position-reference source."""
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Frozen position-source physics requires the retained compute step')
    if not json.loads((RUN / 'RESULT.json').read_text())['execution_completed']:
        raise RuntimeError('Complete the preceding matched Generator prediction stage first')
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())['post_prediction_position_floor']
    out = BASE / 'original_position_feedback_floor'
    out.mkdir(exist_ok=False)
    write(out / 'PROTOCOL.json', plan)
    bootstrap = ROOT / 'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
    checkpoint = BASE / 'matched_train_coverage128/broad_train/training/model_607.pt'
    asset = BASE / 'scene_runtime/converted_g1/g1_29dof_rev_1_0_with_rubber_hand.usd'
    records, results = [], {}
    for source in plan['sources']:
        previous = BASE / f'matched_reference_feedback96/heldout_native_validation/source_{source:03d}/evaluation'
        case = out / f'source_{source:03d}'; case.mkdir()
        case_plan = json.loads((previous / 'PROTOCOL.json').read_text())
        if case_plan['reference_alignment'] != 'none':
            raise RuntimeError('Position source comparison requires the unchanged native clock')
        write(case / 'PROTOCOL.json', case_plan)
        (case / 'motions').symlink_to(previous / 'motions', target_is_directory=True)
        endpoints = {}
        for arm in plan['arms']:
            target = case / arm
            command = [sys.executable, str(bootstrap),
                       'scripts.sugar.demo_following.demo_future.collect_refiner_pair',
                       '--headless', '--controller', 'tracker', '--tracker-checkpoint', str(checkpoint),
                       '--robot-usd', str(asset), '--run', str(case), '--arm', 'original', '--output', str(target)]
            if arm == 'original_numeric_demo':
                command.append('--original-position-feedback')
            with (case / f'{arm}.log').open('x') as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                row = dict(source=source, arm=arm, pid=proc.pid, pgid=os.getpgid(proc.pid), command=command,
                           slurm_job_id=os.environ['SLURM_JOB_ID'], slurm_step_id=os.environ['SLURM_STEP_ID'], host=socket.gethostname())
                records.append(row); write(out / 'CHILDREN.json', records)
                row['returncode'] = proc.wait(); write(out / 'CHILDREN.json', records)
            if row['returncode'] or not (target / 'RESULT.json').exists():
                raise RuntimeError('Inspect exact frozen position-source runtime failure')
            endpoints[arm] = json.loads((target / 'RESULT.json').read_text())
            expected_source = 'original_numeric_demo' if arm == 'original_numeric_demo' else 'unchanged_measured_reference'
            if json.loads((target / 'PROTOCOL.json').read_text())['position_feedback_source'] != expected_source:
                raise RuntimeError('Actual position input source differs from the declared arm')
        left, right = [case / arm for arm in plan['arms']]
        with np.load(left / 'STARTUP.npz') as a, np.load(right / 'STARTUP.npz') as b:
            startup = a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
        fields = ('robot_body_state_before_w', 'robot_root_state_before_w', 'object_state_before_w',
                  'joint_pos_before', 'joint_vel_before', 'teacher_observation')
        with np.load(left / 'TRACE.npz') as a, np.load(right / 'TRACE.npz') as b:
            initial = {k: np.array_equal(a[k][0], b[k][0]) for k in fields}
        if not startup or not all(initial.values()):
            raise RuntimeError('Matched position-source startup or actual initial world differs')
        results[str(source)] = dict(endpoints=endpoints, startup_exact=startup, initial_world_exact=initial,
                                    physical_passed={arm: passed(endpoint) for arm, endpoint in endpoints.items()})
        write(out / 'PARTIAL_RESULT.json', results)
        print(json.dumps(dict(position_floor_source=source, physical_passed=results[str(source)]['physical_passed'])), flush=True)
    counts = {arm: sum(row['physical_passed'][arm] for row in results.values()) for arm in plan['arms']}
    result = dict(execution_completed=True, optimizer_updates=0, supported_source_denominator=8,
                  original_validation_denominator=10, failed_teacher_sources=[18, 48], per_source=results,
                  physical_pass_counts=counts, original_position_floor_passed=all(count == 8 for count in counts.values()),
                  scope=plan['scope'], next_action=plan['automatic_next_action'])
    write(out / 'RESULT.json', result)
    return result


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--arm', choices=ARMS)
    mode.add_argument('--evaluate-only', action='store_true')
    mode.add_argument('--pipeline', action='store_true')
    mode.add_argument('--original-position-floor', action='store_true')
    mode.add_argument('--saved-conditioning-readback', action='store_true')
    mode.add_argument('--prepare-history-dropout', action='store_true')
    mode.add_argument('--preflight-history-dropout', action='store_true')
    mode.add_argument('--prepare-independent-noise', action='store_true')
    mode.add_argument('--prepare-goal-removal', action='store_true')
    mode.add_argument('--preflight-goal-removal', action='store_true')
    args = parser.parse_args()
    RUN = args.run.resolve()
    if RUN.parent != BASE:
        raise RuntimeError('Keep Generator experiments in the existing local experiment root')
    if args.prepare_history_dropout:
        prepare_history_dropout()
    elif args.prepare_independent_noise:
        prepare_independent_noise()
    elif args.prepare_goal_removal:
        prepare_goal_removal()
    elif args.preflight_goal_removal:
        preflight_goal_removal()
    elif args.preflight_history_dropout:
        preflight_history_dropout()
    elif args.prepare_only:
        prepare()
    elif args.pipeline:
        pipeline()
    elif args.evaluate_only:
        evaluate()
    elif args.original_position_floor:
        original_position_floor()
    elif args.saved_conditioning_readback:
        diagnose_saved_conditioning()
    else:
        train_arm(args.arm)


if __name__ == '__main__':
    main()
