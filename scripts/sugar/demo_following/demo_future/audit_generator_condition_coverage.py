"""Frozen full-Generator condition coverage; neighbors are descriptive only.

Compare the exact eight fitted cases, preceding native replay, and the complete
available native corpus separately. Never fit a retrieval predictor or use a
target to select an input neighbor. Full learned encoders are loaded unchanged.
"""
import argparse
import json
from pathlib import Path

import dill
import numpy as np
import torch
from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
from scripts.sugar.demo_following.demo_future.generator_dataset import ActualDemoGeometryDataset, ActualBranchGeometryDataset
from scripts.sugar.demo_following.demo_future.generator_demo_geometry import CONTEXT_KEY, restore_geometry_state
from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import write

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'matched_generator_branch_interpolation_fit512'
PARENT = ROOT / 'SUGAR/demo_ckpts/CarryBox/generator.ckpt'


def main():
    global RUN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    RUN=parser.parse_args().run.resolve()
    current_replay=RUN==BASE/'matched_generator_branch_latent_replay01512'
    if not current_replay and RUN!=BASE/'matched_generator_branch_interpolation_fit512':
        raise RuntimeError('Use only the declared completed condition-coverage diagnostic')
    torch.set_num_threads(1)
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    endpoint = json.loads((RUN / 'RESULT.json').read_text())
    probe_path='frozen_evaluation/rank_component_audit/RESULT.json' if current_replay else 'frozen_evaluation/geometry_phase_probe/RESULT.json'
    probe = json.loads((RUN / probe_path).read_text())
    if not endpoint['horizon_plots_inspected'] or not probe['checks_passed']:
        raise RuntimeError('Complete preceding frozen endpoint and geometry probe first')
    if current_replay and not probe['plot_inspected']:raise RuntimeError('Inspect complete denoising curves first')
    out = RUN / ('frozen_evaluation/joint_condition_coverage' if current_replay else 'frozen_evaluation/joint_condition_coverage_r1'); out.mkdir(exist_ok=False)
    model_sources=[('full025',BASE/'matched_generator_branch_paired_rank025512'),('latent_replay01',RUN)] if current_replay else [('fit512',RUN),('replay656',BASE/'matched_generator_branch_interpolation_replay16')]
    fitted_name='current_fitted_fourteen' if current_replay else 'current_fitted_eight'
    native = ActualDemoGeometryDataset(BASE / 'tracker_train_generator_corpus76', PARENT,
        BASE / 'matched_reference_feedback96/reference_feedback/evaluation')
    fitted = ActualBranchGeometryDataset(**plan['branch_dataset'])
    phases = plan['frozen_evaluation']['evaluation_phases']
    candidates = [native[i] for i in range(len(native))] + [fitted[i] for i in range(fitted.real_case_count)]
    metadata = []
    for episode, start, context in native.indices:
        row = native.timeline_metadata[episode]
        metadata.append(dict(source=row['source'], role=row['role'], official_row=start,
            context_row=context, fitted_in_current_run=False,
            included_in_preceding_native_replay=row['source'] not in (96, 90),
            label_path=row['label_path'], context_path=row['context_path']))
    metadata.extend(dict(row, role='actual_fitted_branch', fitted_in_current_run=True,
                         included_in_preceding_native_replay=False) for row in fitted.timeline_metadata)
    queries = []
    for phase in phases:
        dataset = ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)], plan['normalizer_state'])
        queries.extend([dataset[i] for i in (0, 1)])
    keys = list(native[0]['obs'])
    all_samples = candidates + queries
    obs = {k: torch.stack([s['obs'][k] for s in all_samples]) for k in keys}
    targets = torch.stack([s['action'] for s in all_samples])
    n = len(candidates)
    subsets = dict(current_fitted=np.arange(len(native), n),
        preceding_native_replay=np.asarray([i for i,m in enumerate(metadata) if m['included_in_preceding_native_replay']]),
        all_available_native=np.arange(len(native)))
    subsets[fitted_name]=subsets.pop('current_fitted')
    write(out / 'PROTOCOL.json', dict(run=str(RUN), phases=phases, candidate_metadata=metadata,
        subset_counts={k:len(v) for k,v in subsets.items()},
        model_checkpoints=[str(source/'demo_geometry/checkpoints/endpoint.ckpt') for name,source in model_sources],
        current_fitted_cases=fitted.real_case_count,
        automatic_next_action='Inspect fullstates, every fitted self-neighbor and all four reused-check query neighborhoods across exact fitted/native subsets. Raw geometry, causal state/history and both full learned encoders are descriptive coverage evidence. Choose one bounded conditional-representation or actual-data hypothesis from this evidence, preserving sevenTRAIN fitting gains and both failed reused checks. No retrieval replacement, coefficient sweep, optimizer update or generated physics.',
        scope=f'CPU read-only exact available-data and full learned condition-token distances. Current model fitted only{fitted.real_case_count}actual cases. The all-available native subset includes96/90 labels and overlaps reused phase checks; it is available data, not current TRAIN or independent validation. Mean-square distances within fields/tokens, then equal group weights. All neighbors selected without labels; command distances afterward are descriptive. Label-aware oracle is not a deployed model or a model-class bound.',
        new_optimizer_updates=0, new_sample_draws=0, new_physics_steps=0))
    policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
    initial = torch.load(plan['normalizer_state'], map_location='cpu')['model']
    restore_geometry_state(policy, initial, 'demo_geometry', goal_condition_mode='zero_diagnostic')
    normalized = policy.normalizer.normalize(obs)
    normalized_targets = policy.normalizer['action'].normalize(targets).numpy().astype(np.float64)
    fields = dict(object=('obj_pos_b','obj_ori_b'), history=('last_action',), geometry=(CONTEXT_KEY,))
    features = {'raw': {name:torch.cat([normalized[k] for k in ks],-1).reshape(len(all_samples),-1).numpy().astype(np.float64)
                        for name,ks in fields.items()}}
    checks = dict(current_fitted_real_count=fitted.real_case_count == (14 if current_replay else 8),
        preceding_native_count=len(subsets['preceding_native_replay']) == 3897,
        all_available_native_count=len(native) == 4125,
        candidate_metadata_aligned=len(metadata) == n)
    for name, source in model_sources:
        with (source / 'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
            state = torch.load(f, pickle_module=dill, map_location='cpu')['state_dicts']['model']
        # The restoration adapter accepts a freshly loaded official9-D encoder;
        # never apply it twice to an already expanded177-D instance.
        policy = GeneratorWrapper.load(str(PARENT), device='cpu').policy
        restore_geometry_state(policy, state, 'demo_geometry', goal_condition_mode='zero_diagnostic')
        policy.eval().requires_grad_(False)
        checks[name+'_frozen_normalizer_exact'] = all(torch.equal(v,state[k]) for k,v in initial.items() if k.startswith('normalizer.'))
        with torch.inference_mode():
            tokens = torch.cat([policy.obs_encoder({k:v[i:i+128] for k,v in normalized.items()}, training=False)
                                for i in range(0,len(all_samples),128)]).numpy().astype(np.float64)
        checks[name+'_full_state_unchanged'] = all(torch.equal(v,state[k]) for k,v in policy.state_dict().items())
        checks[name+'_complete_condition_shape'] = tokens.shape == (len(all_samples),3,256)
        checks[name+'_finite_tokens'] = bool(np.isfinite(tokens).all())
        features[name] = {field:tokens[:,i] for i,field in enumerate(fields)}
    rows, arrays = {}, {}
    for qi in range(len(queries)):
        phase, branch = phases[qi//2], qi%2
        own = ((normalized_targets[:n]-normalized_targets[n+qi])**2).mean((1,2))
        other = ((normalized_targets[:n]-normalized_targets[n+(qi^1)])**2).mean((1,2))
        entry = dict(phase=phase, branch=branch, split=endpoint['phases'][str(phase)]['evaluation_split'], representations={})
        for representation, groups in features.items():
            distances = {field:((values[:n]-values[n+qi])**2).mean(1) for field,values in groups.items()}
            distances['state'] = (distances['object']+distances['history'])/2
            distances['joint'] = (distances['object']+distances['history']+distances['geometry'])/3
            selected = {}
            for subset, indices in subsets.items():
                selected[subset] = {}
                for field, distance in distances.items():
                    nearest = indices[np.argsort(distance[indices],kind='stable')[:5]]
                    selected[subset][field] = [dict(candidate_index=int(j), input_rms=float(distance[j]**.5),
                        target_mse=float(own[j]), other_target_mse=float(other[j]), own_branch_preferred=bool(own[j]<other[j])) for j in nearest]
                oracle = int(indices[np.argmin(own[indices])])
                selected[subset]['label_aware_oracle'] = dict(candidate_index=oracle, target_mse=float(own[oracle]))
            entry['representations'][representation] = selected
            arrays[f'{representation}_query{qi}_joint_distance'] = distances['joint']
            if phase in plan['data']['train']['phases']:
                checks[f'{representation}_{phase}_{branch}_fitted_self_neighbor'] = float(distances['joint'][subsets[fitted_name]].min()) < 1e-12
        rows[f'{phase}_{branch}'] = entry
    np.savez_compressed(out / 'JOINT_DISTANCES.npz', **arrays)
    report = dict(execution_completed=True, checks=checks, checks_passed=all(checks.values()), queries=rows,
        subset_counts={k:len(v) for k,v in subsets.items()}, new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,
        scope='Full learned encoder and raw-input neighborhood coverage only. Neither nearest-neighbor agreement nor a label-aware oracle proves learnability, generalization or physical success.')
    write(out / 'RESULT.json', report)
    if not report['checks_passed']:
        raise RuntimeError('Complete candidate or full learned-encoder checks failed')
    print(json.dumps(dict(checks=checks, subset_counts=report['subset_counts'],
        withheld={k:{rep:{subset:{field:rows[k]['representations'][rep][subset][field][0] for field in ['state','geometry','joint']}
        for subset in subsets} for rep in features} for k in rows if rows[k]['split']=='heldout_phase'})),flush=True)


if __name__ == '__main__':
    main()
