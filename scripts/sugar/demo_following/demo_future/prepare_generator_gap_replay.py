"""Prepare the declared actual two-gap coverage experiment; no model updates."""
import json

import numpy as np

from scripts.sugar.demo_following.demo_future import collect_generator_branch_coverage as collection_adapter

BASE = collection_adapter.BASE
CORPUS = BASE / 'generator_actual_branch_coverage_gaps2'
PREVIOUS = BASE / 'matched_generator_branch_latent_replay01512'
REPLAY = BASE / 'generator_train_diffusion_replay8_gaps2'
RUN = BASE / 'matched_generator_branch_latent_replay01_gaps512'


def main():
    physical = json.loads((CORPUS / 'RESULT.json').read_text())
    collection = json.loads((CORPUS / 'PROTOCOL.json').read_text())
    comparison = json.loads((PREVIOUS / 'frozen_evaluation/LATENT_REPLAY_COMPARISON.json').read_text())
    if not (physical['all_phase_data_passed'] and physical['physical_plots_inspected']
            and physical['all_new_xyz_plots_inspected'] and physical['actual_new_control_steps'] == 2400
            and physical['admitted_phase_groups'] == 11 and physical['cumulative_distinct_attempted_phase_groups'] == 13
            and comparison['checks_passed'] and comparison['decision']['all_seven_train_pass']
            and not comparison['decision']['both_reused_checks_pass']):
        raise RuntimeError('Require complete actual gap data and the full preceding fit/transfer evidence')
    if (CORPUS / 'NEXT_MATCHED_PROTOCOL.json').exists():
        raise RuntimeError('Never overwrite the declared next experiment')
    plan = json.loads((PREVIOUS / 'PROTOCOL.json').read_text())
    old_data = plan['data']
    collection_adapter.RUN = CORPUS
    data, disjoint = collection_adapter.prepare_phase_arrays(collection)
    assert data['train']['phases'] == [158,178,197,221,245,261,277,298,318]
    assert data['train']['real_examples'] == 18 and data['heldout_phase']['phases'] == [218,258]
    checks = dict(all_actual_visuals=physical['physical_plots_inspected'], source_clocks_disjoint=all(disjoint.values()))
    for split in ('train', 'heldout_phase'):
        with np.load(old_data[split]['arrays']) as old, np.load(data[split]['arrays']) as new:
            checks[f'{split}_schema_exact'] = set(old.files) == set(new.files)
            for i, row in enumerate(old_data[split]['samples']):
                j = next(j for j, candidate in enumerate(data[split]['samples'])
                         if (candidate['phase'], candidate['arm']) == (row['phase'], row['arm']))
                checks[f'{split}_{row["phase"]}_{row["arm"]}_old_arrays_exact'] = all(np.array_equal(old[k][i], new[k][j]) for k in old.files)
                checks[f'{split}_{row["phase"]}_{row["arm"]}_old_provenance_exact'] = row == data[split]['samples'][j]
    assert all(checks.values())
    description = ('Separate full official Generator coverage experiment. Preserve preceding full release initialization, '
        'old frozen normalization, full8319216parameter encoder/denoiser, IID q noise, normalized goal9 zero, '
        'intact causal history,0.25scaledsoftplus rank and0.1latent epsilon loss in both arms. Add only four '
        'real TRAIN cases at221/261, keeping old14cases and four reused218/258check arrays exact. '
        'Same512updates per arm, seed272084, AdamW1e-4 and cosine warmup16,8repetitions per case. '
        'Balanced batch grows112to144; total q and auxiliary rows per arm grow57344to73728, '
        'each realcase still4096exposures. Different batch changes random-number consumption and stochastic '
        'updates: no identical q trajectory, oldzeroendpoint identity, equalFLOPs or independent-motion claim. '
        'Same frozenoldfull025teacher, old14latent arrays reused exactly; only new4cases receive fresh '
        'official16step trajectories with the same8seeds. Teacher/replay compute remains additional.')
    plan.update(run_name=RUN.name, data=data, real_branch_examples=18, batch_size=144,
        repeated_examples_per_update=144, repetitions_per_real_training_example=8,
        branch_dataset=dict(corpus=str(CORPUS),normalizer_state=plan['normalizer_state'],repetitions=8,split='train',paired_supervision=True),
        original_source_future_clocks_disjoint=disjoint,
        phase_corpora={str(p):str(CORPUS/f'switch_{p}/branch_samples') for p in collection['switch_frames']},
        data_coverage_predecessor=str(PREVIOUS),diagnostic_predecessor=str(PREVIOUS),
        generator_training_started=False,implementation_status='requires_new4case_replay_and_complete144row_CPU_BF16_preflights',
        training=description,optimization=description,comparison_scope=description,
        actual_collection_coverage=dict(cumulative_admitted_phases=11,cumulative_attempted_phases=13,retained_failed_phases=[237,358]),
        objective_decision_scope='Reuse the already executed fixed0.25rank+0.1latent objective; no new coefficient choice. The old gradient audit used14cases, not the enlarged18case batch. Full new batch preflight and matched experiment remain required.',
        automatic_next_action='Complete frozenoldfull025 replay for only new221/261cases and exactold14latent reuse, fullstate/scheduler/originalsample/saved checks and all9TRAIN replay curves. Generalize only dataset/factory/preflight/comparison glue for18cases; no model modules. Complete actual144row CPU and H200BF16 preflights, then separate both512arms with full model/Adam/batch/exposure records and all11phase5condition32draws, savedreadbacks and88horizon panels. Require all original per-phase criteria; checks218/258 remain reused. If both pass, next resolve broader native and original goal/position prerequisites before generated physics and independent officialSMP comparison. If negative, diagnose full endpoint evidence without update513 or coefficient sweep.')
    replay = dict(plan['generated_state_objective'])
    replay.update(replay_source=str(REPLAY),input_arrays=str(REPLAY/'TRAIN_GENERATED_STATE_INPUTS.npz'),
        seed_assignment='Original dataset index//18 assigns eight repeated rows per actual case to the same eight frozen replay seeds, no indexing RNG.',
        time_assignment='TRAIN forward counter modulo16, common across all144rows.8seeds x16times x18cases covered32times over512updates.',
        expected_total_auxiliary_rows=73728,expected_exposures_per_real_case=4096,
        teacher_run=str(BASE/'matched_generator_branch_paired_rank025512'),
        reused_replay_source=str(BASE/'generator_train_diffusion_replay8'),reused_actual_cases=14,new_actual_cases=4)
    plan['generated_state_objective'] = replay
    plan['frozen_evaluation'] = dict(plan['frozen_evaluation'],evaluation_phases=collection['switch_frames'],
        scope='Nine TRAIN phases/18realcases and reused218/258fourcheckcases, all original criteria unchanged. Same96/90motions and old normalization, only raw-source future label clocks disjoint. Not untouched test, independent motions, generated physics or SMP benefit. Historical237/358failures retained in11/13actual coverage.')
    collection_adapter.write(CORPUS/'NEXT_MATCHED_PROTOCOL.json',plan)
    collection_adapter.write(CORPUS/'MODEL_PREPARATION_READBACK.json',dict(checks=checks,checks_passed=all(checks.values()),real_train_cases=18,batch_size=144,updates_per_arm=512,total_rows_per_arm=73728,old_rows_per_arm=57344,optimizer_updates_executed=0,new_generator_started=False,next_run=str(RUN),replay_destination=str(REPLAY)))
    print(json.dumps(dict(checks_passed=all(checks.values()),checks=len(checks),real_train_cases=18,batch_size=144,next_run=str(RUN))),flush=True)


if __name__ == '__main__':
    main()
