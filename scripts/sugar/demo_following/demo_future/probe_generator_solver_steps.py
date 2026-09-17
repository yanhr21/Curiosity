"""Zero-update full official DDPM16-versus50 diagnostic on retained endpoints."""
import argparse
import copy
import json

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import (
    BASE, ARMS, evaluate_phases, write,
)

SOURCE = BASE / 'matched_generator_branch_paired_rank025512'
REJECTED = BASE / 'matched_generator_branch_encoder_rank025512'
RUN = BASE / 'matched_generator_branch_rank025_solver50'


def prepare():
    comparison=json.loads((REJECTED/'frozen_evaluation/PAIRED_COMPARISON.json').read_text())
    component=json.loads((REJECTED/'frozen_evaluation/rank_component_audit/RESULT.json').read_text())
    assert comparison['checks_passed'] and component['checks_passed'] and component['plot_inspected']
    train=[v for v in comparison['phases'].values() if v['split']=='train']
    assert all(v['rank_mse']>v['base_mse'] for v in train)
    plan=copy.deepcopy(json.loads((SOURCE/'PROTOCOL.json').read_text()))
    plan.update(run_name=RUN.name,solver_comparison=dict(source_run=str(SOURCE),
        rejected_routing_run=str(REJECTED),original_steps=16,new_steps=50,
        intervention='Only official policy.num_inference_steps changes16to50. The released scheduler supplies all50 reverse steps. No custom solver, new model, optimizer, normalization, clipping, data or loss change.',
        random_control='Same declared32seeds and per-branch reset. Initial Gaussian noise follows the unchanged official call. Random step innovations are not time-matched across different step counts. First two original16step draws per phase/condition replay exactly before50step sampling.',
        selection='Retain full0.25endpoint because all seven TRAIN generation errors regressed under encoder-only routing. No reused-check metric selects model or step count;50is the full declared training-time grid, not a searched optimum.'),
        generator_training_started=False,new_optimizer_updates=0,new_physics_steps=0,
        checkpoint='Read-only references to already completed source arms; no new training or checkpoint selection.',
        automatic_next_action='Complete all9phases/fiveconditions/32draws with full official50stepDDPM, exact original16step two-draw replay, frozen-state/input checks, all72horizon panels and saved metric readback. Compare all seven TRAIN and reused checks with unchanged criteria and no old-pass regression. If still negative, inspect saved reverse-process trajectories before another training budget; do not claim physical execution/SMP benefit or automatically sweep solver steps.')
    plan['frozen_evaluation']['inference_steps']=50
    plan['frozen_evaluation']['scope']+=' Zero-update official16-to50step solver diagnostic; same initial seed, different reverse stochastic paths and compute. Source training updates are historical, not repeated.'
    RUN.mkdir(exist_ok=False)
    write(RUN/'PROTOCOL.json',plan)
    for arm in ARMS:
        (RUN/arm).symlink_to(SOURCE/arm,target_is_directory=True)
    print(str(RUN),flush=True)


def evaluate():
    assert json.loads((RUN/'PROTOCOL.json').read_text())['solver_comparison']['new_steps']==50
    result=evaluate_phases(RUN)
    result['new_optimizer_updates']=0
    write(RUN/'RESULT.json',result)


def compare():
    new=json.loads((RUN/'RESULT.json').read_text())
    old=json.loads((SOURCE/'RESULT.json').read_text())
    assert new['execution_completed'] and new['horizon_plots_inspected']
    assert json.loads((RUN/'frozen_evaluation/SAVED_PHASE_READBACK.json').read_text())['saved_metrics_reproduced']
    checks={};rows={}
    for phase,v in new['phases'].items():
        checks[phase+'_all_integrity']=all(v['matched_and_intervention_checks'].values())
        a=old['phases'][phase];x=a['variants']['trained_demo_geometry_correct'];y=v['variants']['trained_demo_geometry_correct']
        rows[phase]=dict(split=v['evaluation_split'],original16_mse=x['normalized_mse'],full50_mse=y['normalized_mse'],
            full50_over_original16=y['normalized_mse']/x['normalized_mse'],
            original16_correct_draws=x['per_branch_correct_preference_draws'],full50_correct_draws=y['per_branch_correct_preference_draws'],
            original16_pass=a['branch_identifiability_passed'],full50_pass=v['branch_identifiability_passed'])
    decision=dict(all_train_pass=all(new['training_phase_checks'].values()),both_reused_checks_pass=all(new['heldout_phase_checks'].values()),
        old_passing_mse_not_worse=all(v['full50_mse']<=v['original16_mse'] for v in rows.values() if v['original16_pass']))
    decision['all_prediction_requirements_passed']=all(checks.values()) and all(decision.values())
    result=dict(checks=checks,checks_passed=all(checks.values()),phases=rows,decision=decision,
        new_optimizer_updates=0,new_physics_steps=0,
        scope='Frozen full official solver comparison.50steps changes reverse-time grid, random innovation path and inference compute. Two16step replay seeds validate unchanged loader per phase/condition, not extra training replicates. Reused phase checks remain non-independent.')
    out=RUN/'frozen_evaluation/SOLVER_COMPARISON.json';assert not out.exists();write(out,result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','evaluate','compare'],required=True)
    args=parser.parse_args()
    {'prepare':prepare,'evaluate':evaluate,'compare':compare}[args.mode]()


if __name__=='__main__':main()
