"""Prepare one bounded teacher-refresh experiment after the full own-path audit."""
import json
import copy
import numpy as np

from scripts.sugar.demo_following.demo_future.audit_generator_own_replay import BASE, RUN as OLD, OWN, LEGACY, OUT, write

RUN=BASE/'matched_generator_branch_self_replay01_gaps512'


def main():
    audit=json.loads((OUT/'RESULT.json').read_text())
    direction=json.loads((OUT/'DIRECTION_DECISION.json').read_text())
    readback=json.loads((OUT/'SAVED_MEAN_READBACK.json').read_text())
    checks=dict(full128_gradient_audit=audit['checks_passed'] and audit['plot_inspected'] and len(audit['rows'])==128,
        all18_predeclared_direction_checks=direction['raw_gradient_support'] and len(direction['criteria'])==18 and all(direction['criteria'].values()),
        independent_saved_gradient_readback=readback['checks_passed'])
    with np.load(LEGACY/'TRAIN_GENERATED_STATE_INPUTS.npz') as a,np.load(OWN/'TRAIN_GENERATED_STATE_INPUTS.npz') as b:
        for key in a.files:
            checks['replay_'+key]=not np.array_equal(a[key],b[key]) if key=='xt' else np.array_equal(a[key],b[key])
    assert all(checks.values())
    decision=OUT/'DECISION.json';assert not decision.exists() and not RUN.exists()
    write(decision,dict(checks=checks,checks_passed=True,permits_preparing_matched_objective_experiment=True,
        weight=.1,teacher_refresh=True,next_run=str(RUN),direction_criteria=direction['criteria'],
        scope='Only one separate teacher-refresh experiment supported by fixed raw-gradient checks; actual full CPU/BF16 preflights and unchanged full prediction criteria required. Local gradient compatibility is not generation benefit.'))
    plan=copy.deepcopy(json.loads((OLD/'PROTOCOL.json').read_text()))
    scope=('Same complete8319216 official model/release initialization/old frozen normalizer/18realTRAIN cases/144rows/seed272084/AdamW/schedule/512updates perarm. '
        'Only auxiliary diffusion teacher changes from oldfull025 to the frozen ranked18case endpoint. All144saved xt paths refreshed; actual targets, initialGaussians,8seeds,16times,.25rank+.1aux and both-arm auxiliary clocks unchanged. '
        'Zero-context auxiliary changes too, so zero endpoint/model/Adam/samples are not expected to equal predecessor. Original q/rank loss, full gradients, forwards and RNG must remain exact with auxiliary disabled. '
        'Teacher training/trajectory sampling and audits are extra compute, not equal totalFLOPs or independent-motion evidence. Original endpoints are immutable; both new arms start from the full release initialization.')
    plan.update(run_name=RUN.name,data_coverage_predecessor=str(OLD),diagnostic_predecessor=str(OLD),
        objective_decision=str(decision),objective_audit=str(OUT),teacher_refresh=True,
        generator_training_started=False,implementation_status='requires_full144row_CPU_and_BF16_preflights',
        training=scope,optimization=scope,comparison_scope=scope,objective_decision_scope=scope,
        automatic_next_action='Complete actual full144row initial+learned CPU and H200BF16 preflights, then both512arms from release, fullmodel+Adam/all11phase5condition32draws/all88panels/loss-exposure readback and strict predecessor comparison. Preserve all9TRAIN and2reused-check criteria plus previous passing TRAIN nonregression. Negative results trigger bounded diagnostics, never endpoint extension or a teacher/weight sweep. No generated physics or SMP benefit claim.')
    replay=plan['generated_state_objective']
    replay.update(replay_source=str(OWN),input_arrays=str(OWN/'TRAIN_GENERATED_STATE_INPUTS.npz'),teacher_run=str(OLD),
        reused_replay_source=str(LEGACY),reused_actual_cases=18,new_actual_cases=0)
    RUN.mkdir();write(RUN/'PROTOCOL.json',plan)
    write(RUN/'PREPARATION.json',dict(checks=checks,checks_passed=True,status='protocol_prepared_preflights_pending',
        source_run=str(OLD),replay_source=str(OWN),new_optimizer_updates=0,new_physics_steps=0))
    print(json.dumps(dict(checks=len(checks),next_run=str(RUN))),flush=True)


if __name__=='__main__':main()
