"""Bounded actual shared-world branch collection with the frozen full Tracker.

Uses existing official PhysX collection, independent physical/reference readback
and official Generator formatting. Creates no model and performs no optimization.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import write

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'generator_actual_branch_coverage4'
PARENT = BASE / 'matched_train_coverage128/broad_train'
SOURCE = PARENT / 'evaluation'
CHECKPOINT = PARENT / 'training/model_607.pt'
USD = BASE / 'scene_runtime/converted_g1/g1_29dof_rev_1_0_with_rubber_hand.usd'
BOOT = ROOT / 'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
ARMS = ('original', 'repeat', 'alternate')


def prepare():
    evidence = BASE / 'matched_generator_goal_removed_iid16/frozen_branch_transfer/RESULT.json'
    result = json.loads(evidence.read_text())
    if not result['execution_completed'] or result['branch_identifiability_passed'] or not result['horizon_plot_inspected']:
        raise RuntimeError('Complete and inspect the failed broader-model actual branch transfer first')
    if not json.loads((SOURCE / 'READBACK.json').read_text())['data_pair_feasibility_passed']:
        raise RuntimeError('Require the frozen full607 successful paired execution')
    if not json.loads((SOURCE / 'REFERENCE_FOLLOWING_READBACK.json').read_text())['exploratory_reference_following_checks_passed']:
        raise RuntimeError('Require independent actual reference following for the existing pair')
    RUN.mkdir(exist_ok=False)
    plan = dict(checkpoint=str(CHECKPOINT), sources=[96, 90], switch_frames=[158, 178, 218, 258],
                reused_switch_frame=158, new_switch_frames=[178, 218, 258], arms=list(ARMS),
                source_experiment=str(SOURCE), seed=272012, steps_per_arm=400,
                new_rollouts=9, maximum_new_control_steps=3600, optimizer_updates=0,
                intervention='Only the predeclared reference-selection/causal reference-alignment frame changes between phase groups. Within each group full weights, initial world, physics, command history and seed are shared; only selected reference differs after the switch. Never reset or write actual world state at the switch.',
                data_admission='For each of all4declared phase groups, require complete original/repeat/alternate400step frozen finite execution with sustained lift, exact startup/common prefix/switch world, and independent own-versus-other reference following. Export only admitted actual episodes with the official formatter. Recover the exact phase3 shared-world sample, preserving8x36 reference labels and separate29-D actual actions. All failures stay in4group denominator.',
                new_budget_pass='All3newphase groups must satisfy data admission, giving4/4including the reused158group and8real branch-point examples. This is within-one-TRAIN-pair phase coverage, not independent motions or policy-training replicates.',
                scope='Full601629parameter846-D frozen official-derived Tracker with known reference commands and48-D measured reference positions. This is real paired-data feasibility, not generated closed-loop control, human-video following or SMP benefit. Existing158results are reused explicitly, never counted as new physics.',
                automatic_next_action='Finish all9bounded actual rollouts and independent readback, even if a scientific admission check fails. If all4groups pass, inspect all physical curves and exact formatter/context/label data before declaring a new matched Generator data experiment. If any fail, diagnose its actual reference/physics evidence with the full saved model, keeping failures in the denominator. No Generator updates are launched by this collector.')
    write(RUN / 'PROTOCOL.json', plan)
    parent = json.loads((SOURCE / 'PROTOCOL.json').read_text())
    for frame in plan['switch_frames']:
        out = RUN / f'switch_{frame:03d}'; out.mkdir()
        group = dict(parent, switch_control_frame=frame,
                     purpose='Frozen fullmodel607 actual same-world branch phase coverage',
                     alignment=f'At{frame} apply the same single causal reference yaw/boxXY transform in both arms. Actual world and sensor history are never written.',
                     checkpoint_origin=str(CHECKPOINT),
                     automatic_next_action='Complete original/repeat/alternate400steps and independent readbacks. Scientific failures remain explicit; no model update follows automatically.')
        (out / 'motions').symlink_to((SOURCE / 'motions').resolve(), target_is_directory=True)
        if frame == 158:
            for arm in ARMS:
                (out / arm).symlink_to((SOURCE / arm).resolve(), target_is_directory=True)
            group['reused_existing_actual_results'] = str(SOURCE)
        write(out / 'PROTOCOL.json', group)
    print(json.dumps(plan), flush=True)


def pipeline():
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Actual PhysX collection requires the retained compute step')
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    if not json.loads((RUN / 'FORMATTER_PREFLIGHT.json').read_text())['passed']:
        raise RuntimeError('Complete actual saved-branch formatter regression before new collection')
    if (RUN / 'CHILDREN.json').exists():
        raise RuntimeError('Inspect recorded collection children; do not duplicate actual rollouts')
    children, results = [], {}

    def child(tag, module, args):
        command = [sys.executable, str(BOOT), 'scripts.sugar.demo_following.demo_future.' + module, *map(str, args)]
        with (RUN / (tag + '.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(tag=tag, pid=process.pid, pgid=os.getpgid(process.pid), command=command)
            children.append(row); write(RUN / 'CHILDREN.json', children)
            row['returncode'] = process.wait(); write(RUN / 'CHILDREN.json', children)
        if row['returncode']:
            raise RuntimeError('Execution failure in exact recorded child ' + tag)

    for frame in plan['switch_frames']:
        out = RUN / f'switch_{frame:03d}'
        if plan.get('reused_complete_groups') and frame not in plan['new_switch_frames']:
            previous = json.loads((Path(plan['reused_complete_groups']) / 'RESULT.json').read_text())
            row = dict(previous['per_switch'][str(frame)], reused_existing_rollouts=True)
            if not row['admitted'] or not row['physical_plots_inspected']:
                raise RuntimeError('Reused complete actual group lost its data or visual checks')
            results[str(frame)] = row
            write(RUN / 'PARTIAL_RESULT.json', results)
            continue
        if frame in plan['new_switch_frames']:
            for arm in ARMS:
                child(f'switch{frame}_{arm}', 'collect_refiner_pair', [
                    '--headless', '--controller', 'tracker', '--tracker-checkpoint', CHECKPOINT,
                    '--robot-usd', USD, '--run', out, '--arm', arm])
        child(f'switch{frame}_pair_readback', 'readback_refiner_pair', ['--run', out])
        child(f'switch{frame}_reference_readback', 'readback_reference_following', ['--run', out])
        pair = json.loads((out / 'READBACK.json').read_text())
        following = json.loads((out / 'REFERENCE_FOLLOWING_READBACK.json').read_text())
        admitted = pair['data_pair_feasibility_passed'] and following['exploratory_reference_following_checks_passed']
        row = dict(admitted=admitted, pair_checks=pair['checks'], following_checks=following['checks'],
                   steps={arm: json.loads((out / arm / 'RESULT.json').read_text())['actual_steps'] for arm in ARMS},
                   reused_existing_rollouts=frame not in plan['new_switch_frames'], physical_plots_inspected=False)
        if admitted:
            child(f'switch{frame}_official_export', 'export_tracker_il', ['--run', out])
            from scripts.sugar.demo_following.demo_future.prepare_generator_demo_context import prepare_branch_point
            prepare_branch_point(out, out / 'branch_samples')
            row['branch_sample_checks'] = json.loads((out / 'branch_samples/RESULT.json').read_text())['checks']
            row['admitted'] = row['admitted'] and all(row['branch_sample_checks'].values())
        results[str(frame)] = row
        write(RUN / 'PARTIAL_RESULT.json', results)
    final = dict(execution_completed=True, per_switch=results, declared_phase_groups=len(plan['switch_frames']),
                 admitted_phase_groups=sum(row['admitted'] for row in results.values()),
                 all_phase_data_passed=all(row['admitted'] for row in results.values()),
                 actual_new_rollouts=3 * len(plan['new_switch_frames']), actual_new_control_steps=sum(sum(row['steps'].values()) for row in results.values() if not row['reused_existing_rollouts']),
                 optimizer_updates=0, distinct_source_motions=2, physical_plots_inspected=False,
                 scope=plan['scope'], next_action=plan['automatic_next_action'])
    if plan.get('retained_failed_phase_groups'):
        final.update(retained_failed_phase_groups=plan['retained_failed_phase_groups'],
            cumulative_distinct_attempted_phase_groups=plan['cumulative_distinct_attempted_phase_groups'],
            cumulative_admitted_phase_groups=final['admitted_phase_groups'],
            preceding_failed_collection=plan['preceding_failed_collection'])
    write(RUN / 'RESULT.json', final)
    print(json.dumps(final), flush=True)


def prepare_gap_extension():
    """Two largest TRAIN gaps; complete existing physics/formatter path only."""
    import numpy as np
    previous=BASE/'generator_actual_branch_coverage_compatible'
    model_run=BASE/'matched_generator_branch_latent_replay01512'
    evidence=model_run/'frozen_evaluation/phase_gap_screen_r1'
    screen=json.loads((evidence/'RESULT.json').read_text())
    coverage=json.loads((model_run/'frozen_evaluation/joint_condition_coverage/RESULT.json').read_text())
    comparison=json.loads((model_run/'frozen_evaluation/LATENT_REPLAY_COMPARISON.json').read_text())
    old=json.loads((previous/'RESULT.json').read_text())
    if not (screen['checks_passed'] and screen['permits_preparing_bounded_actual_collection'] and screen['selected_new_phases']==[221,261]
            and coverage['checks_passed'] and coverage['all18_queries_and_54checks_inspected']
            and comparison['checks_passed'] and comparison['decision']['all_seven_train_pass'] and not comparison['decision']['both_reused_checks_pass']):
        raise RuntimeError('Require complete full-model fit/transfer/coverage and static gap evidence')
    if RUN!=BASE/'generator_actual_branch_coverage_gaps2':raise RuntimeError('Use the separately declared two-gap collection')
    plan=json.loads((previous/'PROTOCOL.json').read_text())
    reused=plan['switch_frames']
    assert len(reused)==9 and old['physical_plots_inspected'] and all(old['per_switch'][str(p)]['admitted'] and old['per_switch'][str(p)]['physical_plots_inspected'] for p in reused)
    assert plan['retained_failed_phase_groups']==[237,358]
    train=sorted([158,178,197,245,277,298,318,221,261])
    plan.update(switch_frames=sorted(reused+[221,261]),new_switch_frames=[221,261],reused_switch_frames=reused,
        reused_complete_groups=str(previous),new_rollouts=6,maximum_new_control_steps=2400,
        prospective_train_phases=train,frozen_check_phases=[218,258],
        cumulative_distinct_attempted_phase_groups=13,retained_failed_phase_groups=[237,358],
        preceding_failed_collection=str(previous),static_admission_evidence=str(evidence/'RESULT.json'),
        phase_selection=screen['selections'],phase_selection_rule=json.loads((evidence/'PROTOCOL.json').read_text())['selection_rule'],
        rationale='Current full latent replay objective fits all7TRAIN phases, but reused218/258 fail with systematic error. Complete fullencoder coverage still favors original branch in joint neighborhoods. Fill the two largest adjacentTRAIN gaps (197,245) and (245,277) at221/261, chosen by original clocks and static official compatibility, never target-error ranking.',
        data_admission='Unchanged full frozen601629parameter846-D model607, sameworld/seed/initialhistory, original single causal reference yaw/boxXY alignment. Each new group original/repeat/alternate400steps, finite fullstate, sustainedlift, exact prefix/switchworld, independent own-versus-other reference following and complete official formatter. Never alter world, referenceZ, terms, thresholds or labels. Failed237/358 remain excluded from labels and included in cumulative denominator.',
        new_budget_pass='Exactly two new groups, six400step attempts,max2400controls. Both must pass unchanged admission and every physical/reference/XYZ curve must be inspected. Only then11required groups available; maximum11/13cumulative, never reversal of earlier9/11.',
        scope='New actual data feasibility only.27old admitted rollouts read-only reused,6newattempts; preserve2failed old groups. Same96/90TRAIN sources and onefull frozenTracker seed, not independent motion generalization, generated future execution or SMP benefit. Static compatibility doesnot establish finite-horizon following.',
        automatic_next_action='Complete all6bounded new actualattempts and independent readbacks, even after scientific failures. Inspect physical/reference/XYZ evidence and exact clock/formatter exports. If both pass, prepare a separate18realcase full Generator coverage experiment retaining the demonstrated0.25rank+0.1latent objective and oldnormalizer, with fresh replay only for new cases from the samefull025teacher and old14replay arrays exact. Explicitly predeclare changed batch/total exposure and all11phase evaluation; do not append existing512 budgets. Collector never starts Generator training; any failure remains in13cumulative denominator.')
    provenance={}
    for arm,source in (('original',96),('alternate',90)):
        with np.load(BASE/f'refiner_aligned96_90_feasibility/{arm}/TRACE.npz') as a:provenance[source]=a['reference_frame'].copy()
    clocks={split:{source:{int(raw[f]) for phase in phases for f in phase+np.arange(8)*5} for source,raw in provenance.items()} for split,phases in (('train',train),('check',[218,258]))}
    plan['prospective_future_source_clocks_disjoint']={str(source):not bool(clocks['train'][source]&clocks['check'][source]) for source in provenance}
    assert all(plan['prospective_future_source_clocks_disjoint'].values())
    RUN.mkdir(exist_ok=False);write(RUN/'PROTOCOL.json',plan)
    parent=json.loads((SOURCE/'PROTOCOL.json').read_text())
    for frame in plan['switch_frames']:
        out=RUN/f'switch_{frame:03d}'
        if frame in reused:
            out.symlink_to((previous/out.name).resolve(),target_is_directory=True);continue
        out.mkdir();(out/'motions').symlink_to((SOURCE/'motions').resolve(),target_is_directory=True)
        group=dict(parent,switch_control_frame=frame,purpose='Frozenfull607 two-largest-TRAIN-gap actual branch coverage',
            alignment=f'At{frame} use the original single causal reference yaw/boxXY transform, never write world or history.',
            budget_reason=f'Exactly400controls,{400-frame}post-selection controls,lastlabel{frame+35};no padding.',
            checkpoint_origin=str(CHECKPOINT),automatic_next_action='Complete original/repeat/alternate actual endpoints and unchanged independent readbacks; no optimizer or generated controller.')
        write(out/'PROTOCOL.json',group)
    from scripts.sugar.demo_following.demo_future.prepare_generator_demo_context import prepare_branch_point
    regression=RUN/'formatter_regression';regression.mkdir();checks={}
    for phase in reused:
        target=regression/f'phase_{phase}';prepare_branch_point(previous/f'switch_{phase}',target)
        with np.load(target/'BRANCH_SAMPLES.npz') as a,np.load(previous/f'switch_{phase}/branch_samples/BRANCH_SAMPLES.npz') as old_arrays:
            checks[str(phase)]=set(a.files)==set(old_arrays.files) and all(np.array_equal(a[k],old_arrays[k]) for k in a.files)
    write(RUN/'FORMATTER_PREFLIGHT.json',dict(passed=all(checks.values()),checks=checks,reused_groups=9,reused_actual_branches=18,new_optimizer_updates=0,new_physics_steps=0,scope='All saved9groups official formatter arrays exactly reproduced in new regression directories; source arrays never overwritten.'))
    assert all(checks.values())
    print(json.dumps(dict(run=str(RUN),new_phases=[221,261],new_rollouts=6,max_new_steps=2400,all9formatter_regressions_exact=True)),flush=True)


def prepare_extension(bracket=False, dense=False, compatible=False):
    import numpy as np
    previous = BASE / ('generator_actual_branch_coverage6' if bracket else 'generator_actual_branch_coverage4')
    if dense:
        previous = BASE / 'generator_actual_branch_coverage_bracket'
    if compatible:
        previous = BASE / 'generator_actual_branch_coverage_dense'
    result = json.loads((previous / 'RESULT.json').read_text())
    model_run = BASE / 'matched_generator_branch_replay16_r1'
    model = json.loads((model_run / 'RESULT.json').read_text())
    gradients = json.loads((model_run / 'objective_probe/RESULT.json').read_text())
    coverage = json.loads((model_run / 'frozen_evaluation/REPLAY_COVERAGE_READBACK.json').read_text())
    reused_phases = [158, 178, 218, 258, 318] if bracket else [158, 178, 218, 258]
    if dense:
        reused_phases = [158, 178, 218, 258, 298, 318]
    if compatible:
        reused_phases = [158,178,197,218,258,277,298,318]
    if not all(result['per_switch'][str(p)]['admitted'] for p in reused_phases) or not result['physical_plots_inspected'] or not model['all_horizon_plots_inspected'] or model['combined_conditioning_passed'] or not gradients['checks_passed'] or not coverage['checks_passed']:
        raise RuntimeError('Complete actual four-phase data and negative replay diagnostics first')
    expected_name = 'generator_actual_branch_coverage_dense' if dense else ('generator_actual_branch_coverage_bracket' if bracket else 'generator_actual_branch_coverage6')
    if compatible:
        expected_name = 'generator_actual_branch_coverage_compatible'
    if RUN.name != expected_name:
        raise RuntimeError('Use the separate six-phase coverage directory')
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    plan.pop('reused_switch_frame', None)
    plan.update(switch_frames=[158, 178, 218, 258, 318, 358], new_switch_frames=[318, 358],
        reused_switch_frames=[158, 178, 218, 258], reused_complete_groups=str(previous),
        new_rollouts=6, maximum_new_control_steps=2400,
        prospective_train_phases=[158, 178, 318, 358], frozen_check_phases=[218, 258],
        rationale='Only early shared-world branches were fitted; later phases collapse to the original despite distinct geometry neighbors. Add real later branches to bracket the same already-used check phases, instead of more identical noise repeats. This is known-source interpolation coverage, not a new untouched test.',
        data_admission='Same frozen fullmodel607, all original/repeat/alternate400step physical and independent reference-following checks, exact common world/history and official formatter labels. All6groups remain in denominator. Reuse4complete groups read-only; collect only2newgroups. Late branches have42/82post-switch controls, no horizon padding.',
        new_budget_pass='Both newly collected318/358groups must pass existing actual-data admission. Then inspect all new physical curves and recover4new real branch examples. Existing heldout218/258source future-label frames remain excluded from prospective fitting data.',
        scope='Complete frozen601629parameter846-D official-derived Tracker with known reference plans, actual PhysX only,0optimizer updates. Six new400step rollout attempts maximum2400steps;12oldrollouts reused explicitly. Same96/90TRAIN sources and policy seed, not independent source coverage or generated/SMP success.',
        automatic_next_action='Finish all6newbounded rollouts and full data/independent physical readback regardless of scientific failures. If both late groups pass, inspect all plots and exact future-label clock separation before a new matched interpolation protocol; no Generator training inside this collector. Otherwise diagnose actual late physical/reference failures without dropping them from the denominator or adding undeclared rollouts.')
    if bracket:
        if result['all_phase_data_passed'] or result['per_switch']['358']['admitted']:
            raise RuntimeError('Bracket follow-up requires the inspected358following failure')
        plan.update(switch_frames=[158, 178, 218, 258, 298, 318], new_switch_frames=[298],
            reused_switch_frames=reused_phases, new_rollouts=3, maximum_new_control_steps=1200,
            prospective_train_phases=[158, 178, 298, 318],
            preceding_failed_collection=str(previous), retained_failed_phase_groups=[358],
            cumulative_distinct_attempted_phase_groups=7,
            rationale='Phase358 has42post-switch frames and fails independent body/box following despite lift. Keep its negative6group outcome. New298has102post-switch frames and disjoint future-label clocks, extending the observed interval without extending or relabeling any old rollout. Pair with already admitted318to bracket fixed218/258checks.',
            new_budget_pass='The one new298group must pass unchanged full physical/common-world and independent reference criteria. Prospective six admitted data phases are158/178/218/258/298/318. Cumulative attempted phases include failed358: report at most6/7, never7/7 or reversal of the previous5/6negative.',
            data_admission='Unchanged per-group criteria, full model607 and original/repeat/alternate400steps. Read-only reuse five admitted groups; retain358failure separately in cumulative7phase denominator. No label export from failed358.',
            scope='Exactly3newfrozen fullTracker400step PhysX attempts, max1200newsteps,0updates;15oldrollouts reused,358failure remains in cumulative7phase attempt history. Same96/90TRAIN motions, no generated control or SMP claim.',
            automatic_next_action='Complete all3new298attempts and unchanged independent readbacks, retaining any failure. If298passes, inspect all8physical/reference panels and source clocks before preparing matched interpolation on the6admitted groups; report cumulative6/7including358failure. If negative, diagnose actual following evidence without further undeclared collection. No Generator training in this collector.')
    if dense:
        fitting_run = BASE / 'matched_generator_branch_interpolation_fit512'
        fitting = json.loads((fitting_run / 'RESULT.json').read_text())
        joint = json.loads((fitting_run / 'frozen_evaluation/joint_condition_coverage_r1/RESULT.json').read_text())
        donor = json.loads((fitting_run / 'frozen_evaluation/geometry_phase_probe/RESULT.json').read_text())
        if not fitting['horizon_plots_inspected'] or fitting['heldout_phase_transfer_passed'] or not joint['checks_passed'] or not donor['checks_passed']:
            raise RuntimeError('Complete negative fitting and full joint-coverage evidence before dense collection')
        source_root = BASE / 'refiner_aligned96_90_feasibility'
        provenance = {}
        for arm, source in (('original',96),('alternate',90)):
            with np.load(source_root / arm / 'TRACE.npz') as trace:
                provenance[source] = trace['reference_frame'].copy()
        used = {s: {int(provenance[s][f]) for p in (218,258) for f in p+np.arange(8)*5} for s in provenance}
        selection = {}
        for center in (198,238,278):
            eligible = [p for p in range(center-15,center+16)
                        if all(not (set(map(int,frames[p+np.arange(8)*5])) & used[s]) for s,frames in provenance.items())]
            if not eligible:
                raise RuntimeError('No disjoint source-clock phase in the declared neighborhood')
            selection[str(center)] = dict(eligible_phases=eligible, selected=min(eligible,key=lambda p:(abs(p-center),p)))
        new_phases = [row['selected'] for row in selection.values()]
        if new_phases != [197,237,277]:
            raise RuntimeError('Actual raw source clocks differ from the bounded coverage proposal')
        plan.update(switch_frames=sorted(reused_phases+new_phases), new_switch_frames=new_phases,
            reused_switch_frames=reused_phases, reused_complete_groups=str(previous),
            new_rollouts=9, maximum_new_control_steps=3600,
            prospective_train_phases=[158,178,197,237,277,298,318], frozen_check_phases=[218,258],
            preceding_failed_collection=str(BASE / 'generator_actual_branch_coverage6'),
            retained_failed_phase_groups=[358], cumulative_distinct_attempted_phase_groups=10,
            phase_selection=selection,
            phase_selection_rule='Nearest phase to198/238/278 within15controls with both raw-source future-label frame sets disjoint from fixed218/258; ties choose lower phase. Uses clocks only, no targets, scores or physical outcome selection.',
            rationale='Eight-case fit and geometry-only interventions fail middle-state branches. Complete learned-token/native-data coverage shows nearby demo geometry and available futures, but joint state/history conditions often prefer original. Add three actual shared-state reference interventions near the missing middle phases; no synthetic input mixtures or extra noise repeats as data.',
            data_admission='Same full frozen601629parameter846-D model607 and original/repeat/alternate400step PhysX, exact common world/history, sustained lift, independent own-versus-other reference following and full official formatter checks. All nine required groups remain;358failure stays in cumulative ten attempted phases.',
            new_budget_pass='All three new197/237/277groups must pass unchanged complete-data criteria, then inspect all physical/XYZ curves and exact source-label separation. Maximum9/10 cumulative including failed358; never relabel as10/10.',
            scope='Nine new actual400step attempts, maximum3600newcontrols,0updates;18oldrollouts read-only reused. Same96/90TRAIN motions, same frozen policy and seed, no independent source/seed replication, generated execution or SMP benefit.',
            automatic_next_action='Complete all nine bounded actual attempts and unchanged independent readbacks regardless of scientific failures. Inspect new physical/XYZ evidence and exact formatter/source clocks. If all new groups pass, prepare a separate full official Generator coverage protocol preserving fixed218/258 checks and explicitly accounting for fourteen real fitting cases; do not use the older eight-case protocol unchanged. If any fail, preserve denominator and diagnose saved actual evidence; no automatic undeclared collection or training.')
    if compatible:
        admission = json.loads((previous / 'switch_admission_audit/RESULT.json').read_text())
        if not admission['checks_passed'] or admission['selected_next_phase'] != 245 or result['per_switch']['237']['admitted']:
            raise RuntimeError('Require exact static237failure reconstruction and declared245candidate')
        plan.update(switch_frames=sorted(reused_phases+[245]), new_switch_frames=[245],
            reused_switch_frames=reused_phases, reused_complete_groups=str(previous),
            new_rollouts=3, maximum_new_control_steps=1200,
            prospective_train_phases=[158,178,197,245,277,298,318], frozen_check_phases=[218,258],
            preceding_failed_collection=str(previous), retained_failed_phase_groups=[237,358],
            cumulative_distinct_attempted_phase_groups=11,
            static_admission_evidence=str(previous / 'switch_admission_audit/RESULT.json'),
            phase_selection=dict(failed_phase=237,selected_phase=245,center=238),
            phase_selection_rule=admission['selection_rule'],
            rationale='237switch already violates unchanged0.3m object and end-effector terms before its first action. Six real pre-action term sets and aligned arrays reproduce exactly.245is nearest-to238clock-disjoint candidate with both reference term sets false on a saved unswitched real state. Static compatibility is not rollout success.',
            data_admission='Same complete frozen model607, initial world/seed/physics/history, original causal yaw/boxXY-only reference alignment, all400steps and independent reference following. Never shift reference Z, relax terms or reset actual world. Preserve both237and358failures.',
            new_budget_pass='Exactly one new245original/repeat/alternate group, max1200steps. If all unchanged data checks pass and plots inspected, selected nine groups are complete; cumulative remains at most9/11 including237/358. The preceding8/9negative dense experiment remains unchanged.',
            scope='Three new actual400step attempts,24oldrollouts read-only reused,0updates. Same96/90TRAIN motions and full model607. New245feasibility unproven, no generated control or SMP benefit.',
            automatic_next_action='Complete all3new245attempts and original independent readbacks; retain any failure. Inspect physical/reference/XYZ curves and exact source clocks. If245passes, prepare the separate fourteen-case full Generator protocol using seven TRAIN and fixed218/258checks while reporting9/11cumulative. If it fails, diagnose saved reference/physics evidence without undeclared rollout or training. No model training inside collector.')
    clocks = {name: {source: set() for source in (96, 90)} for name in ('train', 'check')}
    source_root = BASE / 'refiner_aligned96_90_feasibility'
    for name, phases in (('train', plan['prospective_train_phases']), ('check', plan['frozen_check_phases'])):
        for arm, source in (('original', 96), ('alternate', 90)):
            with np.load(source_root / arm / 'TRACE.npz') as trace:
                for phase in phases:
                    frames = phase + np.arange(8) * 5
                    if not np.all(trace['reference_id'][frames] == source):
                        raise RuntimeError('Prospective raw-demo source clock changes inside a phase')
                    clocks[name][source].update(trace['reference_frame'][frames].tolist())
    plan['prospective_future_source_clocks_disjoint'] = {str(source): not bool(clocks['train'][source] & clocks['check'][source]) for source in (96, 90)}
    if not all(plan['prospective_future_source_clocks_disjoint'].values()):
        raise RuntimeError('Late coverage overlaps retained check future-label source clocks')
    RUN.mkdir(exist_ok=False)
    write(RUN / 'PROTOCOL.json', plan)
    parent = json.loads((SOURCE / 'PROTOCOL.json').read_text())
    for frame in plan['switch_frames']:
        out = RUN / f'switch_{frame:03d}'
        if frame in plan['reused_switch_frames']:
            out.symlink_to((previous / out.name).resolve(), target_is_directory=True)
            continue
        out.mkdir()
        (out / 'motions').symlink_to((SOURCE / 'motions').resolve(), target_is_directory=True)
        group = dict(parent, switch_control_frame=frame, purpose='Frozen fullmodel607 late actual shared-world branch coverage',
            alignment=f'At{frame} apply the existing single causal reference yaw/boxXY transform; never write actual state or history.',
            budget_reason=f'Exactly400controls within450reference frames, {400-frame}post-selection frames, last label frame{frame+35}; no padding.',
            checkpoint_origin=str(CHECKPOINT), automatic_next_action='Complete all original/repeat/alternate endpoints and independent readbacks; no model updates.')
        write(out / 'PROTOCOL.json', group)
    # Regression re-exports go in a new folder; never rewrite the four reused groups.
    from scripts.sugar.demo_following.demo_future.prepare_generator_demo_context import prepare_branch_point
    regression = RUN / 'formatter_regression'; regression.mkdir()
    checks = {}
    for frame in plan['reused_switch_frames']:
        out = regression / f'phase_{frame}'
        prepare_branch_point(previous / f'switch_{frame}', out)
        with np.load(out / 'BRANCH_SAMPLES.npz') as actual, np.load(previous / f'switch_{frame}/branch_samples/BRANCH_SAMPLES.npz') as original:
            checks[str(frame)] = set(actual.files) == set(original.files) and all(np.array_equal(actual[k], original[k]) for k in original.files)
    report = dict(passed=all(checks.values()), existing_actual_phase_arrays_exact=checks,
        prospective_source_future_clocks_disjoint=plan['prospective_future_source_clocks_disjoint'], new_physics_steps=0,
        scope=f'Read-only re-export of all{2*len(plan["reused_switch_frames"])}existing actual examples into separate regression folder. New actual feasibility not yet established.')
    write(RUN / 'FORMATTER_PREFLIGHT.json', report)
    if not report['passed']:
        raise RuntimeError('Existing actual formatter regression failed')
    print(json.dumps(dict(run=str(RUN), new_rollouts=plan['new_rollouts'], max_new_steps=plan['maximum_new_control_steps'], preflight=report)), flush=True)


def prepare_phase_arrays(collection):
    import numpy as np
    data = {}
    for split, phases in (('train', collection['prospective_train_phases']), ('heldout_phase', collection['frozen_check_phases'])):
        arrays, samples = {}, []
        for phase in phases:
            folder = RUN / f'switch_{phase}/branch_samples'
            report = json.loads((folder / 'RESULT.json').read_text())
            if not all(report['checks'].values()):
                raise RuntimeError('Actual branch data checks failed')
            with np.load(folder / 'BRANCH_SAMPLES.npz') as source:
                for key in source.files: arrays.setdefault(key, []).append(source[key].copy())
            for arm in ('original', 'alternate'):
                samples.append(dict(phase=phase, arm=arm, **report['per_arm'][arm]))
        path = RUN / ('TRAIN_BRANCH_SAMPLES.npz' if split == 'train' else 'HELDOUT_PHASE_BRANCH_SAMPLES.npz')
        combined = {key: np.concatenate(values) for key, values in arrays.items()}
        if path.exists():
            with np.load(path) as saved:
                if set(saved.files) != set(combined) or not all(np.array_equal(saved[k], v) for k, v in combined.items()):
                    raise RuntimeError('Previously prepared actual arrays differ; never overwrite')
        else:
            np.savez_compressed(path, **combined)
        data[split] = dict(phases=phases, real_examples=len(samples), arrays=str(path), samples=samples)
    clocks = {name: {source: {f for s in row['samples'] if s['source'] == source for f in s['source_frames']} for source in (96, 90)} for name, row in data.items()}
    disjoint = {str(source): not bool(clocks['train'][source] & clocks['heldout_phase'][source]) for source in (96, 90)}
    if not all(disjoint.values()):
        raise RuntimeError('Actual admitted training and check future source clocks overlap')
    return data, disjoint


def prepare_interpolation_protocol():
    import numpy as np
    result = json.loads((RUN / 'RESULT.json').read_text())
    collection = json.loads((RUN / 'PROTOCOL.json').read_text())
    if not result['all_phase_data_passed'] or not result['physical_plots_inspected']:
        raise RuntimeError('Complete all late actual feasibility and physical curve inspection first')
    if (RUN / 'NEXT_MATCHED_PROTOCOL.json').exists():
        raise RuntimeError('Do not overwrite the next matched protocol')
    previous = BASE / 'matched_generator_branch_replay16_r1'
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    data, disjoint = prepare_phase_arrays(collection)
    for key in ('preceding_execution_failure', 'spent_aborted_updates_separate', 'schema_correction'):
        plan.pop(key, None)
    plan.update(run_name='matched_generator_branch_interpolation_replay16', data=data,
        branch_dataset=dict(corpus=str(RUN), normalizer_state=plan['normalizer_state'], repetitions=8, split='train'),
        repetitions_per_real_training_example=8, real_branch_examples=8,
        original_source_future_clocks_disjoint=disjoint,
        phase_corpora={str(p): str(RUN / f'switch_{p}/branch_samples') for p in collection['switch_frames']},
        data_coverage_predecessor=str(previous),
        training=f"Same full release initialization/frozen old TRAIN normalizer, fresh AdamW5e-5, seed272051,16epochs,batch128,656updates each arm, full official loop and16step DDPM. Same3897native rows plus1296branch noise rows per epoch; replace half the former early branch exposures with4actual cases from late phases{[p for p in collection['prospective_train_phases'] if p >= 200]}. Each of8real branches repeated162times per epoch; not independent rollouts. IID/goalzero/intact history unchanged.",
        comparison_scope=f"Same model, optimizer settings, total data rows/native rows, branch row weight, update budget and sampling as preceding replay. Only actual branch supervision phase coverage changes; each arm starts from full release state. Already-used218/258phase checks remain fixed, now between fitted phases{collection['prospective_train_phases']}. No untouched-test or motion-disjoint phase-generalization claim.",
        automatic_next_action='Complete all12actual branch-array and full mixed-batch preflight, then both fixed656update arms and all native/fourTRAIN/twoheldout-phase frozen sampling, saved readbacks and horizon inspection. Require original native conditioning and both heldout phase criteria; do not relax thresholds or add updates. If positive, separately resolve original position/goal and generated-controller prerequisites before an independent SMP comparison. If negative, inspect remaining fitting/context/physical evidence before a different bounded experiment; no same-budget extension.')
    plan['replay_dataset']['corpus'] = str(RUN)
    plan['phase_frozen_evaluation']['evaluation_phases'] = collection['switch_frames']
    plan['phase_frozen_evaluation']['scope'] = f"Known-source interpolation across actual branch phases, fitted{collection['prospective_train_phases']} versus already-used218/258 checks with disjoint future source-label frames. Same96/90TRAIN sources and previous normalization, not untouched/pretraining-held-out evaluation, generated physics or SMP benefit. Failed358physical collection remains excluded and reported in cumulative7phase denominator."
    plan['actual_collection_coverage'] = dict(required_admitted_phases=collection['switch_frames'],
        cumulative_attempted_phases=result.get('cumulative_distinct_attempted_phase_groups', result['declared_phase_groups']),
        cumulative_admitted_phases=result['admitted_phase_groups'], retained_failed_phases=result.get('retained_failed_phase_groups', []),
        preceding_failed_collection=result.get('preceding_failed_collection'))
    plan['optimization'] = plan['training'] + ' ' + plan['local_method']
    old_composition = plan['replay_composition']
    count = data['train']['real_examples']
    repetitions = old_composition['native_real_chunks'] // (3 * count)
    rows = old_composition['native_real_chunks'] + count * repetitions
    if rows != old_composition['total_rows_per_epoch']:
        raise RuntimeError('Late coverage unexpectedly changes total row budget')
    plan['replay_composition'] = dict(old_composition, real_branch_examples=count,
        branch_phases=data['train']['phases'], branch_repetitions=repetitions,
        total_rows_per_epoch=rows, branch_row_fraction=count * repetitions / rows)
    # Full actual dataset construction and model preflight run in the training
    # preparation module, where heavy imports finish before any main() writes.
    write(RUN / 'NEXT_MATCHED_PROTOCOL.json', plan)
    print(json.dumps(dict(next_run=plan['run_name'], composition=plan['replay_composition'], exact_updates_per_arm=656, source_clocks_disjoint=disjoint)), flush=True)


def prepare_dense_fit_protocol():
    if RUN.name not in ('generator_actual_branch_coverage_dense','generator_actual_branch_coverage_compatible'):
        raise RuntimeError('Use the separate declared dense actual corpus')
    collection = json.loads((RUN / 'PROTOCOL.json').read_text())
    result = json.loads((RUN / 'RESULT.json').read_text())
    if not result['all_phase_data_passed'] or not result['physical_plots_inspected']:
        raise RuntimeError('Complete all actual endpoints, independent following and plot inspection first')
    if (RUN / 'NEXT_MATCHED_PROTOCOL.json').exists():
        raise RuntimeError('Preserve the declared next protocol')
    previous = BASE / 'matched_generator_branch_interpolation_fit512'
    plan = json.loads((previous / 'PROTOCOL.json').read_text())
    data, disjoint = prepare_phase_arrays(collection)
    count = data['train']['real_examples']
    cumulative = collection['cumulative_distinct_attempted_phase_groups']
    if count != 14 or result['cumulative_admitted_phase_groups'] != 9 or result['cumulative_distinct_attempted_phase_groups'] != cumulative:
        raise RuntimeError('Require all fourteen real fitted cases while retaining358failure')
    description = ('Separate full official Generator dense-phase fitting diagnostic. Same full released '
        'initialization/frozen old normalizer/IID noise/zero normalized goal/intact actual history, '
        'seed272084, AdamW1e-4, cosine warmup16, exactly512updates each arm. Fourteen actual '
        'cases each repeated8times per balanced112row batch;4096 noise exposures per case, '
        'same as previous eight-case512. Six real cases are added and batch grows64to112; '
        'total noise rows increase32768to57344 per arm. This is not a fixed-total-compute or '
        'single-variable native-replay comparison, independent motions or physical execution.')
    plan.update(run_name='matched_generator_branch_dense_fit512', data=data,
        branch_dataset=dict(corpus=str(RUN), normalizer_state=plan['normalizer_state'], repetitions=8, split='train'),
        real_branch_examples=count, batch_size=112, repeated_examples_per_update=112,
        repetitions_per_real_training_example=8, original_source_future_clocks_disjoint=disjoint,
        phase_corpora={str(p):str(RUN / f'switch_{p}/branch_samples') for p in collection['switch_frames']},
        training=description, optimization=description, comparison_scope=description,
        diagnostic_predecessor=str(previous), generator_training_started=False,
        actual_collection_coverage=dict(cumulative_admitted_phases=9,cumulative_attempted_phases=cumulative,retained_failed_phases=collection['retained_failed_phase_groups']),
        automatic_next_action='Complete full112row preflight and both exact512update arms, then all9phases/five conditions/32draws, saved readbacks and all curves. Keep seven TRAIN checks separate from the original reused218/258 checks. If only fitting improves, report phase-transfer failure and diagnose remaining conditional representation; no update513 or success from loss alone. If both check phases pass, separately evaluate native prediction and original goal/position execution prerequisites before generated physics; no automatic SMP-benefit claim.')
    plan['frozen_evaluation']['evaluation_phases'] = collection['switch_frames']
    plan['frozen_evaluation']['scope'] = (f"Fourteen TRAIN cases at{collection['prospective_train_phases']} and four "
        'already-used218/258 check cases. Actual future-source-label sets disjoint, same96/90 motions '
        'and old normalization. Seven TRAIN results separate from two check results. Not untouched '
        f"test, independent source generalization, generated execution or SMP benefit; failed phases{collection['retained_failed_phase_groups']} retained.")
    write(RUN / 'NEXT_MATCHED_PROTOCOL.json', plan)
    print(json.dumps(dict(next_run=plan['run_name'],real_training_cases=count,batch_size=112,updates_per_arm=512,source_clocks_disjoint=disjoint)),flush=True)


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--pipeline', action='store_true')
    mode.add_argument('--prepare-extension', action='store_true')
    mode.add_argument('--prepare-interpolation-protocol', action='store_true')
    mode.add_argument('--prepare-bracket', action='store_true')
    mode.add_argument('--prepare-dense', action='store_true')
    mode.add_argument('--prepare-dense-fit-protocol', action='store_true')
    mode.add_argument('--prepare-compatible', action='store_true')
    mode.add_argument('--prepare-gap', action='store_true')
    args = parser.parse_args()
    RUN = args.run.resolve()
    if RUN.parent != BASE:
        raise RuntimeError('Use the local experiment root')
    if args.prepare_gap: prepare_gap_extension()
    elif args.prepare_compatible: prepare_extension(compatible=True)
    elif args.prepare_dense_fit_protocol: prepare_dense_fit_protocol()
    elif args.prepare_dense: prepare_extension(dense=True)
    elif args.prepare_bracket: prepare_extension(bracket=True)
    elif args.prepare_interpolation_protocol: prepare_interpolation_protocol()
    elif args.prepare_extension: prepare_extension()
    elif args.prepare: prepare()
    else: pipeline()


if __name__ == '__main__':
    main()
