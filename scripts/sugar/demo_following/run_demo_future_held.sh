#!/usr/bin/env bash
set -euo pipefail
ROOT=/public/home/yanhongru/Curiosity
cd "$ROOT"
test -n "${SLURM_JOB_ID:-}"
test -n "${SLURM_STEP_ID:-}"
case "$(hostname)" in login*|mgmtserver*) exit 2 ;; esac
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec 9>"$ROOT/experiments/demo_following/demo_future_smp_v1/pipeline.lock"
flock -n 9
stage="${1:-audit}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
BOOTSTRAP=scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py
case "$stage" in
    generator-actual-state-rollouts)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_generator_actual_state_supervision rollout_pipeline "$@"
        ;;
    generator-actual-state-supervision)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_generator_actual_state_supervision pipeline "$@"
        ;;
    generator-rollout-smp)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_generated_rollout_smp "$@"
        ;;
    generator-generated-command-rollout)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_generated_command_rollout pipeline "$@"
        ;;
    generator-measured-robot-pipeline)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_generator_measured_robot "$@"
        ;;
    generator-terminal-objective-qualification)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_terminal_objective --mode run "$@"
        ;;
    generator-fixed-teacher-seed-replication)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.replicate_generator_training_seed --mode pipeline "$@"
        ;;
    generator-existing-checkpoint-profile)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.profile_generator_existing_checkpoints --mode run "$@"
        ;;
    generator-optimizer-trajectory-evaluation)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.inspect_generator_optimizer_trajectory --mode evaluate "$@"
        ;;
    generator-exact-optimizer-replay)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.replay_generator_optimizer_trajectory "$@"
        ;;
    generator-self-objective-vs-sampler)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_self_objective_vs_sampler "$@"
        ;;
    generator-full-sampler-gradient)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_full_sampler_gradient "$@"
        ;;
    generator-self-path-transfer)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_generator_self_path_transfer --mode run "$@"
        ;;
    generator-gap-objective-gradient)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_gap_objective "$@"
        ;;
    generator-training-replay-probe)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_generator_training_replay "$@"
        ;;
    generator-frozen-phase-recovery)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.recover_generator_phase_evaluation --pipeline "$@"
        ;;
    generator-phase-gap-screen)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_phase_gaps "$@"
        ;;
    generator-latent-replay-cpu-preflight)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.preflight_generator_latent_replay --device cpu "$@"
        ;;
    generator-latent-replay-preflight)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.preflight_generator_latent_replay --device cuda "$@"
        ;;
    generator-train-diffusion-replay)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.collect_generator_train_diffusion_replay "$@"
        ;;
    generator-own-replay-gradient)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_own_replay "$@"
        ;;
    generator-generated-state-gradient)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_generated_state_gradient "$@"
        ;;
    generator-path-feedback)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.readback_generator_path_feedback "$@"
        ;;
    generator-hinge-gradient)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_paired_condition_objective --hinge-margin-audit "$@"
        ;;
    generator-recorded-condition)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_generator_recorded_condition --mode run "$@"
        ;;
    generator-reverse-probe)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_generator_reverse_process --mode run "$@"
        ;;
    generator-solver-probe)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_generator_solver_steps --mode evaluate "$@"
        ;;
    generator-endpoint-conflict)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_paired_condition_objective --endpoint-conflict "$@"
        ;;
    generator-rank-components)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_rank_components --mode frozen "$@"
        ;;
    audit)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_frozen "$@"
        ;;
    matched)
        "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.preflight
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.train_matched "$@"
        ;;
    gradient-probe)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_gradients "$@"
        ;;
    paired)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.train_matched \
            --experiment paired --output "$ROOT/experiments/demo_following/demo_future_smp_v1/matched_same_task_gap" "$@"
        ;;
    feature-audit)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_feature_contract "$@"
        ;;
    repair-reference-roles)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.repair_reference_roles "$@"
        ;;
    canonical-data)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.canonical_data "$@"
        ;;
    canonical-matched)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.train_matched \
            --experiment canonical --output "$ROOT/experiments/demo_following/demo_future_smp_v1/matched_corrected_canonical" "$@"
        ;;
    canonical-endpoint)
        RUN="$ROOT/experiments/demo_following/demo_future_smp_v1/matched_corrected_canonical"
        "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.readback --run "$RUN"
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.render_contrast --run "$RUN"
        ;;
    fit-information)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_fit_information "$@"
        ;;
    observable-geometry)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_observable_geometry "$@"
        ;;
    observable-hold)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_observable_hold "$@"
        ;;
    observable-hold-current-phase)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.probe_observable_hold --clock-mode current-phase-only "$@"
        ;;
    geometry-residual-matched)
        "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.preflight_residual
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.train_matched \
            --experiment geometry-residual --output "$ROOT/experiments/demo_following/demo_future_smp_v1/matched_geometry_residual" "$@"
        ;;
    geometry-residual-endpoint)
        RUN="$ROOT/experiments/demo_following/demo_future_smp_v1/matched_geometry_residual"
        "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.readback --run "$RUN" --control direct_geometry --treatment residual_geometry
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.render_contrast --run "$RUN" --control direct_geometry --treatment residual_geometry
        ;;
    residual-component)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.readback_residual_component "$@"
        ;;
    residual-information)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_residual_information "$@"
        ;;
    refiner-pair)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.collect_refiner_pair --headless "$@"
        ;;
    heldout-native-validation)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_heldout_reference_validation "$@"
        ;;
    generator-demo-preflight)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_generator_demo_geometry "$@"
        ;;
    generator-geometry-training)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.train_generator_geometry "$@"
        ;;
    generator-paired-training-preflight)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.prepare_paired_generator_training --preflight --device cuda "$@"
        ;;
    generator-paired-objective-audit)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_paired_condition_objective --audit "$@"
        ;;
    generator-denoising-probe)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.generator_branch_diagnostic --phase-denoising-probe "$@"
        ;;
    generator-geometry-phase-probe)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.generator_branch_diagnostic --phase-geometry-probe "$@"
        ;;
    generator-branch-transfer)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.generator_branch_diagnostic --transfer-only "$@"
        ;;
    generator-phase-input-probe)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.generator_branch_diagnostic --phase-input-probe "$@"
        ;;
    generator-replay-objective-probe)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.generator_branch_diagnostic --replay-objective-probe "$@"
        ;;
    generator-branch-coverage)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.collect_generator_branch_coverage "$@"
        ;;
    generator-data-and-matched)
        "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.collect_train_reference_corpus --resume-tracker
        "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.collect_train_reference_corpus --tracker-readback-only
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.train_generator_geometry --pipeline
        ;;
    train-reference-corpus)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.collect_train_reference_corpus "$@"
        ;;
    train-coverage-pipeline)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_matched_train_coverage "$@"
        ;;
    refiner-pair-readback)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.readback_refiner_pair "$@"
        ;;
    refiner-smp)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.score_refiner_smp "$@"
        ;;
    refiner-following)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.readback_reference_following "$@"
        ;;
    official-generator-audit)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.audit_official_generator "$@"
        ;;
    export-refined-motion)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.export_refined_motion "$@"
        ;;
    export-tracker-il)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.export_tracker_il "$@"
        ;;
    aligned-teacher-bank)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.build_aligned_teacher_motion "$@"
        ;;
    tracker-bcppo)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_tracker_bcppo "$@"
        ;;
    tracker-online-future)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.train_tracker_online_future --headless "$@"
        ;;
    tracker-online-future-pipeline)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_online_tracker_future "$@"
        ;;
    tracker-reference-feedback-pipeline)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_matched_reference_feedback "$@"
        ;;
    tracker-fixed-replay-bc)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.train_tracker_replay_bc "$@"
        ;;
    tracker-future-matched)
        exec "$PYTHON" "$BOOTSTRAP" scripts.sugar.demo_following.demo_future.run_matched_tracker_future "$@"
        ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
