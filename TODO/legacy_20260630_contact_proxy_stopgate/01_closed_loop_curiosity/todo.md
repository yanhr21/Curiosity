# Phase 01 Closed-Loop Curiosity TODO

## Active Rules

- Phase 01 starts from the accepted Phase 00 dataset.
- Do not run training, simulation, dataset conversion, model loading, or
  NumPy/PyTorch-heavy checks on the login node.
- Use Curiosity-owned tmux-held H200 Slurm allocations for all real compute.
- Keep new paths short and grouped under `phase01/core/`.
- Do not claim curiosity success from preflight, manifest building, lower loss,
  checkpoint existence, or videos alone.
- If five real one-hour training attempts have no positive held-out result,
  stop and report to the user before any sixth attempt.
- Do not commit unless the user explicitly asks.

## Setup

- [x] Create active Phase 01 plan directly under `PLAN/`.
- [x] Create active Phase 01 TODO directly under `TODO/`.
- [x] Write the five one-hour negative training attempt stop gate into
      `AGENTS.md`.
- [x] Establish short grouped Phase 01 artifact paths.
- [x] Build Phase 01 transition manifest from Phase 00 train/validation data
      inside H200 allocation.
- [x] Start first Phase 01 manifest build in H200 allocation.
      Progress: Slurm job `157899` on `server64`, run tag
      `phase01_core_manifest_20260629_235852`. Fresh official Newton
      SensorContact sanity passed. The build failed before producing training
      data because the first builder expected `newton.object.body_q`, while
      the real Phase 00 schema uses `newton.panda.object_body_q`. This is an
      invalid preflight/schema-adapter failure, not a training attempt and not
      a negative one-hour result.
- [x] Rerun Phase 01 manifest build after schema repair.
      Progress: repaired builder to use real Phase 00 schema
      `newton.panda.object_body_q`; rerun `phase01_core_manifest_r1_20260630_000040`
      completed in Slurm job `157899` on `server64` with exit code 0.
      Manifest status is `pass`, with 14392 train transitions and 5397
      validation transitions.
- [x] Verify held-out Phase 00 cells are excluded from train/validation
      transition artifacts.
      Evidence: manifest lists the four held-out cells only under
      `held_out_seen_but_excluded` / `held_out_excluded_from_training`.
- [x] Write Phase 01 manifest report under
      `experiments/reports/phase01/core/`.
      Evidence: `experiments/reports/phase01/core/transition_manifest.md`.

## Baselines

- [x] Declare the strongest available baseline set for Phase 01:
      no-adaptation, scripted feedback, no-curiosity residual adaptation, and
      any compatible serious/mainstream method or documented blocker.
      Evidence: `experiments/configs/phase01/baselines.json`.
- [x] Add grouped Phase 01 held-out baseline runner/launcher for the scripted
      non-curiosity baselines.
      Evidence:
      `experiments/configs/phase01/run_baselines_in_alloc.sh`,
      `experiments/configs/phase01/launch_baselines_tmux.sh`. These scripts
      are prepared but not yet run.
- [x] Run or refresh held-out baseline metrics with full MP4 evidence.
      Evidence:
      `experiments/reports/phase01/core/baselines/p01_base_heldout_r1_20260630_0120_summary.md`,
      `experiments/outputs/phase01/core/baselines/p01_base_mp4_20260630_0132_mp4_summary.json`.
      Completed 8/8 held-out method-cell evaluations and 8/8 MP4 videos.
- [x] Record baseline safety metrics: lift/hold, drop, contact loss, slip,
      acceleration, force/contact proxy, and no-op/instability indicators.
      Evidence is in the metrics JSON/CSV under
      `experiments/outputs/phase01/core/baselines/`.
- [x] Backfill Phase 01 baseline MP4 videos after the first baseline run
      produced GIF/video-frame evidence but no MP4 files.
      Evidence:
      `experiments/outputs/phase01/core/baselines/p01_base_mp4_20260630_0132_mp4_summary.json`,
      with 8/8 MP4 exports passing at 601 frames and 20 FPS.

## Real Training Attempts

- [x] Attempt ledger exists at
      `experiments/reports/phase01/core/training_attempts.json`.
- [x] Prepare grouped Phase 01 learning-progress scoring config/launcher for
      use after a valid forward-model checkpoint exists.
      Evidence:
      `experiments/configs/phase01/learning_progress.json`,
      `experiments/configs/phase01/run_learning_progress_in_alloc.sh`,
      `experiments/configs/phase01/launch_learning_progress_tmux.sh`. This is
      not yet run and does not update a policy.
- [x] Compute Phase 01 learning-progress scores from the valid forward-model
      checkpoint pair.
      Evidence:
      `experiments/outputs/phase01/core/lp/curiosity_learning_progress_summary.json`
      and `experiments/outputs/phase01/core/lp/curiosity_learning_progress_scores.csv`.
      Result: status pass, 19789 scores, mean learning progress
      `0.6390625392953463`, policy_updated=false. This is not a policy result
      and not a positive curiosity claim.
- [x] Attempt to build Phase 01 residual-controller manifest from real Phase 00
      Newton feedback fields.
      Evidence:
      `data/processed/phase01/resid/manifest.json`,
      `experiments/reports/phase01/core/residual_manifest.md`,
      `logs/newton/phase01/core/p01_resid_manifest_20260630_0138.srun.log`.
      Result: failed data gate with `train_active_feedback_count=0` and
      `validation_active_feedback_count=0`. This is not a training attempt and
      does not count toward the five one-hour negative-training stop gate. It
      blocks residual policy training from the current Phase 00 source rows
      because empty/inactive feedback labels would be a fake residual target.
- [x] Add Phase 01 train-only corrective source collection and advantage gate
      for repairing the residual-label blocker without touching held-out data.
      Evidence:
      `experiments/configs/phase01/src_collect.json`,
      `experiments/configs/phase01/run_src_collect_in_alloc.sh`,
      `experiments/configs/phase01/launch_src_collect_tmux.sh`,
      `experiments/configs/phase01/build_src_gate.py`. This path is data
      repair only: it pairs train-cell no-adaptation with official scripted
      feedback, requires active feedback labels, and admits only paired
      advantage without safety regression.
- [x] Run Phase 01 train-only corrective source collection inside a
      Curiosity-owned tmux-held H200 allocation.
      Evidence: run tag `p01_src_a3r2_20260630_0221`, Slurm job `157951` on
      `server51`, log
      `logs/newton/phase01/core/src/p01_src_a3r2_20260630_0221.srun.log`.
      Earlier partial runs `p01_src_a1_20260630_0152` and
      `p01_src_a2_20260630_0201` were stopped as parameter diagnostics:
      a1 was overactive, a2 had zero labels on the first checked cell. They
      are not training attempts.
- [x] Inspect `data/processed/phase01/src/manifest.json` and
      `experiments/reports/phase01/core/src_collect.md`.
      Result: source gate passed with 6 admitted train-only sources, 2 rejected
      sources, 9000 train rows, 1800 validation rows, 4061 train active
      feedback labels, 1616 validation active feedback labels, and all held-out
      cells excluded.
- [x] If the source gate passes, prepare one-hour no-curiosity residual
      baseline training from the gated source before curiosity-weighted
      residual training.
      Evidence:
      `experiments/configs/phase01/resid_base_train.json`,
      `experiments/configs/phase01/run_resid_base_train_in_alloc.sh`,
      `experiments/configs/phase01/launch_resid_base_train_tmux.sh`.
- [x] Run one-hour no-curiosity residual baseline training from the gated
      source.
      Evidence: `p01_resid_base_a1_20260630_0307` ran in Slurm job `157951`
      on `server51`; fresh official Newton sanity passed; summary
      `experiments/outputs/phase01/core/resid/base/p01_resid_base_a1_20260630_0307_summary.json`
      reports `real_training_result=true`, `elapsed_seconds=3600.239132165909`,
      `optimizer_steps=14460`, `checkpoint_written=true`, validation loss
      `0.30022132396698`, validation active accuracy `0.967222273349762`, and
      GPU utilization mean `90.32231404958678%`. Checkpoint:
      `checkpoints/phase01/core/resid/base/p01_resid_base_a1_20260630_0307.pt`.
      This is a learned non-curiosity baseline component, not curiosity
      success.
- [x] Evaluate the no-curiosity residual baseline checkpoint on the locked
      held-out Phase 01 cells before training/interpreting curiosity-weighted
      residual results.
      Evidence:
      `experiments/outputs/phase01/core/resid/base_eval/p01_resid_eval_a1_20260630_0342_summary.json`,
      `experiments/reports/phase01/core/resid/p01_resid_eval_a1_20260630_0342_summary.md`,
      and
      `experiments/reports/phase01/core/resid/p01_resid_eval_a1_20260630_0342_comparison.md`.
      Result: 4/4 held-out successes, but not a clean improvement over the
      strongest existing baseline set. This is learned baseline evidence, not
      curiosity success.
- [x] Prepare matched learning-progress evidence for the gated source rows so
      curiosity-weighted residual training does not use stale/mismatched
      Phase00 run tags.
      Evidence:
      `experiments/configs/phase01/src_lp_manifest.json`,
      `experiments/configs/phase01/src_lp_scores.json`,
      `experiments/configs/phase01/build_src_lp_manifest.py`,
      `experiments/configs/phase01/run_src_lp_in_alloc.sh`,
      `experiments/configs/phase01/launch_src_lp_tmux.sh`.
      Run `p01_src_lp_a1_20260630_0405` in Slurm job `157999` on `server64`
      passed fresh official Newton sanity, built
      `data/processed/phase01/src_lp/manifest.json`, and wrote matched scores
      to
      `experiments/outputs/phase01/core/src_lp/curiosity_learning_progress_summary.json`.
      Result: 8995 train transitions, 1799 validation transitions, 10794
      scores, mean learning progress `0.5819649252997756`,
      `policy_updated=false`, and held-out cells excluded. This is scoring
      evidence only, not policy training and not curiosity success.
- [x] Train the first curiosity-weighted residual candidate for a real one-hour run
      only after the matched score/preflight gate passes.
      Progress: config and launch path added at
      `experiments/configs/phase01/resid_curiosity_train.json`,
      `experiments/configs/phase01/run_resid_curiosity_train_in_alloc.sh`,
      and
      `experiments/configs/phase01/launch_resid_curiosity_train_tmux.sh`.
      Run `p01_resid_cur_a1_20260630_0407` launched in Curiosity H200 Slurm
      job `157999` on `server64`; fresh official Newton sanity passed and
      early GPU utilization samples reached 82% and 94%. Classification is
      now complete: summary
      `experiments/outputs/phase01/core/resid/curiosity/p01_resid_cur_a1_20260630_0407_summary.json`
      reports `real_training_result=true`, `elapsed_seconds=3600.2106466293335`,
      `optimizer_steps=13716`, checkpoint
      `checkpoints/phase01/core/resid/curiosity/p01_resid_cur_a1_20260630_0407.pt`,
      and GPU utilization mean `87.53781512605042%`. This is still not success
      without held-out comparison.
- [ ] If the source gate fails or admits too few cells, record the blocker and
      repair the data/objective before any residual training.
- [x] Attempt 1 forward-model component: first launch
      `p01_fwd_a1_20260630_001144` failed before real
      one-hour training because the Phase 01 transition CSV missed the
      `run_tag` compatibility column expected by the trainer. This is recorded
      as invalid in the attempt ledger and does not count as a negative
      one-hour result. Retry `p01_fwd_a1r1_20260630_001700` then reached
      PyTorch but OOMed before one-hour training because the old trainer
      expanded repeated full-sequence batches. This second invalid run is also
      recorded in the ledger and does not count as a negative one-hour result.
      Trainer has been repaired to use mini-batch sequence training.
      Retry `p01_fwd_a1r2_20260630_002030` completed one-hour H200 forward
      model training with fresh official Newton sanity, checkpoint, and GPU
      utilization pass. Evidence:
      `experiments/outputs/phase01/core/fwd/p01_fwd_a1r2_20260630_002030_summary.json`.
      This is a valid forward-model component only, not a policy update and
      not a positive curiosity result. Negative real one-hour policy-result
      count remains 0.
- [x] Attempt 2 curiosity-weighted residual candidate:
      `p01_resid_cur_a1_20260630_0407` completed real one-hour training,
      held-out evaluation, MP4 export, and strongest-baseline comparison.
      Result: negative evidence, not curiosity success.
      Evidence:
      `experiments/outputs/phase01/core/resid/curiosity/p01_resid_cur_a1_20260630_0407_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_eval_a1r1_20260630_0511_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_cmp_a1_20260630_0518_comparison.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_mp4_a1_20260630_0519_mp4_summary.json`.
      Comparison: `positive_curiosity_result=false`,
      `safety_regression_cell_count=4`, `useful_improvement_count=1`.
      Negative real one-hour stop-gate count: 1/5. Continue with faithful
      repair before any next candidate.
- [x] Attempt 3 safety-anchor curiosity repair:
      `p01_resid_cur_sa_a2_20260630_0521` completed as the second real
      curiosity policy candidate after Attempt 2 failed held-out comparison.
      Repair config:
      `experiments/configs/phase01/resid_curiosity_sa_train.json`.
      Design change: restart from the no-curiosity residual baseline, lower
      curiosity weight and learning rate, and activate baseline-preservation
      anchor with lower contact/phase thresholds.
      Evidence:
      `experiments/outputs/phase01/core/resid/curiosity_sa/p01_resid_cur_sa_a2_20260630_0521_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_sa_eval_a2_20260630_0622_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_sa_cmp_a2_20260630_0630_comparison.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_sa_mp4_a2_20260630_0639_mp4_summary.json`.
      Result: negative evidence, not curiosity success.
      Comparison: `positive_curiosity_result=false`,
      `safety_regression_cell_count=4`, `useful_improvement_count=0`.
      Limitation: the intended preservation anchor did not activate
      (`train_anchor_weight_mean=0.0`, `validation_anchor_weight_mean=0.0`)
      because Phase 01 rows contain `newton.panda.rigid_contact_count` while
      the trainer only checked `newton.contact.rigid_contact_count`.
      Negative real one-hour stop-gate count: 2/5.
- [x] Export real MP4 videos for Attempt 3 held-out evaluation.
      Evidence:
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_sa_mp4_a2_20260630_0639_mp4_summary.json`
      reports `status=pass`, `passed_count=4`, and `failed_count=0`.
- [x] Attempt 4 contact-fallback safety-anchor curiosity repair:
      Prepared config:
      `experiments/configs/phase01/resid_curiosity_sa2_train.json`.
      Trainer repair:
      `experiments/configs/train_curiosity_weighted_residual_adapter_v1.py`
      now falls back from `newton.contact.rigid_contact_count` to
      `newton.panda.rigid_contact_count` for anchor activation and records
      `anchor_contact_count_columns` in the training summary. This attempt
      completed in Curiosity H200 Slurm job `157999` on `server64` with run
      tag `p01_resid_cur_sa2_a3_20260630_0641`; fresh official Newton
      SensorContact sanity passed, real training lasted 3600.0938968658447
      seconds, and mean H200 utilization was 87.19327731092437%.
      Evidence:
      `experiments/outputs/phase01/core/resid/curiosity_sa2/p01_resid_cur_sa2_a3_20260630_0641_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_sa2_eval_a3_20260630_0739_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_sa2_cmp_a3_20260630_0747_comparison.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_sa2_mp4_a3_20260630_0748_mp4_summary.json`.
      Result: negative evidence, not curiosity success.
      The contact fallback worked: `train_anchor_weight_mean=0.33101025223731995`
      and `validation_anchor_weight_mean=0.38709700107574463`. But the
      active neutral anchor damaged validation behavior
      (`validation_loss=4.01609468460083`,
      `validation_active_accuracy=0.5355555415153503`), and held-out
      comparison still reported `positive_curiosity_result=false`,
      `safety_regression_cell_count=4`, `useful_improvement_count=1`.
      Negative real one-hour stop-gate count: 3/5.
- [x] Attempt 5 base-policy distillation/trust-region repair:
      Do not repeat the same neutral-anchor objective. Prepare a fourth real
      curiosity policy candidate that preserves the learned no-curiosity
      residual policy outputs on stable or low-curiosity frames, instead of
      forcing those frames to neutral residual targets. It must use the same
      train-only source rows, source-matched learning-progress scores, locked
      held-out cells, and strongest-baseline comparison contract. If this
      fourth real one-hour policy candidate is also negative, the stop-gate
      count will become 4/5; do not start a sixth real one-hour candidate if
      the count reaches 5/5.
      Progress: config
      `experiments/configs/phase01/resid_curiosity_distill_train.json` is
      prepared and run `p01_resid_cur_distill_a4_20260630_0752` completed in
      Curiosity H200 Slurm job `157999` on `server64`. Fresh official Newton
      SensorContact sanity passed, real training lasted 3600.3141434192657
      seconds, and mean H200 utilization was 89.89830508474576%.
      Evidence:
      `experiments/outputs/phase01/core/resid/curiosity_distill/p01_resid_cur_distill_a4_20260630_0752_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_distill_eval_a4_20260630_0853_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_distill_cmp_a4_20260630_0902_comparison.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_distill_mp4_a4_20260630_0903_mp4_summary.json`.
      Result: negative evidence, not curiosity success.
      The base-policy distillation anchor preserved validation behavior better
      than the neutral anchor (`validation_loss=0.3044162690639496`,
      `validation_active_accuracy=0.9661111235618591`), but held-out
      evaluation succeeded on only 3/4 cells and the contact-only heavy
      cylinder failed with `max_object_accel_m_s2=9.535886263570077`.
      Strongest-baseline comparison reported
      `positive_curiosity_result=false`, `safety_regression_cell_count=4`,
      `useful_improvement_count=2`.
      Negative real one-hour stop-gate count: 4/5.
- [ ] Attempt 6 gate:
      Do not start this as a casual continuation. Only one real one-hour
      negative candidate remains before the mandatory stop/report gate. The
      next action should be a more substantive data/objective repair focused on
      train-only evidence and the observed held-out failure pattern, without
      using held-out rows for training, tuning, source selection, or threshold
      selection. If a fifth real one-hour curiosity policy candidate is
      negative, stop and report to the user before any sixth real training
      attempt.
      Current audit:
      `experiments/reports/phase01/core/resid/p01_four_negative_repair_audit.md`.
      Finding: admitted train-only corrective sources reduce some
      slip/acceleration metrics but consistently trade off hold/lift, matching
      the repeated held-out safety/hold/lift regressions. The final allowed
      attempt must repair the source objective or document a blocker before
      training.
- [x] Run strict final-attempt preflight on the existing corrective source
      before allowing the fifth real one-hour curiosity policy candidate.
      Evidence: `data/processed/phase01/src_strict/manifest.json` and
      `experiments/reports/phase01/core/resid/p01_strict_source_repair_preflight.md`.
      Result: blocked, with `admitted_count=0`, `rejected_count=6`,
      `train_row_count=0`, `validation_row_count=0`, and
      `final_one_hour_attempt_allowed_from_this_preflight=false`. This is a
      data gate, not training, and does not change the stop-gate count.
- [x] Run a gentle train-only strict source collection to test whether a less
      aggressive official scripted correction can provide safer source labels
      without touching held-out cells.
      Evidence: run tag `p01_src_gentle_a1_20260630_0913`,
      `data/processed/phase01/src_gentle/manifest.json`, and
      `experiments/reports/phase01/core/src_gentle_collect.md`.
      Result: failed final-attempt gate. It admitted only
      `train_cylinder_heavy_low`, rejected 7 cells, wrote 1800 train rows and
      0 validation rows, and failed with `admitted_cells_below_min:1` plus
      `no_validation_rows_from_admitted_sources`. This is useful evidence for
      the source blocker, not a trainable final-attempt dataset.
- [x] Repair the source-collection runner boolean handling after discovering
      that JSON `false` was printed as `False` and the shell wrapper treated it
      as enabled.
      Evidence:
      `experiments/configs/phase01/run_src_collect_in_alloc.sh` now normalizes
      boolean-like JSON values to `0`/`1`, and
      `experiments/configs/phase01/launch_src_collect_tmux.sh` creates the
      parent directory for custom `LOG_PATH`.
- [x] Run balanced strict train-only source collection with initial waypoint
      adjustment disabled and the strict final-attempt gate unchanged.
      Evidence: run tag `p01_src_bal_a1r1_20260630_0945`,
      config `experiments/configs/phase01/src_collect_balanced_strict.json`,
      and `data/processed/phase01/src_bal/manifest.json`.
      Result: failed final-attempt gate with `admitted_count=1`,
      `rejected_count=7`, `train_row_count=1800`,
      `validation_row_count=0`, `train_active_feedback_count=29`,
      `validation_active_feedback_count=0`, and failures
      `admitted_cells_below_min:1`,
      `no_validation_rows_from_admitted_sources`.
- [x] Run focused box/light-cylinder strict train-only source collection after
      balanced evidence showed the box cells were close to passing only on
      active feedback frame count.
      Evidence: run tag `p01_src_box_a1_20260630_1006`,
      config `experiments/configs/phase01/src_collect_box_strict.json`, and
      `data/processed/phase01/src_box/manifest.json`.
      Result: failed final-attempt gate with `admitted_count=1`,
      `rejected_count=2`, `train_row_count=1800`,
      `validation_row_count=0`, `train_active_feedback_count=23`,
      `validation_active_feedback_count=0`, and failures
      `admitted_cells_below_min:1`,
      `no_validation_rows_from_admitted_sources`.
- [x] Build a local-advantage segment-mask source repair without using
      held-out cells after source-level gates failed.
      Evidence: config `experiments/configs/phase01/local_adv_segments.json`,
      builder `experiments/configs/phase01/build_local_adv_segments.py`, run
      tag `p01_local_adv_a1_20260630_1024`, and manifest
      `data/processed/phase01/local_adv/manifest.json`.
      Result: passed preflight with 3 train segments, 1 validation segment,
      576 train rows, 192 validation rows, 58 train active feedback labels,
      29 validation active feedback labels, no held-out leakage, and
      `generated_trex_fields=[]`.
- [x] Compute learning-progress curiosity scores for the local-advantage
      segments using the existing real one-hour Newton-native forward-model
      checkpoint pair.
      Evidence:
      `experiments/outputs/phase01/core/local_adv_lp/curiosity_learning_progress_summary.json`.
      Result: status pass, 768 scores, mean learning progress
      `0.7639258076936434`, mean bounded curiosity reward
      `0.5006128434669032`, `policy_updated=false`, and no fake score fields.
- [x] Run a smoke diagnostic for the local-advantage curiosity residual
      trainer before spending the final allowed real one-hour attempt.
      Evidence:
      `experiments/outputs/phase01/core/resid/curiosity_local_adv/p01_resid_cur_local_adv_smoke_a1_20260630_1026_summary.json`.
      Result: status pass, `smoke_diagnostic_only=true`,
      `real_training_result=false`, `checkpoint_written=false`, train and
      validation score coverage 1.0. This does not count toward the five
      real-training stop gate.
- [x] Fifth real curiosity policy candidate:
      `p01_resid_cur_local_adv_a5_20260630_1028` completed in Curiosity H200
      Slurm job `158247` on `server29` from config
      `experiments/configs/phase01/resid_curiosity_local_adv_train.json`.
      Evidence:
      `experiments/outputs/phase01/core/resid/curiosity_local_adv/p01_resid_cur_local_adv_a5_20260630_1028_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_local_adv_eval/p01_resid_cur_local_adv_eval_a5_20260630_1323_summary.json`,
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_local_adv_cmp_a5_20260630_1340_comparison.json`,
      and
      `experiments/outputs/phase01/core/resid/curiosity_local_adv_eval/p01_resid_cur_local_adv_mp4_a5_20260630_1343_mp4_summary.json`.
      Result: negative evidence, not curiosity success. Training lasted
      3600.0268936157227 seconds, GPU mean utilization was
      99.18333333333334%, held-out evaluation succeeded on 4/4 cells, MP4
      export passed 4/4 videos, but strongest-baseline comparison reported
      `positive_curiosity_result=false`, `safety_regression_cell_count=4`,
      and `useful_improvement_count=2`.
- [x] Before any fifth real one-hour curiosity policy training, obtain a
      train-only source/objective repair that passes the final-attempt gate:
      at least two admitted train-only sources, a held-back validation source
      split, active feedback labels in both train and validation, no held-out
      leakage, and paired metrics that do not trade away lift/hold/safety for
      slip or acceleration.
      Evidence: `data/processed/phase01/local_adv/manifest.json` passed via
      local-advantage segment masking.
- [x] Stop gate before attempt 6: reached. Five real one-hour curiosity policy
      candidates are negative. Do not start a sixth real one-hour curiosity
      training attempt without explicit user instruction.

Each counted attempt must be at least one hour of real training, not a smoke
test. Invalid/blocked runs must be recorded but do not count as negative
one-hour attempts.

## Evaluation

- [x] Evaluate any trained curiosity-weighted policy on all 4 held-out Phase 00
      cells.
      Prepared launcher:
      `experiments/configs/phase01/launch_resid_curiosity_eval_tmux.sh`.
      It reuses the parameterized held-out evaluator and writes curiosity
      candidate outputs under `phase01/core/resid/curiosity_eval`.
- [x] Generate real MP4 rollout videos under `experiments/visuals/phase01/core/`.
      Evidence:
      `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_mp4_a1_20260630_0519_mp4_summary.json`
      passed 4/4 MP4 exports, each with 601 encoded frames at 20 FPS.
- [x] Compare against strongest declared baseline with safety metrics.
      Prepared comparison builder:
      `experiments/configs/phase01/build_resid_curiosity_comparison.py`,
      `experiments/configs/phase01/run_resid_curiosity_compare_in_alloc.sh`,
      and
      `experiments/configs/phase01/launch_resid_curiosity_compare_tmux.sh`.
      The comparison must run after held-out curiosity evaluation and must
      classify `positive_curiosity_result=false` unless the candidate beats the
      strongest baseline set without safety regression.
- [ ] Classify result as positive, negative, incomplete, invalid, or blocked.

## Current Status

Phase 01 has produced one valid one-hour forward-model component, one valid
one-hour no-curiosity residual baseline component, source-matched
learning-progress scores for the gated corrective source rows, complete
held-out non-curiosity baseline metrics with MP4 evidence, and four real
one-hour curiosity-weighted residual candidates. All four curiosity candidates
are negative evidence: they succeeded on all four held-out cells but failed the
strongest-baseline safety/performance comparison, except the fourth
distillation candidate succeeded on only 3/4 held-out cells. The second
safety-anchor candidate revealed that the intended preservation anchor was
inactive due to a contact-key mismatch. The third candidate fixed that mismatch
and proved the anchor can activate, but the neutral-anchor objective damaged
validation behavior and still regressed held-out safety metrics. The fourth
candidate used base-policy distillation and restored validation behavior, but
failed the contact-only heavy cylinder held-out cell and still regressed safety
against the strongest baseline set. There is no curiosity success claim.
Stop-gate count is 5/5. The source-quality blocker was repaired by
local-advantage segment masking, not by weakening the source-level safety
contract, but the fifth real candidate
`p01_resid_cur_local_adv_a5_20260630_1028` was still negative after held-out
comparison. There is no curiosity success claim. Do not run a sixth real
one-hour training attempt without explicit user instruction.
