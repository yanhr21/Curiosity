# Phase 04 TODO: Closed-Loop Adaptation

- [x] Run no-adaptation policy on nominal cup.
      Evidence: Phase 02 nominal cup baseline.
- [x] Run no-adaptation policy across mass/fill variants.
      Evidence: Phase 02 3x3 mass/friction grid.
- [x] Use the user-approved short-term stable route: start adaptation from the
      official Newton Panda hydro scripted infant prior, not from an unverified
      pretrained checkpoint.
      Evidence: scripted feedback baseline config and controller mode.
- [x] Run scripted adaptive baseline across mass/fill variants.
      Nominal cup gate complete:
      `lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545`.
      Ordinary grid cells complete: `empty_low`, `empty_medium`, `half_low`,
      `half_medium`, `half_high`, `full_medium`, `full_high`.
      Held-out `full_low` evaluation complete:
      `lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845`.
      This run passed fresh official Newton sanity, camera export, visual
      validation, manual visual inspection, lift, hold, slip, drop,
      contact-loss, and contact-proxy gates. Full metrics still failed only on
      `object_accel_above_threshold` with
      `max_object_accel_m_s2=8.308707788010144`. Feedback did not trigger
      (`feedback_trigger_count=0`).
      Held-out `empty_high` evaluation complete:
      `lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955`.
      This run passed fresh official Newton sanity, camera export, visual
      validation, manual visual inspection, lift, hold, slip, drop,
      contact-loss, and contact-proxy gates. Full metrics still failed only on
      `object_accel_above_threshold` with
      `max_object_accel_m_s2=8.308498000056417`. Feedback did not trigger
      (`feedback_trigger_count=0`). Scripted feedback evaluation is complete,
      but no adaptation-improvement claim is valid because feedback never
      triggered.
- [x] Run contact-aware curiosity diagnostic across mass/fill variants.
      Evidence: Phase 03 replay diagnostic
      `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`
      has `status=pass`, `rollout_count=9`, and rows for the full 3x3
      mass/friction grid, including held-out `full_low` and `empty_high`.
      This is diagnostic replay only: no model training, no policy update, no
      placeholder T-Rex/VQ-VAE/world model, and tactile source is
      `newton.contact_proxy_only`.
- [ ] Train the first learned adapter as residual controller-parameter output,
      not full low-level torque control.
      Current status: blocked, training not started. One promoted source
      candidate now exists, but the formal residual-label source runner and
      adapter-training runner do not exist yet.
      Evidence:
      `experiments/configs/residual_adapter_training_readiness_v1.json` and
      `experiments/reports/2026-06-27_phase04_residual_adapter_training_readiness_v1.md`.
      Blocker: training now would skip source-runner review, held-out split
      enforcement, and additional ordinary-cell collection. Next step is the
      runner/source-gate path, not direct learned-adapter training.
- [x] Run first ordinary-cell diagnostic to verify whether the official Newton
      path can emit nonzero residual controller-parameter labels.
      Evidence:
      `experiments/reports/2026-06-27_phase04_residual_label_source_sensitive_feedback_half_low.md`.
      Run `residual_label_source_sensitive_feedback_half_low_20260627_030145`
      used ordinary `half_low`, passed official Newton sanity and visual
      validation, produced final `feedback_trigger_count=241`, and logged
      nonzero `candidate.controller.*` residual fields. It is not promoted as a
      training-label source because standard metrics failed on
      `hold_duration_below_threshold` and `object_accel_above_threshold`.
- [x] Run bounded residual-label threshold sweep on ordinary cells only.
      Goal: keep `feedback_trigger_count > 0` while recovering lift, hold,
      drop, contact-loss, visual, and sanity gates. Do not use held-out
      `full_low` or `empty_high` for label collection.
      Current sweep status: first promoted source candidate found.
      `contact64` and `contact58` produce nonzero labels but fail hold;
      `accel_threshold_5p5` preserves lift/hold/drop/contact behavior but has
      `feedback_trigger_count=0`; `contact58_gentle` produces nonzero labels
      and preserves lift/hold/drop/contact/visual/manual gates, but strict
      metrics still fail on `object_accel_above_threshold`.
      Latest diagnostic:
      `residual_label_sweep_half_low_contact58_20260627_0310` on ordinary
      `half_low` passed fresh official Newton sanity, automated visual
      validation, and manual nonblank inspection, and produced
      `feedback_trigger_count=241`. It is not promoted because the longest
      hold remained `0.9833333333333333` s, below the formal 2s gate.
      Evidence:
      `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58.md`.
      Follow-up:
      `residual_label_sweep_half_low_contact58_gentle_20260627_0345` preserved
      nonzero residual labels and passed lift/hold/drop/contact/visual gates,
      but strict metrics still failed only on `object_accel_above_threshold`
      with `max_object_accel_m_s2=8.308707788010144`. Evidence:
      `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle.md`.
      Second follow-up:
      `residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355`
      increased initial lift duration scale to `1.8`, preserved nonzero
      residual labels and task gates, but still failed strict metrics on
      `object_accel_above_threshold` with
      `max_object_accel_m_s2=8.308972018193668`. Evidence:
      `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle_smooth.md`.
      Peak-analysis follow-up showed the repeated acceleration value was a
      recorded initial settling artifact: the non-warmup top event occurred at
      step 2, phase 0, before feedback was active. Warmup source candidate:
      `residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006`
      uses `PRE_RECORD_WARMUP_STEPS=15`, preserves nonzero feedback labels,
      and passes strict metrics with `max_object_accel_m_s2=0.5063306543767194`,
      `hold_duration_s=2.5333309173583984`, and
      `feedback_trigger_count=241`. Evidence:
      `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle_lift165_warmup15.md`.
      Decision record updated:
      `experiments/reports/2026-06-27_phase04_object_acceleration_gate_decision.md`.
- [x] Build the formal residual-label source runner and source-gate checks.
      Input manifest:
      `experiments/configs/residual_label_source_manifest_v1.json`.
      Requirements: run only in the held tmux allocation or later equivalent
      compute allocation, activate a prebuilt local `envs/` venv, rerun fresh
      official Newton sanity, enforce ordinary-cell source collection, exclude
      held-out `full_low` and `empty_high`, record commands/logs/configs, and
      keep `generated_trex_fields=[]` with `schema_promotion=blocked`.
      This is still not learned-adapter training.
      Evidence:
      `experiments/reports/2026-06-27_phase04_residual_label_source_runner_v1.md`.
      Final runner `residual_label_source_runner_v1_20260627_0401` ran inside
      tmux-held allocation `154142`, passed fresh official Newton sanity, and
      generated
      `data/processed/residual_label_source_runner_v1_20260627/manifest.json`
      with `status=pass`, `source_run_count=3`, `record_count=1080`,
      `total_feedback_trigger_count=722`, `failures=[]`,
      `generated_trex_fields=[]`, `schema_promotion=blocked`, and
      `training_started=false`.
- [ ] Collect additional ordinary-cell residual-label source candidates after
      the runner gates are in place.
      Progress: ordinary `empty_low` and `half_medium` were collected after
      the runner gate and both passed fresh official Newton sanity, automated
      visual validation, manual visual inspection, strict metrics, and peak
      analysis. They were added to
      `experiments/configs/residual_label_source_manifest_v1.json` and the
      source runner passed with all three sources. Continue with ordinary
      cells such as `full_high`; keep held-out cells evaluation-only.
- [x] Plan nonzero residual correction data collection before learned-adapter
      training.
      Evidence:
      `experiments/configs/residual_correction_collection_plan_v1.json` and
      `experiments/reports/2026-06-27_phase04_residual_correction_collection_plan_v1.md`.
      The plan now includes the first promoted warmup15 source candidate and
      the source-runner route while preserving held-out `full_low` and
      `empty_high`.
- [x] Reserve held-out mass/friction cells for generalization evaluation.
      Evidence: Phase 02/04 configs preserve `full_low` and `empty_high` as
      held-out cells.
- [ ] Compare adaptation speed and failure modes.
      Waiting on a valid learned residual-adapter run. Current scripted
      feedback and curiosity/contact replay diagnostics are baselines only.
- [x] Save direct visual paths for success and failure cases.
      Evidence: nominal scripted feedback report records contact sheet and
      frame browser paths; grid reports still pending.
