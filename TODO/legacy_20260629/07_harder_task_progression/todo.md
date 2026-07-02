# Phase 07 TODO: Harder Task Progression

- [ ] Enforce the no-early-exit harder-training gate.
      Phase 03 V1 is only an intermediate pipeline result. Do not mark the
      overall curiosity objective complete until a harder-task closed-loop
      curiosity policy beats the declared baseline and the comparison against
      serious/mainstream methods is either passed or explicitly blocked by
      faithful incompatibility. Sparse contact sheets or smoke tests are not
      sufficient.
- [ ] Keep the harder-training persistence lock active in project memory.
      The requirement is now written into `IDEA/idea.md`, `AGENTS.md`,
      `PLAN/07_harder_task_progression/plan.md`, and this TODO. Future agents
      must not downgrade the objective into an easy rerun, toy model, smoke
      test, offline score, checkpoint-exists claim, single-video result, or
      negative result relabeled as completion. If the hard contract is not
      satisfied, continue faithful training/evaluation, objective repair,
      stronger data collection, ablations, baseline audit, official-method
      setup, or record a concrete blocker.
      User reaffirmation 2026-06-28: keep this lock visible specifically to
      prevent another downgrade or fast exit. Do not call Phase 07 complete
      because an ablation queue finished, a checkpoint exists, a video rendered,
      or curiosity produced negative evidence. Completion requires closed-loop
      curiosity to beat the strongest declared baseline on harder held-out
      tasks without safety regression, plus faithful serious-method comparison
      or documented blocker evidence.
      Evidence challenge 2026-06-28: any completion claim must cite the exact
      training command/log, checkpoint, official sanity check, held-out
      harder-task metrics, strongest-baseline comparison, safety comparison,
      full rollout videos, manual visual inspection, and faithful
      serious-method comparison or blocker. If the evidence is diagnostic,
      validation-only, stale, or negative, keep this TODO open and continue.
- [ ] Enforce the latest 2026-06-28 no-downgrade/no-fast-exit lock.
      The harder-training requirement is an active execution gate, not a note.
      Do not simplify the objective into an easier task, toy substitute, smoke
      test, offline score, validation-only threshold repair, queue/script
      completion, checkpoint-exists claim, single-video demo, or negative
      result relabeled as success. Any completion claim must pass harder
      closed-loop held-out evaluation, beat the strongest declared baseline
      without safety regression, include full videos and strict metrics, cover
      required ablations, and include faithful serious-method comparison or a
      documented blocker. If that evidence is missing or negative, keep this
      TODO open and continue with faithful repair, stronger data collection,
      training/evaluation, ablation, baseline audit, official-method setup, or
      a concrete blocker.
- [ ] Write a hard-training contract before the next Phase 07 training run.
      The contract must name the exact harder task cells, train/validation/
      held-out split, training/adaptation budget, baseline policies, ablations,
      success metrics, safety metrics, checkpoint/log/output paths, and required
      full-rollout videos. Do not start a success-claiming run without this
      contract.
      Progress 2026-06-27: contract written at
      `experiments/configs/phase07_hard_training_contract_v1.json`. It binds
      the variable water-cup task, six train cells, two validation cells, three
      held-out cells, required baselines/ablations, full-video evidence,
      one-GPU-hour training minimums, and forbidden success claims.
- [ ] Prevent downgrade and quick-exit interpretations during training.
      Do not convert the requested hard closed-loop curiosity run into the
      original easy cup benchmark, a one-cell demonstration, offline replay
      scoring, supervised-only reweighting, a smoke test, sparse contact-sheet
      inspection, or a small homemade replacement model. If any of those are
      run, label them as diagnostic/preflight/source collection only and keep
      the objective open.
- [ ] Continue after weak or negative curiosity results instead of exiting.
      A checkpoint, curiosity score file, one-hour training log, one rendered
      video, or negative held-out comparison does not complete the objective.
      If curiosity fails to beat the strongest declared baseline or introduces
      safety regressions, continue with the next faithful step: closed-loop
      objective repair, data collection, baseline audit, required ablations,
      full metrics/video inspection, or documented blocker. Do not re-label
      this as success.
      User reaffirmation 2026-06-27: do not repeat downgrade or quick-exit
      behavior. Keep this TODO open until the harder-task curiosity result
      actually passes the declared comparison, including serious/mainstream
      method comparison or a faithful documented blocker.
      Progress 2026-06-28: after the clean refreshed gate still failed,
      created the validation-only closed-loop threshold-repair config and
      runners:
      `experiments/configs/phase07_closed_loop_threshold_repair_v1.json`,
      `experiments/configs/run_phase07_closed_loop_threshold_repair_in_alloc.sh`,
      and
      `experiments/configs/launch_phase07_closed_loop_threshold_repair_tmux.sh`.
      This runner will use current-policy Newton interaction rollouts on
      validation cells only, with full 360-frame videos, to select a safer
      active threshold before any later held-out re-evaluation. Held-out cells
      are explicitly forbidden for threshold selection. Slurm GPU job `155734`
      was submitted in tmux session
      `curiosity_phase07_threshold_repair_alloc_20260628` and is pending.
      Progress 2026-06-28 later: job `155734` ran on `server37` and completed
      the validation sweep with eight full 360-frame videos plus an initial
      held-out threshold-repaired evaluation. That initial held-out run used
      selected threshold `0.8` and remained `open_not_satisfied`; it did not
      beat no-adaptation on any held-out cell. A selector audit found the
      implementation ranked lift before acceleration despite the safety-first
      config, so the scripts were patched. The `0.8` held-out run is
      diagnostic only. Corrected safety-first threshold `0.65` held-out
      re-evaluation ran as Slurm job `155742` in tmux session
      `curiosity_phase07_threshold065_eval_alloc_20260628`, then the allocation
      was released after the retry completed. The corrected summary is
      `experiments/outputs/phase07_threshold065_heldout_eval_retry_v1_20260628_summary.json`:
      status remains `open_not_satisfied`, with all three held-out cells
      failing to beat `no_adaptation`.
- [ ] Classify every harder-task result before using completion language.
      Mark the result incomplete if it lacks closed-loop curiosity updates,
      harder held-out evaluation, full videos, strict safety metrics, required
      ablations, or faithful serious/mainstream method comparison. Mark it as
      negative evidence if curiosity does not beat the strongest declared
      baseline without safety regression. In either case, the next TODO action
      must be continued faithful work or a documented blocker, not downgrade or
      quick exit.
- [ ] Run the Phase07 hard-training evidence gate after every ablation queue.
      Evidence gate script:
      `experiments/configs/audit_phase07_hard_training_evidence_gate_v1.py`.
      It reads training summaries, held-out metrics, full-video evidence,
      manual/automated visual checks, candidate action-bridge fields,
      curiosity-vs-baseline comparison, and mainstream comparison status. It
      must report `final_curiosity_success_allowed=false` until the full
      hard-training contract passes. This gate is attached to the allocation
      queue and should be rerun through a compute-side allocation refresh, not
      directly on the login node, to remain consistent with the current
      highest-priority cluster rules.
      Progress 2026-06-28: added compute-allocation-only refresh scripts
      `experiments/configs/run_phase07_evidence_refresh_in_alloc.sh` and
      `experiments/configs/launch_phase07_evidence_refresh_tmux.sh`, then ran
      them in Slurm job `155732` on `server13` via tmux session
      `curiosity_phase07_evidence_refresh_alloc_20260628`. Log:
      `logs/newton/phase07_evidence_refresh_v1_20260628.log`; summary:
      `experiments/outputs/phase07_evidence_refresh_v1_20260628_summary.json`.
      The refresh exited with `TMUX_PHASE07_EVIDENCE_REFRESH_EXIT=0`, then the
      allocation was released. The refreshed hard gate now has
      `training_evidence_status=pass`, `evaluation_evidence_status=pass`, and
      `final_curiosity_success_allowed=false`. This removes the stale
      no-learning-progress manual-visual gap but keeps the objective open.
- [ ] Backfill action-bridge fields for every existing Phase07 source/eval NPZ.
      The allocation-only backfill must cover train/validation source rollouts
      plus no-adaptation, scripted feedback, residual baseline,
      curiosity-weighted, random-intrinsic, and object-only held-out NPZs, not
      only the first baseline/candidate pair. The evidence gate may accept a
      provenance-preserving backfilled NPZ as bridge evidence for older
      evaluations, while newly generated evaluations should contain
      `candidate.action.*` directly from the exporter.
- [ ] Build allocation-only mainstream stage-1 dataset indices.
      Script:
      `experiments/configs/build_phase07_mainstream_stage1_dataset_index_v1.py`.
      Runner:
      `experiments/configs/run_phase07_mainstream_stage1_dataset_index_in_alloc.sh`.
      This must run only inside a held Slurm allocation after action-bridge
      backfill. It writes method-specific index/config files for OpenPI
      LeRobot, GR00T LeRobot v2/modality, Diffusion Policy shape_meta, and
      RT-X RGB/task/7D-action mapping. It is not full dataset conversion, not
      training, not inference, and not a mainstream success claim.
- [ ] Pass stage-1 no-held-out-leakage audit before official conversion use.
      Script:
      `experiments/configs/audit_phase07_stage1_no_heldout_leakage_v1.py`.
      Allocation runner:
      `experiments/configs/run_phase07_stage1_no_heldout_leakage_in_alloc.sh`.
      It verifies that held-out cells remain `held_out_eval_only` and
      training-forbidden in the main stage-1 index and all method-specific
      OpenPI/GR00T/Diffusion Policy/RT-X indices. The allocation queue requires
      this audit to pass before continuing. Stage-1 builder must also write
      explicit split files for train, validation, and held-out eval-only so
      later official dataset materializers do not consume the combined index
      for training or normalization by mistake.
- [ ] Pass official-method readiness before any mainstream success claim.
      Script:
      `experiments/configs/audit_phase07_official_method_readiness_v1.py`.
      Allocation runner:
      `experiments/configs/run_phase07_official_method_readiness_in_alloc.sh`.
      It checks official repo checkout, prepared environment under `envs/`,
      official checkpoint or documented faithful checkpoint blocker, stage-1
      dataset index, and closed-loop Phase07 comparison runner for OpenPI,
      GR00T, Diffusion Policy, and RT-X. A repo clone, bridge spec, or
      diagnostic adapter cannot satisfy this gate. The Phase07 remaining
      ablation queue reruns this readiness audit after stage-1 indexing and
      before the hard evidence gate, so the hard gate cannot read stale
      readiness status.
- [ ] Prepare official-method environments and checkpoint/blocker records.
      Plan:
      `experiments/configs/phase07_official_method_env_checkpoint_plan_v1.json`.
      Local dry-run/default setup script:
      `experiments/configs/prepare_phase07_official_method_envs_local.sh`.
      Checkpoint blocker template script:
      `experiments/configs/build_phase07_official_checkpoint_blocker_templates_v1.py`.
      Checkpoint access probe:
      `experiments/configs/audit_phase07_official_checkpoint_access_v1.py`.
      Environments must be created under `envs/` on the shared filesystem
      before compute use; do not install dependencies on compute nodes.
      Official checkpoints must be obtained into an approved checkpoint/cache
      path or recorded as faithful access/incompatibility blockers before any
      mainstream comparison claim.
- [ ] Implement guarded official-method closed-loop comparison runners.
      Current runner gates:
      `experiments/configs/run_phase07_openpi_official_comparison_in_alloc.sh`,
      `experiments/configs/run_phase07_gr00t_official_comparison_in_alloc.sh`,
      `experiments/configs/run_phase07_diffusion_policy_official_comparison_in_alloc.sh`,
      and `experiments/configs/run_phase07_rtx_official_comparison_in_alloc.sh`.
      These are not toy evaluators and not success claims. They refuse to run
      without a Slurm allocation, official env, checkpoint or filled blocker,
      stage-1 method index, and no-held-out-leakage proof. After those gates
      pass, method-specific official inference/fine-tune code still has to be
      implemented faithfully.
- [ ] Build held-out comparison report after every Phase07 queue.
      Script:
      `experiments/configs/build_phase07_heldout_comparison_report_v1.py`.
      It aggregates held-out metrics, video paths, visual inspection status,
      missing ablations, strongest baseline selection, and curiosity safety
      regressions into
      `experiments/outputs/phase07_heldout_comparison_report_v1_20260627.json`
      plus a markdown report. This report is evidence only; it does not train,
      render, infer, or claim success.
      Progress 2026-06-28: evidence refresh reran the report after manual
      visual JSONs existed for contact-only, shuffled-contact, delayed-contact,
      and no-learning-progress. The report now has
      `missing_or_failed_entry_count=0`, but
      `curiosity_beats_all_strongest_baselines_without_safety_regression=false`;
      therefore the comparison is cleanly negative rather than incomplete due
      to stale missing visual records.
- [x] Define the variable water-cup weight/fill task family.
      Include empty, quarter, half, three-quarter, and full mass settings;
      low/medium/high friction; visual fill cues; pose randomization; and
      held-out combinations. Record the task spec under `experiments/configs/`
      or `docs/`.
      Evidence: `experiments/configs/variable_water_cup_harder_task_v1.json`
      defines five fill/mass levels, three friction levels, truthful/hidden/
      misleading visual-cue conditions, train/validation/held-out splits,
      required policies, required metrics, and no-early-exit completion gates.
- [x] Add complete rollout video export for harder-task evaluations.
      Final harder-task evidence must include full visual rollout videos or a
      dense-frame video-equivalent artifact, in addition to sampled frame
      browsers and contact sheets. Record video paths in summaries and reports.
      Evidence: `experiments/configs/newton_panda_hydro_tiled_camera_export.py`
      now accepts `--video-frame-stride` and `--video-fps` and writes
      `rollout_video.gif` plus dense `video_frames/` when enabled. The tmux
      launcher and in-allocation runner pass `VIDEO_FRAME_STRIDE` and
      `VIDEO_FPS`. Syntax checks pass. Compute-side Phase 07 runs produced
      360-frame GIFs and manual visual inspections:
      `experiments/visuals/phase07_watercup_three_quarter_low_residual_baseline_video_rerun_20260627/rollout_video.gif`,
      `experiments/outputs/phase07_watercup_three_quarter_low_residual_baseline_video_rerun_20260627_manual_visual_inspection.json`,
      `experiments/visuals/phase07_watercup_three_quarter_low_curiosity_weighted_video_20260627/rollout_video.gif`,
      and
      `experiments/outputs/phase07_watercup_three_quarter_low_curiosity_weighted_video_20260627_manual_visual_inspection.json`.
- [ ] Build Newton-native source generation for harder water-cup variants.
      Use the official Newton scripted infant prior plus approved adapters.
      Run only inside tmux-held compute allocation, with fresh official Newton
      sanity, camera export, visual validation, manual visual inspection, and
      strict metrics. Keep outputs under `newton.*` and `candidate.*`
      namespaces. Include train/validation cells and held-out cells; held-out
      cells must not affect training, labels, thresholds, or hyperparameters.
      Progress: first harder physical cell `three_quarter_low` ran with
      `object_mass_kg=0.29`, `object_friction_mu=0.35`, fresh official Newton
      sanity, automated visual validation, manual visual inspection, strict
      lift-hold metrics, and full rollout videos for residual baseline and
      curiosity-weighted policy. Report:
      `experiments/reports/2026-06-27_phase07_watercup_video_baseline_curiosity_first_cell.md`.
      This item remains incomplete until the full variable water-cup source
      set, including visual fill-cue conditions and split coverage, is built.
      Progress 2026-06-27: Phase07 train-source collection now has four
      promoted scripted-feedback source candidates:
      `quarter_low_truthful`, `quarter_medium_hidden`, `half_low_hidden`, and
      `half_medium_truthful`. Each run used fresh official Newton sanity,
      real Newton cup mass/friction adapter settings, automated visual
      validation, direct manual visual inspection, strict metrics, accel peak
      analysis, nonzero feedback labels, and a 360-frame rollout GIF. The
      Phase07 source runner passed with `source_run_count=4`,
      `record_count=1440`, `total_feedback_active_frames=964`, and
      `failures=[]` at
      `data/processed/phase07_residual_label_source_runner_v1_20260627/manifest.json`.
      This is still incomplete because 4 required train/validation source
      cells remain, visual fill cues are currently metadata-only rather than
      rendered, and no closed-loop curiosity training has started from this
      Phase07 source set.
      Progress 2026-06-27 update: all eight planned train/validation
      source candidates are now promoted:
      `quarter_low_truthful`, `quarter_medium_hidden`, `half_low_hidden`,
      `half_medium_truthful`, `three_quarter_medium_misleading`,
      `three_quarter_high_truthful`, `empty_medium_hidden`, and
      `full_medium_misleading`. The Phase07 source runner passed with
      `source_run_count=8`, `record_count=2880`,
      `total_feedback_active_frames=1927`, and `failures=[]` at
      `data/processed/phase07_residual_label_source_runner_v1_20260627/manifest.json`.
      This source-data gate is complete for train/validation rows, but the
      TODO remains open because visual fill cues are still metadata-only and
      no closed-loop curiosity training has started.
- [ ] Establish baselines for variable water-cup weight/fill.
      Required baselines: no-adaptation scripted prior, scripted feedback, and
      trained residual adapter without curiosity. Add serious/mainstream
      reference method or official checkpoint comparison when a faithful
      compatible implementation exists; otherwise write a blocker/audit. Do
      not claim curiosity improvement until these are complete.
      Progress: first `three_quarter_low` physical cell now has
      no-adaptation, scripted-feedback, residual-adapter-without-curiosity,
      and curiosity-weighted 360-frame videos with manual visual inspections.
      Report:
      `experiments/reports/2026-06-27_phase07_watercup_video_baseline_curiosity_first_cell.md`.
      This item remains incomplete until these baselines cover the required
      Phase 07 train/validation/held-out variable water-cup cells and the
      serious/mainstream reference comparison is audited or run.
- [ ] Train curiosity in a complete closed loop on the variable water-cup task.
      The next result must go beyond offline replay scoring and one-shot
      supervised reweighting. It must update the policy/adaptation behavior
      from curiosity-driven interaction evidence under a documented training
      loop, with fresh sanity checks, held allocation, GPU utilization records,
      checkpoints, logs, and no held-out leakage. If using a residual adapter,
      record exactly how curiosity changes data collection, weighting,
      exploration, curriculum, or policy update decisions.
      Progress: the existing Phase 03 curiosity-weighted residual checkpoint
      was evaluated with a full video on the first `three_quarter_low` harder
      physical cell. This is not enough: it reuses the earlier supervised
      curiosity-weighted checkpoint and is not a new complete closed-loop
      Phase 07 curiosity training run.
      Progress 2026-06-27: Phase07 source data collection/preflight is now
      being rebuilt from harder-task interaction data instead of reusing the
      Phase03 easy-grid runner. Four train cells have passed source gates and
      a 1440-row source-runner manifest exists, but this is still
      pre-training data collection, not the required closed-loop curiosity
      policy update.
      Progress 2026-06-27 update: Phase07 source collection reached 8/8
      train/validation sources and the source runner passed with 2880 rows.
      Phase07 residual-adapter preflight passed with 2160 train rows and 720
      validation rows at
      `data/processed/phase07_residual_adapter_training_preflight_v1_20260627/manifest.json`.
      A real one-GPU-hour Phase07 no-curiosity residual baseline training run
      has started as `phase07_residual_adapter_v1_train_20260627` using
      `experiments/configs/phase07_residual_adapter_trainer_v1.json`. This is
      baseline training, not curiosity success.
      Progress 2026-06-27 later update: the Phase07 no-curiosity residual
      baseline training completed with `status=pass`, `real_training_result=true`,
      checkpoint
      `checkpoints/phase07_residual_adapter_trainer_v1_20260627/phase07_residual_adapter_v1_train_20260627.pt`,
      `optimizer_steps=21604`, validation loss `0.012411288917064667`, and
      GPU utilization status `pass` with mean `98.90833333333333%`.
      Phase07 curiosity-forward preflight then passed with 2872 transition
      records, and the real one-GPU-hour Phase07 curiosity forward-model
      training run `phase07_curiosity_forward_model_v1_train_20260627` has
      started from
      `experiments/configs/phase07_curiosity_forward_model_trainer_v1.json`.
      Progress 2026-06-27 later update: Phase07 curiosity forward-model
      training completed with `status=pass`, `real_training_result=true`,
      checkpoint
      `checkpoints/phase07_curiosity_forward_model_v1_20260627/phase07_curiosity_forward_model_v1_train_20260627.pt`,
      initial snapshot
      `checkpoints/phase07_curiosity_forward_model_v1_20260627/phase07_curiosity_forward_model_v1_train_20260627_initial_snapshot.pt`,
      `optimizer_steps=20915`, validation loss `0.2959863543510437`, and
      GPU utilization status `pass` with mean `98.975%`. Phase07
      learning-progress scoring then passed with 2872 scores, mean learning
      progress `0.7043353064857361`, and
      `not_raw_prediction_error_only=true`. The real one-GPU-hour
      curiosity-weighted residual training run
      `phase07_curiosity_weighted_residual_adapter_v1_train_20260627` has
      started from
      `experiments/configs/phase07_curiosity_weighted_residual_adapter_trainer_v1.json`.
      Progress 2026-06-27 held-out update: curiosity-weighted residual
      training completed with `status=pass`, `real_training_result=true`,
      checkpoint
      `checkpoints/phase07_curiosity_weighted_residual_adapter_trainer_v1_20260627/phase07_curiosity_weighted_residual_adapter_v1_train_20260627.pt`,
      `optimizer_steps=21618`, `train_score_coverage=0.9972222222222222`,
      validation loss `0.015233036130666733`, and GPU utilization status
      `pass` with mean `98.8%`. Three held-out cells were evaluated with
      four policies and full 360-frame videos:
      `empty_high_misleading`, `full_low_hidden`, and
      `three_quarter_low_misleading`. Result: incomplete/negative. The
      curiosity-weighted checkpoint does not beat the strongest baseline,
      because no-adaptation has higher lift and longer hold on all three
      held-out cells. Report:
      `experiments/reports/2026-06-27_phase07_heldout_curiosity_weighted_eval_v1.md`.
      Progress 2026-06-27 evidence update: manual visual inspection JSONs,
      standard metrics JSON/CSV, and acceleration peak analysis JSONs now pass
      for all 12 held-out policy videos. This strengthens the negative result
      rather than completing the objective: curiosity-weighted residual still
      fails to beat no-adaptation on lift/hold and has the highest acceleration
      peak on `full_low_hidden`.
- [ ] Run curiosity ablations on the harder task.
      Compare against residual adapter without curiosity, random intrinsic
      reward, object-only curiosity, contact-only curiosity, vision/contact
      curiosity, shuffled contact, delayed contact, and no learning-progress
      term. Do not use a missing ablation as permission to claim success.
      Progress 2026-06-27: generated six ablation score CSV variants from real
      Phase07 Newton transition records under
      `experiments/outputs/phase07_curiosity_ablation_scores_v1_20260627/`
      with fresh official Newton sanity pass and `policy_updated=false`.
      The `random_intrinsic` ablation completed a real one-GPU-hour training
      run `phase07_random_intrinsic_residual_adapter_v1_train_20260627` with
      checkpoint
      `checkpoints/phase07_random_intrinsic_residual_adapter_trainer_v1_20260627/phase07_random_intrinsic_residual_adapter_v1_train_20260627.pt`,
      `optimizer_steps=21642`, mean GPU utilization `98.88333333333334%`,
      and held-out evaluation on `empty_high_misleading`, `full_low_hidden`,
      and `three_quarter_low_misleading` with 360-frame videos, manual visual
      JSONs, metrics, and acceleration analysis. It does not beat the
      no-adaptation baseline. Remaining ablations still to train/evaluate:
      object-only, contact-only, shuffled contact, delayed contact, and no
      learning-progress.
      Progress 2026-06-27 later update: the `object_only` ablation completed
      real one-GPU-hour training run
      `phase07_object_only_residual_adapter_v1_train_20260627` with checkpoint
      `checkpoints/phase07_object_only_residual_adapter_trainer_v1_20260627/phase07_object_only_residual_adapter_v1_train_20260627.pt`,
      `optimizer_steps=21634`, mean GPU utilization `98.91666666666667%`, and
      held-out evaluation on all three configured held-out cells with
      360-frame videos, manual visual JSONs, metrics, and acceleration
      analysis. It also does not beat no-adaptation. Remaining ablations still
      to train/evaluate: contact-only, shuffled contact, delayed contact, and
      no learning-progress.
      Progress 2026-06-27 contact-only attempt: contact-only config was added
      at `experiments/configs/phase07_contact_only_residual_adapter_trainer_v1.json`.
      The first training attempt
      `phase07_contact_only_residual_adapter_v1_train_20260627` passed fresh
      official Newton sanity but failed before a valid training result with
      CUDA OOM because allocation `154290` was occupied by an OpenPI process
      using about 129393 MiB. This is a resource conflict, not a model result.
      It is recorded in
      `experiments/outputs/phase07_contact_only_residual_adapter_v1_train_20260627_resource_conflict.json`.
      A new Curiosity-dedicated allocation was requested in tmux session
      `curiosity_contact_ablation_alloc_20260627_222339`, Slurm job `155039`,
      and is now running on `server07`.
      Live 2026-06-27 23:54 CST: the remaining-ablation queue passed fresh
      official Newton sensor/contact sanity, action-bridge backfill,
      mainstream adapter conversion preflight, stage-1 dataset indexing, and
      no-held-out-leakage audit, then entered
      `phase07_contact_only_residual_adapter_v1_train_retry_20260627`.
      GPU utilization is 99-100% after startup. No valid retry
      summary/checkpoint exists yet, so this is in-progress evidence only and
      cannot be called completed curiosity training.
      Progress 2026-06-28 contact-only completion: retry training completed
      with summary
      `experiments/outputs/phase07_contact_only_residual_adapter_trainer_v1_20260627/phase07_contact_only_residual_adapter_v1_train_retry_20260627_summary.json`,
      checkpoint
      `checkpoints/phase07_contact_only_residual_adapter_trainer_v1_20260627/phase07_contact_only_residual_adapter_v1_train_retry_20260627.pt`,
      `elapsed_seconds=3600.155266523361`, `optimizer_steps=21494`,
      `real_training_result=true`, `checkpoint_written=true`, and mean GPU
      utilization `98.80165289256199%`. Held-out evals for
      `empty_high_misleading`, `full_low_hidden`, and
      `three_quarter_low_misleading` all wrote passing metrics, 360-frame
      videos, acceleration analysis, and candidate action-bridge validation.
      Manual visual inspection remains `pending_direct_agent_check`, and this
      ablation does not prove curiosity beats `no_adaptation`; keep the hard
      objective open.
      Progress 2026-06-28 contact-only manual visual update: inspected the
      three contact-only held-out contact sheets directly. They are nonblank
      three-camera triptychs with visible gripper/object, start/middle/final
      frames, and no obvious drop or render failure. Added manual inspection
      JSONs for all three contact-only held-out cells; each is marked
      `curiosity_success_claim_valid=false`.
      Progress 2026-06-28 shuffled-contact completion: real one-GPU-hour
      training completed with summary
      `experiments/outputs/phase07_shuffled_contact_residual_adapter_trainer_v1_20260627/phase07_shuffled_contact_residual_adapter_v1_train_20260627_summary.json`,
      checkpoint
      `checkpoints/phase07_shuffled_contact_residual_adapter_trainer_v1_20260627/phase07_shuffled_contact_residual_adapter_v1_train_20260627.pt`,
      `elapsed_seconds=3600.129097223282`, `optimizer_steps=21651`,
      `real_training_result=true`, `checkpoint_written=true`, and mean GPU
      utilization `98.875%`. All three held-out evals produced passing
      metrics, 360-frame videos, action-bridge validation, and manual visual
      JSONs. This remains ablation evidence only, not a final success claim.
      Progress 2026-06-28 delayed-contact completion: real one-GPU-hour
      training completed with summary
      `experiments/outputs/phase07_delayed_contact_residual_adapter_trainer_v1_20260627/phase07_delayed_contact_residual_adapter_v1_train_20260627_summary.json`,
      checkpoint
      `checkpoints/phase07_delayed_contact_residual_adapter_trainer_v1_20260627/phase07_delayed_contact_residual_adapter_v1_train_20260627.pt`,
      `elapsed_seconds=3600.1080799102783`, `optimizer_steps=21745`,
      `real_training_result=true`, `checkpoint_written=true`, and mean GPU
      utilization `98.825%`. All three held-out evals produced passing
      metrics, 360-frame videos, action-bridge validation, and manual visual
      JSONs. This remains ablation evidence only.
      Progress 2026-06-28 no-learning-progress completion: real one-GPU-hour
      training completed with summary
      `experiments/outputs/phase07_no_learning_progress_residual_adapter_trainer_v1_20260627/phase07_no_learning_progress_residual_adapter_v1_train_20260627_summary.json`,
      checkpoint
      `checkpoints/phase07_no_learning_progress_residual_adapter_trainer_v1_20260627/phase07_no_learning_progress_residual_adapter_v1_train_20260627.pt`,
      `elapsed_seconds=3600.085962533951`, `optimizer_steps=21607`,
      `real_training_result=true`, `checkpoint_written=true`, and mean GPU
      utilization `99.0%`. All three held-out evals produced passing metrics,
      360-frame videos, action-bridge validation, and manual visual JSONs. The
      queue reran official readiness, held-out comparison, and hard gate, but
      the result remains incomplete/negative: curiosity still does not beat the
      strongest baseline, and official/mainstream comparison readiness remains
      open. The final gate's no-learning manual-visual missing item is stale
      because the manual JSONs were added after the queue gate run.
      Progress 2026-06-27 remaining-ablation config update: configs are now
      prepared and syntax-checked for `shuffled_contact`, `delayed_contact`,
      and `no_learning_progress`:
      `experiments/configs/phase07_shuffled_contact_residual_adapter_trainer_v1.json`,
      `experiments/configs/phase07_delayed_contact_residual_adapter_trainer_v1.json`,
      and `experiments/configs/phase07_no_learning_progress_residual_adapter_trainer_v1.json`.
      Their score summaries are pass with `policy_updated=false`, but they
      still need real one-GPU-hour training and held-out video evaluation.
      Progress 2026-06-27 queue automation: added
      `experiments/configs/run_phase07_remaining_ablation_queue_in_alloc.sh`
      and `experiments/configs/launch_phase07_remaining_ablation_queue_tmux.sh`.
      These scripts wait for a running tmux-held Slurm allocation, then run the
      four remaining ablation trainings and their three held-out evaluations
      with metrics and acceleration analysis. This only prepares/executes the
      required evidence path; manual visual inspection, report update, and
      stronger closed-loop curiosity repair remain open.
      Added lightweight autolaunch watcher
      `experiments/configs/watch_phase07_remaining_ablation_queue_autolaunch.sh`
      so pending job `155039` can start the queue automatically once it becomes
      RUNNING, without doing training or rendering on the login node.
      Queue runner now writes `logs/newton/*_env.sh` for each training and
      held-out evaluation job to preserve exact reproducibility.
- [ ] Satisfy the serious/mainstream method comparison gate.
      The final harder-task curiosity claim must either beat a faithful
      mainstream comparison or document official incompatibility blockers. Gate
      file: `experiments/configs/phase07_mainstream_comparison_gate_v1.json`.
      Current required candidates are OpenPI/pi0, Diffusion Policy, Open X/RT-X,
      and NVIDIA Isaac GR00T. Do not implement toy replacements for any of
      these methods. This item remains open until each candidate is either
      compared using official code/checkpoints or blocked with concrete
      evidence.
      Progress 2026-06-27 audit: added
      `experiments/outputs/phase07_mainstream_comparison_audit_v1_20260627.json`
      and `experiments/reports/2026-06-27_phase07_mainstream_comparison_audit_v1.md`.
      Official repos are reachable, but no faithful Phase07 comparison or
      concrete incompatibility blocker has been completed, so the gate remains
      open.
      Added and ran `experiments/configs/audit_phase07_mainstream_repos_v1.py`.
      It wrote
      `experiments/outputs/phase07_mainstream_repo_reachability_audit_v1_20260627.json`
      as a repeatable official repo/checkpoint availability audit.
      Progress 2026-06-27 official-code update: shallow-cloned official
      OpenPI/pi0, Diffusion Policy, Open X/RT-X, and NVIDIA Isaac GR00T code
      under `external/` with `GIT_LFS_SKIP_SMUDGE=1`; no large model weights
      were downloaded. Added
      `experiments/configs/phase07_mainstream_official_code_compatibility_matrix_v1.json`
      and
      `experiments/reports/2026-06-27_phase07_mainstream_official_code_compatibility_matrix_v1.md`.
      Gate remains open because no official checkpoint comparison or concrete
      incompatibility blocker is complete.
      Progress 2026-06-27 bridge spec: added
      `experiments/configs/phase07_mainstream_adapter_bridge_spec_v1.json` and
      `experiments/reports/2026-06-27_phase07_mainstream_adapter_bridge_spec_v1.md`.
      These specify required OpenPI/GR00T LeRobot mappings, Diffusion Policy
      Dataset/EnvRunner/shape_meta requirements, and RT-X 7D gripper-frame
      action bridge requirements. Residual-only imitation is explicitly
      diagnostic and cannot close the mainstream gate.
      Progress 2026-06-27 bridge readiness audit: added and ran
      `experiments/configs/audit_phase07_mainstream_bridge_readiness_v1.py`,
      writing
      `experiments/outputs/phase07_mainstream_bridge_readiness_audit_v1_20260627.json`.
      Required Phase07 source/context columns and current held-out videos exist,
      but preferred 7D EEF/gripper action columns are missing. Next required
      work is a provenance-preserving Newton Panda EEF/gripper action bridge or
      a concrete official incompatibility blocker; residual-only imitation is
      not enough.
      Progress 2026-06-27 action-bridge implementation: updated
      `experiments/configs/newton_panda_hydro_tiled_camera_export.py` so future
      rollout NPZs emit `candidate.action.eef_delta_x/y/z/roll/pitch/yaw`,
      `candidate.action.gripper`, and
      `candidate.action.eef_delta_xyzrpy_gripper`. Existing older held-out NPZs
      still lack these fields; reruns are required before mainstream adapter
      conversion can use them.
      Queue runner now validates those fields after each new held-out eval and
      writes `<run_tag>_candidate_action_bridge_validation.json`; missing bridge
      fields fail the queue.
      Added allocation-only backfill scripts
      `experiments/configs/backfill_phase07_candidate_action_bridge_v1.py` and
      `experiments/configs/run_phase07_candidate_action_bridge_backfill_in_alloc.sh`.
      They create bridge-bearing copies of existing no-adaptation and
      curiosity-weighted held-out NPZs under
      `experiments/outputs/phase07_action_bridge_backfill_v1_20260627/` without
      overwriting originals. The remaining-ablation queue runs this first once
      allocation `155039` starts.
      Added allocation-only mainstream adapter conversion preflight:
      `experiments/configs/build_phase07_mainstream_adapter_conversion_preflight_v1.py`
      and
      `experiments/configs/run_phase07_mainstream_adapter_conversion_preflight_in_alloc.sh`.
      The remaining-ablation queue runs it after backfill to validate
      bridge-bearing NPZ shapes and write method-specific mapping specs before
      any full official dataset conversion or checkpoint run.
- [ ] Evaluate held-out variable water-cup combinations.
      Held-out variants must not be used for labels, training, hyperparameter
      tuning, or threshold selection. Required gates: fresh official Newton
      sanity, automated visual validation, manual visual inspection, strict
      lift/hold/slip/drop/contact/acceleration/safety metrics, complete
      rollout videos, and direct visual paths.
      Progress: `three_quarter_low` side-by-side videos for no-adaptation,
      scripted feedback, residual baseline, and curiosity-weighted policy
      passed visual and lift-hold gates. Curiosity had slightly higher lift
      and lower xy drift than the residual baseline, but worse acceleration
      proxy; the no-adaptation scripted prior also succeeded with higher lift
      and longer hold. Therefore this is not a success claim. Full held-out
      variable water-cup evaluation remains incomplete.
      Progress 2026-06-27 update: all three configured held-out cells now have
      no-adaptation, scripted-feedback, no-curiosity residual, and
      curiosity-weighted residual 360-frame GIFs. Current comparison is
      negative for curiosity because no-adaptation remains stronger on lift
      and hold duration for every held-out cell.
      Progress 2026-06-27 evidence update: manual visual JSONs, acceleration
      peak analysis, and metrics CSV/JSON now pass for all 12 held-out policy
      videos. Ablations, mainstream comparison, visual fill cue rendering, and
      stronger closed-loop curiosity training remain incomplete.
      Progress 2026-06-27 random-ablation update: random-intrinsic residual
      ablation added three more held-out 360-frame videos with metrics,
      acceleration analysis, and manual visual JSONs. It remains below the
      no-adaptation baseline on lift/hold and is not a success claim.
      Progress 2026-06-27 object-only update: object-only residual ablation
      added three more held-out 360-frame videos with metrics, acceleration
      analysis, and manual visual JSONs. It remains below no-adaptation on
      lift/hold and is not a success claim.
      Progress 2026-06-28 evidence refresh: allocation-only refresh wrote
      `experiments/outputs/phase07_evidence_refresh_v1_20260628_summary.json`
      and
      `experiments/reports/2026-06-28_phase07_evidence_refresh_v1.md`.
      Stale evidence was cleared, but the hard gate remains
      `open_not_satisfied` because curiosity still does not beat the strongest
      baseline and official serious-method readiness remains open.
      Progress 2026-06-28 threshold repair: validation-only threshold repair
      wrote
      `experiments/outputs/phase07_closed_loop_threshold_repair_v1_20260628_summary.json`.
      After fixing the selector to prioritize safety before lift, the corrected
      selected threshold is `0.65`; this is not training and not a success
      claim.
      Progress 2026-06-28 corrected 0.65 held-out rerun:
      `experiments/outputs/phase07_threshold065_heldout_eval_retry_v1_20260628_summary.json`
      remains `open_not_satisfied`. All three held-out full-video runs are
      visually valid and list no safety regression, but none beat
      `no_adaptation` on the ordered held-out comparison. Continue with a real
      training/objective fix rather than treating threshold repair as
      completion.
- [ ] Collect Phase07 V2 stabilization-range train/validation source data.
      The corrected 0.65 failure exposed a structural limit: current source
      labels and learned residual evaluation clamp stabilization extension to
      `0.3s`, while the strongest no-adaptation held-out baseline holds for
      about `3.1s`. V2 raises the train/validation source action range to
      `0.9s` with safety metrics and full 420-frame videos. This is source
      collection only; it must not be called training or success.
      Progress 2026-06-28: added
      `experiments/configs/phase07_v2_stabilization_source_collection_v1.json`,
      `experiments/configs/run_phase07_v2_stabilization_source_collection_in_alloc.sh`,
      and
      `experiments/configs/launch_phase07_v2_stabilization_source_collection_tmux.sh`.
      Started tmux-held Slurm job `155749` in session
      `curiosity_phase07_v2_source_alloc_20260628`; log is
      `logs/newton/phase07_v2_stabilization_source_collection_v1_20260628.log`.
- [x] Manually inspect all Phase07 V2 source contact sheets before source-runner promotion.
      The V2 source collector writes manual visual JSONs as
      `pending_direct_agent_check`. Do not run the V2 source runner or any V2
      training until all eight train/validation source videos are directly
      checked and the manifest status is promoted from
      `phase07_v2_stabilization_source_candidates_manual_visual_pending` to
      `phase07_v2_stabilization_source_candidates_complete_training_not_started`.
      Progress 2026-06-28: all eight V2 source contact sheets were directly
      inspected and passed as nonblank three-camera visual evidence. The
      manual JSONs were updated to `pass_nonblank_success_with_feedback`; this
      validates source visuals only and is not a curiosity success claim.
- [x] Run Phase07 V2 source runner and residual-adapter preflight.
      Progress 2026-06-28: V2 source runner passed, writing
      `data/processed/phase07_v2_stabilization_residual_label_source_runner_v1_20260628/manifest.json`
      with `3360` records, eight source runs, and `2405` feedback-trigger
      frames. V2 residual preflight passed, writing
      `data/processed/phase07_v2_residual_adapter_training_preflight_v1_20260628/manifest.json`
      with `2520` train records and `840` validation records.
- [ ] Run Phase07 V2 source runner, preflights, and real training after manual checks pass.
      Added V2 configs for source runner, residual preflight, no-curiosity
      residual baseline training, curiosity forward-model preflight/training,
      learning-progress scoring, and curiosity-weighted residual training:
      `experiments/configs/phase07_v2_stabilization_residual_label_source_runner_v1.json`,
      `experiments/configs/phase07_v2_residual_adapter_training_preflight_v1.json`,
      `experiments/configs/phase07_v2_residual_adapter_trainer_v1.json`,
      `experiments/configs/phase07_v2_curiosity_forward_model_preflight_v1.json`,
      `experiments/configs/phase07_v2_curiosity_forward_model_trainer_v1.json`,
      `experiments/configs/phase07_v2_curiosity_learning_progress_v1.json`, and
      `experiments/configs/phase07_v2_curiosity_weighted_residual_adapter_trainer_v1.json`.
      Each real training run still requires at least one GPU-hour and fresh
      official Newton sanity. Held-out cells remain forbidden until final eval.
      Progress 2026-06-28: V2 no-curiosity residual baseline real training
      started as `phase07_v2_residual_adapter_v1_train_20260628` in Slurm job
      `155749`; log:
      `logs/newton/phase07_v2_residual_adapter_v1_train_20260628.log`.
      This is a stronger baseline training run, not curiosity success.
      Progress 2026-06-28 later: V2 no-curiosity residual training completed
      with `real_training_result=true`, `elapsed_seconds=3600.183328151703`,
      `optimizer_steps=18668`, mean GPU utilization `99.07964601769912%`, and
      checkpoint
      `checkpoints/phase07_v2_residual_adapter_trainer_v1_20260628/phase07_v2_residual_adapter_v1_train_20260628.pt`.
      V2 forward-model preflight then passed with `3352` transition records,
      and real V2 curiosity forward-model training started as
      `phase07_v2_curiosity_forward_model_v1_train_20260628`.
      Progress 2026-06-28 later: V2 forward-model training completed with
      `real_training_result=true`, `elapsed_seconds=3600.023451566696`,
      `optimizer_steps=17970`, checkpoint and initial snapshot under
      `checkpoints/phase07_v2_curiosity_forward_model_v1_20260628/`, and mean
      GPU utilization `99.130081300813%`. V2 learning-progress scoring passed
      for `3352` records with mean bounded curiosity reward
      `0.643422921641614`. V2 curiosity-weighted residual real training
      started as
      `phase07_v2_curiosity_weighted_residual_adapter_v1_train_20260628`.
      Progress 2026-06-28 later: V2 curiosity-weighted residual real training
      completed with `real_training_result=true`,
      `elapsed_seconds=3600.0821080207825`, checkpoint
      `checkpoints/phase07_v2_curiosity_weighted_residual_adapter_trainer_v1_20260628/phase07_v2_curiosity_weighted_residual_adapter_v1_train_20260628.pt`,
      and mean GPU utilization `99.81666666666666%`. Held-out V2 evaluation
      then completed as `phase07_v2_heldout_eval_v1_20260628`, writing summary
      `experiments/outputs/phase07_v2_heldout_eval_v1_20260628_summary.json`,
      report `experiments/reports/2026-06-28_phase07_v2_heldout_eval_v1.md`,
      and manual visual inspection
      `experiments/outputs/phase07_v2_heldout_eval_v1_20260628_manual_visual_inspection.json`.
      Result: `open_not_satisfied`. The videos/contact sheets are valid, but
      curiosity-weighted residual did not beat `no_adaptation` across all
      held-out cells and did not beat `no_curiosity_residual` without safety
      regression. This is negative/incomplete evidence, not completion.
- [ ] Repair the V2 objective after negative held-out evidence.
      V2 widened stabilization action range and produced real checkpoints, but
      held-out comparison still shows the learned residual sacrifices the
      strong no-adaptation hold/lift behavior. The next training attempt must
      add an explicit baseline-preservation or behavior-cloning anchor against
      the official scripted/no-adaptation trajectory while keeping
      curiosity-driven safety gains. Do not solve this by lowering the
      held-out bar, changing the held-out split, reporting only threshold
      tuning, or calling the V2 checkpoint complete.
      Progress 2026-06-28: implemented and trained a first neutral-residual
      baseline-preservation anchor as
      `phase07_v2_curiosity_weighted_residual_adapter_anchor_v1_train_20260628`.
      The training artifact is valid and ran for one GPU-hour, but validation
      quality was poor (`active_accuracy=0.6595238447189331`,
      `continuous_mse=1.8517640829086304`, `loss=4.608899116516113`). Held-out
      anchor evaluation
      `experiments/outputs/phase07_v2_anchor_heldout_eval_v1_20260628_summary.json`
      remained `open_not_satisfied`: it did not beat no-adaptation across all
      cells and did not beat the no-curiosity residual baseline without safety
      regressions. All nine contact sheets were directly inspected and passed
      as nonblank multi-camera rollouts; see
      `experiments/outputs/phase07_v2_anchor_heldout_eval_v1_20260628_manual_visual_inspection.json`.
      This is negative evidence and must not be called completed curiosity
      training.
- [ ] Repair the anchor objective after the negative V2 anchor held-out result.
      The first anchor likely over-constrained the residual by pushing stable
      contact frames toward a neutral continuous target. Continue with a
      faithful repair rather than downgrading the task: test a softer/separate
      anchor, keep the harder held-out cells fixed, preserve the strongest
      baseline gate, and require another real one-hour GPU training run before
      making any training claim.
      Progress 2026-06-28: added separated anchor loss controls to
      `experiments/configs/train_curiosity_weighted_residual_adapter_v1.py`
      and created
      `experiments/configs/phase07_v2_curiosity_weighted_residual_adapter_active_anchor_trainer_v1.json`.
      This next attempt uses the same architecture/data/eval gate but replaces
      the failed hard neutral-continuous anchor with a soft active-only anchor.
      It still requires smoke verification, a real one-hour GPU training run,
      harder held-out evaluation, full videos, and direct visual inspection
      before any success language is allowed.
      Progress 2026-06-28: requested a one-day tmux-held GPU allocation for
      this repair as Slurm job `155785`, session
      `curiosity_phase07_active_anchor_alloc_20260628_101625`. The job is for
      smoke, real training, and held-out evaluation only; it is not completion
      evidence.
      Progress 2026-06-28 later: active-anchor smoke
      `phase07_v2_curiosity_weighted_residual_adapter_active_anchor_v1_smoke_20260628`
      passed with fresh official Newton sanity, `smoke_diagnostic_only=true`,
      `checkpoint_written=false`, and validation
      `active_accuracy=0.997619092464447`. Real one-hour training started as
      `phase07_v2_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628`
      in the same allocation. This remains in progress and is not success
      evidence until real-training, held-out evaluation, full videos, and
      manual visual gates pass.
      Progress 2026-06-28 later: active-anchor real training completed with
      `real_training_result=true`, `elapsed_seconds=3600.1757838726044`,
      `optimizer_steps=18649`, checkpoint
      `checkpoints/phase07_v2_curiosity_weighted_residual_adapter_active_anchor_trainer_v1_20260628/phase07_v2_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628.pt`,
      and mean GPU utilization `98.79166666666667%`. Validation remained weak
      (`active_accuracy=0.663095235824585`, `active_bce=0.6922045350074768`),
      so this is not a success claim. Held-out evaluation started as
      `phase07_v2_active_anchor_heldout_eval_v1_20260628` with the fixed harder
      cells and full video export.
      Progress 2026-06-28 later: active-anchor held-out evaluation completed
      9/9 videos and direct contact-sheet inspection. Summary:
      `experiments/outputs/phase07_v2_active_anchor_heldout_eval_v1_20260628_summary.json`.
      Manual inspection:
      `experiments/outputs/phase07_v2_active_anchor_heldout_eval_v1_20260628_manual_visual_inspection.json`.
      Result remains `open_not_satisfied`: curiosity did not beat no-adaptation
      across all cells and did not beat no-curiosity residual without safety
      regression. This is negative evidence, not completion.
- [ ] Inspect and repair learned-residual controller baseline transition.
      The soft active-only anchor still cannot recover no-adaptation hold
      duration, which suggests the learned-residual controller mode may not
      reproduce the official scripted waypoint/contact transition even when
      the residual should be neutral. Before another anchor-only training run,
      audit the controller path and run a neutral-residual/no-feedback parity
      check against no-adaptation on the same harder held-out cells.
      Progress 2026-06-28: confirmed `lift_hold_learned_residual` was
      initialized through `_configure_lift_hold_feedback_waypoints`, applying
      `FEEDBACK_INITIAL_LIFT_DURATION_SCALE` and an initial stabilization
      extension before any learned residual activation. Patched
      `experiments/configs/newton_panda_hydro_tiled_camera_export.py` so
      learned-residual mode starts from `_configure_lift_hold_waypoints`.
      Syntax check passed. Launched neutral parity eval
      `phase07_v2_learned_neutral_parity_eval_v1_20260628` with
      `ACTIVE_THRESHOLD=2.0`; this is a controller sanity check, not curiosity
      success.
      Progress 2026-06-28 later: neutral parity restored hold-duration parity
      with no-adaptation on all three harder cells
      (`4.133329391479492`, `4.099996089935303`, `4.099996089935303`). This is
      not curiosity success. Launched post-repair held-out eval
      `phase07_v2_fixed_controller_active_anchor_heldout_eval_v1_20260628`
      with the active-anchor checkpoint and `ACTIVE_THRESHOLD=0.5`.
      Progress 2026-06-28 later: post-repair held-out eval completed 9/9
      videos and direct visual inspection. Summary:
      `experiments/outputs/phase07_v2_fixed_controller_active_anchor_heldout_eval_v1_20260628_summary.json`.
      Result remains `open_not_satisfied`: neutral controller bias is repaired,
      but the active checkpoint still reduces hold/lift relative to
      no-adaptation and introduces safety regressions. This is negative
      evidence, not completion.
- [ ] Regenerate residual source labels from the repaired official base.
      The current V2 residual checkpoints were trained from source labels
      collected before the learned-residual neutral-path repair. Build the next
      source collection so the base trajectory is official no-adaptation and
      residual labels are contact-triggered changes relative to that base.
      Then rerun source runner, preflights, real no-curiosity training,
      curiosity forward training, learning-progress scoring,
      curiosity-weighted training, and the same harder held-out video gate.
      Progress 2026-06-28: added
      `experiments/configs/phase07_v3_repaired_base_source_collection_v1.json`
      and wired
      `FEEDBACK_APPLY_INITIAL_WAYPOINT_ADJUSTMENT=0` through the Newton export
      runner. Started source collection
      `phase07_v3_repaired_base_source_collection_v1_20260628` in Slurm job
      `155785`, tmux window `phase07_v3_repaired_source`. This is source data
      collection only; downstream source runner/training remains blocked until
      all source videos are manually inspected.
      Progress 2026-06-28 later: source collection completed 8/8, and all
      eight contact sheets were directly inspected as nonblank multi-camera
      robot/object rollouts. Manifest
      `experiments/configs/phase07_v3_repaired_base_source_manifest_v1.json`
      was promoted to
      `phase07_v3_repaired_base_source_candidates_complete_training_not_started`.
      Generated V3 source runner/preflight configs and launched
      `phase07_v3_repaired_base_residual_label_source_runner_v1_20260628`.
      Progress 2026-06-28 later: V3 source runner passed with fresh official
      Newton sanity and produced 3360 source records from 8 source runs
      (`total_feedback_active_frames=2405`), while keeping
      `schema_promotion=blocked` and `training_started=false`. V3 residual
      adapter preflight also passed with 2520 train records and 840 validation
      records, with the three harder held-out cells still reserved. Added V3
      repaired-base configs for no-curiosity residual training, curiosity
      forward preflight/training, learning-progress scoring, and active-anchor
      curiosity-weighted residual fine-tuning. Started the V3 no-curiosity
      residual baseline real training as
      `phase07_v3_repaired_base_residual_adapter_v1_train_20260628` in Slurm
      job `155785`, tmux window `phase07_v3_residual_train`. This is required
      baseline training, not curiosity success evidence.
      Progress 2026-06-28 later: V3 no-curiosity residual baseline completed
      as a real one-hour run with `elapsed_seconds=3600.1102674007416`,
      `optimizer_steps=18651`, checkpoint
      `checkpoints/phase07_v3_repaired_base_residual_adapter_trainer_v1_20260628/phase07_v3_repaired_base_residual_adapter_v1_train_20260628.pt`,
      and mean GPU utilization `98.96666666666667%`. V3 curiosity forward
      preflight passed with 3352 transition records (`2514` train, `838`
      validation), no model creation, and `schema_promotion=blocked`. Started
      V3 curiosity forward-model real training as
      `phase07_v3_repaired_base_curiosity_forward_model_v1_train_20260628` in
      Slurm job `155785`, tmux window `phase07_v3_forward_train`. This is a
      dynamics/learning-progress prerequisite only, not a policy update or
      curiosity success claim.
      Progress 2026-06-28 later: V3 curiosity forward-model training completed
      as a real one-hour run with `elapsed_seconds=3600.0523829460144`,
      `optimizer_steps=17989`, initial snapshot
      `checkpoints/phase07_v3_repaired_base_curiosity_forward_model_v1_20260628/phase07_v3_repaired_base_curiosity_forward_model_v1_train_20260628_initial_snapshot.pt`,
      trained checkpoint
      `checkpoints/phase07_v3_repaired_base_curiosity_forward_model_v1_20260628/phase07_v3_repaired_base_curiosity_forward_model_v1_train_20260628.pt`,
      and mean GPU utilization `98.9%`. V3 learning-progress scoring passed
      with 3352 scores, train mean bounded curiosity reward
      `0.8107618392954183`, validation mean `0.6631783638273964`, and
      `policy_updated=false`. Started the V3 repaired-base
      curiosity-weighted residual policy fine-tune as
      `phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628`
      in Slurm job `155785`, tmux window
      `phase07_v3_curiosity_residual_train`. This is the policy-update
      training stage, but it is still not success evidence until the one-hour
      gate, checkpoint, held-out videos, strict metrics, and strongest-baseline
      comparisons pass without safety regression.
      Progress 2026-06-28 later: V3 repaired-base curiosity-weighted residual
      policy fine-tune completed as a real one-hour run with
      `elapsed_seconds=3600.164398908615`, `optimizer_steps=18549`,
      `train_score_coverage=0.9976190476190476`, checkpoint
      `checkpoints/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_trainer_v1_20260628/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_active_anchor_v1_train_20260628.pt`,
      and mean GPU utilization `98.76666666666667%`. Started V3 repaired-base
      held-out evaluation
      `phase07_v3_repaired_base_curiosity_heldout_eval_v1_20260628` with eval
      prefix `phase07_v3_repaired_base_eval`, active threshold `0.5`, paired
      V3 no-curiosity/curiosity checkpoints, full videos, and the same harder
      held-out cells. This evaluation is still pending and no success claim is
      allowed until metrics, videos, manual visual inspection, and
      strongest-baseline comparison pass without safety regression.
      Progress 2026-06-28 later: V3 held-out evaluation completed 9/9 rollouts
      and direct contact-sheet inspection. Manual inspection passed as
      nonblank multi-camera robot/object rollouts, but the aggregate status is
      `open_not_satisfied`. Curiosity failed both all-cell gates: it did not
      beat no-adaptation and did not beat no-curiosity residual without safety
      regression. Key failures: `empty_high_misleading` matched no-adaptation
      hold but had lower lift and higher acceleration; `full_low_hidden` was
      lower than both baselines on hold/lift; `three_quarter_low_misleading`
      had similar hold to no-curiosity but a slip regression. Treat this as
      valid negative evidence. Next task: diagnose score selectivity,
      active-frequency/residual magnitude, and validation threshold
      sensitivity before another held-out run.
      Progress 2026-06-28 later: diagnostics found weak score selectivity
      (`41.795942720763724%` of scores `>=0.95`, median
      `0.8553736656904221`) and weak active validation behavior
      (`active_accuracy=0.6571428775787354`, active BCE
      `1.1270439624786377`). Held-out residual traces also show unstable
      activation: `empty_high_misleading` curiosity active rate only
      `0.2380952380952381%`, while the other two harder cells are about
      `71.19047619047619%`. Added
      `experiments/configs/phase07_v3_repaired_base_closed_loop_threshold_repair_v1.json`
      and launched validation-only threshold repair
      `phase07_v3_repaired_base_closed_loop_threshold_repair_v1_20260628` in
      Slurm job `155785`, tmux window `phase07_v3_threshold_repair`. Held-out
      cells remain forbidden for threshold selection. This is diagnostic
      threshold repair only, not training and not success evidence.
      Progress 2026-06-28 later: V3 validation-only threshold repair completed
      8/8 validation rollouts and selected threshold `0.95`. All validation
      rollouts were `status_ok=true` and `success=true`, but the high selected
      threshold indicates the current curiosity checkpoint is safest when
      mostly suppressed, so this supports the active-head/score-selectivity
      diagnosis rather than a success claim. Started post-validation held-out
      re-evaluation
      `phase07_v3_repaired_base_threshold095_heldout_eval_v1_20260628` with
      eval prefix `phase07_v3_repaired_base_thr095_eval`, active threshold
      `0.95`, and the paired V3 no-curiosity/curiosity checkpoints. This
      remains held-out evaluation only and must still pass metrics, videos,
      manual visual inspection, and baseline comparisons before any success
      language.
      Progress 2026-06-28 later: threshold `0.95` held-out re-evaluation
      completed 9/9 rollouts and direct contact-sheet inspection, but remains
      `open_not_satisfied`. It makes `empty_high_misleading` nearly match
      no-adaptation, but `full_low_hidden` and
      `three_quarter_low_misleading` remain below no-adaptation on hold/lift
      and show acceleration regressions. Threshold repair alone is not enough.
      Generated rank-calibrated score artifact
      `experiments/outputs/phase07_v3_repaired_base_curiosity_learning_progress_rank_calibrated_v1_20260628/curiosity_learning_progress_summary.json`
      to reduce saturated curiosity weighting, added
      `experiments/configs/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_trainer_v1.json`,
      and launched real one-hour rank-calibrated residual fine-tune
      `phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_v1_train_20260628`
      in Slurm job `155785`, tmux window `phase07_v3_rank_residual_train`.
      This is a weighting/anchor repair attempt, not success evidence.
      Progress 2026-06-28 later: rank-calibrated residual training completed a
      real one-hour run with `elapsed_seconds=3600.0904426574707`,
      `optimizer_steps=18576`, `train_score_coverage=0.9976190476190476`,
      checkpoint
      `checkpoints/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_trainer_v1_20260628/phase07_v3_repaired_base_curiosity_weighted_residual_adapter_rank_active_anchor_v1_train_20260628.pt`,
      and mean GPU utilization `98.73333333333333%`. Validation active
      behavior improved (`active_accuracy=0.9297619462013245`,
      `active_bce=0.18072140216827393`). Started held-out evaluation
      `phase07_v3_repaired_base_rank_curiosity_heldout_eval_v1_20260628` with
      eval prefix `phase07_v3_repaired_base_rank_eval`, active threshold `0.5`,
      and the same no-adaptation/no-curiosity/rank-curiosity comparison gate.
      This remains evaluation pending, not success evidence.
      Progress 2026-06-28 later: first rank held-out eval attempt failed
      before performance evidence because the Newton export loader allow-list
      did not accept checkpoint classification
      `newton_native_rank_calibrated_curiosity_weighted_residual_controller_adapter_v1_checkpoint`.
      Patched
      `experiments/configs/newton_panda_hydro_tiled_camera_export.py` to accept
      that rank-calibrated checkpoint classification, syntax check passed, and
      restarted eval as
      `phase07_v3_repaired_base_rank_curiosity_heldout_eval_retry_v1_20260628`
      with eval prefix `phase07_v3_repaired_base_rank_retry_eval`. This is a
      glue fix/retry, not model progress or success evidence.
      Progress 2026-06-28 later: rank held-out retry completed 9/9 rollouts
      and direct contact-sheet inspection. Manual inspection JSON:
      `experiments/outputs/phase07_v3_repaired_base_rank_curiosity_heldout_eval_retry_v1_20260628_manual_visual_inspection.json`.
      Visual evidence is valid/nonblank, but the aggregate status remains
      `open_not_satisfied`: rank curiosity fails both all-cell gates and does
      not beat no-adaptation on harder held-out hold/lift. Key failures:
      `empty_high_misleading` has lower hold/lift and much higher acceleration
      than no-adaptation; `full_low_hidden` remains below no-adaptation and
      no-curiosity residual on hold/lift; `three_quarter_low_misleading`
      remains below no-adaptation on hold/lift and has an acceleration
      regression versus no-curiosity residual. Treat this as negative evidence,
      not completion. Next task: repair the residual target or active
      intervention source so curiosity can improve held-out hold/lift without
      safety regression; do not claim success or exit quickly from the V3
      rank-calibrated checkpoint.
- [ ] Run V3 closed-loop DAgger-style teacher repair instead of another
      offline curiosity-score reweighting pass.
      Added `--record-scripted-teacher-labels` to
      `experiments/configs/newton_panda_hydro_tiled_camera_export.py` so the
      learned residual policy controls Newton while scripted corrective labels
      are recorded under `candidate.teacher.*` and not applied. Added
      `experiments/configs/build_phase07_closed_loop_teacher_preflight_v1.py`,
      `experiments/configs/phase07_v3_closed_loop_teacher_preflight_v1.json`,
      `experiments/configs/phase07_v3_closed_loop_teacher_residual_adapter_trainer_v1.json`,
      `experiments/configs/run_phase07_v3_closed_loop_teacher_training_in_alloc.sh`,
      and
      `experiments/configs/launch_phase07_v3_closed_loop_teacher_training_tmux.sh`.
      Static validation passed for Python syntax, shell syntax, JSON configs,
      and required checkpoints. Slurm job `156649` was submitted through tmux
      session `curiosity_phase07_closed_loop_teacher_alloc_20260628` as
      `phase07_closed_loop_teacher_1gpu_1day` and is pending for resources.
      This task remains open until the chain collects on-policy train/validation
      teacher rollouts, preflight passes with nonzero teacher labels, real
      one-hour training writes a checkpoint, and harder held-out full-video
      evaluation plus manual visual/baseline/mainstream gates pass.
      Progress 2026-06-28 later: job `156649` started on `server57` and the
      chain `phase07_v3_closed_loop_teacher_chain_v1_20260628` completed all
      eight train/validation on-policy source rollouts with learned-residual
      control, non-applied scripted teacher labels, and video export pass.
      Preflight passed at
      `data/processed/phase07_v3_closed_loop_teacher_preflight_v1_20260628/manifest.json`
      with `2520` train rows, `840` validation rows,
      `total_teacher_active_frames=2407`, and `failures=[]`. The one-hour
      closed-loop teacher residual training run
      `phase07_v3_closed_loop_teacher_residual_adapter_v1_train_20260628` has
      started after fresh official Newton sanity. Still open: checkpoint, GPU
      utilization gate, harder held-out full videos, direct manual visual
      inspection, strongest-baseline comparison, and mainstream/official method
      gate.
      Progress 2026-06-28 later: the training run completed a real one-hour
      policy update with `elapsed_seconds=3600.0265502929688`,
      `optimizer_steps=18576`, checkpoint
      `checkpoints/phase07_v3_closed_loop_teacher_residual_adapter_trainer_v1_20260628/phase07_v3_closed_loop_teacher_residual_adapter_v1_train_20260628.pt`,
      mean GPU utilization `98.56198347107438%`, and validation active accuracy
      `0.9952381253242493`. The chain has entered harder held-out full-video
      evaluation `phase07_v3_closed_loop_teacher_heldout_eval_v1_20260628`.
      Still not complete: need held-out metrics, full videos, manual visual
      inspection, no-adaptation/no-curiosity comparisons without safety
      regression, and mainstream/official comparison gate.
      Progress 2026-06-28 later: held-out evaluation completed 9/9 full-video
      rollouts and direct contact-sheet inspection. Manual inspection JSON:
      `experiments/outputs/phase07_v3_closed_loop_teacher_heldout_eval_v1_20260628_manual_visual_inspection.json`.
      The result is `open_not_satisfied`, not success. Closed-loop teacher does
      not beat no-adaptation on any held-out cell: `empty_high_misleading`
      hold/lift `3.9333295822143555`/`0.16152004897594452` versus no-adaptation
      `4.133329391479492`/`0.1663660854101181`; `full_low_hidden`
      `3.8666629791259766`/`0.1542869359254837` versus no-adaptation
      `4.099996089935303`/`0.15918205678462982`; and
      `three_quarter_low_misleading`
      `3.8833296298980713`/`0.15595264732837677` versus no-adaptation
      `4.099996089935303`/`0.16022272408008575`. Diagnostics show about `72%`
      held-out active rate, nearly the same as no-curiosity residual, so the
      failure is over-dense corrective labels/over-intervention. Next repair:
      do not repeat dense teacher imitation; build advantage-gated intervention
      labels from paired train/validation evidence or expand to genuinely
      harder object/task families where no-adaptation fails.
- [ ] Find a harder task distribution where no-adaptation actually leaves room
      for closed-loop curiosity to improve.
      Progress 2026-06-28: ran diagnostic
      `phase08_harder_candidate_probe_v1_20260628` on ultra-low-friction
      `full_ultralow_hidden` and `three_quarter_ultralow_misleading` with full
      rollout videos for no-adaptation and scripted feedback. Manual inspection
      JSON:
      `experiments/outputs/phase08_harder_candidate_probe_v1_20260628_manual_visual_inspection.json`.
      Result is negative for source selection: no-adaptation still beats
      scripted feedback on hold/lift (`4.1`/`0.1592235416173935` vs
      `2.966666666666667`/`0.15194369852542877` on full-ultralow, and
      `4.1`/`0.1602979302406311` vs `2.9833333333333334`/`0.1528860181570053`
      on three-quarter-ultralow). Do not use lower friction alone as the next
      training distribution. Next candidates should alter object geometry,
      contact patch, deformability, off-center torque, or use advantage-gated
      labels.
- [ ] Build advantage-gated residual data before the next training attempt.
      Added `experiments/configs/build_advantage_gated_residual_preflight_v1.py`
      and `experiments/configs/phase08_advantage_gated_residual_preflight_v1.json`.
      Static checks passed. The gate must compare paired no-adaptation and
      intervention rollouts, reject cells whose intervention does not improve
      hold/lift or causes safety regression, and fail instead of emitting
      harmful dense-feedback training rows. This task remains open until paired
      train/validation rollouts have been collected, the manifest passes with
      accepted cells and active frames, and the resulting dataset is used in a
      real one-hour training run.
      Progress 2026-06-28: contact-patch source selection found a positive
      official Newton `pen` region. Accepted paired evidence includes
      `pen_end_bias` from the retry2 contact-patch probe,
      `pen_end_bias_train_d`, `pen_end_bias_val_c`, and `pen_end_bias_val_e`;
      `pen_end_bias_val_d` was rejected by the strict gate because the
      intervention failed. The strict preflight passed at
      `data/processed/phase08_advantage_gated_residual_preflight_v1_20260628/manifest.json`
      with `900` train records, `900` validation records, `4` accepted cells,
      `1313` accepted active frames, and no failures. This is a data gate, not
      final curiosity success.
      Progress 2026-06-28 later: started real one-hour training
      `phase08_advantage_gated_residual_adapter_v1_train_20260628` in Slurm
      job `156696` on `server02` after fresh official Newton sanity. GPU
      utilization checks showed about `97%` during the early training window.
      Still open: wait for trainer summary/checkpoint, verify elapsed time and
      mean GPU utilization, then run held-out full-video evaluation.
- [ ] Run the official Newton pen/object-family source-selection probe.
      Added `experiments/configs/run_phase08_object_family_probe_in_alloc.sh`
      for the official Newton `pen` scene with paired no-adaptation and
      scripted-feedback rollouts. `bash -n` passed. Slurm job `156688` is
      pending in tmux session `curiosity_phase08_object_probe_alloc_20260628`
      for one GPU/one day. This is diagnostic only: it may identify a harder
      object/task distribution where no-adaptation leaves room for improvement,
      but it is not training and must not be reported as a curiosity success.
      Result 2026-06-28: completed four full-video diagnostic rollouts with
      fresh official Newton sanity checks and direct visual inspection. Manual
      inspection JSON:
      `experiments/outputs/phase08_object_family_probe_v1_20260628_manual_visual_inspection.json`.
      The probe is negative for source selection: scripted feedback triggered
      `0` times in both pen cells and had lower hold than no-adaptation
      (`3.8666666666666667` versus `4.25`) while only slightly increasing lift.
      Do not admit these pen cells into advantage-gated training data.
- [ ] Run off-center/contact-patch source-selection probe.
      Added `--grasp-offset-delta-xyz` to the Newton exporter and
      `GRASP_OFFSET_DELTA_XYZ` to the allocation runner. The perturbation is
      applied before IK waypoint capture, preserves official object geometry
      and body state, and writes `candidate.task.grasp_offset_delta_xyz` plus
      `grasp_perturbation_adapter` provenance. Added
      `experiments/configs/run_phase08_contact_patch_probe_in_alloc.sh` to run
      paired no-adaptation versus guarded scripted feedback on `cube_edge_x`,
      `cube_corner_xy`, and `pen_end_bias`. Static checks passed. This remains
      open until full videos, metrics, direct visual inspection, and paired
      advantage-gate eligibility are recorded.
      Progress 2026-06-28: the retry2 probe completed and found `pen_end_bias`
      as the positive cell; cube cells were rejected or noncompetitive. Direct
      contact-sheet inspection confirmed nonblank multi-camera evidence and
      the no-adaptation failure versus guarded-feedback stable hold/lift.
- [ ] Run Phase08 advantage-gated residual held-out evaluation after training.
      Added `experiments/configs/run_phase08_advantage_gated_heldout_eval_in_alloc.sh`.
      It requires the Phase08 advantage-gated checkpoint, runs only inside a
      Slurm allocation, and evaluates held-out `pen_end_bias_heldout_center`,
      `pen_end_bias_heldout_high_y`, and `pen_end_bias_heldout_low_x` with
      full videos for `no_adaptation`, `guarded_feedback`, and
      `advantage_gated_residual`. It also validates NPZ fields, runs strict
      metrics and acceleration analysis, and reports whether the trained
      residual beats the strongest baseline without safety regression. This is
      evaluation of the residual repair, not final curiosity success; if it is
      negative, continue repair instead of stopping.
      Result 2026-06-28: the Phase08 advantage-gated residual training passed
      the one-hour real-training gate with checkpoint
      `checkpoints/phase08_advantage_gated_residual_adapter_trainer_v1_20260628/phase08_advantage_gated_residual_adapter_v1_train_20260628.pt`,
      `elapsed_seconds=3600.1261126995087`, `optimizer_steps=25629`, and mean
      GPU utilization `97.14876033057851%`. The first held-out evaluation
      attempt failed only because the NPZ validation used stale field names;
      the runner was fixed to validate `newton.panda.rigid_contact_count` and
      `newton.panda.object_body_q`, then retry1 completed all 9/9 rollouts.
      Summary:
      `experiments/outputs/phase08_advantage_gated_heldout_eval_retry1_v1_20260628_summary.json`.
      Manual visual inspection:
      `experiments/outputs/phase08_advantage_gated_heldout_eval_retry1_v1_20260628_manual_visual_inspection.json`.
      Performance status is `open_not_satisfied`: `guarded_feedback` was the
      strongest baseline on all three held-out cells, and the trained
      advantage-gated residual did not beat it on any cell without safety
      regression. Treat this as negative residual-repair evidence. Next action:
      continue with a curiosity-specific closed-loop objective/source repair or
      stronger intervention distribution; do not stop and do not call this
      curiosity training complete.
- [ ] Continue Phase08 with a real curiosity forward-model and learning-progress chain.
      Added
      `experiments/configs/build_phase08_advantage_source_compat_v1.py`,
      `experiments/configs/phase08_advantage_source_compat_v1.json`,
      `experiments/configs/run_phase08_curiosity_preflight_chain_in_alloc.sh`,
      `experiments/configs/phase08_curiosity_forward_model_preflight_v1.json`,
      `experiments/configs/phase08_curiosity_forward_model_trainer_v1.json`,
      `experiments/configs/phase08_curiosity_learning_progress_v1.json`, and
      `experiments/configs/phase08_curiosity_weighted_residual_adapter_trainer_v1.json`.
      Source compat passed at
      `data/processed/phase08_advantage_source_compat_v1_20260628/manifest.json`
      with `1800` records, four source runs, and no failures. The first
      forward-model preflight attempt exposed a missing `valid_use` config key;
      this was fixed, and retry passed at
      `data/processed/phase08_curiosity_forward_model_preflight_v1_20260628/manifest.json`
      with `1796` transitions, `898` train transitions, `898` validation
      transitions, and no failures. Started real one-hour forward-model
      training `phase08_curiosity_forward_model_v1_train_20260628` in the held
      Slurm allocation after fresh official Newton sanity; early GPU
      utilization was about `97%`. Still open: complete the one-hour forward
      model, compute learning-progress scores, train the curiosity-weighted
      residual policy for at least one GPU-hour, and evaluate it on the same
      held-out full-video strongest-baseline gate.
      Progress 2026-06-28 later: forward-model training completed with
      `real_training_result=true`, `elapsed_seconds=3600.0301122665405`,
      `optimizer_steps=24798`, checkpoint
      `checkpoints/phase08_curiosity_forward_model_v1_20260628/phase08_curiosity_forward_model_v1_train_20260628.pt`,
      initial snapshot
      `checkpoints/phase08_curiosity_forward_model_v1_20260628/phase08_curiosity_forward_model_v1_train_20260628_initial_snapshot.pt`,
      and mean GPU utilization `97.34166666666667%`. Learning-progress scoring
      passed at
      `experiments/outputs/phase08_curiosity_learning_progress_v1_20260628/curiosity_learning_progress_summary.json`
      with `1796` scores, mean learning progress `0.5845009502902926`, and no
      failures. Started real one-hour curiosity-weighted residual policy
      training `phase08_curiosity_weighted_residual_adapter_v1_train_20260628`
      in the same held allocation after fresh official Newton sanity; early GPU
      utilization was about `97%`. Still open: complete the policy checkpoint,
      verify GPU/time gates, then run held-out full-video comparison against
      no-adaptation, guarded feedback, and the non-curiosity/advantage-gated
      residual baseline.
      Progress 2026-06-29: policy training completed earlier with
      `real_training_result=true`, `elapsed_seconds=3600.114011287689`,
      `optimizer_steps=25749`, checkpoint
      `checkpoints/phase08_curiosity_weighted_residual_adapter_trainer_v1_20260628/phase08_curiosity_weighted_residual_adapter_v1_train_20260628.pt`,
      and mean GPU utilization `96.90833333333333%`. The held-out full-video
      comparison then ran in Slurm job `156696` through GPU `srun` after a
      first non-srun launch failed official Newton CUDA sanity. Valid rerun log:
      `logs/newton/phase08_curiosity_weighted_heldout_eval_v1_20260628.srun.log`;
      summary:
      `experiments/outputs/phase08_curiosity_weighted_heldout_eval_v1_20260628_summary.json`;
      report:
      `experiments/reports/2026-06-28_phase08_curiosity_weighted_heldout_eval_v1.md`;
      manual visual inspection:
      `experiments/outputs/phase08_curiosity_weighted_heldout_eval_v1_20260628_manual_visual_inspection.json`.
      Status remains `open_not_satisfied`: curiosity-weighted residual did not
      beat the strongest `guarded_feedback` baseline across the three held-out
      `pen_end_bias` cells without safety regression. Center improved over
      advantage-gated residual but still lost to guarded/no-adaptation safety
      and hold behavior; high-y and low-x failed the strongest-baseline gate.
      Next TODO: do not repeat the same supervised residual weighting target.
      Build the next closed-loop repair around preserving the successful
      baseline hold/lift prior while applying curiosity only to contact/slip
      correction windows, or collect a stronger advantage-gated source
      distribution where the intervention actually beats the paired baseline.
      Progress 2026-06-29 later: added and trained guarded-anchor repair config
      `experiments/configs/phase08_guarded_anchor_curiosity_weighted_residual_adapter_trainer_v1.json`.
      Real training passed with checkpoint
      `checkpoints/phase08_guarded_anchor_curiosity_weighted_residual_adapter_trainer_v1_20260629/phase08_guarded_anchor_curiosity_repair_v1_train_20260629.pt`,
      `elapsed_seconds=3600.1154675483704`, `optimizer_steps=25422`, and mean
      GPU utilization `97.26666666666667%`. The first held-out eval failed at
      exporter checkpoint classification allowlist; patched
      `experiments/configs/newton_panda_hydro_tiled_camera_export.py` and reran
      retry1 successfully. Retry summary:
      `experiments/outputs/phase08_guarded_anchor_heldout_eval_retry1_v1_20260629_summary.json`;
      manual visual inspection:
      `experiments/outputs/phase08_guarded_anchor_heldout_eval_retry1_v1_20260629_manual_visual_inspection.json`.
      Status remains `open_not_satisfied`: stronger anchor improved some
      safety terms but crushed hold/lift and still lost to the strongest
      `guarded_feedback` baseline. Next TODO: train selective-anchor repair
      from the previous curiosity checkpoint, not from the over-anchored
      checkpoint, preserving lift velocity while weakly anchoring only
      hold-height/stabilization in high-contact stable phases. Config:
      `experiments/configs/phase08_selective_anchor_curiosity_weighted_residual_adapter_trainer_v1.json`.
      Progress 2026-06-29 final in this sequence: selective-anchor training
      passed with checkpoint
      `checkpoints/phase08_selective_anchor_curiosity_weighted_residual_adapter_trainer_v1_20260629/phase08_selective_anchor_curiosity_repair_v1_train_20260629.pt`,
      `elapsed_seconds=3600.115711927414`, `optimizer_steps=25592`,
      validation `active_accuracy=0.9977778196334839`, validation
      `continuous_mse=0.00719881895929575`, and mean GPU utilization
      `98.13333333333334%`. Held-out eval completed with summary
      `experiments/outputs/phase08_selective_anchor_heldout_eval_v1_20260629_summary.json`,
      report
      `experiments/reports/2026-06-29_phase08_selective_anchor_heldout_eval_v1.md`,
      full rollout videos under
      `experiments/visuals/phase08_selective_anchor_eval_pen_end_bias_heldout_*_curiosity_weighted_residual_20260629/`,
      and manual visual inspection
      `experiments/outputs/phase08_selective_anchor_heldout_eval_v1_20260629_manual_visual_inspection.json`.
      Status is still `open_not_satisfied`: selective anchor recovered some
      hold/lift compared with guarded-anchor but still lost to strongest
      `guarded_feedback` and retained slip/acceleration regressions. Next TODO:
      stop treating anchor tuning as sufficient; build a guarded-hold-prior
      plus local curiosity slip/contact overlay source, and require paired
      train/validation evidence that the overlay beats guarded feedback before
      another one-hour policy update.
- [ ] Expand guarded-overlay source evidence before the next training run.
      Direct guarded-overlay source probe
      `phase08_guarded_overlay_probe_direct_v1_20260629` completed in Slurm
      allocation `156696` and wrote
      `experiments/outputs/phase08_guarded_overlay_probe_direct_v1_20260629_summary.json`.
      It is source selection only, not training and not a success claim. It
      accepted only `pen_end_bias_train_c`; both validation overlay cells were
      rejected because they lost hold/lift and had acceleration regressions.
      Therefore the next one-hour training run is forbidden until expanded
      train/validation evidence passes. Do not train on a single accepted
      train cell and do not relabel this probe as completed curiosity.
      Progress 2026-06-29: added
      `experiments/configs/run_phase08_guarded_overlay_expanded_probe_direct_in_alloc.sh`
      and launched it in tmux-held Slurm allocation `156696`, tmux window
      `phase08_overlay_expanded_direct`, with log
      `logs/newton/phase08_guarded_overlay_expanded_probe_direct_v1_20260629.srun.log`.
      It uses no held-out cells and must find multiple accepted train cells
      plus at least one accepted validation cell before an overlay-training
      preflight may be built. If it does not, continue source/control repair
      instead of doing narrow one-cell training.
      Result 2026-06-29: expanded probe completed and wrote
      `experiments/outputs/phase08_guarded_overlay_expanded_probe_direct_v1_20260629_summary.json`
      plus report
      `experiments/reports/2026-06-29_phase08_guarded_overlay_expanded_probe_direct_v1.md`.
      It accepted one validation source, `pen_end_bias_overlay_val_c0`, but
      accepted zero train sources under the strict hold/lift/slip/accel
      non-regression gate. Training remains forbidden. Next action: run a
      train-focused guarded-overlay probe around the successful validation
      offset and the earlier direct accepted `pen_end_bias_train_c` region.
      Progress 2026-06-29: added
      `experiments/configs/run_phase08_guarded_overlay_train_focus_probe_direct_in_alloc.sh`
      and launched it in Slurm allocation `156696`, tmux window
      `phase08_overlay_train_focus`, log
      `logs/newton/phase08_guarded_overlay_train_focus_probe_direct_v1_20260629.srun.log`.
      It probes five train-only offsets and must produce accepted train
      source coverage before any overlay-training preflight can start.
      Result 2026-06-29: train-focused probe completed and wrote
      `experiments/outputs/phase08_guarded_overlay_train_focus_probe_direct_v1_20260629_summary.json`
      plus report
      `experiments/reports/2026-06-29_phase08_guarded_overlay_train_focus_probe_direct_v1.md`.
      Status is `open_no_train_overlay_source_candidates`; accepted train
      source count is `0`. This confirms that direct strict-gate source
      expansion alone is insufficient.
      Progress 2026-06-29: added a stricter failure-repair source gate rather
      than training on weak evidence:
      `experiments/configs/build_phase08_guarded_overlay_failure_repair_preflight_v1.py`,
      `experiments/configs/phase08_guarded_overlay_failure_repair_preflight_v1.json`,
      and
      `experiments/configs/run_phase08_guarded_overlay_failure_repair_preflight_in_alloc.sh`.
      This gate preserves old strict non-regression when baseline succeeds,
      but when baseline fails it accepts only overlay runs that are successful,
      not dropped, meet absolute hold/lift/drop/slip/accel/contact thresholds,
      and do not regress safety. It was run inside Slurm job `156696` and
      passed, writing
      `data/processed/phase08_guarded_overlay_failure_repair_preflight_v1_20260629/manifest.json`
      with 4 accepted cells, 3 train accepted cells, 1 validation accepted
      cell, 1350 train records, 450 validation records, and 1318 accepted
      active frames. This is preflight only, not training and not success.
      Progress 2026-06-29: added source-compat and forward-model preflight
      configs:
      `experiments/configs/phase08_guarded_overlay_failure_repair_source_compat_v1.json`
      and
      `experiments/configs/phase08_guarded_overlay_curiosity_forward_model_preflight_v1.json`.
      The allocation-only chain passed and wrote
      `data/processed/phase08_guarded_overlay_failure_repair_source_compat_v1_20260629/manifest.json`
      and
      `data/processed/phase08_guarded_overlay_curiosity_forward_model_preflight_v1_20260629/manifest.json`.
      Forward preflight contains 1796 transitions, split as 1347 train and
      449 validation, with no held-out cells.
      Progress 2026-06-29: added new training configs
      `experiments/configs/phase08_guarded_overlay_curiosity_forward_model_trainer_v1.json`,
      `experiments/configs/phase08_guarded_overlay_curiosity_learning_progress_v1.json`,
      and
      `experiments/configs/phase08_guarded_overlay_curiosity_weighted_residual_adapter_trainer_v1.json`.
      Started the sequential training chain in Slurm job `156696`, tmux window
      `phase08_overlay_train_chain`, log
      `logs/newton/phase08_guarded_overlay_training_chain_v1_20260629.srun.log`.
      The chain must complete real forward-model training, learning-progress
      scoring, and real policy training before held-out evaluation can start.
      Do not mark this item or the project complete from preflight or training
      start alone.
      Result 2026-06-29: the training chain completed with exit `0`.
      Forward-model training passed with checkpoint
      `checkpoints/phase08_guarded_overlay_curiosity_forward_model_v1_20260629/phase08_guarded_overlay_curiosity_forward_model_v1_train_20260629.pt`,
      `elapsed_seconds=3600.0973856449127`, `optimizer_steps=17503`,
      validation loss `0.16814851760864258`, and mean GPU utilization
      `98.25833333333334%`. Learning-progress scoring passed with 1796 scores,
      mean bounded curiosity reward `0.7238376709891858`, and
      `not_raw_prediction_error_only=true`. Policy training passed with
      checkpoint
      `checkpoints/phase08_guarded_overlay_curiosity_weighted_residual_adapter_trainer_v1_20260629/phase08_guarded_overlay_curiosity_weighted_residual_adapter_v1_train_20260629.pt`,
      `elapsed_seconds=3600.164947986603`, `optimizer_steps=17665`, and mean
      GPU utilization `98.24166666666666%`. Validation was weak
      (`active_accuracy=0.5822222232818604`,
      `continuous_mse=0.5252048373222351`), so this is a valid training
      artifact but not success evidence.
      Progress 2026-06-29: held-out eval first attempt failed before producing
      a result because `lift_hold_feedback_residual_overlay` did not receive
      `--residual-adapter-checkpoint` through the v2 wrapper. Patched
      `experiments/configs/run_phase08_curiosity_weighted_heldout_eval_in_alloc.sh`
      to expose `CURIOSITY_CONTROLLER_MODE` and use the direct controller
      runner for residual-overlay candidate evaluation.
      Result 2026-06-29 retry1: full-video held-out eval completed with exit
      `0`, summary
      `experiments/outputs/phase08_guarded_overlay_curiosity_heldout_eval_retry1_v1_20260629_summary.json`,
      report
      `experiments/reports/2026-06-29_phase08_guarded_overlay_curiosity_heldout_eval_retry1_v1.md`,
      and manual visual inspection
      `experiments/outputs/phase08_guarded_overlay_curiosity_heldout_eval_retry1_v1_20260629_manual_visual_inspection.json`.
      Status is still `open_not_satisfied`: the new guarded-overlay curiosity
      policy succeeds and beats the strongest baseline on `heldout_center` and
      `heldout_high_y`, and beats advantage-gated residual on all three cells,
      but fails `heldout_low_x` with zero hold and large slip. Continue with
      targeted low-x repair/source expansion; do not claim final curiosity
      success.
      Progress 2026-06-29 continuation: ran a train-only repair-coverage probe
      with the latest guarded-overlay checkpoint:
      `phase08_guarded_overlay_repair_coverage_probe_direct_v1_20260629`.
      It produced full 450-frame videos for five paired train offsets and
      found useful failed-baseline repairs on `train_focus_a` and
      `train_focus_d`, while `train_focus_b`, `train_focus_c`, and
      `train_focus_e` were rejected as harmful or insufficient. Added v2
      source/preflight configs and ran allocation-only gates. The v2
      failure-repair preflight passed at
      `data/processed/phase08_guarded_overlay_failure_repair_preflight_v2_20260629/manifest.json`
      with 6 accepted cells, 5 train accepted cells, 1 validation accepted
      cell, 2250 train records, 450 validation records, and 1973 accepted
      active frames. The v2 source-compat and forward-model preflight passed
      at
      `data/processed/phase08_guarded_overlay_curiosity_forward_model_preflight_v2_20260629/manifest.json`
      with 2694 transition records and no held-out leakage.
      Progress 2026-06-29 continuation: added v2 real-training configs
      `experiments/configs/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_trainer_v2.json`,
      `experiments/configs/phase08_guarded_overlay_repair_coverage_curiosity_learning_progress_v2.json`,
      and
      `experiments/configs/phase08_guarded_overlay_repair_coverage_curiosity_weighted_residual_adapter_trainer_v2.json`.
      Started the sequential v2 training chain in Slurm job `156696`, tmux
      window `phase08_overlay_train_v2`, log
      `logs/newton/phase08_guarded_overlay_repair_coverage_training_chain_v2_20260629.srun.log`.
      It must complete real forward-model training, learning-progress scoring,
      real policy training, and then held-out full-video evaluation before any
      improvement claim is allowed.
      Progress 2026-06-29 continuation: v2 forward-model training completed
      as a real one-hour result with summary
      `experiments/outputs/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_v2_20260629/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_v2_train_20260629_summary.json`,
      checkpoint
      `checkpoints/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_v2_20260629/phase08_guarded_overlay_repair_coverage_curiosity_forward_model_v2_train_20260629.pt`,
      `elapsed_seconds=3600.151699781418`, `optimizer_steps=10537`, and
      validation loss `0.21219861507415771`. V2 learning-progress scoring
      passed with 2694 scores, mean bounded curiosity reward
      `0.5845975945337932`, train mean `0.6020250130443103`, validation mean
      `0.49746050198120656`, and `not_raw_prediction_error_only=true` at
      `experiments/outputs/phase08_guarded_overlay_repair_coverage_curiosity_learning_progress_v2_20260629/curiosity_learning_progress_summary.json`.
      V2 policy training is now running in the same allocation as
      `phase08_guarded_overlay_repair_coverage_curiosity_weighted_residual_adapter_v2_train_20260629`.
      Added and launched wait-for-policy held-out eval launcher
      `experiments/configs/launch_phase08_guarded_overlay_repair_coverage_heldout_eval_v2_wait_tmux.sh`
      in tmux window `phase08_overlay_eval_v2_wait`; it waits for a passing
      policy summary/checkpoint before running full-video held-out eval. This
      is still in progress, not a success claim.
- [ ] Add a slippery/low-contact object family.
      Examples: smooth cup, metal cylinder, laminated card, thin box, and
      small-contact-patch object. Evaluate slip, delayed contact loss, and
      over-grip/excessive-force failures.
- [ ] Add a deformable/compliant object family.
      Examples: pouch, sponge block, soft bottle, or partially filled
      container. Do not use fake tactile or synthetic T-Rex fields. Use real
      Newton/Taccel evidence under explicit namespaces and record deformation
      or contact provenance.
- [ ] Add handled or off-center object tasks.
      Include handle orientation or asymmetric center-of-mass variants.
      Evaluate torque-induced slip, wrong grasp point, and lift-success but
      hold-failure cases.
- [ ] Add fragile or safety-constrained object tasks.
      Define force/contact-proxy limits and require failure reports when an
      approach succeeds visually but violates safety.
- [ ] Produce a harder-task failure-mode comparison report.
      Compare no-adaptation, scripted feedback, residual adapter without
      curiosity, curiosity-trained policy, ablations, and serious/mainstream
      reference methods or documented blockers. The report must state whether
      curiosity improves adaptation beyond residual training and must not make
      broad generalization, T-Rex, or tactile F6 claims unless those gates have
      separately passed.
- [ ] Prove improvement before stopping.
      The final harder-task result must beat the declared baseline on the
      agreed metrics without hiding safety regressions. If the result only
      matches baseline, lacks complete videos, lacks held-out harder tasks, or
      lacks faithful mainstream-method comparison, keep the objective open and
      continue training or report the blocker.
