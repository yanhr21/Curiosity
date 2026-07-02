# Phase 03 TODO: Curiosity Reward

- [x] Write post-pivot curiosity reward spec.
      Evidence: `docs/curiosity_reward_spec_v1.md`.
- [x] Add object-motion prediction error.
      Evidence: replay evaluator logs constant-velocity diagnostic object
      prediction error.
- [x] Add contact prediction error.
      Evidence: replay evaluator logs contact-persistence diagnostic error from
      `newton.panda.rigid_contact_count`.
- [x] Add bounded impact/useful-change reward.
      Evidence: replay evaluator logs clipped useful lift progress while
      contact is present.
- [x] Add safety/excessive-force penalty.
      Evidence: replay evaluator logs acceleration, drop, and contact-force
      proxy penalties.
- [x] Add no-op penalty.
      Evidence: replay evaluator logs active-command frames with no contact and
      no object motion.
- [x] Add learning-progress metric.
      Evidence: replay evaluator logs a replay-window learning-progress proxy;
      this is diagnostic only, not policy learning.
- [x] Prefer learning progress and bounded useful change over raw prediction
      error before using curiosity for policy updates.
      Evidence: `docs/curiosity_reward_spec_v1.md` and
      `experiments/configs/curiosity_reward_baseline_replay_v1.json`.
- [x] Add learned forward-model targets for object pose delta, object velocity,
      contact proxy, slip/contact-loss risk, and tactile-marker response when
      tactile evidence exists. Current V1 only adds diagnostic replay predictors
      and must not be treated as a learned world model.
      Evidence: `docs/residual_adapter_forward_model_contract_v1.md` and
      `experiments/configs/residual_adapter_forward_model_contract_v1.json`
      define target fields for object pose delta, object velocity, contact
      proxy next step, slip risk, contact-loss risk, lift-response residual,
      success/failure risk, and a blocked tactile-marker target for future real
      `taccel.marker.*` evidence. This is a target contract only; no learned
      model has been trained or claimed.
- [x] Run curiosity ablations: no curiosity, random intrinsic reward,
      object-motion-only, contact-only, tactile-only, vision+tactile, shuffled
      tactile, and delayed tactile.
      Evidence: `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`.
- [x] Validate reward on baseline rollouts before policy adaptation.
      Evidence: compute run in `logs/newton/curiosity_reward_baseline_replay_v1_20260627.log`
      passed all 9 Phase 02 rollout gates before evaluation.
- [x] Build the learned curiosity training manifest.
      Use the existing residual-label source runner and preflight split:
      `data/processed/residual_label_source_runner_v1_20260627/manifest.json`
      and
      `data/processed/residual_adapter_training_preflight_v1_20260627/manifest.json`.
      Preserve held-out `full_low` and `empty_high`; do not use them for
      forward-model training, policy training, hyperparameter selection, or
      label-source construction. Output belongs under `data/processed/`.
      Evidence:
      `experiments/configs/curiosity_forward_model_preflight_v1.json`,
      `experiments/configs/build_curiosity_forward_model_preflight.py`,
      `experiments/configs/launch_curiosity_forward_model_preflight_tmux.sh`,
      `logs/newton/curiosity_forward_model_preflight_v1_20260627_110711.log`,
      and
      `data/processed/curiosity_forward_model_preflight_v1_20260627/manifest.json`.
      Result: status pass, fresh official Newton sanity pass, 1795 transition
      records, 1436 train transitions, 359 validation transitions,
      `training_started=false`, `generated_trex_fields=[]`, and
      `schema_promotion=blocked`.
- [x] Train the Newton-native learned curiosity forward model.
      Targets must include object pose delta, object velocity, contact proxy
      next step, slip risk, contact-loss risk, lift-response residual, and
      success/failure risk. This must run in a tmux-held compute allocation
      with fresh official Newton sanity and prebuilt `envs/` venvs. It must
      report validation loss per target and must keep
      `generated_trex_fields=[]` and `schema_promotion=blocked`.
      Evidence:
      `experiments/configs/curiosity_forward_model_trainer_v1.json`,
      `experiments/configs/train_curiosity_forward_model_v1.py`, and
      `experiments/configs/launch_curiosity_forward_model_trainer_tmux.sh`.
      Smoke diagnostic passed in
      `experiments/outputs/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_smoke_20260627_110806_summary.json`
      with `smoke_diagnostic_only=true` and no checkpoint. Real training ran
      for one hour under tmux session
      `curiosity_forward_alloc_20260627_105456`, job `154290`, run tag
      `curiosity_forward_model_v1_train_20260627`. Fresh official Newton
      sanity passed, final checkpoint was written to
      `checkpoints/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_train_20260627.pt`,
      summary to
      `experiments/outputs/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_train_20260627_summary.json`,
      and GPU utilization record to
      `experiments/outputs/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_train_20260627_gpu_utilization.json`.
      Result: status pass, `real_training_result=true`, elapsed 3600.10 s,
      30959 optimizer steps, validation loss 0.10705970227718353,
      mean GPU utilization 99.0%, `generated_trex_fields=[]`, and
      `schema_promotion=blocked`.
- [x] Compute real learning-progress or controllable-disagreement curiosity.
      The signal must come from model improvement over time, frozen model
      snapshots, or an explicitly configured ensemble. Raw prediction error
      alone is not enough because it can reward drops, collisions, or noise.
      Evidence:
      `experiments/configs/curiosity_learning_progress_v1.json`,
      `experiments/configs/compute_curiosity_learning_progress_v1.py`,
      `experiments/configs/run_curiosity_learning_progress_in_alloc.sh`, and
      `experiments/configs/launch_curiosity_learning_progress_tmux.sh`.
      Scoring run `curiosity_learning_progress_v1_20260627` passed fresh
      official Newton sanity and wrote
      `experiments/outputs/curiosity_learning_progress_v1_20260627/curiosity_learning_progress_summary.json`
      and
      `experiments/outputs/curiosity_learning_progress_v1_20260627/curiosity_learning_progress_scores.csv`.
      Result: status pass, 1795 scores, mean learning progress
      0.6249577405558987, mean bounded curiosity reward
      0.6250462863355618, train split score count 1436, validation split
      score count 359, `not_raw_prediction_error_only=true`,
      `policy_updated=false`, `generated_trex_fields=[]`, and
      `schema_promotion=blocked`.
- [x] Train or fine-tune a residual-controller policy with bounded curiosity
      reward.
      Compare against the existing trained residual adapter without curiosity.
      Do not call this T-Rex, VQ-VAE, or a generic world model. Record command,
      config, checkpoint, environment, GPU utilization, output paths, and
      sanity checks.
      Evidence:
      `experiments/configs/curiosity_weighted_residual_adapter_trainer_v1.json`,
      `experiments/configs/train_curiosity_weighted_residual_adapter_v1.py`,
      `experiments/configs/run_curiosity_weighted_residual_adapter_trainer_in_alloc.sh`,
      and
      `experiments/configs/launch_curiosity_weighted_residual_adapter_trainer_tmux.sh`.
      This is explicitly a curiosity-weighted supervised fine-tune from the
      existing Newton-native residual checkpoint, not an RL algorithm. Smoke
      diagnostic passed in
      `experiments/outputs/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_smoke_20260627_summary.json`.
      Real training run
      `curiosity_weighted_residual_adapter_v1_train_20260627` passed fresh
      official Newton sanity, ran for one hour, wrote checkpoint
      `checkpoints/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627.pt`,
      summary
      `experiments/outputs/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627_summary.json`,
      and GPU utilization record
      `experiments/outputs/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627_gpu_utilization.json`.
      Result: status pass, `real_training_result=true`, elapsed 3600.06 s,
      32480 optimizer steps, validation loss 6.058789585949853e-05,
      active accuracy 1.0, mean GPU utilization 99.07563025210084%,
      `not_rl_algorithm=true`, `generated_trex_fields=[]`, and
      `schema_promotion=blocked`.
- [x] Evaluate curiosity-trained policy on held-out cup cells.
      Required held-out cells: `full_low` and `empty_high`. Evaluation must
      pass fresh official Newton sanity, automated visual validation, manual
      visual inspection, strict lift/hold/slip/drop/contact/acceleration
      metrics, and direct visual path reporting. Compare against:
      no-adaptation, scripted feedback, trained residual adapter without
      curiosity, random intrinsic reward, object-only curiosity, contact-only
      curiosity, shuffled contact, delayed contact, and no learning-progress
      term.
      Evidence:
      `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_summary.json`,
      `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_run_status.json`,
      `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_visual_validation.json`,
      `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_manual_visual_inspection.json`,
      `experiments/visuals/curiosity_weighted_eval_full_low_heldout_rerun_20260627/contact_sheet.png`,
      `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_summary.json`,
      `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_run_status.json`,
      `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_visual_validation.json`,
      `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_manual_visual_inspection.json`,
      and
      `experiments/visuals/curiosity_weighted_eval_empty_high_heldout_20260627/contact_sheet.png`.
      Both held-out cells passed fresh official Newton sanity, automated
      visual validation, manual visual inspection, and strict lift-hold task
      metrics. Full-low result: success true, final lift
      0.15441761910915375 m, hold 2.5 s, drop from max 0.0 m. Empty-high
      result: success true, final lift 0.161421999335289 m, hold
      2.566666666666667 s, drop from max 0.0 m.
- [x] Decide whether curiosity improves adaptation beyond residual training.
      A valid claim requires held-out improvement without hiding safety
      failures. If results only match the residual adapter or improve one
      metric while worsening drop/slip/force, record the limitation and do not
      claim curiosity success.
      Decision: do not claim improvement beyond residual training. The
      curiosity-weighted residual adapter passes both held-out cells, but the
      direct residual baseline also passes both. Full-low baseline final lift
      is 0.1548849195241928 m versus curiosity-weighted 0.15441761910915375 m;
      full-low peak acceleration proxy improves from 0.8005321025848389 to
      0.6401360034942627. Empty-high is essentially tied: baseline final lift
      0.1613951474428177 m versus curiosity-weighted 0.161421999335289 m,
      with similar acceleration proxy. This supports "curiosity-weighted
      training is valid and stable on the held-out cells" but not "curiosity
      improves adaptation beyond the no-curiosity residual adapter."
