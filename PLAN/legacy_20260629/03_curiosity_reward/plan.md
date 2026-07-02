# Phase 03: Curiosity Reward

## Goal

Define and implement intrinsic objectives around physical prediction, not raw
pixel prediction.

## Reward Components

- object-motion prediction error;
- contact prediction error;
- unexpected object acceleration or slip proxy;
- lift-response mismatch under expected mass;
- bounded impact/useful-change reward;
- safety penalty for excessive force or unstable motion;
- no-op penalty;
- learning progress over time.

## Required Design

Curiosity must target physical learning progress, not raw RGB novelty. The
intrinsic reward should use bounded and controllable prediction improvement:

```text
intrinsic_reward =
  learning_progress(object/contact/tactile prediction)
+ controllable_disagreement
+ bounded_useful_change
- safety_penalty
- no_op_penalty
- excessive_force_penalty
```

The first forward models should predict object pose delta, object velocity,
contact count/proxy, slip or contact-loss risk, tactile-marker response when
available, and success/failure risk. Raw prediction error can be logged, but it
must not be the only reward because chaotic collisions and unstable drops can
produce large errors without useful learning.

## Required Ablations

- no curiosity;
- random intrinsic reward;
- object-motion-only curiosity;
- contact-only curiosity;
- tactile-only curiosity;
- vision+tactile curiosity;
- shuffled tactile;
- delayed tactile.

## Completion Criteria

- Reward spec exists.
- Components are logged separately.
- Reward can be evaluated on baseline rollouts before it is used for policy
  adaptation.
- A learned forward model is trained on real Newton-native rollout records and
  passes validation on held-out ordinary cells.
- A learning-progress signal is computed from the learned forward model over
  temporally separated data or frozen/checkpointed model snapshots, not only
  from replay heuristics.
- Intrinsic reward is used for an actual policy/adaptation update inside the
  Newton-native residual-controller framework.
- Held-out mass/friction results show whether curiosity-driven adaptation
  improves over no-curiosity residual adaptation without hiding safety
  failures.

## Completed Replay Evaluation V1

2026-06-27: implemented and ran the first Phase 03 replay evaluator on the
validated Phase 02 cup mass/friction grid.

- Spec: `docs/curiosity_reward_spec_v1.md`.
- Config: `experiments/configs/curiosity_reward_baseline_replay_v1.json`.
- Evaluator: `experiments/configs/evaluate_curiosity_reward_baseline_replay.py`.
- Launcher: `experiments/configs/launch_curiosity_reward_baseline_replay_tmux.sh`.
- Compute runner: `experiments/configs/run_curiosity_reward_baseline_replay_in_alloc.sh`.
- Output JSON: `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.json`.
- Output CSV: `experiments/outputs/curiosity_reward_baseline_replay_v1_20260627.csv`.
- Log: `logs/newton/curiosity_reward_baseline_replay_v1_20260627.log`.

Execution used existing tmux-held allocation `154023` on `server56`, activated
the prebuilt local shared-filesystem Newton venv at `envs/newton/.venv`, and
reread `AGENTS.md` inside the compute job.

V1 is a diagnostic replay reward evaluator only. It trains no model, creates no
placeholder VQ-VAE/world model, and does not claim exact T-Rex schema
compatibility. It uses Newton contact count as `newton.contact_proxy_only`
because tactile marker fields are not present in the Phase 02 rollouts.

Replay gate result:

- status: pass;
- evaluated rollouts: 9;
- held-out cells included: `full_low`, `empty_high`;
- required gates checked per rollout: fresh official Newton sanity, camera
  summary, automated visual validation, manual visual inspection, and Phase 02
  lift-hold metrics;
- ablations logged: no curiosity, random intrinsic, object-motion-only,
  contact-only, tactile-only via contact proxy, vision+tactile via object plus
  contact proxy, shuffled tactile, delayed tactile.

This clears the baseline-rollout reward-shape gate before policy adaptation.
It does not clear the later learned forward-model or true tactile-marker
representation requirement.

## Forward-Model Target Contract V1

2026-06-27: added the first learned forward-model target contract without
implementing or claiming a learned world model.

- Spec: `docs/residual_adapter_forward_model_contract_v1.md`.
- Config: `experiments/configs/residual_adapter_forward_model_contract_v1.json`.
- Status: `target_contract_ready_training_not_started`.

Defined targets:

- object pose delta;
- object velocity;
- contact proxy next step;
- slip risk;
- contact-loss risk;
- lift-response residual;
- success/failure risk;
- tactile-marker response, blocked until real `taccel.marker.*` evidence is
  added.

This satisfies the target-definition gate for learned curiosity, but not the
training gate. No placeholder MLP, VQ-VAE, Transformer, or world model has been
created.

## Required Learned Curiosity Training V1

The Phase 03 replay evaluator is not enough. Complete curiosity training must
include a real learned forward-model path and a policy/adaptation update that
uses intrinsic reward.

The first implementation should stay Newton-native and use the already
validated residual-label/source-runner data. It must not pretend to be T-Rex,
VQ-VAE, or a generic world model. A simple supervised forward model is allowed
only if it is explicitly named as the Newton-native curiosity forward model,
uses the documented source columns, and is evaluated against held-out cells;
it must not be represented as a faithful T-Rex or VQ-VAE implementation.

Training stages:

1. Build a curiosity training manifest from
   `data/processed/residual_label_source_runner_v1_20260627/manifest.json` and
   the held-out-safe split from
   `data/processed/residual_adapter_training_preflight_v1_20260627/manifest.json`.
2. Train a learned forward model to predict:
   object pose delta, object velocity, contact proxy next step, slip risk,
   contact-loss risk, lift-response residual, and success/failure risk.
3. Compute learning-progress and controllable-disagreement scores from model
   improvement or ensemble disagreement. Do not use raw prediction error alone.
4. Train or fine-tune a residual-controller policy with extrinsic lift-hold
   reward plus bounded curiosity reward.
5. Evaluate on held-out `full_low` and `empty_high`, plus the Phase 07 harder
   tasks once their baselines exist.
6. Report ablations:
   no curiosity, random intrinsic reward, object-only curiosity, contact-only
   curiosity, vision/contact curiosity, shuffled contact, delayed contact, and
   no learning-progress term.

Required gates:

- fresh official Newton sanity before compute-side training/evaluation;
- run inside a tmux-held allocation using prebuilt `envs/` venvs;
- no held-out cells in training, hyperparameter selection, or label-source
  construction;
- direct visual paths for success and failure cases;
- strict metrics for lift, hold, slip, drop, contact loss, and acceleration;
- `generated_trex_fields=[]` and `schema_promotion=blocked`;
- explicit report stating whether the result is curiosity training, residual
  training, or only diagnostic replay.

Do not claim curiosity training complete until all of these gates pass.

## Completed Learned Curiosity Training V1

2026-06-27: completed the first Newton-native learned curiosity training chain.

Primary report:

- `experiments/reports/2026-06-27_phase03_curiosity_training_v1.md`.

Completed stages:

1. Built the held-out-safe curiosity transition manifest from residual-label
   source records.
2. Trained `newton_native_curiosity_forward_model_v1` for one hour in tmux-held
   Slurm allocation `154290`.
3. Computed learning-progress curiosity scores from the initial model snapshot
   and trained checkpoint, not raw prediction error alone.
4. Fine-tuned the existing Newton-native residual adapter with bounded
   curiosity weights. This was a supervised fine-tune, not RL.
5. Evaluated the curiosity-weighted residual adapter on held-out `full_low` and
   `empty_high` cup cells with fresh official Newton sanity, automated visual
   validation, manual visual inspection, strict lift-hold metrics, and direct
   visual paths.

Key artifacts:

- forward checkpoint:
  `checkpoints/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_train_20260627.pt`;
- learning-progress summary:
  `experiments/outputs/curiosity_learning_progress_v1_20260627/curiosity_learning_progress_summary.json`;
- curiosity-weighted residual checkpoint:
  `checkpoints/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627.pt`;
- full-low held-out summary:
  `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_summary.json`;
- full-low manual visual inspection:
  `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_manual_visual_inspection.json`;
- empty-high held-out summary:
  `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_summary.json`;
- empty-high manual visual inspection:
  `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_manual_visual_inspection.json`.

Decision:

The curiosity-weighted policy is stable and passes both V1 held-out cells, but
it does not demonstrate improvement over the no-curiosity residual baseline.
Do not claim curiosity improves adaptation beyond residual training from this
V1 result. The correct claim is that the training/evaluation pipeline is valid
and ready for stronger ablations or harder held-out tasks.
