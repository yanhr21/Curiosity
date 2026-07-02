# Phase 03 Curiosity Training V1

## Scope

This report records the first complete Newton-native learned curiosity
training chain:

1. train a Newton-native curiosity forward model;
2. compute learning-progress curiosity rewards from initial and trained model
   snapshots;
3. fine-tune the existing Newton-native residual controller with curiosity
   weights;
4. evaluate the curiosity-weighted residual controller on held-out cup cells.

This is not T-Rex, not VQ-VAE, not a generic world model, and not RL. The
policy update is a curiosity-weighted supervised fine-tune of the existing
Newton-native residual adapter.

## Allocation

```text
tmux_session=curiosity_forward_alloc_20260627_105456
slurm_job_id=154290
node=server37
job_name=curiosity_forward_1gpu_1day
```

All compute-side runs reread `AGENTS.md`, used prebuilt venvs under `envs/`,
and ran fresh official Newton sensor/contact sanity before downstream work.

## Forward Model

Command:

```text
ALLOW_REAL_TRAINING=1 RUN_TAG=curiosity_forward_model_v1_train_20260627 RUN_MODE=train JOB_ID=154290 TMUX_SESSION=curiosity_forward_alloc_20260627_105456 WINDOW_NAME=curiosity_forward_train_1h bash experiments/configs/launch_curiosity_forward_model_trainer_tmux.sh
```

Artifacts:

- log:
  `logs/newton/curiosity_forward_model_v1_train_20260627.log`;
- fresh sanity:
  `experiments/outputs/curiosity_forward_model_v1_train_20260627_fresh_newton_sensor_contact_sanity.json`;
- initial snapshot:
  `checkpoints/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_train_20260627_initial_snapshot.pt`;
- final checkpoint:
  `checkpoints/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_train_20260627.pt`;
- summary:
  `experiments/outputs/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_train_20260627_summary.json`;
- GPU utilization:
  `experiments/outputs/curiosity_forward_model_v1_20260627/curiosity_forward_model_v1_train_20260627_gpu_utilization.json`.

Result:

- status: pass;
- real training result: true;
- elapsed seconds: 3600.097998380661;
- optimizer steps: 30959;
- validation loss: 0.10705970227718353;
- mean GPU utilization: 99.0%;
- `generated_trex_fields=[]`;
- `schema_promotion=blocked`.

Validation per-target normalized MSE:

- object delta z: 0.007835397496819496;
- object velocity z: 0.005930713377892971;
- rigid contact count next: 0.15082810819149017;
- contact delta count next: 0.6826992630958557;
- contact-loss risk next: 0.0007851792615838349;
- slip risk next: 0.0006978070014156401;
- lift-response residual next: 0.007065158803015947;
- success final: 0.0006359718972817063.

## Learning Progress

Command:

```text
RUN_TAG=curiosity_learning_progress_v1_20260627 JOB_ID=154290 TMUX_SESSION=curiosity_forward_alloc_20260627_105456 WINDOW_NAME=curiosity_learning_progress bash experiments/configs/launch_curiosity_learning_progress_tmux.sh
```

Artifacts:

- log:
  `logs/newton/curiosity_learning_progress_v1_20260627.log`;
- summary:
  `experiments/outputs/curiosity_learning_progress_v1_20260627/curiosity_learning_progress_summary.json`;
- scores:
  `experiments/outputs/curiosity_learning_progress_v1_20260627/curiosity_learning_progress_scores.csv`.

Result:

- status: pass;
- score count: 1795;
- mean learning progress: 0.6249577405558987;
- mean bounded curiosity reward: 0.6250462863355618;
- train split: 1436 scores, mean reward 0.6395169950803755;
- validation split: 359 scores, mean reward 0.5671634513563103;
- `not_raw_prediction_error_only=true`;
- `policy_updated=false`;
- `generated_trex_fields=[]`;
- `schema_promotion=blocked`.

The score is based on improvement from the initial forward-model snapshot to
the trained checkpoint, not raw prediction error alone.

## Curiosity-Weighted Residual Fine-Tune

Command:

```text
ALLOW_REAL_TRAINING=1 RUN_TAG=curiosity_weighted_residual_adapter_v1_train_20260627 RUN_MODE=train JOB_ID=154290 TMUX_SESSION=curiosity_forward_alloc_20260627_105456 WINDOW_NAME=curiosity_weighted_train_1h bash experiments/configs/launch_curiosity_weighted_residual_adapter_trainer_tmux.sh
```

Artifacts:

- log:
  `logs/newton/curiosity_weighted_residual_adapter_v1_train_20260627.log`;
- checkpoint:
  `checkpoints/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627.pt`;
- summary:
  `experiments/outputs/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627_summary.json`;
- GPU utilization:
  `experiments/outputs/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627_gpu_utilization.json`.

Result:

- status: pass;
- real training result: true;
- elapsed seconds: 3600.0575489997864;
- optimizer steps: 32480;
- validation loss: 6.058789585949853e-05;
- weighted validation loss: 6.529475649585947e-05;
- active accuracy: 1.0;
- train score coverage: 0.9972222222222222;
- validation score coverage: 0.9972222222222222;
- mean GPU utilization: 99.07563025210084%;
- `not_rl_algorithm=true`;
- `generated_trex_fields=[]`;
- `schema_promotion=blocked`.

## Held-Out Evaluation

The first full-low launch failed before producing an evaluation result because
the evaluator rejected the new checkpoint classification
`newton_native_curiosity_weighted_residual_controller_adapter_v1_checkpoint`.
The loader was updated to accept this Newton-native curiosity-weighted residual
checkpoint classification. The corrected full-low rerun and empty-high run
both passed.

Full-low command:

```text
RUN_TAG=curiosity_weighted_eval_full_low_heldout_rerun_20260627 JOB_ID=154290 TMUX_SESSION=curiosity_forward_alloc_20260627_105456 WINDOW_NAME=curiosity_weighted_eval_full_low_rerun SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold_learned_residual FINAL_HOLD_DURATION=2.5 PHYSICS_VARIANT_LABEL=curiosity_weighted_full_low_heldout_rerun OBJECT_MASS_KG=0.35 OBJECT_FRICTION_MU=0.35 FEEDBACK_MIN_CONTACT_COUNT=58 FEEDBACK_ACCEL_THRESHOLD=6.5 FEEDBACK_INITIAL_LIFT_DURATION_SCALE=1.65 FEEDBACK_LIFT_DURATION_SCALE_MAX=1.05 FEEDBACK_HOLD_HEIGHT_STEP=0.0005 FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=0.005 FEEDBACK_STABILIZATION_STEP=0.05 FEEDBACK_STABILIZATION_MAX=0.3 PRE_RECORD_WARMUP_STEPS=15 RESIDUAL_ADAPTER_CHECKPOINT=/public/home/yanhongru/Curiosity/checkpoints/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627.pt RESIDUAL_ADAPTER_ACTIVE_THRESHOLD=0.5 NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 bash experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh
```

Full-low artifacts:

- log:
  `logs/newton/curiosity_weighted_eval_full_low_heldout_rerun_20260627.log`;
- summary:
  `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_summary.json`;
- run status:
  `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_run_status.json`;
- automated visual validation:
  `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_visual_validation.json`;
- manual visual inspection:
  `experiments/outputs/curiosity_weighted_eval_full_low_heldout_rerun_20260627_manual_visual_inspection.json`;
- contact sheet:
  `experiments/visuals/curiosity_weighted_eval_full_low_heldout_rerun_20260627/contact_sheet.png`.

Full-low result:

- mass/friction: 0.35 kg, mu 0.35;
- success: true;
- final lift: 0.15441761910915375 m;
- hold: 2.5 s;
- drop from max: 0.0 m;
- max xy drift: 0.014668777585029602 m;
- contact count mean: 51.522222222222226;
- acceleration proxy max: 0.6401360034942627;
- automated visual validation: pass;
- manual visual inspection: pass.

Empty-high command:

```text
RUN_TAG=curiosity_weighted_eval_empty_high_heldout_20260627 JOB_ID=154290 TMUX_SESSION=curiosity_forward_alloc_20260627_105456 WINDOW_NAME=curiosity_weighted_eval_empty_high SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold_learned_residual FINAL_HOLD_DURATION=2.5 PHYSICS_VARIANT_LABEL=curiosity_weighted_empty_high_heldout OBJECT_MASS_KG=0.08 OBJECT_FRICTION_MU=1.2 FEEDBACK_MIN_CONTACT_COUNT=58 FEEDBACK_ACCEL_THRESHOLD=6.5 FEEDBACK_INITIAL_LIFT_DURATION_SCALE=1.65 FEEDBACK_LIFT_DURATION_SCALE_MAX=1.05 FEEDBACK_HOLD_HEIGHT_STEP=0.0005 FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=0.005 FEEDBACK_STABILIZATION_STEP=0.05 FEEDBACK_STABILIZATION_MAX=0.3 PRE_RECORD_WARMUP_STEPS=15 RESIDUAL_ADAPTER_CHECKPOINT=/public/home/yanhongru/Curiosity/checkpoints/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627.pt RESIDUAL_ADAPTER_ACTIVE_THRESHOLD=0.5 NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 bash experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh
```

Empty-high artifacts:

- log:
  `logs/newton/curiosity_weighted_eval_empty_high_heldout_20260627.log`;
- summary:
  `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_summary.json`;
- run status:
  `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_run_status.json`;
- automated visual validation:
  `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_visual_validation.json`;
- manual visual inspection:
  `experiments/outputs/curiosity_weighted_eval_empty_high_heldout_20260627_manual_visual_inspection.json`;
- contact sheet:
  `experiments/visuals/curiosity_weighted_eval_empty_high_heldout_20260627/contact_sheet.png`.

Empty-high result:

- mass/friction: 0.08 kg, mu 1.2;
- success: true;
- final lift: 0.161421999335289 m;
- hold: 2.566666666666667 s;
- drop from max: 0.0 m;
- max xy drift: 0.0133469607681036 m;
- contact count mean: 54.03333333333333;
- acceleration proxy max: 0.4577457904815674;
- automated visual validation: pass;
- manual visual inspection: pass.

## Baseline Comparison

Direct baseline is the no-curiosity residual adapter checkpoint evaluated in
Phase 04:

- full-low summary:
  `experiments/outputs/residual_adapter_eval_v1_full_low_heldout_20260627_0613_summary.json`;
- empty-high summary:
  `experiments/outputs/residual_adapter_eval_v1_empty_high_heldout_20260627_0620_summary.json`.

| Cell | Method | Success | Final lift m | Hold s | Drop m | Max xy drift m | Contact mean | Accel proxy max |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_low | residual baseline | true | 0.1548849195241928 | 2.5 | 0.0 | 0.011278465390205383 | 50.03333333333333 | 0.8005321025848389 |
| full_low | curiosity-weighted | true | 0.15441761910915375 | 2.5 | 0.0 | 0.014668777585029602 | 51.522222222222226 | 0.6401360034942627 |
| empty_high | residual baseline | true | 0.1613951474428177 | 2.566666666666667 | 0.0 | 0.013317948207259178 | 54.24722222222222 | 0.4572629928588867 |
| empty_high | curiosity-weighted | true | 0.161421999335289 | 2.566666666666667 | 0.0 | 0.0133469607681036 | 54.03333333333333 | 0.4577457904815674 |

## Decision

The curiosity-weighted training chain is complete and valid for this
Newton-native V1: it trained a forward model, computed learning-progress
scores, used those scores in a residual-adapter fine-tune, and passed both
held-out evaluation cells with automated and manual visual gates.

It does not prove that curiosity improves adaptation beyond the no-curiosity
residual adapter. The held-out success rate is tied. Full-low has slightly
lower lift and higher xy drift but lower acceleration proxy; empty-high is
essentially tied. The correct claim is:

```text
curiosity-weighted Newton-native residual training is stable and passes the
held-out V1 cells, but V1 does not demonstrate improvement over the existing
no-curiosity residual baseline.
```
