# Phase 04 Residual Adapter Held-Out Evaluation V1

## Scope

This report records held-out evaluation of the trained Newton-native residual
controller adapter on `full_low` and `empty_high`. These cells were reserved
from residual-label collection and training. This is not a T-Rex result and
does not claim broad object-family generalization.

## Checkpoint

```text
checkpoint=checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt
controller_mode=lift_hold_learned_residual
method=newton_native_residual_controller_adapter_v1
generated_trex_fields=[]
schema_promotion=blocked
```

## Full/Low Held-Out

Run:

```bash
RUN_TAG=residual_adapter_eval_v1_full_low_heldout_20260627_0613 \
WINDOW_NAME=residual_adapter_eval_full_low \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
PHYSICS_VARIANT_LABEL=learned_residual_full_low_heldout \
OBJECT_MASS_KG=0.35 \
OBJECT_FRICTION_MU=0.35 \
bash experiments/configs/launch_residual_adapter_evaluation_tmux.sh
```

Evidence:

- fresh official Newton sanity:
  `experiments/outputs/residual_adapter_eval_v1_full_low_heldout_20260627_0613_fresh_newton_sensor_contact_sanity.json`;
- summary:
  `experiments/outputs/residual_adapter_eval_v1_full_low_heldout_20260627_0613_summary.json`;
- visual validation:
  `experiments/outputs/residual_adapter_eval_v1_full_low_heldout_20260627_0613_visual_validation.json`;
- manual visual inspection:
  `experiments/outputs/residual_adapter_eval_v1_full_low_heldout_20260627_0613_manual_visual_inspection.json`;
- metrics:
  `experiments/outputs/residual_adapter_eval_v1_full_low_heldout_20260627_0613_metrics.json`;
- frame browser:
  `experiments/visuals/residual_adapter_eval_v1_full_low_heldout_20260627_0613/frame_browser.html`;
- contact sheet:
  `experiments/visuals/residual_adapter_eval_v1_full_low_heldout_20260627_0613/contact_sheet.png`.

Result:

- fresh official Newton sanity: pass;
- visual validation: pass;
- manual visual inspection: `pass_nonblank_success_learned_residual`;
- metrics status: pass;
- lift height: `0.1548849195241928` m;
- hold duration: `2.499997615814209` s;
- max slip: `0.0034206882392378247` m;
- contact-loss frames: `0`;
- max contact proxy: `61`;
- max object acceleration: `1.5345948979069628` m/s^2;
- object not dropped: true.

Baselines on the same held-out cell:

- no adaptation:
  `lift_hold_no_adaptation_scripted_baseline_v1_cup_full_low_prefinalize_20260627_1350`
  failed full metrics only on `object_accel_above_threshold`, with
  max object acceleration `8.308390712127508` m/s^2;
- scripted feedback:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845`
  failed full metrics only on `object_accel_above_threshold`, with
  max object acceleration `8.308707788010144` m/s^2 and
  `feedback_trigger_count=0`.

## Empty/High Held-Out

Run:

```bash
RUN_TAG=residual_adapter_eval_v1_empty_high_heldout_20260627_0620 \
WINDOW_NAME=residual_adapter_eval_empty_high \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
PHYSICS_VARIANT_LABEL=learned_residual_empty_high_heldout \
OBJECT_MASS_KG=0.08 \
OBJECT_FRICTION_MU=1.2 \
bash experiments/configs/launch_residual_adapter_evaluation_tmux.sh
```

Evidence:

- fresh official Newton sanity:
  `experiments/outputs/residual_adapter_eval_v1_empty_high_heldout_20260627_0620_fresh_newton_sensor_contact_sanity.json`;
- summary:
  `experiments/outputs/residual_adapter_eval_v1_empty_high_heldout_20260627_0620_summary.json`;
- visual validation:
  `experiments/outputs/residual_adapter_eval_v1_empty_high_heldout_20260627_0620_visual_validation.json`;
- manual visual inspection:
  `experiments/outputs/residual_adapter_eval_v1_empty_high_heldout_20260627_0620_manual_visual_inspection.json`;
- metrics:
  `experiments/outputs/residual_adapter_eval_v1_empty_high_heldout_20260627_0620_metrics.json`;
- frame browser:
  `experiments/visuals/residual_adapter_eval_v1_empty_high_heldout_20260627_0620/frame_browser.html`;
- contact sheet:
  `experiments/visuals/residual_adapter_eval_v1_empty_high_heldout_20260627_0620/contact_sheet.png`.

Result:

- fresh official Newton sanity: pass;
- visual validation: pass;
- manual visual inspection: `pass_nonblank_success_learned_residual`;
- metrics status: pass;
- lift height: `0.1613951474428177` m;
- hold duration: `2.566664218902588` s;
- max slip: `0.003700697622575275` m;
- contact-loss frames: `0`;
- max contact proxy: `62`;
- max object acceleration: `0.4686260874870734` m/s^2;
- object not dropped: true.

Baselines on the same held-out cell:

- no adaptation:
  `lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_high_prefinalize_20260627_1445`
  failed full metrics only on `object_accel_above_threshold`, with
  max object acceleration `8.308498000056417` m/s^2;
- scripted feedback:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955`
  failed full metrics only on `object_accel_above_threshold`, with
  max object acceleration `8.308498000056417` m/s^2 and
  `feedback_trigger_count=0`.

## Interpretation

The trained residual adapter passes the current full lift-hold metrics on both
reserved held-out cells, while the no-adaptation and scripted-feedback
baselines fail the same full metrics only on object acceleration. Under the
current two-cell held-out cup benchmark, this is valid evidence of learned
residual-controller improvement over those baselines.

This is still a narrow claim. It does not prove broad object-family
generalization, tactile F6 availability, T-Rex compatibility, or curiosity
policy learning beyond this Newton-native residual-controller experiment.
