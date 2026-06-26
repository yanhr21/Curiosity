# Phase 04 Scripted Feedback Full-Medium Variant

## Scope

This is the sixth scripted feedback ordinary mass/friction grid cell for Phase
04 closed-loop adaptation.

This is not a learned policy and not a curiosity result. It uses the official
Newton Panda hydro scripted infant prior plus scripted controller feedback.

## Run

- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805`
- Mass label: `full`
- Friction label: `medium`
- Requested object mass: `0.35` kg
- Requested friction mu: `0.80`
- Observed body mass: `0.3499999940395355` kg
- Observed shape material mu: `0.800000011920929`
- Held-out cell: false

## Gate Evidence

- Fresh official Newton sanity:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805_fresh_newton_sensor_contact_sanity.json`
- Visual validation:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805_visual_validation.json`
- Manual visual inspection:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805_manual_visual_inspection.json`
- Metrics:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805_metrics.json`
- Rollout:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805.npz`

Direct visual paths:

- Contact sheet:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805/contact_sheet.png`
- Frame browser:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805/frame_browser.html`
- Key inspected frames:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805/frame_0225.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805/frame_0315.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805/frame_0359.png`

## Result

Visual gate: pass.

Strict metrics: fail, only because `object_accel_above_threshold`.

Metric values:

- lift height: `0.15381594002246857` m;
- hold duration: `2.7833306789398193` s;
- max slip: `0.003283848177821986` m;
- drop height loss: `0.0` m;
- contact loss frames: `0`;
- max contact proxy: `62.0`;
- max object acceleration: `8.308707937632189` m/s^2.

Scripted feedback summary:

- feedback trigger count: `0`;
- feedback reason labels: only `none`.

## Interpretation

The feedback grid path is runnable and visually valid on the full/medium
mass/friction variant. Real Newton mass/friction values were applied before
finalize.

This cell still does not prove adaptation improvement because no feedback event
was triggered. Continue running the remaining ordinary grid cell before making
adaptation claims.
