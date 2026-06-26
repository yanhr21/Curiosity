# Phase 04 Scripted Feedback Full-High Variant

## Scope

This is the seventh and final scripted feedback ordinary mass/friction grid
cell for Phase 04 closed-loop adaptation.

This is not a learned policy and not a curiosity result. It uses the official
Newton Panda hydro scripted infant prior plus scripted controller feedback.

## Run

- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820`
- Mass label: `full`
- Friction label: `high`
- Requested object mass: `0.35` kg
- Requested friction mu: `1.20`
- Observed body mass: `0.3499999940395355` kg
- Observed shape material mu: `1.2000000476837158`
- Held-out cell: false

## Gate Evidence

- Fresh official Newton sanity:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_fresh_newton_sensor_contact_sanity.json`
- Visual validation:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_visual_validation.json`
- Manual visual inspection:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_manual_visual_inspection.json`
- Metrics:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820_metrics.json`
- Rollout:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820.npz`

Direct visual paths:

- Contact sheet:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820/contact_sheet.png`
- Frame browser:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820/frame_browser.html`
- Key inspected frames:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820/frame_0225.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820/frame_0315.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820/frame_0359.png`

## Result

Visual gate: pass.

Strict metrics: fail, only because `object_accel_above_threshold`.

Metric values:

- lift height: `0.1542350798845291` m;
- hold duration: `2.7833306789398193` s;
- max slip: `0.0032414356600358944` m;
- drop height loss: `0.0` m;
- contact loss frames: `0`;
- max contact proxy: `63.0`;
- max object acceleration: `8.308127374067027` m/s^2.

Scripted feedback summary:

- feedback trigger count: `0`;
- feedback reason labels: only `none`.

## Interpretation

The feedback grid path is runnable and visually valid on the full/high
mass/friction variant. Real Newton mass/friction values were applied before
finalize.

The ordinary scripted feedback grid is complete. No ordinary cell triggered
the current feedback rule, so this grid does not support an adaptation
improvement claim. The next valid Phase 04 work is either held-out evaluation
or revising the feedback trigger design with the failure evidence preserved.
