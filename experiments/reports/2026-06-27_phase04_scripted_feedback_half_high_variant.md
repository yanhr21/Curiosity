# Phase 04 Scripted Feedback Half-High Variant

## Scope

This is the fifth scripted feedback ordinary mass/friction grid cell for Phase
04 closed-loop adaptation.

This is not a learned policy and not a curiosity result. It uses the official
Newton Panda hydro scripted infant prior plus scripted controller feedback.

## Run

- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745`
- Mass label: `half`
- Friction label: `high`
- Requested object mass: `0.20` kg
- Requested friction mu: `1.20`
- Observed body mass: `0.20000000298023224` kg
- Observed shape material mu: `1.2000000476837158`
- Held-out cell: false

## Gate Evidence

- Fresh official Newton sanity:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745_fresh_newton_sensor_contact_sanity.json`
- Visual validation:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745_visual_validation.json`
- Manual visual inspection:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745_manual_visual_inspection.json`
- Metrics:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745_metrics.json`
- Rollout:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745.npz`

Direct visual paths:

- Contact sheet:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745/contact_sheet.png`
- Frame browser:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745/frame_browser.html`
- Key inspected frames:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745/frame_0225.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745/frame_0315.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745/frame_0359.png`

## Result

Visual gate: pass.

Strict metrics: fail, only because `object_accel_above_threshold`.

Metric values:

- lift height: `0.15871746838092804` m;
- hold duration: `2.816663980484009` s;
- max slip: `0.0034686670254336456` m;
- drop height loss: `4.32133674621582e-06` m;
- contact loss frames: `0`;
- max contact proxy: `61.0`;
- max object acceleration: `8.308550048022228` m/s^2.

Scripted feedback summary:

- feedback trigger count: `0`;
- feedback reason labels: only `none`.

## Interpretation

The feedback grid path is runnable and visually valid on the half/high
mass/friction variant. Real Newton mass/friction values were applied before
finalize.

This cell still does not prove adaptation improvement because no feedback event
was triggered. Continue running the remaining ordinary grid cells before making
adaptation claims.
