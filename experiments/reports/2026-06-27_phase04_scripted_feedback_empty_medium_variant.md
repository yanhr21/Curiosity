# Phase 04 Scripted Feedback Empty-Medium Variant

## Scope

This is the second scripted feedback ordinary mass/friction grid cell for Phase
04 closed-loop adaptation.

This is not a learned policy and not a curiosity result. It uses the official
Newton Panda hydro scripted infant prior plus scripted controller feedback.

## Run

- Canonical run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635`
- Duplicate noncanonical run also exists:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1630`
- Mass label: `empty`
- Friction label: `medium`
- Requested object mass: `0.08` kg
- Requested friction mu: `0.80`
- Observed body mass: `0.07999999821186066` kg
- Observed shape material mu: `0.800000011920929`
- Held-out cell: false

## Command

The canonical run used the existing tmux-held Curiosity allocation `154023`.

```bash
RUN_TAG=lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635 \
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
PHYSICS_VARIANT_LABEL=feedback_empty_medium_mass0p08_mu0p80 \
OBJECT_MASS_KG=0.08 \
OBJECT_FRICTION_MU=0.80 \
bash experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh
```

Metrics were extracted after manual visual inspection:

```bash
JOB_ID=154023 \
TMUX_SESSION=curiosity_next_source_alloc_20260626_232937 \
RUN_TAG=lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635 \
WINDOW_NAME=lift_hold_feedback_metrics \
BASELINE_NAME=scripted_feedback_adaptation_grasp_lift \
MASS_LABEL=empty \
FRICTION_LABEL=medium \
POSE_SEED=feedback_empty_medium \
MANUAL_VISUAL_INSPECTION=pass \
bash experiments/configs/launch_lift_hold_metrics_tmux.sh
```

## Gate Evidence

- Fresh official Newton sanity:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_fresh_newton_sensor_contact_sanity.json`
- Visual validation:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_visual_validation.json`
- Manual visual inspection:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_manual_visual_inspection.json`
- Metrics:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635_metrics.json`
- Rollout:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635.npz`

Direct visual paths:

- Contact sheet:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635/contact_sheet.png`
- Frame browser:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635/frame_browser.html`
- Key inspected frames:
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635/frame_0225.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635/frame_0315.png`
  `experiments/visuals/lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635/frame_0359.png`

## Result

Visual gate: pass.

Strict metrics: fail, only because `object_accel_above_threshold`.

Metric values:

- lift height: `0.160252645611763` m;
- hold duration: `2.8333306312561035` s;
- max slip: `0.0035626504907293466` m;
- drop height loss: `0.0` m;
- contact loss frames: `0`;
- max contact proxy: `62.0`;
- max object acceleration: `8.308392358032673` m/s^2.

Scripted feedback summary:

- feedback trigger count: `0`;
- feedback reason labels: only `none`.

## Interpretation

The feedback grid path is runnable and visually valid on the empty/medium
mass/friction variant. Real Newton mass/friction values were applied before
finalize.

This cell still does not prove adaptation improvement because no feedback event
was triggered. Continue running the remaining ordinary grid cells before making
adaptation claims.
