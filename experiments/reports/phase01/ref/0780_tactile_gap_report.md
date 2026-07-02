# Phase 01 Reference Video Tactile Gap Report

Date: 2026-07-01

Reference video:
`0780e5ec3fdb26b63ae63de0f49f07c4.mp4`

Extracted inspection assets:
`experiments/visuals/phase01/ref/0780/contact_sheet.jpg`

Inspection metadata:
`experiments/visuals/phase01/ref/0780/inspect.json`

## 1. What the reference video contains

The reference video is a 24 s, 30 FPS, 720-frame, 2846x1510 MP4. It is not just
a rollout video. It is a synchronized tactile diagnostic view. Each sampled
frame compares three material/sensor settings:

- `metal rigid 82 fps`
- `wood rigid 82 fps`
- `rubber soft FEM 5 fps`

Each column includes:

- visual interaction scene with Panda hydro tactile contact patch overlay;
- left/right tactile pad maps;
- contact pressure or compression heatmaps;
- normal force `Fn`;
- tangential/shear force `Ft`;
- shear direction vectors and pen/rod geometry overlays;
- grip force time series;
- shear-on-object time series and left/right shear direction agreement;
- contact area and mean penetration/compression curves;
- for soft FEM, rod bend and height dynamics.

This means the target tactile standard is multi-material, left/right pad,
spatially resolved, force-aware, shear-aware, time-synchronized, and visually
auditable. It also explicitly separates rigid hydroelastic contact from soft FEM
deformation behavior.

## 2. What the current Phase 01 tactile actually is

The active Phase 01 data used by the curiosity policy candidate is much weaker.
The current training rows contain:

- `newton.panda.rigid_contact_count`
- `newton.object.body_q.z`
- controller phase and command fields;
- object mass and friction labels;
- `candidate.modality.vision_available_mask`
- `candidate.modality.contact_available_mask`
- feedback action targets;
- next-step object/contact/slip risk targets.

The current local-advantage CSV header confirms there is no tactile image,
deformation field, per-finger pad map, left/right force vector, shear vector,
pressure map, marker flow, or soft-body compression state in the actual policy
training data.

Current tactile proxy is therefore best described as:

> coarse Newton contact count + contact availability mask + next-step contact
> risk labels.

It is not equivalent to the tactile shown in the reference video.

## 3. Current curiosity design

Current Phase 01 curiosity has three components.

### 3.1 Newton-native forward model

Config: `experiments/configs/phase01/fwd_train.json`

Model:

- GRU sequence forward model;
- input dim 10;
- hidden dim 512;
- 2 GRU layers;
- targets are next-step transition/contact quantities:
  - `target.object.delta_z_next`
  - `target.object.velocity_z_next`
  - `target.contact.count_next`
  - `target.contact.delta_count_next`
  - `target.contact_loss_risk_next`
  - `target.slip_risk_next`

Evidence:
`experiments/outputs/phase01/core/fwd/p01_fwd_a1r2_20260630_002030_summary.json`

Result:

- real training: true;
- elapsed: 3600.12 s;
- optimizer steps: 28762;
- checkpoint:
  `checkpoints/phase01/core/fwd/p01_fwd_a1r2_20260630_002030.pt`;
- mean GPU utilization: 68.88%;
- validation loss: 0.01977 normalized MSE.

This is a valid Newton-native forward-model component, but it is not a tactile
world model and not proof of curiosity policy success.

### 3.2 Learning-progress curiosity score

Config: `experiments/configs/phase01/learning_progress.json`

The score compares the initial and trained forward-model prediction errors. The
reward is bounded learning progress, with penalties for no-op/contact-loss/slip
risk. It is an offline score over recorded transitions. It does not update a
policy by itself.

Latest local-advantage evidence:
`experiments/outputs/phase01/core/local_adv_lp/curiosity_learning_progress_summary.json`

Result:

- status: pass;
- score count: 768;
- mean learning progress: 0.7639;
- mean bounded curiosity reward: 0.5006;
- policy updated: false.

### 3.3 Curiosity-weighted residual controller

Config:
`experiments/configs/phase01/resid_curiosity_local_adv_train.json`

Model:

- GRU residual controller;
- input dim 10;
- hidden dim 512;
- 2 GRU layers;
- active head predicts whether feedback should be active;
- continuous head predicts:
  - lift velocity scale;
  - hold height offset;
  - stabilization extension.

Approximate parameter count: 2.38 M parameters.

The fifth attempt loads the no-curiosity residual baseline checkpoint and
fine-tunes it with sample weights:

- base weight: 1.0;
- curiosity scale: 0.08;
- min/max sample weights: 0.85 to 1.15;
- base-policy distillation anchor enabled to reduce drift.

Evidence:
`experiments/outputs/phase01/core/resid/curiosity_local_adv/p01_resid_cur_local_adv_a5_20260630_1028_summary.json`

Result:

- real training: true;
- elapsed: 3600.03 s;
- optimizer steps: 16922;
- checkpoint:
  `checkpoints/phase01/core/resid/curiosity_local_adv/p01_resid_cur_local_adv_a5_20260630_1028.pt`;
- mean GPU utilization: 99.18%;
- validation active accuracy: 0.4792.

## 4. Exploration / progress so far

Positive technical progress:

1. Phase 00 created a fixed 15-cell Newton asset/evaluation basis.
2. Phase 01 created train/validation/held-out separation.
3. Official Newton sanity checks were kept as gates.
4. A no-adaptation/scripted-feedback/no-curiosity-residual baseline contract was
   declared before curiosity success claims.
5. A real one-hour Newton-native forward model was trained.
6. Learning-progress scores were generated from forward-model error changes.
7. A learned no-curiosity residual baseline was trained.
8. Five real one-hour curiosity-weighted residual policy candidates were run and
   evaluated.
9. Full held-out rollout evidence and MP4/contact-sheet outputs were generated
   for the later attempts.
10. The project correctly classified the latest result as negative/incomplete
    instead of claiming success.

Negative result ledger:
`experiments/reports/phase01/core/training_attempts.json`

Current status:

- `negative_real_one_hour_attempt_count = 5`
- `stop_before_attempt_6 = true`

No sixth real one-hour curiosity policy training attempt is allowed without user
instruction.

## 5. Why the current method does not work

### 5.1 The tactile representation is too weak

The reference video shows tactile as dense pad-level pressure/compression/shear
signals with material-specific temporal behavior. Current Phase 01 only exposes
contact count and contact masks. This loses:

- contact location on each pad;
- pressure distribution;
- force direction;
- normal/tangential force separation;
- left/right tactile asymmetry;
- contact area;
- penetration/compression;
- shear/slip precursor geometry;
- soft-body deformation state;
- material compliance signatures.

Therefore the model cannot learn the same contact physics that the reference
video makes visible. It can only correlate a scalar contact count with a small
set of scripted controller corrections.

### 5.2 Curiosity is offline reweighting, not true closed-loop exploration

The current score is computed from recorded transitions. The policy is then
fine-tuned in a supervised way. This can prioritize samples, but it does not
let the agent actively choose new contacts, probe uncertain areas, or collect
new tactile evidence based on prediction error.

This is why the project records `policy_updated=false` for learning-progress
scoring and marks the residual training as a policy candidate, not as complete
closed-loop curiosity.

### 5.3 The base task leaves little room for improvement

The strongest non-curiosity baselines are already strong on held-out cells. The
latest curiosity candidate succeeded on 4/4 held-out cells, but it did not beat
the best baseline safely.

Latest comparison:
`experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_local_adv_cmp_a5_20260630_1340_comparison.json`

Result:

- `positive_curiosity_result = false`
- `safety_regression_cell_count = 4`
- `useful_improvement_count = 2`
- classification: `negative_or_incomplete_candidate`

Per-cell failures were mainly slip, acceleration, or hold-duration regressions
against the strongest baseline.

### 5.4 The residual action space is narrow

The learned residual controller can only adjust:

- whether feedback is active;
- lift velocity scale;
- hold height offset;
- stabilization extension.

It cannot learn rich contact probing motions, finger repositioning, tactile
servoing, pressure balancing, or shear-minimizing exploratory behavior. With
the current tactile signal, the model also has no dense state input that would
justify those richer actions.

### 5.5 Dataset diversity is not enough for curiosity

Current local-advantage repair data admitted only four short segments:

- 3 train segments;
- 1 validation segment;
- 576 train rows;
- 192 validation rows;
- 58 train active feedback frames;
- 29 validation active feedback frames.

This is not enough to learn a general tactile curiosity behavior, especially
when the tactile input is only a scalar contact proxy.

## 6. What must change to match the reference video

The next tactile target should be upgraded from contact-count proxy to a
reference-video-aligned representation:

1. left/right pad tactile frames;
2. pressure or compression heatmap per pad;
3. normal force `Fn` per pad;
4. tangential/shear force `Ft` per pad;
5. shear direction vectors;
6. contact area and center of pressure;
7. mean/max penetration or compression;
8. temporal derivatives for pressure, shear, and area;
9. material/compliance labels or inferred embedding;
10. visual scene and tactile channels synchronized at every policy step;
11. tactile-mask training where vision can be masked while tactile remains
    online;
12. explicit balance gates so policy performance must hold under:
    - vision+tactile;
    - tactile-only masked vision;
    - vision-only ablation;
    - mismatched/noisy tactile ablation.

The curiosity objective should then move from offline sample reweighting toward
closed-loop active probing:

- forward model predicts tactile + object transition, not only object z/contact
  count;
- intrinsic reward is bounded learning progress over tactile prediction and
  task-relevant contact dynamics;
- exploration is constrained by safety/contact stability;
- policy can choose probe/regrasp/pressure-balancing actions, not just residual
  lift/hold tweaks;
- success requires held-out improvement over the strongest non-curiosity
  baseline without safety regression.

## 7. Bottom line

The current work has produced useful infrastructure and honest negative
evidence, but it has not achieved the intended curiosity learning claim.

The main reason is not just a bad hyperparameter. The representation and action
space are below the standard shown in the reference video. Current "tactile" is
a coarse contact proxy; the reference tactile is dense, bilateral,
force/shear/compliance-aware, and time-synchronized. Until Phase 01 upgrades to
that level of tactile evidence and uses it in closed-loop exploration, the
curiosity policy is unlikely to beat a strong scripted/no-curiosity baseline on
harder grasping tasks.
