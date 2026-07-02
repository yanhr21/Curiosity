# Phase 07 Water-Cup Video Baseline/Curiosity First Cell

## Scope

This is the first Phase 07 harder-task progress record after the no-early-exit
rule was added. It is not a final curiosity success claim.

The run establishes:

- a harder variable water-cup task spec;
- complete rollout GIF video export in the Newton camera exporter;
- one new harder physical cell with residual baseline and curiosity-weighted
  policy evaluated under the same mass/friction settings;
- no-adaptation and scripted-feedback video baselines for the same physical
  cell;
- manual visual inspection for both videos.

The visual fill-cue condition is not implemented yet, so this is a harder
physical mass/friction cell, not the full variable-fill family.

## Task Spec

Spec:

- `experiments/configs/variable_water_cup_harder_task_v1.json`.

The spec defines five fill/mass levels, three friction levels,
truthful/hidden/misleading visual-cue conditions, train/validation/held-out
splits, required policies, metrics, and no-early-exit completion gates.

## Video Export

Implemented in:

- `experiments/configs/newton_panda_hydro_tiled_camera_export.py`;
- `experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh`;
- `experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh`.

New arguments/environment:

- `--video-frame-stride` / `VIDEO_FRAME_STRIDE`;
- `--video-fps` / `VIDEO_FPS`.

When enabled, the exporter writes:

- `rollout_video.gif`;
- dense `video_frames/video_frame_*.png`;
- `video_export` metadata in the summary and run-status JSON.

Syntax and lightweight checks passed:

```text
jq empty experiments/configs/variable_water_cup_harder_task_v1.json
envs/residual_adapter/.venv/bin/python -m py_compile experiments/configs/newton_panda_hydro_tiled_camera_export.py
bash -n experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh
git diff --check
```

## Runs

Allocation:

```text
tmux_session=curiosity_forward_alloc_20260627_105456
slurm_job_id=154290
node=server37
```

Cell:

```text
cell=three_quarter_low_physical
object_mass_kg=0.29
object_friction_mu=0.35
num_steps=360
video_frame_stride=1
video_fps=18
```

### No-Adaptation Scripted Prior

Artifacts:

- summary:
  `experiments/outputs/phase07_watercup_three_quarter_low_noadapt_video_20260627_summary.json`;
- run status:
  `experiments/outputs/phase07_watercup_three_quarter_low_noadapt_video_20260627_run_status.json`;
- manual visual inspection:
  `experiments/outputs/phase07_watercup_three_quarter_low_noadapt_video_20260627_manual_visual_inspection.json`;
- rollout video:
  `experiments/visuals/phase07_watercup_three_quarter_low_noadapt_video_20260627/rollout_video.gif`.

Result:

- fresh official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: pass;
- video export: pass, 360 frames;
- success: true;
- final lift: 0.16070151329040527 m;
- hold: 3.1 s;
- drop from max: 0.0 m;
- max xy drift: 0.009837419725954533 m;
- contact count mean: 50.02777777777778.

The no-adaptation run does not compute the same feedback acceleration observer,
so its default `candidate.controller.feedback_observed_object_accel_m_s2=0`
must not be treated as a comparable safety advantage.

### Scripted Feedback

Artifacts:

- summary:
  `experiments/outputs/phase07_watercup_three_quarter_low_scripted_feedback_video_20260627_summary.json`;
- run status:
  `experiments/outputs/phase07_watercup_three_quarter_low_scripted_feedback_video_20260627_run_status.json`;
- manual visual inspection:
  `experiments/outputs/phase07_watercup_three_quarter_low_scripted_feedback_video_20260627_manual_visual_inspection.json`;
- rollout video:
  `experiments/visuals/phase07_watercup_three_quarter_low_scripted_feedback_video_20260627/rollout_video.gif`.

Result:

- fresh official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: pass;
- video export: pass, 360 frames;
- success: true;
- final lift: 0.1552736610174179 m;
- hold: 2.5 s;
- drop from max: 0.0 m;
- max xy drift: 0.01256692223250866 m;
- contact count mean: 50.330555555555556;
- acceleration proxy max: 0.48623085021972656.

### Residual Baseline

Command:

```text
RUN_TAG=phase07_watercup_three_quarter_low_residual_baseline_video_rerun_20260627 JOB_ID=154290 TMUX_SESSION=curiosity_forward_alloc_20260627_105456 WINDOW_NAME=phase07_tq_low_residual_video_rerun SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold_learned_residual FINAL_HOLD_DURATION=2.5 PHYSICS_VARIANT_LABEL=phase07_three_quarter_low_video_baseline_rerun OBJECT_MASS_KG=0.29 OBJECT_FRICTION_MU=0.35 FEEDBACK_MIN_CONTACT_COUNT=58 FEEDBACK_ACCEL_THRESHOLD=6.5 FEEDBACK_INITIAL_LIFT_DURATION_SCALE=1.65 FEEDBACK_LIFT_DURATION_SCALE_MAX=1.05 FEEDBACK_HOLD_HEIGHT_STEP=0.0005 FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=0.005 FEEDBACK_STABILIZATION_STEP=0.05 FEEDBACK_STABILIZATION_MAX=0.3 PRE_RECORD_WARMUP_STEPS=15 RESIDUAL_ADAPTER_CHECKPOINT=/public/home/yanhongru/Curiosity/checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt RESIDUAL_ADAPTER_ACTIVE_THRESHOLD=0.5 NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 VIDEO_FRAME_STRIDE=1 VIDEO_FPS=18 bash experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh
```

Artifacts:

- log:
  `logs/newton/phase07_watercup_three_quarter_low_residual_baseline_video_rerun_20260627.log`;
- summary:
  `experiments/outputs/phase07_watercup_three_quarter_low_residual_baseline_video_rerun_20260627_summary.json`;
- run status:
  `experiments/outputs/phase07_watercup_three_quarter_low_residual_baseline_video_rerun_20260627_run_status.json`;
- manual visual inspection:
  `experiments/outputs/phase07_watercup_three_quarter_low_residual_baseline_video_rerun_20260627_manual_visual_inspection.json`;
- rollout video:
  `experiments/visuals/phase07_watercup_three_quarter_low_residual_baseline_video_rerun_20260627/rollout_video.gif`.

Result:

- fresh official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: pass;
- video export: pass, 360 frames;
- success: true;
- final lift: 0.15535058081150055 m;
- hold: 2.5 s;
- drop from max: 0.0 m;
- max xy drift: 0.012301909737288952 m;
- contact count mean: 50.25;
- acceleration proxy max: 0.5592405796051025.

### Curiosity-Weighted Policy

Command:

```text
RUN_TAG=phase07_watercup_three_quarter_low_curiosity_weighted_video_20260627 JOB_ID=154290 TMUX_SESSION=curiosity_forward_alloc_20260627_105456 WINDOW_NAME=phase07_tq_low_curiosity_video SCENE=cube TRACKED_OBJECT=existing_cup_asset CONTROLLER_MODE=lift_hold_learned_residual FINAL_HOLD_DURATION=2.5 PHYSICS_VARIANT_LABEL=phase07_three_quarter_low_curiosity_weighted OBJECT_MASS_KG=0.29 OBJECT_FRICTION_MU=0.35 FEEDBACK_MIN_CONTACT_COUNT=58 FEEDBACK_ACCEL_THRESHOLD=6.5 FEEDBACK_INITIAL_LIFT_DURATION_SCALE=1.65 FEEDBACK_LIFT_DURATION_SCALE_MAX=1.05 FEEDBACK_HOLD_HEIGHT_STEP=0.0005 FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=0.005 FEEDBACK_STABILIZATION_STEP=0.05 FEEDBACK_STABILIZATION_MAX=0.3 PRE_RECORD_WARMUP_STEPS=15 RESIDUAL_ADAPTER_CHECKPOINT=/public/home/yanhongru/Curiosity/checkpoints/curiosity_weighted_residual_adapter_trainer_v1_20260627/curiosity_weighted_residual_adapter_v1_train_20260627.pt RESIDUAL_ADAPTER_ACTIVE_THRESHOLD=0.5 NUM_STEPS=360 SAMPLE_STEPS=0,45,90,135,180,225,270,315,359 VIDEO_FRAME_STRIDE=1 VIDEO_FPS=18 bash experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh
```

Artifacts:

- log:
  `logs/newton/phase07_watercup_three_quarter_low_curiosity_weighted_video_20260627.log`;
- summary:
  `experiments/outputs/phase07_watercup_three_quarter_low_curiosity_weighted_video_20260627_summary.json`;
- run status:
  `experiments/outputs/phase07_watercup_three_quarter_low_curiosity_weighted_video_20260627_run_status.json`;
- manual visual inspection:
  `experiments/outputs/phase07_watercup_three_quarter_low_curiosity_weighted_video_20260627_manual_visual_inspection.json`;
- rollout video:
  `experiments/visuals/phase07_watercup_three_quarter_low_curiosity_weighted_video_20260627/rollout_video.gif`.

Result:

- fresh official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: pass;
- video export: pass, 360 frames;
- success: true;
- final lift: 0.1556989997625351 m;
- hold: 2.5 s;
- drop from max: 0.0 m;
- max xy drift: 0.01036224327981472 m;
- contact count mean: 50.68055555555556;
- acceleration proxy max: 1.031738519668579.

## Comparison

| Policy | Success | Final lift m | Hold s | Drop m | Max xy drift m | Contact mean | Accel proxy max | Video |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no-adaptation scripted prior | true | 0.16070151329040527 | 3.1 | 0.0 | 0.009837419725954533 | 50.02777777777778 | N/A | 360-frame GIF |
| scripted feedback | true | 0.1552736610174179 | 2.5 | 0.0 | 0.01256692223250866 | 50.330555555555556 | 0.48623085021972656 | 360-frame GIF |
| residual baseline | true | 0.15535058081150055 | 2.5 | 0.0 | 0.012301909737288952 | 50.25 | 0.5592405796051025 | 360-frame GIF |
| curiosity-weighted | true | 0.1556989997625351 | 2.5 | 0.0 | 0.01036224327981472 | 50.68055555555556 | 1.031738519668579 | 360-frame GIF |

The curiosity-weighted run has slightly higher lift and lower xy drift on this
single harder physical cell than the no-curiosity residual baseline, but it
has a worse acceleration proxy. The no-adaptation scripted prior also succeeds
with higher lift and longer hold on this cell. This is not enough to claim
curiosity improvement. It is evidence that the video path and first Phase 07
side-by-side evaluation are working.

## Status

Phase 07 remains incomplete. Missing requirements include:

- full visual fill-cue implementation;
- complete closed-loop curiosity training beyond the current supervised
  curiosity-weighted residual checkpoint;
- more held-out harder cells;
- no-adaptation and scripted-feedback video baselines for more Phase 07 cells;
- curiosity ablations;
- serious/mainstream method comparison or documented faithful blocker;
- evidence that curiosity beats the declared baseline without safety
  regression.
