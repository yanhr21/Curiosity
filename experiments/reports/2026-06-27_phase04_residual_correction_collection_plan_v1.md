# Phase 04 Residual Correction Collection Plan V1

## Scope

This plan records the residual-label collection route required before training
the first residual controller-parameter adapter. It does not train a model or
claim adaptation.

## Why This Is Needed

The learned adapter should output residual controller parameters around the
official Newton scripted infant prior, not low-level torques and not a toy
policy. Training requires source rollouts with nonzero residual labels,
official sanity checks, visual evidence, strict metrics, and held-out split
enforcement.

## Current Feedback Rule

The feedback function is `_apply_scripted_feedback` in
`experiments/configs/newton_panda_hydro_tiled_camera_export.py`.

It triggers on:

- low contact count after the lift phase starts;
- online framewise object-z acceleration above threshold;
- object height drop during hold.

Default scripted-feedback grid runs were visually valid but had
`feedback_trigger_count=0`. Later ordinary-cell diagnostics produced nonzero
labels. Aggressive contact thresholds failed hold; gentler thresholds preserved
task behavior but exposed a recorded initial-settling acceleration artifact.

## Promoted Source Candidate

The first promoted source candidate is:

```text
residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006
```

It uses ordinary cell `half_low`, not a held-out cell, and ran inside allocation
`154023` through tmux session
`curiosity_next_source_alloc_20260626_232937`.

Parameter setting:

- `FEEDBACK_MIN_CONTACT_COUNT=58`;
- `FEEDBACK_ACCEL_THRESHOLD=6.5`;
- `FEEDBACK_INITIAL_LIFT_DURATION_SCALE=1.65`;
- gentle correction caps:
  `FEEDBACK_LIFT_DURATION_SCALE_MAX=1.05`,
  `FEEDBACK_HOLD_HEIGHT_STEP=0.0005`,
  `FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=0.005`,
  `FEEDBACK_STABILIZATION_STEP=0.05`,
  `FEEDBACK_STABILIZATION_MAX=0.3`;
- `PRE_RECORD_WARMUP_STEPS=15`;
- `OBJECT_MASS_KG=0.20`;
- `OBJECT_FRICTION_MU=0.35`.

Evidence:

- official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: `pass_nonblank_success_with_feedback`;
- metrics status: pass;
- final feedback trigger count: 241;
- feedback-active frames: 241;
- object dropped: false;
- contact-loss frames: 0;
- lift height: 0.15815936028957367 m;
- hold duration: 2.5333309173583984 s;
- max slip: 0.0030417809728431086 m;
- max object acceleration: 0.5063306543767194 m/s^2.

Report:
`experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle_lift165_warmup15.md`.

Manifest:
`experiments/configs/residual_label_source_manifest_v1.json`.

## Historical Diagnostics

- `residual_label_source_sensitive_feedback_half_low_20260627_030145`
  produced nonzero labels but failed hold and object-acceleration metrics.
- `residual_label_sweep_half_low_contact58_20260627_0310` produced nonzero
  labels but still failed the 2s hold gate.
- `residual_label_source_accel_sensitive_half_low_20260627_030748` preserved
  task behavior but had `feedback_trigger_count=0`.
- `residual_label_sweep_half_low_contact58_gentle_20260627_0345` and
  `residual_label_sweep_half_low_contact58_gentle_smooth_20260627_0355`
  preserved task behavior and labels but still included the initial settling
  acceleration artifact in the recorded metric window.

Peak analysis showed the non-warmup top acceleration event was at step 2,
phase 0, before feedback was active. The warmup15 source candidate resolves
that artifact without changing to a toy model or training anything.

## Promotion Rules

A diagnostic can become a residual-label source only if it passes sanity,
visual, manual, drop/contact gates, strict metrics, and produces nonzero
residual corrections on ordinary cells.

It must remain diagnostic if the threshold is intentionally aggressive, if it
triggers without preserving task gates, or if it relies on held-out cells.

## Forbidden Claims

Do not claim:

- adaptation improvement;
- policy update;
- held-out generalization;
- tactile F6;
- T-Rex compatibility.

## Training Follow-Up

The formal residual-label source runner, residual-adapter training preflight,
trainer smoke, and real one-GPU one-hour training run now pass.

Real training evidence:

- run tag: `residual_adapter_trainer_v1_train_20260627_0548`;
- summary:
  `experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548_summary.json`;
- checkpoint:
  `checkpoints/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_train_20260627_0548.pt`;
- training report:
  `experiments/reports/2026-06-27_phase04_residual_adapter_training_v1.md`;
- elapsed seconds: `3600.0302035808563`;
- optimizer steps: `32685`;
- validation loss: `6.241170922294259e-05`;
- mean GPU utilization: `99.08333333333333%`;
- generated T-Rex fields: `[]`;
- schema promotion: `blocked`.

## Next Step

Do not continue treating source mismatch as the active gate. Wire the trained
checkpoint into the Newton residual-controller evaluation path, run a
non-held-out validation rollout with visual/browser and metric gates, then run
held-out `full_low` and `empty_high` evaluation. Do not use held-out cells for
labels, training, tuning, or threshold selection.
