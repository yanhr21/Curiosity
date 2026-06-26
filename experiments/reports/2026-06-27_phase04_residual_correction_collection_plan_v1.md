# Phase 04 Residual Correction Collection Plan V1

## Scope

This plan defines the next data-collection step required before training the
first residual controller-parameter adapter. It also records the first executed
diagnostic after the plan was created. It does not train a model or claim
adaptation.

## Why This Is Needed

The residual adapter training readiness audit is blocked because all
scripted-feedback runs have `feedback_trigger_count=0`. Current data therefore
contains no nonzero residual controller-parameter corrections.

Training now would teach a no-op residual adapter.

## Current Feedback Rule

The current feedback function is
`_apply_scripted_feedback` in
`experiments/configs/newton_panda_hydro_tiled_camera_export.py`.

It triggers on:

- low contact count after the lift phase starts;
- online framewise object-z acceleration above threshold;
- object height drop during hold.

Observed evidence:

- contact counts are high enough;
- height drop is zero;
- strict metrics fail only on object acceleration;
- online feedback still never triggered.

## Proposed Diagnostic Collection

Plan file:
`experiments/configs/residual_correction_collection_plan_v1.json`.

First diagnostic route:

- name: `accel_sensitive_training_diagnostic_v1`;
- purpose: collect nonzero residual corrections, not train;
- candidate acceleration threshold: `5.5` m/s^2;
- keep default contact/drop gates;
- keep held-out `full_low` and `empty_high` untouched;
- first ordinary diagnostic cells: `empty_low`, `half_medium`, `full_high`.

Required outputs:

- `candidate.controller.feedback_active > 0`;
- `candidate.controller.feedback_trigger_count > 0`;
- nonzero feedback residual fields;
- fresh official Newton sanity;
- visual validation;
- manual visual inspection;
- lift-hold metrics;
- direct visual paths.

## Executed Diagnostics

Run tag:
`residual_label_source_sensitive_feedback_half_low_20260627_030145`.

This run used ordinary cell `half_low`, not a held-out cell. It executed inside
allocation `154023` through tmux session
`curiosity_next_source_alloc_20260626_232937`.

Parameter change actually tested:

- `FEEDBACK_MIN_CONTACT_COUNT=64`;
- `FEEDBACK_ACCEL_THRESHOLD=6.5`;
- `OBJECT_MASS_KG=0.20`;
- `OBJECT_FRICTION_MU=0.35`.

Evidence:

- official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: `pass_nonblank_but_task_failure`;
- final feedback trigger count: 241;
- feedback-active frames: 241;
- feedback reason: `low_contact_count`;
- object dropped: false;
- contact-loss frames: 0;
- metrics status: fail;
- failure reasons: `hold_duration_below_threshold`,
  `object_accel_above_threshold`.

Report:
`experiments/reports/2026-06-27_phase04_residual_label_source_sensitive_feedback_half_low.md`.

Interpretation: this diagnostic proves nonzero residual labels can be produced
through the official Newton rollout path, but the contact threshold is too
aggressive to promote as a training-label source. The next step is a bounded
ordinary-cell threshold sweep, not training.

Additional sweep evidence now exists:

- `residual_label_sweep_half_low_contact58_20260627_0310` lowered the contact
  threshold from 64 to 58. It still produced nonzero labels
  (`feedback_trigger_count=241`) but failed the 2s hold gate, so it is not
  promoted.
- `residual_label_source_accel_sensitive_half_low_20260627_030748` used
  `FEEDBACK_ACCEL_THRESHOLD=5.5` and default contact threshold 20. It preserved
  lift/hold/drop/contact behavior, but `feedback_trigger_count=0`, so it is
  not a residual-label source.
- Accel-sensitive report:
  `experiments/reports/2026-06-27_phase04_residual_label_source_accel_sensitive_half_low.md`.
- `residual_label_sweep_half_low_contact58_gentle_20260627_0345` keeps the
  contact58 trigger but caps correction magnitude. It preserves lift, hold,
  drop, contact-loss, visual, and manual gates while producing
  `feedback_trigger_count=241`. Strict metrics still fail on
  `object_accel_above_threshold`.
- Gentle contact58 report:
  `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle.md`.

Current interpretation: `contact58_gentle` is the best current residual-label
candidate but is not fully promoted because the strict object-acceleration gate
still fails.

## Promotion Rules

The diagnostic can become a residual-label source only if it passes sanity,
visual, manual, drop/contact gates and produces nonzero residual corrections on
ordinary cells.

It must remain diagnostic if the threshold is intentionally aggressive or if it
triggers without improving metrics.

It must block if it still produces `feedback_trigger_count=0`, uses held-out
cells for labels, fails visual/sanity gates, or drops the object.

## Forbidden Claims

Do not claim:

- learned adapter training;
- adaptation improvement;
- policy update;
- tactile F6;
- T-Rex compatibility.

## Next Step

After these diagnostics, the next aligned action is to reduce object
acceleration around the `contact58_gentle` candidate while preserving nonzero
feedback, lift, hold, drop, contact, visual, and manual gates. Do not use
held-out cells for label collection.
