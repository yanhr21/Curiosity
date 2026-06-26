# Phase 04 Residual Adapter Training Readiness V1

## Scope

This report audits whether the first learned residual controller-parameter
adapter can be trained now. It does not start training and does not create a
model.

## Result

Status: blocked, training not started. A best current residual-label candidate
now exists, but it is not fully promoted because the strict object-acceleration
gate still fails.

The original blocker was that every scripted-feedback evaluation recorded
`feedback_trigger_count=0`. A follow-up ordinary-cell diagnostic,
`residual_label_source_sensitive_feedback_half_low_20260627_030145`, produced
nonzero residual corrections with final `feedback_trigger_count=241`. It still
cannot be used as a training source because metrics fail on
`hold_duration_below_threshold` and `object_accel_above_threshold`.

The best current follow-up candidate,
`residual_label_sweep_half_low_contact58_gentle_20260627_0345`, preserves lift,
hold, drop, contact-loss, visual, and manual gates while producing
`feedback_trigger_count=241`, but strict metrics still fail on
`object_accel_above_threshold`.

## Ready Evidence

- Official Newton Panda hydro scripted infant prior is available and visually
  validated.
- Real Newton mass/friction grid exists across nominal, ordinary, and held-out
  cells.
- Held-out `full_low` and `empty_high` are preserved for generalization.
- Phase 05 contact source manifest passed with `source_run_count=10`,
  `record_count=3600`, `generated_trex_fields=[]`, and
  `schema_promotion=blocked`.
- Phase 03 curiosity replay diagnostic passed with `rollout_count=9`.
- Residual adapter and forward-model target contract exists:
  `experiments/configs/residual_adapter_forward_model_contract_v1.json`.
- First nonzero residual diagnostic exists:
  `experiments/reports/2026-06-27_phase04_residual_label_source_sensitive_feedback_half_low.md`.
- Best current residual-label candidate exists:
  `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle.md`.

## Blocking Gaps

- No fully promoted nonzero residual demonstration:
  the best current contact58-gentle diagnostic produces nonzero feedback labels
  and preserves lift/hold/drop/contact gates, but still fails the strict
  object-acceleration gate.
- Training on the current promoted labels would still teach no valid
  adaptation source.
- No approved residual-adapter training implementation exists yet.
- No compute-side residual-adapter training runner exists with source-gate
  checks, held-out split enforcement, official Newton sanity, and report
  generation.

## Allowed Next Routes

1. Reduce object acceleration around the `contact58_gentle` candidate with
   documented rationale, then repeat the same sanity/visual/metrics gates.
2. After nonzero residual labels or another approved objective exists, build a
   formal residual-adapter training runner that preserves held-out cells and
   reports all source gates.
3. If switching to diffusion policy, ACT, OpenPI, or another serious policy
   method, first audit official code/checkpoints and observation/action
   contracts.

## Forbidden Next Steps

- Do not train a zero-residual no-op adapter and claim adaptation.
- Do not introduce a placeholder MLP/policy and present it as official-method
  progress.
- Do not train on held-out `full_low` or `empty_high`.
- Do not rename Newton contact proxy into T-Rex tactile F6.

## Interpretation

The project is ready for one more focused residual-label diagnostic, not for a
valid learned-adapter training run. The next concrete Phase 04 action should be
to reduce object acceleration around `contact58_gentle` while preserving
nonzero residual corrections.
