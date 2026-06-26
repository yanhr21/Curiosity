# Phase 04 Residual Adapter Training Readiness V1

## Scope

This report audits whether the first learned residual controller-parameter
adapter can be trained now. It does not start training and does not create a
model.

## Result

Status: residual-label source runner passed, learned-adapter training runner
not started.

The original blocker was that every default scripted-feedback evaluation had
`feedback_trigger_count=0`. Follow-up ordinary-cell diagnostics proved that
the official Newton rollout path can emit nonzero `candidate.controller.*`
residual fields, and the formal source runner now validates four ordinary
source candidates:

- `half_low`:
  `residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006`;
- `empty_low`:
  `residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345`;
- `half_medium`:
  `residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352`.
- `full_high`:
  `residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410`.

Final source-runner output:

- manifest:
  `data/processed/residual_label_source_runner_v1_20260627/manifest.json`;
- records:
  `data/processed/residual_label_source_runner_v1_20260627/residual_label_records.csv`;
- status: pass;
- source run count: 4;
- record count: 1440;
- total feedback trigger count: 963;
- failures: [];
- generated T-Rex fields: [];
- schema promotion: blocked;
- training started: false.

Training still has not started because no learned residual-adapter training
implementation or adapter sanity runner has been reviewed.

## Ready Evidence

- Official Newton Panda hydro scripted infant prior is available and visually
  validated.
- Real Newton mass/friction variants exist across nominal, ordinary, and
  held-out cells.
- Held-out `full_low` and `empty_high` remain reserved for generalization.
- Phase 05 contact source manifest passed with `source_run_count=10`,
  `record_count=3600`, `generated_trex_fields=[]`, and
  `schema_promotion=blocked`.
- Phase 03 curiosity replay diagnostic passed with `rollout_count=9`.
- Residual adapter and forward-model target contract exists:
  `experiments/configs/residual_adapter_forward_model_contract_v1.json`.
- Source manifest exists:
  `experiments/configs/residual_label_source_manifest_v1.json`.
- Source runner report exists:
  `experiments/reports/2026-06-27_phase04_residual_label_source_runner_v1.md`.

## Resolved Blockers

- The previous strict acceleration blocker was traced to a recorded initial
  settling artifact and resolved by `PRE_RECORD_WARMUP_STEPS=15`.
- The lack of a promoted nonzero residual-label source is resolved for four
  ordinary cells.
- The formal source-runner blocker is resolved: the runner passed after fresh
  official Newton sanity and held-out split checks.

## Blocking Gaps

- No approved learned residual-adapter training implementation exists.
- No compute-side adapter training runner exists with official Newton sanity,
  source-gate checks, held-out split enforcement, and report generation.
- Source coverage is still limited to four ordinary cells, so no general
  learned-adaptation claim is valid yet.

## Allowed Next Routes

1. Continue collecting any remaining ordinary-cell sources if needed,
   excluding held-out `full_low` and `empty_high`.
2. Design and review the learned residual-adapter runner using the source
   manifest and runner output as inputs.
3. Only after runner review, start a real training run that follows the
   no-placeholder-model and GPU-duration rules.

## Forbidden Next Steps

- Do not train a zero-residual no-op adapter and claim adaptation.
- Do not introduce a placeholder MLP/policy and present it as official-method
  progress.
- Do not train on held-out `full_low` or `empty_high`.
- Do not rename Newton contact proxy into T-Rex tactile F6.

## Interpretation

The project has moved past source-runner construction. The next technical
blocker is the learned residual-adapter runner, not source availability for the
first four ordinary cells.
