# Phase 04 Residual Adapter Training Readiness V1

## Scope

This report audits whether the first learned residual controller-parameter
adapter can be trained now. It does not start training and does not create a
model.

## Result

Status: residual-adapter trainer smoke passed, real learned-adapter training
not started.

The original blocker was that every default scripted-feedback evaluation had
`feedback_trigger_count=0`. Follow-up ordinary-cell diagnostics proved that
the official Newton rollout path can emit nonzero `candidate.controller.*`
residual fields, and the formal source runner now validates five ordinary
source candidates:

- `half_low`:
  `residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006`;
- `empty_low`:
  `residual_label_sweep_empty_low_contact58_gentle_lift165_warmup15_20260627_0345`;
- `half_medium`:
  `residual_label_sweep_half_medium_contact58_gentle_lift165_warmup15_20260627_0352`.
- `full_high`:
  `residual_label_sweep_full_high_contact58_gentle_lift165_warmup15_20260627_0410`.
- `empty_medium`:
  `residual_label_sweep_empty_medium_contact58_gentle_lift165_warmup15_20260627_0425`.

Final source-runner output:

- manifest:
  `data/processed/residual_label_source_runner_v1_20260627/manifest.json`;
- records:
  `data/processed/residual_label_source_runner_v1_20260627/residual_label_records.csv`;
- status: pass;
- source run count: 5;
- record count: 1800;
- total feedback trigger count: 1203;
- failures: [];
- generated T-Rex fields: [];
- schema promotion: blocked;
- training started: false.

Real training still has not started because no `RUN_MODE=train` run has
completed the required one-GPU one-hour training, checkpoint writing, and
held-out evaluation gates.

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
- Training-input preflight exists:
  `experiments/configs/residual_adapter_training_preflight_v1.json`.
- Training-input preflight manifest passed:
  `data/processed/residual_adapter_training_preflight_v1_20260627/manifest.json`.
- Training-input preflight report exists:
  `experiments/reports/2026-06-27_phase04_residual_adapter_training_preflight_v1.md`.
- Trainer config exists:
  `experiments/configs/residual_adapter_trainer_v1.json`.
- Trainer smoke summary passed:
  `experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_smoke_20260627_0539_summary.json`.
- Trainer smoke report exists:
  `experiments/reports/2026-06-27_phase04_residual_adapter_trainer_smoke_v1.md`.

## Resolved Blockers

- The previous strict acceleration blocker was traced to a recorded initial
  settling artifact and resolved by `PRE_RECORD_WARMUP_STEPS=15`.
- The lack of a promoted nonzero residual-label source is resolved for five
  ordinary cells.
- The formal source-runner blocker is resolved: the runner passed after fresh
  official Newton sanity and held-out split checks.
- The training-input preflight blocker is resolved: the preflight runner passed
  after fresh official Newton sanity with 1440 train records and 360 validation
  records, while excluding held-out `full_low` and `empty_high`.
- The trainer smoke blocker is resolved: the CUDA/PyTorch trainer path passed
  after fresh official Newton sanity with optimizer steps and validation metrics
  while writing no checkpoint and making no real-training claim.

## Blocking Gaps

- No real one-GPU one-hour adapter training run has completed.
- No trained adapter checkpoint exists.
- No held-out adapter evaluation exists on `full_low` or `empty_high`.
- Source coverage is five ordinary cells. This limits broad learned-adaptation
  claims, but it is not a blocker for designing the learned residual-adapter
  runner.

## Allowed Next Routes

1. Run `RUN_MODE=train` only after confirming the GPU-utilization plan and
   preserving the one-GPU one-hour rule.
2. Optionally collect more ordinary-cell sources later, excluding held-out
   `full_low` and `empty_high`, but do not treat this as the active gate.
3. Only after runner review, start a real training run that follows the
   no-placeholder-model and GPU-duration rules.

## Forbidden Next Steps

- Do not train a zero-residual no-op adapter and claim adaptation.
- Do not introduce a placeholder MLP/policy and present it as official-method
  progress.
- Do not train on held-out `full_low` or `empty_high`.
- Do not rename Newton contact proxy into T-Rex tactile F6.

## Interpretation

The project has moved past source-runner construction, training-input
preflight, and trainer smoke. The next technical blocker is the real one-hour
training run plus held-out evaluation, not source availability, split
construction, or trainer import.
