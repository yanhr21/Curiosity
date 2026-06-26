# Phase 04 Residual Adapter Training Readiness V1

## Scope

This report audits whether the first learned residual controller-parameter
adapter can be trained now. It does not start training and does not create a
model.

## Result

Status: residual-label source candidate ready, training runner not started.

The original blocker was that every default scripted-feedback evaluation had
`feedback_trigger_count=0`. Follow-up ordinary-cell diagnostics proved that
the official Newton rollout path can emit nonzero `candidate.controller.*`
residual fields. The first fully promoted source candidate is now:

```text
residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006
```

This run uses ordinary `half_low`, not a held-out cell. It passed fresh
official Newton sanity, automated visual validation, manual visual inspection,
and strict lift-hold metrics while producing `feedback_trigger_count=241`.

Training still has not started because the formal residual-label source runner
and adapter-training runner are not implemented or reviewed yet.

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
- First source manifest exists:
  `experiments/configs/residual_label_source_manifest_v1.json`.
- First promoted source candidate report exists:
  `experiments/reports/2026-06-27_phase04_residual_label_sweep_half_low_contact58_gentle_lift165_warmup15.md`.

Key source metrics:

- metrics status: pass;
- feedback trigger count: 241;
- lift height: 0.15815936028957367 m;
- hold duration: 2.5333309173583984 s;
- max slip: 0.0030417809728431086 m;
- contact-loss frames: 0;
- max object acceleration: 0.5063306543767194 m/s^2.

## Resolved Blocker

The previous strict acceleration blocker was traced to a recorded initial
settling artifact. Peak analysis on the non-warmup diagnostic found the top
acceleration event at step 2, phase 0, before feedback was active. The warmup15
source candidate excludes that artifact from the recorded metric window and
passes strict metrics.

## Blocking Gaps

- No formal residual-label source runner exists yet.
- No compute-side residual-adapter training runner exists with official Newton
  sanity, source-gate checks, held-out split enforcement, and report
  generation.
- Only one ordinary `half_low` source candidate is promoted, so no general
  adaptation claim is valid yet.
- No approved learned-adapter training implementation exists. A placeholder
  MLP/policy must not be introduced as learned adaptation progress.

## Allowed Next Routes

1. Build the formal residual-label source runner around
   `experiments/configs/residual_label_source_manifest_v1.json`.
2. Collect additional ordinary-cell source candidates after the runner gates
   are in place, still excluding held-out `full_low` and `empty_high`.
3. Only after source gates and runner review, implement the learned residual
   adapter runner with official Newton sanity and report generation.

## Forbidden Next Steps

- Do not train a zero-residual no-op adapter and claim adaptation.
- Do not introduce a placeholder MLP/policy and present it as official-method
  progress.
- Do not train on held-out `full_low` or `empty_high`.
- Do not rename Newton contact proxy into T-Rex tactile F6.

## Interpretation

The project is no longer blocked on the first nonzero residual-label source.
It is now blocked on formal runner construction and broader ordinary-cell
collection before any learned residual-adapter training claim.
