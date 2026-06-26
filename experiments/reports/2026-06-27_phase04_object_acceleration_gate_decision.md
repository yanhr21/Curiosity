# Phase 04 Object Acceleration Gate Decision Record

Date: 2026-06-27

## Status

Resolved for the first residual-label source candidate. The project should not
remain blocked on the earlier object-acceleration gate issue.

## Context

Earlier ordinary-cell diagnostics produced nonzero feedback labels but failed
strict metrics on `object_accel_above_threshold` around `8.309` m/s^2. That
looked like a possible physical or safety blocker.

Peak analysis showed the top acceleration event in the non-warmup diagnostic
occurred at step 2, phase 0, before scripted feedback was active. That means
the strict metric was dominated by an initial settling artifact in the recorded
window, not by the residual feedback behavior during lift/hold.

## Resolution

The source path now supports `PRE_RECORD_WARMUP_STEPS`. The warmup is executed
before the recorded rollout and does not advance the scripted waypoint
sequence. The first warmup15 source candidate is:

```text
residual_label_sweep_half_low_contact58_gentle_lift165_warmup15_20260627_032006
```

Result:

- official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: `pass_nonblank_success_with_feedback`;
- metrics status: pass;
- feedback trigger count: 241;
- max object acceleration: 0.5063306543767194 m/s^2;
- top peak after warmup: step 136, phase 2, feedback active.

## Decision

Use the warmup15 run as the first residual-label source candidate and move on
to formal source-runner construction. Do not continue blind threshold sweeps
for the old `8.309` m/s^2 artifact.

The previous non-warmup and smooth diagnostics remain historical evidence, not
current blockers.

## Next Step

Use `experiments/configs/residual_label_source_manifest_v1.json` as the first
runner input. Build source-gate checks, preserve held-out `full_low` and
`empty_high`, and collect more ordinary cells before any learned-adapter
training claim.
