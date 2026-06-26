# Phase 04 Residual Adapter Failure-Mode Comparison V1

## Scope

This report compares validated Newton cup rollouts across three policies:

- no-adaptation scripted prior;
- scripted feedback baseline;
- learned Newton-native residual controller adapter.

The comparison covers four cells:

- held-out: `full_low`, `empty_high`;
- ordinary: `half_high`, `full_medium`.

This is a post-hoc analysis of already validated outputs. It is not training,
not model creation, not a T-Rex result, and not tactile F6 evidence.
`generated_trex_fields=[]` and `schema_promotion=blocked` remain required.

## Execution

- Config:
  `experiments/configs/residual_adapter_failure_mode_comparison_v1.json`.
- Script:
  `experiments/configs/compare_residual_adapter_failure_modes.py`.
- Launcher:
  `experiments/configs/launch_residual_adapter_failure_mode_comparison_tmux.sh`.
- Allocation: existing tmux-held Slurm job `154142` on `server56`.
- Output JSON:
  `experiments/outputs/residual_adapter_failure_mode_comparison_v1_20260627.json`.
- Output CSV:
  `experiments/outputs/residual_adapter_failure_mode_comparison_v1_20260627.csv`.
- Status: pass.
- Rows: `12`.
- Cell comparisons: `4`.

## Main Result

Across all four cells, the no-adaptation and scripted-feedback baselines fail
only on `object_accel_above_threshold`. The scripted-feedback baseline never
activates feedback in these runs: `final_feedback_trigger_count=0`.

The learned residual adapter passes all four cells. It activates feedback in
every learned run with `final_feedback_trigger_count=240`,
`max_feedback_active_probability=1`, and first active time
`2.2666666507720947` s. The first lift-threshold crossing occurs at
`0.3499999940395355` s for learned runs, so the learned residual is acting as a
post-lift stabilization correction in this benchmark, not as an earlier
regrasp or pre-lift exploration action.

## Per-Cell Comparison

### Held-Out Full-Low

- No adaptation: fail, `max_object_accel_m_s2=8.308390712127508`.
- Scripted feedback: fail, `max_object_accel_m_s2=8.308707788010144`,
  `final_feedback_trigger_count=0`.
- Learned residual: pass, `max_object_accel_m_s2=1.5345948979069628`,
  `final_feedback_trigger_count=240`.
- Learned acceleration reduction vs no-adaptation: `6.773795814220545`
  m/s^2.
- Learned acceleration reduction vs scripted feedback: `6.774112890103181`
  m/s^2.

### Held-Out Empty-High

- No adaptation: fail, `max_object_accel_m_s2=8.308498000056417`.
- Scripted feedback: fail, `max_object_accel_m_s2=8.308498000056417`,
  `final_feedback_trigger_count=0`.
- Learned residual: pass, `max_object_accel_m_s2=0.4686260874870734`,
  `final_feedback_trigger_count=240`.
- Learned acceleration reduction vs no-adaptation: `7.839871912569344`
  m/s^2.
- Learned acceleration reduction vs scripted feedback: `7.839871912569344`
  m/s^2.

### Ordinary Half-High

- No adaptation: fail, `max_object_accel_m_s2=8.308498000056417`.
- Scripted feedback: fail, `max_object_accel_m_s2=8.308550048022228`,
  `final_feedback_trigger_count=0`.
- Learned residual: pass, `max_object_accel_m_s2=0.4709508074000259`,
  `final_feedback_trigger_count=240`.
- Learned acceleration reduction vs no-adaptation: `7.837547192656391`
  m/s^2.
- Learned acceleration reduction vs scripted feedback: `7.837599240622202`
  m/s^2.

### Ordinary Full-Medium

- No adaptation: fail, `max_object_accel_m_s2=8.308498000056417`.
- Scripted feedback: fail, `max_object_accel_m_s2=8.308707937632189`,
  `final_feedback_trigger_count=0`.
- Learned residual: pass, `max_object_accel_m_s2=2.6287727996680594`,
  `final_feedback_trigger_count=240`.
- Learned acceleration reduction vs no-adaptation: `5.679725200388358`
  m/s^2.
- Learned acceleration reduction vs scripted feedback: `5.679935137964129`
  m/s^2.

## Interpretation

The observed failure mode is consistent across these four cells: the baselines
are already visually valid and satisfy lift/hold/slip/drop/contact gates, but
they fail the strict full metric because the acceleration gate records an
excessive peak. The scripted feedback rule does not help because it never
triggers.

The learned residual adapter changes this failure mode. It passes the strict
metric by lowering recorded max object acceleration while preserving lift,
hold, slip, contact-loss, and visual gates. Its active timing shows that this
checkpoint currently behaves like a stabilization residual after the object is
already lifted, not like a full curiosity-driven exploration policy.

This completes the current four-cell adaptation-speed and failure-mode
comparison for the cup benchmark. Remaining broad-claim blockers are still
multi-seed stability, richer object families, and tactile/T-Rex-compatible
signals.
