# Phase 04 Residual Label Sweep: Half Low Contact58 Diagnostic

Date: 2026-06-27

## Status

Diagnostic completed, not promoted to a training-label source.

This run belongs to the short-term stable route: keep the official Newton Panda
hydro scripted grasp/lift controller as the infant prior, then collect only
validated nonzero residual controller-parameter labels before starting any
learned adapter. No learned model was trained, no placeholder policy was
created, and no T-Rex schema fields were generated.

## Run

- Run tag: `residual_label_sweep_half_low_contact58_20260627_0310`
- Slurm job: `154023`
- Tmux session: `curiosity_next_source_alloc_20260626_232937`
- Cell: ordinary `half_low`
- Held-out cell: false
- Controller mode: `lift_hold_feedback`
- Object mass: `0.20` kg
- Object friction: `0.35`
- Feedback min contact count: `58`
- Feedback acceleration threshold: `6.5` m/s^2
- Feedback height-drop threshold: `0.015` m
- Final hold duration target: `2.5` s

## Evidence

- Official Newton sanity:
  `experiments/outputs/residual_label_sweep_half_low_contact58_20260627_0310_fresh_newton_sensor_contact_sanity.json`
- Summary:
  `experiments/outputs/residual_label_sweep_half_low_contact58_20260627_0310_summary.json`
- Automated visual validation:
  `experiments/outputs/residual_label_sweep_half_low_contact58_20260627_0310_visual_validation.json`
- Manual visual inspection:
  `experiments/outputs/residual_label_sweep_half_low_contact58_20260627_0310_manual_visual_inspection.json`
- NPZ:
  `experiments/outputs/residual_label_sweep_half_low_contact58_20260627_0310.npz`
- Contact sheet:
  `experiments/visuals/residual_label_sweep_half_low_contact58_20260627_0310/contact_sheet.png`
- Frame browser:
  `experiments/visuals/residual_label_sweep_half_low_contact58_20260627_0310/frame_browser.html`
- Log:
  `logs/newton/residual_label_sweep_half_low_contact58_20260627_0310.log`

## Observed Results

- Fresh official Newton sanity: pass
- Camera export: pass
- Automated visual validation: pass
- Manual visual inspection: `pass_nonblank_but_task_failure`
- Feedback trigger count: `241`
- Feedback-active frames: `241`
- Feedback reason: low contact count
- Contact proxy range: `31` to `62`
- Lift height: `0.12752966582775116` m
- Longest hold: `0.9833333333333333` s
- Drop from max: `0.0` m
- Task metric status: fail
- Failure reason: `hold_duration_below_min`

## Interpretation

The contact58 sweep keeps the useful property from the first sensitive
diagnostic: the official Newton rollout path can emit nonzero
`candidate.controller.*` residual fields. However, it still fails the 2s hold
gate. That means it is useful as threshold-sweep evidence, but it must not be
used as a residual adapter training-label source.

The selected short-term stable method remains:

1. Use the official Newton Panda hydro scripted controller as the infant prior.
2. Restrict residual-label collection to ordinary cells, excluding held-out
   `full_low` and `empty_high`.
3. Sweep feedback thresholds until nonzero residual labels are produced while
   preserving official sanity, automated/manual visual checks, lift, hold,
   drop, and contact gates.
4. Only after at least one promoted nonzero residual-label source exists,
   implement the learned residual adapter runner with source gates and
   held-out split checks.

## Decision

Not promoted to training-label source.

Next action: continue the bounded ordinary-cell sweep with a less disruptive
trigger strategy. Prefer acceleration-sensitive or milder contact thresholds
that preserve the 2s hold gate while still producing nonzero residual
controller labels.
