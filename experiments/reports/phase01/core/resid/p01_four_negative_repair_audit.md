# Phase 01 Five-Negative Stop-Gate Audit

Status: incomplete curiosity objective; five negative real one-hour curiosity
policy candidates are recorded and the stop gate is triggered.

This report is not a success claim. It exists to prevent the final allowed
one-hour attempt from repeating the same failure mode.

## Stop-Gate State

- Negative real one-hour curiosity policy candidates: 5/5.
- Latest ledger:
  `experiments/reports/phase01/core/training_attempts.json`.
- Do not start a sixth real one-hour curiosity policy candidate without
  explicit user instruction.

## Negative Attempts

- `p01_resid_cur_a1_20260630_0407`: negative; 4/4 held-out safety regressions.
- `p01_resid_cur_sa_a2_20260630_0521`: negative; safety anchor intended but
  inactive due to contact-key mismatch.
- `p01_resid_cur_sa2_a3_20260630_0641`: negative; contact fallback activated
  the neutral anchor, but validation behavior collapsed and 4/4 held-out cells
  still regressed safety.
- `p01_resid_cur_distill_a4_20260630_0752`: negative; base-policy
  distillation restored validation behavior, but held-out evaluation succeeded
  on only 3/4 cells and strongest-baseline comparison still reported 4/4
  safety regressions.

## Source-Data Finding

The train-only corrective source gate admitted 6 sources and rejected 2 box
sources:

- source manifest: `data/processed/phase01/src/manifest.json`
- train rows: 9000
- validation rows: 1800
- train active feedback labels: 4061
- validation active feedback labels: 1616

The admitted train-only sources improve some slip/acceleration metrics, but
their paired source metrics consistently trade off hold/lift. Examples from
the admitted-source manifest:

- `train_cylinder_heavy_low`: slip delta `-0.010330062057485857`, acceleration
  delta `-2.8153249401739315`, but hold delta `-0.49999237060546875` and lift
  delta `-0.0026956871151924133`.
- `train_cup_half_medium_truthful`: acceleration delta `-0.2258575846767753`,
  but hold delta `-0.5499916076660156` and lift delta `-0.0038487017154693604`.
- `train_cup_half_low_misleading`: acceleration delta `-0.1890888614941555`,
  but hold delta `-0.5499916076660156` and lift delta `-0.003790229558944702`.

This means the current residual labels teach a tradeoff: reduce slip or
acceleration by accepting weaker lift/hold behavior. That tradeoff matches the
held-out failure pattern in repeated curiosity-weighted residual candidates.

## Final-Attempt Source Repair Evidence

Two train-only repair gates were run before allowing any fifth real one-hour
curiosity policy candidate.

Strict existing-source preflight:

- manifest: `data/processed/phase01/src_strict/manifest.json`
- report:
  `experiments/reports/phase01/core/resid/p01_strict_source_repair_preflight.md`
- status: `blocked`
- admitted sources: 0
- rejected sources: 6
- train rows: 0
- validation rows: 0
- final one-hour attempt allowed: false

Gentle strict recollection:

- run tag: `p01_src_gentle_a1_20260630_0913`
- manifest: `data/processed/phase01/src_gentle/manifest.json`
- report: `experiments/reports/phase01/core/src_gentle_collect.md`
- status: `fail`
- admitted sources: 1
- rejected sources: 7
- train rows: 1800
- validation rows: 0
- train active feedback labels: 39
- validation active feedback labels: 0
- failures: `admitted_cells_below_min:1`,
  `no_validation_rows_from_admitted_sources`

The only admitted gentle source was `train_cylinder_heavy_low`. Its paired
metrics improved acceleration and slip, but still had small lift and hold
regressions (`hold_delta_fb_minus_no=-0.049999237060546875`,
`lift_delta_fb_minus_no=-0.0007306858897209167`). It is not enough to train
the final allowed candidate because there is no validation source and the
gate admitted fewer than two cells.

Balanced strict recollection:

- run tag: `p01_src_bal_a1r1_20260630_0945`
- config:
  `experiments/configs/phase01/src_collect_balanced_strict.json`
- manifest: `data/processed/phase01/src_bal/manifest.json`
- status: `fail`
- admitted sources: 1
- rejected sources: 7
- train rows: 1800
- validation rows: 0
- train active feedback labels: 29
- validation active feedback labels: 0
- failures: `admitted_cells_below_min:1`,
  `no_validation_rows_from_admitted_sources`

The admitted balanced source was `train_cylinder_light_medium`. The two box
cells improved acceleration without safety regression, but remained below the
active-feedback frame gate (`18` and `17` active frames versus the required
`20`). A prior aborted balanced run, `p01_src_bal_a1_20260630_0939`, exposed a
runner bug: JSON `false` was printed as `False`, while the shell wrapper only
treated `0` as disabled, so initial waypoint adjustment stayed enabled. The
runner was repaired before `a1r1`; the aborted run is invalid source evidence
and must not be used for training.

Focused box/light-cylinder strict recollection:

- run tag: `p01_src_box_a1_20260630_1006`
- config: `experiments/configs/phase01/src_collect_box_strict.json`
- manifest: `data/processed/phase01/src_box/manifest.json`
- status: `fail`
- admitted sources: 1
- rejected sources: 2
- train rows: 1800
- validation rows: 0
- train active feedback labels: 23
- validation active feedback labels: 0
- failures: `admitted_cells_below_min:1`,
  `no_validation_rows_from_admitted_sources`

The focused run admitted only `train_box_heavy_low_offset`. Lowering the
feedback acceleration trigger increased active frames, but made
`train_box_light_medium_center` regress acceleration and lift, while
`train_cylinder_light_medium` regressed lift. This is negative source-repair
evidence, not a training result.

Local-advantage segment repair:

- run tag: `p01_local_adv_a1_20260630_1024`
- segment config: `experiments/configs/phase01/local_adv_segments.json`
- segment manifest: `data/processed/phase01/local_adv/manifest.json`
- LP score config: `experiments/configs/phase01/local_adv_lp_scores.json`
- LP score summary:
  `experiments/outputs/phase01/core/local_adv_lp/curiosity_learning_progress_summary.json`
- status: `pass`
- train segments: 3
- validation segments: 1
- train rows: 576
- validation rows: 192
- train active feedback labels: 58
- validation active feedback labels: 29

This repair does not lower the source-level safety/lift/hold contract. It
masks the objective to fixed-length local segments from source rollouts whose
paired metrics are safe enough, including near-miss sources that failed only
because full-rollout active feedback frames were below the source-level gate.
Learning-progress scores are still computed from the existing real one-hour
Newton-native forward-model checkpoint pair; they are not fabricated.

Local-advantage smoke diagnostic:

- run tag: `p01_resid_cur_local_adv_smoke_a1_20260630_1026`
- summary:
  `experiments/outputs/phase01/core/resid/curiosity_local_adv/p01_resid_cur_local_adv_smoke_a1_20260630_1026_summary.json`
- status: `pass`
- train score coverage: 1.0
- validation score coverage: 1.0
- real training result: false
- checkpoint written: false

Fifth real one-hour candidate:

- run tag: `p01_resid_cur_local_adv_a5_20260630_1028`
- config:
  `experiments/configs/phase01/resid_curiosity_local_adv_train.json`
- training summary:
  `experiments/outputs/phase01/core/resid/curiosity_local_adv/p01_resid_cur_local_adv_a5_20260630_1028_summary.json`
- held-out evaluation:
  `experiments/outputs/phase01/core/resid/curiosity_local_adv_eval/p01_resid_cur_local_adv_eval_a5_20260630_1323_summary.json`
- strongest-baseline comparison:
  `experiments/outputs/phase01/core/resid/curiosity_eval/p01_resid_cur_local_adv_cmp_a5_20260630_1340_comparison.json`
- MP4 summary:
  `experiments/outputs/phase01/core/resid/curiosity_local_adv_eval/p01_resid_cur_local_adv_mp4_a5_20260630_1343_mp4_summary.json`
- result: negative
- positive curiosity result: false
- safety regression cell count: 4
- useful improvement count: 2

Stop-gate status: triggered. This is the fifth negative real one-hour
curiosity policy candidate. No sixth real one-hour curiosity policy training
may start without the user's instruction.

## Required Final-Attempt Repair

The fifth real one-hour candidate must not repeat the current source objective
unchanged. Before using the final allowed training attempt, repair at least one
of the following with train-only evidence:

- collect a stricter corrective source whose paired train metrics do not trade
  away lift/hold for slip/acceleration;
- or build a loss/source mask that trains only on source segments with
  demonstrated local advantage and does not supervise the harmful hold/lift
  tradeoff;
- or document a blocker if no train-only source can provide positive advantage
  without safety/lift/hold regression.

Held-out cells remain locked. They must not be used for training,
hyperparameter selection, threshold tuning, source selection, or label
construction.

Current final-attempt status: completed and negative. Local-advantage segment
masking repaired the source gate enough to allow the fifth attempt, but the
held-out strongest-baseline comparison was still negative. The required action
is to stop and report before any sixth real one-hour curiosity policy training
attempt.
