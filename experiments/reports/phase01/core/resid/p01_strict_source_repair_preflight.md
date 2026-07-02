# Phase 01 Strict Source Repair Preflight

- status: `blocked`
- run tag: `p01_src_strict_a1_20260630_0910`
- admitted strict sources: `0`
- rejected strict sources: `6`
- train rows: `0`
- validation rows: `0`
- manifest: `data/processed/phase01/src_strict/manifest.json`

This is a train-only data/objective preflight. It is not training and not a curiosity success claim.

## Gate

- max lift regression m: `0.001`
- max hold regression s: `0.1`
- min slip improvement m: `0.00025`
- min accel improvement m/s^2: `0.25`

## Rejected

- `train_cup_quarter_low_hidden`: lift_regression:-0.003832399845123291, hold_regression:-0.5333251953125
- `train_cup_half_medium_truthful`: lift_regression:-0.0038487017154693604, hold_regression:-0.5499916076660156
- `train_cup_three_quarter_high_truthful`: lift_regression:-0.00379122793674469, hold_regression:-0.5499916076660156
- `train_cylinder_light_medium`: lift_regression:-0.002358578145503998, hold_regression:-0.4666595458984375
- `train_cylinder_heavy_low`: lift_regression:-0.0026956871151924133, hold_regression:-0.49999237060546875
- `train_cup_half_low_misleading`: lift_regression:-0.003790229558944702, hold_regression:-0.5499916076660156

## Blockers

- `strict_admitted_sources_below_min:0`
