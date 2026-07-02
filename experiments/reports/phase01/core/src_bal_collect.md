# Phase 01 Train-Only Corrective Source Gate

- status: `fail`
- run tag: `p01_src_bal_a1r1_20260630_0945`
- admitted sources: `1`
- rejected sources: `7`
- train rows: `1800`
- validation rows: `0`
- train active feedback labels: `29`
- validation active feedback labels: `0`
- train csv: `data/processed/phase01/src_bal/train.csv`
- validation csv: `data/processed/phase01/src_bal/validation.csv`

This is corrective source data preparation only. It is not residual training and not curiosity success evidence.

## Admitted

- `train_cylinder_light_medium` active `29` triggers `29`

## Rejected

- `train_cup_quarter_low_hidden`: lift_regression, no_paired_advantage
- `train_cup_half_medium_truthful`: lift_regression, no_paired_advantage
- `train_cup_three_quarter_high_truthful`: lift_regression, no_paired_advantage
- `train_box_light_medium_center`: active_feedback_frames_below_min:18
- `train_box_heavy_low_offset`: active_feedback_frames_below_min:17
- `train_cylinder_heavy_low`: accel_regression
- `train_cup_half_low_misleading`: lift_regression, no_paired_advantage

## Failures

- `admitted_cells_below_min:1`
- `no_validation_rows_from_admitted_sources`
