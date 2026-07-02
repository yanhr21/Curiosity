# Phase 01 Train-Only Corrective Source Gate

- status: `fail`
- run tag: `p01_src_gentle_a1_20260630_0913`
- admitted sources: `1`
- rejected sources: `7`
- train rows: `1800`
- validation rows: `0`
- train active feedback labels: `39`
- validation active feedback labels: `0`
- train csv: `data/processed/phase01/src_gentle/train.csv`
- validation csv: `data/processed/phase01/src_gentle/validation.csv`

This is corrective source data preparation only. It is not residual training and not curiosity success evidence.

## Admitted

- `train_cylinder_heavy_low` active `39` triggers `39`

## Rejected

- `train_cup_quarter_low_hidden`: slip_regression, accel_regression, no_paired_advantage
- `train_cup_half_medium_truthful`: no_paired_advantage
- `train_cup_three_quarter_high_truthful`: accel_regression, no_paired_advantage
- `train_box_light_medium_center`: active_feedback_frames_below_min:11
- `train_box_heavy_low_offset`: slip_regression, active_feedback_frames_below_min:13
- `train_cylinder_light_medium`: slip_regression, accel_regression, no_paired_advantage, active_feedback_frames_below_min:17
- `train_cup_half_low_misleading`: no_paired_advantage

## Failures

- `admitted_cells_below_min:1`
- `no_validation_rows_from_admitted_sources`
