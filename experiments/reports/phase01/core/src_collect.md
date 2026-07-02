# Phase 01 Train-Only Corrective Source Gate

- status: `pass`
- run tag: `p01_src_a3r2_20260630_0221`
- admitted sources: `6`
- rejected sources: `2`
- train rows: `9000`
- validation rows: `1800`
- train active feedback labels: `4061`
- validation active feedback labels: `1616`
- train csv: `data/processed/phase01/src/train.csv`
- validation csv: `data/processed/phase01/src/validation.csv`

This is corrective source data preparation only. It is not residual training and not curiosity success evidence.

## Admitted

- `train_cup_quarter_low_hidden` active `239` triggers `239`
- `train_cup_half_medium_truthful` active `1647` triggers `1647`
- `train_cup_three_quarter_high_truthful` active `1530` triggers `1530`
- `train_cylinder_light_medium` active `23` triggers `23`
- `train_cylinder_heavy_low` active `622` triggers `622`
- `train_cup_half_low_misleading` active `1616` triggers `1616`

## Rejected

- `train_box_light_medium_center`: active_feedback_frames_below_min:10
- `train_box_heavy_low_offset`: accel_regression, active_feedback_frames_below_min:10
