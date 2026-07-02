# Phase 01 Train-Only Corrective Source Gate

- status: `fail`
- run tag: `p01_src_box_a1_20260630_1006`
- admitted sources: `1`
- rejected sources: `2`
- train rows: `1800`
- validation rows: `0`
- train active feedback labels: `23`
- validation active feedback labels: `0`
- train csv: `data/processed/phase01/src_box/train.csv`
- validation csv: `data/processed/phase01/src_box/validation.csv`

This is corrective source data preparation only. It is not residual training and not curiosity success evidence.

## Admitted

- `train_box_heavy_low_offset` active `23` triggers `23`

## Rejected

- `train_box_light_medium_center`: lift_regression, accel_regression, no_paired_advantage
- `train_cylinder_light_medium`: lift_regression

## Failures

- `admitted_cells_below_min:1`
- `no_validation_rows_from_admitted_sources`
