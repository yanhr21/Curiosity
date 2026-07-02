# Phase 01 Residual Manifest

- status: `fail`
- train rows: `14400`
- validation rows: `5400`
- train active feedback labels: `0`
- validation active feedback labels: `0`
- train csv: `data/processed/phase01/resid/train.csv`
- validation csv: `data/processed/phase01/resid/validation.csv`
- held-out excluded: `heldout_box_heavy_low_large_offset, heldout_cup_empty_high_misleading, heldout_cup_full_low_hidden, heldout_cylinder_heavy_low_masked_vision`

This is data preparation only. It is not residual training and not curiosity success evidence.

## Failures

- `no_active_feedback_labels_in_train`
- `no_active_feedback_labels_in_validation`
