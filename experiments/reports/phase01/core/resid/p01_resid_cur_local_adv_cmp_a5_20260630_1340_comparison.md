# Phase 01 Curiosity Residual Strongest-Baseline Comparison

- status: `pass`
- result classification: `negative_or_incomplete_candidate`
- positive curiosity result: `False`
- candidate cells: `4`
- safety-regression cells: `4`
- useful improvements: `2`

This comparison is strict. A checkpoint or successful rollout is not a curiosity success unless this comparison is positive without safety regression.

## Cells

- `heldout_box_heavy_low_large_offset` pass `False` regressions `slip_worse_than_best_baseline, accel_worse_than_best_baseline` improvements `none`
- `heldout_cup_empty_high_misleading` pass `False` regressions `slip_worse_than_best_baseline, hold_below_best_baseline_tolerance` improvements `accel_better_than_best_baseline`
- `heldout_cup_full_low_hidden` pass `False` regressions `slip_worse_than_best_baseline, accel_worse_than_best_baseline, hold_below_best_baseline_tolerance` improvements `none`
- `heldout_cylinder_heavy_low_masked_vision` pass `False` regressions `accel_worse_than_best_baseline, hold_below_best_baseline_tolerance` improvements `slip_better_than_best_baseline`
