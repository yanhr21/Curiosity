# Phase 01 No-Curiosity Residual Baseline Comparison

- run tag: `p01_resid_eval_a1_20260630_0342`
- checkpoint: `checkpoints/phase01/core/resid/base/p01_resid_base_a1_20260630_0307.pt`
- status: held-out evaluation completed on 4/4 locked held-out cells
- classification: learned non-curiosity baseline evidence, not curiosity success

## Result

The learned no-curiosity residual baseline succeeded on all four held-out
cells, but it did not clearly dominate the existing strongest non-curiosity
baseline set. It is therefore useful as a learned baseline, not as a positive
result.

## Per-Cell Comparison

- `heldout_box_heavy_low_large_offset`: residual kept success and slightly
  reduced slip versus no-adaptation (`0.00864` vs `0.00875`), but increased
  acceleration versus no-adaptation (`3.668` vs `2.838`). It is not a clean
  safety improvement.
- `heldout_cup_empty_high_misleading`: residual matched success/lift/hold
  closely to no-adaptation, but did not beat scripted feedback on acceleration
  (`0.837` vs scripted `0.618`).
- `heldout_cup_full_low_hidden`: residual reduced slip and acceleration versus
  no-adaptation (`0.00352` vs `0.00404`, `0.956` vs `1.120`), but lowered lift
  and hold (`0.1554` vs `0.1595`, `26.10s` vs `27.10s`).
- `heldout_cylinder_heavy_low_masked_vision`: residual remained successful but
  underperformed scripted feedback on slip and acceleration (`0.02417` vs
  scripted `0.01699`, `6.664` vs scripted `3.706`).

## Next Step

The next faithful step is not to claim improvement from this baseline. It is to
train the curiosity-weighted residual candidate from matched learning-progress
evidence and evaluate it against the strongest per-cell baseline without
safety regression.
