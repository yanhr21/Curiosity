# Phase 01 Curiosity-Weighted Residual Held-Out Eval

- status: `pass`
- run tag: `p01_resid_cur_sa2_eval_a3_20260630_0739`
- checkpoint: `checkpoints/phase01/core/resid/curiosity_sa2/p01_resid_cur_sa2_a3_20260630_0641.pt`
- held-out cells: `4`
- successes: `4`

This is held-out evaluation of the curiosity-weighted residual candidate. It is not a success claim until compared against the strongest baseline set without safety regression.

## Cells

- `heldout_box_heavy_low_large_offset` status `success` lift `0.23116485029459` hold `27.332916259765625` slip `0.00876767305259826` contact_loss `1` accel `3.3813573093985267`
- `heldout_cup_empty_high_misleading` status `success` lift `0.1663738340139389` hold `27.132919311523438` slip `0.003988934361141126` contact_loss `1` accel `0.8367688064134504`
- `heldout_cup_full_low_hidden` status `success` lift `0.15691174566745758` hold `26.09960174560547` slip `0.0035519907619746094` contact_loss `0` accel `2.1730084564959387`
- `heldout_cylinder_heavy_low_masked_vision` status `success` lift `0.19995779544115067` hold `27.16625213623047` slip `0.01397912922941359` contact_loss `0` accel `3.8402607303074534`
