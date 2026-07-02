# Phase 01 Curiosity-Weighted Residual Held-Out Eval

- status: `pass`
- run tag: `p01_resid_cur_distill_eval_a4_20260630_0853`
- checkpoint: `checkpoints/phase01/core/resid/curiosity_distill/p01_resid_cur_distill_a4_20260630_0752.pt`
- held-out cells: `4`
- successes: `3`

This is held-out evaluation of the curiosity-weighted residual candidate. It is not a success claim until compared against the strongest baseline set without safety regression.

## Cells

- `heldout_box_heavy_low_large_offset` status `success` lift `0.2311554104089737` hold `27.332916259765625` slip `0.008784643082939646` contact_loss `1` accel `3.359825724102936`
- `heldout_cup_empty_high_misleading` status `success` lift `0.16635611653327942` hold `27.132919311523438` slip `0.003989424922389694` contact_loss `1` accel `0.8370831474296004`
- `heldout_cup_full_low_hidden` status `success` lift `0.1554131805896759` hold `26.09960174560547` slip `0.0035159890738519993` contact_loss `0` accel `2.2852551882707908`
- `heldout_cylinder_heavy_low_masked_vision` status `fail` lift `0.20075052231550217` hold `27.16625213623047` slip `0.0107705642565621` contact_loss `0` accel `9.535886263570077`
