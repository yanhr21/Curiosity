# Phase 01 Curiosity-Weighted Residual Held-Out Eval

- status: `pass`
- run tag: `p01_resid_cur_sa_eval_a2_20260630_0622`
- checkpoint: `checkpoints/phase01/core/resid/curiosity_sa/p01_resid_cur_sa_a2_20260630_0521.pt`
- held-out cells: `4`
- successes: `4`

This is held-out evaluation of the curiosity-weighted residual candidate. It is not a success claim until compared against the strongest baseline set without safety regression.

## Cells

- `heldout_box_heavy_low_large_offset` status `success` lift `0.23109791427850723` hold `27.332916259765625` slip `0.008698736654404836` contact_loss `1` accel `4.660896354407887`
- `heldout_cup_empty_high_misleading` status `success` lift `0.16635002195835114` hold `27.132919311523438` slip `0.003990586643387253` contact_loss `1` accel `0.8370469897805163`
- `heldout_cup_full_low_hidden` status `success` lift `0.15515045821666718` hold `26.09960174560547` slip `0.003621361992955404` contact_loss `0` accel `0.9682264222664955`
- `heldout_cylinder_heavy_low_masked_vision` status `success` lift `0.19894565641880035` hold `27.16625213623047` slip `0.024545211491614254` contact_loss `0` accel `6.662776097266313`
