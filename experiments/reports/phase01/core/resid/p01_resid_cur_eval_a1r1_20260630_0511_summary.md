# Phase 01 Curiosity-Weighted Residual Held-Out Eval

- status: `pass`
- run tag: `p01_resid_cur_eval_a1r1_20260630_0511`
- checkpoint: `checkpoints/phase01/core/resid/curiosity/p01_resid_cur_a1_20260630_0407.pt`
- held-out cells: `4`
- successes: `4`

This is held-out evaluation of the curiosity-weighted residual candidate. It is not a success claim until compared against the strongest baseline set without safety regression.

## Cells

- `heldout_box_heavy_low_large_offset` status `success` lift `0.23134008795022964` hold `27.332916259765625` slip `0.00878326005722852` contact_loss `1` accel `2.9951313175151437`
- `heldout_cup_empty_high_misleading` status `success` lift `0.16638021171092987` hold `27.132919311523438` slip `0.003988460714382876` contact_loss `1` accel `0.8369675870183162`
- `heldout_cup_full_low_hidden` status `success` lift `0.15534420311450958` hold `26.09960174560547` slip `0.003585376586989635` contact_loss `0` accel `1.2717283841865368`
- `heldout_cylinder_heavy_low_masked_vision` status `success` lift `0.19985529780387878` hold `27.16625213623047` slip `0.016512854789349603` contact_loss `0` accel `3.8401647928384754`
