# Phase 01 Curiosity-Weighted Residual Held-Out Eval

- status: `pass`
- run tag: `p01_resid_cur_local_adv_eval_a5_20260630_1323`
- checkpoint: `checkpoints/phase01/core/resid/curiosity_local_adv/p01_resid_cur_local_adv_a5_20260630_1028.pt`
- held-out cells: `4`
- successes: `4`

This is held-out evaluation of the curiosity-weighted residual candidate. It is not a success claim until compared against the strongest baseline set without safety regression.

## Cells

- `heldout_box_heavy_low_large_offset` status `success` lift `0.2312484085559845` hold `27.332916259765625` slip `0.008888607676829311` contact_loss `1` accel `4.595734910700008`
- `heldout_cup_empty_high_misleading` status `success` lift `0.1655988246202469` hold `26.599594116210938` slip `0.004129981722406402` contact_loss `1` accel `0.4832204527288278`
- `heldout_cup_full_low_hidden` status `success` lift `0.1592714488506317` hold `26.649593353271484` slip `0.00386093686118682` contact_loss `0` accel `2.168506427457192`
- `heldout_cylinder_heavy_low_masked_vision` status `success` lift `0.1999811977148056` hold `26.732925415039062` slip `0.010107224730615752` contact_loss `0` accel `5.069453198170307`
