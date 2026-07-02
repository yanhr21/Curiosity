# Phase 01 No-Curiosity Residual Held-Out Eval

- status: `pass`
- run tag: `p01_resid_eval_a1_20260630_0342`
- checkpoint: `checkpoints/phase01/core/resid/base/p01_resid_base_a1_20260630_0307.pt`
- held-out cells: `4`
- successes: `4`

This is learned non-curiosity baseline evaluation, not curiosity success.

## Cells

- `heldout_box_heavy_low_large_offset` status `success` lift `0.2301599532365799` hold `27.332916259765625` slip `0.008636475950326438` contact_loss `1` accel `3.668397494094045`
- `heldout_cup_empty_high_misleading` status `success` lift `0.16635854542255402` hold `27.132919311523438` slip `0.003989112433078776` contact_loss `1` accel `0.8371069571664718`
- `heldout_cup_full_low_hidden` status `success` lift `0.155398428440094` hold `26.09960174560547` slip `0.0035224140569761397` contact_loss `0` accel `0.9557586251038963`
- `heldout_cylinder_heavy_low_masked_vision` status `success` lift `0.19894390553236008` hold `27.16625213623047` slip `0.024165773241598177` contact_loss `0` accel `6.663840041900521`
