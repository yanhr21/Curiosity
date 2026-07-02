# T-Rex Data Contract

- created_at: `2026-07-01`
- classification: `metadata_contract_not_training_not_checkpoint_load`
- official source: `external/T-Rex_43ff`
- validator: `src/newton_tactile_curiosity/trex_contract_validate.py`

This records the minimum metadata contract for any future Newton-to-T-Rex
bridge. It is source/schema work only: no T-Rex checkpoint was downloaded,
loaded, or trained.

## Required Features

The official T-Rex LeRobot path requires:

- `observation.images.head`: video, `[3,H,W]`
- `observation.images.wrist_right`: video, `[3,H,W]`
- `observation.images.wrist_left`: video, `[3,H,W]`
- `observation.state`: float32, `[62]`
- `action`: float32, `[16,62]`
- `action_abs`: float32, `[62]`
- `observation.tactile_f6`: float32, `[10,6]`
- `observation.tactile_deform.l0` ... `l4`: video, `[3,H,W]`
- `observation.tactile_deform.r0` ... `r4`: video, `[3,H,W]`

The validator also checks q01/q99/mask normalization stats for `action`,
`state`, and `tactile_f6` when a stats file is supplied.

## Current Mismatch

The current Newton candidate must not be promoted to T-Rex compatibility yet:

- it is single-arm Panda evidence, not bimanual eef-62;
- its tactile fields are candidate Newton force maps/proxies, not validated
  10-finger F6 plus deformation streams;
- it is not a LeRobot v3.0 T-Rex contract dataset;
- Gate 00F official semantic validation is still open.

The first valid future claim is only a metadata-contract pass on a converted
held-out dataset. A policy success claim still requires official checkpoint
load, held-out rollouts, ablations, and improvement over the strongest baseline
without safety regression.
