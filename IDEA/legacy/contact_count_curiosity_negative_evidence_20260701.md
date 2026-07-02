# Contact-Count Curiosity Negative Evidence

Date: 2026-07-01

This file preserves the old Newton-native contact-count curiosity route as an
explicit wrong-path / negative-evidence record. It is not the active research
idea and must not be used to justify a future success claim.

## Old Pipeline

1. Use an existing base controller to complete grasp/lift/hold.
2. Record low-dimensional rollout state: object height, contact count,
   controller phase, mass, and friction labels.
3. Train a GRU forward model to predict next object motion, contact count, and
   slip/contact-loss risk.
4. Compute learning-progress curiosity reward from forward-model error change.
5. Use that reward to weight supervised residual-controller samples.
6. Train each candidate for about one hour, then evaluate held-out cells.

## Why It Failed The Claim

This was not true closed-loop curiosity. It was offline learning-progress
scoring plus supervised residual fine-tuning. The intrinsic signal did not
drive online exploration, did not change future rollout data through policy
optimization, and did not make the agent actively choose probing, regrasping,
grip-force adjustment, pressure balancing, or shear-minimizing behavior.

The tactile input was also not tactile. The old fields were scalar contact
proxies:

```text
newton.panda.rigid_contact_count
candidate.modality.contact_available_mask
```

The model input was low-dimensional and did not include left/right pad pressure
maps, compression maps, `Fn`, `Ft`, shear direction, contact area,
penetration/compression, marker flow, or tactile images.

## Result

The five real one-hour curiosity policy candidates did not pass the strongest
baseline comparison:

```text
positive_curiosity_result = false
safety_regression_cell_count = 4
```

Therefore:

- old contact-count curiosity pipeline = legacy negative evidence;
- do not describe old Phase 01 results as curiosity success;
- do not run a sixth old-style one-hour contact-count residual training attempt
  unless the user explicitly resets that stop gate;
- do not call sample reweighting closed-loop curiosity;
- do not call `rigid_contact_count` tactile.
