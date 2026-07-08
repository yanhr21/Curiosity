# G1 Low-Carry Load Robustness Status

Timestamp: 2026-07-07 07:04 CST.

This is diagnostic evidence only. It is not a final carrying-success claim.

## Current Best

- Fresh verified low-carry baseline at `0.50 kg`:
  `20260707_g1_targetwindow_lowcarry_repro_lowcarry_targethold819`.
- It uses real G1 USD, AGILE ONNX policy, spawned free box,
  `attach_box=none`, collision-enabled front tray/top lid, and no rollout
  root pose, root velocity, or box pose writes.
- Key result: 819 steps, fall/drop `0/0`, final robot/box target-directed
  travel `2.29876/2.34645 m`, target-window end streak `164`, max robot/box
  tilt `0.20860/0.41361 rad`.

## Load Validation

Suite: `20260707_g1_lowcarry_load_validation_fresh`.

| Mass | Status | Key outcome |
| --- | --- | --- |
| 0.25 kg | fail | `384` falls / `225` drops after entering the target-window region. |
| 0.50 kg | pass | Reproduced the 819-step low-carry baseline. |
| 0.75 kg | fail | `346` falls / `284` drops before final hold. |

## Mass Band

Suite: `20260707_g1_lowcarry_mass_band_fresh`.

| Mass | Status | Key outcome |
| --- | --- | --- |
| 0.35 kg | pass | Fall/drop `0/0`, target-window end streak `108`, max robot/box tilt `0.24340/0.41646 rad`. |
| 0.40 kg | fail | Early lateral/roll fall: `398` falls. |
| 0.45 kg | fail | Briefly reached target window, then late fall/drop: `87` falls / `60` drops. |
| 0.55 kg | fail | `383` falls / `170` drops. |
| 0.60 kg | fail-near | Fall/drop `0/0`, target-window end streak `108`, but box tilt `0.63855 rad > 0.45`. |
| 0.65 kg | fail | `414` falls / `154` drops. |

## Repair Attempts

Suite `20260707_g1_lowcarry_load_repair_fresh`: `0/3` pass.

- `0.25 kg` final-window freeze: `418` falls / `102` drops.
- `0.25 kg` policy-then-stand: `550` falls / `536` drops.
- `0.75 kg` chestpad/retention/slow: `930` falls / `856` drops.

Suite `20260707_g1_lowcarry_edge_repair_fresh`: `0/4` pass.

- `0.60 kg` tight lid variants and chestpad hold all worsened the original
  near-miss, producing falls/drops.
- `0.45 kg` tight-lid/final-zero improved fall/drop to `0/0`, but under-traveled
  to about `1.52 m`, had target-window streak `0`, and exceeded tilt gates.

Suite `20260707_g1_lowcarry_edge_repair_v2_fresh`: `0/4` pass.

- Delaying `0.45 kg` final-hold moved farther but crossed the stability
  boundary and produced falls/drops.
- `0.60 kg` side-rail-only and no-lid/tall-rail variants also failed with
  falls/drops.

## Decision

The current G1 low-carry front-tray setup has discontinuous narrow stable
islands, not load robustness. Stop sweeping scalar final thresholds, tight or
lower lids, chestpad, side rails, and no-lid geometry as the main route.

Next valid work should switch to a materially different controller/backend or
policy adaptation for payload/contact dynamics.
