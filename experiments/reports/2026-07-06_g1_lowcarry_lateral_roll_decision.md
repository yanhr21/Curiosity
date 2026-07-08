# G1 Low-Carry Lateral/Roll Decision Notes

Date: 2026-07-06

This note summarizes the current light-box low-carry branch. It is not success
evidence. All runs below keep rollout root pose, rollout root velocity, and box
pose writes at zero.

## Baseline

- `168472`
  - Stamp:
    `20260706_g1_agile_lowcarry_lightbox025_retention_step420_targethold819_strict_targetnegx1`
  - Strong retention plus terminal/final min step `420/420`.
  - Result: fail.
  - Fall/drop: `55/39`; first fall/drop: `764/780`.
  - Final robot/box lateral error: `-2.340681/-2.443050 m`.
  - Final relative offset: `0.359277 m`.
  - Max robot/box tilt: `2.705030/1.518419 rad`.
  - Interpretation: best delayed-failure timing, but severe late lateral drift
    and roll/pitch collapse.

## Lateral Correction Variants

- `168475`
  - Stamp:
    `20260706_g1_agile_lowcarry_lightbox025_retention_step420_latcorr_targethold819_strict_targetnegx1`
  - Same retention and step gates as `168472`; lateral correction always-on
    across hold, relaxed tilt gate, sign `1.0`, limit `0.003`.
  - Result: fail.
  - Fall/drop: `87/39`; first fall/drop: `732/780`.
  - Final robot/box lateral error: `-2.313470/-2.372427 m`.
  - Lateral correction active steps: `242`.
  - Interpretation: enabling more lateral correction with the old sign does
    not fix drift and worsens delayed-fall timing.

- `168478`
  - Stamp:
    `20260706_g1_agile_lowcarry_lightbox025_retention_step420_latrev_targethold819_strict_targetnegx1`
  - Reversed lateral sign, limit `0.003`.
  - Result: fail.
  - Fall/drop: `150/100`; first fall/drop: `657/719`.
  - Final robot/box lateral error: `-1.079545/-1.217948 m`.
  - Final relative offset: `0.382696 m`.
  - Max robot/box tilt: `1.273742/1.348365 rad`.
  - Interpretation: reversed sign reduces lateral drift and tilt, but is too
    aggressive and causes earlier fall/drop plus box lag.

- `168479`
  - Stamp:
    `20260706_g1_agile_lowcarry_lightbox025_retention_step420_latrev_mild_targethold819_strict_targetnegx1`
  - Reversed lateral sign, reduced limit `0.0015`.
  - Result: fail.
  - Fall/drop: `74/0`; first fall/drop: `745/null`.
  - Final robot/box target-directed travel: `2.013973/1.655340 m`.
  - Final robot/box lateral error: `-1.437272/-1.531827 m`.
  - Final relative offset: `0.371662 m`.
  - Max robot/box tilt: `1.424887/1.671567 rad`.
  - Interpretation: best current lateral-command variant because it preserves
    the box and restores some forward progress, but still fails from late roll,
    lateral drift, and box lag.

## Active Pending Diagnostic

- `168482`
  - Stamp:
    `20260706_g1_agile_lowcarry_lightbox025_retention_step420_latrev_mild_rollpos_targethold819_strict_targetnegx1`
  - Same as `168479`, but `BALANCE_ROLL_SIGN=1.0`.
  - Result: fail.
  - Fall/drop: `219/199`.
  - Final robot/box target-directed travel: `1.316626/1.191437 m`.
  - Final robot/box lateral error: `0.923641/1.107086 m`.
  - Final relative offset: `0.281507 m`.
  - Max robot/box tilt: `0.955348/0.915332 rad`.
  - Interpretation: positive roll sign worsens retention and does not solve
    lateral/roll behavior. Stop roll-sign sweeping from this branch.

## Decision Rule

- Since `168482` worsened the run, stop lateral/roll sign sweeping and change
  the contact or hold formulation. Candidate next contact changes are
  chest-pad support, top-lid height adjustment, or a different front-tray
  geometry while keeping no rollout root/velocity/box pose shortcuts.
