# G1 Close-Front Tilt-Escape Result - 2026-07-07

This report records the strict result of the close-front final-hold
tilt-escape probes. These are Isaac diagnostics only. They are not learned
unknown-load carrying and not arbitrary-posture humanoid carrying.

## Baseline Boundary

The prior useful close-front near-miss was `steps1050_final120`:

- fall/drop: `0/0`
- final robot/box target-directed travel: about `2.026/2.103 m`
- final robot/box lateral error: about `0.081/0.095 m`
- max robot/box tilt: about `0.486/0.493 rad`
- target-window stable steps: `76 < 80`
- final-hold active steps: `268 < 399`

The blocker was not gross locomotion collapse; it was terminal hold, box/robot
tilt margin, and slightly insufficient target-window dwell.

## Late Tilt-Escape Probe

Entrypoint:
`scripts/isaac/run_core_world_g1_lowcarry_close_front_tilt_escape_suite.sh`

Suite:
`20260707_g1_lowcarry_close_front_tilt_escape`

Result: aggregate `fail`, `0/2` strict cases passed.

Both cases preserved fall/drop `0/0`, rollout writes `0/0/0`, and final
robot/box travel about `2.026/2.103 m`, but did not improve target-window dwell
or terminal tilt enough. The late thresholds only activated escape in the last
`11-18` steps, too late to arrest the final tilt.

## Early Tilt-Escape Probe

Suite:
`20260707_g1_lowcarry_close_front_tilt_escape_early2`

Result: aggregate `fail`, `0/2` strict cases passed.

| Case | fall/drop | robot/box travel m | lateral robot/box m | max tilt robot/box rad | target stable/longest/end | escape active | Main failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `escape_robot022_box030_scale015` | `0/0` | `1.963/2.014` | `-0.118/-0.207` | `0.610/0.500` | `77/75/75` | `34` from step `887` | robot tilt, box tilt, target dwell, final-hold length |
| `escape_robot018_box024_scale020` | `0/0` | `2.118/2.130` | `0.013/0.150` | `0.267/0.544` | `81/80/80` | `83` from step `792` | box tilt and final-hold length |

## Interpretation

Early tilt escape is a useful mechanism because it recovered the close-front
target-window dwell without falls, drops, or rollout root/velocity/box pose
writes. It is still not a pass because the box pitched past the strict
`0.45 rad` gate and the 1050-step run cannot satisfy the `399` final-hold
active-step requirement.

The next valid close-front probe is therefore not another lateral sign/gain
scan. It should test physical box attitude support and a longer 1200-step
terminal hold while preserving the same strict gates. The active follow-up is:

`scripts/isaac/run_core_world_g1_lowcarry_close_front_chestpad_tilt_support_suite.sh`

Pending jobs:

- full two-case suite: Slurm `170370` / `g1_chestpad`
- one-case quick/backfill suite: Slurm `170372` / `g1_chestquick`

## Chest-Pad Tilt-Support Follow-Up

Suite:
`20260707_g1_lowcarry_close_front_chestpad_tilt_support`

Result: aggregate `fail`, `0/2` strict cases passed. The quick/backfill job
`170372` was cancelled after the full suite started, to avoid duplicate GPU use.

| Case | fall/drop | first fall/drop | robot/box travel m | max tilt robot/box rad | target stable | final hold | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pad_box022_z012_x006` | `891/593` | `277/309` | `0.192/0.065` | `3.138/3.141` | `0` | `0` | early collapse |
| `pad_box026_z014_x008` | `879/387` | `277/309` | `0.838/0.733` | `3.105/3.141` | `0` | `0` | early collapse |

This follow-up was a negative isolation result. Chest-pad collision enabled at
step `650`, after the first fall/drop, while the modified lower/thicker top lid
enabled at step `116`. Therefore this run should not be interpreted as evidence
that an earlier chest pad is bad. It shows that the lower-lid / thicker-pad
geometry destabilized the carry before the intended terminal support mechanism
could matter.

Next valid test:
`scripts/isaac/run_core_world_g1_lowcarry_close_front_early_escape_1200_suite.sh`

That suite keeps the original support geometry from the no-fall 1050-step
near-miss, extends to `1200` steps, and only isolates whether early tilt escape
plus original-size chest-pad triggering can satisfy the final-hold duration and
box-tilt gates.

## Early-Escape 1200-Step Isolation

Suite:
`20260707_g1_lowcarry_close_front_early_escape_1200`

Result: aggregate `fail`, `0/2` strict cases passed. Both cases produced the
same metrics, meaning the original-size box-tilt chest-pad trigger did not
alter behavior relative to target-window-only triggering.

| Case | fall/drop | first fall/drop | target stable/longest/end | final hold | robot/box travel m | lateral robot/box m | max tilt robot/box rad | escape active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_target_window_pad` | `103/72` | `1097/1128` | `108/107/0` | `418@782` | `3.129/2.968` | `1.014/1.090` | `2.120/2.061` | `197` from step `792` |
| `baseline_box_tilt_pad` | `103/72` | `1097/1128` | `108/107/0` | `418@782` | `3.129/2.968` | `1.014/1.090` | `2.120/2.061` | `197` from step `792` |

This is a useful late-failure boundary. It restores the original no-early-fall
behavior, satisfies the final-hold active-step count, and enters the target
window for more than the required number of steps. It then leaves the window,
over-travels, drifts laterally, and falls/drops late. The likely immediate
failure mode is that final tilt escape continues to apply a nonzero command
after the target window has already been stable.

Next valid test:
`scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_suite.sh`

That suite suppresses final tilt escape after target-window streaks of `60` or
`80` steps while keeping the same original support geometry and strict gates.

## Escape-Suppression Follow-Up

Suite:
`20260707_g1_lowcarry_close_front_escape_suppression`

Result: aggregate `fail`, `0/2` strict cases passed.

| Case | fall/drop | first fall/drop | target stable/longest/end | final hold | robot/box travel m | lateral robot/box m | max tilt robot/box rad | escape active/suppressed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `suppress_after_streak60` | `64/46` | `1136/1154` | `114/113/0` | `418@782` | `3.200/3.140` | `0.923/0.879` | `2.734/2.592` | `167/54` |
| `suppress_after_streak80` | `102/80` | `1098/1120` | `108/107/0` | `418@782` | `3.117/2.891` | `0.738/0.582` | `3.139/3.132` | `189/28` |

Suppressing final tilt escape after target-window dwell helps but is still not
sufficient. The `60`-step suppression delayed first fall by about `39` steps
relative to the unsuppressed 1200-step run and reduced fall/drop counts, but
the robot still left the window, over-traveled, drifted laterally, and collapsed
late.

The next terminal-control test should add a window-latched brake or stand
handoff after sufficient target-window dwell. Continuing to increase forward
escape, lower top-lid contact, or pad thickness is not supported by these
results.

Active next suite:
`scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_brake_suite.sh`

It keeps the better `suppress_after_streak60` setup and applies a negative
final brake command after `240` final-hold steps for `120` steps.

## Escape-Suppression Brake Follow-Up

Suite:
`20260707_g1_lowcarry_close_front_escape_suppression_brake`

Result: aggregate `fail`, `0/2` strict cases passed.

| Case | fall/drop | first fall/drop | target stable/longest/end | robot/box travel m | lateral robot/box m | max tilt robot/box rad | brake active |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `suppress60_brake240_neg004` | `110/81` | `1090/1119` | `110/109/0` | `3.177/3.039` | `0.653/0.559` | `3.134/3.141` | `120`, steps `1022-1141` |
| `suppress60_brake240_neg008` | `33/7` | `1167/1193` | `108/107/0` | `3.583/3.587` | `0.902/1.064` | `0.673/0.739` | `120`, steps `1022-1141` |

The stronger brake delayed collapse and reduced tilt/drop severity, but it did
not preserve the target-window hold. It also allowed more target-direction
over-travel. This suggests the next terminal test should switch the controller
to stand targets after sufficient window dwell instead of continuing to tune
brake magnitude alone.

Active next suite:
`scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_suppression_stand_suite.sh`

It keeps the `suppress_after_streak60` boundary and switches to final stand
targets after `240` or `300` final-hold steps.

## Escape-Suppression Stand Follow-Up

Suite:
`20260707_g1_lowcarry_close_front_escape_suppression_stand`

Result: aggregate `fail`, `0/2` strict cases passed.

| Case | fall/drop | first fall/drop | target stable/longest/end | final-stand stable/longest | robot/box travel m | lateral robot/box m | max tilt robot/box rad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `suppress60_stand240_blend002` | `56/33` | `1144/1167` | `144/143/0` | `91/91` | `2.750/2.510` | `0.765/0.869` | `2.924/2.198` |
| `suppress60_stand300_blend004` | `39/12` | `1161/1188` | `114/113/0` | `1/1` | `3.312/3.240` | `0.739/0.945` | `1.108/1.110` |

The earlier stand handoff is the best terminal-control direction so far because
it increased total target-window dwell and produced `91` stable target-window
steps during final stand. It still fails strict gates: fall/drop are nonzero,
the target-window streak at the end is `0`, lateral error remains too high, and
tilt eventually exceeds the limit.

The next step should combine this earlier stand handoff with lateral/roll
stabilization or replace the support backend. This is still not close-front
carrying success.

Active next suite:
`scripts/isaac/run_core_world_g1_lowcarry_close_front_escape_stand_lateral_suite.sh`

It starts from `suppress60_stand240_blend002` and compares the two signs of an
existing lateral-error-driven balance roll target.

## Escape-Stand Lateral Follow-Up

Suite:
`20260707_g1_lowcarry_close_front_escape_stand_lateral`

Result: aggregate `fail`, `0/2` strict cases passed.

| Case | fall/drop | first fall/drop | target stable/longest/end | final-stand stable/longest | robot/box travel m | lateral robot/box m | max tilt robot/box rad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stand240_lateral_sign_neg` | `57/34` | `1143/1166` | `145/144/0` | `92/92` | `2.735/2.491` | `0.745/0.877` | `3.130/2.462` |
| `stand240_lateral_sign_pos` | `56/33` | `1144/1167` | `145/144/0` | `92/92` | `2.766/2.517` | `0.755/0.860` | `3.122/2.280` |

The lateral roll target did not materially change the failure. The terminal
branch has produced useful boundaries but not a strict close-front pass. The
next meaningful step is not more scalar tuning on lateral sign/gain, brake
magnitude, or escape threshold. It should be a materially different terminal
support policy or a support-backend replacement under the same strict gates.
