# 2026-07-06 G1 yaw-zero free-box carry status

This report records the current direct Isaac G1 free-box carrying state after
the yaw-zero terminal-hold diagnostics. It is not a final success claim.

## Best current evidence

- Best target-hold run:
  `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`
  (`168398`, `server39`, build `0`, checker `0`, status `pass`).
- Duration: `819/819` steps.
- Shortcut writes: rollout root pose `0`, root velocity `0`, box pose `0`.
- Safety: fall/drop `0/0`.
- Target hold: final robot/box target-directed travel
  `2.298755/2.346454 m`; target-window both-final-hold stable steps,
  longest streak, and end streak were all `164`.
- Height/tilt: min robot/box z `0.752112/0.808381 m`; max robot/box tilt
  `0.208595/0.413612 rad`.
- Carry quality: final relative offset `0.079615 m`; final robot/box lateral
  error `0.133809/0.191820 m`.

Interpretation: this is the strongest direct-G1 evidence so far that the
robot can carry a free dynamic box in the low cradle into a target window and
hold it there for a nontrivial interval without falling, dropping, or using
rollout root/box shortcuts.

## Second-Posture Evidence

- Best current `chestpad` run:
  `20260706_g1_agile_chestpad_oppositeyaw_nearstop_targetwindow_900_targetnegx1`
  (`168419`, `server39`, build `0`, checker `0`, status `pass`).
- Duration: `900/900` steps.
- Shortcut writes: rollout root pose `0`, root velocity `0`, box pose `0`.
- Safety: fall/drop `0/0`.
- Target window: final robot/box target-directed travel
  `1.730244/1.759363 m`; target-window both stable steps, longest streak, and
  end streak were all `33`.
- Height/tilt: min robot/box z `0.721562/0.825034 m`; max robot/box tilt
  `0.307758/0.384690 rad`.
- Carry quality: final relative offset `0.108737 m`; max relative offset
  `0.205432 m`; final robot/box lateral error `0.258455/0.362250 m`.

Interpretation: this shows a second manually selected carrying posture/contact
configuration can end inside the target window without falling, dropping, or
using rollout shortcuts. The evidence is weaker than the low-cradle result
because the end streak is only `33` steps; it does not prove autonomous
posture selection, active probing, or robust multi-posture carrying.

Pending strengthening run:
`20260706_g1_agile_chestpad_oppositeyaw_nearstop_targetwindow_1000_targetnegx1`
(`168420`) keeps the same `chestpad` controller but extends to `1000` steps
and requires target-window both stable, longest streak, and end streak all
`>=100`. It completed on `server21` with build status `0` and checker status
`1`. The good part: `1000/1000` steps, fall/drop `0/0`, no rollout root/box
shortcut writes, final robot/box target-directed travel
`1.912552/1.774289 m`, and target-window both stable/longest/end streak
`133/133/133`. The failure: carry quality drifted beyond strict gates, with
max robot tilt `0.485765`, max box tilt `0.713845`, final relative offset
`0.311690 m`, and final box lateral error `0.785696 m`. This is a long
target-window hold but not a strict second-posture pass.

## 900-step stable-carry evidence

- Best run: `20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_latsmall_strict_900_targetnegx1`
  (`168335`, `server39`, build `0`, checker `1`).
- Duration: `900/900` steps.
- Shortcut writes: rollout root pose `0`, root velocity `0`, box pose `0`.
- Safety: fall/drop `0/0`.
- Height/tilt: min robot/box z `0.660195/0.759457 m`; max robot/box tilt
  `0.208595/0.413612 rad`.
- Carry quality: final relative offset `0.049381 m`; final robot/box lateral
  error `0.259286/0.290766 m`.
- Limitation: precise target stop failed. Final robot/box target-directed
  travel was `2.713489/2.747867 m`, above the strict `2.35 m` upper gate.

Interpretation: this shows the same policy remains upright and retains the box
for 900 steps, but it overruns the target window. It is useful stable-carry
evidence, not precise target stopping.

## Negative follow-ups

- `168339` (`0.45 m` cutoff) was too early: first fall/drop `624/644`, final
  robot/box target-directed travel `1.554673/1.521620 m`.
- `168343` and `168350` (`0.55/0.545 m` cutoff) were close but failed late:
  first fall `882`, no box drop, final robot target-directed travel
  `2.437910 m`, and final-hold target-window longest streak `100`.
- `168352` extended the best setting to `1200` steps and failed after the
  900-step horizon: first fall/drop `945/965`.
- `168356` proved late rescue wiring but did not improve the 1200-step
  collapse: first fall/drop `946/963`.
- `168358` made roll feedback stronger and destabilized much earlier:
  first fall/drop `598/645`.
- `168402` tested the same `819`-step target-hold gate in `chestpad` mode and
  failed: fall/drop `87/35`, first fall/drop `732/784`, final robot/box
  lateral error `-1.672259/-1.909126 m`, final relative offset `0.427477 m`,
  and max robot/box tilt `3.032566/3.033201 rad`, with no rollout root/box
  shortcut writes.
- `168403` tested the same gate in `boxtilt` mode and failed: fall/drop
  `4/0`, first fall `815`, final robot/box target-directed travel only
  `0.560830/0.411779 m`, target-window hold counts `0`, and final relative
  offset `0.362352 m`, again with no rollout root/box shortcut writes.
- `168420` extended the best chest-pad setting to `1000` steps and achieved
  target-window both end streak `133`, but failed strict posture/carry-quality
  gates from late drift: max robot/box tilt `0.485765/0.713845 rad`, final
  relative offset `0.311690 m`, and final box lateral error `0.785696 m`.
- `168431` added a target-window final stop to the chest-pad setting. It
  fixed the major `168420` tilt/relative-offset drift and reached final-hold
  end streak `132` with last command `[0, 0, 0]`, fall/drop `0/0`, no rollout
  root/box shortcut writes, max robot/box tilt `0.307758/0.384690 rad`, and
  final relative offset `0.144021 m`. It still failed by `1.4 cm` on final
  box lateral error: `0.614122 > 0.6`.
- `168432` moved the final-stop trigger earlier to `1.55 m`. It preserved
  fall/drop `0/0`, no shortcut writes, target-window/final-hold end streak
  `103`, and good tilt/relative offset, but final box lateral error worsened
  to `0.692755 m`. The useful next search is a smaller trigger shift around
  `1.62 m`, not a much earlier stop.
- `168433` tried the intermediate `1.62 m` final-stop trigger. It preserved
  fall/drop `0/0`, no shortcut writes, target-window/final-hold end streak
  `133`, and good tilt/relative offset, but final box lateral error worsened
  further to `0.715806 m`. The next branch should keep the better `1.65 m`
  trigger and allow corrective yaw/lateral commands during final hold.
- `168435` kept the `1.65 m` trigger and allowed yaw/lateral corrections
  during final hold. It still had fall/drop `0/0`, no shortcut writes, and
  good tilt/relative offset, but final box lateral error was `0.708802 m`.
  Corrections did not fix the drift; the next branch should try final stand
  targets after the `1.65 m` trigger.
- `168436` added final stand targets after the `1.65 m` final trigger. It
  fixed lateral drift (`final_box_target_lateral_error_m=0.232397`) and kept
  fall/drop `0/0`, no shortcut writes, target-window/final-hold/final-stand
  end streak `133/132/132`, and final relative offset `0.187513 m`. It failed
  only tilt gates: max robot/box tilt `0.755765/0.749709 rad`. The next branch
  should make the stand transition gentler.
- `168437` made the final stand transition gentler (`delay=40`,
  `blend=0.005`). It fixed tilt (`0.309353/0.384690 rad`) and kept fall/drop
  `0/0`, no shortcut writes, and final relative offset `0.165383 m`, but
  final box lateral error remained `0.650740 m`. A middle transition
  (`delay=20`, `blend=0.01`) is the next reasonable interpolation.
- `168438` tried that middle transition (`delay=20`, `blend=0.01`) and failed
  both tilt and lateral gates: max robot/box tilt `0.658000/0.658923 rad` and
  final box lateral error `0.712096 m`. This suggests delay is the damaging
  factor; the next branch should keep zero delay and only reduce blend rate.
- `168440` kept zero delay and lowered blend to `0.005`. It fixed lateral
  (`0.129922 m`) and kept fall/drop `0/0` and no shortcut writes, but tilt was
  unacceptable (`0.827365/0.979325 rad`). Final stand is therefore useful for
  lateral correction but not yet a strict second-posture solution.
- `168450` returned to final-stop with corrective commands but flipped the
  lateral sign. It completed `1000` steps with fall/drop `0/0` and no shortcut
  writes, but it was clearly worse than `168431`: target-window end streak
  collapsed to `0`, final robot/box lateral error was
  `1.710366/1.626656 m`, final robot/box target-directed travel was
  `2.548142/2.555898 m`, and max robot/box tilt was
  `0.391817/0.542518 rad`. Do not continue this lateral-sign branch.
- `168451` disabled lateral correction and allowed yaw correction during final
  hold. It failed much earlier and harder: fall/drop `232/193`, first
  fall/drop step `768/807`, target-window end streak `0`, final robot/box
  lateral error `3.699152/3.710816 m`, and max robot/box tilt
  `3.135358/3.139102 rad`, with no shortcut writes. This shows the lateral
  correction in `168431` is necessary for this chest-pad setup.
- `168452` widened the chest pad to `0.44 m`. It preserved fall/drop `0/0`
  and no shortcut writes, but final latch was delayed to step `991`, leaving
  only `9` final-hold steps; target-window end streak was `10`, max robot/box
  tilt was `0.412648/0.501682 rad`, final relative offset was `0.322406 m`,
  and final box lateral error was `0.650154 m`. This is worse than `168431`;
  increasing pad width is not the next fix.
- `168453` increased lateral gain to `0.10`. It overcorrected and failed:
  fall/drop `95/82`, first fall/drop step `905/918`, no terminal/final latch,
  target-window end streak `0`, max robot/box tilt `0.971987/1.016840 rad`,
  and final robot/box lateral error `-1.621430/-1.625827 m`, with no shortcut
  writes. Do not use high lateral gain.
- `168454` tried a tiny lateral-gain interpolation `0.085`. It kept fall/drop
  `0/0` and no shortcut writes, but final latch was false, target-window end
  streak was `0`, final robot/box target-directed travel was only
  `1.188529/1.227951 m`, final robot/box lateral error was
  `0.658880/0.774984 m`, and max robot/box tilt was
  `0.371441/0.630405 rad`. Even small gain changes are too disruptive here.
- `168455` added base `AGILE_COMMAND_Y=0.005`. It kept fall/drop `0/0` and no
  shortcut writes, but over-biased the path: final latch was step `960`,
  final-hold end streak was only `40`, max box tilt was `0.486637 rad`, and
  final robot/box lateral error became `-0.858138/-0.730904 m`. This suggests
  command-y bias may be useful only at a much smaller magnitude.
- `168456` reduced the base bias to `AGILE_COMMAND_Y=0.001`, but the branch
  failed badly: fall/drop `347/272`, first fall/drop step `653/686`,
  target-window end streak `0`, final robot/box target-directed travel
  `4.135339/4.167482 m`, final robot/box lateral error
  `-1.190266/-1.049643 m`, and max robot/box tilt
  `3.140074/3.138898 rad`, with no shortcut writes. Stop the command-y bias
  branch.

## Current conclusion

The immediate verified milestone is low-cradle free-box carry-to-target-hold,
not final task success. The remaining direct-G1 blockers are:

- long-horizon stabilization after about 900 steps;
- precise target-window stopping without freezing joints or using root/box
  pose writes over a longer fixed horizon;
- extension beyond one low-cradle posture to multiple carrying postures and
  load/shape variations.

The new `chestpad` and `boxtilt` negatives are important: the current result
does not yet generalize across carrying posture/contact geometry. The next
valid claim-improving step is not another low-cradle target-hold repeat, but a
controlled second-posture or load/shape generalization that passes the same
fall/drop, target-hold, relative-offset, and no-shortcut gates.

For the next chest-pad branch, the best anchor remains `168431`, which missed
only the final box lateral gate by `0.014122 m`. Since both lateral-sign
changes, yaw-only correction, and wider contact geometry are negative, the
next test should preserve the `168431` structure and slightly increase
pre-final lateral gain only by a tiny interpolation, not to `0.10`. That
interpolation also failed, so the next low-risk branch is a tiny base
command-y path bias with the original gain. The first `0.005` bias
overcorrected, and `0.001` failed worse. Further chest-pad progress likely
requires contact/controller redesign rather than scalar micro-tuning. The next
useful non-overclaiming step is held-out validation on the already verified
low-carry route.

The reusable compute-side launcher
`scripts/isaac/run_core_world_g1_targetwindow_posture_validation_suite.sh`
now captures this validation path. It refuses to run on login nodes, runs
selected target-window posture/load cases through the larger-box strict suite,
and writes `targetwindow_posture_validation_summary.json` for audit.

Separate pending active-probing path:
`20260706_g1_probe_selected_targetwindow_diag1` (`168429`) runs the new
front-bumper probe -> non-hidden-telemetry posture selector -> selected
target-window validation pipeline. It completed on `server44` with Slurm exit
`0:0`: probe `front_bumper` active steps `220`, probe motion `0.511708 m`,
selector `status=pass`, selected posture `lowcarry`,
`selection_uses_hidden_ground_truth=false`, selected validation checker
`status=pass`, `819` steps, fall/drop `0/0`, target-window end streak `164`,
and rollout root pose/root velocity/box pose writes `0/0/0`. This is useful
active-probing/selection plumbing evidence for one low-risk case, not final
autonomous posture-selection success.
