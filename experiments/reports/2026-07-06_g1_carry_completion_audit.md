# 2026-07-06 G1 Carry Completion Audit

This audit tracks the gap between the current direct Isaac G1 diagnostics and
the full project objective. It is not a success claim.

## Full Objective

The target system must demonstrate a robot in simulation that:

- walks and remains balanced;
- carries a free box while walking;
- remains balanced while carrying in any selected carrying posture;
- does not rely on rollout root pose writes, root velocity writes, or box pose
  writes as carrying shortcuts;
- eventually supports unknown-load probing and autonomous posture selection.

## Current Verified Evidence

### Low-carry target-hold

Best run:
`20260706_g1_agile_largerbox_lowcarry_terminal015_finalearly060_yawzero_targethold_819_targetnegx1`
(`168398`).

Evidence:

- completed `819/819` steps;
- fall/drop `0/0`;
- rollout root pose/root velocity/box pose writes `0/0/0`;
- final robot/box target-directed travel `2.298755/2.346454 m`;
- target-window both-final-hold stable steps, longest streak, and end streak
  all `164`;
- final relative offset `0.079615 m`;
- min robot/box z `0.752112/0.808381 m`;
- max robot/box tilt `0.208595/0.413612 rad`.

Conclusion: direct Isaac G1 low-carry free-box carry-to-target-hold is
verified for this one posture and box setting.

### Chest-pad short target-window hold

Best run:
`20260706_g1_agile_chestpad_oppositeyaw_nearstop_targetwindow_900_targetnegx1`
(`168419`).

Evidence:

- completed `900/900` steps;
- fall/drop `0/0`;
- rollout root pose/root velocity/box pose writes `0/0/0`;
- final robot/box target-directed travel `1.730244/1.759363 m`;
- target-window both stable steps, longest streak, and end streak all `33`;
- final relative offset `0.108737 m`;
- max relative offset `0.205432 m`;
- min robot/box z `0.721562/0.825034 m`;
- max robot/box tilt `0.307758/0.384690 rad`.

Conclusion: a second manually selected carrying posture/contact configuration
can end inside the target window, but the hold evidence is short.

## Pending Evidence

`168420`
(`20260706_g1_agile_chestpad_oppositeyaw_nearstop_targetwindow_1000_targetnegx1`)
extended the chest-pad posture to `1000` steps and raised target-window both
stable, longest-streak, and end-streak gates to `>=100`. It completed with
fall/drop `0/0`, no rollout root/box shortcut writes, and target-window both
stable/longest/end streak `133/133/133`. It still failed strict
posture/carry-quality gates: max robot/box tilt `0.485765/0.713845 rad`,
final relative offset `0.311690 m`, and final box lateral error
`0.785696 m`. It is therefore long-window evidence, not a strict second-
posture pass.

`scripts/isaac/select_core_world_g1_carry_posture_from_probe.py` now provides
a diagnostic bridge from a G1 probe summary to a low-carry or chest-pad
validation configuration. It uses visible box size and logged probe
displacement, explicitly ignores hidden `box_mass_kg`, and writes a selection
report plus optional shell env exports. This is useful plumbing, but it is not
success evidence until paired with a real G1 probe run and a selected-posture
carry validation.

`scripts/isaac/run_core_world_g1_probe_selected_targetwindow_validation.sh`
now connects that bridge into a compute-side pipeline: front-bumper probe,
diagnostic posture selection, then selected target-window validation. It is
not success evidence until a compute-node run passes.

Slurm job `168429`
(`20260706_g1_probe_selected_targetwindow_diag1`) is the first submitted run
of that pipeline. It completed with Slurm exit `0:0`. The probe used
`probe_mode=front_bumper` for `220` active steps and measured probe motion
`0.511708 m`. The selector passed, selected `lowcarry`, reported
`selection_uses_hidden_ground_truth=false`, and explicitly ignored the present
hidden `box_mass_kg`. The selected validation passed with `819` steps,
fall/drop `0/0`, target-window end streak `164`, and rollout root pose/root
velocity/box pose writes `0/0/0`. This is a passing diagnostic pipeline for a
low-risk box, not full autonomous posture-selection success.

## Negative Evidence

- `168402` showed yaw-zero `chestpad` does not transfer the low-carry
  target-hold behavior: fall/drop `87/35`, large lateral drift, and late fall.
- `168403` showed default `boxtilt` does not reach the target window and fails
  near the end.
- `168352`, `168356`, and `168358` showed low-carry long-horizon stabilization
  is not solved beyond the current target-hold horizon.
- `168420` showed chest-pad target-window occupancy can be extended to `133`
  end-streak steps, but late drift violates tilt, relative-offset, and lateral
  gates.
- `168431` showed target-window final stop fixes the major chest-pad
  tilt/relative-offset drift and gives final-hold end streak `132`, but it
  still misses the strict final box lateral gate by `0.014122 m`.
- `168450` showed flipping final lateral command direction is not a valid
  solution: target-window end streak became `0`, final robot/box lateral error
  grew to `1.710366/1.626656 m`, final target-directed travel overshot to
  `2.548142/2.555898 m`, and max robot/box tilt exceeded gates
  (`0.391817/0.542518 rad`), although fall/drop and shortcut-write counts
  remained `0`.
- `168451` showed yaw-only final correction without lateral correction is also
  invalid: fall/drop `232/193`, first fall/drop step `768/807`, target-window
  end streak `0`, final robot/box lateral error `3.699152/3.710816 m`, and
  max robot/box tilt `3.135358/3.139102 rad`. This preserves no-shortcut
  evidence but contradicts any claim that yaw-only correction solves the
  second-posture hold.
- `168452` showed simply widening chest-pad support to `0.44 m` is not a
  valid second-posture solution: fall/drop and shortcut gates remained clean,
  but final latch was too late (`991`), final active steps were `9`,
  target-window end streak was `10`, max robot/box tilt exceeded gates
  (`0.412648/0.501682 rad`), final relative offset was `0.322406 m`, and
  final box lateral error was `0.650154 m`.
- `168453` showed `AGILE_COMMAND_HOLD_LATERAL_GAIN=0.10` is not a valid
  second-posture solution: it preserved no-shortcut evidence but caused
  fall/drop `95/82`, no terminal/final latch, target-window end streak `0`,
  max robot/box tilt `0.971987/1.016840 rad`, and final robot/box lateral
  error `-1.621430/-1.625827 m`.
- `168454` showed `AGILE_COMMAND_HOLD_LATERAL_GAIN=0.085` is also not a valid
  second-posture solution: fall/drop and shortcut gates stayed clean, but the
  run never latched final hold, target-window end streak was `0`, final box
  target-directed travel was only `1.227951 m`, final box lateral error was
  `0.774984 m`, and max box tilt was `0.630405 rad`.
- `168455` showed `AGILE_COMMAND_Y=0.005` is not a valid second-posture
  solution: fall/drop and shortcut gates stayed clean, but final active steps
  fell to `40`, target-window end streak was `41`, max box tilt was
  `0.486637 rad`, and final robot/box lateral error overcorrected to
  `-0.858138/-0.730904 m`.
- `168456` showed `AGILE_COMMAND_Y=0.001` is not a valid second-posture
  solution either: no-shortcut evidence stayed clean, but fall/drop was
  `347/272`, target-window end streak was `0`, final robot/box
  target-directed travel overshot to `4.135339/4.167482 m`, and max robot/box
  tilt was `3.140074/3.138898 rad`.
- `168458` showed the verified low-carry route does not yet generalize to a
  lighter held-out mass (`FREE_BOX_MASS=0.25`): no-shortcut evidence stayed
  clean, but fall/drop was `384/225`, target-window and final-hold end streaks
  were `0`, final relative offset was `0.449969 m`, final robot/box
  target-directed travel overshot to `4.399167/3.986899 m`, and max robot/box
  tilt was `2.710347/2.745914 rad`.
- `168462` showed the verified low-carry route also does not generalize to a
  heavier held-out mass (`FREE_BOX_MASS=0.75`): no-shortcut evidence stayed
  clean, but fall/drop was `346/284`, terminal/final latch were false,
  target-window and final-hold end streaks were `0`, final relative offset was
  `0.405846 m`, final box target-directed travel was `-0.169243 m`, and max
  robot/box tilt was `1.996009/3.139406 rad`.

## Missing For Full Completion

- Strong second-posture hold: `chestpad` must pass a longer target-window end
  hold with strict tilt, relative-offset, and lateral gates. `168420` is not
  enough because it only passed the window occupancy part.
- More posture coverage: `boxtilt` or another distinct posture/contact
  geometry must be stabilized, or the claim must be explicitly narrowed away
  from "any posture."
- Load/shape generalization: at least one held-out mass or box shape should
  pass the same no-fall/no-drop/no-shortcut target-window gates for each
  claimed posture.
- Active unknown-load probing: current G1 runs use manually selected posture
  and controller settings except for the new `168429` diagnostic pipeline,
  which passed on one low-risk case but does not yet prove robust unknown-load
  inference.
- Autonomous posture selection: current posture choice is manual. The full
  objective requires posture selection that responds to body/load constraints.
  The new G1 probe selector is a diagnostic heuristic; `168429` validates its
  wiring on one low-risk case only.
- Video-conditioned learning: current runs are controller diagnostics, not
  video-conditioned RL.

## Next Execution Rule

Do not mark the goal complete until every claimed posture/load setting has
current check/summary evidence for:

- completed horizon;
- fall/drop `0/0`;
- rollout root pose/root velocity/box pose writes `0/0/0`;
- target-window end hold meeting the declared threshold;
- bounded relative offset and lateral error;
- robot/box height and tilt inside gates.

For the next chest-pad branch, do not loosen gates. Add a target-window final
stop/hold or other stabilization that preserves the `168420`/`168431` window
occupancy while reducing final box lateral drift below `0.6 m`.
