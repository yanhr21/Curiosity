# 2026-07-07 G1 Close-Front Final-Stabilize Result

This is a negative diagnostic result, not robot carrying success.

## Run

- Script: `scripts/isaac/run_core_world_g1_lowcarry_close_front_final_stabilize_suite.sh`
- Slurm job: `170306` / `g1_finstab45`
- Tmux: `curiosity_g1_final_stabilize_quick45_0707`
- Suite prefix: `20260707_g1_lowcarry_close_front_final_stabilize_quick45`
- Summary:
  `experiments/outputs/core_world_g1_lowcarry_close_front_final_stabilize/20260707_g1_lowcarry_close_front_final_stabilize_quick45/close_front_final_stabilize_summary.json`

## Result

`steps1200_final120_tilt030` failed strict gates:

- fall/drop: `142/0`
- first fall/drop step: `924/-`
- final robot/box target-directed travel: `0.7305/0.6500 m`
- max robot/box target-directed travel: `1.5190/1.4691 m`
- final robot/box lateral error: `-0.0611/-0.2004 m`
- max robot/box tilt: `3.1296/3.1289 rad`
- target-window stable/longest/end steps: `0/0/0`
- final-hold active: `418` steps from step `782`
- chest pad trigger: step `887`
- rollout root/velocity/box pose writes: `0/0/0`

## Interpretation

Earlier box-tilt chest-pad triggering did not recover the close-front posture.
The branch initially stayed upright through the pre-fall window, but the final
target travel collapsed by the end and the robot fell at step `924`.

The unified posture-conditioned gate confirms the same boundary:
`low_front_060` still passes, but `close_front_060_conditioned` fails with the
same fall/travel/target-window profile as this quick45 case. The current gate
therefore preserves one tuned posture but does not generalize to close-front.

Do not repeat the unchanged quick45 chest-pad/final-stabilize scalar branch.
The useful close-front boundaries remain:

- `steps1050_final120` from hold-delay: fall/drop `0/0`, final robot/box
  travel about `2.026/2.103 m`, but target-window stable `76 < 80` and max
  robot/box tilt `0.486/0.493 rad`.
- `support_timing_no_runtime_pad`: target-window stable `130`, but late fall
  at step `901` and no end streak.

Next action should change mechanism rather than only scalar thresholds:
posture-conditioned support/command selection, target-window arrest that does
not induce lateral drift, or a controller-backed support posture.
