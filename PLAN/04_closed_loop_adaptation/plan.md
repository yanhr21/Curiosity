# Phase 04: Closed-Loop Adaptation

## Goal

Test the cup-mass example directly.

## Experiment

1. Start from a basic grasping prior.
2. Evaluate on nominal cup.
3. Evaluate on varied mass/fill-level cups.
4. Observe mismatch between expected and actual lift/contact response.
5. Adapt grip force, lift speed, stabilization, or regrasp timing.
6. Compare against no-adaptation and scripted feedback baselines.

## Residual Adaptation Policy

The first learned policy should output residual controller parameters rather
than full low-level torques:

- gripper closure target;
- lift velocity scale;
- hold height target;
- regrasp trigger threshold;
- stabilization duration.

This keeps the problem focused on contact-rich adaptation while the official
Newton Panda hydro scripted prior handles basic approach and grasp structure.
A pretrained checkpoint must not replace this short-term prior unless a
separate checkpoint audit proves code, weights, embodiment, action semantics,
and visual/metric behavior inside Newton.

User-approved short-term route as of 2026-06-27: begin closed-loop adaptation
from the official Newton scripted infant prior. The first adaptation policies
should tune controller parameters around that prior, not learn end-to-end
grasping from scratch and not depend on an unverified checkpoint.

## Dataset And Evaluation

Training rollouts should cover nominal and randomized cup properties. Held-out
mass/friction cells are reserved for testing whether the policy learned a
physical adaptation rule rather than memorizing the grid.

Evaluation must report:

- lift success;
- slip/drop rate;
- excessive-force rate;
- adaptation speed after mismatch;
- success per contact-proxy integral;
- visual success and failure cases with direct paths.

## Completion Criteria

- Adaptation improves at least one key metric without hiding safety failures.
- Results include success, under-grip/drop, over-force, wrong-mass expectation,
  and corrected-adaptation visual cases.
- Direct image paths are recorded in the report.

## Completed Scripted Feedback Nominal Gate

2026-06-27: configured and ran the first scripted feedback baseline on the
nominal existing-cup lift-hold task.

- Config: `experiments/configs/lift_hold_scripted_feedback_baseline_v1.json`.
- Launcher: `experiments/configs/launch_lift_hold_scripted_feedback_baseline_tmux.sh`.
- Controller mode: `lift_hold_feedback`.
- Run tag: `lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545`.
- Output NPZ:
  `experiments/outputs/lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545.npz`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_nominal_cup.md`.

Result:

- fresh official Newton sanity: pass;
- automated visual validation: pass;
- manual visual inspection: pass;
- visual lift-hold behavior: pass;
- strict metric status: fail, only `object_accel_above_threshold`;
- feedback trigger count: 0.

Interpretation: nominal cup verifies the feedback path and logged controller
fields, but it does not yet demonstrate adaptation because no feedback event
was triggered. The next Phase 04 step is to run the scripted feedback baseline
across the same mass/friction grid used by Phase 02, preserving held-out cells.

## Completed Scripted Feedback Ordinary Grid Cells

2026-06-27: ran the first scripted feedback ordinary grid cell.

- Cell: `empty_low`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_low_prefinalize_20260627_1615`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_empty_low_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.07999999821186066` kg observed.
- Applied friction mu: `0.3499999940395355` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: the feedback grid path is runnable on a real mass/friction
variant. This cell still does not demonstrate adaptation because the current
feedback thresholds did not trigger. Continue through the grid before drawing
adaptation claims.

2026-06-27: validated the second scripted feedback ordinary grid cell.

- Cell: `empty_medium`.
- Canonical run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1635`.
- Duplicate run kept as noncanonical output:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_medium_prefinalize_20260627_1630`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_empty_medium_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.07999999821186066` kg observed.
- Applied friction mu: `0.800000011920929` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this second ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the third scripted feedback ordinary grid cell.

- Cell: `half_low`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_half_low_prefinalize_20260627_1700`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_half_low_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.20000000298023224` kg observed.
- Applied friction mu: `0.3499999940395355` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this third ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the fourth scripted feedback ordinary grid cell.

- Cell: `half_medium`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_half_medium_prefinalize_20260627_1725`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_half_medium_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.20000000298023224` kg observed.
- Applied friction mu: `0.800000011920929` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this fourth ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the fifth scripted feedback ordinary grid cell.

- Cell: `half_high`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_half_high_prefinalize_20260627_1745`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_half_high_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.20000000298023224` kg observed.
- Applied friction mu: `1.2000000476837158` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this fifth ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the sixth scripted feedback ordinary grid cell.

- Cell: `full_medium`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_medium_prefinalize_20260627_1805`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_full_medium_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.3499999940395355` kg observed.
- Applied friction mu: `0.800000011920929` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: this sixth ordinary cell remains visually valid and physically
parameterized, but still does not demonstrate feedback adaptation because no
feedback event was triggered.

2026-06-27: validated the seventh and final scripted feedback ordinary grid
cell.

- Cell: `full_high`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_high_prefinalize_20260627_1820`.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_full_high_variant.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.3499999940395355` kg observed.
- Applied friction mu: `1.2000000476837158` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Feedback trigger count: 0.

Interpretation: the ordinary scripted feedback grid is now complete. All cells
are visually valid and correctly parameterized, but no cell triggered the
current feedback rule, so no adaptation-improvement claim is valid yet.

## Held-Out Scripted Feedback Evaluation

2026-06-27: evaluated the first held-out scripted feedback cell.

- Cell: `full_low`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845`.
- Held-out generalization cell: true.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_full_low_heldout.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.3499999940395355` kg observed.
- Applied friction mu: `0.3499999940395355` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Lift height: `0.15313686430454254` m.
- Hold duration: `2.7833306789398193` s.
- Max slip: `0.0034078387381632435` m.
- Contact-loss frames: `0`.
- Max contact proxy: `62.0`.
- Max object acceleration: `8.308707788010144` m/s^2.
- Feedback trigger count: `0`.

Interpretation: `full_low` remains held-out evidence. It is visually valid and
correctly parameterized, but still does not demonstrate feedback adaptation
because the feedback rule did not trigger. `empty_high` remains the last
held-out scripted feedback evaluation cell.

2026-06-27: evaluated the second and final held-out scripted feedback cell.

- Cell: `empty_high`.
- Run tag:
  `lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955`.
- Held-out generalization cell: true.
- Report:
  `experiments/reports/2026-06-27_phase04_scripted_feedback_empty_high_heldout.md`.
- Fresh official Newton sanity: pass.
- Automated visual validation: pass.
- Manual visual inspection: pass.
- Applied object mass: `0.07999999821186066` kg observed.
- Applied friction mu: `1.2000000476837158` observed.
- Strict metric status: fail, only `object_accel_above_threshold`.
- Lift height: `0.16016103327274323` m.
- Hold duration: `2.8333306312561035` s.
- Max slip: `0.0035689078921667837` m.
- Contact-loss frames: `0`.
- Max contact proxy: `62.0`.
- Max object acceleration: `8.308498000056417` m/s^2.
- Feedback trigger count: `0`.

Interpretation: scripted feedback evaluation now covers the nominal cup, seven
ordinary cells, and two held-out cells. All are visually valid and correctly
parameterized, but no cell triggered the current feedback rule, so no
adaptation-improvement claim is valid. The next adaptation step must either
revise the feedback trigger with documented rationale or move to the planned
residual learned controller-parameter adapter.
