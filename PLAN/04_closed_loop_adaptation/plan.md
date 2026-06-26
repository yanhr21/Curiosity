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
