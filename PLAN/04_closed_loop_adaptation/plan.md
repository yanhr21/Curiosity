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

## Completion Criteria

- Adaptation improves at least one key metric without hiding safety failures.
- Results include success, under-grip/drop, over-force, wrong-mass expectation,
  and corrected-adaptation visual cases.
- Direct image paths are recorded in the report.
