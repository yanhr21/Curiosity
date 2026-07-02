# Phase 01 Newton-Only Dense Tactile Curiosity Plan

## Status

Status: blocked by user pause on 2026-07-01. Do not start allocations,
training, evaluation, data conversion, or further implementation until the
user gives the next instruction.

This phase does not claim Gate 00F completion. It uses the current Newton
`8c501...` grasp/lift/dense tactile evidence as the engineering base and keeps
official UniVTAC/TaCauchy/IsaacLab TacSL validation as a pending comparison
gap.

Gate 00F is a low-priority final validation/comparison-gap track for now. It
must not be treated as the active high-priority blocker or used as a reason to
keep delaying Newton-only training once work resumes.

## Objective

Train a Newton-native curiosity system for the tactile infant:

- closed-loop dense visuo-tactile prediction;
- active probing over grasp/lift/hold and contact-rich perturbations;
- policy improvement over declared baselines without safety regression;
- tactile-mask robustness so the policy cannot collapse to pure vision or
  pure tactile.

## Current Base Evidence

- Newton source: `external/newton_8c501`.
- Runtime: around 80 FPS is accepted for continuation.
- Dense tactile chain:
  `experiments/reports/phase00/ref_tactile/newton_8c501_cont_chain_status.md`.
- Candidate tactile summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile_summary.json`.
- Gate review:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_8c501_cont_20260701_1927/phase00_gate_review_summary.json`.

## Boundary

Official semantic validation remains open:

- UniVTAC official runtime sanity pending.
- TaCauchy official runtime sanity pending.
- IsaacLab TacSL official runtime sanity pending.

Phase 01 may train under this accepted blocker, but results must be labeled
`Newton-only`. They are not final reference-video tactile validation and not
Gate 00F completion.

## Training Contract

Before a success claim, each real training attempt must declare:

- data source and train/validation/held-out split;
- base controller/model and strongest baseline;
- dense tactile fields used: scene RGB, `Fn`, `Ft`, shear direction,
  contact-normal overlay, contact-area proxy, penetration/compression proxy,
  object pose/lift, and force/mechanics time series where available;
- curiosity objective and active probing rule;
- safety metrics: lift/hold/drop/slip/contact force/acceleration;
- videos and dense tactile visualizations for held-out rollouts;
- ablations: vision+tactile, tactile-only masked vision, vision-only, and
  noisy/mismatched tactile.

The five real-training attempt stop gate applies.

## Initial Execution Path

1. Build a Phase 01 training data manifest from current Newton dense tactile
   evidence and any new compute-side rollout generation.
2. Run no-curiosity/scripted baseline evaluation on the same held-out tasks.
3. Train dense visuo-tactile forward prediction and learning-progress scoring.
4. Run closed-loop active probing/adaptation.
5. Compare against baseline and ablations with videos and strict metrics.

All compute work must run inside Curiosity-owned tmux-held Slurm allocations.
No simulation, training, dataset conversion, model loading, rendering, or
heavy Python may run on the login node.
