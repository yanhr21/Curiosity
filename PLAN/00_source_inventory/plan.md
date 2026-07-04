# Plan 00: Source Inventory And Boundaries

## Purpose

Build a defensible source inventory before any simulation or training. The
goal is to know which codebases are executable candidates, which are only
references, and which assets are forbidden for the first phase.

## Current Local Sources

- `external/IsaacLab-Arena` commit `8a74e79`: primary simulation platform.
- `external/GR00T-VisualSim2Real` commit `92bf086`: fallback G1
  loco-manipulation control stack.
- `external/WBC-AGILE` commit `7259792`: fallback whole-body RL/control and
  evaluation stack.
- `external/SUGAR` commit `01fe123`: video-driven humanoid
  loco-manipulation reference/baseline, not first core.
- `external/google-research-xirl` commit `62457e1`: cross-embodiment video
  reward component.
- `external/graph-inverse-rl` commit `7d06634`: object-centric video reward
  component.
- `external/WholebodyVLA` commit `7a86f5c`: resource list and conceptual VLA
  reference; README says no concrete codebase release timeline.

## Deferred Or URL-Only Sources

- FALCON: force-adaptive humanoid loco-manipulation reference. Clone later
  with sparse checkout if needed.
- SplitAdapter: load-aware factorized adaptation paper. Treat as reference
  until official code is available and verified.
- VIP: visual reward reference. Do not download Ego4D or other real video
  data in the first phase.
- Vid2Robot: video-conditioned policy reference. Do not treat as RL or
  unknown-load carrying solution.

## Forbidden In This Phase

- T-Rex and real-scene-only models.
- Real robot datasets.
- Real human video datasets.
- SUGAR processed data and demo checkpoints.
- GraphIRL datasets or trained rewards.
- XIRL datasets.
- VIP pretrained weights or Ego4D data.
- Any model/data download that requires loading, conversion, or processing on
  a login node.

## Exit Criteria

- `docs/execution_path_2026-07-02.md` exists and names the selected primary
  platform and fallback stacks.
- Every local external repository has commit recorded.
- The first executable platform is selected before compute-node preflight.
- No real dataset or real-scene-only model has been downloaded.

