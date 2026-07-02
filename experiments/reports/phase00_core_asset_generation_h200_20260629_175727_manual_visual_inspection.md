# Phase 00 Core Asset Generation H200 Manual Visual Inspection

Status: partial pass with blockers.

This is asset-generation evidence only. It is not training, not a curiosity
success claim, and not a final dataset completion claim.

## Result

- H200 run: `phase00_core_asset_generation_h200_20260629_175727`
- Slurm job: `157615`
- Host: `server36`
- GPU: `NVIDIA H200`
- Generated cells with visual artifacts: `12`
- Blocked cells: `3`
- Manual visual pass: `12`

All generated contact sheets inspected were nonblank and showed the Panda
robot, tabletop, target object, and three camera panels. The raw visual sheets
are suitable as first visual evidence for the generated cells.

## Important Limitations

- The three blocked cells are all center-of-mass offset box cells. The current
  exporter does not yet implement faithful COM-offset authoring.
- Box and cylinder families currently use available official Newton proxy
  objects: cube proxy for box and pen/cylinder-like proxy for cylinder.
- Modality masks are stored in exported arrays under `candidate.modality.*`;
  the raw contact sheets intentionally remain unmasked Newton rollout visuals
  for visual inspection.
- Mass/friction changes are physics/data parameters and are not necessarily
  visually obvious in contact sheets.

## Evidence

- Aggregate H200 summary:
  `experiments/outputs/phase00_core_asset_generation_h200_20260629_175727_phase00_core_asset_generation_h200_summary.json`
- Manual visual inspection JSON:
  `experiments/outputs/phase00_core_asset_generation_h200_20260629_175727_manual_visual_inspection.json`
- H200 run report:
  `experiments/reports/phase00_core_asset_generation_h200_20260629_175727_phase00_core_asset_generation_h200.md`

## Next Required Repair

Implement faithful center-of-mass offset authoring for box assets, then rerun
the three blocked COM-offset cells on a tmux-held H200 allocation.
