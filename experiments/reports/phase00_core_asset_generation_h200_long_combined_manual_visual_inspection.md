# Phase 00 Long-Horizon Manual Visual Inspection

Date: 2026-06-29

Status: pass for long-horizon asset-generation visual inspection.

This is manual visual inspection evidence for Phase 00 asset generation only.
It is not training evidence and not a curiosity success claim.

## Runs Inspected

- Primary long run:
  `phase00_core_asset_generation_h200_long_20260629_182052`
- Filtered long repair run:
  `phase00_core_asset_generation_h200_long_repair2_20260629_183216`
- Slurm job: `157630`
- H200 node: `server29`
- GPU evidence: `NVIDIA H200`

## Inspection Criteria

Each contact sheet was checked for:

- nonblank rendered frames;
- visible Panda robot;
- visible tabletop/workspace;
- visible target object or official proxy object;
- visible `head_proxy`, `right_wrist_proxy`, and `left_wrist_proxy` panels;
- no obvious broken camera panel, missing object, or blank rollout.

Raw contact sheets are unmasked visual evidence. Modality masks are represented
in exported arrays under `candidate.modality.*`.

## Cells

- `train/train_cup_quarter_low_hidden`: pass.
- `train/train_cup_half_medium_truthful`: pass.
- `train/train_cup_three_quarter_high_truthful`: pass.
- `train/train_box_light_medium_center`: pass.
- `train/train_box_heavy_low_offset`: pass; COM offset recorded in data.
- `train/train_cylinder_light_medium`: pass.
- `train/train_cylinder_heavy_low`: pass.
- `train/train_cup_half_low_misleading`: pass.
- `validation/val_cup_empty_medium_hidden`: pass.
- `validation/val_box_medium_high_offset`: pass; COM offset recorded in data.
- `validation/val_cylinder_medium_low`: pass.
- `held_out/heldout_cup_full_low_hidden`: pass.
- `held_out/heldout_cup_empty_high_misleading`: pass.
- `held_out/heldout_box_heavy_low_large_offset`: pass; COM offset recorded in data.
- `held_out/heldout_cylinder_heavy_low_masked_vision`: pass.

## Remaining Limitations

- Box and cylinder families currently use available official Newton proxy
  objects rather than custom authored USD assets.
- This validates Phase 00 asset generation, long-horizon rollout media, contact
  sheets, and metric artifact presence. It does not validate or complete any
  downstream curiosity training claim.
