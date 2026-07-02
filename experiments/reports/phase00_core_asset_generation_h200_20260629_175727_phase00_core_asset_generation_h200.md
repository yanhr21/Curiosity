# Phase 00 Core Asset Generation H200 Report

- run tag: `phase00_core_asset_generation_h200_20260629_175727`
- status: `incomplete_manual_review_or_blockers`
- slurm job: `157615`
- hostname: `server36`
- gpu names: `NVIDIA H200`
- generated pending manual review: `12`
- blocked or failed: `3`

This report is asset-generation evidence only. It is not training and not a curiosity success claim.

## Cells

- `train/train_cup_quarter_low_hidden` family `cup_like_official` proxy `official_cup_asset` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_cup_half_medium_truthful` family `cup_like_official` proxy `official_cup_asset` mask `alternating_mask` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_cup_three_quarter_high_truthful` family `cup_like_official` proxy `official_cup_asset` mask `contact_only_masked_vision` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_box_light_medium_center` family `box_procedural` proxy `official_cube_box_proxy` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_box_heavy_low_offset` family `box_procedural` proxy `official_cube_box_proxy` mask `alternating_mask` status `blocked_com_offset_export_not_yet_supported` blocker `Current exporter does not yet implement center-of-mass offset authoring for cell=train_box_heavy_low_offset.` video `False` contact_sheet `False`
- `train/train_cylinder_light_medium` family `cylinder_procedural` proxy `official_pen_cylinder_like_proxy` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_cylinder_heavy_low` family `cylinder_procedural` proxy `official_pen_cylinder_like_proxy` mask `contact_only_masked_vision` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_cup_half_low_misleading` family `cup_like_official` proxy `official_cup_asset` mask `alternating_mask` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `validation/val_cup_empty_medium_hidden` family `cup_like_official` proxy `official_cup_asset` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `validation/val_box_medium_high_offset` family `box_procedural` proxy `official_cube_box_proxy` mask `alternating_mask` status `blocked_com_offset_export_not_yet_supported` blocker `Current exporter does not yet implement center-of-mass offset authoring for cell=val_box_medium_high_offset.` video `False` contact_sheet `False`
- `validation/val_cylinder_medium_low` family `cylinder_procedural` proxy `official_pen_cylinder_like_proxy` mask `contact_only_masked_vision` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `held_out/heldout_cup_full_low_hidden` family `cup_like_official` proxy `official_cup_asset` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `held_out/heldout_cup_empty_high_misleading` family `cup_like_official` proxy `official_cup_asset` mask `alternating_mask` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `held_out/heldout_box_heavy_low_large_offset` family `box_procedural` proxy `official_cube_box_proxy` mask `vision_contact` status `blocked_com_offset_export_not_yet_supported` blocker `Current exporter does not yet implement center-of-mass offset authoring for cell=heldout_box_heavy_low_large_offset.` video `False` contact_sheet `False`
- `held_out/heldout_cylinder_heavy_low_masked_vision` family `cylinder_procedural` proxy `official_pen_cylinder_like_proxy` mask `contact_only_masked_vision` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
