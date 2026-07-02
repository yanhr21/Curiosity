# Phase 00 Core Asset Generation H200 Report

- run tag: `phase00_core_asset_generation_h200_long_repair2_20260629_183216`
- status: `incomplete_manual_review_or_blockers`
- slurm job: `157630`
- hostname: `server29`
- gpu names: `NVIDIA H200`
- generation profile: `{'minimum_num_steps': 1800, 'num_steps': 1800, 'sample_steps': None, 'pre_record_warmup_steps': '60', 'final_hold_duration_s': '12.0', 'hold_duration_min_s': '8.0', 'video_frame_stride': '3', 'video_fps': '20', 'short_rollout_refusal_enabled': True}`
- generated pending manual review: `11`
- blocked or failed: `0`

This report is asset-generation evidence only. It is not training and not a curiosity success claim.

## Cells

- `train/train_box_heavy_low_offset` family `box_procedural` proxy `official_cube_box_proxy` mask `alternating_mask` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_cylinder_light_medium` family `cylinder_procedural` proxy `official_pen_cylinder_like_proxy` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_cylinder_heavy_low` family `cylinder_procedural` proxy `official_pen_cylinder_like_proxy` mask `contact_only_masked_vision` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `train/train_cup_half_low_misleading` family `cup_like_official` proxy `official_cup_asset` mask `alternating_mask` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `validation/val_cup_empty_medium_hidden` family `cup_like_official` proxy `official_cup_asset` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `validation/val_box_medium_high_offset` family `box_procedural` proxy `official_cube_box_proxy` mask `alternating_mask` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `validation/val_cylinder_medium_low` family `cylinder_procedural` proxy `official_pen_cylinder_like_proxy` mask `contact_only_masked_vision` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `held_out/heldout_cup_full_low_hidden` family `cup_like_official` proxy `official_cup_asset` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `held_out/heldout_cup_empty_high_misleading` family `cup_like_official` proxy `official_cup_asset` mask `alternating_mask` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `held_out/heldout_box_heavy_low_large_offset` family `box_procedural` proxy `official_cube_box_proxy` mask `vision_contact` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
- `held_out/heldout_cylinder_heavy_low_masked_vision` family `cylinder_procedural` proxy `official_pen_cylinder_like_proxy` mask `contact_only_masked_vision` status `generated_pending_manual_review` blocker `` video `True` contact_sheet `True`
