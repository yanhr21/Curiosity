# Phase 00 Gate 00D/00E/00F Review

- run_tag: `p00_gate_8c501_cont_20260701_1927`
- status: `open_not_curiosity_ready`
- Gate 00D: `open_reference_semantics_blocked`
- Gate 00E: `open_tactile_validation_blocked`
- Gate 00F: `open_official_semantic_validation_blocked`
- curiosity_training_allowed: `False`

## Passed Checks

- `official_newton_runtime_around80_fps`
- `base_grasp_lift_final_test`
- `steel_spec_material`
- `candidate_direct_fn_ft`
- `sensorcontact_alignment`
- `normal_and_area_proxy_overlay`
- `candidate_gel_marker_render`
- `reference_comparison_assets`
- `channel_semantic_layout_audit`
- `semantic_reference_matrix_available`
- `semantic_bridge_spec_available`
- `reference_env_availability`
- `reference_asset_availability`
- `reference_asset_reuse_plan_available`

## Failed Checks

- `univtac_official_reference_sanity`
- `tacauchy_official_reference_sanity`
- `isaaclab_tacsl_official_reference_sanity`

## Hard Blockers

- validated gel/marker photometric semantics comparable to the reference video
- validated photometric/deformation marker tracking on the pad surface
- validated real contact-area semantics beyond the current point-contact-density proxy
- validated channel-level semantic equivalence beyond current layout audit
- UniVTAC official reference sanity not passed
- TaCauchy official reference sanity not passed
- Official IsaacLab TacSL reference sanity not passed

## Evidence

- benchmark_summary: `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_8c501_hot_r2_v1_20260701_162800/newton_hydro_benchmark_summary.json`
- candidate_summary: `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile_summary.json`
- candidate_video: `/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile.avi`
- candidate_sheet: `/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile_sheet.jpg`
- reference_compare_summary: `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_8c501_cont_20260701_1925/reference_video_compare_summary.json`
- reference_compare_sheet: `/public/home/yanhongru/Curiosity/experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_8c501_cont_20260701_1925/reference_vs_candidate_sheet.jpg`
- alignment_summary: `experiments/outputs/phase00/ref_tactile/mujoco_align/p00_mjw_align_v1_20260701_055200/mjw_sensor_alignment_summary.json`
- channel_audit_summary: `experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_8c501_cont_20260701_1926/channel_semantic_audit_summary.json`
- semantic_reference_matrix: `experiments/configs/phase00/ref_tactile/semantic_validation_reference_matrix_v1.json`
- semantic_bridge_spec: `experiments/configs/phase00/ref_tactile/semantic_bridge_spec_v1.json`
- reference_env_availability_summary: `experiments/outputs/phase00/ref_tactile/envprep/availability/reference_env_availability_status.json`
- reference_asset_availability_summary: `experiments/configs/phase00/ref_tactile/envprep/reference_asset_availability_v1.json`
- reference_asset_reuse_plan: `experiments/configs/phase00/ref_tactile/envprep/reference_asset_reuse_plan_v1.json`
- univtac_sanity_summary: `None`
- tacauchy_sanity_summary: `None`
- isaaclab_tacsl_sanity_summary: `None`

This review is intentionally conservative. Candidate force-derived tactile visuals do not close the gate until tactile semantics are validated against the reference standard.
