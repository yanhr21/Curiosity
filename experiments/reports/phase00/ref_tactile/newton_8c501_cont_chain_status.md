# Newton 8c501 Continuation Chain Status

Date: 2026-07-01

Classification: partial positive 8c501 candidate tactile evidence. This is
not Gate 00D/00E/00F completion, not training, and not curiosity success.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/newton_8c501_cont_chain_status_v1.json`

## Execution Context

- Slurm job: `160924`
- Host: `server30`
- GPU: `NVIDIA H200`
- tmux session: `curiosity_phase00_ref_tactile`
- Allocation window: `alloc_8c501_cont`

## Runtime Policy

The historical `82 FPS` reference did not block this run. The active threshold
is around `80 FPS`, and the existing 8c501 benchmark evidence was acceptable:

- `p00_bench_8c501_hot_v1_20260701_162700`: `80.1 FPS`
- `p00_bench_8c501_hot_r2_v1_20260701_162800`: `80.8 FPS`

## Evidence Chain

- Dense tactile export:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile_summary.json`
- Dense tactile video:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile.avi`
- Dense tactile sheet:
  `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_8c501_cont_20260701_1924/candidate_mjw_direct_tactile_sheet.jpg`
- Reference comparison:
  `experiments/outputs/phase00/ref_tactile/ref_compare/p00_refcmp_8c501_cont_20260701_1925/reference_video_compare_summary.json`
- Reference comparison sheet:
  `experiments/visuals/phase00/ref_tactile/ref_compare/p00_refcmp_8c501_cont_20260701_1925/reference_vs_candidate_sheet.jpg`
- Channel audit:
  `experiments/outputs/phase00/ref_tactile/channel_audit/p00_chan_8c501_cont_20260701_1926/channel_semantic_audit_summary.json`
- Gate review:
  `experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_8c501_cont_20260701_1927/phase00_gate_review_summary.json`

## Positive Results

- Dense tactile export status: `pass_candidate_direct_force_export`
- Frames: `240`
- Contact frames: `147`
- Max object lift: `0.22243839502334595 m`
- Max candidate Fn sum: `40.09991455078125`
- Max candidate Ft sum: `12.027889251708984`
- Steel candidate material: `mu=0.30000001192092896`, `kh=999999995904.0`
- Reference comparison assets: `pass_reference_comparison_assets`
- Channel audit: `pass_channel_audit_open_validation`
- Gate review status: `open_not_curiosity_ready`
- Gate review passed checks include:
  `official_newton_runtime_around80_fps`,
  `base_grasp_lift_final_test`,
  `steel_spec_material`,
  `candidate_direct_fn_ft`,
  `sensorcontact_alignment`,
  `normal_and_area_proxy_overlay`,
  `candidate_gel_marker_render`,
  `reference_comparison_assets`,
  and `channel_semantic_layout_audit`.

## Remaining Blockers

- UniVTAC official compute-side sanity has not passed.
- TaCauchy official compute-side sanity has not passed.
- Official IsaacLab TacSL compute-side sanity has not passed.
- Gel/marker photometric semantics are still not validated.
- Photometric/deformation marker tracking is still not validated.
- Contact area is still a point-contact-density proxy, not validated real
  sensor contact area.

## Decision

The 8c501 chain is now the latest-source positive candidate tactile evidence
chain, but it is not final tactile/base/curiosity success. Continue Gate 00F
official semantic validation or faithful blocker documentation before any
curiosity training.
