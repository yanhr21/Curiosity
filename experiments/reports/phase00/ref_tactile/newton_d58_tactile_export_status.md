# Newton d58 Tactile Export Status

Status: candidate dense tactile/mechanics evidence only. This is not curiosity
training, not curiosity success, and not Gate 00D/00E completion.

Run:
- tag: `p00_mjw_d58_marker_v1_20260701_071248`
- Newton root: `external/newton_d58`
- Newton commit: `d58e70266be0db803261f3e46a2f7d923a43db37`
- Slurm job: `160467`
- summary: `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_summary.json`
- video: `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile.avi`
- sheet: `experiments/visuals/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_sheet.jpg`
- npz: `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_mjw_d58_marker_v1_20260701_071248/candidate_mjw_direct_tactile_timeseries.npz`

Observed metrics:
- frames: `240`
- frames with pad-object contacts: `147`
- max object lift: `0.22254392504692078 m`
- max candidate Fn sum: `40.08497619628906`
- max candidate Ft sum: `12.025492668151855`
- left/right marker-flow norms: `3.722446918487549` / `3.3947927951812744`
- left/right contact-area proxy cell ratios: `0.279296875` / `0.2880859375`
- material override request: `mu=0.3`, `kh=1000000000000`
- material override observed: `mu=0.30000001192092896`, `kh=999999995904`

Manual visual inspection:
the sheet is nonblank. It shows synchronized scene frames, left/right
marker-flow panels, Fn/Ft heatmaps, normal/shear overlays, force curves, and
contact-area proxy response during grasp/lift/hold.

Remaining gates:
`direct_tactile_claim_allowed=false`. Reference-video comparison, photometric
marker semantics, true contact-area semantics, and Phase 00 Gate review are
still required before this can be promoted over the older active evidence chain.
