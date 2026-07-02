# Active Dense Tactile Evidence Manifest

Date: 2026-07-01

Machine-readable record:
`experiments/configs/phase00/dense_tactile_infant/active_evidence_manifest_20260701_v1.json`

## What Ran

Inside Curiosity tmux-held Slurm job `160989` on `server64`, the active Phase
00 runner produced a new Newton 8c501 dense tactile/base export under the
active `dense_tactile_infant` layout.

Base run:
`p00_dense_8c501_base_cop_20260701_2030`

Reference comparison:
`p00_dense_refcmp_8c501_base_cop_20260701_2032`

Hydro compression/penetration proxy run:
`p00_dense_hydro_compress_8c501_20260701_2040`

## Positive Evidence

- Base export status: `pass_candidate_direct_force_export`.
- Official final test status: `pass`.
- Frames: `240`.
- Frames with pad-object contact: `146`.
- Max object lift: `0.22242838144302368 m`.
- Max pad-object contact count: `50`.
- Max candidate `Fn` sum: `40.21227264404297`.
- Max candidate `Ft` sum: `12.041539192199707`.
- Left/right candidate `Fn` map max:
  `8.891204833984375` / `9.644493103027344`.
- Left/right candidate `Ft` map max:
  `2.6673614978790283` / `2.893348455429077`.
- Left/right contact-area proxy map max:
  `8.957447052001953` / `8.968476295471191`.
- Left/right marker-flow norm max:
  `3.7791855335235596` / `5.611800193786621`.
- Candidate center-of-pressure proxy valid frames:
  left `146`, right `146`.
- Steel candidate material update passed with requested `mu=0.3` and
  `kh=1e12`.
- Reference-vs-candidate comparison assets passed and both videos are nonblank.
- Hydro compression/penetration proxy export passed with nonblank scene camera,
  max left/right deform proxy maps
  `0.0023627502378076315` / `0.0015500170411542058`, max contact-area proxy
  sum `0.0033083378802984953 m^2`, max Fn proxy `22603.349609375`, and max
  stress proxy `6832237.5`.

## Manual Visual Inspection

The candidate contact sheet shows a nonblank Panda scene with object contact.
Before contact, the pad force panels are dark. During grasp/lift/hold,
left/right candidate `Fn`, `Ft`, area-proxy, marker-flow, and mechanics curves
activate in sync with the visual scene.

The reference-vs-candidate sheet shows that the active Newton candidate now has
the right broad structure: scene frames, left/right tactile panels, force maps,
marker-like rendering, and time-series curves. The reference video remains
richer and more semantically complete, especially in photometric marker
semantics and tactile channel density.

The hydro compression sheet shows synchronized scene camera frames plus
left/right hydro proxy Fn, stress, shear, and deform maps. Contact phases show
nonzero deform/penetration proxy maps aligned with the visual grasp/lift/hold.

## Boundary

This is positive active Phase 00 candidate evidence only.

It is not:

- official tactile semantic validation;
- real contact-area validation;
- photometric GelSight/marker validation;
- T-Rex schema compatibility;
- training;
- curiosity success.

Known remaining gaps:

- compression/penetration are present as Newton hydro deform/penetration
  proxies, but official tactile semantic validation is still missing;
- contact area is still a proxy;
- center of pressure is a candidate force-map weighted proxy, not validated
  hardware CoP;
- candidate `Fn/Ft` is not a validated official tactile force field;
- closed-loop curiosity training has not started.
