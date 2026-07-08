# Prismatic Showcase Visual Manifest

Timestamp: 2026-07-07 03:58 CST.

This is a presentation manifest only. It is not a new success claim.

## Best Current Visual

- One-page showcase:
  `slides/2026-07-07_isaac_carry_showcase.html`
- GIF:
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_fallback.gif`
- Poster:
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_fallback_poster.png`
- Render summary:
  `experiments/visuals/prismatic_reference_showcase/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/prismatic_reference_presentation_fallback_summary.json`
- Source rollout summary:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/core_world_prismatic_carrier_stand_summary.json`
- Source state CSV:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/core_world_prismatic_carrier_stand_state.csv`
- Corrected checker:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_probe_adaptive_10kg_mid_matched_cpu/reference_check_corrected.json`
- Posture/load after-retry suite:
  `experiments/outputs/core_world_prismatic_carrier_stand/20260707_prismatic_reference_posture_load_suite/prismatic_reference_posture_load_suite_after_retry_summary.json`

## What It Shows

- A side-view prismatic carrier scaffold with carrier body, four driven
  prismatic legs/feet, physical cradle, free box, target line, path trace,
  phase label, support-foot count, and rollout metrics.
- The source run used a 10 kg free box in `cradle_free_box` mode,
  `guarded_prelift_quasistatic_step_cycle`, active probe for 80 steps, and
  probe-adaptive gait/posture decisions.
- Key source metrics: fall/drop `0/0`, body root pose/velocity writes `0/0`,
  box pose writes `0`, post-settle payload travel `-0.1799 m`, final
  post-settle target distance `0.0099 m`, max post-settle relative offset
  `0.0116 m`, and max tilt `0.1064 rad`.
- Additional corrected-checker robustness: the after-retry posture/load suite
  is `status=pass` with 4/4 cases passing: nominal 10 kg mid carry,
  near-chest high 12 kg, long-reach low 8 kg, and bulky 10 kg. This strengthens
  the scaffold evidence but does not change the non-success boundary.

## What Must Not Be Claimed

- Not an Isaac camera render.
- Not humanoid walking.
- Not learned carrying.
- Not final box-carrying success.
- Not evidence that the G1 wrapper generalizes across postures and loads.

## Fresh Validation Status

- `169008` / `prism_ref` was an invalid configuration failure because
  horizontal legs were not enabled.
- `169019` / `prism_ref_cpu` was a negative short 760-step rerun.
- `169026` / `prism_ref_mcpu` produced the matched 2880-step rollout. Slurm
  recorded it as failed because the old checker used an over-strict global
  relative-error gate, but `reference_check_corrected.json` passes with the
  corrected post-settle relative-error gate.
