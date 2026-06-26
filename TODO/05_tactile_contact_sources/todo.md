# Phase 05 TODO: Tactile And Contact Sources

- [x] Decide whether the first post-pivot tactile/contact run uses only Newton
      contact proxies.
      Decision: use Newton contact proxies first, because the Phase 04 lift-hold
      rollouts have real `newton.panda.rigid_contact_count`, object pose,
      controller, camera, visual-validation, and manual-inspection evidence.
      Evidence:
      `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`.
- [ ] If using Taccel marker evidence, keep it under `taccel.marker.*`.
- [ ] Do not create `observation.tactile_f6` without calibrated nonzero F6.
- [ ] Do not create `observation.tactile_deform.*` without real dense tactile
      deformation streams and visual validation.
- [x] Convert the first real Newton contact/tactile source without T-Rex schema
      promotion.
      Evidence:
      `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`
      and
      `experiments/reports/2026-06-27_phase05_newton_lift_hold_contact_source_manifest.md`.
      Status: pass; `record_count=3600`; `source_run_count=10`;
      `generated_trex_fields=[]`; `schema_promotion=blocked`.
- [ ] Add tactile stream to both policy input and forward-model prediction
      target once a real tactile/contact source passes gates.
- [ ] Add training-time modality masks: both visible, vision masked, tactile
      masked, partial vision mask, and partial tactile mask.
- [ ] Add post-contact pure tactile windows so stabilization and slip detection
      can run without continuous visual input.
- [ ] Report vision-only, tactile-only, vision+tactile, shuffled tactile, and
      delayed tactile ablations.
- [x] Record direct image paths for every tactile/contact visual.
      Evidence: the Phase 05 manifest links each source run to its Phase 04
      contact sheet, frame browser, visual-validation JSON, manual visual
      inspection JSON, metrics JSON, and source NPZ.
