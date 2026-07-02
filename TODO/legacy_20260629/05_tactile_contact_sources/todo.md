# Phase 05 TODO: Tactile And Contact Sources

- [x] Decide whether the first post-pivot tactile/contact run uses only Newton
      contact proxies.
      Decision: use Newton contact proxies first, because the Phase 04 lift-hold
      rollouts have real `newton.panda.rigid_contact_count`, object pose,
      controller, camera, visual-validation, and manual-inspection evidence.
      Evidence:
      `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`.
- [x] If using Taccel marker evidence, keep it under `taccel.marker.*`.
      Evidence: current Phase 05/06 mainline keeps Newton contact evidence
      under `newton.*` / `candidate.*`, keeps Taccel marker evidence
      provenance-only, and does not promote marker evidence into
      `observation.tactile_f6` or `observation.tactile_deform.*`.
      The Phase 06 reassessment records this as still blocked for strict T-Rex
      promotion:
      `experiments/reports/2026-06-27_phase06_trex_bridge_source_reassessment_v1.md`.
- [x] Do not create `observation.tactile_f6` without calibrated nonzero F6.
      Evidence: Phase 05 Newton contact manifest generated no
      `observation.tactile_f6` field; `generated_trex_fields=[]` and
      `schema_promotion=blocked`.
- [x] Do not create `observation.tactile_deform.*` without real dense tactile
      deformation streams and visual validation.
      Evidence: Phase 05 Newton contact manifest generated no
      `observation.tactile_deform.*` fields; `generated_trex_fields=[]` and
      `schema_promotion=blocked`.
- [x] Convert the first real Newton contact/tactile source without T-Rex schema
      promotion.
      Evidence:
      `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`
      and
      `experiments/reports/2026-06-27_phase05_newton_contact_source_manifest_v1.md`.
      Status: pass; `record_count=3600`; `source_run_count=10`;
      `generated_trex_fields=[]`; `schema_promotion=blocked`.
- [x] Add tactile stream to both policy input and forward-model prediction
      target once a real tactile/contact source passes gates.
      Evidence: `experiments/configs/residual_adapter_forward_model_contract_v1.json`
      routes the active Newton contact stream
      `newton.contact.rigid_contact_count` into both residual-adapter inputs
      and forward-model target `contact_proxy_next_step`. Real
      `taccel.marker.*` and `taccel.ftac.*` remain blocked until nonzero,
      visually inspected source evidence exists.
- [x] Add training-time modality masks: both visible, vision masked, tactile
      masked, partial vision mask, and partial tactile mask.
      Evidence: `experiments/configs/residual_adapter_forward_model_contract_v1.json`
      defines `both_visible`, `vision_masked_touch_visible`,
      `touch_masked_vision_visible`, `partial_vision_mask`, and
      `partial_touch_mask`.
- [x] Add post-contact pure tactile windows so stabilization and slip detection
      can run without continuous visual input.
      Evidence: `experiments/configs/residual_adapter_forward_model_contract_v1.json`
      defines `post_contact_pure_touch_window` with vision-mask probability
      curriculum `0.3 -> 0.6` and touch-mask probability curriculum
      `0.1 -> 0.2`.
- [x] Report vision-only, tactile-only, vision+tactile, shuffled tactile, and
      delayed tactile ablations.
      Evidence:
      `experiments/reports/2026-06-27_phase05_contact_proxy_ablation_report_v1.md`
      summarizes existing validated Phase 03 replay ablations across 9
      mass/friction rollouts. Current labels are contact-proxy diagnostics:
      object-motion-only proxy, Newton contact-proxy tactile/contact-only,
      object+contact proxy, shuffled contact proxy, and delayed contact proxy.
      This is not a trained policy ablation and not tactile F6 evidence.
- [x] Record direct image paths for every tactile/contact visual.
      Evidence: the Phase 05 manifest links each source run to its Phase 04
      contact sheet, frame browser, visual-validation JSON, manual visual
      inspection JSON, metrics JSON, and source NPZ.
