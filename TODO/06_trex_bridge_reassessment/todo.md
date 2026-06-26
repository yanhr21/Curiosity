# Phase 06 TODO: T-Rex Bridge Reassessment

- [x] Reassess T-Rex bridge only after Newton-native adaptation shows useful
      behavior.
      Evidence: Phase 04 learned residual adapter passed held-out `full_low`
      and `empty_high`, extra ordinary `half_high` and `full_medium`, and the
      four-cell failure-mode comparison. Reassessment report:
      `experiments/reports/2026-06-27_phase06_trex_bridge_source_reassessment_v1.md`.
- [x] Run reference-only sanity for currently staged official T-Rex checkpoint
      assets without changing the Newton-native mainline.
      Evidence:
      `experiments/outputs/trex_checkpoint_current_sanity_20260627_2055_integrity.json`,
      `experiments/outputs/trex_checkpoint_current_sanity_20260627_2055_midtrain_model_load.json`,
      and
      `experiments/reports/2026-06-27_phase06_trex_checkpoint_current_sanity.md`.
      Both sanity checks passed. This is not a bridge go decision, not
      training, and not Newton data compatibility.
- [x] Verify whether a bimanual 62D source exists.
      Result: missing for the current Newton mainline. Newton strict
      inventories list `observation.state`, `action`, and `action_abs` as
      missing.
- [x] Verify whether accepted synchronized cameras exist.
      Result: missing as accepted T-Rex keys for the current Newton mainline.
      Real `newton.camera.*` evidence exists and has visual gates, but it is
      not promoted to `observation.images.*`.
- [x] Verify whether calibrated nonzero `[10,6]` F6 exists.
      Result: missing for the current Newton mainline. Newton contact proxy
      remains `newton.*` evidence and is not promoted to F6.
- [x] Verify whether ten dense deformation streams exist.
      Result: missing. Strict inventories list all ten
      `observation.tactile_deform.l0-l4/r0-r4` streams as missing.
- [x] Run strict inventory only if all source groups exist.
      Result: no new strict promotion inventory was run because source groups
      are missing. Existing dense Newton strict inventories remain `blocked`
      with 17 missing fields and 0 incompatible fields.
- [x] Make explicit go/no-go decision.
      Decision: no-go for strict T-Rex bridge promotion now; keep T-Rex as
      reference-only and continue Newton-native adaptation.
