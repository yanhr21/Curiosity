# Phase 06 TODO: T-Rex Bridge Reassessment

- [ ] Reassess T-Rex bridge only after Newton-native adaptation shows useful
      behavior.
- [x] Run reference-only sanity for currently staged official T-Rex checkpoint
      assets without changing the Newton-native mainline.
      Evidence:
      `experiments/outputs/trex_checkpoint_current_sanity_20260627_2055_integrity.json`,
      `experiments/outputs/trex_checkpoint_current_sanity_20260627_2055_midtrain_model_load.json`,
      and
      `experiments/reports/2026-06-27_phase06_trex_checkpoint_current_sanity.md`.
      Both sanity checks passed. This is not a bridge go decision, not
      training, and not Newton data compatibility.
- [ ] Verify whether a bimanual 62D source exists.
- [ ] Verify whether accepted synchronized cameras exist.
- [ ] Verify whether calibrated nonzero `[10,6]` F6 exists.
- [ ] Verify whether ten dense deformation streams exist.
- [ ] Run strict inventory only if all source groups exist.
- [ ] Make explicit go/no-go decision.
