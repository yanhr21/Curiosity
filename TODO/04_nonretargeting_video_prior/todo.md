# TODO 04: Non-Retargeting Video Prior

- [ ] Generate simulation-only reference videos after the base environment is
  reproducible.
- [ ] Define allowed video-derived signals: phase, progress, object motion,
  coarse contact affordance, success/failure.
- [ ] Implement or adapt XIRL-style embedding only after reviewing official
  config and code paths.
- [ ] Implement or adapt GraphIRL-style object graph reward only after mapping
  entities available in the simulator.
- [ ] Add wrong-video and mismatched-video evaluation sets.
- [ ] Add checks that video reward does not encode joint pose or end-effector
  trajectory targets.
- [ ] Report every video experiment as diagnostic until held-out baselines and
  safety metrics exist.

