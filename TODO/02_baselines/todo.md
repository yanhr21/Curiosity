# Phase 02 TODO: Baselines

- [x] Select short-term infant prior: official Newton Panda hydro scripted
      grasp/lift path, not a pretrained checkpoint.
- [x] Implement no-adaptation scripted grasp baseline launch path from the
      selected Newton Panda hydro prior:
      `experiments/configs/lift_hold_no_adaptation_baseline_v1.json` and
      `experiments/configs/launch_lift_hold_no_adaptation_baseline_tmux.sh`.
- [ ] Implement scripted feedback adaptation baseline.
- [ ] Decide whether the first learned baseline is BC, diffusion policy,
      ACT-style, or another documented method.
- [x] Search for real Newton-compatible basic grasping checkpoints before
      training any grasp policy from scratch; no directly usable Newton-native
      Panda grasp/lift checkpoint found.
- [ ] If a pretrained policy is considered, write a checkpoint audit covering
      codebase, checkpoint path, license, embodiment, action semantics, camera
      requirements, and smoke-test command.
- [x] Keep the official Newton Panda hydro scripted grasp/lift controller as
      the default infant prior for the short-term baseline path.
- [x] Define shared metrics table schema:
      `experiments/configs/lift_hold_metrics_schema_v1.json`.
- [x] Add controller provenance fields to rollout export under
      `candidate.controller.*`.
- [x] Run no-adaptation baseline on the official nominal cube prior:
      `lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_20260627_0210`.
- [x] Write Phase 02 no-adaptation nominal cube baseline report:
      `experiments/reports/2026-06-27_phase02_no_adaptation_nominal_baseline.md`.
- [x] Implement metrics extractor for `lift_hold_metrics_schema_v1.json`
      covering hold duration, slip, drop, contact-loss, and acceleration terms.
- [x] Rerun metrics extraction for nominal official cube baseline. The
      extractor completed in allocation `154023` and correctly reported the
      baseline as `fail` with `hold_duration_s=1.3166654109954834`,
      `max_slip_m=0.09295262564260072`, and failure reasons
      `hold_duration_below_threshold` and `slip_above_threshold`.
- [x] Add `controller_mode=lift_hold` to the no-adaptation baseline launcher
      so the baseline uses official approach/grasp/lift waypoints without the
      release/place segment.
- [ ] Rerun no-adaptation nominal cube baseline with `controller_mode=lift_hold`
      and extract full metrics.
- [ ] Run baseline on nominal cup.
- [ ] Run baseline across mass/fill variants.
- [ ] Write Phase 02 no-adaptation nominal cup report after compute run and
      manual visual inspection.
