# Phase 04 TODO: Closed-Loop Adaptation

- [x] Run no-adaptation policy on nominal cup.
      Evidence: Phase 02 nominal cup baseline.
- [x] Run no-adaptation policy across mass/fill variants.
      Evidence: Phase 02 3x3 mass/friction grid.
- [x] Use the user-approved short-term stable route: start adaptation from the
      official Newton Panda hydro scripted infant prior, not from an unverified
      pretrained checkpoint.
      Evidence: scripted feedback baseline config and controller mode.
- [ ] Run scripted adaptive baseline across mass/fill variants.
      Nominal cup gate complete:
      `lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545`.
      Ordinary grid cells complete: `empty_low`, `empty_medium`, `half_low`,
      `half_medium`, `half_high`, `full_medium`.
      Remaining ordinary cells: `full_high`.
- [ ] Run contact-aware curiosity diagnostic across mass/fill variants.
- [ ] Train the first learned adapter as residual controller-parameter output,
      not full low-level torque control.
- [x] Reserve held-out mass/friction cells for generalization evaluation.
      Evidence: Phase 02/04 configs preserve `full_low` and `empty_high` as
      held-out cells.
- [ ] Compare adaptation speed and failure modes.
- [x] Save direct visual paths for success and failure cases.
      Evidence: nominal scripted feedback report records contact sheet and
      frame browser paths; grid reports still pending.
