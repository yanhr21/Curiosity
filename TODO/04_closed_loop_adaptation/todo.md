# Phase 04 TODO: Closed-Loop Adaptation

- [x] Run no-adaptation policy on nominal cup.
      Evidence: Phase 02 nominal cup baseline.
- [x] Run no-adaptation policy across mass/fill variants.
      Evidence: Phase 02 3x3 mass/friction grid.
- [x] Use the user-approved short-term stable route: start adaptation from the
      official Newton Panda hydro scripted infant prior, not from an unverified
      pretrained checkpoint.
      Evidence: scripted feedback baseline config and controller mode.
- [x] Run scripted adaptive baseline across mass/fill variants.
      Nominal cup gate complete:
      `lift_hold_scripted_feedback_baseline_v1_nominal_cup_20260627_1545`.
      Ordinary grid cells complete: `empty_low`, `empty_medium`, `half_low`,
      `half_medium`, `half_high`, `full_medium`, `full_high`.
      Held-out `full_low` evaluation complete:
      `lift_hold_scripted_feedback_baseline_v1_cup_full_low_heldout_prefinalize_20260627_1845`.
      This run passed fresh official Newton sanity, camera export, visual
      validation, manual visual inspection, lift, hold, slip, drop,
      contact-loss, and contact-proxy gates. Full metrics still failed only on
      `object_accel_above_threshold` with
      `max_object_accel_m_s2=8.308707788010144`. Feedback did not trigger
      (`feedback_trigger_count=0`).
      Held-out `empty_high` evaluation complete:
      `lift_hold_scripted_feedback_baseline_v1_cup_empty_high_heldout_prefinalize_20260627_1955`.
      This run passed fresh official Newton sanity, camera export, visual
      validation, manual visual inspection, lift, hold, slip, drop,
      contact-loss, and contact-proxy gates. Full metrics still failed only on
      `object_accel_above_threshold` with
      `max_object_accel_m_s2=8.308498000056417`. Feedback did not trigger
      (`feedback_trigger_count=0`). Scripted feedback evaluation is complete,
      but no adaptation-improvement claim is valid because feedback never
      triggered.
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
