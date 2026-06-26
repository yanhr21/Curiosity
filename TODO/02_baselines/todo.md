# Phase 02 TODO: Baselines

- [x] Select short-term infant prior: official Newton Panda hydro scripted
      grasp/lift path, not a pretrained checkpoint.
- [x] Record short-term stable method: use the official Newton Panda hydro
      scripted controller as the non-learned infant prior, fix real Newton
      physics variation before mass/fill baselines, then train residual
      controller adaptation instead of an end-to-end grasp policy first.
- [x] Record user approval on 2026-06-27 to use the short-term stable Newton
      scripted-prior route now; checkpoint audits are secondary and must not
      block the feedback-adaptation and residual-controller path.
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
- [x] Rerun no-adaptation nominal cube baseline with `controller_mode=lift_hold`
      and extract full metrics:
      `lift_hold_no_adaptation_scripted_baseline_v1_nominal_cube_lifthold_v2_20260627_0255`.
      This run passed official sanity, camera export, visual validation,
      manual visual inspection, and full metrics with `hold_duration_s=4.316662549972534`,
      `max_slip_m=0.007660537484248558`, and `status=success`.
- [x] Run baseline on nominal cup:
      `lift_hold_no_adaptation_scripted_baseline_v1_nominal_cup_existing_asset_lifthold_20260627_0915`.
      This run passed official sanity, camera export, visual validation, and
      manual visual inspection. Full metrics correctly marked it as `fail`
      because `max_object_accel_m_s2=8.308498000056417` exceeded the schema
      threshold `8.0`, even though lift, hold duration, slip, drop, and contact
      loss passed.
- [x] Implement and sanity-check a real mass/friction variant adapter before
      running mass/fill variants. Do not create variant runs by changing only
      labels such as `MASS_LABEL` or `FRICTION_LABEL`; the Newton model must
      actually change object mass/inertia and contact friction with provenance
      recorded in the summary.
      Attempted runtime model-array mutation on 2026-06-27:
      `physics_variant_adapter_sanity_cup_mass15_friction06_20260627_0945`
      proved the requested values were written into summary/NPZ provenance
      (`body_mass_scale=1.5`, `shape_friction_scale=0.6`) but failed visual
      validation because only 4 frames were sampled. Follow-up 5-frame runs
      repeatedly failed with Warp CUDA illegal memory access during
      SensorTiledCamera/export cleanup. Stop this runtime mutation path; next
      implementation must move physics changes to the official builder/finalize
      path or a documented Newton API, then rerun the fresh sanity/export/visual
      gate.
      Completed replacement path:
      `physics_variant_adapter_prefinalize_sanity_cup_mass15_friction06_v2_20260627_1125`
      passed fresh official Newton sanity, camera export, visual validation, and
      manual visual inspection for adapter sanity. Summary provenance confirmed
      cup mass changed from `0.10100987856276333` to observed
      `0.15151481330394745`, and shape friction changed from `1.0` to observed
      `0.6000000238418579`. This is not a task-success claim because diagnostic
      task metrics still failed with `hold_duration_below_min`.
- [ ] Run baseline across mass/fill variants after the real physics-parameter
      adapter passes a fresh official sanity/export/visual gate.
      Started grid execution with `empty_medium`:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_medium_prefinalize_20260627_1140`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.307760545609415`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. Continue remaining grid cells without
      lowering thresholds.
      Continued grid execution with `half_medium`:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_half_medium_prefinalize_20260627_1155`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.308443857335977`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. Continue remaining grid cells without
      lowering thresholds.
      Completed the medium-friction mass axis with `full_medium`:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_full_medium_prefinalize_20260627_1205`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.308498000056417`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. The remaining grid cells are low/high
      friction variants and held-out generalization cells.
      Started low-friction axis with `empty_low`:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_low_prefinalize_20260627_1320`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.308707937632189`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. Continue low/high friction cells without
      using held-out `full_low` or `empty_high` as training/grid-completion cells.
      Continued low-friction axis with `half_low`:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_half_low_prefinalize_20260627_1335`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.308498000056417`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. `full_low` remains held out.
      Evaluated held-out `full_low` as no-adaptation evidence:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_full_low_prefinalize_20260627_1350`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.308390712127508`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. Keep it labeled as held-out evidence for
      later learned adaptation comparisons.
      Started ordinary high-friction axis with `half_high`:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_half_high_prefinalize_20260627_1415`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.308498000056417`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. `empty_high` remains held out.
      Completed ordinary high-friction axis with `full_high`:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_full_high_prefinalize_20260627_1430`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.308498000056417`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. This completes the ordinary mass/friction
      grid while preserving held-out `full_low` and `empty_high`.
      Evaluated held-out `empty_high` as no-adaptation evidence:
      `lift_hold_no_adaptation_scripted_baseline_v1_cup_empty_high_prefinalize_20260627_1445`.
      The run passed fresh official Newton sanity, camera export, visual
      validation, and manual visual inspection. Full metrics correctly marked
      it as `fail` only because `max_object_accel_m_s2=8.308498000056417`
      exceeded the schema threshold `8.0`; lift, hold, slip, drop, contact-loss,
      and contact-proxy gates passed. This completes the 3x3 no-adaptation
      physics-variant evaluation grid; `full_low` and `empty_high` remain
      held-out evidence for later learned adaptation comparisons.
- [x] Write Phase 02 no-adaptation nominal cup report after compute run and
      manual visual inspection:
      `experiments/reports/2026-06-27_phase02_no_adaptation_nominal_cup_baseline.md`.
