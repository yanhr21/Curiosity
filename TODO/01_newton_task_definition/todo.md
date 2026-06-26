# Phase 01 TODO: Newton Task Definition

- [x] Write the variable-mass cup lift-and-hold task spec:
      `docs/lift_hold_variable_mass_cup_task_spec.md`.
- [x] Define mass/fill-level grid in
      `experiments/configs/lift_hold_variable_mass_cup_task_v1.json`.
- [x] Define friction grid in
      `experiments/configs/lift_hold_variable_mass_cup_task_v1.json`.
- [x] Define object pose randomization in
      `experiments/configs/lift_hold_variable_mass_cup_task_v1.json`.
- [x] Define success metrics.
- [x] Define failure metrics.
- [x] Define visual gate requirements and direct image output paths. The gate
      is defined, and the first official Panda hydro gate passed as
      `lift_hold_variable_mass_cup_v1_official_panda_gate_20260627_0021`.
- [x] Select the first Newton scene entry point:
      `external/newton/newton/examples/robot/example_robot_panda_hydro.py`.
- [x] Validate the task spec:
      `python3 experiments/configs/validate_lift_hold_variable_mass_cup_task_v1.py`
      passed with `failures=[]`, 3 mass levels, 3 friction levels, 13 required
      signals, 6 curiosity terms, and `schema_promotion=blocked`.
- [x] Run the first clean Newton visual gate for Phase 01: reused allocation
      `154023`, reran fresh official Newton `sensor_contact` sanity, exported
      9 SensorTiledCamera frames, manually inspected the contact sheet plus
      frames `0000`, `0120`, and `0239`, and cleared downstream use only for
      task-spec/visual evidence.
- [x] Run the next cup-asset gate where the grasped object path is adapted to
      the local cup asset, after fresh official sanity and manual visual
      inspection. The gate passed for cup-asset retarget and visual evidence as
      `lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105`,
      with `tracked_object=existing_cup_asset`,
      `generated_trex_fields=[]`, and `schema_promotion=blocked`.
- [ ] Tune cup grasp initialization and object physical parameters. Current
      cup-asset gate shows the cup visible and retargeted, with
      `max_lift=0.15901388227939606`, but the final inspected frame shows the
      cup tilted/fallen, so stable cup grasp success is still pending.
- [x] Add an extended final-hold controller parameter and rerun the cup hold
      visual gate. Run
      `lift_hold_variable_mass_cup_v1_existing_cup_hold_gate_20260627_0145`
      passed official sanity and visual validation with
      `final_hold_duration=999.0`, `final_object_z=0.30836987495422363`, and
      `max_lift=0.15986861288547516`.
- [ ] Run a longer-window cup hold metric gate, for example 360 frames or more,
      before claiming the configured full two-second hold success metric.
