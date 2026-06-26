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
      is defined but remains `pending_compute_run`.
- [x] Select the first Newton scene entry point:
      `external/newton/newton/examples/robot/example_robot_panda_hydro.py`.
- [x] Validate the task spec:
      `python3 experiments/configs/validate_lift_hold_variable_mass_cup_task_v1.py`
      passed with `failures=[]`, 3 mass levels, 3 friction levels, 13 required
      signals, 6 curiosity terms, and `schema_promotion=blocked`.
