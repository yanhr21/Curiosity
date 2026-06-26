# Phase 04 Residual Adapter Training Preflight V1

## Scope

This report records the compute-side preflight for the first learned
residual-controller adapter input pipeline. It does not train a model, create a
checkpoint, or claim learned adaptation.

## Allocation

- tmux session: `curiosity_residual_source_alloc_20260627_034021`.
- Slurm job: `154142`.
- Node: `server56`.
- Environment: prebuilt local `envs/newton/.venv` activated on compute node.
- Resource policy: reused the existing tmux-held one-day GPU allocation; no
  new allocation and no `sbatch`.

## Files

- Config:
  `experiments/configs/residual_adapter_training_preflight_v1.json`.
- Builder:
  `experiments/configs/build_residual_adapter_training_preflight.py`.
- Compute runner:
  `experiments/configs/run_residual_adapter_training_preflight_in_alloc.sh`.
- tmux launcher:
  `experiments/configs/launch_residual_adapter_training_preflight_tmux.sh`.
- Output manifest:
  `data/processed/residual_adapter_training_preflight_v1_20260627/manifest.json`.
- Train split:
  `data/processed/residual_adapter_training_preflight_v1_20260627/residual_adapter_train_records.csv`.
- Validation split:
  `data/processed/residual_adapter_training_preflight_v1_20260627/residual_adapter_validation_records.csv`.

## Final Command

```bash
RUN_TAG=residual_adapter_training_preflight_v1_20260627_0523 \
WINDOW_NAME=residual_adapter_preflight_v1_contract \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
bash experiments/configs/launch_residual_adapter_training_preflight_tmux.sh
```

Final log:
`logs/newton/residual_adapter_training_preflight_v1_20260627_0523.log`.

Fresh official Newton sanity:
`experiments/outputs/residual_adapter_training_preflight_v1_20260627_0523_fresh_newton_sensor_contact_sanity.json`.

## Result

- status: pass;
- fresh official Newton sanity: pass;
- source records: 1800;
- train records: 1440;
- validation records: 360;
- failures: [];
- generated T-Rex fields: [];
- schema promotion: blocked;
- training started: false;
- model created: false.

Train cells:

- `half_low`;
- `empty_low`;
- `half_medium`;
- `full_high`.

Validation cell:

- `empty_medium`.

Held-out cells still reserved for future evaluation:

- `full_low`;
- `empty_high`.

## Columns

Feature columns:

- `newton.panda.sim_time`;
- `newton.contact.rigid_contact_count`;
- `newton.object.body_q.z`;
- `candidate.controller.phase_index`.

Teacher context columns:

- `candidate.controller.commanded_gripper_target`;
- `candidate.controller.commanded_lift_target`.

Target columns:

- `candidate.controller.feedback_active`;
- `candidate.controller.feedback_lift_velocity_scale`;
- `candidate.controller.feedback_hold_height_offset_m`;
- `candidate.controller.feedback_stabilization_extension_s`.

The final manifest confirms nonzero `feedback_active` targets in both splits:
963 nonzero frames in train and 240 nonzero frames in validation.

## Failed Preflight Attempts

- `residual_adapter_training_preflight_v1_20260627_0517`: failed because CSV
  writing did not project source rows to the configured output columns.
- `residual_adapter_training_preflight_v1_20260627_0520`: failed because the
  adapter contract did not yet explicitly expose `generated_trex_fields=[]` and
  `schema_promotion=blocked`.

Both failures were fixed directly. The final pass did not relax source, split,
or no-training gates.

## Interpretation

The training-input preflight is now functional and compute-verified. The next
technical step is the actual learned residual-adapter trainer implementation.
That trainer must consume this preflight manifest, rerun fresh official Newton
sanity, preserve held-out split checks, train for the required GPU duration,
and report metrics/visual evidence before any learned-adaptation claim.
