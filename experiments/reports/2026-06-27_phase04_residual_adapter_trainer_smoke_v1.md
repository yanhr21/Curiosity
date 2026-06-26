# Phase 04 Residual Adapter Trainer Smoke V1

## Scope

This report records the first compute-side smoke test of the Newton-native
residual controller-parameter adapter trainer. It is a trainer-path diagnostic,
not a real training result and not a learned-adaptation claim.

## Environment

- Newton sanity venv: `envs/newton/.venv`.
- Trainer venv: `envs/residual_adapter/.venv`.
- Trainer dependency setup was done locally on the shared filesystem, not on a
  compute node.
- Installed trainer packages include `torch==2.6.0` and `numpy==2.2.6`.
- Final compute device: `cuda:0`, NVIDIA H200.

The first smoke attempt,
`residual_adapter_trainer_v1_smoke_20260627_0527`, passed fresh official
Newton sanity but failed because `envs/newton/.venv` did not contain `torch`.
The runner was then fixed to separate `NEWTON_VENV` and `TRAINER_VENV`.

The second smoke attempt,
`residual_adapter_trainer_v1_smoke_20260627_0534`, passed with torch but showed
a missing-NumPy warning in the trainer venv. `numpy<3` was installed locally in
`envs/residual_adapter/.venv`.

## Final Command

```bash
RUN_TAG=residual_adapter_trainer_v1_smoke_20260627_0539 \
RUN_MODE=smoke \
WINDOW_NAME=residual_adapter_trainer_smoke3 \
JOB_ID=154142 \
TMUX_SESSION=curiosity_residual_source_alloc_20260627_034021 \
bash experiments/configs/launch_residual_adapter_trainer_tmux.sh
```

Final log:
`logs/newton/residual_adapter_trainer_v1_smoke_20260627_0539.log`.

Summary:
`experiments/outputs/residual_adapter_trainer_v1_20260627/residual_adapter_trainer_v1_smoke_20260627_0539_summary.json`.

Fresh official Newton sanity:
`experiments/outputs/residual_adapter_trainer_v1_smoke_20260627_0539_fresh_newton_sensor_contact_sanity.json`.

## Result

- status: pass;
- run mode: smoke;
- fresh official Newton sanity: pass;
- optimizer steps: 3;
- torch version: `2.6.0+cu124`;
- CUDA available: true;
- device: `cuda:0`;
- CUDA device name: `NVIDIA H200`;
- checkpoint written: false;
- real training result: false;
- generated T-Rex fields: [];
- schema promotion: blocked;
- failures: [].

Validation metrics from the smoke diagnostic:

- loss: `1.626072883605957`;
- active BCE: `0.6624462008476257`;
- continuous MSE: `0.9636266231536865`;
- feedback-active accuracy: `0.6666666865348816`.

## Interpretation

The actual trainer entrypoint now imports PyTorch/CUDA, consumes the preflight
manifest, runs the GRU residual-controller adapter path, computes validation
metrics, and preserves all no-T-Rex/no-checkpoint smoke gates. This clears the
trainer-import/smoke blocker.

It does not clear the real-training gate. A real run must use
`RUN_MODE=train`, satisfy the one-GPU one-hour rule, monitor GPU utilization,
write a checkpoint, and then run held-out evaluation/visual gates before any
learned-adaptation claim.
