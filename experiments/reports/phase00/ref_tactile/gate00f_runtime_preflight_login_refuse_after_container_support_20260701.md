# Gate 00F Runtime Preflight Login Refuse After Container Support

- Date: `2026-07-01`
- Classification: `login_refuse_check_not_runtime_not_gate_completion`
- Script:
  `experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`

## Command

```bash
RUN_TAG=p00_runtime_preflight_login_refuse_after_container_support_20260701 \
bash experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh
```

## Result

- Exit code: `2`
- Stderr:
  `ERROR: must run inside a Slurm allocation.`

## Gate Effect

The runtime preflight still refuses login-node execution after the
registry-authoritative path and container module-preflight support were added.
No registry validation, container command, module import, simulation,
rendering, training, or official sanity ran.
