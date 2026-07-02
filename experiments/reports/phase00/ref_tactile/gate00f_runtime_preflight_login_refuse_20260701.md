# Gate 00F Runtime Preflight Login Refuse Check

- Date: `2026-07-01`
- Classification: `launcher_safety_check_not_training_not_gate_completion`
- Command:
  `bash experiments/configs/phase00/ref_tactile/run_gate00f_runtime_preflight_in_alloc.sh`
- Expected exit code: `2`
- Observed exit code: `2`
- Observed message: `ERROR: must run inside a Slurm allocation.`

Result: `pass_refused_login_node_execution`.

This confirms the runtime preflight will not run on the login node without a
Slurm allocation. It does not clear Gate 00F.
