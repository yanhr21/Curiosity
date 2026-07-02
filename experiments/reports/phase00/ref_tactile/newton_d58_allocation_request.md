# Newton d58 Allocation Request

Date: 2026-07-01

This records the d58 allocation lifecycle. It is not training, curiosity
success, or Gate completion.

Machine-readable status:
`experiments/configs/phase00/ref_tactile/newton_d58_allocation_request_v1.json`

## Request

- tmux session: `curiosity_phase00_ref_tactile`
- tmux window: `alloc_d58`
- job name: `curiosity_p00_d58_1gpu_1day`
- observed Slurm job id: `160467`
- observed state: granted on `server02`; evidence runs completed; cancellation
  requested after the d58 Gate review
- resource request: `gpu:NVIDIAH200:1`, `1-00:00:00`, `8` CPUs, `64G`
- log:
  `logs/newton/phase00/ref_tactile/allocation/curiosity_p00_d58_1gpu_1day_20260701_070340.log`

## Runs Completed

- `p00_bench_d58_v1_20260701_070459`
- `p00_bench_d58_hot_v1_20260701_070611`
- `p00_mjw_d58_marker_v1_20260701_071248`
- `p00_refcmp_d58_marker_v1_20260701_071521`
- `p00_chan_d58_marker_v1_20260701_071757`
- `p00_gate_d58_marker_v1_20260701_071843`

The allocation was not used for training. It was released after the evidence
chain reached the current official semantic-validation blocker.
