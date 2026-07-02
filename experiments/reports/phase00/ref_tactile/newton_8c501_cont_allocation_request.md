# Newton 8c501 Continuation Allocation Request

Date: 2026-07-01

Classification: allocation request only. This is not tactile export, not Gate
review, not training, and not curiosity success.

Machine-readable record:
`experiments/configs/phase00/ref_tactile/newton_8c501_cont_allocation_request_v1.json`

## Request

- tmux session: `curiosity_phase00_ref_tactile`
- tmux window: `alloc_8c501_cont`
- job name: `curiosity_p00_8c501_cont_1gpu_1day`
- job id: `160924`
- partition: `gpu`
- gres: `gpu:NVIDIAH200:1`
- time limit: `1-00:00:00`
- initial state: `PENDING`
- initial reason: `Priority`
- final observed state after release request: `COMPLETING`
- Slurm observed `EndTime`: `2026-07-01T19:27:36`
- Slurm observed `ExitCode`: `0:9`
- log:
  `logs/newton/phase00/ref_tactile/allocation/curiosity_p00_8c501_cont_1gpu_1day_20260701_192127.log`

## Intended Use

Run the corrected `8c501...` continuation chain after the allocation is
running:

1. Dense tactile export.
2. Reference-video comparison.
3. Channel semantic audit.
4. Gate review.

The `82 FPS` historical reference must not block this chain; the existing
`80.1/80.8 FPS` runtime evidence is acceptable for continuation.

The allocation was released after the 8c501 dense tactile export, reference
comparison, channel audit, and Gate review completed, to avoid idle GPU usage.
