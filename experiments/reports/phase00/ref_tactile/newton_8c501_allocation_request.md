# Newton 8c501 Allocation Request

Date: 2026-07-01

Classification: Slurm allocation request only. This is not a benchmark result,
not tactile export, not Gate review, and not training.

## Request

- tmux session: `curiosity_phase00_ref_tactile`
- tmux window: `alloc_8c501`
- job name: `curiosity_p00_8c501_1gpu_1day`
- job id: `160854`
- partition: `gpu`
- gres: `gpu:NVIDIAH200:1`
- time limit: `1-00:00:00`
- cpus per task: `8`
- memory: `64G`
- log:
  `logs/newton/phase00/ref_tactile/allocation/curiosity_p00_8c501_1gpu_1day_20260701_162519.log`

Initial `squeue` state: `PENDING`, reason `Priority`.

## Intended Use

Run the `external/newton_8c501` handoff sequence after the job is RUNNING:
runtime benchmark first, then dense tactile export, reference-video comparison,
channel audit, and Gate review if each previous stage passes.
