# Newton 8c501 Benchmark Status

Date: 2026-07-01

Classification: runtime sanity acceptable around 80 FPS. This is not
dense tactile export, not Gate review, not training, and not curiosity success.

## Runs

`external/newton_8c501` at
`8c501b47847569fecdda97a9f7f01205c6f7964f` was tested inside Curiosity Slurm
job `160854` on `server30` with `NVIDIA H200`.

| Run | Duration | FPS | Runtime Decision |
| --- | ---: | ---: | --- |
| `p00_bench_8c501_hot_v1_20260701_162700` | `30.01s` | `80.1` | acceptable |
| `p00_bench_8c501_hot_r2_v1_20260701_162800` | `60.00s` | `80.8` | acceptable |

Both runs executed successfully and are around 80 FPS. The old `82 FPS` number
is a historical diagnostic reference, not a hard blocker.

## Decision

Do not treat `8c501...` as blocked by FPS. It may proceed to dense tactile
export/reference comparison/channel audit/Gate review inside a Curiosity
tmux-held Slurm allocation.

The current stronger runtime evidence remains d58:
`p00_bench_d58_hot_v1_20260701_070611` at `82.7 FPS`.

## Next Action

Continue with the 8c501 dense tactile export path when compute is available.
Gate 00F still requires official UniVTAC/TaCauchy/IsaacLab TacSL sanity or an
accepted faithful blocker.
