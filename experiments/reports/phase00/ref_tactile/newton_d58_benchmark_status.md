# Newton d58 Benchmark Status

Date: 2026-07-01

This records the latest-upstream Newton runtime sanity. It is not training, not
curiosity success, and not Gate 00D/00E completion.

Machine-readable status:
`experiments/configs/phase00/ref_tactile/newton_d58_benchmark_status_v1.json`

## Runs

Run `p00_bench_d58_v1_20260701_070459`:

- commit: `d58e70266be0db803261f3e46a2f7d923a43db37`
- host/job: `server02` / `160467`
- FPS: `70.8`
- frames/elapsed: `717` / `10.13` s
- status: pass execution, below 82 FPS target
- summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_d58_v1_20260701_070459/newton_hydro_benchmark_summary.json`

Run `p00_bench_d58_hot_v1_20260701_070611`:

- commit: `d58e70266be0db803261f3e46a2f7d923a43db37`
- host/job: `server02` / `160467`
- FPS: `82.7`
- frames/elapsed: `2482` / `30.01` s
- status: pass, meets 82 FPS target
- summary:
  `experiments/outputs/phase00/ref_tactile/newton_hydro/p00_bench_d58_hot_v1_20260701_070611/newton_hydro_benchmark_summary.json`

## Interpretation

The latest upstream Newton worktree now has runtime-positive evidence on H200:
the hot/longer benchmark meets the 82 FPS target. Follow-up d58 tactile export,
reference comparison, channel audit, and Gate review have also run:

- tactile export: `p00_mjw_d58_marker_v1_20260701_071248`
- reference compare: `p00_refcmp_d58_marker_v1_20260701_071521`
- channel audit: `p00_chan_d58_marker_v1_20260701_071757`
- Gate review: `p00_gate_d58_marker_v1_20260701_071843`,
  `open_not_curiosity_ready`

The remaining blocker is semantic validation, not runtime.
