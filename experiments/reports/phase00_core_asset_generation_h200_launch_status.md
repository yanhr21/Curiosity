# Phase 00 Core Asset Generation H200 Launch Status

Date: 2026-06-29

## Status

Status: long-horizon Phase 00 asset generation has completed across the primary
long run plus the filtered long repair run. Phase 00 asset generation evidence
is now complete for the 15-cell catalog, but this is still not training
evidence and not a curiosity success claim.

This is not simulation evidence, not asset validation evidence, not training,
and not a curiosity success claim.

## Allocation Request

### Long-Horizon Rerun Queue

- tmux session: `curiosity_phase00_h200_asset_long_alloc`
- allocation window: `alloc`
- watcher window: `phase00_long_wait`
- Slurm job: `157630`
- job name: `curiosity_phase00_long_asset_generation`
- requested resources: `--partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:NVIDIAH200:1 --time=1-00:00:00`
- current observed state: `RUNNING`
- current observed node: `server29`
- run tag: `phase00_core_asset_generation_h200_long_20260629_182052`
- main log: `logs/newton/phase00_core_asset_generation_h200_long_20260629_182052.srun.log`
- first observed result: fresh official Newton SensorContact sanity passed in
  the H200 allocation before the first cell export started
- first completed long-profile cell:
  `train/train_cup_quarter_low_hidden` passed with 1800 simulated steps,
  601 rollout GIF frames, longest hold `26.883333333333333` seconds, and
  `success_all_worlds=true`
- second completed long-profile cell:
  `train/train_cup_half_medium_truthful` passed with 1800 simulated steps,
  601 rollout GIF frames, longest hold `26.85` seconds, and
  `success_all_worlds=true`
- current observed progress: 2 cell rows written; third train cell is running
  in the H200 allocation
- allocation GPU utilization spot check through
  `srun --jobid=157630 --overlap ... nvidia-smi`: `NVIDIA H200, 67 %, 941 MiB`
- later failure: the first long run reached 5 rows but stopped after
  `train_cylinder_light_medium` because an in-allocation script invocation saw
  a transient malformed argument line and exited `127`; this run remains
  partial evidence, not completion.
- COM repair: exporter glue now passes `OBJECT_COM_OFFSET_XYZ` to Newton
  `ModelBuilder.body_com` under `candidate.physics.object_com_offset_xyz`,
  with observed COM written back to summaries.
- filtered repair run: `phase00_core_asset_generation_h200_long_repair2_20260629_183216`
  launched in the same tmux-held H200 allocation, window
  `phase00_asset_h200_long_repair2`, for the 11 missing/blocked cells.
- repair2 first observed state: fresh official Newton SensorContact sanity
  passed for `train_box_heavy_low_offset`, then entered Newton camera export;
  H200 utilization spot check reported `NVIDIA H200, 68 %, 941 MiB`.
- repair2 first completed COM-offset cell:
  `train/train_box_heavy_low_offset` generated under the 1800-step profile with
  requested/updated/observed COM x offset `0.014` m, 601 rollout GIF frames,
  longest hold `27.116666666666667` seconds, `success_all_worlds=true`, and
  visual/metric artifacts present.
- repair2 progress update: 4/11 filtered missing/blocked cells have generated
  rows; completed cells are `train_box_heavy_low_offset`,
  `train_cylinder_light_medium`, `train_cylinder_heavy_low`, and
  `train_cup_half_low_misleading`; current observed cell is
  `validation/val_cup_empty_medium_hidden`.
- repair2 progress update: 7/11 filtered rows written. Validation cells now
  include `val_cup_empty_medium_hidden`, `val_box_medium_high_offset`, and
  `val_cylinder_medium_low`; the run has entered held-out generation.
- repair2 progress update: 10/11 filtered rows written. Held-out COM-offset
  cell `heldout_box_heavy_low_large_offset` generated a row; current observed
  final cell is `heldout_cylinder_heavy_low_masked_vision`.
- repair2 completion: 11/11 filtered missing/blocked cells generated rows and
  wrote aggregate summary/report paths:
  `experiments/outputs/phase00_core_asset_generation_h200_long_repair2_20260629_183216_phase00_core_asset_generation_h200_summary.json`
  and
  `experiments/reports/phase00_core_asset_generation_h200_long_repair2_20260629_183216_phase00_core_asset_generation_h200.md`.
- combined coverage check: 15/15 catalog cells have long-horizon generated rows
  across `phase00_core_asset_generation_h200_long_20260629_182052` and
  `phase00_core_asset_generation_h200_long_repair2_20260629_183216`.
- long-horizon checks: every generated cell summary reports `num_steps=1800`,
  rollout GIF `frame_count=601`, and video status `pass`.
- COM checks: the three COM-offset cells have matching requested, updated, and
  observed COM offsets: `0.014`, `0.012`, and `0.018` m on x.
- manual visual inspection: 15/15 contact sheets passed. Evidence:
  `experiments/reports/phase00_core_asset_generation_h200_long_combined_manual_visual_inspection.md`.
- intended profile: `PHASE00_NUM_STEPS=1800`,
  `PHASE00_PRE_RECORD_WARMUP_STEPS=60`,
  `PHASE00_FINAL_HOLD_DURATION=12.0`,
  `PHASE00_HOLD_DURATION_MIN=8.0`,
  `PHASE00_VIDEO_FRAME_STRIDE=3`,
  `PHASE00_VIDEO_FPS=20`
- watcher behavior: job `157630` reached `RUNNING`, and the watcher invoked
  `experiments/configs/launch_phase00_core_asset_generation_h200_tmux.sh` with
  `WINDOW_NAME=phase00_asset_h200_long` and the long-horizon profile.

### Completed Short-Horizon Run

- tmux session: `curiosity_phase00_h200_asset_alloc`
- allocation window: `alloc`
- watcher window: `phase00_wait_launch`
- Slurm job: `157615`
- requested resources: `--partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=1-00:00:00`
- initial observed state: `PENDING`
- initial observed reason: `Priority`
- final observed execution: job `157615` ran on `server36` with `NVIDIA H200`
  and completed the Phase 00 runner with exit code `0`
- allocation release: `scancel 157615` issued after runner completion; the job
  no longer appears in `squeue`

## Launch Path

When job `157615` became `RUNNING`, the watcher launched:

```bash
JOB_ID=157615 \
TMUX_SESSION=curiosity_phase00_h200_asset_alloc \
WINDOW_NAME=phase00_asset_h200 \
RUN_TAG=phase00_core_asset_generation_h200_<timestamp> \
bash experiments/configs/launch_phase00_core_asset_generation_h200_tmux.sh
```

That launcher then runs the actual validation through:

```bash
srun --jobid=157615 --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  bash -lc 'bash experiments/configs/run_phase00_core_asset_generation_h200_in_alloc.sh'
```

## Guardrails

- The runner refuses to run without `SLURM_JOB_ID`.
- The runner refuses to run unless `nvidia-smi` reports H200.
- The runner writes logs under `logs/newton/`.
- The runner writes aggregate evidence under `experiments/outputs/` and
  `experiments/reports/`.
- Login-node Newton simulation/rendering/training remains forbidden.

## Known Limitation

The current exporter can generate official cup, official cube-proxy, and
official pen/cylinder-like proxy cells. It does not yet faithfully author
center-of-mass offsets for box cells, so COM-offset cells will be recorded as
blockers unless the exporter is extended before the H200 run reaches them.

The completed run used the old 450-step short profile. It is pipeline evidence
only and does not satisfy the active long-horizon Phase 00 completion gate.

## Result Artifacts

- Aggregate summary:
  `experiments/outputs/phase00_core_asset_generation_h200_20260629_175727_phase00_core_asset_generation_h200_summary.json`
- H200 report:
  `experiments/reports/phase00_core_asset_generation_h200_20260629_175727_phase00_core_asset_generation_h200.md`
- Manual visual inspection:
  `experiments/outputs/phase00_core_asset_generation_h200_20260629_175727_manual_visual_inspection.json`
