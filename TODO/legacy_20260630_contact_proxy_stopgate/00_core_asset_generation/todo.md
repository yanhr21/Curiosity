# Phase 00 Core Data Asset Generation TODO

## Active Rules

- Phase 00 is the new starting point for data and asset generation; all earlier
  phase-numbered work is legacy evidence only.
- Active TODO files live directly under `TODO/`; `legacy_<date>/` folders are
  historical archives.
- Do not run simulation, rendering, validation builders, dataset conversion,
  model loading, training, or NumPy/PyTorch-heavy asset checks on the login
  node.
- Real asset generation and validation must run in a Curiosity-specific
  tmux-held H200 allocation using `srun`/`salloc`.
- Future visual outputs must be grouped by phase, for example
  `experiments/visuals/phase00/<run_tag>/`; do not add new flat run
  directories directly under `experiments/visuals/`.
- Full Phase 00 asset generation must use the long H200 profile: at least
  1800 simulated steps per catalog cell, 60 warmup steps, 12 second final hold,
  8 second minimum hold, and dense rollout video evidence. Shorter runs are
  diagnostics only and cannot satisfy completion.
- Do not claim curiosity training is complete from this asset phase.
- Do not create fake tactile, fake T-Rex, or downgraded placeholder fields.
- Do not commit unless the user explicitly asks.

## Completed Preflight

- [x] Move previous phase records into archive directories.
- [x] Rename old date archive to `legacy_20260629` so it cannot be mistaken
      for active work.
- [x] Keep active Phase 00 plan/todo directly under `PLAN/` and `TODO/`.
- [x] Create a first structured core asset catalog.
- [x] Generate static design visualizations for arena/split inspection.
- [x] Record that static visualizations are not real Newton asset evidence.
- [x] Search and record official Newton/OpenUSD asset-generation references.

## Required H200 Preparation

- [x] Add tmux-held H200 Phase 00 launch script.
      Evidence:
      `experiments/configs/launch_phase00_core_asset_generation_h200_tmux.sh`.
- [x] Add in-allocation Phase 00 asset validation runner.
      Evidence:
      `experiments/configs/run_phase00_core_asset_generation_h200_in_alloc.sh`.
- [x] Add Phase 00 report template or aggregate report writer.
      Evidence: aggregate report is written by the in-allocation runner to
      `experiments/reports/<run_tag>_phase00_core_asset_generation_h200.md`.
- [x] Require H200 evidence before generation starts.
      Evidence: runner refuses to proceed unless `nvidia-smi` reports H200.
- [x] Require fresh official Newton sanity in the same allocation.
      Evidence: runner delegates each generated cell through
      `run_newton_panda_hydro_camera_export_v2_in_alloc.sh`, whose first step
      is fresh official Newton SensorContact sanity in the same allocation.
- [x] Require cached official Newton assets and prebuilt local venvs.
      Evidence: launcher and delegated exporter check local venvs and cached
      Newton assets before running.
- [x] Refuse login-node execution for any simulation/rendering/validation.
      Evidence: in-allocation runner exits if `SLURM_JOB_ID` is absent; launcher
      only creates tmux `srun --jobid=<held_job>` work.
- [x] Add compute-side modality mask postprocess.
      Evidence: in-allocation runner applies `candidate.modality.*` masks on
      H200 after Newton export for contact-only, vision-only, and alternating
      mask cells.
- [x] Refuse short full-generation runs.
      Evidence: `run_phase00_core_asset_generation_h200_in_alloc.sh` now exits
      if `PHASE00_NUM_STEPS` is below `PHASE00_MIN_NUM_STEPS` and defaults to
      the 1800-step full H200 generation profile.

## Required Real Asset Generation

- [x] Start or reuse a Curiosity-specific tmux-held H200 allocation.
      Progress 2026-06-29: requested Slurm job `157615` in tmux session
      `curiosity_phase00_h200_asset_alloc`, window `alloc`; watcher window
      `phase00_wait_launch` will launch Phase 00 generation when the job becomes
      RUNNING. Current known state at request time: PENDING `(Priority)`.
      Completion update: job `157615` ran on `server36` with `NVIDIA H200`,
      launched window `phase00_asset_h200`, completed the Phase 00 runner with
      exit code 0, and was released after the run.
- [x] Record Slurm job ID, hostname, GPU model, CUDA visibility, env paths,
      and exact commands.
      Progress: recorded in
      `logs/newton/phase00_core_asset_generation_h200_20260629_175727.srun.log`
      and aggregate summary
      `experiments/outputs/phase00_core_asset_generation_h200_20260629_175727_phase00_core_asset_generation_h200_summary.json`.
- [x] Rerun generated cells under the active long-horizon H200 profile.
      Progress: the completed 2026-06-29 H200 run used the old 450-step short
      profile. It remains useful pipeline evidence, but it is not sufficient
      full Phase 00 asset generation evidence.
      Queue update: requested Slurm job `157630` in tmux session
      `curiosity_phase00_h200_asset_long_alloc`, window `alloc`, with watcher
      window `phase00_long_wait`; resources requested are
      `--gres=gpu:NVIDIAH200:1` for one day.
      Running update: job `157630` is now `RUNNING` on `server29`; run tag
      `phase00_core_asset_generation_h200_long_20260629_182052`; main log
      `logs/newton/phase00_core_asset_generation_h200_long_20260629_182052.srun.log`;
      fresh official Newton SensorContact sanity passed before the first cell
      export started. First completed long-profile cell
      `train/train_cup_quarter_low_hidden` passed with 1800 simulated steps,
      601 rollout GIF frames, longest hold `26.883333333333333` seconds, and
      `success_all_worlds=true`. H200 utilization spot check through the held
      allocation reported `NVIDIA H200, 67 %, 941 MiB`.
      Latest progress: 2 long-profile cell rows are written; the second cell
      `train/train_cup_half_medium_truthful` passed with longest hold `26.85`
      seconds and 601 rollout GIF frames; the third train cell is running.
      Failure/repair update: the first long run later reached 5 rows but exited
      `127` after `train_cylinder_light_medium`; COM-offset glue was then added
      through `OBJECT_COM_OFFSET_XYZ` -> `candidate.physics.object_com_offset_xyz`
      -> Newton `ModelBuilder.body_com`. Filtered repair run
      `phase00_core_asset_generation_h200_long_repair2_20260629_183216`
      is running in the same H200 allocation for the 11 missing/blocked cells;
      its first COM-offset cell passed official Newton SensorContact sanity and
      entered camera export. H200 utilization spot check: `NVIDIA H200, 68 %, 941 MiB`.
      COM evidence: `train/train_box_heavy_low_offset` completed under the
      1800-step profile with requested/updated/observed COM x offset `0.014` m,
      601 rollout GIF frames, longest hold `27.116666666666667` seconds, and
      `success_all_worlds=true`.
      Progress update: repair2 has written 4/11 filtered rows and is currently
      running `validation/val_cup_empty_medium_hidden`.
      Progress update: repair2 has written 7/11 filtered rows, including the
      validation COM-offset cell `val_box_medium_high_offset`, and has entered
      held-out generation.
      Progress update: repair2 has written 10/11 filtered rows, including the
      held-out COM-offset cell `heldout_box_heavy_low_large_offset`; final
      filtered cell `heldout_cylinder_heavy_low_masked_vision` is running.
      Completion update: combined primary long run plus filtered repair2 run
      covers 15/15 catalog cells. Every generated summary reports
      `num_steps=1800`, rollout GIF `frame_count=601`, and video status `pass`.
- [x] Render every train cell from
      `experiments/configs/phase00_core_tabletop_asset_catalog_v1.json`.
      Progress: completed 8/8 train cells under the long-horizon H200 profile.
- [x] Render every validation cell from the catalog.
      Progress: completed 3/3 validation cells under the long-horizon H200
      profile.
- [x] Render every held-out cell from the catalog without using it for tuning.
      Progress: completed 4/4 held-out cells under the long-horizon H200
      profile as evidence only, not for tuning.
- [x] Produce per-cell frame browser, contact sheet, and full rollout video or
      dense-frame equivalent.
      Progress: produced these artifacts for 15/15 long-horizon cells.
- [x] Produce real MP4 video visualizations, not GIF-only evidence.
      Progress: generated `rollout_video.mp4` for 15/15 Phase 00 catalog cells
      in Curiosity-owned tmux-held H200 Slurm job `157730` on `server53`;
      run tag `phase00_video_mp4_export_h200_20260629_203527`; aggregate
      summary
      `experiments/outputs/phase00_video_mp4_export_h200_20260629_203527_phase00_video_mp4_summary.json`
      reports `status=pass`, `passed_count=15`, `failed_count=0`, 601 frames
      per video at 20 FPS. This is visualization evidence only, not curiosity
      training evidence.
- [x] Update future visual-output layout to avoid flat phase mixing.
      Progress: the Newton camera export launcher now derives
      `VISUAL_PHASE_DIR` from `RUN_TAG` by default, so future runs write under
      `experiments/visuals/<phase>/...`; Phase 00 aggregate rows now record
      `experiments/visuals/phase00/...`.
- [x] Clean current `experiments/visuals` directory.
      Progress: `experiments/visuals` now keeps only the final Phase 00 MP4
      visualization set under `experiments/visuals/phase00/`: 15 final
      long-horizon cell directories, each with `rollout_video.mp4`. Non-final
      short-run, legacy phase07/phase08, residual, design, and superseded
      visual directories were moved out of `experiments/visuals` to
      `legacy/experiments_visuals_archive_20260629_cleanup/`.
- [x] Produce per-cell contact/contact-proxy traces and lift/hold metrics.
      Progress: produced metrics JSON for 15/15 long-horizon cells.
- [x] Produce per-cell visual validation JSON.
      Progress: produced visual validation JSON for 15/15 long-horizon cells.
- [x] Produce per-cell manual visual inspection record.
      Progress: manual visual inspection passed for 15/15 contact sheets at
      `experiments/reports/phase00_core_asset_generation_h200_long_combined_manual_visual_inspection.md`.
- [x] Produce aggregate Phase 00 H200 report.
      Progress: primary and filtered-repair aggregate reports exist; combined
      evidence is summarized in
      `experiments/reports/phase00_core_asset_generation_h200_launch_status.md`.

## Required Gates Before Training

- [x] Confirm all real generation artifacts were produced on H200, not the
      login node.
- [x] Confirm no held-out leakage into source collection, threshold tuning,
      hyperparameter selection, or controller repair.
- [x] Confirm contact-only masked-vision cells have H200-applied modality masks.
      Evidence is `candidate.modality.*` metadata in generated summaries and
      masked exported arrays. This is an asset/data gate, not proof that a
      trained policy has learned to use contact.
- [x] Confirm visual/contact balance is represented in the asset catalog.
      Evidence: the generated cells include `vision_contact`,
      `contact_only_masked_vision`, `vision_only_masked_contact`, and
      `alternating_mask` modes. This does not claim downstream training success.
- [x] Confirm no contact-limit overflow, mass/inertia blocker, blank video, or
      mismatched object-family cell remains unresolved.
- [x] Record that Phase 00 asset gates now allow planning the next real
      closed-loop curiosity data collection/training stage.
      This does not start or complete that training stage.

## Current Status

Phase 00 asset generation evidence and real MP4 video visualization export are
complete for the 15-cell catalog under the active 1800-step H200 profile. This
does not complete curiosity training. The next stage is still blocked from
making any curiosity success claim until real closed-loop training/evaluation
satisfies the harder-training contract.
