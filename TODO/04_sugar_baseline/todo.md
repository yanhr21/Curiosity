# TODO 04: Official SUGAR Baseline

> Active boundary (2026-07-13): stop the official refiner at `model_10000.pt`.
> Do not resume Refiner training. The operator clarified that normal functional
> behavior plus visual evidence is sufficient and exact paper numeric equality
> is not required, so official downstream rollout/processing/Tracker/Generator
> work is active from the unmodified model-10000 checkpoint. Keep current and
> pending Curiosity SUGAR allocations instead of releasing them.

- [x] Move the active SUGAR clone from `external/SUGAR` to root-level `SUGAR/`.
- [x] Move the matching IsaacLab v2.3.0 source from its external location to
  root-level `IsaacLab/`, retaining only a compatibility symlink for the active
  editable environment.
- [x] Move reproduction outputs and logs to
  `experiments/sugar_reproduction/{outputs,logs}` without interrupting the
  active Tracker process.
- [x] Keep the entire `experiments/` tree local-only and ignored by Git; never
  commit or push reports, outputs, logs, checkpoints, datasets, videos, or
  visualizations below it.

- [x] Make SUGAR the highest-priority mainline in `AGENTS.md`.
- [x] Confirm official SUGAR repository and local clone:
  `SUGAR`, remote `https://github.com/tianshuwu/SUGAR.git`,
  commit `01fe123` (`fix inference bug`).
- [x] Clone official IsaacLab dependency:
  `IsaacLab`, tag `v2.3.0`, commit `3c6e67b`.
- [x] Download official SUGAR assets on a compute node using
  `scripts/sugar/download_official_sugar_assets.sh`.
- [x] Verify the downloaded official asset directories exist:
  `SUGAR/data/CarryBox`,
  `SUGAR/descriptions/robots/g1`,
  `SUGAR/descriptions/objects/small_box`,
  and `SUGAR/demo_ckpts/CarryBox`.
- [x] Identify or prepare a cluster-safe Python 3.11 environment matching
  SUGAR's official requirements without running dependency installation on the
  login node. Current audit: only `gr00t_n16_py310` and `isaac_arena_py312`
  venvs were found; no prebuilt SUGAR py311 env with IsaacSim 5.1.0 and
  IsaacLab 2.3.0 is available yet. Compute-node preflight log:
  `experiments/sugar_reproduction/logs/20260711_sugar_env_preflight.log`.
- [x] Add guarded official environment preparation recipe:
  `scripts/sugar/prepare_official_sugar_env.sh`. It encodes the official
  Python 3.11, IsaacSim 5.1.0, IsaacLab v2.3.0, `rsl_rl`, `sugar_rl`, and
  `sugar_il` install path, but refuses login nodes and refuses execution unless
  `SUGAR_ENV_BUILD_APPROVED=1` is set.
- [x] Install or activate official dependencies on a compute node:
  `isaacsim[all,extscache]==5.1.0`, IsaacLab `v2.3.0`, `rsl_rl`, editable
  `source/sugar_rl`, and editable `source/sugar_il`.
- [x] Add a compute-node-only environment preflight script:
  `scripts/sugar/preflight_official_sugar_env.sh`. It checks official asset
  paths and package metadata before inference, without importing IsaacSim or
  launching simulation.
- [x] Run a finite official-code CarryBox inference smoke on a GPU compute node
  with `scripts/sugar/run_official_sugar_carrybox_inference.sh`.
- [x] Record the SUGAR inference log, host, allocation/job id, exact command,
  SUGAR commit, IsaacLab tag, asset presence, and pass/fail cause.
- [x] Record the SUGAR inference rendering path and boundary: official
  `play.py --headless --video` used IsaacLab/Gymnasium `RecordVideo` with
  `render_mode="rgb_array"`; Vulkan/GLFW warnings were present but non-fatal;
  this is not evidence that the separate viewport/rendering-manager true-render
  path is fixed.
- [ ] If inference is blocked, record the blocker exactly. Do not substitute a
  Curiosity G1 scaffold, toy policy, or simplified controller.
- [x] Add a compute-node-only official refiner training launcher:
  `scripts/sugar/run_official_sugar_carrybox_refiner_train.sh`.
- [x] Run a short official refiner training smoke with official SUGAR code,
  data, robot, object, and task:
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_train_smoke_iter1.log`.
- [x] Add a compute-node-only official CarryBox training pipeline launcher:
  `scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh`. It follows
  the official `SUGAR/train.sh CarryBox` stage order: refiner train,
  refiner rollout, refiner rollout processing, tracker train, tracker rollout,
  tracker rollout processing, and generator train.
- [ ] Run the full official CarryBox refiner train stage at official scale
  (`4096` envs, `30001` iterations). Current job:
  `curiosity_sugar_refiner_full_0712`, Slurm `177561`, job name
  `sugar_reffull`, running on `server23` through persistent `tmux+srun`.
- [ ] After inference is verified, run official CarryBox training/reproduction
  path from `SUGAR/train.sh CarryBox` on appropriate compute resources.
- [x] Add a continuation watcher for the remaining official CarryBox pipeline:
  `scripts/sugar/wait_and_run_official_sugar_carrybox_remaining_pipeline.sh`.
  It waits for the full refiner checkpoint and then launches
  `START_STAGE=refiner_rollout` through persistent `tmux+srun`.
- [x] Add a lightweight SUGAR status checker:
  `scripts/sugar/check_official_sugar_carrybox_status.sh`. It only inspects
  Slurm, checkpoint files, logs, and Curiosity-owned SUGAR tmux panes.
- [ ] Run the remaining official CarryBox training stages after refiner
  completion: refiner rollout, refiner rollout processing, tracker training,
  tracker rollout, tracker rollout processing, and generator training.

## 2026-07-12 Inference Smoke Evidence

- Environment: `/public/home/yanhongru/envs/sugar_py311_isaacsim510`.
- Fixed preflight PASS:
  `experiments/sugar_reproduction/logs/20260712_sugar_env_preflight_fixed.log`.
- Official IsaacSim extscache installed from verified NVIDIA wheels:
  `experiments/sugar_reproduction/logs/20260712_sugar_extscache_local_install.log`.
- GPU inference job: Slurm `177522`, `server35`.
- Successful smoke log:
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_local_assets.log`.
- Output video:
  `experiments/sugar_reproduction/outputs/released_inference/CarryBox/videos/play/rl-video-step-0.mp4`.
- Glue changes used only local replacement assets for cluster-offline Isaac
  visual dependencies: local ground plane USD and local frame marker. SUGAR
  tracker/generator checkpoints, task, data, robot, and object assets remained
  official.

## 2026-07-12 Refiner Training Smoke Evidence

- Launcher:
  `scripts/sugar/run_official_sugar_carrybox_refiner_train.sh`.
- GPU training smoke job: Slurm `177539`, `server35`.
- Command parameters: `NUM_ENVS=64`, `MAX_ITERATIONS=1`,
  task `Sugar-G129dof-CarryBox-Refiner`, motion folder `data/CarryBox`.
- Log:
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_train_smoke_iter1.log`.
- Evidence in log: environment setup completed, RSL-RL actor/critic built,
  `Learning iteration 0/1`, `Total timesteps: 1536`, final `status=0`.
- Output files:
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_sugar_carrybox_refiner_train_smoke_iter1/logs/refiner/model_0.pt`,
  TensorBoard event, `params/env.yaml`, and `params/agent.yaml`.
- This is only a short official training-path smoke. It is not the official
  4096-env, 30001-iteration refiner reproduction and not the complete
  refiner-rollout/tracker/generator pipeline.

## 2026-07-12 Full Official Pipeline Launch State

- Launcher:
  `scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh`.
- Stage defaults match the official SUGAR CarryBox training path:
  refiner/tracker `4096` envs and `30001` training iterations, refiner/tracker
  rollout with `1000` envs, and generator `1001` epochs.
- Formal refiner stage request:
  `START_STAGE=refiner_train`, `STOP_AFTER_STAGE=refiner_train`,
  `REFINER_NUM_ENVS=4096`, `REFINER_MAX_ITERATIONS=30001`.
- Persistent allocation state:
  `tmux` session `curiosity_sugar_refiner_full_0712`, Slurm job `177561`,
  job name `sugar_reffull`. The original pending interactive-shell request
  `177557` was cancelled and replaced so the official refiner stage starts
  automatically once Slurm grants the allocation.
- Current Slurm state: `RUNNING` on `server23`; the job started before the
  earlier `server45` estimate.
- Full-refiner log:
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_full_official.log`.
- Observed running evidence before the deliberate stop: Slurm job `177561`
  ran on `server23`; the output directory contains
  `logs/refiner/model_0.pt`, `logs/refiner/model_1000.pt`,
  `logs/refiner/model_2000.pt`, `logs/refiner/model_3000.pt`, and
  `logs/refiner/model_4000.pt`, and `logs/refiner/model_5000.pt`
  (`14,957,429` bytes each), TensorBoard events, `params/env.yaml`,
  `params/agent.yaml`, and a stored SUGAR git diff.
- No fatal traceback, missing-file, Boost Python argument, runtime, CUDA OOM,
  or `[Error]` pattern was found in the refiner log during this check.
- Completion still requires
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/model_30000.pt`
  and
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/ckpts/refiner.pt`;
  these do not exist because the run was intentionally stopped after
  `model_5000.pt` instead of continuing to `model_30000.pt`.
- Continuation watcher:
  `tmux` session `curiosity_sugar_remaining_after_refiner_0712`, log
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_remaining_pipeline_after_refiner_wait.log`.
  It checks for `ckpts/refiner.pt` every `300` seconds and then launches the
  official remaining pipeline with `START_STAGE=refiner_rollout`. This is a
  pending follow-on, not evidence that the remaining stages have run; as of
  `2026-07-12 14:29:05 CST`, it is still waiting and has not launched a
  remaining-pipeline Slurm job.
- Remaining-pipeline trigger-chain audit: the watcher waits for the exact
  `ckpts/refiner.pt` path; once present, it launches `srun` with
  `START_STAGE=refiner_rollout`. The pipeline stage index maps
  `refiner_train=0` and `refiner_rollout=1`, so the follow-on run starts from
  the official refiner rollout instead of re-running refiner training.
- Watcher maintenance: before launching the remaining pipeline, the watcher now
  logs the Python path, pipeline script, and Slurm request parameters, then
  performs login-safe existence checks for the Curiosity root, official SUGAR
  clone, executable Python, pipeline script, and still-present nonempty
  `ckpts/refiner.pt`. These checks do not import Python, launch Isaac, or run
  training on the login node.
- Watcher runtime state: the old watcher process was no longer present when
  checked at `2026-07-12 14:14 CST`, while `ckpts/refiner.pt` was still
  missing. A fresh Curiosity-owned watcher session
  `curiosity_sugar_remaining_after_refiner_0712` was created with the
  maintained script at `2026-07-12 14:14:18 CST`; the live status checker now
  shows the Python path, pipeline script, Slurm request, and wait target from
  that active watcher.
- Text audit: `scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh`
  preserves the official `SUGAR/train.sh CarryBox` stage order and
  command arguments while adding only cluster guards, logging, stage controls,
  and local visual-asset environment variables. `bash -n` passed for the
  pipeline wrapper, watcher, inference launcher, refiner smoke launcher,
  preflight, asset download, and environment preparation scripts.
- Pipeline wrapper maintenance: official SUGAR rollout completion raises
  `SystemExit(msg)` from the rollout command after printing
  `[Rollout] ====== All ... envs completed ...`. The original
  `SUGAR/train.sh` has no `set -e`, so this completion signal does
  not stop the official shell sequence. The cluster wrapper now inspects only
  the current rollout stage log and treats that explicit completion message as
  success only when no fatal pattern was detected in the same stage log; it
  still treats tracebacks, missing-file errors, Boost Python argument errors,
  `RuntimeError`, CUDA OOM, and `[Error]` log lines as fatal.
- Status-check script audit:
  `bash -n scripts/sugar/check_official_sugar_carrybox_status.sh` passed, and
  a live run reported Slurm/checkpoint/log/tmux status without launching
  Python, Isaac, training, rendering, dataset conversion, or model loading.
- Status-check script maintenance: refiner log/tmux tails now strip ANSI
  control codes before printing, keeping future status captures readable while
  still only doing text inspection.
- Status-check script maintenance: the checker now parses the latest refiner
  iteration from text logs and reports the next `model_*.pt` checkpoint and
  remaining iterations, without running Python or loading models.
- Status-check script maintenance: the checker now prints a
  `watcher_wait_log_config` block by grepping stable watcher configuration
  lines before the rolling wait-log tail. This keeps watcher launch parameters
  visible even after long waits, while still only doing text inspection.
- Status-check script maintenance: the checker now separates the tracked full
  refiner Slurm job from all current user `sugar*` Slurm jobs, so the eventual
  `sugar_remaining` follow-on allocation will be visible without a separate
  manual `squeue` command.
- Status-check script maintenance: Slurm rows are now de-duplicated, and the
  checker explicitly reports whether the Curiosity refiner and remaining
  watcher tmux sessions are present. `bash -n` passed after this change.
- Status-check script maintenance: Slurm rows now carry explicit
  `slurm_tracked_refiner_row=` and `slurm_user_sugar_job_row=` prefixes, so the
  same running refiner job is not confused for two separate jobs in filtered
  status summaries. `bash -n` passed after this change.
- Status-check script maintenance: the checker now prints
  `latest_refiner_periodic_checkpoint=model_*.pt` from existing refiner
  checkpoint files, reducing the need for a separate manual `ls` when tracking
  periodic refiner progress. `bash -n` passed after this change.
- Status-check script maintenance: the checker now parses and prints
  `latest_refiner_elapsed=` and `latest_refiner_eta=` from the refiner log, so
  one status command reports checkpoint, iteration, elapsed time, and ETA.
  `bash -n` passed after this change.
- Status-check script maintenance: the checker now estimates
  `estimated_time_to_next_checkpoint=` from elapsed time and current iteration,
  giving a rough time-to-next-periodic-checkpoint without launching any project
  code. `bash -n` passed after this change.
- Status-check script maintenance: `SUGAR_STATUS_COMPACT=1` now suppresses long
  refiner/tmux tails and shortens the watcher wait-log tail while preserving
  Slurm, checkpoint, progress, ETA, tmux presence, and watcher launch
  configuration lines. `bash -n` passed after this change.
- Status-check script maintenance: compact status now also prints
  `next_ckpt_watcher_tmux=` and `next_ckpt_wait_log_tail`, so the current
  periodic checkpoint watcher is visible from the same status entrypoint.
  `bash -n` passed after this change.
- Status-check script maintenance: the default next-checkpoint watcher and
  wait-log paths now follow the parsed `next_checkpoint` instead of being
  hard-coded to `model_3000.pt`. Environment variables can still override
  them. `bash -n` passed after this change.
- Periodic watcher maintenance: `scripts/sugar/wait_for_official_sugar_refiner_checkpoint.sh`
  now supports opt-in `AUTO_CHAIN_NEXT=1`, `CHAIN_STEP`, and
  `CHAIN_UNTIL_ITERATION`. After a watched checkpoint is found, it starts the
  next checkpoint watcher in a Curiosity-owned tmux session if one is not
  already present. This remains login-node safe: it only checks files, Slurm
  liveness, and tmux sessions. `bash -n` passed after this change.
- Downstream dependency audit: wrapper arguments for refiner rollout
  processing, tracker training, tracker rollout processing, and generator
  training match the official `SUGAR/train.sh` and script argument
  parsers. File-level metadata checks found the expected downstream packages
  for processing and generator training, including `zarr==2.12.0`,
  `numcodecs==0.12.1`, `hydra-core`, `omegaconf`, `diffusers==0.32.1`,
  `accelerate==1.2.1`, `timm==1.0.12`, `datasets==2.6.1`, `numba`, and
  `pydantic==2.11.4`.
- Preflight maintenance: `scripts/sugar/preflight_official_sugar_env.sh` now
  checks metadata for the downstream rollout-processing and generator
  dependencies above. It still refuses login nodes; only `bash -n` was run on
  the login node after this edit.
- [x] Add a login-node-safe reproduction artifact audit:
  `scripts/sugar/audit_official_sugar_reproduction.sh`. It checks files,
  directories, git metadata, Python dist-info metadata, log text, and Slurm
  state without importing Python or launching Isaac/training/rendering/data
  conversion/model loading.
- Latest audit log:
  `experiments/sugar_reproduction/logs/20260712_sugar_official_reproduction_audit_latest.log`.
  Latest result at `2026-07-12 17:01:45 CST`: `summary_present=29`,
  `summary_missing=10`, `summary_notes=1`, and
  `reproduction_status=incomplete`.
- Latest compact status snapshot:
  `experiments/sugar_reproduction/logs/20260712_sugar_official_status_compact_latest.log`. Latest
  snapshot at `2026-07-12 18:00:10 CST`: refiner progress `4130/30001`,
  latest checkpoint `model_4000.pt`, next checkpoint `model_5000.pt`,
  estimated time to next checkpoint `01:12:30`, and `ckpts/refiner.pt` still
  missing.
- [x] Add a login-node-safe periodic checkpoint watcher:
  `scripts/sugar/wait_for_official_sugar_refiner_checkpoint.sh`.
- `model_3000.pt` watcher result:
  tmux session `curiosity_sugar_model3000_watch_0712` exited after finding
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/model_3000.pt`
  at `2026-07-12 16:19:28 CST`.
- `model_4000.pt` watcher result:
  tmux session `curiosity_sugar_model4000_watch_0712` found
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/model_4000.pt`
  at `2026-07-12 17:48 CST` and automatically started the next watcher.
- `model_5000.pt` watcher runtime:
  tmux session `curiosity_sugar_model5000_watch_0712`, log
  `experiments/sugar_reproduction/logs/20260712_sugar_refiner_model_5000_watch.log`, target
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/model_5000.pt`.
  It started at `2026-07-12 17:49:35 CST` with `AUTO_CHAIN_NEXT=1` and only
  sleeps/checks files plus Slurm liveness; it does not run Python, Isaac,
  training, rendering, dataset conversion, or model loading.
- `model_5000.pt` landed at `2026-07-12 19:19 CST`. The periodic watcher,
  remaining-pipeline watcher, and refiner Slurm job were then stopped so the
  run would not continue beyond the requested 5000-step boundary.
- [x] Add a compute-node-only 5000-step refiner eval wrapper:
  `scripts/sugar/run_official_sugar_carrybox_refiner5000_eval.sh`.
- [x] Run 5000-step refiner video smoke on Slurm job `177780`, `server23`:
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner5000_eval.log`. This loaded
  `model_5000.pt` and ran 200 video steps, producing
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/videos/play/rl-video-step-0.mp4`,
  but it exited before rollout completion by design, so
  `trajectory_complete_count=0`.
- [x] Run 5000-step refiner no-video rollout eval on Slurm job `177782`,
  `server23`:
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner5000_rollout_eval.log`. Result:
  all `16/16` envs completed and `13` trajectory-complete `.npz` files were
  saved under
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/eval/refiner_model5000_rollout_eval_novideo/raw_npz/trajectory_complete`.
- [x] Add a login-node-safe 5000-step eval summarizer:
  `scripts/sugar/summarize_official_sugar_refiner5000_eval.sh`. It reports
  `16` expected sampled rollout windows, `13` saved complete trajectories,
  and an `81.25%` sampled refiner-window completion rate, then marks
  `comparable_to_paper=false`.
- [x] Do not claim paper-level SUGAR CarryBox reproduction from the 5000-step
  refiner result. Paper Table 1 reports SUGAR Carry Box train SR/Err
  `84.5/0.280` and test SR/Err `69.6/0.326`; those are final
  object-target success/error metrics for the full refiner/tracker/generator
  policy, not the sampled refiner rollout-window completion metric from
  `model_5000.pt`.
- The audit confirms the current missing official reproduction artifacts:
  refiner `model_30000.pt`, `ckpts/refiner.pt`, refiner rollout raw
  trajectories, refiner processed RL dataset, tracker `model_30000.pt`,
  `ckpts/tracker.pt`, tracker rollout raw trajectories, tracker processed IL
  zarr dataset, generator `epoch=1000.ckpt`, and `ckpts/generator.ckpt`.
- Audit maintenance at `2026-07-12 17:01 CST`: the audit now emits an
  explicit PASS for the latest observed full-refiner periodic checkpoint
  (`model_3000.pt` at this check), instead of only printing it as a status
  line. This improves evidence accounting but does not change the completion
  gate.

## 2026-07-13 Active Continuation

- [x] Add exact checkpoint-path resume support and validate the RSL-RL
  iteration arithmetic: resume `5000` plus `25001` requested iterations targets
  final checkpoint `30000`.
- [x] Preserve an immutable canonical copy of the original successful
  server23 `model_5000.pt`, and archive failed retry versions separately.
- [x] Add an output-directory `flock` so multiple retained/pending allocations
  cannot write the official checkpoint tree concurrently and backups can take
  over after the active pipeline exits.
- [x] Retain one-day allocations `178073` (`server36`) and `178091`
  (`server53`) instead of releasing backup resources.
- [x] Submit lower-CPU, one-day requests on both accessible partitions and a
  flexible any-healthy-node request: `178129`, `178133`, `178134`, `178136`,
  `178137`, and `178143`.
- [ ] Obtain a clean GPU and complete the official refiner continuation from
  `model_5000.pt` through `model_30000.pt`. Clean resources are now active:
  `178129` owns the pipeline lock and produced verified `model_6000.pt` on
  server23 and has now also produced verified `model_7000.pt` and
  `model_8000.pt`; `178136` and `178133` wait on the lock as retained
  failover.
  Continue polling until the final checkpoint exists.
- [ ] Export `ckpts/refiner.pt` and complete refiner rollout plus processed RL
  dataset.
- [ ] Complete official tracker training through `model_30000.pt`, export
  `ckpts/tracker.pt`, then complete tracker rollout and processed IL dataset.
- [ ] Complete official generator training through `epoch=1000.ckpt` and
  export `ckpts/generator.ckpt`.
- [x] Generate and visually inspect the current official refiner training
  curves on a compute allocation; de-duplicate failed short retry event files.
- [x] Generate and visually inspect the intermediate model-5000 rollout sample
  summary on a compute allocation, explicitly labeling it non-comparable to
  paper CarryBox SR/Err.
- [ ] Re-render full refiner/tracker/generator curves and final inference/eval
  video after all official artifacts exist.
- [ ] Run the final artifact audit; completion still requires all ten missing
  official reproduction artifacts.
- Remaining-pipeline handoff audit at `2026-07-12 17:03 CST`: the active
  watcher waits for the exact refiner checkpoint path
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/ckpts/refiner.pt`.
  After that file appears, it launches the wrapper with
  `START_STAGE=refiner_rollout`, whose refiner rollout checkpoint argument
  points to the same output tree and whose subsequent stages still match the
  official `SUGAR/train.sh CarryBox` order and key arguments.
- Pipeline wrapper maintenance at `2026-07-12 17:08 CST`: added post-stage
  artifact checks to
  `scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh`. The wrapper
  now refuses to continue if refiner/tracker/generator checkpoints, rollout
  complete-trajectory files, or processed RL/IL datasets are missing after the
  official stage that should create them. `bash -n scripts/sugar/*.sh` passed
  after this change.
- SUGAR shell entrypoints under `scripts/sugar/*.sh` are executable, so future
  tmux/srun launches can call them directly or through `bash`.
- This remains pending until the refiner stage writes
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/logs/refiner/model_30000.pt`
  and the launcher copies it to
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/ckpts/refiner.pt`.

## 2026-07-13 Operator-Selected 10000-Step Endpoint

- [x] Save and stabilize official `logs/refiner/model_10000.pt`.
- [x] Stop the official training child immediately after that checkpoint while
  leaving Slurm `178129` allocated in its persistent shell.
- [x] Verify no `model_11000.pt` exists and remove/stop the periodic auto-chain
  that could continue training.
- [x] Preserve read-only `ckpts/refiner_model10000.pt`; verify its SHA256 is
  `a398a7293fcea0ef948234e5de47b990fa586d2efd4e54ad7e481151c16124c3` and
  its internal checkpoint iteration is exactly `10000` on a compute node.
- [x] Run the official no-video refiner rollout diagnostic: `16/16` sampled
  environments completed and `16` complete trajectory files were saved.
- [x] Run the official video diagnostic and preserve
  `visualizations/refiner_model10000_rollout_video.mp4`.
- [x] Generate and visually inspect
  `visualizations/refiner_model10000_rollout_summary.png` plus its JSON
  sidecar, explicitly labeled refiner-only and non-comparable to paper SR/Err.
- [x] Regenerate and visually inspect `refiner_training_curves.png` through
  iteration 10000.
- [x] Retain running allocation `178129` and pending backup request `178137`;
  do not manually release either.
- [x] Refresh the artifact audit: `45` items present, the 10000 stop boundary
  passes, and the same `10` dormant full-reproduction artifacts remain absent.
- [ ] Dormant unless re-authorized: resume Refiner from 10000 to 30000.
- [x] Export the unmodified model-10000 checkpoint to the official downstream
  `ckpts/refiner.pt` path with an explicit provenance sidecar.
- [x] Complete official 1000-env Refiner rollout: `922/1000` complete
  trajectories, then generate the official processed RL dataset.
- [x] Generate and inspect the full-rollout distribution visualization and JSON
  (`92.2%`, derived discrepancy mean/median `0.05787/0.05153`).
- [ ] Complete official Tracker training to the operator-selected
  `model_10000.pt`, then generate final Tracker curves and rollout video.
- [ ] Complete Tracker rollout/processing and official Generator training,
  followed by final inference video and visual audit.
