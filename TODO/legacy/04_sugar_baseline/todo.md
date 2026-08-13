# TODO 04: Official SUGAR Baseline + High-Fidelity Tactile Extension

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

## 2026-07-14 SUGAR + High-Fidelity Tactile Mainline

### T0 — Freeze the Accepted SUGAR Control

- [x] Write a frozen baseline manifest containing the accepted SUGAR commit,
  IsaacLab v2.3.0 commit, local compatibility diffs, checkpoint hashes, task
  names, observation dimensions, evaluation seeds, and artifact paths.
- [ ] Preserve a matched no-tactile evaluation set before changing any task or
  observation configuration.
- [x] Add new tactile task/config names; do not modify or overwrite official
  baseline task registrations and outputs.

### T1 — Official IsaacLab Tactile Stack

- [x] Audit the current local SUGAR/IsaacLab path: IsaacLab v2.3.0 has only
  `net_forces_w`/`force_matrix_w` ContactSensor outputs here, and SUGAR derives
  hand-contact booleans using force thresholds; neither counts as tactile.
- [x] Identify the official implementation: IsaacLab v2.3.2
  `isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor`, tag commit
  `37ddf626871758333d6ed89cf64ad702aef127d0`.
- [x] Verify from official API documentation that it exposes per-taxel normal
  force, two-axis shear force, tactile RGB/depth, taxel pose, and penetration
  depth, rather than only contact labels or an aggregate wrench.
- [x] Fetch and audit the official v2.3.2 source and exact tactile assets/configs
  without replacing the current accepted v2.3.0 baseline provenance.
- [x] Produce a v2.3.0 -> v2.3.2 SUGAR compatibility audit covering task APIs,
  RSL-RL interfaces, render/camera interfaces, sensor changes, and the
  cluster-local integration glue in
  `DOCS/isaaclab_v232_tacsl_backport_audit.md`.
- [ ] Prepare a separate prebuilt tactile environment on an explicitly approved
  compute allocation; never install or resolve dependencies on the login node.
- [ ] Run compute-node SUGAR registration, reset, official-checkpoint inference,
  and short-rollout compatibility diagnostics on a wholly upgraded v2.3.2
  core. A direct full-core attempt exposed Manager API incompatibilities, so
  the audited minimal backport route below is the active path.
- [x] If the upgrade fails, backport only the official v2.3.2 tactile sensor,
  assets, and configs with commit-level provenance; record the blocker instead
  of creating a simplified sensor.

### T2 — Dual-Palm GelSight Integration

- [x] Select official GelSight Mini or R15 assets by measured palm fit and
  active sensing area.
- [x] Mount independent left/right elastomer sensors on the G1 palm contact
  surfaces with explicit transforms, camera intrinsics, and active taxel area.
- [x] Create/validate the CarryBox SDF contact representation without changing
  the frozen baseline visual geometry, mass, or inertia.
- [ ] Configure an initial `20 x 25` taxel grid per palm and 60 Hz update rate.
  The `20 x 25` grid is active; the current SUGAR 20 ms control step yields
  50 Hz, so the 60 Hz rate claim remains open.
- [ ] Expose and log left/right normal-force, pressure, two-axis shear,
  RGB/depth, taxel pose, timestamp, and validity-mask tensors. Force,
  pressure/shear policy observations, RGB/depth/deformation, taxel geometry,
  and timestamps now exist; an explicit validity mask is still pending.
- [x] Implement pressure conversion from normal taxel force using the represented
  surface area; do not label raw force as pressure.
- [x] Verify that no policy observation uses binary contact, thresholded force,
  hand-object distance, object pose, or SDF penetration as a tactile substitute.

### T3 — Tactile Fidelity Gates

- [ ] No-contact test: quantify pressure/shear noise floor and GelSight baseline
  stability. A geometrically separated nominal render is recorded for both
  cameras, but the stability/noise statistics are not yet complete.
- [ ] Normal-load sweep: verify monotonic integrated pressure, plausible patch
  growth, and declared agreement tolerance against an independent simulator
  wrench.
- [ ] Tangential-load sweep: verify shear direction, tangential-stiffness trend,
  and Coulomb saturation.
- [ ] Stick-slip ramp: verify a repeatable transition before gross sliding.
- [ ] Spatial sweep: verify pressure centroid and shear-field motion under known
  box translations and rotations.
- [ ] Mirrored-hand/reset/seed tests: verify left-right consistency and
  repeatability.
- [ ] Benchmark sensor update and rendering at 1, 16, 64, 256, and larger env
  counts; label these diagnostics, not training results. Force-field checks
  currently pass at 1, 8, 128, and 512 environments; the requested matched
  benchmark grid and renderer scaling remain open.
- [ ] Generate tactile validation visualizations: pressure heatmap, shear
  quiver/heatmap, RGB/depth frame, contact-patch centroid, and synchronized
  normal/tangential load curves.
- [ ] Acquire or record a blocker for physical GelSight calibration data.
- [ ] Before any real-tactile/sim-to-real claim, calibrate load response,
  footprint, shear/slip trend, latency/noise, and image statistics against the
  physical sensor.

### T4 — SUGAR Observation and Dataset Integration

- [ ] Register tactile CarryBox Refiner/Tracker configs derived from official
  SUGAR configs while keeping baseline configs unchanged. Refiner and camera
  validation tasks are registered; Tracker remains pending.
- [x] Add direct spatial pressure/shear observation terms for both hands.
- [ ] Add GelSight RGB/depth observation terms behind an explicit configuration
  flag. Separate official left/right camera validation tasks and recorders now
  pass, but images are intentionally not yet actor inputs.
- [x] Add a spatial tactile encoder and per-hand fusion with SUGAR's existing
  motion/proprioceptive representation; preserve raw maps in rollout logs.
- [x] Keep baseline reward and termination terms fixed for the first tactile
  training comparison.
- [ ] Extend Refiner rollout files with synchronized tactile tensors, masks,
  calibration metadata, and sensor-randomization parameters.
- [ ] Extend Tracker processing and datasets without reducing tactile maps to
  contact labels or single pre-encoder scalars.
- [ ] Decide on Generator tactile conditioning only after Refiner/Tracker
  observation-only results are valid.

### T5 — Tactile Training

- [x] Run official-code sensor and observation-shape smoke tests on a compute
  node.
- [x] Add checkpoint loading that reuses compatible accepted-SUGAR weights and
  initializes only new tactile encoder/fusion parameters, with explicit load
  reports.
- [x] Save a true pre-update RSL checkpoint and require deterministic
  zero-taxel warm-start audits for both actor and critic. The corrected loader
  reports exactly zero actor error, critic error, and zero-input tactile
  feature magnitude against accepted `model_10000.pt`.
- [x] Remove the unintended PhysX fallback mass on the two collision-free R15
  coordinate-frame tips. Each virtual `elastomer_tip` is now explicitly
  `1e-6 kg`; the corrected runtime robot mass is `33.351139 kg`, while the
  official elastomer collision, SDF, taxels, and camera transform are retained.
- [x] Freeze the accepted SUGAR actor during tactile finetuning. Only the
  non-bias spatial tactile encoder weights and tactile columns of the first
  actor layer may update; the original 890 input columns remain frozen, and
  the critic remains trainable.
- [x] Run the strict `model_pre_update.pt -> model_0.pt` causal checkpoint
  audit across full, zero, and pressure-only branches. All three pre-update
  model states are bitwise identical; all 455,680 original first-layer actor
  weights and every other frozen actor tensor remain exact; the zero branch's
  entire 844,794-parameter actor remains exact; and only the permitted tactile
  path changes in full and pressure-only.
- [x] Audit the three live training-process mount environments and serialized
  `env.yaml` files. All roles use the exact v3 offsets; full versus zero differs
  only at the observation function/zero mode, and zero versus pressure differs
  only at the policy-boundary mode.
- [x] Generate a machine-readable matched-training identity audit and require
  the final advantage analyzer to cross-bind its run paths/pre-update SHA with
  the final weight audit and evaluated checkpoint SHAs. The current identity
  report passes with no failures.
- [x] Restore the active tactile training/evaluation env module byte-for-byte
  to the training-time snapshot before final evaluation (SHA256 `c6dfdae3...`).
  Preserve the later synchronized dual-camera validation config in a separate
  module and repoint only its task registration, so visualization remains
  available without changing the core task provenance.
- [x] Repeat the strict causal checkpoint audit at `model_1000.pt`. The report
  passes all three roles: zero preserves its entire actor bitwise, full and
  pressure-only change only the permitted tactile path, and all original actor
  inputs/frozen tensors remain exact.
- [x] Repeat the strict causal checkpoint audit at `model_2000.pt`. It again
  passes all three roles, preserves the complete zero actor bitwise, and finds
  only permitted tactile-path changes in full and pressure-only.
- [ ] Repeat the strict causal checkpoint audit at the final checkpoint before
  admitting any final performance comparison.
- [x] Run a paired 64-environment nominal full-horizon `model_0.pt` diagnostic.
  All three roles retain 57/64 accepted-control successes; full touch affects
  actions in exactly the 59.375% of environments with real taxel exposure but
  does not yet change success. This proves no first-update collapse and no
  early advantage.
- [x] Verify the zero control end to end at iteration 1000: its nominal rollout
  NPZ is byte-identical to iteration 0 even though the checkpoint/critic SHA
  changed. The full iteration-1000 probe also retains 57/64 nominal success;
  neither result establishes an advantage.
- [x] Run paired three-role model1000 nominal, low-friction, contact-phase
  lateral-pulse, and combined-stress diagnostics. Full and zero have identical
  per-environment success sets in the first three (57/64, 56/64, and 57/64),
  with pressure-only worse (55/64, 53/64, and 55/64). Under combined stress,
  full/zero/pressure reach 30/64, 34/64, and 29/64. Full has real taxel exposure
  and same-state action dependence in every condition but no task-performance
  advantage at this checkpoint.
- [x] Run the paired three-role model2000 combined-stress diagnostic. Full,
  zero, and pressure-only reach 28/64, 34/64, and 34/64. Full loses six net
  successes, all within the 50 touched/action-dependent environments; the 14
  untouched environments are tied. This remains checkpoint diagnostic
  evidence and is excluded from the final gate.
- [x] Fix and validate the held-out COM-offset evaluator. PhysX rigid-object
  COM tensors are `[env, 7]`, not articulation-style `[env, body, 7]`; the
  evaluator now supports both official layouts and requires exact readback. A
  1-env combined smoke verified a requested +0.05 m local-Y shift.
- [ ] Train a matched from-scratch tactile control to separate warm-start gains.
- [x] Predeclare the next latent-contact-dynamics learning branch before the
  active reference-only result. It keeps rewards and all matched controls fixed
  while extending SUGAR's existing actor-hidden startup mass/friction draws
  with COM/impulse training and an explicitly audited PhysX/TacSL friction
  model; implementation remains gated on the current branch outcome.
- [x] Prepare the latent-contact-dynamics candidate only in isolated,
  unregistered files. Its per-environment adapter delegates the complete force
  calculation to the existing TacSL parent, while explicit average-combine
  materials and a stratified startup event align CarryBox/R15 PhysX dynamic
  friction with the TacSL Coulomb coefficient. Syntax/static checks pass and
  the active optimizer-clean 21-file manifest remains byte-exact.
- [ ] Run the prepared official Isaac App latent-dynamics preflight only if the
  sequential optimizer-clean 42/43/44 replication reaches its first fully
  admissible negative report; every earlier attempted seed must have a complete
  positive report. It must pass
  material-binding/combine-mode, exact mass/inertia/COM, object plus all four
  palm-interface static/dynamic/zero-restitution material, and
  TacSL-coefficient readback,
  partial-environment refresh, one reference-contact-phase pulse per episode,
  actor-hidden-tuple invariance, and bilateral spatial pressure/shear exposure
  before any latent task registration or formal update is allowed. The
  candidate event and preflight now contain independent fail-closed checks for
  all of these physical readbacks, including exact first-pulse linear velocity,
  unchanged angular velocity, audit-state agreement and second-call
  idempotence, plus a live per-environment taxel-shear
  check against `mu_dynamic * normal_force`. The USD coverage check also traverses R15
  instance proxies, resolves inherited physics-purpose bindings, and requires
  collision coverage under every declared object/palm root, rather than
  allowing valid records from other roots to hide a missing elastomer binding;
  the guarded runner now also locks the v3 left/right mount offsets and the
  official R15 USD SHA, while the preflight checks corrected robot/tip masses,
  29 actions, 500 taxels per hand, update rate, and force-field stiffness.
  All checks remain unexecuted until activation.
- [x] Arm a login-only negative-result watcher that consumes no GPU while
  waiting. It follows seeds 42/43/44 in order, exits without a request if all
  three pass, rejects any
  provenance/exposure/action-dependence/noninferiority gate failure, requires
  evaluation seed 42 plus a passing suite-manifest audit and exact current
  optimizer-clean executor SHA at both watcher and compute-runner boundaries,
  and only for a fully admissible negative result requests one finite four-hour
  `tmux+srun` allocation for the guarded latent preflight. The Isaac preflight
  also independently reloads the admitted report, verifies the complete
  negative gate and executor bindings, and for a later negative revalidates and
  records the complete earlier positive-report chain. It records all report and
  executor SHA values and includes that executor in its source manifest.
- [ ] Only after the guarded latent preflight writes a rehashed
  `overall_pass=true` report, add a process-local latent full/zero/pressure task
  registration and training entry. Both must require the exact preflight path
  and SHA, independently revalidate its admission/source bindings, reuse the
  current `ReferenceOnlyTactileActorCritic`, fixed-group optimizer-clean PPO,
  frozen official actor boundary, v3 mounts, and official R15 asset, and remain
  absent before the pass. The current watcher intentionally stops after
  preflight rather than auto-starting RL.
- [ ] After that registration exists, run a retained-compute-node full-role
  `1 env x 1 update` official-App smoke and audit bitwise-equal three-role
  pre-update checkpoints, normalized source/config manifests, fixed optimizer
  groups, and the complete actor-freeze boundary. A failed or stale diagnostic
  keeps formal latent training blocked.
- [ ] If every latent admission diagnostic passes, train fresh matched latent
  full/zero/pressure roles at 512 environments and 7098 uninterrupted updates,
  beginning with independent latent training seed42 and admitting latent
  seeds43/44 sequentially only after the preceding latent single-seed gate
  passes. The optimizer-clean seed that triggered the fallback is provenance,
  not a substitute for this independent 42/43/44 sequence.
- [ ] Add a latent-specific paired closed-loop evaluator and analyzers before
  the first latent formal update. Record the hidden mass/friction/COM/pulse
  tuple for audit without placing it in actor observations; fix evaluation seed
  42, require exact cross-role initial-state and tuple pairing, cover held-out
  latent dynamics plus the existing six-condition suite, and keep all latent
  statistics separate from current-distribution reports.
- [x] Train pressure+shear before enabling tactile RGB/depth at scale. The
  512-environment run completed 1000 iterations (`12,288,000` environment
  steps) from the official `model_10000.pt` warm start, followed by a clean
  100-iteration continuation to `model_1098.pt` with no dropped contact query.
  The old continuation failed at the previously unset reference-to-pool
  schedule boundary and a resumed diagnostic was deliberately stopped because
  the state pool/curriculum counter is process-local. Fresh uninterrupted
  full, zero, and pressure-only 7098-iteration matched runs were then stopped
  after 4727, 4787, and 1171 updates because the tip-mass and zero-input
  identity defects made them invalid for performance comparison. Corrected
  `tacsl_frozen_adapter_v3_full`, `zero`, and `pressure` branches are active
  from fresh accepted-checkpoint warm starts.
- [x] Verify at iteration 3000 that live taxel input reaches the learned actor:
  the first tactile Conv has relative L2 drift `0.13936` in the full branch and
  exactly zero drift in the zero-input branch. This is pathway-learning
  evidence from the superseded branch only, not a valid performance advantage.
- [ ] Preserve the SUGAR Refiner -> rollout/process -> Tracker ->
  rollout/process -> Generator stage order and artifact gates.
- [ ] Add calibrated randomization for stiffness, friction, tangential
  stiffness, offset, noise, latency, dead taxels, and camera appearance.
- [x] Use one-day persistent `tmux+srun/salloc` allocations, retain acquired
  backups, and switch CPU count/partition if requests do not start promptly.
- [x] Arm a no-race final-evaluation allocation fallback because the active
  jobs end around 13:30 and users cannot extend their wall clock. Jobs
  `180475/180476/180477` are deferred until 09:00 and consume no GPU while
  `BeginTime`-pending. A login-side switch activates them only if all three are
  RUNNING before any `model_7097.pt` exists; otherwise the five original
  sleeping watchers remain authoritative. The replacement workers wait on one
  sentinel and preserve the same audit/shard/analysis commands and outputs.
- [x] Extend those replacement workers into a conditional research loop. A
  valid exact-state `advantage_proven=true` stops the fallback; a valid false
  decision triggers one official-code diagnostic per reference-only role.
  Only all three diagnostic sentinels activate simultaneous 512-env/7098-update
  full/zero/pressure training, followed automatically by reference identity,
  strict weight audit, the 54-run final suite, and success-primary analysis.
  No diagnostic/formal output or sentinel existed when this was armed.

### T6 — Matched Ablations and Evaluation

- [x] Register compute-matched full, zero, and pressure-only Refiner branches;
  all retain the official dual-R15 TacSL scene and differ only at the policy
  observation boundary.
- [x] Add an online closed-loop evaluator with same-policy tactile
  interventions, per-environment outputs, official termination reasons, and
  held-out mass/friction/COM/lateral-pulse conditions.
- [x] Extend the evaluator with per-environment/per-hand all-taxel minimum SDF
  and closest-step records, then calibrate the palm mounts on natural policy
  trajectories. Lock the safe `v3` hand-frame offsets for matched training;
  record that natural pretraining exposure is `59.375%` overall but only
  `12.5%` on the right hand, so bilateral-balanced coverage is not yet proven.
- [x] Harden the final paired protocol against stale geometry and mismatched
  runs: require the locked `v3` offsets, corrected `33.351139 kg` runtime robot
  and `1e-6 kg` virtual tips, final checkpoint iteration, strict final weight
  audit, identical per-environment motion IDs and full robot/object initial
  states, and exact checkpoint SHA-to-policy-role binding.
- [x] Add an isolated `0.05 m` lateral COM-offset condition and include COM in
  the combined held-out condition; the final gate now requires six exact
  nominal/held-out conditions rather than silently omitting COM.
- [x] Require all seven same-policy tactile interventions for the full policy,
  while evaluating the matched zero and pressure policies only in their
  trained observation modes.
- [x] Add fail-closed final-role, sharded-suite, strict-weight-audit, and final
  paired-analysis entry points so the three retained GPU allocations can run
  disjoint evaluation shards without manifest or result-name collisions.
- [x] Bind each final-role shard to its own training-time source manifest. The
  entry point now checks all eight sensor/training/object/robot/PPO/observation/
  task/actor source digests before loading simulation and fails closed on a
  missing, duplicate, or mismatched entry. `SOURCE_GUARD_ONLY=1` passed for
  full, zero, and pressure on server13; final suite manifests record the source
  manifest path/SHA, passed guard, and final-role/suite-runner SHAs.
- [x] Cross-bind those five completed shard manifests in the final advantage
  analyzer. Admission now requires successful shard status, exact disjoint mode
  coverage, all six conditions, matching checkpoint/result identity, and
  current evaluator/TacSL/final-role/suite-runner/analyzer plus role-specific
  training-manifest SHAs; a manifest can no longer be recorded and then ignored.
- [x] Run a `1 env x 2 step` compute-node schema smoke with an existing
  iteration-0 checkpoint. It generated both JSON and NPZ and serialized the
  checkpoint gate, v3 mounts, corrected body masses, source hashes, motion ID,
  and complete paired initial-state arrays. This is interface evidence only.
- [x] Isolate the mutable Unitree asset links and IsaacLab USD-converter cache
  under a unique per-run temporary root. A same-second full/zero concurrent
  `1 env x 2 step` smoke completed both manifests with status zero; this closes
  the evaluator startup race but carries no performance meaning.
- [x] Separate policy `training_seed` from evaluation-environment seed and add
  a predeclared multi-training-seed audit. It requires at least three distinct
  matched policy seeds, every single-seed causal/provenance gate, and positive
  seed-bootstrap held-out plus same-policy-intervention success gains; it does
  not count 256 evaluation environments as independent policy trainings.
- [ ] Train and evaluate at least two additional matched full/zero/pressure
  policy seeds after the seed-42 result passes the single-seed gate.
- [x] Prepare the exact-state positive branch rather than stopping at one seed:
  seed42/43/44 use the same parameterized final role, each follow-up is a fresh
  matched 512-env/7098-update full/zero/pressure trio with a 14-hour admission
  gate and the unchanged 54-run suite, and seed44 invokes the strict exact-state
  multi-seed audit. Login watchers request seed43/44 only after the immediately
  preceding exact-state seed strictly passes.
- [x] Bind the existing multi-training-seed audit to one explicit protocol
  profile so exact-state and reference-only reports cannot be mixed. The
  reference-only runner fixes seed 42/43/44 inputs; every report must retain
  passing suite-manifest, weight, training-identity, and single-seed advantage
  gates before seed-level bootstrap evidence can pass. The final aggregator
  also requires report contents in exact 42/43/44 order and rehashes the
  single-seed analyzer, weight/identity audits, and every shard manifest.
- [x] Separate policy-training seed from evaluation seed in both exact-state
  and reference-only final roles. Training uses 42/43/44, every final suite
  fixes evaluation seed 42, and the multi-seed analyzer rejects mixed
  evaluation seeds.
- [x] Bind reference-only training seed to serialized training artifacts. Its
  identity audit now requires both environment and agent YAML seeds to equal
  the requested seed, and final evaluation rejects a mismatched identity
  report instead of trusting the directory label.
- [x] Make the common single-seed advantage analyzer independently require the
  identity report's expected seed and all three serialized environment/agent
  seed maps to equal the rollout training seed. This applies to exact-state and
  reference-only reports before any performance gate is evaluated.
- [x] Bind reference-only runtime geometry and allocation identity. Every
  formal run now records its compute host/Slurm job, task/role, 512x7098
  budget, training seed, both locked v3 offsets, source manifest, negative
  exact-state audit, timestamps, and terminal status; the identity audit
  requires three distinct jobs and rehashes the recorded files.
- [x] Close the late-replacement recovery gap for reference seed42. If a final
  checkpoint appears before the scheduled workers activate, old watchers may
  finish exact evaluation; a login watcher now requests three fresh one-day
  jobs only after that exact report is negative and primary activation is
  absent, then enters the existing reference diagnostics/formal pipeline via a
  bounded matched barrier without repeating exact evaluation.
- [x] Isolate mutable Unitree USD and IsaacLab converter caches for concurrent
  formal training, not only evaluation. Reference runtime metadata and identity
  checks bind unique `/tmp/IsaacLab/reference_train_<job>_<run>` roots per role;
  exact follow-up workers use the analogous per-job/per-run layout.
- [x] Reuse retained replacement allocations for seed43 only if reference-only
  seed42 passes and every role independently has at least fourteen wall-clock
  hours remaining. Current measured 7098-update training alone projects to
  about 11 hours, so this preserves room for audits and the 54-run suite. A
  three-role ready barrier publishes one activation sentinel, so
  a borderline allocation cannot start only two of the three matched runs;
  formal training, audits, and the full seed43 suite then run automatically.
- [x] Arm a login-node-only seed43 fallback watcher. It gives retained workers
  ten minutes to activate, waits out and verifies cleanup of any partial
  fourteen-hour-boundary barrier, then requests three fresh one-day tmux+srun
  workers only when seed42 strictly passed and no matched seed43 is active.
  Unactivated readiness now times out after 20 minutes, below the cluster's
  three-hour low-GPU-utilization eviction window. The watcher was reloaded at
  2026-07-15 14:16 CST and excludes server21/server35/server43 after observed
  Vulkan device-loss failures there.
- [x] Arm a login-node-only seed44 allocation watcher that remains inert until
  the strict reference-only seed43 report passes. It then requests three
  tmux-retained one-day GPU workers, requires the follow-up prerequisite seed
  to equal the current seed minus one, runs the same matched formal protocol,
  and lets seed44 role A invoke the fixed seed42/43/44 audit automatically. It
  was reloaded with the same device-loss node exclusion set as seed43.
- [x] Preserve the predeclared generic seed43/44 worker byte-for-byte while
  applying the seed42 completion-metadata and peer-failure barrier through a
  fail-closed executor. It verifies the locked worker and analysis-path SHA
  values before generating an ephemeral runtime copy; follow-up identity audit
  cannot race a partially written final checkpoint, and one failed
  training/audit/shard releases the other roles instead of leaving idle GPU
  allocations waiting indefinitely.
- [x] Audit actor-side privileged object information. The official Refiner
  actor sees exact object pose/orientation/linear/angular velocity and future
  reference terms computed from exact current object state, so the active
  branch is an exact-state simulation upper bound rather than an occlusion
  experiment.
- [x] Predeclare the follow-up reference-only actor protocol without modifying
  the running exact-state branch. Actual object pose/orientation/velocities are
  replaced by reference-plan values for the actor, future object commands are
  recomputed in the current reference frame to close the exact-state leak, and
  the critic stays privileged. Noise/latency remain later fixed ablations.
- [x] Prepare and compute-audit its isolated actor-critic adapter without
  registering the task. The official 890-wide actor/critic warm start is exact
  at zero tactile, the actor alone gains 256 spatial tactile features, critic
  output is bitwise tactile-invariant, and all 16 architecture checks pass.
- [x] Audit the six reference-only coordinate transforms with IsaacLab math.
  They exactly match official observations at a tracked frame, remain bitwise
  invariant to actual-box pose/velocity perturbations, preserve every width,
  and leave all six official privileged terms measurably responsive.
- [x] Run a direct-config `1 env` Manager/PhysX observation audit without
  registering the branch. At fixed reference frame 425, object pose/velocity
  writes pass exact buffer/PhysX readback, the complete 890-wide actor policy
  group remains bitwise unchanged, and the 890-wide privileged critic changes
  by `0.60784`; all 16 checks pass.
- [x] Prepare an isolated fail-closed reference-only matched-training wrapper.
  It fixes the official checkpoint, v3 mounts, dual R15 assets, 512 environments,
  7098 uninterrupted updates, role-specific task IDs, and an 18-file source
  manifest. It is intentionally not runnable until the exact-state final suite
  is frozen and the separately audited task/class registration is activated;
  reduced environment/update overrides are rejected in formal mode and allowed
  only behind an explicit diagnostic mode whose label and artifact checks
  cannot be confused with matched training.
- [x] Implement the reference-only task/class activation as process-local glue
  rather than modifying official `train.py` or the exact task registry. The
  three registrations are rejected without an explicit enable variable and
  passed a compute-node no-enable/enable registry audit. The formal wrapper
  additionally refuses to run until the exact-state final audit exists and
  records `advantage_proven=false`.
- [x] Prepare a reference-only-specific matched-training identity audit. It
  requires all six actor object terms to use the reference plan, all six critic
  terms to retain exact state, policy/critic groups to include/exclude tactile
  as declared, exact zero-tactile warm start and frozen adapter parameters,
  identical normalized role configs, a passing live observation-leak audit,
  the negative exact-state gate, and all 18 training-source digests.
- [x] Bind all three reference-only preflights into that formal identity audit,
  not only the live runtime report. Architecture, transform, and Manager/PhysX
  reports must each pass every check; their recorded observation/config/actor
  sources and accepted `model_10000.pt` are rehashed against current bytes.
- [x] Prepare isolated reference-only final-role and closed-loop suite runners.
  They remain non-runnable before real training, require the negative exact
  gate plus passing reference training identity and 18-file source guard, fix
  the reference-only full task at 256 environments/1501 steps, retain the same
  six conditions and seven full-policy interventions, and serialize the
  process-local evaluator/registration/reference-source identities.
- [x] Reuse one statistical advantage implementation through explicit
  `exact_state` (default) and `reference_only` profiles. They share paired
  bootstrap/metrics plus nominal, exposure, action-dependence, and provenance
  machinery; profile-specific provenance is explicit, and the reference-only
  admission rule may only tighten the frozen exact-state rule. An isolated
  reference-only analysis runner selects the second profile.
- [x] Route seed42 and every follow-up reference-only single-seed report through
  the fail-closed bootstrap-axis executor. It rehashes the immutable analyzer
  and repairs only its `[condition, bootstrap, env]` reduction axis; later
  seeds can no longer call the known-broken locked entry point directly.
- [x] Harden the optimizer-clean seed43/44 admission and final multi-seed
  provenance before seed42 completes. Both login watchers now wait for the
  executor-complete field rather than racing the base JSON write; the positive
  watcher and compute worker require every causal gate, locked analyzer SHA,
  current executor SHA, and all four no-change flags. The final executor keeps
  the frozen seed-bootstrap rules, while additionally requiring one identical
  official `model_10000` source, identical source/prerequisite identities,
  distinct pre-update checkpoints across seeds, matched pre-update bytes
  within each seed, and nine distinct role allocations.
- [x] Enforce the stronger predeclared reference-only success rule in that
  shared implementation. Unlike the frozen exact-state secondary path,
  reference-only cannot pass on failure reduction or progress: held-out
  full-vs-zero success and same-full-policy live-vs-zero success must each have
  a positive paired 95% CI lower bound. Secondary metrics remain diagnostic.
- [x] Add fixed-path reference-only training-identity and strict checkpoint
  audit runners. They consume the real three-role run labels and produce the
  exact reports required by the final-role and shared statistical analyzer;
  they remain fail-closed while those artifacts do not exist.
- [x] Harden the live seed-42 post-training barrier against checkpoint/metadata
  races. Role A now requires all three final checkpoints plus completed runtime
  metadata with `finished` and final `status=0` before identity audit; any
  training, audit, shard, or analysis failure is published to the peer workers
  so they fail closed instead of idling their GPU allocations indefinitely.
- [x] Audit the fresh-restart seed42 `model_0.pt` checkpoints after observing
  that full/pressure were not bitwise identical to the earlier time-limit
  partial while zero was. The strict audit passes: all pre-update states match,
  zero preserves its entire actor, and full/pressure change only the permitted
  seven tactile encoder weights plus tactile input columns; every base column
  and other frozen actor tensor remains exact. This is restart/contract
  evidence only, not performance evidence.
- [x] Audit the running reference-only optimizer dynamics before any final
  result. Although all three `agent.yaml` files are byte-identical, actor KL
  controls one Adam shared by actor and critic: zero is at `1e-2` for all 637
  audited steps, while full/pressure rolling-100 rates are about `1.25e-4` and
  `1.19e-4`; zero value loss spikes to `43226.6`. Freeze the step-636 report and
  predeclare `DOCS/sugar_tactile_optimizer_deconfounding_protocol.md`.
- [x] Run the fresh uninterrupted common-`model_1000.pt` strict weight audit.
  All actor-freeze checks pass, while zero critic relative L2 drift reaches
  `11.967` and maximum absolute change `23.006`, versus full/pressure relative
  `1.152/1.318` and maxima near `1.39`; retain this as optimizer diagnosis, not
  performance evidence.
- [x] Close the shared-optimizer reference-only branch without making a tactile
  claim. The completed exact-state seed-42 branch remains a separate negative
  54-run result. The matched reference-only full/zero/pressure diagnostics
  and bitwise-identical pre-update/source-manifest checks pass. The first formal
  attempt activated at 2026-07-15 11:46 CST but inherited allocations near
  their 24-hour limits; full/zero/pressure stopped at iterations
  1157/1289/1309 and are archived as `time_limit_partial`, not resumable formal
  results. Fresh 512-environment, 7098-update training restarted from the
  bitwise-identical warm start at 13:44 CST in jobs 181009/181011; zero job
  181010 lost its server43 Vulkan device before iteration 0, was isolated, and
  restarted fresh as job 181025. After its common model-1000 audit exposed the
  same confound and the user authorized reuse of the GPUs, all three were
  stopped at 16:08 CST; their metadata records `status=22` because model7097 is
  absent. Retain those checkpoints as optimizer diagnostics only, do not run
  their frozen performance suite, and do not activate same-contract seeds
  43/44.
- [ ] Complete the predeclared optimizer-clean continuation. Its isolated
  implementation and official-checkpoint structure audit pass: fixed `1e-3`
  privileged-critic and `1e-4` tactile-actor Adam groups are disjoint and
  exhaustive, the inherited official PPO update emits complete finite
  LR/KL/gradient metrics, and no frozen tensor changes. Guarded three-task Gym
  entry/source auditing also passes. The fixed full-role official Isaac App
  `1 env x 1 update` smoke and fail-closed runtime audit now pass all checks,
  including the two serialized optimizer groups/rates, permitted tactile-path
  change, exact base actor, finite LR/KL/gradient metrics, and the current
  21-file source manifest. Its report SHA256 is
  `2a9e3af4231e5710cfdaf88357c4f10972af234fa9e2ab4707f0a69c82426956`.
  Matched formal training, strict identity/weight audits, and the frozen
  six-condition/seven-intervention evaluation remain. Matched seed42 formal
  jobs `181297/181298/181299` started at 16:10 CST with identical 21-file
  manifests, identical agent configs, and byte-identical pre-update checkpoint
  SHA256 `a189ae75238f51c6f15addab07d765f4698b0ae41f8d62631e10ba17b10cdf80`.
  The immediate model-0 strict audit passes: the zero actor remains bitwise
  exact, full/pressure change exactly the seven encoder weights plus tactile
  input columns, and every base/frozen actor tensor remains exact (report
  SHA256 `af2c52b5020d2e155049a4dc3a37a75e76ecd36ddc8d54c4e482d6d29a9d553d`).
  The clean model-1000 fail-fast audit repeats the same pass: zero's full actor
  is bitwise exact, full/pressure change only the permitted tactile path, and
  all base columns/frozen actor tensors remain exact (report SHA256
  `5790a5c0faa68ee046796a342f4450746b95873692c4a430047b847f6c81fc5b`).
  A separate intermediate dynamics audit covers exact steps `0..1000` for all
  seven optimizer metrics: both learning rates are fixed on all 1001 records,
  zero tactile gradients are identically zero, full/pressure gradients are
  positive throughout, and all cross-role identities/error checks pass
  (report SHA256
  `26a6a09b6586414b5bd98ceadf06d9f2e8d0c066093429916b4e732dfb3f46c4`).
  Before final identity admission, require the newly bound formal-dynamics
  audit to verify all 7098 TensorBoard steps per role, exact fixed group rates,
  finite loss/KL/gradient series, final serialized optimizer groups, matched
  source/agent/pre-update identities, and zero error signatures. Every final
  suite manifest must bind its report and audit-source digest.
  Before any final rollout, the suite runner now also fail-closes on the exact
  core evaluator, optimizer-clean evaluator entrypoint, optimizer-clean
  executor, locked common analyzer, and TacSL sensor SHA values; this is a
  provenance-only guard and does not change modes, conditions, horizons, or
  statistics. Its guarded runner SHA256 is
  `c58c65c9249177d5c8ef62e46adf47993648c9fbe806a356f7679ddc11e2b408`.
  The prepared three-role
  worker and optimizer-clean analyzer executor retain the original success-
  primary gates. Do no task-success-based LR search. Only a passing clean
  seed42 may activate clean seeds 43/44.
- [x] Reject forced contact-midpoint reset as a performance protocol after its
  settling diagnostic crossed official failure thresholds; use natural
  trajectory-start contact for the final comparison.
- [ ] Evaluate frozen SUGAR and a non-tactile contact-label/body-force proxy
  control.
- [x] Predeclare the frozen official SUGAR actor control without retraining.
  Bind the exact-state zero checkpoint and its passing identity/weight audits,
  reject the older evaluator files as insufficient for a new cross-profile
  pairing claim, and conditionally rerun all six zero-taxel conditions with the
  current evaluator only after optimizer-clean seed42 passes. The paired
  analyzer requires ten bitwise-equal initial arrays, held-out success-gain CI
  lower bound above zero, and nominal success/failure noninferiority. This is a
  single-seed gate and does not replace seeds43/44 or proxy controls.
- [x] Predeclare the two proxy controls before any final tactile result and
  implement them in isolated, unregistered source files. Both consume only the
  official three-step hand `ContactSensor.force_matrix_w_history`, emit an
  explicit 18-D non-tactile layout, and use a shared per-hand
  `9 -> 477 -> 55 -> 320 -> 128` zero-preserving MLP with exactly 89,536 total
  and 89,088 trainable parameters, matching both the total and trainable
  spatial TacSL encoder capacity. The aggregate mode is explicitly limited to
  the filtered normal-contact force vector; it is neither total frictional
  force nor shear. Its reference-only actor base directly inherits the active
  tactile `ReferenceOnlyPolicyCfg`, avoiding a duplicated observation
  definition. The proxy configs now inherit the exact SDF
  CarryBox plus corrected dual-R15 G1 tactile scene, while their observation
  groups never expose TacSL, preventing robot/object physics from confounding
  the information-only control. The reference-only proxy no longer inherits
  the shared adaptive-rate base PPO: its isolated adapter calls the unchanged
  official PPO update and matches the active tactile optimizer-clean contract
  with fixed critic/proxy-actor rates `1e-3/1e-4`. The compute-node
  official-checkpoint audit now binds both environment and optimizer sources,
  proves named groups exhaust trainable tensors, and rejects local PPO-loss
  reimplementation; it is prepared but intentionally not run. A future
  official-App audit must verify scene identity and the 18-D ContactSensor-only
  boundary before any proxy task is activated. That audit is now predeclared
  for both proxy modes and fail-closes on the SDF/R15 scene identity, exact
  reference-only policy terms, absence of a TacSL observation group, direct
  18-D ContactSensor recomputation, and fixed optimizer config; it remains
  unexecuted and is explicitly a wiring diagnostic only.
- [ ] Evaluate pressure-only, pressure+shear, GelSight-only, and full multimodal
  tactile policies under matched seeds and training budgets.
- [x] Lock a post-gate modality analyzer before seed42 completes. It reloads the
  same paired suite and separately gates full-live versus trained pressure-only,
  full-live versus same-policy pressure masking, pressure-only versus zero, and
  full-live versus shear-only masking. Incremental shear requires both held-out
  success CI lower bounds above zero, positive per-condition directions,
  nominal noninferiority, shear exposure, and live-versus-pressure action
  dependence; a complete pressure control alone is not a shear claim. A
  login-only zero-GPU watcher is prepared to request a short CPU compute
  allocation only after a positive primary report.
- [ ] Evaluate tactile dropout/noise/latency and masked-at-test variants.
- [x] Predeclare the post-gate robustness suite before optimizer-clean seed42
  completes. It freezes the current evaluator instead of modifying it, fixes
  20/40/100 ms delay, 10/25% frame and dead-taxel loss, left/right complete
  failure, and 1/5 kPa actor-map noise modes, uses exact paired initial arrays
  and deterministic serialized corruption schedules, and labels every level as
  an algorithmic stress test rather than physical GelSight noise. The suite
  remains inactive until the current seed42 causal gate passes.
- [ ] Evaluate held-out object mass, friction, geometry, center-of-mass offset,
  lateral disturbance, and partial sensor failure.
- [x] Predeclare the first held-out-geometry gate without using the task-
  mismatched official KickBox asset as CarryBox evidence. After a positive
  current seed42 gate only, process-local configs may apply fixed local-Y
  0.90/1.10 scales to the pinned official small-box USD through the existing SDF
  spawner. A compute smoke must first prove exact geometry/physics/SDF readback,
  bilateral TacSL exposure, zero actor geometry leakage, and stable natural
  trajectory starts; performance then uses paired full-live, same-policy zero,
  matched zero, and pressure controls with no retraining or mount tuning.
- [ ] Report official SUGAR success/error plus fall/drop rate, slip distance,
  pressure imbalance/peaks, shear ratio, contact-patch stability, object
  oscillation, torque/energy, throughput, GPU memory, and inference latency.
- [ ] Generate synchronized rollout videos containing world view, left/right
  pressure and shear maps, GelSight frames, slip metrics, and task status.
- [ ] Claim a tactile benefit only if it survives matched-budget, proxy-control,
  and sensor-masking ablations.

### T7 — Higher-Fidelity Validation and Claim Boundary

- [ ] Run a small TacEx soft-body GelSight comparison if its official code and
  Isaac compatibility can be reproduced without destabilizing the SUGAR stack.
- [ ] Compare TacSL and TacEx pressure footprint/deformation/image trends on the
  same controlled contacts; do not use TacEx as the first large-scale backend.
- [ ] Keep results labeled `high-fidelity simulated tactile` until physical
  calibration passes.
- [ ] Record any unavailable official assets, incompatible runtime, licensing,
  or physical-hardware dependency as a blocker; never substitute a contact
  label, force threshold, or toy taxel model.

### T8 — Official Tactile Genesis Candidate Backend

- [x] Admit Tactile Genesis as an independent candidate backend and link the
  detailed P0-C ladder in `PLAN/05_direct_tactile_rgb_policy/plan.md` and
  `TODO/05_direct_tactile_rgb_policy/todo.md` from the active baseline plan.
- [x] Freeze and audit the official paper snapshot, maintained Genesis World
  source, licenses, per-taxel force/torque contract, elastomer-marker
  displacement contract, and source hashes in
  `DOCS/tactile_genesis_candidate_backend_audit_20260717.md`.
- [x] Record the current runtime blocker exactly: no compatible prebuilt
  Genesis/Quadrants environment exists; the SUGAR environment is Python
  3.11/Torch 2.7, while the paper task requires Python 3.12 and embedded Eden
  requires Torch >=2.9.2. Do not modify the accepted SUGAR environment or
  install dependencies inside its compute allocation.
- [ ] After a separate compatible environment is provisioned, run the
  untouched official kinematic and elastomer tactile sandbox on a retained
  compute allocation and preserve native tensor outputs and provenance.
- [ ] Add only a minimal audited G1 palm/probe adapter in Genesis and reproduce
  the zero/onset, normal-load, signed two-axis shear, speed/load, and footprint
  translation gates for both palms.
- [ ] Compare TacSL and Tactile Genesis only on declared shared observables,
  units, and coordinate transforms; keep marker displacement distinct from
  photometric GelSight RGB/depth.
- [ ] Decide from measured batching, reset determinism, latency, and
  action/state equivalence whether to use cross-backend calibration/domain
  randomization, pursue a Genesis-native SUGAR port, or stop at independent
  validation. Do not start unvalidated Isaac/Genesis policy fusion.
- [x] Keep every Genesis claim bounded to `high-fidelity simulated tactile
  under Genesis physics`, never physical GelSight validation, IsaacLab
  numerical equivalence, or sim-to-real.
