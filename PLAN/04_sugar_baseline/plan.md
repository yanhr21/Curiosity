# Plan 04: Official SUGAR Baseline

## Current Operator Boundary — 2026-07-13

The active refiner endpoint is now `model_10000.pt`. Per the operator's latest
instruction, do not resume refiner training beyond iteration 10000. The
operator clarified that functional behavior and visual evidence are the
acceptance criterion; exact paper numeric equality is not required. Continue
the official downstream rollout, processing, Tracker, and Generator stages
from the unmodified model-10000 checkpoint, while labeling this as an
operator-selected truncated schedule. Keep acquired and pending Curiosity
SUGAR resources rather than releasing them.

## Canonical Workspace Layout — 2026-07-13

- Active source clone: `SUGAR/` at the workspace root.
- Active IsaacLab source: `IsaacLab/` at the workspace root.
- Reproduction outputs: `experiments/sugar_reproduction/outputs/`.
- Reproduction logs: `experiments/sugar_reproduction/logs/`.
- The entire experiments tree is local-only and intentionally ignored by
  `.gitignore`; no file below it may be committed or pushed.
- `external/SUGAR`, the legacy IsaacLab link under `external/`,
  `SUGAR/outputs`, and `logs/sugar` are transitional compatibility symlinks for
  the already-running pipeline, editable environments, and historical
  commands; all new work uses the canonical paths above.

## Priority

As of 2026-07-11, SUGAR is the highest-priority mainline baseline for this
repository. The immediate goal is to reproduce official SUGAR CarryBox before
adding Curiosity-specific changes.

SUGAR is the closest public baseline because it combines:

- human-video-driven humanoid loco-manipulation;
- IsaacLab manager-based environments;
- Unitree G1-style humanoid assets;
- CarryBox-like object interaction tasks;
- released processed data and demo checkpoints.

## Official Sources

- Code: `https://github.com/tianshuwu/SUGAR`
- Local clone: `SUGAR`
- Current local SUGAR commit: `01fe123` (`fix inference bug`)
- IsaacLab dependency: official `isaac-sim/IsaacLab` tag `v2.3.0`
- Local IsaacLab clone: `IsaacLab`
- Paper: `arXiv:2605.20373`

## Fidelity Rules

- Use official SUGAR code, task names, data, descriptions, and checkpoints.
- Do not substitute Curiosity's previous G1/AGILE scaffold for SUGAR.
- Do not hand-roll a smaller controller, toy policy, toy refiner, toy tracker,
  or toy generator and call it SUGAR progress.
- Any reduced run is only a cluster smoke test if it uses official SUGAR code
  and official released assets/checkpoints.

## Cluster Execution Rules

- Login node work is limited to source clone, text inspection, documentation,
  and allocation setup.
- Data download/unzip, package/environment setup checks, IsaacSim launch,
  SUGAR inference, SUGAR training, and visualization must run on compute nodes.
- GPU work must be held through a Curiosity-owned `tmux` session with persistent
  `srun` or `salloc`. Do not use `sbatch`, `sspath`, or unrelated sessions.
- Compute nodes should activate prebuilt shared-filesystem environments where
  possible. Do not silently create an incompatible environment or downgrade
  SUGAR.

## Reproduction Path

1. Confirm official SUGAR and IsaacLab clones:
   - `SUGAR`
   - `IsaacLab` at `v2.3.0`
2. Download official SUGAR assets on a compute node:
   - `data.zip` from Google Drive ID `1AIJWqS5rFGl5u2Qq6jCCTHKdh51SX2Sc`
   - `descriptions.zip` from ID `1wXNAjNMrfV0e-d2pQ6m9dm4xrG5lSoyD`
   - `demo_ckpts.zip` from ID `1Uc2SPPVvTboEgw4Scyuz3TmzNKDg-dx-`
3. Configure/activate an environment matching official requirements:
   - Python 3.11
   - `isaacsim[all,extscache]==5.1.0`
   - IsaacLab `v2.3.0`
   - editable installs for `source/sugar_rl` and `source/sugar_il`
   - guarded preparation recipe:
     `scripts/sugar/prepare_official_sugar_env.sh`
4. Reproduce official CarryBox inference:
   - preflight: `scripts/sugar/preflight_official_sugar_env.sh`
   - task: `Sugar-G129dof-CarryBox-Inference`
   - motion folder: `SUGAR/data/CarryBox`
   - tracker checkpoint: `SUGAR/demo_ckpts/CarryBox/tracker.pt`
   - generator checkpoint: `SUGAR/demo_ckpts/CarryBox/generator.ckpt`
5. Only after inference runs, attempt official CarryBox training using
   `SUGAR/train.sh CarryBox`, with outputs clearly labeled as SUGAR
   reproduction rather than Curiosity success.
6. Use `scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh` for the
   cluster execution of the official CarryBox training stages. The wrapper
   preserves the official stage order and task names while adding cluster
   guards, logging, local Isaac visual asset resolution, and explicit stage
   start/stop controls.

## Current Status

- Official SUGAR code is cloned at `SUGAR`.
- Official IsaacLab `v2.3.0` code is cloned at `IsaacLab`.
- Official SUGAR data, descriptions, and demo checkpoints are downloaded and
  unpacked under `SUGAR`:
  - `data/CarryBox`
  - `descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf`
  - `descriptions/objects/small_box/obj_aligned.usd`
  - `demo_ckpts/CarryBox/tracker.pt`
  - `demo_ckpts/CarryBox/generator.ckpt`
- Asset download log:
  `experiments/sugar_reproduction/logs/20260711_sugar_assets_download.log`.
- A faithful SUGAR Python 3.11 environment now exists at
  `/public/home/yanhongru/envs/sugar_py311_isaacsim510`.
- The existing `isaac_arena_py312` environment is not a faithful SUGAR
  environment: setup logs and package metadata show Python 3.12,
  `isaacsim==6.0.1.0`, and `isaaclab==4.5.24`.
- A guarded official environment preparation script exists at
  `scripts/sugar/prepare_official_sugar_env.sh`. It refuses login nodes and
  refuses normal execution unless `SUGAR_ENV_BUILD_APPROVED=1` is set, because
  normal dependency installation/venv creation is not allowed on ordinary
  project compute allocations.
- Official CarryBox inference smoke ran successfully on GPU Slurm job `177522`
  (`server35`) with official SUGAR code/data/checkpoints:
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_official_inference_local_assets.log`.
  Output video:
  `experiments/sugar_reproduction/outputs/released_inference/CarryBox/videos/play/rl-video-step-0.mp4`.
- Official CarryBox refiner training smoke ran successfully on GPU Slurm job
  `177539` (`server35`) with official SUGAR train code and data:
  `experiments/sugar_reproduction/logs/20260712_sugar_carrybox_refiner_train_smoke_iter1.log`.
  It used `NUM_ENVS=64`, `MAX_ITERATIONS=1`, produced `model_0.pt`, and only
  verifies the training path.
- A compute-node-only full training pipeline launcher now exists at
  `scripts/sugar/run_official_sugar_carrybox_train_pipeline.sh`. It implements
  the official CarryBox sequence from `SUGAR/train.sh CarryBox`:
  refiner train, refiner rollout, refiner rollout processing, tracker train,
  tracker rollout, tracker rollout processing, and generator train.
- A formal refiner stage was run through persistent `tmux+srun`: session
  `curiosity_sugar_refiner_full_0712`, Slurm job `177561`, job name
  `sugar_reffull`, node `server23`, with `4096` envs. Per the updated
  operator decision on `2026-07-12`, this run was intentionally stopped after
  `model_5000.pt` was produced instead of continuing to the original
  `30001`-iteration setting. Produced checkpoints:
  `model_0.pt`, `model_1000.pt`, `model_2000.pt`, `model_3000.pt`,
  `model_4000.pt`, and `model_5000.pt`.
- The previous remaining-pipeline watcher and periodic checkpoint auto-chain
  were stopped before they could continue beyond the 5000-step boundary.
  `ckpts/refiner.pt` was not exported from this run, and the full official
  refiner-rollout/tracker/generator pipeline was not launched.
- Remaining-pipeline path handoff was rechecked at `2026-07-12 17:03 CST`.
  The active watcher target
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/ckpts/refiner.pt`
  matches the pipeline wrapper's refiner rollout checkpoint argument, and the
  wrapper keeps the official sequence from `SUGAR/train.sh CarryBox`:
  refiner rollout, refiner rollout processing, tracker train, tracker rollout,
  tracker rollout processing, then generator train.
- Pipeline wrapper maintenance at `2026-07-12 17:08 CST`: after each official
  stage, the wrapper now verifies the required next-stage artifact before
  continuing. These checks cover final/exported refiner and tracker
  checkpoints, rollout complete-trajectory `.npz` files, processed RL/IL
  datasets, and the generator final/exported checkpoint. This does not change
  official task names, stage order, data paths, or training parameters; it
  prevents a silent continuation if an official stage exits without producing
  the artifact needed by the next stage.
- Downstream rollout-processing and generator dependencies were text-audited
  before the remaining pipeline starts. The official wrapper arguments match
  `SUGAR/train.sh`, and environment metadata shows the expected
  `zarr`, `numcodecs`, `hydra-core`, `omegaconf`, `diffusers`, `accelerate`,
  `timm`, `datasets`, `numba`, and `pydantic` packages. The preflight script
  now checks these metadata entries inside compute allocations. This is
  readiness evidence only; it is not a substitute for the actual remaining
  official stages.
- Login-node-safe artifact audit added at
  `scripts/sugar/audit_official_sugar_reproduction.sh`. Latest saved audit
  `experiments/sugar_reproduction/logs/20260712_sugar_official_reproduction_audit_latest.log` reported
  `summary_present=29`, `summary_missing=10`, and
  `reproduction_status=incomplete`. The missing artifacts are the full refiner
  final/exported checkpoint, refiner rollout/process outputs, tracker
  train/export outputs, tracker rollout/process outputs, and generator
  final/exported checkpoint.
  The audit now also records the latest observed periodic refiner checkpoint as
  an explicit PASS, while keeping final completion gated on `model_30000.pt`,
  `ckpts/refiner.pt`, and the remaining official pipeline artifacts.
- Lightweight status checker added at
  `scripts/sugar/check_official_sugar_carrybox_status.sh`. It only inspects
  Slurm, files, logs, and Curiosity-owned SUGAR tmux panes; it does not launch
  Python, Isaac, training, rendering, dataset conversion, or model loading.
- A 5000-step refiner eval wrapper now exists:
  `scripts/sugar/run_official_sugar_carrybox_refiner5000_eval.sh`. It uses
  official `scripts/sugar_rl/play.py` with
  `Sugar-G129dof-CarryBox-Refiner-Rollout` and the `model_5000.pt`
  checkpoint. The no-video rollout eval on Slurm job `177782` completed on
  `server23`: all `16/16` envs completed and `13` `trajectory_complete`
  `.npz` files were saved under
  `experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full/eval/refiner_model5000_rollout_eval_novideo/raw_npz/trajectory_complete`.
- A login-node-safe 5000-step eval summarizer now exists:
  `scripts/sugar/summarize_official_sugar_refiner5000_eval.sh`. It only reads
  logs and file counts. The rollout log reports `16` expected sampled rollout
  windows and `13` saved complete trajectories, i.e. an `81.25%` sampled
  refiner-window completion rate for this small `model_5000.pt` check.
- Paper comparability was explicitly audited against arXiv Table 1
  (`https://arxiv.org/html/2605.20373v1#S4.T1`). The paper defines Carry Box
  success as the final object position being within a predefined target
  threshold, and Err as final object-target Euclidean distance. Paper Table 1
  reports SUGAR Carry Box train SR/Err `84.5/0.280` and test SR/Err
  `69.6/0.326`. The 5000-step refiner sampled-window completion rate is not
  the same metric and must not be compared as if it were paper SR/Err.
- This inference smoke required only cluster-local Isaac visual asset glue:
  `SUGAR/descriptions/terrain/sugar_ground_plane.usda`,
  `ISAACLAB_GROUND_PLANE_USD`, and `ISAACLAB_USE_LOCAL_FRAME_MARKER=1`.
- The produced SUGAR mp4 came from official `play.py --headless --video` using
  IsaacLab/Gymnasium `RecordVideo` with `render_mode="rgb_array"`. The log
  contains headless Vulkan/GLFW warnings, but they were non-fatal on this path.
  This should not be conflated with the older unresolved viewport/rendering
  manager true-render dependency issue.
- The 5000-step refiner result is a partial refiner checkpoint/eval only. It is
  not the paper-level SUGAR CarryBox result, because the paper's final
  inference/evaluation uses the full refiner, tracker, and generator sequence.

## 2026-07-13 Active Continuation

- Resume the original successful server23 `model_5000.pt` to the official
  `model_30000.pt` refiner target. The canonical starting checkpoint is stored
  separately at
  `outputs/CarryBox_20260712_official_carrybox_full/resume_sources/server23_original/model_5000.pt`
  with SHA256
  `175f4df698ca2f7e04bc94072ef2dcdd23172243a2529cbd8f84704ff615720d`.
- Keep the one-day allocations `178073` (`server36`) and `178091`
  (`server53`) alive as backup resources. Do not cancel them when switching
  work; only terminate an already-crashed Curiosity child process when needed.
- Continue polling one-day low-CPU requests across both accessible partitions:
  `178129`, `178133`, `178134`, `178136`, `178137`, and flexible request
  `178143`. Do not cancel pending backups.
- The first clean GPU to start acquires the output-directory pipeline lock and
  runs the official refiner continuation. Other acquired allocations wait for
  the lock and remain available for failover.
- After `model_30000.pt` and `ckpts/refiner.pt`, continue without reordering:
  refiner rollout, refiner processing, tracker training, tracker rollout,
  tracker processing, and generator training.
- Produce compute-node-generated visual evidence for every available stage.
  Current intermediate visualizations are `refiner_training_curves.png` and
  `refiner_model5000_rollout_summary.png`; both are explicitly partial and not
  paper-level reproduction claims.
- Active execution: Slurm `178129` on `server23` physical GPU 3 owns the
  pipeline lock and is stably continuing the official 4096-env refiner from
  iteration 5000; Slurm `178143` on physical GPU 6 holds a second one-day
  allocation and waits on the lock for immediate failover. Continue polling
  iteration progress and artifact creation without releasing either resource.
- First resumed artifact: `model_6000.pt` landed at 2026-07-13 06:11 CST and
  the watcher auto-chained to `model_7000.pt`. Continue the same persistent
  execution through `model_30000.pt`; regenerate/refinspect curves at material
  checkpoints and keep the surviving server23/server45 failover allocations.
- `model_7000.pt` landed at 2026-07-13 07:42 CST; the watcher auto-chained to
  `model_8000.pt`. The cluster automatically cancelled the idle server53
  allocation, but no backup was manually released.
- `model_8000.pt` landed at 2026-07-13 09:13 CST; the watcher auto-chained to
  `model_9000.pt`. Surviving failovers are now `178136` and `178133` on
  server35, both waiting on the output lock after the cluster automatically
  cancelled the older idle-lock allocations.

## 2026-07-13 Operator-Selected Stop At 10000

- The exact-stop watcher detected stable
  `logs/refiner/model_10000.pt` at `2026-07-13 12:13:47 CST`, terminated only
  the official refiner training child, and left Slurm allocation `178129`
  active. No `model_11000.pt` exists and no checkpoint auto-chain remains.
- The checkpoint and its read-only named copy
  `ckpts/refiner_model10000.pt` have identical SHA256
  `a398a7293fcea0ef948234e5de47b990fa586d2efd4e54ad7e481151c16124c3`.
  A compute-node `torch.load` audit recorded keys `infos`, `iter`,
  `model_state_dict`, and `optimizer_state_dict`, with internal `iter=10000`,
  in `experiments/sugar_reproduction/logs/20260713_sugar_refiner_model10000_checkpoint_audit.log`.
- The official no-video refiner rollout diagnostic loaded this checkpoint,
  completed all `16/16` sampled environments, and saved `16` complete
  trajectories. This is a refiner-only sampled diagnostic, not the paper's
  full-policy CarryBox SR/Err.
- Compute-node-generated visual evidence now includes:
  `visualizations/refiner_training_curves.png`,
  `visualizations/refiner_model10000_rollout_summary.png` with its JSON
  sidecar, and `visualizations/refiner_model10000_rollout_video.mp4`.
- Slurm `178129` remains retained in its persistent shell on `server23`; the
  low-CPU one-day backup request `178137` remains pending. Other acquired
  backups were cancelled automatically by the scheduler's idle-allocation
  policy, not manually released.
- Paper-schedule reproduction remains incomplete by operator choice because
  the official `model_30000.pt` is not being produced. Functional downstream
  reproduction is active from the byte-identical model-10000 export; this does
  not change or resume Refiner training.
- The refreshed artifact audit reports `summary_present=45`,
  `summary_missing=10`, and explicitly passes both `model_10000.pt` and the
  absence of `model_11000.pt`; the ten missing items are the deliberately
  unrun full-reproduction artifacts above.

## 2026-07-13 Functional Downstream Reproduction From Model 10000

- `ckpts/refiner.pt` is a byte-identical, read-only export of
  `ckpts/refiner_model10000.pt`, with provenance sidecar and SHA256
  `a398a7293fcea0ef948234e5de47b990fa586d2efd4e54ad7e481151c16124c3`.
- Official Refiner rollout on Slurm `178916` completed `1000/1000`
  environments and saved `922` complete trajectories (`92.2%`). Official
  `process_refiner_rollout.py` then created the RL dataset with status 0.
- The full rollout visualization reports derived final relative-position
  discrepancy mean/median `0.05787/0.05153` and remains explicitly
  non-comparable to paper SR/Err:
  `visualizations/refiner_model10000_full_rollout_summary.png`.
- Official Tracker training is active with `4096` environments and an
  operator-selected final checkpoint of `model_10000.pt`. Early reward and
  episode length rise while motion errors fall; live visual evidence is
  `visualizations/tracker_training_curves.png`.
- One-day allocations `178916` (`cpu/server13`), `178917` (`gpu/server45`),
  and `178918` (`gpux/server13`) are retained. Only `178916` runs the pipeline;
  the others remain persistent backups. Request `178137` remains pending.
