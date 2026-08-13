# Plan 04: Official SUGAR Baseline + High-Fidelity Tactile Extension

## Current Operator Boundary — 2026-07-13

The active refiner endpoint is now `model_10000.pt`. Per the operator's latest
instruction, do not resume refiner training beyond iteration 10000. The
operator clarified that functional behavior and visual evidence are the
acceptance criterion; exact paper numeric equality is not required. Continue
the official downstream rollout, processing, Tracker, and Generator stages
from the unmodified model-10000 checkpoint, while labeling this as an
operator-selected truncated schedule. Keep acquired and pending Curiosity
SUGAR resources rather than releasing them.

## Active Matched Tactile Experiment — 2026-07-14

The accepted SUGAR control remains frozen. The first three matched branches
were stopped and superseded after runtime audits found two invalidating
conditions: each collision-free R15 camera-tip frame acquired a 1 kg PhysX
fallback mass when attached to the mobile G1, and the tactile network did not
preserve the accepted actor under zero input. The tip frames are now
dynamically negligible while the official elastomer collision/SDF geometry is
unchanged. Zero taxels must pass an exact actor/critic equivalence gate before
training.

Three corrected Refiner branches now run official dual-palm TacSL in the same
scene with seed 42, 512 environments, and an uninterrupted 7098-iteration
budget: full pressure+shear, policy-boundary zero, and policy-boundary
pressure-only. All start from the frozen official `model_10000.pt` without
resuming its optimizer or iteration. The official SUGAR actor and action noise
are frozen; only the spatial tactile encoder weights and tactile input columns
can update, with all zero-preserving tactile biases fixed. This makes the zero
branch an exact persistent SUGAR actor rather than a separately relearned
controller.

The training mount is the natural-policy SDF-calibrated `v3` offset (hand-frame
meters: left `-0.004606,-0.041890,0.005119`, right
`-0.005480,0.063320,0.025027`). Reference scans bound worst penetration near
5.2 mm and non-proxy activation below 1%; a 64-environment natural rollout
retained 89.06% success and exposed real taxel pressure/shear in 59.38% of
episodes. Right-hand coverage was only 12.5% and remains an open limitation.

The strict causal checkpoint audit now passes at `model_0.pt`,
`model_1000.pt`, and `model_2000.pt`. The three
`model_pre_update.pt` model states are bitwise identical across all 45 tensors
(`1,685,947` parameters). From pre-update through the first PPO update, the
`455,680` accepted-SUGAR first-layer actor weights and every other frozen actor
tensor remain bitwise exact in all branches; the zero branch's entire
`844,794`-parameter actor is exact; and full/pressure-only modify only the
permitted tactile encoder weights and tactile input columns. This proves the
training gate is operating, not that tactile improves closed-loop performance.
The same audit remains mandatory at the final checkpoint. Matched 64-environment
model1000 diagnostics are negative: full and zero have identical success sets
under nominal (57/64), low-friction (56/64), and contact-phase lateral-pulse
(57/64) conditions, while pressure-only reaches 55/64, 53/64, and 55/64
respectively. Under combined stress, full/zero/pressure reach 30/64, 34/64,
and 29/64. These diagnostics therefore justify continuing the uninterrupted
budget but cannot support an advantage claim. The model2000 combined-stress
diagnostic remains negative: full/zero/pressure reach 28/64, 34/64, and 34/64.
Among the 50 touched and action-dependent environments, full succeeds in 24
and zero in 30; among the 14 untouched environments both succeed in four.
The loss therefore lies on the causal tactile subset rather than in a
no-contact null.

Final comparison uses online execution from trajectory frame zero so contact
emerges naturally. Forced contact-frame reset is sensor validation only and
was empirically rejected as a performance protocol. The evaluation suite uses
official success/failure terms, matched live/zero/pressure interventions, and
held-out mass, friction, isolated COM offset, and contact-phase lateral
disturbances, plus a combined stress condition. Every admitted result must
serialize the locked v3 mounts, corrected robot/tip masses, checkpoint
iteration and SHA, source hashes, per-environment motion ID, and full initial
robot/object state. The full policy requires live, zero, pressure-only,
shear-only, swapped-hand, wrong-environment, and one-step-lagged executions;
the zero and pressure policies run only their trained modes. A
predeclared paired-bootstrap audit must pass before any tactile-advantage
claim, and it refuses results not bound to a passing strict final checkpoint
weight audit. The old iteration-3000 weight drift is retained only as superseded
diagnostic evidence. The corrected branches require completed matched
evaluation before any task-advantage claim.

The first completed triplet is explicitly one policy-training seed. Evaluation
environment bootstraps quantify paired rollout uncertainty but are not treated
as independent trainings. A final multi-seed claim requires at least three
distinct matched full/zero/pressure training seeds, every seed to pass the
single-seed causal/provenance gate, and positive seed-level bootstrap lower
bounds for held-out matched-control and same-policy zero-intervention success
gains.

Policy-training seed and evaluation-environment seed are distinct protocol
fields. The three policy triplets use training seeds 42/43/44, while every
final suite fixes evaluation seed 42 so the seed-level effect is not confounded
by different held-out initial-state draws. The multi-seed analyzer rejects
reports with differing evaluation seeds.

Before the active reference-only seed-42 result, a read-only TensorBoard audit
found a cross-role optimizer confound: the frozen zero actor holds the shared
actor/critic Adam at `1e-2`, whereas live tactile roles run near `1e-4`. The
running triplet and frozen suite remain unchanged, but their result is
as-implemented rather than clean modality attribution. Same-contract seed43/44
follow-ups are disarmed. A fixed two-group, fixed-rate continuation is
predeclared in `DOCS/sugar_tactile_optimizer_deconfounding_protocol.md`; clean
seed42 must pass before clean seeds 43/44 can start.

A separate latent-contact-dynamics follow-up is predeclared in
`DOCS/sugar_tactile_latent_contact_dynamics_protocol.md`. It keeps official
SUGAR rewards/terminations, the same reference-only actor, and the deconfounded
optimizer contract, but trains matched full/zero/pressure roles across
actor-hidden, coherently sampled mass, friction, COM, and contact-phase
impulses. It is activated only after the optimizer-clean current-distribution
branch and is never pooled with the current protocol. Its primary endpoint
remains paired task success, not reward or a contact proxy.

Its inactive implementation candidate is now isolated in new asset, sensor,
event, and environment-config modules. The sensor adapter does not reproduce
the TacSL force equation: it selects the current environment subset's
coefficient and delegates to the existing parent implementation. No task is
registered. A finite official Isaac App preflight is prepared but must not run
unless the sequential optimizer-clean 42/43/44 replication reaches its first
fully admissible negative report; every prior attempted seed must be a complete
positive. A passing preflight is still only an admission diagnostic, never
tactile-advantage evidence.

A login-only watcher now follows that sequential result chain without holding GPU resources.
It requests one finite persistent `tmux+srun` preflight allocation only when
the clean report is performance-negative while every provenance, exposure,
action-dependence, control-completeness, and nominal-noninferiority gate has
passed. Earlier positive reports are rehashed and revalidated before a later
negative is admitted. Three positive reports leave the latent branch inactive;
an invalid result is a blocker rather than an excuse to launch the fallback.

The watcher deliberately stops after a passing preflight instead of chaining
into training. A pass authorizes a separate audited implementation step, not an
automatic experiment: only then add process-local latent task registration and
a training entry bound to the exact preflight SHA, run a full-role `1 env x 1
update` official-App smoke plus three-role pre-update/source/optimizer/freeze
audits, and only after those pass launch fresh matched latent full/zero/pressure
training at `512 x 7098`. The latent training-seed sequence is independently
fixed at 42, 43, and 44 regardless of which current-distribution seed admitted
the fallback. Its paired evaluator must serialize, but never expose to the
actor, the hidden mass/friction/COM/pulse tuple and must keep its statistics
separate from the current-distribution suite.

The exact-state seed-42 gate is negative. The confounded reference-only seed-42
triplet was stopped after its common model-1000 diagnostic rather than spending
the remaining allocation on a result that could not support clean attribution;
its runtime records correctly lack a final checkpoint. The optimizer-clean
branch restarted from the same official warm start at 2026-07-15 16:10 CST in
three independent 24-hour jobs. Only a clean seed42 pass can request clean seeds
43/44. Every three-role activation retains a bounded barrier below the cluster
low-utilization eviction interval.

The optimizer-clean admission chain now passes the official checkpoint
structure audit, inherited official-PPO update audit, guarded task-registration
audit, and a full-role official Isaac App `1 env x 1 update` runtime smoke. The
formal three-role worker is bound to the two fixed named optimizer groups,
bitwise warm-start identity, 21-file source manifests, strict final checkpoint
audit, and the unchanged six-condition/seven-intervention success-primary
evaluation. The smoke is integration evidence only; formal RL and its closed-
loop result remain required.

Before the first final rollout, the optimizer-clean suite runner was hardened
to reject any change to the frozen core evaluator, optimizer-clean evaluator
entrypoint, optimizer-clean analysis executor, locked common analyzer, or TacSL
sensor. This adds only pre-execution SHA guards; the six conditions, seven
interventions, 256 environments, 1,501-step horizon, and statistics are
unchanged.

The conditional seed43/44 chain is now fail-closed on the completed
single-seed executor record rather than mere JSON existence. It requires the
current executor and locked analyzer hashes, all causal/provenance gates, and
all no-change flags before requesting GPUs. The frozen multi-seed statistics
are unchanged; its optimizer-clean executor additionally proves that seeds
42/43/44 use one official warm start and identical sources/prerequisites, have
matched role initialization within each seed but distinct initialization
across seeds, and occupy nine different Slurm allocations.

The official Refiner actor currently consumes the same privileged group as the
critic: exact current object pose/orientation/linear/angular velocity plus
future object-reference terms computed relative to exact current object state.
The active seed-42 branch preserves that official baseline and therefore tests
whether tactile improves an exact-state controller. It cannot establish the
separate occlusion hypothesis. If the completed exact-state branch remains
negative, activate the separately predeclared reference-only actor protocol in
`DOCS/sugar_tactile_reference_only_actor_protocol.md`. It substitutes the
reference-plan object state for actual actor object state and recomputes future
object commands in the current reference frame, while the critic remains
privileged. Compare full, zero, and pressure-only under identical 7098-update
budgets; keep noise/latency as later fixed ablations rather than result-seeking
alternatives. Do not retrofit this condition into the already-running branch
or call object state a tactile signal.

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

## 2026-07-14 SUGAR + High-Fidelity Tactile Mainline

The accepted SUGAR CarryBox reproduction is now the frozen control for the next
research phase. The active research mainline is SUGAR plus direct,
spatially-resolved tactile sensing. Existing SUGAR contact labels remain in the
baseline only; they are not evidence of tactile sensing and cannot be renamed
or reused as the new modality.

### Technical Research Result

The current local IsaacLab `v2.3.0` only provides rigid-body contact reporting
for this use case. Its net-force/force-matrix path is useful for dynamics and
validation, but it does not provide a taxel pressure field, a two-dimensional
shear field, or elastomer images.

Although the `v2.3.2` release separately adds friction-force reporting to the
ordinary `ContactSensor`, that output is still a contact-report tensor rather
than a taxel-resolved elastomer measurement. It remains a non-tactile proxy or
validation channel under this plan.

Official IsaacLab `v2.3.2` adds the TacSL-based
`isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor`. The official API
provides:

- `tactile_normal_force`: one normal-force value per tactile point;
- `tactile_shear_force`: two tangential components per tactile point;
- `tactile_rgb_image` and `tactile_depth_image` for GelSight-style sensing;
- taxel positions/orientations and penetration depth for sensor diagnostics;
- configurable taxel grid, normal stiffness, tangential stiffness, and
  friction coefficient; and
- official GelSight R15 and GelSight Mini render configurations.

The force field is a TacSL penalty model over SDF queries:
`F_n = k_n * depth` and
`F_t = min(k_t * ||v_t||, mu * F_n)`. This is a direct high-dimensional
simulated tactile signal, not a binary contact label, but it is not a full FEM
model of gel mechanics. Any real-tactile or sim-to-real claim therefore also
requires physical sensor calibration.

Pinned sources:

- IsaacLab `v2.3.2`, commit
  `37ddf626871758333d6ed89cf64ad702aef127d0`;
- [v2.3.2 release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.2);
- [official visuo-tactile guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/core-concepts/sensors/visuo_tactile_sensor.html);
- [official tactile API](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab_contrib/isaaclab_contrib.sensors.html);
- [TacSL paper](https://arxiv.org/abs/2408.06506);
- [TacEx paper](https://arxiv.org/abs/2411.04776) as a slower soft-body
  validation reference, not the first training backend.

### Tactile Truth Contract

A SUGAR+tactile run is valid only when the policy observation originates from
the tactile sensor and retains spatial structure. At minimum each hand must
provide a normal pressure map and a two-axis shear map. Pressure in pascals is
derived from normal taxel force divided by the represented surface area; raw
normal force must not be mislabeled as pressure.

At least one GelSight-style RGB or depth deformation stream must also be
recorded during sensor validation. The first policy ablation may use force
maps without tactile RGB for throughput, but it must still consume the full
spatial pressure/shear maps rather than a contact bit or aggregate wrench.

Forbidden policy inputs presented as tactile:

- contact/no-contact labels and thresholded force histories;
- body-level net contact force or a six-axis wrench alone;
- hand-object distance, object pose, SDF penetration flags, or reward terms;
- a privileged-state latent that has no deployable tactile measurement; and
- a single scalar obtained by pooling the map before it reaches the tactile
  encoder.

PhysX contact reports may be logged only for independent force-balance
validation. Penetration depth and exact simulator geometry are diagnostic or
critic-only privileged quantities, not actor tactile observations.

### 2026-07-14 Implementation Status

- The accepted SUGAR control remains unchanged. The tactile branch is a new
  `Sugar-G129dof-CarryBox-Tactile-Refiner` task warm-started from the frozen
  official-code `model_10000.pt`; optimizer state and iteration count are not
  resumed from that control checkpoint.
- The official IsaacLab v2.3.2 `isaaclab_contrib` TacSL implementation and
  GelSight R15 configs are pinned to commit
  `37ddf626871758333d6ed89cf64ad702aef127d0` and minimally backported beside
  the preserved v2.3.0 SUGAR stack.
- Each official G1 rubber hand now carries an independent referenced R15
  elastomer/camera-tip assembly. The CarryBox uses its official visual/rigid
  geometry with an SDF collision approximation for TacSL queries. The actor
  receives two hands by three channels by `20 x 25` taxels: non-negative normal
  pressure plus signed two-axis frictional shear, followed by a per-hand CNN.
- The current SUGAR control step is 20 ms, so the implemented force-field stream
  is 50 Hz, not yet the planned 60 Hz. This discrepancy remains an explicit
  rate-validation item; no result may claim 60 Hz until the task timing is
  changed and revalidated.
- Compute-node checks passed at 1, 8, 128, and 512 environments. Contact-frame
  diagnostics recorded nonzero spatial pressure/shear from both hands. The
  final 20-step record has shapes `[20, 1, 2, 20, 25]` for normal force and
  `[20, 1, 2, 20, 25, 2]` for shear, a `2.190237 mN` maximum taxel normal
  force, a `4.313159 mN` maximum taxel shear norm, and no reset or dropped
  contact-object query.
- The first persistent 512-environment pressure+shear run completed 1000
  tactile-branch iterations (`12,288,000` environment steps). A clean
  100-iteration continuation with a 128 MiB contact stack completed at
  `model_1098.pt`, SHA256
  `149beb524f7c67de00bf57a7f2aa76914a140c1de41d2fc07151853329715ca6`.
  A further 6000-iteration continuation from that checkpoint is active in
  persistent Slurm job `178916` on `server13`; it is ongoing work rather than
  a completed result.
- Official R15 TAXIM `bg.jpg` and `polycalib.npz` were downloaded from the
  Isaac 5.1 asset root with hashes recorded locally. Official renderer-backed
  `320 x 240` RGB, depth, nominal depth, and deformation streams passed
  separate left- and right-R15 runtime validations without making images a
  policy input. The two camera processes are sequential because simultaneous
  v2.3.0 `TiledCamera`/Replicator instances conflict during initialization;
  each record still contains both hands' normal/shear fields.
- All current claims remain `high-fidelity simulated tactile`. Normal-load,
  tangential, stick-slip, force-balance, image-response, and physical GelSight
  calibration gates remain open.
- Source, checkpoint, force-field, and RGB/depth evidence is frozen in
  `DOCS/sugar_tactile_branch_record.md`; experiment binaries remain local-only.

### Phase T0 — Preserve the Accepted Baseline

1. Freeze the accepted SUGAR code/config/checkpoint provenance and preserve
   the model-10000 Refiner, rollout dataset, Tracker/Generator artifacts, and
   visualizations under the ignored `experiments/` tree.
2. Record a no-tactile baseline evaluation manifest before changing IsaacLab
   or SUGAR observation dimensions.
3. Keep the official SUGAR stage order and task names as the control. Tactile
   work uses new task/config names and must never overwrite baseline outputs.

### Phase T1 — IsaacLab v2.3.2 Compatibility Gate

1. Preserve the root `IsaacLab/` v2.3.0 source as the faithful reproduction
   boundary, then prepare an auditable v2.3.2 tactile adaptation from the
   official tag.
2. Use a separate prebuilt shared-filesystem environment for the tactile stack.
   Any environment build or dependency resolution requires an explicitly
   approved compute allocation; never install on the login node.
3. Reapply only the two audited cluster-local ground/marker compatibility
   changes and the minimum SUGAR API compatibility changes.
4. On a compute node, run SUGAR task-registration, reset, inference-checkpoint,
   and short rollout diagnostics before adding a sensor. Exact baseline numbers
   are not required, but task behavior must remain functionally normal.
5. If a direct v2.3.2 upgrade is incompatible, backport the official v2.3.2
   TacSL sensor, assets, and configs with commit-level provenance. Do not write
   a simplified local tactile replacement.

### Phase T2 — Dual-Palm Sensor Asset Integration

1. Add separate left/right GelSight-style elastomer patches to the G1 palm
   surfaces. Start with official GelSight Mini or R15 assets/configs; do not
   approximate them with collision boxes plus contact labels.
2. Keep sensor transforms, elastomer geometry, active taxel area, taxel layout,
   and camera intrinsics explicit in configuration.
3. Give the CarryBox contact object a valid SDF collision representation while
   preserving the baseline rigid-body mass/inertia and visual geometry.
4. Start with a `20 x 25` force grid per palm and 60 Hz sensor updates, then
   change resolution/rate only from measured accuracy-throughput evidence.
5. Preserve left/right maps separately. Record normal force, pressure, shear,
   RGB/depth, taxel pose, sensor timestamp, and validity mask in diagnostics.

### Phase T3 — Physical Fidelity and Calibration Gates

Run these tests on compute nodes before policy training:

1. **No contact:** pressure/shear remain at declared noise floor and tactile
   images match the no-contact baseline.
2. **Normal load sweep:** integrated pressure rises monotonically with known
   normal load, the contact patch grows/moves plausibly, and the integrated
   taxel force agrees with an independent simulator wrench within a declared
   tolerance.
3. **Tangential sweep:** shear direction opposes relative tangential motion;
   magnitude follows the configured tangential stiffness and saturates at the
   Coulomb limit.
4. **Stick-slip:** a controlled lateral ramp exhibits a repeatable transition
   from sticking to sliding rather than only a binary contact transition.
5. **Spatial test:** known box translations/rotations move the pressure centroid
   and shear pattern in the expected taxel coordinates.
6. **Symmetry and repeatability:** mirrored left/right loading produces
   correspondingly mirrored fields across seeds and resets.
7. **Rate/performance:** verify 60 Hz temporal output and benchmark 1, 16, 64,
   256, and larger vectorized environment counts without calling a short run
   training evidence.
8. **Physical sensor calibration:** compare load, contact footprint, shear/slip
   trend, and image statistics to a real GelSight-class sensor. Without this
   gate, use only the label `high-fidelity simulated tactile`.

### Phase T4 — SUGAR Observation and Data Integration

1. Register a new tactile CarryBox task derived from official SUGAR configs;
   leave official task names and baseline configs unchanged.
2. Add sensor-backed observation terms for left/right pressure and shear maps,
   plus GelSight RGB/depth where enabled. The actor must not read geometric SDF
   depth or object state as a substitute for tactile data.
3. Use a spatial tactile encoder and fuse its per-hand embeddings with SUGAR's
   existing proprioceptive/motion representation. Retain raw maps in logged
   rollout data for audit and alternative encoders.
4. First modify observations only. Keep baseline rewards and terminations fixed
   to isolate the value of sensing. Tactile-aware rewards are a later, separate
   ablation.
5. Extend Refiner and Tracker rollout schemas to store synchronized tactile
   sequences, masks, calibration metadata, and sensor randomization parameters.
6. Decide whether Generator receives tactile history only after the
   observation-only Refiner/Tracker path is validated; do not silently alter
   every SUGAR stage at once.

### Phase T5 — Training Schedule

1. Run sensor-only and observation-shape smoke tests using official
   SUGAR/IsaacLab code, clearly labeled diagnostics.
2. Warm-start from the accepted SUGAR checkpoint where tensor compatibility
   permits, initializing only the new tactile encoder/fusion parameters.
   Compare against a matched from-scratch tactile run to expose warm-start
   effects.
3. Preserve the SUGAR Refiner -> rollout/process -> Tracker -> rollout/process
   -> Generator order. Each tactile artifact uses a separate experiment name
   and provenance manifest.
4. Train first with pressure+shear, then enable GelSight RGB/depth after the
   force-map path meets throughput and learning gates.
5. Randomize sensor stiffness, friction coefficient, tangential stiffness,
   zero offset, noise, latency, dead taxels, and camera appearance within ranges
   supported by calibration. Do not randomize away physically implausible
   failures.
6. Use persistent one-day `tmux` plus `srun`/`salloc` resources. Retain acquired
   backups and choose lower-CPU or alternate partitions if a request does not
   start promptly.

### Phase T6 — Ablations and Evaluation

Use matched seeds, environment steps, checkpoint selection, and evaluation
episodes for:

1. frozen official SUGAR;
2. contact-label/body-force proxy control, explicitly labeled non-tactile;
3. normal pressure only;
4. normal pressure plus two-axis shear;
5. GelSight RGB/depth only;
6. full pressure + shear + GelSight;
7. full tactile with sensor dropout/noise/latency randomization; and
8. full tactile with the tactile stream masked at evaluation.

Pressure/shear attribution is locked separately in
`DOCS/sugar_tactile_modality_advantage_protocol.md`. After the primary gate
only, its unchanged-suite analyzer requires both full versus a trained matched
pressure policy and full-live versus same-policy pressure masking to have
positive held-out success CI lower bounds, plus nominal noninferiority, shear
exposure, and action dependence. A shear-only mask is not misrepresented as a
separately trained shear policy.

The post-gate actor-input stress suite is fixed in
`DOCS/sugar_tactile_postgate_robustness_protocol.md` before the optimizer-clean
seed42 result. It forks rather than edits the frozen current evaluator and uses
paired nominal/combined runs for 20/40/100 ms delays, 10/25% frame and dead-
taxel loss, complete left/right loss, and 1/5 kPa numerical stress noise. These
are reproducible simulated input corruptions, not a calibrated physical sensor
noise model.

The first held-out geometry gate is separately fixed in
`DOCS/sugar_tactile_heldout_geometry_protocol.md`: evaluation-only local-Y
0.90/1.10 width variants of the pinned official CarryBox USD, with matched
visual/SDF scaling, exact physics readback, no actor geometry input, no
retraining, and no post-result mount or scale tuning. The official big-box USD
belongs to KickBox/PushBox and is not treated as matched CarryBox evidence.

For every learned proxy control, match the selected tactile profile's scene,
warm start, actor-freeze boundary, seeds, update budget, and optimizer dynamics.
In particular, the reference-only proxy uses the same fixed `1e-3` critic and
`1e-4` adapter-actor Adam groups as the optimizer-clean tactile branch while
retaining the unchanged official RSL-RL PPO update.

The frozen official SUGAR comparison reuses the audited exact-state zero
checkpoint actor without retraining, but reruns its six zero-taxel conditions
with the current evaluator. Require bitwise equality of motion IDs, start
frames, robot/object state, material tensors, last action, and scene origins
before paired statistics. Do not use the older exact-state NPZ files for this
new cross-profile claim because they predate several of those serialized
initial-state fields.

Evaluate both nominal CarryBox and held-out mass, object friction, geometry,
center-of-mass offset, disturbance direction/magnitude, and partial sensor
failure. Report official SUGAR success/error together with:

- drop and humanoid-fall rate;
- cumulative and peak slip distance;
- peak/mean pressure and left-right pressure imbalance;
- shear-to-normal ratio and Coulomb-margin violations;
- contact-patch area, centroid drift, and contact-loss duration;
- object tilt/oscillation and energy/torque cost; and
- tactile inference latency, simulator steps per second, GPU memory, and
  achievable parallel environment count.

### Phase T7 — Claim Boundary and Fallbacks

- A gain over SUGAR must survive matched-budget and proxy-control ablations.
- TacSL results without physical calibration are simulated-tactile results,
  not real-hardware tactile claims.
- TacEx may be used on a small validation scene to compare gel deformation and
  image/contact trends. Its known multi-environment and dependency risks block
  using it as the first large-scale SUGAR training backend.
- The commercial Tashan tactile asset path in Isaac Sim is not the primary
  choice until its data contract, licensing, reproducibility, and cluster
  deployment are auditable.
- If official tactile assets/code cannot run in the matching Isaac stack,
  record the compatibility blocker. Never replace them with contact labels,
  thresholded forces, or a hand-written toy tactile model.

### Phase T8 — Official Tactile Genesis Candidate Backend

Admit the official Tactile Genesis stack as a second, independent
high-fidelity simulated tactile backend. The detailed executable ladder is
frozen in `PLAN/05_direct_tactile_rgb_policy/plan.md` under P0-C, with source
and runtime evidence in
`DOCS/tactile_genesis_candidate_backend_audit_20260717.md`.

1. Keep the accepted IsaacLab/TacSL implementation as the primary
   SUGAR-native path. Tactile Genesis runs under Genesis World and is not a
   drop-in IsaacLab sensor or a source of interchangeable tensors.
2. Use only the official paper snapshot and maintained official Genesis World
   implementation. Preserve per-taxel force/torque and native elastomer-marker
   displacement; binary contact, depth-only, proximity, and aggregate wrench
   remain non-tactile proxy controls.
3. After a compatible prebuilt environment is provisioned, run the untouched
   official tactile sandbox, then reproduce the TacSL zero/onset, normal-load,
   signed `+U/-U` and `+V/-V` shear, speed/load, and footprint-translation
   benches with a minimal audited G1 palm adapter.
4. Compare TacSL and Tactile Genesis only on declared shared units and
   coordinate frames. Do not relabel marker displacement as GelSight RGB/depth.
5. Consider cross-backend calibration/domain randomization or a Genesis-native
   SUGAR port only after batching, reset determinism, latency, and action/state
   equivalence pass. Do not start implicit Isaac/Genesis co-simulation or mix
   backend observations without simulator-specific ablations.

The current exact blocker is the absence of a compatible prebuilt Genesis
environment: the approved SUGAR stack is Python 3.11/Torch 2.7, while the
paper task requires Python 3.12 and its embedded Eden requires Torch >=2.9.2.
Do not install or resolve that dependency stack inside an active SUGAR compute
allocation. Any passing result must be called `high-fidelity simulated tactile
under Genesis physics`, not physical GelSight validation, IsaacLab equivalence,
or sim-to-real.
