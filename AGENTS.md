# Global Agent Rules

This file contains active operating rules only. Historical experiment records,
wrong-path evidence, and old phase narratives must not be appended here. Put
them under `IDEA/legacy/`, `PLAN/legacy_*`, `TODO/legacy_*`, or
`experiments/reports/`.

## Highest Priority Cluster Safety Rules

These rules override all other project instructions.

### Login Node Hard Limit

- Never run Python experiments, data processing, validation builders, model
  loading, rendering, simulation, training, evaluation, visualization
  generation, dataset conversion, NumPy/PyTorch-heavy scripts, or any other
  compute-heavy project task on a login or management node such as
  `mgmtserver02`.
- Login nodes are only for lightweight operations: editing files, `git`
  commands, `git clone`, `git push`, small text inspection with tools such as
  `sed`/`rg`, lightweight file listing, and job/allocation submission.
- Keep login-node CPU below 300% and memory within lightweight interactive
  limits. If a command can plausibly exceed those limits, do not run it on the
  login node.
- If a project Python command is needed and it is not a trivial import-free
  syntax check, submit or run it inside a compute allocation instead.
- If a Python or experiment process is found running on the login node for this
  project, report it immediately and stop or move the workflow to compute
  resources unless the user explicitly says otherwise.

### Compute Node Requirements

- All simulation, rendering, validation builders, source-manifest builders,
  dataset conversion, training, evaluation, and other compute tasks must run on
  compute nodes.
- GPU resources must be obtained and kept through `tmux` plus persistent
  `srun`/`salloc` allocation workflow. Do not use one-shot submission paths
  such as `sbatch`/single-use wrappers for experiments unless the user
  explicitly approves.
- Do not use `sspath` or other one-shot resource paths for this project.
- Compute nodes should only activate prebuilt local shared-filesystem
  environments. Do not perform normal dependency installation, venv creation,
  package builds, or dependency resolution on compute nodes.
- During real GPU work, maintain GPU utilization above 30%. If utilization
  stays below 30% for more than 3 hours, release the allocation or fix the job.
- Short runs must be labeled as diagnostics or smoke tests, not as real
  training or real experiment results.

### Experiment Naming And Directory Layout

- This rule is second only to the login/compute-node safety requirements.
- Do not create long, flat experiment-output directories that differ only by a
  repeated run prefix.
- Group outputs first by active phase, then by experiment family, then by the
  shortest useful variant name.
- Prefer compact paths such as
  `experiments/visuals/phase00/dense_tactile_infant/core/train_box_offset/`.
- The same grouping rule applies to visuals, outputs, reports, logs, configs,
  and future experiment artifacts.
- Run tags may remain long inside metadata, reports, JSON fields, or log
  content for reproducibility, but visible directory names should be compact.
- If moving or renaming evidence directories, update affected metadata,
  reports, summaries, manifests, and TODO/PLAN references.
- If an old flat layout is kept for history, move it into `legacy/` or document
  why it must remain.

### Resource Exclusion Zone

- Do not touch, inspect, stop, reuse, attach to, or modify any `reflex`,
  `ICLR2027/Reflex`, OpenPI, Cosmos, or other non-Curiosity tmux sessions,
  allocations, processes, logs, scripts, or resources.
- Reflex-related jobs and tmux sessions are outside this project even if they
  appear in process listings. Ignore them except to avoid interfering with
  them.
- Curiosity work must use only Curiosity-specific sessions, allocations, paths,
  and logs.

## Active Research Direction

- Active idea: `IDEA/idea.md`.
- Active plan: `PLAN/00_dense_tactile_infant/plan.md`.
- Active todo: `TODO/00_dense_tactile_infant/todo.md`.
- Current target: reference-video-aligned dense tactile environment plus base
  grasp/lift/hold.
- Success claim condition: harder held-out tasks beat the strongest baseline
  without safety regression.
- Archived wrong-path records must not be treated as the current solution,
  training recipe, success evidence, or proof of curiosity learning. Use
  `IDEA/legacy/` only for error boundaries and audit history.

## Dense Tactile Infant Requirements

- Before curiosity training, produce a base controller or model that completes
  basic grasp/lift/hold and exports dense tactile/mechanics evidence.
- Required evidence: visual scene, left/right tactile pad maps,
  pressure/compression heatmaps, `Fn`, `Ft`, shear direction, contact area,
  center of pressure, penetration/compression, material/friction/stiffness
  statistics, and grip/shear/contact time-series.
- Required representation:
  `left_pad.pressure [T,H,W]`,
  `left_pad.compression [T,H,W]`,
  `left_pad.shear_u/v [T,H,W]`,
  `left_pad.contact_mask [T,H,W]`,
  `left_pad.Fn/Ft [T]` or `[T,H,W]`, and the same `right_pad.*` fields.
- Preserve candidate Newton/MJWarp provenance under explicit namespaces such
  as `candidate.newton_mjw.Fn`, `candidate.newton_mjw.Ft`,
  `candidate.newton_mjw.area_proxy`, `candidate.newton_mjw.marker_flow`, and
  `candidate.newton_mjw.contact_normal`.
- Do not rename proxy fields into official tactile semantics:
  `area_proxy != real contact area`,
  `marker_flow render != photometric GelSight marker output`,
  `contact_count != tactile map`, and
  `candidate Fn/Ft != validated official tactile force field`.
- Gate 00F is low-priority final semantic validation/comparison-gap work.
  UniVTAC, TaCauchy, and IsaacLab TacSL are useful for final proof, but must
  not block current dense tactile infant/base-evidence work unless the user
  explicitly reopens that track.
- The old 82 FPS number is a diagnostic reference, not a hard blocker. Stable
  runtime around 80 FPS is acceptable for continuing dense tactile export and
  evidence work.

## Closed-Loop Curiosity Requirements

- Future curiosity must be true closed-loop dense visuo-tactile prediction with
  active probing and safety-constrained exploration.
- Sample reweighting alone is not closed-loop curiosity.
- Intrinsic reward must affect policy optimization and change future rollout
  data.
- Forward/world models must predict tactile/contact/mechanics, not only object
  height or contact count.
- Policies must support meaningful exploratory actions such as probing,
  regrasping, grip-force adjustment, pressure balancing, and shear-minimizing
  behavior.
- Required ablations before any success claim: vision+tactile, tactile-only
  masked vision, vision-only, and noisy/delayed/shuffled/mismatched tactile.
- The policy must not collapse to pure vision or pure tactile.
- If the base controller already solves easy grasp/lift/hold, move to harder
  held-out tasks or finer metrics before claiming curiosity improvement.

## Evaluation And Stop Gates

- A counted real-training attempt must be at least one hour, inside a
  Curiosity-owned tmux-held Slurm allocation, with GPU utilization evidence,
  exact command/log, config, checkpoint or failure record, and held-out or
  validation metrics.
- A positive result must improve the declared strongest baseline on declared
  metrics without safety regression.
- Lower training loss, script exit, checkpoint creation, rendered videos, or a
  single improved auxiliary metric do not count as success.
- Maintain an explicit attempt ledger for any real training after dense
  tactile/base evidence exists. Classify each real one-hour attempt as
  positive, negative, invalid, or blocked with evidence paths.
- If five real one-hour attempts after the current reset are negative, do not
  start attempt six. Stop, report the evidence, and wait for user instruction.
- Required metrics: lift, hold duration, slip, drop, contact loss, object
  acceleration, force/contact cost, safety regression, and strongest-baseline
  comparison.
- Sparse contact sheets are not enough for final harder-task evidence. Final
  evaluation must include full rollout videos or a documented dense-frame video
  equivalent.

## No Degraded Placeholder Model Rule

- Never write downgraded placeholder MLP/VAE/Transformer/world-model
  implementations and present them as T-Rex-style, VQ-VAE-style, or
  world-model progress.
- Do not hand-roll a toy VQ-VAE, toy expert, toy Transformer, toy world model,
  or simplified replacement when the user asked to use or replicate an
  existing serious method such as T-Rex.
- For T-Rex/VQ-VAE/world-model work, first use the official repository,
  official released checkpoints, official embedded VQ-VAE path, and faithful
  architecture/code adaptation. Only write adapter/glue code needed to connect
  official code and project data.
- If official weights/code are unavailable or incompatible, record that as a
  blocker or compatibility issue. Do not silently substitute a smaller homemade
  model to make an experiment run.
- Any simplified diagnostic must be explicitly labeled as a diagnostic and must
  not be represented as a faithful T-Rex/VQ-VAE/world-model implementation.

## Official Code, Checkpoint, And Sanity Rules

- Preserve official settings, code paths, configs, checkpoints, and version
  requirements as a 1:1 match whenever possible.
- If an official config, checkpoint, dependency, or asset is missing, obtain
  the official version if feasible. If it cannot be obtained or is
  incompatible, document it as a blocker.
- Before each experiment attempt, run a minimal official sanity check when the
  official method is part of the claim. Record command, expected output,
  observed output, and pass/fail result.
- If the official sanity check fails, downstream results are not valid for that
  official-method claim until fixed or recorded as a blocker.
- Dataset, embodiment, or schema mismatch is not a stop gate for
  namespace-preserving conversion work. Continue converting real source
  evidence under explicit provenance namespaces such as `taccel.marker.*`,
  `newton.contact.*`, or `candidate.*`; only block exact official-schema
  promotion when required fields are missing.

## Git And Commit Rules

- Do not commit unless the user explicitly asks for a commit.
- The worktree may already be dirty. Do not revert user or unrelated changes.
- Never run destructive commands such as `git reset --hard` or
  `git checkout --` unless the user explicitly requests that operation.

## Experiment Reporting Rules

- Every experiment action must be recorded in the relevant plan/TODO or
  experiment note so the work is reproducible, traceable, and auditable.
- Record commands, configs, checkpoint paths, repository commits, environment
  details, output directories, logs, and sanity-check results.
- If the same blocker or debugging loop repeats more than 3 times without
  resolution, stop, list the issue clearly for the user, and wait for approval
  or next instructions.
- Do not claim curiosity training is complete unless the evidence chain proves
  closed-loop curiosity, harder held-out evaluation, strongest-baseline
  improvement, safety metrics, videos, ablations, and faithful serious-method
  comparison or documented blocker.
- All newly generated rollout/visualization videos must be MP4 files. Do not
  generate AVI videos for this project. If a tool can only emit AVI, convert or
  replace that path before recording the artifact as project evidence.

## Workspace Layout

- Source code belongs under `src/`.
- Official external repositories belong under `external/`.
- Documentation belongs under `docs/`.
- The active research idea belongs under `IDEA/idea.md`.
- Active plans belong under `PLAN/`.
- Active task tracking belongs under `TODO/`.
- Old records belong under `legacy/`, `IDEA/legacy/`, `PLAN/legacy_*`, or
  `TODO/legacy_*`.
- Logs belong under `logs/`.
- Experiment outputs belong under `experiments/outputs/`.
- Visual outputs belong under `experiments/visuals/`, grouped by active phase.
- Experiment configs belong under `experiments/configs/`.
- Experiment reports belong under `experiments/reports/`.
- Large datasets belong under `data/`.
- Checkpoints belong under `checkpoints/`.
