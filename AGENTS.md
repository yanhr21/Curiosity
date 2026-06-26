# Global Agent Rules

## Highest Priority Cluster Safety Rules

These rules override all other project instructions. If any lower-priority
instruction conflicts with this section, follow this section and record the
conflict.

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

### Resource Exclusion Zone

- Do not touch, inspect, stop, reuse, attach to, or modify any `reflex`,
  `ICLR2027/Reflex`, OpenPI, Cosmos, or other non-Curiosity tmux sessions,
  allocations, processes, logs, scripts, or resources.
- Reflex-related jobs and tmux sessions are outside this project even if they
  appear in process listings. Ignore them except to avoid interfering with
  them.
- Curiosity work must use only Curiosity-specific sessions, allocations, paths,
  and logs.

## No Degraded Placeholder Model Rule

- Hard user rule from 2026-06-24: never casually write downgraded placeholder MLP/VAE/Transformer/world-model implementations and present them as T-Rex-style, VQ-VAE-style, or world-model progress.
- Do not hand-roll a toy VQ-VAE, toy expert, toy Transformer, toy world model, or simplified replacement when the user asked to use or replicate an existing serious method such as T-Rex.
- For T-Rex/VQ-VAE/world-model work, first use the official repository, official released checkpoints, official embedded VQ-VAE path, and faithful architecture/code adaptation. Only write adapter/glue code needed to connect official code and project data.
- If official weights/code are unavailable or incompatible, record that as a blocker or compatibility issue. Do not silently substitute a smaller homemade model to make an experiment run.
- Any simplified diagnostic must be explicitly labeled as a diagnostic and must not be represented as a faithful T-Rex/VQ-VAE/world-model implementation.
- Strictly follow `IDEA/idea.md`, staged plans under `PLAN/`, and staged tasks under `TODO/`. Do not collapse active planning back into one monolithic plan or todo file. Complete the current phase before advancing unless the user explicitly approves a parallel track. The old `plans/` contents are archived under `PLAN/legacy/` and `TODO/legacy/` as historical records. Do not downgrade, shortcut, fake progress, or silently change the research direction to make an experiment easier.

## Official Code, Checkpoint, and Sanity-Check Rules

- When downloading Newton, T-Rex, Taccel, or any checkpoint/model repository, preserve the original official settings, code paths, configs, checkpoints, and version requirements as a 1:1 match whenever possible.
- If an official config, checkpoint, dependency, or asset is missing, download or obtain the official version before running the experiment. If it cannot be obtained or is incompatible, document it as a blocker.
- Never train a small replacement MLP, VAE, VQ-VAE, Transformer, world model, tactile encoder, or policy and present it as equivalent to an official method or checkpoint.
- Before each experiment attempt, run a minimal sanity check that verifies the original official path produces expected outputs. Record the sanity-check command, expected output, observed output, and pass/fail result.
- If the official sanity check fails, stop treating downstream results as valid until the failure is fixed or documented as a blocker.
- Dataset, embodiment, or schema mismatch is not a stop gate for namespace-preserving conversion work. Continue converting real source evidence under explicit provenance namespaces such as `taccel.marker.*`, `newton.contact.*`, or `candidate.*`; only block exact T-Rex schema promotion when required official fields are missing. Do not rename partial source evidence into official T-Rex keys.

## Cluster Usage Rules

- The login-node restrictions in `Highest Priority Cluster Safety Rules` are mandatory and take precedence over this section.
- GPU resources must be acquired and held through `tmux` or an equivalent persistent interactive session. Do not use one-shot `sbatch` runs for experiments unless the user explicitly approves a different cluster workflow.
- Keep GPU jobs inside `tmux` or an equivalent persistent terminal/session mechanism so the job is not released or interrupted when the connection drops.
- Each GPU resource request must reserve at least one full day by default, for example one GPU for one day. Reuse the same held allocation for multiple sanity checks, visualization jobs, and follow-up diagnostics instead of repeatedly submitting short jobs and wasting time in the queue.
- All experiment environments must always be prepared first as local shared-filesystem venvs under `envs/` before using compute resources. Never configure environments directly on compute nodes: no normal `pip install`, dependency resolution, venv creation, package build, or environment setup should consume compute-node allocation time. Compute nodes should only activate existing venvs and run sanity checks, visualization, evaluation, or training. This applies to all future experiments without exception unless the user explicitly approves a different workflow.
- If local venv installation is blocked by proxy, network, or package-index issues, first try switching proxy settings and/or package sources; when the proxy route is unreliable, prefer Tsinghua or other China-accessible mirrors before any other workaround. Do not move dependency installation or environment setup onto compute nodes just because the default local install path is slow or blocked.
- While holding GPU resources in `tmux`, keep GPU utilization above 30% during real training. If utilization stays below 30% for more than 3 hours, release or fix the job instead of wasting the allocation.
- Training runs must use at least one GPU for at least one hour unless the user explicitly requests a short diagnostic/smoke test.
- Any short run must be clearly labeled as a diagnostic or smoke test, not as a real training result.

## Experiment Execution and Reporting Rules

- Every experiment action must be directly reported and recorded in the relevant plan/TODO or experiment note so the work is reproducible, traceable, and auditable.
- Record commands, configs, checkpoint paths, repository commits, environment details, output directories, logs, and sanity-check results.
- If the same blocker or debugging loop repeats more than 3 times without resolution, stop, list the issue clearly for the user, and wait for approval or next instructions. Do not drift into unrelated work.
- Maintain a clean workspace:
  - source code belongs under `src/`;
  - official external repositories belong under `external/`;
  - documentation belongs under `docs/`;
  - the active research idea belongs under `IDEA/`;
  - active plans belong under `PLAN/`;
  - active task tracking belongs under `TODO/`;
  - old pre-pivot records belong under root `legacy/`;
  - logs belong under `logs/`;
  - experiment outputs belong under `experiments/outputs/`;
  - visual outputs belong under `experiments/visuals/`;
  - experiment configs belong under `experiments/configs/`;
  - experiment reports belong under `experiments/reports/`;
  - large datasets belong under `data/`;
  - checkpoints belong under `checkpoints/`.
