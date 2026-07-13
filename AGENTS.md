# Global Agent Rules

## No Degraded Placeholder Model Rule

- Never write downgraded placeholder MLP/VAE/Transformer/world-model
  implementations and present them as T-Rex-style, VQ-VAE-style, SUGAR-style,
  or world-model progress.
- For SUGAR work, use the official repository, official released data,
  official descriptions, official checkpoints, official task registration, and
  the matching IsaacLab stack. Only write adapter or cluster glue needed to run
  that official code in this workspace.
- If official weights, code, assets, or runtime requirements are unavailable
  or incompatible, record the blocker exactly. Do not silently substitute a
  smaller homemade model or local controller to make an experiment run.
- Any reduced run must be explicitly labeled as a smoke test or diagnostic and
  must still use official SUGAR code and official SUGAR assets.

## Active Workspace Scope

This workspace's active mainline consists of the official SUGAR source and its
matching official IsaacLab source.

- Active idea: `IDEA/idea.md`.
- Active plan: `PLAN/04_sugar_baseline/plan.md`.
- Active TODO: `TODO/04_sugar_baseline/todo.md`.
- Active scripts: `scripts/sugar/`.
- Active mainline source trees at the workspace root:
  - `SUGAR/`
  - `IsaacLab/`
- Active reproduction outputs: `experiments/sugar_reproduction/outputs/`.
- Active logs: `experiments/sugar_reproduction/logs/`.
- Active reproduction record: `DOCS/sugar_carrybox_reproduction_full_record.md`.
- Active local report: `experiments/reports/sugar_baseline_status_20260711.md`
  (ignored; never commit or push it).
- Active dependency cache: `external/wheelhouse` for local dependency cache
  only.

`SUGAR/` and `IsaacLab/` must remain at the workspace root because both are
active mainline source trees, not external baselines. Reproduction outputs, checkpoints, datasets,
visualizations, videos, and logs belong under `experiments/sugar_reproduction/`
and are local-only. Transitional symlinks at
`external/SUGAR`, the legacy IsaacLab link under `external/`, `SUGAR/outputs`,
and `logs/sugar` exist only so the already-running pipeline, historical
commands, and the prebuilt environment's editable installs remain valid; new
scripts and docs must use the canonical root paths above. The editable-install
finders currently record old absolute paths below `external/`, so do not remove
those compatibility links until that environment is rebuilt or reinstalled in
an approved compute allocation.

## Experiment Git Exclusion Rule

- The entire root-level `experiments/` tree must remain ignored by Git.
- Never stage, commit, or push any file or directory below `experiments/`,
  including reports, logs, checkpoints, datasets, videos, and visualizations.
- Never use `git add -f`, a `.gitignore` negation rule, or any other mechanism
  to force `experiments/` content into Git history.
- Keep experiment artifacts on the shared filesystem. Put any concise result
  summary that must be version-controlled under `DOCS/`, `PLAN/`, or `TODO/`
  instead of `experiments/`.

Old dense-tactile Curiosity materials were archived outside the repository at:

```text
/public/home/yanhongru/Curiosity_archive_20260702_pre_video_guided_carrying/
```

Old non-SUGAR plans, TODOs, experiments, logs, scripts, docs, top-level
artifacts, and external repos from this workspace are stored outside the
repository under:

```text
/public/home/yanhongru/Curiosity_legacy/20260712_pre_sugar_workspace_cleanup/
```

The root-level `legacy/` path must remain absent and ignored. Never move this
archive back into Curiosity, and never stage, commit, or push any legacy
content. `Curiosity_legacy` is a sibling archive directory, not repository
source.

Do not treat archived Curiosity, AGILE, MuJoCo, tactile, prismatic, or failed
rendering results as current success evidence. They may only be used as
historical negative evidence or for comparison after the SUGAR reproduction is
complete.

## Highest Priority Research Mainline: SUGAR Baseline

- The highest-priority mainline is the official SUGAR baseline:
  `SUGAR: A Scalable Human-Video-Driven Generalizable Humanoid
  Loco-Manipulation Learning Framework`.
- SUGAR is the closest public baseline because it combines human-video-driven
  loco-manipulation, IsaacLab, G1-style humanoids, and CarryBox-like tasks.
- First reproduce official SUGAR CarryBox faithfully. Only then add
  Curiosity-specific changes on top of SUGAR, with ablations against the
  reproduced baseline.
- Do not continue old direct Isaac G1/AGILE scalar tuning, MuJoCo carrying
  paths, tactile-only paths, or non-SUGAR proxy scaffolds as active work.

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

### Compute Node Requirements

- All simulation, rendering, dataset conversion, training, evaluation, model
  loading, and visualization generation must run on compute nodes.
- GPU resources must be obtained and kept through `tmux` plus persistent
  `srun`/`salloc` allocation workflow. Do not use one-shot submission paths
  such as `sbatch` or single-use wrappers for experiments unless the user
  explicitly approves.
- Do not use `sspath` or other one-shot resource paths for this project.
- Compute nodes should only activate prebuilt local shared-filesystem
  environments. Do not perform normal dependency installation, venv creation,
  package builds, or dependency resolution on compute nodes.
- Short runs must be labeled as diagnostics or smoke tests, not as real
  training or real experiment results.

### Resource Exclusion Zone

- Do not touch, inspect, stop, reuse, attach to, or modify any `reflex`,
  `ICLR2027/Reflex`, OpenPI, Cosmos, or other non-Curiosity tmux sessions,
  allocations, processes, logs, scripts, or resources.
- If non-project sessions appear in process listings, ignore them except to
  avoid interference.

## SUGAR Fidelity Gates

- Use official SUGAR task names, official CarryBox data, official robot/object
  descriptions, official checkpoints, and the official training stage order.
- Keep SUGAR and IsaacLab local changes minimal, auditable, and limited to
  cluster/runtime compatibility unless the user explicitly asks for research
  modifications.
- **Reproduction accepted/passed by the user on 2026-07-13.** The acceptance
  criterion is that the official SUGAR CarryBox pipeline and reproduced effect
  are functionally normal; exact equality with paper-reported numbers is not
  required.
- The accepted local Refiner boundary is the official-code `model_10000.pt`.
  Do not resume Refiner training beyond iteration 10000. Its successful
  rollout, processed dataset, and visualizations are valid reproduction
  evidence.
- Tracker and Generator continuation may remain active to complete and improve
  the local artifact chain, but their unfinished or numerically non-identical
  results do not revoke the user-approved SUGAR reproduction pass. Record their
  actual status truthfully and do not claim exact paper-number reproduction.
- Any future research claim must compare against the faithful SUGAR CarryBox
  reproduction, not against archived local proxy tasks.
