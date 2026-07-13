# Curiosity SUGAR Workspace

This repository is now organized around one active mainline: faithful official
SUGAR CarryBox reproduction, followed by modifications made directly on top of
SUGAR.

Active files and directories:

- `IDEA/idea.md`: SUGAR-only research direction and claim gates.
- `PLAN/04_sugar_baseline/plan.md`: official SUGAR reproduction plan.
- `TODO/04_sugar_baseline/todo.md`: current SUGAR task list.
- `scripts/sugar/`: compute-node-safe SUGAR setup, preflight, inference,
  training, watcher, status, and audit scripts.
- `SUGAR/`: active official SUGAR source tree at the workspace root.
- `IsaacLab/`: active official IsaacLab source tree at the workspace root.
- `experiments/sugar_reproduction/outputs/`: checkpoints, datasets, videos,
  visualizations, and other reproduction outputs.
- `experiments/sugar_reproduction/logs/`: active SUGAR logs.
- `experiments/reports/sugar_baseline_status_20260711.md`: current SUGAR
  baseline status report.
- `DOCS/sugar_carrybox_reproduction_full_record.md`: end-to-end reproduction
  and H200 rendering record.
- `external/wheelhouse`: local dependency cache, not a research baseline.

Archived non-SUGAR workspace contents are under:

```text
/public/home/yanhongru/Curiosity_legacy/20260712_pre_sugar_workspace_cleanup/
```

That archive contains old failed experiments, old plans/TODOs, non-SUGAR docs,
old scripts, old logs, top-level artifacts, and external repos that are no
longer active. It is outside this repository and must never be committed or
pushed. These files are historical context only, not current success evidence.

Cluster rule: do not run simulation, rendering, training, dataset conversion,
model loading, or other heavy Python work on the login node. Use the
Curiosity-owned `tmux` plus persistent `srun`/`salloc` workflow for compute
work.

The entire `experiments/` tree is local-only and intentionally ignored by Git;
do not commit or push its reports, models, datasets, logs, videos, or
visualizations. Large downloaded SUGAR assets and runtime compatibility links
under `SUGAR/` also remain ignored.
