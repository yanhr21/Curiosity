# Plan 01: Simulation Platform Preflight

## Purpose

Select and verify the first executable simulation stack on a compute node. This
phase is platform validation only, not a scientific experiment.

## Primary Track: Isaac Lab-Arena

Use Isaac Lab-Arena first because it directly names a G1 loco-manipulation box
pick-and-place task. The expected stack is:

- Isaac Lab-Arena at Arena's pinned source state;
- Isaac Sim 6.0.1 installed from the official PyPI/NVIDIA wheel path;
- Isaac Lab at Arena's pinned submodule commit;
- Isaac-GR00T at Arena's pinned submodule commit;
- split local shared-filesystem environments:
  `/public/home/yanhongru/envs/isaac_arena_py312` for Isaac/Arena and
  `/public/home/yanhongru/envs/gr00t_n16_py310` for the official GR00T server.

Official first smoke target:

- Arena closed-loop policy runner, inside a Curiosity-owned tmux-held Slurm
  compute allocation, using the official tuned GR00T checkpoint:

```bash
bash scripts/isaac/run_arena_g1_locomanip_eval.sh
```

Do not run this on a login node.

The initial run is a diagnostic smoke test only. It is not evidence that the
research idea works, because it tests a published Arena/G1 policy on the
official Galileo box pick-and-place task rather than unknown-load active
probing or non-retargeting video-conditioned learning.

## Fallback Track A: GR00T-VisualSim2Real

Use if Arena versioning or Docker support blocks progress. Expected stack:

- Isaac Sim 5.1;
- Isaac Lab installed from source;
- Python 3.11;
- Unitree G1 loco-manipulation configs.

Official import smoke test from README:

```bash
python -c "from gr00t.rl.envs.base_task.base_task import BaseTask; print('OK')"
```

Only run inside compute allocation because it may load project dependencies.

## Fallback Track B: WBC-AGILE

Use if a stronger whole-body controller/evaluation workflow is needed before
box manipulation.

Expected stack:

- Isaac Lab v2.3.2;
- Isaac Sim 5.1.

Official commands are train/eval commands; do not run them until explicitly
planned as smoke tests in compute allocation.

## Version Isolation Rule

Do not mix Arena's Isaac Sim 6.0 / Isaac Lab 3.0 environment with the
Isaac Sim 5.1 / Isaac Lab 2.3.x environment used by SUGAR, VIRAL, and
WBC-AGILE. Treat them as separate environment tracks.

## Exit Criteria

- One platform has a successful official import or test smoke run in a
  Curiosity-owned tmux-held Slurm allocation.
- Exact command, allocation, environment, log path, commit, and result are
  recorded.
- Failures are recorded as blockers, not papered over by rewriting toy code.

## Current State On 2026-07-02

- Official Arena source is cloned at
  `8a74e794b621b0f8d3627d096a1bae9ce11e7b56`.
- Official IsaacLab submodule is cloned at
  `55df2c34390ba94b22d41879514c5485c5115462`.
- Official Isaac-GR00T submodule is cloned at
  `e29d8fc50b0e4745120ae3fb72447986fe638aa6`.
- Official inference checkpoint is downloaded under
  `/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial/checkpoint-20000`.
- GR00T environment dependency check passes.
- Isaac/Arena environment has unresolved metadata incompatibilities; this is a
  risk to be tested on the compute node, not silently ignored.
- Slurm allocation is pending in the Curiosity-owned tmux session
  `curiosity_isaac_arena_g1_run_0702`.
