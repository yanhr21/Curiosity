# Global Agent Rules

This repository is now for video-guided, embodiment-aware active
loco-manipulation for unknown-load carrying. Old dense-tactile Curiosity
materials were archived outside the repository at:

```text
/public/home/yanhongru/Curiosity_archive_20260702_pre_video_guided_carrying/
```

Do not treat archived Curiosity results as current success evidence. They may
only be used as historical caution about overclaiming, weak held-out transfer,
and proxy-field promotion.

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

## Active Research Direction

- Active idea: `IDEA/idea.md`.
- Main survey: `docs/2026-07-02_research_overview.md`.
- Working title:
  `Video-guided, embodiment-aware active loco-manipulation for unknown-load carrying`.
- Core claim: video can provide task semantics, progress, object-motion, and
  contact-affordance priors, but the robot must actively probe unknown object
  dynamics and choose a stable, low-cost posture for its own body.
- Current negative conclusion: as of 2026-07-02, no known system fully solves
  cross-morphology humanoid box carrying with unknown weight/shape, active
  self-selected posture, long-duration carrying, and non-retargeting
  video-conditioned RL.

## Non-Retargeting Rule

- Do not turn the project into human-to-robot joint retargeting, motion
  shadowing, teleoperation replay, or end-effector trajectory cloning.
- Human, robot, or simulation video may be used as a weak reference for task
  phase, progress, object displacement, contact-location priors, and success
  or failure cues.
- Video must not be treated as a command to copy human joint angles, body
  posture, arm trajectories, footstep timing, or grasp geometry.
- Retargeting, teleoperation, and behavior cloning methods may be used only as
  baselines or data sources when explicitly labeled as such.

## Active-Probing Requirement

- RGB or RGB-D video alone cannot determine object mass, center of mass,
  friction, internal fill, stiffness, or required carrying force.
- A valid policy must include active probing behaviors such as micro-lift,
  push-pull, grip-force ramping, stance adjustment, footstep repositioning,
  hold-height adjustment, arm/torso contact redistribution, and gait-speed
  modulation.
- A valid world or belief model must represent uncertainty over object
  dynamics and update that belief from probing feedback.
- Do not claim video-conditioned success if probing is absent or if unknown
  load properties are secretly provided as privileged inputs.

## Embodiment-Aware Carrying Requirement

- The policy must adapt to robot morphology and limits: height, mass, limb
  lengths, joint ranges, torque limits, hand/forearm/chest contact geometry,
  foot support polygon, balance controller, and actuator thermal or effort
  limits.
- The same reference video should not force the same posture across different
  robot bodies. A successful method should choose different feasible carrying
  strategies when morphology or load changes.
- Required strategy space includes at least: front carry, low carry,
  chest/torso-supported carry, asymmetric carry, regrasp, stance widening,
  squat depth adjustment, and walking-speed reduction.

## Evidence And Metrics

- Required evidence for any real claim:
  synchronized scene video, object pose, estimated load belief, contact state,
  robot joint states, torque or effort cost, CoM/ZMP or balance margin,
  footsteps, slip/drop/contact-loss events, and safety events.
- Required metrics:
  carry distance, carry duration, drop rate, slip, contact loss, fall rate,
  recovery after perturbation, object acceleration, energy or torque cost,
  peak joint torque, balance margin, probing attempts, and posture diversity
  across robot bodies and load distributions.
- Harder held-out tests must vary object weight, center of mass, shape,
  size, friction, handle availability, robot morphology, and reference-video
  embodiment.

## Success Claim Gate

A success claim requires all of the following:

- It beats the strongest declared baseline on harder held-out tasks.
- It has no safety regression in falls, drops, excessive torque, object
  acceleration, or collision/contact-force limits.
- It shows that video conditioning improves over no-video RL or scripted
  probing without collapsing into retargeting.
- It shows that active probing improves over video-only or privileged-static
  inference.
- It shows morphology-dependent posture selection, not one fixed pose copied
  across robots.
- It includes ablations for no-video, wrong-video, mismatched embodiment
  video, retargeting baseline, behavior-cloning baseline, no-probing,
  oracle-load, and corrupted or delayed force/contact feedback.

Anything weaker is a diagnostic, engineering milestone, or negative result.

## Official Code And Serious Method Rule

- Use official repositories, released checkpoints, and faithful configs when
  claiming comparison to a serious method.
- Do not hand-roll toy VQ-VAE, toy Transformer, toy world model, toy humanoid
  controller, or simplified video-conditioned policy and present it as
  serious-method progress.
- If official weights, code, assets, or environments are unavailable or
  incompatible, document that as a blocker or comparison gap.
- Simplified code is allowed only when clearly labeled as a diagnostic or
  interface smoke test.

## Experiment Reporting Rules

- Every experiment action must be recorded in the relevant plan, TODO, or
  report with command, config, environment, output path, and status.
- A counted real-training attempt must be at least one hour inside a
  Curiosity-owned tmux-held Slurm allocation, with GPU-utilization evidence,
  exact command/log, config, checkpoint or failure record, and held-out
  evaluation.
- If the same blocker or debugging loop repeats more than 3 times without
  resolution, stop, list the issue clearly for the user, and wait for approval
  or next instructions.
- Newly generated rollout or visualization videos must be MP4 files. Do not
  generate AVI as active evidence.

## Git And Commit Rules

- Do not commit unless the user explicitly asks for a commit.
- The worktree may already be dirty. Do not revert user or unrelated changes.
- Never run destructive commands such as `git reset --hard` or
  `git checkout --` unless the user explicitly requests that operation.

## Workspace Layout

- Source code belongs under `src/`.
- Official external repositories belong under `external/`.
- Documentation belongs under `docs/`.
- The active research idea belongs under `IDEA/idea.md`.
- Active plans belong under `PLAN/`.
- Active task tracking belongs under `TODO/`.
- Old local material belongs outside the repo archive unless the user
  explicitly asks to restore it.
- Logs belong under `logs/`.
- Experiment outputs belong under `experiments/outputs/`.
- Visual outputs belong under `experiments/visuals/`.
- Experiment configs belong under `experiments/configs/`.
- Experiment reports belong under `experiments/reports/`.
- Large datasets belong under `data/`.
- Checkpoints belong under `checkpoints/`.
